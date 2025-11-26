//===--- SemaRipple.cpp - Semantic Analysis for Ripple constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements semantic analysis for Ripple directives.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaRipple.h"
#include "TreeTransform.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/AttrIterator.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifierBase.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/StmtRipple.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaOpenMP.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>

namespace clang { class Scope; }

using namespace clang;

#define DEBUG_TYPE "semaripple"

/// We don't want to modify SemaOpenMP for now so a copy is the only way to
/// expose these static methods from clang/lib/Sema/SemaOpenMP.cpp
/// The loop analysis part is mostly copied from SemaOpenMP.cpp with
/// simplifications:
/// - We don't capture variables
/// - We don't handle nested loops (used by OpenMP for collapse)
/// - We don't support c++ range loop, i.e., for (auto x : range)

//====----------------- Begin OpenMP copied section ---------------------====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

namespace {

/// Iteration space of a single for loop.
struct LoopIterationSpace final {
  /// True if the condition operator is the strict compare operator (<, > or
  /// !=).
  bool IsStrictCompare = false;
  /// True if CounterVar is declared as part of the loop init
  bool IsDeclStmtInit = false;
  /// Condition of the loop (LB <op> UB, i.e., do we enter the loop?)
  Expr *PreCond = nullptr;
  /// This expression calculates the number of iterations in the loop.
  /// It is always possible to calculate it before starting the loop.
  Expr *NumIterations = nullptr;
  /// The loop counter variable.
  Expr *CounterVar = nullptr;
  /// Private loop counter variable.
  Expr *PrivateCounterVar = nullptr;
  /// This is initializer for the initial value of #CounterVar.
  Expr *CounterInit = nullptr;
  /// This is step for the #CounterVar used to generate its update:
  /// #CounterVar = #CounterInit + #CounterStep * CurrentIteration.
  Expr *CounterStep = nullptr;
  /// Should step be subtracted?
  bool Subtract = false;
  /// Source range of the loop init.
  SourceRange InitSrcRange;
  /// Source range of the loop condition.
  SourceRange CondSrcRange;
  /// Source range of the loop increment.
  SourceRange IncSrcRange;
};

/// Helper class for checking canonical form of the Ripple loops and
/// extracting iteration space of each loop in the loop nest, that will be used
/// for IR generation.
class RippleIterationSpaceChecker {
  /// Reference to Sema.
  Sema &SemaRef;
  /// A location for diagnostics (when there is no some better location).
  SourceLocation DefaultLoc;
  /// A location for diagnostics (when increment is not compatible).
  SourceLocation ConditionLoc;
  /// A source location for referring to loop init later.
  SourceRange InitSrcRange;
  /// A source location for referring to condition later.
  SourceRange ConditionSrcRange;
  /// A source location for referring to increment later.
  SourceRange IncrementSrcRange;
  /// Loop variable.
  ValueDecl *LCDecl = nullptr;
  /// Reference to loop variable.
  Expr *LCRef = nullptr;
  /// Lower bound (initializer for the var).
  Expr *LB = nullptr;
  /// Upper bound.
  Expr *UB = nullptr;
  /// Loop step (increment).
  Expr *Step = nullptr;
  /// This flag is true when condition is one of:
  ///   Var <  UB
  ///   Var <= UB
  ///   UB  >  Var
  ///   UB  >= Var
  /// This will have no value when the condition is !=
  std::optional<bool> TestIsLessOp;
  /// This flag is true when condition is strict ( < or > ).
  bool TestIsStrictOp = false;
  /// This flag is true when step is subtracted on each iteration.
  bool SubtractStep = false;
  /// Checks if the provide statement depends on the loop counter.
  std::optional<unsigned> doesDependOnLoopCounter(const Stmt *S,
                                                  unsigned DiagKindSelect);
  /// Original condition required for checking of the exit condition for
  /// non-rectangular loop.
  Expr *Condition = nullptr;
  /// This flag is true when Init is a DeclStmt for(<type> var = LB)
  bool LCDeclDeclaredByInit = false;

public:
  RippleIterationSpaceChecker(Sema &SemaRef, SourceLocation DefaultLoc)
      : SemaRef(SemaRef), DefaultLoc(DefaultLoc), ConditionLoc(DefaultLoc) {}
  /// Check init-expr for canonical loop form and save loop counter
  /// variable - #Var and its initialization value - #LB.
  bool checkAndSetInit(Stmt *S, bool EmitDiags = true);
  /// Check test-expr for canonical form, save upper-bound (#UB), flags
  /// for less/greater and for strict/non-strict comparison.
  bool checkAndSetCond(Expr *S);
  /// Check incr-expr for canonical loop form and return true if it
  /// does not conform, otherwise save loop step (#Step).
  bool checkAndSetInc(Expr *S);
  /// Return the loop counter variable.
  ValueDecl *getLoopDecl() const { return LCDecl; }
  /// Return the reference expression to loop counter variable.
  Expr *getLoopDeclRefExpr() const { return LCRef; }
  /// Source range of the loop init.
  SourceRange getInitSrcRange() const { return InitSrcRange; }
  /// Source range of the loop condition.
  SourceRange getConditionSrcRange() const { return ConditionSrcRange; }
  /// Source range of the loop increment.
  SourceRange getIncrementSrcRange() const { return IncrementSrcRange; }
  /// True if the step should be subtracted.
  bool shouldSubtractStep() const { return SubtractStep; }
  /// True, if the compare operator is strict (<, > or !=).
  bool isStrictTestOp() const { return TestIsStrictOp; }
  /// Build the expression to calculate the number of iterations.
  Expr *buildNumIterations(Scope *S) const;
  /// Build the precondition expression for the loops.
  Expr *buildPreCond(Scope *S, Expr *Cond) const;
  /// Build reference expression to the counter be used for codegen.
  DeclRefExpr *buildCounterVar() const;
  /// Build reference expression to the private counter be used for
  /// codegen.
  Expr *buildPrivateCounterVar() const;
  /// Build initialization of the counter be used for codegen.
  Expr *buildCounterInit() const;
  /// Build step of the counter be used for codegen.
  Expr *buildCounterStep() const;
  /// Build loop data with counter value for depend clauses in ordered
  /// directives.
  Expr *
  buildOrderedLoopData(Scope *S, Expr *Counter,
                       llvm::MapVector<const Expr *, DeclRefExpr *> &Captures,
                       SourceLocation Loc, Expr *Inc = nullptr,
                       OverloadedOperatorKind OOK = OO_Amp);
  /// Return true if any expression is dependent.
  bool dependent() const;
  /// Return true if the init is a DeclStmt
  bool hasDeclStmtInit() const;

private:
  /// Check the right-hand side of an assignment in the increment
  /// expression.
  bool checkAndSetIncRHS(Expr *RHS);
  /// Helper to set loop counter variable and its initializer.
  bool setLCDeclAndLB(ValueDecl *NewLCDecl, Expr *NewDeclRefExpr, Expr *NewLB,
                      bool EmitDiags);
  /// Helper to set upper bound.
  bool setUB(Expr *NewUB, std::optional<bool> LessOp, bool StrictOp,
             SourceRange SR, SourceLocation SL);
  /// Helper to set loop increment.
  bool setStep(Expr *NewStep, bool Subtract);
};

} // namespace

static const Expr *getExprAsWritten(const Expr *E) {
  if (const auto *FE = dyn_cast<FullExpr>(E))
    E = FE->getSubExpr();

  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
    E = MTE->getSubExpr();

  while (const auto *Binder = dyn_cast<CXXBindTemporaryExpr>(E))
    E = Binder->getSubExpr();

  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExprAsWritten();
  return E->IgnoreParens();
}

static Expr *getExprAsWritten(Expr *E) {
  return const_cast<Expr *>(getExprAsWritten(const_cast<const Expr *>(E)));
}

bool RippleIterationSpaceChecker::dependent() const {
  if (!LCDecl) {
    assert(!LB && !UB && !Step);
    return false;
  }
  return LCDecl->getType()->isDependentType() ||
         (LB && LB->isValueDependent()) || (UB && UB->isValueDependent()) ||
         (Step && Step->isValueDependent());
}

bool RippleIterationSpaceChecker::hasDeclStmtInit() const {
  return LCDeclDeclaredByInit;
}

bool RippleIterationSpaceChecker::setUB(Expr *NewUB, std::optional<bool> LessOp,
                                        bool StrictOp, SourceRange SR,
                                        SourceLocation SL) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && UB == nullptr &&
         Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewUB || NewUB->containsErrors())
    return true;
  UB = NewUB;
  if (LessOp)
    TestIsLessOp = LessOp;
  TestIsStrictOp = StrictOp;
  ConditionSrcRange = SR;
  ConditionLoc = SL;
  doesDependOnLoopCounter(UB, /*Cond = 1*/ 1);
  return false;
}

static DeclRefExpr *buildDeclRefExpr(Sema &S, ValueDecl *D, QualType Ty,
                                     SourceLocation Loc,
                                     bool RefersToCapture = false) {
  D->setReferenced();
  D->markUsed(S.Context);
  return DeclRefExpr::Create(S.getASTContext(), NestedNameSpecifierLoc(),
                             SourceLocation(), D, RefersToCapture, Loc, Ty,
                             VK_LValue);
}

ExprResult
SemaRipple::PerformRippleImplicitIntegerConversion(SourceLocation Loc,
                                                   Expr *Op) {
  // We can reuse the OpenMP method here since we follow the same semantics and
  // the error/note diagnostics messages are not OpenMP specific.
  return SemaRef.OpenMP().PerformOpenMPImplicitIntegerConversion(
      Loc, getExprAsWritten(Op));
}

bool RippleIterationSpaceChecker::setStep(Expr *NewStep, bool Subtract) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl != nullptr && LB != nullptr && Step == nullptr);
  if (!NewStep || NewStep->containsErrors())
    return true;
  if (!NewStep->isValueDependent()) {
    // Check that the step is integer expression.
    SourceLocation StepLoc = NewStep->getBeginLoc();
    ExprResult Val = SemaRef.Ripple().PerformRippleImplicitIntegerConversion(
        StepLoc, getExprAsWritten(NewStep));
    if (Val.isInvalid())
      return true;
    NewStep = Val.get();

    // Ripple Canonical Loop Form Restrictions (following OpenMP 2.6 spec)
    //  If test-expr is of form var relational-op b and relational-op is < or
    //  <= then incr-expr must cause var to increase on each iteration of the
    //  loop. If test-expr is of form var relational-op b and relational-op is
    //  > or >= then incr-expr must cause var to decrease on each iteration of
    //  the loop.
    //  If test-expr is of form b relational-op var and relational-op is < or
    //  <= then incr-expr must cause var to decrease on each iteration of the
    //  loop. If test-expr is of form b relational-op var and relational-op is
    //  > or >= then incr-expr must cause var to increase on each iteration of
    //  the loop.
    std::optional<llvm::APSInt> Result =
        NewStep->getIntegerConstantExpr(SemaRef.Context);
    bool IsUnsigned = !NewStep->getType()->hasSignedIntegerRepresentation();
    bool IsConstNeg =
        Result && Result->isSigned() && (Subtract != Result->isNegative());
    bool IsConstPos =
        Result && Result->isSigned() && (Subtract == Result->isNegative());
    bool IsConstZero = Result && !Result->getBoolValue();

    // != with increment is treated as <; != with decrement is treated as >
    if (!TestIsLessOp)
      TestIsLessOp = IsConstPos || (IsUnsigned && !Subtract);
    if (UB && (IsConstZero ||
               (*TestIsLessOp ? (IsConstNeg || (IsUnsigned && Subtract))
                              : (IsConstPos || (IsUnsigned && !Subtract))))) {
      SemaRef.Diag(NewStep->getExprLoc(),
                   diag::err_ripple_loop_incr_not_compatible)
          << LCDecl << *TestIsLessOp << NewStep->getSourceRange();
      // Reuse OMP diagnostic
      SemaRef.Diag(ConditionLoc,
                   diag::note_omp_loop_cond_requires_compatible_incr)
          << *TestIsLessOp << ConditionSrcRange;
      return true;
    }
    if (*TestIsLessOp == Subtract) {
      NewStep =
          SemaRef.CreateBuiltinUnaryOp(NewStep->getExprLoc(), UO_Minus, NewStep)
              .get();
      Subtract = !Subtract;
    }
  }

  Step = NewStep;
  SubtractStep = Subtract;
  // Ripple does not capture, so we have to check that the step does not depend
  // on the loop IV
  return doesDependOnLoopCounter(Step, /*Inc = 2*/ 2).value_or(0);
}

static bool checkRippleIterationSpace(ForStmt *For, Sema &SemaRef,
                                      LoopIterationSpace &ResultIterSpaces) {
  // We don't need a scope for expressions used by the Ripple compute construct:
  // we are only building expressions from the already well formed ForSmt.
  Scope *S = nullptr;
  // Canonical Loop Form
  //   for (init-expr; test-expr; incr-expr) structured-block
  RippleIterationSpaceChecker ISC(SemaRef, For->getForLoc());

  // Check init.
  Stmt *Init = For->getInit();
  if (ISC.checkAndSetInit(Init))
    return true;

  bool HasErrors = false;

  // Check loop variable's type.
  if (ValueDecl *LCDecl = ISC.getLoopDecl()) {
    // Ripple [2.6, Canonical Loop Form]
    // Var is one of the following:
    //   A variable of signed or unsigned integer type.
    QualType VarType = LCDecl->getType().getNonReferenceType();
    if (!VarType->isDependentType() && !VarType->isIntegerType() &&
        !VarType->isPointerType() &&
        !(SemaRef.getLangOpts().CPlusPlus && VarType->isOverloadableType())) {
      // Reuse OMP diagnostic
      SemaRef.Diag(Init->getBeginLoc(), diag::err_omp_loop_variable_type)
          << false;
      HasErrors = true;
    }

    // Check test-expr.
    HasErrors |= ISC.checkAndSetCond(For->getCond());

    // Check incr-expr.
    HasErrors |= ISC.checkAndSetInc(For->getInc());
  }

  if (ISC.dependent() || SemaRef.CurContext->isDependentContext() || HasErrors)
    return HasErrors;

  // Build the loop's iteration space representation.
  ResultIterSpaces.PreCond = ISC.buildPreCond(S, For->getCond());
  ResultIterSpaces.NumIterations = ISC.buildNumIterations(S);
  ResultIterSpaces.CounterVar = ISC.buildCounterVar();
  ResultIterSpaces.PrivateCounterVar = ISC.buildPrivateCounterVar();
  ResultIterSpaces.CounterInit = ISC.buildCounterInit();
  ResultIterSpaces.CounterStep = ISC.buildCounterStep();
  ResultIterSpaces.InitSrcRange = ISC.getInitSrcRange();
  ResultIterSpaces.CondSrcRange = ISC.getConditionSrcRange();
  ResultIterSpaces.IncSrcRange = ISC.getIncrementSrcRange();
  ResultIterSpaces.Subtract = ISC.shouldSubtractStep();
  ResultIterSpaces.IsStrictCompare = ISC.isStrictTestOp();
  ResultIterSpaces.IsDeclStmtInit = ISC.hasDeclStmtInit();

  HasErrors |= (ResultIterSpaces.PreCond == nullptr ||
                ResultIterSpaces.NumIterations == nullptr ||
                ResultIterSpaces.CounterVar == nullptr ||
                ResultIterSpaces.PrivateCounterVar == nullptr ||
                ResultIterSpaces.CounterInit == nullptr ||
                ResultIterSpaces.CounterStep == nullptr);

  return HasErrors;
}

bool RippleIterationSpaceChecker::checkAndSetInit(Stmt *S, bool EmitDiags) {
  // Check init-expr for canonical loop form and save loop counter
  // variable - #Var and its initialization value - #LB.
  // Ripple Canonical loop form following OpenMP spec [2.6]
  // init-expr may be one of the following:
  //   var = lb
  //   integer-type var = lb
  //   random-access-iterator-type var = lb
  //   pointer-type var = lb
  //
  if (!S) {
    if (EmitDiags) {
      SemaRef.Diag(DefaultLoc, diag::err_ripple_loop_not_canonical_init);
    }
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  InitSrcRange = S->getSourceRange();
  if (Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign) {
      Expr *LHS = BO->getLHS()->IgnoreParens();
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        return setLCDeclAndLB(DRE->getDecl(), DRE, BO->getRHS(), EmitDiags);
      }
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                EmitDiags);
      }
    }
  } else if (auto *DS = dyn_cast<DeclStmt>(S)) {
    if (DS->isSingleDecl()) {
      if (auto *Var = dyn_cast_or_null<VarDecl>(DS->getSingleDecl())) {
        if (Var->hasInit() && !Var->getType()->isReferenceType()) {
          // Accept non-canonical init form here but emit ext. warning.
          if (Var->getInitStyle() != VarDecl::CInit && EmitDiags)
            SemaRef.Diag(S->getBeginLoc(),
                         diag::ext_ripple_loop_not_canonical_init)
                << S->getSourceRange();
          LCDeclDeclaredByInit = true;
          return setLCDeclAndLB(
              Var,
              buildDeclRefExpr(SemaRef, Var,
                               Var->getType().getNonReferenceType(),
                               DS->getBeginLoc()),
              Var->getInit(), EmitDiags);
        }
      }
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getOperator() == OO_Equal) {
      Expr *LHS = CE->getArg(0);
      if (auto *ME = dyn_cast<MemberExpr>(LHS)) {
        if (ME->isArrow() &&
            isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
          return setLCDeclAndLB(ME->getMemberDecl(), ME, BO->getRHS(),
                                EmitDiags);
      }
    }
  }

  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  if (EmitDiags)
    SemaRef.Diag(S->getBeginLoc(), diag::err_ripple_loop_not_canonical_init)
        << S->getSourceRange();
  return true;
}

static const ValueDecl *getCanonicalDecl(const ValueDecl *D) {
  const auto *VD = dyn_cast<VarDecl>(D);
  const auto *FD = dyn_cast<FieldDecl>(D);
  if (VD != nullptr) {
    VD = VD->getCanonicalDecl();
    D = VD;
  } else {
    assert(FD);
    FD = FD->getCanonicalDecl();
    D = FD;
  }
  return D;
}

static ValueDecl *getCanonicalDecl(ValueDecl *D) {
  return const_cast<ValueDecl *>(
      getCanonicalDecl(const_cast<const ValueDecl *>(D)));
}

namespace {
/// Checker for the non-rectangular loops. Checks if the initializer or
/// condition expression references loop counter variable.
class LoopCounterRefChecker final
    : public ConstStmtVisitor<LoopCounterRefChecker, bool> {
private:
  Sema &SemaRef;
  const ValueDecl *CurLCDecl = nullptr;
  unsigned DiagKindSelect = 0;
  unsigned BaseLoopId = 0;
  bool checkDecl(const Expr *E, const ValueDecl *VD) {
    if (getCanonicalDecl(VD) == getCanonicalDecl(CurLCDecl)) {
      SemaRef.Diag(E->getExprLoc(),
                   diag::err_ripple_stmt_depends_on_loop_counter)
          << DiagKindSelect;
      return false;
    }
    return true;
  }

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const ValueDecl *VD = E->getDecl();
    if (isa<VarDecl>(VD))
      return checkDecl(E, VD);
    return false;
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (isa<CXXThisExpr>(E->getBase()->IgnoreParens())) {
      const ValueDecl *VD = E->getMemberDecl();
      if (isa<VarDecl>(VD) || isa<FieldDecl>(VD))
        return checkDecl(E, VD);
    }
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    bool Res = false;
    for (const Stmt *Child : S->children())
      Res = (Child && Visit(Child)) || Res;
    return Res;
  }
  explicit LoopCounterRefChecker(Sema &SemaRef, const ValueDecl *CurLCDecl,
                                 unsigned DiagKindSelect)
      : SemaRef(SemaRef), CurLCDecl(CurLCDecl), DiagKindSelect(DiagKindSelect) {
  }
  unsigned getBaseLoopId() const {
    assert(CurLCDecl && "Expected loop dependency.");
    return BaseLoopId;
  }
};
} // namespace

std::optional<unsigned>
RippleIterationSpaceChecker::doesDependOnLoopCounter(const Stmt *S,
                                                     unsigned DiagKindSelect) {
  // Check for the non-rectangular loops.
  LoopCounterRefChecker LoopStmtChecker(SemaRef, LCDecl, DiagKindSelect);
  if (LoopStmtChecker.Visit(S)) {
    return LoopStmtChecker.getBaseLoopId();
  }
  return std::nullopt;
}

bool RippleIterationSpaceChecker::setLCDeclAndLB(ValueDecl *NewLCDecl,
                                                 Expr *NewLCRefExpr,
                                                 Expr *NewLB, bool EmitDiags) {
  // State consistency checking to ensure correct usage.
  assert(LCDecl == nullptr && LB == nullptr && LCRef == nullptr &&
         UB == nullptr && Step == nullptr && !TestIsLessOp && !TestIsStrictOp);
  if (!NewLCDecl || !NewLB || NewLB->containsErrors())
    return true;
  LCDecl = getCanonicalDecl(NewLCDecl);
  LCRef = NewLCRefExpr;
  if (auto *CE = dyn_cast_or_null<CXXConstructExpr>(NewLB))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        NewLB = CE->getArg(0)->IgnoreParenImpCasts();
  LB = NewLB;
  if (EmitDiags)
    doesDependOnLoopCounter(LB, /*Init = 0*/ 0);
  return false;
}

DeclRefExpr *RippleIterationSpaceChecker::buildCounterVar() const {
  return cast_if_present<DeclRefExpr>(LCRef);
}

static Expr *calculateNumIters(Sema &SemaRef, Scope *S,
                               SourceLocation DefaultLoc, Expr *Lower,
                               Expr *Upper, Expr *Step, QualType LCTy,
                               bool TestIsStrictOp, bool RoundToStep) {
  ExprResult NewStep = Step;
  if (!NewStep.isUsable())
    return nullptr;
  llvm::APSInt LRes, SRes;
  bool IsLowerConst = false, IsStepConst = false;
  if (std::optional<llvm::APSInt> Res =
          Lower->getIntegerConstantExpr(SemaRef.Context)) {
    LRes = *Res;
    IsLowerConst = true;
  }
  if (std::optional<llvm::APSInt> Res =
          Step->getIntegerConstantExpr(SemaRef.Context)) {
    SRes = *Res;
    IsStepConst = true;
  }
  bool NoNeedToConvert = IsLowerConst && !RoundToStep &&
                         ((!TestIsStrictOp && LRes.isNonNegative()) ||
                          (TestIsStrictOp && LRes.isStrictlyPositive()));
  bool NeedToReorganize = false;
  // Check if any subexpressions in Lower -Step [+ 1] lead to overflow.
  if (!NoNeedToConvert && IsLowerConst &&
      (TestIsStrictOp || (RoundToStep && IsStepConst))) {
    NoNeedToConvert = true;
    if (RoundToStep) {
      unsigned BW = LRes.getBitWidth() > SRes.getBitWidth()
                        ? LRes.getBitWidth()
                        : SRes.getBitWidth();
      LRes = LRes.extend(BW + 1);
      LRes.setIsSigned(true);
      SRes = SRes.extend(BW + 1);
      SRes.setIsSigned(true);
      LRes -= SRes;
      NoNeedToConvert = LRes.trunc(BW).extend(BW + 1) == LRes;
      LRes = LRes.trunc(BW);
    }
    if (TestIsStrictOp) {
      unsigned BW = LRes.getBitWidth();
      LRes = LRes.extend(BW + 1);
      LRes.setIsSigned(true);
      ++LRes;
      NoNeedToConvert =
          NoNeedToConvert && LRes.trunc(BW).extend(BW + 1) == LRes;
      // truncate to the original bitwidth.
      LRes = LRes.trunc(BW);
    }
    NeedToReorganize = NoNeedToConvert;
  }
  llvm::APSInt URes;
  bool IsUpperConst = false;
  if (std::optional<llvm::APSInt> Res =
          Upper->getIntegerConstantExpr(SemaRef.Context)) {
    URes = *Res;
    IsUpperConst = true;
  }
  if (NoNeedToConvert && IsLowerConst && IsUpperConst &&
      (!RoundToStep || IsStepConst)) {
    unsigned BW = LRes.getBitWidth() > URes.getBitWidth() ? LRes.getBitWidth()
                                                          : URes.getBitWidth();
    LRes = LRes.extend(BW + 1);
    LRes.setIsSigned(true);
    URes = URes.extend(BW + 1);
    URes.setIsSigned(true);
    URes -= LRes;
    NoNeedToConvert = URes.trunc(BW).extend(BW + 1) == URes;
    NeedToReorganize = NoNeedToConvert;
  }
  // If the boundaries are not constant or (Lower - Step [+ 1]) is not constant
  // or less than zero (Upper - (Lower - Step [+ 1]) may overflow) - promote to
  // unsigned.
  if ((!NoNeedToConvert || (LRes.isNegative() && !IsUpperConst)) &&
      !LCTy->isDependentType() && LCTy->isIntegerType()) {
    QualType LowerTy = Lower->getType();
    QualType UpperTy = Upper->getType();
    uint64_t LowerSize = SemaRef.Context.getTypeSize(LowerTy);
    uint64_t UpperSize = SemaRef.Context.getTypeSize(UpperTy);
    if ((LowerSize <= UpperSize && UpperTy->hasSignedIntegerRepresentation()) ||
        (LowerSize > UpperSize && LowerTy->hasSignedIntegerRepresentation())) {
      QualType CastType = SemaRef.Context.getIntTypeForBitwidth(
          LowerSize > UpperSize ? LowerSize : UpperSize, /*Signed=*/0);
      Upper =
          SemaRef
              .PerformImplicitConversion(
                  SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Upper).get(),
                  CastType, AssignmentAction::Converting)
              .get();
      Lower = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Lower).get();
      NewStep = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, NewStep.get());
    }
  }
  if (!Lower || !Upper || NewStep.isInvalid())
    return nullptr;

  ExprResult Diff;
  // If need to reorganize, then calculate the form as Upper - (Lower - Step [+
  // 1]).
  if (NeedToReorganize) {
    Diff = Lower;

    if (RoundToStep) {
      // Lower - Step
      Diff =
          SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Diff.get(), NewStep.get());
      if (!Diff.isUsable())
        return nullptr;
    }

    // Lower - Step [+ 1]
    if (TestIsStrictOp)
      Diff = SemaRef.BuildBinOp(
          S, DefaultLoc, BO_Add, Diff.get(),
          SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!Diff.isUsable())
      return nullptr;

    Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
    if (!Diff.isUsable())
      return nullptr;

    // Upper - (Lower - Step [+ 1]).
    Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Upper, Diff.get());
    if (!Diff.isUsable())
      return nullptr;
  } else {
    Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Sub, Upper, Lower);

    if (!Diff.isUsable() && LCTy->getAsCXXRecordDecl()) {
      // BuildBinOp already emitted error, this one is to point user to upper
      // and lower bound, and to tell what is passed to 'operator-'.
      // This diagnostic is not OpenMP specific
      SemaRef.Diag(Upper->getBeginLoc(), diag::err_omp_loop_diff_cxx)
          << Upper->getSourceRange() << Lower->getSourceRange();
      return nullptr;
    }

    if (!Diff.isUsable())
      return nullptr;

    // Upper - Lower [- 1]
    if (TestIsStrictOp)
      Diff = SemaRef.BuildBinOp(
          S, DefaultLoc, BO_Sub, Diff.get(),
          SemaRef.ActOnIntegerConstant(SourceLocation(), 1).get());
    if (!Diff.isUsable())
      return nullptr;

    if (RoundToStep) {
      // Upper - Lower [- 1] + Step
      Diff =
          SemaRef.BuildBinOp(S, DefaultLoc, BO_Add, Diff.get(), NewStep.get());
      if (!Diff.isUsable())
        return nullptr;
    }
  }

  // Parentheses (for dumping/debugging purposes only).
  Diff = SemaRef.ActOnParenExpr(DefaultLoc, DefaultLoc, Diff.get());
  if (!Diff.isUsable())
    return nullptr;

  // (Upper - Lower [- 1] + Step) / Step or (Upper - Lower) / Step
  Diff = SemaRef.BuildBinOp(S, DefaultLoc, BO_Div, Diff.get(), NewStep.get());
  if (!Diff.isUsable())
    return nullptr;

  return Diff.get();
}

/// Ignore parenthesizes, implicit casts, copy constructor and return the
/// variable (which may be the loop variable) if possible.
static const ValueDecl *getInitLCDecl(const Expr *E) {
  if (!E)
    return nullptr;
  E = getExprAsWritten(E);
  if (const auto *CE = dyn_cast_or_null<CXXConstructExpr>(E))
    if (const CXXConstructorDecl *Ctor = CE->getConstructor())
      if ((Ctor->isCopyOrMoveConstructor() ||
           Ctor->isConvertingConstructor(/*AllowExplicit=*/false)) &&
          CE->getNumArgs() > 0 && CE->getArg(0) != nullptr)
        E = CE->getArg(0)->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast_or_null<DeclRefExpr>(E)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return getCanonicalDecl(VD);
  }
  if (const auto *ME = dyn_cast_or_null<MemberExpr>(E))
    if (ME->isArrow() && isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
      return getCanonicalDecl(ME->getMemberDecl());
  return nullptr;
}

bool RippleIterationSpaceChecker::checkAndSetCond(Expr *S) {
  // Check test-expr for canonical form, save upper-bound UB, flags for
  // less/greater and for strict/non-strict comparison.
  // Ripple Canonical loop form. Following OpenMP spec [2.9]
  // Test-expr may be one of the following:
  //   var relational-op b
  //   b relational-op var
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_ripple_loop_not_canonical_cond)
        << LCDecl;
    return true;
  }
  Condition = S;
  S = getExprAsWritten(S);
  SourceLocation CondLoc = S->getBeginLoc();
  auto &&CheckAndSetCond = [this](BinaryOperatorKind Opcode, const Expr *LHS,
                                  const Expr *RHS, SourceRange SR,
                                  SourceLocation OpLoc) -> std::optional<bool> {
    if (BinaryOperator::isRelationalOp(Opcode)) {
      if (getInitLCDecl(LHS) == LCDecl)
        return setUB(const_cast<Expr *>(RHS),
                     (Opcode == BO_LT || Opcode == BO_LE),
                     (Opcode == BO_LT || Opcode == BO_GT), SR, OpLoc);
      if (getInitLCDecl(RHS) == LCDecl)
        return setUB(const_cast<Expr *>(LHS),
                     (Opcode == BO_GT || Opcode == BO_GE),
                     (Opcode == BO_LT || Opcode == BO_GT), SR, OpLoc);
    } else if (Opcode == BO_NE) {
      return setUB(const_cast<Expr *>(getInitLCDecl(LHS) == LCDecl ? RHS : LHS),
                   /*LessOp=*/std::nullopt,
                   /*StrictOp=*/true, SR, OpLoc);
    }
    return std::nullopt;
  };
  std::optional<bool> Res;
  if (auto *RBO = dyn_cast<CXXRewrittenBinaryOperator>(S)) {
    CXXRewrittenBinaryOperator::DecomposedForm DF = RBO->getDecomposedForm();
    Res = CheckAndSetCond(DF.Opcode, DF.LHS, DF.RHS, RBO->getSourceRange(),
                          RBO->getOperatorLoc());
  } else if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    Res = CheckAndSetCond(BO->getOpcode(), BO->getLHS(), BO->getRHS(),
                          BO->getSourceRange(), BO->getOperatorLoc());
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (CE->getNumArgs() == 2) {
      Res = CheckAndSetCond(
          BinaryOperator::getOverloadedOpcode(CE->getOperator()), CE->getArg(0),
          CE->getArg(1), CE->getSourceRange(), CE->getOperatorLoc());
    }
  }
  if (Res)
    return *Res;
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(CondLoc, diag::err_ripple_loop_not_canonical_cond)
      << S->getSourceRange() << LCDecl;
  return true;
}

bool RippleIterationSpaceChecker::checkAndSetIncRHS(Expr *RHS) {
  // RHS of canonical loop form increment can be:
  //   var + incr
  //   incr + var
  //   var - incr
  //
  RHS = RHS->IgnoreParenImpCasts();
  if (auto *BO = dyn_cast<BinaryOperator>(RHS)) {
    if (BO->isAdditiveOp()) {
      bool IsAdd = BO->getOpcode() == BO_Add;
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), !IsAdd);
      if (IsAdd && getInitLCDecl(BO->getRHS()) == LCDecl)
        return setStep(BO->getLHS(), /*Subtract=*/false);
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    bool IsAdd = CE->getOperator() == OO_Plus;
    if ((IsAdd || CE->getOperator() == OO_Minus) && CE->getNumArgs() == 2) {
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), !IsAdd);
      if (IsAdd && getInitLCDecl(CE->getArg(1)) == LCDecl)
        return setStep(CE->getArg(0), /*Subtract=*/false);
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(RHS->getBeginLoc(), diag::err_ripple_loop_not_canonical_incr)
      << RHS->getSourceRange() << LCDecl;
  return true;
}

bool RippleIterationSpaceChecker::checkAndSetInc(Expr *S) {
  // Check incr-expr for canonical loop form and return true if it
  // does not conform.
  // Ripple Canonical loop form. Following the OpenMP spec [2.6]
  // Test-expr may be one of the following:
  //   ++var
  //   var++
  //   --var
  //   var--
  //   var += incr
  //   var -= incr
  //   var = var + incr
  //   var = incr + var
  //   var = var - incr
  //
  if (!S) {
    SemaRef.Diag(DefaultLoc, diag::err_ripple_loop_not_canonical_incr)
        << LCDecl;
    return true;
  }
  if (auto *ExprTemp = dyn_cast<ExprWithCleanups>(S))
    if (!ExprTemp->cleanupsHaveSideEffects())
      S = ExprTemp->getSubExpr();

  IncrementSrcRange = S->getSourceRange();
  S = S->IgnoreParens();
  if (auto *UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementDecrementOp() &&
        getInitLCDecl(UO->getSubExpr()) == LCDecl)
      return setStep(SemaRef
                         .ActOnIntegerConstant(UO->getBeginLoc(),
                                               (UO->isDecrementOp() ? -1 : 1))
                         .get(),
                     /*Subtract=*/false);
  } else if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return setStep(BO->getRHS(), BO->getOpcode() == BO_SubAssign);
      break;
    case BO_Assign:
      if (getInitLCDecl(BO->getLHS()) == LCDecl)
        return checkAndSetIncRHS(BO->getRHS());
      break;
    default:
      break;
    }
  } else if (auto *CE = dyn_cast<CXXOperatorCallExpr>(S)) {
    switch (CE->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(SemaRef
                           .ActOnIntegerConstant(
                               CE->getBeginLoc(),
                               ((CE->getOperator() == OO_MinusMinus) ? -1 : 1))
                           .get(),
                       /*Subtract=*/false);
      break;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return setStep(CE->getArg(1), CE->getOperator() == OO_MinusEqual);
      break;
    case OO_Equal:
      if (getInitLCDecl(CE->getArg(0)) == LCDecl)
        return checkAndSetIncRHS(CE->getArg(1));
      break;
    default:
      break;
    }
  }
  if (dependent() || SemaRef.CurContext->isDependentContext())
    return false;
  SemaRef.Diag(S->getBeginLoc(), diag::err_ripple_loop_not_canonical_incr)
      << S->getSourceRange() << LCDecl;
  return true;
}

Expr *RippleIterationSpaceChecker::buildCounterInit() const { return LB; }

Expr *RippleIterationSpaceChecker::buildCounterStep() const { return Step; }

Expr *RippleIterationSpaceChecker::buildPreCond(Scope *S, Expr *Cond) const {

  // Try to build LB <op> UB, where <op> is <, >, <=, or >=.
  Sema::TentativeAnalysisScope Trap(SemaRef);

  ExprResult NewLB = LB;
  ExprResult NewUB = UB;
  if (!NewLB.isUsable() || !NewUB.isUsable())
    return nullptr;

  ExprResult CondExpr =
      SemaRef.BuildBinOp(S, DefaultLoc,
                         *TestIsLessOp ? (TestIsStrictOp ? BO_LT : BO_LE)
                                       : (TestIsStrictOp ? BO_GT : BO_GE),
                         NewLB.get(), NewUB.get());
  if (CondExpr.isUsable()) {
    if (!SemaRef.Context.hasSameUnqualifiedType(CondExpr.get()->getType(),
                                                SemaRef.Context.BoolTy))
      CondExpr = SemaRef.PerformImplicitConversion(
          CondExpr.get(), SemaRef.Context.BoolTy,
          /*Action=*/AssignmentAction::Converting,
          /*AllowExplicit=*/true);
  }

  // Otherwise use original loop condition and evaluate it in runtime.
  return CondExpr.isUsable() ? CondExpr.get() : Cond;
}

Expr *RippleIterationSpaceChecker::buildNumIterations(Scope *S) const {
  QualType VarType = LCDecl->getType().getNonReferenceType();
  if (!VarType->isIntegerType() && !VarType->isPointerType() &&
      !SemaRef.getLangOpts().CPlusPlus)
    return nullptr;
  Expr *LBVal = LB;
  Expr *UBVal = UB;
  Expr *UBExpr = *TestIsLessOp ? UBVal : LBVal;
  Expr *LBExpr = *TestIsLessOp ? LBVal : UBVal;
  Expr *Upper = UBExpr;
  Expr *Lower = LBExpr;
  if (!Upper || !Lower)
    return nullptr;

  ExprResult Diff = calculateNumIters(SemaRef, S, DefaultLoc, Lower, Upper,
                                      Step, VarType, TestIsStrictOp,
                                      /*RoundToStep=*/true);
  if (!Diff.isUsable())
    return nullptr;

  QualType Type = Diff.get()->getType();
  ASTContext &C = SemaRef.Context;
  bool UseVarType = VarType->hasIntegerRepresentation() &&
                    C.getTypeSize(Type) > C.getTypeSize(VarType);
  if (!Type->isIntegerType() || UseVarType) {
    unsigned NewSize =
        UseVarType ? C.getTypeSize(VarType) : C.getTypeSize(Type);
    bool IsSigned = UseVarType ? VarType->hasSignedIntegerRepresentation()
                               : Type->hasSignedIntegerRepresentation();
    Type = C.getIntTypeForBitwidth(NewSize, IsSigned);
    if (!SemaRef.Context.hasSameType(Diff.get()->getType(), Type)) {
      Diff = SemaRef.PerformImplicitConversion(Diff.get(), Type,
                                               AssignmentAction::Converting,
                                               /*AllowExplicit=*/true);
      if (!Diff.isUsable())
        return nullptr;
    }
  }

  return Diff.get();
}

/// Build a variable declaration for Ripple loop iteration variable.
static VarDecl *buildVarDecl(Sema &SemaRef, SourceLocation Loc, QualType Type,
                             StringRef Name, const AttrVec *Attrs = nullptr) {
  DeclContext *DC = SemaRef.CurContext;
  IdentifierInfo *II = &SemaRef.PP.getIdentifierTable().get(Name);
  TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(Type, Loc);
  auto *Decl =
      VarDecl::Create(SemaRef.Context, DC, Loc, Loc, II, Type, TInfo, SC_None);
  if (Attrs) {
    for (specific_attr_iterator<AlignedAttr> I(Attrs->begin()), E(Attrs->end());
         I != E; ++I)
      Decl->addAttr(*I);
  }
  Decl->setImplicit();
  return Decl;
}

Expr *RippleIterationSpaceChecker::buildPrivateCounterVar() const {
  if (LCDecl && !LCDecl->isInvalidDecl()) {
    QualType Type = LCDecl->getType().getNonReferenceType();
    VarDecl *PrivateVar =
        buildVarDecl(SemaRef, DefaultLoc, Type, "ripple.par.iv",
                     LCDecl->hasAttrs() ? &LCDecl->getAttrs() : nullptr);
    if (PrivateVar->isInvalidDecl())
      return nullptr;
    return buildDeclRefExpr(SemaRef, PrivateVar, Type, DefaultLoc);
  }
  return nullptr;
}

//====----------------- End OpenMP copied section ---------------------====//

StmtResult SemaRipple::CreateRippleParallelComputeStmt(
    SourceRange PragmaLoc, SourceRange BlockShapeLoc, SourceRange DimsLoc,
    ValueDecl *BlockShape, ArrayRef<uint64_t> Dims, Stmt *AssociatedStatement,
    bool NoRemainder) {
  auto ForLoop = dyn_cast<ForStmt>(AssociatedStatement);
  if (!ForLoop) {
    Diag(AssociatedStatement->getBeginLoc(),
         diag::err_ripple_loop_not_for_loop);
    Diag(PragmaLoc.getBegin(), diag::note_pragma_entered_here) << PragmaLoc;
    return StmtError();
  }

  auto *RPC = RippleComputeConstruct::Create(
      getASTContext(), PragmaLoc, BlockShapeLoc, DimsLoc, BlockShape, Dims,
      cast<ForStmt>(AssociatedStatement), NoRemainder);

  ActOnDuplicateDimensionIndex(*RPC);
  ActOnRippleComputeConstruct(*RPC);

  // Mark the block shape as being used to avoid warnings about possible unused
  // block shape in dependant contexts (we only generate a declrefexpr in
  // non-dependant contexts)
  BlockShape->markUsed(getASTContext());

  return RPC;
}

namespace {

class LoopBodyCloneChecker final
    : public ConstStmtVisitor<LoopBodyCloneChecker, bool> {
private:
  Sema &SemaRef;
  bool CheckCallExpr(const CallExpr *Call) {
    if (const FunctionDecl *FD = Call->getDirectCallee()) {
      for (const auto *Attr : FD->attrs()) {
        if (isa<NoDuplicateAttr>(Attr)) {
          SemaRef.Diag(Call->getBeginLoc(),
                       diag::err_ripple_loop_body_cannot_contain)
              << "calls to function with the attribute "
                 "'noduplicate'"
              << Call->getSourceRange();
          SemaRef.Diag(FD->getBeginLoc(), diag::note_declared_at);
          return true;
        }
      }
    }
    return false;
  }
  bool CheckLabelStmt(const LabelStmt *Label) {
    SemaRef.Diag(Label->getBeginLoc(),
                 diag::err_ripple_loop_body_cannot_contain)
        << "labels" << Label->getDecl()->getSourceRange();
    return true;
  }

public:
  bool VisitCallExpr(const CallExpr *Call) { return CheckCallExpr(Call); }
  bool VisitLabelStmt(const LabelStmt *Label) { return CheckLabelStmt(Label); }
  bool VisitStmt(const Stmt *S) {
    bool Res = false;
    for (const Stmt *Child : S->children())
      Res = (Child && Visit(Child)) || Res;
    return Res;
  }
  explicit LoopBodyCloneChecker(Sema &SemaRef) : SemaRef(SemaRef) {}
};

bool checkAssociatedLoopBody(const ForStmt *For, Sema &SemaRef) {
  LoopBodyCloneChecker CloneChecker(SemaRef);
  return CloneChecker.VisitStmt(For->getBody());
}

bool ActOnAssociatedLoop(Sema &SemaRef, ForStmt *ForLoop,
                         LoopIterationSpace &LIS) {
  return checkRippleIterationSpace(ForLoop, SemaRef, LIS) ||
         checkAssociatedLoopBody(ForLoop, SemaRef);
}

std::pair<Expr *, Expr *>
createIndexAndSizeExprs(Sema &SemaRef, SourceLocation Loc,
                        const RippleComputeConstruct &S, QualType ExprTypes) {
  QualType SizeTType = SemaRef.Context.getSizeType();
  auto &Context = SemaRef.getASTContext();
  QualType BSType = S.getBlockShape()->getType();
  LLVM_DEBUG(llvm::dbgs() << "Block shape is " << *S.getBlockShape()
                          << " with type " << BSType << "\n");
  auto *BSRef =
      buildDeclRefExpr(SemaRef, S.getBlockShape(), BSType, Loc, false);
  Expr *BlockSizeExpr = ImplicitCastExpr::Create(
      SemaRef.getASTContext(), BSRef->getType(), CK_LValueToRValue, BSRef,
      nullptr, VK_LValue, FPOptionsOverride());
  Expr *BlockSizeExprAsVoidPtr = ImplicitCastExpr::Create(
      Context, Context.VoidPtrTy, CK_BitCast, BlockSizeExpr, nullptr,
      VK_PRValue, FPOptionsOverride());

  ExprResult RippleIndexSum;
  ExprResult RippleSizeProduct;
  for (auto DimId : S.getDimensionIds()) {
    auto DimIdExpr = IntegerLiteral::Create(
        SemaRef.Context,
        llvm::APInt(SemaRef.Context.getTypeSize(SizeTType), DimId), SizeTType,
        Loc);
    SmallVector<Expr *, 2> Args{BlockSizeExprAsVoidPtr, DimIdExpr};
    Expr *CallGetIndex = SemaRef.BuildBuiltinCallExpr(
        Loc, Builtin::ID::BI__builtin_ripple_get_index, Args);
    Expr *CallGetSize = SemaRef.BuildBuiltinCallExpr(
        Loc, Builtin::ID::BI__builtin_ripple_get_size, Args);
    if (RippleIndexSum.isUnset()) {
      RippleIndexSum = CallGetIndex;
    } else {
      // The current index * size of previous dimensions
      Expr *IndexTimesInnerSize =
          SemaRef
              .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Mul,
                                  CallGetIndex, RippleSizeProduct.get())
              .get();
      RippleIndexSum =
          SemaRef.CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Add,
                                     RippleIndexSum.get(), IndexTimesInnerSize);
    }
    if (RippleSizeProduct.isUnset()) {
      RippleSizeProduct = CallGetSize;
    } else {
      RippleSizeProduct =
          SemaRef
              .ActOnParenExpr(
                  Loc, Loc,
                  SemaRef
                      .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Mul,
                                          RippleSizeProduct.get(), CallGetSize)
                      .get())
              .get();
    }
  }
  RippleIndexSum = SemaRef.ActOnParenExpr(Loc, Loc, RippleIndexSum.get());

  assert(
      SemaRef.Context.hasSameType(RippleIndexSum.get()->getType(), SizeTType));
  if (!SemaRef.Context.hasSameType(SizeTType, ExprTypes)) {
    RippleIndexSum = SemaRef.PerformImplicitConversion(
        RippleIndexSum.get(), ExprTypes, AssignmentAction::Converting);
    RippleSizeProduct = SemaRef.PerformImplicitConversion(
        RippleSizeProduct.get(), ExprTypes, AssignmentAction::Converting);
  }
  return {RippleIndexSum.get(), RippleSizeProduct.get()};
}

Expr *createRippleInit(Sema &SemaRef, SourceLocation Loc, Expr *RippleIndex,
                       Expr *LoopInit) {
  QualType T = RippleIndex->getType();
  Expr *RippleInit =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Add, RippleIndex,
                              LoopInit)
          .get();
  VarDecl *ParallelInit =
      buildVarDecl(SemaRef, Loc, T, "ripple.par.init", nullptr);
  assert(!ParallelInit->isInvalidDecl());
  LLVM_DEBUG(llvm::dbgs() << "Ripple parallel init: "; ParallelInit->print(
      llvm::dbgs(), SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << " = "; RippleInit->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  SemaRef.AddInitializerToDecl(ParallelInit, RippleInit, true);
  return buildDeclRefExpr(SemaRef, ParallelInit, T, Loc);
}

Expr *createRippleStep(Sema &SemaRef, SourceLocation Loc, Expr *RippleSize,
                       Expr *LoopStep) {
  QualType T = RippleSize->getType();
  Expr *RippleStep =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Mul, RippleSize,
                              LoopStep)
          .get();
  VarDecl *ParStep = buildVarDecl(SemaRef, Loc, T, "ripple.par.step", nullptr);
  assert(!ParStep->isInvalidDecl());
  LLVM_DEBUG(llvm::dbgs() << "Ripple parallel Step: ";
             ParStep->print(llvm::dbgs(), SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << " = "; RippleStep->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  SemaRef.AddInitializerToDecl(ParStep, RippleStep, true);
  return buildDeclRefExpr(SemaRef, ParStep, T, Loc);
}

Expr *createLoopIVUpdate(Sema &SemaRef, SourceRange Loc, Expr *LoopIV,
                         Expr *RippleIV, Expr *RippleInit, Expr *RippleStep,
                         bool Subtract) {
  // Create an expression:
  // LoopIV = RippleInit +- (RippleStep * RippleIV)
  auto RippleOffset =
      SemaRef
          .ActOnParenExpr(
              Loc.getBegin(), Loc.getEnd(),
              SemaRef
                  .CreateBuiltinBinOp(Loc.getBegin(),
                                      BinaryOperator::Opcode::BO_Mul,
                                      RippleStep, RippleIV)
                  .get())
          .get();
  auto CounterVal =
      SemaRef
          .CreateBuiltinBinOp(Loc.getBegin(),
                              Subtract ? BinaryOperator::Opcode::BO_Sub
                                       : BinaryOperator::Opcode::BO_Add,
                              RippleInit, RippleOffset)
          .get();
  auto LoopIVUpdate =
      SemaRef
          .CreateBuiltinBinOp(Loc.getBegin(), BinaryOperator::Opcode::BO_Assign,
                              LoopIV, CounterVal)
          .get();
  LLVM_DEBUG(llvm::dbgs() << "Loop IV update: "; LoopIVUpdate->printPretty(
      llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  return LoopIVUpdate;
}

Expr *createRippleParallelLoopCond(Sema &SemaRef, SourceLocation Loc, Expr *IV,
                                   Expr *NumIter) {
  auto Cond =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_LT, IV, NumIter)
          .get();
  LLVM_DEBUG(llvm::dbgs() << "Loop condition: "; Cond->printPretty(
      llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  return Cond;
}

Expr *createRippleParallelInc(Sema &SemaRef, SourceLocation Loc, Expr *IV) {
  // Add 1 to the induction variable!
  auto IncExpr =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_AddAssign, IV,
                              SemaRef.ActOnIntegerConstant(Loc, 1).get())
          .get();
  LLVM_DEBUG(llvm::dbgs() << "Ripple Increment: "; IncExpr->printPretty(
      llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  return IncExpr;
}

Expr *createNumIterVar(Sema &SemaRef, SourceLocation Loc, Expr *NumIterations) {
  QualType T = NumIterations->getType();
  VarDecl *PV = buildVarDecl(SemaRef, Loc, T, "ripple.loop.iters", nullptr);
  assert(!PV->isInvalidDecl());

  LLVM_DEBUG(llvm::dbgs() << "Ripple loop iterations: ";
             PV->print(llvm::dbgs(), SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << " = "; NumIterations->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  SemaRef.AddInitializerToDecl(PV, NumIterations, true);
  return buildDeclRefExpr(SemaRef, PV, T, Loc);
}

Expr *createParallelNumIterVar(Sema &SemaRef, SourceLocation Loc,
                               Expr *NumIterations, Expr *RippleSize) {
  QualType T = NumIterations->getType();
  VarDecl *PV = buildVarDecl(SemaRef, Loc, T, "ripple.par.loop.iters", nullptr);
  assert(!PV->isInvalidDecl());
  Expr *NumParallelIters =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Div,
                              NumIterations, RippleSize)
          .get();
  LLVM_DEBUG(llvm::dbgs() << "Num ripple parallel iterations: ";
             PV->print(llvm::dbgs(), SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << " = "; NumParallelIters->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  SemaRef.AddInitializerToDecl(PV, NumParallelIters, true);
  return buildDeclRefExpr(SemaRef, PV, T, Loc);
}

Expr *createParallelBlockSize(Sema &SemaRef, SourceLocation Loc,
                              Expr *RippleSize) {
  QualType T = RippleSize->getType();
  VarDecl *PV = buildVarDecl(SemaRef, Loc, T, "ripple.par.block.size", nullptr);
  assert(!PV->isInvalidDecl());
  LLVM_DEBUG(llvm::dbgs() << "Ripple parallel block size: ";
             PV->print(llvm::dbgs(), SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << " = "; RippleSize->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  SemaRef.AddInitializerToDecl(PV, RippleSize, true);
  return buildDeclRefExpr(SemaRef, PV, T, Loc);
}

Expr *createRippleIV(Sema &SemaRef, Expr *RippleIV) {
  // Start at zero
  LLVM_DEBUG(llvm::dbgs() << "Ripple IV: ";
             cast<DeclRefExpr>(RippleIV)->getDecl()->print(
                 llvm::dbgs(), SemaRef.Context.getPrintingPolicy()));
  SemaRef.AddInitializerToDecl(
      cast<DeclRefExpr>(RippleIV)->getDecl(),
      SemaRef.ActOnIntegerConstant(RippleIV->getBeginLoc(), 0).get(),
      /*DirectInit*/ true);
  LLVM_DEBUG(llvm::dbgs() << " = ";
             cast<VarDecl>(cast<DeclRefExpr>(RippleIV)->getDecl())
                 ->getInit()
                 ->printPretty(llvm::dbgs(), nullptr,
                               SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  return RippleIV;
}

Expr *createRemainderEntryCondition(Sema &SemaRef, SourceLocation Loc,
                                    Expr *RippleIV, Expr *RippleSize,
                                    Expr *LoopIters) {
  Expr *RippleToLoopIter =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Mul, RippleIV,
                              RippleSize)
          .get();
  // RippleToLoopIter != LoopIters
  Expr *EnterRemainderCond =
      SemaRef
          .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_NE,
                              RippleToLoopIter, LoopIters)
          .get();
  LLVM_DEBUG(llvm::dbgs() << "EnterRemainderCond: ";
             EnterRemainderCond->printPretty(
                 llvm::dbgs(), nullptr, SemaRef.Context.getPrintingPolicy());
             llvm::dbgs() << "\n");
  return EnterRemainderCond;
}

Stmt *createRippleLoopBody(Sema &SemaRef, Expr *LoopIVUpdate, Stmt *LoopBody) {
  return SemaRef
      .ActOnCompoundStmt(LoopBody->getBeginLoc(), LoopBody->getEndLoc(),
                         {LoopIVUpdate, LoopBody},
                         /*IsStmtExpr*/ false)
      .get();
}

ForStmt *createRippleLoop(Sema &SemaRef, const ForStmt *InitialFor, Expr *Cond,
                          Expr *Inc, Stmt *Body) {
  return cast_if_present<ForStmt>(
      SemaRef
          .ActOnForStmt(
              InitialFor->getBeginLoc(), InitialFor->getLParenLoc(), nullptr,
              SemaRef.ActOnCondition(nullptr, Cond->getBeginLoc(), Cond,
                                     Sema::ConditionKind::Boolean),
              SemaRef.MakeFullExpr(Inc), InitialFor->getRParenLoc(), Body)
          .get());
}

Expr *createStoreUBtoIV(Sema &SemaRef, SourceLocation Loc, Expr *LoopIV,
                        Expr *IVLowerBound, Expr *NumIterations, Expr *Step) {
  // Create the expression representing IV = LB + Step * NumIterations
  Expr *AllSteps =
      SemaRef
          .ActOnParenExpr(
              Loc, Loc,
              SemaRef
                  .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Mul,
                                      NumIterations, Step)
                  .get())
          .get();
  Expr *UB = SemaRef
                 .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Add,
                                     IVLowerBound, AllSteps)
                 .get();
  return SemaRef
      .CreateBuiltinBinOp(Loc, BinaryOperator::Opcode::BO_Assign, LoopIV, UB)
      .get();
}

VarDecl *newImplicitVarDecl(Sema &SemaRef, VarDecl *VD) {
  auto *NewVD = VarDecl::Create(SemaRef.Context, VD->getDeclContext(),
                                VD->getInnerLocStart(), VD->getLocation(),
                                VD->getIdentifier(), VD->getType(),
                                VD->getTypeSourceInfo(), VD->getStorageClass());
  if (VD->hasAttrs())
    NewVD->setAttrs(VD->getAttrs());
  NewVD->setInitStyle(VD->getInitStyle());

  // VarDeclBitfields
  NewVD->setTSCSpec(VD->getTSCSpec());
  NewVD->setARCPseudoStrong(VD->isARCPseudoStrong());

  // NonParmVarDeclBitfields
  // Cannot be a paramvardecl VD->isThisDeclarationADemotedDefinition()) {
  NewVD->setExceptionVariable(VD->isExceptionVariable());
  NewVD->setNRVOVariable(VD->isNRVOVariable());
  NewVD->setCXXForRangeDecl(VD->isCXXForRangeDecl());
  NewVD->setObjCForDecl(VD->isObjCForDecl());
  if (VD->isInlineSpecified())
    NewVD->setInlineSpecified();
  if (VD->isInline())
    NewVD->setImplicitlyInline();
  NewVD->setConstexpr(VD->isConstexpr());
  NewVD->setInitCapture(VD->isInitCapture());
  NewVD->setPreviousDeclInSameBlockScope(VD->isPreviousDeclInSameBlockScope());
  if (VD->isEscapingByref())
    NewVD->setEscapingByref();
  if (VD->isCXXCondDecl())
    NewVD->setCXXCondDecl();

  NewVD->setImplicit();
  SemaRef.CheckVariableDeclarationType(NewVD);
  return NewVD;
}

class HasParallelIdxMatchingRippleParallel final
    : public ConstStmtVisitor<HasParallelIdxMatchingRippleParallel, bool> {
private:
  Sema &SemaRef;
  const RippleComputeConstruct &RCC;
  bool CheckCallExpr(const CallExpr *Call) {
    if (auto *Fun = Call->getDirectCallee()) {
      if (Fun->getBuiltinID() == Builtin::BI__builtin_ripple_parallel_idx) {
        auto RippleConstructDimIds = RCC.getDimensionIds();
        if (Call->getNumArgs() == RippleConstructDimIds.size() + 1) {
          // Check that we reference the same BlockShape declaration
          auto *BlockShapeExpr = Call->getArg(0)->IgnoreParenImpCasts();
          auto *DRE = dyn_cast<DeclRefExpr>(BlockShapeExpr);
          if (!DRE || DRE->getDecl() != RCC.getBlockShape())
            return false;

          // And that all the ripple_parallel_idx dimensions match the
          // ripple_parallel ones as well
          return all_of_zip(
              RCC.getDimensionIds(),
              make_range(std::next(Call->arg_begin()), Call->arg_end()),
              [this](uint64_t ParallelDim, const Expr *Arg) {
                Expr::EvalResult R;
                // Checked by SemaRipple::CheckBuiltinFunctionCall
                if (!Arg->EvaluateAsInt(R, SemaRef.getASTContext()))
                  return false;
                return R.Val.getInt().getZExtValue() == ParallelDim;
              });
        }
      }
    }
    return false;
  }

public:
  bool VisitCallExpr(const CallExpr *Call) { return CheckCallExpr(Call); }
  bool VisitStmt(const Stmt *S) {
    bool Res = false;
    for (const Stmt *Child : S->children())
      Res = (Child && Visit(Child)) || Res;
    return Res;
  }
  explicit HasParallelIdxMatchingRippleParallel(
      Sema &SemaRef, const RippleComputeConstruct &RCC)
      : SemaRef(SemaRef), RCC(RCC) {}
};

class RippleParallelIdxTransformer final
    : public clang::TreeTransform<RippleParallelIdxTransformer> {
  VarDecl *DeclContainingParallelIdx;
  HasParallelIdxMatchingRippleParallel CallsParallelIdxForRippleParallel;

public:
  RippleParallelIdxTransformer(clang::Sema &SemaRef,
                               const RippleComputeConstruct &RCC,
                               VarDecl *ParallelIdxDecl)
      : TreeTransform<RippleParallelIdxTransformer>(SemaRef),
        DeclContainingParallelIdx(ParallelIdxDecl),
        CallsParallelIdxForRippleParallel(SemaRef, RCC) {}

  bool AlwaysRebuild() { return false; }

  ExprResult TransformCallExpr(CallExpr *Call) {
    if (CallsParallelIdxForRippleParallel.VisitCallExpr(Call)) {
      Expr *ReplacementExpr = buildDeclRefExpr(
          SemaRef, DeclContainingParallelIdx,
          DeclContainingParallelIdx->getType(), Call->getBeginLoc());
      if (!SemaRef.Context.hasSameUnqualifiedType(ReplacementExpr->getType(),
                                                  Call->getType()))
        ReplacementExpr =
            SemaRef
                .PerformImplicitConversion(ReplacementExpr, Call->getType(),
                                           AssignmentAction::Converting)
                .get();
      return ReplacementExpr;
    }
    return TreeTransform::TransformCallExpr(Call);
  }

  Decl *TransformDefinition(SourceLocation Loc, Decl *D) {
    if (auto *VD = dyn_cast<VarDecl>(D)) {
      if (auto *Init = VD->getInit()) {
        // We visit the init first to make sure we'll do a change
        // We need to generate the new vardecl before transforming the init
        // because there may be init self-references to it
        if (CallsParallelIdxForRippleParallel.VisitExpr(Init)) {
          auto *NewVD = newImplicitVarDecl(SemaRef, VD);
          if (auto *LocalD = VD->getPreviousDecl())
            NewVD->setPreviousDecl(cast<VarDecl>(
                TransformDecl(LocalD->getBeginLoc(), cast<Decl>(LocalD))));
          transformedLocalDecl(VD, {NewVD});
          auto NewInit = TransformExpr(Init);
          if (!NewInit.isUsable())
            return nullptr;
          SemaRef.AddInitializerToDecl(NewVD, NewInit.get(),
                                       VD->isDirectInit());
          return NewVD;
        }
      }
    }
    return TreeTransform::TransformDefinition(Loc, D);
  }

  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl,
                                [[maybe_unused]] TypeSourceInfo *Declarator,
                                [[maybe_unused]] SourceLocation StartLoc,
                                [[maybe_unused]] SourceLocation IdLoc,
                                [[maybe_unused]] IdentifierInfo *Id) {
    return ExceptionDecl;
  }
};

// A local transformer to redeclare VarDecl for the remainder iteration of the
// for body
class VarDeclRedeclTransformer final
    : public clang::TreeTransform<VarDeclRedeclTransformer> {

public:
  VarDeclRedeclTransformer(clang::Sema &SemaRef)
      : TreeTransform<VarDeclRedeclTransformer>(SemaRef) {}

  bool AlwaysRebuild() { return false; }

  Decl *TransformDefinition(SourceLocation Loc, Decl *D) {
    if (auto *VD = dyn_cast<VarDecl>(D)) {
      // Cannot be in a DeclStmt but better check!
      assert(!isa<ParmVarDecl>(VD) && !isa<ImplicitParamDecl>(VD));
      LLVM_DEBUG(llvm::dbgs() << "Visiting var decl: "; VD->print(llvm::dbgs());
                 llvm::dbgs() << "\n");

      auto *NewVD = newImplicitVarDecl(SemaRef, VD);
      transformedLocalDecl(VD, {NewVD});
      if (auto *LocalD = VD->getPreviousDecl())
        NewVD->setPreviousDecl(cast<VarDecl>(
            TransformDecl(LocalD->getBeginLoc(), cast<Decl>(LocalD))));
      if (VD->getInit()) {
        auto NewInit = TransformExpr(VD->getInit());
        if (!NewInit.isUsable())
          return nullptr;
        SemaRef.AddInitializerToDecl(NewVD, NewInit.get(), VD->isDirectInit());
      }
      return NewVD;
    }
    return TreeTransform::TransformDefinition(Loc, D);
  }

  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl,
                                TypeSourceInfo *Declarator,
                                SourceLocation StartLoc, SourceLocation IdLoc,
                                IdentifierInfo *Id) {
    VarDecl *Var = getSema().BuildExceptionDeclaration(nullptr, Declarator,
                                                       StartLoc, IdLoc, Id);
    transformedLocalDecl(ExceptionDecl, {Var});
    return Var;
  }
};

} // namespace

void SemaRipple::ActOnDuplicateDimensionIndex(const RippleComputeConstruct &S) {
  llvm::DenseSet<uint64_t> DimensionIds;
  for (auto DimIndex : S.getDimensionIds()) {
    if (DimensionIds.contains(DimIndex)) {
      SemaRef.Diag(S.getDimsRange().getBegin(),
                   diag::err_ripple_duplicate_parallel_index)
          << DimIndex << S.getDimsRange();
    } else
      DimensionIds.insert(DimIndex);
  }
}

bool SemaRipple::CheckHasRippleBlockType(const Expr *E, unsigned BuiltinID) {
  auto *ENoCast = E->IgnoreParenImpCasts();
  bool HasValidRippleBlockShapeType = false;
  QualType PtrTy = ENoCast->getType().getDesugaredType(SemaRef.getASTContext());
  LLVM_DEBUG(
      llvm::dbgs() << "Type of BS\n\tExpr(";
      E->printPretty(llvm::dbgs(), nullptr, SemaRef.getPrintingPolicy());
      llvm::dbgs() << ")\n\tw/o parenthesis and impl casts\n\tExpr(";
      ENoCast->printPretty(llvm::dbgs(), nullptr, SemaRef.getPrintingPolicy());
      llvm::dbgs() << ")\n\tis\n\t" << PtrTy << "\n");
  if (PtrTy->isPointerType())
    if (const RecordType *RT = PtrTy->getPointeeType()->getAs<RecordType>())
      if (RT->getDecl()->getName() == "ripple_block_shape")
        HasValidRippleBlockShapeType = true;
  if (!HasValidRippleBlockShapeType) {
    StringRef OperationName = "ripple constructs";
    switch (BuiltinID) {
    case Builtin::BI__builtin_ripple_get_index:
      OperationName = "ripple_id";
      break;
    case Builtin::BI__builtin_ripple_get_size:
      OperationName = "ripple_get_block_size";
      break;
    case Builtin::BI__builtin_ripple_parallel_idx:
      OperationName = "ripple_parallel_idx";
      break;
    case Builtin::BI__builtin_ripple_broadcast_i8:
    case Builtin::BI__builtin_ripple_broadcast_u8:
    case Builtin::BI__builtin_ripple_broadcast_i16:
    case Builtin::BI__builtin_ripple_broadcast_u16:
    case Builtin::BI__builtin_ripple_broadcast_i32:
    case Builtin::BI__builtin_ripple_broadcast_u32:
    case Builtin::BI__builtin_ripple_broadcast_i64:
    case Builtin::BI__builtin_ripple_broadcast_u64:
    case Builtin::BI__builtin_ripple_broadcast_f16:
    case Builtin::BI__builtin_ripple_broadcast_bf16:
    case Builtin::BI__builtin_ripple_broadcast_f32:
    case Builtin::BI__builtin_ripple_broadcast_f64:
    case Builtin::BI__builtin_ripple_broadcast_p:
      OperationName = "ripple_broadcast";
      break;
    default:
      break;
    }
    SemaRef.Diag(E->getBeginLoc(), diag::err_ripple_block_shape_argument)
        << OperationName << PtrTy;
  }
  return !HasValidRippleBlockShapeType;
}

bool SemaRipple::CheckBuiltinFunctionCall(const FunctionDecl *FDecl,
                                          unsigned BuiltinID,
                                          const CallExpr *RippleBICall) {
  auto &ASTCtx = SemaRef.getASTContext();
  bool FoundErrors = false;
  switch (BuiltinID) {
  default:
    llvm_unreachable("Non-implemented check");
  case Builtin::BI__builtin_ripple_get_index:
  case Builtin::BI__builtin_ripple_get_size:
  case Builtin::BI__builtin_ripple_parallel_idx: {
    auto *BlockShapeArg = RippleBICall->getArg(0);
    if (CheckHasRippleBlockType(BlockShapeArg, BuiltinID))
      FoundErrors = true;

    int ArgNo = 2;
    // Arg
    for (auto *Arg : make_range(std::next(RippleBICall->arg_begin()),
                                RippleBICall->arg_end())) {
      Expr::EvalResult R;
      if (!Arg->getType()->isIntegralType(ASTCtx)) {
        SemaRef.Diag(RippleBICall->getBeginLoc(),
                     diag::err_builtin_invalid_arg_type)
            << ArgNo << /* scalar */ 1 << /* 'integer' ty */ 1 << /* no fp */ 0
            << Arg->getType() << Arg->getSourceRange();
        FoundErrors = true;
      } else {
        if (!Arg->isValueDependent() &&
            !Arg->EvaluateAsInt(R, SemaRef.getASTContext())) {
          SemaRef.Diag(RippleBICall->getBeginLoc(),
                       diag::err_constant_integer_arg_type)
              << FDecl->getName() << Arg->getSourceRange();
          FoundErrors = true;
        }
      }
      ArgNo++;
    }
  } break;
  }
  return FoundErrors;
}

void SemaRipple::ActOnRippleComputeConstruct(RippleComputeConstruct &S) {

  // We only have ripple 'parallel' applying on for loops for now
  LoopIterationSpace LIS;

  if (ActOnAssociatedLoop(SemaRef, S.getAssociatedForStmt(), LIS)) {
    LLVM_DEBUG(llvm::dbgs() << "Ripple parallel has errors!\n");
    return;
  }

  // Fill the expressions needed for codegen
  if (SemaRef.CurContext->isDependentContext() || !LIS.NumIterations)
    return;

  Expr *RippleIdx, *RippleSize;
  std::tie(RippleIdx, RippleSize) = createIndexAndSizeExprs(
      SemaRef, S.getBeginLoc(), S, LIS.NumIterations->getType());

  S.setLoopInit(createRippleIV(SemaRef, LIS.PrivateCounterVar));
  S.setParallelBlockSize(createParallelBlockSize(
      SemaRef, S.getDimsRange().getBegin(), RippleSize));
  S.setAssociatedLoopIters(createNumIterVar(
      SemaRef, LIS.InitSrcRange.getBegin(), LIS.NumIterations));
  S.setRippleFullBlockIters(createParallelNumIterVar(
      SemaRef, LIS.InitSrcRange.getBegin(), S.getAssociatedLoopIters(),
      S.getParallelBlockSize()));

  S.setRippleLowerBound(createRippleInit(SemaRef, LIS.InitSrcRange.getBegin(),
                                         RippleIdx, LIS.CounterInit));
  S.setRippleStep(createRippleStep(SemaRef, LIS.InitSrcRange.getBegin(),
                                   S.getParallelBlockSize(), LIS.CounterStep));

  // We only codegen the declaration when the loop init is a DeclStmt
  if (LIS.IsDeclStmtInit)
    S.setLoopIVOrigin(LIS.CounterVar);

  auto *RippleLoopCond = createRippleParallelLoopCond(
      SemaRef, LIS.CondSrcRange.getBegin(), LIS.PrivateCounterVar,
      S.getRippleFullBlockIters());
  auto *RippleLoopInc = createRippleParallelInc(
      SemaRef, LIS.IncSrcRange.getBegin(), LIS.PrivateCounterVar);
  S.setLoopIVUpdate(createLoopIVUpdate(
      SemaRef, LIS.InitSrcRange, LIS.CounterVar, S.getLoopInit(),
      S.getRippleLowerBound(), S.getRippleStep(), LIS.Subtract));

  RippleParallelIdxTransformer RPIT(
      SemaRef, S, cast<VarDecl>(cast<DeclRefExpr>(S.getLoopInit())->getDecl()));
  auto BodyWithRippleParallelIdx =
      RPIT.TransformStmt(S.getAssociatedForStmt()->getBody());
  // There may be recovery expressions (parse errors) that makes this transform
  // invalid, stop here if we cannot transform!
  if (!BodyWithRippleParallelIdx.isUsable())
    return;
  // We can't check at this point that there is no more
  // __builtin_ripple_parallel_idx because enclosing ripple_parallel may exist.
  // Defer the error until codegen.

  auto *RippleLoopBody = createRippleLoopBody(SemaRef, S.getLoopIVUpdate(),
                                              BodyWithRippleParallelIdx.get());

  S.setRippleLoopStmt(createRippleLoop(SemaRef, S.getAssociatedForStmt(),
                                       RippleLoopCond, RippleLoopInc,
                                       RippleLoopBody));

  if (S.generateRemainder()) {
    S.setRemainderRuntimeCond(createRemainderEntryCondition(
        SemaRef, LIS.CondSrcRange.getBegin(), S.getLoopInit(),
        S.getParallelBlockSize(), S.getAssociatedLoopIters()));
    VarDeclRedeclTransformer VDT(SemaRef);
    auto ClonedBody = VDT.TransformStmt(BodyWithRippleParallelIdx.get());
    if (!ClonedBody.isUsable())
      return;
    S.setRemainderBody(ClonedBody.get());
    // Compute the real IV UB at the end of the remainder
    Expr *SetLoopIVToUB = createStoreUBtoIV(
        SemaRef, S.getAssociatedForStmt()->getInc()->getBeginLoc(),
        LIS.CounterVar, LIS.CounterInit, LIS.NumIterations, LIS.CounterStep);
    S.setEndLoopIVUpdate(SetLoopIVToUB);
  }
}

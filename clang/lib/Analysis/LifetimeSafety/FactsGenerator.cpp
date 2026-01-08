//===- FactsGenerator.cpp - Lifetime Facts Generation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <string>

#include "clang/AST/OperationKinds.h"
#include "clang/Analysis/Analyses/LifetimeSafety/FactsGenerator.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {
using llvm::isa_and_present;

OriginList *FactsGenerator::getOriginsList(const ValueDecl &D) {
  return FactMgr.getOriginMgr().getOrCreateList(&D);
}
OriginList *FactsGenerator::getOriginsList(const Expr &E) {
  return FactMgr.getOriginMgr().getOrCreateList(&E);
}

/// Propagates origin information from Src to Dst through all levels of
/// indirection, creating OriginFlowFacts at each level.
///
/// This function enforces a critical type-safety invariant: both lists must
/// have the same shape (same depth/structure). This invariant ensures that
/// origins flow only between compatible types during expression evaluation.
///
/// Examples:
///   - `int* p = &x;` flows origins from `&x` (depth 1) to `p` (depth 1)
///   - `int** pp = &p;` flows origins from `&p` (depth 2) to `pp` (depth 2)
///     * Level 1: pp <- p's address
///     * Level 2: (*pp) <- what p points to (i.e., &x)
///   - `View v = obj;` flows origins from `obj` (depth 1) to `v` (depth 1)
void FactsGenerator::flow(OriginList *Dst, OriginList *Src, bool Kill) {
  if (!Dst)
    return;
  assert(Src &&
         "Dst is non-null but Src is null. List must have the same length");
  assert(Dst->getLength() == Src->getLength() &&
         "Lists must have the same length");

  while (Dst && Src) {
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        Dst->getOuterOriginID(), Src->getOuterOriginID(), Kill));
    Dst = Dst->peelOuterOrigin();
    Src = Src->peelOuterOrigin();
  }
}

/// Creates a loan for the storage path of a given declaration reference.
/// This function should be called whenever a DeclRefExpr represents a borrow.
/// \param DRE The declaration reference expression that initiates the borrow.
/// \return The new Loan on success, nullptr otherwise.
static const PathLoan *createLoan(FactManager &FactMgr,
                                  const DeclRefExpr *DRE) {
  if (const auto *VD = dyn_cast<ValueDecl>(DRE->getDecl())) {
    AccessPath Path(VD);
    // The loan is created at the location of the DeclRefExpr.
    return FactMgr.getLoanMgr().createLoan<PathLoan>(Path, DRE);
  }
  return nullptr;
}

void FactsGenerator::run() {
  llvm::TimeTraceScope TimeProfile("FactGenerator");
  const CFG &Cfg = *AC.getCFG();
  llvm::SmallVector<Fact *> PlaceholderLoanFacts = issuePlaceholderLoans();
  // Iterate through the CFG blocks in reverse post-order to ensure that
  // initializations and destructions are processed in the correct sequence.
  for (const CFGBlock *Block : *AC.getAnalysis<PostOrderCFGView>()) {
    CurrentBlockFacts.clear();
    EscapesInCurrentBlock.clear();
    if (Block == &Cfg.getEntry())
      CurrentBlockFacts.append(PlaceholderLoanFacts.begin(),
                               PlaceholderLoanFacts.end());
    for (unsigned I = 0; I < Block->size(); ++I) {
      const CFGElement &Element = Block->Elements[I];
      if (std::optional<CFGStmt> CS = Element.getAs<CFGStmt>())
        Visit(CS->getStmt());
      else if (std::optional<CFGLifetimeEnds> LifetimeEnds =
                   Element.getAs<CFGLifetimeEnds>())
        handleLifetimeEnds(*LifetimeEnds);
    }
    CurrentBlockFacts.append(EscapesInCurrentBlock.begin(),
                             EscapesInCurrentBlock.end());
    FactMgr.addBlockFacts(Block, CurrentBlockFacts);
  }
}

/// Simulates LValueToRValue conversion by peeling the outer lvalue origin
/// if the expression is a GLValue. For pointer/view GLValues, this strips
/// the origin representing the storage location to get the origins of the
/// pointed-to value.
///
/// Example: For `View& v`, returns the origin of what v points to, not v's
/// storage.
static OriginList *getRValueOrigins(const Expr *E, OriginList *List) {
  if (!List)
    return nullptr;
  return E->isGLValue() ? List->peelOuterOrigin() : List;
}

void FactsGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(D))
      if (const Expr *InitExpr = VD->getInit()) {
        OriginList *VDList = getOriginsList(*VD);
        if (!VDList)
          continue;
        OriginList *InitList = getOriginsList(*InitExpr);
        assert(InitList && "VarDecl had origins but InitExpr did not");
        flow(VDList, InitList, /*Kill=*/true);
      }
}

void FactsGenerator::VisitDeclRefExpr(const DeclRefExpr *DRE) {
  // Skip function references as their lifetimes are not interesting. Skip non
  // GLValues (like EnumConstants).
  if (DRE->getFoundDecl()->isFunctionOrFunctionTemplate() || !DRE->isGLValue())
    return;
  handleUse(DRE);
  // For all declarations with storage (non-references), we issue a loan
  // representing the borrow of the variable's storage itself.
  //
  // Examples:
  //   - `int x; x` issues loan to x's storage
  //   - `int* p; p` issues loan to p's storage (the pointer variable)
  //   - `View v; v` issues loan to v's storage (the view object)
  //   - `int& r = x; r` issues no loan (r has no storage, it's an alias to x)
  if (doesDeclHaveStorage(DRE->getDecl())) {
    const Loan *L = createLoan(FactMgr, DRE);
    assert(L);
    OriginList *List = getOriginsList(*DRE);
    assert(List &&
           "gl-value DRE of non-pointer type should have an origin list");
    // This loan specifically tracks borrowing the variable's storage location
    // itself and is issued to outermost origin (List->OID).
    CurrentBlockFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), List->getOuterOriginID()));
  }
}

void FactsGenerator::VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
  if (isGslPointerType(CCE->getType())) {
    handleGSLPointerConstruction(CCE);
    return;
  }
}

void FactsGenerator::VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE) {
  // Specifically for conversion operators,
  // like `std::string_view p = std::string{};`
  if (isGslPointerType(MCE->getType()) &&
      isa_and_present<CXXConversionDecl>(MCE->getCalleeDecl()) &&
      isGslOwnerType(MCE->getImplicitObjectArgument()->getType())) {
    // The argument is the implicit object itself.
    handleFunctionCall(MCE, MCE->getMethodDecl(),
                       {MCE->getImplicitObjectArgument()},
                       /*IsGslConstruction=*/true);
    return;
  }
  if (const CXXMethodDecl *Method = MCE->getMethodDecl()) {
    // Construct the argument list, with the implicit 'this' object as the
    // first argument.
    llvm::SmallVector<const Expr *, 4> Args;
    Args.push_back(MCE->getImplicitObjectArgument());
    Args.append(MCE->getArgs(), MCE->getArgs() + MCE->getNumArgs());

    handleFunctionCall(MCE, Method, Args, /*IsGslConstruction=*/false);
  }
}

void FactsGenerator::VisitCallExpr(const CallExpr *CE) {
  handleFunctionCall(CE, CE->getDirectCallee(),
                     {CE->getArgs(), CE->getNumArgs()});
}

void FactsGenerator::VisitCXXNullPtrLiteralExpr(
    const CXXNullPtrLiteralExpr *N) {
  /// TODO: Handle nullptr expr as a special 'null' loan. Uninitialized
  /// pointers can use the same type of loan.
  getOriginsList(*N);
}

void FactsGenerator::VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
  OriginList *Dest = getOriginsList(*ICE);
  if (!Dest)
    return;
  OriginList *SrcList = getOriginsList(*ICE->getSubExpr());

  switch (ICE->getCastKind()) {
  case CK_LValueToRValue:
    // TODO: Decide what to do for x-values here.
    if (!ICE->getSubExpr()->isLValue())
      return;

    assert(SrcList && "LValue being cast to RValue has no origin list");
    // The result of an LValue-to-RValue cast on a pointer lvalue (like `q` in
    // `int *p, *q; p = q;`) should propagate the inner origin (what the pointer
    // points to), not the outer origin (the pointer's storage location). Strip
    // the outer lvalue origin.
    flow(getOriginsList(*ICE), getRValueOrigins(ICE->getSubExpr(), SrcList),
         /*Kill=*/true);
    return;
  case CK_NullToPointer:
    getOriginsList(*ICE);
    // TODO: Flow into them a null origin.
    return;
  case CK_NoOp:
  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
    flow(Dest, SrcList, /*Kill=*/true);
    return;
  case CK_FunctionToPointerDecay:
  case CK_BuiltinFnToFnPtr:
  case CK_ArrayToPointerDecay:
    // Ignore function-to-pointer decays.
    return;
  default:
    return;
  }
}

void FactsGenerator::VisitUnaryOperator(const UnaryOperator *UO) {
  switch (UO->getOpcode()) {
  case UO_AddrOf: {
    const Expr *SubExpr = UO->getSubExpr();
    // The origin of an address-of expression (e.g., &x) is the origin of
    // its sub-expression (x). This fact will cause the dataflow analysis
    // to propagate any loans held by the sub-expression's origin to the
    // origin of this UnaryOperator expression.
    killAndFlowOrigin(*UO, *SubExpr);
    return;
  }
  case UO_Deref: {
    const Expr *SubExpr = UO->getSubExpr();
    killAndFlowOrigin(*UO, *SubExpr);
    return;
  }
  default:
    return;
  }
}

void FactsGenerator::VisitReturnStmt(const ReturnStmt *RS) {
  if (const Expr *RetExpr = RS->getRetValue()) {
    if (OriginList *List = getOriginsList(*RetExpr))
      for (OriginList *L = List; L != nullptr; L = L->peelOuterOrigin())
        EscapesInCurrentBlock.push_back(FactMgr.createFact<OriginEscapesFact>(
            L->getOuterOriginID(), RetExpr));
  }
}

void FactsGenerator::handleAssignment(const Expr *LHSExpr,
                                      const Expr *RHSExpr) {
  if (const auto *DRE_LHS =
          dyn_cast<DeclRefExpr>(LHSExpr->IgnoreParenImpCasts())) {
    OriginList *LHSList = getOriginsList(*DRE_LHS);
    assert(LHSList && "LHS is a DRE and should have an origin list");
    OriginList *RHSList = getOriginsList(*RHSExpr);

    // For operator= with reference parameters (e.g.,
    // `View& operator=(const View&)`), the RHS argument stays an lvalue,
    // unlike built-in assignment where LValueToRValue cast strips the outer
    // lvalue origin. Strip it manually to get the actual value origins being
    // assigned.
    RHSList = getRValueOrigins(RHSExpr, RHSList);

    markUseAsWrite(DRE_LHS);
    // Kill the old loans of the destination origin and flow the new loans
    // from the source origin.
    flow(LHSList->peelOuterOrigin(), RHSList, /*Kill=*/true);
  }
}

void FactsGenerator::VisitBinaryOperator(const BinaryOperator *BO) {
  // TODO: Handle pointer arithmetic (e.g., `p + 1` or `1 + p`) where the
  // result should have the same loans as the pointer operand.
  if (BO->isCompoundAssignmentOp())
    return;
  if (BO->isAssignmentOp())
    handleAssignment(BO->getLHS(), BO->getRHS());
  // TODO: Handle assignments involving dereference like `*p = q`.
}

void FactsGenerator::VisitConditionalOperator(const ConditionalOperator *CO) {
  if (hasOrigins(CO)) {
    // Merge origins from both branches of the conditional operator.
    // We kill to clear the initial state and merge both origins into it.
    killAndFlowOrigin(*CO, *CO->getTrueExpr());
    flowOrigin(*CO, *CO->getFalseExpr());
  }
}

void FactsGenerator::VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
  // Assignment operators have special "kill-then-propagate" semantics
  // and are handled separately.
  if (OCE->getOperator() == OO_Equal && OCE->getNumArgs() == 2 &&
      hasOrigins(OCE->getArg(0)->getType())) {
    handleAssignment(OCE->getArg(0), OCE->getArg(1));
    return;
  }
  VisitCallExpr(OCE);
}

void FactsGenerator::VisitCXXFunctionalCastExpr(
    const CXXFunctionalCastExpr *FCE) {
  // Check if this is a test point marker. If so, we are done with this
  // expression.
  if (handleTestPoint(FCE))
    return;
  if (isGslPointerType(FCE->getType()))
    killAndFlowOrigin(*FCE, *FCE->getSubExpr());
}

void FactsGenerator::VisitInitListExpr(const InitListExpr *ILE) {
  if (!hasOrigins(ILE))
    return;
  // For list initialization with a single element, like `View{...}`, the
  // origin of the list itself is the origin of its single element.
  if (ILE->getNumInits() == 1)
    killAndFlowOrigin(*ILE, *ILE->getInit(0));
}

void FactsGenerator::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *MTE) {
  OriginList *MTEList = getOriginsList(*MTE);
  if (!MTEList)
    return;
  OriginList *SubExprList = getOriginsList(*MTE->getSubExpr());
  if (MTE->isGLValue()) {
    assert(!SubExprList ||
           MTEList->getLength() == SubExprList->getLength() + 1 &&
               "MTE top level origin should contain a loan to the MTE itself");
    MTEList = getRValueOrigins(MTE, MTEList);
    // TODO: Issue a loan to the MTE.
    flow(MTEList, SubExprList, /*Kill=*/true);
  } else {
    assert(MTE->isXValue());
    flow(MTEList, SubExprList, /*Kill=*/true);
  }
}

void FactsGenerator::handleLifetimeEnds(const CFGLifetimeEnds &LifetimeEnds) {
  /// TODO: Handle loans to temporaries.
  const VarDecl *LifetimeEndsVD = LifetimeEnds.getVarDecl();
  if (!LifetimeEndsVD)
    return;
  // Iterate through all loans to see if any expire.
  for (const auto *Loan : FactMgr.getLoanMgr().getLoans()) {
    if (const auto *BL = dyn_cast<PathLoan>(Loan)) {
      // Check if the loan is for a stack variable and if that variable
      // is the one being destructed.
      if (BL->getAccessPath().D == LifetimeEndsVD)
        CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
            BL->getID(), LifetimeEnds.getTriggerStmt()->getEndLoc()));
    }
  }
}

void FactsGenerator::handleGSLPointerConstruction(const CXXConstructExpr *CCE) {
  assert(isGslPointerType(CCE->getType()));
  if (CCE->getNumArgs() != 1)
    return;

  const Expr *Arg = CCE->getArg(0);
  if (isGslPointerType(Arg->getType())) {
    OriginList *ArgList = getOriginsList(*Arg);
    assert(ArgList && "GSL pointer argument should have an origin list");
    // GSL pointer is constructed from another gsl pointer.
    // Example:
    //  View(View v);
    //  View(const View &v);
    ArgList = getRValueOrigins(Arg, ArgList);
    flow(getOriginsList(*CCE), ArgList, /*Kill=*/true);
  } else {
    // This could be a new borrow.
    // TODO: Add code example here.
    handleFunctionCall(CCE, CCE->getConstructor(),
                       {CCE->getArgs(), CCE->getNumArgs()},
                       /*IsGslConstruction=*/true);
  }
}

/// Checks if a call-like expression creates a borrow by passing a value to a
/// reference parameter, creating an IssueFact if it does.
/// \param IsGslConstruction True if this is a GSL construction where all
///   argument origins should flow to the returned origin.
void FactsGenerator::handleFunctionCall(const Expr *Call,
                                        const FunctionDecl *FD,
                                        ArrayRef<const Expr *> Args,
                                        bool IsGslConstruction) {
  OriginList *CallList = getOriginsList(*Call);
  // Ignore functions returning values with no origin.
  if (!FD || !CallList)
    return;
  auto IsArgLifetimeBound = [FD](unsigned I) -> bool {
    const ParmVarDecl *PVD = nullptr;
    if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
        Method && Method->isInstance()) {
      if (I == 0)
        // For the 'this' argument, the attribute is on the method itself.
        return implicitObjectParamIsLifetimeBound(Method) ||
               shouldTrackImplicitObjectArg(Method);
      if ((I - 1) < Method->getNumParams())
        // For explicit arguments, find the corresponding parameter
        // declaration.
        PVD = Method->getParamDecl(I - 1);
    } else if (I == 0 && shouldTrackFirstArgument(FD)) {
      return true;
    } else if (I < FD->getNumParams()) {
      // For free functions or static methods.
      PVD = FD->getParamDecl(I);
    }
    return PVD ? PVD->hasAttr<clang::LifetimeBoundAttr>() : false;
  };
  if (Args.empty())
    return;
  bool KillSrc = true;
  for (unsigned I = 0; I < Args.size(); ++I) {
    OriginList *ArgList = getOriginsList(*Args[I]);
    if (!ArgList)
      continue;
    if (IsGslConstruction) {
      // TODO: document with code example.
      // std::string_view(const std::string_view& from)
      if (isGslPointerType(Args[I]->getType())) {
        assert(!Args[I]->isGLValue() || ArgList->getLength() >= 2);
        ArgList = getRValueOrigins(Args[I], ArgList);
      }
      if (isGslOwnerType(Args[I]->getType())) {
        // GSL construction creates a view that borrows from arguments.
        // This implies flowing origins through the list structure.
        flow(CallList, ArgList, KillSrc);
        KillSrc = false;
      }
    } else if (IsArgLifetimeBound(I)) {
      // Lifetimebound on a non-GSL-ctor function means the returned
      // pointer/reference itself must not outlive the arguments. This
      // only constraints the top-level origin.
      CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
          CallList->getOuterOriginID(), ArgList->getOuterOriginID(), KillSrc));
      KillSrc = false;
    }
  }
}

/// Checks if the expression is a `void("__lifetime_test_point_...")` cast.
/// If so, creates a `TestPointFact` and returns true.
bool FactsGenerator::handleTestPoint(const CXXFunctionalCastExpr *FCE) {
  if (!FCE->getType()->isVoidType())
    return false;

  const auto *SubExpr = FCE->getSubExpr()->IgnoreParenImpCasts();
  if (const auto *SL = dyn_cast<StringLiteral>(SubExpr)) {
    llvm::StringRef LiteralValue = SL->getString();
    const std::string Prefix = "__lifetime_test_point_";

    if (LiteralValue.starts_with(Prefix)) {
      StringRef Annotation = LiteralValue.drop_front(Prefix.length());
      CurrentBlockFacts.push_back(
          FactMgr.createFact<TestPointFact>(Annotation));
      return true;
    }
  }
  return false;
}

// A DeclRefExpr will be treated as a use of the referenced decl. It will be
// checked for use-after-free unless it is later marked as being written to
// (e.g. on the left-hand side of an assignment).
void FactsGenerator::handleUse(const DeclRefExpr *DRE) {
  OriginList *List = getOriginsList(*DRE);
  if (!List)
    return;
  // Remove the outer layer of origin which borrows from the decl directly. This
  // is a use of the underlying decl.
  List = getRValueOrigins(DRE, List);
  // Skip if there is no inner origin (e.g., when it is not a pointer type).
  if (!List)
    return;
  UseFact *UF = FactMgr.createFact<UseFact>(DRE, List);
  CurrentBlockFacts.push_back(UF);
  assert(!UseFacts.contains(DRE));
  UseFacts[DRE] = UF;
}

void FactsGenerator::markUseAsWrite(const DeclRefExpr *DRE) {
  if (UseFacts.contains(DRE))
    UseFacts[DRE]->markAsWritten();
}

// Creates an IssueFact for a new placeholder loan for each pointer or reference
// parameter at the function's entry.
llvm::SmallVector<Fact *> FactsGenerator::issuePlaceholderLoans() {
  const auto *FD = dyn_cast<FunctionDecl>(AC.getDecl());
  if (!FD)
    return {};

  llvm::SmallVector<Fact *> PlaceholderLoanFacts;
  for (const ParmVarDecl *PVD : FD->parameters()) {
    OriginList *List = getOriginsList(*PVD);
    if (!List)
      continue;
    const PlaceholderLoan *L =
        FactMgr.getLoanMgr().createLoan<PlaceholderLoan>(PVD);
    PlaceholderLoanFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), List->getOuterOriginID()));
  }
  return PlaceholderLoanFacts;
}

} // namespace clang::lifetimes::internal

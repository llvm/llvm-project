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
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {
using llvm::isa_and_present;

OriginTree *FactsGenerator::getTree(const ValueDecl &D) {
  return FactMgr.getOriginMgr().getOrCreateTree(&D);
}
OriginTree *FactsGenerator::getTree(const Expr &E) {
  return FactMgr.getOriginMgr().getOrCreateTree(&E);
}

/// Propagates origin information from Src to Dst through all levels of
/// indirection, creating OriginFlowFacts at each level.
///
/// This function enforces a critical type-safety invariant: both trees must
/// have the same shape (same depth/structure). This invariant ensures that
/// origins flow only between compatible types during expression evaluation.
///
/// Examples:
///   - `int* p = &x;` flows origins from `&x` (depth 1) to `p` (depth 1)
///   - `int** pp = &p;` flows origins from `&p` (depth 2) to `pp` (depth 2)
///     * Level 1: pp <- p's address
///     * Level 2: (*pp) <- what p points to (i.e., &x)
///   - `View v = obj;` flows origins from `obj` (depth 1) to `v` (depth 1)
void FactsGenerator::flow(OriginTree *Dst, OriginTree *Src, bool Kill) {
  if (!Dst)
    return;
  assert(Src &&
         "Dst is non-null but Src is null. Trees must have the same shape");
  assert(Dst->getDepth() == Src->getDepth() &&
         "Trees must have the same shape");

  while (Dst && Src) {
    CurrentBlockFacts.push_back(
        FactMgr.createFact<OriginFlowFact>(Dst->OID, Src->OID, Kill));
    Dst = Dst->Pointee;
    Src = Src->Pointee;
  }
}

/// Creates a loan for the storage path of a given declaration reference.
/// This function should be called whenever a DeclRefExpr represents a borrow.
/// \param DRE The declaration reference expression that initiates the borrow.
/// \return The new Loan on success, nullptr otherwise.
static const Loan *createLoan(FactManager &FactMgr, const DeclRefExpr *DRE) {
  if (const auto *VD = dyn_cast<ValueDecl>(DRE->getDecl())) {
    AccessPath Path(VD);
    // The loan is created at the location of the DeclRefExpr.
    return &FactMgr.getLoanMgr().addLoan(Path, DRE);
  }
  return nullptr;
}

void FactsGenerator::run() {
  llvm::TimeTraceScope TimeProfile("FactGenerator");
  // Iterate through the CFG blocks in reverse post-order to ensure that
  // initializations and destructions are processed in the correct sequence.
  for (const CFGBlock *Block : *AC.getAnalysis<PostOrderCFGView>()) {
    CurrentBlockFacts.clear();
    EscapesInCurrentBlock.clear();
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

void FactsGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(D))
      if (const Expr *InitExpr = VD->getInit()) {
        OriginTree *VDTree = getTree(*VD);
        if (!VDTree)
          continue;
        OriginTree *InitTree = getTree(*InitExpr);
        assert(InitTree && "VarDecl had origins but InitExpr did not");
        // Special handling for rvalue references initialized with xvalues.
        // For declarations like `Ranges&& r = std::move(ranges);`, the rvalue
        // reference should directly refer to the object being moved from,
        // rather than creating a new indirection level. We skip the outer
        // reference level and flow the pointee origins directly.
        if (VD->getType()->isRValueReferenceType() && InitExpr->isXValue()) {
          flow(VDTree->Pointee, InitTree->Pointee, /*Kill=*/true);
          continue;
        }
        flow(VDTree, InitTree, /*Kill=*/true);
      }
}

void FactsGenerator::VisitDeclRefExpr(const DeclRefExpr *DRE) {
  // Skip function references and PR values.
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
  //   - `int& r = x; r`issues no loan (r has no storage, it's an alias to x)
  if (doesDeclHaveStorage(DRE->getDecl())) {
    const Loan *L = createLoan(FactMgr, DRE);
    assert(L);
    OriginTree *Tree = getTree(*DRE);
    assert(Tree &&
           "gl-value DRE of non-pointer type should have an origin tree");
    // This loan specifically tracks borrowing the variable's storage location
    // itself and is issued to outermost origin (Tree->OID).
    CurrentBlockFacts.push_back(
        FactMgr.createFact<IssueFact>(L->ID, Tree->OID));
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

static bool isStdMove(const FunctionDecl *FD) {
  return FD && FD->isInStdNamespace() && FD->getIdentifier() &&
         FD->getName() == "move";
}

void FactsGenerator::VisitCallExpr(const CallExpr *CE) {
  handleFunctionCall(CE, CE->getDirectCallee(),
                     {CE->getArgs(), CE->getNumArgs()});
  // Remember accessPath which moved using std::move.
  // TODO: If there is need, this could flow-sensitive
  if (isStdMove(CE->getDirectCallee()))
    if (CE->getNumArgs() == 1)
      if (auto *DRE =
              dyn_cast<DeclRefExpr>(CE->getArg(0)->IgnoreParenImpCasts()))
        MovedDecls.insert(DRE->getDecl());
}

void FactsGenerator::VisitCXXNullPtrLiteralExpr(
    const CXXNullPtrLiteralExpr *N) {
  /// TODO: Handle nullptr expr as a special 'null' loan. Uninitialized
  /// pointers can use the same type of loan.
  getTree(*N);
}

void FactsGenerator::VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
  OriginTree *Dest = getTree(*ICE);
  if (!Dest)
    return;
  OriginTree *SrcTree = getTree(*ICE->getSubExpr());

  if (ICE->getCastKind() == CK_LValueToRValue) {
    // TODO: Decide what to do for x-values here.
    if (!ICE->getSubExpr()->isLValue())
      return;

    assert(SrcTree && "LValue being cast to RValue has no origin tree");
    // The result of an LValue-to-RValue cast on a reference-to-pointer like
    // has the inner origin. Get rid of the outer origin.
    flow(getTree(*ICE), SrcTree->Pointee, /*Kill=*/true);
    return;
  }
  if (ICE->getCastKind() == CK_NullToPointer) {
    getTree(*ICE);
    // TODO: Flow into them a null origin.
    return;
  }
  if (ICE->getCastKind() == CK_NoOp ||
      ICE->getCastKind() == CK_ConstructorConversion ||
      ICE->getCastKind() == CK_UserDefinedConversion)
    flow(Dest, SrcTree, /*Kill=*/true);
  if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
      ICE->getCastKind() == CK_BuiltinFnToFnPtr ||
      ICE->getCastKind() == CK_ArrayToPointerDecay) {
    // Ignore function-to-pointer decays.
    return;
  }
}

void FactsGenerator::VisitUnaryOperator(const UnaryOperator *UO) {
  if (UO->getOpcode() == UO_AddrOf) {
    const Expr *SubExpr = UO->getSubExpr();
    // The origin of an address-of expression (e.g., &x) is the origin of
    // its sub-expression (x). This fact will cause the dataflow analysis
    // to propagate any loans held by the sub-expression's origin to the
    // origin of this UnaryOperator expression.
    killAndFlowOrigin(*UO, *SubExpr);
  }
  if (UO->getOpcode() == UO_Deref) {
    const Expr *SubExpr = UO->getSubExpr();
    killAndFlowOrigin(*UO, *SubExpr);
  }
}

void FactsGenerator::VisitReturnStmt(const ReturnStmt *RS) {
  if (const Expr *RetExpr = RS->getRetValue()) {
    if (OriginTree *Tree = getTree(*RetExpr))
      for (OriginTree *T = Tree; T; T = T->Pointee)
        EscapesInCurrentBlock.push_back(
            FactMgr.createFact<OriginEscapesFact>(T->OID, RetExpr));
  }
}

void FactsGenerator::VisitBinaryOperator(const BinaryOperator *BO) {
  if (BO->isCompoundAssignmentOp())
    return;
  if (BO->isAssignmentOp()) {
    const Expr *LHSExpr = BO->getLHS();
    const Expr *RHSExpr = BO->getRHS();

    if (const auto *DRE_LHS =
            dyn_cast<DeclRefExpr>(LHSExpr->IgnoreParenImpCasts())) {
      OriginTree *LHSTree = getTree(*DRE_LHS);
      assert(LHSTree && "LHS is a DRE and should have an origin tree");
      OriginTree *RHSTree = getTree(*RHSExpr);
      markUseAsWrite(DRE_LHS);
      // Kill the old loans of the destination origin and flow the new loans
      // from the source origin.
      flow(LHSTree->Pointee, RHSTree, /*Kill=*/true);
    }
  }
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
  if (OCE->getOperator() == OO_Equal && OCE->getNumArgs() == 2) {

    const Expr *LHSExpr = OCE->getArg(0);
    const Expr *RHSExpr = OCE->getArg(1);

    if (const auto *DRE_LHS =
            dyn_cast<DeclRefExpr>(LHSExpr->IgnoreParenImpCasts())) {
      OriginTree *LHSTree = getTree(*DRE_LHS);
      assert(LHSTree && "LHS is a DRE and should have an origin tree");
      OriginTree *RHSTree = getTree(*RHSExpr);

      // For operator= with reference parameters (e.g.,
      // `View& operator=(const View&)`), the RHS argument stays an lvalue,
      // unlike built-in assignment where LValueToRValue cast strips the outer
      // lvalue origin. Strip it manually to get the actual value origins being
      // assigned.
      if (RHSExpr->isGLValue())
        RHSTree = RHSTree->Pointee;
      markUseAsWrite(DRE_LHS);
      // Kill the old loans of the destination origin and flow the new loans
      // from the source origin.
      flow(LHSTree->Pointee, RHSTree, /*Kill=*/true);
    }
    return;
  }
  handleFunctionCall(OCE, OCE->getDirectCallee(),
                     {OCE->getArgs(), OCE->getNumArgs()},
                     /*IsGslConstruction=*/false);
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
  OriginTree *MTETree = getTree(*MTE);
  OriginTree *SubExprTree = getTree(*MTE->getSubExpr());
  if (!MTETree)
    return;
  if (MTE->isGLValue()) {
    assert(!SubExprTree ||
           MTETree->getDepth() == SubExprTree->getDepth() + 1 && "todo doc.");
    // Issue a loan to the MTE.
    // const Loan *L = createLoan(FactMgr, MTE);
    // CurrentBlockFacts.push_back(
    //     FactMgr.createFact<IssueFact>(L->ID, MTETree->OID));
    if (SubExprTree)
      flow(MTETree->Pointee, SubExprTree, /*Kill=*/true);
  } else {
    assert(MTE->isXValue());
    flow(MTETree, SubExprTree, /*Kill=*/true);
  }
  // TODO: MTE top level origin should contain a loan to the MTE itself.
}

void FactsGenerator::handleLifetimeEnds(const CFGLifetimeEnds &LifetimeEnds) {
  /// TODO: Handle loans to temporaries.
  const VarDecl *LifetimeEndsVD = LifetimeEnds.getVarDecl();
  if (!LifetimeEndsVD)
    return;
  // Iterate through all loans to see if any expire.
  for (const auto &Loan : FactMgr.getLoanMgr().getLoans()) {
    const AccessPath &LoanPath = Loan.Path;
    if (MovedDecls.contains(LoanPath.D))
      continue;
    // Check if the loan is for a stack variable and if that variable
    // is the one being destructed.
    if (LoanPath.D == LifetimeEndsVD)
      CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
          Loan.ID, LifetimeEnds.getTriggerStmt()->getEndLoc()));
  }
}

void FactsGenerator::handleGSLPointerConstruction(const CXXConstructExpr *CCE) {
  assert(isGslPointerType(CCE->getType()));
  if (CCE->getNumArgs() != 1)
    return;

  if (isGslPointerType(CCE->getArg(0)->getType())) {
    OriginTree *ArgTree = getTree(*CCE->getArg(0));
    assert(ArgTree && "GSL pointer argument should have an origin tree");
    // GSL pointer is constructed from another gsl pointer.
    // Example:
    //  View(View v);
    //  View(const View &v);
    if (ArgTree->getDepth() == 2)
      ArgTree = ArgTree->Pointee;
    flow(getTree(*CCE), ArgTree, /*Kill=*/true);
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
  OriginTree *CallTree = getTree(*Call);
  // Ignore functions returning values with no origin.
  if (!FD || !CallTree)
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
    OriginTree *ArgTree = getTree(*Args[I]);
    if (!ArgTree)
      continue;
    if (IsGslConstruction) {
      // TODO: document with code example.
      // std::string_view(const std::string_view& from)
      if (isGslPointerType(Args[I]->getType()) && Args[I]->isGLValue()) {
        assert(ArgTree->getDepth() >= 2);
        ArgTree = ArgTree->Pointee;
      }
      if (isGslOwnerType(Args[I]->getType())) {
        // GSL construction creates a view that borrows from arguments.
        // This implies flowing origins through the tree structure.
        flow(CallTree, ArgTree, KillSrc);
        KillSrc = false;
      }
    } else if (IsArgLifetimeBound(I)) {
      // Lifetimebound on a non-GSL-ctor function means the returned
      // pointer/reference itself must not outlive the arguments. This
      // only constraints the top-level origin.
      CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
          CallTree->OID, ArgTree->OID, KillSrc));
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
  OriginTree *Tree = getTree(*DRE);
  if (!Tree)
    return;
  // Remove the outer layer of origin which borrows from the decl directly. This
  // is a use of the underlying decl.
  Tree = Tree->Pointee;
  // Skip if there is no inner origin (e.g., when it is not a pointer type).
  if (!Tree)
    return;
  llvm::SmallVector<OriginID, 1> UsedOrigins;
  OriginTree *T = Tree;
  while (T) {
    UsedOrigins.push_back(T->OID);
    T = T->Pointee;
  }
  UseFact *UF = FactMgr.createFact<UseFact>(DRE, UsedOrigins);
  CurrentBlockFacts.push_back(UF);
  assert(!UseFacts.contains(DRE));
  UseFacts[DRE] = UF;
}

void FactsGenerator::markUseAsWrite(const DeclRefExpr *DRE) {
  if (UseFacts.contains(DRE))
    UseFacts[DRE]->markAsWritten();
}

} // namespace clang::lifetimes::internal

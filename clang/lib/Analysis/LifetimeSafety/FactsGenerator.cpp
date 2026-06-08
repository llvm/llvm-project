//===- FactsGenerator.cpp - Lifetime Facts Generation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <string>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/FactsGenerator.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/OperatorKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {
using llvm::isa_and_present;

OriginNode *FactsGenerator::getOriginNode(const ValueDecl &D) {
  return FactMgr.getOriginMgr().getOrCreateNode(&D);
}

OriginNode *FactsGenerator::getOriginNode(const Expr &E) {
  return FactMgr.getOriginMgr().getOrCreateNode(&E);
}

bool FactsGenerator::hasOrigins(QualType QT) const {
  return FactMgr.getOriginMgr().hasOrigins(QT);
}

bool FactsGenerator::hasOrigins(const Expr *E) const {
  return FactMgr.getOriginMgr().hasOrigins(E);
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
void FactsGenerator::flow(OriginNode *Dst, OriginNode *Src, bool Kill) {
  if (!Dst)
    return;
  assert(Src &&
         "Dst is non-null but Src is null. List must have the same length");
  assert(Dst->getLength() == Src->getLength() &&
         "Lists must have the same length");

  while (Dst && Src) {
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        Dst->getOriginID(), Src->getOriginID(), Kill));
    Dst = Dst->getPointeeChild();
    Src = Src->getPointeeChild();
  }
}

/// Creates a loan for the storage path of a given declaration reference.
/// This function should be called whenever a DeclRefExpr represents a borrow.
/// \param DRE The declaration reference expression that initiates the borrow.
/// \return The new Loan on success, nullptr otherwise.
static const Loan *createLoan(FactManager &FactMgr, const DeclRefExpr *DRE) {
  const ValueDecl *VD = DRE->getDecl();
  AccessPath Path(VD);
  // The loan is created at the location of the DeclRefExpr.
  return FactMgr.getLoanMgr().createLoan(Path, DRE);
}

/// Creates a loan for the storage location of a temporary object.
/// \param MTE The MaterializeTemporaryExpr that represents the temporary
/// binding. \return The new Loan.
static const Loan *createLoan(FactManager &FactMgr,
                              const MaterializeTemporaryExpr *MTE) {
  AccessPath Path(MTE);
  return FactMgr.getLoanMgr().createLoan(Path, MTE);
}

/// Creates a loan for an allocation through 'new'
/// \param NE The CXXNewExpr that represents the allocation
/// \return The new Loan on success, nullptr otherwise
static const Loan *createLoan(FactManager &FactMgr, const CXXNewExpr *NE) {
  AccessPath Path(NE);
  return FactMgr.getLoanMgr().createLoan(Path, NE);
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
    CurrentBlock = Block;
    if (Block == &Cfg.getEntry())
      CurrentBlockFacts.append(PlaceholderLoanFacts.begin(),
                               PlaceholderLoanFacts.end());
    for (unsigned I = 0; I < Block->size(); ++I) {
      const CFGElement &Element = Block->Elements[I];
      if (std::optional<CFGStmt> CS = Element.getAs<CFGStmt>())
        Visit(CS->getStmt());
      else if (std::optional<CFGInitializer> Initializer =
                   Element.getAs<CFGInitializer>())
        handleCXXCtorInitializer(Initializer->getInitializer());
      else if (std::optional<CFGLifetimeEnds> LifetimeEnds =
                   Element.getAs<CFGLifetimeEnds>())
        handleLifetimeEnds(*LifetimeEnds);
      else if (std::optional<CFGFullExprCleanup> FullExprCleanup =
                   Element.getAs<CFGFullExprCleanup>()) {
        handleFullExprCleanup(*FullExprCleanup);
      }
    }
    if (Block == &Cfg.getExit())
      handleExitBlock();

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
static OriginNode *getRValueOrigins(const Expr *E, OriginNode *Node) {
  if (!Node)
    return nullptr;
  return E->isGLValue() ? Node->getPointeeChild() : Node;
}

void FactsGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(D))
      if (const Expr *InitExpr = VD->getInit()) {
        OriginNode *VDNode = getOriginNode(*VD);
        if (!VDNode)
          continue;
        OriginNode *InitNode = getOriginNode(*InitExpr);
        assert(InitNode && "VarDecl had origins but InitExpr did not");
        flow(VDNode, InitNode, /*Kill=*/true);
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
    OriginNode *Node = getOriginNode(*DRE);
    assert(Node &&
           "gl-value DRE of non-pointer type should have an origin list");
    // This loan specifically tracks borrowing the variable's storage location
    // itself and is issued to outermost origin (Node->OID).
    CurrentBlockFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), Node->getOriginID()));
  }
}

void FactsGenerator::VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
  if (isGslPointerType(CCE->getType())) {
    handleGSLPointerConstruction(CCE);
    return;
  }
  // For defaulted (implicit or `= default`) copy/move constructors, propagate
  // origins directly. User-defined copy/move constructors are not handled here
  // as they have opaque semantics.
  if (CCE->getConstructor()->isCopyOrMoveConstructor() &&
      CCE->getConstructor()->isDefaulted() && CCE->getNumArgs() == 1 &&
      hasOrigins(CCE->getType())) {
    const Expr *Arg = CCE->getArg(0);
    if (OriginNode *ArgNode = getRValueOrigins(Arg, getOriginNode(*Arg))) {
      flow(getOriginNode(*CCE), ArgNode, /*Kill=*/true);
      return;
    }
  }
  // Standard library callable wrappers (e.g., std::function) propagate the
  // stored lambda's origins.
  if (const auto *RD = CCE->getType()->getAsCXXRecordDecl();
      RD && isStdCallableWrapperType(RD) && CCE->getNumArgs() == 1) {
    const Expr *Arg = CCE->getArg(0);
    if (OriginNode *ArgNode = getRValueOrigins(Arg, getOriginNode(*Arg))) {
      flow(getOriginNode(*CCE), ArgNode, /*Kill=*/true);
      return;
    }
  }
  handleFunctionCall(CCE, CCE->getConstructor(),
                     {CCE->getArgs(), CCE->getNumArgs()},
                     /*IsGslConstruction=*/false);
}

void FactsGenerator::VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *DIE) {
  if (const Expr *Init = DIE->getExpr())
    killAndFlowOrigin(*DIE, *Init);
}

void FactsGenerator::handleCXXCtorInitializer(const CXXCtorInitializer *CII) {
  // Flows origins from the initializer expression to the field.
  // Example: `MyObj(std::string s) : view(s) {}`
  if (const FieldDecl *FD = CII->getAnyMember())
    killAndFlowOrigin(*FD, *CII->getInit());
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

void FactsGenerator::VisitMemberExpr(const MemberExpr *ME) {
  auto *MD = ME->getMemberDecl();
  if (isa<FieldDecl>(MD) && doesDeclHaveStorage(MD)) {
    assert(ME->isGLValue() && "Field member should be GL value");
    OriginNode *Dst = getOriginNode(*ME);
    assert(Dst && "Field member should have an origin list as it is GL value");
    OriginNode *Src = getOriginNode(*ME->getBase());
    assert(Src && "Base expression should be a pointer/reference type");
    // The field's glvalue (outermost origin) holds the same loans as the base
    // expression.
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        Dst->getOriginID(), Src->getOriginID(),
        /*Kill=*/true));
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
  getOriginNode(*N);
}

void FactsGenerator::VisitCastExpr(const CastExpr *CE) {
  OriginNode *Dest = getOriginNode(*CE);
  if (!Dest)
    return;
  const Expr *SubExpr = CE->getSubExpr();
  OriginNode *Src = getOriginNode(*SubExpr);

  switch (CE->getCastKind()) {
  case CK_LValueToRValue:
    if (!SubExpr->isGLValue())
      return;

    assert(Src && "LValue being cast to RValue has no origin list");
    // The result of an LValue-to-RValue cast on a pointer lvalue (like `q` in
    // `int *p, *q; p = q;`) should propagate the inner origin (what the pointer
    // points to), not the outer origin (the pointer's storage location). Strip
    // the outer lvalue origin.
    flow(getOriginNode(*CE), getRValueOrigins(SubExpr, Src),
         /*Kill=*/true);
    return;
  case CK_NullToPointer:
    getOriginNode(*CE);
    // TODO: Flow into them a null origin.
    return;
  case CK_NoOp:
  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
    flow(Dest, Src, /*Kill=*/true);
    return;
  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase:
    // It is possible that the derived class and base class have different
    // gsl::Pointer annotations. Skip if their origin shape differ.
    if (Dest && Src && Dest->getLength() == Src->getLength())
      flow(Dest, Src, /*Kill=*/true);
    return;
  case CK_ArrayToPointerDecay:
    assert(Src && "Array expression should have origins as it is GL value");
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        Dest->getOriginID(), Src->getOriginID(), /*Kill=*/true));
    return;
  case CK_FunctionToPointerDecay:
  case CK_BuiltinFnToFnPtr:
    // Ignore function-to-pointer decays.
    return;
  case CK_BitCast:
    // OriginLists for Src and Dst may differ here. For example when casting
    // from int** to void*
    if (Src && Dest && Dest->getLength() == Src->getLength())
      flow(Dest, Src, /*Kill=*/true);
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
    if (OriginNode *Node = getOriginNode(*RetExpr))
      for (OriginNode *L = Node; L != nullptr; L = L->getPointeeChild())
        EscapesInCurrentBlock.push_back(
            FactMgr.createFact<ReturnEscapeFact>(L->getOriginID(), RetExpr));
  }
}

void FactsGenerator::handleAssignment(const Expr *TargetExpr,
                                      const Expr *LHSExpr,
                                      const Expr *RHSExpr) {
  LHSExpr = LHSExpr->IgnoreParenImpCasts();
  OriginNode *LHSNode = nullptr;

  if (const auto *DRE_LHS = dyn_cast<DeclRefExpr>(LHSExpr)) {
    LHSNode = getOriginNode(*DRE_LHS);
    assert(LHSNode && "LHS is a DRE and should have an origin list");
  }
  // Handle assignment to member fields (e.g., `this->view = s` or `view = s`).
  // This enables detection of dangling fields when local values escape to
  // fields.
  if (const auto *ME_LHS = dyn_cast<MemberExpr>(LHSExpr)) {
    LHSNode = getOriginNode(*ME_LHS);
    assert(LHSNode && "LHS is a MemberExpr and should have an origin list");
  }
  if (!LHSNode)
    return;
  OriginNode *RHSNode = getOriginNode(*RHSExpr);
  // For operator= with reference parameters (e.g.,
  // `View& operator=(const View&)`), the RHS argument stays an lvalue,
  // unlike built-in assignment where LValueToRValue cast strips the outer
  // lvalue origin. Strip it manually to get the actual value origins being
  // assigned.
  RHSNode = getRValueOrigins(RHSExpr, RHSNode);

  if (const auto *DRE_LHS = dyn_cast<DeclRefExpr>(LHSExpr)) {
    QualType QT = DRE_LHS->getDecl()->getType();
    if (QT->isReferenceType()) {
      if (hasOrigins(QT->getPointeeType())) {
        // Writing through a reference uses the binding but overwrites the
        // pointee. Model this as a Read of the outer origin (keeping the
        // binding live) and a Write of the inner origins (killing the pointee's
        // liveness).
        if (UseFact *UF = UseFacts.lookup(DRE_LHS)) {
          const OriginNode *FullNode = UF->getUsedOrigins();
          assert(FullNode);
          UF->setUsedOrigins(FactMgr.getOriginMgr().createSingleOriginNode(
              FullNode->getOriginID()));
          if (const OriginNode *InnerNode = FullNode->getPointeeChild()) {
            UseFact *WriteUF = FactMgr.createFact<UseFact>(DRE_LHS, InnerNode);
            WriteUF->markAsWritten();
            CurrentBlockFacts.push_back(WriteUF);
          }
        }
      }
    } else
      markUseAsWrite(DRE_LHS);
  }
  if (!RHSNode) {
    // RHS has no tracked origins (e.g., assigning a callable without origins
    // to std::function). Clear loans of the destination.
    for (OriginNode *LHSInner = LHSNode->getPointeeChild(); LHSInner;
         LHSInner = LHSInner->getPointeeChild())
      CurrentBlockFacts.push_back(
          FactMgr.createFact<KillOriginFact>(LHSInner->getOriginID()));
    return;
  }
  // Kill the old loans of the destination origin and flow the new loans
  // from the source origin.
  flow(LHSNode->getPointeeChild(), RHSNode, /*Kill=*/true);
  killAndFlowOrigin(*TargetExpr, *LHSExpr);
}

void FactsGenerator::handlePointerArithmetic(const BinaryOperator *BO) {
  if (Expr *RHS = BO->getRHS(); RHS->getType()->isPointerType()) {
    killAndFlowOrigin(*BO, *RHS);
    return;
  }
  Expr *LHS = BO->getLHS();
  assert(LHS->getType()->isPointerType() &&
         "Pointer arithmetic must have a pointer operand");
  killAndFlowOrigin(*BO, *LHS);
}

void FactsGenerator::VisitBinaryOperator(const BinaryOperator *BO) {
  if (BO->isCompoundAssignmentOp())
    return;
  if (BO->getType()->isPointerType() && BO->isAdditiveOp())
    handlePointerArithmetic(BO);
  handleUse(BO->getRHS());
  if (BO->isAssignmentOp())
    handleAssignment(BO, BO->getLHS(), BO->getRHS());
  // TODO: Handle assignments involving dereference like `*p = q`.
}

void FactsGenerator::VisitConditionalOperator(const ConditionalOperator *CO) {
  if (!hasOrigins(CO))
    return;

  const Expr *TrueExpr = CO->getTrueExpr();
  const Expr *FalseExpr = CO->getFalseExpr();

  const auto Preds = CurrentBlock->preds();

  // Skip origin flow from conditional operator arms that cannot produce the
  // result value: throw arms and calls to noreturn functions.
  bool TBHasEdge = true;
  bool FBHasEdge = true;

  switch (CurrentBlock->pred_size()) {
  case 0:
    return;
  case 1: {
    TBHasEdge = llvm::any_of(**Preds.begin(),
                             [ExpectedStmt = TrueExpr->IgnoreParenImpCasts()](
                                 const CFGElement &Elt) {
                               if (auto CS = Elt.getAs<CFGStmt>())
                                 return CS->getStmt() == ExpectedStmt;
                               return false;
                             });
    FBHasEdge = !TBHasEdge;
    break;
  }
  case 2: {
    const auto *It = Preds.begin();
    TBHasEdge = It->isReachable();
    FBHasEdge = (++It)->isReachable();
    break;
  }
  default:
    llvm_unreachable("expected at most 2 predecessors");
    return;
  }

  bool FirstFlow = true;
  auto HandleFlow = [&](const Expr *E) {
    if (FirstFlow) {
      killAndFlowOrigin(*CO, *E);
      FirstFlow = false;
    } else {
      flowOrigin(*CO, *E);
    }
  };

  if (TBHasEdge)
    HandleFlow(TrueExpr);
  if (FBHasEdge)
    HandleFlow(FalseExpr);
}

void FactsGenerator::VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
  // Assignment operators have special "kill-then-propagate" semantics
  // and are handled separately.
  if (OCE->getOperator() == OO_Equal && OCE->getNumArgs() == 2 &&
      hasOrigins(OCE->getArg(0)->getType())) {
    // Pointer-like types: assignment inherently propagates origins.
    QualType LHSTy = OCE->getArg(0)->getType();
    if (LHSTy->isPointerOrReferenceType() || isGslPointerType(LHSTy) ||
        isGslOwnerType(LHSTy)) {
      handleAssignment(OCE, OCE->getArg(0), OCE->getArg(1));
      return;
    }
    // Standard library callable wrappers (e.g., std::function) can propagate
    // the stored lambda's origins.
    if (const auto *RD = LHSTy->getAsCXXRecordDecl();
        RD && isStdCallableWrapperType(RD)) {
      handleAssignment(OCE, OCE->getArg(0), OCE->getArg(1));
      return;
    }
    // Other tracked types: only defaulted operator= propagates origins.
    // User-defined operator= has opaque semantics, so don't handle them now.
    if (const auto *MD =
            dyn_cast_or_null<CXXMethodDecl>(OCE->getDirectCallee());
        MD && MD->isDefaulted()) {
      handleAssignment(OCE, OCE->getArg(0), OCE->getArg(1));
      return;
    }
  }

  ArrayRef Args = {OCE->getArgs(), OCE->getNumArgs()};
  // For `static operator()`, the first argument is the object argument,
  // remove it from the argument list to avoid off-by-one errors.
  if (OCE->getOperator() == OO_Call && OCE->getDirectCallee()->isStatic())
    Args = Args.slice(1);
  handleFunctionCall(OCE, OCE->getDirectCallee(), Args);
}

void FactsGenerator::VisitCXXFunctionalCastExpr(
    const CXXFunctionalCastExpr *FCE) {
  // Check if this is a test point marker. If so, we are done with this
  // expression.
  if (handleTestPoint(FCE))
    return;
  VisitCastExpr(FCE);
}

void FactsGenerator::VisitInitListExpr(const InitListExpr *ILE) {
  if (!hasOrigins(ILE))
    return;
  // For list initialization with a single element, like `View{...}`, the
  // origin of the list itself is the origin of its single element.
  if (ILE->getNumInits() == 1)
    killAndFlowOrigin(*ILE, *ILE->getInit(0));
}

void FactsGenerator::VisitCXXBindTemporaryExpr(
    const CXXBindTemporaryExpr *BTE) {
  killAndFlowOrigin(*BTE, *BTE->getSubExpr());
}

void FactsGenerator::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *MTE) {
  assert(MTE->isGLValue());
  OriginNode *MTENode = getOriginNode(*MTE);
  if (!MTENode)
    return;
  OriginNode *SubExprNode = getOriginNode(*MTE->getSubExpr());
  assert((!SubExprNode ||
          MTENode->getLength() == (SubExprNode->getLength() + 1)) &&
         "MTE top level origin should contain a loan to the MTE itself");

  OriginNode *RValMTENode = getRValueOrigins(MTE, MTENode);
  flow(RValMTENode, SubExprNode, /*Kill=*/true);
  OriginID OuterMTEID = MTENode->getOriginID();
  if (MTE->getStorageDuration() == SD_FullExpression) {
    // Issue a loan to MTE for the storage location represented by MTE.
    const Loan *L = createLoan(FactMgr, MTE);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), OuterMTEID));
  }
}

void FactsGenerator::VisitLambdaExpr(const LambdaExpr *LE) {
  // The lambda gets a single merged origin that aggregates all captured
  // pointer-like origins. Currently we only need to detect whether the lambda
  // outlives any capture.
  OriginNode *LambdaNode = getOriginNode(*LE);
  if (!LambdaNode)
    return;
  bool Kill = true;
  for (const Expr *Init : LE->capture_inits()) {
    if (!Init)
      continue;
    OriginNode *InitNode = getOriginNode(*Init);
    if (!InitNode)
      continue;
    // FIXME: Consider flowing all origin levels once lambdas support more than
    // one origin. Currently only the outermost origin is flowed, so by-ref
    // captures like `[&p]` (where p is string_view) miss inner-level
    // invalidation.
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        LambdaNode->getOriginID(), InitNode->getOriginID(), Kill));
    Kill = false;
  }
}

void FactsGenerator::VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE) {
  assert(ASE->isGLValue() && "Array subscript should be a GL value");
  OriginNode *Dst = getOriginNode(*ASE);
  assert(Dst && "Array subscript should have origins as it is a GL value");
  OriginNode *Src = getOriginNode(*ASE->getBase());
  assert(Src && "Base of array subscript should have origins");
  CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
      Dst->getOriginID(), Src->getOriginID(), /*Kill=*/true));
}

void FactsGenerator::handlePlacementNew(const CXXNewExpr *NE,
                                        OriginNode *NewNode) {
  // Model only the standard single-argument placement new form, where the
  // placement argument corresponds to a void* allocation-function parameter.
  // Other placement forms, such as std::nothrow, are not modeled as providing
  // storage for the returned pointer.
  if (NE->getNumPlacementArgs() != 1)
    return;

  const FunctionDecl *OperatorNew = NE->getOperatorNew();
  if (OperatorNew->getNumParams() <= 1)
    return;

  const auto *Arg =
      OperatorNew->getParamDecl(1)->getType()->getAs<PointerType>();
  if (!Arg || !Arg->isVoidPointerType())
    return;

  // Use the placement argument before the implicit conversion to void*, so
  // inner origins are still available.
  const Expr *PlacementArg = NE->getPlacementArg(0);
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(PlacementArg);
      ICE && ICE->getCastKind() == CK_BitCast &&
      PlacementArg->getType()->isVoidPointerType())
    PlacementArg = ICE->getSubExpr();
  OriginNode *PlacementNode = getOriginNode(*PlacementArg);
  // FIXME: General placement arguments need separate handling to overwrite
  // the right origins.

  // The pointer returned by placement new comes from the placement
  // argument.
  if (PlacementNode)
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        NewNode->getOriginID(), PlacementNode->getOriginID(), true));
}

void FactsGenerator::VisitCXXNewExpr(const CXXNewExpr *NE) {
  OriginNode *NewNode = getOriginNode(*NE);
  const Expr *Init = NE->getInitializer();

  if (NE->getNumPlacementArgs() == 1) {
    handlePlacementNew(NE, NewNode);
  } else {
    const Loan *L = createLoan(FactMgr, NE);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), NewNode->getOriginID()));
  }

  NewNode = NewNode->getPointeeChild();

  if (!NewNode || !Init)
    return;

  // FIXME: OriginNode is null for `new[]` initializers. Remove this `Init`
  // check once array origins are supported.
  if (OriginNode *InitNode = getOriginNode(*Init); InitNode)
    flow(NewNode, InitNode, true);
}

void FactsGenerator::VisitCXXDeleteExpr(const CXXDeleteExpr *DE) {
  OriginNode *Node = getOriginNode(*DE->getArgument());
  CurrentBlockFacts.push_back(
      FactMgr.createFact<InvalidateOriginFact>(Node->getOriginID(), DE));
}

bool FactsGenerator::escapesViaReturn(OriginID OID) const {
  return llvm::any_of(EscapesInCurrentBlock, [OID](const Fact *F) {
    if (const auto *EF = F->getAs<ReturnEscapeFact>())
      return EF->getEscapedOriginID() == OID;
    return false;
  });
}

void FactsGenerator::handleLifetimeEnds(const CFGLifetimeEnds &LifetimeEnds) {
  const VarDecl *LifetimeEndsVD = LifetimeEnds.getVarDecl();
  if (!LifetimeEndsVD)
    return;
  // Expire the origin when its variable's lifetime ends to ensure liveness
  // doesn't persist through loop back-edges.
  std::optional<OriginID> ExpiredOID;
  if (OriginNode *Node = getOriginNode(*LifetimeEndsVD)) {
    OriginID OID = Node->getOriginID();
    // Skip origins that escape via return; the escape checker needs their loans
    // to remain until the return statement is processed.
    if (!escapesViaReturn(OID))
      ExpiredOID = OID;
  }
  CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
      AccessPath(LifetimeEndsVD), LifetimeEnds.getTriggerStmt()->getEndLoc(),
      ExpiredOID));
}

void FactsGenerator::handleFullExprCleanup(
    const CFGFullExprCleanup &FullExprCleanup) {
  for (const auto *MTE : FullExprCleanup.getExpiringMTEs())
    CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
        AccessPath(MTE), FullExprCleanup.getCleanupLoc()));
}

void FactsGenerator::handleExitBlock() {
  for (const Origin &O : FactMgr.getOriginMgr().getOrigins())
    if (auto *FD = dyn_cast_if_present<FieldDecl>(O.getDecl()))
      // Create FieldEscapeFacts for all field origins that remain live at exit.
      EscapesInCurrentBlock.push_back(
          FactMgr.createFact<FieldEscapeFact>(O.ID, FD));
    else if (auto *VD = dyn_cast_if_present<VarDecl>(O.getDecl())) {
      // Create GlobalEscapeFacts for all origins with global-storage that
      // remain live at exit.
      if (VD->hasGlobalStorage()) {
        EscapesInCurrentBlock.push_back(
            FactMgr.createFact<GlobalEscapeFact>(O.ID, VD));
      }
    }
}

void FactsGenerator::handleGSLPointerConstruction(const CXXConstructExpr *CCE) {
  assert(isGslPointerType(CCE->getType()));
  if (CCE->getNumArgs() != 1)
    return;

  const Expr *Arg = CCE->getArg(0);
  if (isGslPointerType(Arg->getType())) {
    OriginNode *ArgNode = getOriginNode(*Arg);
    assert(ArgNode && "GSL pointer argument should have an origin list");
    // GSL pointer is constructed from another gsl pointer.
    // Example:
    //  View(View v);
    //  View(const View &v);
    ArgNode = getRValueOrigins(Arg, ArgNode);
    flow(getOriginNode(*CCE), ArgNode, /*Kill=*/true);
  } else if (Arg->getType()->isPointerType()) {
    // GSL pointer is constructed from a raw pointer. Flow only the outermost
    // raw pointer. Example:
    //  View(const char*);
    //  Span<int*>(const in**);
    OriginNode *ArgNode = getOriginNode(*Arg);
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        getOriginNode(*CCE)->getOriginID(), ArgNode->getOriginID(),
        /*Kill=*/true));
  } else {
    // This could be a new borrow.
    // TODO: Add code example here.
    handleFunctionCall(CCE, CCE->getConstructor(),
                       {CCE->getArgs(), CCE->getNumArgs()},
                       /*IsGslConstruction=*/true);
  }
}

void FactsGenerator::handleMovedArgsInCall(const FunctionDecl *FD,
                                           ArrayRef<const Expr *> Args) {
  unsigned IsInstance = 0;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD);
      MD && MD->isInstance() && !isa<CXXConstructorDecl>(FD)) {
    IsInstance = 1;
    // std::unique_ptr::release() transfers ownership.
    // Treat it as a move to prevent false-positive warnings when the unique_ptr
    // destructor runs after ownership has been transferred.
    if (isUniquePtrRelease(*MD)) {
      const Expr *UniquePtrExpr = Args[0];
      OriginNode *MovedOrigins = getOriginNode(*UniquePtrExpr);
      if (MovedOrigins)
        CurrentBlockFacts.push_back(FactMgr.createFact<MovedOriginFact>(
            UniquePtrExpr, MovedOrigins->getOriginID()));
    }
  }

  // Skip 'this' arg as it cannot be moved.
  for (unsigned I = IsInstance;
       I < Args.size() && I < FD->getNumParams() + IsInstance; ++I) {
    const ParmVarDecl *PVD = FD->getParamDecl(I - IsInstance);
    if (!PVD->getType()->isRValueReferenceType())
      continue;
    const Expr *Arg = Args[I];
    OriginNode *MovedOrigins = getOriginNode(*Arg);
    assert(MovedOrigins->getLength() >= 1 &&
           "unexpected length for r-value reference param");
    // Arg is being moved to this parameter. Mark the origin as moved.
    CurrentBlockFacts.push_back(
        FactMgr.createFact<MovedOriginFact>(Arg, MovedOrigins->getOriginID()));
  }
}

void FactsGenerator::handleInvalidatingCall(const Expr *Call,
                                            const FunctionDecl *FD,
                                            ArrayRef<const Expr *> Args) {
  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (!MD || !MD->isInstance())
    return;

  if (!isInvalidationMethod(*MD))
    return;

  // Heuristics to turn-down false positives. Skip member field expressions for
  // now. This is not a perfect filter and will still surface some false
  // positives (e.g. `auto& r = s.v`).
  if (!isa<DeclRefExpr>(Args[0]->IgnoreImpCasts()))
    return;

  OriginNode *ThisNode = getOriginNode(*Args[0]);
  if (ThisNode)
    CurrentBlockFacts.push_back(FactMgr.createFact<InvalidateOriginFact>(
        ThisNode->getOriginID(), Call));
}

void FactsGenerator::handleDestructiveCall(const Expr *Call,
                                           const FunctionDecl *FD,
                                           ArrayRef<const Expr *> Args) {
  if (!destructsFirstArg(*FD))
    return;
  OriginNode *ArgNode = getOriginNode(*Args[0]);
  if (ArgNode)
    CurrentBlockFacts.push_back(
        FactMgr.createFact<InvalidateOriginFact>(ArgNode->getOriginID(), Call));
}

void FactsGenerator::handleImplicitObjectFieldUses(const Expr *Call,
                                                   const FunctionDecl *FD) {
  const auto *MemberCall = dyn_cast_or_null<CXXMemberCallExpr>(Call);
  if (!MemberCall)
    return;

  if (!isa_and_present<CXXThisExpr>(
          MemberCall->getImplicitObjectArgument()->IgnoreImpCasts()))
    return;

  const auto *MD = dyn_cast<CXXMethodDecl>(FD);
  assert(MD && "Function must be a CXXMethodDecl for member calls");

  const auto *ClassDecl = MD->getParent()->getDefinition();
  if (!ClassDecl)
    return;

  const auto UseFields = [&](const CXXRecordDecl *RD) {
    for (const auto *Field : RD->fields())
      if (auto *FieldNode = getOriginNode(*Field))
        CurrentBlockFacts.push_back(
            FactMgr.createFact<UseFact>(Call, FieldNode));
  };

  UseFields(ClassDecl);

  ClassDecl->forallBases([&](const CXXRecordDecl *Base) {
    UseFields(Base);
    return true;
  });
}

void FactsGenerator::handleLifetimeCaptureBy(const FunctionDecl *FD,
                                             ArrayRef<const Expr *> Args) {
  if (Args.empty())
    return;
  // FIXME: Add support for capture_by on constructors.
  if (isa<CXXConstructorDecl>(FD))
    return;
  const auto *Method = dyn_cast<CXXMethodDecl>(FD);
  bool IsInstance =
      Method && Method->isInstance() && !isa<CXXConstructorDecl>(FD);
  auto getArgCaptureBy = [FD,
                          IsInstance](unsigned I) -> LifetimeCaptureByAttr * {
    const ParmVarDecl *PVD = nullptr;
    if (IsInstance) {
      // FIXME: Add support for I == 0 i.e. capture_by on function declarations
      if (I > 0 && I - 1 < FD->getNumParams())
        PVD = FD->getParamDecl(I - 1);
    } else {
      if (I < FD->getNumParams())
        PVD = FD->getParamDecl(I);
    }
    return PVD ? PVD->getAttr<LifetimeCaptureByAttr>() : nullptr;
  };
  for (unsigned I = 0; I < Args.size(); ++I) {
    const LifetimeCaptureByAttr *Attr = getArgCaptureBy(I);
    if (!Attr)
      continue;
    OriginNode *CapturedOriginNode = getOriginNode(*Args[I]);
    if (!CapturedOriginNode)
      continue;
    if (!CapturedOriginNode)
      continue;
    for (int CapturingArgIdx : Attr->params()) {
      // FIXME: Add support for capturing to Global/unknown.
      if (CapturingArgIdx == LifetimeCaptureByAttr::Global ||
          CapturingArgIdx == LifetimeCaptureByAttr::Unknown ||
          CapturingArgIdx == LifetimeCaptureByAttr::Invalid)
        continue;
      ArrayRef<const Expr *> CallArgs = IsInstance ? Args.drop_front() : Args;
      const Expr *CapturedByArg =
          (CapturingArgIdx == LifetimeCaptureByAttr::This)
              ? Args[0]
              : CallArgs[CapturingArgIdx];
      assert(CapturedByArg && "Capturer expression must be valid");

      OriginNode *CapturingOriginNode = getOriginNode(*CapturedByArg);
      OriginNode *Dest = getRValueOrigins(CapturedByArg, CapturingOriginNode);
      if (!Dest)
        continue;
      // KillDest=false because we cannot know if previous captures are being
      // replaced or accumulated. Multiple successive captures into the same
      // destination must all be tracked, so captured lifetimes are always
      // merged.
      CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
          Dest->getOriginID(), CapturedOriginNode->getOriginID(),
          /*KillDest=*/false));
    }
  }
}

void FactsGenerator::handleFunctionCall(const Expr *Call,
                                        const FunctionDecl *FD,
                                        ArrayRef<const Expr *> Args,
                                        bool IsGslConstruction) {
  OriginNode *CallNode = getOriginNode(*Call);
  // Ignore functions returning values with no origin.
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  if (!FD)
    return;
  // All arguments to a function are a use of the corresponding expressions.
  for (const Expr *Arg : Args)
    handleUse(Arg);
  handleInvalidatingCall(Call, FD, Args);
  handleDestructiveCall(Call, FD, Args);
  handleMovedArgsInCall(FD, Args);
  handleImplicitObjectFieldUses(Call, FD);
  handleLifetimeCaptureBy(FD, Args);
  if (!CallNode)
    return;
  if (isStdReferenceCast(FD)) {
    assert(Args.size() == 1 &&
           "std reference cast builtins take exactly one argument");
    // std reference-cast functions like std::move return a result that refers
    // to the same object as the argument, so propagate the full origins.
    flow(CallNode, getOriginNode(*Args[0]), /*Kill=*/true);
    return;
  }
  auto IsArgLifetimeBound = [FD, &Args](unsigned I) -> bool {
    const ParmVarDecl *PVD = nullptr;
    if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
        Method && Method->isInstance() && !isa<CXXConstructorDecl>(FD)) {
      if (I == 0)
        // For the 'this' argument, the attribute is on the method itself.
        return implicitObjectParamIsLifetimeBound(Method) ||
               shouldTrackImplicitObjectArg(
                   *Args[0], Method, /*RunningUnderLifetimeSafety=*/true);
      if ((I - 1) < Method->getNumParams())
        // For explicit arguments, find the corresponding parameter
        // declaration.
        PVD = Method->getParamDecl(I - 1);
    } else if (I == 0 && shouldTrackFirstArgument(FD)) {
      return true;
    } else if (I == 1 && shouldTrackSecondArgument(FD)) {
      return true;
    } else if (I < FD->getNumParams()) {
      // For free functions or static methods.
      PVD = FD->getParamDecl(I);
    }
    return PVD ? PVD->hasAttr<clang::LifetimeBoundAttr>() : false;
  };
  auto shouldTrackPointerImplicitObjectArg = [FD, &Args](unsigned I) -> bool {
    const auto *Method = dyn_cast<CXXMethodDecl>(FD);
    if (!Method || !Method->isInstance())
      return false;
    return I == 0 &&
           isGslPointerType(Method->getFunctionObjectParameterType()) &&
           shouldTrackImplicitObjectArg(*Args[0], Method,
                                        /*RunningUnderLifetimeSafety=*/true);
  };
  if (Args.empty())
    return;
  bool KillSrc = true;
  for (unsigned I = 0; I < Args.size(); ++I) {
    OriginNode *ArgNode = getOriginNode(*Args[I]);
    if (!ArgNode)
      continue;
    if (IsGslConstruction) {
      // TODO: document with code example.
      // std::string_view(const std::string_view& from)
      if (isGslPointerType(Args[I]->getType())) {
        assert(!Args[I]->isGLValue() || ArgNode->getLength() >= 2);
        ArgNode = getRValueOrigins(Args[I], ArgNode);
      }
      if (isGslOwnerType(Args[I]->getType())) {
        // The constructed gsl::Pointer borrows from the Owner's storage, not
        // from what the Owner itself borrows, so only the outermost origin is
        // needed.
        CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
            CallNode->getOriginID(), ArgNode->getOriginID(), KillSrc));
        KillSrc = false;
      } else if (IsArgLifetimeBound(I)) {
        // Only flow the outer origin here. For lifetimebound args in
        // gsl::Pointer construction, we do not have enough information to
        // safely match inner origins, so the source and
        // destination origin lists may have different lengths.
        // FIXME: Handle origin-shape mismatches gracefully so we can also flow
        // inner origins.
        CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
            CallNode->getOriginID(), ArgNode->getOriginID(), KillSrc));
        KillSrc = false;
      }
    } else if (shouldTrackPointerImplicitObjectArg(I)) {
      assert(ArgNode->getLength() >= 2 &&
             "Object arg of pointer type should have at least two origins");
      // See through the GSLPointer reference to see the pointer's value.
      CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
          CallNode->getOriginID(), ArgNode->getPointeeChild()->getOriginID(),
          KillSrc));
      KillSrc = false;
    } else if (IsArgLifetimeBound(I)) {
      // Lifetimebound on a non-GSL-ctor function means the returned
      // pointer/reference itself must not outlive the arguments. This
      // only constrains the top-level origin.
      CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
          CallNode->getOriginID(), ArgNode->getOriginID(), KillSrc));
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

void FactsGenerator::handleUse(const Expr *E) {
  OriginNode *Node = getOriginNode(*E);
  if (!Node)
    return;
  // For DeclRefExpr: Remove the outer layer of origin which borrows from the
  // decl directly (e.g., when this is not a reference). This is a use of the
  // underlying decl.
  if (auto *DRE = dyn_cast<DeclRefExpr>(E);
      DRE && !DRE->getDecl()->getType()->isReferenceType())
    Node = getRValueOrigins(DRE, Node);
  // Skip if there is no inner origin (e.g., when it is not a pointer type).
  if (!Node)
    return;
  if (!UseFacts.contains(E)) {
    UseFact *UF = FactMgr.createFact<UseFact>(E, Node);
    CurrentBlockFacts.push_back(UF);
    UseFacts[E] = UF;
  }
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
  if (auto ThisOrigins = FactMgr.getOriginMgr().getThisOrigins()) {
    OriginNode *Node = *ThisOrigins;
    const Loan *L = FactMgr.getLoanMgr().createLoan(
        AccessPath::Placeholder(cast<CXXMethodDecl>(FD)),
        /*IssuingExpr=*/nullptr);
    PlaceholderLoanFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), Node->getOriginID()));
  }
  for (const ParmVarDecl *PVD : FD->parameters()) {
    OriginNode *Node = getOriginNode(*PVD);
    if (!Node)
      continue;
    const Loan *L = FactMgr.getLoanMgr().createLoan(
        AccessPath::Placeholder(PVD), /*IssuingExpr=*/nullptr);
    PlaceholderLoanFacts.push_back(
        FactMgr.createFact<IssueFact>(L->getID(), Node->getOriginID()));
  }
  return PlaceholderLoanFacts;
}

} // namespace clang::lifetimes::internal

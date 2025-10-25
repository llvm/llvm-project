//===- FactsGenerator.cpp - Lifetime Facts Generation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/FactsGenerator.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {
using llvm::isa_and_present;

static bool isGslPointerType(QualType QT) {
  if (const auto *RD = QT->getAsCXXRecordDecl()) {
    // We need to check the template definition for specializations.
    if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
      return CTSD->getSpecializedTemplate()
          ->getTemplatedDecl()
          ->hasAttr<PointerAttr>();
    return RD->hasAttr<PointerAttr>();
  }
  return false;
}

static bool isPointerType(QualType QT) {
  return QT->isPointerOrReferenceType() || isGslPointerType(QT);
}
// Check if a type has an origin.
static bool hasOrigin(const Expr *E) {
  return E->isGLValue() || isPointerType(E->getType());
}

static bool hasOrigin(const VarDecl *VD) {
  return isPointerType(VD->getType());
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
    for (unsigned I = 0; I < Block->size(); ++I) {
      const CFGElement &Element = Block->Elements[I];
      if (std::optional<CFGStmt> CS = Element.getAs<CFGStmt>())
        Visit(CS->getStmt());
      else if (std::optional<CFGAutomaticObjDtor> DtorOpt =
                   Element.getAs<CFGAutomaticObjDtor>())
        handleDestructor(*DtorOpt);
    }
    FactMgr.addBlockFacts(Block, CurrentBlockFacts);
  }
}

void FactsGenerator::VisitDeclStmt(const DeclStmt *DS) {
  for (const Decl *D : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(D))
      if (hasOrigin(VD))
        if (const Expr *InitExpr = VD->getInit())
          killAndFlowOrigin(*VD, *InitExpr);
}

void FactsGenerator::VisitDeclRefExpr(const DeclRefExpr *DRE) {
  handleUse(DRE);
  // For non-pointer/non-view types, a reference to the variable's storage
  // is a borrow. We create a loan for it.
  // For pointer/view types, we stick to the existing model for now and do
  // not create an extra origin for the l-value expression itself.

  // TODO: A single origin for a `DeclRefExpr` for a pointer or view type is
  // not sufficient to model the different levels of indirection. The current
  // single-origin model cannot distinguish between a loan to the variable's
  // storage and a loan to what it points to. A multi-origin model would be
  // required for this.
  if (!isPointerType(DRE->getType())) {
    if (const Loan *L = createLoan(FactMgr, DRE)) {
      OriginID ExprOID = FactMgr.getOriginMgr().getOrCreate(*DRE);
      CurrentBlockFacts.push_back(
          FactMgr.createFact<IssueFact>(L->ID, ExprOID));
    }
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
      isa_and_present<CXXConversionDecl>(MCE->getCalleeDecl())) {
    // The argument is the implicit object itself.
    handleFunctionCall(MCE, MCE->getMethodDecl(),
                       {MCE->getImplicitObjectArgument()},
                       /*IsGslConstruction=*/true);
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
  FactMgr.getOriginMgr().getOrCreate(*N);
}

void FactsGenerator::VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
  if (!hasOrigin(ICE))
    return;
  // An ImplicitCastExpr node itself gets an origin, which flows from the
  // origin of its sub-expression (after stripping its own parens/casts).
  killAndFlowOrigin(*ICE, *ICE->getSubExpr());
}

void FactsGenerator::VisitUnaryOperator(const UnaryOperator *UO) {
  if (UO->getOpcode() == UO_AddrOf) {
    const Expr *SubExpr = UO->getSubExpr();
    // Taking address of a pointer-type expression is not yet supported and
    // will be supported in multi-origin model.
    if (isPointerType(SubExpr->getType()))
      return;
    // The origin of an address-of expression (e.g., &x) is the origin of
    // its sub-expression (x). This fact will cause the dataflow analysis
    // to propagate any loans held by the sub-expression's origin to the
    // origin of this UnaryOperator expression.
    killAndFlowOrigin(*UO, *SubExpr);
  }
}

void FactsGenerator::VisitReturnStmt(const ReturnStmt *RS) {
  if (const Expr *RetExpr = RS->getRetValue()) {
    if (hasOrigin(RetExpr)) {
      OriginID OID = FactMgr.getOriginMgr().getOrCreate(*RetExpr);
      CurrentBlockFacts.push_back(FactMgr.createFact<ReturnOfOriginFact>(OID));
    }
  }
}

void FactsGenerator::VisitBinaryOperator(const BinaryOperator *BO) {
  if (BO->isAssignmentOp())
    handleAssignment(BO->getLHS(), BO->getRHS());
}

void FactsGenerator::VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
  // Assignment operators have special "kill-then-propagate" semantics
  // and are handled separately.
  if (OCE->isAssignmentOp() && OCE->getNumArgs() == 2) {
    handleAssignment(OCE->getArg(0), OCE->getArg(1));
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
  if (!hasOrigin(ILE))
    return;
  // For list initialization with a single element, like `View{...}`, the
  // origin of the list itself is the origin of its single element.
  if (ILE->getNumInits() == 1)
    killAndFlowOrigin(*ILE, *ILE->getInit(0));
}

void FactsGenerator::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *MTE) {
  if (!hasOrigin(MTE))
    return;
  // A temporary object's origin is the same as the origin of the
  // expression that initializes it.
  killAndFlowOrigin(*MTE, *MTE->getSubExpr());
}

void FactsGenerator::handleDestructor(const CFGAutomaticObjDtor &DtorOpt) {
  /// TODO: Also handle trivial destructors (e.g., for `int`
  /// variables) which will never have a CFGAutomaticObjDtor node.
  /// TODO: Handle loans to temporaries.
  /// TODO: Consider using clang::CFG::BuildOptions::AddLifetime to reuse the
  /// lifetime ends.
  const VarDecl *DestructedVD = DtorOpt.getVarDecl();
  if (!DestructedVD)
    return;
  // Iterate through all loans to see if any expire.
  /// TODO(opt): Do better than a linear search to find loans associated with
  /// 'DestructedVD'.
  for (const Loan &L : FactMgr.getLoanMgr().getLoans()) {
    const AccessPath &LoanPath = L.Path;
    // Check if the loan is for a stack variable and if that variable
    // is the one being destructed.
    if (LoanPath.D == DestructedVD)
      CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
          L.ID, DtorOpt.getTriggerStmt()->getEndLoc()));
  }
}

void FactsGenerator::handleGSLPointerConstruction(const CXXConstructExpr *CCE) {
  assert(isGslPointerType(CCE->getType()));
  if (CCE->getNumArgs() != 1)
    return;
  if (hasOrigin(CCE->getArg(0)))
    killAndFlowOrigin(*CCE, *CCE->getArg(0));
  else
    // This could be a new borrow.
    handleFunctionCall(CCE, CCE->getConstructor(),
                       {CCE->getArgs(), CCE->getNumArgs()},
                       /*IsGslConstruction=*/true);
}

/// Checks if a call-like expression creates a borrow by passing a value to a
/// reference parameter, creating an IssueFact if it does.
/// \param IsGslConstruction True if this is a GSL construction where all
///   argument origins should flow to the returned origin.
void FactsGenerator::handleFunctionCall(const Expr *Call,
                                        const FunctionDecl *FD,
                                        ArrayRef<const Expr *> Args,
                                        bool IsGslConstruction) {
  // Ignore functions returning values with no origin.
  if (!FD || !hasOrigin(Call))
    return;
  auto IsArgLifetimeBound = [FD](unsigned I) -> bool {
    const ParmVarDecl *PVD = nullptr;
    if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
        Method && Method->isInstance()) {
      if (I == 0)
        // For the 'this' argument, the attribute is on the method itself.
        return implicitObjectParamIsLifetimeBound(Method);
      if ((I - 1) < Method->getNumParams())
        // For explicit arguments, find the corresponding parameter
        // declaration.
        PVD = Method->getParamDecl(I - 1);
    } else if (I < FD->getNumParams())
      // For free functions or static methods.
      PVD = FD->getParamDecl(I);
    return PVD ? PVD->hasAttr<clang::LifetimeBoundAttr>() : false;
  };
  if (Args.empty())
    return;
  bool killedSrc = false;
  for (unsigned I = 0; I < Args.size(); ++I)
    if (IsGslConstruction || IsArgLifetimeBound(I)) {
      if (!killedSrc) {
        killedSrc = true;
        killAndFlowOrigin(*Call, *Args[I]);
      } else
        flowOrigin(*Call, *Args[I]);
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

void FactsGenerator::handleAssignment(const Expr *LHSExpr,
                                      const Expr *RHSExpr) {
  if (!hasOrigin(LHSExpr))
    return;
  // Find the underlying variable declaration for the left-hand side.
  if (const auto *DRE_LHS =
          dyn_cast<DeclRefExpr>(LHSExpr->IgnoreParenImpCasts())) {
    markUseAsWrite(DRE_LHS);
    if (const auto *VD_LHS = dyn_cast<ValueDecl>(DRE_LHS->getDecl())) {
      // Kill the old loans of the destination origin and flow the new loans
      // from the source origin.
      killAndFlowOrigin(*VD_LHS, *RHSExpr);
    }
  }
}

// A DeclRefExpr will be treated as a use of the referenced decl. It will be
// checked for use-after-free unless it is later marked as being written to
// (e.g. on the left-hand side of an assignment).
void FactsGenerator::handleUse(const DeclRefExpr *DRE) {
  if (isPointerType(DRE->getType())) {
    UseFact *UF = FactMgr.createFact<UseFact>(DRE);
    CurrentBlockFacts.push_back(UF);
    assert(!UseFacts.contains(DRE));
    UseFacts[DRE] = UF;
  }
}

void FactsGenerator::markUseAsWrite(const DeclRefExpr *DRE) {
  if (!isPointerType(DRE->getType()))
    return;
  assert(UseFacts.contains(DRE));
  UseFacts[DRE]->markAsWritten();
}

} // namespace clang::lifetimes::internal

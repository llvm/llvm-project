//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerToRefCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

/// Visitor that classifies how a pointer parameter is used in a function body.
/// It detects:
///   - Dereferences (operator*, operator->)
///   - Null checks (if(ptr), ptr != nullptr, assert(ptr), etc.)
///   - Pointer arithmetic
///   - Being passed to other functions as a pointer
///   - Address-of or array subscript usage
class PointerUsageVisitor : public RecursiveASTVisitor<PointerUsageVisitor> {
public:
  PointerUsageVisitor(const ParmVarDecl *Param, ASTContext &Ctx)
      : Param(Param), Ctx(Ctx) {}

  bool isDereferenced() const { return Dereferenced; }
  bool isNullChecked() const { return NullChecked; }
  bool isUsedAsPointer() const { return UsedAsPointer; }

  bool VisitUnaryOperator(const UnaryOperator *UO) {
    const Expr *SubExpr = UO->getSubExpr();
    if (UO->getOpcode() == UO_Deref && refersToParam(SubExpr))
      Dereferenced = true;
    if (UO->getOpcode() == UO_AddrOf && refersToParam(SubExpr))
      UsedAsPointer = true;
    return true;
  }

  bool VisitMemberExpr(const MemberExpr *ME) {
    if (ME->isArrow() && refersToParam(ME->getBase()))
      Dereferenced = true;
    return true;
  }

  // Detect null checks: if(ptr), ptr == nullptr, !ptr, etc.
  bool VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
    if (ICE->getCastKind() == CK_PointerToBoolean &&
        refersToParam(ICE->getSubExpr()))
      NullChecked = true;
    return true;
  }

  bool VisitBinaryOperator(const BinaryOperator *BO) {
    const Expr *LHS = BO->getLHS()->IgnoreImplicit();
    const Expr *RHS = BO->getRHS()->IgnoreImplicit();

    // ptr == nullptr, nullptr == ptr, ptr != nullptr, etc.
    if (BO->isEqualityOp()) {
      const bool LHSIsParam = refersToParam(LHS);
      const bool RHSIsParam = refersToParam(RHS);
      if ((LHSIsParam && RHS->isNullPointerConstant(
                             Ctx, Expr::NPC_ValueDependentIsNotNull)) ||
          (RHSIsParam &&
           LHS->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull)))
        NullChecked = true;
      // Comparing pointer to another non-null pointer (e.g. p == q).
      else if (LHSIsParam || RHSIsParam)
        UsedAsPointer = true;
    }

    // Pointer comparison (relational).
    if (BO->isRelationalOp() && (refersToParam(LHS) || refersToParam(RHS)))
      UsedAsPointer = true;

    // Pointer arithmetic: ptr + n, ptr - n.
    if ((BO->getOpcode() == BO_Add || BO->getOpcode() == BO_Sub) &&
        (refersToParam(LHS) || refersToParam(RHS)))
      UsedAsPointer = true;

    // Reassignment: ptr = something.
    if (BO->getOpcode() == BO_Assign && refersToParam(LHS))
      UsedAsPointer = true;

    return true;
  }

  // Detect being passed to a function expecting a pointer.
  bool VisitCallExpr(const CallExpr *CE) {
    for (unsigned I = 0; I < CE->getNumArgs(); ++I)
      if (refersToParam(CE->getArg(I)->IgnoreImplicit()))
        UsedAsPointer = true;
    return true;
  }

  // Detect array subscript: ptr[i]
  bool VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE) {
    if (refersToParam(ASE->getBase()->IgnoreImplicit()))
      UsedAsPointer = true;
    return true;
  }

  // Detect delete expression: delete ptr.
  bool VisitCXXDeleteExpr(const CXXDeleteExpr *DE) {
    if (refersToParam(DE->getArgument()))
      UsedAsPointer = true;
    return true;
  }

  // Detect return of pointer value.
  bool VisitReturnStmt(const ReturnStmt *RS) {
    if (RS->getRetValue() && refersToParam(RS->getRetValue()))
      UsedAsPointer = true;
    return true;
  }

  // Detect pointer stored to a variable: int *q = ptr;
  bool VisitVarDecl(const VarDecl *VD) {
    if (VD != Param && VD->hasInit() && refersToParam(VD->getInit()))
      UsedAsPointer = true;
    return true;
  }

  // Detect pointer passed to constructor: SomeClass obj(ptr);
  bool VisitCXXConstructExpr(const CXXConstructExpr *CE) {
    for (unsigned I = 0; I < CE->getNumArgs(); ++I)
      if (refersToParam(CE->getArg(I)))
        UsedAsPointer = true;
    return true;
  }

  // Detect pointer captured by lambda.
  bool TraverseLambdaExpr(LambdaExpr *LE) {
    for (const LambdaCapture &Capture : LE->captures())
      if (Capture.capturesVariable() && Capture.getCapturedVar() == Param)
        UsedAsPointer = true;
    // Don't descend into the lambda body -- it's a different scope.
    return true;
  }

  // Skip unevaluated contexts like sizeof/alignof.
  bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
    return true;
  }

  // Skip unevaluated context: decltype(P).
  bool TraverseDecltypeTypeLoc(DecltypeTypeLoc TL, bool = false) {
    return true;
  }

private:
  bool refersToParam(const Expr *E) const {
    if (!E)
      return false;
    const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    return DRE && DRE->getDecl() == Param;
  }

  const ParmVarDecl *Param;
  ASTContext &Ctx;
  bool Dereferenced = false;
  bool NullChecked = false;
  bool UsedAsPointer = false;
};

AST_MATCHER(CXXMethodDecl, isOverloadedOperator) {
  return Node.isOverloadedOperator();
}

AST_MATCHER(NamedDecl, hasIdentifier) {
  return Node.getIdentifier() != nullptr;
}

AST_MATCHER(ParmVarDecl, pointsToCompleteType) {
  const auto *PT = Node.getType()->getAs<PointerType>();
  return PT && !PT->getPointeeType()->isIncompleteType();
}

} // namespace

void PointerToRefCheck::registerMatchers(MatchFinder *Finder) {
  // Match pointer parameters in non-trivial function definitions,
  // excluding void pointers and function pointers.
  const auto PointerParam = parmVarDecl(
      hasIdentifier(), pointsToCompleteType(),
      hasType(pointerType(pointee(unless(voidType()), unless(functionType())))),
      decl().bind("param"));
  Finder->addMatcher(functionDecl(isDefinition(), unless(isImplicit()),
                                  unless(isDeleted()), unless(isMain()),
                                  unless(isExternC()),
                                  unless(cxxMethodDecl(isVirtual())),
                                  unless(cxxMethodDecl(isOverloadedOperator())),
                                  has(typeLoc(forEach(PointerParam))))
                         .bind("func"),
                     this);
}

void PointerToRefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  assert(Func && Param && Func->hasBody());

  // Analyze usage in the body.
  PointerUsageVisitor Visitor(Param, *Result.Context);
  Visitor.TraverseStmt(Func->getBody());

  // Only flag if the pointer is dereferenced but never null-checked
  // and not used as a raw pointer (arithmetic, passed to functions, etc.).
  if (Visitor.isDereferenced() && !Visitor.isNullChecked() &&
      !Visitor.isUsedAsPointer()) {
    diag(Param->getLocation(),
         "pointer parameter '%0' can be a reference; it is always "
         "dereferenced but never checked for null")
        << Param->getName();
  }
}

} // namespace clang::tidy::readability

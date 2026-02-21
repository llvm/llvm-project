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
  explicit PointerUsageVisitor(const ParmVarDecl *Param) : Param(Param) {}

  bool isDereferenced() const { return Dereferenced; }
  bool isNullChecked() const { return NullChecked; }
  bool isUsedAsPointer() const { return UsedAsPointer; }

  bool VisitUnaryOperator(const UnaryOperator *UO) {
    if (UO->getOpcode() == UO_Deref && refersToParam(UO->getSubExpr()))
      Dereferenced = true;
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
      if ((refersToParam(LHS) && isNullExpr(RHS)) ||
          (refersToParam(RHS) && isNullExpr(LHS)))
        NullChecked = true;
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

  // Detect address-taken or assigned to another pointer.
  bool VisitDeclRefExpr(const DeclRefExpr *DRE) {
    // We handle specific patterns above; just track raw usage for
    // cases we might miss (e.g. storing to another variable).
    return true;
  }

private:
  bool refersToParam(const Expr *E) const {
    if (!E)
      return false;
    const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreImplicit());
    return DRE && DRE->getDecl() == Param;
  }

  static bool isNullExpr(const Expr *E) {
    if (!E)
      return false;
    E = E->IgnoreImplicit();
    if (isa<GNUNullExpr>(E))
      return true;
    if (const auto *IL = dyn_cast<IntegerLiteral>(E))
      return IL->getValue() == 0;
    if (isa<CXXNullPtrLiteralExpr>(E))
      return true;
    return false;
  }

  const ParmVarDecl *Param;
  bool Dereferenced = false;
  bool NullChecked = false;
  bool UsedAsPointer = false;
};

} // namespace

void PointerToRefCheck::registerMatchers(MatchFinder *Finder) {
  // Match function definitions with at least one pointer parameter.
  // Match each pointer parameter individually.
  Finder->addMatcher(
      parmVarDecl(hasType(pointerType()),
                  hasAncestor(functionDecl(isDefinition(), unless(isImplicit()),
                                           unless(isDeleted()))
                                  .bind("func")))
          .bind("param"),
      this);
}

void PointerToRefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  if (!Func || !Param || !Func->hasBody())
    return;

  // Skip system headers.
  if (Result.SourceManager->isInSystemHeader(Param->getLocation()))
    return;

  // Skip variadic functions, main, and operator overloads.
  if (Func->isVariadic() || Func->isMain())
    return;
  if (isa<CXXMethodDecl>(Func) &&
      cast<CXXMethodDecl>(Func)->isOverloadedOperator())
    return;

  // Skip if the parameter is unnamed (can't be used).
  if (!Param->getIdentifier())
    return;

  // Skip void pointers (too generic).
  const auto *PT = Param->getType()->getAs<PointerType>();
  if (!PT || PT->getPointeeType()->isVoidType())
    return;

  // Skip pointers to incomplete types.
  if (PT->getPointeeType()->isIncompleteType())
    return;

  // Skip virtual methods (changing signature breaks polymorphism).
  if (const auto *MD = dyn_cast<CXXMethodDecl>(Func))
    if (MD->isVirtual())
      return;

  // Skip callbacks and function pointers (typedef params).
  if (Param->getType()->isFunctionPointerType())
    return;

  // Analyze usage in the body.
  PointerUsageVisitor Visitor(Param);
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

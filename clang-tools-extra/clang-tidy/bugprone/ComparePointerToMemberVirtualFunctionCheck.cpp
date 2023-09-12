//===--- ComparePointerToMemberVirtualFunctionCheck.cpp - clang-tidy ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ComparePointerToMemberVirtualFunctionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

// Matches a `UnaryOperator` whose operator is pre-increment:
AST_MATCHER(CXXMethodDecl, isVirtual) { return Node.isVirtual(); }

void ComparePointerToMemberVirtualFunctionCheck::registerMatchers(
    MatchFinder *Finder) {

  auto DirectMemberPointer = unaryOperator(
      allOf(hasOperatorName("&"),
            hasUnaryOperand(declRefExpr(to(cxxMethodDecl(isVirtual()))))));
  auto IndirectMemberPointer = ignoringImpCasts(declRefExpr());

  auto BinaryOperatorMatcher = [](auto &&Matcher) {
    return binaryOperator(allOf(hasAnyOperatorName("==", "!="),
                                hasEitherOperand(hasType(memberPointerType(
                                    pointee(functionType())))),
                                hasEitherOperand(Matcher)),
                          unless(hasEitherOperand(
                              castExpr(hasCastKind(CK_NullToMemberPointer)))));
  };

  Finder->addMatcher(
      BinaryOperatorMatcher(DirectMemberPointer).bind("direct_member_pointer"),
      this);

  Finder->addMatcher(BinaryOperatorMatcher(IndirectMemberPointer)
                         .bind("indirect_member_pointer"),
                     this);
}

static const char *const ErrorMsg =
    "A pointer to member virtual function shall only be tested for equality "
    "with null-pointer-constant.";

static const Expr *removeImplicitCast(const Expr *E) {
  while (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
    E = ICE->getSubExpr();
  return E;
}

void ComparePointerToMemberVirtualFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *DirectCompare =
          Result.Nodes.getNodeAs<BinaryOperator>("direct_member_pointer")) {
    diag(DirectCompare->getOperatorLoc(), ErrorMsg);
  } else if (const auto *IndirectCompare =
                 Result.Nodes.getNodeAs<BinaryOperator>(
                     "indirect_member_pointer")) {
    const Expr *E = removeImplicitCast(IndirectCompare->getLHS());
    if (!isa<DeclRefExpr>(E))
      E = removeImplicitCast(IndirectCompare->getRHS());
    const auto *MPT = cast<MemberPointerType>(E->getType().getCanonicalType());
    llvm::SmallVector<SourceLocation, 4U> SameSignatureVirtualMethods{};
    for (const auto *D : MPT->getClass()->getAsCXXRecordDecl()->decls())
      if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
        if (MD->isVirtual() && MD->getType() == MPT->getPointeeType())
          SameSignatureVirtualMethods.push_back(MD->getBeginLoc());
    if (!SameSignatureVirtualMethods.empty()) {
      diag(IndirectCompare->getOperatorLoc(), ErrorMsg);
      for (const auto Loc : SameSignatureVirtualMethods)
        diag(Loc, "potential member virtual function.", DiagnosticIDs::Note);
    }
  }
}

} // namespace clang::tidy::bugprone

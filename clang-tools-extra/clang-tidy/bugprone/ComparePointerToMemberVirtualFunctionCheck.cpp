//===--- ComparePointerToMemberVirtualFunctionCheck.cpp - clang-tidy ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ComparePointerToMemberVirtualFunctionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(CXXMethodDecl, isVirtual) { return Node.isVirtual(); }

static constexpr llvm::StringLiteral ErrorMsg =
    "comparing a pointer to member virtual function with other pointer is "
    "unspecified behavior, only compare it with a null-pointer constant for "
    "equality.";

} // namespace

void ComparePointerToMemberVirtualFunctionCheck::registerMatchers(
    MatchFinder *Finder) {

  auto DirectMemberVirtualFunctionPointer = unaryOperator(
      allOf(hasOperatorName("&"),
            hasUnaryOperand(declRefExpr(to(cxxMethodDecl(isVirtual()))))));
  auto IndirectMemberPointer =
      ignoringImpCasts(declRefExpr().bind("indirect_member_pointer"));

  Finder->addMatcher(
      binaryOperator(
          allOf(hasAnyOperatorName("==", "!="),
                hasEitherOperand(
                    hasType(memberPointerType(pointee(functionType())))),
                anyOf(hasEitherOperand(DirectMemberVirtualFunctionPointer),
                      hasEitherOperand(IndirectMemberPointer)),
                unless(hasEitherOperand(
                    castExpr(hasCastKind(CK_NullToMemberPointer))))))
          .bind("binary_operator"),
      this);
}

void ComparePointerToMemberVirtualFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BO = Result.Nodes.getNodeAs<BinaryOperator>("binary_operator");
  const auto *DRE =
      Result.Nodes.getNodeAs<DeclRefExpr>("indirect_member_pointer");

  if (DRE == nullptr) {
    // compare with pointer to member virtual function.
    diag(BO->getOperatorLoc(), ErrorMsg);
    return;
  }
  // compare with variable which type is pointer to member function.
  llvm::SmallVector<SourceLocation, 12U> SameSignatureVirtualMethods{};
  const auto *MPT = cast<MemberPointerType>(DRE->getType().getCanonicalType());
  const CXXRecordDecl *RD = MPT->getMostRecentCXXRecordDecl();
  if (RD == nullptr)
    return;

  constexpr bool StopVisit = false;

  auto VisitSameSignatureVirtualMethods =
      [&](const CXXRecordDecl *CurrentRecordDecl) -> bool {
    bool Ret = !StopVisit;
    for (const auto *MD : CurrentRecordDecl->methods()) {
      if (MD->isVirtual() && MD->getType() == MPT->getPointeeType()) {
        SameSignatureVirtualMethods.push_back(MD->getBeginLoc());
        Ret = StopVisit;
      }
    }
    return Ret;
  };

  if (StopVisit != VisitSameSignatureVirtualMethods(RD)) {
    RD->forallBases(VisitSameSignatureVirtualMethods);
  }

  if (!SameSignatureVirtualMethods.empty()) {
    diag(BO->getOperatorLoc(), ErrorMsg);
    for (const auto Loc : SameSignatureVirtualMethods)
      diag(Loc, "potential member virtual function is declared here.",
           DiagnosticIDs::Note);
  }
}

} // namespace clang::tidy::bugprone

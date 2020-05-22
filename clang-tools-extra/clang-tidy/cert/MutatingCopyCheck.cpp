//===--- MutatingCopyCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MutatingCopyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

static constexpr llvm::StringLiteral SourceDeclName = "ChangedPVD";
static constexpr llvm::StringLiteral MutatingOperatorName = "MutatingOp";
static constexpr llvm::StringLiteral MutatingCallName = "MutatingCall";

void MutatingCopyCheck::registerMatchers(MatchFinder *Finder) {
  const auto MemberExprOrSourceObject = anyOf(
      memberExpr(),
      declRefExpr(to(decl(equalsBoundNode(std::string(SourceDeclName))))));

  const auto IsPartOfSource =
      allOf(unless(hasDescendant(expr(unless(MemberExprOrSourceObject)))),
            MemberExprOrSourceObject);

  const auto IsSourceMutatingAssignment = traverse(
      ast_type_traits::TK_AsIs,
      expr(anyOf(binaryOperator(isAssignmentOperator(), hasLHS(IsPartOfSource))
                     .bind(MutatingOperatorName),
                 cxxOperatorCallExpr(isAssignmentOperator(),
                                     hasArgument(0, IsPartOfSource))
                     .bind(MutatingOperatorName))));

  const auto MemberExprOrSelf = anyOf(memberExpr(), cxxThisExpr());

  const auto IsPartOfSelf = allOf(
      unless(hasDescendant(expr(unless(MemberExprOrSelf)))), MemberExprOrSelf);

  const auto IsSelfMutatingAssignment =
      expr(anyOf(binaryOperator(isAssignmentOperator(), hasLHS(IsPartOfSelf)),
                 cxxOperatorCallExpr(isAssignmentOperator(),
                                     hasArgument(0, IsPartOfSelf))));

  const auto IsSelfMutatingMemberFunction =
      functionDecl(hasBody(hasDescendant(IsSelfMutatingAssignment)));

  const auto IsSourceMutatingMemberCall =
      cxxMemberCallExpr(on(IsPartOfSource),
                        callee(IsSelfMutatingMemberFunction))
          .bind(MutatingCallName);

  const auto MutatesSource = allOf(
      hasParameter(
          0, parmVarDecl(hasType(lValueReferenceType())).bind(SourceDeclName)),
      anyOf(forEachDescendant(IsSourceMutatingAssignment),
            forEachDescendant(IsSourceMutatingMemberCall)));

  Finder->addMatcher(cxxConstructorDecl(isCopyConstructor(), MutatesSource),
                     this);

  Finder->addMatcher(cxxMethodDecl(isCopyAssignmentOperator(), MutatesSource),
                     this);
}

void MutatingCopyCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MemberCall =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>(MutatingCallName))
    diag(MemberCall->getBeginLoc(), "call mutates copied object");
  else if (const auto *Assignment =
               Result.Nodes.getNodeAs<Expr>(MutatingOperatorName))
    diag(Assignment->getBeginLoc(), "mutating copied object");
}

} // namespace cert
} // namespace tidy
} // namespace clang

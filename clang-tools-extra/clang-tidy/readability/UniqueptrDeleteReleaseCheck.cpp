//===--- UniqueptrDeleteReleaseCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UniqueptrDeleteReleaseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void UniqueptrDeleteReleaseCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "PreferResetCall", PreferResetCall);
}

UniqueptrDeleteReleaseCheck::UniqueptrDeleteReleaseCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      PreferResetCall(Options.get("PreferResetCall", false)) {}

void UniqueptrDeleteReleaseCheck::registerMatchers(MatchFinder *Finder) {

  auto UniquePtrWithDefaultDelete = classTemplateSpecializationDecl(
      hasName("::std::unique_ptr"),
      hasTemplateArgument(1, refersToType(hasDeclaration(cxxRecordDecl(
                                 hasName("::std::default_delete"))))));

  Finder->addMatcher(
      cxxDeleteExpr(
          unless(isInTemplateInstantiation()),
          has(cxxMemberCallExpr(
                  callee(memberExpr(hasObjectExpression(anyOf(
                                        hasType(UniquePtrWithDefaultDelete),
                                        hasType(pointsTo(
                                            UniquePtrWithDefaultDelete)))),
                                    member(cxxMethodDecl(hasName("release"))))
                             .bind("release_expr")))
                  .bind("release_call")))
          .bind("delete"),
      this);
}

void UniqueptrDeleteReleaseCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *DeleteExpr = Result.Nodes.getNodeAs<CXXDeleteExpr>("delete");
  const auto *ReleaseExpr = Result.Nodes.getNodeAs<MemberExpr>("release_expr");
  const auto *ReleaseCallExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("release_call");

  if (ReleaseExpr->getBeginLoc().isMacroID())
    return;

  auto D =
      diag(DeleteExpr->getBeginLoc(), "prefer '%select{= nullptr|reset()}0' "
                                      "to reset 'unique_ptr<>' objects");
  D << PreferResetCall << DeleteExpr->getSourceRange()
    << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
           DeleteExpr->getBeginLoc(),
           DeleteExpr->getArgument()->getBeginLoc()));
  if (PreferResetCall) {
    D << FixItHint::CreateReplacement(ReleaseExpr->getMemberLoc(), "reset");
  } else {
    if (ReleaseExpr->isArrow())
      D << FixItHint::CreateInsertion(ReleaseExpr->getBase()->getBeginLoc(),
                                      "*");
    D << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(ReleaseExpr->getOperatorLoc(),
                                       ReleaseCallExpr->getEndLoc()),
        " = nullptr");
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang

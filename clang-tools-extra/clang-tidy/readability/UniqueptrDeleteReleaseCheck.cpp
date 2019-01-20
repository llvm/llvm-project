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
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void UniqueptrDeleteReleaseCheck::registerMatchers(MatchFinder *Finder) {
  auto IsSusbstituted = qualType(anyOf(
      substTemplateTypeParmType(), hasDescendant(substTemplateTypeParmType())));

  auto UniquePtrWithDefaultDelete = classTemplateSpecializationDecl(
      hasName("std::unique_ptr"),
      hasTemplateArgument(1, refersToType(qualType(hasDeclaration(cxxRecordDecl(
                                 hasName("std::default_delete")))))));

  Finder->addMatcher(
      cxxDeleteExpr(has(ignoringParenImpCasts(cxxMemberCallExpr(
                        on(expr(hasType(UniquePtrWithDefaultDelete),
                                unless(hasType(IsSusbstituted)))
                               .bind("uptr")),
                        callee(cxxMethodDecl(hasName("release")))))))
          .bind("delete"),
      this);
}

void UniqueptrDeleteReleaseCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PtrExpr = Result.Nodes.getNodeAs<Expr>("uptr");
  const auto *DeleteExpr = Result.Nodes.getNodeAs<Expr>("delete");

  if (PtrExpr->getBeginLoc().isMacroID())
    return;

  // Ignore dependent types.
  // It can give us false positives, so we go with false negatives instead to
  // be safe.
  if (PtrExpr->getType()->isDependentType())
    return;

  SourceLocation AfterPtr = Lexer::getLocForEndOfToken(
      PtrExpr->getEndLoc(), 0, *Result.SourceManager, getLangOpts());

  diag(DeleteExpr->getBeginLoc(),
       "prefer '= nullptr' to 'delete x.release()' to reset unique_ptr<> "
       "objects")
      << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
             DeleteExpr->getBeginLoc(), PtrExpr->getBeginLoc()))
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(AfterPtr, DeleteExpr->getEndLoc()),
             " = nullptr");
}

} // namespace readability
} // namespace tidy
} // namespace clang

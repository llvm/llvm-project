//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TriviallyDestructibleCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;
using namespace clang::tidy::matchers;

namespace clang::tidy::performance {

namespace {

AST_MATCHER(Decl, isFirstDecl) { return Node.isFirstDecl(); }

AST_MATCHER_P(CXXRecordDecl, hasBase, Matcher<QualType>, InnerMatcher) {
  for (const CXXBaseSpecifier &BaseSpec : Node.bases()) {
    QualType BaseType = BaseSpec.getType();
    if (InnerMatcher.matches(BaseType, Finder, Builder))
      return true;
  }
  return false;
}

} // namespace

void TriviallyDestructibleCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxDestructorDecl(
          isDefaulted(),
          unless(anyOf(isFirstDecl(), isVirtual(),
                       ofClass(cxxRecordDecl(
                           anyOf(hasBase(unless(isTriviallyDestructible())),
                                 has(fieldDecl(unless(
                                     hasType(isTriviallyDestructible()))))))))))
          .bind("decl"),
      this);
}

void TriviallyDestructibleCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXDestructorDecl>("decl");

  // Get locations of both first and out-of-line declarations.
  SourceManager &SM = *Result.SourceManager;
  const auto *FirstDecl = cast<CXXMethodDecl>(MatchedDecl->getFirstDecl());
  const SourceLocation FirstDeclEnd = utils::lexer::findNextTerminator(
      FirstDecl->getEndLoc(), SM, getLangOpts());
  const CharSourceRange SecondDeclRange = CharSourceRange::getTokenRange(
      MatchedDecl->getBeginLoc(),
      utils::lexer::findNextTerminator(MatchedDecl->getEndLoc(), SM,
                                       getLangOpts()));
  if (FirstDeclEnd.isInvalid() || SecondDeclRange.isInvalid())
    return;

  // Report diagnostic.
  diag(FirstDecl->getLocation(),
       "class %0 can be made trivially destructible by defaulting the "
       "destructor on its first declaration")
      << FirstDecl->getParent()
      << FixItHint::CreateInsertion(FirstDeclEnd, " = default")
      << FixItHint::CreateRemoval(SecondDeclRange);
  diag(MatchedDecl->getLocation(), "destructor definition is here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::performance

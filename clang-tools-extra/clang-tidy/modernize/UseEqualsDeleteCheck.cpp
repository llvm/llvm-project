//===--- UseEqualsDeleteCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseEqualsDeleteCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

static const char SpecialFunction[] = "SpecialFunction";
static const char DeletedNotPublic[] = "DeletedNotPublic";

void UseEqualsDeleteCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseEqualsDeleteCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  auto PrivateSpecialFn = cxxMethodDecl(
      isPrivate(),
      anyOf(cxxConstructorDecl(anyOf(isDefaultConstructor(),
                                     isCopyConstructor(), isMoveConstructor())),
            cxxMethodDecl(
                anyOf(isCopyAssignmentOperator(), isMoveAssignmentOperator())),
            cxxDestructorDecl()));

  Finder->addMatcher(
      cxxMethodDecl(
          PrivateSpecialFn,
          unless(anyOf(hasBody(stmt()), isDefaulted(), isDeleted(),
                       ast_matchers::isTemplateInstantiation(),
                       // Ensure that all methods except private special member
                       // functions are defined.
                       hasParent(cxxRecordDecl(hasMethod(unless(
                           anyOf(PrivateSpecialFn, hasBody(stmt()), isPure(),
                                 isDefaulted(), isDeleted()))))))))
          .bind(SpecialFunction),
      this);

  Finder->addMatcher(
      cxxMethodDecl(isDeleted(), unless(isPublic())).bind(DeletedNotPublic),
      this);
}

void UseEqualsDeleteCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Func =
          Result.Nodes.getNodeAs<CXXMethodDecl>(SpecialFunction)) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        Func->getEndLoc(), 0, *Result.SourceManager, getLangOpts());

    if (Func->getLocation().isMacroID() && IgnoreMacros)
      return;
    // FIXME: Improve FixItHint to make the method public.
    diag(Func->getLocation(),
         "use '= delete' to prohibit calling of a special member function")
        << FixItHint::CreateInsertion(EndLoc, " = delete");
  } else if (const auto *Func =
                 Result.Nodes.getNodeAs<CXXMethodDecl>(DeletedNotPublic)) {
    // Ignore this warning in macros, since it's extremely noisy in code using
    // DISALLOW_COPY_AND_ASSIGN-style macros and there's no easy way to
    // automatically fix the warning when macros are in play.
    if (Func->getLocation().isMacroID() && IgnoreMacros)
      return;
    // FIXME: Add FixItHint to make the method public.
    diag(Func->getLocation(), "deleted member function should be public");
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang

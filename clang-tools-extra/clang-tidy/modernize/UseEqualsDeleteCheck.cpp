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

namespace clang::tidy::modernize {

namespace {
AST_MATCHER(FunctionDecl, hasAnyDefinition) {
  if (Node.hasBody() || Node.isPureVirtual() || Node.isDefaulted() ||
      Node.isDeleted())
    return true;

  if (const FunctionDecl *Definition = Node.getDefinition())
    if (Definition->hasBody() || Definition->isPureVirtual() ||
        Definition->isDefaulted() || Definition->isDeleted())
      return true;

  return false;
}

AST_MATCHER(Decl, isUsed) { return Node.isUsed(); }

AST_MATCHER(CXXMethodDecl, isSpecialFunction) {
  if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(&Node))
    return Constructor->isDefaultConstructor() ||
           Constructor->isCopyOrMoveConstructor();

  return isa<CXXDestructorDecl>(Node) || Node.isCopyAssignmentOperator() ||
         Node.isMoveAssignmentOperator();
}
} // namespace

static const char SpecialFunction[] = "SpecialFunction";
static const char DeletedNotPublic[] = "DeletedNotPublic";

UseEqualsDeleteCheck::UseEqualsDeleteCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", true)) {}

void UseEqualsDeleteCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseEqualsDeleteCheck::registerMatchers(MatchFinder *Finder) {
  auto PrivateSpecialFn = cxxMethodDecl(isPrivate(), isSpecialFunction());

  Finder->addMatcher(
      cxxMethodDecl(
          PrivateSpecialFn, unless(hasAnyDefinition()), unless(isUsed()),
          // Ensure that all methods except private special member functions are
          // defined.
          unless(ofClass(hasMethod(cxxMethodDecl(unless(PrivateSpecialFn),
                                                 unless(hasAnyDefinition()))))))
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

    if (IgnoreMacros && Func->getLocation().isMacroID())
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
    if (IgnoreMacros && Func->getLocation().isMacroID())
      return;
    // FIXME: Add FixItHint to make the method public.
    diag(Func->getLocation(), "deleted member function should be public");
  }
}

} // namespace clang::tidy::modernize

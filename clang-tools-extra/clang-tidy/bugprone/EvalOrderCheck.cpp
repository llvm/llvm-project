//===--- EvalOrderCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvalOrderCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>

using namespace clang::ast_matchers;
using ::clang::ast_matchers::internal::Matcher;

namespace clang::tidy::bugprone {

EvalOrderCheck::EvalOrderCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

bool EvalOrderCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

void EvalOrderCheck::registerMatchers(MatchFinder *Finder) {
  auto Ctor = cxxConstructExpr(unless(isListInitialization())).bind("ctor");
  auto Fun = callExpr().bind("fun");
  auto Mut = unless(hasDeclaration(cxxMethodDecl(isConst())));
  auto Mc1 = hasDescendant(
      cxxMemberCallExpr(
          Mut, callee(cxxMethodDecl(hasParent(recordDecl().bind("rd1")))))
          .bind("mc1"));
  auto Mc2 = hasDescendant(
      cxxMemberCallExpr(
          unless(equalsBoundNode("mc1")),
          callee(cxxMethodDecl(hasParent(recordDecl(equalsBoundNode("rd1"))))))
          .bind("mc2"));
  Finder->addMatcher(expr(anyOf(Ctor, Fun), allOf(Mc1, Mc2)), this);
}

void EvalOrderCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<Expr>("ctor");
  const auto *Fun = Result.Nodes.getNodeAs<Expr>("fun");

  if (Ctor) {
    diag(Ctor->getExprLoc(), "Order of evaluation of constructor "
                             "arguments is unspecified.");
  } else {
    diag(Fun->getExprLoc(), "Order of evaluation of function "
                            "arguments is unspecified. ");
  }
}

} // namespace clang::tidy::bugprone

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantParenthesesCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER_P(ParenExpr, subExpr, ast_matchers::internal::Matcher<Expr>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
}

AST_MATCHER(ParenExpr, isInMacro) {
  const Expr *E = Node.getSubExpr();
  return Node.getLParen().isMacroID() || Node.getRParen().isMacroID() ||
         E->getBeginLoc().isMacroID() || E->getEndLoc().isMacroID();
}

} // namespace

RedundantParenthesesCheck::RedundantParenthesesCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedDecls(utils::options::parseStringList(
          Options.get("AllowedDecls", "std::max;std::min"))) {}

void RedundantParenthesesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedDecls",
                utils::options::serializeStringList(AllowedDecls));
}

void RedundantParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  const auto ConstantExpr =
      expr(anyOf(integerLiteral(), floatLiteral(), characterLiteral(),
                 cxxBoolLiteral(), stringLiteral(), cxxNullPtrLiteralExpr()));
  Finder->addMatcher(
      parenExpr(
          subExpr(anyOf(parenExpr(), ConstantExpr,
                        declRefExpr(to(namedDecl(unless(
                            matchers::matchesAnyListedName(AllowedDecls))))))),
          unless(anyOf(isInMacro(),
                       // sizeof(...) is common used.
                       hasParent(unaryExprOrTypeTraitExpr()))))
          .bind("dup"),
      this);
}

void RedundantParenthesesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *PE = Result.Nodes.getNodeAs<ParenExpr>("dup");
  diag(PE->getBeginLoc(), "redundant parentheses around expression")
      << FixItHint::CreateRemoval(PE->getLParen())
      << FixItHint::CreateRemoval(PE->getRParen());
}

} // namespace clang::tidy::readability

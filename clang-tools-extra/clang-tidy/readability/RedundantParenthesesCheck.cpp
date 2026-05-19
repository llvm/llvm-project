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

// Matches if this ParenExpr is the base object of a member access
// (.field or ->field), possibly through implicit casts or temporary
// materialization that Clang inserts (e.g. qualification casts for const
// methods, MaterializeTemporaryExpr for prvalue bases).
AST_MATCHER(ParenExpr, isBaseOfMemberAccess) {
  if (!isa<CXXOperatorCallExpr>(Node.getSubExpr()->IgnoreImpCasts()))
    return false;
  ASTContext &Ctx = Finder->getASTContext();
  DynTypedNodeList Parents = Ctx.getParents(Node);
  while (!Parents.empty()) {
    const auto *E = Parents[0].get<Expr>();
    if (!E)
      break;
    if (isa<MemberExpr>(E))
      return true;
    if (!isa<ImplicitCastExpr>(E) && !isa<MaterializeTemporaryExpr>(E))
      break;
    Parents = Ctx.getParents(*E);
  }
  return false;
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
      parenExpr(subExpr(anyOf(
                    parenExpr(), ConstantExpr,
                    declRefExpr(to(namedDecl(unless(
                        matchers::matchesAnyListedRegexName(AllowedDecls))))),
                    memberExpr(), callExpr())),
                unless(anyOf(isInMacro(),
                             // sizeof(...) is common used.
                             hasParent(unaryExprOrTypeTraitExpr()),
                             // Don't warn when parens are the object of a
                             // member access: (expr).foo or (expr)->foo.
                             isBaseOfMemberAccess())))
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

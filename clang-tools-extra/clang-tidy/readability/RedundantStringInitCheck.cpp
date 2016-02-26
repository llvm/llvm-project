//===- RedundantStringInitCheck.cpp - clang-tidy ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantStringInitCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

AST_MATCHER(StringLiteral, lengthIsZero) { return Node.getLength() == 0; }

} // namespace

void RedundantStringInitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto StringCtorExpr = cxxConstructExpr(
      hasDeclaration(cxxMethodDecl(hasName("basic_string"))),
      argumentCountIs(2),
      hasArgument(0, ignoringParenImpCasts(stringLiteral(lengthIsZero()))),
      hasArgument(1, cxxDefaultArgExpr()));

  // string foo = "";
  // OR
  // string bar("");
  Finder->addMatcher(
      namedDecl(varDecl(hasType(cxxRecordDecl(hasName("basic_string"))),
                        hasInitializer(
                            expr(anyOf(StringCtorExpr,
                                       exprWithCleanups(has(expr(anyOf(
                                           StringCtorExpr,
                                           cxxConstructExpr(hasArgument(
                                               0, cxxBindTemporaryExpr(has(
                                                      StringCtorExpr))))))))))
                                .bind("expr"))))
          .bind("decl"),
      this);
}

void RedundantStringInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CtorExpr = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *Decl = Result.Nodes.getNodeAs<NamedDecl>("decl");
  diag(CtorExpr->getExprLoc(), "redundant string initialization")
      << FixItHint::CreateReplacement(CtorExpr->getSourceRange(),
                                      Decl->getName());
}

} // namespace readability
} // namespace tidy
} // namespace clang

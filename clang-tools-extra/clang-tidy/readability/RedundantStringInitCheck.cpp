//===- RedundantStringInitCheck.cpp - clang-tidy ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantStringInitCheck.h"
#include "../utils/Matchers.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace readability {

void RedundantStringInitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Match string constructor.
  const auto StringConstructorExpr = expr(anyOf(
      cxxConstructExpr(argumentCountIs(1),
                       hasDeclaration(cxxMethodDecl(hasName("basic_string")))),
      // If present, the second argument is the alloc object which must not
      // be present explicitly.
      cxxConstructExpr(argumentCountIs(2),
                       hasDeclaration(cxxMethodDecl(hasName("basic_string"))),
                       hasArgument(1, cxxDefaultArgExpr()))));

  // Match a string constructor expression with an empty string literal.
  const auto EmptyStringCtorExpr = cxxConstructExpr(
      StringConstructorExpr,
      hasArgument(0, ignoringParenImpCasts(stringLiteral(hasSize(0)))));

  const auto EmptyStringCtorExprWithTemporaries =
      cxxConstructExpr(StringConstructorExpr,
                       hasArgument(0, ignoringImplicit(EmptyStringCtorExpr)));

  // Match a variable declaration with an empty string literal as initializer.
  // Examples:
  //     string foo = "";
  //     string bar("");
  Finder->addMatcher(
      namedDecl(
          varDecl(hasType(hasUnqualifiedDesugaredType(recordType(
                      hasDeclaration(cxxRecordDecl(hasName("basic_string")))))),
                  hasInitializer(expr(ignoringImplicit(anyOf(
                                          EmptyStringCtorExpr,
                                          EmptyStringCtorExprWithTemporaries)))
                                     .bind("expr"))),
          unless(parmVarDecl()))
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

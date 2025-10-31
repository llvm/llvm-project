//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousStringviewDataUsageCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

SuspiciousStringviewDataUsageCheck::SuspiciousStringviewDataUsageCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringViewTypes(utils::options::parseStringList(Options.get(
          "StringViewTypes", "::std::basic_string_view;::llvm::StringRef"))),
      AllowedCallees(
          utils::options::parseStringList(Options.get("AllowedCallees", ""))) {}

void SuspiciousStringviewDataUsageCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringViewTypes",
                utils::options::serializeStringList(StringViewTypes));
  Options.store(Opts, "AllowedCallees",
                utils::options::serializeStringList(AllowedCallees));
}

bool SuspiciousStringviewDataUsageCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind>
SuspiciousStringviewDataUsageCheck::getCheckTraversalKind() const {
  return TK_AsIs;
}

void SuspiciousStringviewDataUsageCheck::registerMatchers(MatchFinder *Finder) {

  auto AncestorCall = anyOf(
      cxxConstructExpr(), callExpr(unless(cxxOperatorCallExpr())), lambdaExpr(),
      initListExpr(
          hasType(qualType(hasCanonicalType(hasDeclaration(recordDecl()))))));

  auto DataMethod =
      cxxMethodDecl(hasName("data"),
                    ofClass(matchers::matchesAnyListedName(StringViewTypes)));

  auto SizeCall = cxxMemberCallExpr(
      callee(cxxMethodDecl(hasAnyName("size", "length"))),
      on(ignoringParenImpCasts(
          matchers::isStatementIdenticalToBoundNode("self"))));

  auto DescendantSizeCall = expr(hasDescendant(
      expr(SizeCall, hasAncestor(expr(AncestorCall).bind("ancestor-size")),
           hasAncestor(expr(equalsBoundNode("parent"),
                            equalsBoundNode("ancestor-size"))))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          on(ignoringParenImpCasts(expr().bind("self"))), callee(DataMethod),
          expr().bind("data-call"),
          hasParent(expr(anyOf(
              invocation(
                  expr().bind("parent"), unless(cxxOperatorCallExpr()),
                  hasAnyArgument(
                      ignoringParenImpCasts(equalsBoundNode("data-call"))),
                  unless(hasAnyArgument(ignoringParenImpCasts(SizeCall))),
                  unless(hasAnyArgument(DescendantSizeCall)),
                  hasDeclaration(namedDecl(
                      unless(matchers::matchesAnyListedName(AllowedCallees))))),
              initListExpr(expr().bind("parent"),
                           hasType(qualType(hasCanonicalType(hasDeclaration(
                               recordDecl(unless(matchers::matchesAnyListedName(
                                   AllowedCallees))))))),
                           unless(DescendantSizeCall)))))),
      this);
}

void SuspiciousStringviewDataUsageCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *DataCallExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("data-call");
  diag(DataCallExpr->getExprLoc(),
       "result of a `data()` call may not be null terminated, provide size "
       "information to the callee to prevent potential issues")
      << DataCallExpr->getCallee()->getSourceRange();
}

} // namespace clang::tidy::bugprone

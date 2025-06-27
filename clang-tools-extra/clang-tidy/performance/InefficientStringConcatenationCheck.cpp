//===--- InefficientStringConcatenationCheck.cpp - clang-tidy--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientStringConcatenationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void InefficientStringConcatenationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

InefficientStringConcatenationCheck::InefficientStringConcatenationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)) {}

void InefficientStringConcatenationCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto BasicStringType =
      hasType(qualType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(cxxRecordDecl(hasName("::std::basic_string")))))));

  const auto BasicStringPlusOperator = cxxOperatorCallExpr(
      hasOverloadedOperatorName("+"),
      hasAnyArgument(ignoringImpCasts(declRefExpr(BasicStringType))));

  const auto PlusOperator =
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("+"),
          hasAnyArgument(ignoringImpCasts(declRefExpr(BasicStringType))),
          hasDescendant(BasicStringPlusOperator))
          .bind("plusOperator");

  const auto AssignOperator = cxxOperatorCallExpr(
      hasOverloadedOperatorName("="),
      hasArgument(0, declRefExpr(BasicStringType,
                                 hasDeclaration(decl().bind("lhsStrT")))
                         .bind("lhsStr")),
      hasArgument(1, stmt(hasDescendant(declRefExpr(
                         hasDeclaration(decl(equalsBoundNode("lhsStrT"))))))),
      hasDescendant(BasicStringPlusOperator));

  if (StrictMode) {
    Finder->addMatcher(cxxOperatorCallExpr(anyOf(AssignOperator, PlusOperator)),
                       this);
  } else {
    Finder->addMatcher(
        cxxOperatorCallExpr(anyOf(AssignOperator, PlusOperator),
                            hasAncestor(stmt(anyOf(cxxForRangeStmt(),
                                                   whileStmt(), forStmt())))),
        this);
  }
}

void InefficientStringConcatenationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *LhsStr = Result.Nodes.getNodeAs<DeclRefExpr>("lhsStr");
  const auto *PlusOperator =
      Result.Nodes.getNodeAs<CXXOperatorCallExpr>("plusOperator");
  const char *DiagMsg =
      "string concatenation results in allocation of unnecessary temporary "
      "strings; consider using 'operator+=' or 'string::append()' instead";

  if (LhsStr)
    diag(LhsStr->getExprLoc(), DiagMsg);
  else if (PlusOperator)
    diag(PlusOperator->getExprLoc(), DiagMsg);
}

} // namespace clang::tidy::performance

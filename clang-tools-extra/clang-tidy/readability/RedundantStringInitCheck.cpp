//===- RedundantStringInitCheck.cpp - clang-tidy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantStringInitCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace readability {

const char DefaultStringNames[] = "::std::basic_string";

RedundantStringInitCheck::RedundantStringInitCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringNames(utils::options::parseStringList(
          Options.get("StringNames", DefaultStringNames))) {}

void RedundantStringInitCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringNames", DefaultStringNames);
}

void RedundantStringInitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;
  const auto hasStringTypeName = hasAnyName(
      SmallVector<StringRef, 3>(StringNames.begin(), StringNames.end()));

  // Version of StringNames with namespaces removed
  std::vector<std::string> stringNamesNoNamespace;
  for (const std::string &name : StringNames) {
    std::string::size_type colonPos = name.rfind(':');
    stringNamesNoNamespace.push_back(
        name.substr(colonPos == std::string::npos ? 0 : colonPos + 1));
  }
  const auto hasStringCtorName = hasAnyName(SmallVector<StringRef, 3>(
      stringNamesNoNamespace.begin(), stringNamesNoNamespace.end()));

  // Match string constructor.
  const auto StringConstructorExpr = expr(
      anyOf(cxxConstructExpr(argumentCountIs(1),
                             hasDeclaration(cxxMethodDecl(hasStringCtorName))),
            // If present, the second argument is the alloc object which must
            // not be present explicitly.
            cxxConstructExpr(argumentCountIs(2),
                             hasDeclaration(cxxMethodDecl(hasStringCtorName)),
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
          varDecl(
              hasType(hasUnqualifiedDesugaredType(recordType(
                  hasDeclaration(cxxRecordDecl(hasStringTypeName))))),
              hasInitializer(expr(ignoringImplicit(anyOf(
                  EmptyStringCtorExpr, EmptyStringCtorExprWithTemporaries)))))
              .bind("vardecl"),
          unless(parmVarDecl())),
      this);
}

void RedundantStringInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VDecl = Result.Nodes.getNodeAs<VarDecl>("vardecl");
  // VarDecl's getSourceRange() spans 'string foo = ""' or 'string bar("")'.
  // So start at getLocation() to span just 'foo = ""' or 'bar("")'.
  SourceRange ReplaceRange(VDecl->getLocation(), VDecl->getEndLoc());
  diag(VDecl->getLocation(), "redundant string initialization")
      << FixItHint::CreateReplacement(ReplaceRange, VDecl->getName());
}

} // namespace readability
} // namespace tidy
} // namespace clang

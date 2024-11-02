//===--- FasterStringFindCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FasterStringFindCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

namespace {

llvm::Optional<std::string> makeCharacterLiteral(const StringLiteral *Literal) {
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    Literal->outputString(OS);
  }
  // Now replace the " with '.
  auto Pos = Result.find_first_of('"');
  if (Pos == Result.npos)
    return llvm::None;
  Result[Pos] = '\'';
  Pos = Result.find_last_of('"');
  if (Pos == Result.npos)
    return llvm::None;
  Result[Pos] = '\'';
  return Result;
}

AST_MATCHER_FUNCTION(ast_matchers::internal::Matcher<Expr>,
                     hasSubstitutedType) {
  return hasType(qualType(anyOf(substTemplateTypeParmType(),
                                hasDescendant(substTemplateTypeParmType()))));
}

} // namespace

FasterStringFindCheck::FasterStringFindCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses",
                      "::std::basic_string;::std::basic_string_view"))) {}

void FasterStringFindCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
}

void FasterStringFindCheck::registerMatchers(MatchFinder *Finder) {
  const auto SingleChar =
      expr(ignoringParenCasts(stringLiteral(hasSize(1)).bind("literal")));
  const auto StringFindFunctions =
      hasAnyName("find", "rfind", "find_first_of", "find_first_not_of",
                 "find_last_of", "find_last_not_of");

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(functionDecl(StringFindFunctions).bind("func")),
          anyOf(argumentCountIs(1), argumentCountIs(2)),
          hasArgument(0, SingleChar),
          on(expr(hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                      recordDecl(hasAnyName(StringLikeClasses)))))),
                  unless(hasSubstitutedType())))),
      this);
}

void FasterStringFindCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("literal");
  const auto *FindFunc = Result.Nodes.getNodeAs<FunctionDecl>("func");

  auto Replacement = makeCharacterLiteral(Literal);
  if (!Replacement)
    return;

  diag(Literal->getBeginLoc(), "%0 called with a string literal consisting of "
                               "a single character; consider using the more "
                               "effective overload accepting a character")
      << FindFunc
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(Literal->getBeginLoc(),
                                            Literal->getEndLoc()),
             *Replacement);
}

} // namespace performance
} // namespace tidy
} // namespace clang

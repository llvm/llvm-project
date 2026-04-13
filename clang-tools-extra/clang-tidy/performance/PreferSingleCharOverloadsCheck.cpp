//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferSingleCharOverloadsCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static std::string makeCharacterLiteral(const StringLiteral *Literal) {
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    Literal->outputString(OS);
  }
  // Now replace the " with '.
  const size_t OpenPos = Result.find_first_of('"');
  assert(OpenPos != std::string::npos);
  Result[OpenPos] = '\'';

  const size_t ClosePos = Result.find_last_of('"');
  assert(ClosePos != std::string::npos);
  Result[ClosePos] = '\'';

  // "'" is OK, but ''' is not, so add a backslash
  if ((ClosePos - OpenPos) == 2 && Result[OpenPos + 1] == '\'')
    Result.replace(OpenPos + 1, 1, "\\'");

  return Result;
}

PreferSingleCharOverloadsCheck::PreferSingleCharOverloadsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringLikeClasses(utils::options::parseStringList(
          Options.get("StringLikeClasses",
                      "::std::basic_string;::std::basic_string_view"))) {}

void PreferSingleCharOverloadsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StringLikeClasses",
                utils::options::serializeStringList(StringLikeClasses));
}

void PreferSingleCharOverloadsCheck::registerMatchers(MatchFinder *Finder) {
  const auto SingleChar = expr().bind("literal");

  const auto StringExpr = expr(hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(recordDecl(hasAnyName(StringLikeClasses)))))));

  const auto InterestingStringFunction = hasAnyName(
      "find", "rfind", "find_first_of", "find_first_not_of", "find_last_of",
      "find_last_not_of", "starts_with", "ends_with", "contains", "operator+=");

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(functionDecl(InterestingStringFunction).bind("func")),
          anyOf(argumentCountIs(1), argumentCountIs(2)),
          hasArgument(0, SingleChar), on(StringExpr)),
      this);

  Finder->addMatcher(cxxOperatorCallExpr(hasOperatorName("+="),
                                         hasLHS(StringExpr), hasRHS(SingleChar),
                                         callee(functionDecl().bind("func"))),
                     this);
}

void PreferSingleCharOverloadsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FindFunc = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *Literal = Result.Nodes.getNodeAs<Expr>("literal");

  if (!utils::forAllLeavesOfTernaryTree(Literal, [](const Expr *E) {
        const auto *Literal = dyn_cast<StringLiteral>(E);
        return Literal && Literal->getLength() == 1;
      }))
    return;

  const auto Diag = diag(Literal->getBeginLoc(),
                         "%0 called with a string literal consisting of "
                         "a single character; consider using the more "
                         "efficient overload accepting a character")
                    << FindFunc;

  utils::forAllLeavesOfTernaryTree(Literal, [&](const Expr *E) {
    Diag << FixItHint::CreateReplacement(
        E->getSourceRange(), makeCharacterLiteral(cast<StringLiteral>(E)));
    return true;
  });
}

} // namespace clang::tidy::performance

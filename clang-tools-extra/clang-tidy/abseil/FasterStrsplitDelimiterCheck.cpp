//===--- FasterStrsplitDelimiterCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FasterStrsplitDelimiterCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

namespace {

AST_MATCHER(StringLiteral, lengthIsOne) { return Node.getLength() == 1; }

::internal::Matcher<Expr>
constructExprWithArg(llvm::StringRef ClassName,
                     const ::internal::Matcher<Expr> &Arg) {
  auto ConstrExpr = cxxConstructExpr(hasType(recordDecl(hasName(ClassName))),
                                     hasArgument(0, ignoringParenCasts(Arg)));

  return anyOf(ConstrExpr, cxxBindTemporaryExpr(has(ConstrExpr)));
}

::internal::Matcher<Expr>
copyConstructExprWithArg(llvm::StringRef ClassName,
                         const ::internal::Matcher<Expr> &Arg) {
  return constructExprWithArg(ClassName, constructExprWithArg(ClassName, Arg));
}

llvm::Optional<std::string> makeCharacterLiteral(const StringLiteral *Literal) {
  std::string Result;
  {
    llvm::raw_string_ostream Stream(Result);
    Literal->outputString(Stream);
  }

  // Special case: If the string contains a single quote, we just need to return
  // a character of the single quote. This is a special case because we need to
  // escape it in the character literal.
  if (Result == R"("'")")
    return std::string(R"('\'')");

  assert(Result.size() == 3 || (Result.size() == 4 && Result.substr(0, 2) == "\"\\"));

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

} // anonymous namespace

void FasterStrsplitDelimiterCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Binds to one character string literals.
  const auto SingleChar =
      expr(ignoringParenCasts(stringLiteral(lengthIsOne()).bind("Literal")));

  // Binds to a string_view (either absl or std) that was passed by value and
  // contructed from string literal.
  auto StringViewArg =
      copyConstructExprWithArg("::absl::string_view", SingleChar);

  auto ByAnyCharArg =
      expr(copyConstructExprWithArg("::absl::ByAnyChar", StringViewArg))
          .bind("ByAnyChar");

  // Find uses of absl::StrSplit(..., "x") and absl::StrSplit(...,
  // absl::ByAnyChar("x")) to transform them into absl::StrSplit(..., 'x').
  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::absl::StrSplit"))),
                              hasArgument(1, anyOf(ByAnyCharArg, SingleChar)),
                              unless(isInTemplateInstantiation()))
                         .bind("StrSplit"),
                     this);

  // Find uses of absl::MaxSplits("x", N) and
  // absl::MaxSplits(absl::ByAnyChar("x"), N) to transform them into
  // absl::MaxSplits('x', N).
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::absl::MaxSplits"))),
          hasArgument(0, anyOf(ByAnyCharArg, ignoringParenCasts(SingleChar))),
          unless(isInTemplateInstantiation())),
      this);
}

void FasterStrsplitDelimiterCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("Literal");

  if (Literal->getBeginLoc().isMacroID() || Literal->getEndLoc().isMacroID())
    return;

  llvm::Optional<std::string> Replacement = makeCharacterLiteral(Literal);
  if (!Replacement)
    return;
  SourceRange Range = Literal->getSourceRange();

  if (const auto *ByAnyChar = Result.Nodes.getNodeAs<Expr>("ByAnyChar"))
    Range = ByAnyChar->getSourceRange();

  diag(
      Literal->getBeginLoc(),
      "%select{absl::StrSplit()|absl::MaxSplits()}0 called with a string "
      "literal "
      "consisting of a single character; consider using the character overload")
      << (Result.Nodes.getNodeAs<CallExpr>("StrSplit") ? 0 : 1)
      << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(Range),
                                      *Replacement);
}

} // namespace abseil
} // namespace tidy
} // namespace clang

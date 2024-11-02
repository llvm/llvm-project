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
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

namespace {

AST_MATCHER(StringLiteral, lengthIsOne) { return Node.getLength() == 1; }

llvm::Optional<std::string> makeCharacterLiteral(const StringLiteral *Literal,
                                                 const ASTContext &Context) {
  assert(Literal->getLength() == 1 &&
         "Only single character string should be matched");
  assert(Literal->getCharByteWidth() == 1 &&
         "StrSplit doesn't support wide char");
  std::string Result = clang::tooling::fixit::getText(*Literal, Context).str();
  bool IsRawStringLiteral = StringRef(Result).startswith(R"(R")");
  // Since raw string literal might contain unescaped non-printable characters,
  // we normalize them using `StringLiteral::outputString`.
  if (IsRawStringLiteral) {
    Result.clear();
    llvm::raw_string_ostream Stream(Result);
    Literal->outputString(Stream);
  }
  // Special case: If the string contains a single quote, we just need to return
  // a character of the single quote. This is a special case because we need to
  // escape it in the character literal.
  if (Result == R"("'")")
    return std::string(R"('\'')");

  // Now replace the " with '.
  std::string::size_type Pos = Result.find_first_of('"');
  if (Pos == Result.npos)
    return std::nullopt;
  Result[Pos] = '\'';
  Pos = Result.find_last_of('"');
  if (Pos == Result.npos)
    return std::nullopt;
  Result[Pos] = '\'';
  return Result;
}

} // anonymous namespace

void FasterStrsplitDelimiterCheck::registerMatchers(MatchFinder *Finder) {
  // Binds to one character string literals.
  const auto SingleChar =
      expr(ignoringParenCasts(stringLiteral(lengthIsOne()).bind("Literal")));

  // Binds to a string_view (either absl or std) that was passed by value and
  // constructed from string literal.
  auto StringViewArg = ignoringElidableConstructorCall(ignoringImpCasts(
      cxxConstructExpr(hasType(recordDecl(hasName("::absl::string_view"))),
                       hasArgument(0, ignoringParenImpCasts(SingleChar)))));

  // Need to ignore the elidable constructor as otherwise there is no match for
  // c++14 and earlier.
  auto ByAnyCharArg =
      expr(has(ignoringElidableConstructorCall(
               ignoringParenCasts(cxxBindTemporaryExpr(has(cxxConstructExpr(
                   hasType(recordDecl(hasName("::absl::ByAnyChar"))),
                   hasArgument(0, StringViewArg))))))))
          .bind("ByAnyChar");

  // Find uses of absl::StrSplit(..., "x") and absl::StrSplit(...,
  // absl::ByAnyChar("x")) to transform them into absl::StrSplit(..., 'x').
  Finder->addMatcher(
      traverse(TK_AsIs,
               callExpr(callee(functionDecl(hasName("::absl::StrSplit"))),
                        hasArgument(1, anyOf(ByAnyCharArg, SingleChar)),
                        unless(isInTemplateInstantiation()))
                   .bind("StrSplit")),
      this);

  // Find uses of absl::MaxSplits("x", N) and
  // absl::MaxSplits(absl::ByAnyChar("x"), N) to transform them into
  // absl::MaxSplits('x', N).
  Finder->addMatcher(
      traverse(TK_AsIs,
               callExpr(callee(functionDecl(hasName("::absl::MaxSplits"))),
                        hasArgument(0, anyOf(ByAnyCharArg,
                                             ignoringParenCasts(SingleChar))),
                        unless(isInTemplateInstantiation()))),
      this);
}

void FasterStrsplitDelimiterCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("Literal");

  if (Literal->getBeginLoc().isMacroID() || Literal->getEndLoc().isMacroID())
    return;

  llvm::Optional<std::string> Replacement =
      makeCharacterLiteral(Literal, *Result.Context);
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

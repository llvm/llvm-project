//===--- UseDigitSeparatorCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <numeric>
#include <regex>

#include "UseDigitSeparatorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace {
std::vector<std::basic_string<char>>
splitStringByGroupSize(const std::basic_string<char> &String, size_t GroupSize) {
  std::vector<std::basic_string<char>> Result;
  std::basic_string<char> ReversedString(String.rbegin(), String.rend());

  for (size_t I = 0; I < ReversedString.size(); I += GroupSize) {
    Result.push_back(ReversedString.substr(I, GroupSize));
  }

  std::reverse(Result.begin(), Result.end());
  std::for_each(Result.begin(), Result.end(), [](std::basic_string<char> &Str) {return std::reverse(Str.begin(), Str.end());});

  return Result;
}
} // namespace

namespace clang::tidy::modernize {

void UseDigitSeparatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(integerLiteral().bind("integerLiteral"), this);
  Finder->addMatcher(floatLiteral().bind("floatLiteral"), this);
}

void UseDigitSeparatorCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();
  const IntegerLiteral *MatchedInteger = Result.Nodes.getNodeAs<IntegerLiteral>("integerLiteral");
  const FloatingLiteral *MatchedFloat = Result.Nodes.getNodeAs<FloatingLiteral>("floatLiteral");

  if (MatchedInteger != nullptr) {
    // Get original literal source text
    const StringRef OriginalLiteralString = Lexer::getSourceText(
        CharSourceRange::getTokenRange(MatchedInteger->getSourceRange()),
        Source, Context.getLangOpts());

    // Configure formatting
    unsigned int Radix;
    size_t GroupSize;
    std::string Prefix;
    std::string Postfix;
    if (OriginalLiteralString.starts_with("0b")) {
      Radix = 2;
      GroupSize = 4;
      Prefix = "0b";
    } else if (OriginalLiteralString.starts_with("0x")) {
      Radix = 16;
      GroupSize = 4;
      Prefix = "0x";
    } else if (OriginalLiteralString.starts_with("0") &&
               OriginalLiteralString != "0") {
      Radix = 8;
      GroupSize = 3;
      Prefix = "0";
    } else {
      Radix = 10;
      GroupSize = 3;
    }

    if (OriginalLiteralString.ends_with("L") ||
        OriginalLiteralString.ends_with("l") ||
        OriginalLiteralString.ends_with("U") ||
        OriginalLiteralString.ends_with("u")) {
      Postfix = OriginalLiteralString.back();
    }

    // Get formatting literal text
    const llvm::APInt IntegerValue = MatchedInteger->getValue();
    const std::vector<std::string> SplittedIntegerLiteral =
        splitStringByGroupSize(toString(IntegerValue, Radix, true), GroupSize);
    const std::string FormatedLiteralString =
        Prefix +
        std::accumulate(
            SplittedIntegerLiteral.begin(), SplittedIntegerLiteral.end(),
            std::string(""),
            [](std::basic_string<char> S1, std::basic_string<char> S2) {
              return S1 + "\'" + S2;
            })
            .erase(0, 1) +
        Postfix;

    // Compare the original and formatted representation of a literal
    if (OriginalLiteralString != FormatedLiteralString) {
      diag(MatchedInteger->getLocation(),
           "unformatted representation of integer literal '%0'")
          << OriginalLiteralString
          << FixItHint::CreateInsertion(MatchedInteger->getLocation(),
                                        FormatedLiteralString);
    }
  } else if (MatchedFloat != nullptr) {
    // Get original literal source text
    const std::string OriginalLiteralString =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(MatchedFloat->getSourceRange()),
            Source, Context.getLangOpts())
            .str();

    // Get formatting literal text
    std::string::size_type DotPosition = OriginalLiteralString.find('.');
    std::string FirstSubString = OriginalLiteralString.substr(0, DotPosition);
    std::string SecondSubString = OriginalLiteralString.substr(
        DotPosition + 1, OriginalLiteralString.size());
    std::reverse(SecondSubString.begin(), SecondSubString.end());

    std::vector<std::string> SplitedString = {FirstSubString, SecondSubString};
    std::vector<std::string> SplittedFloatLiteral0 =
        splitStringByGroupSize(SplitedString[0], 3);
    std::vector<std::string> SplittedFloatLiteral1 =
        splitStringByGroupSize(SplitedString[1], 3);
    std::string SecondReversedString =
        std::accumulate(
            SplittedFloatLiteral1.begin(), SplittedFloatLiteral1.end(),
            std::string(""),
            [](std::basic_string<char> S1, std::basic_string<char> S2) {
              return S1 + "\'" + S2;
            })
            .erase(0, 1);
    std::reverse(SecondReversedString.begin(), SecondReversedString.end());
    const std::string FormatedLiteralString =
        std::accumulate(
            SplittedFloatLiteral0.begin(), SplittedFloatLiteral0.end(),
            std::string(""),
            [](std::basic_string<char> S1, std::basic_string<char> S2) {
              return S1 + "\'" + S2;
            })
            .erase(0, 1) +
        '.' + SecondReversedString;

    // Compare the original and formatted representation of a literal
    if (OriginalLiteralString != FormatedLiteralString) {
      diag(MatchedFloat->getLocation(),
           "unformatted representation of integer literal '%0'")
          << OriginalLiteralString
          << FixItHint::CreateInsertion(MatchedFloat->getLocation(),
                                        FormatedLiteralString);
    }
  } else {
    assert(0);
  }
}

} // namespace clang::tidy::modernize

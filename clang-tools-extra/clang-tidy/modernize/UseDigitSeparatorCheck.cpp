//===--- UseDigitSeparatorCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <numeric>

#include "UseDigitSeparatorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace {
std::vector<std::basic_string<char>>
splitString3Symbols(const std::basic_string<char> &String) {
  std::vector<std::basic_string<char>> Result;
  std::basic_string<char> ReversedString(String.rbegin(), String.rend());

  for (size_t I = 0; I < ReversedString.size(); I += 3) {
    Result.push_back(ReversedString.substr(I, 3));
  }

  std::reverse(Result.begin(), Result.end());

  return Result;
}
} // namespace

namespace clang::tidy::modernize {

void UseDigitSeparatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(integerLiteral().bind("integerLiteral"), this);
}

void UseDigitSeparatorCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedInteger = Result.Nodes.getNodeAs<IntegerLiteral>("integerLiteral");
  const auto IntegerValue = MatchedInteger->getValue();
  const auto IntegerString =
      splitString3Symbols(toString(IntegerValue, 10, true));
  const auto FinalString =
      std::accumulate(IntegerString.begin(), IntegerString.end(),
                      std::string(""),
                      [](std::basic_string<char> S1,
                         std::basic_string<char> S2) { return S1 + "\'" + S2; })
          .erase(0, 1);
  diag(MatchedInteger->getLocation(), "integer warning %0")
      << FinalString
      << FixItHint::CreateInsertion(MatchedInteger->getLocation(),
                                    "this is integer");
  diag(MatchedInteger->getLocation(), "integer", DiagnosticIDs::Note);
}

} // namespace clang::tidy::modernize

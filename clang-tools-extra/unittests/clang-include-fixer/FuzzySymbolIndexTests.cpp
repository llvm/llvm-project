//===-- FuzzySymbolIndexTests.cpp - Fuzzy symbol index unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FuzzySymbolIndex.h"
#include "gmock/gmock.h"
#include "llvm/Support/Regex.h"
#include "gtest/gtest.h"

using testing::ElementsAre;
using testing::Not;

namespace clang {
namespace include_fixer {
namespace {

TEST(FuzzySymbolIndexTest, Tokenize) {
  EXPECT_THAT(FuzzySymbolIndex::tokenize("URLHandlerCallback"),
              ElementsAre("url", "handler", "callback"));
  EXPECT_THAT(FuzzySymbolIndex::tokenize("snake_case11"),
              ElementsAre("snake", "case", "11"));
  EXPECT_THAT(FuzzySymbolIndex::tokenize("__$42!!BOB\nbob"),
              ElementsAre("42", "bob", "bob"));
}

MATCHER_P(MatchesSymbol, Identifier, "") {
  llvm::Regex Pattern("^" + arg);
  std::string err;
  if (!Pattern.isValid(err)) {
    *result_listener << "invalid regex: " << err;
    return false;
  }
  auto Tokens = FuzzySymbolIndex::tokenize(Identifier);
  std::string Target = llvm::join(Tokens.begin(), Tokens.end(), " ");
  *result_listener << "matching against '" << Target << "'";
  return llvm::Regex("^" + arg).match(Target);
}

TEST(FuzzySymbolIndexTest, QueryRegexp) {
  auto QueryRegexp = [](const std::string &query) {
    return FuzzySymbolIndex::queryRegexp(FuzzySymbolIndex::tokenize(query));
  };
  EXPECT_THAT(QueryRegexp("uhc"), MatchesSymbol("URLHandlerCallback"));
  EXPECT_THAT(QueryRegexp("urhaca"), MatchesSymbol("URLHandlerCallback"));
  EXPECT_THAT(QueryRegexp("uhcb"), Not(MatchesSymbol("URLHandlerCallback")))
      << "Non-prefix";
  EXPECT_THAT(QueryRegexp("uc"), Not(MatchesSymbol("URLHandlerCallback")))
      << "Skip token";

  EXPECT_THAT(QueryRegexp("uptr"), MatchesSymbol("unique_ptr"));
  EXPECT_THAT(QueryRegexp("UniP"), MatchesSymbol("unique_ptr"));
}

} // namespace
} // namespace include_fixer
} // namespace clang

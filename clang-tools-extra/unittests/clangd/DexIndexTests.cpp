//===-- DexIndexTests.cpp  ----------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/dex/Token.h"
#include "index/dex/Trigram.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

using ::testing::ElementsAre;

testing::Matcher<std::vector<Token>>
trigramsAre(std::initializer_list<std::string> Trigrams) {
  std::vector<Token> Tokens;
  for (const auto &Symbols : Trigrams) {
    Tokens.push_back(Token(Token::Kind::Trigram, Symbols));
  }
  return testing::UnorderedElementsAreArray(Tokens);
}

TEST(DexIndexTrigrams, IdentifierTrigrams) {
  EXPECT_THAT(generateIdentifierTrigrams("X86"), trigramsAre({"x86"}));

  EXPECT_THAT(generateIdentifierTrigrams("nl"), trigramsAre({}));

  EXPECT_THAT(generateIdentifierTrigrams("clangd"),
              trigramsAre({"cla", "lan", "ang", "ngd"}));

  EXPECT_THAT(generateIdentifierTrigrams("abc_def"),
              trigramsAre({"abc", "abd", "ade", "bcd", "bde", "cde", "def"}));

  EXPECT_THAT(
      generateIdentifierTrigrams("a_b_c_d_e_"),
      trigramsAre({"abc", "abd", "acd", "ace", "bcd", "bce", "bde", "cde"}));

  EXPECT_THAT(
      generateIdentifierTrigrams("unique_ptr"),
      trigramsAre({"uni", "unp", "upt", "niq", "nip", "npt", "iqu", "iqp",
                   "ipt", "que", "qup", "qpt", "uep", "ept", "ptr"}));

  EXPECT_THAT(generateIdentifierTrigrams("TUDecl"),
              trigramsAre({"tud", "tde", "ude", "dec", "ecl"}));

  EXPECT_THAT(generateIdentifierTrigrams("IsOK"),
              trigramsAre({"iso", "iok", "sok"}));

  EXPECT_THAT(generateIdentifierTrigrams("abc_defGhij__klm"),
              trigramsAre({
                  "abc", "abd", "abg", "ade", "adg", "adk", "agh", "agk", "bcd",
                  "bcg", "bde", "bdg", "bdk", "bgh", "bgk", "cde", "cdg", "cdk",
                  "cgh", "cgk", "def", "deg", "dek", "dgh", "dgk", "dkl", "efg",
                  "efk", "egh", "egk", "ekl", "fgh", "fgk", "fkl", "ghi", "ghk",
                  "gkl", "hij", "hik", "hkl", "ijk", "ikl", "jkl", "klm",
              }));
}

TEST(DexIndexTrigrams, QueryTrigrams) {
  EXPECT_THAT(generateQueryTrigrams("X86"), trigramsAre({"x86"}));

  EXPECT_THAT(generateQueryTrigrams("nl"), trigramsAre({}));

  EXPECT_THAT(generateQueryTrigrams("clangd"),
              trigramsAre({"cla", "lan", "ang", "ngd"}));

  EXPECT_THAT(generateQueryTrigrams("abc_def"),
              trigramsAre({"abc", "bcd", "cde", "def"}));

  EXPECT_THAT(generateQueryTrigrams("a_b_c_d_e_"),
              trigramsAre({"abc", "bcd", "cde"}));

  EXPECT_THAT(generateQueryTrigrams("unique_ptr"),
              trigramsAre({"uni", "niq", "iqu", "que", "uep", "ept", "ptr"}));

  EXPECT_THAT(generateQueryTrigrams("TUDecl"),
              trigramsAre({"tud", "ude", "dec", "ecl"}));

  EXPECT_THAT(generateQueryTrigrams("IsOK"), trigramsAre({"iso", "sok"}));

  EXPECT_THAT(generateQueryTrigrams("abc_defGhij__klm"),
              trigramsAre({"abc", "bcd", "cde", "def", "efg", "fgh", "ghi",
                           "hij", "ijk", "jkl", "klm"}));
}

} // namespace dex
} // namespace clangd
} // namespace clang

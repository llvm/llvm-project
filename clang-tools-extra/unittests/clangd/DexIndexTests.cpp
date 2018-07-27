//===-- DexIndexTests.cpp  ----------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/dex/Iterator.h"
#include "index/dex/Token.h"
#include "index/dex/Trigram.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace dex {

using ::testing::ElementsAre;

TEST(DexIndexIterators, DocumentIterator) {
  const PostingList L = {4, 7, 8, 20, 42, 100};
  auto DocIterator = create(L);

  EXPECT_EQ(DocIterator->peek(), 4U);
  EXPECT_EQ(DocIterator->reachedEnd(), false);

  DocIterator->advance();
  EXPECT_EQ(DocIterator->peek(), 7U);
  EXPECT_EQ(DocIterator->reachedEnd(), false);

  DocIterator->advanceTo(20);
  EXPECT_EQ(DocIterator->peek(), 20U);
  EXPECT_EQ(DocIterator->reachedEnd(), false);

  DocIterator->advanceTo(65);
  EXPECT_EQ(DocIterator->peek(), 100U);
  EXPECT_EQ(DocIterator->reachedEnd(), false);

  DocIterator->advanceTo(420);
  EXPECT_EQ(DocIterator->reachedEnd(), true);
}

TEST(DexIndexIterators, AndWithEmpty) {
  const PostingList L0;
  const PostingList L1 = {0, 5, 7, 10, 42, 320, 9000};

  auto AndEmpty = createAnd(create(L0));
  EXPECT_EQ(AndEmpty->reachedEnd(), true);

  auto AndWithEmpty = createAnd(create(L0), create(L1));
  EXPECT_EQ(AndWithEmpty->reachedEnd(), true);

  EXPECT_THAT(consume(*AndWithEmpty), ElementsAre());
}

TEST(DexIndexIterators, AndTwoLists) {
  const PostingList L0 = {0, 5, 7, 10, 42, 320, 9000};
  const PostingList L1 = {0, 4, 7, 10, 30, 60, 320, 9000};

  auto And = createAnd(create(L1), create(L0));

  EXPECT_EQ(And->reachedEnd(), false);
  EXPECT_THAT(consume(*And), ElementsAre(0U, 7U, 10U, 320U, 9000U));

  And = createAnd(create(L0), create(L1));

  And->advanceTo(0);
  EXPECT_EQ(And->peek(), 0U);
  And->advanceTo(5);
  EXPECT_EQ(And->peek(), 7U);
  And->advanceTo(10);
  EXPECT_EQ(And->peek(), 10U);
  And->advanceTo(42);
  EXPECT_EQ(And->peek(), 320U);
  And->advanceTo(8999);
  EXPECT_EQ(And->peek(), 9000U);
  And->advanceTo(9001);
}

TEST(DexIndexIterators, AndThreeLists) {
  const PostingList L0 = {0, 5, 7, 10, 42, 320, 9000};
  const PostingList L1 = {0, 4, 7, 10, 30, 60, 320, 9000};
  const PostingList L2 = {1, 4, 7, 11, 30, 60, 320, 9000};

  auto And = createAnd(create(L0), create(L1), create(L2));
  EXPECT_EQ(And->peek(), 7U);
  And->advanceTo(300);
  EXPECT_EQ(And->peek(), 320U);
  And->advanceTo(100000);

  EXPECT_EQ(And->reachedEnd(), true);
}

TEST(DexIndexIterators, OrWithEmpty) {
  const PostingList L0;
  const PostingList L1 = {0, 5, 7, 10, 42, 320, 9000};

  auto OrEmpty = createOr(create(L0));
  EXPECT_EQ(OrEmpty->reachedEnd(), true);

  auto OrWithEmpty = createOr(create(L0), create(L1));
  EXPECT_EQ(OrWithEmpty->reachedEnd(), false);

  EXPECT_THAT(consume(*OrWithEmpty),
              ElementsAre(0U, 5U, 7U, 10U, 42U, 320U, 9000U));
}

TEST(DexIndexIterators, OrTwoLists) {
  const PostingList L0 = {0, 5, 7, 10, 42, 320, 9000};
  const PostingList L1 = {0, 4, 7, 10, 30, 60, 320, 9000};

  auto Or = createOr(create(L0), create(L1));

  EXPECT_EQ(Or->reachedEnd(), false);
  EXPECT_EQ(Or->peek(), 0U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 4U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 5U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 7U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 10U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 30U);
  Or->advanceTo(42);
  EXPECT_EQ(Or->peek(), 42U);
  Or->advanceTo(300);
  EXPECT_EQ(Or->peek(), 320U);
  Or->advanceTo(9000);
  EXPECT_EQ(Or->peek(), 9000U);
  Or->advanceTo(9001);
  EXPECT_EQ(Or->reachedEnd(), true);

  Or = createOr(create(L0), create(L1));

  EXPECT_THAT(consume(*Or),
              ElementsAre(0U, 4U, 5U, 7U, 10U, 30U, 42U, 60U, 320U, 9000U));
}

TEST(DexIndexIterators, OrThreeLists) {
  const PostingList L0 = {0, 5, 7, 10, 42, 320, 9000};
  const PostingList L1 = {0, 4, 7, 10, 30, 60, 320, 9000};
  const PostingList L2 = {1, 4, 7, 11, 30, 60, 320, 9000};

  auto Or = createOr(create(L0), create(L1), create(L2));

  EXPECT_EQ(Or->reachedEnd(), false);
  EXPECT_EQ(Or->peek(), 0U);

  Or->advance();
  EXPECT_EQ(Or->peek(), 1U);

  Or->advance();
  EXPECT_EQ(Or->peek(), 4U);

  Or->advanceTo(7);

  Or->advanceTo(59);
  EXPECT_EQ(Or->peek(), 60U);

  Or->advanceTo(9001);
  EXPECT_EQ(Or->reachedEnd(), true);
}

// FIXME(kbobyrev): The testcase below is similar to what is expected in real
// queries. It should be updated once new iterators (such as boosting, limiting,
// etc iterators) appear. However, it is not exhaustive and it would be
// beneficial to implement automatic generation of query trees for more
// comprehensive testing.
TEST(DexIndexIterators, QueryTree) {
  // An example of more complicated query
  //
  //                      +-----------------+
  //                      |And Iterator:1, 5|
  //                      +--------+--------+
  //                               |
  //                               |
  //                 +------------------------------------+
  //                 |                                    |
  //                 |                                    |
  //      +----------v----------+              +----------v---------+
  //      |And Iterator: 1, 5, 9|              |Or Iterator: 0, 1, 5|
  //      +----------+----------+              +----------+---------+
  //                 |                                    |
  //          +------+-----+                    +---------+-----------+
  //          |            |                    |         |           |
  //  +-------v-----+ +----v-----+           +--v--+    +-V--+    +---v---+
  //  |1, 3, 5, 8, 9| |1, 5, 7, 9|           |Empty|    |0, 5|    |0, 1, 5|
  //  +-------------+ +----------+           +-----+    +----+    +-------+

  const PostingList L0 = {1, 3, 5, 8, 9};
  const PostingList L1 = {1, 5, 7, 9};
  const PostingList L2 = {0, 5};
  const PostingList L3 = {0, 1, 5};
  const PostingList L4;

  // Root of the query tree: [1, 5]
  auto Root = createAnd(
      // Lower And Iterator: [1, 5, 9]
      createAnd(create(L0), create(L1)),
      // Lower Or Iterator: [0, 1, 5]
      createOr(create(L2), create(L3), create(L4)));

  EXPECT_EQ(Root->reachedEnd(), false);
  EXPECT_EQ(Root->peek(), 1U);
  Root->advanceTo(0);
  // Advance multiple times. Shouldn't do anything.
  Root->advanceTo(1);
  Root->advanceTo(0);
  EXPECT_EQ(Root->peek(), 1U);
  Root->advance();
  EXPECT_EQ(Root->peek(), 5U);
  Root->advanceTo(5);
  EXPECT_EQ(Root->peek(), 5U);
  Root->advanceTo(9000);
  EXPECT_EQ(Root->reachedEnd(), true);
}

TEST(DexIndexIterators, StringRepresentation) {
  const PostingList L0 = {4, 7, 8, 20, 42, 100};
  const PostingList L1 = {1, 3, 5, 8, 9};
  const PostingList L2 = {1, 5, 7, 9};
  const PostingList L3 = {0, 5};
  const PostingList L4 = {0, 1, 5};
  const PostingList L5;

  EXPECT_EQ(llvm::to_string(*(create(L0))), "[4, 7, 8, 20, 42, 100]");

  auto Nested = createAnd(createAnd(create(L1), create(L2)),
                          createOr(create(L3), create(L4), create(L5)));

  EXPECT_EQ(llvm::to_string(*Nested),
            "(& (& [1, 3, 5, 8, 9] [1, 5, 7, 9]) (| [0, 5] [0, 1, 5] []))");
}

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

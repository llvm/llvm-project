//===- unittests/Serialization/SourceLocationEncodingTests.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/SourceLocationEncoding.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {
using LocSeq = SourceLocationSequence;

// Convert a single source location into encoded form and back.
// If ExpectedEncoded is provided, verify the encoded value too.
// Loc is the raw (in-memory) form of SourceLocation.
void roundTrip(SourceLocation::UIntTy Loc,
               llvm::Optional<uint64_t> ExpectedEncoded = llvm::None) {
  uint64_t ActualEncoded =
      SourceLocationEncoding::encode(SourceLocation::getFromRawEncoding(Loc));
  if (ExpectedEncoded)
    ASSERT_EQ(ActualEncoded, *ExpectedEncoded) << "Encoding " << Loc;
  SourceLocation::UIntTy DecodedEncoded =
      SourceLocationEncoding::decode(ActualEncoded).getRawEncoding();
  ASSERT_EQ(DecodedEncoded, Loc) << "Decoding " << ActualEncoded;
}

// As above, but use sequence encoding for a series of locations.
void roundTrip(std::vector<SourceLocation::UIntTy> Locs,
               std::vector<uint64_t> ExpectedEncoded = {}) {
  std::vector<uint64_t> ActualEncoded;
  {
    LocSeq::State Seq;
    for (auto L : Locs)
      ActualEncoded.push_back(SourceLocationEncoding::encode(
          SourceLocation::getFromRawEncoding(L), Seq));
    if (!ExpectedEncoded.empty())
      ASSERT_EQ(ActualEncoded, ExpectedEncoded)
          << "Encoding " << testing::PrintToString(Locs);
  }
  std::vector<SourceLocation::UIntTy> DecodedEncoded;
  {
    LocSeq::State Seq;
    for (auto L : ActualEncoded) {
      SourceLocation Loc = SourceLocationEncoding::decode(L, Seq);
      DecodedEncoded.push_back(Loc.getRawEncoding());
    }
    ASSERT_EQ(DecodedEncoded, Locs)
        << "Decoding " << testing::PrintToString(ActualEncoded);
  }
}

constexpr SourceLocation::UIntTy MacroBit =
    1 << (sizeof(SourceLocation::UIntTy) * CHAR_BIT - 1);
constexpr SourceLocation::UIntTy Big = MacroBit >> 1;
constexpr SourceLocation::UIntTy Biggest = -1;

TEST(SourceLocationEncoding, Individual) {
  roundTrip(1, 2);
  roundTrip(100, 200);
  roundTrip(MacroBit, 1);
  roundTrip(MacroBit | 5, 11);
  roundTrip(Big);
  roundTrip(Big + 1);
  roundTrip(MacroBit | Big);
  roundTrip(MacroBit | Big + 1);
}

TEST(SourceLocationEncoding, Sequence) {
  roundTrip({1, 2, 3, 3, 2, 1},
            {2, // 1
             5, // +2 (+1 of non-raw)
             5, // +2
             1, // +0
             4, // -2
             4} // -2
  );
  roundTrip({100, 0, 100},
            {200, // 100
             0,   // 0
             1}   // +0
  );

  roundTrip({1, Big}, {2, ((Big - 1) << 2) + 1});
  roundTrip({2, MacroBit | Big}, {4, ((Big - 1) << 2) - 1});

  roundTrip({3, MacroBit | 5, MacroBit | 4, 3},
            {6,  // 3
             11, // +5 (+2 of non-raw + set macro bit)
             4,  // -2
             6}  // -3 (-2 of non-raw, clear macro bit)
  );

  roundTrip(
      {123 | MacroBit, 1, 9, Biggest, Big, Big + 1, 0, MacroBit | Big, 0});
}

} // namespace

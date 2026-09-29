//===- unittests/Serialization/SourceLocationEncodingTests.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/SourceLocationEncoding.h"

#include "gtest/gtest.h"
#include <climits>
#include <optional>

using namespace llvm;
using namespace clang;

namespace {

// Convert a single source location into encoded form and back.
// If ExpectedEncoded is provided, verify the encoded value too.
// Loc is the raw (in-memory) form of SourceLocation.
void roundTrip(SourceLocation::UIntTy Loc,
               std::optional<uint64_t> ExpectedEncoded = std::nullopt) {
  uint64_t ActualEncoded = SourceLocationEncoding::encode(
      SourceLocation::getFromRawEncoding(Loc), /*BaseOffset=*/0,
      /*BaseModuleFileIndex=*/0);
  if (ExpectedEncoded) {
    ASSERT_EQ(ActualEncoded, *ExpectedEncoded) << "Encoding " << Loc;
  }
  SourceLocation::UIntTy DecodedEncoded =
      SourceLocationEncoding::decode(ActualEncoded).first.getRawEncoding();
  ASSERT_EQ(DecodedEncoded, Loc) << "Decoding " << ActualEncoded;
}

constexpr SourceLocation::UIntTy MacroBit =
    1 << (sizeof(SourceLocation::UIntTy) * CHAR_BIT - 1);
constexpr SourceLocation::UIntTy Big = MacroBit >> 1;

TEST(SourceLocationEncoding, Individual) {
  roundTrip(1, 2);
  roundTrip(100, 200);
  roundTrip(MacroBit, 1);
  roundTrip(MacroBit | 5, 11);
  roundTrip(Big);
  roundTrip(Big + 1);
  roundTrip(MacroBit | Big);
  roundTrip(MacroBit | (Big + 1));
}

} // namespace

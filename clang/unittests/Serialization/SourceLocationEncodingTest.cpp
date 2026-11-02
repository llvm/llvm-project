//===- unittests/Serialization/SourceLocationEncodingTests.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/SourceLocationEncoding.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
constexpr SourceLocation::UIntTy Biggest = ~SourceLocation::UIntTy(0);

using Chain = SourceLocationEncoding::Chain;

uint64_t encodeLocal(SourceLocation::UIntTy Loc) {
  return SourceLocationEncoding::encode(SourceLocation::getFromRawEncoding(Loc),
                                        /*BaseOffset=*/0,
                                        /*BaseModuleFileIndex=*/0);
}

// Round-trip a run of locations through a chain, the way ASTWriter and
// ASTReader use it for SM_SLOC_EXPANSION_ENTRY.
void roundTripChain(SourceLocation::UIntTy RecordOffset,
                    ArrayRef<SourceLocation::UIntTy> Locs) {
  SmallVector<uint64_t> Encoded;
  Chain Enc(RecordOffset + 2);
  for (SourceLocation::UIntTy Loc : Locs) {
    uint64_t E = Enc.deltaEncode(encodeLocal(Loc));
    // Nothing may spill into the module file index.
    ASSERT_EQ(E >> 32, 0u) << "Encoding " << Loc;
    Encoded.push_back(E);
  }

  Chain Dec(RecordOffset + 2);
  for (auto [E, Loc] : llvm::zip(Encoded, Locs)) {
    auto [Decoded, ModuleFileIndex] =
        SourceLocationEncoding::decode(Dec.deltaDecode(E));
    ASSERT_EQ(ModuleFileIndex, 0u) << "Decoding " << E;
    ASSERT_EQ(Decoded.getRawEncoding(), Loc) << "Decoding " << E;
  }
}

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

TEST(SourceLocationEncoding, Chained) {
  // The chain must work wherever in the module the record happens to sit.
  for (SourceLocation::UIntTy Off : {0u, 1u, 100u, 1u << 20, 1u << 30}) {
    roundTripChain(Off, {1, 2, 3});
    roundTripChain(Off, {0, 0, 0});             // all null
    roundTripChain(Off, {MacroBit | 5, 0, 17}); // null in the middle
    roundTripChain(Off, {7, 7, 7});             // repeats
    roundTripChain(Off, {Big, Big + 1, MacroBit | Big});
    roundTripChain(Off, {Biggest, 1, Biggest}); // large jumps
    roundTripChain(Off, {MacroBit, MacroBit | 1, 1});
  }
}

TEST(SourceLocationEncoding, NoSpillIntoModuleFileIndex) {
  // No encoded value should spill to the upper 32 bit of the encoding.
  // See llvm/llvm-project#145529.
  roundTripChain(0, {1, (1u << 30) + 1});
  roundTripChain(0, {1, 9, Biggest, Big, Big + 1, 0, MacroBit | Big, 0});

  // Sweep the extremes: whatever the seed, an encoded value stays in 32 bits.
  for (SourceLocation::UIntTy Off : {0u, 1u, 1u << 30, (1u << 31) - 3}) {
    for (SourceLocation::UIntTy Loc : {1u, 2u, MacroBit, MacroBit | 1u, Big,
                                       Big + 1, Biggest, Biggest - 1}) {
      Chain Enc(Off + 2);
      uint64_t Raw = encodeLocal(Loc);
      uint64_t E = Enc.deltaEncode(Raw);
      ASSERT_LE(E, 0xFFFFFFFFull);
      Chain Dec(Off + 2);
      ASSERT_EQ(Dec.deltaDecode(E), Raw);
    }
  }
}

// deltaEncode() and deltaDecode() split on either side of the hole, so an
// off-by-one there is a single-character mistake that silently turns a valid
// location into a null one. Pin both codes adjacent to the hole. The location
// whose raw value equals the seed lands just above it; MacroBit | (seed - 1)
// lands just below.
TEST(SourceLocationEncoding, HoleBoundary) {
  for (SourceLocation::UIntTy Off : {0u, 1u, 100u, 4096u, 1u << 20}) {
    // What the chain seeds to: the rotation of a file location at the
    // anchor, i.e. encodeRaw(Off + 2) == 2 * (Off + 2). Spelled out because
    // Chain keeps encodeRaw private. Since hole == 2 * Seed - 1, the location
    // landing just above the hole is the one whose raw value equals Seed,
    // and the one just below is MacroBit | (Seed - 1).
    SourceLocation::UIntTy Seed = 2 * (Off + 2);
    roundTripChain(Off, {Seed});                  // code == hole + 1
    roundTripChain(Off, {MacroBit | (Seed - 1)}); // code == hole
    roundTripChain(Off, {MacroBit | (Seed - 1), Seed});
  }
}

// Locations owned by an imported module file keep their module file index and
// are stored verbatim, and they must not disturb the chain around them.
TEST(SourceLocationEncoding, ImportedLocationsBypassChain) {
  constexpr SourceLocation::UIntTy RecordOffset = 64;
  uint64_t Imported = SourceLocationEncoding::encode(
      SourceLocation::getFromRawEncoding(4242), /*BaseOffset=*/100,
      /*BaseModuleFileIndex=*/3);
  ASSERT_NE(Imported >> 32, 0u);

  Chain WithImport(RecordOffset + 2);
  uint64_t A1 = WithImport.deltaEncode(encodeLocal(70));
  uint64_t I = WithImport.deltaEncode(Imported);
  uint64_t B1 = WithImport.deltaEncode(encodeLocal(90));

  EXPECT_EQ(I, Imported) << "Imported location must pass through unchanged";

  // Dropping the imported location leaves the other two encodings untouched.
  Chain WithoutImport(RecordOffset + 2);
  EXPECT_EQ(WithoutImport.deltaEncode(encodeLocal(70)), A1);
  EXPECT_EQ(WithoutImport.deltaEncode(encodeLocal(90)), B1);

  Chain Dec(RecordOffset + 2);
  EXPECT_EQ(SourceLocationEncoding::decode(Dec.deltaDecode(A1))
                .first.getRawEncoding(),
            70u);
  EXPECT_EQ(Dec.deltaDecode(I), Imported);
  EXPECT_EQ(SourceLocationEncoding::decode(Dec.deltaDecode(B1))
                .first.getRawEncoding(),
            90u);
}

} // namespace

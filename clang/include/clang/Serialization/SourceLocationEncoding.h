//===--- SourceLocationEncoding.h - Small serialized locations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// We wish to encode the SourceLocation from other module file not dependent
// on the other module file. So that the source location changes from other
// module file may not affect the contents of the current module file. Then the
// users don't need to recompile the whole project due to a new line in a module
// unit in the root of the dependency graph.
//
// To achieve this, we need to encode the index of the module file into the
// encoding of the source location. The encoding of the source location may be:
//
//      |-----------------------|-----------------------|
//      |          A            |         B         | C |
//
//  * A: 32 bit. The index of the module file in the module manager + 1. The +1
//  here is necessary since we wish 0 stands for the current module file.
//  * B: 31 bit. The offset of the source location to the module file containing
//  it.
//  * C: The macro bit. We rotate it to the lowest bit so that we can save some
//  space in case the index of the module file is 0.
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_SOURCELOCATIONENCODING_H
#define LLVM_CLANG_SERIALIZATION_SOURCELOCATIONENCODING_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/MathExtras.h"
#include <climits>

namespace clang {

/// Serialized encoding of SourceLocations without context.
/// Optimized to have small unsigned values (=> small after VBR encoding).
///
// Macro locations have the top bit set, we rotate by one so it is the low bit.
class SourceLocationEncoding {
  using UIntTy = SourceLocation::UIntTy;
  constexpr static unsigned UIntBits = CHAR_BIT * sizeof(UIntTy);

  static UIntTy encodeRaw(UIntTy Raw) {
    return (Raw << 1) | (Raw >> (UIntBits - 1));
  }
  static UIntTy decodeRaw(UIntTy Raw) {
    return (Raw >> 1) | (Raw << (UIntBits - 1));
  }

public:
  using RawLocEncoding = uint64_t;

  static RawLocEncoding encode(SourceLocation Loc, UIntTy BaseOffset,
                               unsigned BaseModuleFileIndex);
  static std::pair<SourceLocation, unsigned> decode(RawLocEncoding);

  /// A delta encoder for a run of source locations.
  /// The high level strategy of the chain to encode the run is the following.
  /// Locations:           SL0          SL1          SL2
  /// map to   :   (SL0 - seed)  (SL1 - SL0)  (SL2 - SL1)
  ///
  /// Two wrinkles complicate this strategy's implementation.
  /// First, since the delta-encoded values are meant for VBR compression, they
  /// are mapped from int to unsigned int with the zigZag method.
  ///
  /// Second, we are encoding an input value of 0 by 0, instead of using a
  /// delta, and we are encoding a delta of zero by 1. This is because both
  /// value 0 and a delta of zero show up frequently in source location delta
  /// encoding. It would be wasteful to represent an absolute 0 using a delta
  /// from a previous value in the run (the delta's absolute value may be big).
  /// A consequence is that for a given Prev, zigZag(0 - Prev) should not be
  /// used for any input, because 0 is encoded directly. This naturally allows
  /// us to represent possible delta values in the following way:
  ///
  /// delta = V - prev
  /// hole = zigZag(0 - prev)
  /// encoded = zigZag(delta) + 1 if zigZag(delta) < hole
  ///         = zigZag(delta)     if zigZag(delta) > hole
  ///
  /// Or pictorially:
  /// zigZag(delta): 0, 1, ... hole - 1, hole + 1, hole + 2, ...
  ///       encoded: 1, 2, ...     hole, hole + 1, hole + 2, ...
  ///
  /// Note it is impossible for zigZag(delta) to be equal to the hole as that
  /// would imply V == 0. In other words, if zigZag(delta) < hole, we increment
  /// the encoded value by 1 to leave 0 to represent V == 0. If zigZag(delta)
  /// is larger than hole, we do not need to add 1. Therefore, we will
  /// not accidentally add 1 to values that may overflow beyond 2^32 - 1,
  /// which may lead to accidental change of the ModuleIndex bits.
  /// This mapping avoids the issue llvm/llvm-project#145529 attempted to fix by
  /// design.
  class Chain {
    UIntTy Prev;

    /// Maps an int to an unsigned int.
    /// Explicitly, zigZag does the following mapping:
    /// From: 0, -1, +1, -2, +2, ...
    ///   To: 0,  1,  2,  3,  4, ...
    /// In other words, the mapping is the following:
    /// zigZag(V) = 2 * V if V >= 0
    ///           = 2 * |V| - 1 if V < 0
    static UIntTy zigZag(UIntTy V) {
      return (V << 1) ^ (UIntTy(0) - (V >> (UIntBits - 1)));
    }

    /// Reverse mapping of zigZag.
    static UIntTy zagZig(UIntTy V) { return (V >> 1) ^ (UIntTy(0) - (V & 1)); }

    /// Computes the hole left by 0 - prev.
    UIntTy hole() const { return zigZag(UIntTy(0) - Prev); }

  public:
    /// Get the seed for a chain from an SM_SLOC_EXPANSION_ENTRY record's first
    /// field, Offset. That field holds the entry's adjusted module-local offset
    /// with the dummy entry subtracted out, so adding 2 recovers the entry's
    /// own position in the source location space. encodeRaw then puts it in the
    /// same rotated space as the locations the chain encodes, so the deltas
    /// line up. Reader and writer both derive the seed from this one field, so
    /// they cannot drift apart.
    static UIntTy getSeedFrom(SourceLocation::UIntTy RecordOffset) {
      return encodeRaw(RecordOffset + 2);
    }

    explicit Chain(UIntTy Seed) : Prev(Seed) {
      // Using zero as seed could make the chain very expensive since
      // the deltas may be big.
      assert(Seed != 0 && "Chain seed should anchor the run");
    }

    RawLocEncoding deltaEncode(RawLocEncoding V) {
      // If the source location is external, do not encode.
      if (V >> 32)
        return V;

      // Use 0 to encode an input of 0.
      if (V == 0)
        return 0;

      // Delta encode the rest of the possible input values.
      UIntTy SL = static_cast<UIntTy>(V);
      UIntTy E = zigZag(SL - Prev);
      UIntTy H = hole();
      assert(E != H && "Non-zero location cannot be mapped to the hole");
      Prev = SL;
      return E < H ? static_cast<RawLocEncoding>(E) + 1 : E;
    }

    RawLocEncoding deltaDecode(RawLocEncoding V) {
      // If the source location is external, it is not delta encoded.
      if (V >> 32)
        return V;

      // If V is 0, it is not delta encoded.
      if (V == 0)
        return 0;

      // Delta-decode the value.
      UIntTy H = hole();
      UIntTy D = static_cast<UIntTy>(V);
      Prev += zagZig(D <= H ? D - 1 : D);
      return Prev;
    }
  };
};

inline SourceLocationEncoding::RawLocEncoding
SourceLocationEncoding::encode(SourceLocation Loc, UIntTy BaseOffset,
                               unsigned BaseModuleFileIndex) {
  // If the source location is a local source location, we can try to optimize
  // the similar sequences to only record the differences.
  if (!BaseOffset)
    return encodeRaw(Loc.getRawEncoding());
  if (Loc.isInvalid())
    return 0;

  // Otherwise, the higher bits are used to store the module file index,
  // so it is meaningless to optimize the source locations into small
  // integers. Let's try to always use the raw encodings.
  assert(Loc.getOffset() >= BaseOffset);
  Loc = Loc.getLocWithOffset(-BaseOffset);
  RawLocEncoding Encoded = encodeRaw(Loc.getRawEncoding());

  // 16 bits should be sufficient to store the module file index.
  assert(BaseModuleFileIndex < (1 << 16));
  Encoded |= (RawLocEncoding)BaseModuleFileIndex << 32;
  return Encoded;
}
inline std::pair<SourceLocation, unsigned>
SourceLocationEncoding::decode(RawLocEncoding Encoded) {
  unsigned ModuleFileIndex = Encoded >> 32;

  if (!ModuleFileIndex)
    return {SourceLocation::getFromRawEncoding(decodeRaw(Encoded)),
            ModuleFileIndex};

  Encoded &= llvm::maskTrailingOnes<RawLocEncoding>(32);
  SourceLocation Loc = SourceLocation::getFromRawEncoding(decodeRaw(Encoded));

  return {Loc, ModuleFileIndex};
}

} // namespace clang
#endif

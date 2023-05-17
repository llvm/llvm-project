//===--- SourceLocationEncoding.h - Small serialized locations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Source locations are stored pervasively in the AST, making up a third of
// the size of typical serialized files. Storing them efficiently is important.
//
// We use integers optimized by VBR-encoding, because:
//  - when abbreviations cannot be used, VBR6 encoding is our only choice
//  - in the worst case a SourceLocation can be ~any 32-bit number, but in
//    practice they are highly predictable
//
// We encode the integer so that likely values encode as small numbers that
// turn into few VBR chunks:
//  - the invalid sentinel location is a very common value: it encodes as 0
//  - the "macro or not" bit is stored at the bottom of the integer
//    (rather than at the top, as in memory), so macro locations can have
//    small representations.
//  - related locations (e.g. of a left and right paren pair) are usually
//    similar, so when encoding a sequence of locations we store only
//    differences between successive elements.
//
//===----------------------------------------------------------------------===//

#include <climits>
#include "clang/Basic/SourceLocation.h"

#ifndef LLVM_CLANG_SERIALIZATION_SOURCELOCATIONENCODING_H
#define LLVM_CLANG_SERIALIZATION_SOURCELOCATIONENCODING_H

namespace clang {
class SourceLocationSequence;

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
  friend SourceLocationSequence;

public:
  static uint64_t encode(SourceLocation Loc,
                         SourceLocationSequence * = nullptr);
  static SourceLocation decode(uint64_t, SourceLocationSequence * = nullptr);
};

/// Serialized encoding of a sequence of SourceLocations.
///
/// Optimized to produce small values when locations with the sequence are
/// similar. Each element can be delta-encoded against the last nonzero element.
///
/// Sequences should be started by creating a SourceLocationSequence::State,
/// and then passed around as SourceLocationSequence*. Example:
///
///   // establishes a sequence
///   void EmitTopLevelThing() {
///     SourceLocationSequence::State Seq;
///     EmitContainedThing(Seq);
///     EmitRecursiveThing(Seq);
///   }
///
///   // optionally part of a sequence
///   void EmitContainedThing(SourceLocationSequence *Seq = nullptr) {
///     Record.push_back(SourceLocationEncoding::encode(SomeLoc, Seq));
///   }
///
///   // establishes a sequence if there isn't one already
///   void EmitRecursiveThing(SourceLocationSequence *ParentSeq = nullptr) {
///     SourceLocationSequence::State Seq(ParentSeq);
///     Record.push_back(SourceLocationEncoding::encode(SomeLoc, Seq));
///     EmitRecursiveThing(Seq);
///   }
///
class SourceLocationSequence {
  using UIntTy = SourceLocation::UIntTy;
  using EncodedTy = uint64_t;
  constexpr static auto UIntBits = SourceLocationEncoding::UIntBits;
  static_assert(sizeof(EncodedTy) > sizeof(UIntTy), "Need one extra bit!");

  // Prev stores the rotated last nonzero location.
  UIntTy &Prev;

  // Zig-zag encoding turns small signed integers into small unsigned integers.
  // 0 => 0, -1 => 1, 1 => 2, -2 => 3, ...
  static UIntTy zigZag(UIntTy V) {
    UIntTy Sign = (V & (1 << (UIntBits - 1))) ? UIntTy(-1) : UIntTy(0);
    return Sign ^ (V << 1);
  }
  static UIntTy zagZig(UIntTy V) { return (V >> 1) ^ -(V & 1); }

  SourceLocationSequence(UIntTy &Prev) : Prev(Prev) {}

  EncodedTy encodeRaw(UIntTy Raw) {
    if (Raw == 0)
      return 0;
    UIntTy Rotated = SourceLocationEncoding::encodeRaw(Raw);
    if (Prev == 0)
      return Prev = Rotated;
    UIntTy Delta = Rotated - Prev;
    Prev = Rotated;
    // Exactly one 33 bit value is possible! (1 << 32).
    // This is because we have two representations of zero: trivial & relative.
    return 1 + EncodedTy{zigZag(Delta)};
  }
  UIntTy decodeRaw(EncodedTy Encoded) {
    if (Encoded == 0)
      return 0;
    if (Prev == 0)
      return SourceLocationEncoding::decodeRaw(Prev = Encoded);
    return SourceLocationEncoding::decodeRaw(Prev += zagZig(Encoded - 1));
  }

public:
  SourceLocation decode(EncodedTy Encoded) {
    return SourceLocation::getFromRawEncoding(decodeRaw(Encoded));
  }
  EncodedTy encode(SourceLocation Loc) {
    return encodeRaw(Loc.getRawEncoding());
  }

  class State;
};

/// This object establishes a SourceLocationSequence.
class SourceLocationSequence::State {
  UIntTy Prev = 0;
  SourceLocationSequence Seq;

public:
  // If Parent is provided and non-null, then this root becomes part of that
  // enclosing sequence instead of establishing a new one.
  State(SourceLocationSequence *Parent = nullptr)
      : Seq(Parent ? Parent->Prev : Prev) {}

  // Implicit conversion for uniform use of roots vs propagated sequences.
  operator SourceLocationSequence *() { return &Seq; }
};

inline uint64_t SourceLocationEncoding::encode(SourceLocation Loc,
                                               SourceLocationSequence *Seq) {
  return Seq ? Seq->encode(Loc) : encodeRaw(Loc.getRawEncoding());
}
inline SourceLocation
SourceLocationEncoding::decode(uint64_t Encoded, SourceLocationSequence *Seq) {
  return Seq ? Seq->decode(Encoded)
             : SourceLocation::getFromRawEncoding(decodeRaw(Encoded));
}

} // namespace clang
#endif

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
//      |---------------|---------------------------|
//      |        A      |           B           | C |
//
//  * A: 24 bit. The index of the module file in the module manager + 1. The +1
//  here is necessary since we wish 0 stands for the current module file.
//  * B: 39 bit. The offset of the source location to the module file containing
//  it.
//  * C: The macro bit. We rotate it to the lowest bit so that we can save some
//  space in case the index of the module file is 0.
//
// Together, B and C form SourceLocation::Bits (40 bits total).
//
// Currently, only the lower 16 bits of A are used to store the module file
// index, leaving the upper 8 bits of A unused (reserved for future use).
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

  static UIntTy encodeRaw(UIntTy Raw) {
    return ((Raw & llvm::maskTrailingOnes<uint64_t>(SourceLocation::Bits - 1))
            << 1) |
           (Raw >> (SourceLocation::Bits - 1));
  }
  static UIntTy decodeRaw(UIntTy Raw) {
    return (Raw >> 1) | ((Raw & 1) << (SourceLocation::Bits - 1));
  }

public:
  using RawLocEncoding = uint64_t;
  // 16 bits should be sufficient to store the module file index.
  constexpr static unsigned ModuleFileIndexBits = 16;
  constexpr static unsigned SourceLocationEncodingBits = SourceLocation::Bits;
  static_assert(ModuleFileIndexBits + SourceLocationEncodingBits <
                    sizeof(RawLocEncoding) * CHAR_BIT,
                "Insufficient encoding bits");

  static RawLocEncoding encode(SourceLocation Loc, UIntTy BaseOffset,
                               unsigned BaseModuleFileIndex);
  static std::pair<SourceLocation, unsigned> decode(RawLocEncoding);
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

  assert(BaseModuleFileIndex < (1 << ModuleFileIndexBits));
  Encoded |= (RawLocEncoding)BaseModuleFileIndex << SourceLocation::Bits;
  return Encoded;
}
inline std::pair<SourceLocation, unsigned>
SourceLocationEncoding::decode(RawLocEncoding Encoded) {
  unsigned ModuleFileIndex = Encoded >> SourceLocation::Bits;

  if (!ModuleFileIndex)
    return {SourceLocation::getFromRawEncoding(decodeRaw(Encoded)),
            ModuleFileIndex};

  Encoded &= llvm::maskTrailingOnes<RawLocEncoding>(SourceLocation::Bits);
  SourceLocation Loc = SourceLocation::getFromRawEncoding(decodeRaw(Encoded));

  return {Loc, ModuleFileIndex};
}

} // namespace clang
#endif

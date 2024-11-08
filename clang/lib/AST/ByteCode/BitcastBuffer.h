//===--------------------- BitcastBuffer.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H
#define LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H

#include <cassert>
#include <cstddef>
#include <memory>

namespace clang {
namespace interp {

enum class Endian { Little, Big };

/// Returns the value of the bit in the given sequence of bytes.
static inline bool bitof(const std::byte *B, unsigned BitIndex) {
  return (B[BitIndex / 8] & (std::byte{1} << (BitIndex % 8))) != std::byte{0};
}

/// Returns whether \p N is a full byte offset or size.
static inline bool fullByte(unsigned N) { return N % 8 == 0; }

/// Track what bits have been initialized to known values and which ones
/// have indeterminate value.
/// All offsets are in bits.
struct BitcastBuffer {
  size_t FinalBitSize = 0;
  std::unique_ptr<std::byte[]> Data;

  BitcastBuffer(size_t FinalBitSize) : FinalBitSize(FinalBitSize) {
    assert(fullByte(FinalBitSize));
    unsigned ByteSize = FinalBitSize / 8;
    Data = std::make_unique<std::byte[]>(ByteSize);
  }

  /// Returns the buffer size in bits.
  size_t size() const { return FinalBitSize; }

  /// Returns \c true if all bits in the buffer have been initialized.
  bool allInitialized() const {
    // FIXME: Implement.
    return true;
  }

  /// Push \p BitWidth bits at \p BitOffset from \p In into the buffer.
  /// \p TargetEndianness is the endianness of the target we're compiling for.
  /// \p In must hold at least \p BitWidth many bits.
  void pushData(const std::byte *In, size_t BitOffset, size_t BitWidth,
                Endian TargetEndianness);

  /// Copy \p BitWidth bits at offset \p BitOffset from the buffer.
  /// \p TargetEndianness is the endianness of the target we're compiling for.
  ///
  /// The returned output holds exactly (\p FullBitWidth / 8) bytes.
  std::unique_ptr<std::byte[]> copyBits(unsigned BitOffset, unsigned BitWidth,
                                        unsigned FullBitWidth,
                                        Endian TargetEndianness) const;
};

} // namespace interp
} // namespace clang
#endif

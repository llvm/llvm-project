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

/// A quantity in bits.
struct Bits {
  size_t N = 0;
  Bits() = default;
  static Bits zero() { return Bits(0); }
  explicit Bits(size_t Quantity) : N(Quantity) {}
  size_t getQuantity() const { return N; }
  size_t roundToBytes() const { return N / 8; }
  size_t getOffsetInByte() const { return N % 8; }
  bool isFullByte() const { return N % 8 == 0; }
  bool nonZero() const { return N != 0; }

  Bits operator-(Bits Other) { return Bits(N - Other.N); }
  Bits operator+(Bits Other) { return Bits(N + Other.N); }
  Bits operator+=(size_t O) {
    N += O;
    return *this;
  }

  bool operator>=(Bits Other) { return N >= Other.N; }
};

/// A quantity in bytes.
struct Bytes {
  size_t N;
  explicit Bytes(size_t Quantity) : N(Quantity) {}
  size_t getQuantity() const { return N; }
  Bits toBits() const { return Bits(N * 8); }
};

/// Track what bits have been initialized to known values and which ones
/// have indeterminate value.
struct BitcastBuffer {
  Bits FinalBitSize;
  std::unique_ptr<std::byte[]> Data;

  BitcastBuffer(Bits FinalBitSize) : FinalBitSize(FinalBitSize) {
    assert(FinalBitSize.isFullByte());
    unsigned ByteSize = FinalBitSize.roundToBytes();
    Data = std::make_unique<std::byte[]>(ByteSize);
  }

  /// Returns the buffer size in bits.
  Bits size() const { return FinalBitSize; }

  /// Returns \c true if all bits in the buffer have been initialized.
  bool allInitialized() const {
    // FIXME: Implement.
    return true;
  }

  /// Push \p BitWidth bits at \p BitOffset from \p In into the buffer.
  /// \p TargetEndianness is the endianness of the target we're compiling for.
  /// \p In must hold at least \p BitWidth many bits.
  void pushData(const std::byte *In, Bits BitOffset, Bits BitWidth,
                Endian TargetEndianness);

  /// Copy \p BitWidth bits at offset \p BitOffset from the buffer.
  /// \p TargetEndianness is the endianness of the target we're compiling for.
  ///
  /// The returned output holds exactly (\p FullBitWidth / 8) bytes.
  std::unique_ptr<std::byte[]> copyBits(Bits BitOffset, Bits BitWidth,
                                        Bits FullBitWidth,
                                        Endian TargetEndianness) const;
};

} // namespace interp
} // namespace clang
#endif

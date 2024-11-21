//===-------------------- Bitcastbuffer.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "BitcastBuffer.h"

using namespace clang;
using namespace clang::interp;

/// Returns the value of the bit in the given sequence of bytes.
static inline bool bitof(const std::byte *B, Bits BitIndex) {
  return (B[BitIndex.roundToBytes()] &
          (std::byte{1} << BitIndex.getOffsetInByte())) != std::byte{0};
}

void BitcastBuffer::pushData(const std::byte *In, Bits BitOffset, Bits BitWidth,
                             Endian TargetEndianness) {
  for (unsigned It = 0; It != BitWidth.getQuantity(); ++It) {
    bool BitValue = bitof(In, Bits(It));
    if (!BitValue)
      continue;

    Bits DstBit;
    if (TargetEndianness == Endian::Little)
      DstBit = BitOffset + Bits(It);
    else
      DstBit = size() - BitOffset - BitWidth + Bits(It);

    size_t DstByte = DstBit.roundToBytes();
    Data[DstByte] |= std::byte{1} << DstBit.getOffsetInByte();
  }
}

std::unique_ptr<std::byte[]>
BitcastBuffer::copyBits(Bits BitOffset, Bits BitWidth, Bits FullBitWidth,
                        Endian TargetEndianness) const {
  assert(BitWidth.getQuantity() <= FullBitWidth.getQuantity());
  assert(FullBitWidth.isFullByte());
  auto Out = std::make_unique<std::byte[]>(FullBitWidth.roundToBytes());

  for (unsigned It = 0; It != BitWidth.getQuantity(); ++It) {
    Bits BitIndex;
    if (TargetEndianness == Endian::Little)
      BitIndex = BitOffset + Bits(It);
    else
      BitIndex = size() - BitWidth - BitOffset + Bits(It);

    bool BitValue = bitof(Data.get(), BitIndex);
    if (!BitValue)
      continue;

    Bits DstBit = Bits(It);
    size_t DstByte = DstBit.roundToBytes();
    Out[DstByte] |= std::byte{1} << DstBit.getOffsetInByte();
  }

  return Out;
}

#if 0
  template<typename T>
  static std::string hex(T t) {
    std::stringstream stream;
    stream << std::hex << (int)t;
    return std::string(stream.str());
  }


  void BitcastBuffer::dump(bool AsHex = true) const {
    llvm::errs() << "LSB\n  ";
    unsigned LineLength = 0;
    for (unsigned I = 0; I != (FinalBitSize / 8); ++I) {
      std::byte B = Data[I];
      if (AsHex) {
        std::stringstream stream;
        stream << std::hex << (int)B;
        llvm::errs() << stream.str();
        LineLength += stream.str().size() + 1;
      } else {
        llvm::errs() << std::bitset<8>((int)B).to_string();
        LineLength += 8 + 1;
        // llvm::errs() << (int)B;
      }
      llvm::errs() << ' ';
    }
    llvm::errs() << '\n';

    for (unsigned I = 0; I != LineLength; ++I)
      llvm::errs() << ' ';
    llvm::errs() << "MSB\n";
  }
#endif

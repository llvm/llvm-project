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

void BitcastBuffer::pushData(const std::byte *In, size_t BitOffset,
                             size_t BitWidth, Endian TargetEndianness) {
  for (unsigned It = 0; It != BitWidth; ++It) {
    bool BitValue = bitof(In, It);
    if (!BitValue)
      continue;

    unsigned DstBit;
    if (TargetEndianness == Endian::Little)
      DstBit = BitOffset + It;
    else
      DstBit = size() - BitOffset - BitWidth + It;

    unsigned DstByte = (DstBit / 8);
    Data[DstByte] |= std::byte{1} << (DstBit % 8);
  }
}

std::unique_ptr<std::byte[]>
BitcastBuffer::copyBits(unsigned BitOffset, unsigned BitWidth,
                        unsigned FullBitWidth, Endian TargetEndianness) const {
  assert(BitWidth <= FullBitWidth);
  assert(fullByte(FullBitWidth));
  auto Out = std::make_unique<std::byte[]>(FullBitWidth / 8);

  for (unsigned It = 0; It != BitWidth; ++It) {
    unsigned BitIndex;
    if (TargetEndianness == Endian::Little)
      BitIndex = BitOffset + It;
    else
      BitIndex = size() - BitWidth - BitOffset + It;

    bool BitValue = bitof(Data.get(), BitIndex);
    if (!BitValue)
      continue;
    unsigned DstBit = It;
    unsigned DstByte = (DstBit / 8);
    Out[DstByte] |= std::byte{1} << (DstBit % 8);
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

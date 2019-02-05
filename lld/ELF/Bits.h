//===- Bits.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_BITS_H
#define LLD_ELF_BITS_H

#include "Config.h"
#include "llvm/Support/Endian.h"

namespace lld {
namespace elf {

inline uint64_t readUint(uint8_t *Buf) {
  if (Config->Is64)
    return llvm::support::endian::read64(Buf, Config->Endianness);
  return llvm::support::endian::read32(Buf, Config->Endianness);
}

inline void writeUint(uint8_t *Buf, uint64_t Val) {
  if (Config->Is64)
    llvm::support::endian::write64(Buf, Val, Config->Endianness);
  else
    llvm::support::endian::write32(Buf, Val, Config->Endianness);
}

} // namespace elf
} // namespace lld

#endif

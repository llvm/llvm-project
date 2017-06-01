//===-- scudo_crc32.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_crc32.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_CRC32_H_
#define SCUDO_CRC32_H_

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __scudo {

enum : u8 {
  CRC32Software = 0,
  CRC32Hardware = 1,
};

u32 computeCRC32(u32 Crc, uptr Data, u8 HashType);

}  // namespace __scudo

#endif  // SCUDO_CRC32_H_

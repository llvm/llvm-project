//===-- scudo_crc32.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// CRC32 function leveraging hardware specific instructions. This has to be
/// kept separated to restrict the use of compiler specific flags to this file.
///
//===----------------------------------------------------------------------===//

#include "scudo_crc32.h"

namespace __scudo {

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
u32 computeHardwareCRC32(u32 Crc, uptr Data) {
  return CRC32_INTRINSIC(Crc, Data);
}
#endif  // defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)

}  // namespace __scudo

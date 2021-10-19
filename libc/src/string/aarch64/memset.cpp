//===-- Implementation of memset ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memset.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/memset_utils.h"

namespace __llvm_libc {

using namespace __llvm_libc::aarch64_memset;

inline static void AArch64Memset(char *dst, int value, size_t count) {
  if (count == 0)
    return;
  if (count <= 3) {
    SplatSet<_1>(dst, value);
    if (count > 1)
      SplatSet<Tail<_2>>(dst, value, count);
    return;
  }
  if (count <= 8)
    return SplatSet<HeadTail<_4>>(dst, value, count);
  if (count <= 16)
    return SplatSet<HeadTail<_8>>(dst, value, count);
  if (count <= 32)
    return SplatSet<HeadTail<_16>>(dst, value, count);
  if (count <= 96) {
    SplatSet<_32>(dst, value);
    if (count <= 64)
      return SplatSet<Tail<_32>>(dst, value, count);
    SplatSet<Skip<32>::Then<_32>>(dst, value);
    SplatSet<Tail<_32>>(dst, value, count);
    return;
  }
  if (count < 448 || value != 0 || !AArch64ZVA(dst, count))
    return SplatSet<Align<_16, Arg::_1>::Then<Loop<_64>>>(dst, value, count);
}

LLVM_LIBC_FUNCTION(void *, memset, (void *dst, int value, size_t count)) {
  AArch64Memset((char *)dst, value, count);
  return dst;
}

} // namespace __llvm_libc

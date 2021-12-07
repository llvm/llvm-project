//===-- Implementation of memset and bzero --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H

#include "src/__support/architectures.h"
#include "src/string/memory_utils/elements.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

// A general purpose implementation assuming cheap unaligned writes for sizes:
// 1, 2, 4, 8, 16, 32 and 64 Bytes. Note that some architecture can't store 32
// or 64 Bytes at a time, the compiler will expand them as needed.
//
// This implementation is subject to change as we benchmark more processors. We
// may also want to customize it for processors with specialized instructions
// that performs better (e.g. `rep stosb`).
//
// A note on the apparent discrepancy in the use of 32 vs 64 Bytes writes.
// We want to balance two things here:
//  - The number of redundant writes (when using `SetBlockOverlap`),
//  - The number of conditionals for sizes <=128 (~90% of memset calls are for
//    such sizes).
//
// For the range 64-128:
//  - SetBlockOverlap<64> uses no conditionals but always writes 128 Bytes this
//  is wasteful near 65 but efficient toward 128.
//  - SetAlignedBlocks<32> would consume between 3 and 4 conditionals and write
//  96 or 128 Bytes.
//  - Another approach could be to use an hybrid approach copy<64>+Overlap<32>
//  for 65-96 and copy<96>+Overlap<32> for 97-128
//
// Benchmarks showed that redundant writes were cheap (for Intel X86) but
// conditional were expensive, even on processor that do not support writing 64B
// at a time (pre-AVX512F). We also want to favor short functions that allow
// more hot code to fit in the iL1 cache.
//
// Above 128 we have to use conditionals since we don't know the upper bound in
// advance. SetAlignedBlocks<64> may waste up to 63 Bytes, SetAlignedBlocks<32>
// may waste up to 31 Bytes. Benchmarks showed that SetAlignedBlocks<64> was not
// superior for sizes that mattered.
inline static void inline_memset(char *dst, unsigned char value, size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_X86
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::x86;
  if (count == 0)
    return;
  if (count == 1)
    return splat_set<_1>(dst, value);
  if (count == 2)
    return splat_set<_2>(dst, value);
  if (count == 3)
    return splat_set<_3>(dst, value);
  if (count <= 8)
    return splat_set<HeadTail<_4>>(dst, value, count);
  if (count <= 16)
    return splat_set<HeadTail<_8>>(dst, value, count);
  if (count <= 32)
    return splat_set<HeadTail<_16>>(dst, value, count);
  if (count <= 64)
    return splat_set<HeadTail<_32>>(dst, value, count);
  if (count <= 128)
    return splat_set<HeadTail<_64>>(dst, value, count);
  return splat_set<Align<_32, Arg::Dst>::Then<Loop<_32>>>(dst, value, count);
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_AARCH64
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::aarch64_memset;
  if (count == 0)
    return;
  if (count <= 3) {
    splat_set<_1>(dst, value);
    if (count > 1)
      splat_set<Tail<_2>>(dst, value, count);
    return;
  }
  if (count <= 8)
    return splat_set<HeadTail<_4>>(dst, value, count);
  if (count <= 16)
    return splat_set<HeadTail<_8>>(dst, value, count);
  if (count <= 32)
    return splat_set<HeadTail<_16>>(dst, value, count);
  if (count <= 96) {
    splat_set<_32>(dst, value);
    if (count <= 64)
      return splat_set<Tail<_32>>(dst, value, count);
    splat_set<Skip<32>::Then<_32>>(dst, value);
    splat_set<Tail<_32>>(dst, value, count);
    return;
  }
  if (count < 448 || value != 0 || !AArch64ZVA(dst, count))
    return splat_set<Align<_16, Arg::_1>::Then<Loop<_64>>>(dst, value, count);
#else
  /////////////////////////////////////////////////////////////////////////////
  // Default
  /////////////////////////////////////////////////////////////////////////////
  using namespace ::__llvm_libc::scalar;

  if (count == 0)
    return;
  if (count == 1)
    return splat_set<_1>(dst, value);
  if (count == 2)
    return splat_set<_2>(dst, value);
  if (count == 3)
    return splat_set<_3>(dst, value);
  if (count <= 8)
    return splat_set<HeadTail<_4>>(dst, value, count);
  if (count <= 16)
    return splat_set<HeadTail<_8>>(dst, value, count);
  if (count <= 32)
    return splat_set<HeadTail<_16>>(dst, value, count);
  if (count <= 64)
    return splat_set<HeadTail<_32>>(dst, value, count);
  if (count <= 128)
    return splat_set<HeadTail<_64>>(dst, value, count);
  return splat_set<Align<_32, Arg::Dst>::Then<Loop<_32>>>(dst, value, count);
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H

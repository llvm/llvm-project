//===-- Memset utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_UTILS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_UTILS_H

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
//  - Another approach could be to use an hybrid approach Copy<64>+Overlap<32>
//  for 65-96 and Copy<96>+Overlap<32> for 97-128
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
inline static void GeneralPurposeMemset(char *dst, unsigned char value,
                                        size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return SplatSet<scalar::_1>(dst, value);
  if (count == 2)
    return SplatSet<scalar::_2>(dst, value);
  if (count == 3)
    return SplatSet<scalar::_3>(dst, value);
  if (count == 4)
    return SplatSet<scalar::_4>(dst, value);
  if (count <= 8)
    return SplatSet<HeadTail<scalar::_4>>(dst, value, count);
  if (count <= 16)
    return SplatSet<HeadTail<scalar::_8>>(dst, value, count);
  if (count <= 32)
    return SplatSet<HeadTail<scalar::_16>>(dst, value, count);
  if (count <= 64)
    return SplatSet<HeadTail<scalar::_32>>(dst, value, count);
  if (count <= 128)
    return SplatSet<HeadTail<scalar::_64>>(dst, value, count);
  return SplatSet<Align<scalar::_32, Arg::Dst>::Then<Loop<scalar::_32>>>(
      dst, value, count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_UTILS_H

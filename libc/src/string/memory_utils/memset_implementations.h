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
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
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
inline static void inline_memset(Ptr dst, uint8_t value, size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_X86
  /////////////////////////////////////////////////////////////////////////////
  static constexpr size_t kMaxSize = x86::kAvx512F ? 64
                                     : x86::kAvx   ? 32
                                     : x86::kSse2  ? 16
                                                   : 8;
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memset<1, kMaxSize>::block(dst, value);
  if (count == 2)
    return generic::Memset<2, kMaxSize>::block(dst, value);
  if (count == 3)
    return generic::Memset<3, kMaxSize>::block(dst, value);
  if (count <= 8)
    return generic::Memset<4, kMaxSize>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<8, kMaxSize>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<16, kMaxSize>::head_tail(dst, value, count);
  if (count <= 64)
    return generic::Memset<32, kMaxSize>::head_tail(dst, value, count);
  if (count <= 128)
    return generic::Memset<64, kMaxSize>::head_tail(dst, value, count);
  // Aligned loop
  generic::Memset<32, kMaxSize>::block(dst, value);
  align_to_next_boundary<32>(dst, count);
  return generic::Memset<32, kMaxSize>::loop_and_tail(dst, value, count);
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_AARCH64
  /////////////////////////////////////////////////////////////////////////////
  static constexpr size_t kMaxSize = aarch64::kNeon ? 16 : 8;
  if (count == 0)
    return;
  if (count <= 3) {
    generic::Memset<1, kMaxSize>::block(dst, value);
    if (count > 1)
      generic::Memset<2, kMaxSize>::tail(dst, value, count);
    return;
  }
  if (count <= 8)
    return generic::Memset<4, kMaxSize>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<8, kMaxSize>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<16, kMaxSize>::head_tail(dst, value, count);
  if (count <= (32 + 64)) {
    generic::Memset<32, kMaxSize>::block(dst, value);
    if (count <= 64)
      return generic::Memset<32, kMaxSize>::tail(dst, value, count);
    generic::Memset<32, kMaxSize>::block(dst + 32, value);
    generic::Memset<32, kMaxSize>::tail(dst, value, count);
    return;
  }
  if (count >= 448 && value == 0 && aarch64::neon::hasZva()) {
    generic::Memset<64, kMaxSize>::block(dst, 0);
    align_to_next_boundary<64>(dst, count);
    return aarch64::neon::BzeroCacheLine<64>::loop_and_tail(dst, 0, count);
  } else {
    generic::Memset<16, kMaxSize>::block(dst, value);
    align_to_next_boundary<16>(dst, count);
    return generic::Memset<64, kMaxSize>::loop_and_tail(dst, value, count);
  }
#else
  /////////////////////////////////////////////////////////////////////////////
  // Default
  /////////////////////////////////////////////////////////////////////////////
  static constexpr size_t kMaxSize = 8;
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memset<1, kMaxSize>::block(dst, value);
  if (count == 2)
    return generic::Memset<2, kMaxSize>::block(dst, value);
  if (count == 3)
    return generic::Memset<3, kMaxSize>::block(dst, value);
  if (count <= 8)
    return generic::Memset<4, kMaxSize>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<8, kMaxSize>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<16, kMaxSize>::head_tail(dst, value, count);
  if (count <= 64)
    return generic::Memset<32, kMaxSize>::head_tail(dst, value, count);
  if (count <= 128)
    return generic::Memset<64, kMaxSize>::head_tail(dst, value, count);
  // Aligned loop
  generic::Memset<32, kMaxSize>::block(dst, value);
  align_to_next_boundary<32>(dst, count);
  return generic::Memset<32, kMaxSize>::loop_and_tail(dst, value, count);
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H

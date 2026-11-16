//===-- Bcmp implementation for aarch64 -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE, LIBC_NOINLINE
#include "src/__support/macros/config.h"       // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/generic/bcmp_utils.h"
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType, load_unaligned

#include <stddef.h> // size_t

#ifdef __ARM_NEON
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128
#include <arm_neon.h>
#endif

namespace LIBC_NAMESPACE_DECL {

#ifdef __ARM_NEON
namespace internal {

// Comparing `__uint128_t` keeps 17..32 in GPRs (`ldp` + `cmp` + `ccmp`)
// without Neon-to-GPR transfer latency; LLVM would otherwise vectorize 64-bit
// comparisons into Neon. Checking the tail with a predicted-not-taken branch
// keeps tail address computation off the equal-path critical path.
LIBC_INLINE BcmpReturnType bcmp_aarch64_17_to_32(CPtr p1, CPtr p2,
                                                 size_t count) {
#ifdef LIBC_TYPES_HAS_INT128
  if (LIBC_UNLIKELY(bcmp_xor<__uint128_t>(p1 + count - 16, p2 + count - 16) !=
                    0))
    return BcmpReturnType::nonzero();
  return static_cast<uint32_t>(bcmp_xor<__uint128_t>(p1, p2) != 0);
#else
  return bcmp_17_to_32_64bit(p1, p2, count);
#endif
}

LIBC_INLINE uint8x16_t aarch64_xor16(CPtr p1, CPtr p2) {
  return veorq_u8(load_unaligned<uint8x16_t>(p1),
                  load_unaligned<uint8x16_t>(p2));
}

LIBC_INLINE uint8x16_t aarch64_xor32(CPtr p1, CPtr p2) {
  return vorrq_u8(aarch64_xor16(p1, p2), aarch64_xor16(p1 + 16, p2 + 16));
}

// Pairwise `umaxp` folds four 32-bit lanes into 64 bits (`umaxp + fmov`) with
// lower latency than a full horizontal `umaxv` reduction.
LIBC_INLINE bool aarch64_neon_nonzero(uint8x16_t diff) {
  const uint32x4_t d32 = vreinterpretq_u32_u8(diff);
  return vgetq_lane_u64(vreinterpretq_u64_u32(vpmaxq_u32(d32, d32)), 0) != 0;
}

LIBC_INLINE bool aarch64_neon_64_ne(CPtr p1, CPtr p2) {
  return aarch64_neon_nonzero(
      vorrq_u8(aarch64_xor32(p1, p2), aarch64_xor32(p1 + 32, p2 + 32)));
}

// Sizes > 32 pair Neon early-exit checks over `[4, count)` with a GPR `[0, 4)`
// final return (hiding Neon-to-GPR latency on equal inputs). The > 64 handler
// is kept out-of-line and tail-called so the 0..64 fast path stays compact.
LIBC_INLINE BcmpReturnType bcmp_aarch64_33_to_64(CPtr p1, CPtr p2,
                                                 size_t count) {
  const uint8x16_t head =
      vorrq_u8(aarch64_xor16(p1 + 4, p2 + 4), aarch64_xor16(p1 + 16, p2 + 16));
  if (LIBC_UNLIKELY(aarch64_neon_nonzero(
          vorrq_u8(head, aarch64_xor32(p1 + count - 32, p2 + count - 32)))))
    return BcmpReturnType::nonzero();
  return bcmp_xor<uint32_t>(p1, p2);
}

LIBC_NOINLINE LIBC_INLINE uint32_t bcmp_aarch64_over_64(CPtr p1, CPtr p2,
                                                        size_t count) {
  const uint8x16_t head =
      vorrq_u8(aarch64_xor16(p1 + 4, p2 + 4), aarch64_xor16(p1 + 16, p2 + 16));
  if (LIBC_UNLIKELY(aarch64_neon_nonzero(
          vorrq_u8(head, aarch64_xor32(p1 + 32, p2 + 32)))))
    return 1;
  if (count > 128) {
    if (LIBC_UNLIKELY(bcmp_xor<uint32_t>(p1, p2) != 0))
      return 1;
    const size_t last = count - 64;
    for (size_t i = 64 - static_cast<size_t>(distance_to_align_down<16>(p1));
         i < last; i += 64) {
      if (LIBC_UNLIKELY(aarch64_neon_64_ne(p1 + i, p2 + i)))
        return 1;
    }
  }
  if (LIBC_UNLIKELY(aarch64_neon_64_ne(p1 + count - 64, p2 + count - 64)))
    return 1;
  return bcmp_xor<uint32_t>(p1, p2);
}

} // namespace internal
#endif // __ARM_NEON

[[maybe_unused]] LIBC_INLINE
    BcmpReturnType inline_bcmp_aarch64(CPtr p1, CPtr p2, size_t count) {
#ifdef __ARM_NEON
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_aarch64_17_to_32(p1, p2, count);
  if (count <= 64)
    return internal::bcmp_aarch64_33_to_64(p1, p2, count);
  return internal::bcmp_aarch64_over_64(p1, p2, count);
#else
  return inline_bcmp_unaligned_64bit(p1, p2, count);
#endif // __ARM_NEON
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_BCMP_H

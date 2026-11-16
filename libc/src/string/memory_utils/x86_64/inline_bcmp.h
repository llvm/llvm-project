//===-- Bcmp implementation for x86 -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_BCMP_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_NOINLINE
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY, LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_SSE2
#include "src/string/memory_utils/generic/bcmp_utils.h"
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType, load_unaligned

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_CPU_HAS_SSE2)
#include <immintrin.h>
#endif

namespace LIBC_NAMESPACE_DECL {

#if defined(LIBC_TARGET_CPU_HAS_SSE2)
namespace internal {

LIBC_INLINE __m128i sse2_eq_16(CPtr a, CPtr b) {
  return _mm_cmpeq_epi8(load_unaligned<__m128i>(a), load_unaligned<__m128i>(b));
}

LIBC_INLINE __m128i sse2_eq_32(CPtr a, CPtr b) {
  return _mm_and_si128(sse2_eq_16(a, b), sse2_eq_16(a + 16, b + 16));
}

LIBC_INLINE uint32_t sse2_mismatch_mask(__m128i eq) {
  return static_cast<uint32_t>(_mm_movemask_epi8(eq)) ^ 0xffffU;
}

// 17..32 and 33..64 return a 16- or 32-bit mismatch bitmask directly in `int`
// (avoiding `ptest` + `setcc`). With AVX-512VL+BW, `_mm*_cmpneq_epi8_mask` runs
// both comparisons in parallel and combines with `kor` instead of LLVM lowering
// `and(cmpeq, cmpeq)` to a serialized masked `vpcmpeqb %k0{%k1}`.
LIBC_INLINE BcmpReturnType bcmp_x86_17_to_32(CPtr p1, CPtr p2, size_t count) {
#if defined(__AVX512VL__) && defined(__AVX512BW__)
  const __mmask16 head_ne = _mm_cmpneq_epi8_mask(load_unaligned<__m128i>(p1),
                                                 load_unaligned<__m128i>(p2));
  const __mmask16 tail_ne =
      _mm_cmpneq_epi8_mask(load_unaligned<__m128i>(p1 + count - 16),
                           load_unaligned<__m128i>(p2 + count - 16));
  return static_cast<uint32_t>(head_ne | tail_ne);
#else
  return sse2_mismatch_mask(_mm_and_si128(
      sse2_eq_16(p1, p2), sse2_eq_16(p1 + count - 16, p2 + count - 16)));
#endif
}

#if defined(LIBC_TARGET_ARCH_IS_X86_32)

// Compact 32-byte SSE2 loop + tail for 32-bit x86 (low GPR count & code size).
LIBC_INLINE BcmpReturnType bcmp_x86_32_gt32(CPtr p1, CPtr p2, size_t count) {
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < count - 32; i += 32) {
    if (LIBC_UNLIKELY(_mm_movemask_epi8(sse2_eq_32(p1 + i, p2 + i)) != 0xffff))
      return BcmpReturnType::nonzero();
  }
  return sse2_mismatch_mask(sse2_eq_32(p1 + count - 32, p2 + count - 32));
}

#else // x86-64

LIBC_INLINE BcmpReturnType bcmp_x86_33_to_64(CPtr p1, CPtr p2, size_t count) {
#if defined(__AVX512VL__) && defined(__AVX512BW__)
  const __mmask32 head_ne = _mm256_cmpneq_epi8_mask(
      load_unaligned<__m256i>(p1), load_unaligned<__m256i>(p2));
  const __mmask32 tail_ne =
      _mm256_cmpneq_epi8_mask(load_unaligned<__m256i>(p1 + count - 32),
                              load_unaligned<__m256i>(p2 + count - 32));
  return static_cast<uint32_t>(head_ne | tail_ne);
#elif defined(__AVX2__)
  const __m256i eq = _mm256_and_si256(
      _mm256_cmpeq_epi8(load_unaligned<__m256i>(p1),
                        load_unaligned<__m256i>(p2)),
      _mm256_cmpeq_epi8(load_unaligned<__m256i>(p1 + count - 32),
                        load_unaligned<__m256i>(p2 + count - 32)));
  return ~static_cast<uint32_t>(_mm256_movemask_epi8(eq));
#else
  return sse2_mismatch_mask(_mm_and_si128(
      sse2_eq_32(p1, p2), sse2_eq_32(p1 + count - 32, p2 + count - 32)));
#endif
}

// Out-of-line tail-called > 64-byte handler keeping the 0..64 fast path small
// and its instruction layout and `vzeroupper` placement clean. Medium sizes
// (`65..256` AVX2 / `65..128` SSE2) return the mismatch mask of the first block
// directly while checking remaining bytes with predicted early-exit branches,
// hiding tail load latency on equal inputs. Larger sizes check the head before
// entering the aligned loop so early mismatches exit immediately.
#if defined(__AVX2__)
LIBC_NOINLINE LIBC_INLINE uint32_t bcmp_x86_over_64(CPtr p1, CPtr p2,
                                                    size_t count) {
  auto xor_32 = [](CPtr a, CPtr b) -> __m256i {
    return _mm256_xor_si256(load_unaligned<__m256i>(a),
                            load_unaligned<__m256i>(b));
  };
  auto xor_128 = [&](CPtr a, CPtr b) -> __m256i {
    return _mm256_or_si256(
        _mm256_or_si256(xor_32(a, b), xor_32(a + 32, b + 32)),
        _mm256_or_si256(xor_32(a + 64, b + 64), xor_32(a + 96, b + 96)));
  };
  const __m256i head_eq = _mm256_cmpeq_epi8(load_unaligned<__m256i>(p1),
                                            load_unaligned<__m256i>(p2));
  const __m256i second_xor = xor_32(p1 + 32, p2 + 32);
  if (count <= 128) {
    const __m256i rest_xor = _mm256_or_si256(
        _mm256_or_si256(second_xor, xor_32(p1 + count - 64, p2 + count - 64)),
        xor_32(p1 + count - 32, p2 + count - 32));
    if (LIBC_UNLIKELY(!_mm256_testz_si256(rest_xor, rest_xor)))
      return 1;
    return ~static_cast<uint32_t>(_mm256_movemask_epi8(head_eq));
  }
  const __m256i head_rest =
      _mm256_or_si256(second_xor, _mm256_or_si256(xor_32(p1 + 64, p2 + 64),
                                                  xor_32(p1 + 96, p2 + 96)));
  const size_t last = count - 128;
  if (count <= 256) {
    const __m256i rest_xor =
        _mm256_or_si256(head_rest, xor_128(p1 + last, p2 + last));
    if (LIBC_UNLIKELY(!_mm256_testz_si256(rest_xor, rest_xor)))
      return 1;
    return ~static_cast<uint32_t>(_mm256_movemask_epi8(head_eq));
  }
  if (LIBC_UNLIKELY(~static_cast<uint32_t>(_mm256_movemask_epi8(head_eq)) !=
                        0 ||
                    !_mm256_testz_si256(head_rest, head_rest)))
    return 1;
  for (size_t i = 128 - static_cast<size_t>(distance_to_align_down<32>(p1));
       i < last; i += 128) {
    const __m256i block_xor = xor_128(p1 + i, p2 + i);
    if (LIBC_UNLIKELY(!_mm256_testz_si256(block_xor, block_xor)))
      return 1;
  }
  const __m256i tail_xor = xor_128(p1 + last, p2 + last);
  return ~static_cast<uint32_t>(_mm256_movemask_epi8(
      _mm256_cmpeq_epi8(tail_xor, _mm256_setzero_si256())));
}
#else
LIBC_NOINLINE LIBC_INLINE uint32_t bcmp_x86_over_64(CPtr p1, CPtr p2,
                                                    size_t count) {
  auto eq_64 = [](CPtr a, CPtr b) -> __m128i {
    return _mm_and_si128(sse2_eq_32(a, b), sse2_eq_32(a + 32, b + 32));
  };
  const __m128i head_eq = sse2_eq_32(p1, p2);
  const __m128i second_eq = sse2_eq_32(p1 + 32, p2 + 32);
  const size_t last = count - 64;
  if (count <= 128) {
    if (LIBC_UNLIKELY(_mm_movemask_epi8(_mm_and_si128(
                          second_eq, eq_64(p1 + last, p2 + last))) != 0xffff))
      return 1;
    return sse2_mismatch_mask(head_eq);
  }
  if (LIBC_UNLIKELY(_mm_movemask_epi8(_mm_and_si128(head_eq, second_eq)) !=
                    0xffff))
    return 1;
  for (size_t i = 64 - static_cast<size_t>(distance_to_align_down<16>(p1));
       i < last; i += 64) {
    if (LIBC_UNLIKELY(_mm_movemask_epi8(eq_64(p1 + i, p2 + i)) != 0xffff))
      return 1;
  }
  return sse2_mismatch_mask(eq_64(p1 + last, p2 + last));
}
#endif // __AVX2__ / SSE2

#endif // LIBC_TARGET_ARCH_IS_X86_32

} // namespace internal
#endif // LIBC_TARGET_CPU_HAS_SSE2

[[maybe_unused]] LIBC_INLINE BcmpReturnType inline_bcmp_x86(CPtr p1, CPtr p2,
                                                            size_t count) {
#if defined(LIBC_TARGET_CPU_HAS_SSE2)
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
#if defined(LIBC_TARGET_ARCH_IS_X86_32)
  if (count <= 16)
    return internal::bcmp_9_to_16_32bit(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_x86_17_to_32(p1, p2, count);
  return internal::bcmp_x86_32_gt32(p1, p2, count);
#else
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_x86_17_to_32(p1, p2, count);
  if (count <= 64)
    return internal::bcmp_x86_33_to_64(p1, p2, count);
  return internal::bcmp_x86_over_64(p1, p2, count);
#endif
#elif defined(LIBC_TARGET_ARCH_IS_X86_32)
  return inline_bcmp_unaligned_32bit(p1, p2, count);
#else
  return inline_bcmp_unaligned_64bit(p1, p2, count);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_BCMP_H

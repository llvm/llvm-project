//===-- Shared bcmp building blocks -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/attributes.h" // LIBC_ALWAYS_INLINE, LIBC_INLINE
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY, LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h" // LIBC_COMPILER_IS_MSVC
#include "src/__support/macros/properties/types.h"    // LIBC_TYPES_HAS_INT64
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType, load_unaligned

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {
namespace internal {

template <typename T> LIBC_INLINE T bcmp_xor(CPtr p1, CPtr p2) {
  return load_unaligned<T>(p1) ^ load_unaligned<T>(p2);
}

// Compares count bytes for count in 1..3 without reading outside [0, count).
LIBC_ALWAYS_INLINE BcmpReturnType bcmp_exact_1_to_3(CPtr p1, CPtr p2,
                                                    size_t count) {
  return cpp::to_integer<uint32_t>((p1[0] ^ p2[0]) |
                                   (p1[count - 1] ^ p2[count - 1]) |
                                   (p1[count >> 1] ^ p2[count >> 1]));
}

// Compares count bytes for count in 2..3 without reading outside [0, count).
LIBC_INLINE BcmpReturnType bcmp_2_to_3(CPtr p1, CPtr p2, size_t count) {
  const size_t tail = count - sizeof(uint16_t);
  return static_cast<uint32_t>(bcmp_xor<uint16_t>(p1, p2) |
                               bcmp_xor<uint16_t>(p1 + tail, p2 + tail));
}

// Compares count bytes for count in 4..8 without reading outside [0, count).
// On x86-64, checking the tail with a predicted-not-taken branch and returning
// the head XOR keeps the `(count - 4)` AGU off the equal-path critical path and
// lets the compare fuse with a load and branch. Hoisting both head loads while
// keeping the XOR after the branch avoids both sinking the loads and tail-
// merging an empty `ret` with AVX `vzeroupper; ret`. On AArch64, LLVM
// if-converts a tail branch into `cmp` + `csinc`, so OR-ing both XORs stays on
// general ALU pipes instead.
LIBC_INLINE BcmpReturnType bcmp_4_to_8(CPtr p1, CPtr p2, size_t count) {
  const size_t tail = count - sizeof(uint32_t);
#if defined(LIBC_TARGET_ARCH_IS_X86_64) && !defined(LIBC_COMPILER_IS_MSVC)
  uint32_t h1 = load_unaligned<uint32_t>(p1);
  uint32_t h2 = load_unaligned<uint32_t>(p2);
  asm("" : "+r"(h1), "+r"(h2));
  if (LIBC_UNLIKELY(load_unaligned<uint32_t>(p1 + tail) !=
                    load_unaligned<uint32_t>(p2 + tail)))
    return BcmpReturnType::nonzero();
  return h1 ^ h2;
#else
  return bcmp_xor<uint32_t>(p1, p2) | bcmp_xor<uint32_t>(p1 + tail, p2 + tail);
#endif
}

// Compares count bytes for count in 0..8 without reading outside [0, count).
// Checking `count >= 4` then `count >= 2` keeps the rare `count == 0` exit off
// the 2..8 paths and uses minimal loads for predictable short sizes.
LIBC_INLINE BcmpReturnType bcmp_0_to_8(CPtr p1, CPtr p2, size_t count) {
  if (count >= 4)
    return bcmp_4_to_8(p1, p2, count);
  if (count >= 2)
    return bcmp_2_to_3(p1, p2, count);
  if (LIBC_UNLIKELY(count == 0))
    return BcmpReturnType::zero();
  return cpp::to_integer<uint32_t>(p1[0] ^ p2[0]);
}

// Compares count in 9..16 using non-adjacent 32-bit loads so LLVM keeps all
// loads in GPRs (avoiding NEON SLP `vmov` and unaligned `ldrd`/`ldm` on Arm).
LIBC_INLINE BcmpReturnType bcmp_9_to_16_32bit(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY((bcmp_xor<uint32_t>(p1 + 4, p2 + 4) |
                     bcmp_xor<uint32_t>(p1 + count - 8, p2 + count - 8)) != 0))
    return BcmpReturnType::nonzero();
  return bcmp_xor<uint32_t>(p1, p2) |
         bcmp_xor<uint32_t>(p1 + count - 4, p2 + count - 4);
}

#ifdef LIBC_TYPES_HAS_INT64

// Compares count in 9..16: predicted early exit on `[4, count)` and direct
// 32-bit XOR return on `[0, 4)` (no `setne`/`cset` or `count`-indexed AGU).
LIBC_INLINE BcmpReturnType bcmp_9_to_16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY((bcmp_xor<uint32_t>(p1 + 4, p2 + 4) |
                     bcmp_xor<uint64_t>(p1 + count - 8, p2 + count - 8)) != 0))
    return BcmpReturnType::nonzero();
  return bcmp_xor<uint32_t>(p1, p2);
}

// Branchless 17..32 comparison using 64-bit loads. Interleaving head and tail
// pairs prevents LLVM from SLP-vectorizing adjacent loads on RVV targets.
LIBC_INLINE BcmpReturnType bcmp_17_to_32_64bit(CPtr p1, CPtr p2, size_t count) {
  const uint64_t d0 = bcmp_xor<uint64_t>(p1, p2) |
                      bcmp_xor<uint64_t>(p1 + count - 16, p2 + count - 16);
  const uint64_t d1 = bcmp_xor<uint64_t>(p1 + 8, p2 + 8) |
                      bcmp_xor<uint64_t>(p1 + count - 8, p2 + count - 8);
  return static_cast<uint32_t>((d0 | d1) != 0);
}

LIBC_INLINE bool bcmp_32_64bit_ne(CPtr p1, CPtr p2) {
  return (bcmp_xor<uint64_t>(p1, p2) | bcmp_xor<uint64_t>(p1 + 8, p2 + 8) |
          bcmp_xor<uint64_t>(p1 + 16, p2 + 16) |
          bcmp_xor<uint64_t>(p1 + 24, p2 + 24)) != 0;
}

#endif // LIBC_TYPES_HAS_INT64

} // namespace internal

// Generic unaligned 32-bit scalar `bcmp` (32-bit Arm with unaligned support,
// x86-32 without SSE2, RV32 with `__riscv_misaligned_fast`). Stride-8 pairing
// keeps all loads in GPRs (no NEON SLP `vmov` or unaligned `ldrd`/`ldm`).
[[maybe_unused]] LIBC_ALWAYS_INLINE BcmpReturnType
inline_bcmp_unaligned_32bit(CPtr p1, CPtr p2, size_t count) {
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16_32bit(p1, p2, count);
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < count - 16; i += 16) {
    if (LIBC_UNLIKELY((internal::bcmp_xor<uint32_t>(p1 + i, p2 + i) |
                       internal::bcmp_xor<uint32_t>(p1 + i + 8, p2 + i + 8)) !=
                      0))
      return BcmpReturnType::nonzero();
    if (LIBC_UNLIKELY(
            (internal::bcmp_xor<uint32_t>(p1 + i + 4, p2 + i + 4) |
             internal::bcmp_xor<uint32_t>(p1 + i + 12, p2 + i + 12)) != 0))
      return BcmpReturnType::nonzero();
  }
  const CPtr tail1 = p1 + count - 16;
  const CPtr tail2 = p2 + count - 16;
  if (LIBC_UNLIKELY((internal::bcmp_xor<uint32_t>(tail1 + 4, tail2 + 4) |
                     internal::bcmp_xor<uint32_t>(tail1 + 12, tail2 + 12)) !=
                    0))
    return BcmpReturnType::nonzero();
  return internal::bcmp_xor<uint32_t>(tail1, tail2) |
         internal::bcmp_xor<uint32_t>(tail1 + 8, tail2 + 8);
}

#ifdef LIBC_TYPES_HAS_INT64

// Generic unaligned 64-bit scalar `bcmp` (x86-64 without SSE2, AArch64
// `-mgeneral-regs-only`, RV64 with `__riscv_misaligned_fast`).
[[maybe_unused]] LIBC_ALWAYS_INLINE BcmpReturnType
inline_bcmp_unaligned_64bit(CPtr p1, CPtr p2, size_t count) {
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_17_to_32_64bit(p1, p2, count);
  if (LIBC_UNLIKELY(internal::bcmp_32_64bit_ne(p1, p2)))
    return BcmpReturnType::nonzero();
  if (count > 64) {
    const size_t last = count - 32;
    LIBC_LOOP_NOUNROLL
    for (size_t i = 32 - static_cast<size_t>(distance_to_align_down<8>(p1));
         i < last; i += 32) {
      if (LIBC_UNLIKELY(internal::bcmp_32_64bit_ne(p1 + i, p2 + i)))
        return BcmpReturnType::nonzero();
    }
  }
  return static_cast<uint32_t>(
      internal::bcmp_32_64bit_ne(p1 + count - 32, p2 + count - 32));
}

#endif // LIBC_TYPES_HAS_INT64

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_BCMP_UTILS_H

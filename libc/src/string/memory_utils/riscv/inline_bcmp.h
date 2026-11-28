//===-- Bcmp implementation for riscv ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_RISCV_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_RISCV_INLINE_BCMP_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/config.h"       // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/generic/bcmp_utils.h"
#include "src/string/memory_utils/utils.h" // CPtr, BcmpReturnType

#include <stddef.h> // size_t

#if !defined(LIBC_TARGET_ARCH_IS_RISCV64) &&                                   \
    !defined(LIBC_TARGET_ARCH_IS_RISCV32)
#error "Unimplemented"
#endif

#if defined(__riscv_vector) && defined(__riscv_v_intrinsic) &&                 \
    (__riscv_v_intrinsic >= 11000)
#include <riscv_vector.h>
#define LIBC_RISCV_HAS_RVV
#elif !defined(__riscv_misaligned_fast)
#include "src/string/memory_utils/generic/aligned_access.h"
#endif

namespace LIBC_NAMESPACE_DECL {

[[maybe_unused]] LIBC_INLINE BcmpReturnType inline_bcmp_riscv(CPtr p1, CPtr p2,
                                                              size_t count) {
#ifdef LIBC_RISCV_HAS_RVV
#ifdef __riscv_misaligned_fast
  if (count <= 8)
    return internal::bcmp_0_to_8(p1, p2, count);
#ifdef LIBC_TARGET_ARCH_IS_RISCV64
  if (count <= 16)
    return internal::bcmp_9_to_16(p1, p2, count);
  if (count <= 32)
    return internal::bcmp_17_to_32_64bit(p1, p2, count);
#else
  if (count <= 16)
    return internal::bcmp_9_to_16_32bit(p1, p2, count);
#endif
#else
  if (count == 0)
    return BcmpReturnType::zero();
  if (count <= 3)
    return internal::bcmp_exact_1_to_3(p1, p2, count);
#endif
  // VLA RVV 1.0 loop (LMUL=4); `do..while` avoids a loop-back branch on
  // single-chunk inputs after short sizes exit above.
  do {
    const size_t vl = __riscv_vsetvl_e8m4(count);
    const vuint8m4_t v1 =
        __riscv_vle8_v_u8m4(reinterpret_cast<const uint8_t *>(p1), vl);
    const vuint8m4_t v2 =
        __riscv_vle8_v_u8m4(reinterpret_cast<const uint8_t *>(p2), vl);
    const vbool2_t ne = __riscv_vmsne_vv_u8m4_b2(v1, v2, vl);
    if (LIBC_UNLIKELY(__riscv_vfirst_m_b2(ne, vl) >= 0))
      return BcmpReturnType::nonzero();
    p1 += vl;
    p2 += vl;
    count -= vl;
  } while (count > 0);
  return BcmpReturnType::zero();
#elif defined(__riscv_misaligned_fast) && defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_bcmp_unaligned_64bit(p1, p2, count);
#elif defined(__riscv_misaligned_fast)
  return inline_bcmp_unaligned_32bit(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_bcmp_aligned_access_64bit(p1, p2, count);
#else
  return inline_bcmp_aligned_access_32bit(p1, p2, count);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

#undef LIBC_RISCV_HAS_RVV

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_RISCV_INLINE_BCMP_H

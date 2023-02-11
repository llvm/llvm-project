//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {
[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_embedded_tiny(CPtr p1, CPtr p2, size_t count) {
  LIBC_LOOP_NOUNROLL
  for (size_t offset = 0; offset < count; ++offset)
    if (auto value = generic::Memcmp<1>::block(p1 + offset, p2 + offset))
      return value;
  return MemcmpReturnType::ZERO();
}

#if defined(LIBC_TARGET_ARCH_IS_X86) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_generic_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = generic::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return generic::Memcmp<16>::loop_and_tail(p1, p2, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_X86) ||
       // defined(LIBC_TARGET_ARCH_IS_AARCH64)

#if defined(LIBC_TARGET_ARCH_IS_X86)
[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_x86_sse2_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = x86::sse2::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return x86::sse2::Memcmp<16>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_x86_avx2_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count <= 32)
    return x86::sse2::Memcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Memcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx2::Memcmp<64>::head_tail(p1, p2, count);
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = x86::avx2::Memcmp<32>::block(p1, p2))
      return value;
    align_to_next_boundary<32, Arg::P1>(p1, p2, count);
  }
  return x86::avx2::Memcmp<32>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_x86_avx512bw_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count <= 32)
    return x86::sse2::Memcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Memcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx512bw::Memcmp<64>::head_tail(p1, p2, count);
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = x86::avx512bw::Memcmp<64>::block(p1, p2))
      return value;
    align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  }
  return x86::avx512bw::Memcmp<64>::loop_and_tail(p1, p2, count);
}

#endif // defined(LIBC_TARGET_ARCH_IS_X86)

#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aarch64_neon_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 128)) { // [128, ∞]
    if (auto value = generic::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
    return generic::Memcmp<32>::loop_and_tail(p1, p2, count);
  }
  if (generic::Bcmp<16>::block(p1, p2)) // [16, 16]
    return generic::Memcmp<16>::block(p1, p2);
  if (count < 32) // [17, 31]
    return generic::Memcmp<16>::tail(p1, p2, count);
  if (generic::Bcmp<16>::block(p1 + 16, p2 + 16)) // [32, 32]
    return generic::Memcmp<16>::block(p1 + 16, p2 + 16);
  if (count < 64) // [33, 63]
    return generic::Memcmp<32>::tail(p1, p2, count);
  // [64, 127]
  return generic::Memcmp<16>::loop_and_tail(p1 + 32, p2 + 32, count - 32);
}
#endif // defined(LIBC_TARGET_ARCH_IS_AARCH64)

LIBC_INLINE MemcmpReturnType inline_memcmp(CPtr p1, CPtr p2, size_t count) {
#if defined(LIBC_TARGET_ARCH_IS_X86) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
  if (count == 0)
    return MemcmpReturnType::ZERO();
  if (count == 1)
    return generic::Memcmp<1>::block(p1, p2);
  if (count == 2)
    return generic::Memcmp<2>::block(p1, p2);
  if (count == 3)
    return generic::Memcmp<3>::block(p1, p2);
  if (count <= 8)
    return generic::Memcmp<4>::head_tail(p1, p2, count);
  if (count <= 16)
    return generic::Memcmp<8>::head_tail(p1, p2, count);
#if defined(LIBC_TARGET_ARCH_IS_X86)
  if constexpr (x86::kAvx512BW)
    return inline_memcmp_x86_avx512bw_gt16(p1, p2, count);
  else if constexpr (x86::kAvx2)
    return inline_memcmp_x86_avx2_gt16(p1, p2, count);
  else if constexpr (x86::kSse2)
    return inline_memcmp_x86_sse2_gt16(p1, p2, count);
  else
    return inline_memcmp_generic_gt16(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  if constexpr (aarch64::kNeon)
    return inline_memcmp_aarch64_neon_gt16(p1, p2, count);
  else
    return inline_memcmp_generic_gt16(p1, p2, count);
#endif
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
  return inline_memcmp_embedded_tiny(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_GPU)
  return inline_memcmp_embedded_tiny(p1, p2, count);
#else
#error "Unsupported platform"
#endif
}

LIBC_INLINE int inline_memcmp(const void *p1, const void *p2, size_t count) {
  return static_cast<int>(inline_memcmp(reinterpret_cast<CPtr>(p1),
                                        reinterpret_cast<CPtr>(p2), count));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H

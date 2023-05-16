//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_byte_per_byte(CPtr p1, CPtr p2, size_t offset, size_t count) {
  LIBC_LOOP_NOUNROLL
  for (; offset < count; ++offset)
    if (p1[offset] != p2[offset])
      return BcmpReturnType::NONZERO();
  return BcmpReturnType::ZERO();
}

[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_aligned_access_64bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_bcmp_byte_per_byte(p1, p2, 0, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_bcmp_byte_per_byte(p1, p2, 0, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint64_t a;
    if (p2_alignment == 0)
      a = load64_aligned<uint64_t>(p2, offset);
    else if (p2_alignment == 4)
      a = load64_aligned<uint32_t, uint32_t>(p2, offset);
    else if (p2_alignment == 2)
      a = load64_aligned<uint16_t, uint16_t, uint16_t, uint16_t>(p2, offset);
    else
      a = load64_aligned<uint8_t, uint16_t, uint16_t, uint16_t, uint8_t>(
          p2, offset);
    uint64_t b = load64_aligned<uint64_t>(p1, offset);
    if (a != b)
      return BcmpReturnType::NONZERO();
  }
  return inline_bcmp_byte_per_byte(p1, p2, offset, count);
}

[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_aligned_access_32bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_bcmp_byte_per_byte(p1, p2, 0, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_bcmp_byte_per_byte(p1, p2, 0, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint32_t a;
    if (p2_alignment == 0)
      a = load32_aligned<uint32_t>(p2, offset);
    else if (p2_alignment == 2)
      a = load32_aligned<uint16_t, uint16_t>(p2, offset);
    else
      a = load32_aligned<uint8_t, uint16_t, uint8_t>(p2, offset);
    uint32_t b = load32_aligned<uint32_t>(p1, offset);
    if (a != b)
      return BcmpReturnType::NONZERO();
  }
  return inline_bcmp_byte_per_byte(p1, p2, offset, count);
}

#if defined(LIBC_TARGET_ARCH_IS_X86) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_generic_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count < 256)
    return generic::Bcmp<16>::loop_and_tail(p1, p2, count);
  if (auto value = generic::Bcmp<64>::block(p1, p2))
    return value;
  align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  return generic::Bcmp<64>::loop_and_tail(p1, p2, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_X86) ||
       // defined(LIBC_TARGET_ARCH_IS_AARCH64)

#if defined(LIBC_TARGET_ARCH_IS_X86)
[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_x86_sse2_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count < 256)
    return x86::sse2::Bcmp<16>::loop_and_tail(p1, p2, count);
  if (auto value = x86::sse2::Bcmp<16>::block(p1, p2))
    return value;
  align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  return x86::sse2::Bcmp<64>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_x86_avx2_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Bcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx2::Bcmp<64>::head_tail(p1, p2, count);
  if (LIBC_UNLIKELY(count >= 256)) {
    if (auto value = x86::avx2::Bcmp<64>::block(p1, p2))
      return value;
    align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  }
  return x86::avx2::Bcmp<64>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE BcmpReturnType
inline_bcmp_x86_avx512bw_gt16(CPtr p1, CPtr p2, size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Bcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx512bw::Bcmp<64>::head_tail(p1, p2, count);
  if (LIBC_UNLIKELY(count >= 256)) {
    if (auto value = x86::avx512bw::Bcmp<64>::block(p1, p2))
      return value;
    align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  }
  return x86::avx512bw::Bcmp<64>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE BcmpReturnType inline_bcmp_x86(CPtr p1, CPtr p2,
                                                            size_t count) {
  if (count == 0)
    return BcmpReturnType::ZERO();
  if (count == 1)
    return generic::Bcmp<1>::block(p1, p2);
  if (count == 2)
    return generic::Bcmp<2>::block(p1, p2);
  if (count <= 4)
    return generic::Bcmp<2>::head_tail(p1, p2, count);
  if (count <= 8)
    return generic::Bcmp<4>::head_tail(p1, p2, count);
  if (count <= 16)
    return generic::Bcmp<8>::head_tail(p1, p2, count);
  if constexpr (x86::kAvx512BW)
    return inline_bcmp_x86_avx512bw_gt16(p1, p2, count);
  else if constexpr (x86::kAvx2)
    return inline_bcmp_x86_avx2_gt16(p1, p2, count);
  else if constexpr (x86::kSse2)
    return inline_bcmp_x86_sse2_gt16(p1, p2, count);
  else
    return inline_bcmp_generic_gt16(p1, p2, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_X86)

#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
[[maybe_unused]] LIBC_INLINE BcmpReturnType inline_bcmp_aarch64(CPtr p1,
                                                                CPtr p2,
                                                                size_t count) {
  if (LIBC_LIKELY(count <= 32)) {
    if (LIBC_UNLIKELY(count >= 16)) {
      return aarch64::Bcmp<16>::head_tail(p1, p2, count);
    }
    switch (count) {
    case 0:
      return BcmpReturnType::ZERO();
    case 1:
      return generic::Bcmp<1>::block(p1, p2);
    case 2:
      return generic::Bcmp<2>::block(p1, p2);
    case 3:
      return generic::Bcmp<2>::head_tail(p1, p2, count);
    case 4:
      return generic::Bcmp<4>::block(p1, p2);
    case 5:
    case 6:
    case 7:
      return generic::Bcmp<4>::head_tail(p1, p2, count);
    case 8:
      return generic::Bcmp<8>::block(p1, p2);
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
      return generic::Bcmp<8>::head_tail(p1, p2, count);
    }
  }

  if (count <= 64)
    return aarch64::Bcmp<32>::head_tail(p1, p2, count);

  // Aligned loop if > 256, otherwise normal loop
  if (LIBC_UNLIKELY(count > 256)) {
    if (auto value = aarch64::Bcmp<32>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return aarch64::Bcmp<32>::loop_and_tail(p1, p2, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_AARCH64)

LIBC_INLINE BcmpReturnType inline_bcmp(CPtr p1, CPtr p2, size_t count) {
#if defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_bcmp_x86(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_bcmp_aarch64(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_bcmp_aligned_access_64bit(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
  return inline_bcmp_aligned_access_32bit(p1, p2, count);
#else
  return inline_bcmp_byte_per_byte(p1, p2, 0, count);
#endif
}

LIBC_INLINE int inline_bcmp(const void *p1, const void *p2, size_t count) {
  return static_cast<int>(inline_bcmp(reinterpret_cast<CPtr>(p1),
                                      reinterpret_cast<CPtr>(p2), count));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H

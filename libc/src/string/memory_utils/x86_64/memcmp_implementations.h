//===-- Memcmp implementation for x86_64 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H
#define LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H

#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h" // MemcmpReturnType

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_generic_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = generic::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return generic::Memcmp<16>::loop_and_tail(p1, p2, count);
}

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

LIBC_INLINE MemcmpReturnType inline_memcmp_x86(CPtr p1, CPtr p2, size_t count) {

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
  if constexpr (x86::kAvx512BW)
    return inline_memcmp_x86_avx512bw_gt16(p1, p2, count);
  else if constexpr (x86::kAvx2)
    return inline_memcmp_x86_avx2_gt16(p1, p2, count);
  else if constexpr (x86::kSse2)
    return inline_memcmp_x86_sse2_gt16(p1, p2, count);
  else
    return inline_memcmp_generic_gt16(p1, p2, count);
}
} // namespace __llvm_libc

#endif // LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H

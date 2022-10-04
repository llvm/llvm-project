//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

static inline BcmpReturnType inline_bcmp_generic_gt16(CPtr p1, CPtr p2,
                                                      size_t count) {
  if (count < 256)
    return generic::Bcmp<16>::loop_and_tail(p1, p2, count);
  if (auto value = generic::Bcmp<64>::block(p1, p2))
    return value;
  align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  return generic::Bcmp<64>::loop_and_tail(p1, p2, count);
}

#if defined(LLVM_LIBC_ARCH_X86)
static inline BcmpReturnType inline_bcmp_x86_sse2_gt16(CPtr p1, CPtr p2,
                                                       size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count < 256)
    return x86::sse2::Bcmp<16>::loop_and_tail(p1, p2, count);
  if (auto value = x86::sse2::Bcmp<16>::block(p1, p2))
    return value;
  align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  return x86::sse2::Bcmp<64>::loop_and_tail(p1, p2, count);
}

static inline BcmpReturnType inline_bcmp_x86_avx2_gt16(CPtr p1, CPtr p2,
                                                       size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Bcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx2::Bcmp<64>::head_tail(p1, p2, count);
  if (unlikely(count >= 256)) {
    if (auto value = x86::avx2::Bcmp<64>::block(p1, p2))
      return value;
    align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  }
  return x86::avx2::Bcmp<64>::loop_and_tail(p1, p2, count);
}

static inline BcmpReturnType inline_bcmp_x86_avx512bw_gt16(CPtr p1, CPtr p2,
                                                           size_t count) {
  if (count <= 32)
    return x86::sse2::Bcmp<16>::head_tail(p1, p2, count);
  if (count <= 64)
    return x86::avx2::Bcmp<32>::head_tail(p1, p2, count);
  if (count <= 128)
    return x86::avx512bw::Bcmp<64>::head_tail(p1, p2, count);
  if (unlikely(count >= 256)) {
    if (auto value = x86::avx512bw::Bcmp<64>::block(p1, p2))
      return value;
    align_to_next_boundary<64, Arg::P1>(p1, p2, count);
  }
  return x86::avx512bw::Bcmp<64>::loop_and_tail(p1, p2, count);
}
#endif // defined(LLVM_LIBC_ARCH_X86)

static inline BcmpReturnType inline_bcmp(CPtr p1, CPtr p2, size_t count) {
#if defined(LLVM_LIBC_ARCH_AARCH64)
  if (likely(count <= 32)) {
    if (unlikely(count >= 16)) {
      return generic::Bcmp<16>::head_tail(p1, p2, count);
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
    case 5 ... 7:
      return generic::Bcmp<4>::head_tail(p1, p2, count);
    case 8:
      return generic::Bcmp<8>::block(p1, p2);
    case 9 ... 15:
      return generic::Bcmp<8>::head_tail(p1, p2, count);
    }
  }

  if (count <= 64)
    return generic::Bcmp<32>::head_tail(p1, p2, count);

  // Aligned loop if > 256, otherwise normal loop
  if (count > 256) {
    if (auto value = generic::Bcmp<32>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return generic::Bcmp<32>::loop_and_tail(p1, p2, count);
#else
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
#if defined(LLVM_LIBC_ARCH_X86)
  if constexpr (x86::kAvx512BW)
    return inline_bcmp_x86_avx512bw_gt16(p1, p2, count);
  else if constexpr (x86::kAvx2)
    return inline_bcmp_x86_avx2_gt16(p1, p2, count);
  else if constexpr (x86::kSse2)
    return inline_bcmp_x86_sse2_gt16(p1, p2, count);
  else
    return inline_bcmp_generic_gt16(p1, p2, count);
#else
  return inline_bcmp_generic_gt16(p1, p2, count);
#endif
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BCMP_IMPLEMENTATIONS_H

//===-- Elementary operations for aarch64 --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_AARCH64_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_AARCH64_H

#if defined(__arm__) || defined(__aarch64__)

#include <src/string/memory_utils/elements.h>
#include <stddef.h> // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace __llvm_libc {
namespace aarch64_memset {
#ifdef __ARM_NEON
struct Splat8 {
  static constexpr size_t kSize = 8;
  static void SplatSet(char *dst, const unsigned char value) {
    vst1_u8((uint8_t *)dst, vdup_n_u8(value));
  }
};

struct Splat16 {
  static constexpr size_t kSize = 16;
  static void SplatSet(char *dst, const unsigned char value) {
    vst1q_u8((uint8_t *)dst, vdupq_n_u8(value));
  }
};

using _8 = Splat8;
using _16 = Splat16;
#else
using _8 = __llvm_libc::scalar::_8;
using _16 = Repeated<_8, 2>;
#endif // __ARM_NEON

using _1 = __llvm_libc::scalar::_1;
using _2 = __llvm_libc::scalar::_2;
using _3 = __llvm_libc::scalar::_3;
using _4 = __llvm_libc::scalar::_4;
using _32 = Chained<_16, _16>;
using _64 = Chained<_32, _32>;

struct ZVA {
  static constexpr size_t kSize = 64;
  static void SplatSet(char *dst, const unsigned char value) {
    asm("dc zva, %[dst]" : : [dst] "r"(dst) : "memory");
  }
};

inline static bool AArch64ZVA(char *dst, size_t count) {
  uint64_t zva_val;
  asm("mrs %[zva_val], dczid_el0" : [zva_val] "=r"(zva_val));
  if ((zva_val & 31) != 4)
    return false;
  SplatSet<Align<_64, Arg::_1>::Then<Loop<ZVA, _64>>>(dst, 0, count);
  return true;
}

} // namespace aarch64_memset

namespace aarch64 {

using _1 = __llvm_libc::scalar::_1;
using _2 = __llvm_libc::scalar::_2;
using _3 = __llvm_libc::scalar::_3;
using _4 = __llvm_libc::scalar::_4;
using _8 = __llvm_libc::scalar::_8;
using _16 = __llvm_libc::scalar::_16;

#ifdef __ARM_NEON
struct N32 {
  static constexpr size_t kSize = 32;
  static bool Equals(const char *lhs, const char *rhs) {
    uint8x16_t l_0 = vld1q_u8((const uint8_t *)lhs);
    uint8x16_t r_0 = vld1q_u8((const uint8_t *)rhs);
    uint8x16_t l_1 = vld1q_u8((const uint8_t *)(lhs + 16));
    uint8x16_t r_1 = vld1q_u8((const uint8_t *)(rhs + 16));
    uint8x16_t temp = vpmaxq_u8(veorq_u8(l_0, r_0), veorq_u8(l_1, r_1));
    uint64_t res =
        vgetq_lane_u64(vreinterpretq_u64_u8(vpmaxq_u8(temp, temp)), 0);
    return res == 0;
  }
  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    uint8x16_t l_0 = vld1q_u8((const uint8_t *)lhs);
    uint8x16_t r_0 = vld1q_u8((const uint8_t *)rhs);
    uint8x16_t l_1 = vld1q_u8((const uint8_t *)(lhs + 16));
    uint8x16_t r_1 = vld1q_u8((const uint8_t *)(rhs + 16));
    uint8x16_t temp = vpmaxq_u8(veorq_u8(l_0, r_0), veorq_u8(l_1, r_1));
    uint64_t res =
        vgetq_lane_u64(vreinterpretq_u64_u8(vpmaxq_u8(temp, temp)), 0);
    if (res == 0)
      return 0;
    size_t index = (__builtin_ctzl(res) >> 3) << 2;
    uint32_t l = *((const uint32_t *)(lhs + index));
    uint32_t r = *((const uint32_t *)(rhs + index));
    return __llvm_libc::scalar::_4::ScalarThreeWayCompare(l, r);
  }
};

using _32 = N32;
#else
using _32 = __llvm_libc::scalar::_32;
#endif // __ARM_NEON

} // namespace aarch64
} // namespace __llvm_libc

#endif // defined(__arm__) || defined(__aarch64__)

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_AARCH64_H

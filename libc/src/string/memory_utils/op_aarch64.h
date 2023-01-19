//===-- aarch64 implementation of memory function building blocks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides aarch64 specific building blocks to compose memory
// functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_AARCH64_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_AARCH64_H

#include "src/__support/architectures.h"

#if defined(LLVM_LIBC_ARCH_AARCH64)

#include "src/__support/common.h"
#include "src/string/memory_utils/op_generic.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif //__ARM_NEON

namespace __llvm_libc::aarch64 {

static inline constexpr bool kNeon = LLVM_LIBC_IS_DEFINED(__ARM_NEON);

namespace neon {

template <size_t Size> struct BzeroCacheLine {
  static constexpr size_t SIZE = Size;

  LIBC_INLINE void block(Ptr dst, uint8_t) {
    static_assert(Size == 64);
#if __SIZEOF_POINTER__ == 4
    asm("dc zva, %w[dst]" : : [dst] "r"(dst) : "memory");
#else
    asm("dc zva, %[dst]" : : [dst] "r"(dst) : "memory");
#endif
  }

  LIBC_INLINE void loop_and_tail(Ptr dst, uint8_t value, size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    size_t offset = 0;
    do {
      block(dst + offset, value);
      offset += SIZE;
    } while (offset < count - SIZE);
    // Unaligned store, we can't use 'dc zva' here.
    static constexpr size_t kMaxSize = kNeon ? 16 : 8;
    generic::Memset<Size, kMaxSize>::tail(dst, value, count);
  }
};

inline static bool hasZva() {
  uint64_t zva_val;
  asm("mrs %[zva_val], dczid_el0" : [zva_val] "=r"(zva_val));
  // DC ZVA is permitted if DZP, bit [4] is zero.
  // BS, bits [3:0] is log2 of the block count in words.
  // So the next line checks whether the instruction is permitted and block
  // count is 16 words (i.e. 64 bytes).
  return (zva_val & 0b11111) == 0b00100;
}

} // namespace neon

///////////////////////////////////////////////////////////////////////////////
// Bcmp
template <size_t Size> struct Bcmp {
  static constexpr size_t SIZE = Size;
  static constexpr size_t BlockSize = 32;

  LIBC_INLINE static const unsigned char *as_u8(CPtr ptr) {
    return reinterpret_cast<const unsigned char *>(ptr);
  }

  LIBC_INLINE static BcmpReturnType block(CPtr p1, CPtr p2) {
    if constexpr (Size == 16) {
      auto _p1 = as_u8(p1);
      auto _p2 = as_u8(p2);
      uint8x16_t a = vld1q_u8(_p1);
      uint8x16_t n = vld1q_u8(_p2);
      uint8x16_t an = veorq_u8(a, n);
      uint32x2_t an_reduced = vqmovn_u64(vreinterpretq_u64_u8(an));
      return vmaxv_u32(an_reduced);
    } else if constexpr (Size == 32) {
      auto _p1 = as_u8(p1);
      auto _p2 = as_u8(p2);
      uint8x16_t a = vld1q_u8(_p1);
      uint8x16_t b = vld1q_u8(_p1 + 16);
      uint8x16_t n = vld1q_u8(_p2);
      uint8x16_t o = vld1q_u8(_p2 + 16);
      uint8x16_t an = veorq_u8(a, n);
      uint8x16_t bo = veorq_u8(b, o);
      // anbo = (a ^ n) | (b ^ o).  At least one byte is nonzero if there is
      // a difference between the two buffers.  We reduce this value down to 4
      // bytes in two steps. First, calculate the saturated move value when
      // going from 2x64b to 2x32b. Second, compute the max of the 2x32b to get
      // a single 32 bit nonzero value if a mismatch occurred.
      uint8x16_t anbo = vorrq_u8(an, bo);
      uint32x2_t anbo_reduced = vqmovn_u64(vreinterpretq_u64_u8(anbo));
      return vmaxv_u32(anbo_reduced);
    } else if constexpr ((Size % BlockSize) == 0) {
      for (size_t offset = 0; offset < Size; offset += BlockSize)
        if (auto value = Bcmp<BlockSize>::block(p1 + offset, p2 + offset))
          return value;
    } else {
      deferred_static_assert("SIZE not implemented");
    }
    return BcmpReturnType::ZERO();
  }

  LIBC_INLINE static BcmpReturnType tail(CPtr p1, CPtr p2, size_t count) {
    return block(p1 + count - SIZE, p2 + count - SIZE);
  }

  LIBC_INLINE static BcmpReturnType head_tail(CPtr p1, CPtr p2, size_t count) {
    if constexpr (Size == 16) {
      auto _p1 = as_u8(p1);
      auto _p2 = as_u8(p2);
      uint8x16_t a = vld1q_u8(_p1);
      uint8x16_t b = vld1q_u8(_p1 + count - 16);
      uint8x16_t n = vld1q_u8(_p2);
      uint8x16_t o = vld1q_u8(_p2 + count - 16);
      uint8x16_t an = veorq_u8(a, n);
      uint8x16_t bo = veorq_u8(b, o);
      // anbo = (a ^ n) | (b ^ o)
      uint8x16_t anbo = vorrq_u8(an, bo);
      uint32x2_t anbo_reduced = vqmovn_u64(vreinterpretq_u64_u8(anbo));
      return vmaxv_u32(anbo_reduced);
    } else if constexpr (Size == 32) {
      auto _p1 = as_u8(p1);
      auto _p2 = as_u8(p2);
      uint8x16_t a = vld1q_u8(_p1);
      uint8x16_t b = vld1q_u8(_p1 + 16);
      uint8x16_t c = vld1q_u8(_p1 + count - 16);
      uint8x16_t d = vld1q_u8(_p1 + count - 32);
      uint8x16_t n = vld1q_u8(_p2);
      uint8x16_t o = vld1q_u8(_p2 + 16);
      uint8x16_t p = vld1q_u8(_p2 + count - 16);
      uint8x16_t q = vld1q_u8(_p2 + count - 32);
      uint8x16_t an = veorq_u8(a, n);
      uint8x16_t bo = veorq_u8(b, o);
      uint8x16_t cp = veorq_u8(c, p);
      uint8x16_t dq = veorq_u8(d, q);
      uint8x16_t anbo = vorrq_u8(an, bo);
      uint8x16_t cpdq = vorrq_u8(cp, dq);
      // abnocpdq = ((a ^ n) | (b ^ o)) | ((c ^ p) | (d ^ q)).  Reduce this to
      // a nonzero 32 bit value if a mismatch occurred.
      uint64x2_t abnocpdq = vreinterpretq_u64_u8(anbo | cpdq);
      uint32x2_t abnocpdq_reduced = vqmovn_u64(abnocpdq);
      return vmaxv_u32(abnocpdq_reduced);
    } else {
      deferred_static_assert("SIZE not implemented");
    }
    return BcmpReturnType::ZERO();
  }

  LIBC_INLINE static BcmpReturnType loop_and_tail(CPtr p1, CPtr p2,
                                                  size_t count) {
    static_assert(Size > 1, "a loop of size 1 does not need tail");
    size_t offset = 0;
    do {
      if (auto value = block(p1 + offset, p2 + offset))
        return value;
      offset += SIZE;
    } while (offset < count - SIZE);
    return tail(p1, p2, count);
  }
};

} // namespace __llvm_libc::aarch64

#endif // LLVM_LIBC_ARCH_AARCH64

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_AARCH64_H

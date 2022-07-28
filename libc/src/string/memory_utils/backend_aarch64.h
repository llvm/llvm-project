//===-- Elementary operations for aarch64 ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_AARCH64_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_AARCH64_H

#if !defined(LLVM_LIBC_ARCH_AARCH64)
#include "src/string/memory_utils/backend_scalar.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace __llvm_libc {

struct Aarch64Backend : public Scalar64BitBackend {
  static constexpr bool IS_BACKEND_TYPE = true;

  template <typename T, Temporality TS, Aligned AS,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline T load(const T *src) {
    return Scalar64BitBackend::template load<T, TS, AS>(src);
  }
};

// Implementation of the SizedOp abstraction for the set operation.
struct Zva64 {
  static constexpr size_t SIZE = 64;

  template <typename DstAddrT>
  static inline void set(DstAddrT dst, ubyte value) {
#if __SIZEOF_POINTER__ == 4
    asm("dc zva, %w[dst]" : : [dst] "r"(dst) : "memory");
#else
    asm("dc zva, %[dst]" : : [dst] "r"(dst) : "memory");
#endif
  }
};

inline static bool hasZva() {
  uint64_t zva_val;
  asm("mrs %[zva_val], dczid_el0" : [zva_val] "=r"(zva_val));
  // DC ZVA is permitted if DZP, bit [4] is zero.
  // BS, bits [3:0] is log2 of the block size in words.
  // So the next line checks whether the instruction is permitted and block size
  // is 16 words (i.e. 64 bytes).
  return (zva_val & 0b11111) == 0b00100;
}

namespace aarch64 {
using _1 = SizedOp<Aarch64Backend, 1>;
using _2 = SizedOp<Aarch64Backend, 2>;
using _3 = SizedOp<Aarch64Backend, 3>;
using _4 = SizedOp<Aarch64Backend, 4>;
using _8 = SizedOp<Aarch64Backend, 8>;
using _16 = SizedOp<Aarch64Backend, 16>;
using _32 = SizedOp<Aarch64Backend, 32>;
using _64 = SizedOp<Aarch64Backend, 64>;
using _128 = SizedOp<Aarch64Backend, 128>;
} // namespace aarch64

} // namespace __llvm_libc

#endif // LLVM_LIBC_ARCH_AARCH64

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_AARCH64_H

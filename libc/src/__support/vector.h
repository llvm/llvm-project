//===-- Helper functions for SIMD extensions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_VECTOR_H
#define LLVM_LIBC_SRC___SUPPORT_VECTOR_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(LIBC_HAS_VECTOR_TYPE, "Compiler does not support vector types.");

namespace vector {

template <size_t N> struct BitmaskTy;
template <> struct BitmaskTy<1> {
  using type = uint8_t;
};
template <> struct BitmaskTy<8> {
  using type = uint8_t;
};
template <> struct BitmaskTy<16> {
  using type = uint16_t;
};
template <> struct BitmaskTy<32> {
  using type = uint32_t;
};
template <> struct BitmaskTy<64> {
  using type = uint64_t;
};

template <typename T> struct SSE2 {
  static constexpr size_t WIDTH = 16;
  static constexpr size_t NUM_ELEMENTS = WIDTH / sizeof(T);
};
template <typename T> struct AVX2 {
  static constexpr size_t WIDTH = 32;
  static constexpr size_t NUM_ELEMENTS = WIDTH / sizeof(T);
};
template <typename T> struct AVX512 {
  static constexpr size_t WIDTH = 64;
  static constexpr size_t NUM_ELEMENTS = WIDTH / sizeof(T);
};
template <typename T> struct Neon {
  static constexpr size_t WIDTH = 16;
  static constexpr size_t NUM_ELEMENTS = WIDTH / sizeof(T);
};

#if defined(LIBC_TARGET_CPU_HAS_AVX512F)
template <typename T> using Platform = AVX512<T>;
#elif defined(LIBC_TARGET_CPU_HAS_AVX2)
template <typename T> using Platform = AVX2<T>;
#elif defined(LIBC_TARGET_CPU_HAS_SSE2)
template <typename T> using Platform = SSE2<T>;
#elif defined(LIBC_TARGET_CPU_HAS_ARM_NEON)
template <typename T> using Platform = Neon<T>;
#endif

template <typename T, size_t N = Platform<T>::NUM_ELEMENTS>
using Vector = T LIBC_VECTOR_TYPE(N);

template <typename To, typename From, size_t N>
LIBC_INLINE Vector<To, N> convert(const Vector<From, N> &v) {
  return __builtin_convertvector(v, Vector<To, N>);
}

template <typename T, size_t N>
LIBC_INLINE typename BitmaskTy<N>::type to_bitmask(const Vector<T, N> &v) {
  return cpp::bit_cast<typename BitmaskTy<N>::type>(convert<bool, T, N>(v));
}
} // namespace vector

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_VECTOR_H

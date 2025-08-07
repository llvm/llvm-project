//===-- Portable SIMD library similar to stdx::simd -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a generic interface into fixed-size SIMD instructions
// using the clang vector type. The API shares some similarities with the
// stdx::simd proposal, but instead chooses to use vectors as primitive types
// with several extra helper functions.
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits/integral_constant.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_SIMD_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_SIMD_H

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

static_assert(LIBC_HAS_VECTOR_TYPE, "compiler does not support vector types");

namespace internal {

template <size_t Size> struct get_as_integer_type;

template <> struct get_as_integer_type<1> {
  using type = uint8_t;
};
template <> struct get_as_integer_type<2> {
  using type = uint16_t;
};
template <> struct get_as_integer_type<4> {
  using type = uint32_t;
};
template <> struct get_as_integer_type<8> {
  using type = uint64_t;
};

template <class T>
using get_as_integer_type_t = typename get_as_integer_type<sizeof(T)>::type;

#if defined(LIBC_TARGET_CPU_HAS_AVX512F)
template <typename T>
inline constexpr size_t native_vector_size = 64 / sizeof(T);
#elif defined(LIBC_TARGET_CPU_HAS_AVX2)
template <typename T>
inline constexpr size_t native_vector_size = 32 / sizeof(T);
#elif defined(LIBC_TARGET_CPU_HAS_SSE2) || defined(LIBC_TARGET_CPU_HAS_ARM_NEON)
template <typename T>
inline constexpr size_t native_vector_size = 16 / sizeof(T);
#else
template <typename T> inline constexpr size_t native_vector_size = 1;
#endif
} // namespace internal

// Type aliases.
template <typename T, size_t N>
using fixed_size_simd = T [[clang::ext_vector_type(N)]];
template <typename T, size_t N = internal::native_vector_size<T>>
using simd = T [[clang::ext_vector_type(N)]];
template <typename T>
using simd_mask = simd<bool, internal::native_vector_size<T>>;

// Type trait helpers.
template <typename T> struct simd_size : cpp::integral_constant<size_t, 1> {};
template <typename T, unsigned N>
struct simd_size<T [[clang::ext_vector_type(N)]]>
    : cpp::integral_constant<size_t, N> {};
template <class T> constexpr size_t simd_size_v = simd_size<T>::value;

template <typename T> struct is_simd : cpp::integral_constant<bool, false> {};
template <typename T, unsigned N>
struct is_simd<T [[clang::ext_vector_type(N)]]>
    : cpp::integral_constant<bool, true> {};
template <class T> constexpr bool is_simd_v = is_simd<T>::value;

template <typename T>
struct is_simd_mask : cpp::integral_constant<bool, false> {};
template <unsigned N>
struct is_simd_mask<bool [[clang::ext_vector_type(N)]]>
    : cpp::integral_constant<bool, true> {};
template <class T> constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

template <typename T>
using enable_if_simd_t = cpp::enable_if_t<is_simd_v<T>, T>;

// Casting.
template <typename To, typename From, size_t N>
LIBC_INLINE constexpr simd<To, N> simd_cast(simd<From, N> v) {
  return __builtin_convertvector(v, simd<To, N>);
}

// SIMD mask operations.
template <size_t N> LIBC_INLINE constexpr bool all_of(simd<bool, N> m) {
  return __builtin_reduce_and(m);
}
template <size_t N> LIBC_INLINE constexpr bool any_of(simd<bool, N> m) {
  return __builtin_reduce_or(m);
}
template <size_t N> LIBC_INLINE constexpr bool none_of(simd<bool, N> m) {
  return !any_of(m);
}
template <size_t N> LIBC_INLINE constexpr bool some_of(simd<bool, N> m) {
  return any_of(m) && !all_of(m);
}
template <size_t N> LIBC_INLINE constexpr int popcount(simd<bool, N> m) {
  return __builtin_popcountg(m);
}
template <size_t N> LIBC_INLINE constexpr int find_first_set(simd<bool, N> m) {
  return __builtin_ctzg(m);
}
template <size_t N> LIBC_INLINE constexpr int find_last_set(simd<bool, N> m) {
  constexpr size_t size = simd_size_v<simd<bool, N>>;
  return size - __builtin_clzg(m);
}

// Elementwise operations.
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> min(simd<T, N> x, simd<T, N> y) {
  return __builtin_elementwise_min(x, y);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> max(simd<T, N> x, simd<T, N> y) {
  return __builtin_elementwise_max(x, y);
}

// Reduction operations.
template <typename T, size_t N, typename Op = cpp::plus<>>
LIBC_INLINE constexpr T reduce(simd<T, N> v, Op op = {}) {
  return reduce(v, op);
}
template <typename T, size_t N>
LIBC_INLINE constexpr T reduce(simd<T, N> v, cpp::plus<>) {
  return __builtin_reduce_add(v);
}
template <typename T, size_t N>
LIBC_INLINE constexpr T reduce(simd<T, N> v, cpp::multiplies<>) {
  return __builtin_reduce_mul(v);
}
template <typename T, size_t N>
LIBC_INLINE constexpr T reduce(simd<T, N> v, cpp::bit_and<>) {
  return __builtin_reduce_and(v);
}
template <typename T, size_t N>
LIBC_INLINE constexpr T reduce(simd<T, N> v, cpp::bit_or<>) {
  return __builtin_reduce_or(v);
}
template <typename T, size_t N>
LIBC_INLINE constexpr T reduce(simd<T, N> v, cpp::bit_xor<>) {
  return __builtin_reduce_xor(v);
}
template <typename T, size_t N> LIBC_INLINE constexpr T hmin(simd<T, N> v) {
  return __builtin_reduce_min(v);
}
template <typename T, size_t N> LIBC_INLINE constexpr T hmax(simd<T, N> v) {
  return __builtin_reduce_max(v);
}

// Accessor helpers.
template <typename T>
LIBC_INLINE enable_if_simd_t<T> load_unaligned(const void *ptr) {
  T tmp;
  __builtin_memcpy(&tmp, ptr, sizeof(T));
  return tmp;
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> load_aligned(const void *ptr) {
  return *reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignof(T)));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> store_unaligned(T v, void *ptr) {
  __builtin_memcpy(ptr, &v, sizeof(T));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> store_aligned(T v, void *ptr) {
  *reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignof(T))) = v;
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> masked_load(simd<bool, simd_size_v<T>> m,
                                            void *ptr) {
  return __builtin_masked_load(
      m, reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignof(T))));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> masked_store(simd<bool, simd_size_v<T>> m, T v,
                                             void *ptr) {
  __builtin_masked_store(
      m, v, reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignof(T))));
}

// Construction helpers.
template <typename T, size_t N> LIBC_INLINE constexpr simd<T, N> splat(T v) {
  return simd<T, N>(v);
}
template <typename T> LIBC_INLINE constexpr simd<T> splat(T v) {
  return splat<T, simd_size_v<simd<T>>>(v);
}
template <typename T, unsigned N>
LIBC_INLINE constexpr simd<T, N> iota(T base = T(0), T step = T(1)) {
  simd<T, N> v{};
  for (unsigned i = 0; i < N; ++i)
    v[i] = base + T(i) * step;
  return v;
}
template <typename T>
LIBC_INLINE constexpr simd<T> iota(T base = T(0), T step = T(1)) {
  return iota<T, simd_size_v<simd<T>>>(base, step);
}

// Conditional helpers.
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> select(simd<bool, N> m, simd<T, N> x,
                                        simd<T, N> y) {
  return m ? x : y;
}

// TODO: where expressions, scalar overloads, ABI types.

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif

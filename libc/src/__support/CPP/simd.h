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
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_SIMD_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_SIMD_H

#if LIBC_HAS_VECTOR_TYPE

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

namespace internal {

template <typename T>
using get_as_integer_type_t = unsigned _BitInt(sizeof(T) * CHAR_BIT);

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

template <typename T> LIBC_INLINE constexpr T poison() {
  return __builtin_nondeterministic_value(T());
}
} // namespace internal

// Type aliases.
template <typename T, size_t N>
using fixed_size_simd = T [[clang::ext_vector_type(N)]];
template <typename T, size_t N = internal::native_vector_size<T>>
using simd = T [[clang::ext_vector_type(N)]];
template <typename T>
using simd_mask = simd<bool, internal::native_vector_size<T>>;

// Type trait helpers.
template <typename T>
struct simd_size : cpp::integral_constant<size_t, __builtin_vectorelements(T)> {
};
template <class T> constexpr size_t simd_size_v = simd_size<T>::value;

template <typename T> struct is_simd : cpp::integral_constant<bool, false> {};
template <typename T, unsigned N>
struct is_simd<simd<T, N>> : cpp::integral_constant<bool, true> {};
template <class T> constexpr bool is_simd_v = is_simd<T>::value;

template <typename T>
struct is_simd_mask : cpp::integral_constant<bool, false> {};
template <unsigned N>
struct is_simd_mask<simd<bool, N>> : cpp::integral_constant<bool, true> {};
template <class T> constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

template <typename T> struct simd_element_type;
template <typename T, size_t N> struct simd_element_type<simd<T, N>> {
  using type = T;
};
template <typename T>
using simd_element_type_t = typename simd_element_type<T>::type;

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

template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> abs(simd<T, N> x) {
  return __builtin_elementwise_abs(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> fma(simd<T, N> x, simd<T, N> y, simd<T, N> z) {
  return __builtin_elementwise_fma(x, y, z);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> ceil(simd<T, N> x) {
  return __builtin_elementwise_ceil(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> floor(simd<T, N> x) {
  return __builtin_elementwise_floor(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> roundeven(simd<T, N> x) {
  return __builtin_elementwise_roundeven(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> round(simd<T, N> x) {
  return __builtin_elementwise_round(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> trunc(simd<T, N> x) {
  return __builtin_elementwise_trunc(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> nearbyint(simd<T, N> x) {
  return __builtin_elementwise_nearbyint(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> rint(simd<T, N> x) {
  return __builtin_elementwise_rint(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> canonicalize(simd<T, N> x) {
  return __builtin_elementwise_canonicalize(x);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> copysign(simd<T, N> x, simd<T, N> y) {
  return __builtin_elementwise_copysign(x, y);
}
template <typename T, size_t N>
LIBC_INLINE constexpr simd<T, N> fmod(simd<T, N> x, simd<T, N> y) {
  return __builtin_elementwise_fmod(x, y);
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
  return load_unaligned<T>(__builtin_assume_aligned(ptr, alignof(T)));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> store_unaligned(T v, void *ptr) {
  __builtin_memcpy(ptr, &v, sizeof(T));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> store_aligned(T v, void *ptr) {
  store_unaligned<T>(v, __builtin_assume_aligned(ptr, alignof(T)));
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T>
masked_load(simd<bool, simd_size_v<T>> m, void *ptr,
            T passthru = internal::poison<simd_element_type<T>>()) {
  return __builtin_masked_load(m, ptr, passthru);
}
template <typename T>
LIBC_INLINE enable_if_simd_t<T> masked_store(simd<bool, simd_size_v<T>> m, T v,
                                             void *ptr) {
  __builtin_masked_store(
      m, v, static_cast<T *>(__builtin_assume_aligned(ptr, alignof(T))));
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

#endif // LIBC_HAS_VECTOR_TYPE
#endif

//===----- detail.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_DETAILS_H_
#define _HLSL_HLSL_DETAILS_H_

namespace hlsl {

namespace __detail {

template <typename T, typename U> struct is_same {
  static const bool value = false;
};

template <typename T> struct is_same<T, T> {
  static const bool value = true;
};

template <bool B, typename T> struct enable_if {};

template <typename T> struct enable_if<true, T> {
  using Type = T;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::Type;

template <typename U, typename T, int N>
constexpr enable_if_t<sizeof(U) == sizeof(T), vector<U, N>>
bit_cast(vector<T, N> V) {
  return __builtin_bit_cast(vector<U, N>, V);
}

template <typename U, typename T>
constexpr enable_if_t<sizeof(U) == sizeof(T), U> bit_cast(T F) {
  return __builtin_bit_cast(U, F);
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_impl(T X) {
  return __builtin_elementwise_abs(X);
}

template <typename T, int N>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_vec_impl(vector<T, N> X) {
  return __builtin_elementwise_sqrt(__builtin_hlsl_dot(X, X));
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
distance_impl(T X, T Y) {
  return length_impl(X - Y);
}

template <typename T, int N>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
distance_vec_impl(vector<T, N> X, vector<T, N> Y) {
#if (__has_builtin(__builtin_spirv_distance))
  return __builtin_spirv_distance(X, Y);
#else
  return length_vec_impl(X - Y);
#endif
}
} // namespace __detail
} // namespace hlsl
#endif //_HLSL_HLSL_DETAILS_H_

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

constexpr vector<uint, 4> d3d_color_to_ubyte4_impl(vector<float, 4> V) {
  // Use the same scaling factor used by FXC, and DXC for DXIL
  // (i.e., 255.001953)
  // https://github.com/microsoft/DirectXShaderCompiler/blob/070d0d5a2beacef9eeb51037a9b04665716fd6f3/lib/HLSL/HLOperationLower.cpp#L666C1-L697C2
  // The DXC implementation refers to a comment on the following stackoverflow
  // discussion to justify the scaling factor: "Built-in rounding, necessary
  // because of truncation. 0.001953 * 256 = 0.5"
  // https://stackoverflow.com/questions/52103720/why-does-d3dcolortoubyte4-multiplies-components-by-255-001953f
  return V.zyxw * 255.001953f;
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_impl(T X) {
  return __builtin_elementwise_abs(X);
}

template <typename T, int N>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_vec_impl(vector<T, N> X) {
#if (__has_builtin(__builtin_spirv_length))
  return __builtin_spirv_length(X);
#else
  return __builtin_elementwise_sqrt(__builtin_hlsl_dot(X, X));
#endif
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
distance_impl(T X, T Y) {
  return length_impl(X - Y);
}

template <typename T, int N>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
distance_vec_impl(vector<T, N> X, vector<T, N> Y) {
  return length_vec_impl(X - Y);
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
reflect_impl(T I, T N) {
  return I - 2 * N * I * N;
}

template <typename T, int L>
constexpr vector<T, L> reflect_vec_impl(vector<T, L> I, vector<T, L> N) {
#if (__has_builtin(__builtin_spirv_reflect))
  return __builtin_spirv_reflect(I, N);
#else
  return I - 2 * N * __builtin_hlsl_dot(I, N);
#endif
}

} // namespace __detail
} // namespace hlsl
#endif //_HLSL_HLSL_DETAILS_H_

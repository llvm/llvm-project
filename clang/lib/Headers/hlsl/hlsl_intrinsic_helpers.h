//===----- hlsl_intrinsic_helpers.h - HLSL helpers intrinsics -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSIC_HELPERS_H_
#define _HLSL_HLSL_INTRINSIC_HELPERS_H_

namespace hlsl {
namespace __detail {

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_impl(T X) {
  return abs(X);
}

template <typename T, int N>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
length_impl(vector<T, N> X) {
#if (__has_builtin(__builtin_spirv_length))
  return __builtin_spirv_length(X);
#else
  return sqrt(dot(X, X));
#endif
}

constexpr float dot2add_impl(half2 a, half2 b, float c) {
#if (__has_builtin(__builtin_dx_dot2add))
  return __builtin_dx_dot2add(a, b, c);
#else
  return dot(a, b) + c;
#endif
}

template <typename T, int N>
constexpr enable_if_t<!is_same<double, T>::value, T>
mul_vec_impl(vector<T, N> x, vector<T, N> y) {
  return dot(x, y);
}

// Double vectors do not have a dot intrinsic, so expand manually.
template <typename T, int N>
enable_if_t<is_same<double, T>::value, T> mul_vec_impl(vector<T, N> x,
                                                       vector<T, N> y) {
  T sum = x[0] * y[0];
  [unroll] for (int i = 1; i < N; ++i) sum = mad(x[i], y[i], sum);
  return sum;
}

template <typename T>
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value, T>
reflect_impl(T I, T N) {
  return I - 2 * N * I * N;
}

template <typename T, int L>
constexpr vector<T, L> reflect_impl(vector<T, L> I, vector<T, L> N) {
#if (__has_builtin(__builtin_spirv_reflect))
  return __builtin_spirv_reflect(I, N);
#else
  return I - 2 * N * dot(I, N);
#endif
}

template <typename T, typename U> constexpr T refract_impl(T I, T N, U Eta) {
#if (__has_builtin(__builtin_spirv_refract))
  return __builtin_spirv_refract(I, N, Eta);
#endif
  T Mul = dot(N, I);
  T K = 1 - Eta * Eta * (1 - Mul * Mul);
  T Result = (Eta * I - (Eta * Mul + sqrt(K)) * N);
  return select<T>(K < 0, static_cast<T>(0), Result);
}

template <typename T> constexpr T fmod_impl(T X, T Y) {
#if !defined(__DIRECTX__)
  return __builtin_elementwise_fmod(X, Y);
#else
  T div = X / Y;
  bool ge = div >= 0;
  T frc = frac(abs(div));
  return select<T>(ge, frc, -frc) * Y;
#endif
}

template <typename T, int N>
constexpr vector<T, N> fmod_vec_impl(vector<T, N> X, vector<T, N> Y) {
#if !defined(__DIRECTX__)
  return __builtin_elementwise_fmod(X, Y);
#else
  vector<T, N> div = X / Y;
  vector<bool, N> ge = div >= 0;
  vector<T, N> frc = frac(abs(div));
  return select<T>(ge, frc, -frc) * Y;
#endif
}

template <typename T> constexpr T smoothstep_impl(T Min, T Max, T X) {
#if (__has_builtin(__builtin_spirv_smoothstep))
  return __builtin_spirv_smoothstep(Min, Max, X);
#else
  T S = saturate((X - Min) / (Max - Min));
  return (3 - 2 * S) * S * S;
#endif
}

template <typename T> constexpr vector<T, 4> lit_impl(T NDotL, T NDotH, T M) {
  bool DiffuseCond = NDotL < 0;
  T Diffuse = select<T>(DiffuseCond, 0, NDotL);
  vector<T, 4> Result = {1, Diffuse, 0, 1};
  // clang-format off
  bool SpecularCond = or(DiffuseCond, (NDotH < 0));
  // clang-format on
  T SpecularExp = exp(log(NDotH) * M);
  Result[2] = select<T>(SpecularCond, 0, SpecularExp);
  return Result;
}

template <typename T> constexpr T faceforward_impl(T N, T I, T Ng) {
  return select<T>(dot(I, Ng) < 0, N, -N);
}

template <typename K, typename T, int BitWidth>
constexpr K firstbithigh_impl(T X) {
  K FBH = __builtin_hlsl_elementwise_firstbithigh(X);
#if defined(__DIRECTX__)
  // The firstbithigh DXIL ops count bits from the wrong side, so we need to
  // invert it for DirectX.
  K Inversion = (BitWidth - 1) - FBH;
  FBH = select(FBH == -1, FBH, Inversion);
#endif
  return FBH;
}

template <typename T> constexpr T ddx_impl(T input) {
#if (__has_builtin(__builtin_spirv_ddx))
  return __builtin_spirv_ddx(input);
#else
  return __builtin_hlsl_elementwise_ddx_coarse(input);
#endif
}

template <typename T> constexpr T ddy_impl(T input) {
#if (__has_builtin(__builtin_spirv_ddy))
  return __builtin_spirv_ddy(input);
#else
  return __builtin_hlsl_elementwise_ddy_coarse(input);
#endif
}

template <typename T> constexpr T fwidth_impl(T input) {
#if (__has_builtin(__builtin_spirv_fwidth))
  return __builtin_spirv_fwidth(input);
#else
  T derivCoarseX = ddx_coarse(input);
  derivCoarseX = abs(derivCoarseX);
  T derivCoarseY = ddy_coarse(input);
  derivCoarseY = abs(derivCoarseY);
  return derivCoarseX + derivCoarseY;
#endif
}

} // namespace __detail
} // namespace hlsl

#endif // _HLSL_HLSL_INTRINSIC_HELPERS_H_

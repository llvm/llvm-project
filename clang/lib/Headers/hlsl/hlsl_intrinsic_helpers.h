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

template <typename T, int N> constexpr bool all_impl(vector<T, N> V) {
  return __builtin_reduce_and(
      __builtin_convertvector(V != static_cast<T>(0), vector<bool, N>));
}

template <typename T, int N> constexpr bool any_impl(vector<T, N> V) {
  return __builtin_reduce_or(
      __builtin_convertvector(V != static_cast<T>(0), vector<bool, N>));
}

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
constexpr enable_if_t<is_same<float, T>::value || is_same<half, T>::value,
                      vector<T, 3>>
cross_impl(vector<T, 3> x, vector<T, 3> y) {
  return vector<T, 3>(x[1] * y[2] - y[1] * x[2], x[2] * y[0] - y[2] * x[0],
                      x[0] * y[1] - y[0] * x[1]);
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
#if defined(__DIRECTX__)
  T Div = X / Y;
  T Frc = frac(abs(Div));
  return select(Div >= 0, Frc, (T)-Frc) * Y;
#else
  return __builtin_elementwise_fmod(X, Y);
#endif
}

// Returns the mantissa of X, which has the sign of X and an absolute value in
// [0.5, 1), and writes the exponent to Exp such that X = Mant * 2^Exp. X can be
// a scalar, vector or matrix of float.
template <typename T> constexpr T frexp_impl(T X, out T Exp) {
  auto Bits = bit_cast<uint>(X);
  auto ExpBits = Bits & 0x7f800000u;
  // Zeros, denormals, infinities and NaNs are returned as they are, with an
  // exponent of 0. These are exactly the cases where the exponent field is 0 or
  // all ones, so ExpBits - 1 is at least 0x7f7fffff, whether it wraps around or
  // not. This avoids ||, which is not supported on vectors.
  auto IsSpecial = (ExpBits - 1u) >= 0x7f7fffffu;
  // Keep the sign and the fraction, and replace the exponent with the one for
  // 0.5, which scales X into [0.5, 1).
  auto MantBits = (Bits & 0x807fffffu) | 0x3f000000u;
  T Mant = select(IsSpecial, X, bit_cast<float>(MantBits));
  auto RawExp = (ExpBits >> 23u) - 126u;
  Exp = select(IsSpecial, static_cast<T>(0),
               static_cast<T>(bit_cast<int>(RawExp)));
  return Mant;
}

// Matrices do not support the bitwise operators, so frexp_impl works on the
// elements of the matrix as a vector.
template <int R, int C>
constexpr matrix<float, R, C> frexp_impl(matrix<float, R, C> X,
                                         out matrix<float, R, C> Exp) {
  vector<float, R * C> VecExp;
  vector<float, R * C> VecMant =
      frexp_impl(__builtin_bit_cast(vector<float, R * C>, X), VecExp);
  Exp = __builtin_bit_cast(matrix<float, R, C>, VecExp);
  return __builtin_bit_cast(matrix<float, R, C>, VecMant);
}

template <typename T> constexpr T smoothstep_impl(T Min, T Max, T X) {
#if (__has_builtin(__builtin_spirv_smoothstep))
  return __builtin_spirv_smoothstep(Min, Max, X);
#else
  T S = saturate((X - Min) / (Max - Min));
  return (3 - 2 * S) * S * S;
#endif
}

template <typename T> constexpr T step_impl(T Y, T X) {
  return select(X < Y, (T)(__detail::elem_type_t<T>)0,
                (T)(__detail::elem_type_t<T>)1);
}

template <typename T> constexpr T lerp_impl(T X, T Y, T S) {
  return X + S * (Y - X);
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

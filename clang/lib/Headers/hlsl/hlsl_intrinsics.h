//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_INTRINSICS_H_
#define _HLSL_HLSL_INTRINSICS_H_

#include "hlsl/hlsl_intrinsic_helpers.h"

namespace hlsl {

//===----------------------------------------------------------------------===//
// asfloat builtins
//===----------------------------------------------------------------------===//

/// \fn float asfloat(T Val)
/// \brief Interprets the bit pattern of x as float point number.
/// \param Val The input value.

template <typename T, int N>
constexpr vector<float, N> asfloat(vector<T, N> V) {
  return __detail::bit_cast<float, T, N>(V);
}

template <typename T> constexpr float asfloat(T F) {
  return __detail::bit_cast<float, T>(F);
}

//===----------------------------------------------------------------------===//
// asint builtins
//===----------------------------------------------------------------------===//

/// \fn int asint(T Val)
/// \brief Interprets the bit pattern of x as an integer.
/// \param Val The input value.

template <typename T, int N> constexpr vector<int, N> asint(vector<T, N> V) {
  return __detail::bit_cast<int, T, N>(V);
}

template <typename T> constexpr int asint(T F) {
  return __detail::bit_cast<int, T>(F);
}

//===----------------------------------------------------------------------===//
// asint16 builtins
//===----------------------------------------------------------------------===//

/// \fn int16_t asint16(T X)
/// \brief Interprets the bit pattern of \a X as an 16-bit integer.
/// \param X The input value.

#ifdef __HLSL_ENABLE_16_BIT

template <typename T, int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
constexpr __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                    __detail::is_same<uint16_t, T>::value ||
                                    __detail::is_same<half, T>::value,
                                vector<int16_t, N>> asint16(vector<T, N> V) {
  return __detail::bit_cast<int16_t, T, N>(V);
}

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
constexpr __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                    __detail::is_same<uint16_t, T>::value ||
                                    __detail::is_same<half, T>::value,
                                int16_t> asint16(T F) {
  return __detail::bit_cast<int16_t, T>(F);
}
#endif

//===----------------------------------------------------------------------===//
// asuint builtins
//===----------------------------------------------------------------------===//

/// \fn uint asuint(T Val)
/// \brief Interprets the bit pattern of x as an unsigned integer.
/// \param Val The input value.

template <typename T, int N> constexpr vector<uint, N> asuint(vector<T, N> V) {
  return __detail::bit_cast<uint, T, N>(V);
}

template <typename T> constexpr uint asuint(T F) {
  return __detail::bit_cast<uint, T>(F);
}

//===----------------------------------------------------------------------===//
// asuint splitdouble builtins
//===----------------------------------------------------------------------===//

/// \fn void asuint(double D, out uint lowbits, out int highbits)
/// \brief Split and interprets the lowbits and highbits of double D into uints.
/// \param D The input double.
/// \param lowbits The output lowbits of D.
/// \param highbits The output highbits of D.
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double, out uint, out uint);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double2, out uint2, out uint2);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double3, out uint3, out uint3);
_HLSL_BUILTIN_ALIAS(__builtin_hlsl_elementwise_splitdouble)
void asuint(double4, out uint4, out uint4);

//===----------------------------------------------------------------------===//
// asuint16 builtins
//===----------------------------------------------------------------------===//

/// \fn uint16_t asuint16(T X)
/// \brief Interprets the bit pattern of \a X as an 16-bit unsigned integer.
/// \param X The input value.

#ifdef __HLSL_ENABLE_16_BIT

template <typename T, int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
constexpr __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                    __detail::is_same<uint16_t, T>::value ||
                                    __detail::is_same<half, T>::value,
                                vector<uint16_t, N>> asuint16(vector<T, N> V) {
  return __detail::bit_cast<uint16_t, T, N>(V);
}

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
constexpr __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                    __detail::is_same<uint16_t, T>::value ||
                                    __detail::is_same<half, T>::value,
                                uint16_t> asuint16(T F) {
  return __detail::bit_cast<uint16_t, T>(F);
}
#endif

//===----------------------------------------------------------------------===//
// distance builtins
//===----------------------------------------------------------------------===//

/// \fn K distance(T X, T Y)
/// \brief Returns a distance scalar between \a X and \a Y.
/// \param X The X input value.
/// \param Y The Y input value.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> distance(T X, T Y) {
  return __detail::distance_impl(X, Y);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
distance(T X, T Y) {
  return __detail::distance_impl(X, Y);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half distance(__detail::HLSL_FIXED_VECTOR<half, N> X,
                           __detail::HLSL_FIXED_VECTOR<half, N> Y) {
  return __detail::distance_vec_impl(X, Y);
}

template <int N>
const inline float distance(__detail::HLSL_FIXED_VECTOR<float, N> X,
                            __detail::HLSL_FIXED_VECTOR<float, N> Y) {
  return __detail::distance_vec_impl(X, Y);
}

//===----------------------------------------------------------------------===//
// dot2add builtins
//===----------------------------------------------------------------------===//

/// \fn float dot2add(half2 A, half2 B, float C)
/// \brief Dot product of 2 vector of type half and add a float scalar value.
/// \param A The first input value to dot product.
/// \param B The second input value to dot product.
/// \param C The input value added to the dot product.

_HLSL_AVAILABILITY(shadermodel, 6.4)
const inline float dot2add(half2 A, half2 B, float C) {
  return __detail::dot2add_impl(A, B, C);
}

//===----------------------------------------------------------------------===//
// dst builtins
//===----------------------------------------------------------------------===//

/// \fn vector<T, 4> dst(vector<T, 4>, vector<T, 4>)
/// \brief Calculates a distance vector.
/// \param Src0 [in] Contains the squared distance
/// \param Src1 [in] Contains the reciprocal distance
///
/// Return the computed distance vector

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half4 dst(half4 Src0, half4 Src1) {
  return __detail::dst_impl(Src0, Src1);
}

const inline float4 dst(float4 Src0, float4 Src1) {
  return __detail::dst_impl(Src0, Src1);
}

const inline double4 dst(double4 Src0, double4 Src1) {
  return __detail::dst_impl(Src0, Src1);
}

//===----------------------------------------------------------------------===//
// faceforward builtin
//===----------------------------------------------------------------------===//

/// \fn T faceforward(T N, T I, T Ng)
/// \brief Flips the surface-normal (if needed) to face in a direction opposite
/// to \a I. Returns the result in terms of \a N.
/// \param N The resulting floating-point surface-normal vector.
/// \param I A floating-point, incident vector that points from the view
/// position to the shading position.
/// \param Ng A floating-point surface-normal vector.
///
/// Return a floating-point, surface normal vector that is facing the view
/// direction.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> faceforward(T N, T I, T Ng) {
  return __detail::faceforward_impl(N, I, Ng);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
faceforward(T N, T I, T Ng) {
  return __detail::faceforward_impl(N, I, Ng);
}

template <int L>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, L> faceforward(
    __detail::HLSL_FIXED_VECTOR<half, L> N,
    __detail::HLSL_FIXED_VECTOR<half, L> I,
    __detail::HLSL_FIXED_VECTOR<half, L> Ng) {
  return __detail::faceforward_impl(N, I, Ng);
}

template <int L>
const inline __detail::HLSL_FIXED_VECTOR<float, L>
faceforward(__detail::HLSL_FIXED_VECTOR<float, L> N,
            __detail::HLSL_FIXED_VECTOR<float, L> I,
            __detail::HLSL_FIXED_VECTOR<float, L> Ng) {
  return __detail::faceforward_impl(N, I, Ng);
}

//===----------------------------------------------------------------------===//
// firstbithigh builtins
//===----------------------------------------------------------------------===//

/// \fn T firstbithigh(T Val)
/// \brief Returns the location of the first set bit starting from the lowest
/// order bit and working upward, per component.
/// \param Val the input value.

#ifdef __HLSL_ENABLE_16_BIT

template <typename T>
_HLSL_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                       __detail::is_same<uint16_t, T>::value,
                                   uint> firstbithigh(T X) {
  return __detail::firstbithigh_impl<uint, T, 16>(X);
}

template <typename T, int N>
_HLSL_AVAILABILITY(shadermodel, 6.2)
const
    inline __detail::enable_if_t<__detail::is_same<int16_t, T>::value ||
                                     __detail::is_same<uint16_t, T>::value,
                                 vector<uint, N>> firstbithigh(vector<T, N> X) {
  return __detail::firstbithigh_impl<vector<uint, N>, vector<T, N>, 16>(X);
}

#endif

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_same<int, T>::value || __detail::is_same<uint, T>::value, uint>
firstbithigh(T X) {
  return __detail::firstbithigh_impl<uint, T, 32>(X);
}

template <typename T, int N>
const inline __detail::enable_if_t<__detail::is_same<int, T>::value ||
                                       __detail::is_same<uint, T>::value,
                                   vector<uint, N>>
firstbithigh(vector<T, N> X) {
  return __detail::firstbithigh_impl<vector<uint, N>, vector<T, N>, 32>(X);
}

template <typename T>
const inline __detail::enable_if_t<__detail::is_same<int64_t, T>::value ||
                                       __detail::is_same<uint64_t, T>::value,
                                   uint>
firstbithigh(T X) {
  return __detail::firstbithigh_impl<uint, T, 64>(X);
}

template <typename T, int N>
const inline __detail::enable_if_t<__detail::is_same<int64_t, T>::value ||
                                       __detail::is_same<uint64_t, T>::value,
                                   vector<uint, N>>
firstbithigh(vector<T, N> X) {
  return __detail::firstbithigh_impl<vector<uint, N>, vector<T, N>, 64>(X);
}

//===----------------------------------------------------------------------===//
// fmod builtins
//===----------------------------------------------------------------------===//

/// \fn T fmod(T x, T y)
/// \brief Returns the linear interpolation of x to y.
/// \param x [in] The dividend.
/// \param y [in] The divisor.
///
/// Return the floating-point remainder of the x parameter divided by the y
/// parameter.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> fmod(T X, T Y) {
  return __detail::fmod_impl(X, Y);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
fmod(T X, T Y) {
  return __detail::fmod_impl(X, Y);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, N> fmod(
    __detail::HLSL_FIXED_VECTOR<half, N> X,
    __detail::HLSL_FIXED_VECTOR<half, N> Y) {
  return __detail::fmod_vec_impl(X, Y);
}

template <int N>
const inline __detail::HLSL_FIXED_VECTOR<float, N>
fmod(__detail::HLSL_FIXED_VECTOR<float, N> X,
     __detail::HLSL_FIXED_VECTOR<float, N> Y) {
  return __detail::fmod_vec_impl(X, Y);
}

//===----------------------------------------------------------------------===//
// ldexp builtins
//===----------------------------------------------------------------------===//

/// \fn T ldexp(T X, T Exp)
/// \brief Returns the result of multiplying the specified value by two raised
/// to the power of the specified exponent.
/// \param X [in] The specified value.
/// \param Exp [in] The specified exponent.
///
/// This function uses the following formula: X * 2^Exp

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> ldexp(T X, T Exp) {
  return __detail::ldexp_impl(X, Exp);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
ldexp(T X, T Exp) {
  return __detail::ldexp_impl(X, Exp);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, N> ldexp(
    __detail::HLSL_FIXED_VECTOR<half, N> X,
    __detail::HLSL_FIXED_VECTOR<half, N> Exp) {
  return __detail::ldexp_impl(X, Exp);
}

template <int N>
const inline __detail::HLSL_FIXED_VECTOR<float, N>
ldexp(__detail::HLSL_FIXED_VECTOR<float, N> X,
      __detail::HLSL_FIXED_VECTOR<float, N> Exp) {
  return __detail::ldexp_impl(X, Exp);
}

//===----------------------------------------------------------------------===//
// length builtins
//===----------------------------------------------------------------------===//

/// \fn T length(T x)
/// \brief Returns the length of the specified floating-point vector.
/// \param x [in] The vector of floats, or a scalar float.
///
/// Length is based on the following formula: sqrt(x[0]^2 + x[1]^2 + ...).

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> length(T X) {
  return __detail::length_impl(X);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
length(T X) {
  return __detail::length_impl(X);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half length(__detail::HLSL_FIXED_VECTOR<half, N> X) {
  return __detail::length_vec_impl(X);
}

template <int N>
const inline float length(__detail::HLSL_FIXED_VECTOR<float, N> X) {
  return __detail::length_vec_impl(X);
}

//===----------------------------------------------------------------------===//
// lit builtins
//===----------------------------------------------------------------------===//

/// \fn vector<T, 4> lit(T NDotL, T NDotH, T M)
/// \brief Returns a lighting coefficient vector.
/// \param NDotL The dot product of the normalized surface normal and the
/// light vector.
/// \param NDotH The dot product of the half-angle vector and the surface
/// normal.
/// \param M A specular exponent.
///
/// This function returns a lighting coefficient vector (ambient, diffuse,
/// specular, 1).

_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline half4 lit(half NDotL, half NDotH, half M) {
  return __detail::lit_impl(NDotL, NDotH, M);
}

const inline float4 lit(float NDotL, float NDotH, float M) {
  return __detail::lit_impl(NDotL, NDotH, M);
}

//===----------------------------------------------------------------------===//
// D3DCOLORtoUBYTE4 builtin
//===----------------------------------------------------------------------===//

/// \fn T D3DCOLORtoUBYTE4(T x)
/// \brief Converts a floating-point, 4D vector set by a D3DCOLOR to a UBYTE4.
/// \param x [in] The floating-point vector4 to convert.
///
/// The return value is the UBYTE4 representation of the \a x parameter.
///
/// This function swizzles and scales components of the \a x parameter. Use this
/// function to compensate for the lack of UBYTE4 support in some hardware.

constexpr int4 D3DCOLORtoUBYTE4(float4 V) {
  return __detail::d3d_color_to_ubyte4_impl(V);
}

//===----------------------------------------------------------------------===//
// NonUniformResourceIndex builtin
//===----------------------------------------------------------------------===//

/// \fn uint NonUniformResourceIndex(uint I)
/// \brief A compiler hint to indicate that a resource index varies across
/// threads within a wave (i.e., it is non-uniform).
/// \param I [in] Resource array index
///
/// The return value is the \Index parameter.
///
/// When indexing into an array of shader resources (e.g., textures, buffers),
/// some GPU hardware and drivers require the compiler to know whether the index
/// is uniform (same for all threads) or non-uniform (varies per thread).
///
/// Using NonUniformResourceIndex explicitly marks an index as non-uniform,
/// disabling certain assumptions or optimizations that could lead to incorrect
/// behavior when dynamically accessing resource arrays with non-uniform
/// indices.

constexpr uint32_t NonUniformResourceIndex(uint32_t Index) {
  return __builtin_hlsl_resource_nonuniformindex(Index);
}

//===----------------------------------------------------------------------===//
// reflect builtin
//===----------------------------------------------------------------------===//

/// \fn T reflect(T I, T N)
/// \brief Returns a reflection using an incident ray, \a I, and a surface
/// normal, \a N.
/// \param I The incident ray.
/// \param N The surface normal.
///
/// The return value is a floating-point vector that represents the reflection
/// of the incident ray, \a I, off a surface with the normal \a N.
///
/// This function calculates the reflection vector using the following formula:
/// V = I - 2 * N * dot(I N) .
///
/// N must already be normalized in order to achieve the desired result.
///
/// The operands must all be a scalar or vector whose component type is
/// floating-point.
///
/// Result type and the type of all operands must be the same type.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> reflect(T I, T N) {
  return __detail::reflect_impl(I, N);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
reflect(T I, T N) {
  return __detail::reflect_impl(I, N);
}

template <int L>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, L> reflect(
    __detail::HLSL_FIXED_VECTOR<half, L> I,
    __detail::HLSL_FIXED_VECTOR<half, L> N) {
  return __detail::reflect_vec_impl(I, N);
}

template <int L>
const inline __detail::HLSL_FIXED_VECTOR<float, L>
reflect(__detail::HLSL_FIXED_VECTOR<float, L> I,
        __detail::HLSL_FIXED_VECTOR<float, L> N) {
  return __detail::reflect_vec_impl(I, N);
}

//===----------------------------------------------------------------------===//
// refract builtin
//===----------------------------------------------------------------------===//

/// \fn T refract(T I, T N, T eta)
/// \brief Returns a refraction using an entering ray, \a I, a surface
/// normal, \a N and refraction index \a eta
/// \param I The entering ray.
/// \param N The surface normal.
/// \param eta The refraction index.
///
/// The return value is a floating-point vector that represents the refraction
/// using the refraction index, \a eta, for the direction of the entering ray,
/// \a I, off a surface with the normal \a N.
///
/// This function calculates the refraction vector using the following formulas:
/// k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I))
/// if k < 0.0 the result is 0.0
/// otherwise, the result is eta * I - (eta * dot(N, I) + sqrt(k)) * N
///
/// I and N must already be normalized in order to achieve the desired result.
///
/// I and N must be a scalar or vector whose component type is
/// floating-point.
///
/// eta must be a 16-bit or 32-bit floating-point scalar.
///
/// Result type, the type of I, and the type of N must all be the same type.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> refract(T I, T N, T eta) {
  return __detail::refract_impl(I, N, eta);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
refract(T I, T N, T eta) {
  return __detail::refract_impl(I, N, eta);
}

template <int L>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, L> refract(
    __detail::HLSL_FIXED_VECTOR<half, L> I,
    __detail::HLSL_FIXED_VECTOR<half, L> N, half eta) {
  return __detail::refract_impl(I, N, eta);
}

template <int L>
const inline __detail::HLSL_FIXED_VECTOR<float, L>
refract(__detail::HLSL_FIXED_VECTOR<float, L> I,
        __detail::HLSL_FIXED_VECTOR<float, L> N, float eta) {
  return __detail::refract_impl(I, N, eta);
}

//===----------------------------------------------------------------------===//
// smoothstep builtin
//===----------------------------------------------------------------------===//

/// \fn T smoothstep(T Min, T Max, T X)
/// \brief Returns a smooth Hermite interpolation between 0 and 1, if \a X is in
/// the range [\a Min, \a Max].
/// \param Min The minimum range of the x parameter.
/// \param Max The maximum range of the x parameter.
/// \param X The specified value to be interpolated.
///
/// The return value is 0.0 if \a X ≤ \a Min and 1.0 if \a X ≥ \a Max. When \a
/// Min < \a X < \a Max, the function performs smooth Hermite interpolation
/// between 0 and 1.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> smoothstep(T Min, T Max, T X) {
  return __detail::smoothstep_impl(Min, Max, X);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
smoothstep(T Min, T Max, T X) {
  return __detail::smoothstep_impl(Min, Max, X);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, N> smoothstep(
    __detail::HLSL_FIXED_VECTOR<half, N> Min,
    __detail::HLSL_FIXED_VECTOR<half, N> Max,
    __detail::HLSL_FIXED_VECTOR<half, N> X) {
  return __detail::smoothstep_vec_impl(Min, Max, X);
}

template <int N>
const inline __detail::HLSL_FIXED_VECTOR<float, N>
smoothstep(__detail::HLSL_FIXED_VECTOR<float, N> Min,
           __detail::HLSL_FIXED_VECTOR<float, N> Max,
           __detail::HLSL_FIXED_VECTOR<float, N> X) {
  return __detail::smoothstep_vec_impl(Min, Max, X);
}

//===----------------------------------------------------------------------===//
// fwidth builtin
//===----------------------------------------------------------------------===//

/// \fn T fwidth(T x)
/// \brief Computes the sum of the absolute values of the partial derivatives
/// with regard to the x and y screen space coordinates.
/// \param x [in] The floating-point scalar or vector to process.
///
/// The return value is a floating-point scalar or vector where each element
/// holds the computation of the matching element in the input.

template <typename T>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::enable_if_t<__detail::is_arithmetic<T>::Value &&
                                       __detail::is_same<half, T>::value,
                                   T> fwidth(T input) {
  return __detail::fwidth_impl(input);
}

template <typename T>
const inline __detail::enable_if_t<
    __detail::is_arithmetic<T>::Value && __detail::is_same<float, T>::value, T>
fwidth(T input) {
  return __detail::fwidth_impl(input);
}

template <int N>
_HLSL_16BIT_AVAILABILITY(shadermodel, 6.2)
const inline __detail::HLSL_FIXED_VECTOR<half, N> fwidth(
    __detail::HLSL_FIXED_VECTOR<half, N> input) {
  return __detail::fwidth_impl(input);
}

template <int N>
const inline __detail::HLSL_FIXED_VECTOR<float, N>
fwidth(__detail::HLSL_FIXED_VECTOR<float, N> input) {
  return __detail::fwidth_impl(input);
}

} // namespace hlsl
#endif //_HLSL_HLSL_INTRINSICS_H_

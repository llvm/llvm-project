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

constexpr vector<uint, 4> d3d_color_to_ubyte4(vector<float, 4> V) {
  // Use the same scaling factor used by FXC (i.e., 255.001953)
  // Excerpt from stackoverflow discussion:
  // "Built-in rounding, necessary because of truncation. 0.001953 * 256 = 0.5"
  // https://stackoverflow.com/questions/52103720/why-does-d3dcolortoubyte4-multiplies-components-by-255-001953f
  return V.zyxw * 255.001953f;
}

} // namespace __detail
} // namespace hlsl
#endif //_HLSL_HLSL_DETAILS_H_

//===--- hlsl_compat_overloads.h - Extra HLSL overloads for intrinsics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_COMPAT_OVERLOADS_H_
#define _HLSl_COMPAT_OVERLOADS_H_

namespace hlsl {

// Note: Functions in this file are sorted alphabetically, then grouped by base
// element type, and the element types are sorted by size, then singed integer,
// unsigned integer and floating point. Keeping this ordering consistent will
// help keep this file manageable as it grows.

//===----------------------------------------------------------------------===//
// clamp builtins overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, vector<T, N> p1, T p2) {
  return clamp(p0, p1, (vector<T, N>)p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, vector<T, N> p2) {
  return clamp(p0, (vector<T, N>)p1, p2);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, T p1, T p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}

//===----------------------------------------------------------------------===//
// max builtin overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(vector<T, N> p0, T p1) {
  return max(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
max(T p0, vector<T, N> p1) {
  return max((vector<T, N>)p0, p1);
}

//===----------------------------------------------------------------------===//
// min builtin overloads
//===----------------------------------------------------------------------===//

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(vector<T, N> p0, T p1) {
  return min(p0, (vector<T, N>)p1);
}

template <typename T, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
min(T p0, vector<T, N> p1) {
  return min((vector<T, N>)p0, p1);
}

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_

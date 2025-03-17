//===--- hlsl_compat_overloads.h - Extra HLSL overloads for intrinsics --===//
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

template <typename T, typename R, typename U, uint N>
constexpr __detail::enable_if_t<
    __detail::is_arithmetic<U>::Value && (N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, vector<R, N> p1, U p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}
template <typename T, typename R, typename U, uint N>
constexpr __detail::enable_if_t<
    __detail::is_arithmetic<U>::Value && (N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, U p1, vector<R, N> p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}
template <typename T, typename U, typename V, uint N>
constexpr __detail::enable_if_t<__detail::is_arithmetic<U>::Value &&
                                    __detail::is_arithmetic<V>::Value &&
                                    (N > 1 && N <= 4),
                                vector<T, N>>
clamp(vector<T, N> p0, U p1, V p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}
template <typename T, typename R, typename S, uint N>
constexpr __detail::enable_if_t<(N > 1 && N <= 4), vector<T, N>>
clamp(vector<T, N> p0, vector<R, N> p1, vector<S, N> p2) {
  return clamp(p0, (vector<T, N>)p1, (vector<T, N>)p2);
}
template <typename U, typename V, typename W>
constexpr __detail::enable_if_t<__detail::is_arithmetic<U>::Value &&
                                    __detail::is_arithmetic<V>::Value &&
                                    __detail::is_arithmetic<W>::Value,
                                U>
clamp(U p0, V p1, W p2) {
  return clamp(p0, (U)p1, (U)p2);
}

} // namespace hlsl
#endif // _HLSL_COMPAT_OVERLOADS_H_

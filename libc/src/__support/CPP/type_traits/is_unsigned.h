//===-- is_unsigned type_traits ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNSIGNED_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNSIGNED_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/type_traits/bool_constant.h"
#include "src/__support/CPP/type_traits/is_arithmetic.h"
#include "src/__support/CPP/type_traits/is_fixed_point.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

// Primary template: handles arithmetic and signed fixed-point types
template <typename T>
struct is_unsigned : bool_constant<(is_arithmetic_v<T> && (T(-1) > T(0)))> {
  LIBC_INLINE constexpr operator bool() const { return is_unsigned::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_unsigned::value; }
};

#ifdef LIBC_COMPILER_HAS_FIXED_POINT
// Specializations for unsigned fixed-point types
template <typename T, bool IsUnsigned>
struct fixed_point_is_unsigned : bool_constant<IsUnsigned> {
  LIBC_INLINE constexpr operator bool() const { return fixed_point_is_unsigned::value; }
  LIBC_INLINE constexpr bool operator()() const { return fixed_point_is_unsigned::value; }
};
template <> struct is_unsigned<fract> : fixed_point_is_unsigned<fract, false> {};
template <> struct is_unsigned<unsigned short fract> : fixed_point_is_unsigned<unsigned short fract, true> {};
template <> struct is_unsigned<unsigned fract> : fixed_point_is_unsigned<unsigned fract, true> {};
template <> struct is_unsigned<unsigned long fract> : fixed_point_is_unsigned<unsigned long fract, true> {};
template <> struct is_unsigned<short fract> : fixed_point_is_unsigned<short fract, false> {};
template <> struct is_unsigned<long fract> : fixed_point_is_unsigned<long fract, false> {};
template <> struct is_unsigned<accum> : fixed_point_is_unsigned<accum, false> {};
template <> struct is_unsigned<unsigned short accum> : fixed_point_is_unsigned<unsigned short accum, true> {};
template <> struct is_unsigned<unsigned accum> : fixed_point_is_unsigned<unsigned accum, true> {};
template <> struct is_unsigned<unsigned long accum> : fixed_point_is_unsigned<unsigned long accum, true> {};
template <> struct is_unsigned<short accum> : fixed_point_is_unsigned<short accum, false> {};
template <> struct is_unsigned<long accum> : fixed_point_is_unsigned<long accum, false> {};
template <> struct is_unsigned<sat fract> : fixed_point_is_unsigned<sat fract, false> {};
template <> struct is_unsigned<unsigned short sat fract> : fixed_point_is_unsigned<unsigned short sat fract, true> {};
template <> struct is_unsigned<unsigned sat fract> : fixed_point_is_unsigned<unsigned sat fract, true> {};
template <> struct is_unsigned<unsigned long sat fract> : fixed_point_is_unsigned<unsigned long sat fract, true> {};
template <> struct is_unsigned<short sat fract> : fixed_point_is_unsigned<short sat fract, false> {};
template <> struct is_unsigned<long sat fract> : fixed_point_is_unsigned<long sat fract, false> {};
template <> struct is_unsigned<sat accum> : fixed_point_is_unsigned<sat accum, false> {};
template <> struct is_unsigned<unsigned short sat accum> : fixed_point_is_unsigned<unsigned short sat accum, true> {};
template <> struct is_unsigned<unsigned sat accum> : fixed_point_is_unsigned<unsigned sat accum, true> {};
template <> struct is_unsigned<unsigned long sat accum> : fixed_point_is_unsigned<unsigned long sat accum, true> {};
template <> struct is_unsigned<short sat accum> : fixed_point_is_unsigned<short sat accum, false> {};
template <> struct is_unsigned<long sat accum> : fixed_point_is_unsigned<long sat accum, false> {};
#endif // LIBC_COMPILER_HAS_FIXED_POINT

template <typename T>
LIBC_INLINE_VAR constexpr bool is_unsigned_v = is_unsigned<T>::value;

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_UNSIGNED_H

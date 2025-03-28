//===-- is_signed type_traits -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_SIGNED_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_SIGNED_H

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
struct is_signed : bool_constant<(is_arithmetic_v<T> && (T(-1) < T(0)))> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};

#ifdef LIBC_COMPILER_HAS_FIXED_POINT
// Specializations for unsigned fixed-point types
template <typename T, bool IsSigned>
struct fixed_point_is_signed : bool_constant<IsSigned> {
  LIBC_INLINE constexpr operator bool() const { return fixed_point_is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return fixed_point_is_signed::value; }
};
template <> struct is_signed<fract> : fixed_point_is_signed<fract, true> {};
template <> struct is_signed<unsigned short fract> : fixed_point_is_signed<unsigned short fract, false> {};
template <> struct is_signed<unsigned fract> : fixed_point_is_signed<unsigned fract, false> {};
template <> struct is_signed<unsigned long fract> : fixed_point_is_signed<unsigned long fract, false> {};
template <> struct is_signed<short fract> : fixed_point_is_signed<short fract, true> {};
template <> struct is_signed<long fract> : fixed_point_is_signed<long fract, true> {};
template <> struct is_signed<accum> : fixed_point_is_signed<accum, true> {};
template <> struct is_signed<unsigned short accum> : fixed_point_is_signed<unsigned short accum, false> {};
template <> struct is_signed<unsigned accum> : fixed_point_is_signed<unsigned accum, false> {};
template <> struct is_signed<unsigned long accum> : fixed_point_is_signed<unsigned long accum, false> {};
template <> struct is_signed<short accum> : fixed_point_is_signed<short accum, true> {};
template <> struct is_signed<long accum> : fixed_point_is_signed<long accum, true> {};
template <> struct is_signed<sat fract> : fixed_point_is_signed<sat fract, true> {};
template <> struct is_signed<unsigned short sat fract> : fixed_point_is_signed<unsigned short sat fract, false> {};
template <> struct is_signed<unsigned sat fract> : fixed_point_is_signed<unsigned sat fract, false> {};
template <> struct is_signed<unsigned long sat fract> : fixed_point_is_signed<unsigned long sat fract, false> {};
template <> struct is_signed<short sat fract> : fixed_point_is_signed<short sat fract, true> {};
template <> struct is_signed<long sat fract> : fixed_point_is_signed<long sat fract, true> {};
template <> struct is_signed<sat accum> : fixed_point_is_signed<sat accum, true> {};
template <> struct is_signed<unsigned short sat accum> : fixed_point_is_signed<unsigned short sat accum, false> {};
template <> struct is_signed<unsigned sat accum> : fixed_point_is_signed<unsigned sat accum, false> {};
template <> struct is_signed<unsigned long sat accum> : fixed_point_is_signed<unsigned long sat accum, false> {};
template <> struct is_signed<short sat accum> : fixed_point_is_signed<short sat accum, true> {};
template <> struct is_signed<long sat accum> : fixed_point_is_signed<long sat accum, true> {};
#endif // LIBC_COMPILER_HAS_FIXED_POINT

template <typename T>
LIBC_INLINE_VAR constexpr bool is_signed_v = is_signed<T>::value;

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_SIGNED_H

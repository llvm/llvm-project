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
struct is_signed : bool_constant<((is_fixed_point_v<T> || is_arithmetic_v<T>) && (T(-1) < T(0)))> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};

// Specializations for unsigned fixed-point types
template <>
struct is_signed<unsigned short fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned long fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned short accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned long accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned short sat fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned sat fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned long sat fract> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned short sat accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned sat accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};
template <>
struct is_signed<unsigned long sat accum> : bool_constant<false> {
  LIBC_INLINE constexpr operator bool() const { return is_signed::value; }
  LIBC_INLINE constexpr bool operator()() const { return is_signed::value; }
};

template <typename T>
LIBC_INLINE_VAR constexpr bool is_signed_v = is_signed<T>::value;

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_IS_SIGNED_H

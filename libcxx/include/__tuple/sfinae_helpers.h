//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TUPLE_SFINAE_HELPERS_H
#define _LIBCPP___TUPLE_SFINAE_HELPERS_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_CXX03_LANG

struct __check_tuple_constructor_fail {
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __enable_explicit_default() { return false; }
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __enable_implicit_default() { return false; }
  template <class...>
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __enable_explicit() {
    return false;
  }
  template <class...>
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __enable_implicit() {
    return false;
  }
  template <class...>
  static _LIBCPP_HIDE_FROM_ABI constexpr bool __enable_assign() {
    return false;
  }
};
#endif // !defined(_LIBCPP_CXX03_LANG)

#if _LIBCPP_STD_VER >= 17

template <bool _CanCopy, bool _CanMove>
struct __sfinae_ctor_base {};
template <>
struct __sfinae_ctor_base<false, false> {
  __sfinae_ctor_base()                                     = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&)            = delete;
  __sfinae_ctor_base(__sfinae_ctor_base&&)                 = delete;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&)      = default;
};
template <>
struct __sfinae_ctor_base<true, false> {
  __sfinae_ctor_base()                                     = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&)            = default;
  __sfinae_ctor_base(__sfinae_ctor_base&&)                 = delete;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&)      = default;
};
template <>
struct __sfinae_ctor_base<false, true> {
  __sfinae_ctor_base()                                     = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&)            = delete;
  __sfinae_ctor_base(__sfinae_ctor_base&&)                 = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&)      = default;
};

template <bool _CanCopy, bool _CanMove>
struct __sfinae_assign_base {};
template <>
struct __sfinae_assign_base<false, false> {
  __sfinae_assign_base()                                       = default;
  __sfinae_assign_base(__sfinae_assign_base const&)            = default;
  __sfinae_assign_base(__sfinae_assign_base&&)                 = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = delete;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&)      = delete;
};
template <>
struct __sfinae_assign_base<true, false> {
  __sfinae_assign_base()                                       = default;
  __sfinae_assign_base(__sfinae_assign_base const&)            = default;
  __sfinae_assign_base(__sfinae_assign_base&&)                 = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&)      = delete;
};
template <>
struct __sfinae_assign_base<false, true> {
  __sfinae_assign_base()                                       = default;
  __sfinae_assign_base(__sfinae_assign_base const&)            = default;
  __sfinae_assign_base(__sfinae_assign_base&&)                 = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = delete;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&)      = default;
};
#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TUPLE_SFINAE_HELPERS_H

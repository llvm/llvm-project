// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___FLAT_MAP_CONTAINER_TRAITS_H
#define _LIBCPP___FLAT_MAP_CONTAINER_TRAITS_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __is_stl_container = false;

template <class _Tp>
struct __container_traits {
  static constexpr bool __emplacement_has_strong_exception_safety_guarantee = false;
};

template <class _Tp>
  requires __is_stl_container<_Tp>
struct __container_traits<_Tp> {
  // http://eel.is/c++draft/container.reqmts
  // 66 Unless otherwise specified (see [associative.reqmts.except], [unord.req.except], [deque.modifiers],
  // [inplace.vector.modifiers], and [vector.modifiers]) all container types defined in this Clause meet the following
  // additional requirements:
  // - (66.1) If an exception is thrown by an insert() or emplace() function while inserting a single element, that
  // function has no effects.
  static constexpr bool __emplacement_has_strong_exception_safety_guarantee = true;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___FLAT_MAP_CONTAINER_TRAITS_H

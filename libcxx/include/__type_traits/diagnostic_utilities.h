//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_DIAGNOSTIC_UTILITIES_H
#define _LIBCPP___TYPE_TRAITS_DIAGNOSTIC_UTILITIES_H

#include <__config>
#include <__type_traits/is_array.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_unbounded_array.h>
#include <__type_traits/is_void.h>
#include <__type_traits/is_volatile.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20
#  define _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_IS_NOT_ARRAY_BEFORE_CXX20(_Container, _Tp)
#else
#  define _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_IS_NOT_ARRAY_BEFORE_CXX20(_Container, _Tp)                                \
    ;                                                                                                                  \
    static_assert(!is_array<_Tp>::value, "'std::" _Container "' cannot hold C arrays before C++20")
#endif

// Per https://eel.is/c++draft/containers#container.reqmts-64, allocator-aware containers must have an
// allocator that meets the Cpp17Allocator requirements (https://eel.is/c++draft/allocator.requirements).
// In particular, this means that containers should only accept non-cv-qualified object types, and
// types that are Cpp17Erasable.
#define _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_REQUIREMENTS_BASE(_Container, _Tp)                                          \
  static_assert(!is_const<_Tp>::value, "'std::" _Container "' cannot hold const types");                               \
  static_assert(!is_volatile<_Tp>::value, "'std::" _Container "' cannot hold volatile types");                         \
  static_assert(!is_reference<_Tp>::value, "'std::" _Container "' cannot hold references");                            \
  static_assert(!is_function<_Tp>::value, "'std::" _Container "' cannot hold functions");                              \
  static_assert(                                                                                                       \
      !__libcpp_is_unbounded_array<_Tp>::value, "'std::" _Container "' cannot hold C arrays of an unknown size")       \
      _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_IS_NOT_ARRAY_BEFORE_CXX20(_Container, _Tp)

#define _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_REQUIREMENTS_FULL(_Container, _Tp)                                          \
  _LIBCPP_CHECK_CONTAINER_VALUE_TYPE_REQUIREMENTS_BASE(_Container, _Tp);                                               \
  static_assert(!is_void<_Tp>::value, "'std::" _Container "' cannot hold 'void'")

#endif // _LIBCPP___TYPE_TRAITS_DIAGNOSTIC_UTILITIES_H

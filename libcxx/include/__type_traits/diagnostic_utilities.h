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
#include <__type_traits/decay.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_bounded_array.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_unbounded_array.h>
#include <__type_traits/is_void.h>
#include <__type_traits/is_volatile.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Many templates require their type parameters to be cv-unqualified objects.
template <template <class...> class _Template, class _Tp, bool = is_same<__decay_t<_Tp>, _Tp>::value>
struct __requires_cv_unqualified_object_type : true_type {};

#define _LIBCPP_DEFINE__REQUIRES_CV_UNQUALIFIED_OBJECT_TYPE(_Template, _Verb)                                          \
  template <class _Tp>                                                                                                 \
  struct __requires_cv_unqualified_object_type<_Template, _Tp, false>                                                  \
      : integral_constant<bool,                                                                                        \
                          !(is_const<_Tp>::value || is_volatile<_Tp>::value || is_reference<_Tp>::value ||             \
                            is_function<_Tp>::value)> {                                                                \
    static_assert(!is_const<_Tp>::value, "'std::" #_Template "' cannot " _Verb " const types");                        \
    static_assert(!is_volatile<_Tp>::value, "'std::" #_Template "' cannot " _Verb " volatile types");                  \
    static_assert(!is_reference<_Tp>::value, "'std::" #_Template "' cannot " _Verb " references");                     \
    static_assert(!is_function<_Tp>::value, "'std::" #_Template "' cannot " _Verb " functions");                       \
  }

// Per https://eel.is/c++draft/containers#container.reqmts-64, allocator-aware containers must have an
// allocator that meets the Cpp17Allocator requirements (https://eel.is/c++draft/allocator.requirements).
// In particular, this means that containers should only accept non-cv-qualified object types, and
// types that are Cpp17Erasable.
template <template <class...> class _Template, class _Tp, bool = is_same<__decay_t<_Tp>, _Tp>::value>
struct __allocator_requirements : true_type {};

#if _LIBCPP_STD_VER >= 20
template <class _Tp>
inline const bool __bounded_arrays_allowed_only_after_cxx20 = false;
#else
template <class _Tp>
inline const bool __bounded_arrays_allowed_only_after_cxx20 = __libcpp_is_bounded_array<_Tp>::value;
#endif

#define _LIBCPP_DEFINE__ALLOCATOR_VALUE_TYPE_REQUIREMENTS(_Template, _Verb)                                            \
  _LIBCPP_DEFINE__REQUIRES_CV_UNQUALIFIED_OBJECT_TYPE(_Template, _Verb);                                               \
  template <class _Tp>                                                                                                 \
  struct __allocator_requirements<_Template, _Tp, false>                                                               \
      : integral_constant<bool,                                                                                        \
                          __requires_cv_unqualified_object_type<_Template, _Tp>::value &&                              \
                              !__bounded_arrays_allowed_only_after_cxx20<_Tp> > {                                      \
    static_assert(!__bounded_arrays_allowed_only_after_cxx20<_Tp>,                                                     \
                  "'std::" #_Template "' cannot " _Verb " C arrays before C++20");                                     \
  }

template <template <class...> class, class>
struct __container_requirements : false_type {
  static_assert(
      false,
      "a new container has been defined; please define '_LIBCPP_DEFINE__CONTAINER_VALUE_TYPE_REQUIREMENTS' for "
      "that container");
};

#define _LIBCPP_DEFINE__CONTAINER_VALUE_TYPE_REQUIREMENTS(_Template)                                                   \
  _LIBCPP_DEFINE__ALLOCATOR_VALUE_TYPE_REQUIREMENTS(_Template, "hold");                                                \
  template <class _Tp>                                                                                                 \
  struct __container_requirements<_Template, _Tp>                                                                      \
      : integral_constant<bool,                                                                                        \
                          __allocator_requirements<_Template, _Tp>::value &&                                           \
                              !(is_void<_Tp>::value || __libcpp_is_unbounded_array<_Tp>::value)> {                     \
    static_assert(!is_void<_Tp>::value, "'std::" #_Template "' cannot hold 'void'");                                   \
    static_assert(!__libcpp_is_unbounded_array<_Tp>::value,                                                            \
                  "'std::" #_Template "' cannot hold C arrays of an unknown size");                                    \
  }

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_DIAGNOSTIC_UTILITIES_H

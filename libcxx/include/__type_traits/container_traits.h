// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_CONTAINER_TRAITS_H
#define _LIBCPP___TYPE_TRAITS_CONTAINER_TRAITS_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// __container_traits is a general purpose struct contains traits of containers' different operations.
// It currently only has one trait: `__emplacement_has_strong_exception_safety_guarantee`, but it's
// intended to be extended in the future.
// If a container does not support an operation. For example, `std::array` does not support `insert`
// or `emplace`, the trait of that operation will return false.
template <class _Container>
struct __container_traits {
  // A trait that tells whether a single element insertion/emplacement via member function
  // `insert(...)` or `emplace(...)` has strong exception guarantee, that is, if the function
  // exits via an exception, the original container is unaffected
  static _LIBCPP_CONSTEXPR const bool __emplacement_has_strong_exception_safety_guarantee = false;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_CONTAINER_TRAITS_H

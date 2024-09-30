// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_START_LIFETIME_AS_H
#define _LIBCPP___MEMORY_START_LIFETIME_AS_H

#include <__config>
#include <__type_traits/is_array.h>
#include <__type_traits/remove_all_extents.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS __start_lifetime_as_impl {
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR _Tp* __call(void* __p) _NOEXCEPT { return static_cast<_Tp*>(__p); }
};

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp* start_lifetime_as(void* __p) _NOEXCEPT {
  return __start_lifetime_as_impl<_Tp>::__call(__p);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const _Tp* start_lifetime_as(const void* __p) _NOEXCEPT {
  return std::start_lifetime_as<_Tp>(const_cast<void*>(__p));
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR volatile _Tp* start_lifetime_as(volatile void* __p) _NOEXCEPT {
  return std::start_lifetime_as<_Tp>(const_cast<void*>(__p));
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const volatile _Tp* start_lifetime_as(const volatile void* __p) _NOEXCEPT {
  return std::start_lifetime_as<_Tp>(const_cast<void*>(__p));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_START_LIFETIME_AS_H

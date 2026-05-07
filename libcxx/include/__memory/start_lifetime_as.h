//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_START_LIFETIME_AS_H
#define _LIBCPP___MEMORY_START_LIFETIME_AS_H

#include "__configuration/attributes.h"
#include <__config>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* start_lifetime_as(void* __p) _NOEXCEPT {
  return __builtin_start_lifetime_as(static_cast<_Tp*>(__p));
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI const _Tp* start_lifetime_as(const void* __p) _NOEXCEPT {
  return __builtin_start_lifetime_as(static_cast<const _Tp*>(__p));
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI volatile _Tp* start_lifetime_as(volatile void* __p) _NOEXCEPT {
  return __builtin_start_lifetime_as(static_cast<volatile _Tp*>(__p));
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI const volatile _Tp* start_lifetime_as(const volatile void* __p) _NOEXCEPT {
  return __builtin_start_lifetime_as(static_cast<const volatile _Tp*>(__p));
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* start_lifetime_as_array(void* __p, size_t __n) _NOEXCEPT {
  static_cast<void>(__n);
  return __builtin_start_lifetime_as(static_cast<_Tp*>(__p));
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI const _Tp* start_lifetime_as_array(const void* __p, size_t __n) _NOEXCEPT {
  static_cast<void>(__n);
  return __builtin_start_lifetime_as(static_cast<const _Tp*>(__p));
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI volatile _Tp* start_lifetime_as_array(volatile void* __p, size_t __n) _NOEXCEPT {
  static_cast<void>(__n);
  return __builtin_start_lifetime_as(static_cast<volatile _Tp*>(__p));
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI const volatile _Tp* start_lifetime_as_array(const volatile void* __p, size_t __n) _NOEXCEPT {
  static_cast<void>(__n);
  return __builtin_start_lifetime_as(static_cast<const volatile _Tp*>(__p));
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_START_LIFETIME_AS_H

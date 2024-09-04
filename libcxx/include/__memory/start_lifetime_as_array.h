// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_START_LIFETIME_AS_ARRAY_H
#define _LIBCPP___MEMORY_START_LIFETIME_AS_ARRAY_H

#include <__config>
#include <__memory/start_lifetime_as.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp*
start_lifetime_as_array(void* __p, [[__maybe_unused__]] size_t __n) _NOEXCEPT {
  return static_cast<_Tp*>(__p);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const _Tp* start_lifetime_as_array(const void* __p, size_t __n) _NOEXCEPT {
  return std::start_lifetime_as_array<_Tp>(const_cast<void*>(__p), __n);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR volatile _Tp*
start_lifetime_as_array(volatile void* __p, size_t __n) _NOEXCEPT {
  return std::start_lifetime_as_array<_Tp>(const_cast<void*>(__p), __n);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const volatile _Tp*
start_lifetime_as_array(const volatile void* __p, size_t __n) _NOEXCEPT {
  return std::start_lifetime_as_array<_Tp>(const_cast<void*>(__p), __n);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_START_LIFETIME_AS_ARRAY_H

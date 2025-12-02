//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MEMORY_TEMP_VALUE_H
#define _LIBCPP___CXX03___MEMORY_TEMP_VALUE_H

#include <__cxx03/__config>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__memory/allocator_traits.h>
#include <__cxx03/__type_traits/aligned_storage.h>
#include <__cxx03/__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Alloc>
struct __temp_value {
  typedef allocator_traits<_Alloc> _Traits;

  typename aligned_storage<sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp)>::type __v;
  _Alloc& __a;

  _LIBCPP_HIDE_FROM_ABI _Tp* __addr() { return reinterpret_cast<_Tp*>(std::addressof(__v)); }

  _LIBCPP_HIDE_FROM_ABI _Tp& get() { return *__addr(); }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_CFI __temp_value(_Alloc& __alloc, _Args&&... __args) : __a(__alloc) {
    _Traits::construct(__a, __addr(), std::forward<_Args>(__args)...);
  }

  _LIBCPP_HIDE_FROM_ABI ~__temp_value() { _Traits::destroy(__a, __addr()); }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___MEMORY_TEMP_VALUE_H

//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FWD_ARRAY_H
#define _LIBCPP___CXX03___FWD_ARRAY_H

#include <__cxx03/__config>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, size_t _Size>
struct _LIBCPP_TEMPLATE_VIS array;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCPP_HIDE_FROM_ABI _Tp& get(array<_Tp, _Size>&) _NOEXCEPT;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCPP_HIDE_FROM_ABI const _Tp& get(const array<_Tp, _Size>&) _NOEXCEPT;

template <class>
struct __is_std_array : false_type {};

template <class _Tp, size_t _Size>
struct __is_std_array<array<_Tp, _Size> > : true_type {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FWD_ARRAY_H

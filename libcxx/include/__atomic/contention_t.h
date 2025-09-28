//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_CONTENTION_T_H
#define _LIBCPP___ATOMIC_CONTENTION_T_H

#include <__atomic/support.h>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_standard_layout.h>
#include <cstddef>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class = void>
struct __is_atomic_wait_native_type : false_type {};

#if defined(__linux__) || (defined(_AIX) && !defined(__64BIT__))
using __cxx_contention_t _LIBCPP_NODEBUG = int32_t;

template <class _Tp>
struct __is_atomic_wait_native_type<_Tp, __enable_if_t<is_standard_layout<_Tp>::value && sizeof(_Tp) == 4> > : true_type {};

#else
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;

template <class _Tp>
struct __is_atomic_wait_native_type<_Tp,
                                    __enable_if_t<is_standard_layout<_Tp>::value && (sizeof(_Tp) == 4 || sizeof(_Tp) == 8)> >
    : true_type {};

#endif // __linux__ || (_AIX && !__64BIT__)

using __cxx_atomic_contention_t _LIBCPP_NODEBUG = __cxx_atomic_impl<__cxx_contention_t>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_CONTENTION_T_H

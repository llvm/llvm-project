//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: __builtin_popcountg is available since Clang 19 and GCC 14. When support for older versions is dropped, we can
//  refactor this code to exclusively use __builtin_popcountg.

#ifndef _LIBCPP___CXX03___BIT_POPCOUNT_H
#define _LIBCPP___CXX03___BIT_POPCOUNT_H

#include <__cxx03/__bit/rotate.h>
#include <__cxx03/__config>
#include <__cxx03/limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

inline _LIBCPP_HIDE_FROM_ABI int __libcpp_popcount(unsigned __x) _NOEXCEPT { return __builtin_popcount(__x); }

inline _LIBCPP_HIDE_FROM_ABI int __libcpp_popcount(unsigned long __x) _NOEXCEPT { return __builtin_popcountl(__x); }

inline _LIBCPP_HIDE_FROM_ABI int __libcpp_popcount(unsigned long long __x) _NOEXCEPT {
  return __builtin_popcountll(__x);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___BIT_POPCOUNT_H

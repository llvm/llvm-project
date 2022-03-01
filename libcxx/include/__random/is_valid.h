//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANDOM_IS_VALID_H
#define _LIBCPP___RANDOM_IS_VALID_H

#include <__config>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#  pragma clang include_instead(<random>)
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// [rand.req.genl]/1.5:
// The effect of instantiating a template that has a template type parameter
// named IntType is undefined unless the corresponding template argument is
// cv-unqualified and is one of short, int, long, long long, unsigned short,
// unsigned int, unsigned long, or unsigned long long.

template<class> struct __libcpp_random_is_valid_inttype : false_type {};
template<> struct __libcpp_random_is_valid_inttype<short> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<int> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<long> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<long long> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<unsigned short> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<unsigned int> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<unsigned long> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<unsigned long long> : true_type {};

#ifndef _LIBCPP_HAS_NO_INT128
template<> struct __libcpp_random_is_valid_inttype<__int128_t> : true_type {};
template<> struct __libcpp_random_is_valid_inttype<__uint128_t> : true_type {};
#endif // _LIBCPP_HAS_NO_INT128

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANDOM_IS_VALID_H

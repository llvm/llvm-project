//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TYPE_TRAITS_RESULT_OF_H
#define _LIBCPP___CXX03___TYPE_TRAITS_RESULT_OF_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/invoke.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// result_of

template <class _Callable>
class result_of;

template <class _Fp, class... _Args>
class _LIBCPP_TEMPLATE_VIS result_of<_Fp(_Args...)> : public __invoke_of<_Fp, _Args...> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TYPE_TRAITS_RESULT_OF_H

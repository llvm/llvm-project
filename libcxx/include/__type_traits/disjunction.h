//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_DISJUNCTION_H
#define _LIBCPP___TYPE_TRAITS_DISJUNCTION_H

#include <__config>
#include <__type_traits/conditional.h>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 14

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Arg, class... _Args>
struct __disjunction_impl {
  using type = conditional_t<bool(_Arg::value), _Arg, typename __disjunction_impl<_Args...>::type>;
};

template <class _Arg>
struct __disjunction_impl<_Arg> {
  using type = _Arg;
};

template <class... _Args>
struct disjunction : __disjunction_impl<false_type, _Args...>::type {};
template<class... _Args>
inline constexpr bool disjunction_v = disjunction<_Args...>::value;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 14

#endif // _LIBCPP___TYPE_TRAITS_DISJUNCTION_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_MINMAX_H
#define _LIBCPP___CXX03___ALGORITHM_MINMAX_H

#include <__cxx03/__algorithm/comp.h>
#include <__cxx03/__algorithm/minmax_element.h>
#include <__cxx03/__config>
#include <__cxx03/__functional/identity.h>
#include <__cxx03/__type_traits/is_callable.h>
#include <__cxx03/__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Compare>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI pair<const _Tp&, const _Tp&>
minmax(_LIBCPP_LIFETIMEBOUND const _Tp& __a, _LIBCPP_LIFETIMEBOUND const _Tp& __b, _Compare __comp) {
  return __comp(__b, __a) ? pair<const _Tp&, const _Tp&>(__b, __a) : pair<const _Tp&, const _Tp&>(__a, __b);
}

template <class _Tp>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI pair<const _Tp&, const _Tp&>
minmax(_LIBCPP_LIFETIMEBOUND const _Tp& __a, _LIBCPP_LIFETIMEBOUND const _Tp& __b) {
  return std::minmax(__a, __b, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ALGORITHM_MINMAX_H

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FUNCTIONAL_IDENTITY_H
#define _LIBCPP___CXX03___FUNCTIONAL_IDENTITY_H

#include <__cxx03/__config>
#include <__cxx03/__fwd/functional.h>
#include <__cxx03/__type_traits/integral_constant.h>
#include <__cxx03/__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_identity : false_type {};

struct __identity {
  template <class _Tp>
  _LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _Tp&& operator()(_Tp&& __t) const _NOEXCEPT {
    return std::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

template <>
struct __is_identity<__identity> : true_type {};
template <>
struct __is_identity<reference_wrapper<__identity> > : true_type {};
template <>
struct __is_identity<reference_wrapper<const __identity> > : true_type {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FUNCTIONAL_IDENTITY_H

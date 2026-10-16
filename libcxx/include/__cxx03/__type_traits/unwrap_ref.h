//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___TYPE_TRAITS_UNWRAP_REF_H
#define _LIBCPP___CXX03___TYPE_TRAITS_UNWRAP_REF_H

#include <__cxx03/__config>
#include <__cxx03/__fwd/functional.h>
#include <__cxx03/__type_traits/decay.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __unwrap_reference {
  typedef _LIBCPP_NODEBUG _Tp type;
};

template <class _Tp>
struct __unwrap_reference<reference_wrapper<_Tp> > {
  typedef _LIBCPP_NODEBUG _Tp& type;
};

template <class _Tp>
struct __unwrap_ref_decay : __unwrap_reference<__decay_t<_Tp> > {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___TYPE_TRAITS_UNWRAP_REF_H

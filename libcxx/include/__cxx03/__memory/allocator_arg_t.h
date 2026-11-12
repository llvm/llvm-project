// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FUNCTIONAL_ALLOCATOR_ARG_T_H
#define _LIBCPP___CXX03___FUNCTIONAL_ALLOCATOR_ARG_T_H

#include <__cxx03/__config>
#include <__cxx03/__memory/uses_allocator.h>
#include <__cxx03/__type_traits/integral_constant.h>
#include <__cxx03/__type_traits/is_constructible.h>
#include <__cxx03/__type_traits/remove_cvref.h>
#include <__cxx03/__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct _LIBCPP_TEMPLATE_VIS allocator_arg_t {
  explicit allocator_arg_t() = default;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FUNCTIONAL_ALLOCATOR_ARG_T_H

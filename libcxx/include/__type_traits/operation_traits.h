//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_OPERATION_TRAITS_H
#define _LIBCPP___TYPE_TRAITS_OPERATION_TRAITS_H

#include <__config>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// Tags to represent the canonical operations
struct __equal_tag {};
struct __plus_tag {};

// In the general case, __desugars_to is false

template <class _CanonicalTag, class _Operation, class... _Args>
struct __desugars_to : false_type {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_OPERATION_TRAITS_H

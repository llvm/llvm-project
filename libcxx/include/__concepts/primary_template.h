//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
#define _LIBCPP___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H

#include <__config>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
concept __primary_template = requires {
  typename _Tp::__primary_template;
  requires _IsSame<typename _Tp::__primary_template, _Tp>::value;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H

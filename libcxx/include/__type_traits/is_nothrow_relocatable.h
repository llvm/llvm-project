//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_NOTHROW_RELOCATABLE_H
#define _LIBCPP___TYPE_TRAITS_IS_NOTHROW_RELOCATABLE_H

#include <__config>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_nothrow_destructible.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__type_traits/remove_all_extents.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_nothrow_relocatable
    : integral_constant<bool,
                        __libcpp_is_trivially_relocatable<_Tp>::value ||
                            (is_nothrow_move_constructible<__remove_all_extents_t<_Tp> >::value &&
                             is_nothrow_destructible<__remove_all_extents_t<_Tp> >::value)> {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_NOTHROW_RELOCATABLE_H

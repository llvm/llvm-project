//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_REGISTER_PASSABLE_H
#define _LIBCPP___TYPE_TRAITS_REGISTER_PASSABLE_H

#include <__config>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_trivially_constructible.h>
#include <__type_traits/is_trivially_destructible.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _Arg>
concept __itanium_trivial_for_calls =
    is_trivially_destructible_v<_Arg> && is_trivially_copy_constructible_v<_Arg> &&
    is_trivially_move_constructible_v<_Arg>;

template <class _Arg>
concept __register_passable =
    !is_reference_v<_Arg> && sizeof(_Arg) <= 2 * sizeof(void*) && __itanium_trivial_for_calls<_Arg>;

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_REGISTER_PASSABLE_H

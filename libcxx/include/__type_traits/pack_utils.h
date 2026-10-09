//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_PACK_UTILS_H
#define _LIBCPP___TYPE_TRAITS_PACK_UTILS_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/type_list.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class, class>
inline const bool __contains_type_v = [] -> bool { static_assert(false); }();

template <class... _Args, class _SearchT>
inline const bool __contains_type_v<__type_list<_Args...>, _SearchT> = (__is_same(_Args, _SearchT) || ...);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_PACK_UTILS_H

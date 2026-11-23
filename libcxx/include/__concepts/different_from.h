//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONCEPTS_DIFFERENT_FROM_H
#define _LIBCPP___CONCEPTS_DIFFERENT_FROM_H

#include <__concepts/same_as.h>
#include <__config>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Up>
concept __different_from = !same_as<remove_cvref_t<_Tp>, remove_cvref_t<_Up>>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___CONCEPTS_DIFFERENT_FROM_H

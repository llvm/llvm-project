//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONCEPTS_COMPARISON_COMMON_TYPE_H
#define _LIBCPP___CONCEPTS_COMPARISON_COMMON_TYPE_H

#include <__concepts/convertible_to.h>
#include <__concepts/same_as.h>
#include <__config>
#include <__type_traits/common_reference.h>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _Tp, class _Up, class _CommonRef = common_reference_t<const _Tp&, const _Up&>>
concept __comparison_common_type_with_impl =
    same_as<common_reference_t<const _Tp&, const _Up&>, common_reference_t<const _Up&, const _Tp&>> && requires {
      requires convertible_to<const _Tp&, const _CommonRef&> || convertible_to<_Tp, const _CommonRef&>;
      requires convertible_to<const _Up&, const _CommonRef&> || convertible_to<_Up, const _CommonRef&>;
    };

template <class _Tp, class _Up>
concept __comparison_common_type_with = __comparison_common_type_with_impl<remove_cvref_t<_Tp>, remove_cvref_t<_Up>>;

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CONCEPTS_COMPARISON_COMMON_TYPE_H

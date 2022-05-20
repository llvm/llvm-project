//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_CONDITIONAL_H
#define _LIBCPP___TYPE_TRAITS_CONDITIONAL_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <bool _Bp, class _If, class _Then>
    struct _LIBCPP_TEMPLATE_VIS conditional {typedef _If type;};
template <class _If, class _Then>
    struct _LIBCPP_TEMPLATE_VIS conditional<false, _If, _Then> {typedef _Then type;};

#if _LIBCPP_STD_VER > 11
template <bool _Bp, class _If, class _Then> using conditional_t = typename conditional<_Bp, _If, _Then>::type;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_CONDITIONAL_H

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FMT_PAIR_LIKE_H
#define _LIBCPP___FORMAT_FMT_PAIR_LIKE_H

#include <__config>
#include <__fwd/pair.h>
#include <__fwd/tuple.h>
#include <__tuple/tuple_size.h>
#include <__type_traits/is_specialization.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

// [tuple.like] defines a tuple-like exposition only concept. This concept is not related to that. Therefore it uses a
// different name for the concept.
//
// TODO FMT Add a test to validate we fail when using that concept after P2165 has been implemented.

// [format.range.fmtkind]/2.2.1 and [tab:formatter.range.type]:
// "U is either a specialization of pair or a specialization of tuple such that tuple_size_v<U> is 2."
template <class _Tp>
concept __fmt_pair_like =
    __is_specialization_v<_Tp, pair> || (__is_specialization_v<_Tp, tuple> && tuple_size_v<_Tp> == 2);

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FORMAT_FMT_PAIR_LIKE_H

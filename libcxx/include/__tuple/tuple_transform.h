// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TUPLE_TUPLE_TRANSFORM_H
#define _LIBCPP___TUPLE_TUPLE_TRANSFORM_H

#include <__config>

#include <__functional/invoke.h>
#include <__utility/forward.h>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <class _Fun, class _Tuple>
_LIBCPP_HIDE_FROM_ABI constexpr auto __tuple_transform(_Fun&& __f, _Tuple&& __tuple) {
  return std::apply(
      [&]<class... _Types>(_Types&&... __elements) {
        return tuple<invoke_result_t<_Fun&, _Types>...>(std::invoke(__f, std::forward<_Types>(__elements))...);
      },
      std::forward<_Tuple>(__tuple));
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TUPLE_TUPLE_TRANSFORM_H

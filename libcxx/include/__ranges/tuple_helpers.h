// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_TUPLE_HELPERS_H
#define _LIBCPP___RANGES_TUPLE_HELPERS_H

#include <__functional/invoke.h>
#include <__utility/forward.h>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

namespace ranges {

template <class _Fun, class _Tuple>
_LIBCPP_HIDE_FROM_ABI constexpr auto __tuple_transform(_Fun&& __f, _Tuple&& __tuple) {
  return std::apply(
      [&]<class... _Types>(_Types&&... __elements) {
        return tuple<invoke_result_t<_Fun&, _Types>...>(std::invoke(__f, std::forward<_Types>(__elements))...);
      },
      std::forward<_Tuple>(__tuple));
}

} // namespace ranges

#endif // _LIBCPP___RANGES_TUPLE_HELPERS_H
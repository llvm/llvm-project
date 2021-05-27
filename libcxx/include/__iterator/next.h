// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_NEXT_H
#define _LIBCPP___ITERATOR_NEXT_H

#include <__config>
#include <__function_like.h>
#include <__iterator/advance.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {
struct __next_fn final : private __function_like {
  constexpr explicit __next_fn(__tag __x) noexcept : __function_like(__x) {}

  template <input_or_output_iterator _Ip>
  constexpr _Ip operator()(_Ip __x) const {
    ++__x;
    return __x;
  }

  template <input_or_output_iterator _Ip>
  constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n) const {
    ranges::advance(__x, __n);
    return __x;
  }

  template <input_or_output_iterator _Ip, sentinel_for<_Ip> _Sp>
  constexpr _Ip operator()(_Ip __x, _Sp __bound) const {
    ranges::advance(__x, __bound);
    return __x;
  }

  template <input_or_output_iterator _Ip, sentinel_for<_Ip> _Sp>
  constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n, _Sp __bound) const {
    ranges::advance(__x, __n, __bound);
    return __x;
  }
};

inline constexpr auto next = __next_fn(__function_like::__tag());
} // namespace ranges

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_PRIMITIVES_H

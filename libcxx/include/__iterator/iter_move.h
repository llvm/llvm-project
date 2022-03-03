// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_ITER_MOVE_H
#define _LIBCPP___ITERATOR_ITER_MOVE_H

#include <__concepts/class_or_enum.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_CONCEPTS)

// [iterator.cust.move]

namespace ranges {
namespace __iter_move {

void iter_move();

template <class _Tp>
concept __unqualified_iter_move =
  __class_or_enum<remove_cvref_t<_Tp>> &&
  requires (_Tp&& __t) {
    iter_move(std::forward<_Tp>(__t));
  };

// [iterator.cust.move]

struct __fn {
  template<class _Ip>
    requires __unqualified_iter_move<_Ip>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator()(_Ip&& __i) const
    noexcept(noexcept(iter_move(std::forward<_Ip>(__i))))
  {
    return iter_move(std::forward<_Ip>(__i));
  }

  template<class _Ip>
    requires (!__unqualified_iter_move<_Ip>) &&
             requires { *declval<_Ip>(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator()(_Ip&& __i) const
    noexcept(noexcept(*std::forward<_Ip>(__i)))
  {
    if constexpr (is_lvalue_reference_v<decltype(*declval<_Ip>())>) {
      return std::move(*std::forward<_Ip>(__i));
    } else {
      return *std::forward<_Ip>(__i);
    }
  }
};
} // namespace __iter_move

inline namespace __cpo {
  inline constexpr auto iter_move = __iter_move::__fn{};
} // namespace __cpo
} // namespace ranges

template<__dereferenceable _Tp>
  requires requires(_Tp& __t) { { ranges::iter_move(__t) } -> __can_reference; }
using iter_rvalue_reference_t = decltype(ranges::iter_move(declval<_Tp&>()));

#endif // !defined(_LIBCPP_HAS_NO_CONCEPTS)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_ITER_MOVE_H

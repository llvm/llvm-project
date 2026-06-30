// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_RESERVE_HINT_H
#define _LIBCPP___RANGES_RESERVE_HINT_H

#include <__concepts/class_or_enum.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__ranges/size.h>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26
// [range.prim.size.hint]

namespace ranges {
namespace __reserve_hint {
void reserve_hint() = delete;

template <typename _Tp>
concept __sized = requires(_Tp&& __t) { ranges::size(__t); };

template <typename _Tp>
concept __member_reserve_hint = !__sized<_Tp> && requires(_Tp&& __t) {
  { auto(__t.reserve_hint()) } -> __integer_like;
};

template <typename _Tp>
concept __unqualified_reserve_hint =
    !__sized<_Tp> && !__member_reserve_hint<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
      { auto(reserve_hint(__t)) } -> __integer_like;
    };

struct __fn {
  // `[range.prim.size.hint]`: `ranges::size(t)` is a valid expression
  template <__sized _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __integer_like auto operator()(_Tp&& __t) const
      noexcept(noexcept(ranges::size(__t))) {
    return ranges::size(__t);
  }

  // `[range.prim.size.hint]`: `auto(t.reserve_hint())` is a valid expression
  template <__member_reserve_hint _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __integer_like auto operator()(_Tp&& __t) const
      noexcept(noexcept(auto(__t.reserve_hint()))) {
    return auto(__t.reserve_hint());
  }

  // `[range.prim.size.hint]`: `auto(reserve_hint(t))` is a valid expression
  template <__unqualified_reserve_hint _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __integer_like auto operator()(_Tp&& __t) const
      noexcept(noexcept(auto(reserve_hint(__t)))) {
    return auto(reserve_hint(__t));
  }
};
} // namespace __reserve_hint

inline namespace __cpo {
inline constexpr auto reserve_hint = __reserve_hint::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_RESERVE_HINT_H

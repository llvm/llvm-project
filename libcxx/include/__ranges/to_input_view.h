// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_TO_INPUT_VIEW_H
#define _LIBCPP___RANGES_TO_INPUT_VIEW_H

#include <__algorithm/iter_swap.h>
#include <__concepts/constructible.h>
#include <__config>
#include <__iterator/indirectly_comparable.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/view_interface.h>
#include <__type_traits/maybe_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

namespace ranges {

template <input_range _V>
  requires view<_V>
class to_input_view : public view_interface<to_input_view<_V>> {
private:
  _V __base_;

  template <bool _Cont>
  class __iterator;

public:
  _LIBCPP_HIDE_FROM_ABI to_input_view()
    requires default_initializable<_V>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit to_input_view(_V __base) : __base_(std::move(__base)) {}

  // base
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _V base() const&
    requires copy_constructible<_V>
  {
    return __base_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _V base() && { return std::move(__base_); }

  // begin
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_V>)
  {
    return __iterator<false>(ranges::begin(__base_));
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _V>
  {
    return __iterator<true>(ranges::begin(__base_));
  }

  // end
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_V>)
  {
    return ranges::end(__base_);
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _V>
  {
    return ranges::end(__base_);
  }

  // size
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_V>
  {
    return ranges::size(__base_);
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _V>
  {
    return ranges::size(__base_);
  }
};

template <class _R>
to_input_view(_R&&) -> to_input_view<ranges::views::all_t<_R>>;

template <input_range _V>
  requires view<_V>
template <bool _Const>
class to_input_view<_V>::__iterator {
  using _Base                  = __maybe_const<_Const, _V>;
  iterator_t<_Base> __current_ = iterator_t<_Base>();

public:
  using difference_type  = range_difference_t<_Base>;
  using value_type       = range_value_t<_Base>;
  using iterator_concept = input_iterator_tag;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __iterator(iterator_t<_Base> __current) : __current_(std::move(__current)) {}

  _LIBCPP_HIDE_FROM_ABI __iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;

  _LIBCPP_HIDE_FROM_ABI __iterator(__iterator&&)            = default;
  _LIBCPP_HIDE_FROM_ABI __iterator& operator=(__iterator&&) = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_V>, iterator_t<_Base>>
      : __current_(std::move(__i.__current_)) {}

  // base
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() && { return std::move(__current_); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_Base>& base() const& noexcept { return __current_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const noexcept(noexcept(*__current_)) {
    return *__current_;
  }

  // operator ++
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    ++__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  // operator==
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iterator& __x, const sentinel_t<_Base>& __y) {
    return __x.__current_ == __y;
  }

  // operator --
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const sentinel_t<_Base>& __y, const __iterator& __x)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __y - __x.__current_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __x, const sentinel_t<_Base>& __y)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __x.__current_ - __y;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_Base>
  iter_move(const __iterator& __i) noexcept(noexcept(ranges::iter_move(__i.__current_))) {
    return ranges::iter_move(__i.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(const __iterator& __x, const __iterator& __y) noexcept(noexcept(iter_swap(__x.__current_, __y.__current_)))
    requires indirectly_swappable<iterator_t<_Base>>
  {
    ranges::iter_swap(__x.__current_, __y.__current_);
  }
};

template <class _V>
constexpr bool enable_borrowed_range<to_input_view<_V>> = enable_borrowed_range<_V>;

inline namespace __cpo {
struct __to_input_range_adaptor {
  template <ranges::input_range _V>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI inline constexpr auto operator()(_V&& __r) const {
    return to_input_view<ranges::views::all_t<_V>>(std::forward<_V>(__r));
  }

  template <ranges::input_range _V>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend inline constexpr auto operator|(_V&& __r, __to_input_range_adaptor) {
    return to_input_view<ranges::views::all_t<_V>>(std::forward<_V>(__r));
  }
};
} // namespace __cpo

namespace views {
inline constexpr auto to_input = ranges::__cpo::__to_input_range_adaptor{};
} // namespace views

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_TO_INPUT_VIEW_H

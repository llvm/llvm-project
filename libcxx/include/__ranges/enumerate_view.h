// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ENUMERATE_VIEW_H
#define _LIBCPP___RANGES_ENUMERATE_VIEW_H

#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/iter_move.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__type_traits/maybe_const.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

// [concept.object]

template <class _Rp>
concept __range_with_movable_references =
    input_range<_Rp> && std::move_constructible<range_reference_t<_Rp>> &&
    std::move_constructible<range_rvalue_reference_t<_Rp>>;

// [range.enumerate.view]

template <view _View>
  requires __range_with_movable_references<_View>
class enumerate_view : public view_interface<enumerate_view<_View>> {
  _View __base_ = _View();

  // [range.enumerate.iterator]
  template <bool _Const>
  class __iterator;

  // [range.enumerate.sentinel]
  template <bool _Const>
  class __sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr enumerate_view()
    requires default_initializable<_View>
  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit enumerate_view(_View __base) : __base_(std::move(__base)) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>(ranges::begin(__base_), 0);
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires __range_with_movable_references<const _View>
  {
    return __iterator<true>(ranges::begin(__base_), 0);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    if constexpr (forward_range<_View> && common_range<_View> && sized_range<_View>)
      return __iterator<false>(ranges::end(__base_), ranges::distance(__base_));
    else
      return __sentinel<false>(ranges::end(__base_));
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires __range_with_movable_references<const _View>
  {
    if constexpr (forward_range<_View> && common_range<const _View> && sized_range<const _View>)
      return __iterator<true>(ranges::end(__base_), ranges::distance(__base_));
    else
      return __sentinel<true>(ranges::end(__base_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return ranges::size(__base_);
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return ranges::size(__base_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }
};

template <class _Range>
enumerate_view(_Range&&) -> enumerate_view<views::all_t<_Range>>;

// [range.enumerate.iterator]

template <view _View>
  requires __range_with_movable_references<_View>
template <bool _Const>
class enumerate_view<_View>::__iterator {
  using _Base _LIBCPP_NODEBUG = __maybe_const<_Const, _View>;

  static consteval auto __get_iterator_concept() {
    if constexpr (random_access_range<_Base>) {
      return random_access_iterator_tag{};
    } else if constexpr (bidirectional_range<_Base>) {
      return bidirectional_iterator_tag{};
    } else if constexpr (forward_range<_Base>) {
      return forward_iterator_tag{};
    } else {
      return input_iterator_tag{};
    }
  }

  friend class enumerate_view<_View>;

public:
  using iterator_category = input_iterator_tag;
  using iterator_concept  = decltype(__get_iterator_concept());
  using difference_type   = range_difference_t<_Base>;
  using value_type        = tuple<difference_type, range_value_t<_Base>>;

private:
  using __reference_type _LIBCPP_NODEBUG = tuple<difference_type, range_reference_t<_Base>>;

  iterator_t<_Base> __current_ = iterator_t<_Base>();
  difference_type __pos_       = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __iterator(iterator_t<_Base> __current, difference_type __pos)
      : __current_(std::move(__current)), __pos_(__pos) {}

public:
  _LIBCPP_HIDE_FROM_ABI __iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>>
      : __current_(std::move(__i.__current_)), __pos_(__i.__pos_) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_Base>& base() const& noexcept { return __current_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() && { return std::move(__current_); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr difference_type index() const noexcept { return __pos_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator*() const { return __reference_type(__pos_, *__current_); }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    ++__current_;
    ++__pos_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { return ++*this; }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires forward_range<_Base>
  {
    auto __temp = *this;
    ++*this;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires bidirectional_range<_Base>
  {
    --__current_;
    --__pos_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __temp = *this;
    --*this;
    return *__temp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n)
    requires random_access_range<_Base>
  {
    __current_ += __n;
    __pos_ += __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n)
    requires random_access_range<_Base>
  {
    __current_ -= __n;
    __pos_ -= __n;
    return *this;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator[](difference_type __n) const
    requires random_access_range<_Base>
  {
    return __reference_type(__pos_ + __n, __current_[__n]);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) noexcept {
    return __x.__pos_ == __y.__pos_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr strong_ordering
  operator<=>(const __iterator& __x, const __iterator& __y) noexcept {
    return __x.__pos_ <=> __y.__pos_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __temp = __i;
    __temp += __n;
    return __temp;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, const __iterator& __i)
    requires random_access_range<_Base>
  {
    return __i + __n;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __temp = __i;
    __temp -= __n;
    return __temp;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __x, const __iterator& __y) noexcept {
    return __x.__pos_ - __y.__pos_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto iter_move(const __iterator& __i) noexcept(
      noexcept(ranges::iter_move(__i.__current_)) && is_nothrow_move_constructible_v<range_rvalue_reference_t<_Base>>) {
    return tuple<difference_type, range_rvalue_reference_t<_Base>>(__i.__pos_, ranges::iter_move(__i.__current_));
  }
};

// [range.enumerate.sentinel]

template <view _View>
  requires __range_with_movable_references<_View>
template <bool _Const>
class enumerate_view<_View>::__sentinel {
  using _Base _LIBCPP_NODEBUG = __maybe_const<_Const, _View>;

  sentinel_t<_Base> __end_ = sentinel_t<_Base>();

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __sentinel(sentinel_t<_Base> __end) : __end_(std::move(__end)) {}

  friend class enumerate_view<_View>;

public:
  _LIBCPP_HIDE_FROM_ABI __sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_Const> __other)
    requires _Const && convertible_to<sentinel_t<_View>, sentinel_t<_Base>>
      : __end_(std::move(__other.__end_)) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr sentinel_t<_Base> base() const { return __end_; }

  template <bool _OtherConst>
    requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__current_ == __y.__end_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _View>>
  operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__current_ - __y.__end_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _View>>
  operator-(const __sentinel& __x, const __iterator<_OtherConst>& __y) {
    return __x.__end_ - __y.__current_;
  }
};

template <class _View>
constexpr bool enable_borrowed_range<enumerate_view<_View>> = enable_borrowed_range<_View>;

namespace views {
namespace __enumerate {

// [range.enumerate.overview]

struct __fn : __range_adaptor_closure<__fn> {
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(/**/ enumerate_view<views::all_t<_Range>>(__range)))
      -> decltype(/*-------------------------------*/ enumerate_view<views::all_t<_Range>>(__range)) {
    return /*--------------------------------------*/ enumerate_view<views::all_t<_Range>>(__range);
  }
};

} // namespace __enumerate

inline namespace __cpo {

inline constexpr auto enumerate = __enumerate::__fn{};

} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ENUMERATE_VIEW_H

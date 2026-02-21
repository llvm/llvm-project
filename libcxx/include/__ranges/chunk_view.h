// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_CHUNK_VIEW_H
#define _LIBCPP___RANGES_CHUNK_VIEW_H

#include <__algorithm/ranges_min.h>
#include <__assert>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__functional/bind_back.h>
#include <__iterator/advance.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/take_view.h>
#include <__ranges/view_interface.h>
#include <__type_traits/conditional.h>
#include <__type_traits/decay.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/make_unsigned.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <class _Integral>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto __div_ceil(_Integral __num, _Integral __denom) {
  _Integral __r = __num / __denom;
  if (__num % __denom)
    ++__r;
  return __r;
}

template <view _View>
  requires input_range<_View>
class chunk_view : public view_interface<chunk_view<_View>> {
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_;
  _LIBCPP_NO_UNIQUE_ADDRESS range_difference_t<_View> __n_;
  _LIBCPP_NO_UNIQUE_ADDRESS range_difference_t<_View> __remainder_;
  _LIBCPP_NO_UNIQUE_ADDRESS __non_propagating_cache<iterator_t<_View>> __current_;

  class __outer_iterator;
  class __inner_iterator;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit chunk_view(_View __base, range_difference_t<_View> __n)
      : __base_(std::move(__base)), __n_(__n), __remainder_(0) {
    _LIBCPP_ASSERT_PEDANTIC(__n > 0, "Trying to construct a chunk_view with chunk size <= 0");
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires std::copy_constructible<_View>
  {
    return __base_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __outer_iterator begin() {
    __current_.__emplace(ranges::begin(__base_));
    __remainder_ = __n_;
    return __outer_iterator(*this);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr default_sentinel_t end() const noexcept {
    return std::default_sentinel;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __n_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __n_));
  }
};

template <view _View>
  requires input_range<_View>
class chunk_view<_View>::__outer_iterator {
  friend chunk_view;

  chunk_view* __parent_;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __outer_iterator(chunk_view& __parent)
      : __parent_(std::addressof(__parent)) {}

public:
  class value_type;
  using iterator_concept = input_iterator_tag;
  using difference_type  = range_difference_t<_View>;

  _LIBCPP_HIDE_FROM_ABI __outer_iterator(__outer_iterator&&) = default;

  _LIBCPP_HIDE_FROM_ABI __outer_iterator& operator=(__outer_iterator&&) = default;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
    _LIBCPP_ASSERT_PEDANTIC(*this != default_sentinel, "Trying to dereference past-the-end chunk_view iterator");
    return value_type(*__parent_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __outer_iterator& operator++() {
    _LIBCPP_ASSERT_PEDANTIC(*this != default_sentinel, "Trying to increment past-the-end chunk_view iterator");
    ranges::advance(*__parent_->__current_, __parent_->__remainder_, ranges::end(__parent_->__base_));
    __parent_->__remainder_ = __parent_->__n_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __outer_iterator& __i, default_sentinel_t) {
    return *__i.__parent_->__current_ == ranges::end(__i.__parent_->__base_) && __i.__parent_->__remainder_ != 0;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(default_sentinel_t, const __outer_iterator& __i)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    const auto __dist = ranges::end(__i.__parent_->__base_) - *__i.__parent_->__current_;
    if (__dist < __i.__parent_->__remainder_)
      return __dist == 0 ? 0 : 1;
    return ranges::__div_ceil(__dist - __i.__parent_->__remainder_, __i.__parent_->__n_) + 1;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __outer_iterator& __i, default_sentinel_t __s)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return -(__s - __i);
  }
};

template <view _View>
  requires input_range<_View>
class chunk_view<_View>::__outer_iterator::value_type : public view_interface<value_type> {
  friend __outer_iterator;

  chunk_view* __parent_;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit value_type(chunk_view& __parent) : __parent_(std::addressof(__parent)) {}

public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __inner_iterator begin() const noexcept {
    return __inner_iterator(*__parent_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr default_sentinel_t end() const noexcept { return default_sentinel; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return std::__to_unsigned_like(
        ranges::min(__parent_->__remainder_, ranges::end(__parent_->__base_) - *__parent_->__current_));
  }
};

template <view _View>
  requires input_range<_View>
class chunk_view<_View>::__inner_iterator {
  friend chunk_view;

  chunk_view* __parent_;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __inner_iterator(chunk_view& __parent) noexcept
      : __parent_(std::addressof(__parent)) {}

public:
  using iterator_concept = input_iterator_tag;
  using difference_type  = range_difference_t<_View>;
  using value_type       = range_value_t<_View>;

  _LIBCPP_HIDE_FROM_ABI __inner_iterator(__inner_iterator&&) = default;

  _LIBCPP_HIDE_FROM_ABI __inner_iterator& operator=(__inner_iterator&&) = default;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_View> base() const& { return *__parent_->__current_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr range_reference_t<_View> operator*() const {
    _LIBCPP_ASSERT_PEDANTIC(*this != default_sentinel, "Trying to dereference past-the-end chunk_view iterator");
    return **__parent_->__current_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __inner_iterator& operator++() {
    _LIBCPP_ASSERT_PEDANTIC(*this != default_sentinel, "Trying to increment past-the-end chunk_view iterator");
    ++*__parent_->__current_;
    if (*__parent_->__current_ == ranges::end(__parent_->__base_))
      __parent_->__remainder_ = 0;
    else
      --__parent_->__remainder_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __inner_iterator& __i, default_sentinel_t) {
    return __i.__parent_->__remainder_ == 0;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(default_sentinel_t, const __inner_iterator& __i)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return ranges::min(__i.__parent_->__remainder_, ranges::end(__i.__parent_->__base_) - *__i.__parent_->__current_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __inner_iterator& __i, default_sentinel_t __s)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return -(__s - __i);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  iter_move(const __inner_iterator& __i) noexcept(noexcept(ranges::iter_move(*__i.__parent_->__current_))) {
    return ranges::iter_move(*__i.__parent_->__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(const __inner_iterator& __x, const __inner_iterator& __y) noexcept(
      noexcept((ranges::iter_swap(*__x.__parent_->__current_, *__y.__parent_->__current_))))
    requires indirectly_swappable<iterator_t<_View>>
  {
    return ranges::iter_swap(*__x.__parent_->__current_, *__y.__parent_->__current_);
  }
};

template <view _View>
  requires forward_range<_View>
class chunk_view<_View> : public view_interface<chunk_view<_View>> {
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_;
  _LIBCPP_NO_UNIQUE_ADDRESS range_difference_t<_View> __n_;

  template <bool _Const>
  class __iterator;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit chunk_view(_View __base, range_difference_t<_View> __n)
      : __base_(std::move(__base)), __n_(__n) {
    _LIBCPP_ASSERT_PEDANTIC(__n > 0, "Trying to construct a chunk_view with chunk size <= 0");
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>(this, ranges::begin(__base_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires forward_range<const _View>
  {
    return __iterator<true>(this, ranges::begin(__base_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    if constexpr (common_range<_View> && sized_range<_View>) {
      auto __missing = (__n_ - ranges::distance(__base_) % __n_) % __n_;
      return __iterator<false>(this, ranges::end(__base_), __missing);
    } else if constexpr (common_range<_View> && !bidirectional_range<_View>)
      return __iterator<false>(this, ranges::end(__base_));
    else
      return default_sentinel;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires forward_range<const _View>
  {
    if constexpr (common_range<const _View> && sized_range<const _View>) {
      auto __missing = (__n_ - ranges::distance(__base_) % __n_) % __n_;
      return __iterator<true>(this, ranges::end(__base_), __missing);
    } else if constexpr (common_range<const _View> && !bidirectional_range<const _View>)
      return __iterator<true>(this, ranges::end(__base_));
    else
      return default_sentinel;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __n_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __n_));
  }
};

template <view _View>
  requires forward_range<_View>
template <bool _Const>
class chunk_view<_View>::__iterator {
  friend chunk_view;

  using _Parent _LIBCPP_NODEBUG = __maybe_const<_Const, chunk_view>;
  using _Base _LIBCPP_NODEBUG   = __maybe_const<_Const, _View>;

  iterator_t<_Base> __current_         = iterator_t<_Base>();
  sentinel_t<_Base> __end_             = sentinel_t<_Base>();
  range_difference_t<_Base> __n_       = 0;
  range_difference_t<_Base> __missing_ = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(
      _Parent* __parent, iterator_t<_Base> __current, range_difference_t<_Base> __missing = 0)
      : __current_(__current), __end_(ranges::end(__parent->__base_)), __n_(__parent->__n_), __missing_(__missing) {}

  [[nodiscard]] static consteval auto __get_iterator_concept() {
    if constexpr (random_access_range<_Base>)
      return random_access_iterator_tag{};
    else if constexpr (bidirectional_range<_Base>)
      return bidirectional_iterator_tag{};
    else
      return forward_iterator_tag{};
  }

public:
  using iterator_category = input_iterator_tag;
  using iterator_concept  = decltype(__iterator::__get_iterator_concept());
  using value_type        = decltype(views::take(subrange(__current_, __end_), __n_));
  using difference_type   = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>> &&
                 convertible_to<sentinel_t<_View>, sentinel_t<_Base>>
      : __current_(std::move(__i.__current_)),
        __end_(std::move(__i.__end_)),
        __n_(__i.__n_),
        __missing_(__i.__missing_) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() const { return __current_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
    _LIBCPP_ASSERT_PEDANTIC(__current_ != __end_, "Trying to dereference past-the-end chunk_view iterator");
    return views::take(subrange(__current_, __end_), __n_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](difference_type __pos) const
    requires random_access_range<_Base>
  {
    return *(*this + __pos);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    _LIBCPP_ASSERT_PEDANTIC(__current_ != __end_, "Trying to increment past-the-end chunk_view iterator");
    __missing_ = ranges::advance(__current_, __n_, __end_);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int) {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires bidirectional_range<_Base>
  {
    ranges::advance(__current_, __missing_ - __n_);
    __missing_ = 0;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int) {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __x)
    requires random_access_range<_Base>
  {
    if (__x > 0) {
      _LIBCPP_ASSERT_PEDANTIC(ranges::distance(__current_, __end_) > __n_ * (__x - 1),
                              "Trying to advance chunk_view iterator out of range");
      ranges::advance(__current_, __n_ * (__x - 1));
      __missing_ = ranges::advance(__current_, __n_, __end_);
    } else if (__x < 0) {
      ranges::advance(__current_, __n_ * __x + __missing_);
      __missing_ = 0;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __x)
    requires random_access_range<_Base>
  {
    return *this += -__x;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
    return __x.__current_ == __y.__current_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, default_sentinel_t) {
    return __x.__current_ == __x.__end_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__current_ < __y.__current_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __y < __x;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__y < __x);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__x < __y);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator+(const __iterator& __i, difference_type __pos)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __pos;
    return __r;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator+(difference_type __pos, const __iterator& __i)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __pos;
    return __r;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator-(const __iterator& __i, difference_type __pos)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r -= __pos;
    return __r;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __i, const __iterator& __j)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>>
  {
    return (__i.__current_ - __j.__current_ + __i.__missing_ - __j.__missing_) / __i.__n_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(default_sentinel_t, const __iterator& __i)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return ranges::__div_ceil(__i.__end_ - __i.__current_, __i.__n_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __i, default_sentinel_t __s)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return -(__s - __i);
  }
};

template <class _Range>
chunk_view(_Range&&, range_difference_t<_Range>) -> chunk_view<views::all_t<_Range>>;

template <class _View>
inline constexpr bool enable_borrowed_range<chunk_view<_View>> = forward_range<_View> && enable_borrowed_range<_View>;

namespace views {
namespace __chunk {
struct __fn {
  template <viewable_range _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range, range_difference_t<_Range> __n) noexcept(
      noexcept(/*-----*/ chunk_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n))))
      -> decltype(/*--*/ chunk_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n))) {
    return /*---------*/ chunk_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n));
  }

  template <class _DifferenceType>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_DifferenceType __n) noexcept(is_nothrow_constructible_v<decay_t<_DifferenceType>, _DifferenceType>) {
    return __pipeable(std::__bind_back(__fn{}, std::forward<_DifferenceType>(__n)));
  }
};

} // namespace __chunk

inline namespace __cpo {
inline constexpr auto chunk = __chunk::__fn{};

} // namespace __cpo
} // namespace views

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CHUNK_VIEW_H

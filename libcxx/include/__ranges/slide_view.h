// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_SLIDE_VIEW_H
#define _LIBCPP___RANGES_SLIDE_VIEW_H

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
#include <__iterator/next.h>
#include <__iterator/prev.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/counted.h>
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

template <class _View>
concept __slide_caches_nothing = random_access_range<_View> && sized_range<_View>;

template <class _View>
concept __slide_caches_last = !__slide_caches_nothing<_View> && bidirectional_range<_View> && common_range<_View>;

template <class _View>
concept __slide_caches_first = !__slide_caches_nothing<_View> && !__slide_caches_last<_View>;

template <forward_range _View>
  requires view<_View>
class slide_view : public view_interface<slide_view<_View>> {
public:
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_;
  _LIBCPP_NO_UNIQUE_ADDRESS range_difference_t<_View> __n_;
  _LIBCPP_NO_UNIQUE_ADDRESS _If<__slide_caches_first<_View>, __non_propagating_cache<iterator_t<_View>>, __empty_cache>
      __cached_begin_;
  _LIBCPP_NO_UNIQUE_ADDRESS _If<__slide_caches_last<_View>, __non_propagating_cache<iterator_t<_View>>, __empty_cache>
      __cached_end_;

  template <bool _Const>
  class __iterator;
  class __sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit slide_view(_View __base, range_difference_t<_View> __n)
      : __base_(std::move(__base)), __n_(__n) {
    _LIBCPP_ASSERT_PEDANTIC(__n > 0, "Trying to construct a slide_view with slide size <= 0");
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!(__simple_view<_View> && __slide_caches_nothing<const _View>))
  {
    if constexpr (__slide_caches_first<_View>) {
      __cached_begin_ = __iterator<false>(
          ranges::begin(__base_), ranges::next(ranges::begin(__base_), __n_ - 1, ranges::end(__base_)), __n_);
      return __cached_begin_;
    } else
      return __iterator<false>(ranges::begin(__base_), __n_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires __slide_caches_nothing<const _View>
  {
    return __iterator<true>(ranges::begin(__base_), __n_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!(__simple_view<_View> && __slide_caches_nothing<const _View>))
  {
    if constexpr (__slide_caches_nothing<_View>)
      return __iterator<false>(ranges::begin(__base_) + range_difference_t<_View>(size()), __n_);
    else if constexpr (__slide_caches_last<_View>) {
      __cached_end_ = __iterator<false>(ranges::prev(ranges::end(__base_), __n_ - 1, ranges::begin(__base_)), __n_);
      return __cached_end_;
    } else if constexpr (common_range<_View>)
      return __iterator<false>(ranges::end(__base_), ranges::end(__base_), __n_);
    else
      return __sentinel(ranges::end(__base_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires __slide_caches_nothing<const _View>
  {
    return begin() + range_difference_t<const _View>(size());
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    auto __sz = ranges::distance(__base_) - __n_ + 1;
    if (__sz < 0)
      __sz = 0;
    return std::__to_unsigned_like(__sz);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    auto __sz = ranges::distance(__base_) - __n_ + 1;
    if (__sz < 0)
      __sz = 0;
    return std::__to_unsigned_like(__sz);
  }
};

template <forward_range _View>
  requires view<_View>
template <bool _Const>
class slide_view<_View>::__iterator {
  friend slide_view;
  using _Base _LIBCPP_NODEBUG = _If<_Const, const _View, _View>;

  _LIBCPP_NO_UNIQUE_ADDRESS iterator_t<_Base> __current_ = iterator_t<_Base>();
  _LIBCPP_NO_UNIQUE_ADDRESS _If<__slide_caches_first<_Base>, iterator_t<_Base>, __empty_cache> __last_ele_ =
      _If<__slide_caches_first<_Base>, iterator_t<_Base>, __empty_cache>();
  _LIBCPP_NO_UNIQUE_ADDRESS range_difference_t<_Base> __n_ = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(iterator_t<_Base> __current, range_difference_t<_Base> __n)
    requires(!__slide_caches_first<_Base>)
      : __current_(__current), __n_(__n) {}

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(
      iterator_t<_Base> __current, iterator_t<_Base> __last_ele, range_difference_t<_Base> __n)
    requires __slide_caches_first<_Base>
      : __current_(__current), __last_ele_(__last_ele), __n_(__n) {}

  [[nodiscard]] static consteval auto __get_iterator_concept() {
    if constexpr (random_access_range<_Base>)
      return random_access_iterator_tag{};
    else if constexpr (bidirectional_range<_Base>)
      return bidirectional_iterator_tag{};
    else
      return forward_iterator_tag{};
  }

public:
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = decltype(__get_iterator_concept());
  using value_type        = decltype(views::counted(__current_, __n_));
  using difference_type   = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>>
      : __current_(std::move(__i.__current_)), __n_(__i.__n_) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator*() const { return views::counted(__current_, __n_); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator[](difference_type __pos) const
    requires random_access_range<_Base>
  {
    return views::counted(__current_ + __pos, __n_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    __current_ = ranges::next(__current_);
    if constexpr (__slide_caches_first<_Base>)
      __last_ele_ = ranges::next(__last_ele_);
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
    __current_ = ranges::prev(__current_);
    if constexpr (__slide_caches_first<_Base>)
      __last_ele_ = ranges::prev(__last_ele_);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n)
    requires random_access_range<_Base>
  {
    __current_ = __current_ + __n;
    if constexpr (__slide_caches_first<_Base>)
      __last_ele_ = __last_ele_ + __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n)
    requires random_access_range<_Base>
  {
    __current_ = __current_ - __n;
    if constexpr (__slide_caches_first<_Base>)
      __last_ele_ = __last_ele_ - __n;
    return *this;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
    if constexpr (__slide_caches_first<_Base>)
      return __x.__last_ele_ == __y.__last_ele_;
    else
      return __x.__current_ == __y.__current_;
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
    if constexpr (__slide_caches_first<_View>)
      return __i.__last_ele_ - __j.__last_ele_;
    else
      return __i.__current_ - __j.__current_;
  }
};

template <forward_range _View>
  requires view<_View>
class slide_view<_View>::__sentinel {
  sentinel_t<_View> __end_;

  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel(sentinel_t<_View> __end) : __end_(__end) {}

public:
  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel() = default;

  template <bool _Const>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iterator<_Const>& __x, const __sentinel& __y) {
    return __x.__last_ele_ == __y.__end_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<_View>
  operator-(const __iterator<false>& __x, const __sentinel& __y)
    requires disable_sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __x.__last_ele_ - __y.__end_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<_View>
  operator-(const __sentinel& __y, const __iterator<false>& __x)
    requires disable_sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __y.__end_ - __x.__last_ele_;
  }
};

template <class _Range>
slide_view(_Range&&, range_difference_t<_Range>) -> slide_view<views::all_t<_Range>>;

template <class _View>
inline constexpr bool enable_borrowed_range<slide_view<_View>> = enable_borrowed_range<_View>;

namespace views {
namespace __slide {
struct __fn {
  template <viewable_range _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range, range_difference_t<_Range> __n) noexcept(
      noexcept(/*-----*/ slide_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n))))
      -> decltype(/*--*/ slide_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n))) {
    return /*---------*/ slide_view(std::forward<_Range>(__range), std::forward<range_difference_t<_Range>>(__n));
  }

  template <class _DifferenceType>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_DifferenceType __n) noexcept(is_nothrow_constructible_v<decay_t<_DifferenceType>, _DifferenceType>) {
    return __pipeable(std::__bind_back(__fn{}, std::forward<_DifferenceType>(__n)));
  }
};

} // namespace __slide

inline namespace __cpo {
inline constexpr auto slide = __slide::__fn{};

} // namespace __cpo
} // namespace views

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_SLIDE_VIEW_H

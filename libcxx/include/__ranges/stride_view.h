// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_STRIDE_VIEW_H
#define _LIBCPP___RANGES_STRIDE_VIEW_H

#include <__config>

#include <__compare/three_way_comparable.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/derived_from.h>
#include <__functional/bind_back.h>
#include <__iterator/advance.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/reverse_iterator.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/subrange.h>
#include <__ranges/view_interface.h>
#include <__ranges/views.h>
#include <__type_traits/conditional.h>
#include <__type_traits/maybe_const.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <class _Value>
_LIBCPP_HIDE_FROM_ABI constexpr _Value __div_ceil(_Value __left, _Value __right) {
  _Value __r = __left / __right;
  if (__left % __right) {
    __r++;
  }
  return __r;
}

template <input_range _View>
  requires view<_View>
class stride_view : public view_interface<stride_view<_View>> {
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_ = _View();
  range_difference_t<_View> __stride_     = 0;

  template <bool _Const>
  class __iterator;
  template <bool _Const>
  class __sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit stride_view(_View __base, range_difference_t<_View> __stride)
      : __base_(std::move(__base)), __stride_(__stride) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto stride() { return __stride_; }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires ranges::sized_range<_View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __stride_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires ranges::sized_range<const _View>
  {
    return std::__to_unsigned_like(ranges::__div_ceil(ranges::distance(__base_), __stride_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>{*this, ranges::begin(__base_)};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires ranges::range<const _View>
  {
    return __iterator<true>{*this, ranges::begin(__base_)};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View> && common_range<_View> && sized_range<_View> && forward_range<_View>)
  {
    auto __missing = (__stride_ - ranges::distance(__base_) % __stride_) % __stride_;
    return __iterator<false>{*this, ranges::end(__base_), __missing};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View> && common_range<_View> && !bidirectional_range<_View>)
  {
    return __iterator<false>{*this, ranges::end(__base_)};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    return std::default_sentinel;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires(range<const _View> && common_range<const _View> && sized_range<const _View> && forward_range<const _View>)
  {
    auto __missing = (__stride_ - ranges::distance(__base_) % __stride_) % __stride_;
    return __iterator<true>{*this, ranges::end(__base_), __missing};
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires(range<const _View> && common_range<_View> && !bidirectional_range<_View>)
  {
    return __iterator<true>{*this, ranges::end(__base_)};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires(range<const _View>)
  {
    return std::default_sentinel;
  }
}; // class stride_view

template <class _View>
struct __stride_iterator_category {};

template <forward_range _View>
struct __stride_iterator_category<_View> {
  using _Cat = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using iterator_category =
      _If<derived_from<_Cat, random_access_iterator_tag>,
          random_access_iterator_tag,
          /* else */ _Cat >;
};

template <input_range _View>
  requires view<_View>
template <bool _Const>
class stride_view<_View>::__iterator : public __stride_iterator_category<_View> {
  using _Parent = __maybe_const<_Const, stride_view<_View>>;
  using _Base   = __maybe_const<_Const, _View>;

public:
  using difference_type = range_difference_t<_Base>;
  using value_type      = range_value_t<_Base>;
  using iterator_concept =
      _If<random_access_range<_Base>,
          random_access_iterator_tag,
          _If<bidirectional_range<_Base>,
              bidirectional_iterator_tag,
              _If<forward_range<_Base>,
                  forward_iterator_tag,
                  /* else */ input_iterator_tag >>>;

  _LIBCPP_NO_UNIQUE_ADDRESS iterator_t<_Base> __current_     = iterator_t<_Base>();
  _LIBCPP_NO_UNIQUE_ADDRESS ranges::sentinel_t<_Base> __end_ = ranges::sentinel_t<_Base>();
  difference_type __stride_                                  = 0;
  difference_type __missing_                                 = 0;

  // using iterator_category = inherited;
  _LIBCPP_HIDE_FROM_ABI __iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && std::convertible_to<ranges::iterator_t<_View>, ranges::iterator_t<_Base>> &&
                 std::convertible_to<ranges::sentinel_t<_View>, ranges::sentinel_t<_Base>>
      : __current_(std::move(__i.__current_)),
        __end_(std::move(__i.__end_)),
        __stride_(__i.__stride_),
        __missing_(__i.__missing_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(
      _Parent& __parent, ranges::iterator_t<_Base> __current, difference_type __missing = 0)
      : __current_(std::move(__current)),
        __end_(ranges::end(__parent.__base_)),
        __stride_(__parent.__stride_),
        __missing_(__missing) {}

  _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_View> const& base() const& noexcept { return __current_; }
  _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_View> base() && { return std::move(__current_); }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const { return *__current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    __missing_ = ranges::advance(__current_, __stride_, __end_);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires forward_range<_Base>
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires bidirectional_range<_Base>
  {
    ranges::advance(__current_, __missing_ - __stride_);
    __missing_ = 0;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __s)
    requires random_access_range<_Base>
  {
    if (__s > 0) {
      ranges::advance(__current_, __stride_ * (__s - 1));
      __missing_ = ranges::advance(__current_, __stride_, __end_);
    } else if (__s < 0) {
      ranges::advance(__current_, __stride_ * __s + __missing_);
      __missing_ = 0;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __s)
    requires random_access_range<_Base>
  {
    return *this += -__s;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __s) const
    requires random_access_range<_Base>
  {
    return *(*this + __s);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(__iterator const& __x, default_sentinel_t) {
    return __x.__current_ == __x.__end_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(__iterator const& __x, __iterator const& __y)
    requires equality_comparable<iterator_t<_Base>>
  {
    return __x.__current_ == __y.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(__iterator const& __x, __iterator const& __y)
    requires random_access_range<_Base>
  {
    return __x.__current_ < __y.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(__iterator const& __x, __iterator const& __y)
    requires random_access_range<_Base>
  {
    return __y < __x;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(__iterator const& __x, __iterator const& __y)
    requires random_access_range<_Base>
  {
    return !(__y < __x);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(__iterator const& __x, __iterator const& __y)
    requires random_access_range<_Base>
  {
    return !(__x < __y);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=>(__iterator const& __x, __iterator const& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(__iterator const& __i, difference_type __s)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __s;
    return __r;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __s, __iterator const& __i)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __s;
    return __r;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(__iterator const& __i, difference_type __s)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r -= __s;
    return __r;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(__iterator const& __x, __iterator const& __y)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>> && forward_range<_Base>
  {
    auto __n = __x.__current_ - __y.__current_;
    return (__n + __x.__missing_ - __y.__missing_) / __x.__stride_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(__iterator const& __x, __iterator const& __y)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>>
  {
    auto __n = __x.__current_ - __y.__current_;
    if (__n < 0) {
      return -ranges::__div_ceil(-__n, __x.__stride_);
    }
    return ranges::__div_ceil(__n, __x.__stride_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(default_sentinel_t, __iterator const& __x)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return ranges::__div_ceil(__x.__end_ - __x.__current_, __x.__stride_);
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(__iterator const& __x, default_sentinel_t __y)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return -(__y - __x);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_Base>
  iter_move(__iterator const& __it) noexcept(noexcept(ranges::iter_move(__it.__current_))) {
    return ranges::iter_move(__it.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(__iterator const& __x,
            __iterator const& __y) noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
    requires indirectly_swappable<iterator_t<_Base>>
  {
    return ranges::iter_swap(__x.__current_, __y.__current_);
  }
}; // class stride_view::__iterator

template <input_range _View>
  requires view<_View>
template <bool _Const>
class stride_view<_View>::__sentinel {
public:
  sentinel_t<_View> __end_ = sentinel_t<_View>();

  _LIBCPP_HIDE_FROM_ABI __sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __sentinel(stride_view<_View>& __parent)
      : __end_(ranges::end(__parent.__base_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr sentinel_t<_View> base() const { return __end_; }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(__iterator<true> const& __x, __sentinel const& __y) {
    return __x.__current_ == __y.__end_;
  }
}; // class stride_view::__sentinel

template <class _Range>
stride_view(_Range&&) -> stride_view<views::all_t<_Range>>;

namespace views {
namespace __stride {
// removed this.
struct __fn {
  template <viewable_range _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, range_difference_t<_Range> __n) const
      noexcept(noexcept(stride_view{std::forward<_Range>(__range), __n}))
          -> decltype(stride_view{std::forward<_Range>(__range), __n}) {
    return stride_view{std::forward<_Range>(__range), __n};
  }

  template <class _Np>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Np&& __n) const {
    return __range_adaptor_closure_t(std::__bind_back(*this, std::forward<_Np>(__n)));
  }
};
} // namespace __stride

inline namespace __cpo {
inline constexpr auto stride = __stride::__fn{};
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_STRIDE_VIEW_H

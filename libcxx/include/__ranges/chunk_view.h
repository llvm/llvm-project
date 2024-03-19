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

#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__functional/bind_back.h>
#include <__iterator/advance.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/subrange.h>
#include <__ranges/take_view.h>
#include <__ranges/view_interface.h>
#include <__type_traits/make_unsigned.h>
#include <__type_traits/maybe_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <class _Integer>
constexpr _Integer __div_ceil(_Integer __num, _Integer __denom) {
  _Integer __r = __num / __denom;
  if (__num % __denom) {
    ++__r;
  }
  return __r;
}

template <view _View>
  requires input_range<_View>
class chunk_view : public view_interface<chunk_view<_View>> {};

template <view _View>
  requires forward_range<_View>
class chunk_view<_View> : public view_interface<chunk_view<_View>> {
private:
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_;
  range_difference_t<_View> __n_;

  template <bool>
  class __iterator;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit chunk_view(_View __base, range_difference_t<_View> __n)
      : __base_(std::move(__base)), __n_(__n) {
    _LIBCPP_ASSERT_UNCATEGORIZED(__n > 0, "__n must be greater than 0");
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>(this, ranges::begin(__base_));
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires forward_range<const _View>
  {
    return __iterator<true>(this, ranges::begin(__base_));
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    if constexpr (common_range<_View> && sized_range<_View>) {
      auto __missing = (__n_ - ranges::distance(__base_) % __n_) % __n_;
      return __iterator<false>(this, ranges::end(__base_), __missing);
    } else if constexpr (common_range<_View> && !bidirectional_range<_View>) {
      return __iterator<false>(this, ranges::end(__base_));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires forward_range<_View>
  {
    if constexpr (common_range<const _View> && sized_range<const _View>) {
      auto __missing = (__n_ - ranges::distance(__base_) % __n_) % __n_;
      return __iterator<true>(this, ranges::end(__base_), __missing);
    } else if constexpr (common_range<const _View> && !bidirectional_range<const _View>) {
      return __iterator<true>(this, ranges::end(__base_));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return std::__to_unsigned_like(__div_ceil(ranges::distance(__base_), __n_));
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return std::__to_unsigned_like(__div_ceil(ranges::distance(__base_), __n_));
  }
};

template <class _View>
chunk_view(_View&&, range_difference_t<_View>) -> chunk_view<views::all_t<_View>>;

template <class _View>
inline constexpr bool enable_borrowed_range<chunk_view<_View>> = enable_borrowed_range<_View> && forward_range<_View>;

template <view _View>
  requires forward_range<_View>
template <bool _Const>
class chunk_view<_View>::__iterator {
private:
  friend chunk_view;

  using _Parent = __maybe_const<_Const, chunk_view>;
  using _Base   = __maybe_const<_Const, _View>;

  iterator_t<_Base> __current_         = iterator_t<_Base>();
  sentinel_t<_Base> __end_             = sentinel_t<_Base>();
  range_difference_t<_Base> __n_       = 0;
  range_difference_t<_Base> __missing_ = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(
      _Parent* __parent, iterator_t<_Base> __current, range_difference_t<_Base> __missing = 0)
      : __current_(std::move(__current)),
        __end_(ranges::end(__parent->__base_)),
        __n_(__parent->__n_),
        __missing_(__missing) {}

public:
  using iterator_category = input_iterator_tag;
  using iterator_concept =
      conditional_t<random_access_range<_Base>,
                    random_access_iterator_tag,
                    conditional_t<bidirectional_range<_Base>, bidirectional_iterator_tag, forward_iterator_tag>>;
  using value_type      = decltype(views::take(subrange(__current_, __end_), __n_));
  using difference_type = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>> &&
                 convertible_to<sentinel_t<_View>, sentinel_t<_Base>>
      : __current_(std::move(__i.__current_)),
        __end_(std::move(__i.__end_)),
        __n_(__i.__n_),
        __missing_(__i.__missing_) {}

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() const { return __current_; }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
    _LIBCPP_ASSERT_PEDANTIC(__current_ != __end_, "Dereferencing past-the-end chunk_view iterator");
    return views::take(subrange(__current_, __end_), __n_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    _LIBCPP_ASSERT_PEDANTIC(__current_ != __end_, "Incrementing past-the-end chunk_view iterator");
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

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(const difference_type __offset)
    requires random_access_range<_Base>
  {
    if (__offset > 0) {
      ranges::advance(__current_, __n_ * (__offset - 1));
      __missing_ = ranges::advance(__current_, __n_, __end_);
    } else if (__offset < 0) {
      ranges::advance(__current_, __n_ * __offset + __missing_);
      __missing_ = 0;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(const difference_type __offset)
    requires random_access_range<_Base>
  {
    return *this += -__offset;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](const difference_type __offset) const
    requires random_access_range<_Base>
  {
    return *(*this + __offset);
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iterator& __x, const __iterator& __y) {
    return __x.__current_ == __y.__current_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iterator& __x, default_sentinel_t) {
    return __x.__current_ == __x.__end_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__current_ < __y.__current_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __y < __x;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator<=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__y < __x);
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator>=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__x < __y);
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator+(const __iterator& __x, const difference_type __y)
    requires random_access_range<_Base>
  {
    return __iterator{__x} += __y;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator+(const difference_type __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __y + __x;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator
  operator-(const __iterator& __x, difference_type __y)
    requires random_access_range<_Base>
  {
    return __iterator{__x} -= __y;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __x, const __iterator& __y)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>>
  {
    return (__x.__current_ - __y.__current_ + __x.__missing_ - __y.__missing_) / __x.n_;
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const default_sentinel_t, const __iterator& __x)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __div_ceil(__x.__end_ - __x.__current_, __x.__n_);
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __iterator& __x, const default_sentinel_t __y)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return -(__y - __x);
  }
};

namespace views {
namespace __chunk {
struct __fn {
  template <class _Range, convertible_to<range_difference_t<_Range>> _Np>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Np&& __n) const
      noexcept(noexcept(chunk_view(std::forward<_Range>(__range), std::forward<_Np>(__n))))
          -> decltype(chunk_view(std::forward<_Range>(__range), std::forward<_Np>(__n))) {
    return chunk_view(std::forward<_Range>(__range), std::forward<_Np>(__n));
  }

  template <class _Np>
    requires constructible_from<decay_t<_Np>, _Np>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Np&& __n) const
      noexcept(is_nothrow_constructible_v<decay_t<_Np>, _Np>) {
    return __range_adaptor_closure_t(std::__bind_back(*this, std::forward<_Np>(__n)));
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

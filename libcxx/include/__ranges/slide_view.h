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

#include <__assert>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__config>
#include <__functional/bind_back.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/prev.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/counted.h>
#include <__ranges/empty_view.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/view_interface.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <typename _V>
concept __slide_caches_nothing = random_access_range<_V> && sized_range<_V>;
template <typename _V>
concept __slide_caches_last = !__slide_caches_nothing<_V> && bidirectional_range<_V> && common_range<_V>;
template <typename _V>
concept __slide_caches_first = !__slide_caches_nothing<_V> && !__slide_caches_last<_V>;

template <forward_range _View>
  requires view<_View>
class slide_view : public view_interface<slide_view<_View>> {
public:
  template <bool>
  class __iterator;
  class __sentinel;

private:
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_ = _View();
  range_difference_t<_View> __n_          = 0;
  using _Cache = _If<!(__slide_caches_nothing<_View>), __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _Cache __cached_begin_;
  _Cache __cached_end_;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit slide_view(_View __base, range_difference_t<_View> __n)
      : __base_(std::move(__base)), __n_(__n) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!(__simple_view<_View> && __slide_caches_nothing<const _View>) && __slide_caches_last<_View>)
  {
    auto __first = ranges::begin(__base_);
    if (!__cached_begin_.__has_value()) {
      __cached_begin_.__emplace(ranges::next(__first, __n_ - 1, ranges::end(__base_)));
    }
    return __iterator<false>(__first, __cached_begin_, __n_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!(__simple_view<_View> && __slide_caches_nothing<const _View>))
  {
    return __iterator<false>(ranges::begin(__base_), __n_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires __slide_caches_nothing<_View>
  {
    return __iterator<true>(ranges::begin(__base_), __n_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!(__simple_view<_View> && __slide_caches_nothing<const _View>))
  {
    if constexpr (__slide_caches_nothing<_View>) {
      return __iterator<false>(ranges::begin(__base_) + range_difference_t<_View>(size()), __n_);
    } else if constexpr (__slide_caches_last<_View>) {
      return __iterator<false>(ranges::prev(ranges::end(__base_), __n_ - 1, ranges::begin(__base_)), __n_);
    } else if constexpr (common_range<_View>) {
      return __iterator<false>(ranges::end(__base_), ranges::end(__base_), __n_);
    } else {
      return sentinel<false>(ranges::end(__base_));
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires __slide_caches_nothing<const _View>
  {
    return begin() + range_difference_t<const _View>(size());
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    auto __size = ranges::distance(__base_) - __n_ + 1;
    if (__size < 0) {
      __size = 0;
    }
    return std::__to_unsigned_like(__size);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    auto __size = ranges::distance(__base_) - __n_ + 1;
    if (__size < 0) {
      __size = 0;
    }
    return __size;
  }
};

template <class _Range>
slide_view(_Range&&) -> slide_view<views::all_t<_Range>>;

template <class _View>
struct __slide_view_iterator_concept {
  using type = forward_iterator_tag;
};

template <random_access_range _View>
struct __slide_view_iterator_concept<_View> {
  using type = random_access_iterator_tag;
};

template <bidirectional_range _View>
struct __slide_view_iterator_concept<_View> {
  using type = bidirectional_iterator_tag;
};

template <forward_range _View>
  requires view<_View>
template <bool _Const>
class slide_view<_View>::__iterator : public __slide_view_iterator_concept<_View> {
  friend slide_view;

  using _Base           = __maybe_const<_Const, _View>;
  using _Last           = _If<__slide_caches_first<_Base>, iterator_t<_Base>, empty_view<_Base>>;
  slide_view* __parent_ = nullptr;
  _LIBCPP_NO_UNIQUE_ADDRESS iterator_t<_Base> __current_ = iterator_t<_Base>();
  _LIBCPP_NO_UNIQUE_ADDRESS _Last __last_                = {};
  range_difference_t<_Base> __n_                         = 0;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(iterator_t<_Base> __current, range_difference_t<_Base> __n)
    requires(!(__slide_caches_first<_Base>))
      : __current_(__current), __n_(__n) {}

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(iterator_t<_Base> __current, _Last __last, range_difference_t<_Base> __n)
    requires(__slide_caches_first<_Base>)
      : __current_(__current), __last_(__last), __n_(__n) {}

public:
  using value_type        = decltype(views::counted(__current_, __n_));
  using difference_type   = range_difference_t<_Base>;
  using iterator_category = input_iterator_tag;
  using iterator_concept  = __slide_view_iterator_concept<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>>
      : __current_(std::move(__i.__current_)), __n_(__i.__n_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr auto operator*() const { return views::counted(__current_, __n_); }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    __current_ = ranges::next(__current_);
    if constexpr (__slide_caches_first<_Base>) {
      __last_ = ranges::next(__last_);
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int) {
    auto __tmp = *this;
    ++*this;
    return __tmp;
    // TODO ...
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires bidirectional_range<_View>
  {
    __current_ = ranges::prev(__current_);
    if constexpr (__slide_caches_first<_Base>) {
      __last_ = std::prev(__last_);
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_View>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __x)
    requires random_access_range<_Base>
  {
    __current_ = std::prev(__current_, __x);
    if constexpr (__slide_caches_first<_Base>) {
      __last_ = std::prev(__last_, __x);
    }
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator+=(difference_type __x)
    requires random_access_range<_Base>
  {
    __current_ = std::next(__current_, __x);
    if constexpr (__slide_caches_first<_Base>) {
      __last_ = std::next(__last_, __x);
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto operator[](difference_type __x) const
    requires random_access_range<_Base>
  {
    return views::counted(__current_ + __x, __n_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
    if constexpr (__slide_caches_first<_Base>) {
      __x.__last_ == __y.__last_;
    }
    return __x.__current_ == __y.__current_;
  }
};

template <forward_range _View>
  requires view<_View>
class slide_view<_View>::__sentinel {
  sentinel_t<_View> __end_ = sentinel_t<_View>();
  constexpr explicit __sentinel(sentinel_t<_View> __end) : __end_(__end) {}

public:
  __sentinel() = default;

  friend constexpr bool operator==(const slide_view::__iterator<false>& __x, const __sentinel& __y) {
    return __x.__last_ == __y.__end_;
  };
  friend constexpr range_difference_t<_View> operator-(const slide_view::__iterator<false>& __x, const __sentinel& __y)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __x.__last_ - __y.__end_;
  };

  friend constexpr range_difference_t<_View> operator-(const __sentinel& __y, const slide_view::__iterator<false>& __x)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __y.__end_ - __x.__last_;
  }
};

namespace views {
namespace __slide {
struct __fn {
  template <viewable_range _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, range_difference_t<_Range> __n) const
      noexcept(noexcept(slide_view{std::forward<_Range>(__range), __n}))
          -> decltype(slide_view{std::forward<_Range>(__range), __n}) {
    return slide_view{std::forward<_Range>(__range), __n};
  }

  template <class _Np>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Np&& __n) const {
    return __range_adaptor_closure_t(std::__bind_back(*this, std::forward<_Np>(__n)));
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

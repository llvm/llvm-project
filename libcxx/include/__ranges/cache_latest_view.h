// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_TO_CACHE_LATEST_VIEW_H
#define _LIBCPP___RANGES_TO_CACHE_LATEST_VIEW_H

#include <__concepts/constructible.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__type_traits/add_pointer.h>
#include <__type_traits/conditional.h>
#include <__type_traits/is_reference.h>
#include <__utility/as_lvalue.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

namespace ranges {

// [range.cache.latest.view]

template <input_range _View>
  requires view<_View>
class cache_latest_view : public view_interface<cache_latest_view<_View>> {
  _View __base_ = _View(); // exposition only
  using __cache_t _LIBCPP_NODEBUG =
      conditional_t<is_reference_v<range_reference_t<_View>>, // exposition only
                    add_pointer_t<range_reference_t<_View>>,
                    range_reference_t<_View>>;

  __non_propagating_cache<__cache_t> __cache_; // exposition only

  // [range.cache.latest.iterator], class cache_latest_view::iterator
  class iterator; // exposition only
  // [range.cache.latest.sentinel], class cache_latest_view::sentinel
  class sentinel; // exposition only

public:
  _LIBCPP_HIDE_FROM_ABI cache_latest_view()
    requires default_initializable<_View>
  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit cache_latest_view(_View __base) : __base_{std::move(__base)} {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() { return iterator(*this); }
  _LIBCPP_HIDE_FROM_ABI constexpr auto end() { return sentinel{*this}; }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return ranges::size(__base_);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return ranges::size(__base_);
  }

  // TODO: Implement when P2846R6 is available.
  // constexpr auto reserve_hint()
  //   requires approximately_sized_range<_View>
  // {
  //   return ranges::reserve_hint(__base_);
  // }
  // constexpr auto reserve_hint() const
  //   requires approximately_sized_range<const _View>
  // {
  //   return ranges::reserve_hint(__base_);
  // }
};

template <class _Range>
cache_latest_view(_Range&&) -> cache_latest_view<views::all_t<_Range>>;

// [range.cache.latest.iterator]

template <input_range _View>
  requires view<_View>
class cache_latest_view<_View>::iterator {
  cache_latest_view* __parent_; // exposition only
  iterator_t<_View> __current_; // exposition only

  _LIBCPP_HIDE_FROM_ABI constexpr explicit iterator(cache_latest_view& __parent) // exposition only
      : __parent_{std::addressof(__parent)}, __current_{ranges::begin(__parent.__base_)} {}

  friend class cache_latest_view<_View>;

public:
  using difference_type  = range_difference_t<_View>;
  using value_type       = range_value_t<_View>;
  using iterator_concept = input_iterator_tag;

  _LIBCPP_HIDE_FROM_ABI iterator(iterator&&)            = default;
  _LIBCPP_HIDE_FROM_ABI iterator& operator=(iterator&&) = default;

  _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_View> base() && { return std::move(__current_); }
  _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_View>& base() const& noexcept { return __current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr range_reference_t<_View>& operator*() const {
    if constexpr (is_reference_v<range_reference_t<_View>>) {
      if (!__parent_->__cache_.__has_value()) {
        __parent_->__cache_.__emplace(std::addressof(std::__as_lvalue(*__current_)));
      }
      return **__parent_->__cache_;
    } else {
      if (!__parent_->__cache_.__has_value()) {
        __parent_->__cache_.__emplace_from([&]() -> decltype(auto) { return *__current_; });
      }
      return *__parent_->__cache_;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr iterator& operator++() {
    __parent_->__cache_.__reset();
    ++__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  _LIBCPP_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_View>
  iter_move(const iterator& __i) noexcept(noexcept(ranges::iter_move(__i.__current_))) {
    return ranges::iter_move(__i.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(const iterator& __x,
            const iterator& __y) noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
    requires indirectly_swappable<iterator_t<_View>>
  {
    ranges::iter_swap(__x.__current_, __y.__current_);
  }
};

// [range.cache.latest.sentinel]

template <input_range _View>
  requires view<_View>
class cache_latest_view<_View>::sentinel {
  sentinel_t<_View> __end_ = sentinel_t<_View>(); // exposition only

  _LIBCPP_HIDE_FROM_ABI constexpr explicit sentinel(cache_latest_view& __parent) // exposition only
      : __end_{ranges::end(__parent.__base_)} {}

  friend class cache_latest_view<_View>;

public:
  _LIBCPP_HIDE_FROM_ABI sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr sentinel_t<_View> base() const { return __end_; }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const iterator& __x, const sentinel& __y) {
    return __x.__current_ == __y.__end_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<_View> operator-(const iterator& __x, const sentinel& __y)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __x.__current_ - __y.__end_;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<_View> operator-(const sentinel& __x, const iterator& __y)
    requires sized_sentinel_for<sentinel_t<_View>, iterator_t<_View>>
  {
    return __x.__end_ - __y.__current_;
  }
};

namespace views {
namespace __cache_latest_view {

struct __fn : __range_adaptor_closure<__fn> {
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(/**/ cache_latest_view(std::forward<_Range>(__range))))
      -> decltype(/*-------------------------------*/ cache_latest_view(std::forward<_Range>(__range))) {
    return /*--------------------------------------*/ cache_latest_view(std::forward<_Range>(__range));
  }
};

} // namespace __cache_latest_view

inline namespace __cpo {
inline constexpr auto cache_latest = __cache_latest_view::__fn{};
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_TO_CACHE_LATEST_VIEW_H

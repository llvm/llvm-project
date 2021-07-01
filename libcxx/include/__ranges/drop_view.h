// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___RANGES_DROP_VIEW_H
#define _LIBCPP___RANGES_DROP_VIEW_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/concepts.h>
#include <__ranges/access.h>
#include <__ranges/view_interface.h>
#include <__ranges/all.h>
#include <type_traits>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {
  template<view _View>
  class drop_view
    : public view_interface<drop_view<_View>> {

    // We cache begin() whenever ranges::next is not guaranteed O(1) to provide an
    // amortized O(1) begin() method. If this is an input_range, then we cannot cache
    // begin because begin is not equality preserving.
    // Note: drop_view<input-range>::begin() is still trivially amortized O(1) because
    // one can't call begin() on it more than once.
    static constexpr bool _UseCache = forward_range<_View> && !(random_access_range<_View> && sized_range<_View>);
    using _Cache = optional<iterator_t<_View>>;
    struct _Empty { };

    // For forward ranges use std::optional to cache the begin iterator.
    // No unique address + _Empty means we don't use any extra space when this
    // is not a forward iterator.
    [[no_unique_address]] conditional_t<_UseCache, _Cache, _Empty> __cached_begin_;
    range_difference_t<_View> __count_ = 0;
    _View __base_ = _View();

public:
    drop_view() requires default_initializable<_View> = default;

    constexpr drop_view(_View __base, range_difference_t<_View> __count)
      : __cached_begin_()
      , __count_(__count)
      , __base_(_VSTD::move(__base))
    {
      _LIBCPP_ASSERT(__count_ >= 0, "count must be greater than or equal to zero.");
    }

    constexpr drop_view(drop_view const& __other)
      : __cached_begin_() // Intentionally not propagating the cached begin iterator.
      , __count_(__other.__count_)
      , __base_(__other.__base_)
    { }

    constexpr drop_view(drop_view&& __other)
      : __cached_begin_() // Intentionally not propagating the cached begin iterator.
      , __count_(_VSTD::move(__other.__count_))
      , __base_(_VSTD::move(__other.__base_))
    { }

    constexpr drop_view& operator=(drop_view const& __other) {
      if constexpr (_UseCache) {
        __cached_begin_.reset();
      }
      __base_ = __other.__base_;
      __count_ = __other.__count_;
      return *this;
    }

    constexpr drop_view& operator=(drop_view&& __other) {
      if constexpr (_UseCache) {
        __cached_begin_.reset();
        __other.__cached_begin_.reset();
      }
      __base_ = _VSTD::move(__other.__base_);
      __count_ = _VSTD::move(__other.__count_);
      return *this;
    }

    constexpr _View base() const& requires copy_constructible<_View> { return __base_; }
    constexpr _View base() && { return _VSTD::move(__base_); }

    constexpr auto begin()
      requires (!(__simple_view<_View> &&
                  random_access_range<const _View> && sized_range<const _View>))
    {
      if constexpr (_UseCache)
        if (__cached_begin_)
          return *__cached_begin_;

      auto __tmp = ranges::next(ranges::begin(__base_), __count_, ranges::end(__base_));
      if constexpr (_UseCache)
        __cached_begin_ = __tmp;
      return __tmp;
    }

    constexpr auto begin() const
      requires random_access_range<const _View> && sized_range<const _View>
    {
      return ranges::next(ranges::begin(__base_), __count_, ranges::end(__base_));
    }

    constexpr auto end()
      requires (!__simple_view<_View>)
    { return ranges::end(__base_); }

    constexpr auto end() const
      requires range<const _View>
    { return ranges::end(__base_); }

    static constexpr auto __size(auto& __self) {
      const auto __s = ranges::size(__self.__base_);
      const auto __c = static_cast<decltype(__s)>(__self.__count_);
      return __s < __c ? 0 : __s - __c;
    }

    constexpr auto size()
      requires sized_range<_View>
    { return __size(*this); }

    constexpr auto size() const
      requires sized_range<const _View>
    { return __size(*this); }
  };

  template<class _Range>
  drop_view(_Range&&, range_difference_t<_Range>)
  // TODO: this is just recreating all_t.
    -> drop_view<decltype(views::all(std::declval<_Range>()))>;

  template<class _Tp>
  inline constexpr bool enable_borrowed_range<drop_view<_Tp>> = enable_borrowed_range<_Tp>;

} // namespace ranges

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_DROP_VIEW_H

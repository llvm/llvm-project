// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCPP___RANGES_REVERSE_VIEW_H
#define _LIBCPP___RANGES_REVERSE_VIEW_H

#include <__concepts/constructible.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/next.h>
#include <__iterator/reverse_iterator.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__utility/move.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_RANGES)

namespace ranges {
  template<view _View>
    requires bidirectional_range<_View>
  class reverse_view : public view_interface<reverse_view<_View>> {
    // We cache begin() whenever ranges::next is not guaranteed O(1) to provide an
    // amortized O(1) begin() method.
    static constexpr bool _UseCache = !random_access_range<_View> && !common_range<_View>;
    using _Cache = _If<_UseCache, __non_propagating_cache<reverse_iterator<iterator_t<_View>>>, __empty_cache>;
    [[no_unique_address]] _Cache __cached_begin_ = _Cache();
    [[no_unique_address]] _View __base_ = _View();

  public:
    _LIBCPP_HIDE_FROM_ABI
    reverse_view() requires default_initializable<_View> = default;

    _LIBCPP_HIDE_FROM_ABI
    constexpr explicit reverse_view(_View __view) : __base_(_VSTD::move(__view)) {}

    _LIBCPP_HIDE_FROM_ABI
    constexpr _View base() const& requires copy_constructible<_View> { return __base_; }

    _LIBCPP_HIDE_FROM_ABI
    constexpr _View base() && { return _VSTD::move(__base_); }

    _LIBCPP_HIDE_FROM_ABI
    constexpr reverse_iterator<iterator_t<_View>> begin() {
      if constexpr (_UseCache)
        if (__cached_begin_.__has_value())
          return *__cached_begin_;

      auto __tmp = _VSTD::make_reverse_iterator(ranges::next(ranges::begin(__base_), ranges::end(__base_)));
      if constexpr (_UseCache)
        __cached_begin_.__set(__tmp);
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr reverse_iterator<iterator_t<_View>> begin() requires common_range<_View> {
      return _VSTD::make_reverse_iterator(ranges::end(__base_));
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr auto begin() const requires common_range<const _View> {
      return _VSTD::make_reverse_iterator(ranges::end(__base_));
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr reverse_iterator<iterator_t<_View>> end() {
      return _VSTD::make_reverse_iterator(ranges::begin(__base_));
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr auto end() const requires common_range<const _View> {
      return _VSTD::make_reverse_iterator(ranges::begin(__base_));
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr auto size() requires sized_range<_View> {
      return ranges::size(__base_);
    }

    _LIBCPP_HIDE_FROM_ABI
    constexpr auto size() const requires sized_range<const _View> {
      return ranges::size(__base_);
    }
  };

  template<class _Range>
  reverse_view(_Range&&) -> reverse_view<views::all_t<_Range>>;

  template<class _Tp>
  inline constexpr bool enable_borrowed_range<reverse_view<_Tp>> = enable_borrowed_range<_Tp>;
} // namespace ranges

#endif // !defined(_LIBCPP_HAS_NO_RANGES)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_REVERSE_VIEW_H

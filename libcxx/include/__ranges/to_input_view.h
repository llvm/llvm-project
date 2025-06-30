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

#include <__concepts/constructible.h>
#include <__config>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/view_interface.h>

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
  _LIBCPP_HIDE_FROM_ABI constexpr _V base() const&
    requires copy_constructible<_V>
  {
    return __base_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _V base() && { return std::move(__base_); }

  // begin
  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_V>)
  {
    return __iterator<false>(ranges::begin(__base_));
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _V>
  {
    return __iterator<true>(ranges::begin(__base_));
  }

  // end
  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_V>)
  {
    return ranges::end(__base_);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _V>
  {
    return ranges::end(__base_);
  }

  // size
  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_V>
  {
    return ranges::size(__base_);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _V>
  {
    return ranges::size(__base_);
  }
};

template <class _R>
to_input_view(_R&&) -> to_input_view<ranges::views::all_t<_R>>;

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_FROM_RANGE_H

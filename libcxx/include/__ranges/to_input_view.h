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
#include <__concepts/convertible_to.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

namespace ranges {

// [range.to.input.view

template <input_range _View>
  requires view<_View>
class to_input_view : public view_interface<to_input_view<_View>> {
  _View __base_ = _View(); // exposition only

  // [range.to.input.iterator], class template to_input_view::iterator
  template <bool _Const>
  class iterator; // exposition only

public:
  _LIBCPP_HIDE_FROM_ABI to_input_view()
    requires default_initializable<_View>
  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit to_input_view(_View __base) : __base_(std::move(__base)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return iterator<false>{ranges::begin(__base_)};
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _View>
  {
    return iterator<true>{ranges::begin(__base_)};
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    return ranges::end(__base_);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _View>
  {
    return ranges::end(__base_);
  }
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
to_input_view(_Range&&) -> to_input_view<views::all_t<_Range>>;

// [range.to.input.iterator]

template <input_range _View>
  requires view<_View>
template <bool _Const>
class to_input_view<_View>::iterator {
  using _Base _LIBCPP_NODEBUG = __maybe_const<_Const, _View>; // exposition only

  iterator_t<_Base> __current_ = iterator_t<_Base>(); // exposition only

  _LIBCPP_HIDE_FROM_ABI constexpr explicit iterator(iterator_t<_Base> __current)
      : __current_(std::move(__current)) {} // exposition only

  friend class to_input_view<_View>;

public:
  using difference_type  = range_difference_t<_Base>;
  using value_type       = range_value_t<_Base>;
  using iterator_concept = input_iterator_tag;

  _LIBCPP_HIDE_FROM_ABI iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;

  _LIBCPP_HIDE_FROM_ABI iterator(iterator&&)            = default;
  _LIBCPP_HIDE_FROM_ABI iterator& operator=(iterator&&) = default;

  _LIBCPP_HIDE_FROM_ABI constexpr iterator(iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<_Base>>
      : __current_(std::move(__i.__current_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() && { return std::move(__current_); }
  _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_Base>& base() const& noexcept { return __current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const { return *__current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr iterator& operator++() {
    ++__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const iterator& __x, const sentinel_t<_Base>& __y) {
    return __x.__current_ == __y;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const sentinel_t<_Base>& __y, const iterator& __x)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __y - __x.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const iterator& __x, const sentinel_t<_Base>& __y)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __x.__current_ - __y;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_Base> _LIBCPP_HIDE_FROM_ABI
  iter_move(const iterator& __i) noexcept(noexcept(ranges::iter_move(__i.__current_))) {
    return ranges::iter_move(__i.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(const iterator& __x,
            const iterator& __y) noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
    requires indirectly_swappable<iterator_t<_Base>>
  {
    ranges::iter_swap(__x.__current_, __y.__current_);
  }
};

template <class _View>
constexpr bool enable_borrowed_range<to_input_view<_View>> = enable_borrowed_range<_View>;

namespace views {
namespace __to_input_view {

struct __fn : __range_adaptor_closure<__fn> {
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(/**/ to_input_view(std::forward<_Range>(__range))))
      -> decltype(/*--*/ to_input_view(std::forward<_Range>(__range))) {
    return /*---------*/ to_input_view(std::forward<_Range>(__range));
  }

  template <class _Range>
    requires(!common_range<_Range> && !forward_range<_Range>)
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(views::all(std::forward<_Range>(__range))))
      -> decltype(/*--------------------------*/ views::all(std::forward<_Range>(__range))) {
    return /*---------------------------------*/ views::all(std::forward<_Range>(__range));
  }
};

} // namespace __to_input_view

inline namespace __cpo {
inline constexpr auto to_input = __to_input_view::__fn{};
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_TO_INPUT_VIEW_H

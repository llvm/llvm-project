// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ADJACENT_VIEW_H
#define _LIBCPP___RANGES_ADJACENT_VIEW_H

#include <__config>

#include <__algorithm/min.h>
#include <__compare/three_way_comparable.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/equality_comparable.h>
#include <__cstddef/size_t.h>
#include <__functional/invoke.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/prev.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/empty_view.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__tuple/tuple_transform.h>
#include <__type_traits/common_type.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/make_unsigned.h>
#include <__type_traits/maybe_const.h>
#include <__utility/declval.h>
#include <__utility/forward.h>
#include <__utility/integer_sequence.h>
#include <__utility/move.h>
#include <array>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <forward_range _View, size_t _Np>
  requires view<_View> && (_Np > 0)
class adjacent_view : public view_interface<adjacent_view<_View, _Np>> {
private:
  _LIBCPP_NO_UNIQUE_ADDRESS _View __base_ = _View();

  template <bool>
  class __iterator;

  template <bool>
  class __sentinel;

  struct __as_sentinel {};

public:
  _LIBCPP_HIDE_FROM_ABI adjacent_view()
    requires default_initializable<_View>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit adjacent_view(_View __base) : __base_(std::move(__base)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>(ranges::begin(__base_), ranges::end(__base_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _View> // LWG4482 This is under-constrained.
  {
    return __iterator<true>(ranges::begin(__base_), ranges::end(__base_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    if constexpr (common_range<_View>) {
      return __iterator<false>(__as_sentinel{}, ranges::begin(__base_), ranges::end(__base_));
    } else {
      return __sentinel<false>(ranges::end(__base_));
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _View> // LWG4482 This is under-constrained.
  {
    if constexpr (common_range<const _View>) {
      return __iterator<true>(__as_sentinel{}, ranges::begin(__base_), ranges::end(__base_));
    } else {
      return __sentinel<true>(ranges::end(__base_));
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    using _ST = decltype(ranges::size(__base_));
    using _CT = common_type_t<_ST, size_t>;
    auto __sz = static_cast<_CT>(ranges::size(__base_));
    __sz -= std::min<_CT>(__sz, _Np - 1);
    return static_cast<_ST>(__sz);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    using _ST = decltype(ranges::size(__base_));
    using _CT = common_type_t<_ST, size_t>;
    auto __sz = static_cast<_CT>(ranges::size(__base_));
    __sz -= std::min<_CT>(__sz, _Np - 1);
    return static_cast<_ST>(__sz);
  }
};

template <forward_range _View, size_t _Np>
  requires view<_View> && (_Np > 0)
template <bool _Const>
class adjacent_view<_View, _Np>::__iterator {
  friend adjacent_view;
  using _Base _LIBCPP_NODEBUG              = __maybe_const<_Const, _View>;
  array<iterator_t<_Base>, _Np> __current_ = array<iterator_t<_Base>, _Np>();

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(iterator_t<_Base> __first, sentinel_t<_Base> __last) {
    __current_[0] = __first;
    for (size_t __i = 1; __i < _Np; ++__i) {
      __current_[__i] = ranges::next(__current_[__i - 1], 1, __last);
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__as_sentinel, iterator_t<_Base> __first, iterator_t<_Base> __last) {
    if constexpr (!bidirectional_range<_Base>) {
      __current_.fill(__last);
    } else {
      __current_[_Np - 1] = __last;
      for (int __i = static_cast<int>(_Np) - 2; __i >= 0; --__i) {
        __current_[__i] = ranges::prev(__current_[__i + 1], 1, __first);
      }
    }
  }

  template <class _Iter, size_t... _Is>
  _LIBCPP_HIDE_FROM_ABI explicit constexpr __iterator(_Iter&& __i, index_sequence<_Is...>)
      : __current_{std::move(__i.__current_[_Is])...} {}

  static consteval auto __get_iterator_concept() {
    if constexpr (random_access_range<_Base>)
      return random_access_iterator_tag{};
    else if constexpr (bidirectional_range<_Base>)
      return bidirectional_iterator_tag{};
    else
      return forward_iterator_tag{};
  }

  template <class _Tp, size_t _Index>
  using __always _LIBCPP_NODEBUG = _Tp;

  template <class _Tp, size_t... _Is>
  static auto __repeat_tuple_helper(index_sequence<_Is...>) -> tuple<__always<_Tp, _Is>...>;

public:
  using iterator_category = input_iterator_tag;
  using iterator_concept  = decltype(__get_iterator_concept());
  using value_type        = decltype(__repeat_tuple_helper<range_value_t<_Base>>(make_index_sequence<_Np>{}));
  using difference_type   = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<iterator_t<_View>, iterator_t<const _View>>
      : __iterator(std::move(__i), make_index_sequence<_Np>{}) {}

  _LIBCPP_HIDE_FROM_ABI constexpr auto operator*() const {
    return std::__tuple_transform([](auto& __i) -> decltype(auto) { return *__i; }, __current_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    for (auto& __i : __current_) {
      ++__i;
    }
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
    for (auto& __i : __current_) {
      --__i;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __x)
    requires random_access_range<_Base>
  {
    for (auto& __i : __current_) {
      __i += __x;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __x)
    requires random_access_range<_Base>
  {
    for (auto& __i : __current_) {
      __i -= __x;
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto operator[](difference_type __n) const
    requires random_access_range<_Base>
  {
    return std::__tuple_transform([&](auto& __i) -> decltype(auto) { return __i[__n]; }, __current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
    return __x.__current_.back() == __y.__current_.back();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__current_.back() < __y.__current_.back();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __y < __x;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__y < __x);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__x < __y);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_.back() <=> __y.__current_.back();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, const __iterator& __i)
    requires random_access_range<_Base>
  {
    return __i + __n;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r -= __n;
    return __r;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>>
  {
    return __x.__current_.back() - __y.__current_.back();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto iter_move(const __iterator& __i) noexcept(
      noexcept(ranges::iter_move(std::declval<const iterator_t<_Base>&>())) &&
      is_nothrow_move_constructible_v<range_rvalue_reference_t<_Base>>) {
    return std::__tuple_transform(ranges::iter_move, __i.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void iter_swap(const __iterator& __l, const __iterator& __r) noexcept(
      noexcept(ranges::iter_swap(std::declval<iterator_t<_Base>>(), std::declval<iterator_t<_Base>>())))
    requires indirectly_swappable<iterator_t<_Base>>
  {
    for (size_t __i = 0; __i < _Np; ++__i) {
      ranges::iter_swap(__l.__current_[__i], __r.__current_[__i]);
    }
  }
};

template <forward_range _View, size_t _Np>
  requires view<_View> && (_Np > 0)
template <bool _Const>
class adjacent_view<_View, _Np>::__sentinel {
  friend adjacent_view;
  using _Base _LIBCPP_NODEBUG = __maybe_const<_Const, _View>;
  sentinel_t<_Base> __end_    = sentinel_t<_Base>();

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __sentinel(sentinel_t<_Base> __end) { __end_ = std::move(__end); }

public:
  _LIBCPP_HIDE_FROM_ABI __sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_Const> __i)
    requires _Const && convertible_to<sentinel_t<_View>, sentinel_t<_Base>>
      : __end_(std::move(__i.__end_)) {}

  template <bool _OtherConst>
    requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__current_.back() == __y.__end_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _View>>
  operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__current_.back() - __y.__end_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _View>>
  operator-(const __sentinel& __y, const __iterator<_OtherConst>& __x) {
    return __y.__end_ - __x.__current_.back();
  }
};

template <class _View, size_t _Np>
constexpr bool enable_borrowed_range<adjacent_view<_View, _Np>> = enable_borrowed_range<_View>;

namespace views {
namespace __adjacent {

template <size_t _Np>
struct __fn : __range_adaptor_closure<__fn<_Np>> {
  template <class _Range>
    requires(_Np == 0 && forward_range<_Range &&>)
  _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&&) noexcept {
    return empty_view<tuple<>>{};
  }

  template <class _Ranges>
  _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ranges&& __range) noexcept(
      noexcept(adjacent_view<views::all_t<_Ranges&&>, _Np>(std::forward<_Ranges>(__range))))
      -> decltype(adjacent_view<views::all_t<_Ranges&&>, _Np>(std::forward<_Ranges>(__range))) {
    return adjacent_view<views::all_t<_Ranges&&>, _Np>(std::forward<_Ranges>(__range));
  }
};

} // namespace __adjacent
inline namespace __cpo {
template <size_t _Np>
inline constexpr auto adjacent = __adjacent::__fn<_Np>{};
inline constexpr auto pairwise = adjacent<2>;
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ADJACENT_VIEW_H

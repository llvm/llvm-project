// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ADJACENT_TRANSFORM_VIEW_H
#define _LIBCPP___RANGES_ADJACENT_TRANSFORM_VIEW_H

#include <__config>

#include <__algorithm/min.h>
#include <__compare/three_way_comparable.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/derived_from.h>
#include <__concepts/equality_comparable.h>
#include <__concepts/invocable.h>
#include <__cstddef/size_t.h>
#include <__functional/bind_back.h>
#include <__functional/invoke.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/prev.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/adjacent_view.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/empty_view.h>
#include <__ranges/movable_box.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__ranges/zip_transform_view.h>
#include <__type_traits/common_type.h>
#include <__type_traits/decay.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_referenceable.h>
#include <__type_traits/make_unsigned.h>
#include <__type_traits/maybe_const.h>
#include <__utility/declval.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/integer_sequence.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <class _Fn, size_t _Np>
struct __apply_n {
  template <class _Tp, size_t... _Is>
  static auto __apply(index_sequence<_Is...>) -> invoke_result_t<_Fn, decltype((void)_Is, std::declval<_Tp>())...>;

  template <class _Tp>
  static auto operator()(_Tp&&) -> decltype(__apply<_Tp>(make_index_sequence<_Np>{}));
};

template <forward_range _View, move_constructible _Fn, size_t _Np>
  requires view<_View> && (_Np > 0) && is_object_v<_Fn> &&
           regular_invocable<__apply_n<_Fn&, _Np>, range_reference_t<_View>> &&
           __referenceable<invoke_result_t<__apply_n<_Fn&, _Np>, range_reference_t<_View>>>
class adjacent_transform_view : public view_interface<adjacent_transform_view<_View, _Fn, _Np>> {
private:
  _LIBCPP_NO_UNIQUE_ADDRESS adjacent_view<_View, _Np> __inner_;
  _LIBCPP_NO_UNIQUE_ADDRESS __movable_box<_Fn> __fun_;

  using _InnerView _LIBCPP_NODEBUG = adjacent_view<_View, _Np>;

  template <bool _Const>
  using __inner_iterator _LIBCPP_NODEBUG = iterator_t<__maybe_const<_Const, _InnerView>>;

  template <bool _Const>
  using __inner_sentinel _LIBCPP_NODEBUG = sentinel_t<__maybe_const<_Const, _InnerView>>;

  template <bool>
  class __iterator;

  template <bool>
  class __sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI adjacent_transform_view() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit adjacent_transform_view(_View __base, _Fn __fun)
      : __inner_(std::move(__base)), __fun_(std::in_place, std::move(__fun)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __inner_.base();
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__inner_).base(); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() { return __iterator<false>(*this, __inner_.begin()); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _InnerView> && regular_invocable<__apply_n<const _Fn&, _Np>, range_reference_t<const _View>>
  {
    return __iterator<true>(*this, __inner_.begin());
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() {
    if constexpr (common_range<_InnerView>) {
      return __iterator<false>(*this, __inner_.end());
    } else {
      return __sentinel<false>(__inner_.end());
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _InnerView> && regular_invocable<__apply_n<const _Fn&, _Np>, range_reference_t<const _View>>
  {
    if constexpr (common_range<const _InnerView>) {
      return __iterator<true>(*this, __inner_.end());
    } else {
      return __sentinel<true>(__inner_.end());
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_InnerView>
  {
    return __inner_.size();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _InnerView>
  {
    return __inner_.size();
  }
};

template <forward_range _View, move_constructible _Fn, size_t _Np>
  requires view<_View> && (_Np > 0) && is_object_v<_Fn> &&
           regular_invocable<__apply_n<_Fn&, _Np>, range_reference_t<_View>> &&
           __referenceable<invoke_result_t<__apply_n<_Fn&, _Np>, range_reference_t<_View>>>
template <bool _Const>
class adjacent_transform_view<_View, _Fn, _Np>::__iterator {
  friend adjacent_transform_view;

  using _Parent _LIBCPP_NODEBUG = __maybe_const<_Const, adjacent_transform_view>;
  using _Base _LIBCPP_NODEBUG   = __maybe_const<_Const, _View>;

  _Parent* __parent_ = nullptr;
  __inner_iterator<_Const> __inner_;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(_Parent& __parent, __inner_iterator<_Const> __inner)
      : __parent_(std::addressof(__parent)), __inner_(std::move(__inner)) {}

  static consteval auto __get_iterator_category() {
    using _Cat = iterator_traits<iterator_t<_Base>>::iterator_category;
    if constexpr (!is_reference_v<
                      invoke_result_t<__apply_n<__maybe_const<_Const, _Fn>&, _Np>, range_reference_t<_Base>>>)
      return input_iterator_tag{};
    else if constexpr (derived_from<_Cat, random_access_iterator_tag>)
      return random_access_iterator_tag{};
    else if constexpr (derived_from<_Cat, bidirectional_iterator_tag>)
      return bidirectional_iterator_tag{};
    else if constexpr (derived_from<_Cat, forward_iterator_tag>)
      return forward_iterator_tag{};
    else
      return input_iterator_tag{};
  }

  template <size_t... _Is>
  static consteval bool __noexcept_dereference(index_sequence<_Is...>) {
    return noexcept(std::invoke(
        std::declval<__maybe_const<_Const, _Fn>&>(), ((void)_Is, *std::declval<iterator_t<_Base> const&>())...));
  }

public:
  using iterator_category = decltype(__get_iterator_category());
  using iterator_concept  = typename __inner_iterator<_Const>::iterator_concept;
  using value_type =
      remove_cvref_t<invoke_result_t<__apply_n<__maybe_const<_Const, _Fn>&, _Np>, range_reference_t<_Base>>>;
  using difference_type = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<__inner_iterator<false>, __inner_iterator<true>>
      : __parent_(__i.__parent_), __inner_(std::move(__i.__inner_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
      noexcept(__noexcept_dereference(make_index_sequence<_Np>{})) {
    return std::apply(
        [&](const auto&... __iters) -> decltype(auto) { return std::invoke(*__parent_->__fun_, *__iters...); },
        __adjacent_view_iter_access::__get_current(__inner_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    ++__inner_;
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
    --__inner_;
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
    __inner_ += __x;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __x)
    requires random_access_range<_Base>
  {
    __inner_ -= __x;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
    requires random_access_range<_Base>
  {
    return std::apply(
        [&](const auto&... __iters) -> decltype(auto) { return std::invoke(*__parent_->__fun_, __iters[__n]...); },
        __adjacent_view_iter_access::__get_current(__inner_));
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) {
    return __x.__inner_ == __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__inner_ < __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__inner_ > __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__inner_ <= __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__inner_ >= __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<__inner_iterator<_Const>>
  {
    return __x.__inner_ <=> __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    return __iterator(*__i.__parent_, __i.__inner_ + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, const __iterator& __i)
    requires random_access_range<_Base>
  {
    return __iterator(*__i.__parent_, __i.__inner_ + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    return __iterator(*__i.__parent_, __i.__inner_ - __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    requires sized_sentinel_for<__inner_iterator<_Const>, __inner_iterator<_Const>>
  {
    return __x.__inner_ - __y.__inner_;
  }
};

template <forward_range _View, move_constructible _Fn, size_t _Np>
  requires view<_View> && (_Np > 0) && is_object_v<_Fn> &&
           regular_invocable<__apply_n<_Fn&, _Np>, range_reference_t<_View>> &&
           __referenceable<invoke_result_t<__apply_n<_Fn&, _Np>, range_reference_t<_View>>>
template <bool _Const>
class adjacent_transform_view<_View, _Fn, _Np>::__sentinel {
  friend adjacent_transform_view;

  __inner_sentinel<_Const> __inner_;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __sentinel(__inner_sentinel<_Const> __inner)
      : __inner_(std::move(__inner)) {}

public:
  _LIBCPP_HIDE_FROM_ABI __sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_Const> __i)
    requires _Const && convertible_to<__inner_sentinel<false>, __inner_sentinel<_Const>>
      : __inner_(std::move(__i.__inner_)) {}

  template <bool _OtherConst>
    requires sentinel_for<__inner_sentinel<_Const>, __inner_iterator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__inner_ == __y.__inner_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<__inner_sentinel<_Const>, __inner_iterator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
  operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__inner_ - __y.__inner_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<__inner_sentinel<_Const>, __inner_iterator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
  operator-(const __sentinel& __x, const __iterator<_OtherConst>& __y) {
    return __x.__inner_ - __y.__inner_;
  }
};

namespace views {
namespace __adjacent_transform {

template <size_t _Np>
struct __fn : __range_adaptor_closure<__fn<_Np>> {
  template <class _Range, class _Fn>
    requires(_Np == 0 && forward_range<_Range &&>)
  _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&&, _Fn&& __fn) noexcept(noexcept(views::zip_transform(std::forward<_Fn>(__fn))))
      -> decltype(views::zip_transform(std::forward<_Fn>(__fn))) {
    return views::zip_transform(std::forward<_Fn>(__fn));
  }

  template <class _Range, class _Fn>
  _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Range&& __range, _Fn&& __fn) noexcept(
      noexcept(adjacent_transform_view<views::all_t<_Range&&>, decay_t<_Fn>, _Np>(
          std::forward<_Range>(__range), std::forward<_Fn>(__fn))))
      -> decltype(adjacent_transform_view<views::all_t<_Range&&>, decay_t<_Fn>, _Np>(
          std::forward<_Range>(__range), std::forward<_Fn>(__fn))) {
    return adjacent_transform_view<views::all_t<_Range&&>, decay_t<_Fn>, _Np>(
        std::forward<_Range>(__range), std::forward<_Fn>(__fn));
  }

  template <class _Fn>
    requires constructible_from<decay_t<_Fn>, _Fn>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Fn&& __f) const
      noexcept(is_nothrow_constructible_v<decay_t<_Fn>, _Fn>) {
    return __pipeable(std::__bind_back(*this, std::forward<_Fn>(__f)));
  }
};

} // namespace __adjacent_transform
inline namespace __cpo {
template <size_t _Np>
inline constexpr auto adjacent_transform = __adjacent_transform::__fn<_Np>{};
inline constexpr auto pairwise_transform = adjacent_transform<2>;
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ADJACENT_TRANSFORM_VIEW_H

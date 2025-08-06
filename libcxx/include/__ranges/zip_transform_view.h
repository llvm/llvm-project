// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_ZIP_TRANSFORM_VIEW_H
#define _LIBCPP___RANGES_ZIP_TRANSFORM_VIEW_H

#include <__config>

#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/derived_from.h>
#include <__concepts/equality_comparable.h>
#include <__concepts/invocable.h>
#include <__functional/invoke.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/empty_view.h>
#include <__ranges/movable_box.h>
#include <__ranges/view_interface.h>
#include <__ranges/zip_view.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_referenceable.h>
#include <__type_traits/maybe_const.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <tuple> // for std::apply

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>
_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

template <move_constructible _Fn, input_range... _Views>
  requires(view<_Views> && ...) &&
          (sizeof...(_Views) > 0) && is_object_v<_Fn> && regular_invocable<_Fn&, range_reference_t<_Views>...> &&
          __referenceable<invoke_result_t<_Fn&, range_reference_t<_Views>...>>
class zip_transform_view : public view_interface<zip_transform_view<_Fn, _Views...>> {
  _LIBCPP_NO_UNIQUE_ADDRESS zip_view<_Views...> __zip_;
  _LIBCPP_NO_UNIQUE_ADDRESS __movable_box<_Fn> __fun_;

  using _InnerView _LIBCPP_NODEBUG = zip_view<_Views...>;
  template <bool _Const>
  using __ziperator _LIBCPP_NODEBUG = iterator_t<__maybe_const<_Const, _InnerView>>;
  template <bool _Const>
  using __zentinel _LIBCPP_NODEBUG = sentinel_t<__maybe_const<_Const, _InnerView>>;

  template <bool>
  class __iterator;

  template <bool>
  class __sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI zip_transform_view() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit zip_transform_view(_Fn __fun, _Views... __views)
      : __zip_(std::move(__views)...), __fun_(in_place, std::move(__fun)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() { return __iterator<false>(*this, __zip_.begin()); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _InnerView> && regular_invocable<const _Fn&, range_reference_t<const _Views>...>
  {
    return __iterator<true>(*this, __zip_.begin());
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() {
    if constexpr (common_range<_InnerView>) {
      return __iterator<false>(*this, __zip_.end());
    } else {
      return __sentinel<false>(__zip_.end());
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _InnerView> && regular_invocable<const _Fn&, range_reference_t<const _Views>...>
  {
    if constexpr (common_range<const _InnerView>) {
      return __iterator<true>(*this, __zip_.end());
    } else {
      return __sentinel<true>(__zip_.end());
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_InnerView>
  {
    return __zip_.size();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _InnerView>
  {
    return __zip_.size();
  }
};

template <class _Fn, class... _Ranges>
zip_transform_view(_Fn, _Ranges&&...) -> zip_transform_view<_Fn, views::all_t<_Ranges>...>;

template <bool _Const, class _Fn, class... _Views>
struct __zip_transform_iterator_category_base {};

template <bool _Const, class _Fn, class... _Views>
  requires forward_range<__maybe_const<_Const, zip_view<_Views...>>>
struct __zip_transform_iterator_category_base<_Const, _Fn, _Views...> {
private:
  template <class _View>
  using __tag _LIBCPP_NODEBUG = typename iterator_traits<iterator_t<__maybe_const<_Const, _View>>>::iterator_category;

  static consteval auto __get_iterator_category() {
    if constexpr (!is_reference_v<invoke_result_t<__maybe_const<_Const, _Fn>&,
                                                  range_reference_t<__maybe_const<_Const, _Views>>...>>) {
      return input_iterator_tag();
    } else if constexpr ((derived_from<__tag<_Views>, random_access_iterator_tag> && ...)) {
      return random_access_iterator_tag();
    } else if constexpr ((derived_from<__tag<_Views>, bidirectional_iterator_tag> && ...)) {
      return bidirectional_iterator_tag();
    } else if constexpr ((derived_from<__tag<_Views>, forward_iterator_tag> && ...)) {
      return forward_iterator_tag();
    } else {
      return input_iterator_tag();
    }
  }

public:
  using iterator_category = decltype(__get_iterator_category());
};

template <move_constructible _Fn, input_range... _Views>
  requires(view<_Views> && ...) &&
          (sizeof...(_Views) > 0) && is_object_v<_Fn> && regular_invocable<_Fn&, range_reference_t<_Views>...> &&
          __referenceable<invoke_result_t<_Fn&, range_reference_t<_Views>...>>
template <bool _Const>
class zip_transform_view<_Fn, _Views...>::__iterator
    : public __zip_transform_iterator_category_base<_Const, _Fn, _Views...> {
  using _Parent _LIBCPP_NODEBUG = __maybe_const<_Const, zip_transform_view>;
  using _Base _LIBCPP_NODEBUG   = __maybe_const<_Const, _InnerView>;

  friend zip_transform_view<_Fn, _Views...>;

  _Parent* __parent_ = nullptr;
  __ziperator<_Const> __inner_;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(_Parent& __parent, __ziperator<_Const> __inner)
      : __parent_(std::addressof(__parent)), __inner_(std::move(__inner)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr auto __get_deref_and_invoke() const noexcept {
    return [&__fun = *__parent_->__fun_](const auto&... __iters) noexcept(noexcept(std::invoke(
               *__parent_->__fun_, *__iters...))) -> decltype(auto) { return std::invoke(__fun, *__iters...); };
  }

public:
  using iterator_concept = typename __ziperator<_Const>::iterator_concept;
  using value_type =
      remove_cvref_t<invoke_result_t<__maybe_const<_Const, _Fn>&, range_reference_t<__maybe_const<_Const, _Views>>...>>;
  using difference_type = range_difference_t<_Base>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && convertible_to<__ziperator<false>, __ziperator<_Const>>
      : __parent_(__i.__parent_), __inner_(std::move(__i.__inner_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
      noexcept(noexcept(std::apply(__get_deref_and_invoke(), __zip_view_iterator_access::__get_underlying(__inner_)))) {
    return std::apply(__get_deref_and_invoke(), __zip_view_iterator_access::__get_underlying(__inner_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    ++__inner_;
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
        [&]<class... _Is>(const _Is&... __iters) -> decltype(auto) {
          return std::invoke(*__parent_->__fun_, __iters[iter_difference_t<_Is>(__n)]...);
        },
        __zip_view_iterator_access::__get_underlying(__inner_));
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    requires equality_comparable<__ziperator<_Const>>
  {
    return __x.__inner_ == __y.__inner_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
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
    requires sized_sentinel_for<__ziperator<_Const>, __ziperator<_Const>>
  {
    return __x.__inner_ - __y.__inner_;
  }
};

template <move_constructible _Fn, input_range... _Views>
  requires(view<_Views> && ...) &&
          (sizeof...(_Views) > 0) && is_object_v<_Fn> && regular_invocable<_Fn&, range_reference_t<_Views>...> &&
          __referenceable<invoke_result_t<_Fn&, range_reference_t<_Views>...>>
template <bool _Const>
class zip_transform_view<_Fn, _Views...>::__sentinel {
  __zentinel<_Const> __inner_;

  friend zip_transform_view<_Fn, _Views...>;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __sentinel(__zentinel<_Const> __inner) : __inner_(__inner) {}

public:
  _LIBCPP_HIDE_FROM_ABI __sentinel() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_Const> __i)
    requires _Const && convertible_to<__zentinel<false>, __zentinel<_Const>>
      : __inner_(__i.__inner_) {}

  template <bool _OtherConst>
    requires sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__inner_ == __y.__inner_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
  operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y) {
    return __x.__inner_ - __y.__inner_;
  }

  template <bool _OtherConst>
    requires sized_sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>
  _LIBCPP_HIDE_FROM_ABI friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
  operator-(const __sentinel& __x, const __iterator<_OtherConst>& __y) {
    return __x.__inner_ - __y.__inner_;
  }
};

namespace views {
namespace __zip_transform {

struct __fn {
  template <class _Fn>
    requires(move_constructible<decay_t<_Fn>> && regular_invocable<decay_t<_Fn>&> &&
             is_object_v<invoke_result_t<decay_t<_Fn>&>>)
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Fn&&) const
      noexcept(noexcept(auto(views::empty<decay_t<invoke_result_t<decay_t<_Fn>&>>>))) {
    return views::empty<decay_t<invoke_result_t<decay_t<_Fn>&>>>;
  }

  template <class _Fn, class... _Ranges>
    requires(sizeof...(_Ranges) > 0)
  _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Fn&& __fun, _Ranges&&... __rs) const
      noexcept(noexcept(zip_transform_view(std::forward<_Fn>(__fun), std::forward<_Ranges>(__rs)...)))
          -> decltype(zip_transform_view(std::forward<_Fn>(__fun), std::forward<_Ranges>(__rs)...)) {
    return zip_transform_view(std::forward<_Fn>(__fun), std::forward<_Ranges>(__rs)...);
  }
};

} // namespace __zip_transform
inline namespace __cpo {
inline constexpr auto zip_transform = __zip_transform::__fn{};
} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_ZIP_TRANSFORM_VIEW_H

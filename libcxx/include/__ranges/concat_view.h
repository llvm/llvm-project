// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_CONCAT_VIEW_H
#define _LIBCPP___RANGES_CONCAT_VIEW_H

#include <__assert>
#include <__concepts/common_reference_with.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/copyable.h>
#include <__concepts/derived_from.h>
#include <__concepts/equality_comparable.h>
#include <__concepts/swappable.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/movable_box.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__tuple/tuple_transform.h>
#include <__type_traits/conditional.h>
#include <__type_traits/decay.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/make_unsigned.h>
#include <__type_traits/maybe_const.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <tuple>
#include <variant>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

namespace ranges {

#  ifdef __cpp_pack_indexing
template <class... _Tp>
using __extract_last _LIBCPP_NODEBUG = _Tp...[sizeof...(_Tp) - 1];
#  else
template <class _Tp, class... _Tail>
struct __extract_last_impl : __extract_last_impl<_Tail...> {};
template <class _Tp>
struct __extract_last_impl<_Tp> {
  using type _LIBCPP_NODEBUG = _Tp;
};

template <class... _Tp>
using __extract_last _LIBCPP_NODEBUG = __extract_last_impl<_Tp...>::type;
#  endif

template <bool _Const, class... _Tp>
struct __all_but_first_model_sized_range;

template <bool _Const, class _Head, class... _Tail>
struct __all_but_first_model_sized_range<_Const, _Head, _Tail...> {
  static constexpr bool value = (sized_range<__maybe_const<_Const, _Tail>> && ...);
};

template <bool _Const, class... _Views>
concept __all_random_access = (random_access_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class... _Views>
concept __all_bidirectional = (bidirectional_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class... _Views>
concept __all_forward = (forward_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class _First, class... _Tail>
struct __all_common_ignore_last {
  static constexpr bool value =
      common_range<__maybe_const<_Const, _First>> && __all_common_ignore_last<_Const, _Tail...>::value;
};

template <bool _Const, class _Tail>
struct __all_common_ignore_last<_Const, _Tail> {
  static constexpr bool value = true;
};

template <bool _Const, class... _Rs>
concept __concat_is_random_access =
    (__all_random_access<_Const, _Rs...>) && (__all_common_ignore_last<_Const, _Rs...>::value);

template <bool _Const, class... _Rs>
concept __concat_is_bidirectional =
    (__all_bidirectional<_Const, _Rs...>) && (__all_common_ignore_last<_Const, _Rs...>::value);

template <input_range... _Views>
  requires((view<_Views> && ...) && (sizeof...(_Views) > 0) && __concatable<_Views...>)
class concat_view : public view_interface<concat_view<_Views...>> {
  tuple<_Views...> __views_;

  template <bool _Const>
  class __iterator;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr concat_view() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit concat_view(_Views... __views) : __views_(std::move(__views)...) {}

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator<false> begin()
    requires(!(__simple_view<_Views> && ...))
  {
    __iterator<false> __it(this, in_place_index<0>, ranges::begin(std::get<0>(__views_)));
    __it.template __satisfy<0>();
    return __it;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator<true> begin() const
    requires((range<const _Views> && ...) && __concatable<const _Views...>)
  {
    __iterator<true> __it(this, in_place_index<0>, ranges::begin(std::get<0>(__views_)));
    __it.template __satisfy<0>();
    return __it;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!(__simple_view<_Views> && ...))
  {
    if constexpr (__all_forward<false, _Views...> && common_range<__maybe_const<false, __extract_last<_Views...>>>) {
      constexpr auto __n = sizeof...(_Views);
      return __iterator<false>(this, in_place_index<__n - 1>, ranges::end(std::get<__n - 1>(__views_)));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires((range<const _Views> && ...) && __concatable<const _Views...>)
  {
    if constexpr (__all_forward<true, _Views...> && common_range<__maybe_const<true, __extract_last<_Views...>>>) {
      constexpr auto __n = sizeof...(_Views);
      return __iterator<true>(this, in_place_index<__n - 1>, ranges::end(std::get<__n - 1>(__views_)));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires(sized_range<_Views> && ...)
  {
    return std::apply(
        [](auto... __sizes) { return (make_unsigned_t<common_type_t<decltype(__sizes)...>>(__sizes) + ...); },
        std::__tuple_transform(ranges::size, __views_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires(sized_range<const _Views> && ...)
  {
    return std::apply(
        [](auto... __sizes) { return (make_unsigned_t<common_type_t<decltype(__sizes)...>>(__sizes) + ...); },
        std::__tuple_transform(ranges::size, __views_));
  }
};

template <class... _Views>
concat_view(_Views&&...) -> concat_view<views::all_t<_Views>...>;

template <bool _Const, typename... _Views>
struct __concat_view_iterator_category {};

template <bool _Const, typename... _Views>
  requires __all_forward<_Const, _Views...>
struct __concat_view_iterator_category<_Const, _Views...> {
private:
  constexpr static bool __derive_pack_random_iterator =
      (derived_from<typename iterator_traits<iterator_t<__maybe_const<_Const, _Views>>>::iterator_category,
                    random_access_iterator_tag> &&
       ...);
  constexpr static bool __derive_pack_bidirectional_iterator =
      (derived_from<typename iterator_traits<iterator_t<__maybe_const<_Const, _Views>>>::iterator_category,
                    bidirectional_iterator_tag> &&
       ...);
  constexpr static bool __derive_pack_forward_iterator =
      (derived_from<typename iterator_traits< iterator_t<__maybe_const<_Const, _Views>>>::iterator_category,
                    forward_iterator_tag> &&
       ...);

public:
  using iterator_category =
      _If<!is_reference_v<__concat_reference_t<__maybe_const<_Const, _Views>...>>,
          input_iterator_tag,
          _If<__derive_pack_random_iterator,
              random_access_iterator_tag,
              _If<__derive_pack_bidirectional_iterator,
                  bidirectional_iterator_tag,
                  _If<__derive_pack_forward_iterator, forward_iterator_tag, input_iterator_tag > > > >;
};

template <input_range... _Views>
  requires((view<_Views> && ...) && (sizeof...(_Views) > 0) && __concatable<_Views...>)
template <bool _Const>
class concat_view<_Views...>::__iterator : public __concat_view_iterator_category<_Const, _Views...> {
public:
  using iterator_concept =
      _If<__concat_is_random_access<_Const, _Views...>,
          random_access_iterator_tag,
          _If<__concat_is_bidirectional<_Const, _Views...>,
              bidirectional_iterator_tag,
              _If< __all_forward<_Const, _Views...>, forward_iterator_tag, input_iterator_tag > > >;
  using value_type      = __concat_value_t<__maybe_const<_Const, _Views>...>;
  using difference_type = common_type_t<range_difference_t<__maybe_const<_Const, _Views>>...>;

private:
  using __base_iter _LIBCPP_NODEBUG = variant<iterator_t<__maybe_const<_Const, _Views>>...>;
  __base_iter __it_;
  __maybe_const<_Const, concat_view>* __parent_ = nullptr;

  template <size_t _Idx>
  _LIBCPP_HIDE_FROM_ABI constexpr void __satisfy() {
    if constexpr (_Idx < (sizeof...(_Views) - 1)) {
      if (std::get<_Idx>(__it_) == ranges::end(std::get<_Idx>(__parent_->__views_))) {
        __it_.template emplace<_Idx + 1>(ranges::begin(std::get<_Idx + 1>(__parent_->__views_)));
        __satisfy<_Idx + 1>();
      }
    }
  }

  template <size_t _Idx>
  _LIBCPP_HIDE_FROM_ABI constexpr void __prev() {
    if constexpr (_Idx == 0) {
      --std::get<0>(__it_);
    } else {
      if (std::get<_Idx>(__it_) == ranges::begin(std::get<_Idx>(__parent_->__views_))) {
        __it_.template emplace<_Idx - 1>(ranges::end(std::get<_Idx - 1>(__parent_->__views_)));
        __prev<_Idx - 1>();
      } else {
        --std::get<_Idx>(__it_);
      }
    }
  }

  template <size_t _Idx>
  _LIBCPP_HIDE_FROM_ABI constexpr void __advance_fwd(difference_type __offset, difference_type __steps) {
    using __underlying_diff_type = iter_difference_t<variant_alternative_t<_Idx, __base_iter>>;
    if constexpr (_Idx == sizeof...(_Views) - 1) {
      std::get<_Idx>(__it_) += static_cast<__underlying_diff_type>(__steps);
    } else {
      auto __n_size = ranges::distance(std::get<_Idx>(__parent_->__views_));
      if (__offset + __steps < __n_size) {
        std::get<_Idx>(__it_) += static_cast<__underlying_diff_type>(__steps);
      } else {
        __it_.template emplace<_Idx + 1>(ranges::begin(std::get<_Idx + 1>(__parent_->__views_)));
        __advance_fwd<_Idx + 1>(0, __offset + __steps - __n_size);
      }
    }
  }

  template <size_t _Idx>
  _LIBCPP_HIDE_FROM_ABI constexpr void __advance_bwd(difference_type __offset, difference_type __steps) {
    using __underlying_diff_type = iter_difference_t<variant_alternative_t<_Idx, __base_iter>>;
    if constexpr (_Idx == 0) {
      std::get<_Idx>(__it_) -= static_cast<__underlying_diff_type>(__steps);
    } else {
      if (__offset >= __steps) {
        std::get<_Idx>(__it_) -= static_cast<__underlying_diff_type>(__steps);
      } else {
        auto __prev_size = ranges::distance(std::get<_Idx - 1>(__parent_->__views_));
        __it_.template emplace<_Idx - 1>(ranges::end(std::get<_Idx - 1>(__parent_->__views_)));
        __advance_bwd<_Idx - 1>(__prev_size, __steps - __offset);
      }
    }
  }

  template <typename _Func>
  _LIBCPP_HIDE_FROM_ABI constexpr auto __invoke_at_index(_Func&& __func) const {
    // TODO(GCC 16): Just capture `this` when GCC PR113563 and PR121008 are fixed.
    return [&__func, &__view_iter = *this]<std::size_t _Is>(this auto&& __self) {
      if (_Is == __view_iter.__it_.index()) {
        return __func.template operator()<_Is>();
      }
      if constexpr (_Is + 1 < sizeof...(_Views)) {
        return __self.template operator()<_Is + 1>();
      }
      __builtin_unreachable();
    }.template operator()<0>();
  }

  template <size_t... _Is, typename _Func>
  _LIBCPP_HIDE_FROM_ABI constexpr void __apply_at_index(size_t __index, _Func&& __func, index_sequence<_Is...>) const {
    ((__index == _Is ? (static_cast<void>(__func(integral_constant<size_t, _Is>{})), 0) : 0), ...);
  }

  template <size_t _Idx, typename _Func>
  _LIBCPP_HIDE_FROM_ABI constexpr void __apply_at_index(size_t __index, _Func&& __func) const {
    __apply_at_index(__index, std::forward<_Func>(__func), make_index_sequence<_Idx>{});
  }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI explicit constexpr __iterator(__maybe_const<_Const, concat_view>* __parent, _Args&&... __args)
    requires constructible_from<__base_iter, _Args&&...>
      : __it_(std::forward<_Args>(__args)...), __parent_(__parent) {}

  friend class concat_view;
  friend class __iterator<!_Const>;

public:
  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && (convertible_to<iterator_t<_Views>, iterator_t<const _Views>> && ...)
      : __it_([&__src = __i.__it_]<size_t... _Indices>(size_t __idx, index_sequence<_Indices...>) -> __base_iter {
          _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
              !__src.valueless_by_exception(), "Trying to convert from a valueless iterator of concat_view.");
          using __src_lref          = decltype((__src));
          using __construction_fptr = __base_iter (*)(__src_lref);
          static constexpr __construction_fptr __vtable[]{[](__src_lref __src_var) -> __base_iter {
            return __base_iter(in_place_index<_Indices>, std::__unchecked_get<_Indices>(std::move(__src_var)));
          }...};
          return __vtable[__idx](__src);
        }(__i.__it_.index(), make_index_sequence<variant_size_v<__base_iter>>{})),
        __parent_(__i.__parent_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it_.valueless_by_exception(), "Trying to dereference a valueless iterator of concat_view.");
    return __variant_detail::__visitation::__variant::__visit_value(
        [](auto&& __it) -> __concat_reference_t<__maybe_const<_Const, _Views>...> { return *__it; }, __it_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it_.valueless_by_exception(), "Trying to increment a valueless iterator of concat_view.");
    size_t __active_index = __it_.index();
    __apply_at_index<variant_size_v<decltype(__it_)>>(__active_index, [&](auto __index_constant) {
      constexpr size_t __i = __index_constant.value;
      ++std::__unchecked_get<__i>(__it_);
      __satisfy<__i>();
    });
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires(__all_forward<_Const, _Views...>)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires __concat_is_bidirectional<_Const, _Views...>
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it_.valueless_by_exception(), "Trying to decrement a valueless iterator of concat_view.");
    size_t __active_index = __it_.index();
    __apply_at_index<variant_size_v<decltype(__it_)>>(__active_index, [&](auto __index_constant) {
      constexpr size_t __i = __index_constant.value;
      __prev<__i>();
    });
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires __concat_is_bidirectional<_Const, _Views...>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    requires(equality_comparable<iterator_t<__maybe_const<_Const, _Views>>> && ...)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ == __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
    requires __concat_is_random_access<_Const, _Views...>
  {
    return *((*this) + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(const __iterator& __it, difference_type __n)
    requires __concat_is_random_access<_Const, _Views...>
  {
    auto __temp = __it;
    __temp += __n;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, const __iterator& __it)
    requires __concat_is_random_access<_Const, _Views...>
  {
    return __it + __n;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n)
    requires __concat_is_random_access<_Const, _Views...>
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it_.valueless_by_exception(), "Trying to increment a valueless iterator of concat_view.");
    size_t __active_index = __it_.index();
    if (__n > 0) {
      __apply_at_index<tuple_size_v<decltype(__parent_->__views_)>>(__active_index, [&](auto __index_constant) {
        constexpr size_t __i  = __index_constant.value;
        auto& __active_view   = std::get<__i>(__parent_->__views_);
        difference_type __idx = std::get<__i>(__it_) - ranges::begin(__active_view);
        __advance_fwd<__i>(__idx, __n);
      });

    }

    else if (__n < 0) {
      __apply_at_index<tuple_size_v<decltype(__parent_->__views_)>>(__active_index, [&](auto __index_constant) {
        constexpr size_t __i  = __index_constant.value;
        auto& __active_view   = std::get<__i>(__parent_->__views_);
        difference_type __idx = std::get<__i>(__it_) - ranges::begin(__active_view);
        __advance_bwd<__i>(__idx, -__n);
      });
    }

    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n)
    requires __concat_is_random_access<_Const, _Views...>
  {
    *this += -__n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __it, default_sentinel_t) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it.__it_.valueless_by_exception(),
        "Trying to compare a valueless iterator of concat_view with the default sentinel.");
    constexpr auto __last_idx = sizeof...(_Views) - 1;
    return __it.__it_.index() == __last_idx &&
           std::__unchecked_get<__last_idx>(__it.__it_) == ranges::end(std::get<__last_idx>(__it.__parent_->__views_));
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires(__all_random_access<_Const, _Views> && ...)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ < __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires(__all_random_access<_Const, _Views> && ...)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ > __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires(__all_random_access<_Const, _Views> && ...)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ <= __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires(__all_random_access<_Const, _Views> && ...)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ >= __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires((__all_random_access<_Const, _Views> && ...) &&
             (three_way_comparable<iterator_t<__maybe_const<_Const, _Views>>> && ...))
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(!__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
                                        "Trying to compare a valueless iterator of concat_view.");
    return __x.__it_ <=> __y.__it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr decltype(auto) iter_move(const __iterator& __it) noexcept(

      ((is_nothrow_invocable_v< decltype(ranges::iter_move), const iterator_t<__maybe_const<_Const, _Views>>& > &&
        is_nothrow_convertible_v< range_rvalue_reference_t<__maybe_const<_Const, _Views>>,
                                  __concat_rvalue_reference_t<__maybe_const<_Const, _Views>...> >) &&
       ...))

  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it.__it_.valueless_by_exception(), "Trying to apply iter_move to a valueless iterator of concat_view.");
    return __variant_detail::__visitation::__variant::__visit_value(
        [](const auto& __i) -> __concat_rvalue_reference_t<__maybe_const<_Const, _Views>...> {
          return ranges::iter_move(__i);
        },
        __it.__it_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void iter_swap(const __iterator& __x, const __iterator& __y)

      noexcept((noexcept(ranges::swap(*__x, *__y))) &&
               (noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<_Const, _Views>>>(),
                                           std::declval<const iterator_t<__maybe_const<_Const, _Views>>>())) &&
                ...))

    requires swappable_with<iter_reference_t<__iterator>, iter_reference_t<__iterator>> &&
             (... && indirectly_swappable<iterator_t<__maybe_const<_Const, _Views>>>)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
        "Trying to swap iterators of concat_view where at least one iterator is valueless.");
    __variant_detail::__visitation::__variant::__visit_value(
        [&](const auto& __it1, const auto& __it2) {
          if constexpr (is_same_v<decltype(__it1), decltype(__it2)>) {
            ranges::iter_swap(__it1, __it2);
          } else {
            ranges::swap(*__x, *__y);
          }
        },
        __x.__it_,
        __y.__it_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    requires __concat_is_random_access<_Const, _Views...>
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__x.__it_.valueless_by_exception() && !__y.__it_.valueless_by_exception(),
        "Trying to subtract two iterators of concat_view where at least one iterator is valueless.");
    return __x.__invoke_at_index([&]<std::size_t __index_x>() -> difference_type {
      return __y.__invoke_at_index([&]<std::size_t __index_y>() -> difference_type {
        if constexpr (__index_x > __index_y) {
          auto __dx = ranges::distance(
              ranges::begin(std::get<__index_x>(__x.__parent_->__views_)), std::get<__index_x>(__x.__it_));
          auto __dy = ranges::distance(
              std::get<__index_y>(__y.__it_), ranges::end(std::get<__index_y>(__y.__parent_->__views_)));
          difference_type __s = [&]<std::size_t __start, std::size_t __end>(this auto&& __self) -> difference_type {
            if constexpr (__start < __end) {
              return ranges::size(std::get<__start>(__x.__parent_->__views_)) +
                     __self.template operator()<__start + 1, __end>();
            }
            return 0;
          }.template operator()<__index_y + 1, __index_x>();
          return __dy + __s + __dx;
        } else if constexpr (__index_x < __index_y) {
          return -(__y - __x);
        } else {
          return std::get<__index_x>(__x.__it_) - std::get<__index_y>(__y.__it_);
        }
      });
    });
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __it, difference_type __n)
    requires __concat_is_random_access<_Const, _Views...>
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__it.__it_.valueless_by_exception(), "Trying to subtract a valuess iterators of concat_view.");
    auto __temp = __it;
    __temp -= __n;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, default_sentinel_t)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
             ...) &&
            (__all_but_first_model_sized_range<_Const, _Views...>::value)
  {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !__x.__it_.valueless_by_exception(),
        "Trying to subtract a valuess iterators of concat_view from the default sentinel.");
    return __x.__invoke_at_index([&]<std::size_t __index_x>() -> difference_type {
      auto __dx =
          ranges::distance(std::get<__index_x>(__x.__it_), ranges::end(std::get<__index_x>(__x.__parent_->__views_)));
      difference_type __s = [&]<std::size_t __start, std::size_t __end>(this auto&& __self) -> difference_type {
        if constexpr (__start < __end) {
          return ranges::size(std::get<__start>(__x.__parent_->__views_)) +
                 __self.template operator()<__start + 1, __end>();
        }
        return 0;
      }.template operator()<__index_x + 1, sizeof...(_Views)>();
      return -(__dx + __s);
    });
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(default_sentinel_t, const __iterator& __x)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
             ...) &&
            (__all_but_first_model_sized_range<_Const, _Views...>::value)
  {
    return -(__x - default_sentinel);
  }
};

namespace views {
namespace __concat {
struct __fn {
  template <input_range _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_Range&& __range) noexcept(noexcept(views::all((std::forward<_Range>(__range)))))
      -> decltype(views::all((std::forward<_Range>(__range)))) {
    return views::all(std::forward<_Range>(__range));
  }

  template <class _FirstRange, class... _TailRanges>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto
  operator()(_FirstRange&& __first, _TailRanges&&... __tail) noexcept(
      noexcept(concat_view(std::forward<_FirstRange>(__first), std::forward<_TailRanges>(__tail)...)))
      -> decltype(concat_view(std::forward<_FirstRange>(__first), std::forward<_TailRanges>(__tail)...)) {
    return concat_view(std::forward<_FirstRange>(__first), std::forward<_TailRanges>(__tail)...);
  }
};
} // namespace __concat

inline namespace __cpo {
inline constexpr auto concat = __concat::__fn{};
} // namespace __cpo
} // namespace views

} // namespace ranges

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CONCAT_VIEW_H

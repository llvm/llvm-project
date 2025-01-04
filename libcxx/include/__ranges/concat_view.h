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

#include <__algorithm/ranges_find_if.h>
#include <__assert>
#include <__concepts/common_reference_with.h>
#include <__concepts/constructible.h>
#include <__concepts/convertible_to.h>
#include <__concepts/copyable.h>
#include <__concepts/derived_from.h>
#include <__concepts/equality_comparable.h>
#include <__concepts/swappable.h>
#include <__config>
#include <__functional/bind_back.h>
#include <__functional/invoke.h>
#include <__functional/reference_wrapper.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/movable_box.h>
#include <__ranges/non_propagating_cache.h>
#include <__ranges/range_adaptor.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__ranges/zip_view.h>
#include <__type_traits/conditional.h>
#include <__type_traits/decay.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_nothrow_convertible.h>
#include <__type_traits/is_object.h>
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

template <class... __Tp>
using __extract_last = __Tp...[sizeof...(__Tp) - 1];

template <class _Tp, class... _Tail>
constexpr bool __derived_from_pack =
    __derived_from_pack<_Tp, __extract_last<_Tail...>> && __derived_from_pack<_Tail...>;

template <class _Tp, class _IterCategory>
constexpr bool __derived_from_pack<_Tp, _IterCategory> = derived_from<_Tp, _IterCategory>;

template <class _View, class... _Views>
struct __last_view : __last_view<_Views...> {};

template <class _View>
struct __last_view<_View> {
  using type = _View;
};

template <class _Ref, class _RRef, class _It>
concept __concat_indirectly_readable_impl = requires(const _It __it) {
  { *__it } -> convertible_to<_Ref>;
  { ranges::iter_move(__it) } -> convertible_to<_RRef>;
};

template <class... _Rs>
using __concat_reference_t = common_reference_t<range_reference_t<_Rs>...>;

template <class... _Rs>
using __concat_value_t = common_type_t<range_value_t<_Rs>...>;

template <class... _Rs>
using __concat_rvalue_reference_t = common_reference_t<range_rvalue_reference_t<_Rs>...>;

template <class... _Rs>
concept __concat_indirectly_readable =
    common_reference_with<__concat_reference_t<_Rs...>&&, __concat_value_t<_Rs...>&> &&
    common_reference_with<__concat_reference_t<_Rs...>&&, __concat_rvalue_reference_t<_Rs...>&&> &&
    common_reference_with<__concat_rvalue_reference_t<_Rs...>&&, __concat_value_t<_Rs...> const&> &&
    (__concat_indirectly_readable_impl<__concat_reference_t<_Rs...>,
                                       __concat_rvalue_reference_t<_Rs...>,
                                       iterator_t<_Rs>> &&
     ...);

template <class... _Rs>
concept __concatable = requires {
  typename __concat_reference_t<_Rs...>;
  typename __concat_value_t<_Rs...>;
  typename __concat_rvalue_reference_t<_Rs...>;
} && __concat_indirectly_readable<_Rs...>;

template <bool _Const, class... _Rs>
concept __concat_is_random_access =
    (random_access_range<__maybe_const<_Const, _Rs>> && ...) && (sized_range<__maybe_const<_Const, _Rs>> && ...);

template <bool _Const, class... _Rs>
concept __concat_is_bidirectional =
    ((bidirectional_range<__maybe_const<_Const, _Rs>> && ...) && (common_range<__maybe_const<_Const, _Rs>> && ...));

template <bool _Const, class... _Views>
concept __all_forward = (forward_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class... _Tp>
struct __apply_drop_first;

template <bool _Const, class _Head, class... _Tail>
struct __apply_drop_first<_Const, _Head, _Tail...> {
  static constexpr bool value = (sized_range<__maybe_const<_Const, _Tail>> && ...);
};

template <input_range... _Views>
  requires(view<_Views> && ...) && (sizeof...(_Views) > 0) && __concatable<_Views...>
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
    __it.template satisfy<0>();
    return __it;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator<true> begin() const
    requires((range<const _Views> && ...) && __concatable<const _Views...>)
  {
    __iterator<true> __it(this, in_place_index<0>, ranges::begin(std::get<0>(__views_)));
    __it.template satisfy<0>();
    return __it;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!(__simple_view<_Views> && ...))
  {
    if constexpr (common_range<typename __last_view<_Views...>::type>) {
      constexpr auto __n = sizeof...(_Views);
      return __iterator<false>(this, in_place_index<__n - 1>, ranges::end(std::get<__n - 1>(__views_)));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires(range<const _Views> && ...)
  {
    if constexpr (common_range<typename __last_view<_Views...>::type>) {
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
        [](auto... __sizes) {
          using CT = make_unsigned_t<common_type_t<decltype(__sizes)...>>;
          return (CT(__sizes) + ...);
        },
        ranges::__tuple_transform(ranges::size, __views_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires(sized_range<const _Views> && ...)
  {
    return std::apply(
        [](auto... __sizes) {
          using CT = make_unsigned_t<common_type_t<decltype(__sizes)...>>;
          return (CT(__sizes) + ...);
        },
        ranges::__tuple_transform(ranges::size, __views_));
  }
};

template <class... _Views>
concat_view(_Views&&...) -> concat_view<views::all_t<_Views>...>;

template <input_range... _Views>
  requires(view<_Views> && ...) && (sizeof...(_Views) > 0) && __concatable<_Views...>
template <bool _Const>
class concat_view<_Views...>::__iterator {
public:
  constexpr static bool derive_pack_random_iterator =
      __derived_from_pack<typename iterator_traits<iterator_t<__maybe_const<_Const, _Views>>>::iterator_category...,
                          random_access_iterator_tag>;
  constexpr static bool derive_pack_bidirectional_iterator =
      __derived_from_pack<typename iterator_traits<iterator_t<__maybe_const<_Const, _Views>>>::iterator_category...,
                          bidirectional_iterator_tag>;
  constexpr static bool derive_pack_forward_iterator =
      __derived_from_pack<typename iterator_traits< iterator_t<__maybe_const<_Const, _Views>>>::iterator_category...,
                          forward_iterator_tag>;
  using iterator_category =
      _If<!is_reference_v<__concat_reference_t<__maybe_const<_Const, _Views>...>>,
          input_iterator_tag,
          _If<derive_pack_random_iterator,
              random_access_iterator_tag,
              _If<derive_pack_bidirectional_iterator,
                  bidirectional_iterator_tag,
                  _If<derive_pack_forward_iterator, forward_iterator_tag, input_iterator_tag > > > >;
  using iterator_concept =
      _If<__concat_is_random_access<_Const, _Views...>,
          random_access_iterator_tag,
          _If<__concat_is_bidirectional<_Const, _Views...>,
              bidirectional_iterator_tag,
              _If< __all_forward<_Const, _Views...>, forward_iterator_tag, input_iterator_tag > > >;
  using value_type      = __concat_value_t<__maybe_const<_Const, _Views>...>;
  using difference_type = common_type_t<range_difference_t<__maybe_const<_Const, _Views>>...>;
  using base_iter       = variant<iterator_t<__maybe_const<_Const, _Views>>...>;

  base_iter it_;
  __maybe_const<_Const, concat_view>* parent_ = nullptr;

  template <std::size_t _N>
  _LIBCPP_HIDE_FROM_ABI constexpr void satisfy() {
    if constexpr (_N < (sizeof...(_Views) - 1)) {
      if (std::get<_N>(it_) == ranges::end(std::get<_N>(parent_->__views_))) {
        it_.template emplace<_N + 1>(ranges::begin(std::get<_N + 1>(parent_->__views_)));
        satisfy<_N + 1>();
      }
    }
  }

  template <std::size_t _N>
  _LIBCPP_HIDE_FROM_ABI constexpr void prev() {
    if constexpr (_N == 0) {
      --std::get<0>(it_);
    } else {
      if (std::get<_N>(it_) == ranges::begin(std::get<_N>(parent_->__views_))) {
        using prev_view = __maybe_const<_Const, tuple_element_t<_N - 1, tuple<_Views...>>>;
        if constexpr (common_range<prev_view>) {
          it_.emplace<_N - 1>(ranges::end(std::get<_N - 1>(parent_->__views_)));
        } else {
          it_.emplace<_N - 1>(ranges::__next(
              ranges::begin(std::get<_N - 1>(parent_->__views_)), ranges::size(std::get<_N - 1>(parent_->__views_))));
        }
        prev<_N - 1>();
      } else {
        --std::get<_N>(it_);
      }
    }
  }

  template <std::size_t _N>
  _LIBCPP_HIDE_FROM_ABI constexpr void advance_fwd(difference_type __offset, difference_type __steps) {
    using underlying_diff_type = iter_difference_t<variant_alternative_t<_N, base_iter>>;
    if constexpr (_N == sizeof...(_Views) - 1) {
      std::get<_N>(it_) += static_cast<underlying_diff_type>(__steps);
    } else {
      difference_type __n_size = ranges::size(std::get<_N>(parent_->__views_));
      if (__offset + __steps < __n_size) {
        std::get<_N>(it_) += static_cast<underlying_diff_type>(__steps);
      } else {
        it_.template emplace<_N + 1>(ranges::begin(std::get<_N + 1>(parent_->__views_)));
        advance_fwd<_N + 1>(0, __offset + __steps - __n_size);
      }
    }
  }

  template <std::size_t _N>
  _LIBCPP_HIDE_FROM_ABI constexpr void advance_bwd(difference_type __offset, difference_type __steps) {
    using underlying_diff_type = iter_difference_t<variant_alternative_t<_N, base_iter>>;
    if constexpr (_N == 0) {
      std::get<_N>(it_) -= static_cast<underlying_diff_type>(__steps);
    } else {
      if (__offset >= __steps) {
        std::get<_N>(it_) -= static_cast<underlying_diff_type>(__steps);
      } else {
        auto __prev_size = ranges::__distance(std::get<_N - 1>(parent_->__views_));
        it_.emplace<_N - 1>(ranges::begin(std::get<_N - 1>(parent_->__views_)) + __prev_size);
        advance_bwd<_N - 1>(__prev_size, __steps - __offset);
      }
    }
  }

  template <size_t... _Is, typename _Func>
  _LIBCPP_HIDE_FROM_ABI constexpr void
  apply_fn_with_const_index(size_t __index, _Func&& __func, std::index_sequence<_Is...>) {
    ((__index == _Is ? (__func(std::integral_constant<size_t, _Is>{}), 0) : 0), ...);
  }

  template <size_t _N, typename _Func>
  _LIBCPP_HIDE_FROM_ABI constexpr void apply_fn_with_const_index(size_t __index, _Func&& __func) {
    apply_fn_with_const_index(__index, std::forward<_Func>(__func), std::make_index_sequence<_N>{});
  }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI explicit constexpr __iterator(__maybe_const<_Const, concat_view>* __parent, _Args&&... __args)
    requires constructible_from<base_iter, _Args&&...>
      : it_(std::forward<_Args>(__args)...), parent_(__parent) {}

public:
  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_Const> __i)
    requires _Const && (convertible_to<iterator_t<_Views>, iterator_t<const _Views>> && ...)
      : it_(base_iter(in_place_index<__i.index()>, std::get<__i.index()>(std::move(__i.it_)))), parent_(__i.parent_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const {
    using reference = __concat_reference_t<__maybe_const<_Const, _Views>...>;
    return std::visit([](auto&& __it) -> reference { return *__it; }, it_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    size_t __active_index = it_.index();
    apply_fn_with_const_index<std::variant_size_v<decltype(it_)>>(__active_index, [&](auto __index_constant) {
      constexpr size_t __i = __index_constant.value;
      ++std::get<__i>(it_);
      satisfy<__i>();
    });
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires(forward_range<__maybe_const<_Const, _Views>> && ...)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires __concat_is_bidirectional<_Const, _Views...>
  {
    size_t __active_index = it_.index();
    apply_fn_with_const_index<std::variant_size_v<decltype(it_)>>(__active_index, [&](auto __index_constant) {
      constexpr size_t __i = __index_constant.value;
      prev<__i>();
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
    return __x.it_ == __y.it_;
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
    size_t __active_index = it_.index();
    if (__n > 0) {
      std::visit(
          [&](auto& __active_it) {
            apply_fn_with_const_index<std::tuple_size_v<decltype(parent_->__views_)>>(
                __active_index, [&](auto __index_constant) {
                  constexpr size_t __I  = __index_constant.value;
                  auto& __active_view   = std::get<__I>(parent_->__views_);
                  difference_type __idx = __active_it - ranges::begin(__active_view);
                  advance_fwd<__I>(__idx, __n);
                });
          },
          it_);
    }

    else if (__n < 0) {
      std::visit(
          [&](auto& __active_it) {
            apply_fn_with_const_index<std::tuple_size_v<decltype(parent_->__views_)>>(
                __active_index, [&](auto __index_constant) {
                  constexpr size_t __I  = __index_constant.value;
                  auto& __active_view   = std::get<__I>(parent_->__views_);
                  difference_type __idx = __active_it - ranges::begin(__active_view);
                  advance_bwd<__I>(__idx, -__n);
                });
          },
          it_);
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
    constexpr auto __last_idx = sizeof...(_Views) - 1;
    return __it.it_.index() == __last_idx &&
           std::get<__last_idx>(__it.it_) == ranges::end(std::get<__last_idx>(__it.parent_->__views_));
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires(random_access_range<__maybe_const<_Const, _Views>> && ...)
  {
    return __x.it_ < __y.it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires(random_access_range<__maybe_const<_Const, _Views>> && ...)
  {
    return __x.it_ > __y.it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires(random_access_range<__maybe_const<_Const, _Views>> && ...)
  {
    return __x.it_ <= __y.it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires(random_access_range<__maybe_const<_Const, _Views>> && ...)
  {
    return __x.it_ >= __y.it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires((random_access_range<__maybe_const<_Const, _Views>> && ...) &&
             (three_way_comparable<__maybe_const<_Const, _Views>> && ...))
  {
    return __x.it_ <=> __y.it_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr decltype(auto) iter_move(const __iterator& __it) noexcept(

      ((is_nothrow_invocable_v< decltype(ranges::iter_move), const iterator_t<__maybe_const<_Const, _Views>>& >) &&
       ...) &&
      ((is_nothrow_convertible_v< range_rvalue_reference_t<__maybe_const<_Const, _Views>>,
                                  __concat_rvalue_reference_t<__maybe_const<_Const, _Views>...> >) &&
       ...))

  {
    return std::visit(
        [](const auto& __i) -> __concat_rvalue_reference_t<__maybe_const<_Const, _Views>...> {
          return ranges::iter_move(__i);
        },
        __it.it_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void iter_swap(const __iterator& __x, const __iterator& __y)

      noexcept((noexcept(ranges::swap(*__x, *__y))) &&
               (noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<_Const, _Views>>>(),
                                           std::declval<const iterator_t<__maybe_const<_Const, _Views>>>())) &&
                ...))

    requires swappable_with<iter_reference_t<__iterator>, iter_reference_t<__iterator>> &&
             (... && indirectly_swappable<iterator_t<__maybe_const<_Const, _Views>>>)
  {
    std::visit(
        [&](const auto& __it1, const auto& __it2) {
          if constexpr (is_same_v<decltype(__it1), decltype(__it2)>) {
            ranges::iter_swap(__it1, __it2);
          } else {
            ranges::swap(*__x, *__y);
          }
        },
        __x.it_,
        __y.it_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    requires __concat_is_random_access<_Const, _Views...>
  {
    size_t __ix = __x.it_.index();
    size_t __iy = __y.it_.index();

    if (__ix > __iy) {
      std::visit(
          [&](auto& __it_x, auto& __it_y) {
            __it_x.apply_fn_with_const_index<std::tuple_size_v<decltype(__x.parent_->__views_)>>(
                __ix, [&](auto index_constant) {
                  constexpr size_t __index_x = index_constant.value;
                  auto __dx = ranges::__distance(ranges::begin(std::get<__index_x>(__x.parent_->__views_)), __it_x);

                  __it_y.apply_fn_with_const_index<std::tuple_size_v<decltype(__y.parent_->__views_)>>(
                      __iy, [&](auto index_constant) {
                        constexpr size_t __index_y = index_constant.value;
                        auto __dy =
                            ranges::__distance(ranges::begin(std::get<__index_y>(__y.parent_->__views_)), __it_y);
                        difference_type __s = 0;
                        for (size_t __idx = __index_y + 1; __idx < __index_x; __idx++) {
                          __s += ranges::size(std::get<__idx>(__x.parent_->__views_));
                        }
                        return __dy + __s + __dx;
                      });
                });
          },
          __x.it_,
          __y.it_);
    } else if (__ix < __iy) {
      return -(__y - __x);
    } else {
      std::visit([&](const auto& __it1, const auto& __it2) { return __it1 - __it2; }, __x.it_, __y.it_);
    }
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __it, difference_type __n)
    requires __concat_is_random_access<_Const, _Views...>
  {
    auto __temp = __it;
    __temp -= __n;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, default_sentinel_t)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
             ...) &&
            (__apply_drop_first<_Const, _Views...>::value)
  {
    size_t __ix = __x.it_.index();
    std::visit(
        [&](auto& __it_x) {
          __it_x.apply_fn_with_const_index<std::tuple_size_v<decltype(__x.parent_->__views_)>>(
              __ix, [&](auto __index_constant) {
                constexpr size_t __index_x = __index_constant.value;
                auto __dx = ranges::__distance(ranges::begin(std::get<__index_x>(__x.parent_->__views_)), __it_x);

                difference_type __s = 0;
                for (size_t __idx = 0; __idx < __index_x; __idx++) {
                  __s += ranges::size(std::get<__idx>(__x.parent_->__views_));
                }

                return -(__dx + __s);
              });
        },
        __x.it_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(default_sentinel_t, const __iterator& __x)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_Const, _Views>>> &&
             ...) &&
            (__apply_drop_first<_Const, _Views...>::value)
  {
    -(__x - default_sentinel);
  }
};

namespace views {
namespace __concat {
struct __fn {
  template <class... _Views>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator()(_Views... views) const noexcept(
      noexcept(concat_view(std::forward<_Views>(views)...))) -> decltype(concat_view(std::forward<_Views>(views)...)) {
    return concat_view(std::forward<_Views>(views)...);
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

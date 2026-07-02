//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H
#define _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

#include <__concepts/convertible_to.h>
#include <__concepts/equality_comparable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/access.h>
#include <__iterator/concepts.h>
#include <__iterator/default_sentinel.h>
#include <__iterator/distance.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__ranges/all.h>
#include <__ranges/concepts.h>
#include <__ranges/empty.h>
#include <__ranges/single_view.h>
#include <__ranges/size.h>
#include <__ranges/view_interface.h>
#include <__ranges/zip_view.h>
#include <__tuple/tuple_transform.h>
#include <__type_traits/common_type.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/maybe_const.h>
#include <__utility/forward.h>
#include <__utility/integer_sequence.h>
#include <__utility/move.h>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

template <bool _Const, class _First, class... _Vs>
concept __cartesian_product_is_random_access =
    (random_access_range<__maybe_const<_Const, _First>> && ... &&
     (random_access_range<__maybe_const<_Const, _Vs>> && sized_range<__maybe_const<_Const, _Vs>>));

template <class _Rp>
concept __cartesian_product_common_arg = common_range<_Rp> || (sized_range<_Rp> && random_access_range<_Rp>);

template <bool _Const, class _First, class... _Vs>
concept __cartesian_product_is_bidirectional =
    (bidirectional_range<__maybe_const<_Const, _First>> && ... &&
     (bidirectional_range<__maybe_const<_Const, _Vs>> && __cartesian_product_common_arg<__maybe_const<_Const, _Vs>>));

template <class _First, class... _Vs>
concept __cartesian_product_is_common = __cartesian_product_common_arg<_First>;

template <class... _Vs>
concept __cartesian_product_is_sized = (sized_range<_Vs> && ...);

template <bool _Const, template <class> class _FirstSent, class _First, class... _Vs>
concept __cartesian_is_sized_sentinel =
    (sized_sentinel_for<_FirstSent<__maybe_const<_Const, _First>>, iterator_t<__maybe_const<_Const, _First>>> && ... &&
     (sized_range<__maybe_const<_Const, _Vs>> &&
      sized_sentinel_for<iterator_t<__maybe_const<_Const, _Vs>>, iterator_t<__maybe_const<_Const, _Vs>>>));

template <__cartesian_product_common_arg _Rp>
_LIBCPP_HIDE_FROM_ABI constexpr auto __cartesian_common_arg_end(_Rp& __r) {
  if constexpr (common_range<_Rp>) {
    return ranges::end(__r);
  } else {
    return ranges::begin(__r) + ranges::distance(__r);
  }
}

template <bool _Const, class _First, class... _Vs>
concept __cartesian_product_all_random_access =
    (random_access_range<__maybe_const<_Const, _First>> && ... && random_access_range<__maybe_const<_Const, _Vs>>);

template <input_range _First, forward_range... _Vs>
  requires(view<_First> && ... && view<_Vs>)
class cartesian_product_view : public view_interface<cartesian_product_view<_First, _Vs...>> {
private:
  tuple<_First, _Vs...> __bases_;

  template <bool _IsConst>
  class __iterator;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr cartesian_product_view() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit cartesian_product_view(_First __first_base, _Vs... __bases)
      : __bases_{std::move(__first_base), std::move(__bases)...} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator<false> begin()
    requires(!__simple_view<_First> || ... || !__simple_view<_Vs>)
  {
    return __iterator<false>(*this, __tuple_transform(ranges::begin, __bases_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator<true> begin() const
    requires(range<const _First> && ... && range<const _Vs>)
  {
    return __iterator<true>(*this, __tuple_transform(ranges::begin, __bases_));
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator<false> end()
    requires((!__simple_view<_First> || ... || !__simple_view<_Vs>) && __cartesian_product_is_common<_First, _Vs...>)
  {
    constexpr bool __is_const_ = false;
    return __end_impl<__is_const_>(*this);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator<true> end() const
    requires __cartesian_product_is_common<const _First, const _Vs...>
  {
    constexpr bool __is_const_ = true;
    return __end_impl<__is_const_>(*this);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr default_sentinel_t end() const noexcept { return default_sentinel; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires __cartesian_product_is_sized<_First, _Vs...>
  {
    return __size_impl();
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires __cartesian_product_is_sized<const _First, const _Vs...>
  {
    return __size_impl();
  }

private:
  template <bool _IsConst>
  _LIBCPP_HIDE_FROM_ABI static constexpr __iterator<_IsConst>
  __end_impl(__maybe_const<_IsConst, cartesian_product_view>& __self) {
    const auto __ranges_to_iterators = [__end_is_empty = __self.__end_is_empty(),
                                        &__b = __self.__bases_]<std::size_t... _Ip>(std::index_sequence<_Ip...>) {
      const auto __begin_or_first_end = []<class _IsFirst>(_IsFirst, auto& __rng, bool __empty) {
        if constexpr (_IsFirst::value)
          return __empty ? ranges::begin(__rng) : __cartesian_common_arg_end(__rng);
        return ranges::begin(__rng);
      };
      return std::make_tuple(
          __begin_or_first_end(std::bool_constant<_Ip == 0>{}, std::get<_Ip>(__b), __end_is_empty)...);
    };
    __iterator<_IsConst> __it(__self, __ranges_to_iterators(std::make_index_sequence<1 + sizeof...(_Vs)>{}));
    return __it;
  }

  template <std::size_t _Np = 1>
  _LIBCPP_HIDE_FROM_ABI constexpr bool __end_is_empty() const {
    if constexpr (_Np == 1 + sizeof...(_Vs))
      return false;
    else {
      if (const auto& __v = std::get<_Np>(__bases_); ranges::empty(__v))
        return true;
      return __end_is_empty<_Np + 1>();
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto __size_impl() const {
    return std::apply(
        [](auto&&... __bases) {
          using _SizeType = std::common_type_t<std::ranges::range_size_t<decltype(__bases)>...>;
          return (static_cast<_SizeType>(std::ranges::size(__bases)) * ...);
        },
        __bases_);
  }
};

template <class... _Vs>
cartesian_product_view(_Vs&&...) -> cartesian_product_view<views::all_t<_Vs>...>;

template <input_range _First, forward_range... _Vs>
  requires(view<_First> && ... && view<_Vs>)
template <bool _IsConst>
class cartesian_product_view<_First, _Vs...>::__iterator {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __get_iterator_tag() {
    if constexpr (__cartesian_product_is_random_access<_IsConst, _First, _Vs...>)
      return random_access_iterator_tag{};
    else if constexpr (__cartesian_product_is_bidirectional<_IsConst, _First, _Vs...>)
      return bidirectional_iterator_tag{};
    else if constexpr (forward_range<__maybe_const<_IsConst, _First>>)
      return forward_iterator_tag{};
    else
      return input_iterator_tag{};
  }

  friend cartesian_product_view;

  template <bool>
  friend class cartesian_product_view<_First, _Vs...>::__iterator;

public:
  using iterator_category = input_iterator_tag;
  using iterator_concept  = decltype(__get_iterator_tag());
  using value_type =
      tuple<range_value_t<__maybe_const<_IsConst, _First>>, range_value_t<__maybe_const<_IsConst, _Vs>>...>;
  using reference =
      tuple<range_reference_t<__maybe_const<_IsConst, _First>>, range_reference_t<__maybe_const<_IsConst, _Vs>>...>;
  using difference_type = std::common_type_t<range_difference_t<_First>, range_difference_t<_Vs>...>;

  _LIBCPP_HIDE_FROM_ABI __iterator() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!_IsConst> __i)
    requires _IsConst && (convertible_to<iterator_t<_First>, iterator_t<const _First>> && ... &&
                          convertible_to<iterator_t<_Vs>, iterator_t<const _Vs>>)
      : __parent_(__i.__parent_), __current_(std::move(__i.__current_)) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto operator*() const {
    return __tuple_transform([](auto& __i) -> decltype(auto) { return *__i; }, __current_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    __next();
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { ++*this; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires forward_range<__maybe_const<_IsConst, _First>>
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires __cartesian_product_is_bidirectional<_IsConst, _First, _Vs...>
  {
    __prev();
    return *this;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires __cartesian_product_is_bidirectional<_IsConst, _First, _Vs...>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __x)
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    __advance(__x);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __x)
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    *this += -__x;
    return *this;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr reference operator[](difference_type __n) const
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    return *((*this) + __n);
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    requires equality_comparable<iterator_t<__maybe_const<_IsConst, _First>>>
  {
    return __x.__current_ == __y.__current_;
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __x, default_sentinel_t) {
    return __x.__at_end();
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires __cartesian_product_all_random_access<_IsConst, _First, _Vs...>
  {
    return __x.__current_ <=> __y.__current_;
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator+(const __iterator& __x, difference_type __y)
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    return __iterator(__x) += __y;
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator+(difference_type __x, const __iterator& __y)
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    return __y + __x;
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator-(const __iterator& __x, difference_type __y)
    requires __cartesian_product_is_random_access<_IsConst, _First, _Vs...>
  {
    return __iterator(__x) -= __y;
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr difference_type
  operator-(const __iterator& __x, const __iterator& __y)
    requires __cartesian_is_sized_sentinel<_IsConst, iterator_t, _First, _Vs...>
  {
    return __x.__distance_from(__y.__current_);
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr difference_type
  operator-(const __iterator& __i, default_sentinel_t)
    requires __cartesian_is_sized_sentinel<_IsConst, sentinel_t, _First, _Vs...>
  {
    tuple __end_tuple = [&__b = __i.__parent_->__bases_]<std::size_t... _Ip>(std::index_sequence<_Ip...>) {
      return tuple{ranges::end(std::get<0>(__b)), ranges::begin(std::get<1 + _Ip>(__b))...};
    }(std::make_index_sequence<sizeof...(_Vs)>{});
    return __i.__distance_from(__end_tuple);
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr difference_type
  operator-(default_sentinel_t, const __iterator& __i)
    requires __cartesian_is_sized_sentinel<_IsConst, sentinel_t, _First, _Vs...>
  {
    return -(__i - default_sentinel);
  }

  [[nodiscard]] friend _LIBCPP_HIDE_FROM_ABI constexpr auto iter_move(const __iterator& __i) noexcept(
      noexcept(ranges::iter_move(std::declval<const iterator_t<__maybe_const<_IsConst, _First>>&>())) &&
      (noexcept(ranges::iter_move(std::declval<const iterator_t<__maybe_const<_IsConst, _Vs>>&>())) && ...) &&
      is_nothrow_move_constructible_v<range_rvalue_reference_t<__maybe_const<_IsConst, _First>>> &&
      (is_nothrow_move_constructible_v<range_rvalue_reference_t<__maybe_const<_IsConst, _Vs>>> && ...)) {
    return __tuple_transform(ranges::iter_move, __i.__current_);
  }

  friend _LIBCPP_HIDE_FROM_ABI constexpr void iter_swap(const __iterator& __l, const __iterator& __r) noexcept(
      noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<_IsConst, _First>>&>(),
                                 std::declval<const iterator_t<__maybe_const<_IsConst, _First>>&>())) &&
      (noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<_IsConst, _Vs>>&>(),
                                  std::declval<const iterator_t<__maybe_const<_IsConst, _Vs>>&>())) &&
       ...))
    requires(indirectly_swappable<iterator_t<__maybe_const<_IsConst, _First>>> && ... &&
             indirectly_swappable<iterator_t<__maybe_const<_IsConst, _Vs>>>)
  {
    ranges::__tuple_zip_for_each(ranges::iter_swap, __l.__current_, __r.__current_);
  }

private:
  using _Parent      = __maybe_const<_IsConst, cartesian_product_view>;
  _Parent* __parent_ = nullptr;
  using _MultiIter   = tuple<iterator_t<__maybe_const<_IsConst, _First>>, iterator_t<__maybe_const<_IsConst, _Vs>>...>;
  _MultiIter __current_;

  template <std::size_t _Np = sizeof...(_Vs)>
  _LIBCPP_HIDE_FROM_ABI constexpr void __next() {
    auto& __it = std::get<_Np>(__current_);
    ++__it;
    if constexpr (_Np > 0) {
      if (const auto& __v = std::get<_Np>(__parent_->__bases_); __it == ranges::end(__v)) {
        __it = ranges::begin(__v);
        __next<_Np - 1>();
      }
    }
  }

  template <std::size_t _Np = sizeof...(_Vs)>
  _LIBCPP_HIDE_FROM_ABI constexpr void __prev() {
    auto& __it = std::get<_Np>(__current_);
    if constexpr (_Np > 0) {
      if (const auto& __v = std::get<_Np>(__parent_->__bases_); __it == ranges::begin(__v)) {
        __it = __cartesian_common_arg_end(__v);
        __prev<_Np - 1>();
      }
    }
    --__it;
  }

  template <auto _Np = sizeof...(_Vs)>
  _LIBCPP_HIDE_FROM_ABI constexpr void __advance(difference_type __x) {
    if (__x == 0)
      return;

    const auto& __v    = std::get<_Np>(__parent_->__bases_);
    auto& __it         = std::get<_Np>(__current_);
    const auto __sz    = static_cast<difference_type>(std::ranges::size(__v));
    const auto __first = ranges::begin(__v);

    if (__sz > 0) {
      const auto __idx = static_cast<difference_type>(std::distance(__first, __it));
      __x += __idx;

      difference_type __mod;
      if constexpr (_Np > 0) {
        difference_type __div = __x / __sz;
        __mod                 = __x % __sz;
        if (__mod < 0) {
          __mod += __sz;
          __div--;
        }
        __advance<_Np - 1>(__div);
      } else {
        __mod = (__x >= 0 && __x < __sz) ? __x : __sz;
      }
      __it = std::next(__first, __mod);

    } else {
      if constexpr (_Np > 0) {
        __advance<_Np - 1>(__x);
      }
      __it = __first;
    }
  }

  template <auto _Np = sizeof...(_Vs)>
  _LIBCPP_HIDE_FROM_ABI constexpr bool __at_end() const {
    if (std::get<_Np>(__current_) == ranges::end(std::get<_Np>(__parent_->__bases_)))
      return true;
    if constexpr (_Np > 0)
      return __at_end<_Np - 1>();
    return false;
  }

  template <class _Tuple>
  _LIBCPP_HIDE_FROM_ABI constexpr difference_type __distance_from(const _Tuple& __t) const {
    return __scaled_sum(__t);
  }

  template <auto _Np>
  _LIBCPP_HIDE_FROM_ABI constexpr difference_type __scaled_size() const {
    if constexpr (_Np <= sizeof...(_Vs))
      return static_cast<difference_type>(ranges::size(std::get<_Np>(__parent_->__bases_))) * __scaled_size<_Np + 1>();
    return static_cast<difference_type>(1);
  }

  template <auto _Np>
  _LIBCPP_HIDE_FROM_ABI constexpr difference_type __scaled_distance(const auto& __t) const {
    return static_cast<difference_type>(std::get<_Np>(__current_) - std::get<_Np>(__t)) * __scaled_size<_Np + 1>();
  }

  template <auto _Np = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr difference_type __scaled_sum(const auto& __t) const {
    if constexpr (_Np <= sizeof...(_Vs))
      return __scaled_distance<_Np>(__t) + __scaled_sum<_Np + 1>(__t);
    return static_cast<difference_type>(0);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(_Parent& __parent, _MultiIter __current)
      : __parent_(std::addressof(__parent)), __current_(std::move(__current)) {}
};

namespace views {
namespace __cartesian_product {
struct __fn {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()() { return views::single(tuple()); }

  template <class... _Ranges>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static constexpr auto operator()(_Ranges&&... __rs)
      -> decltype(cartesian_product_view<all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...)) {
    return cartesian_product_view<all_t<_Ranges>...>(std::forward<_Ranges>(__rs)...);
  }
};
} // namespace __cartesian_product
inline namespace __cpo {
inline constexpr auto cartesian_product = __cartesian_product::__fn{};
} // namespace __cpo
} // namespace views
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H
#define _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

#include <__config>
#include <__ranges/concepts.h> // forward_range, view, range_size_t, sized_range, ...
#include <tuple>               // apply
#include <type_traits>         // common_type_t
#include <__type_traits/maybe_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

template <input_range First, forward_range... Vs>
  requires(view<First> && ... && view<Vs>)
class cartesian_product_view : public view_interface<cartesian_product_view<First, Vs...>> {
private:
public: // fixme: remove me
  tuple<First, Vs...> bases_; // exposition only

  template <bool Const>
  class iterator; // exposition only

public:
  constexpr cartesian_product_view() = default;
  constexpr explicit cartesian_product_view(First first_base, Vs... bases)
      : bases_{std::move(first_base), std::move(bases)...} {}

  // constexpr iterator<false> begin()
  //     requires (!simple-view<First> || ... || !simple-view<Vs>)
  // {
  //     return move_iterator(ranges::begin(bases_));
  // }

  // constexpr iterator<true> begin() const
  //     requires (range<const First> && ... && range<const Vs>)
  // {
  //     return move_iterator(ranges::begin(__base_));
  // }

  // constexpr iterator<false> begin()
  //   requires (!simple-view<First> || ... || !simple-view<Vs>);
  // constexpr iterator<true> begin() const
  //   requires (range<const First> && ... && range<const Vs>);

  // constexpr iterator<false> end()
  //   requires ((!simple-view<First> || ... || !simple-view<Vs>) &&
  //     cartesian-product-is-common<First, Vs...>);
  // constexpr iterator<true> end() const
  //   requires cartesian-product-is-common<const First, const Vs...>;
  // constexpr default_sentinel_t end() const noexcept;

  constexpr auto size()
    requires(sized_range<First> && ... && sized_range<Vs>)
  {
    return size_impl();
  }

  constexpr auto size() const
    requires(sized_range<const First> && ... && sized_range<const Vs>)
  {
    return size_impl();
  }

private:
  constexpr auto size_impl() const {
    return std::apply(
        [](auto&&... bases) {
          using size_type = std::common_type_t<std::ranges::range_size_t<decltype(bases)>...>;
          return (static_cast<size_type>(std::ranges::size(bases)) * ...);
        },
        bases_);
  }
};

template<input_range First, forward_range... Vs>
requires (view<First> && ... && view<Vs>)
template<bool Const>
class cartesian_product_view<First, Vs...>::iterator {

  // fixme: implement properly. see section 5.2 in paper.
  template <class... T>
  using tuple_or_pair = tuple<T...>;

public:
//   using iterator_category = input_iterator_tag;
//   using iterator_concept  = see below;
//   using value_type = tuple-or-pair<range_value_t<__maybe_const<Const, First>>,
//     range_value_t<__maybe_const<Const, Vs>>...>;
//   using reference = tuple-or-pair<reference_t<__maybe_const<Const, First>>,
//     reference_t<__maybe_const<Const, Vs>>...>;
//   using difference_type = see below;

  iterator() requires forward_range<__maybe_const<Const, First>> = default;

  constexpr explicit iterator(tuple_or_pair<iterator_t<__maybe_const<Const, First>>,
    iterator_t<__maybe_const<Const, Vs>>...> current) 
      : current_(std::move(current)) {}

  constexpr iterator(iterator<!Const> i) requires Const &&
    (convertible_to<iterator_t<First>, iterator_t<__maybe_const<Const, First>>> &&
      ... && convertible_to<iterator_t<Vs>, iterator_t<__maybe_const<Const, Vs>>>)
      : current_(std::move(i.current_)) {}

//   constexpr auto operator*() const;
//   constexpr iterator& operator++();
//   constexpr void operator++(int);
//   constexpr iterator operator++(int) requires forward_range<__maybe_const<Const, First>>;

//   constexpr iterator& operator--()
//     requires cartesian-product-is-bidirectional<Const, First, Vs...>;
//   constexpr iterator operator--(int)
//     requires cartesian-product-is-bidirectional<Const, First, Vs...>;

//   constexpr iterator& operator+=(difference_type x)
//     requires cartesian-product-is-random-access<Const, First, Vs...>;
//   constexpr iterator& operator-=(difference_type x)
//     requires cartesian-product-is-random-access<Const, First, Vs...>;

//   constexpr reference operator[](difference_type n) const
//     requires cartesian-product-is-random-access<Const, First, Vs...>;

//   friend constexpr bool operator==(const iterator& x, const iterator& y)
//     requires equality_comparable<iterator_t<__maybe_const<Const, First>>>;

//   friend constexpr bool operator==(const iterator& x, default_sentinel_t);

//   friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//     requires all-random-access<Const, First, Vs...>;

//   friend constexpr iterator operator+(const iterator& x, difference_type y)
//     requires cartesian-product-is-random-access<Const, First, Vs...>;
//   friend constexpr iterator operator+(difference_type x, const iterator& y)
//     requires cartesian-product-is-random-access<Const, First, Vs...>;
//   friend constexpr iterator operator-(const iterator& x, difference_type y)
//     requires cartesian-product-is-random-access<Const, First, Vs...>;
//   friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//     requires cartesian-is-sized-sentinel<Const, iterator_t, First, Vs...>;

//   friend constexpr difference_type operator-(iterator i, default_sentinel_t)
//     requires cartesian-is-sized-sentinel<Const, sentinel_t, First, Vs...>;
//   friend constexpr difference_type operator-(default_sentinel_t, iterator i)
//     requires cartesian-is-sized-sentinel<Const, sentinel_t, First, Vs...>;

//   friend constexpr auto iter_move(const iterator& i) noexcept(see below);

//   friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//     requires (indirectly_swappable<iterator_t<__maybe_const<Const, First>>> && ... &&
//       indirectly_swappable<iterator_t<__maybe_const<Const, Vs>>>);

private:
//   __maybe_const<Const, cartesian_product_view>* parent_ = nullptr; // exposition only
  tuple_or_pair<iterator_t<__maybe_const<Const, First>>,
    iterator_t<__maybe_const<Const, Vs>>...> current_; // exposition only

//   template <size_t N = sizeof...(Vs)>
//   constexpr void next(); // exposition only

//   template <size_t N = sizeof...(Vs)>
//   constexpr void prev(); // exposition only

//   template <class Tuple>
//   constexpr difference_type distance-from(Tuple t); // exposition only

//   constexpr explicit iterator(tuple-or-pair<iterator_t<__maybe_const<Const, First>>,
//     iterator_t<__maybe_const<Const, Vs>>...> current); // exposition only
};

template <class... Rs>
cartesian_product_view(Rs&&...) -> cartesian_product_view<std::views::all_t<Rs>...>;

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

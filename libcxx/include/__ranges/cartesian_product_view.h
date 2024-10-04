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
#include <tuple> // apply
#include <type_traits> // common_type_t

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
  tuple<First, Vs...> bases_; // exposition only

  template <bool Const>
  struct iterator; // exposition only

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

  // constexpr see below size()
  //   requires cartesian-product-is-sized<First, Vs...>;
  // constexpr see below size() const
  // requires cartesian-product-is-sized<const First, const Vs...>;
  constexpr auto size() const 
    requires(sized_range<const First> && ... && sized_range<const Vs>)
  {
    return std::apply(
        [](auto&&... bases) {
          using size_type = std::common_type_t<std::ranges::range_size_t<decltype(bases)>...>;
          return (static_cast<size_type>(std::ranges::size(bases)) * ...);
        },
        bases_);
  }
};

template <class... Rs>
cartesian_product_view(Rs&&...) -> cartesian_product_view<std::views::all_t<Rs>...>;

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

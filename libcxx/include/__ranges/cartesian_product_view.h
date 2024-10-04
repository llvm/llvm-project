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
#include <__memory/addressof.h>
#include <__ranges/concepts.h> // forward_range, view, range_size_t, sized_range, ...
#include <__type_traits/maybe_const.h>
#include <tuple>       // apply
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

public: // fixme: remove me
  tuple<First, Vs...> bases_;

  template <bool Const>
  class iterator;

public:
  constexpr cartesian_product_view() = default;
  constexpr explicit cartesian_product_view(First first_base, Vs... bases)
      : bases_{std::move(first_base), std::move(bases)...} {}

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

template <class... Vs>
cartesian_product_view(Vs&&...) -> cartesian_product_view<views::all_t<Vs>...>;

template <input_range First, forward_range... Vs>
  requires(view<First> && ... && view<Vs>)
template <bool Const>
class cartesian_product_view<First, Vs...>::iterator {
public:
  iterator() = default;

  constexpr iterator(iterator<!Const> i)
    requires Const && (convertible_to<iterator_t<First>, iterator_t<const First>> && ... &&
                       convertible_to<iterator_t<Vs>, iterator_t<const Vs>>)
      : parent_(std::addressof(i.parent_)), current_(std::move(i.current_)) {}

private:
  using Parent    = __maybe_const<Const, cartesian_product_view>;
  Parent* parent_ = nullptr;
  tuple<iterator_t<__maybe_const<Const, First>>, iterator_t<__maybe_const<Const, Vs>>...> current_;

  constexpr iterator(Parent& parent, decltype(current_) current)
      : parent_(std::addressof(parent)), current_(std::move(current)) {}
};

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

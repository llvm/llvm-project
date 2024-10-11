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
#include <__iterator/access.h> // begin
#include <__iterator/distance.h>
#include <__iterator/next.h>
#include <__memory/addressof.h>
#include <__ranges/concepts.h> // forward_range, view, range_size_t, sized_range, ...
#include <__ranges/zip_view.h> // tuple_transform
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

template <bool Const, class First, class... Vs>
concept cartesian_product_is_random_access =
    (random_access_range<__maybe_const<Const, First>> && ... &&
     (random_access_range<__maybe_const<Const, Vs>> && sized_range<__maybe_const<Const, Vs>>));

template <class R>
concept cartesian_product_common_arg = common_range<R> || (sized_range<R> && random_access_range<R>);

template <bool Const, class First, class... Vs>
concept cartesian_product_is_bidirectional =
    (bidirectional_range<__maybe_const<Const, First>> && ... &&
     (bidirectional_range<__maybe_const<Const, Vs>> && cartesian_product_common_arg<__maybe_const<Const, Vs>>));

template <cartesian_product_common_arg R>
constexpr auto cartesian_common_arg_end(R& r) {
  if constexpr (common_range<R>) {
    return ranges::end(r);
  } else {
    return ranges::begin(r) + ranges::distance(r);
  }
}

template <input_range First, forward_range... Vs>
  requires(view<First> && ... && view<Vs>)
class cartesian_product_view : public view_interface<cartesian_product_view<First, Vs...>> {
public: // fixme: make private
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
  using iterator_category = input_iterator_tag;
  using value_type = tuple<range_value_t<__maybe_const<Const, First>>, range_value_t<__maybe_const<Const, Vs>>...>;
  using reference =
      tuple<range_reference_t<__maybe_const<Const, First>>, range_reference_t<__maybe_const<Const, Vs>>...>;
  using difference_type = std::common_type_t<range_difference_t<First>, range_difference_t<Vs>...>;

  iterator() = default;

  constexpr iterator(iterator<!Const> i)
    requires Const && (convertible_to<iterator_t<First>, iterator_t<const First>> && ... &&
                       convertible_to<iterator_t<Vs>, iterator_t<const Vs>>)
      : parent_(std::addressof(i.parent_)), current_(std::move(i.current_)) {}

  constexpr auto operator*() const {
    return __tuple_transform([](auto& i) -> decltype(auto) { return *i; }, current_);
  }

  constexpr iterator& operator++() {
    next();
    return *this;
  }

  constexpr void operator++(int) { next(); }

  constexpr iterator operator++(int)
    requires forward_range<__maybe_const<Const, First>>
  {
    auto tmp = *this;
    next();
    return tmp;
  }

  constexpr iterator& operator--()
    requires cartesian_product_is_bidirectional<Const, First, Vs...>
  {
    prev();
    return *this;
  }

  constexpr iterator operator--(int)
    requires cartesian_product_is_bidirectional<Const, First, Vs...>
  {
    auto tmp = *this;
    prev();
    return tmp;
  }

  constexpr iterator& operator+=(difference_type x)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    advance(x);
    return *this;
  }

  constexpr iterator& operator-=(difference_type x)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    *this += -x;
    return *this;
  }

  constexpr reference operator[](difference_type n) const
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    return *((*this) + n);
  }

  friend constexpr bool operator==(const iterator& x, const iterator& y)
    requires equality_comparable<iterator_t<__maybe_const<Const, First>>>
  {
    return x.current_ == y.current_;
  }

  friend constexpr bool operator==(const iterator& x, default_sentinel_t) { return x.at_end(); }

private:
  using Parent    = __maybe_const<Const, cartesian_product_view>;
  Parent* parent_ = nullptr;
  tuple<iterator_t<__maybe_const<Const, First>>, iterator_t<__maybe_const<Const, Vs>>...> current_;

  constexpr iterator(Parent& parent, decltype(current_) current)
      : parent_(std::addressof(parent)), current_(std::move(current)) {}

  template <auto N = sizeof...(Vs)>
  constexpr void next() {
    auto& it = std::get<N>(current_);
    ++it;
    if constexpr (N > 0) {
      if (const auto& v = std::get<N>(parent_->bases_); it == ranges::end(v)) {
        it = ranges::begin(v);
        next<N - 1>();
      }
    }
  }

  template <auto N = sizeof...(Vs)>
  constexpr void prev() {
    auto& it = std::get<N>(current_);
    if constexpr (N > 0) {
      if (const auto& v = std::get<N>(parent_->bases_); it == ranges::begin(v)) {
        it = cartesian_common_arg_end(v);
        prev<N - 1>();
      }
    }
    --it;
  }

  template <std::size_t N = sizeof...(Vs)>
  constexpr void advance(difference_type x) {
    if (x == 0)
      return;

    const auto& v    = std::get<N>(parent_->bases_);
    auto& it         = std::get<N>(current_);
    const auto sz    = static_cast<difference_type>(std::ranges::size(v));
    const auto first = begin(v);

    if (sz > 0) {
      const auto idx = static_cast<difference_type>(std::distance(first, it));
      x += idx;

      difference_type mod;
      if constexpr (N > 0) {
        difference_type div = x / sz;
        mod                 = x % sz;
        if (mod < 0) {
          mod += sz;
          div--;
        }
        advance<N - 1>(div);
      } else {
        mod = (x >= 0 && x < sz) ? x : sz;
      }
      it = std::next(first, mod);

    } else {
      if constexpr (N > 0) {
        advance<N - 1>(x);
      }
      it = first;
    }
  }

  template <auto N = sizeof...(Vs)>
  constexpr bool at_end() const {
    if (std::get<N>(current_) == end(std::get<N>(parent_->bases_)))
      return true;
    if constexpr (N > 0)
      return at_end<N - 1>();
    return false;
  }
};

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

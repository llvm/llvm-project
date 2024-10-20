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
#include <__iterator/iter_move.h>
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

template <class First, class... Vs>
concept cartesian_product_is_common = cartesian_product_common_arg<First>;

template <class... Vs>
concept cartesian_product_is_sized = (sized_range<Vs> && ...);

template <bool Const, template <class> class FirstSent, class First, class... Vs>
concept cartesian_is_sized_sentinel =
    (sized_sentinel_for<FirstSent<__maybe_const<Const, First>>, iterator_t<__maybe_const<Const, First>>> && ... &&
     (sized_range<__maybe_const<Const, Vs>> &&
      sized_sentinel_for<iterator_t<__maybe_const<Const, Vs>>, iterator_t<__maybe_const<Const, Vs>>>));

template <cartesian_product_common_arg R>
constexpr auto cartesian_common_arg_end(R& r) {
  if constexpr (common_range<R>) {
    return ranges::end(r);
  } else {
    return ranges::begin(r) + ranges::distance(r);
  }
}

template <bool Const, class First, class... Vs>
concept __cartesian_product_all_random_access =
    (random_access_range<__maybe_const<Const, First>> && ... && random_access_range<__maybe_const<Const, Vs>>);

template <input_range First, forward_range... Vs>
  requires(view<First> && ... && view<Vs>)
class cartesian_product_view : public view_interface<cartesian_product_view<First, Vs...>> {
private:
  tuple<First, Vs...> bases_;

  template <bool Const>
  class iterator;

public:
  constexpr cartesian_product_view() = default;
  constexpr explicit cartesian_product_view(First first_base, Vs... bases)
      : bases_{std::move(first_base), std::move(bases)...} {}

  constexpr iterator<false> begin()
    requires(!__simple_view<First> || ... || !__simple_view<Vs>)
  {
    return iterator<false>(*this, __tuple_transform(ranges::begin, bases_));
  }

  constexpr iterator<true> begin() const
    requires(range<const First> && ... && range<const Vs>)
  {
    return iterator<true>(*this, __tuple_transform(ranges::begin, bases_));
  }

  constexpr iterator<false> end()
    requires((!__simple_view<First> || ... || !__simple_view<Vs>) && cartesian_product_is_common<First, Vs...>)
  {
    constexpr bool is_const = false;
    return end_impl<is_const>();
  }

  constexpr iterator<true> end() const
    requires(cartesian_product_is_common<const First, const Vs...>)
  {
    constexpr bool is_const = true;
    return end_impl<is_const>();
  }

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
  template <bool is_const>
  constexpr iterator<is_const> end_impl() const {
    bool is_empty                  = end_is_empty();
    const auto ranges_to_iterators = [is_empty, &b = bases_]<std::size_t... I>(std::index_sequence<I...>) {
      const auto begin_or_first_end = [is_empty]<bool is_first>(const auto& rng) {
        if constexpr (is_first)
          return is_empty ? ranges::begin(rng) : cartesian_common_arg_end(rng);
        return ranges::begin(rng);
      };
      return std::make_tuple(begin_or_first_end<I == 0>(std::get<I>(b))...);
    };
    iterator<is_const> it(*this, ranges_to_iterators(std::make_index_sequence<1 + sizeof...(Vs)>{}));
    return it;
  }

  template <auto N = 0>
  constexpr bool end_is_empty() const {
    if constexpr (N == sizeof...(Vs))
      return false;
    if (const auto& v = std::get<N + 1>(bases_); ranges::empty(v))
      return true;
    return end_is_empty<N + 1>();
  }

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

  friend constexpr auto operator<=>(const iterator& x, const iterator& y)
    requires __cartesian_product_all_random_access<Const, First, Vs...>
  {
    return x.current_ <=> y.current_;
  }

  friend constexpr iterator operator+(const iterator& x, difference_type y)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    return iterator(x) += y;
  }

  friend constexpr iterator operator+(difference_type x, const iterator& y)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    return y + x;
  }

  friend constexpr iterator operator-(const iterator& x, difference_type y)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    return iterator(x) -= y;
  }

  friend constexpr iterator operator-(const iterator& x, const iterator& y)
    requires cartesian_product_is_random_access<Const, First, Vs...>
  {
    return x.distance_from(y.current_);
  }

  friend constexpr difference_type operator-(const iterator& i, default_sentinel_t)
    requires cartesian_is_sized_sentinel<Const, sentinel_t, First, Vs...>
  {
    MultiIterator end_tuple;
    std::get<0>(end_tuple) = ranges::end(std::get<0>(i.parent_->bases_));
    for (int N = 1; N <= sizeof...(Vs); N++)
      std::get<N>(end_tuple) = ranges::begin(std::get<N>(i.parent_->bases_));
    return i.distance_from(end_tuple);
  }

  friend constexpr difference_type operator-(default_sentinel_t s, const iterator& i)
    requires cartesian_is_sized_sentinel<Const, sentinel_t, First, Vs...>
  {
    return -(i - s);
  }

  friend constexpr auto iter_move(const iterator& i) /*fixme: noexcept(...) */ {
    return __tuple_transform(ranges::iter_move, i.current_);
  }

  friend constexpr void iter_swap(const iterator& l, const iterator& r) /*fixme: noexcept(...) */
    requires(indirectly_swappable<iterator_t<__maybe_const<Const, First>>> && ... &&
             indirectly_swappable<iterator_t<__maybe_const<Const, Vs>>>)
  {
    iter_swap_helper(l, r);
  }

private:
  using Parent    = __maybe_const<Const, cartesian_product_view>;
  Parent* parent_ = nullptr;
  using MultiIterator = tuple<iterator_t<__maybe_const<Const, First>>, iterator_t<__maybe_const<Const, Vs>>...>;
  MultiIterator current_;

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
    const auto first = ranges::begin(v);

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
    if (std::get<N>(current_) == ranges::end(std::get<N>(parent_->bases_)))
      return true;
    if constexpr (N > 0)
      return at_end<N - 1>();
    return false;
  }

  template <class Tuple>
  constexpr difference_type distance_from(const Tuple& t) const {
    return scaled_sum(t);
  }

  template <auto N>
  constexpr difference_type scaled_size() const {
    if constexpr (N <= sizeof...(Vs))
      return static_cast<difference_type>(ranges::size(std::get<N>(parent_->bases_))) * scaled_size<N + 1>();
    return static_cast<difference_type>(1);
  }

  template <auto N>
  constexpr difference_type scaled_distance(const auto& t) const {
    return static_cast<difference_type>(std::get<N>(current_) - std::get<N>(t)) * scaled_size<N + 1>();
  }

  template <auto N = 0>
  constexpr difference_type scaled_sum(const auto& t) const {
    if constexpr (N <= sizeof...(Vs))
      return scaled_distance<N>(t) + scaled_sum<N + 1>(t);
    return static_cast<difference_type>(0);
  }

  template <auto N = sizeof...(Vs)>
  static constexpr void iter_swap_helper(const iterator& l, const iterator& r) {
    ranges::iter_swap(std::get<N>(l.current_), std::get<N>(r.current_));
    if constexpr (N > 0)
      iter_swap_helper<N - 1>(l, r);
  }
};

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___RANGES_CARTESIAN_PRODUCT_VIEW_H

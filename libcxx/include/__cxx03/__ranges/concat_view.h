// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___RANGES_CONCAT_VIEW_H
#define _LIBCPP___CXX03___RANGES_CONCAT_VIEW_H

#include <__cxx03/__algorithm/ranges_find_if.h>
#include <__cxx03/__assert>
#include <__cxx03/__concepts/constructible.h>
#include <__cxx03/__concepts/copyable.h>
#include <__cxx03/__concepts/derived_from.h>
#include <__cxx03/__concepts/equality_comparable.h>
#include <__cxx03/__config>
#include <__cxx03/__functional/bind_back.h>
#include <__cxx03/__functional/invoke.h>
#include <__cxx03/__functional/reference_wrapper.h>
#include <__cxx03/__iterator/concepts.h>
#include <__cxx03/__iterator/default_sentinel.h>
#include <__cxx03/__iterator/distance.h>
#include <__cxx03/__iterator/iter_move.h>
#include <__cxx03/__iterator/iter_swap.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__iterator/next.h>
#include <__cxx03/__memory/addressof.h>
#include <__cxx03/__ranges/access.h>
#include <__cxx03/__ranges/all.h>
#include <__cxx03/__ranges/concepts.h>
#include <__cxx03/__ranges/movable_box.h>
#include <__cxx03/__ranges/non_propagating_cache.h>
#include <__cxx03/__ranges/range_adaptor.h>
#include <__cxx03/__ranges/view_interface.h>
#include <__cxx03/__type_traits/conditional.h>
#include <__cxx03/__type_traits/decay.h>
#include <__cxx03/__type_traits/is_nothrow_constructible.h>
#include <__cxx03/__type_traits/is_nothrow_convertible.h>
#include <__cxx03/__type_traits/is_object.h>
#include <__cxx03/__type_traits/maybe_const.h>
#include <__cxx03/__utility/forward.h>
#include <__cxx03/__utility/in_place.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/tuple>
#include <__cxx03/variant>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

namespace ranges {

template <class T, class... Ts>
struct extract_last : extract_last<Ts...> {};

template <class T>
struct extract_last<T> {
  using type = T;
};

template <class _T, class... _Ts>
struct derived_from_pack {
  constexpr static bool value =
      derived_from_pack<_T, typename extract_last<_Ts...>::type>::value && derived_from_pack<_Ts...>::value;
};

template <class _T, class _IterCategory>
struct derived_from_pack<_T, _IterCategory> {
  constexpr static bool value = derived_from<_T, _IterCategory>;
};

template <class View, class... Views>
struct last_view : last_view<Views...> {};

template <class View>
struct last_view<View> {
  using type = View;
};

template <class Ref, class RRef, class It>
concept concat_indirectly_readable_impl = requires(const It it) {
  { *it } -> convertible_to<Ref>;
  { ranges::iter_move(it) } -> convertible_to<RRef>;
};

template <class... Rs>
using concat_reference_t = common_reference_t<range_reference_t<Rs>...>;

template <class... Rs>
using concat_value_t = common_type_t<range_value_t<Rs>...>;

template <class... Rs>
using concat_rvalue_reference_t = common_reference_t<range_rvalue_reference_t<Rs>...>;

template <class... Rs>
concept concat_indirectly_readable =
    common_reference_with<concat_reference_t<Rs...>&&, concat_value_t<Rs...>&> &&
    common_reference_with<concat_reference_t<Rs...>&&, concat_rvalue_reference_t<Rs...>&&> &&
    common_reference_with<concat_rvalue_reference_t<Rs...>&&, concat_value_t<Rs...> const&> &&
    (concat_indirectly_readable_impl<concat_reference_t<Rs...>, concat_rvalue_reference_t<Rs...>, iterator_t<Rs>> &&
     ...);

template <class... Rs>
concept concatable = requires { // exposition only
  typename concat_reference_t<Rs...>;
  typename concat_value_t<Rs...>;
  typename concat_rvalue_reference_t<Rs...>;
} && concat_indirectly_readable<Rs...>;

template <bool Const, class... Rs>
concept concat_is_random_access =
    (random_access_range<__maybe_const<Const, Rs>> && ...) && (sized_range<__maybe_const<Const, Rs>> && ...);

template <class R>
concept constant_time_reversible = // exposition only
    (bidirectional_range<R> && common_range<R>) || (sized_range<R> && random_access_range<R>);

template <bool Const, class... Rs>
concept concat_is_bidirectional =
    ((bidirectional_range<__maybe_const<Const, Rs>> && ...) &&
     (constant_time_reversible<__maybe_const<Const, Rs>> && ...));

template <bool Const, class... Views>
concept all_forward = // exposition only
    (forward_range<__maybe_const<Const, Views>> && ...);

template <bool Const, class... Ts>
struct apply_drop_first;

template <bool Const, class Head, class... Tail>
struct apply_drop_first<Const, Head, Tail...> {
  static constexpr bool value = (sized_range<__maybe_const<Const, Tail>> && ...);
};

template <input_range... Views>
  requires(view<Views> && ...) && (sizeof...(Views) > 0) && concatable<Views...>
class concat_view : public view_interface<concat_view<Views...>> {
  tuple<Views...> views_;

  template <bool Const>
  class iterator;
  class sentinel;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr concat_view() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit concat_view(Views... views) : views_(std::move(views)...) {}

  constexpr iterator<false> begin()
    requires(!(__simple_view<Views> && ...))
  {
    iterator<false> it(this, in_place_index<0>, ranges::begin(get<0>(views_)));
    it.template satisfy<0>();
    return it;
  }

  constexpr iterator<true> begin() const
    requires((range<const Views> && ...) && concatable<const Views...>)
  {
    iterator<true> it(this, in_place_index<0>, ranges::begin(get<0>(views_)));
    it.template satisfy<0>();
    return it;
  }

  constexpr auto end()
    requires(!(__simple_view<Views> && ...))
  {
    if constexpr (common_range<typename last_view<Views...>::type>) {
      // last_view to be implemented
      constexpr auto N = sizeof...(Views);
      return iterator<false>(this, in_place_index<N - 1>, ranges::end(get<N - 1>(views_)));
    } else {
      return default_sentinel;
    }
  }

  constexpr auto end() const
    requires(range<const Views> && ...)
  {
    if constexpr (common_range<typename last_view<Views...>::type>) {
      // last_view to be implemented
      constexpr auto N = sizeof...(Views);
      return iterator<true>(this, in_place_index<N - 1>, ranges::end(get<N - 1>(views_)));
    } else {
      return default_sentinel;
    }
  }

  constexpr auto size()
    requires(sized_range<Views> && ...)
  {
    return apply(
        [](auto... sizes) {
          using CT = make_unsigned_t<common_type_t<decltype(sizes)...>>;
          return (CT(sizes) + ...);
        },
        tuple_transform(ranges::size, views_));
  }

  constexpr auto size() const
    requires(sized_range<const Views> && ...)
  {
    return apply(
        [](auto... sizes) {
          using CT = make_unsigned_t<common_type_t<decltype(sizes)...>>;
          return (CT(sizes) + ...);
        },
        tuple_transform(ranges::size, views_));
  }
};

template <class... _Views>
concat_view(_Views&&...) -> concat_view<views::all_t<_Views>...>;

template <input_range... Views>
  requires(view<Views> && ...) && (sizeof...(Views) > 0) && concatable<Views...>
template <bool Const>
class concat_view<Views...>::iterator {
public:
  constexpr static bool derive_pack_random_iterator =
      derived_from_pack<typename iterator_traits<iterator_t<__maybe_const<Const, Views>>>::iterator_category...,
                        random_access_iterator_tag>::value;
  constexpr static bool derive_pack_bidirectional_iterator =
      derived_from_pack<typename iterator_traits<iterator_t<__maybe_const<Const, Views>>>::iterator_category...,
                        bidirectional_iterator_tag>::value;
  constexpr static bool derive_pack_forward_iterator =
      derived_from_pack<typename iterator_traits< iterator_t<__maybe_const<Const, Views>>>::iterator_category...,
                        forward_iterator_tag>::value;
  using iterator_category =
      _If<!is_reference_v<concat_reference_t<__maybe_const<Const, Views>...>>,
          input_iterator_tag,
          _If<derive_pack_random_iterator,
              random_access_iterator_tag,
              _If<derive_pack_bidirectional_iterator,
                  bidirectional_iterator_tag,
                  _If<derive_pack_forward_iterator, forward_iterator_tag, input_iterator_tag > > > >;
  using iterator_concept =
      _If<concat_is_random_access<Const, Views...>,
          random_access_iterator_tag,
          _If<concat_is_bidirectional<Const, Views...>,
              bidirectional_iterator_tag,
              _If< all_forward<Const, Views...>, forward_iterator_tag, input_iterator_tag > > >;
  using value_type      = concat_value_t<__maybe_const<Const, Views>...>;
  using difference_type = common_type_t<range_difference_t<__maybe_const<Const, Views>>...>;
  using base_iter       = variant<iterator_t<__maybe_const<Const, Views>>...>;

  base_iter it_;                                        // exposition only
  __maybe_const<Const, concat_view>* parent_ = nullptr; // exposition only

  template <std::size_t N>
  constexpr void satisfy() {
    if constexpr (N < (sizeof...(Views) - 1)) {
      if (get<N>(it_) == ranges::end(get<N>(parent_->views_))) {
        it_.template emplace<N + 1>(ranges::begin(get<N + 1>(parent_->views_)));
        satisfy<N + 1>();
      }
    }
  }

  template <std::size_t N>
  constexpr void prev() {
    if constexpr (N == 0) {
      --get<0>(it_);
    } else {
      if (get<N>(it_) == ranges::begin(get<N>(parent_->views_))) {
        using prev_view = __maybe_const<Const, tuple_element_t<N - 1, tuple<Views...>>>;
        if constexpr (common_range<prev_view>) {
          it_.emplace<N - 1>(ranges::end(get<N - 1>(parent_->views_)));
        } else {
          it_.emplace<N - 1>(
              ranges::__next(ranges::begin(get<N - 1>(parent_->views_)), ranges::size(get<N - 1>(parent_->views_))));
        }
        prev<N - 1>();
      } else {
        --get<N>(it_);
      }
    }
  }

  template <std::size_t N>
  constexpr void advance_fwd(difference_type offset, difference_type steps) {
    using underlying_diff_type = iter_difference_t<variant_alternative_t<N, base_iter>>;
    if constexpr (N == sizeof...(Views) - 1) {
      get<N>(it_) += static_cast<underlying_diff_type>(steps);
    } else {
      difference_type n_size = ranges::size(get<N>(parent_->views_));
      if (offset + steps < n_size) {
        get<N>(it_) += static_cast<underlying_diff_type>(steps);
      } else {
        it_.template emplace<N + 1>(ranges::begin(get<N + 1>(parent_->views_)));
        advance_fwd<N + 1>(0, offset + steps - n_size);
      }
    }
  }

  template <std::size_t N>
  constexpr void advance_bwd(difference_type offset, difference_type steps) {
    using underlying_diff_type = iter_difference_t<variant_alternative_t<N, base_iter>>;
    if constexpr (N == 0) {
      get<N>(it_) -= static_cast<underlying_diff_type>(steps);
    } else {
      if (offset >= steps) {
        get<N>(it_) -= static_cast<underlying_diff_type>(steps);
      } else {
        auto prev_size = ranges::__distance(get<N - 1>(parent_->views_));
        it_.emplace<N - 1>(ranges::begin(get<N - 1>(parent_->views_)) + prev_size);
        advance_bwd<N - 1>(prev_size, steps - offset);
      }
    }
  }

  template <size_t... Is, typename Func>
  constexpr void apply_fn_with_const_index(size_t index, Func&& func, std::index_sequence<Is...>) {
    ((index == Is ? (func(std::integral_constant<size_t, Is>{}), 0) : 0), ...);
  }

  template <size_t N, typename Func>
  constexpr void apply_fn_with_const_index(size_t index, Func&& func) {
    apply_fn_with_const_index(index, std::forward<Func>(func), std::make_index_sequence<N>{});
  }

  template <class... Args>
  explicit constexpr iterator(__maybe_const<Const, concat_view>* parent, Args&&... args)
    requires constructible_from<base_iter, Args&&...>
      : it_(std::forward<Args>(args)...), parent_(parent) {}

public:
  iterator() = default;

  constexpr iterator(iterator<!Const> i)
    requires Const && (convertible_to<iterator_t<Views>, iterator_t<const Views>> && ...)
      : it_(base_iter(in_place_index<i.index()>, std::get<i.index()>(std::move(i.it_)))), parent_(i.parent_) {}

  constexpr decltype(auto) operator*() const {
    using reference = concat_reference_t<__maybe_const<Const, Views>...>;
    return std::visit([](auto&& it) -> reference { return *it; }, it_);
  }

  constexpr iterator& operator++() {
    size_t active_index = it_.index();
    apply_fn_with_const_index<std::variant_size_v<decltype(it_)>>(active_index, [&](auto index_constant) {
      constexpr size_t i = index_constant.value;
      ++get<i>(it_);
      satisfy<i>();
    });
    return *this;
  }

  constexpr void operator++(int) { ++*this; }

  constexpr iterator operator++(int)
    requires(forward_range<__maybe_const<Const, Views>> && ...)
  {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  constexpr iterator& operator--()
    requires concat_is_bidirectional<Const, Views...>
  {
    size_t active_index = it_.index();
    apply_fn_with_const_index<std::variant_size_v<decltype(it_)>>(active_index, [&](auto index_constant) {
      constexpr size_t i = index_constant.value;
      prev<i>();
    });
    return *this;
  }

  constexpr iterator operator--(int)
    requires concat_is_bidirectional<Const, Views...>
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  friend constexpr bool operator==(const iterator& x, const iterator& y)
    requires(equality_comparable<iterator_t<__maybe_const<Const, Views>>> && ...)
  {
    return x.it_ == y.it_;
  }

  constexpr decltype(auto) operator[](difference_type n) const
    requires concat_is_random_access<Const, Views...>
  {
    return *((*this) + n);
  }

  friend constexpr iterator operator+(const iterator& it, difference_type n)
    requires concat_is_random_access<Const, Views...>
  {
    auto temp = it;
    temp += n;
    return temp;
  }

  friend constexpr iterator operator+(difference_type n, const iterator& it)
    requires concat_is_random_access<Const, Views...>
  {
    return it + n;
  }

  constexpr iterator& operator+=(difference_type n)
    requires concat_is_random_access<Const, Views...>
  {
    size_t active_index = it_.index();
    if (n > 0) {
      std::visit(
          [&](auto& active_it) {
            apply_fn_with_const_index<std::tuple_size_v<decltype(parent_->views_)>>(
                active_index, [&](auto index_constant) {
                  constexpr size_t I  = index_constant.value;
                  auto& active_view   = std::get<I>(parent_->views_);
                  difference_type idx = active_it - ranges::begin(active_view);
                  advance_fwd<I>(idx, n);
                });
          },
          it_);
    }

    else if (n < 0) {
      std::visit(
          [&](auto& active_it) {
            apply_fn_with_const_index<std::tuple_size_v<decltype(parent_->views_)>>(
                active_index, [&](auto index_constant) {
                  constexpr size_t I  = index_constant.value;
                  auto& active_view   = std::get<I>(parent_->views_);
                  difference_type idx = active_it - ranges::begin(active_view);
                  advance_bwd<I>(idx, -n);
                });
          },
          it_);
    }

    return *this;
  }

  constexpr iterator& operator-=(difference_type n)
    requires concat_is_random_access<Const, Views...>
  {
    *this += -n;
    return *this;
  }

  friend constexpr bool operator==(const iterator& it, default_sentinel_t) {
    constexpr auto last_idx = sizeof...(Views) - 1;
    return it.it_.index() == last_idx &&
           std::get<last_idx>(it.it_) == ranges::end(std::get<last_idx>(it.parent_->views_));
  }

  friend constexpr bool operator<(const iterator& x, const iterator& y)
    requires(random_access_range<__maybe_const<Const, Views>> && ...)
  {
    return x.it_ < y.it_;
  }

  friend constexpr bool operator>(const iterator& x, const iterator& y)
    requires(random_access_range<__maybe_const<Const, Views>> && ...)
  {
    return x.it_ > y.it_;
  }

  friend constexpr bool operator<=(const iterator& x, const iterator& y)
    requires(random_access_range<__maybe_const<Const, Views>> && ...)
  {
    return x.it_ <= y.it_;
  }

  friend constexpr bool operator>=(const iterator& x, const iterator& y)
    requires(random_access_range<__maybe_const<Const, Views>> && ...)
  {
    return x.it_ >= y.it_;
  }

  friend constexpr auto operator<=>(const iterator& x, const iterator& y)
    requires((random_access_range<__maybe_const<Const, Views>> && ...) &&
             (three_way_comparable<__maybe_const<Const, Views>> && ...))
  {
    return x.it_ <=> y.it_;
  }

  friend constexpr decltype(auto) iter_move(const iterator& it) noexcept(

      ((is_nothrow_invocable_v< decltype(ranges::iter_move), const iterator_t<__maybe_const<Const, Views>>& >) &&
       ...) &&
      ((is_nothrow_convertible_v< range_rvalue_reference_t<__maybe_const<Const, Views>>,
                                  concat_rvalue_reference_t<__maybe_const<Const, Views>...> >) &&
       ...))

  {
    return std::visit(
        [](const auto& i) -> concat_rvalue_reference_t<__maybe_const<Const, Views>...> { return ranges::iter_move(i); },
        it.it_);
  }

  friend constexpr void iter_swap(const iterator& x, const iterator& y)

      noexcept((noexcept(ranges::swap(*x, *y))) &&
               (noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<Const, Views>>>(),
                                           std::declval<const iterator_t<__maybe_const<Const, Views>>>())) &&
                ...))

    requires swappable_with<iter_reference_t<iterator>, iter_reference_t<iterator>> &&
             (... && indirectly_swappable<iterator_t<__maybe_const<Const, Views>>>)
  {
    std::visit(
        [&](const auto& it1, const auto& it2) {
          if constexpr (is_same_v<decltype(it1), decltype(it2)>) {
            ranges::iter_swap(it1, it2);
          } else {
            ranges::swap(*x, *y);
          }
        },
        x.it_,
        y.it_);
  }

  friend constexpr difference_type operator-(const iterator& x, const iterator& y)
    requires concat_is_random_access<Const, Views...>
  {
    size_t ix = x.it_.index();
    size_t iy = y.it_.index();

    if (ix > iy) {
      std::visit(
          [&](auto& it_x, auto& it_y) {
            it_x.apply_fn_with_const_index<std::tuple_size_v<decltype(x.parent_->views_)>>(
                ix, [&](auto index_constant) {
                  constexpr size_t index_x = index_constant.value;
                  auto dx = ranges::__distance(ranges::begin(std::get<index_x>(x.parent_->views_)), it_x);

                  it_y.apply_fn_with_const_index<std::tuple_size_v<decltype(y.parent_->views_)>>(
                      iy, [&](auto index_constant) {
                        constexpr size_t index_y = index_constant.value;
                        auto dy = ranges::__distance(ranges::begin(std::get<index_y>(y.parent_->views_)), it_y);
                        difference_type s = 0;
                        for (size_t idx = index_y + 1; idx < index_x; idx++) {
                          s += ranges::size(std::get<idx>(x.parent_->views_));
                        }
                        return dy + s + dx;
                      });
                });
          },
          x.it_,
          y.it_);
    } else if (ix < iy) {
      return -(y - x);
    } else {
      std::visit([&](const auto& it1, const auto& it2) { return it1 - it2; }, x.it_, y.it_);
    }
  }

  friend constexpr iterator operator-(const iterator& it, difference_type n)
    requires concat_is_random_access<Const, Views...>
  {
    auto temp = it;
    temp -= n;
    return temp;
  }

  friend constexpr difference_type operator-(const iterator& x, default_sentinel_t)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<Const, Views>>, iterator_t<__maybe_const<Const, Views>>> &&
             ...) &&
            (apply_drop_first<Const, Views...>::value)
  {
    size_t ix = x.it_.index();
    std::visit(
        [&](auto& it_x) {
          it_x.apply_fn_with_const_index<std::tuple_size_v<decltype(x.parent_->views_)>>(ix, [&](auto index_constant) {
            constexpr size_t index_x = index_constant.value;
            auto dx                  = ranges::__distance(ranges::begin(std::get<index_x>(x.parent_->views_)), it_x);

            difference_type s = 0;
            for (size_t idx = 0; idx < index_x; idx++) {
              s += ranges::size(std::get<idx>(x.parent_->views_));
            }

            return -(dx + s);
          });
        },
        x.it_);
  }

  friend constexpr difference_type operator-(default_sentinel_t, const iterator& x)
    requires(sized_sentinel_for<sentinel_t<__maybe_const<Const, Views>>, iterator_t<__maybe_const<Const, Views>>> &&
             ...) &&
            (apply_drop_first<Const, Views...>::value)
  {
    -(x - default_sentinel);
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

#endif /// _LIBCPP___CXX03___RANGES_CONCAT_VIEW_H

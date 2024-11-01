//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// concept checking
//
// template<class T, size_t N>
// concept has-tuple-element =
//   tuple-like<T> && N < tuple_size_v<T>;
//
// template<class T, size_t N>
// concept returnable-element =
//   is_reference_v<T> || move_constructible<tuple_element_t<N, T>>;
//
// template<input_range V, size_t N>
//   requires view<V> && has-tuple-element<range_value_t<V>, N> &&
//            has-tuple-element<remove_reference_t<range_reference_t<V>>, N> &&
//            returnable-element<range_reference_t<V>, N>
// class elements_view;

#include <array>
#include <concepts>
#include <tuple>
#include <ranges>
#include <utility>

#include "test_iterators.h"

template <class It>
using Range = std::ranges::subrange<It, sentinel_wrapper<It>>;

template <class V, size_t N>
concept HasElementsView = requires { typename std::ranges::elements_view<V, N>; };

static_assert(HasElementsView<Range<std::ranges::subrange<int*>*>, 0>);
static_assert(HasElementsView<Range<std::pair<int, int>*>, 1>);
static_assert(HasElementsView<Range<std::tuple<int, int, int>*>, 2>);
static_assert(HasElementsView<Range<std::array<int, 4>*>, 3>);

// !view<V>
static_assert(!std::ranges::view<std::array<std::tuple<int>, 1>>);
static_assert(!HasElementsView<std::array<std::tuple<int>, 1>, 0>);

// !input_range
static_assert(!std::ranges::input_range<Range<cpp20_output_iterator<std::tuple<int>*>>>);
static_assert(!HasElementsView<Range<cpp20_output_iterator<std::tuple<int>*>>, 0>);

// !tuple-like
LIBCPP_STATIC_ASSERT(!std::__tuple_like<int>);
static_assert(!HasElementsView<Range<int*>, 1>);

// !(N < tuple_size_v<T>)
static_assert(!(2 < std::tuple_size_v< std::pair<int, int>>));
static_assert(!HasElementsView<Range<std::pair<int, int>*>, 2>);

// ! (is_reference_v<T> || move_constructible<tuple_element_t<N, T>>)
struct NonMovable {
  NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};
static_assert(!std::move_constructible<NonMovable>);

using NonMovableGenerator =
    decltype(std::views::iota(0, 1) | std::views::transform([](int) {
               return std::pair<NonMovable, int>{1, 1};
             }));

static_assert(!HasElementsView<NonMovableGenerator, 0>);
static_assert(HasElementsView<NonMovableGenerator, 1>);

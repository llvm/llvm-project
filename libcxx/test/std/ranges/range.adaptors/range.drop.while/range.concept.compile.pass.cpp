//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// concept checking
// template<view V, class Pred>
//     requires input_range<V> && is_object_v<Pred> &&
//              indirect_unary_predicate<const Pred, iterator_t<V>>
//   class drop_while_view;

#include <array>
#include <ranges>

#include "test_iterators.h"

template <class It>
using Range = std::ranges::subrange<It, sentinel_wrapper<It>>;

template <class Val = int>
struct Pred {
  bool operator()(const Val&) const;
};

template <class V, class Pred>
concept HasDropWhileView = requires { typename std::ranges::drop_while_view<V, Pred>; };

static_assert(HasDropWhileView<Range<int*>, bool (*)(int)>);
static_assert(HasDropWhileView<Range<int*>, Pred<int>>);

// !view<V>
static_assert(!HasDropWhileView<std::array<int, 5>, Pred<int>>);

// !input_range
static_assert(!HasDropWhileView<Range<cpp20_output_iterator<int*>>, bool (*)(int)>);

// !is_object_v<Pred>
static_assert(!HasDropWhileView<Range<int*>, Pred<int>&>);

// !indirect_unary_predicate<const Pred, iterator_t<V>>
static_assert(!HasDropWhileView<Range<int*>, int>);
static_assert(!HasDropWhileView<Range<int**>, Pred<int>>);

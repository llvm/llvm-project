//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr outer-iterator(outer-iterator<!Const> i)
//   requires Const && convertible_to<iterator_t<View>, iterator_t<Base>>

#include <ranges>

#include <string_view>
#include <type_traits>
#include <utility>
#include "../types.h"

// outer-iterator<Const = true>

template <class Iter>
concept IsConstOuterIter = requires (Iter i) {
  { *(*i).begin() } -> std::same_as<const char&>;
};
static_assert( IsConstOuterIter<OuterIterConst>);

static_assert( std::convertible_to<
    std::ranges::iterator_t<SplitViewDiff>, std::ranges::iterator_t<const SplitViewDiff>>);

// outer-iterator<Const = false>

template <class Iter>
concept IsNonConstOuterIter = requires (Iter i) {
  { *(*i).begin() } -> std::same_as<char&>;
};
static_assert( IsNonConstOuterIter<OuterIterNonConst>);

static_assert(!std::is_constructible_v<OuterIterNonConst, OuterIterConst>);

constexpr bool test() {
  [[maybe_unused]] OuterIterConst i(OuterIterNonConst{});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

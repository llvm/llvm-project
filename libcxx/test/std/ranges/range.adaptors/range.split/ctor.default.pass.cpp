//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr split_view() requires
//   default_initializable<V> && default_initializable<Pattern> = default;

#include <cassert>
#include <ranges>
#include <type_traits>

template <bool DefaultInitializable>
struct View : std::ranges::view_base {
  int i = 42;
  constexpr explicit View()
    requires DefaultInitializable
  = default;
  int* begin() const;
  int* end() const;
};

// clang-format off
static_assert( std::is_default_constructible_v<std::ranges::split_view<View<true >, View<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::split_view<View<false>, View<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::split_view<View<true >, View<false>>>);
static_assert(!std::is_default_constructible_v<std::ranges::split_view<View<false>, View<false>>>);
// clang-format on

constexpr bool test() {
  {
    std::ranges::split_view<View<true>, View<true>> sv = {};
    assert(sv.base().i == 42);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

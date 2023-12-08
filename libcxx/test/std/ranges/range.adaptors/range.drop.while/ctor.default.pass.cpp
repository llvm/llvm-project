//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// drop_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

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

template <bool DefaultInitializable>
struct Pred {
  int i = 42;
  constexpr explicit Pred()
    requires DefaultInitializable
  = default;
  bool operator()(int) const;
};

// clang-format off
static_assert( std::is_default_constructible_v<std::ranges::drop_while_view<View<true >, Pred<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::drop_while_view<View<false>, Pred<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::drop_while_view<View<true >, Pred<false>>>);
static_assert(!std::is_default_constructible_v<std::ranges::drop_while_view<View<false>, Pred<false>>>);
// clang-format on

constexpr bool test() {
  {
    std::ranges::drop_while_view<View<true>, Pred<true>> dwv = {};
    assert(dwv.base().i == 42);
    assert(dwv.pred().i == 42);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

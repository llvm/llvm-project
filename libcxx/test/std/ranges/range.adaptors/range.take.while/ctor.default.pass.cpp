//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// take_while_view() requires default_initializable<V> && default_initializable<Pred> = default;

#include <cassert>
#include <ranges>
#include <type_traits>

template <bool defaultInitable>
struct View : std::ranges::view_base {
  int i;
  constexpr explicit View()
    requires defaultInitable
  = default;
  int* begin() const;
  int* end() const;
};

template <bool defaultInitable>
struct Pred {
  int i;
  constexpr explicit Pred()
    requires defaultInitable
  = default;
  bool operator()(int) const;
};

// clang-format off
static_assert( std::is_default_constructible_v<std::ranges::take_while_view<View<true >, Pred<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::take_while_view<View<false>, Pred<true >>>);
static_assert(!std::is_default_constructible_v<std::ranges::take_while_view<View<true >, Pred<false>>>);
static_assert(!std::is_default_constructible_v<std::ranges::take_while_view<View<false>, Pred<false>>>);
// clang-format on

constexpr bool test() {
  {
    std::ranges::take_while_view<View<true>, Pred<true>> twv = {};
    assert(twv.base().i == 0);
    assert(twv.pred().i == 0);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

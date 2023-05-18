//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr explicit sentinel(sentinel_t<Base> end);

#include <cassert>
#include <ranges>
#include <tuple>
#include <utility>

struct Sent {
  int i;

  friend constexpr bool operator==(std::tuple<int>*, const Sent&) { return true; }
};

struct Range : std::ranges::view_base {
  std::tuple<int>* begin() const;
  Sent end();
};

// Test explicit

static_assert(std::is_constructible_v<std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>, Sent>);
static_assert(!std::is_convertible_v<Sent, std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>>);

constexpr bool test() {
  // base is init correctly
  {
    using R        = std::ranges::elements_view<Range, 0>;
    using Sentinel = std::ranges::sentinel_t<R>;

    Sentinel s1(Sent{5});
    assert(s1.base().i == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

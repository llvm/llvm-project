//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// sentinel() = default;

#include <cassert>
#include <ranges>
#include <tuple>

struct PODSentinel {
  int i; // deliberately uninitialised

  friend constexpr bool operator==(std::tuple<int>*, const PODSentinel&) { return true; }
};

struct Range : std::ranges::view_base {
  std::tuple<int>* begin() const;
  PODSentinel end();
};

constexpr bool test() {
  using EleView  = std::ranges::elements_view<Range, 0>;
  using Sentinel = std::ranges::sentinel_t<EleView>;
  static_assert(!std::is_same_v<Sentinel, std::ranges::iterator_t<EleView>>);

  {
    Sentinel s;
    assert(s.base().i == 0);
  }
  {
    Sentinel s = {};
    assert(s.base().i == 0);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

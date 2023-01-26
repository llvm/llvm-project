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

struct Sent {
  bool b; // deliberately uninitialised

  friend constexpr bool operator==(int*, const Sent& s) { return s.b; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  Sent end();
};

constexpr bool test() {
  {
    using R        = std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = std::ranges::sentinel_t<R>;

    Sentinel s1 = {};
    assert(!s1.base().b);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

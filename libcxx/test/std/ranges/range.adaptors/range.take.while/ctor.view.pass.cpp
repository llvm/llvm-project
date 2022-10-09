//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr take_while_view(V base, Pred pred);

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"

struct View : std::ranges::view_base {
  MoveOnly mo;
  int* begin() const;
  int* end() const;
};

struct Pred {
  bool copied      = false;
  bool moved       = false;
  constexpr Pred() = default;
  constexpr Pred(Pred&&) : moved(true) {}
  constexpr Pred(const Pred&) : copied(true) {}
  bool operator()(int) const;
};

constexpr bool test() {
  {
    std::ranges::take_while_view<View, Pred> twv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(twv.pred().moved);
    assert(!twv.pred().copied);
    assert(std::move(twv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

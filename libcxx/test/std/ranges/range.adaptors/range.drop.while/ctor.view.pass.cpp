//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr drop_while_view(V base, Pred pred);

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
    std::ranges::drop_while_view<View, Pred> dwv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(dwv.pred().moved);
    assert(!dwv.pred().copied);
    assert(std::move(dwv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}

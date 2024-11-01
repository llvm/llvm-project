//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr value_type operator*() const;
//   Effects: Equivalent to return {cur_, next_.begin()};

#include <cassert>
#include <ranges>

#include "../types.h"

struct Iter : ForwardIterBase<Iter> {
  int i;
  constexpr Iter() = default;
  constexpr Iter(int ii) : i(ii) {}
};

constexpr bool test() {
  using SplitView = std::ranges::split_view<std::ranges::subrange<Iter>, std::ranges::subrange<Iter>>;
  using SplitIter = std::ranges::iterator_t<SplitView>;

  {
    SplitView sv;
    Iter current{5};
    std::ranges::subrange next{Iter{6}, Iter{7}};
    const SplitIter it{sv, current, next};
    std::same_as<std::ranges::subrange<Iter>> decltype(auto) value = *it;
    assert(value.begin().i == 5);
    assert(value.end().i == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

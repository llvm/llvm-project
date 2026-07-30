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
  int i            = 0;
  constexpr Iter() = default;
  constexpr Iter(int ii) : i(ii) {}
  constexpr int operator*() const { return i; }
  constexpr Iter& operator++() {
    ++i;
    return *this;
  }
  constexpr Iter operator++(int) {
    Iter tmp = *this;
    ++*this;
    return tmp;
  }
  friend constexpr bool operator==(const Iter& x, const Iter& y) { return x.i == y.i; }
};

constexpr bool test() {
  using SplitView = std::ranges::split_view<std::ranges::subrange<Iter>, std::ranges::subrange<Iter>>;
  using SplitIter = std::ranges::iterator_t<SplitView>;

  SplitView sv{std::ranges::subrange<Iter>{Iter{5}, Iter{8}}, std::ranges::subrange<Iter>{Iter{7}, Iter{8}}};
  const SplitIter it                                             = sv.begin();
  std::same_as<std::ranges::subrange<Iter>> decltype(auto) value = *it;
  assert(value.begin().i == 5);
  assert(value.end().i == 7);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

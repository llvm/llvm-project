//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator_t<V> base() const;

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
  // base only has one const overload
  using SplitView = std::ranges::split_view<std::ranges::subrange<Iter>, std::ranges::subrange<Iter>>;
  using SplitIter = std::ranges::iterator_t<SplitView>;

  SplitView sv{std::ranges::subrange<Iter>{Iter{5}, Iter{8}},
               std::ranges::subrange<Iter>{Iter{8}, Iter{9}}};

  // const &
  {
    const SplitIter it                     = sv.begin();
    std::same_as<Iter> decltype(auto) base = it.base();
    assert(base.i == 5);
  }

  // &
  {
    SplitIter it                           = sv.begin();
    std::same_as<Iter> decltype(auto) base = it.base();
    assert(base.i == 5);
  }

  // &&
  {
    SplitIter it                           = sv.begin();
    std::same_as<Iter> decltype(auto) base = std::move(it).base();
    assert(base.i == 5);
  }

  // const &&
  {
    const SplitIter it                     = sv.begin();
    std::same_as<Iter> decltype(auto) base = std::move(it).base();
    assert(base.i == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

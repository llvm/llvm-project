//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() = default;

#include <cassert>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

struct PODIter {
  int i; // deliberately uninitialised because we're testing that default ctor of the iterator zero initialises the underlying iterators

  using iterator_category = std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = std::intptr_t;

  constexpr int operator*() const { return i; }

  constexpr PODIter& operator++() { return *this; }
  constexpr PODIter operator++(int) { return *this; }

  friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
};

struct IterDefaultCtrView : std::ranges::view_base {
  PODIter begin() const;
  PODIter end() const;
};

struct IterNoDefaultCtrView : std::ranges::view_base {
  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class... Views>
using adjacent_iter = std::ranges::iterator_t<std::ranges::adjacent_view<Views...>>;

template <std::size_t N>
constexpr void test() {
  using View = std::ranges::adjacent_view<IterDefaultCtrView, N>;
  using Iter = std::ranges::iterator_t<View>;
  {
    Iter iter;
    auto tuple = *iter;
    assert((std::get<0>(tuple) == 0));
    if constexpr (N >= 2)
      assert((std::get<1>(tuple) == 0));
    if constexpr (N >= 3)
      assert((std::get<2>(tuple) == 0));
    if constexpr (N >= 4)
      assert((std::get<3>(tuple) == 0));
    if constexpr (N >= 5)
      assert((std::get<4>(tuple) == 0));
  }

  {
    Iter iter  = {};
    auto tuple = *iter;
    assert((std::get<0>(tuple) == 0));
    if constexpr (N >= 2)
      assert((std::get<1>(tuple) == 0));
    if constexpr (N >= 3)
      assert((std::get<2>(tuple) == 0));
    if constexpr (N >= 4)
      assert((std::get<3>(tuple) == 0));
    if constexpr (N >= 5)
      assert((std::get<4>(tuple) == 0));
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

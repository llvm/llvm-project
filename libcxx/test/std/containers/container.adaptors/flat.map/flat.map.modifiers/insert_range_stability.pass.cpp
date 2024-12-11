//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<container-compatible-range<value_type> R>
//   void insert_range(R&& rg);
//
// libc++ uses stable_sort to ensure that flat_map's behavior matches map's,
// in terms of which duplicate items are kept.
// This tests a conforming extension.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <flat_map>
#include <random>
#include <ranges>
#include <map>
#include <vector>
#include <utility>

#include "test_macros.h"

struct Mod256 {
  bool operator()(int x, int y) const { return (x % 256) < (y % 256); }
};

int main(int, char**) {
  {
    std::mt19937 randomness;
    std::pair<uint16_t, uint16_t> pairs[400];
    for (int i = 0; i < 400; ++i) {
      uint16_t r = randomness();
      pairs[i]   = {r, r};
    }

    std::map<uint16_t, uint16_t, Mod256> m(pairs, pairs + 200);
    std::flat_map<uint16_t, uint16_t, Mod256> fm(std::sorted_unique, m.begin(), m.end());
    assert(std::ranges::equal(fm, m));

    fm.insert_range(std::views::counted(pairs + 200, 200));
    m.insert(pairs + 200, pairs + 400);
    assert(fm.size() == m.size());
    LIBCPP_ASSERT(std::ranges::equal(fm, m));
  }

  {
    std::vector<std::pair<int, int>> v{{1, 2}, {1, 3}};
    std::flat_map<int, int> m;
    m.insert_range(v);
    assert(m.size() == 1);
    LIBCPP_ASSERT(m[1] == 2);
  }
  return 0;
}

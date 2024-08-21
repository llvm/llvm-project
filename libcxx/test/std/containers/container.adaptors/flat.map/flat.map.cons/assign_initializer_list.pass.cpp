//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map& operator=(initializer_list<value_type> il);

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <ranges>
#include <vector>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

int main(int, char**) {
  {
    using C = std::flat_map<int, int>;
    C m     = {{8, 8}, {10, 10}};
    assert(m.size() == 2);
    m                              = {{3, 0}, {1, 0}, {2, 0}, {2, 1}, {3, 1}, {4, 0}, {3, 2}, {5, 0}, {6, 0}, {5, 1}};
    std::pair<int, int> expected[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    using C = std::
        flat_map<int, int, std::less<>, std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>;
    C m = {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}, {8, 1}, {9, 1}, {10, 1}};
    assert(m.size() == 10);
    m                              = {{1, 1}, {3, 2}, {4, 3}, {5, 4}, {6, 5}, {5, 6}, {2, 7}};
    std::pair<int, int> expected[] = {{1, 1}, {2, 7}, {3, 2}, {4, 3}, {5, 4}, {6, 5}};
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    using C =
        std::flat_map<double,
                      int,
                      std::less<>,
                      std::deque<double, min_allocator<double>>,
                      std::vector<int, min_allocator<int>>>;
    C m = {};
    assert(m.size() == 0);
    m = {{3, 0}, {1, 0}, {2, 0}, {2, 1}, {3, 1}, {4, 0}, {3, 2}, {5, 0}, {6, 0}, {5, 1}};
    std::pair<double, int> expected[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};
    assert(std::ranges::equal(m.keys(), expected | std::views::elements<0>));
    LIBCPP_ASSERT(std::ranges::equal(m, expected));
  }
  {
    using C = std::flat_map<double, double, std::less<>, std::deque<double>, std::deque<double>>;
    C m     = {{10, 1}, {8, 1}};
    assert(m.size() == 2);
    m                                    = {{3, 2}};
    std::pair<double, double> expected[] = {{3, 2}};
    assert(std::ranges::equal(m, expected));
  }

  return 0;
}

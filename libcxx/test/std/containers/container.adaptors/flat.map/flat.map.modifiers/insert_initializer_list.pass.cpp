//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// void insert(initializer_list<value_type> il);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    using V                      = std::pair<const int, double>;
    std::flat_map<int, double> m = {{1, 1}, {1, 1.5}, {1, 2}, {3, 1}, {3, 1.5}, {3, 2}};
    m.insert({
        {4, 1},
        {4, 1.5},
        {4, 2},
        {1, 1},
        {1, 1.5},
        {1, 2},
        {2, 1},
        {2, 1.5},
        {2, 2},
    });
    assert(m.size() == 4);
    assert(std::distance(m.begin(), m.end()) == 4);
    assert(*m.begin() == V(1, 1));
    assert(*std::next(m.begin()) == V(2, 1));
    assert(*std::next(m.begin(), 2) == V(3, 1));
    assert(*std::next(m.begin(), 3) == V(4, 1));
  }
  {
    using V = std::pair<const int, double>;
    using M =
        std::flat_map<int,
                      double,
                      std::less<int>,
                      std::deque<int, min_allocator<int>>,
                      std::deque<double, min_allocator<double>>>;
    M m = {{1, 1}, {1, 1.5}, {1, 2}, {3, 1}, {3, 1.5}, {3, 2}};
    m.insert({
        {4, 1},
        {4, 1.5},
        {4, 2},
        {1, 1},
        {1, 1.5},
        {1, 2},
        {2, 1},
        {2, 1.5},
        {2, 2},
        {2, 1},
        {2, 1.5},
        {2, 2},
    });
    assert(m.size() == 4);
    assert(std::distance(m.begin(), m.end()) == 4);
    assert(*m.begin() == V(1, 1));
    assert(*std::next(m.begin()) == V(2, 1));
    assert(*std::next(m.begin(), 2) == V(3, 1));
    assert(*std::next(m.begin(), 3) == V(4, 1));
  }

  return 0;
}

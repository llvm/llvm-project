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
    m.insert(std::sorted_unique,
             {
                 {0, 1},
                 {1, 2},
                 {2, 1},
                 {4, 1},
             });
    assert(m.size() == 5);
    assert(std::distance(m.begin(), m.end()) == 5);
    assert(*m.begin() == V(0, 1));
    assert(*std::next(m.begin()) == V(1, 1));
    assert(*std::next(m.begin(), 2) == V(2, 1));
    assert(*std::next(m.begin(), 3) == V(3, 1));
    assert(*std::next(m.begin(), 4) == V(4, 1));
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
    m.insert(std::sorted_unique,
             {
                 {0, 1},
                 {1, 2},
                 {2, 1},
                 {4, 1},
             });
    assert(m.size() == 5);
    assert(std::distance(m.begin(), m.end()) == 5);
    assert(*m.begin() == V(0, 1));
    assert(*std::next(m.begin()) == V(1, 1));
    assert(*std::next(m.begin(), 2) == V(2, 1));
    assert(*std::next(m.begin(), 3) == V(3, 1));
    assert(*std::next(m.begin(), 4) == V(4, 1));
  }

  return 0;
}

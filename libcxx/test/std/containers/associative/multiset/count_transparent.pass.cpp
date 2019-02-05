//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <set>

// class multiset

// template<typename K>
//     iterator lower_bound(const K& x);              // C++14
// template<typename K>
//     const_iterator lower_bound(const K& x) const;  // C++14

#include <cassert>
#include <set>
#include <utility>

#include "min_allocator.h"
#include "private_constructor.hpp"
#include "test_macros.h"

struct Comp {
  using is_transparent = void;

  bool operator()(const std::pair<int, int> &lhs,
                  const std::pair<int, int> &rhs) const {
    return lhs < rhs;
  }

  bool operator()(const std::pair<int, int> &lhs, int rhs) const {
    return lhs.first < rhs;
  }

  bool operator()(int lhs, const std::pair<int, int> &rhs) const {
    return lhs < rhs.first;
  }
};

int main(int, char**) {
  std::multiset<std::pair<int, int>, Comp> s{{2, 1}, {1, 1}, {1, 1}, {1, 1}, {2, 2}};

  auto cnt = s.count(1);
  assert(cnt == 3);

  return 0;
}

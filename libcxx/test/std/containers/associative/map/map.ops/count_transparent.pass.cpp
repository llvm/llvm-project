//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <map>

// class map

// template<typename K>
//   size_type count(const K& x) const;        // C++14

#include <cassert>
#include <map>
#include <utility>

#include "min_allocator.h"
#include "private_constructor.h"
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
  std::map<std::pair<int, int>, int, Comp> s{
      {{2, 1}, 1}, {{1, 2}, 2}, {{1, 3}, 3}, {{1, 4}, 4}, {{2, 2}, 5}};

  auto cnt = s.count(1);
  assert(cnt == 3);

  return 0;
}

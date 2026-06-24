//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <map>

// class map

// template<typename K>
//         constexpr pair<iterator,iterator>             equal_range(const K& x);        // C++14, constexpr since C++26
// template<typename K>
//         constexpr pair<const_iterator,const_iterator> equal_range(const K& x) const;  // C++14, constexpr since C++26

#include <cassert>
#include <map>
#include <utility>

#include "test_macros.h"

struct Comp {
  using is_transparent = void;

  TEST_CONSTEXPR_CXX26 bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs < rhs;
  }

  TEST_CONSTEXPR_CXX26 bool operator()(const std::pair<int, int>& lhs, int rhs) const { return lhs.first < rhs; }

  TEST_CONSTEXPR_CXX26 bool operator()(int lhs, const std::pair<int, int>& rhs) const { return lhs < rhs.first; }
};

TEST_CONSTEXPR_CXX26 bool test() {
  std::map<std::pair<int, int>, int, Comp> s{{{2, 1}, 1}, {{1, 2}, 2}, {{1, 3}, 3}, {{1, 4}, 4}, {{2, 2}, 5}};

  auto er   = s.equal_range(1);
  long nels = 0;

  for (auto it = er.first; it != er.second; it++) {
    assert(it->first.first == 1);
    nels++;
  }

  assert(nels == 3);
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

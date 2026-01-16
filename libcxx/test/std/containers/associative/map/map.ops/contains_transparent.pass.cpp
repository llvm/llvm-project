//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cassert>
#include <map>

// <map>

// template<class K> constexpr bool contains(const K& x) const; // C++20, constexpr since C++26

#include "test_macros.h"

struct Comp {
  using is_transparent = void;

  TEST_CONSTEXPR_CXX26 bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs < rhs;
  }

  TEST_CONSTEXPR_CXX26 bool operator()(const std::pair<int, int>& lhs, int rhs) const { return lhs.first < rhs; }

  TEST_CONSTEXPR_CXX26 bool operator()(int lhs, const std::pair<int, int>& rhs) const { return lhs < rhs.first; }
};

template <typename Container>
TEST_CONSTEXPR_CXX26 bool test() {
  Container s{{{2, 1}, 1}, {{1, 2}, 2}, {{1, 3}, 3}, {{1, 4}, 4}, {{2, 2}, 5}};

  assert(s.contains(1));
  assert(!s.contains(-1));

  return true;
}

TEST_CONSTEXPR_CXX26 bool test() {
  test<std::map<std::pair<int, int>, int, Comp> >();

  // FIXME: remove when multimap is made constexpr
  if (!TEST_IS_CONSTANT_EVALUATED) {
    test<std::multimap<std::pair<int, int>, int, Comp> >();
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

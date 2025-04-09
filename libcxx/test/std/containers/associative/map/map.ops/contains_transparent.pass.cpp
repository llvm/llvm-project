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

// template<class K> bool contains(const K& x) const; // C++20 // constexpr since C++26

struct Comp {
  using is_transparent = void;

  bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const { return lhs < rhs; }

  bool operator()(const std::pair<int, int>& lhs, int rhs) const { return lhs.first < rhs; }

  bool operator()(int lhs, const std::pair<int, int>& rhs) const { return lhs < rhs.first; }
};

template <typename Container>
void test() {
  Container s{{{2, 1}, 1}, {{1, 2}, 2}, {{1, 3}, 3}, {{1, 4}, 4}, {{2, 2}, 5}};

  assert(s.contains(1));
  assert(!s.contains(-1));
}

TEST_CONSTEXPR_CXX26 bool test() {
  test<std::map<std::pair<int, int>, int, Comp> >();
  test<std::multimap<std::pair<int, int>, int, Comp> >();
return true;
}

int main(int, char**) {
assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

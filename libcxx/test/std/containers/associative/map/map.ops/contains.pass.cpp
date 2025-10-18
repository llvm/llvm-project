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

// bool contains(const key_type& x) const; // constexpr since C++26

#include "test_macros.h"

template <typename T, typename P, typename B, typename... Pairs>
TEST_CONSTEXPR_CXX26 bool test(B bad, Pairs... args) {
  T map;
  P pairs[] = {args...};

  for (auto& p : pairs)
    map.insert(p);
  for (auto& p : pairs)
    assert(map.contains(p.first));

  assert(!map.contains(bad));

  return true;
}

struct E {
  int a    = 1;
  double b = 1;
  char c   = 1;
};

TEST_CONSTEXPR_CXX26 bool test() {
  {
    test<std::map<char, int>, std::pair<char, int> >(
        'e', std::make_pair('a', 10), std::make_pair('b', 11), std::make_pair('c', 12), std::make_pair('d', 13));

    test<std::map<char, char>, std::pair<char, char> >(
        'e', std::make_pair('a', 'a'), std::make_pair('b', 'a'), std::make_pair('c', 'a'), std::make_pair('d', 'b'));

    test<std::map<int, E>, std::pair<int, E> >(
        -1, std::make_pair(1, E{}), std::make_pair(2, E{}), std::make_pair(3, E{}), std::make_pair(4, E{}));
  }
  // FIXME: remove when multimap is made constexpr
  if (!TEST_IS_CONSTANT_EVALUATED) {
    test<std::multimap<char, int>, std::pair<char, int> >(
        'e', std::make_pair('a', 10), std::make_pair('b', 11), std::make_pair('c', 12), std::make_pair('d', 13));

    test<std::multimap<char, char>, std::pair<char, char> >(
        'e', std::make_pair('a', 'a'), std::make_pair('b', 'a'), std::make_pair('c', 'a'), std::make_pair('d', 'b'));

    test<std::multimap<int, E>, std::pair<int, E> >(
        -1, std::make_pair(1, E{}), std::make_pair(2, E{}), std::make_pair(3, E{}), std::make_pair(4, E{}));
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

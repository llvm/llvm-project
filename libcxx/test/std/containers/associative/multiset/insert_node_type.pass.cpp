//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <set>

// class multiset

// iterator insert(node_type&&); // constexpr since C++26

#include <set>
#include <type_traits>
#include "test_macros.h"
#include "min_allocator.h"

template <class Container>
TEST_CONSTEXPR_CXX26 typename Container::node_type node_factory(Container& c, typename Container::key_type const& key) {
  auto it = c.insert(key);
  return c.extract(it);
}

template <class Container>
TEST_CONSTEXPR_CXX26 void test(Container& c) {
  auto* nf = &node_factory<Container>;
  Container c2;

  for (int i = 0; i != 10; ++i) {
    typename Container::node_type node = nf(c, i);
    assert(!node.empty());
    typename Container::iterator it = c.insert(std::move(node));
    assert(node.empty());
    assert(it == c.find(i) && it != c.end());
    assert(*it == i);
    assert(node.empty());
  }

  assert(c.size() == 10);

  { // Insert empty node.
    typename Container::node_type def;
    auto it = c.insert(std::move(def));
    assert(def.empty());
    assert(it == c.end());
  }

  { // Insert duplicate node.
    typename Container::node_type dupl = nf(c2, 0);
    auto it                            = c.insert(std::move(dupl));
    assert(*it == 0);
  }

  assert(c.size() == 11);

  assert(c.count(0) == 2);
  for (int i = 1; i != 10; ++i) {
    assert(c.count(i) == 1);
  }
}

TEST_CONSTEXPR_CXX26 bool test() {
  std::multiset<int> m;
  test(m);
  std::multiset<int, std::less<int>, min_allocator<int>> m2;
  test(m2);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

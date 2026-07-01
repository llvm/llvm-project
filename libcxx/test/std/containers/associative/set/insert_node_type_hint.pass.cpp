//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <set>

// class set

// constexpr iterator insert(const_iterator hint, node_type&&); // constexpr since C++26

#include <set>
#include "test_macros.h"
#include "min_allocator.h"

template <class Container>
TEST_CONSTEXPR_CXX26 typename Container::node_type node_factory(Container& c, typename Container::key_type const& key) {
  auto p = c.insert(key);
  assert(p.second);
  return c.extract(p.first);
}

template <class Container>
TEST_CONSTEXPR_CXX26 void test(Container& c) {
  auto* nf = &node_factory<Container>;

  Container c2;

  for (int i = 0; i != 10; ++i) {
    typename Container::node_type node = nf(c2, i);
    assert(!node.empty());
    std::size_t prev = c.size();
    auto it          = c.insert(c.end(), std::move(node));
    assert(node.empty());
    assert(prev + 1 == c.size());
    assert(*it == i);
  }

  assert(c.size() == 10);

  for (int i = 0; i != 10; ++i) {
    assert(c.count(i) == 1);
  }
}

TEST_CONSTEXPR_CXX26 bool test() {
  std::set<int> m;
  test(m);
  std::set<int, std::less<int>, min_allocator<int>> m2;
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

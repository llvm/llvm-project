//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// explicit queue(const container_type& c);

#include <queue>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#  include "test_convertible.h"
#endif

template <class C>
TEST_CONSTEXPR_CXX26 C make(int n) {
  C c;
  for (int i = 0; i < n; ++i)
    c.push_back(i);
  return c;
}

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::deque<int> Container;
  typedef std::queue<int> Q;
  Container d = make<Container>(5);
  Q q(d);
  assert(q.size() == 5);
  for (std::size_t i = 0; i < d.size(); ++i) {
    assert(q.front() == d[i]);
    q.pop();
  }

#if TEST_STD_VER >= 11
  static_assert(!test_convertible<Q, const Container&>(), "");
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

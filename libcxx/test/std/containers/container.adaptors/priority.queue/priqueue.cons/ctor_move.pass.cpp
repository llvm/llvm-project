//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// priority_queue(priority_queue&& q);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <class C>
TEST_CONSTEXPR_CXX26 C make(int n) {
  C c;
  for (int i = 0; i < n; ++i)
    c.push_back(MoveOnly(i));
  return c;
}

TEST_CONSTEXPR_CXX26 bool test() {
  std::priority_queue<MoveOnly> qo(std::less<MoveOnly>(), make<std::vector<MoveOnly> >(5));
  std::priority_queue<MoveOnly> q = std::move(qo);
  assert(q.size() == 5);
  assert(q.top() == MoveOnly(4));

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue();

// void pop();

#include <queue>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::priority_queue<int> q;
  q.push(1);
  assert(q.top() == 1);
  q.push(3);
  assert(q.top() == 3);
  q.push(2);
  assert(q.top() == 3);
  q.pop();
  assert(q.top() == 2);
  q.pop();
  assert(q.top() == 1);
  q.pop();
  assert(q.empty());

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

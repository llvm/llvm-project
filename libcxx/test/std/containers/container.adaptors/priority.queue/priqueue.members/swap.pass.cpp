//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue();

// void swap(priority_queue& q);

#include <queue>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::priority_queue<int> q1;
  std::priority_queue<int> q2;
  q1.push(1);
  q1.push(3);
  q1.push(2);
  q1.swap(q2);
  assert(q1.empty());
  assert(q2.size() == 3);
  assert(q2.top() == 3);

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

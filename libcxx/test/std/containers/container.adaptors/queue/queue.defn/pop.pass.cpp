//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// void pop();

#include <queue>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::queue<int> q;
  assert(q.size() == 0);
  q.push(1);
  q.push(2);
  q.push(3);
  assert(q.size() == 3);
  assert(q.front() == 1);
  assert(q.back() == 3);
  q.pop();
  assert(q.size() == 2);
  assert(q.front() == 2);
  assert(q.back() == 3);
  q.pop();
  assert(q.size() == 1);
  assert(q.front() == 3);
  assert(q.back() == 3);
  q.pop();
  assert(q.size() == 0);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

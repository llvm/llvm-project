//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// void push(const value_type& v);

#include <stack>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::stack<int> q;
  q.push(1);
  assert(q.size() == 1);
  assert(q.top() == 1);
  q.push(2);
  assert(q.size() == 2);
  assert(q.top() == 2);
  q.push(3);
  assert(q.size() == 3);
  assert(q.top() == 3);

  return 0;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

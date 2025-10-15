//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// bool empty() const;

#include <stack>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::stack<int> q;
  assert(q.empty());
  q.push(1);
  assert(!q.empty());
  q.pop();
  assert(q.empty());

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

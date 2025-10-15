//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// const_reference front() const;

#include <queue>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::queue<int> q;
  assert(q.size() == 0);
  q.push(1);
  q.push(2);
  q.push(3);
  const std::queue<int>& cqr = q;
  const int& cir             = cqr.front();
  assert(cir == 1);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

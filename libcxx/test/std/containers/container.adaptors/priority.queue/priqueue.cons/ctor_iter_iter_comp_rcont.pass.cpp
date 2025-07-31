//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// template <class InputIterator>
//   priority_queue(InputIterator first, InputIterator last,
//                  const Compare& comp, container_type&& c);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

TEST_CONSTEXPR_CXX26 bool test() {
  int a[]     = {3, 5, 2, 0, 6, 8, 1};
  const int n = sizeof(a) / sizeof(a[0]);
  std::priority_queue<MoveOnly> q(a + n / 2, a + n, std::less<MoveOnly>(), std::vector<MoveOnly>(a, a + n / 2));
  assert(q.size() == n);
  assert(q.top() == MoveOnly(8));

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

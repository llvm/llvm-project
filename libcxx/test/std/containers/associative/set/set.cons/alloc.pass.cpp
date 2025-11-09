//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// constexpr set(const allocator_type& a); // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::less<int> C;
  typedef test_allocator<int> A;
  std::set<int, C, A> m(A(5));
  assert(m.empty());
  assert(m.begin() == m.end());
  assert(m.get_allocator() == A(5));

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// constexpr explicit set(const key_compare& comp) const; // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"
#include "../../../test_compare.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef test_less<int> C;
  const std::set<int, C> m(C(3));
  assert(m.empty());
  assert(m.begin() == m.end());
  assert(m.key_comp() == C(3));
  assert(m.value_comp() == C(3));

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

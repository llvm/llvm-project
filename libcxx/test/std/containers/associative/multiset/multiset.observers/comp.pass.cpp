//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// key_compare key_comp() const; // constexpr since C++26
// value_compare value_comp() const; // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  typedef std::multiset<int> set_type;

  set_type s;
  set_type::iterator i1 = s.insert(1);
  set_type::iterator i2 = s.insert(2);

  const set_type& cs = s;

  assert(cs.key_comp()(*i1, *i2));
  assert(!cs.key_comp()(*i2, *i1));

  assert(cs.value_comp()(*i1, *i2));
  assert(!cs.value_comp()(*i2, *i1));

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}

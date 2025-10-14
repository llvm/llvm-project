//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// Check that std::list and its iterators can be instantiated with an incomplete
// type.

#include <list>
#include <cassert>

#include "test_macros.h"

struct A {
  std::list<A> l;
  std::list<A>::iterator it;
  std::list<A>::const_iterator cit;
  std::list<A>::reverse_iterator rit;
  std::list<A>::const_reverse_iterator crit;
};

TEST_CONSTEXPR_CXX26 bool test() {
  A a;
  (void)a;

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}

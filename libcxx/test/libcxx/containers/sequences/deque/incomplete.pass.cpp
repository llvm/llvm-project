//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// deque()
// deque::iterator()

// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ABI_INCOMPLETE_TYPES_IN_DEQUE

#include <deque>
#include <cassert>

#include "test_macros.h"

struct A {
  std::deque<A> d;
  std::deque<A>::iterator it;
  std::deque<A>::reverse_iterator it2;
};

int main(int, char**)
{
  A a;
  assert(a.d.size() == 0);
  a.it = a.d.begin();
  a.it2 = a.d.rend();

  return 0;
}

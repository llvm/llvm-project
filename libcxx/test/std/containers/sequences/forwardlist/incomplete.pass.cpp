//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list()
// forward_list::iterator()
// forward_list::const_iterator()

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

struct A {
  std::forward_list<A> d;
  std::forward_list<A>::iterator it;
  std::forward_list<A>::const_iterator it2;
};

#if TEST_STD_VER >= 11
struct B {
  typedef std::forward_list<B, min_allocator<B>> FList;
  FList d;
  FList::iterator it;
  FList::const_iterator it2;
};
#endif

int main(int, char**)
{
  {
    A a;
    assert(a.d.empty());
    a.it = a.d.begin();
    a.it2 = a.d.cbefore_begin();
  }
#if TEST_STD_VER >= 11
  {
    B b;
    assert(b.d.empty());
    b.it = b.d.begin();
    b.it2 = b.d.cbefore_begin();
  }
#endif

  return 0;
}

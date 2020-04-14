//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib=macosx

// <list>

// Call erase(const_iterator position) with iterator from another container

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cassert>
#include <cstdlib>

#include "test_macros.h"

int main(int, char**)
{
    int a1[] = {1, 2, 3};
    std::list<int> l1(a1, a1+3);
    std::list<int> l2(a1, a1+3);
    std::list<int>::const_iterator i = l2.begin();
    l1.erase(i);
    assert(false);

  return 0;
}

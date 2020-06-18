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

// iterator insert(const_iterator position, size_type n, const value_type& x);

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::list<int> c1(100);
    std::list<int> c2;
    std::list<int>::iterator i = c1.insert(next(c2.cbegin(), 10), 5, 1);
    assert(false);

  return 0;
}

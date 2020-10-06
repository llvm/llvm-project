//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// iterator insert(const_iterator position, const value_type& x);

// UNSUPPORTED: libcxx-no-debug-mode

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"


int main(int, char**)
{
    std::list<int> v1(3);
    std::list<int> v2(3);
    int i = 4;
    v1.insert(v2.begin(), i);
    assert(false);

  return 0;
}

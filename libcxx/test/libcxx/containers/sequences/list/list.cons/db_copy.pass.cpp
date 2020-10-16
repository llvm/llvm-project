//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// list(list&& c);

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::list<int> l1;
    l1.push_back(1); l1.push_back(2); l1.push_back(3);
    std::list<int>::iterator i = l1.begin();
    std::list<int> l2 = l1;
    l2.erase(i);
    assert(false);

  return 0;
}

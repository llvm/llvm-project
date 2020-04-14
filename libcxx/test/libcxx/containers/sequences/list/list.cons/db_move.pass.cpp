//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib=macosx

// <list>

// list(list&& c);

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(1))

#include <list>
#include <cstdlib>
#include <cassert>
#include "test_macros.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    std::list<int> l1 = {1, 2, 3};
    std::list<int>::iterator i = l1.begin();
    std::list<int> l2 = std::move(l1);
    assert(*l2.erase(i) == 2);

  return 0;
}

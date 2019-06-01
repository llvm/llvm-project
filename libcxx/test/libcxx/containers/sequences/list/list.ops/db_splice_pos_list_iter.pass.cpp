//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// <list>

// void splice(const_iterator position, list<T,Allocator>& x, iterator i);

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::list<int> v1(3);
        std::list<int> v2(3);
        v1.splice(v1.begin(), v2, v1.begin());
        assert(false);
    }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Increment local_iterator past end.

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_set>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef int T;
    typedef std::unordered_set<T> C;
    C c(1);
    C::local_iterator i = c.begin(0);
    ++i;
    ++i;
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    typedef int T;
    typedef std::unordered_set<T, min_allocator<T>> C;
    C c(1);
    C::local_iterator i = c.begin(0);
    ++i;
    ++i;
    assert(false);
    }
#endif

}

#else

int main(int, char**)
{

  return 0;
}

#endif

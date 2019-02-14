//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Increment local_iterator past end.

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <string>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::unordered_map<int, std::string> C;
    C c(1);
    C::local_iterator i = c.begin(0);
    ++i;
    ++i;
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>,
                        min_allocator<std::pair<const int, std::string>>> C;
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

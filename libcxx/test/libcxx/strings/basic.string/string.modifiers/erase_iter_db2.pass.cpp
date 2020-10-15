//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator position) with iterator from another container

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <string>
#include <cassert>
#include <cstdlib>
#include <exception>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    std::string l1("123");
    std::string l2("123");
    std::string::const_iterator i = l2.begin();
    l1.erase(i);
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S l1("123");
    S l2("123");
    S::const_iterator i = l2.begin();
    l1.erase(i);
    assert(false);
    }
#endif
}

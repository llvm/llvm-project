//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string& operator=(initializer_list<charT> il);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::string s;
        s = {'a', 'b', 'c'};
        assert(s == "abc");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        S s;
        s = {'a', 'b', 'c'};
        assert(s == "abc");
    }

  return 0;
}

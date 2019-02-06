//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// basic_string& replace(const_iterator i1, const_iterator i2, initializer_list<charT> il);

#include <string>
#include <cassert>

#include "min_allocator.h"

int main()
{
    {
        std::string s("123def456");
        s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        S s("123def456");
        s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
}

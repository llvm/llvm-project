//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <string>

// iterator insert(const_iterator p, initializer_list<charT> il);


#include <string>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
        std::string s("123456");
        std::string::iterator i = s.insert(s.begin() + 3, {'a', 'b', 'c'});
        assert(i - s.begin() == 3);
        assert(s == "123abc456");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        S s("123456");
        S::iterator i = s.insert(s.begin() + 3, {'a', 'b', 'c'});
        assert(i - s.begin() == 3);
        assert(s == "123abc456");
    }

  return 0;
}

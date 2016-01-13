//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string& replace(const_iterator i1, const_iterator i2, initializer_list<charT> il);

#include <string>
#include <cassert>

#include "min_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        std::string s("123def456");
        s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
#if TEST_STD_VER >= 11
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        S s("123def456");
        s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
#endif
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}

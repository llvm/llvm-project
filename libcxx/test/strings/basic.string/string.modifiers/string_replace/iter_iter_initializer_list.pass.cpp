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

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        std::string s("123def456");
        s.replace(s.cbegin() + 3, s.cbegin() + 6, {'a', 'b', 'c'});
        assert(s == "123abc456");
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}

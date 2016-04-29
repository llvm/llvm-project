//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator first, const_iterator last); with a bad range

#if _LIBCPP_DEBUG >= 1

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <string>
#include <cassert>
#include <exception>
#include <cstdlib>

#include "min_allocator.h"

int main()
{
    {
    std::string l1("123");
    std::string::iterator i = l1.erase(l1.cbegin()+1, l1.cbegin());
    assert(false);
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S l1("123");
    S::iterator i = l1.erase(l1.cbegin()+1, l1.cbegin());
    assert(false);
    }
#endif
}

#else

int main()
{
}

#endif

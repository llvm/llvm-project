//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// An string is a contiguous container

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"


template <class C>
void test_contiguous ( const C &c )
{
    for ( size_t i = 0; i < c.size(); ++i )
        assert ( *(c.begin() + static_cast<typename C::difference_type>(i)) == *(std::addressof(*c.begin()) + i));
}

int main(int, char**)
{
    {
    typedef std::string S;
    test_contiguous(S());
    test_contiguous(S("1"));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890"));
    }

    {
    typedef test_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
    test_contiguous(S(A(3)));
    test_contiguous(S("1", A(5)));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)));
    }
#if TEST_STD_VER >= 11
    {
    typedef min_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
    test_contiguous(S(A{}));
    test_contiguous(S("1", A()));
    test_contiguous(S("1234567890123456789012345678901234567890123456789012345678901234567890", A()));
    }
#endif

  return 0;
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// constexpr istream_iterator();

#include <iterator>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
    typedef std::istream_iterator<int> T;
    T it;
    assert(it == T());
#if TEST_STD_VER >= 11
    constexpr T it2;
#endif
    }

}

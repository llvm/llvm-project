//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// Construct with initizializer list

#include <array>
#include <cassert>

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        assert(c.size() == 3);
        assert(c[0] == 1);
        assert(c[1] == 2);
        assert(c[2] == 3.5);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        assert(c.size() == 0);
    }
}

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <vector>
//  vector.bool

// template <class... Args> reference emplace_back(Args&&... args);

#include <vector>
#include <cassert>
#include "min_allocator.h"

int main()
{
    {
        typedef std::vector<bool> C;
        typedef C::reference Ref;
        C c;
        Ref r1 = c.emplace_back();
        assert(c.size() == 1);
        assert(c.front() == false);
        r1 = true;
        assert(c.front() == true);
        r1 = false;
        Ref r2 = c.emplace_back(true);
        assert(c.size() == 2);
        assert(c.front() == false);
        assert(c.back() == true);
        r2 = false;
        assert(c.back() == false);
        r2 = true;
        Ref r3 = c.emplace_back(1 == 1);
        assert(c.size() == 3);
        assert(c.front() == false);
        assert(c[1] == true);
        assert(c.back() == true);
        r3 = false;
        assert(c.back() == false);
    }
    {
        typedef std::vector<bool, min_allocator<bool>> C;
        typedef C::reference Ref;
        C c;

        Ref r1 = c.emplace_back();
        assert(c.size() == 1);
        assert(c.front() == false);
        r1 = true;
        assert(c.front() == true);
        r1 = false;
        Ref r2 = c.emplace_back(true);
        assert(c.size() == 2);
        assert(c.front() == false);
        assert(c.back() == true);
        r2 = false;
        assert(c.back() == false);
        r2 = true;
        c.emplace_back(1 == 1);
        assert(c.size() == 3);
        assert(c.front() == false);
        assert(c[1] == true);
        assert(c.back() == true);
    }
}

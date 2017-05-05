//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// void swap(any &, any &) noexcept

// swap(...) just wraps any::swap(...). That function is tested elsewhere.

#include <experimental/any>
#include <cassert>

using std::experimental::any;
using std::experimental::any_cast;

int main()
{

    { // test noexcept
        any a;
        static_assert(noexcept(swap(a, a)), "swap(any&, any&) must be noexcept");
    }
    {
        any a1(1);
        any a2(2);

        swap(a1, a2);

        // Support testing against system dylibs that don't have bad_any_cast.
        assert(*any_cast<int>(&a1) == 2);
        assert(*any_cast<int>(&a2) == 1);
    }
}

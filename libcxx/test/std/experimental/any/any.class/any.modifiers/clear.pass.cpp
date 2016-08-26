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

// any::clear() noexcept

#include <experimental/any>
#include <cassert>

#include "experimental_any_helpers.h"

int main()
{
    using std::experimental::any;
    using std::experimental::any_cast;
    // empty
    {
        any a;

        // noexcept check
        static_assert(
            noexcept(a.clear())
          , "any.clear() must be noexcept"
          );

        assertEmpty(a);

        a.clear();

        assertEmpty(a);
    }
    // small object
    {
        any a((small(1)));
        assert(small::count == 1);
        assertContains<small>(a, 1);

        a.clear();

        assertEmpty<small>(a);
        assert(small::count == 0);
    }
    // large object
    {
        any a(large(1));
        assert(large::count == 1);
        assertContains<large>(a);

        a.clear();

        assertEmpty<large>(a);
        assert(large::count == 0);
    }
}

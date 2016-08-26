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

// any::empty() noexcept

#include <experimental/any>
#include <cassert>

#include "experimental_any_helpers.h"

int main()
{
    using std::experimental::any;
    // noexcept test
    {
        any a;
        static_assert(noexcept(a.empty()), "any::empty() must be noexcept");
    }
    // empty
    {
        any a;
        assert(a.empty());

        a.clear();
        assert(a.empty());

        a = 42;
        assert(!a.empty());
    }
    // small object
    {
        small const s(1);
        any a(s);
        assert(!a.empty());

        a.clear();
        assert(a.empty());

        a = s;
        assert(!a.empty());
    }
    // large object
    {
        large const l(1);
        any a(l);
        assert(!a.empty());

        a.clear();
        assert(a.empty());

        a = l;
        assert(!a.empty());
    }
}

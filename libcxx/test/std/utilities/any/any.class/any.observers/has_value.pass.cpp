//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <any>

// any::has_value() noexcept

#include <any>
#include <cassert>

#include "any_helpers.h"

int main(int, char**)
{
    using std::any;
    // noexcept test
    {
        any a;
        static_assert(noexcept(a.has_value()), "any::has_value() must be noexcept");
    }
    // empty
    {
        any a;
        assert(!a.has_value());

        a.reset();
        assert(!a.has_value());

        a = 42;
        assert(a.has_value());
    }
    // small object
    {
        small const s(1);
        any a(s);
        assert(a.has_value());

        a.reset();
        assert(!a.has_value());

        a = s;
        assert(a.has_value());
    }
    // large object
    {
        large const l(1);
        any a(l);
        assert(a.has_value());

        a.reset();
        assert(!a.has_value());

        a = l;
        assert(a.has_value());
    }

  return 0;
}

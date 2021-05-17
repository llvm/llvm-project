//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_any_cast is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10 && !no-exceptions
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9 && !no-exceptions

// <any>

// any::reset() noexcept

#include <any>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"

int main(int, char**)
{
    using std::any;
    using std::any_cast;
    // empty
    {
        any a;

        // noexcept check
        static_assert(
            noexcept(a.reset())
          , "any.reset() must be noexcept"
          );

        assertEmpty(a);

        a.reset();

        assertEmpty(a);
    }
    // small object
    {
        any a((small(1)));
        assert(small::count == 1);
        assertContains<small>(a, 1);

        a.reset();

        assertEmpty<small>(a);
        assert(small::count == 0);
    }
    // large object
    {
        any a(large(1));
        assert(large::count == 1);
        assertContains<large>(a, 1);

        a.reset();

        assertEmpty<large>(a);
        assert(large::count == 0);
    }

  return 0;
}

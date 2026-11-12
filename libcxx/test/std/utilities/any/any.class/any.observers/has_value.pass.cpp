//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// any::has_value() noexcept

#include <any>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"

int main(int, char**)
{
    {
        std::any a;
        ASSERT_NOEXCEPT(a.has_value());
    }
    // empty
    {
        std::any a;
        assert(!a.has_value());

        a.reset();
        assert(!a.has_value());

        a = 42;
        assert(a.has_value());
    }
    // small object
    {
        std::any a = small(1);
        assert(a.has_value());

        a.reset();
        assert(!a.has_value());

        a = small(1);
        assert(a.has_value());
    }
    // large object
    {
        std::any a = large(1);
        assert(a.has_value());

        a.reset();
        assert(!a.has_value());

        a = large(1);
        assert(a.has_value());
    }

  return 0;
}

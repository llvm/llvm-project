//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// XFAIL: libcpp-no-rtti

// <any>

// any::type() noexcept

#include <any>
#include <cassert>
#include "any_helpers.h"

int main()
{
    using std::any;
    {
        any const a;
        assert(a.type() == typeid(void));
        static_assert(noexcept(a.type()), "any::type() must be noexcept");
    }
    {
        small const s(1);
        any const a(s);
        assert(a.type() == typeid(small));

    }
    {
        large const l(1);
        any const a(l);
        assert(a.type() == typeid(large));
    }
}

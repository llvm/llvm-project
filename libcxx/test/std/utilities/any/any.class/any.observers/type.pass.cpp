//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: no-rtti

// <any>

// any::type() noexcept

#include <any>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"

int main(int, char**)
{
    {
        const std::any a;
        assert(a.type() == typeid(void));
        ASSERT_NOEXCEPT(a.type());
    }
    {
        std::any a = small(1);
        assert(a.type() == typeid(small));
    }
    {
        std::any a = large(1);
        assert(a.type() == typeid(large));
    }
    {
        int arr[3];
        std::any a = arr;
        assert(a.type() == typeid(int*));  // ensure that it is decayed
    }

  return 0;
}

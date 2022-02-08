//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// any() noexcept;

#include <any>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"
#include "count_new.h"

int main(int, char**)
{
    using std::any;
    {
        static_assert(
            std::is_nothrow_default_constructible<any>::value
          , "Must be default constructible"
          );
    }
    {
        struct TestConstexpr : public std::any {
          constexpr TestConstexpr() : std::any() {}
        };
        TEST_SAFE_STATIC static std::any a;
        ((void)a);
    }
    {
        DisableAllocationGuard g; ((void)g);
        any const a;
        assertEmpty(a);
    }

  return 0;
}

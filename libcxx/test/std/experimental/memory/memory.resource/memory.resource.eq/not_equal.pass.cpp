//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <experimental/memory_resource>

// bool operator!=(memory_resource const &, memory_resource const &) noexcept;

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    // check return types
    {
        ex::memory_resource const * mr1(nullptr);
        ex::memory_resource const * mr2(nullptr);
        static_assert(std::is_same<decltype(*mr1 != *mr2), bool>::value, "");
        static_assert(noexcept(*mr1 != *mr2), "");
    }
    // not equal
    {
        TestResource r1(1);
        TestResource r2(2);
        ex::memory_resource const & mr1 = r1;
        ex::memory_resource const & mr2 = r2;

        assert(mr1 != mr2);
        assert(r1.checkIsEqualCalledEq(1));
        assert(r2.checkIsEqualCalledEq(0));

        assert(mr2 != mr1);
        assert(r1.checkIsEqualCalledEq(1));
        assert(r2.checkIsEqualCalledEq(1));
    }
    // equal
    {
        TestResource r1(1);
        TestResource r2(1);
        ex::memory_resource const & mr1 = r1;
        ex::memory_resource const & mr2 = r2;

        assert(!(mr1 != mr2));
        assert(r1.checkIsEqualCalledEq(1));
        assert(r2.checkIsEqualCalledEq(0));

        assert(!(mr2 != mr1));
        assert(r1.checkIsEqualCalledEq(1));
        assert(r2.checkIsEqualCalledEq(1));
    }
    // equal same object
    {
        TestResource r1(1);
        ex::memory_resource const & mr1 = r1;
        ex::memory_resource const & mr2 = r1;

        assert(!(mr1 != mr2));
        assert(r1.checkIsEqualCalledEq(0));

        assert(!(mr2 != mr1));
        assert(r1.checkIsEqualCalledEq(0));
    }

  return 0;
}

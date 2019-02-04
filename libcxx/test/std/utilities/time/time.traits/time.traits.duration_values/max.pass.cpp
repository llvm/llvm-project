//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::max  // noexcept after C++17

#include <chrono>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::max() ==
           std::numeric_limits<int>::max());
    assert(std::chrono::duration_values<double>::max() ==
           std::numeric_limits<double>::max());
    assert(std::chrono::duration_values<Rep>::max() ==
           std::numeric_limits<Rep>::max());
#if TEST_STD_VER >= 11
    static_assert(std::chrono::duration_values<int>::max() ==
           std::numeric_limits<int>::max(), "");
    static_assert(std::chrono::duration_values<double>::max() ==
           std::numeric_limits<double>::max(), "");
    static_assert(std::chrono::duration_values<Rep>::max() ==
           std::numeric_limits<Rep>::max(), "");
#endif

    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<int>::max());
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<double>::max());
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::max());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(std::chrono::duration_values<int>::max());
    ASSERT_NOEXCEPT(std::chrono::duration_values<double>::max());
    ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::max());
#endif
}

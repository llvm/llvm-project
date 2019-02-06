//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::zero  // noexcept after C++17

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::zero() == 0);
    assert(std::chrono::duration_values<Rep>::zero() == 0);
#if TEST_STD_VER >= 11
    static_assert(std::chrono::duration_values<int>::zero() == 0, "");
    static_assert(std::chrono::duration_values<Rep>::zero() == 0, "");
#endif

    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<int>::zero());
    LIBCPP_ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::zero());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(std::chrono::duration_values<int>::zero());
    ASSERT_NOEXCEPT(std::chrono::duration_values<Rep>::zero());
#endif
}

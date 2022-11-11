//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <functional>

// Make sure that we can use CTAD with operations in <functional>

#include <functional>

#include "test_macros.h"

int main(int, char**) {
    {
        std::plus f;
        ASSERT_SAME_TYPE(decltype(f), std::plus<>);
    }
    {
        std::minus f;
        ASSERT_SAME_TYPE(decltype(f), std::minus<>);
    }
    {
        std::multiplies f;
        ASSERT_SAME_TYPE(decltype(f), std::multiplies<>);
    }
    {
        std::divides f;
        ASSERT_SAME_TYPE(decltype(f), std::divides<>);
    }
    {
        std::modulus f;
        ASSERT_SAME_TYPE(decltype(f), std::modulus<>);
    }
    {
        std::negate f;
        ASSERT_SAME_TYPE(decltype(f), std::negate<>);
    }
    {
        std::bit_and f;
        ASSERT_SAME_TYPE(decltype(f), std::bit_and<>);
    }
    {
        std::bit_not f;
        ASSERT_SAME_TYPE(decltype(f), std::bit_not<>);
    }
    {
        std::bit_or f;
        ASSERT_SAME_TYPE(decltype(f), std::bit_or<>);
    }
    {
        std::bit_xor f;
        ASSERT_SAME_TYPE(decltype(f), std::bit_xor<>);
    }
    {
        std::equal_to f;
        ASSERT_SAME_TYPE(decltype(f), std::equal_to<>);
    }
    {
        std::not_equal_to f;
        ASSERT_SAME_TYPE(decltype(f), std::not_equal_to<>);
    }
    {
        std::less f;
        ASSERT_SAME_TYPE(decltype(f), std::less<>);
    }
    {
        std::less_equal f;
        ASSERT_SAME_TYPE(decltype(f), std::less_equal<>);
    }
    {
        std::greater_equal f;
        ASSERT_SAME_TYPE(decltype(f), std::greater_equal<>);
    }
    {
        std::greater f;
        ASSERT_SAME_TYPE(decltype(f), std::greater<>);
    }
    {
        std::logical_and f;
        ASSERT_SAME_TYPE(decltype(f), std::logical_and<>);
    }
    {
        std::logical_not f;
        ASSERT_SAME_TYPE(decltype(f), std::logical_not<>);
    }
    {
        std::logical_or f;
        ASSERT_SAME_TYPE(decltype(f), std::logical_or<>);
    }

    return 0;
}

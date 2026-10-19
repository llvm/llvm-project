//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420

// <charconv>

// constexpr from_chars_result from_chars(const char* first, const char* last,
//                                        Integral& value, int base = 10)

#include <charconv>
#include <system_error>

#include "test_macros.h"
#include "charconv_test_helpers.h"

struct test_basics : roundtrip_test_base
{
    template <typename T>
    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test<T>(0);
        test<T>(42);
        test<T>(32768);
        test<T>(0, 10);
        test<T>(42, 10);
        test<T>(32768, 10);
        test<T>(0xf, 16);
        test<T>(0xdeadbeaf, 16);
        test<T>(0755, 8);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test<T>(1, b);
            test<T>(-1, b);
            test<T>(xl::lowest(), b);
            test<T>((xl::max)(), b);
            test<T>((xl::max)() / 2, b);
        }
    }
};

struct test_signed : roundtrip_test_base
{
    template <typename T>
    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test<T>(-1);
        test<T>(-12);
        test<T>(-1, 10);
        test<T>(-12, 10);
        test<T>(-21734634, 10);
        test<T>(-2647, 2);
        test<T>(-0xcc1, 16);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test<T>(0, b);
            test<T>(xl::lowest(), b);
            test<T>((xl::max)(), b);
        }
    }
};

TEST_CONSTEXPR_CXX23 bool test()
{
    types::for_each(integrals(), test_basics());
    types::for_each(types::signed_integer_types(), test_signed());

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 20
    static_assert(test());
#endif

    return 0;
}

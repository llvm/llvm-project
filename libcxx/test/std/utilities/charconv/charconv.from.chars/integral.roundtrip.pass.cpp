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
#include "test_macros.h"
#include "charconv_test_helpers.h"

template <typename T>
struct test_basics : roundtrip_test_base<T>
{
    using roundtrip_test_base<T>::test;

    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test(0);
        test(42);
        test(32768);
        test(0, 10);
        test(42, 10);
        test(32768, 10);
        test(0xf, 16);
        test(0xdeadbeaf, 16);
        test(0755, 8);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test(1, b);
            test(-1, b);
            test(xl::lowest(), b);
            test((xl::max)(), b);
            test((xl::max)() / 2, b);
        }
    }
};

template <typename T>
struct test_signed : roundtrip_test_base<T>
{
    using roundtrip_test_base<T>::test;

    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test(-1);
        test(-12);
        test(-1, 10);
        test(-12, 10);
        test(-21734634, 10);
        test(-2647, 2);
        test(-0xcc1, 16);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test(0, b);
            test(xl::lowest(), b);
            test((xl::max)(), b);
        }
    }
};

TEST_CONSTEXPR_CXX23 bool test()
{
    run<test_basics>(integrals);
    run<test_signed>(all_signed);

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 20
    static_assert(test());
#endif

    return 0;
}

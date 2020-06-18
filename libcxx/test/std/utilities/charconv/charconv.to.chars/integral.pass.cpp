//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: !libc++ && c++11
// UNSUPPORTED: !libc++ && c++14

// to_chars requires functions in the dylib that were introduced in Mac OS 10.15.
//
// XFAIL: with_system_cxx_lib=macosx10.14
// XFAIL: with_system_cxx_lib=macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9

// <charconv>

// to_chars_result to_chars(char* first, char* last, Integral value,
//                          int base = 10)

#include <charconv>
#include "test_macros.h"
#include "charconv_test_helpers.h"

template <typename T>
struct test_basics : to_chars_test_base<T>
{
    using to_chars_test_base<T>::test;
    using to_chars_test_base<T>::test_value;

    void operator()()
    {
        test(0, "0");
        test(42, "42");
        test(32768, "32768");
        test(0, "0", 10);
        test(42, "42", 10);
        test(32768, "32768", 10);
        test(0xf, "f", 16);
        test(0xdeadbeaf, "deadbeaf", 16);
        test(0755, "755", 8);

        // Test each len till len of UINT64_MAX = 20 because to_chars algorithm
        // makes branches based on decimal digits count in the value string
        // representation.
        // Test driver automatically skips values not fitting into source type.
        test(1UL, "1");
        test(12UL, "12");
        test(123UL, "123");
        test(1234UL, "1234");
        test(12345UL, "12345");
        test(123456UL, "123456");
        test(1234567UL, "1234567");
        test(12345678UL, "12345678");
        test(123456789UL, "123456789");
        test(1234567890UL, "1234567890");
        test(12345678901UL, "12345678901");
        test(123456789012UL, "123456789012");
        test(1234567890123UL, "1234567890123");
        test(12345678901234UL, "12345678901234");
        test(123456789012345UL, "123456789012345");
        test(1234567890123456UL, "1234567890123456");
        test(12345678901234567UL, "12345678901234567");
        test(123456789012345678UL, "123456789012345678");
        test(1234567890123456789UL, "1234567890123456789");
        test(12345678901234567890UL, "12345678901234567890");

        // Test special cases with zeros inside a value string representation,
        // to_chars algorithm processes them in a special way and should not
        // skip trailing zeros
        // Test driver automatically skips values not fitting into source type.
        test(0UL, "0");
        test(10UL, "10");
        test(100UL, "100");
        test(1000UL, "1000");
        test(10000UL, "10000");
        test(100000UL, "100000");
        test(1000000UL, "1000000");
        test(10000000UL, "10000000");
        test(100000000UL, "100000000");
        test(1000000000UL, "1000000000");
        test(10000000000UL, "10000000000");
        test(100000000000UL, "100000000000");
        test(1000000000000UL, "1000000000000");
        test(10000000000000UL, "10000000000000");
        test(100000000000000UL, "100000000000000");
        test(1000000000000000UL, "1000000000000000");
        test(10000000000000000UL, "10000000000000000");
        test(100000000000000000UL, "100000000000000000");
        test(1000000000000000000UL, "1000000000000000000");
        test(10000000000000000000UL, "10000000000000000000");

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test_value(1, b);
            test_value(xl::lowest(), b);
            test_value((xl::max)(), b);
            test_value((xl::max)() / 2, b);
        }
    }
};

template <typename T>
struct test_signed : to_chars_test_base<T>
{
    using to_chars_test_base<T>::test;
    using to_chars_test_base<T>::test_value;

    void operator()()
    {
        test(-1, "-1");
        test(-12, "-12");
        test(-1, "-1", 10);
        test(-12, "-12", 10);
        test(-21734634, "-21734634", 10);
        test(-2647, "-101001010111", 2);
        test(-0xcc1, "-cc1", 16);

        // Test each len till len of INT64_MAX = 19 because to_chars algorithm
        // makes branches based on decimal digits count in the value string
        // representation.
        // Test driver automatically skips values not fitting into source type.
        test(-1L, "-1");
        test(-12L, "-12");
        test(-123L, "-123");
        test(-1234L, "-1234");
        test(-12345L, "-12345");
        test(-123456L, "-123456");
        test(-1234567L, "-1234567");
        test(-12345678L, "-12345678");
        test(-123456789L, "-123456789");
        test(-1234567890L, "-1234567890");
        test(-12345678901L, "-12345678901");
        test(-123456789012L, "-123456789012");
        test(-1234567890123L, "-1234567890123");
        test(-12345678901234L, "-12345678901234");
        test(-123456789012345L, "-123456789012345");
        test(-1234567890123456L, "-1234567890123456");
        test(-12345678901234567L, "-12345678901234567");
        test(-123456789012345678L, "-123456789012345678");
        test(-1234567890123456789L, "-1234567890123456789");

        // Test special cases with zeros inside a value string representation,
        // to_chars algorithm processes them in a special way and should not
        // skip trailing zeros
        // Test driver automatically skips values not fitting into source type.
        test(-10L, "-10");
        test(-100L, "-100");
        test(-1000L, "-1000");
        test(-10000L, "-10000");
        test(-100000L, "-100000");
        test(-1000000L, "-1000000");
        test(-10000000L, "-10000000");
        test(-100000000L, "-100000000");
        test(-1000000000L, "-1000000000");
        test(-10000000000L, "-10000000000");
        test(-100000000000L, "-100000000000");
        test(-1000000000000L, "-1000000000000");
        test(-10000000000000L, "-10000000000000");
        test(-100000000000000L, "-100000000000000");
        test(-1000000000000000L, "-1000000000000000");
        test(-10000000000000000L, "-10000000000000000");
        test(-100000000000000000L, "-100000000000000000");
        test(-1000000000000000000L, "-1000000000000000000");

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test_value(0, b);
            test_value(xl::lowest(), b);
            test_value((xl::max)(), b);
        }
    }
};

int main(int, char**)
{
    run<test_basics>(integrals);
    run<test_signed>(all_signed);

  return 0;
}

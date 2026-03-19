//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <charconv>

// constexpr from_chars_result from_chars(const char* first, const char* last,
//                                        Integral& value, int base = 10)

#include <charconv>
#include <system_error>

#include "test_macros.h"
#include "charconv_test_helpers.h"

template <typename T>
struct test_basics
{
    TEST_CONSTEXPR_CXX23 void operator()()
    {
        std::from_chars_result r;
        T x;

        {
            char s[] = "001x";

            // the expected form of the subject sequence is a sequence of
            // letters and digits representing an integer with the radix
            // specified by base (C11 7.22.1.4/3)
            r = std::from_chars(s, s + sizeof(s), x);
            assert(r.ec == std::errc{});
            assert(r.ptr == s + 3);
            assert(x == 1);
        }

        {
            // The string has more characters than valid in an 128-bit value.
            char s[] = "0X7BAtSGHDkEIXZgQRfYChLpOzRnM ";

            // The letters from a (or A) through z (or Z) are ascribed the
            // values 10 through 35; (C11 7.22.1.4/3)
            r = std::from_chars(s, s + sizeof(s), x, 36);
            assert(r.ec == std::errc::result_out_of_range);
            // The member ptr of the return value points to the first character
            // not matching the pattern
            assert(r.ptr == s + sizeof(s) - 2);
            assert(x == 1);

            // no "0x" or "0X" prefix shall appear if the value of base is 16
            r = std::from_chars(s, s + sizeof(s), x, 16);
            assert(r.ec == std::errc{});
            assert(r.ptr == s + 1);
            assert(x == 0);

            // only letters and digits whose ascribed values are less than that
            // of base are permitted. (C11 7.22.1.4/3)
            r = std::from_chars(s + 2, s + sizeof(s), x, 12);
            // If the parsed value is not in the range representable by the type
            // of value,
            if (!fits_in<T>(1150))
            {
                // value is unmodified and
                assert(x == 0);
                // the member ec of the return value is equal to
                // errc::result_out_of_range
                assert(r.ec == std::errc::result_out_of_range);
            }
            else
            {
                // Otherwise, value is set to the parsed value,
                assert(x == 1150);
                // and the member ec is value-initialized.
                assert(r.ec == std::errc{});
            }
            assert(r.ptr == s + 5);
        }
    }
};

template <typename T>
struct test_signed
{
    TEST_CONSTEXPR_CXX23 void operator()()
    {
        std::from_chars_result r;
        T x = 42;

        {
            // If the pattern allows for an optional sign,
            // but the string has no digit characters following the sign,
            char s[] = "- 9+12";
            r = std::from_chars(s, s + sizeof(s), x);
            // value is unmodified,
            assert(x == 42);
            // no characters match the pattern.
            assert(r.ptr == s);
            assert(r.ec == std::errc::invalid_argument);
        }

        {
            char s[] = "9+12";
            r = std::from_chars(s, s + sizeof(s), x);
            assert(r.ec == std::errc{});
            // The member ptr of the return value points to the first character
            // not matching the pattern,
            assert(r.ptr == s + 1);
            assert(x == 9);
        }

        {
            char s[] = "12";
            r = std::from_chars(s, s + 2, x);
            assert(r.ec == std::errc{});
            // or has the value last if all characters match.
            assert(r.ptr == s + 2);
            assert(x == 12);
        }

        {
            // '-' is the only sign that may appear
            char s[] = "+30";
            // If no characters match the pattern,
            r = std::from_chars(s, s + sizeof(s), x);
            // value is unmodified,
            assert(x == 12);
            // the member ptr of the return value is first and
            assert(r.ptr == s);
            // the member ec is equal to errc::invalid_argument.
            assert(r.ec == std::errc::invalid_argument);
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

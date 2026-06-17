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

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
// _BitInt wider than 128 bits parses via the generic Horner accumulator. The
// error paths must leave value unmodified; check that with a sentinel.
template <class T, std::size_t N>
TEST_CONSTEXPR_CXX23 void fc_ok(const char (&s)[N], T expect, std::size_t consumed, int base = 10) {
  T v    = 0x55;
  auto r = std::from_chars(s, s + N - 1, v, base);
  assert(r.ec == std::errc{});
  assert(r.ptr == s + consumed);
  assert(v == expect);
}

template <class T, std::size_t N>
TEST_CONSTEXPR_CXX23 void fc_invalid(const char (&s)[N], int base = 10) {
  T v    = 0x55;
  auto r = std::from_chars(s, s + N - 1, v, base);
  assert(r.ec == std::errc::invalid_argument);
  assert(r.ptr == s);   // no characters match -> ptr == first
  assert(v == T(0x55)); // value unmodified
}

template <class T, std::size_t N>
TEST_CONSTEXPR_CXX23 void fc_oor(const char (&s)[N], std::size_t consumed, int base = 10) {
  T v    = 0x55;
  auto r = std::from_chars(s, s + N - 1, v, base);
  assert(r.ec == std::errc::result_out_of_range);
  assert(r.ptr == s + consumed); // past the maximal matching sequence
  assert(v == T(0x55));          // value unmodified
}

TEST_CONSTEXPR_CXX23 bool test_wide() {
  using S = signed _BitInt(256);
  using U = unsigned _BitInt(256);

  // Success and ptr placement.
  fc_ok<U>("42", U(42), 2);
  fc_ok<S>("-42", S(-42), 3);
  fc_ok<S>("-0", S(0), 2);
  fc_ok<U>("007", U(7), 3); // leading zeros consumed
  fc_ok<U>("ff", U(255), 2, 16);
  fc_ok<U>("FF", U(255), 2, 16); // case-insensitive
  fc_ok<U>("123abc", U(123), 3); // base 10: stops at 'a'
  fc_ok<U>("123abc", U(0x123abc), 6, 16);
  fc_ok<U>("0x10", U(0), 1, 16); // no 0x prefix: matches '0', stops at 'x'
  fc_ok<U>("0b10", U(0), 1, 2);  // no 0b prefix: matches '0', stops at 'b'
  fc_ok<U>("12", U(1), 1, 2);    // '2' >= base 2: stops

  // No match: value unmodified, ptr == first.
  fc_invalid<U>("");
  fc_invalid<U>("+5"); // '+' is never accepted
  fc_invalid<S>("+5");
  fc_invalid<U>("-5"); // '-' only for signed types
  fc_invalid<S>("-");  // sign with no digits
  fc_invalid<U>(" 5"); // leading whitespace is not skipped
  fc_invalid<U>("z");

  // Out of range: value unmodified, ptr past every matching digit.
  fc_oor<U>("115792089237316195423570985008687907853269984665640564039457584007913129639936", 78); // 2^256
  fc_oor<U>("115792089237316195423570985008687907853269984665640564039457584007913129639936zzz", 78);
  fc_oor<S>("57896044618658097711785492504343953926634992332820282019728792003956564819968",
            77); // 2^255 = signed max + 1

  // Most-negative -2^255 succeeds: magnitude is signed max + 1 but valid as a
  // negative value.
  fc_ok<S>("-57896044618658097711785492504343953926634992332820282019728792003956564819968",
           std::numeric_limits<S>::min(),
           78);

  // Exact boundaries: max accepted, max + 1 rejected.
  fc_ok<U>("115792089237316195423570985008687907853269984665640564039457584007913129639935",
           std::numeric_limits<U>::max(),
           78); // 2^256 - 1
  fc_ok<S>("57896044618658097711785492504343953926634992332820282019728792003956564819967",
           std::numeric_limits<S>::max(),
           77); // 2^255 - 1
  return true;
}
#endif

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

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
    test_wide();
#  if TEST_STD_VER > 20
    static_assert(test_wide());
#  endif
#endif

    return 0;
}

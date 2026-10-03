//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=50000000

// <charconv>

// constexpr to_chars_result to_chars(char* first, char* last, Integral value,
//                                    int base = 10)

#include <charconv>
#include <cstdint>
#include <system_error>

#include "test_macros.h"
#include "charconv_test_helpers.h"

#ifndef TEST_HAS_NO_INT128
TEST_CONSTEXPR_CXX23 __uint128_t make_u128(__uint128_t a, std::uint64_t b) {
  a *= 1000000000000000000UL;
  a *= 10;
  return a + b;
}

TEST_CONSTEXPR_CXX23 __uint128_t make_u128(__uint128_t a, std::uint64_t b, std::uint64_t c) {
  a *= 10000000000000ULL;
  a += b;
  a *= 10000000000000ULL;
  return a + c;
}

TEST_CONSTEXPR_CXX23 __int128_t make_i128(__int128_t a, std::int64_t b) {
  if (a < 0)
    return -make_u128(-a, b);
  return make_u128(a, b);
}

TEST_CONSTEXPR_CXX23 __int128_t make_i128(__int128_t a, __int128_t b, std::int64_t c) {
  if (a < 0)
    return -make_u128(-a, b, c);
  return make_u128(a, b, c);
}
#endif

template <typename T>
struct test_basics : to_chars_test_base<T>
{
    using to_chars_test_base<T>::test;
    using to_chars_test_base<T>::test_value;

    TEST_CONSTEXPR_CXX23 void operator()()
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
#ifndef TEST_HAS_NO_INT128
        test(make_u128(12UL, 3456789012345678901UL), "123456789012345678901");
        test(make_u128(123UL, 4567890123456789012UL), "1234567890123456789012");
        test(make_u128(1234UL, 5678901234567890123UL), "12345678901234567890123");
        test(make_u128(12345UL, 6789012345678901234UL), "123456789012345678901234");
        test(make_u128(123456UL, 7890123456789012345UL), "1234567890123456789012345");
        test(make_u128(1234567UL, 8901234567890123456UL), "12345678901234567890123456");
        test(make_u128(12345678UL, 9012345678901234567UL), "123456789012345678901234567");
        test(make_u128(123456789UL, 123456789012345678UL), "1234567890123456789012345678");
        test(make_u128(123UL, 4567890123456UL, 7890123456789UL), "12345678901234567890123456789");
        test(make_u128(1234UL, 5678901234567UL, 8901234567890UL), "123456789012345678901234567890");
        test(make_u128(12345UL, 6789012345678UL, 9012345678901UL), "1234567890123456789012345678901");
        test(make_u128(123456UL, 7890123456789UL, 123456789012UL), "12345678901234567890123456789012");
        test(make_u128(1234567UL, 8901234567890UL, 1234567890123UL), "123456789012345678901234567890123");
        test(make_u128(12345678UL, 9012345678901UL, 2345678901234UL), "1234567890123456789012345678901234");
        test(make_u128(123456789UL, 123456789012UL, 3456789012345UL), "12345678901234567890123456789012345");
        test(make_u128(1234567890UL, 1234567890123UL, 4567890123456UL), "123456789012345678901234567890123456");
        test(make_u128(12345678901UL, 2345678901234UL, 5678901234567UL), "1234567890123456789012345678901234567");
        test(make_u128(123456789012UL, 3456789012345UL, 6789012345678UL), "12345678901234567890123456789012345678");
        test(make_u128(1234567890123UL, 4567890123456UL, 7890123456789UL), "123456789012345678901234567890123456789");
#endif

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
#ifndef TEST_HAS_NO_INT128
        test(make_u128(10UL, 0), "100000000000000000000");
        test(make_u128(100UL, 0), "1000000000000000000000");
        test(make_u128(1000UL, 0), "10000000000000000000000");
        test(make_u128(10000UL, 0), "100000000000000000000000");
        test(make_u128(100000UL, 0), "1000000000000000000000000");
        test(make_u128(1000000UL, 0), "10000000000000000000000000");
        test(make_u128(10000000UL, 0), "100000000000000000000000000");
        test(make_u128(100000000UL, 0), "1000000000000000000000000000");
        test(make_u128(100UL, 0, 0), "10000000000000000000000000000");
        test(make_u128(1000UL, 0, 0), "100000000000000000000000000000");
        test(make_u128(10000UL, 0, 0), "1000000000000000000000000000000");
        test(make_u128(100000UL, 0, 0), "10000000000000000000000000000000");
        test(make_u128(1000000UL, 0, 0), "100000000000000000000000000000000");
        test(make_u128(10000000UL, 0, 0), "1000000000000000000000000000000000");
        test(make_u128(100000000UL, 0, 0), "10000000000000000000000000000000000");
        test(make_u128(1000000000UL, 0, 0), "100000000000000000000000000000000000");
        test(make_u128(10000000000UL, 0, 0), "1000000000000000000000000000000000000");
        test(make_u128(100000000000UL, 0, 0), "10000000000000000000000000000000000000");
        test(make_u128(1000000000000UL, 0, 0), "100000000000000000000000000000000000000");
#endif

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

    TEST_CONSTEXPR_CXX23 void operator()()
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
#ifndef TEST_HAS_NO_INT128
        test(make_i128(-1L, 2345678901234567890L), "-12345678901234567890");
        test(make_i128(-12L, 3456789012345678901L), "-123456789012345678901");
        test(make_i128(-123L, 4567890123456789012L), "-1234567890123456789012");
        test(make_i128(-1234L, 5678901234567890123L), "-12345678901234567890123");
        test(make_i128(-12345L, 6789012345678901234L), "-123456789012345678901234");
        test(make_i128(-123456L, 7890123456789012345L), "-1234567890123456789012345");
        test(make_i128(-1234567L, 8901234567890123456L), "-12345678901234567890123456");
        test(make_i128(-12345678L, 9012345678901234567L), "-123456789012345678901234567");
        test(make_i128(-123456789L, 123456789012345678L), "-1234567890123456789012345678");
        test(make_i128(-1234567890L, 1234567890123456789L), "-12345678901234567890123456789");
        test(make_i128(-123L, 4567890123456L, 7890123456789L), "-12345678901234567890123456789");
        test(make_i128(-1234L, 5678901234567L, 8901234567890L), "-123456789012345678901234567890");
        test(make_i128(-12345L, 6789012345678L, 9012345678901L), "-1234567890123456789012345678901");
        test(make_i128(-123456L, 7890123456789L, 123456789012L), "-12345678901234567890123456789012");
        test(make_i128(-1234567L, 8901234567890L, 1234567890123L), "-123456789012345678901234567890123");
        test(make_i128(-12345678L, 9012345678901L, 2345678901234L), "-1234567890123456789012345678901234");
        test(make_i128(-123456789L, 123456789012L, 3456789012345L), "-12345678901234567890123456789012345");
        test(make_i128(-1234567890L, 1234567890123L, 4567890123456L), "-123456789012345678901234567890123456");
        test(make_i128(-12345678901L, 2345678901234L, 5678901234567L), "-1234567890123456789012345678901234567");
        test(make_i128(-123456789012L, 3456789012345L, 6789012345678L), "-12345678901234567890123456789012345678");
        test(make_i128(-1234567890123L, 4567890123456L, 7890123456789L), "-123456789012345678901234567890123456789");
#endif

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
#ifndef TEST_HAS_NO_INT128
        test(make_i128(-1L, 0L), "-10000000000000000000");
        test(make_i128(-10L, 0L), "-100000000000000000000");
        test(make_i128(-100L, 0L), "-1000000000000000000000");
        test(make_i128(-1000L, 0L), "-10000000000000000000000");
        test(make_i128(-10000L, 0L), "-100000000000000000000000");
        test(make_i128(-100000L, 0L), "-1000000000000000000000000");
        test(make_i128(-1000000L, 0L), "-10000000000000000000000000");
        test(make_i128(-10000000L, 0L), "-100000000000000000000000000");
        test(make_i128(-100000000L, 0L), "-1000000000000000000000000000");
        test(make_i128(-1000000000L, 0L), "-10000000000000000000000000000");
        test(make_i128(-100L, 0L, 0L), "-10000000000000000000000000000");
        test(make_i128(-1000L, 0L, 0L), "-100000000000000000000000000000");
        test(make_i128(-10000L, 0L, 0L), "-1000000000000000000000000000000");
        test(make_i128(-100000L, 0L, 0L), "-10000000000000000000000000000000");
        test(make_i128(-1000000L, 0L, 0L), "-100000000000000000000000000000000");
        test(make_i128(-10000000L, 0L, 0L), "-1000000000000000000000000000000000");
        test(make_i128(-100000000L, 0L, 0L), "-10000000000000000000000000000000000");
        test(make_i128(-1000000000L, 0L, 0L), "-100000000000000000000000000000000000");
        test(make_i128(-10000000000L, 0L, 0L), "-1000000000000000000000000000000000000");
        test(make_i128(-100000000000L, 0L, 0L), "-10000000000000000000000000000000000000");
        test(make_i128(-1000000000000L, 0L, 0L), "-100000000000000000000000000000000000000");
#endif

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test_value(0, b);
            test_value(xl::lowest(), b);
            test_value((xl::max)(), b);
        }
    }
};

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 4096
// _BitInt wider than 128 bits has no native promotion target, so the native
// harness above (fixed buffer, strtoll reference) does not apply. Check it with
// hand-verified literals plus exact-fit / one-too-short buffer handling.
template <class T, std::size_t N>
TEST_CONSTEXPR_CXX23 void test_wide_one(T v, const char (&expect)[N], int base = 10) {
  const std::size_t len = N - 1;
  char buf[4200];

  std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v, base);
  assert(r.ec == std::errc{});
  assert(static_cast<std::size_t>(r.ptr - buf) == len);
  for (std::size_t i = 0; i < len; ++i)
    assert(buf[i] == expect[i]);

  // Exact-fit buffer succeeds; one byte short fails with ptr == last and no
  // assertion on the (unspecified) buffer contents.
  r = std::to_chars(buf, buf + len, v, base);
  assert(r.ec == std::errc{} && r.ptr == buf + len);
  r = std::to_chars(buf, buf + len - 1, v, base);
  assert(r.ec == std::errc::value_too_large && r.ptr == buf + len - 1);
}

TEST_CONSTEXPR_CXX23 bool test_wide() {
  using S   = signed _BitInt(256);
  using U   = unsigned _BitInt(256);
  using SLn = std::numeric_limits<S>;
  using ULn = std::numeric_limits<U>;

  test_wide_one(U(0), "0");
  test_wide_one(S(0), "0", 16);
  test_wide_one(U(255), "255");
  test_wide_one(U(255), "ff", 16); // lowercase, no 0x prefix
  test_wide_one(U(0xabcu), "abc", 16);
  test_wide_one(U(10), "1010", 2);
  test_wide_one(S(-12), "-12");
  test_wide_one(S(-255), "-ff", 16);
  test_wide_one(S(-10), "-1010", 2);

  // Trailing/interior zeros must survive (no stray digit dropping).
  test_wide_one(U(1000000000u), "1000000000");

  // Signed min = -2^255: magnitude 2^255 is not representable in S, so it must
  // be formed in the unsigned counterpart. (Base 2 is covered by the round-trip
  // test to avoid a 257-character literal.)
  test_wide_one(SLn::min(), "-57896044618658097711785492504343953926634992332820282019728792003956564819968");
  test_wide_one(SLn::min(), "-8000000000000000000000000000000000000000000000000000000000000000", 16);

  // Signed max = 2^255-1, unsigned max = 2^256-1.
  test_wide_one(SLn::max(), "57896044618658097711785492504343953926634992332820282019728792003956564819967");
  test_wide_one(ULn::max(), "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16);

  return true;
}
#endif

TEST_CONSTEXPR_CXX23 bool test()
{
    run<test_basics>(integrals);
    run<test_signed>(all_signed);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 20
    static_assert(test());
#endif

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 4096
    test_wide();
#  if TEST_STD_VER > 20
    static_assert(test_wide());
#  endif
#endif

    return 0;
}

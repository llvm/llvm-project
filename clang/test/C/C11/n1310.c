// RUN: %clang_cc1 -verify -std=c89 %s
// RUN: %clang_cc1 -verify -std=c99 %s
// RUN: %clang_cc1 -verify -std=c11 %s
// RUN: %clang_cc1 -verify -std=c17 %s
// RUN: %clang_cc1 -verify -std=c23 %s
// expected-no-diagnostics

/* WG14 N1310: Yes
 * Requiring signed char to have no padding bits
 */

/* This is shockingly hard to test, but we're trying our best by checking that
 * setting each bit of an unsigned char, then bit-casting it to signed char,
 * results in a value we expect to see. If we have padding bits, then it's
 * possible (but not mandatory) for the value to not be as we expect, so a
 * failing assertion means the implementation is broken but a passing test does
 * not *prove* there aren't padding bits.
 */
_Static_assert(__CHAR_BIT__ == 8, "");
_Static_assert(sizeof(signed char) == 1, "");

#define TEST(Bit, Expected) __builtin_bit_cast(signed char, (unsigned char)(1 << Bit)) == Expected
_Static_assert(TEST(0, 1), "");
_Static_assert(TEST(1, 2), "");
_Static_assert(TEST(2, 4), "");
_Static_assert(TEST(3, 8), "");
_Static_assert(TEST(4, 16), "");
_Static_assert(TEST(5, 32), "");
_Static_assert(TEST(6, 64), "");
_Static_assert(TEST(7, (signed char)128), "");


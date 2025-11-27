// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xB1, 3) == 0x8D, "");
_Static_assert(__builtin_stdc_rotate_right((unsigned char)0xB1, 3) == 0x36, "");
_Static_assert(__builtin_stdc_rotate_left((unsigned short)0x1234, 4) == 0x2341, "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 4) == 0x4123, "");
_Static_assert(__builtin_stdc_rotate_left(0x12345678U, 8) == 0x34567812U, "");
_Static_assert(__builtin_stdc_rotate_right(0x12345678U, 8) == 0x78123456U, "");
_Static_assert(__builtin_stdc_rotate_left(0x123456789ABCDEF0ULL, 16) == 0x56789ABCDEF01234ULL, "");
_Static_assert(__builtin_stdc_rotate_right(0x123456789ABCDEF0ULL, 16) == 0xDEF0123456789ABCULL, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned __int128)1, 127) == ((unsigned __int128)1 << 127), "");
_Static_assert(__builtin_stdc_rotate_right(((unsigned __int128)1 << 127), 127) == (unsigned __int128)1, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))1, 36) == ((unsigned _BitInt(37))1 << 36), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))1, 36) == ((unsigned _BitInt(37))1 << 1), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(128))1, 1) == ((unsigned _BitInt(128))2), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(13))1, 12) == ((unsigned _BitInt(13))1 << 12), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(17))0x10000, 16) == ((unsigned _BitInt(17))1), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(65))1, 64) == ((unsigned _BitInt(65))1 << 64), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(7))0x01, 1) == ((unsigned _BitInt(7))0x40), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))0x100, 1) == ((unsigned _BitInt(9))0x001), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(9))0x001, 1) == ((unsigned _BitInt(9))0x100), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))0x1FF, 4) == ((unsigned _BitInt(9))0x1FF), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))1, 8) == ((unsigned _BitInt(9))0x100), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(9))0x100, 8) == ((unsigned _BitInt(9))1), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))0x155, 1) == ((unsigned _BitInt(9))0xAB), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))1, 9) == ((unsigned _BitInt(9))1), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x80, 9) == __builtin_stdc_rotate_left((unsigned char)0x80, 1), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x8000, 17) == __builtin_stdc_rotate_right((unsigned short)0x8000, 1), "");
_Static_assert(__builtin_stdc_rotate_left(0x80000000U, 33) == __builtin_stdc_rotate_left(0x80000000U, 1), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 40) == __builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 3), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))0x1000000000ULL, 74) == 0x1000000000ULL, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x80, -1) == __builtin_stdc_rotate_left((unsigned char)0x80, 7), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x8000, -1) == __builtin_stdc_rotate_right((unsigned short)0x8000, 15), "");
_Static_assert(__builtin_stdc_rotate_left(0x80000000U, -5) == __builtin_stdc_rotate_left(0x80000000U, 27), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))0x1000000000ULL, -10) == 0x200ULL, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -8) == (unsigned char)0xAB, "");
_Static_assert(__builtin_stdc_rotate_left(0x12345678U, -32) == 0x12345678U, "");
_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, -25) == __builtin_stdc_rotate_left((unsigned char)0x12, 7), "");
_Static_assert(__builtin_stdc_rotate_right(0x12345678U, -100) == __builtin_stdc_rotate_right(0x12345678U, 28), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned short)0x1234, -65541) == __builtin_stdc_rotate_left((unsigned short)0x1234, -5), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned char)0x80, -261) == __builtin_stdc_rotate_right((unsigned char)0x80, -5), "");
_Static_assert(__builtin_stdc_rotate_left(0x12345678U, -4294967333LL) == __builtin_stdc_rotate_left(0x12345678U, -37), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 0) == (unsigned short)0x1234, "");
_Static_assert(__builtin_stdc_rotate_left((unsigned long long)0x123456789ABCDEF0ULL, 0) == 0x123456789ABCDEF0ULL, "");
_Static_assert(__builtin_stdc_rotate_left(0U, 15) == 0U, "");
_Static_assert(__builtin_stdc_rotate_right(0U, 7) == 0U, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 1000000) == __builtin_stdc_rotate_left((unsigned char)0xAB, 1000000 % 8), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 2147483647) == __builtin_stdc_rotate_right((unsigned short)0x1234, 2147483647 % 16), "");
_Static_assert(__builtin_stdc_rotate_left(0x12345678U, 4294967295U) == __builtin_stdc_rotate_left(0x12345678U, 4294967295U % 32), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -1000000) == __builtin_stdc_rotate_left((unsigned char)0xAB, -1000000 % 8), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, -2147483647) == __builtin_stdc_rotate_right((unsigned short)0x1234, -2147483647 % 16), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(127))1, 1000000) == __builtin_stdc_rotate_left((unsigned _BitInt(127))1, 1000000 % 127), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(127))1, -1000000) == __builtin_stdc_rotate_right((unsigned _BitInt(127))1, -1000000 % 127), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x01, 2147483647) == __builtin_stdc_rotate_left((unsigned char)0x01, 2147483647 % 8), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned char)0x80, -2147483648) == __builtin_stdc_rotate_right((unsigned char)0x80, -2147483648 % 8), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -9) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -17) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -25) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, 64 + 3) == __builtin_stdc_rotate_left((unsigned char)0x12, 3), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 128 + 5) == __builtin_stdc_rotate_right((unsigned short)0x1234, 5), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(31))1, 1000) == __builtin_stdc_rotate_left((unsigned _BitInt(31))1, 1000 % 31), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(61))1, -1000) == __builtin_stdc_rotate_right((unsigned _BitInt(61))1, -1000 % 61), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(128))0xFFFFFFFFFFFFFFFFULL, 50000) == __builtin_stdc_rotate_left((unsigned _BitInt(128))0xFFFFFFFFFFFFFFFFULL, 50000 % 128), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(128))0xAAAAAAAAAAAAAAAAULL, -50000) == __builtin_stdc_rotate_right((unsigned _BitInt(128))0xAAAAAAAAAAAAAAAAULL, -50000 % 128), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 7) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 8) == (unsigned char)0xAB, "");
_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 9) == __builtin_stdc_rotate_left((unsigned char)0xAB, 1), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))0x155, 1000) == __builtin_stdc_rotate_left((unsigned _BitInt(9))0x155, 1000 % 9), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(9))0xAA, -1000) == __builtin_stdc_rotate_right((unsigned _BitInt(9))0xAA, -1000 % 9), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0xFF, 1073741824) == (unsigned char)0xFF, "");
_Static_assert(__builtin_stdc_rotate_right((unsigned short)0xFFFF, -1073741824) == (unsigned short)0xFFFF, "");

_Static_assert(__builtin_stdc_rotate_left((unsigned char)0x42, 1000000000) == __builtin_stdc_rotate_left((unsigned char)0x42, 1000000000 % 8), "");
_Static_assert(__builtin_stdc_rotate_right((unsigned char)0x42, -1000000000) == __builtin_stdc_rotate_right((unsigned char)0x42, -1000000000 % 8), "");

_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(1))0, -1) == (unsigned _BitInt(1))0, "");
_Static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(1))1, -1) == (unsigned _BitInt(1))1, "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(1))0, 5) == (unsigned _BitInt(1))0, "");
_Static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(1))1, 5) == (unsigned _BitInt(1))1, "");

void test_errors(int si, float f) {
  unsigned int ui = 5;

  __builtin_stdc_rotate_left(ui); // expected-error {{too few arguments to function call}}
  __builtin_stdc_rotate_left(ui, 1, 2); // expected-error {{too many arguments to function call}}
}

void test_valid_conversions(_Bool b, int si) {
  unsigned int ui = 5;

  // Valid: bool converts to int for second argument
  (void)__builtin_stdc_rotate_left(ui, b);
  (void)__builtin_stdc_rotate_right(ui, b);

  // Valid: signed int is an integer type for second argument
  (void)__builtin_stdc_rotate_left(ui, si);
  (void)__builtin_stdc_rotate_right(ui, si);
}

void test_invalid_types(float f, int si) {
  unsigned int ui = 5;

  // Invalid: float is not an integer type for second argument
  (void)__builtin_stdc_rotate_left(ui, f); // expected-error {{2nd argument must be a scalar integer type (was 'float')}}
  (void)__builtin_stdc_rotate_left(ui, 1.5); // expected-error {{2nd argument must be a scalar integer type (was 'double')}}
  (void)__builtin_stdc_rotate_right(ui, f); // expected-error {{2nd argument must be a scalar integer type (was 'float')}}

  // Invalid: signed int is not unsigned for first argument
  (void)__builtin_stdc_rotate_left(si, 1); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  (void)__builtin_stdc_rotate_left(-5, 1); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  (void)__builtin_stdc_rotate_right(3.0, 1); // expected-error {{1st argument must be a scalar unsigned integer type (was 'double')}}
}

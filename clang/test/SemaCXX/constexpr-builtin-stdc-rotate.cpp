// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

namespace test_constexpr_stdc_rotate {

static_assert(__builtin_stdc_rotate_left((unsigned char)0b10110001, 3) == (unsigned char)0b10001101, "");
static_assert(__builtin_stdc_rotate_left((unsigned short)0x1234, 4) == (unsigned short)0x2341, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 8) == 0x34567812U, "");
static_assert(__builtin_stdc_rotate_left(0x123456789ABCDEF0ULL, 16) == 0x56789ABCDEF01234ULL, "");
static_assert(__builtin_stdc_rotate_right((unsigned char)0b10110001, 3) == (unsigned char)0b00110110, "");
static_assert(__builtin_stdc_rotate_right(0x12345678U, 8) == 0x78123456U, "");
static_assert(__builtin_stdc_rotate_right(0x123456789ABCDEF0ULL, 16) == 0xDEF0123456789ABCULL, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 0) == 0x12345678U, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 32) == 0x12345678U, "");
static_assert(__builtin_stdc_rotate_left(0x80000000U, 1) == 0x00000001U, "");
static_assert(__builtin_stdc_rotate_right(__builtin_stdc_rotate_left(0x12345678U, 8), 8) == 0x12345678U, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 40) == __builtin_stdc_rotate_left(0x12345678U, 8), "");
static_assert(__builtin_stdc_rotate_left(0x00000000U, 7) == 0x00000000U, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0x01, 2) == (unsigned char)0x04, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xAA, 1) == (unsigned char)0x55, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, 4) == (unsigned char)0x21, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 4) == 0x23456781U, "");

namespace test_int128 {

static_assert(__builtin_stdc_rotate_left((unsigned __int128)1, 127) == (unsigned __int128)1 << 127, "");

constexpr unsigned __int128 test_pattern = 0x123456789ABCDEF0ULL;
static_assert(__builtin_stdc_rotate_left(test_pattern, 1) == test_pattern << 1, "");

} // namespace test_int128

namespace test_bitint {

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))1, 36) == ((unsigned _BitInt(37))1 << 36), "");

constexpr unsigned _BitInt(128) bi128_pattern = 0x123456789ABCDEF0ULL;
static_assert(__builtin_stdc_rotate_left(bi128_pattern, 1) == bi128_pattern << 1, "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(3))0b101, 1) == (unsigned _BitInt(3))0b011, "");

} // namespace test_bitint

namespace test_modulo_behavior {

static_assert(__builtin_stdc_rotate_left((unsigned char)0x80, 9) == __builtin_stdc_rotate_left((unsigned char)0x80, 1), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x8000, 17) == __builtin_stdc_rotate_right((unsigned short)0x8000, 1), "");
static_assert(__builtin_stdc_rotate_left(0x80000000U, 33) == __builtin_stdc_rotate_left(0x80000000U, 1), "");
static_assert(__builtin_stdc_rotate_right(0x8000000000000000ULL, 65) == __builtin_stdc_rotate_right(0x8000000000000000ULL, 1), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 40) == __builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 3), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))0x1000000000ULL, 74) == (unsigned _BitInt(37))0x1000000000ULL, "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))0x123456789ULL, 0) == (unsigned _BitInt(37))0x123456789ULL, "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))0x123456789ULL, 37) == (unsigned _BitInt(37))0x123456789ULL, "");

} // namespace test_modulo_behavior

namespace test_negative_counts {

static_assert(__builtin_stdc_rotate_left((unsigned char)0x80, -1) == __builtin_stdc_rotate_left((unsigned char)0x80, 7), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x8000, -1) == __builtin_stdc_rotate_right((unsigned short)0x8000, 15), "");
static_assert(__builtin_stdc_rotate_left(0x80000000U, -5) == __builtin_stdc_rotate_left(0x80000000U, 27), "");
static_assert(__builtin_stdc_rotate_right(0x8000000000000000ULL, -8) == __builtin_stdc_rotate_right(0x8000000000000000ULL, 56), "");

static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(37))0x1000000000ULL, -10) == (unsigned _BitInt(37))0x200ULL, "");
static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(37))0x800000000ULL, -3) == (unsigned _BitInt(37))0x100000000ULL, "");

} // namespace test_negative_counts

namespace test_boundaries {

static_assert(__builtin_stdc_rotate_left((unsigned char)0x01, 7) == (unsigned char)0x80, "");
static_assert(__builtin_stdc_rotate_right((unsigned char)0x80, 7) == (unsigned char)0x01, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xFF, 1) == (unsigned char)0xFF, "");
static_assert(__builtin_stdc_rotate_left((unsigned short)0x0001, 15) == (unsigned short)0x8000, "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x8000, 15) == (unsigned short)0x0001, "");
static_assert(__builtin_stdc_rotate_left(0x00000001U, 31) == 0x80000000U, "");
static_assert(__builtin_stdc_rotate_right(0x80000000U, 31) == 0x00000001U, "");
static_assert(__builtin_stdc_rotate_left(0x0000000000000001ULL, 63) == 0x8000000000000000ULL, "");
static_assert(__builtin_stdc_rotate_right(0x8000000000000000ULL, 63) == 0x0000000000000001ULL, "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0x01, -7) == (unsigned char)0x02, "");
static_assert(__builtin_stdc_rotate_right((unsigned char)0x80, -7) == (unsigned char)0x40, "");
static_assert(__builtin_stdc_rotate_left(0x00000001U, -31) == 0x00000002U, "");
static_assert(__builtin_stdc_rotate_right(0x80000000U, -31) == 0x40000000U, "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -8) == (unsigned char)0xAB, "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, -32) == 0x12345678U, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, -25) == __builtin_stdc_rotate_left((unsigned char)0x12, 7), "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, -100) == __builtin_stdc_rotate_left(0x12345678U, 28), "");

constexpr unsigned __int128 int128_one = 1;
constexpr unsigned __int128 int128_msb = int128_one << 127;
static_assert(__builtin_stdc_rotate_left(int128_one, 127) == int128_msb, "");
static_assert(__builtin_stdc_rotate_right(int128_msb, 127) == int128_one, "");
static_assert(__builtin_stdc_rotate_left(int128_one, -127) == (int128_one << 1), "");

constexpr unsigned _BitInt(37) bi37_one = 1;
constexpr unsigned _BitInt(37) bi37_msb = bi37_one << 36;
static_assert(__builtin_stdc_rotate_left(bi37_one, 36) == bi37_msb, "");
static_assert(__builtin_stdc_rotate_right(bi37_msb, 36) == bi37_one, "");
static_assert(__builtin_stdc_rotate_left(bi37_one, -36) == (bi37_one << 1), "");

} // namespace test_boundaries

namespace test_extreme_cases {

static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 1000000) == __builtin_stdc_rotate_left((unsigned char)0xAB, 1000000 % 8), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 2147483647) == __builtin_stdc_rotate_right((unsigned short)0x1234, 2147483647 % 16), "");
static_assert(__builtin_stdc_rotate_left(0x12345678U, 4294967295U) == __builtin_stdc_rotate_left(0x12345678U, 4294967295U % 32), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -1000000) == __builtin_stdc_rotate_left((unsigned char)0xAB, -1000000 % 8), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, -2147483647) == __builtin_stdc_rotate_right((unsigned short)0x1234, -2147483647 % 16), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(127))1, 1000000) == __builtin_stdc_rotate_left((unsigned _BitInt(127))1, 1000000 % 127), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(127))1, -1000000) == __builtin_stdc_rotate_right((unsigned _BitInt(127))1, -1000000 % 127), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0x01, 2147483647) == __builtin_stdc_rotate_left((unsigned char)0x01, 2147483647 % 8), "");
static_assert(__builtin_stdc_rotate_right((unsigned char)0x80, -2147483648) == __builtin_stdc_rotate_right((unsigned char)0x80, -2147483648 % 8), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -9) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -17) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, -25) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, 64 + 3) == __builtin_stdc_rotate_left((unsigned char)0x12, 3), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, 128 + 5) == __builtin_stdc_rotate_right((unsigned short)0x1234, 5), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(31))1, 1000) == __builtin_stdc_rotate_left((unsigned _BitInt(31))1, 1000 % 31), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(61))1, -1000) == __builtin_stdc_rotate_right((unsigned _BitInt(61))1, -1000 % 61), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(128))0xFFFFFFFFFFFFFFFFULL, 50000) == __builtin_stdc_rotate_left((unsigned _BitInt(128))0xFFFFFFFFFFFFFFFFULL, 50000 % 128), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(128))0xAAAAAAAAAAAAAAAAULL, -50000) == __builtin_stdc_rotate_right((unsigned _BitInt(128))0xAAAAAAAAAAAAAAAAULL, -50000 % 128), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 7) == __builtin_stdc_rotate_left((unsigned char)0xAB, 7), "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 8) == (unsigned char)0xAB, "");
static_assert(__builtin_stdc_rotate_left((unsigned char)0xAB, 9) == __builtin_stdc_rotate_left((unsigned char)0xAB, 1), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(9))0x155, 1000) == __builtin_stdc_rotate_left((unsigned _BitInt(9))0x155, 1000 % 9), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(9))0xAA, -1000) == __builtin_stdc_rotate_right((unsigned _BitInt(9))0xAA, -1000 % 9), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0xFF, 1073741824) == (unsigned char)0xFF, "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0xFFFF, -1073741824) == (unsigned short)0xFFFF, "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0x42, 1000000000) == __builtin_stdc_rotate_left((unsigned char)0x42, 1000000000 % 8), "");
static_assert(__builtin_stdc_rotate_right((unsigned char)0x42, -1000000000) == __builtin_stdc_rotate_right((unsigned char)0x42, -1000000000 % 8), "");

static_assert(__builtin_stdc_rotate_left((unsigned char)0x12, -1000001) == __builtin_stdc_rotate_right((unsigned char)0x12, 1000001), "");
static_assert(__builtin_stdc_rotate_right((unsigned short)0x1234, -65537) == __builtin_stdc_rotate_left((unsigned short)0x1234, 65537), "");

constexpr unsigned _BitInt(67) large_pattern = 0x123456789ABCDEF0ULL;
static_assert(__builtin_stdc_rotate_left(large_pattern, 1000000) == __builtin_stdc_rotate_left(large_pattern, 1000000 % 67), "");
static_assert(__builtin_stdc_rotate_right(large_pattern, -1000000) == __builtin_stdc_rotate_right(large_pattern, -1000000 % 67), "");

static_assert(__builtin_stdc_rotate_left((unsigned _BitInt(23))0x123456, 1048576) == __builtin_stdc_rotate_left((unsigned _BitInt(23))0x123456, 1048576 % 23), "");
static_assert(__builtin_stdc_rotate_right((unsigned _BitInt(23))0x123456, -1048576) == __builtin_stdc_rotate_right((unsigned _BitInt(23))0x123456, -1048576 % 23), "");

} // namespace test_extreme_cases

} // namespace test_constexpr_stdc_rotate

namespace test_conversions {

struct UnsignedWrapper {
  operator unsigned int() const { return 42; }
};

struct RotateCount {
  operator int() const { return 5; }
};

enum RotateAmount {
  ROTATE_BY_4 = 4,
  ROTATE_BY_8 = 8
};

struct NoConversion {};

void test_implicit_conversions() {
  UnsignedWrapper uw;
  RotateCount rc;

  auto result1 = __builtin_stdc_rotate_left(uw, 3);
  auto result2 = __builtin_stdc_rotate_left(5U, rc);
  auto result3 = __builtin_stdc_rotate_left(uw, rc);
  auto result4 = __builtin_stdc_rotate_right(uw, RotateAmount::ROTATE_BY_4);

  bool b = true;
  auto result5 = __builtin_stdc_rotate_left(10U, b);
}

void test_invalid_types() {
  float f = 3.7f;
  auto result6 = __builtin_stdc_rotate_left(10U, f); // expected-error {{2nd argument must be a scalar integer type (was 'float')}}
  auto result7 = __builtin_stdc_rotate_right(10U, 2.9f); // expected-error {{2nd argument must be a scalar integer type (was 'float')}}

  NoConversion nc;
  auto result1 = __builtin_stdc_rotate_left(5U, nc); // expected-error {{2nd argument must be a scalar integer type (was 'NoConversion')}}
}

} // namespace test_conversions

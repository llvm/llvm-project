// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

namespace test_stdc_leading_zeros {

static_assert(__builtin_stdc_leading_zeros((unsigned char)0) == 8, "");
static_assert(__builtin_stdc_leading_zeros((unsigned char)1) == 7, "");
static_assert(__builtin_stdc_leading_zeros((unsigned char)0x80) == 0, "");
static_assert(__builtin_stdc_leading_zeros((unsigned char)0xFF) == 0, "");
static_assert(__builtin_stdc_leading_zeros((unsigned short)0) == 16, "");
static_assert(__builtin_stdc_leading_zeros((unsigned short)1) == 15, "");
static_assert(__builtin_stdc_leading_zeros((unsigned short)0x8000) == 0, "");
static_assert(__builtin_stdc_leading_zeros(0U) == 32, "");
static_assert(__builtin_stdc_leading_zeros(1U) == 31, "");
static_assert(__builtin_stdc_leading_zeros(0x80000000U) == 0, "");
static_assert(__builtin_stdc_leading_zeros(0ULL) == 64, "");
static_assert(__builtin_stdc_leading_zeros(1ULL) == 63, "");
static_assert(__builtin_stdc_leading_zeros(0x8000000000000000ULL) == 0, "");

} // namespace test_stdc_leading_zeros

namespace test_stdc_leading_ones {

static_assert(__builtin_stdc_leading_ones((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_leading_ones((unsigned char)0xFF) == 8, "");
static_assert(__builtin_stdc_leading_ones((unsigned char)0xF0) == 4, "");
static_assert(__builtin_stdc_leading_ones((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_leading_ones((unsigned short)0xFFFF) == 16, "");
static_assert(__builtin_stdc_leading_ones((unsigned short)0xF000) == 4, "");
static_assert(__builtin_stdc_leading_ones(0U) == 0, "");
static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFU) == 32, "");
static_assert(__builtin_stdc_leading_ones(0xF0000000U) == 4, "");
static_assert(__builtin_stdc_leading_ones(0ULL) == 0, "");
static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

} // namespace test_stdc_leading_ones

namespace test_stdc_trailing_zeros {

static_assert(__builtin_stdc_trailing_zeros((unsigned char)0) == 8, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned char)1) == 0, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned char)0x80) == 7, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned short)0) == 16, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned short)1) == 0, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned short)0x8000) == 15, "");
static_assert(__builtin_stdc_trailing_zeros(0U) == 32, "");
static_assert(__builtin_stdc_trailing_zeros(1U) == 0, "");
static_assert(__builtin_stdc_trailing_zeros(0x80000000U) == 31, "");
static_assert(__builtin_stdc_trailing_zeros(0ULL) == 64, "");
static_assert(__builtin_stdc_trailing_zeros(1ULL) == 0, "");
static_assert(__builtin_stdc_trailing_zeros(0x8000000000000000ULL) == 63, "");

} // namespace test_stdc_trailing_zeros

namespace test_stdc_trailing_ones {

static_assert(__builtin_stdc_trailing_ones((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned char)0xFF) == 8, "");
static_assert(__builtin_stdc_trailing_ones((unsigned char)0x0F) == 4, "");
static_assert(__builtin_stdc_trailing_ones((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned short)0xFFFF) == 16, "");
static_assert(__builtin_stdc_trailing_ones((unsigned short)0x000F) == 4, "");
static_assert(__builtin_stdc_trailing_ones(0U) == 0, "");
static_assert(__builtin_stdc_trailing_ones(1U) == 1, "");
static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFU) == 32, "");
static_assert(__builtin_stdc_trailing_ones(0x0000000FU) == 4, "");
static_assert(__builtin_stdc_trailing_ones(0ULL) == 0, "");
static_assert(__builtin_stdc_trailing_ones(1ULL) == 1, "");
static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

} // namespace test_stdc_trailing_ones

namespace test_stdc_first_leading_zero {

static_assert(__builtin_stdc_first_leading_zero((unsigned char)0) == 1, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xFF) == 0, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xF0) == 5, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned char)0x80) == 2, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned short)0) == 1, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned short)0xFFFF) == 0, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned short)0xF000) == 5, "");
static_assert(__builtin_stdc_first_leading_zero(0xFFFFFFFFU) == 0, "");
static_assert(__builtin_stdc_first_leading_zero(0xF0000000U) == 5, "");
static_assert(__builtin_stdc_first_leading_zero(0ULL) == 1, "");
static_assert(__builtin_stdc_first_leading_zero(0xFFFFFFFFFFFFFFFFULL) == 0, "");
static_assert(__builtin_stdc_first_leading_zero(0xF000000000000000ULL) == 5, "");

} // namespace test_stdc_first_leading_zero

namespace test_stdc_first_leading_one {

static_assert(__builtin_stdc_first_leading_one((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_first_leading_one((unsigned char)0x80) == 1, "");
static_assert(__builtin_stdc_first_leading_one((unsigned char)0x01) == 8, "");
static_assert(__builtin_stdc_first_leading_one((unsigned char)0x0F) == 5, "");
static_assert(__builtin_stdc_first_leading_one((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_first_leading_one((unsigned short)0x8000) == 1, "");
static_assert(__builtin_stdc_first_leading_one((unsigned short)1) == 16, "");
static_assert(__builtin_stdc_first_leading_one(0U) == 0, "");
static_assert(__builtin_stdc_first_leading_one(0x80000000U) == 1, "");
static_assert(__builtin_stdc_first_leading_one(1U) == 32, "");
static_assert(__builtin_stdc_first_leading_one(0ULL) == 0, "");
static_assert(__builtin_stdc_first_leading_one(0x8000000000000000ULL) == 1, "");
static_assert(__builtin_stdc_first_leading_one(1ULL) == 64, "");

} // namespace test_stdc_first_leading_one

namespace test_stdc_first_trailing_zero {

static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0xFF) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x0F) == 5, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x01) == 2, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0xFFFF) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0x000F) == 5, "");
static_assert(__builtin_stdc_first_trailing_zero(0xFFFFFFFFU) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero(0x0000000FU) == 5, "");
static_assert(__builtin_stdc_first_trailing_zero(0ULL) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero(0xFFFFFFFFFFFFFFFFULL) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero(0x000000000000000FULL) == 5, "");

} // namespace test_stdc_first_trailing_zero

namespace test_stdc_first_trailing_one {

static_assert(__builtin_stdc_first_trailing_one((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x01) == 1, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x80) == 8, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned char)0xF0) == 5, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned short)1) == 1, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned short)0x8000) == 16, "");
static_assert(__builtin_stdc_first_trailing_one(0U) == 0, "");
static_assert(__builtin_stdc_first_trailing_one(0x80000000U) == 32, "");
static_assert(__builtin_stdc_first_trailing_one(1U) == 1, "");
static_assert(__builtin_stdc_first_trailing_one(0ULL) == 0, "");
static_assert(__builtin_stdc_first_trailing_one(1ULL) == 1, "");
static_assert(__builtin_stdc_first_trailing_one(0x8000000000000000ULL) == 64, "");

} // namespace test_stdc_first_trailing_one

namespace test_stdc_count_zeros {

static_assert(__builtin_stdc_count_zeros((unsigned char)0) == 8, "");
static_assert(__builtin_stdc_count_zeros((unsigned char)0xFF) == 0, "");
static_assert(__builtin_stdc_count_zeros((unsigned char)0xAA) == 4, "");
static_assert(__builtin_stdc_count_zeros((unsigned short)0) == 16, "");
static_assert(__builtin_stdc_count_zeros((unsigned short)0xFFFF) == 0, "");
static_assert(__builtin_stdc_count_zeros((unsigned short)0xAAAA) == 8, "");
static_assert(__builtin_stdc_count_zeros(0U) == 32, "");
static_assert(__builtin_stdc_count_zeros(0xFFFFFFFFU) == 0, "");
static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAU) == 16, "");
static_assert(__builtin_stdc_count_zeros(0ULL) == 64, "");
static_assert(__builtin_stdc_count_zeros(0xFFFFFFFFFFFFFFFFULL) == 0, "");
static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAAAAAAAAAULL) == 32, "");

} // namespace test_stdc_count_zeros

namespace test_stdc_count_ones {

static_assert(__builtin_stdc_count_ones((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_count_ones((unsigned char)0xFF) == 8, "");
static_assert(__builtin_stdc_count_ones((unsigned char)0xAA) == 4, "");
static_assert(__builtin_stdc_count_ones((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_count_ones((unsigned short)0xFFFF) == 16, "");
static_assert(__builtin_stdc_count_ones((unsigned short)0xAAAA) == 8, "");
static_assert(__builtin_stdc_count_ones(0U) == 0, "");
static_assert(__builtin_stdc_count_ones(0xFFFFFFFFU) == 32, "");
static_assert(__builtin_stdc_count_ones(0xAAAAAAAAU) == 16, "");
static_assert(__builtin_stdc_count_ones(0ULL) == 0, "");
static_assert(__builtin_stdc_count_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");
static_assert(__builtin_stdc_count_ones(0xAAAAAAAAAAAAAAAAULL) == 32, "");

} // namespace test_stdc_count_ones

namespace test_stdc_has_single_bit {

static_assert(__builtin_stdc_has_single_bit((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned char)1) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned char)2) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned char)3) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned char)0x80) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned short)1) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned short)0x8000) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned short)0xFFFF) == 0, "");
static_assert(__builtin_stdc_has_single_bit(0U) == 0, "");
static_assert(__builtin_stdc_has_single_bit(1U) == 1, "");
static_assert(__builtin_stdc_has_single_bit(0x80000000U) == 1, "");
static_assert(__builtin_stdc_has_single_bit(0xFFFFFFFFU) == 0, "");
static_assert(__builtin_stdc_has_single_bit(0ULL) == 0, "");
static_assert(__builtin_stdc_has_single_bit(1ULL) == 1, "");
static_assert(__builtin_stdc_has_single_bit(0x8000000000000000ULL) == 1, "");
static_assert(__builtin_stdc_has_single_bit(0xFFFFFFFFFFFFFFFFULL) == 0, "");

} // namespace test_stdc_has_single_bit

namespace test_stdc_bit_width {

static_assert(__builtin_stdc_bit_width((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned char)1) == 1, "");
static_assert(__builtin_stdc_bit_width((unsigned char)2) == 2, "");
static_assert(__builtin_stdc_bit_width((unsigned char)3) == 2, "");
static_assert(__builtin_stdc_bit_width((unsigned char)0x80) == 8, "");
static_assert(__builtin_stdc_bit_width((unsigned char)0xFF) == 8, "");
static_assert(__builtin_stdc_bit_width((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned short)1) == 1, "");
static_assert(__builtin_stdc_bit_width((unsigned short)0x8000) == 16, "");
static_assert(__builtin_stdc_bit_width(0U) == 0, "");
static_assert(__builtin_stdc_bit_width(1U) == 1, "");
static_assert(__builtin_stdc_bit_width(0x80000000U) == 32, "");
static_assert(__builtin_stdc_bit_width(0ULL) == 0, "");
static_assert(__builtin_stdc_bit_width(1ULL) == 1, "");
static_assert(__builtin_stdc_bit_width(0x8000000000000000ULL) == 64, "");

} // namespace test_stdc_bit_width

namespace test_stdc_bit_floor {

static_assert(__builtin_stdc_bit_floor((unsigned char)0) == 0, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)1) == 1, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)2) == 2, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)3) == 2, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)4) == 4, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)5) == 4, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)0x80) == 0x80, "");
static_assert(__builtin_stdc_bit_floor((unsigned char)0xFF) == 0x80, "");
static_assert(__builtin_stdc_bit_floor((unsigned short)0) == 0, "");
static_assert(__builtin_stdc_bit_floor((unsigned short)1) == 1, "");
static_assert(__builtin_stdc_bit_floor((unsigned short)0xFFFF) == 0x8000, "");
static_assert(__builtin_stdc_bit_floor(0U) == 0U, "");
static_assert(__builtin_stdc_bit_floor(1U) == 1U, "");
static_assert(__builtin_stdc_bit_floor(7U) == 4U, "");
static_assert(__builtin_stdc_bit_floor(0x80000000U) == 0x80000000U, "");
static_assert(__builtin_stdc_bit_floor(0xFFFFFFFFU) == 0x80000000U, "");
static_assert(__builtin_stdc_bit_floor(0ULL) == 0ULL, "");
static_assert(__builtin_stdc_bit_floor(1ULL) == 1ULL, "");
static_assert(__builtin_stdc_bit_floor(0xFFFFFFFFFFFFFFFFULL) == 0x8000000000000000ULL, "");

} // namespace test_stdc_bit_floor

namespace test_stdc_bit_ceil {

static_assert(__builtin_stdc_bit_ceil((unsigned char)0) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)1) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)2) == 2, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)3) == 4, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)4) == 4, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)5) == 8, "");
static_assert(__builtin_stdc_bit_ceil((unsigned char)0x80) == 0x80, "");
static_assert(__builtin_stdc_bit_ceil((unsigned short)0) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned short)1) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned short)7) == 8, "");
static_assert(__builtin_stdc_bit_ceil((unsigned short)0x8000) == 0x8000, "");
static_assert(__builtin_stdc_bit_ceil(0U) == 1U, "");
static_assert(__builtin_stdc_bit_ceil(1U) == 1U, "");
static_assert(__builtin_stdc_bit_ceil(7U) == 8U, "");
static_assert(__builtin_stdc_bit_ceil(0x80000000U) == 0x80000000U, "");
// Overflow: next power of 2 exceeds type width; wraps to zero.
static_assert(__builtin_stdc_bit_ceil(0xFFFFFFFFU) == 0U, "");
static_assert(__builtin_stdc_bit_ceil(0ULL) == 1ULL, "");
static_assert(__builtin_stdc_bit_ceil(1ULL) == 1ULL, "");
static_assert(__builtin_stdc_bit_ceil(7ULL) == 8ULL, "");
static_assert(__builtin_stdc_bit_ceil(0x8000000000000000ULL) == 0x8000000000000000ULL, "");

} // namespace test_stdc_bit_ceil

#ifdef __SIZEOF_INT128__
namespace test_int128 {

static_assert(__builtin_stdc_leading_zeros((unsigned __int128)0) == 128, "");
static_assert(__builtin_stdc_leading_zeros((unsigned __int128)1) == 127, "");
static_assert(__builtin_stdc_leading_ones((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_leading_ones((unsigned __int128)-1) == 128, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned __int128)0) == 128, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned __int128)1) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned __int128)-1) == 128, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned __int128)0) == 1, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned __int128)-1) == 0, "");
static_assert(__builtin_stdc_first_leading_one((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned __int128)0) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned __int128)-1) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned __int128)1) == 1, "");
static_assert(__builtin_stdc_count_ones((unsigned __int128)0xFFFFFFFF) == 32, "");
static_assert(__builtin_stdc_count_zeros((unsigned __int128)0) == 128, "");
static_assert(__builtin_stdc_count_zeros((unsigned __int128)-1) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned __int128)1) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned __int128)3) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned __int128)1) == 1, "");
static_assert(__builtin_stdc_bit_floor((unsigned __int128)0) == 0, "");
static_assert(__builtin_stdc_bit_floor((unsigned __int128)1) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned __int128)0) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned __int128)1) == 1, "");
static_assert(__builtin_stdc_bit_ceil((unsigned __int128)3) == 4, "");

// MSB tests using constexpr to form the value.
constexpr unsigned __int128 int128_one = 1;
constexpr unsigned __int128 int128_msb = int128_one << 127;
static_assert(__builtin_stdc_leading_zeros(int128_msb) == 0, "");
static_assert(__builtin_stdc_leading_ones(int128_msb) == 1, "");
static_assert(__builtin_stdc_trailing_zeros(int128_msb) == 127, "");
static_assert(__builtin_stdc_trailing_ones(int128_msb) == 0, "");
static_assert(__builtin_stdc_first_leading_zero(int128_msb) == 2, "");
static_assert(__builtin_stdc_first_leading_one(int128_msb) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero(int128_msb) == 1, "");
static_assert(__builtin_stdc_first_trailing_one(int128_msb) == 128, "");
static_assert(__builtin_stdc_count_ones(int128_msb) == 1, "");
static_assert(__builtin_stdc_count_zeros(int128_msb) == 127, "");
static_assert(__builtin_stdc_has_single_bit(int128_msb) == 1, "");
static_assert(__builtin_stdc_bit_width(int128_msb) == 128, "");
static_assert(__builtin_stdc_bit_floor(int128_msb) == int128_msb, "");
static_assert(__builtin_stdc_bit_ceil(int128_msb) == int128_msb, "");

} // namespace test_int128
#endif // __SIZEOF_INT128__

namespace test_bitint {

// _BitInt(37): all 14 builtins
static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))0) == 37, "");
static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))1) == 36, "");
static_assert(__builtin_stdc_leading_ones((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_leading_ones((unsigned _BitInt(37))-1) == 37, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(37))0) == 37, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(37))1) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_trailing_ones((unsigned _BitInt(37))-1) == 37, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned _BitInt(37))0) == 1, "");
static_assert(__builtin_stdc_first_leading_zero((unsigned _BitInt(37))-1) == 0, "");
static_assert(__builtin_stdc_first_leading_one((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_first_leading_one((unsigned _BitInt(37))1) == 37, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned _BitInt(37))0) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero((unsigned _BitInt(37))-1) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_first_trailing_one((unsigned _BitInt(37))1) == 1, "");
static_assert(__builtin_stdc_count_ones((unsigned _BitInt(37))0x1F) == 5, "");
static_assert(__builtin_stdc_count_zeros((unsigned _BitInt(37))0) == 37, "");
static_assert(__builtin_stdc_count_zeros((unsigned _BitInt(37))-1) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))0x10) == 1, "");
static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))3) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))0x10) == 5, "");
static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))0) == 0, "");
static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))-1) == 37, "");
static_assert(__builtin_stdc_bit_floor((unsigned _BitInt(37))0x1F) == 0x10, "");
static_assert(__builtin_stdc_bit_ceil((unsigned _BitInt(37))0x11) == 0x20, "");
// Overflow: next power of 2 exceeds _BitInt(17) width; wraps to zero.
static_assert(__builtin_stdc_bit_ceil((unsigned _BitInt(17))(-1)) ==
              (unsigned _BitInt(17))(0), "");

// _BitInt(128): sparse pattern, leading zero count, popcount.
constexpr unsigned _BitInt(128) bi128_pattern = 0x123456789ABCDEF0ULL;
static_assert(__builtin_stdc_count_ones(bi128_pattern) == 32, "");
static_assert(__builtin_stdc_leading_zeros(bi128_pattern) == 67, "");
static_assert(__builtin_stdc_count_zeros(bi128_pattern) == 96, "");

// _BitInt(9): boundary width near a byte.
static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(9))0) == 9, "");
static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(9))0x100) == 8, "");
static_assert(__builtin_stdc_count_ones((unsigned _BitInt(9))0x1FF) == 9, "");
static_assert(__builtin_stdc_count_zeros((unsigned _BitInt(9))0x1FF) == 0, "");
static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(9))0x100) == 1, "");
static_assert(__builtin_stdc_bit_width((unsigned _BitInt(9))0x100) == 9, "");
static_assert(__builtin_stdc_bit_floor((unsigned _BitInt(9))0x1FF) == 0x100, "");

} // namespace test_bitint

namespace test_stdc_ulong {

constexpr unsigned long ULongWidth = __SIZEOF_LONG__ * 8;
constexpr unsigned long ULongMSB = 1UL << (ULongWidth - 1);
constexpr unsigned long ULongAllOnes = ~0UL;
constexpr unsigned long ULongTopNibbleOnes = ULongAllOnes << (ULongWidth - 4);

static_assert(__builtin_stdc_leading_zeros(0UL) == ULongWidth, "");
static_assert(__builtin_stdc_leading_zeros(1UL) == ULongWidth - 1, "");
static_assert(__builtin_stdc_leading_zeros(ULongMSB) == 0, "");
static_assert(__builtin_stdc_leading_ones(0UL) == 0, "");
static_assert(__builtin_stdc_leading_ones(ULongAllOnes) == ULongWidth, "");
static_assert(__builtin_stdc_leading_ones(ULongTopNibbleOnes) == 4, "");
static_assert(__builtin_stdc_trailing_zeros(0UL) == ULongWidth, "");
static_assert(__builtin_stdc_trailing_zeros(1UL) == 0, "");
static_assert(__builtin_stdc_trailing_zeros(ULongMSB) == ULongWidth - 1, "");
static_assert(__builtin_stdc_trailing_ones(0UL) == 0, "");
static_assert(__builtin_stdc_trailing_ones(ULongAllOnes) == ULongWidth, "");
static_assert(__builtin_stdc_trailing_ones(0xFUL) == 4, "");
static_assert(__builtin_stdc_first_leading_zero(0UL) == 1, "");
static_assert(__builtin_stdc_first_leading_zero(ULongAllOnes) == 0, "");
static_assert(__builtin_stdc_first_leading_zero(ULongTopNibbleOnes) == 5, "");
static_assert(__builtin_stdc_first_leading_one(0UL) == 0, "");
static_assert(__builtin_stdc_first_leading_one(ULongMSB) == 1, "");
static_assert(__builtin_stdc_first_leading_one(1UL) == ULongWidth, "");
static_assert(__builtin_stdc_first_trailing_zero(0UL) == 1, "");
static_assert(__builtin_stdc_first_trailing_zero(ULongAllOnes) == 0, "");
static_assert(__builtin_stdc_first_trailing_zero(0xFUL) == 5, "");
static_assert(__builtin_stdc_first_trailing_one(0UL) == 0, "");
static_assert(__builtin_stdc_first_trailing_one(1UL) == 1, "");
static_assert(__builtin_stdc_first_trailing_one(ULongMSB) == ULongWidth, "");
static_assert(__builtin_stdc_count_zeros(0UL) == ULongWidth, "");
static_assert(__builtin_stdc_count_zeros(ULongAllOnes) == 0, "");
static_assert(__builtin_stdc_count_zeros(ULongMSB) == ULongWidth - 1, "");
static_assert(__builtin_stdc_count_ones(0UL) == 0, "");
static_assert(__builtin_stdc_count_ones(ULongAllOnes) == ULongWidth, "");
static_assert(__builtin_stdc_count_ones(ULongMSB) == 1, "");
static_assert(__builtin_stdc_has_single_bit(0UL) == 0, "");
static_assert(__builtin_stdc_has_single_bit(1UL) == 1, "");
static_assert(__builtin_stdc_has_single_bit(ULongMSB) == 1, "");
static_assert(__builtin_stdc_has_single_bit(ULongAllOnes) == 0, "");
static_assert(__builtin_stdc_bit_width(0UL) == 0, "");
static_assert(__builtin_stdc_bit_width(1UL) == 1, "");
static_assert(__builtin_stdc_bit_width(ULongMSB) == ULongWidth, "");
static_assert(__builtin_stdc_bit_floor(0UL) == 0UL, "");
static_assert(__builtin_stdc_bit_floor(1UL) == 1UL, "");
static_assert(__builtin_stdc_bit_floor(ULongAllOnes) == ULongMSB, "");
static_assert(__builtin_stdc_bit_ceil(0UL) == 1UL, "");
static_assert(__builtin_stdc_bit_ceil(1UL) == 1UL, "");
static_assert(__builtin_stdc_bit_ceil(7UL) == 8UL, "");
static_assert(__builtin_stdc_bit_ceil(ULongMSB) == ULongMSB, "");

} // namespace test_stdc_ulong

namespace test_errors {

void test_invalid_types(int si, float f) {
  __builtin_stdc_leading_zeros(si);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_leading_zeros(f);   // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}
  __builtin_stdc_leading_zeros(-1);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}

  __builtin_stdc_leading_ones(si);   // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_leading_ones(f);    // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_trailing_zeros(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_trailing_zeros(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_trailing_ones(si);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_trailing_ones(f);   // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_first_leading_zero(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_first_leading_zero(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_first_leading_one(si);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_first_leading_one(f);   // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_first_trailing_zero(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_first_trailing_zero(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_first_trailing_one(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_first_trailing_one(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_count_ones(si);     // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_count_ones(f);      // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_count_zeros(si);    // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_count_zeros(f);     // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_has_single_bit(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_has_single_bit(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_bit_width(si);      // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_bit_width(f);       // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_bit_floor(si);      // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_bit_floor(f);       // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}

  __builtin_stdc_bit_ceil(si);       // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_bit_ceil(f);        // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}
}

} // namespace test_errors

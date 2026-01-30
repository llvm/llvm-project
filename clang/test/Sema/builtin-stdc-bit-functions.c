// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

// Test stdc_leading_zeros
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)1) == 7, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0x80) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned short)0x8000) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros(0U) == 32, "");
_Static_assert(__builtin_stdc_leading_zeros(1U) == 31, "");
_Static_assert(__builtin_stdc_leading_zeros(0x80000000U) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros(0ULL) == 64, "");
_Static_assert(__builtin_stdc_leading_zeros(1ULL) == 63, "");
_Static_assert(__builtin_stdc_leading_zeros(0x8000000000000000ULL) == 0, "");

// Test stdc_leading_ones
_Static_assert(__builtin_stdc_leading_ones((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned char)0xF0) == 4, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_leading_ones(0xF0000000U) == 4, "");
_Static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

// Test stdc_trailing_zeros
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)1) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0x80) == 7, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_trailing_zeros(0U) == 32, "");
_Static_assert(__builtin_stdc_trailing_zeros(0x80000000U) == 31, "");
_Static_assert(__builtin_stdc_trailing_zeros(0ULL) == 64, "");
_Static_assert(__builtin_stdc_trailing_zeros(0x8000000000000000ULL) == 63, "");

// Test stdc_trailing_ones
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0x0F) == 4, "");
_Static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_trailing_ones(0x0000000FU) == 4, "");
_Static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

// Test stdc_first_leading_zero
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xF0) == 5, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0x80) == 2, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xF0000000U) == 5, "");

// Test stdc_first_leading_one
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x80) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x01) == 8, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x0F) == 5, "");
_Static_assert(__builtin_stdc_first_leading_one(0U) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one(0x80000000U) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one(1U) == 32, "");

// Test stdc_first_trailing_zero
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x0F) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x01) == 2, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0x0000000FU) == 5, "");

// Test stdc_first_trailing_one
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x01) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x80) == 8, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0xF0) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_one(0U) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one(0x80000000U) == 32, "");
_Static_assert(__builtin_stdc_first_trailing_one(1U) == 1, "");

// Test stdc_count_zeros
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0xAA) == 4, "");
_Static_assert(__builtin_stdc_count_zeros(0U) == 32, "");
_Static_assert(__builtin_stdc_count_zeros(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAU) == 16, "");

// Test stdc_count_ones
_Static_assert(__builtin_stdc_count_ones((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_count_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_count_ones((unsigned char)0xAA) == 4, "");
_Static_assert(__builtin_stdc_count_ones(0U) == 0, "");
_Static_assert(__builtin_stdc_count_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_count_ones(0xAAAAAAAAU) == 16, "");

// Test stdc_has_single_bit
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)2) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)3) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0x80) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0U) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(1U) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0x80000000U) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0xFFFFFFFFU) == 0, "");

// Test stdc_bit_width
_Static_assert(__builtin_stdc_bit_width((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)3) == 2, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)0x80) == 8, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_bit_width(0U) == 0, "");
_Static_assert(__builtin_stdc_bit_width(1U) == 1, "");
_Static_assert(__builtin_stdc_bit_width(0x80000000U) == 32, "");

// Test stdc_bit_floor
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)3) == 2, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)4) == 4, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)5) == 4, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0x80) == 0x80, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0xFF) == 0x80, "");
_Static_assert(__builtin_stdc_bit_floor(0U) == 0U, "");
_Static_assert(__builtin_stdc_bit_floor(1U) == 1U, "");
_Static_assert(__builtin_stdc_bit_floor(7U) == 4U, "");
_Static_assert(__builtin_stdc_bit_floor(0x80000000U) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_floor(0xFFFFFFFFU) == 0x80000000U, "");

// Test stdc_bit_ceil
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)3) == 4, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)4) == 4, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)5) == 8, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0x80) == 0x80, "");
_Static_assert(__builtin_stdc_bit_ceil(0U) == 1U, "");
_Static_assert(__builtin_stdc_bit_ceil(1U) == 1U, "");
_Static_assert(__builtin_stdc_bit_ceil(7U) == 8U, "");
_Static_assert(__builtin_stdc_bit_ceil(0x80000000U) == 0x80000000U, "");

// Test with _BitInt types
_Static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))0) == 37, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))1) == 36, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(37))0) == 37, "");
_Static_assert(__builtin_stdc_count_ones((unsigned _BitInt(37))0x1F) == 5, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))0x10) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))0x10) == 5, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned _BitInt(37))0x1F) == 0x10, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned _BitInt(37))0x11) == 0x20, "");

// Test with __int128
_Static_assert(__builtin_stdc_leading_zeros((unsigned __int128)0) == 128, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned __int128)1) == 127, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned __int128)0) == 128, "");
_Static_assert(__builtin_stdc_count_ones((unsigned __int128)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned __int128)1) == 1, "");

// Error cases
void test_errors(int si, float f) {
  unsigned int ui = 5;

  __builtin_stdc_leading_zeros(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_leading_zeros(f);  // expected-error {{1st argument must be a scalar unsigned integer type (was 'float')}}
  __builtin_stdc_leading_zeros(-1); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}

  __builtin_stdc_count_ones(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_has_single_bit(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
  __builtin_stdc_bit_ceil(si); // expected-error {{1st argument must be a scalar unsigned integer type (was 'int')}}
}

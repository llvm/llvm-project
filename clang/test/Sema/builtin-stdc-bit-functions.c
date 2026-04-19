// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c23 -isystem %S/Inputs -fsyntax-only -verify %s

// Test stdc_leading_zeros
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)1) == 7, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0x80) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned short)1) == 15, "");
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
_Static_assert(__builtin_stdc_leading_ones((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned short)0xF000) == 4, "");
_Static_assert(__builtin_stdc_leading_ones(0U) == 0, "");
_Static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_leading_ones(0xF0000000U) == 4, "");
_Static_assert(__builtin_stdc_leading_ones(0ULL) == 0, "");
_Static_assert(__builtin_stdc_leading_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

// Test stdc_trailing_zeros
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)1) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0x80) == 7, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned short)1) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned short)0x8000) == 15, "");
_Static_assert(__builtin_stdc_trailing_zeros(0U) == 32, "");
_Static_assert(__builtin_stdc_trailing_zeros(1U) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros(0x80000000U) == 31, "");
_Static_assert(__builtin_stdc_trailing_zeros(0ULL) == 64, "");
_Static_assert(__builtin_stdc_trailing_zeros(1ULL) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros(0x8000000000000000ULL) == 63, "");

// Test stdc_trailing_ones
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0x0F) == 4, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned short)0x000F) == 4, "");
_Static_assert(__builtin_stdc_trailing_ones(0U) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones(1U) == 1, "");
_Static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_trailing_ones(0x0000000FU) == 4, "");
_Static_assert(__builtin_stdc_trailing_ones(0ULL) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones(1ULL) == 1, "");
_Static_assert(__builtin_stdc_trailing_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");

// Test stdc_first_leading_zero
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xF0) == 5, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0x80) == 2, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned short)0) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned short)0xFFFF) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned short)0xF000) == 5, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xF0000000U) == 5, "");
_Static_assert(__builtin_stdc_first_leading_zero(0ULL) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xFFFFFFFFFFFFFFFFULL) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero(0xF000000000000000ULL) == 5, "");

// Test stdc_first_leading_one
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x80) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x01) == 8, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x0F) == 5, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned short)0x8000) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned short)1) == 16, "");
_Static_assert(__builtin_stdc_first_leading_one(0U) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one(0x80000000U) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one(1U) == 32, "");
_Static_assert(__builtin_stdc_first_leading_one(0ULL) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one(0x8000000000000000ULL) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one(1ULL) == 64, "");

// Test stdc_first_trailing_zero
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x0F) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x01) == 2, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0xFFFF) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0x000F) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0x0000000FU) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0ULL) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0xFFFFFFFFFFFFFFFFULL) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0x000000000000000FULL) == 5, "");

// Test stdc_first_trailing_one
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x01) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x80) == 8, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0xF0) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned short)1) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned short)0x8000) == 16, "");
_Static_assert(__builtin_stdc_first_trailing_one(0U) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one(0x80000000U) == 32, "");
_Static_assert(__builtin_stdc_first_trailing_one(1U) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one(0ULL) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one(1ULL) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one(0x8000000000000000ULL) == 64, "");

// Test stdc_count_zeros
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0xAA) == 4, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned short)0xFFFF) == 0, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned short)0xAAAA) == 8, "");
_Static_assert(__builtin_stdc_count_zeros(0U) == 32, "");
_Static_assert(__builtin_stdc_count_zeros(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAU) == 16, "");
_Static_assert(__builtin_stdc_count_zeros(0ULL) == 64, "");
_Static_assert(__builtin_stdc_count_zeros(0xFFFFFFFFFFFFFFFFULL) == 0, "");
_Static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAAAAAAAAAULL) == 32, "");

// Test stdc_count_ones
_Static_assert(__builtin_stdc_count_ones((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_count_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_count_ones((unsigned char)0xAA) == 4, "");
_Static_assert(__builtin_stdc_count_ones((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_count_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_count_ones((unsigned short)0xAAAA) == 8, "");
_Static_assert(__builtin_stdc_count_ones(0U) == 0, "");
_Static_assert(__builtin_stdc_count_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(__builtin_stdc_count_ones(0xAAAAAAAAU) == 16, "");
_Static_assert(__builtin_stdc_count_ones(0ULL) == 0, "");
_Static_assert(__builtin_stdc_count_ones(0xFFFFFFFFFFFFFFFFULL) == 64, "");
_Static_assert(__builtin_stdc_count_ones(0xAAAAAAAAAAAAAAAAULL) == 32, "");

// Test stdc_has_single_bit
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)2) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)3) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0x80) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned short)1) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned short)0x8000) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned short)0xFFFF) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(0U) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(1U) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0x80000000U) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0xFFFFFFFFU) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(0ULL) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(1ULL) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0x8000000000000000ULL) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0xFFFFFFFFFFFFFFFFULL) == 0, "");

// Test stdc_bit_width
_Static_assert(__builtin_stdc_bit_width((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)3) == 2, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)0x80) == 8, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_bit_width((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned short)1) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned short)0x8000) == 16, "");
_Static_assert(__builtin_stdc_bit_width(0U) == 0, "");
_Static_assert(__builtin_stdc_bit_width(1U) == 1, "");
_Static_assert(__builtin_stdc_bit_width(0x80000000U) == 32, "");
_Static_assert(__builtin_stdc_bit_width(0ULL) == 0, "");
_Static_assert(__builtin_stdc_bit_width(1ULL) == 1, "");
_Static_assert(__builtin_stdc_bit_width(0x8000000000000000ULL) == 64, "");

// Test stdc_bit_floor
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0) == 0, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)3) == 2, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)4) == 4, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)5) == 4, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0x80) == 0x80, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0xFF) == 0x80, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned short)0) == 0, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned short)1) == 1, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned short)0xFFFF) == 0x8000, "");
_Static_assert(__builtin_stdc_bit_floor(0U) == 0U, "");
_Static_assert(__builtin_stdc_bit_floor(1U) == 1U, "");
_Static_assert(__builtin_stdc_bit_floor(7U) == 4U, "");
_Static_assert(__builtin_stdc_bit_floor(0x80000000U) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_floor(0xFFFFFFFFU) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_floor(0ULL) == 0ULL, "");
_Static_assert(__builtin_stdc_bit_floor(1ULL) == 1ULL, "");
_Static_assert(__builtin_stdc_bit_floor(0xFFFFFFFFFFFFFFFFULL) == 0x8000000000000000ULL, "");

// Test stdc_bit_ceil
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)1) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)2) == 2, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)3) == 4, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)4) == 4, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)5) == 8, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0x80) == 0x80, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0x81) == 0, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0xFE) == 0, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0xFF) == 0, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)0) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)1) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)7) == 8, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)0x8000) == 0x8000, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)0x8001) == 0, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)0xFFFF) == 0, "");
_Static_assert(__builtin_stdc_bit_ceil(0U) == 1U, "");
_Static_assert(__builtin_stdc_bit_ceil(1U) == 1U, "");
_Static_assert(__builtin_stdc_bit_ceil(7U) == 8U, "");
_Static_assert(__builtin_stdc_bit_ceil(0x80000000U) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_ceil(0x80000001U) == 0U, "");
_Static_assert(__builtin_stdc_bit_ceil(0xFFFFFFFFU) == 0U, "");
_Static_assert(__builtin_stdc_bit_ceil(0ULL) == 1ULL, "");
_Static_assert(__builtin_stdc_bit_ceil(1ULL) == 1ULL, "");
_Static_assert(__builtin_stdc_bit_ceil(7ULL) == 8ULL, "");
_Static_assert(__builtin_stdc_bit_ceil(0x8000000000000000ULL) == 0x8000000000000000ULL, "");
_Static_assert(__builtin_stdc_bit_ceil(0x8000000000000001ULL) == 0ULL, "");
_Static_assert(__builtin_stdc_bit_ceil(0xFFFFFFFFFFFFFFFFULL) == 0ULL, "");

// Test with _BitInt types - cover all 14 builtins
_Static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))0) == 37, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned _BitInt(37))1) == 36, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned _BitInt(37))-1) == 37, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(37))0) == 37, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned _BitInt(37))1) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned _BitInt(37))-1) == 37, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned _BitInt(37))0) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned _BitInt(37))-1) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned _BitInt(37))1) == 37, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned _BitInt(37))0) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned _BitInt(37))-1) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned _BitInt(37))1) == 1, "");
_Static_assert(__builtin_stdc_count_ones((unsigned _BitInt(37))0x1F) == 5, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned _BitInt(37))0) == 37, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned _BitInt(37))-1) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))0x10) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned _BitInt(37))3) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))0x10) == 5, "");
_Static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))0) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned _BitInt(37))-1) == 37, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned _BitInt(37))0x1F) == 0x10, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned _BitInt(37))0x11) == 0x20, "");

#ifdef __SIZEOF_INT128__
// Test with __int128 - cover all 14 builtins
_Static_assert(__builtin_stdc_leading_zeros((unsigned __int128)0) == 128, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned __int128)1) == 127, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned __int128)-1) == 128, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned __int128)0) == 128, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned __int128)1) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned __int128)-1) == 128, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned __int128)0) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned __int128)-1) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned __int128)0) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned __int128)-1) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_count_ones((unsigned __int128)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned __int128)0) == 128, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned __int128)-1) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned __int128)3) == 0, "");
_Static_assert(__builtin_stdc_bit_width((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned __int128)0) == 0, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned __int128)0) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned __int128)1) == 1, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned __int128)3) == 4, "");
#endif // __SIZEOF_INT128__

// Test with unsigned long across all targets.
enum { ULONG_WIDTH = __SIZEOF_LONG__ * 8 };

_Static_assert(__builtin_stdc_leading_zeros(0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_leading_zeros(1UL) == ULONG_WIDTH - 1, "");
_Static_assert(__builtin_stdc_leading_zeros(1UL << (ULONG_WIDTH - 1)) == 0, "");
_Static_assert(__builtin_stdc_leading_ones(0UL) == 0, "");
_Static_assert(__builtin_stdc_leading_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_leading_ones(~0UL << (ULONG_WIDTH - 4)) == 4, "");
_Static_assert(__builtin_stdc_trailing_zeros(0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_trailing_zeros(1UL) == 0, "");
_Static_assert(__builtin_stdc_trailing_zeros(1UL << (ULONG_WIDTH - 1)) == ULONG_WIDTH - 1, "");
_Static_assert(__builtin_stdc_trailing_ones(0UL) == 0, "");
_Static_assert(__builtin_stdc_trailing_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_trailing_ones(0xFUL) == 4, "");
_Static_assert(__builtin_stdc_first_leading_zero(0UL) == 1, "");
_Static_assert(__builtin_stdc_first_leading_zero(~0UL) == 0, "");
_Static_assert(__builtin_stdc_first_leading_zero(~0UL << (ULONG_WIDTH - 4)) == 5, "");
_Static_assert(__builtin_stdc_first_leading_one(0UL) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one(1UL << (ULONG_WIDTH - 1)) == 1, "");
_Static_assert(__builtin_stdc_first_leading_one(1UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0UL) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_zero(~0UL) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_zero(0xFUL) == 5, "");
_Static_assert(__builtin_stdc_first_trailing_one(0UL) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one(1UL) == 1, "");
_Static_assert(__builtin_stdc_first_trailing_one(1UL << (ULONG_WIDTH - 1)) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_count_zeros(0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_count_zeros(~0UL) == 0, "");
_Static_assert(__builtin_stdc_count_zeros(1UL << (ULONG_WIDTH - 1)) == ULONG_WIDTH - 1, "");
_Static_assert(__builtin_stdc_count_ones(0UL) == 0, "");
_Static_assert(__builtin_stdc_count_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_count_ones(1UL << (ULONG_WIDTH - 1)) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(0UL) == 0, "");
_Static_assert(__builtin_stdc_has_single_bit(1UL) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(1UL << (ULONG_WIDTH - 1)) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(~0UL) == 0, "");
_Static_assert(__builtin_stdc_bit_width(0UL) == 0, "");
_Static_assert(__builtin_stdc_bit_width(1UL) == 1, "");
_Static_assert(__builtin_stdc_bit_width(1UL << (ULONG_WIDTH - 1)) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_bit_floor(0UL) == 0UL, "");
_Static_assert(__builtin_stdc_bit_floor(1UL) == 1UL, "");
_Static_assert(__builtin_stdc_bit_floor(~0UL) == (1UL << (ULONG_WIDTH - 1)), "");
_Static_assert(__builtin_stdc_bit_ceil(0UL) == 1UL, "");
_Static_assert(__builtin_stdc_bit_ceil(1UL) == 1UL, "");
_Static_assert(__builtin_stdc_bit_ceil(7UL) == 8UL, "");
_Static_assert(__builtin_stdc_bit_ceil(1UL << (ULONG_WIDTH - 1)) == (1UL << (ULONG_WIDTH - 1)), "");
_Static_assert(__builtin_stdc_bit_ceil((1UL << (ULONG_WIDTH - 1)) + 1) == 0UL, "");
_Static_assert(__builtin_stdc_bit_ceil(~0UL) == 0UL, "");

// alternating bit patterns
_Static_assert(__builtin_stdc_leading_zeros(0x55555555U) == 1, "");
_Static_assert(__builtin_stdc_leading_ones(0xAAAAAAAAU) == 1, "");
_Static_assert(__builtin_stdc_trailing_zeros(0xAAAAAAAAU) == 1, "");
_Static_assert(__builtin_stdc_trailing_ones(0x55555555U) == 1, "");
_Static_assert(__builtin_stdc_count_zeros(0x55555555U) == 16, "");
_Static_assert(__builtin_stdc_count_ones(0x55555555U) == 16, "");
_Static_assert(__builtin_stdc_count_zeros(0xAAAAAAAAU) == 16, "");
_Static_assert(__builtin_stdc_count_ones(0xAAAAAAAAU) == 16, "");
_Static_assert(__builtin_stdc_bit_width(0x55555555U) == 31, "");
_Static_assert(__builtin_stdc_bit_width(0xAAAAAAAAU) == 32, "");
_Static_assert(__builtin_stdc_bit_floor(0x55555555U) == 0x40000000U, "");
_Static_assert(__builtin_stdc_bit_floor(0xAAAAAAAAU) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_ceil(0x55555555U) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_ceil(0xAAAAAAAAU) == 0U, "");

// nibble patterns
_Static_assert(__builtin_stdc_leading_zeros(0x0F0F0F0FU) == 4, "");
_Static_assert(__builtin_stdc_leading_zeros(0xF0F0F0F0U) == 0, "");
_Static_assert(__builtin_stdc_count_ones(0x0F0F0F0FU) == 16, "");
_Static_assert(__builtin_stdc_count_ones(0xF0F0F0F0U) == 16, "");
_Static_assert(__builtin_stdc_bit_ceil(0x0F0F0F0FU) == 0x10000000U, "");
_Static_assert(__builtin_stdc_bit_ceil(0xF0F0F0F0U) == 0U, "");

// Error cases - all 14 builtins reject bool and enumeration arguments
enum UnsignedEnum { UE_A = 0, UE_B = 1 };
enum SignedEnum { SE_A = -1, SE_B = 1 };
void test_bool_enum_errors(_Bool b, enum UnsignedEnum ue, enum SignedEnum se) {
  __builtin_stdc_leading_zeros(b);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_leading_zeros(ue);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}
  __builtin_stdc_leading_zeros(se);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum SignedEnum')}}

  __builtin_stdc_leading_ones(b);    // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_leading_ones(ue);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_trailing_zeros(b);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_trailing_zeros(ue); // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_trailing_ones(b);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_trailing_ones(ue);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_first_leading_zero(b);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_first_leading_zero(ue);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_first_leading_one(b);    // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_first_leading_one(ue);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_first_trailing_zero(b);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_first_trailing_zero(ue); // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_first_trailing_one(b);   // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_first_trailing_one(ue);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_count_ones(b);      // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_count_ones(ue);     // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_count_zeros(b);     // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_count_zeros(ue);    // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_has_single_bit(b);  // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_has_single_bit(ue); // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_bit_width(b);       // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_bit_width(ue);      // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_bit_floor(b);       // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_bit_floor(ue);      // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}

  __builtin_stdc_bit_ceil(b);        // expected-error {{1st argument must not be a boolean or enumeration type (was 'bool')}}
  __builtin_stdc_bit_ceil(ue);       // expected-error {{1st argument must not be a boolean or enumeration type (was 'enum UnsignedEnum')}}
}

// Error cases - all 14 builtins reject signed and floating-point arguments
void test_errors(int si, float f) {
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

#ifdef __has_include
#if __has_include(<stdbit.h>)
#include <stdbit.h>

_Static_assert(stdc_leading_zeros(0U) == 32, "");
_Static_assert(stdc_leading_zeros(1U) == 31, "");
_Static_assert(stdc_leading_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(stdc_trailing_zeros(0U) == 32, "");
_Static_assert(stdc_trailing_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(stdc_first_leading_zero(0U) == 1, "");
_Static_assert(stdc_first_leading_one(0U) == 0, "");
_Static_assert(stdc_first_trailing_zero(0U) == 1, "");
_Static_assert(stdc_first_trailing_one(0U) == 0, "");
_Static_assert(stdc_count_zeros(0U) == 32, "");
_Static_assert(stdc_count_ones(0xFFFFFFFFU) == 32, "");
_Static_assert(stdc_has_single_bit(4U) == 1, "");
_Static_assert(stdc_bit_width(7U) == 3, "");
_Static_assert(stdc_bit_floor(6U) == 4U, "");
_Static_assert(stdc_bit_ceil(6U) == 8U, "");

// Type-specific: each builtin truncates to the target type before operating.
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned int)0) == 32, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned long)0) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_leading_zeros((unsigned long long)0) == 64, "");

_Static_assert(__builtin_stdc_leading_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned int)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_leading_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned long long)0xFFFFFFFFFFFFFFFF) == 64, "");

_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned int)0) == 32, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned long)0) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned long long)0) == 64, "");

_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned int)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_trailing_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned long long)0xFFFFFFFFFFFFFFFF) == 64, "");

_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xFE) == 8, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned short)0xFFFE) == 16, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned int)0xFFFFFFFE) == 32, "");
_Static_assert(__builtin_stdc_first_leading_zero(~1UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned long long)0xFFFFFFFFFFFFFFFE) == 64, "");

_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0x01) == 8, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned short)0x0001) == 16, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned int)0x00000001) == 32, "");
_Static_assert(__builtin_stdc_first_leading_one(1UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned long long)0x0000000000000001) == 64, "");

_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0x7F) == 8, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned short)0x7FFF) == 16, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned int)0x7FFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_first_trailing_zero(~0UL >> 1) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned long long)0x7FFFFFFFFFFFFFFF) == 64, "");

_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x80) == 8, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned short)0x8000) == 16, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned int)0x80000000) == 32, "");
_Static_assert(__builtin_stdc_first_trailing_one(1UL << (ULONG_WIDTH - 1)) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned long long)0x8000000000000000) == 64, "");

_Static_assert(__builtin_stdc_count_zeros((unsigned char)0) == 8, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned short)0) == 16, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned int)0) == 32, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned long)0) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned long long)0) == 64, "");

_Static_assert(__builtin_stdc_count_ones((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_count_ones((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_count_ones((unsigned int)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_count_ones(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_count_ones((unsigned long long)0xFFFFFFFFFFFFFFFF) == 64, "");

_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0x80) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned short)0x8000) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned int)0x80000000) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit(1UL << (ULONG_WIDTH - 1)) == 1, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned long long)0x8000000000000000) == 1, "");

_Static_assert(__builtin_stdc_bit_width((unsigned char)0xFF) == 8, "");
_Static_assert(__builtin_stdc_bit_width((unsigned short)0xFFFF) == 16, "");
_Static_assert(__builtin_stdc_bit_width((unsigned int)0xFFFFFFFF) == 32, "");
_Static_assert(__builtin_stdc_bit_width(~0UL) == ULONG_WIDTH, "");
_Static_assert(__builtin_stdc_bit_width((unsigned long long)0xFFFFFFFFFFFFFFFF) == 64, "");

_Static_assert(__builtin_stdc_bit_floor((unsigned char)0x80) == 0x80, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned short)0x8000) == 0x8000, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned int)0x80000000) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_floor(~0UL) == (1UL << (ULONG_WIDTH - 1)), "");
_Static_assert(__builtin_stdc_bit_floor((unsigned long long)0x8000000000000000) == 0x8000000000000000ULL, "");

_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0x7F) == 0x80, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned short)0x7FFF) == 0x8000, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned int)0x7FFFFFFF) == 0x80000000U, "");
_Static_assert(__builtin_stdc_bit_ceil(1UL << (ULONG_WIDTH - 1)) == (1UL << (ULONG_WIDTH - 1)), "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned long long)0x7FFFFFFFFFFFFFFF) == 0x8000000000000000ULL, "");

// Truncation: passing a large ull verifies only the low N bits are used.
_Static_assert(__builtin_stdc_leading_zeros((unsigned char)0x8000000000000000ULL) == 8, "");
_Static_assert(__builtin_stdc_leading_ones((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 8, "");
_Static_assert(__builtin_stdc_trailing_zeros((unsigned char)0x8000000000000000ULL) == 8, "");
_Static_assert(__builtin_stdc_trailing_ones((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 8, "");
_Static_assert(__builtin_stdc_first_leading_zero((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 0, "");
_Static_assert(__builtin_stdc_first_leading_one((unsigned char)0xFF00000000000001ULL) == 8, "");
_Static_assert(__builtin_stdc_first_trailing_zero((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 0, "");
_Static_assert(__builtin_stdc_first_trailing_one((unsigned char)0x8000000000000080ULL) == 8, "");
_Static_assert(__builtin_stdc_count_zeros((unsigned char)0x8000000000000000ULL) == 8, "");
_Static_assert(__builtin_stdc_count_ones((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 8, "");
_Static_assert(__builtin_stdc_has_single_bit((unsigned char)0x8000000080008080ULL) == 1, "");
_Static_assert(__builtin_stdc_bit_width((unsigned char)0xFFFFFFFFFFFFFFFFULL) == 8, "");
_Static_assert(__builtin_stdc_bit_floor((unsigned char)0x8000000080008080ULL) == 0x80, "");
_Static_assert(__builtin_stdc_bit_ceil((unsigned char)0x800000008000807FULL) == 0x80, "");
#endif
#endif

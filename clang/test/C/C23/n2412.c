// RUN: %clang_cc1 -verify -std=c23 -ffreestanding %s

/* WG14 N2412: Clang 14
 * Two's complement sign representation
 */
// expected-no-diagnostics

#include <limits.h>

// GH117348 -- BOOL_WIDTH was accidentally expanding to the number of bits in
// the object representation (8) rather than the number of bits in the value
// representation (1).
static_assert(BOOL_WIDTH == 1);

// Validate the other macro requirements.
static_assert(CHAR_WIDTH == SCHAR_WIDTH);
static_assert(CHAR_WIDTH == UCHAR_WIDTH);
static_assert(CHAR_WIDTH == CHAR_BIT);

static_assert(USHRT_WIDTH >= 16);
static_assert(UINT_WIDTH >= 16);
static_assert(ULONG_WIDTH >= 32);
static_assert(ULLONG_WIDTH >= 64);
static_assert(BITINT_MAXWIDTH >= ULLONG_WIDTH);

static_assert(MB_LEN_MAX >= 1);


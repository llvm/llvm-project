// RUN: %clang_cc1 -verify -std=c2y -ffreestanding %s
// RUN: %clang_cc1 -verify -std=c23 -ffreestanding %s

/* WG14 N3496: Clang 20
 * Clarify the specification of the width macros
 *
 * C23 N2412 mandated a two's complement sign representation for integers, and
 * added *_WIDTH macros for all of the various integer types. There was
 * confusion as to whether BOOL_WIDTH specified the minimum number of bits or
 * an exact number of bits. N3496 clarified that it's an exact number of bits.
 */
// expected-no-diagnostics

#include <limits.h>

static_assert(BOOL_WIDTH == 1);


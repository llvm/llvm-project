// RUN: %clang_cc1 -triple x86_64 -ffreestanding -verify -std=c2x %s

/* WG14 N3035: yes
 * _BitInt Fixes
 */

#include <stdint.h>

/* intmax_t and uintmax_t don't need to be able to represent all of the values
 * of a bit-precise integer type. We test this by using a bit-precise integer
 * suffix on some huge values used within the preprocessor.
 */
#if 0x1'FFFF'FFFF'FFFF'FFFFwb == 0 /* expected-error {{integer literal is too large to be represented in any integer type}} */
#endif

/* Yet we can use that value as an initializer... */
_BitInt(66) Val = 0x1'FFFF'FFFF'FFFF'FFFFwb;

/* ...so long as the type is wide enough. */
intmax_t WrongVal = 0x1'FFFF'FFFF'FFFF'FFFFwb; /* expected-warning-re {{implicit conversion from '_BitInt(66)' to 'intmax_t' (aka '{{.*}}') changes value from 36893488147419103231 to -1}} */

/* None of the types in stdint.h may be defined in terms of a bit-precise
 * integer type. This macro presumes that if the type is not one of the builtin
 * scalar integer types, the type must be a bit-precise type. We're using this
 * because C does not have a particularly straightforward way to use _Generic
 * with arbitrary bit-precise integer types.
 */
#define IS_NOT_BIT_PRECISE(TYPE) _Generic((TYPE){ 0 },                          \
                                   short : 1, int : 1, long : 1, long long : 1, \
                                   unsigned short : 1, unsigned int : 1,        \
                                   unsigned long : 1, unsigned long long : 1,   \
                                   char : 1, signed char : 1, unsigned char : 1,\
                                   default : 0)
static_assert(IS_NOT_BIT_PRECISE(int8_t));
static_assert(IS_NOT_BIT_PRECISE(uint8_t));
static_assert(IS_NOT_BIT_PRECISE(int16_t));
static_assert(IS_NOT_BIT_PRECISE(uint16_t));
static_assert(IS_NOT_BIT_PRECISE(int32_t));
static_assert(IS_NOT_BIT_PRECISE(uint32_t));
static_assert(IS_NOT_BIT_PRECISE(int64_t));
static_assert(IS_NOT_BIT_PRECISE(uint64_t));
static_assert(IS_NOT_BIT_PRECISE(intmax_t));
static_assert(IS_NOT_BIT_PRECISE(uintmax_t));
static_assert(IS_NOT_BIT_PRECISE(intptr_t));
static_assert(IS_NOT_BIT_PRECISE(uintptr_t));

/* FIXME: N3035 also added wording disallowing using a bit-precise integer type
 * as the compatible type for an enumerated type. However, we don't have a
 * direct way to test that, so we're claiming conformance without test
 * coverage.
 */

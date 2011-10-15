/* ===-- divsi3.c - Implement __divsi3 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divsi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */
#include "abi.h"

#include "int_lib.h"

su_int COMPILER_RT_ABI __udivsi3(su_int n, su_int d);

/* Returns: a / b */

ARM_EABI_FNALIAS(idiv, divsi3);

COMPILER_RT_ABI si_int
__divsi3(si_int a, si_int b)
{
    const int bits_in_word_m1 = (int)(sizeof(si_int) * CHAR_BIT) - 1;
    si_int s_a = a >> bits_in_word_m1;           /* s_a = a < 0 ? -1 : 0 */
    si_int s_b = b >> bits_in_word_m1;           /* s_b = b < 0 ? -1 : 0 */
    a = (a ^ s_a) - s_a;                         /* negate if s_a == -1 */
    b = (b ^ s_b) - s_b;                         /* negate if s_b == -1 */
    s_a ^= s_b;                                  /* sign of quotient */
    return (__udivsi3(a, b) ^ s_a) - s_a;        /* negate if s_a == -1 */
}

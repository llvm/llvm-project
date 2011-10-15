/* ===-- fixunsdfsi.c - Implement __fixunsdfsi -----------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __fixunsdfsi for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */
#include "abi.h"

#include "int_lib.h"

/* Returns: convert a to a unsigned int, rounding toward zero.
 *          Negative values all become zero.
 */

/* Assumption: double is a IEEE 64 bit floating point type 
 *             su_int is a 32 bit integral type
 *             value in double is representable in su_int or is negative 
 *                 (no range checking performed)
 */

/* seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm */

ARM_EABI_FNALIAS(d2uiz, fixunsdfsi);

COMPILER_RT_ABI su_int
__fixunsdfsi(double a)
{
    double_bits fb;
    fb.f = a;
    int e = ((fb.u.s.high & 0x7FF00000) >> 20) - 1023;
    if (e < 0 || (fb.u.s.high & 0x80000000))
        return 0;
    return (
                0x80000000u                      |
                ((fb.u.s.high & 0x000FFFFF) << 11) |
                (fb.u.s.low >> 21)
           ) >> (31 - e);
}

/* ===-- fixdfti.c - Implement __fixdfti -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __fixdfti for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"

/* Returns: convert a to a signed long long, rounding toward zero. */

/* Assumption: double is a IEEE 64 bit floating point type 
 *             su_int is a 32 bit integral type
 *             value in double is representable in ti_int (no range checking performed)
 */

/* seee eeee eeee mmmm mmmm mmmm mmmm mmmm | mmmm mmmm mmmm mmmm mmmm mmmm mmmm mmmm */

ti_int
__fixdfti(double a)
{
    double_bits fb;
    fb.f = a;
    int e = ((fb.u.s.high & 0x7FF00000) >> 20) - 1023;
    if (e < 0)
        return 0;
    ti_int s = (si_int)(fb.u.s.high & 0x80000000) >> 31;
    ti_int r = 0x0010000000000000uLL | (0x000FFFFFFFFFFFFFuLL & fb.u.all);
    if (e > 52)
        r <<= (e - 52);
    else
        r >>= (52 - e);
    return (r ^ s) - s;
}

#endif

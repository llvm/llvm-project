/* ===-- ctzdi2.c - Implement __ctzdi2 -------------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __ctzdi2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

/* Returns: the number of trailing 0-bits  */

/* Precondition: a != 0 */

COMPILER_RT_ABI si_int
__ctzdi2(di_int a)
{
    dwords x;
    x.all = a;
    const si_int f = -(x.s.low == 0);
    return __builtin_ctz((x.s.high & f) | (x.s.low & ~f)) +
              (f & ((si_int)(sizeof(si_int) * CHAR_BIT)));
}

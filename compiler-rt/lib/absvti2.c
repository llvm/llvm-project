/* ===-- absvti2.c - Implement __absvdi2 -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __absvti2 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"
#include <stdlib.h>

/* Returns: absolute value */

/* Effects: aborts if abs(x) < 0 */

ti_int
__absvti2(ti_int a)
{
    const int N = (int)(sizeof(ti_int) * CHAR_BIT);
    if (a == ((ti_int)1 << (N-1)))
        compilerrt_abort();
    const ti_int s = a >> (N - 1);
    return (a ^ s) - s;
}

#endif

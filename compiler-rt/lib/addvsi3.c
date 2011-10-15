/* ===-- addvsi3.c - Implement __addvsi3 -----------------------------------===
 *
 *                    The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __addvsi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */
#include "abi.h"

#include "int_lib.h"
#include <stdlib.h>

/* Returns: a + b */

/* Effects: aborts if a + b overflows */

COMPILER_RT_ABI si_int
__addvsi3(si_int a, si_int b)
{
    si_int s = a + b;
    if (b >= 0)
    {
        if (s < a)
            compilerrt_abort();
    }
    else
    {
        if (s >= a)
            compilerrt_abort();
    }
    return s;
}

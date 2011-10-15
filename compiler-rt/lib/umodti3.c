/* ===-- umodti3.c - Implement __umodti3 -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __umodti3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#if __x86_64

#include "int_lib.h"

tu_int __udivmodti4(tu_int a, tu_int b, tu_int* rem);

/* Returns: a % b */

tu_int
__umodti3(tu_int a, tu_int b)
{
    tu_int r;
    __udivmodti4(a, b, &r);
    return r;
}

#endif

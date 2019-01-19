/* ===-- umodsi3.c - Implement __umodsi3 -----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __umodsi3 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

/* Returns: a % b */

COMPILER_RT_ABI su_int
__umodsi3(su_int a, su_int b)
{
    return a - __udivsi3(a, b) * b;
}

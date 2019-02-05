/*===-- divmoddi4.c - Implement __divmoddi4 --------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __divmoddi4 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

/* Returns: a / b, *rem = a % b  */

COMPILER_RT_ABI di_int
__divmoddi4(di_int a, di_int b, di_int* rem)
{
  di_int d = __divdi3(a,b);
  *rem = a - (d*b);
  return d;
}

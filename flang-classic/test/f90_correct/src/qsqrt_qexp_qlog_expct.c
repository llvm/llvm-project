/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* C test function for qsqrt_qexp_qlog.f90 */

#include <stdio.h>
#include <math.h>

#define SQRTL_BEGIN 0
#define EXPL_BEGIN 5
#define LOGL_BEGIN 10
#define SQRTL_END 5
#define EXPL_END 10
#define LOGL_END 15
void
get_expected_q(long double src[], long double expct[])
{
  int i;

  for (i = SQRTL_BEGIN; i < SQRTL_END; i++) {
    expct[i] = sqrtl(src[i]);
  }

  for (i = EXPL_BEGIN; i < EXPL_END; i++) {
    expct[i] = expl(src[i]);
  }

  for (i = LOGL_BEGIN; i < LOGL_END; i++) {
    expct[i] = logl(src[i]);
  }
}

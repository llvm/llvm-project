/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !defined(PARAMID) && !defined(_WIN64)
#include <sys/types.h>
#include <fcntl.h>
#include <time.h>
#endif
#include <errno.h>
#include "global.h"

__INT_T
__fort_time(void)
{
  __INT_T s;

#if defined(SUN4SOL2) || defined(SOL86) || defined(HP) || defined(TARGET_OSX)
  s = (int)time((time_t *)0);
#else
  s = time(NULL);
#endif
  if (!LOCAL_MODE) {
    __fort_rbcst(GET_DIST_IOPROC, &s, 1, 1, __CINT);
  }
  return s;
}

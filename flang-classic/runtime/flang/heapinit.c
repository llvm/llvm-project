/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <signal.h>
#include "stdioInterf.h"
#include "fioMacros.h"

/* handler for bus error */

static void sighand(int s)
{
  int lcpu;

  lcpu = __fort_myprocnum();
  fprintf(__io_stderr(), "not enough temporary disk space\n");
  sleep(1); /* wait for message to clear */
  __fort_abort(NULL); /* abort */
}

/* init global heap (maybe) */

void __fort_heapinit(char *beg, char *end, int val)
{
  void (*save)();
  int *pi;

#if !defined(_WIN64)
  save = signal(SIGBUS, sighand);
#endif
  pi = (int *)beg;
  while (pi < (int *)end) {
    *pi++ = val;
  }
#if !defined(_WIN64)
  signal(SIGBUS, save);
#endif
}

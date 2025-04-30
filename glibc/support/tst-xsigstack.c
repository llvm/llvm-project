/* Test of sigaltstack wrappers.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <support/xsignal.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <support/check.h>

#include <stdint.h>
#include <stdio.h>

static volatile uintptr_t handler_stackaddr;

static void
handler (int unused)
{
  int var;
  handler_stackaddr = (uintptr_t) &var;
}

int
do_test (void)
{
  void *sstk = xalloc_sigstack (0);

  unsigned char *sp;
  size_t size;
  xget_sigstack_location (sstk, &sp, &size);
  printf ("signal stack installed: sp=%p size=%zu\n", sp, size);

  struct sigaction sa;
  sa.sa_handler = handler;
  sa.sa_flags   = SA_RESTART | SA_ONSTACK;
  sigfillset (&sa.sa_mask);
  if (sigaction (SIGUSR1, &sa, 0))
    FAIL_RET ("sigaction (SIGUSR1, handler): %m\n");

  raise (SIGUSR1);

  uintptr_t haddr = handler_stackaddr;
  printf ("address of handler local variable: %p\n", (void *)haddr);
  TEST_VERIFY ((uintptr_t)sp < haddr);
  TEST_VERIFY (haddr < (uintptr_t)sp + size);

  xfree_sigstack (sstk);
  return 0;
}

#include <support/test-driver.c>

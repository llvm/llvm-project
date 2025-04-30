/* Test of signal delivery on an alternate stack with MINSIGSTKSZ size.
   Copyright (C) 2020 Free Software Foundation, Inc.
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

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <support/check.h>
#include <support/support.h>

static volatile sig_atomic_t handler_run;

static void
handler (int signo)
{
  /* Clear a bit of on-stack memory.  */
  volatile char buffer[256];
  for (size_t i = 0; i < sizeof (buffer); ++i)
    buffer[i] = 0;
  handler_run = 1;
}

int
do_test (void)
{
  size_t stack_buffer_size = 64 * 1024 * 1024;
  void *stack_buffer = xmalloc (stack_buffer_size);
  void *stack_end = stack_buffer + stack_buffer_size;
  memset (stack_buffer, 0xCC, stack_buffer_size);

  void *stack_bottom = stack_buffer + (stack_buffer_size + MINSIGSTKSZ) / 2;
  void *stack_top = stack_bottom + MINSIGSTKSZ;
  stack_t stack =
    {
      .ss_sp = stack_bottom,
      .ss_size = MINSIGSTKSZ,
    };
  if (sigaltstack (&stack, NULL) < 0)
    FAIL_RET ("sigaltstack: %m\n");

  struct sigaction act =
    {
      .sa_handler = handler,
      .sa_flags = SA_ONSTACK,
    };
  if (sigaction (SIGUSR1, &act, NULL) < 0)
    FAIL_RET ("sigaction: %m\n");

  if (kill (getpid (), SIGUSR1) < 0)
    FAIL_RET ("kill: %m\n");

  if (handler_run != 1)
    FAIL_RET ("handler did not run\n");

  for (void *p = stack_buffer; p < stack_bottom; ++p)
    if (*(unsigned char *) p != 0xCC)
      FAIL_RET ("changed byte %ld bytes below configured stack\n",
		(long) (stack_bottom - p));
  for (void *p = stack_top; p < stack_end; ++p)
    if (*(unsigned char *) p != 0xCC)
      FAIL_RET ("changed byte %ld bytes above configured stack\n",
		(long) (p - stack_top));

  free (stack_buffer);

  return 0;
}

#include <support/test-driver.c>

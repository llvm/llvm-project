/* Test that the GET_PC macro is consistent with the unwinder.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This test searches for the value of the GET_PC macro in the
   addresses obtained from the backtrace function.  */

#include <array_length.h>
#include <execinfo.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <sigcontextinfo.h>

static bool handler_called;

static void
handler (int signal, siginfo_t *info, void *ctx)
{
  TEST_COMPARE (signal, SIGUSR1);

  uintptr_t pc = sigcontext_get_pc (ctx);
  printf ("info: address in signal handler: 0x%" PRIxPTR "\n", pc);

  void *callstack[10];
  int callstack_count = backtrace (callstack, array_length (callstack));
  TEST_VERIFY_EXIT (callstack_count > 0);
  TEST_VERIFY_EXIT (callstack_count <= array_length (callstack));
  bool found = false;
  for (int i = 0; i < callstack_count; ++i)
    {
      const char *marker;
      if ((uintptr_t) callstack[i] == pc)
        {
          found = true;
          marker = " *";
        }
      else
        marker = "";
      printf ("info: call stack entry %d: 0x%" PRIxPTR "%s\n",
              i, (uintptr_t) callstack[i], marker);
    }
  TEST_VERIFY (found);
  handler_called = true;
}

static int
do_test (void)
{
  struct sigaction sa =
    {
     .sa_sigaction = &handler,
     .sa_flags = SA_SIGINFO
    };
  xsigaction (SIGUSR1, &sa, NULL);
  raise (SIGUSR1);
  TEST_VERIFY (handler_called);
  return 0;
}

#include <support/test-driver.c>

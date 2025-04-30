/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <ucontext.h>
#include <assert.h>
#include <unwind.h>
#include <dlfcn.h>
#include <gnu/lib-names.h>

ucontext_t ucp;
char st1[16384];
__thread int thr;

int somevar = -76;
long othervar = -78L;

struct trace_arg
{
  int cnt, size;
};

static _Unwind_Reason_Code
backtrace_helper (struct _Unwind_Context *ctx, void *a)
{
  struct trace_arg *arg = a;
  if (++arg->cnt == arg->size)
    return _URC_END_OF_STACK;
  return _URC_NO_REASON;
}

void
cf (int i)
{
  struct trace_arg arg = { .size = 100, .cnt = -1 };
  void *handle;
  _Unwind_Reason_Code (*unwind_backtrace) (_Unwind_Trace_Fn, void *);

  if (i != othervar || thr != 94)
    {
      printf ("i %d thr %d\n", i, thr);
      exit (1);
    }

  /* Test if callback function of _Unwind_Backtrace is not called infinitely
     times. See Bug 18508 or gcc bug "Bug 66303 - runtime.Caller() returns
     infinitely deep stack frames on s390x.".
     The go runtime calls backtrace_full() in
     <gcc-src>/libbacktrace/backtrace.c, which uses _Unwind_Backtrace().  */
  handle = dlopen (LIBGCC_S_SO, RTLD_LAZY);
  if (handle != NULL)
    {
      unwind_backtrace = dlsym (handle, "_Unwind_Backtrace");
      if (unwind_backtrace != NULL)
	{
	  unwind_backtrace (backtrace_helper, &arg);
	  assert (arg.cnt != -1 && arg.cnt < 100);
	}
      dlclose (handle);
    }

  /* Since uc_link below has been set to NULL, setcontext is supposed to
     terminate the process normally after this function returns.  */
}

int
do_test (void)
{
  if (getcontext (&ucp) != 0)
    {
      if (errno == ENOSYS)
	{
	  puts ("context handling not supported");
	  return 0;
	}

      puts ("getcontext failed");
      return 1;
    }
  thr = 94;
  ucp.uc_link = NULL;
  ucp.uc_stack.ss_sp = st1;
  ucp.uc_stack.ss_size = sizeof st1;
  makecontext (&ucp, (void (*) (void)) cf, 1, somevar - 2);
  if (setcontext (&ucp) != 0)
    {
      puts ("setcontext failed");
      return 1;
    }
  return 2;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

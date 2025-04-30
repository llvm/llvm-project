/* Check stack alignment provided by makecontext.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <ucontext.h>

/* Used for error reporting.  */
static const char *context;

/* Check that ADDRESS is aligned to ALIGNMENT bytes, behind a compiler
   barrier.  */
__attribute__ ((noinline, noclone, weak))
void
check_align (void *address, size_t alignment)
{
  uintptr_t uaddress = (uintptr_t) address;
  if ((uaddress % alignment) != 0)
    {
      support_record_failure ();
      printf ("error: %s: object at address %p is not aligned to %zu bytes\n",
              context, address, alignment);
    }
}

/* Various alignment checking functions.  */

__attribute__ ((noinline, noclone, weak))
void
check_align_int (void)
{
  int a;
  check_align (&a, __alignof__ (a));
}

__attribute__ ((noinline, noclone, weak))
void
check_align_long (void)
{
  long a;
  check_align (&a, __alignof__ (a));
}

__attribute__ ((noinline, noclone, weak))
void
check_align_long_long (void)
{
  long long a;
  check_align (&a, __alignof__ (a));
}

__attribute__ ((noinline, noclone, weak))
void
check_align_double (void)
{
  double a;
  check_align (&a, __alignof__ (a));
}

__attribute__ ((noinline, noclone, weak))
void
check_align_4 (void)
{
  int a __attribute__ ((aligned (4)));
  check_align (&a, 4);
}

__attribute__ ((noinline, noclone, weak))
void
check_align_8 (void)
{
  double a __attribute__ ((aligned (8)));
  check_align (&a, 8);
}

__attribute__ ((noinline, noclone, weak))
void
check_align_16 (void)
{
  struct aligned
  {
    double x0  __attribute__ ((aligned (16)));
    double x1;
  } a;
  check_align (&a, 16);
}

__attribute__ ((noinline, noclone, weak))
void
check_align_32 (void)
{
  struct aligned
  {
    double x0  __attribute__ ((aligned (32)));
    double x1;
    double x2;
    double x3;
  } a;
  check_align (&a, 32);
}

/* Call all the alignment checking functions.  */
__attribute__ ((noinline, noclone, weak))
void
check_alignments (void)
{
  check_align_int ();
  check_align_long ();
  check_align_long_long ();
  check_align_double ();
  check_align_4 ();
  check_align_8 ();
  check_align_16 ();
  check_align_32 ();
}

/* Callback functions for makecontext and their invokers (to be used
   with support_isolate_in_subprocess).  */

static ucontext_t ucp;

static void
callback_0 (void)
{
  context = "callback_0";
  check_alignments ();
  context = "after return from callback_0";
}

static void
invoke_callback_0 (void *closure)
{
  makecontext (&ucp, (void *) callback_0, 0);
  if (setcontext (&ucp) != 0)
    FAIL_EXIT1 ("setcontext");
  FAIL_EXIT1 ("setcontext returned");
}

static void
callback_1 (int arg1)
{
  context = "callback_1";
  check_alignments ();
  TEST_COMPARE (arg1, 101);
  context = "after return from callback_1";
}

static void
invoke_callback_1 (void *closure)
{
  makecontext (&ucp, (void *) callback_1, 1, 101);
  if (setcontext (&ucp) != 0)
    FAIL_EXIT1 ("setcontext");
  FAIL_EXIT1 ("setcontext returned");
}

static void
callback_2 (int arg1, int arg2)
{
  context = "callback_2";
  check_alignments ();
  TEST_COMPARE (arg1, 201);
  TEST_COMPARE (arg2, 202);
  context = "after return from callback_2";
}

static void
invoke_callback_2 (void *closure)
{
  makecontext (&ucp, (void *) callback_2, 2, 201, 202);
  if (setcontext (&ucp) != 0)
    FAIL_EXIT1 ("setcontext");
  FAIL_EXIT1 ("setcontext returned");
}

static void
callback_3 (int arg1, int arg2, int arg3)
{
  context = "callback_3";
  check_alignments ();
  TEST_COMPARE (arg1, 301);
  TEST_COMPARE (arg2, 302);
  TEST_COMPARE (arg3, 303);
  context = "after return from callback_3";
}

static void
invoke_callback_3 (void *closure)
{
  makecontext (&ucp, (void *) callback_3, 3, 301, 302, 303);
  if (setcontext (&ucp) != 0)
    FAIL_EXIT1 ("setcontext");
  FAIL_EXIT1 ("setcontext returned");
}

static int
do_test (void)
{
  context = "direct call";
  check_alignments ();

  atexit (check_alignments);

  if (getcontext (&ucp) != 0)
    FAIL_UNSUPPORTED ("getcontext");

  ucp.uc_link = NULL;
  ucp.uc_stack.ss_size = 512 * 1024;
  ucp.uc_stack.ss_sp = xmmap (NULL, ucp.uc_stack.ss_size,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1);

  support_isolate_in_subprocess (invoke_callback_0, NULL);
  support_isolate_in_subprocess (invoke_callback_1, NULL);
  support_isolate_in_subprocess (invoke_callback_2, NULL);
  support_isolate_in_subprocess (invoke_callback_3, NULL);

  return 0;
}

#include <support/test-driver.c>

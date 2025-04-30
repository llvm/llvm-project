/* Check getcontext and setcontext on the context from makecontext
   with shadow stack.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <x86intrin.h>

static ucontext_t ctx[5];
static atomic_int done;

static void
__attribute__((noinline, noclone))
f2 (void)
{
  printf ("start f2\n");
  done++;
  if (setcontext (&ctx[2]) != 0)
    {
      printf ("%s: setcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
}

static void
f1 (void)
{
  printf ("start f1\n");
  if (getcontext (&ctx[2]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  if (done)
    exit (EXIT_SUCCESS);
  f2 ();
}

static int
do_test (void)
{
  char st1[32768];
  puts ("making contexts");
  if (getcontext (&ctx[0]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  if (getcontext (&ctx[1]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }

  ctx[3].uc_stack.ss_sp = st1;
  ctx[3].uc_stack.ss_size = sizeof st1;
  ctx[3].uc_link = &ctx[0];
  makecontext (&ctx[3], (void (*) (void)) f1, 0);

  ctx[1].uc_stack.ss_sp = st1;
  ctx[1].uc_stack.ss_size = sizeof st1;
  ctx[1].uc_link = &ctx[0];
  makecontext (&ctx[1], (void (*) (void)) f1, 0);

  ctx[4].uc_stack.ss_sp = st1;
  ctx[4].uc_stack.ss_size = sizeof st1;
  ctx[4].uc_link = &ctx[0];
  makecontext (&ctx[4], (void (*) (void)) f1, 0);

  /* NB: When shadow stack is enabled, makecontext calls arch_prctl
     with ARCH_CET_ALLOC_SHSTK to allocate a new shadow stack which
     can be unmapped.  The base address and size of the new shadow
     stack are returned in __ssp[1] and __ssp[2].  makecontext is
     called for CTX1, CTX3 and CTX4.  But only CTX1 is used.  New
     shadow stacks are allocated in the order of CTX3, CTX1, CTX4.
     It is very likely that CTX1's shadow stack is placed between
     CTX3 and CTX4.  We munmap CTX3's and CTX4's shadow stacks to
     create gaps above and below CTX1's shadow stack.  We check
     that setcontext CTX1 works correctly in this case.  */
  if (_get_ssp () != 0)
    {
      if (ctx[3].__ssp[1] != 0
	  && munmap ((void *) (uintptr_t) ctx[3].__ssp[1],
		     (size_t) ctx[3].__ssp[2]) != 0)
	{
	  printf ("%s: munmap: %m\n", __FUNCTION__);
	  exit (EXIT_FAILURE);
	}

      if (ctx[4].__ssp[1] != 0
	  && munmap ((void *) (uintptr_t) ctx[4].__ssp[1],
		     (size_t) ctx[4].__ssp[2]) != 0)
	{
	  printf ("%s: munmap: %m\n", __FUNCTION__);
	  exit (EXIT_FAILURE);
	}
    }

  if (setcontext (&ctx[1]) != 0)
    {
      printf ("%s: setcontext: %m\n", __FUNCTION__);
      exit (EXIT_FAILURE);
    }
  exit (EXIT_FAILURE);
}

#include <support/test-driver.c>

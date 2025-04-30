/* Test if an executable can read from the TLS from an STT_GNU_IFUNC resolver.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <libc-symbols.h>
#include <tls-macros.h>

__thread int bar;
static int *bar_ptr = NULL;

static uint32_t resolver_platform = 0;

int foo (void);

int tcb_test (void);

/* Offsets copied from tcb-offsets.h.  */
#ifdef __powerpc64__
# define __TPREG     "r13"
# define __ATPLATOFF -28764
#else
# define __TPREG     "r2"
# define __ATPLATOFF -28724
#endif

uint32_t
get_platform (void)
{
  register unsigned long tp __asm__ (__TPREG);
  uint32_t tmp;

  __asm__  ("lwz %0,%1(%2)\n"
	    : "=r" (tmp)
	    : "i" (__ATPLATOFF), "b" (tp));

  return tmp;
}

void
init_foo (void)
{
  bar_ptr = TLS_GD (bar);
}

int
my_foo (void)
{
  printf ("&bar = %p and bar_ptr = %p.\n", &bar, bar_ptr);
  return bar_ptr != NULL;
}

__ifunc (foo, foo, my_foo, void, init_foo);

void
init_tcb_test (void)
{
  resolver_platform = get_platform ();
}

int
my_tcb_test (void)
{
  printf ("resolver_platform = 0x%"PRIx32
	  " and current platform = 0x%"PRIx32".\n",
	  resolver_platform, get_platform ());
  return resolver_platform != 0;
}

__ifunc (tcb_test, tcb_test, my_tcb_test, void, init_tcb_test);

static int
do_test (void)
{
  int ret = 0;

  if (foo ())
    printf ("PASS: foo IFUNC resolver called once.\n");
  else
    {
      printf ("FAIL: foo IFUNC resolver not called once.\n");
      ret = 1;
    }

  if (&bar == bar_ptr)
    printf ("PASS: bar address read from IFUNC resolver is correct.\n");
  else
    {
      printf ("FAIL: bar address read from IFUNC resolver is incorrect.\n");
      ret = 1;
    }

  if (tcb_test ())
    printf ("PASS: tcb_test IFUNC resolver called once.\n");
  else
    {
      printf ("FAIL: tcb_test IFUNC resolver not called once.\n");
      ret = 1;
    }

  if (resolver_platform == get_platform ())
    printf ("PASS: platform read from IFUNC resolver is correct.\n");
  else
    {
      printf ("FAIL: platform read from IFUNC resolver is incorrect.\n");
      ret = 1;
    }

  return ret;
}

#include <support/test-driver.c>

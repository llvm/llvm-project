/* Test for large alignment in TLS blocks, BZ#18383.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

static __thread int tdata1 = 1;
static __thread int tdata2 __attribute__ ((aligned (0x10))) = 2;
static __thread int tdata3 __attribute__ ((aligned (0x1000))) = 4;
static __thread int tbss1;
static __thread int tbss2 __attribute__ ((aligned (0x10)));
static __thread int tbss3 __attribute__ ((aligned (0x1000)));

#ifndef NO_LIB
extern __thread int mod_tdata1;
extern __thread int mod_tdata2;
extern __thread int mod_tdata3;
extern __thread int mod_tbss1;
extern __thread int mod_tbss2;
extern __thread int mod_tbss3;
#endif

static int
test_one (const char *which, unsigned int alignment, int *var, int value)
{
  uintptr_t addr = (uintptr_t) var;
  unsigned int misalign = addr & (alignment - 1);

  printf ("%s TLS address %p %% %u = %u\n",
          which, (void *) var, alignment, misalign);

  int got = *var;
  if (got != value)
    {
      printf ("%s value %d should be %d\n", which, got, value);
      return 1;
    }

  return misalign != 0;
}

static int
do_test (void)
{
  int fail = 0;

  fail |= test_one ("tdata1", 4, &tdata1, 1);
  fail |= test_one ("tdata2", 0x10, &tdata2, 2);
  fail |= test_one ("tdata3", 0x1000, &tdata3, 4);

  fail |= test_one ("tbss1", 4, &tbss1, 0);
  fail |= test_one ("tbss2", 0x10, &tbss2, 0);
  fail |= test_one ("tbss3", 0x1000, &tbss3, 0);

#ifndef NO_LIB
  fail |= test_one ("mod_tdata1", 4, &mod_tdata1, 1);
  fail |= test_one ("mod_tdata2", 0x10, &mod_tdata2, 2);
  fail |= test_one ("mod_tdata3", 0x1000, &mod_tdata3, 4);

  fail |= test_one ("mod_tbss1", 4, &mod_tbss1, 0);
  fail |= test_one ("mod_tbss2", 0x10, &mod_tbss2, 0);
  fail |= test_one ("mod_tbss3", 0x1000, &mod_tbss3, 0);
#endif

  return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>

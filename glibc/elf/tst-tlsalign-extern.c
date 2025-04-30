/* Test for large alignment in TLS blocks (extern case), BZ#18383.
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

/* This is the same as tst-tlsalign-static.c, except that it uses
   TLS variables that are defined in a separate translation unit
   (ts-tlsalign-vars.c).  It turned out that the cause of BZ#18383
   on ARM was actually an ARM assembler bug triggered by the ways of
   using .tdata/.tbss sections and relocs referring to them that GCC
   chooses when the variables are defined in the same translation
   unit that contains the references.  */

extern __thread int tdata1;
extern __thread int tdata2;
extern __thread int tdata3;
extern __thread int tbss1;
extern __thread int tbss2;
extern __thread int tbss3;

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

  return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>

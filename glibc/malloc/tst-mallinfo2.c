/* Smoke test for mallinfo2
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* Test that mallinfo2 is properly exported and basically works.  */

#include <array_length.h>
#include <malloc.h>
#include <stdlib.h>
#include <support/check.h>

/* This is not specifically needed for the test, but (1) does
   something to the data so gcc doesn't optimize it away, and (2) may
   help when developing future tests.  */
static void
print_mi (const char *msg, struct mallinfo2 *m)
{
  printf("\n%s...\n", msg);
#define P(f) printf("%s: %zu\n", #f, m->f)
  P(arena);
  P(ordblks);
  P(smblks);
  P(hblks);
  P(hblkhd);
  P(usmblks);
  P(fsmblks);
  P(uordblks);
  P(fordblks);
  P(keepcost);
}

/* We do this to force the call to malloc to not be optimized
   away.  */
volatile void *ptr;

static int
do_test (void)
{
  struct mallinfo2 mi1, mi2;
  int i;
  size_t total = 0;

  /* This is the key difference between mallinfo() and mallinfo2().
     It may be a false positive if int and size_t are the same
     size.  */
  TEST_COMPARE (sizeof (mi1.arena), sizeof (size_t));

  mi1 = mallinfo2 ();
  print_mi ("before", &mi1);

  /* Allocations that are meaningful-sized but not so large as to be
     mmapped, so that they're all accounted for in the field we test
     below.  */
  for (i = 1; i < 20; ++i)
    {
      ptr = malloc (160 * i);
      total += 160 * i;
    }

  mi2 = mallinfo2 ();
  print_mi ("after", &mi2);

  /* Check at least something changed.  */
  TEST_VERIFY (mi2.uordblks >= mi1.uordblks + total);

  return 0;
}

#include <support/test-driver.c>

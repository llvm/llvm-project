/* Test to verify that passing a pointer to an uninitialized object
   to pthread_setspecific doesn't trigger bogus uninitialized warnings.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <stdlib.h>

/* Turn uninitialized warnings into errors to detect the problem.
   See BZ #27714.  */

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wmaybe-uninitialized"
#pragma GCC diagnostic error "-Wuninitialized"

int do_test (void)
{
  void *p = malloc (1);   /* Deliberately uninitialized.  */
  pthread_setspecific (pthread_self (), p);

  void *q = pthread_getspecific (pthread_self ());

  return p == q;
}

#pragma GCC diagnostic pop

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

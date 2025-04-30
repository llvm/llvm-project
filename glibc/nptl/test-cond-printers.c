/* Helper program for testing the pthread_cond_t pretty printer.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* Keep the calls to the pthread_* functions on separate lines to make it easy
   to advance through the program using the gdb 'next' command.  */

#include <time.h>
#include <pthread.h>

#define PASS 0
#define FAIL 1

static int test_status_destroyed (pthread_cond_t *condvar);

int
main (void)
{
  pthread_cond_t condvar;
  pthread_condattr_t attr;
  int result = FAIL;

  if (pthread_condattr_init (&attr) == 0
      && test_status_destroyed (&condvar) == PASS)
    result = PASS;
  /* Else, one of the pthread_cond* functions failed.  */

  return result;
}

/* Initializes CONDVAR, then destroys it.  */
static int
test_status_destroyed (pthread_cond_t *condvar)
{
  int result = FAIL;

  if (pthread_cond_init (condvar, NULL) == 0
      && pthread_cond_destroy (condvar) == 0)
    result = PASS; /* Test status (destroyed).  */

  return result;
}

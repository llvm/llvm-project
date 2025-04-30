/* Helper program for testing the pthread_cond_t and pthread_condattr_t
   pretty printers.

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

static int condvar_reinit (pthread_cond_t *condvar,
			   const pthread_condattr_t *attr);
static int test_setclock (pthread_cond_t *condvar, pthread_condattr_t *attr);
static int test_setpshared (pthread_cond_t *condvar, pthread_condattr_t *attr);

/* Need these so we don't have lines longer than 79 chars.  */
#define SET_SHARED(attr, shared) pthread_condattr_setpshared (attr, shared)

int
main (void)
{
  pthread_cond_t condvar;
  pthread_condattr_t attr;
  int result = FAIL;

  if (pthread_condattr_init (&attr) == 0
      && pthread_cond_init (&condvar, NULL) == 0
      && test_setclock (&condvar, &attr) == PASS
      && test_setpshared (&condvar, &attr) == PASS)
    result = PASS;
  /* Else, one of the pthread_cond* functions failed.  */

  return result;
}

/* Destroys CONDVAR and re-initializes it using ATTR.  */
static int
condvar_reinit (pthread_cond_t *condvar, const pthread_condattr_t *attr)
{
  int result = FAIL;

  if (pthread_cond_destroy (condvar) == 0
      && pthread_cond_init (condvar, attr) == 0)
    result = PASS;

  return result;
}

/* Tests setting the clock ID attribute.  */
__attribute__ ((noinline))
static int
test_setclock (pthread_cond_t *condvar, pthread_condattr_t *attr)
{
  int result = FAIL;

  if (pthread_condattr_setclock (attr, CLOCK_REALTIME) == 0 /* Set clock.  */
      && condvar_reinit (condvar, attr) == PASS)
    result = PASS;

  return result;
}

/* Tests setting whether the condvar can be shared between processes.  */
static int
test_setpshared (pthread_cond_t *condvar, pthread_condattr_t *attr)
{
  int result = FAIL;

  if (SET_SHARED (attr, PTHREAD_PROCESS_SHARED) == 0 /* Set shared.  */
      && condvar_reinit (condvar, attr) == PASS
      && SET_SHARED (attr, PTHREAD_PROCESS_PRIVATE) == 0
      && condvar_reinit (condvar, attr) == PASS)
    result = PASS;

  return result;
}

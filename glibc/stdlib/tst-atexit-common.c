/* Helper file for tst-{atexit,at_quick_exit,cxa_atexit,on_exit}.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* http://pubs.opengroup.org/onlinepubs/000095399/functions/atexit.html
   requires that we support at least 32 atexit handlers.

   The number we actually support is limited by memory. Here we simply
   check that we support at least the minimum required.  */
#define MAX_ATEXIT 32

/* Arbitrary sequence matching current registrations.  */
const char expected[] = "00000000000000000000000003021121130211";

static char crumbs[sizeof (expected)];
static int next_slot = 0;

/* Helper: flush stdout and _exit.  */
static void
_exit_with_flush (int code)
{
  fflush (stdout);
  _exit (code);
}

static void
fn0 (void)
{
  crumbs[next_slot++] = '0';
}

static void
fn1 (void)
{
  crumbs[next_slot++] = '1';
}

static void
fn2 (void)
{
  crumbs[next_slot++] = '2';
  ATEXIT (fn1);
}

static void
fn3 (void)
{
  crumbs[next_slot++] = '3';
  ATEXIT (fn2);
  ATEXIT (fn0);
}

static void
fn_final (void)
{
  if (strcmp (crumbs, expected) == 0)
    _exit_with_flush (0);

  printf ("crumbs:   %s\n", crumbs);
  printf ("expected: %s\n", expected);
  _exit_with_flush (1);
}

static int
do_test (void)
{
  int slots_remaining = MAX_ATEXIT;

  /* Register this first so it can verify expected order of the rest.  */
  ATEXIT (fn_final); --slots_remaining;

  ATEXIT (fn1); --slots_remaining;
  ATEXIT (fn3); --slots_remaining;
  ATEXIT (fn1); --slots_remaining;
  ATEXIT (fn2); --slots_remaining;
  ATEXIT (fn1); --slots_remaining;
  ATEXIT (fn3); --slots_remaining;

  /* Fill the rest of available slots with fn0.  */
  while (slots_remaining > 0)
    {
      ATEXIT (fn0); --slots_remaining;
    }

  /* Verify that handlers registered above are inherited across fork.  */
  const pid_t child = fork ();
  switch (child)
    {
    case -1:
      printf ("fork: %m\n");
      _exit_with_flush (3);
    case 0:  /* Child.  */
      break;
    default:
      {
	int status;
	const pid_t exited = waitpid (child, &status, 0);
	if (child != exited)
	  {
	    printf ("unexpected child: %d, expected %d\n", exited, child);
	    _exit_with_flush (4);
	  }
	if (status != 0)
	  {
	    if (WIFEXITED (status))
	      printf ("unexpected exit status %d from child %d\n",
		      WEXITSTATUS (status), child);
	    else if (WIFSIGNALED (status))
	      printf ("unexpected signal %d from child %d\n",
		      WTERMSIG (status), child);
	    else
	      printf ("unexpected status %d from child %d\n", status, child);
	    _exit_with_flush (5);
	  }
      }
      break;
    }

  EXIT (2);  /* If we see this exit code, fn_final must have not worked.  */
}

#define TEST_FUNCTION do_test
#include <support/test-driver.c>

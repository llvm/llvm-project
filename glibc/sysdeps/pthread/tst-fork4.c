/* Test of fork updating child universe's pthread structures.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>


static int
do_test (void)
{
  pthread_t me = pthread_self ();

  pid_t pid = fork ();

  if (pid < 0)
    {
      printf ("fork: %m\n");
      return 1;
    }

  if (pid == 0)
    {
      int err = pthread_kill (me, SIGTERM);
      printf ("pthread_kill returned: %s\n", strerror (err));
      return 3;
    }

  int status;
  errno = 0;
  if (wait (&status) != pid)
    printf ("wait failed: %m\n");
  else if (WIFSIGNALED (status) && WTERMSIG (status) == SIGTERM)
    {
      printf ("child correctly died with SIGTERM\n");
      return 0;
    }
  else
    printf ("child died with bad status %#x\n", status);

  return 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

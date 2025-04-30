/* Check for failure paths handling for cancellation points.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Check that the cancel syscall points handles both the errno and return code
   correctly for invalid arguments.  */
static void *
tf (void *arg)
{
#ifdef SET_CANCEL_DISABLE
  pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, NULL);
#endif

  /* This is a cancellation point, but we should not be cancelled.  */
  int r = write (-1, 0, 0);

  if (r != -1 || errno != EBADF)
    {
      printf ("error: write returned %d, errno %d\n", r, errno);
      exit (1);
    }

  return NULL;
}

static int
do_test (void)
{
  pthread_t th;

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("error: pthread_create failed");
      exit (1);
    }

  if (pthread_join (th, NULL) != 0)
    {
      puts ("error: pthread_join failed");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

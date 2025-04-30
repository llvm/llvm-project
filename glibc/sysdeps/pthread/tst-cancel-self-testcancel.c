/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "tst-cancel-self-cleanup.c"


static int
do_test (void)
{
  int ret = 0, should_fail = 0;

  pthread_cleanup_push (cleanup, &should_fail);
  if ((ret = pthread_cancel (pthread_self ())) != 0)
    {
      printf ("cancel failed: %s\n", strerror (ret));
      exit (1);
    }

  pthread_testcancel ();

  printf ("Could not cancel self.\n");
  pthread_cleanup_pop (0);

  return 1;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "tst-mqueue.h"

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  if (geteuid () != 0)
    {
      puts ("this test requires root");
      return 0;
    }

  char name[sizeof "/tst-mqueue9-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue9-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 1, .mq_msgsize = 1 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return 0;
    }
  else
    add_temp_mq (name);

  if (seteuid (1) != 0)
    {
      printf ("failed to seteuid (1): %m\n");
      mq_unlink (name);
      return 0;
    }

  int result = 0;
  if (mq_unlink (name) == 0)
    {
      puts ("mq_unlink unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EACCES)
    {
      printf ("mq_unlink did not fail with EACCES: %m\n");
      result = 1;;
    }

  if (seteuid (0) != 0)
    {
      printf ("failed to seteuid (0): %m\n");
      result = 1;
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed with: %m\n");
      result = 1;
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close failed with: %m\n");
      result = 1;
    }

  return result;
}

#include "../test-skeleton.c"

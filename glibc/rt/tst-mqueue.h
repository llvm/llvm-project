/* Common code for message queue passing tests.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <mqueue.h>
#include <search.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

static int temp_mq_fd;

/* Add temporary files in list.  */
static void
__attribute__ ((unused))
add_temp_mq (const char *name)
{
  struct iovec iov[2];
  iov[0].iov_base = (char *) name;
  iov[0].iov_len = strlen (name);
  iov[1].iov_base = (char *) "\n";
  iov[1].iov_len = 1;
  if (writev (temp_mq_fd, iov, 2) != iov[0].iov_len + 1)
    printf ("Could not record temp mq filename %s\n", name);
}

/* Delete all temporary message queues.  */
static void
do_cleanup (void)
{
  if (lseek (temp_mq_fd, 0, SEEK_SET) != 0)
    return;

  FILE *f = fdopen (temp_mq_fd, "r");
  if (f == NULL)
    return;

  char *line = NULL;
  size_t n = 0;
  ssize_t rets;
  while ((rets = getline (&line, &n, f)) > 0)
    {
      if (line[rets - 1] != '\n')
        continue;

      line[rets - 1] = '\0';
      mq_unlink (line);
    }
  fclose (f);
}

static void
do_prepare (void)
{
  char name [] = "/tmp/tst-mqueueN.XXXXXX";
  temp_mq_fd = mkstemp (name);
  if (temp_mq_fd == -1)
    {
      printf ("Could not create temporary file %s: %m\n", name);
      exit (1);
    }
  unlink (name);
}

#define PREPARE(argc, argv) do_prepare ()
#define CLEANUP_HANDLER	do_cleanup ()

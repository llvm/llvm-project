/* Check recvmmsg cancellation.

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

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "tst-cancel4-common.h"

static void *
tf_recvmmsg (void *arg)
{
  struct sockaddr_un sun;

  tempfd = socket (AF_UNIX, SOCK_DGRAM, 0);
  if (tempfd == -1)
    FAIL_EXIT1 ("socket (AF_UNIX, SOCK_DGRAM, 0): %m");

  int tries = 0;
  do
    {
      if (++tries > 10)
	FAIL_EXIT1 ("too many unsuccessful bind calls");

      strcpy (sun.sun_path, "/tmp/tst-cancel4-socket-5-XXXXXX");
      if (mktemp (sun.sun_path) == NULL)
	FAIL_EXIT1 ("cannot generate temp file name");

      sun.sun_family = AF_UNIX;
    }
  while (bind (tempfd, (struct sockaddr *) &sun,
	       offsetof (struct sockaddr_un, sun_path)
	       + strlen (sun.sun_path) + 1) != 0);

  tempfname = strdup (sun.sun_path);

  tempfd2 = socket (AF_UNIX, SOCK_DGRAM, 0);
  if (tempfd2 == -1)
    FAIL_EXIT1 ("socket (AF_UNIX, SOCK_DGRAM, 0): %m");

  int r = pthread_barrier_wait (&b2);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    FAIL_EXIT1 ("pthread_barrier_wait");

  if (arg != NULL)
    {
      r = pthread_barrier_wait (&b2);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	FAIL_EXIT1 ("pthread_barrier_wait");
    }

  pthread_cleanup_push (cl, NULL);

  char mem[70];
  struct iovec iov[1];
  iov[0].iov_base = mem;
  iov[0].iov_len = arg == NULL ? sizeof (mem) : 0;

  struct mmsghdr mm;
  mm.msg_hdr.msg_name = &sun;
  mm.msg_hdr.msg_namelen = sizeof (sun);
  mm.msg_hdr.msg_iov = iov;
  mm.msg_hdr.msg_iovlen = 1;
  mm.msg_hdr.msg_control = NULL;
  mm.msg_hdr.msg_controllen = 0;

  ssize_t ret = recvmmsg (tempfd2, &mm, 1, 0, NULL);
  if (ret == -1 && errno == ENOSYS)
    exit (77);

  pthread_cleanup_pop (0);

  FAIL_EXIT1 ("recvmmsg returned");
}

struct cancel_tests tests[] =
{
  ADD_TEST (recvmmsg, 2, 1),
};
#define ntest_tf (sizeof (tests) / sizeof (tests[0]))

#include "tst-cancel4-common.c"

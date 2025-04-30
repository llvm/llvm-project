/* Common definition for tst-cancel4_* tests.

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

#include <pthread.h>

#include <support/check.h>
#include <support/xthread.h>
#include <support/xunistd.h>

/* Pipe descriptors.  */
static int fds[2];

/* Temporary file descriptor, to be closed after each round.  */
static int tempfd = -1;
static int tempfd2 = -1;
/* Name of temporary file to be removed after each round.  */
static char *tempfname;
/* Temporary message queue.  */
static int tempmsg = -1;

/* Often used barrier for two threads.  */
static pthread_barrier_t b2;

/* The WRITE_BUFFER_SIZE value needs to be chosen such that if we set
   the socket send buffer size to '1', a write of this size on that
   socket will block.

   The Linux kernel imposes a minimum send socket buffer size which
   has changed over the years.  As of Linux 3.10 the value is:

     2 * (2048 + SKB_DATA_ALIGN(sizeof(struct sk_buff)))

   which is attempting to make sure that with standard MTUs,
   TCP can always queue up at least 2 full sized packets.

   Furthermore, there is logic in the socket send paths that
   will allow one more packet (of any size) to be queued up as
   long as some socket buffer space remains.   Blocking only
   occurs when we try to queue up a new packet and the send
   buffer space has already been fully consumed.

   Therefore we must set this value to the largest possible value of
   the formula above (and since it depends upon the size of "struct
   sk_buff", it is dependent upon machine word size etc.) plus some
   slack space.  */

#define WRITE_BUFFER_SIZE 16384

/* Set the send buffer of socket S to 1 byte so any send operation
   done with WRITE_BUFFER_SIZE bytes will force syscall blocking.  */
static void
set_socket_buffer (int s)
{
  int val = 1;
  socklen_t len = sizeof (val);

  TEST_VERIFY_EXIT (setsockopt (s, SOL_SOCKET, SO_SNDBUF, &val,
		    sizeof (val)) == 0);
  TEST_VERIFY_EXIT (getsockopt (s, SOL_SOCKET, SO_SNDBUF, &val, &len) == 0);
  TEST_VERIFY_EXIT (val < WRITE_BUFFER_SIZE);
  printf("got size %d\n", val);
}

/* Cleanup handling test.  */
static int cl_called;

static void
cl (void *arg)
{
  ++cl_called;
}

/* Named pipe used to check for blocking open.  It should be closed
   after the cancellation handling.  */
static char fifoname[] = "/tmp/tst-cancel4-fifo-XXXXXX";
static int fifofd;

static void
__attribute__ ((used))
cl_fifo (void *arg)
{
  ++cl_called;

  unlink (fifoname);
  close (fifofd);
  fifofd = -1;
}

struct cancel_tests
{
  const char *name;
  void *(*tf) (void *);
  int nb;
  int only_early;
};
#define ADD_TEST(name, nbar, early) { #name, tf_##name, nbar, early }

/* Smoke test for SCM_RIGHTS.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* This test passes a file descriptor from a subprocess to the parent
   process, using recvmsg/sendmsg or recvmmsg/sendmmsg.  */

#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <string.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

/* String sent over the socket.  */
static char DATA[] = "descriptor";

/* Path that is to be opened and sent over the socket.  */
#define PATH "/etc"

/* True if sendmmsg/recvmmsg is to be used.  */
static bool use_multi_call;

/* The pair of sockets used for coordination.  The subprocess uses
   sockets[1].  */
static int sockets[2];

/* Subprocess side of one send/receive test.  */
_Noreturn static void
subprocess (void)
{
  /* The file descriptor to send.  */
  int fd = xopen (PATH, O_RDONLY, 0);

  struct iovec iov = { .iov_base = DATA, .iov_len = sizeof (DATA) };
  union
  {
    struct cmsghdr header;
    char bytes[CMSG_SPACE (sizeof (int))];
  } cmsg_storage;
  struct mmsghdr mmhdr =
    {
      .msg_hdr =
      {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = cmsg_storage.bytes,
        .msg_controllen = sizeof (cmsg_storage),
      },
    };

  /* Configure the file descriptor for sending.  */
  struct cmsghdr *cmsg = CMSG_FIRSTHDR (&mmhdr.msg_hdr);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN (sizeof (int));
  memcpy (CMSG_DATA (cmsg), &fd, sizeof (fd));
  mmhdr.msg_hdr.msg_controllen = cmsg->cmsg_len;

  /* Perform the send operation.  */
  int ret;
  if (use_multi_call)
    {
      ret = sendmmsg (sockets[1], &mmhdr, 1, 0);
      if (ret >= 0)
        ret = mmhdr.msg_len;
    }
  else
    ret = sendmsg (sockets[1], &mmhdr.msg_hdr, 0);
  TEST_COMPARE (ret, sizeof (DATA));

  xclose (fd);

  /* Stop the process from exiting.  */
  while (true)
    pause ();
}

/* Performs one send/receive test.  */
static void
one_test (void)
{
  TEST_COMPARE (socketpair (AF_UNIX, SOCK_STREAM, 0, sockets), 0);

  pid_t pid = xfork ();
  if (pid == 0)
    subprocess ();

  char data_storage[sizeof (DATA) + 1];
  struct iovec iov =
    {
      .iov_base = data_storage,
      .iov_len = sizeof (data_storage)
    };
  union
  {
    struct cmsghdr header;
    char bytes[CMSG_SPACE (sizeof (int))];
  } cmsg_storage;
  struct mmsghdr mmhdr =
    {
      .msg_hdr =
      {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = cmsg_storage.bytes,
        .msg_controllen = sizeof (cmsg_storage),
      },
    };

  /* Set up the space for receiving the file descriptor.  */
  struct cmsghdr *cmsg = CMSG_FIRSTHDR (&mmhdr.msg_hdr);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN (sizeof (int));
  mmhdr.msg_hdr.msg_controllen = cmsg->cmsg_len;

  /* Perform the receive operation.  */
  int ret;
  if (use_multi_call)
    {
      ret = recvmmsg (sockets[0], &mmhdr, 1, 0, NULL);
      if (ret >= 0)
        ret = mmhdr.msg_len;
    }
  else
    ret = recvmsg (sockets[0], &mmhdr.msg_hdr, 0);
  TEST_COMPARE (ret, sizeof (DATA));
  TEST_COMPARE_BLOB (data_storage, sizeof (DATA), DATA, sizeof (DATA));

  /* Extract the file descriptor.  */
  TEST_VERIFY (CMSG_FIRSTHDR (&mmhdr.msg_hdr) != NULL);
  TEST_COMPARE (CMSG_FIRSTHDR (&mmhdr.msg_hdr)->cmsg_len,
                CMSG_LEN (sizeof (int)));
  TEST_VERIFY (&cmsg_storage.header == CMSG_FIRSTHDR (&mmhdr.msg_hdr));
  int fd;
  memcpy (&fd, CMSG_DATA (CMSG_FIRSTHDR (&mmhdr.msg_hdr)), sizeof (fd));

  /* Verify the received file descriptor.  */
  TEST_VERIFY (fd > 2);
  struct stat64 st_fd;
  TEST_COMPARE (fstat64 (fd, &st_fd), 0);
  struct stat64 st_path;
  TEST_COMPARE (stat64 (PATH, &st_path), 0);
  TEST_COMPARE (st_fd.st_ino, st_path.st_ino);
  TEST_COMPARE (st_fd.st_dev, st_path.st_dev);
  xclose (fd);

  /* Terminate the subprocess.  */
  TEST_COMPARE (kill (pid, SIGUSR1), 0);
  int status;
  TEST_COMPARE (xwaitpid (pid, &status, 0), pid);
  TEST_VERIFY (WIFSIGNALED (status));
  TEST_COMPARE (WTERMSIG (status), SIGUSR1);

  xclose (sockets[0]);
  xclose (sockets[1]);
}

static int
do_test (void)
{
  one_test ();
  use_multi_call = true;
  one_test ();
  return 0;
}

#include <support/test-driver.c>

/* Test that failing system calls do set errno to the correct value.

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

#include <errno.h>
#include <limits.h>
#include <grp.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/statfs.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <netinet/in.h>
#include <libc-diag.h>

/* This is not an exhaustive test: only system calls that can be
   persuaded to fail with a consistent error code and no side effects
   are included.  Usually these are failures due to invalid arguments,
   with errno code EBADF or EINVAL.  The order of argument checks is
   unspecified, so we must take care to provide arguments that only
   allow _one_ failure mode.

   Note that all system calls that can fail with EFAULT are permitted
   to deliver a SIGSEGV signal instead, so we avoid supplying invalid
   pointers in general, and we do not attempt to test system calls
   that can only fail with EFAULT (e.g. gettimeofday, gethostname).

   Also note that root-only system calls (e.g. acct, reboot) may, when
   the test is run as an unprivileged user, fail due to insufficient
   privileges before bothering to do argument checks, so those are not
   tested either.

   Also, system calls that take enum or a set of flags as argument is
   not tested if POSIX doesn't specify exact binary values for all
   flags, and so any value passed to flags may become valid.

   Some tests assume "/bin/sh" names a file that exists and is not a
   directory.  */

#define test_wrp_rv(rtype, prtype, experr, syscall, ...)	\
  (__extension__ ({						\
    errno = 0xdead;						\
    rtype ret = syscall (__VA_ARGS__);				\
    int err = errno;						\
    int fail;							\
    if (ret == (rtype) -1 && err == experr)			\
      fail = 0;							\
    else							\
      {								\
        fail = 1;						\
        if (ret != (rtype) -1)					\
          printf ("FAIL: " #syscall ": didn't fail as expected"	\
               " (return "prtype")\n", ret);			\
        else if (err == 0xdead)					\
          puts("FAIL: " #syscall ": didn't update errno\n");	\
        else if (err != experr)					\
          printf ("FAIL: " #syscall				\
               ": errno is: %d (%s) expected: %d (%s)\n",	\
               err, strerror (err), experr, strerror (experr));	\
      }								\
    fail;							\
  }))

#define test_wrp(experr, syscall, ...)				\
  test_wrp_rv(int, "%d", experr, syscall, __VA_ARGS__)

static int
do_test (void)
{
  size_t pagesize = sysconf (_SC_PAGESIZE);
  struct statfs sfs;
  struct sockaddr sa;
  socklen_t sl;
  char buf[1];
  struct iovec iov[1] = { { buf, 1 } };
  struct sockaddr_in sin;
  sin.sin_family = AF_INET;
  sin.sin_port = htons (1026);
  sin.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
  struct msghdr msg;
  memset(&msg, 0, sizeof msg);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  int fails = 0;
  fails |= test_wrp (EBADF, accept, -1, &sa, &sl);
  fails |= test_wrp (EINVAL, access, "/", -1);
  fails |= test_wrp (EBADF, bind, -1, (struct sockaddr *)&sin, sizeof sin);
  fails |= test_wrp (ENOTDIR, chdir, "/bin/sh");
  fails |= test_wrp (EBADF, close, -1);
  fails |= test_wrp (EBADF, connect, -1, (struct sockaddr *)&sin, sizeof sin);
  fails |= test_wrp (EBADF, dup, -1);
  fails |= test_wrp (EBADF, dup2, -1, -1);
  fails |= test_wrp (EBADF, fchdir, -1);
  fails |= test_wrp (EBADF, fchmod, -1, 0);
  fails |= test_wrp (EBADF, fcntl, -1, 0);
  fails |= test_wrp (EBADF, fstatfs, -1, &sfs);
  fails |= test_wrp (EBADF, fsync, -1);
  fails |= test_wrp (EBADF, ftruncate, -1, 0);

#if __GNUC_PREREQ (7, 0)
  DIAG_PUSH_NEEDS_COMMENT;
  /* Avoid warnings about the second (size) argument being negative.  */
  DIAG_IGNORE_NEEDS_COMMENT (10.1, "-Wstringop-overflow");
#endif
  fails |= test_wrp (EINVAL, getgroups, -1, 0);
#if __GNUC_PREREQ (7, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif
  fails |= test_wrp (EBADF, getpeername, -1, &sa, &sl);
  fails |= test_wrp (EBADF, getsockname, -1, &sa, &sl);
  fails |= test_wrp (EBADF, getsockopt, -1, 0, 0, buf, &sl);
  fails |= test_wrp (EBADF, ioctl, -1, TIOCNOTTY);
  fails |= test_wrp (EBADF, listen, -1, 1);
  fails |= test_wrp (EBADF, lseek, -1, 0, 0);
  fails |= test_wrp (EINVAL, madvise, (void *) -1, -1, 0);
  fails |= test_wrp_rv (void *, "%p", EBADF,
                        mmap, 0, pagesize, PROT_READ, MAP_PRIVATE, -1, 0);
  fails |= test_wrp (EINVAL, mprotect, (void *) -1, pagesize, -1);
  fails |= test_wrp (EINVAL, msync, (void *) -1, pagesize, -1);
  fails |= test_wrp (EINVAL, munmap, (void *) -1, 0);
  fails |= test_wrp (EISDIR, open, "/bin", EISDIR, O_WRONLY);
  fails |= test_wrp (EBADF, read, -1, buf, 1);
  fails |= test_wrp (EINVAL, readlink, "/", buf, sizeof buf);
  fails |= test_wrp (EBADF, readv, -1, iov, 1);
  fails |= test_wrp (EBADF, recv, -1, buf, 1, 0);
  fails |= test_wrp (EBADF, recvfrom, -1, buf, 1, 0, &sa, &sl);
  fails |= test_wrp (EBADF, recvmsg, -1, &msg, 0);
  fails |= test_wrp (EINVAL, select, -1, 0, 0, 0, 0);
  fails |= test_wrp (EBADF, send, -1, buf, 1, 0);
  fails |= test_wrp (EBADF, sendmsg, -1, &msg, 0);
  fails |= test_wrp (EBADF, sendto, -1, buf, 1, 0, &sa, sl);
  fails |= test_wrp (EBADF, setsockopt, -1, 0, 0, buf, sizeof (*buf));
  fails |= test_wrp (EBADF, shutdown, -1, SHUT_RD);
  fails |= test_wrp (EBADF, write, -1, "Hello", sizeof ("Hello") );
  fails |= test_wrp (EBADF, writev, -1, iov, 1 );

  return fails;
}

#include "support/test-driver.c"

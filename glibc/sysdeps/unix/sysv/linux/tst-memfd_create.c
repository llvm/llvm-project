/* Test for the memfd_create system call.
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
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xunistd.h>
#include <sys/mman.h>

/* Return true if the descriptor has the FD_CLOEXEC flag set.  */
static bool
is_cloexec (int fd)
{
  int flags = fcntl (fd, F_GETFD);
  TEST_VERIFY (flags >= 0);
  return flags & FD_CLOEXEC;
}

/* Return the seals set on FD.  */
static int
get_seals (int fd)
{
  int flags = fcntl (fd, F_GET_SEALS);
  TEST_VERIFY (flags >= 0);
  return flags;
}

/* Return true if the F_SEAL_SEAL flag is set on the descriptor.  */
static bool
is_sealed (int fd)
{
  return get_seals (fd) & F_SEAL_SEAL;
}

static int
do_test (void)
{
  /* Initialized by the first call to memfd_create to 0 (memfd_create
     unsupported) or 1 (memfd_create is implemented in the kernel).
     Subsequent iterations check that the success/failure state is
     consistent.  */
  int supported = -1;

  for (int do_cloexec = 0; do_cloexec < 2; ++do_cloexec)
    for (int do_sealing = 0; do_sealing < 2; ++do_sealing)
      {
        int flags = 0;
        if (do_cloexec)
          flags |= MFD_CLOEXEC;
        if (do_sealing)
          flags |= MFD_ALLOW_SEALING;
        if  (test_verbose > 0)
          printf ("info: memfd_create with flags=0x%x\n", flags);
        int fd = memfd_create ("tst-memfd_create", flags);
        if (fd < 0)
          {
            if (errno == ENOSYS)
              {
                if (supported < 0)
                  {
                    printf ("warning: memfd_create is unsupported\n");
                    supported = 0;
                    continue;
                  }
                TEST_VERIFY (supported == 0);
                continue;
              }
            else
              FAIL_EXIT1 ("memfd_create: %m");
          }
        if (supported < 0)
          supported = 1;
        TEST_VERIFY (supported > 0);

        char *fd_path = xasprintf ("/proc/self/fd/%d", fd);
        char *link = xreadlink (fd_path);
        if (test_verbose > 0)
          printf ("info: memfd link: %s\n", link);
        TEST_VERIFY (strcmp (link, "memfd:tst-memfd_create (deleted)"));
        TEST_VERIFY (is_cloexec (fd) == do_cloexec);
        TEST_VERIFY (is_sealed (fd) == !do_sealing);
        if (do_sealing)
          {
            TEST_VERIFY (fcntl (fd, F_ADD_SEALS, F_SEAL_WRITE) == 0);
            TEST_VERIFY (!is_sealed (fd));
            TEST_VERIFY (get_seals (fd) & F_SEAL_WRITE);
            TEST_VERIFY (fcntl (fd, F_ADD_SEALS, F_SEAL_SEAL) == 0);
            TEST_VERIFY (is_sealed (fd));
          }
        xclose (fd);
        free (fd_path);
        free (link);
      }

  if (supported == 0)
    return EXIT_UNSUPPORTED;
  return 0;
}

#include <support/test-driver.c>

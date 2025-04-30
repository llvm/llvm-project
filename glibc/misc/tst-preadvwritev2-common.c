/* Common function for preadv2 and pwritev2 tests.
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

#include <limits.h>
#include <support/check.h>

#ifndef RWF_HIPRI
# define RWF_HIPRI 0
#endif
#ifndef RWF_DSYNC
# define RWF_DSYNC 0
#endif
#ifndef RWF_SYNC
# define RWF_SYNC 0
#endif
#ifndef RWF_NOWAIT
# define RWF_NOWAIT 0
#endif
#ifndef RWF_APPEND
# define RWF_APPEND 0
#endif
#define RWF_SUPPORTED	(RWF_HIPRI | RWF_DSYNC | RWF_SYNC | RWF_NOWAIT \
			 | RWF_APPEND)

/* Generic uio_lim.h does not define IOV_MAX.  */
#ifndef IOV_MAX
# define IOV_MAX 1024
#endif

static void
do_test_with_invalid_fd (void)
{
  char buf[256];
  struct iovec iov = { buf, sizeof buf };

  /* Check with flag being 0 to use the fallback code which calls pwritev
     or writev.  */
  TEST_VERIFY (preadv2 (-1, &iov, 1, -1, 0) == -1);
  TEST_COMPARE (errno, EBADF);
  TEST_VERIFY (pwritev2 (-1, &iov, 1, -1, 0) == -1);
  TEST_COMPARE (errno, EBADF);

  /* Same tests as before but with flags being different than 0.  Since
     there is no emulation for any flag value, fallback code returns
     ENOTSUP.  This is different running on a kernel with preadv2/pwritev2
     support, where EBADF is returned).  */
  TEST_VERIFY (preadv2 (-1, &iov, 1, 0, RWF_HIPRI) == -1);
  TEST_VERIFY (errno == EBADF || errno == ENOTSUP);
  TEST_VERIFY (pwritev2 (-1, &iov, 1, 0, RWF_HIPRI) == -1);
  TEST_VERIFY (errno == EBADF || errno == ENOTSUP);
}

static void
do_test_with_invalid_iov (void)
{
  {
    char buf[256];
    struct iovec iov;

    iov.iov_base = buf;
    iov.iov_len = (size_t)SSIZE_MAX + 1;

    TEST_VERIFY (preadv2 (temp_fd, &iov, 1, 0, 0) == -1);
    TEST_COMPARE (errno, EINVAL);
    TEST_VERIFY (pwritev2 (temp_fd, &iov, 1, 0, 0) == -1);
    TEST_COMPARE (errno, EINVAL);

    /* Same as for invalid file descriptor tests, emulation fallback
       first checks for flag value and return ENOTSUP.  */
    TEST_VERIFY (preadv2 (temp_fd, &iov, 1, 0, RWF_HIPRI) == -1);
    TEST_VERIFY (errno == EINVAL || errno == ENOTSUP);
    TEST_VERIFY (pwritev2 (temp_fd, &iov, 1, 0, RWF_HIPRI) == -1);
    TEST_VERIFY (errno == EINVAL || errno == ENOTSUP);
  }

  {
    /* An invalid iovec buffer should trigger an invalid memory access
       or an error (Linux for instance returns EFAULT).  */
    struct iovec iov[IOV_MAX+1] = { 0 };

    TEST_VERIFY (preadv2 (temp_fd, iov, IOV_MAX + 1, 0, RWF_HIPRI) == -1);
    TEST_VERIFY (errno == EINVAL || errno == ENOTSUP);
    TEST_VERIFY (pwritev2 (temp_fd, iov, IOV_MAX + 1, 0, RWF_HIPRI) == -1);
    TEST_VERIFY (errno == EINVAL || errno == ENOTSUP);
  }
}

static void
do_test_with_invalid_flags (void)
{
  /* Set the next bit from the mask of all supported flags.  */
  int invalid_flag = RWF_SUPPORTED != 0 ? __builtin_clz (RWF_SUPPORTED) : 2;
  invalid_flag = 0x1 << ((sizeof (int) * CHAR_BIT) - invalid_flag);

  char buf[32];
  const struct iovec vec = { .iov_base = buf, .iov_len = sizeof (buf) };
  if (preadv2 (temp_fd, &vec, 1, 0, invalid_flag) != -1)
    FAIL_EXIT1 ("preadv2 did not fail with an invalid flag");
  if (errno != ENOTSUP)
    FAIL_EXIT1 ("preadv2 failure did not set errno to ENOTSUP (%d)", errno);

  /* This might fail for compat syscall (32 bits running on 64 bits kernel)
     due a kernel issue.  */
  if (pwritev2 (temp_fd, &vec, 1, 0, invalid_flag) != -1)
    FAIL_EXIT1 ("pwritev2 did not fail with an invalid flag");
  if (errno != ENOTSUP)
    FAIL_EXIT1 ("pwritev2 failure did not set errno to ENOTSUP (%d)", errno);
}

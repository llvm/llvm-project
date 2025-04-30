/* Check non representable OFD locks regions in non-LFS mode for compat
   mode (BZ #20251)
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>

#include <support/temp_file.h>
#include <support/check.h>

#include <shlib-compat.h>
compat_symbol_reference (libc, fcntl, fcntl, GLIBC_2_0);

static char *temp_filename;
static int temp_fd;

static void
do_prepare (int argc, char **argv)
{
  temp_fd = create_temp_file ("tst-ofdlocks-compat.", &temp_filename);
  TEST_VERIFY_EXIT (temp_fd != -1);
}

#define PREPARE do_prepare

/* Linux between 4.13 and 4.15 return EOVERFLOW for LFS OFD locks usage
   in compat mode (non-LFS ABI running on a LFS default kernel, such as
   i386 on a x86_64 kernel or s390-32 on a s390-64 kernel) [1].  This is
   a kernel issue because __NR_fcntl64 is the expected way to use OFD locks
   (used on GLIBC for both fcntl and fcntl64).

   [1] https://sourceware.org/ml/libc-alpha/2018-07/msg00243.html  */

static int
do_test (void)
{
  /* The compat fcntl version for architectures which support non-LFS
     operations does not wrap the flock OFD argument, so the struct is passed
     unmodified to kernel.  It means no EOVERFLOW is returned, so operations
     with LFS should not incur in failure.  */

  struct flock64 lck64 = {
    .l_type   = F_WRLCK,
    .l_whence = SEEK_SET,
    .l_start  = (off64_t)INT32_MAX + 1024,
    .l_len    = 1024,
  };
  int ret = fcntl (temp_fd, F_OFD_SETLKW, &lck64);
  if (ret == -1 && errno == EINVAL)
    /* OFD locks are only available on Linux 3.15.  */
    FAIL_UNSUPPORTED ("fcntl (F_OFD_SETLKW) not supported");

  TEST_VERIFY_EXIT (ret == 0);

  /* Open file description locks placed through the same open file description
     (either by same file descriptor or a duplicated one created by fork,
     dup, fcntl F_DUPFD, etc.) overwrites then old lock.  To force a
     conflicting lock combination, it creates a new file descriptor.  */
  int fd = open64 (temp_filename, O_RDWR, 0666);
  TEST_VERIFY_EXIT (fd != -1);

  struct flock64 lck = {
    .l_type   = F_WRLCK,
    .l_whence = SEEK_SET,
    .l_start  = INT32_MAX - 1024,
    .l_len    = 4 * 1024,
  };
  TEST_VERIFY (fcntl (fd, F_OFD_GETLK, &lck) == 0);

  return 0;
}

#include <support/test-driver.c>

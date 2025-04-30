/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ryan S. Arnold <rsa@us.ibm.com>, 2011.

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

#include <fcntl.h>
#include <limits.h>
#include <paths.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/uio.h>


/* The purpose of this test is to verify that the INTERNAL_[V]SYSCALL_NCS
   macros on 64-bit platforms don't cast the return type to (int) which would
   erroneously sign extend the return value should the high bit of the bottom
   half of the word be '1'.  */

#if 0
/* Used to test the non power-of-2 code path.  */
#undef IOV_MAX
#define IOV_MAX 1000
#endif

/* writev() should report that it has written EXPECTED number of bytes.  */
#define EXPECTED ((size_t) INT32_MAX + 1)

static int
do_test (void)
{
  struct iovec iv[IOV_MAX];
  /* POSIX doesn't guarantee that IOV_MAX is pow of 2 but we're optimistic.  */
  size_t bufsz = EXPECTED / IOV_MAX;
  size_t bufrem = EXPECTED % IOV_MAX;

  /* If there's a remainder then IOV_MAX probably isn't a power of 2 and we
     need to make bufsz bigger so that the last iovec, iv[IOV_MAX-1], is free
     for the remainder.  */
  if (bufrem)
    {
      bufsz = bufsz + 1;
      bufrem = EXPECTED - (bufsz * (IOV_MAX - 1));
    }

  /* We writev to /dev/null since we're just testing writev's return value.  */
  int fd = open (_PATH_DEVNULL, O_WRONLY);
  if (fd == -1)
    {
      printf ("Unable to open /dev/null for writing.\n");
      return -1;
    }

  iv[0].iov_base = malloc (bufsz);
  if (iv[0].iov_base == NULL)
    {
      printf ("malloc (%zu) failed.\n", bufsz);
      close (fd);
      return -1;
    }
  iv[0].iov_len = bufsz;

  /* We optimistically presume that there isn't a remainder and set all iovec
     instances to the same base and len as the first instance.  */
  for (int i = 1; i < IOV_MAX; i++)
    {
      /* We don't care what the data is so reuse the allocation from iv[0];  */
      iv[i].iov_base = iv[0].iov_base;
      iv[i].iov_len = iv[0].iov_len;
    }

  /* If there is a remainder then we correct the last iov_len.  */
  if (bufrem)
    iv[IOV_MAX - 1].iov_len = bufrem;

  /* Write junk to /dev/null with the writev syscall in order to get a return
     of INT32_MAX+1 bytes to verify that the INTERNAL_SYSCALL wrappers aren't
     mangling the result if the signbit of a 32-bit number is set.  */
  ssize_t ret = writev (fd, iv, IOV_MAX);

  free (iv[0].iov_base);
  close (fd);

  if (ret != (ssize_t) EXPECTED)
    {
#ifdef ARTIFICIAL_LIMIT
      if (ret != (ssize_t) ARTIFICIAL_LIMIT)
#endif
	{
	  printf ("writev() return value: %zd != EXPECTED: %zd\n",
		  ret, EXPECTED);
	  return 1;
	}
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

/* Tests for preadv2 and pwritev2 (LFS version).
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

#define _FILE_OFFSET_BITS 64

#define PREADV(__fd, __iov, __iovcnt, __offset) \
  preadv2 (__fd, __iov, __iovcnt, __offset, 0)

#define PWRITEV(__fd, __iov, __iovcnt, __offset) \
  pwritev2 (__fd, __iov, __iovcnt, __offset, 0)

#include "tst-preadvwritev-common.c"
#include "tst-preadvwritev2-common.c"

static int
do_test (void)
{
  do_test_with_invalid_flags ();
  do_test_without_offset ();
  do_test_with_invalid_fd ();
  do_test_with_invalid_iov ();

  return do_test_with_offset (0);
}

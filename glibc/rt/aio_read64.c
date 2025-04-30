/* Asynchronous read, 64bit offset version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <bits/wordsize.h>
#if __WORDSIZE != 64

# include <aio.h>
# include <aio_misc.h>
# include <shlib-compat.h>

int
__aio_read64 (struct aiocb64 *aiocbp)
{
  return (__aio_enqueue_request ((aiocb_union *) aiocbp, LIO_READ64) == NULL
	  ? -1 : 0);
}

# if PTHREAD_IN_LIBC
versioned_symbol (libc, __aio_read64, aio_read64, GLIBC_2_34);
#  if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
compat_symbol (librt, __aio_read64, aio_read64, GLIBC_2_1);
#  endif
# else /* !PTHREAD_IN_LIBC */
strong_alias (__aio_read64, aio_read64)
# endif
#endif /* __WORDSIZE != 64 */

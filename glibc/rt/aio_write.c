/* Asynchronous write.
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
#if __WORDSIZE == 64
# define aio_write64 XXX
# include <aio.h>
/* And undo the hack.  */
# undef aio_write64
#else
# include <aio.h>
#endif

#include <aio_misc.h>
#include <shlib-compat.h>

int
__aio_write (struct aiocb *aiocbp)
{
  return (__aio_enqueue_request ((aiocb_union *) aiocbp, LIO_WRITE) == NULL
	  ? -1 : 0);
}

#if PTHREAD_IN_LIBC
versioned_symbol (libc, __aio_write, aio_write, GLIBC_2_34);
# if __WORDSIZE == 64
versioned_symbol (libc, __aio_write, aio_write64, GLIBC_2_34);
# endif
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
compat_symbol (librt, __aio_write, aio_write, GLIBC_2_1);
#  if __WORDSIZE == 64
compat_symbol (librt, __aio_write, aio_write64, GLIBC_2_1);
#  endif
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__aio_write, aio_write)
# if __WORDSIZE == 64
weak_alias (__aio_write, aio_write64)
#endif
#endif  /* !PTHREAD_IN_LIBC */

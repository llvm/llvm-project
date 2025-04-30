/* Return error status of asynchronous I/O request.
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


/* We use an UGLY hack to prevent gcc from finding us cheating.  The
   implementation of aio_error and aio_error64 are identical and so
   we want to avoid code duplication by using aliases.  But gcc sees
   the different parameter lists and prints a warning.  We define here
   a function so that aio_error64 has no prototype.  */
#define aio_error64 XXX
#include <aio.h>
/* And undo the hack.  */
#undef aio_error64

#include <aio_misc.h>
#include <pthreadP.h>
#include <shlib-compat.h>

int
__aio_error (const struct aiocb *aiocbp)
{
  int ret;

  /* Acquire the mutex to make sure all operations for this request are
     complete.  */
  __pthread_mutex_lock (&__aio_requests_mutex);
  ret = aiocbp->__error_code;
  __pthread_mutex_unlock (&__aio_requests_mutex);

  return ret;
}

#if PTHREAD_IN_LIBC
versioned_symbol (libc, __aio_error, aio_error, GLIBC_2_34);
versioned_symbol (libc, __aio_error, aio_error64, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
compat_symbol (librt, __aio_error, aio_error, GLIBC_2_1);
compat_symbol (librt, __aio_error, aio_error64, GLIBC_2_1);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__aio_error, aio_error)
weak_alias (__aio_error, aio_error64)
#endif /* !PTHREAD_IN_LIBC */

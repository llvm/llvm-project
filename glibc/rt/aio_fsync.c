/* Synchronize I/O in given file descriptor.
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
   implementation of aio_fsync and aio_fsync64 are identical and so
   we want to avoid code duplication by using aliases.  But gcc sees
   the different parameter lists and prints a warning.  We define here
   a function so that aio_fsync64 has no prototype.  */
#define aio_fsync64 XXX
#include <aio.h>
/* And undo the hack.  */
#undef aio_fsync64
#include <errno.h>
#include <fcntl.h>

#include <aio_misc.h>
#include <shlib-compat.h>

int
__aio_fsync (int op, struct aiocb *aiocbp)
{
  if (op != O_DSYNC && __builtin_expect (op != O_SYNC, 0))
    {
      __set_errno (EINVAL);
      return -1;
    }

  /* Verify that this is an open file descriptor.  */
  if (__glibc_unlikely (__fcntl (aiocbp->aio_fildes, F_GETFL) == -1))
    {
      __set_errno (EBADF);
      return -1;
    }

  return (__aio_enqueue_request ((aiocb_union *) aiocbp,
				 op == O_SYNC ? LIO_SYNC : LIO_DSYNC) == NULL
	  ? -1 : 0);
}

#if PTHREAD_IN_LIBC
versioned_symbol (libc, __aio_fsync, aio_fsync, GLIBC_2_34);
versioned_symbol (libc, __aio_fsync, aio_fsync64, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
compat_symbol (librt, __aio_fsync, aio_fsync, GLIBC_2_1);
compat_symbol (librt, __aio_fsync, aio_fsync64, GLIBC_2_1);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__aio_fsync, aio_fsync)
weak_alias (__aio_fsync, aio_fsync64)
#endif /* !PTHREAD_IN_LIBC */

/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include "pthreadP.h"
#include <shlib-compat.h>

#ifndef NEW_VERNUM
# define NEW_VERNUM GLIBC_2_3_3
#endif


int
__pthread_attr_setstacksize (pthread_attr_t *attr, size_t stacksize)
{
  struct pthread_attr *iattr;

  iattr = (struct pthread_attr *) attr;

  /* Catch invalid sizes.  */
  int ret = check_stacksize_attr (stacksize);
  if (ret)
    return ret;

  iattr->stacksize = stacksize;

  return 0;
}
versioned_symbol (libc, __pthread_attr_setstacksize,
		  pthread_attr_setstacksize, GLIBC_2_34);


#if PTHREAD_STACK_MIN == 16384
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __pthread_attr_setstacksize,
	       pthread_attr_setstacksize, GLIBC_2_1);
# endif
#else /* PTHREAD_STACK_MIN != 16384 */
# if OTHER_SHLIB_COMPAT (libpthread, NEW_VERNUM, GLIBC_2_34)
compat_symbol (libpthread, __pthread_attr_setstacksize,
	       pthread_attr_setstacksize, NEW_VERNUM);
# endif

# if OTHER_SHLIB_COMPAT(libpthread, GLIBC_2_1, NEW_VERNUM)

int
__old_pthread_attr_setstacksize (pthread_attr_t *attr, size_t stacksize)
{
  struct pthread_attr *iattr;

  iattr = (struct pthread_attr *) attr;

  /* Catch invalid sizes.  */
  if (stacksize < 16384)
    return EINVAL;

#  ifdef STACKSIZE_ADJUST
  STACKSIZE_ADJUST;
#  endif

  iattr->stacksize = stacksize;

  return 0;
}

compat_symbol (libpthread, __old_pthread_attr_setstacksize,
	       pthread_attr_setstacksize, GLIBC_2_1);
# endif /* OTHER_SHLIB_COMPAT */
#endif /* PTHREAD_STACK_MIN != 16384 */

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
#include "pthreadP.h"
#include <shlib-compat.h>


#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_3_2)
int
__pthread_cond_init_2_0 (pthread_cond_2_0_t *cond,
			 const pthread_condattr_t *cond_attr)
{
  struct pthread_condattr *icond_attr = (struct pthread_condattr *) cond_attr;

  /* The type of the first argument is actually that of the old, too
     small pthread_cond_t.  We use only the first word of it, as a
     pointer.  */
  cond->cond = NULL;

  /* We can't support PSHARED condvars in the old pthread_cond_*
     functions and neither clocks other than CLOCK_REALTIME.  */
  if (icond_attr != NULL && icond_attr->value)
    return EINVAL;

  return 0;
}
compat_symbol (libc, __pthread_cond_init_2_0, pthread_cond_init,
	       GLIBC_2_0);
#endif

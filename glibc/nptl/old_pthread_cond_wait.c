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
#include <stdlib.h>
#include "pthreadP.h"
#include <atomic.h>
#include <shlib-compat.h>


#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_3_2)
int
__pthread_cond_wait_2_0 (pthread_cond_2_0_t *cond, pthread_mutex_t *mutex)
{
  if (cond->cond == NULL)
    {
      pthread_cond_t *newcond;

      newcond = (pthread_cond_t *) calloc (sizeof (pthread_cond_t), 1);
      if (newcond == NULL)
	return ENOMEM;

      if (atomic_compare_and_exchange_bool_acq (&cond->cond, newcond, NULL))
	/* Somebody else just initialized the condvar.  */
	free (newcond);
    }

  return __pthread_cond_wait (cond->cond, mutex);
}
compat_symbol (libpthread, __pthread_cond_wait_2_0, pthread_cond_wait,
	       GLIBC_2_0);
#endif

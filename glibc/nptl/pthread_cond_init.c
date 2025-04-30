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

#include <shlib-compat.h>
#include "pthreadP.h"
#include <stap-probe.h>
#include <string.h>
#include <sys/mman.h>


/* See __pthread_cond_wait for details.  */
int
__pthread_cond_init (pthread_cond_t *cond, const pthread_condattr_t *cond_attr)
{
  ASSERT_TYPE_SIZE (pthread_cond_t, __SIZEOF_PTHREAD_COND_T);

  struct pthread_condattr *icond_attr = (struct pthread_condattr *) cond_attr;

  memset (cond, 0, sizeof (pthread_cond_t));

  /* Update the pretty printers if the internal representation of icond_attr
     is changed.  */

  /* Iff not equal to ~0l, this is a PTHREAD_PROCESS_PRIVATE condvar.  */
  if (icond_attr != NULL && (icond_attr->value & 1) != 0)
    cond->__data.__wrefs |= __PTHREAD_COND_SHARED_MASK;
  int clockid = (icond_attr != NULL
		 ? ((icond_attr->value >> 1) & ((1 << COND_CLOCK_BITS) - 1))
		 : CLOCK_REALTIME);
  /* If 0, CLOCK_REALTIME is used; CLOCK_MONOTONIC otherwise.  */
  if (clockid != CLOCK_REALTIME)
    cond->__data.__wrefs |= __PTHREAD_COND_CLOCK_MONOTONIC_MASK;

  LIBC_PROBE (cond_init, 2, cond, cond_attr);

  __try_to_mark_as_unmigratable (cond);

  return 0;
}
libc_hidden_def (__pthread_cond_init)
versioned_symbol (libc, __pthread_cond_init,
		  pthread_cond_init, GLIBC_2_3_2);

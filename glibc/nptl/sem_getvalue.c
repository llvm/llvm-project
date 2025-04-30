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

#include <semaphore.h>
#include <shlib-compat.h>
#include "semaphoreP.h"
#include <atomic.h>


int
__new_sem_getvalue (sem_t *sem, int *sval)
{
  struct new_sem *isem = (struct new_sem *) sem;

  /* XXX Check for valid SEM parameter.  */
  /* FIXME This uses relaxed MO, even though POSIX specifies that this function
     should be linearizable.  However, its debatable whether linearizability
     is the right requirement.  We need to follow up with POSIX and, if
     necessary, use a stronger MO here and elsewhere (e.g., potentially
     release MO in all places where we consume a token).  */

#if __HAVE_64B_ATOMICS
  *sval = atomic_load_relaxed (&isem->data) & SEM_VALUE_MASK;
#else
  *sval = atomic_load_relaxed (&isem->value) >> SEM_VALUE_SHIFT;
#endif

  return 0;
}
versioned_symbol (libc, __new_sem_getvalue, sem_getvalue, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT(libpthread, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libpthread, __new_sem_getvalue, sem_getvalue, GLIBC_2_1);
#endif

#if OTHER_SHLIB_COMPAT(libpthread, GLIBC_2_0, GLIBC_2_1)
int
__old_sem_getvalue (sem_t *sem, int *sval)
{
  struct old_sem *isem = (struct old_sem *) sem;
  *sval = isem->value;
  return 0;
}
compat_symbol (libpthread, __old_sem_getvalue, sem_getvalue, GLIBC_2_0);
#endif

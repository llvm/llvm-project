/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <libc-lock.h>
#include <errno.h>
#include <stdlib.h>
#include <pthreadP.h>

/* This serves as stub "self" pointer for libc locks when TLS is not initialized
   yet.  */
char __libc_lock_self0[0];

/* Placeholder for key creation routine from Hurd cthreads library.  */
int
weak_function
__cthread_keycreate (__cthread_key_t *key)
{
  __set_errno (ENOSYS);
 *key = -1;
  return -1;
}

/* Placeholder for key retrieval routine from Hurd cthreads library.  */
int
weak_function
__cthread_getspecific (__cthread_key_t key, void **pval)
{
  *pval = NULL;
  __set_errno (ENOSYS);
  return -1;
}

/* Placeholder for key setting routine from Hurd cthreads library.  */
int
weak_function
__cthread_setspecific (__cthread_key_t key, void *val)
{
  __set_errno (ENOSYS);
  return -1;
}

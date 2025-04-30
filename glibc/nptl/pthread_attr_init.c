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
#include <string.h>
#include <unistd.h>
#include "pthreadP.h"

#include <shlib-compat.h>
#include <sys/mman.h>


struct pthread_attr *__attr_list;
int __attr_list_lock = LLL_LOCK_INITIALIZER;


int
__pthread_attr_init (pthread_attr_t *attr)
{
  struct pthread_attr *iattr;

  ASSERT_TYPE_SIZE (pthread_attr_t, __SIZEOF_PTHREAD_ATTR_T);
  ASSERT_PTHREAD_INTERNAL_SIZE (pthread_attr_t, struct pthread_attr);

  /* Many elements are initialized to zero so let us do it all at
     once.  This also takes care of clearing the bytes which are not
     internally used.  */
  memset (attr, '\0', __SIZEOF_PTHREAD_ATTR_T);

  iattr = (struct pthread_attr *) attr;

  /* Default guard size specified by the standard.  */
  iattr->guardsize = __getpagesize ();

  __try_to_mark_as_unmigratable (iattr);

  return 0;
}
libc_hidden_def (__pthread_attr_init)
versioned_symbol (libc, __pthread_attr_init, pthread_attr_init, GLIBC_2_1);


#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_1)
int
__pthread_attr_init_2_0 (pthread_attr_t *attr)
{
  /* This code is specific to the old LinuxThread code which has a too
     small pthread_attr_t definition.  The struct looked like
     this:  */
  struct old_attr
  {
    int detachstate;
    int schedpolicy;
    struct sched_param schedparam;
    int inheritsched;
    int scope;
  };
  struct pthread_attr *iattr;

  /* Many elements are initialized to zero so let us do it all at
     once.  This also takes care of clearing the bytes which are not
     internally used.  */
  memset (attr, '\0', sizeof (struct old_attr));

  iattr = (struct pthread_attr *) attr;
  iattr->flags |= ATTR_FLAG_OLDATTR;

  __try_to_mark_as_unmigratable (iattr);

  /* We cannot enqueue the attribute because that member is not in the
     old attribute structure.  */
  return 0;
}
compat_symbol (libc, __pthread_attr_init_2_0, pthread_attr_init, GLIBC_2_0);
#endif

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


int
__pthread_attr_setdetachstate (pthread_attr_t *attr, int detachstate)
{
  struct pthread_attr *iattr;

  iattr = (struct pthread_attr *) attr;

  /* Catch invalid values.  */
  if (detachstate != PTHREAD_CREATE_DETACHED
      && __builtin_expect (detachstate != PTHREAD_CREATE_JOINABLE, 0))
    return EINVAL;

  /* Set the flag.  It is nonzero if threads are created detached.  */
  if (detachstate == PTHREAD_CREATE_DETACHED)
    iattr->flags |= ATTR_FLAG_DETACHSTATE;
  else
    iattr->flags &= ~ATTR_FLAG_DETACHSTATE;

  return 0;
}
strong_alias (__pthread_attr_setdetachstate, pthread_attr_setdetachstate)

/* pthread_mutexattr_setprotocol.  Hurd version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include <pt-internal.h>
#include "pt-mutex.h"
#include <hurdlock.h>

int
pthread_mutexattr_setprotocol (pthread_mutexattr_t *attrp, int proto)
{
  (void) attrp;
  return (proto == PTHREAD_PRIO_NONE
	  ? 0
	  : (proto != PTHREAD_PRIO_INHERIT
	     && proto != PTHREAD_PRIO_PROTECT) ? EINVAL : ENOTSUP);
}

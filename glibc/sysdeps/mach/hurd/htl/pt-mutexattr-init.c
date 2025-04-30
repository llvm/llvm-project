/* pthread_mutexattr_init.  Hurd version.
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

static const pthread_mutexattr_t dfl_attr = {
  .__prioceiling = 0,
  .__protocol = PTHREAD_PRIO_NONE,
  .__pshared = PTHREAD_PROCESS_PRIVATE,
  .__mutex_type = __PTHREAD_MUTEX_TIMED
};

int
__pthread_mutexattr_init (pthread_mutexattr_t *attrp)
{
  ASSERT_TYPE_SIZE (pthread_mutexattr_t, __SIZEOF_PTHREAD_MUTEXATTR_T);

  *attrp = dfl_attr;
  return 0;
}
weak_alias (__pthread_mutexattr_init, pthread_mutexattr_init)

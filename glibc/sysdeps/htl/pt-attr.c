/* Default attributes.  Generic version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <sched.h>
#include <stddef.h>
#include <limits.h>

#include <pt-internal.h>

struct __pthread_attr __pthread_default_attr = {
  __schedparam: { sched_priority: 0 },
  __stacksize: 0,
  __stackaddr: NULL,
#ifdef PAGESIZE
  __guardsize: PAGESIZE,
#else
  __guardsize: 1,
#endif /* PAGESIZE */
  __detachstate: PTHREAD_CREATE_JOINABLE,
  __inheritsched: PTHREAD_EXPLICIT_SCHED,
  __contentionscope: PTHREAD_SCOPE_SYSTEM,
  __schedpolicy: SCHED_OTHER
};

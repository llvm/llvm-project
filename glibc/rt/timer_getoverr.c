/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kaz Kylheku <kaz@ashi.footprints.net>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <pthread.h>
#include <time.h>

#include "posix-timer.h"


/* Get expiration overrun for timer TIMERID.  */
int
timer_getoverrun (timer_t timerid)
{
  struct timer_node *timer;
  int retval = -1;

  pthread_mutex_lock (&__timer_mutex);

  if (! timer_valid (timer = timer_id2ptr (timerid)))
    __set_errno (EINVAL);
  else
    retval = timer->overrun_count;

  pthread_mutex_unlock (&__timer_mutex);

  return retval;
}

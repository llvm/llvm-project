/* Broadcast a condition.  Generic version.
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

#include <pt-internal.h>

/* Unblock all threads that are blocked on condition variable COND.  */
int
__pthread_cond_broadcast (pthread_cond_t *cond)
{
  struct __pthread *wakeup;

  __pthread_spin_wait (&cond->__lock);
  while ((wakeup = cond->__queue))
    {
      __pthread_dequeue (wakeup);
      __pthread_spin_unlock (&cond->__lock);
      /* Wake it up without spin held, so it may have a chance to really
         preempt us */
      __pthread_wakeup (wakeup);
      __pthread_spin_wait (&cond->__lock);
    }
  __pthread_spin_unlock (&cond->__lock);

  return 0;
}

weak_alias (__pthread_cond_broadcast, pthread_cond_broadcast);

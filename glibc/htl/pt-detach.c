/* Detach a thread.
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

#include <errno.h>
#include <pthread.h>
#include <stddef.h>

#include <pt-internal.h>

/* Indicate that the storage for THREAD can be reclaimed when it
   terminates.  */
int
__pthread_detach (pthread_t thread)
{
  struct __pthread *pthread;
  int err = 0;

  /* Lookup the thread structure for THREAD.  */
  pthread = __pthread_getid (thread);
  if (pthread == NULL)
    return ESRCH;

  __pthread_mutex_lock (&pthread->state_lock);

  switch (pthread->state)
    {
    case PTHREAD_JOINABLE:
      /* THREAD still running.  Mark it as detached such that its
         resources can be reclaimed as soon as the thread exits.  */
      pthread->state = PTHREAD_DETACHED;

      /* Broadcast the condition.  This will make threads that are
         waiting to join THREAD continue with hopefully disastrous
         consequences instead of blocking indefinitely.  */
      __pthread_cond_broadcast (&pthread->state_cond);
      __pthread_mutex_unlock (&pthread->state_lock);

      __pthread_dealloc (pthread);
      break;

    case PTHREAD_EXITED:
      __pthread_mutex_unlock (&pthread->state_lock);

      /* THREAD has already exited.  PTHREAD remained after the thread
         exited in order to provide the exit status, but it turns out
         it won't be needed.  */
      __pthread_dealloc (pthread);
      break;

    case PTHREAD_TERMINATED:
      /* Pretend THREAD wasn't there in the first place.  */
      __pthread_mutex_unlock (&pthread->state_lock);
      err = ESRCH;
      break;

    default:
      /* Thou shalt not detach non-joinable threads!  */
      __pthread_mutex_unlock (&pthread->state_lock);
      err = EINVAL;
      break;
    }

  return err;
}
weak_alias (__pthread_detach, pthread_detach)
hidden_def (__pthread_detach)

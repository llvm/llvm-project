/* Wait for thread termination.
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

/* Make calling thread wait for termination of thread THREAD.  Return
   the exit status of the thread in *STATUS.  */
static int
__pthread_join_common (pthread_t thread, void **status, int try,
		       clockid_t clockid,
		       const struct timespec *abstime)
{
  struct __pthread *pthread;
  int err = 0;

  /* Lookup the thread structure for THREAD.  */
  pthread = __pthread_getid (thread);
  if (pthread == NULL)
    return ESRCH;

  if (pthread == _pthread_self ())
    return EDEADLK;

  __pthread_mutex_lock (&pthread->state_lock);

  if (try == 0)
    {
      pthread_cleanup_push ((void (*)(void *)) __pthread_mutex_unlock,
			    &pthread->state_lock);

      /* Rely on pthread_cond_wait being a cancellation point to make
	 pthread_join one too.  */
      while (pthread->state == PTHREAD_JOINABLE && err != ETIMEDOUT)
	err = __pthread_cond_clockwait (&pthread->state_cond,
					&pthread->state_lock,
					clockid, abstime);

      pthread_cleanup_pop (0);
    }

  switch (pthread->state)
    {
    case PTHREAD_JOINABLE:
      __pthread_mutex_unlock (&pthread->state_lock);
      if (err != ETIMEDOUT)
	err = EBUSY;
      break;

    case PTHREAD_EXITED:
      /* THREAD has already exited.  Salvage its exit status.  */
      if (status != NULL)
	*status = pthread->status;

      __pthread_mutex_unlock (&pthread->state_lock);

      __pthread_dealloc (pthread);
      break;

    case PTHREAD_TERMINATED:
      /* Pretend THREAD wasn't there in the first place.  */
      __pthread_mutex_unlock (&pthread->state_lock);
      err = ESRCH;
      break;

    default:
      /* Thou shalt not join non-joinable threads!  */
      __pthread_mutex_unlock (&pthread->state_lock);
      err = EINVAL;
      break;
    }

  return err;
}

int
__pthread_join (pthread_t thread, void **status)
{
  return __pthread_join_common (thread, status, 0, CLOCK_REALTIME, NULL);
}
weak_alias (__pthread_join, pthread_join);

int
__pthread_tryjoin_np (pthread_t thread, void **status)
{
  return __pthread_join_common (thread, status, 1, CLOCK_REALTIME, NULL);
}
weak_alias (__pthread_tryjoin_np, pthread_tryjoin_np);

int
__pthread_timedjoin_np (pthread_t thread, void **status,
			const struct timespec *abstime)
{
  return __pthread_join_common (thread, status, 0, CLOCK_REALTIME, abstime);
}
weak_alias (__pthread_timedjoin_np, pthread_timedjoin_np);

int
__pthread_clockjoin_np (pthread_t thread, void **status,
			clockid_t clockid,
			const struct timespec *abstime)
{
  return __pthread_join_common (thread, status, 0, clockid, abstime);
}
weak_alias (__pthread_clockjoin_np, pthread_clockjoin_np);

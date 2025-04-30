/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sysdep.h>
#include <internaltypes.h>
#include <pthreadP.h>
#include "kernel-posix-timers.h"
#include "kernel-posix-cpu-timers.h"
#include <shlib-compat.h>

int
___timer_create (clockid_t clock_id, struct sigevent *evp, timer_t *timerid)
{
  {
    clockid_t syscall_clockid = (clock_id == CLOCK_PROCESS_CPUTIME_ID
				 ? MAKE_PROCESS_CPUCLOCK (0, CPUCLOCK_SCHED)
				 : clock_id == CLOCK_THREAD_CPUTIME_ID
				 ? MAKE_THREAD_CPUCLOCK (0, CPUCLOCK_SCHED)
				 : clock_id);

    /* If the user wants notification via a thread we need to handle
       this special.  */
    if (evp == NULL
	|| __builtin_expect (evp->sigev_notify != SIGEV_THREAD, 1))
      {
	struct sigevent local_evp;

	if (evp == NULL)
	  {
	    /* The kernel has to pass up the timer ID which is a
	       userlevel object.  Therefore we cannot leave it up to
	       the kernel to determine it.  */
	    local_evp.sigev_notify = SIGEV_SIGNAL;
	    local_evp.sigev_signo = SIGALRM;
	    local_evp.sigev_value.sival_ptr = NULL;

	    evp = &local_evp;
	  }

	kernel_timer_t ktimerid;
	if (INLINE_SYSCALL_CALL (timer_create, syscall_clockid, evp,
				 &ktimerid) == -1)
	  return -1;

	*timerid = kernel_timer_to_timerid (ktimerid);
      }
    else
      {
	/* Create the helper thread.  */
	__pthread_once (&__timer_helper_once, __timer_start_helper_thread);
	if (__timer_helper_tid == 0)
	  {
	    /* No resources to start the helper thread.  */
	    __set_errno (EAGAIN);
	    return -1;
	  }

	struct timer *newp = malloc (sizeof (struct timer));
	if (newp == NULL)
	  return -1;

	/* Copy the thread parameters the user provided.  */
	newp->sival = evp->sigev_value;
	newp->thrfunc = evp->sigev_notify_function;

	/* We cannot simply copy the thread attributes since the
	   implementation might keep internal information for
	   each instance.  */
	__pthread_attr_init (&newp->attr);
	if (evp->sigev_notify_attributes != NULL)
	  {
	    struct pthread_attr *nattr;
	    struct pthread_attr *oattr;

	    nattr = (struct pthread_attr *) &newp->attr;
	    oattr = (struct pthread_attr *) evp->sigev_notify_attributes;

	    nattr->schedparam = oattr->schedparam;
	    nattr->schedpolicy = oattr->schedpolicy;
	    nattr->flags = oattr->flags;
	    nattr->guardsize = oattr->guardsize;
	    nattr->stackaddr = oattr->stackaddr;
	    nattr->stacksize = oattr->stacksize;
	  }

	/* In any case set the detach flag.  */
	__pthread_attr_setdetachstate (&newp->attr, PTHREAD_CREATE_DETACHED);

	/* Create the event structure for the kernel timer.  */
	struct sigevent sev =
	  { .sigev_value.sival_ptr = newp,
	    .sigev_signo = SIGTIMER,
	    .sigev_notify = SIGEV_SIGNAL | SIGEV_THREAD_ID,
	    ._sigev_un = { ._pad = { [0] = __timer_helper_tid } } };

	/* Create the timer.  */
	int res;
	res = INTERNAL_SYSCALL_CALL (timer_create, syscall_clockid, &sev,
				     &newp->ktimerid);
	if (INTERNAL_SYSCALL_ERROR_P (res))
	  {
	    free (newp);
	    __set_errno (INTERNAL_SYSCALL_ERRNO (res));
	    return -1;
	  }

	/* Add to the queue of active timers with thread delivery.  */
	__pthread_mutex_lock (&__timer_active_sigev_thread_lock);
	newp->next = __timer_active_sigev_thread;
	__timer_active_sigev_thread = newp;
	__pthread_mutex_unlock (&__timer_active_sigev_thread_lock);

	*timerid = timer_to_timerid (newp);
      }
  }

  return 0;
}
versioned_symbol (libc, ___timer_create, timer_create, GLIBC_2_34);
libc_hidden_ver (___timer_create, __timer_create)

#if TIMER_T_WAS_INT_COMPAT
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (librt, ___timer_create, timer_create, GLIBC_2_3_3);
# endif

# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_3_3)
timer_t __timer_compat_list[OLD_TIMER_MAX];

int
__timer_create_old (clockid_t clock_id, struct sigevent *evp, int *timerid)
{
  timer_t newp;

  int res = __timer_create (clock_id, evp, &newp);
  if (res == 0)
    {
      int i;
      for (i = 0; i < OLD_TIMER_MAX; ++i)
	if (__timer_compat_list[i] == NULL
	    && ! atomic_compare_and_exchange_bool_acq (&__timer_compat_list[i],
						       newp, NULL))
	  {
	    *timerid = i;
	    break;
	  }

      if (__glibc_unlikely (i == OLD_TIMER_MAX))
	{
	  /* No free slot.  */
	  __timer_delete (newp);
	  __set_errno (EINVAL);
	  res = -1;
	}
    }

  return res;
}
compat_symbol (librt, __timer_create_old, timer_create, GLIBC_2_2);
# endif /* OTHER_SHLIB_COMPAT */

#else /* !TIMER_T_WAS_INT_COMPAT */
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, ___timer_create, timer_create, GLIBC_2_2);
# endif
#endif /* !TIMER_T_WAS_INT_COMPAT */

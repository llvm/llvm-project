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

#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/types.h>


/* Nonzero if the system calls are not available.  */
extern int __no_posix_timers attribute_hidden;

/* Callback to start helper thread.  */
extern void __timer_start_helper_thread (void) attribute_hidden;

/* Control variable for helper thread creation.  */
extern pthread_once_t __timer_helper_once attribute_hidden;

/* Called from fork so that the new subprocess re-creates the
   notification thread if necessary.  */
void __timer_fork_subprocess (void) attribute_hidden;

/* TID of the helper thread.  */
extern pid_t __timer_helper_tid attribute_hidden;

/* List of active SIGEV_THREAD timers.  */
extern struct timer *__timer_active_sigev_thread attribute_hidden;

/* Lock for __timer_active_sigev_thread.  */
extern pthread_mutex_t __timer_active_sigev_thread_lock attribute_hidden;

extern __typeof (timer_create) __timer_create;
libc_hidden_proto (__timer_create)
extern __typeof (timer_delete) __timer_delete;
libc_hidden_proto (__timer_delete)
extern __typeof (timer_getoverrun) __timer_getoverrun;
libc_hidden_proto (__timer_getoverrun)

/* Type of timers in the kernel.  */
typedef int kernel_timer_t;

/* Internal representation of SIGEV_THREAD timer.  */
struct timer
{
  kernel_timer_t ktimerid;

  void (*thrfunc) (sigval_t);
  sigval_t sival;
  pthread_attr_t attr;

  /* Next element in list of active SIGEV_THREAD timers.  */
  struct timer *next;
};


/* For !SIGEV_THREAD, the resulting 'timer_t' is the returned kernel timer
   identifer (kernel_timer_t), while for SIGEV_THREAD it uses the fact malloc
   returns at least _Alignof (max_align_t) pointers plus that valid
   kernel_timer_t are always positive to set the MSB bit of the returned
   'timer_t' to indicate the timer handles a SIGEV_THREAD.  */

static inline timer_t
kernel_timer_to_timerid (kernel_timer_t ktimerid)
{
  return (timer_t) ((intptr_t) ktimerid);
}

static inline timer_t
timer_to_timerid (struct timer *ptr)
{
  return (timer_t) (INTPTR_MIN | (uintptr_t) ptr >> 1);
}

static inline bool
timer_is_sigev_thread (timer_t timerid)
{
  return (intptr_t) timerid < 0;
}

static inline struct timer *
timerid_to_timer (timer_t timerid)
{
  return (struct timer *)((uintptr_t) timerid << 1);
}

static inline kernel_timer_t
timerid_to_kernel_timer (timer_t timerid)
{
  if (timer_is_sigev_thread (timerid))
    return timerid_to_timer (timerid)->ktimerid;
  else
    return (kernel_timer_t) ((uintptr_t) timerid);
}

/* New targets use int instead of timer_t.  The difference only
   matters on 64-bit targets.  */
#include <timer_t_was_int_compat.h>

#if TIMER_T_WAS_INT_COMPAT
# define OLD_TIMER_MAX 256
extern timer_t __timer_compat_list[OLD_TIMER_MAX];
#endif

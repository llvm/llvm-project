/* Definitions for POSIX timer implementation on top of NPTL.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <signal.h>
#include <list.h>


/* Forward declaration.  */
struct timer_node;


/* Definitions for an internal thread of the POSIX timer implementation.  */
struct thread_node
{
  struct list_head links;
  pthread_attr_t attr;
  pthread_t id;
  unsigned int exists;
  struct list_head timer_queue;
  pthread_cond_t cond;
  struct timer_node *current_timer;
  pthread_t captured;
  clockid_t clock_id;
};


/* Internal representation of a timer.  */
struct timer_node
{
  struct list_head links;
  struct sigevent event;
  clockid_t clock;
  struct itimerspec value;
  struct timespec expirytime;
  pthread_attr_t attr;
  unsigned int abstime;
  unsigned int armed;
  enum {
    TIMER_FREE, TIMER_INUSE, TIMER_DELETED
  } inuse;
  struct thread_node *thread;
  pid_t creator_pid;
  int refcount;
  int overrun_count;
};


/* The limit is not published if we are compiled with kernel timer support.
   But we still compiled in this implementation with its limit unless built
   to require the kernel support.  */
#ifndef TIMER_MAX
# define TIMER_MAX 256
#endif

/* Static array with the structures for all the timers.  */
extern struct timer_node __timer_array[TIMER_MAX];

/* Global lock to protect operation on the lists.  */
extern pthread_mutex_t __timer_mutex;

/* Variable to protext initialization.  */
extern pthread_once_t __timer_init_once_control;

/* Nonzero if initialization of timer implementation failed.  */
extern int __timer_init_failed;

/* Node for the thread used to deliver signals.  */
extern struct thread_node __timer_signal_thread_rclk;


/* Return pointer to timer structure corresponding to ID.  */
#define timer_id2ptr(timerid) ((struct timer_node *) timerid)
#define timer_ptr2id(timerid) ((timer_t) timerid)

/* Check whether timer is valid; global mutex must be held. */
static inline int
timer_valid (struct timer_node *timer)
{
  return timer && timer->inuse == TIMER_INUSE;
}

/* Timer refcount functions; need global mutex. */
extern void __timer_dealloc (struct timer_node *timer);

static inline void
timer_addref (struct timer_node *timer)
{
  timer->refcount++;
}

static inline void
timer_delref (struct timer_node *timer)
{
  if (--timer->refcount == 0)
    __timer_dealloc (timer);
}

/* Timespec helper routines.  */
static inline int
__attribute ((always_inline))
timespec_compare (const struct timespec *left, const struct timespec *right)
{
  if (left->tv_sec < right->tv_sec)
    return -1;
  if (left->tv_sec > right->tv_sec)
    return 1;

  if (left->tv_nsec < right->tv_nsec)
    return -1;
  if (left->tv_nsec > right->tv_nsec)
    return 1;

  return 0;
}

static inline void
timespec_add (struct timespec *sum, const struct timespec *left,
	      const struct timespec *right)
{
  sum->tv_sec = left->tv_sec + right->tv_sec;
  sum->tv_nsec = left->tv_nsec + right->tv_nsec;

  if (sum->tv_nsec >= 1000000000)
    {
      ++sum->tv_sec;
      sum->tv_nsec -= 1000000000;
    }
}

static inline void
timespec_sub (struct timespec *diff, const struct timespec *left,
	      const struct timespec *right)
{
  diff->tv_sec = left->tv_sec - right->tv_sec;
  diff->tv_nsec = left->tv_nsec - right->tv_nsec;

  if (diff->tv_nsec < 0)
    {
      --diff->tv_sec;
      diff->tv_nsec += 1000000000;
    }
}


/* We need one of the list functions in the other modules.  */
static inline void
list_unlink_ip (struct list_head *list)
{
  struct list_head *lnext = list->next, *lprev = list->prev;

  lnext->prev = lprev;
  lprev->next = lnext;

  /* The suffix ip means idempotent; list_unlink_ip can be called
   * two or more times on the same node.
   */

  list->next = list;
  list->prev = list;
}


/* Functions in the helper file.  */
extern void __timer_mutex_cancel_handler (void *arg);
extern void __timer_init_once (void);
extern struct timer_node *__timer_alloc (void);
extern int __timer_thread_start (struct thread_node *thread);
extern struct thread_node *__timer_thread_find_matching (const pthread_attr_t *desired_attr, clockid_t);
extern struct thread_node *__timer_thread_alloc (const pthread_attr_t *desired_attr, clockid_t);
extern void __timer_thread_dealloc (struct thread_node *thread);
extern int __timer_thread_queue_timer (struct thread_node *thread,
				       struct timer_node *insert);
extern void __timer_thread_wakeup (struct thread_node *thread);

/* Helper code for POSIX timer implementation on NPTL.
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

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sysdep.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "posix-timer.h"
#include <timer_routines.h>

#ifndef DELAYTIMER_MAX
# define DELAYTIMER_MAX INT_MAX
#endif

/* Number of threads used.  */
#define THREAD_MAXNODES	16

/* Array containing the descriptors for the used threads.  */
static struct thread_node thread_array[THREAD_MAXNODES];

/* Static array with the structures for all the timers.  */
struct timer_node __timer_array[TIMER_MAX];

/* Global lock to protect operation on the lists.  */
pthread_mutex_t __timer_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Variable to protext initialization.  */
pthread_once_t __timer_init_once_control = PTHREAD_ONCE_INIT;

/* Nonzero if initialization of timer implementation failed.  */
int __timer_init_failed;

/* Node for the thread used to deliver signals.  */
struct thread_node __timer_signal_thread_rclk;

/* Lists to keep free and used timers and threads.  */
static struct list_head timer_free_list;
static struct list_head thread_free_list;
static struct list_head thread_active_list;


#ifdef __NR_rt_sigqueueinfo
extern int __syscall_rt_sigqueueinfo (int, int, siginfo_t *);
#endif


/* List handling functions.  */
static inline void
list_append (struct list_head *list, struct list_head *newp)
{
  newp->prev = list->prev;
  newp->next = list;
  list->prev->next = newp;
  list->prev = newp;
}

static inline void
list_insbefore (struct list_head *list, struct list_head *newp)
{
  list_append (list, newp);
}

/*
 * Like list_unlink_ip, except that calling it on a node that
 * is already unlinked is disastrous rather than a noop.
 */

static inline void
list_unlink (struct list_head *list)
{
  struct list_head *lnext = list->next, *lprev = list->prev;

  lnext->prev = lprev;
  lprev->next = lnext;
}

static inline struct list_head *
list_first (struct list_head *list)
{
  return list->next;
}

static inline struct list_head *
list_null (struct list_head *list)
{
  return list;
}

static inline struct list_head *
list_next (struct list_head *list)
{
  return list->next;
}

static inline int
list_isempty (struct list_head *list)
{
  return list->next == list;
}


/* Functions build on top of the list functions.  */
static inline struct thread_node *
thread_links2ptr (struct list_head *list)
{
  return (struct thread_node *) ((char *) list
				 - offsetof (struct thread_node, links));
}

static inline struct timer_node *
timer_links2ptr (struct list_head *list)
{
  return (struct timer_node *) ((char *) list
				- offsetof (struct timer_node, links));
}


/* Initialize a newly allocated thread structure.  */
static void
thread_init (struct thread_node *thread, const pthread_attr_t *attr, clockid_t clock_id)
{
  if (attr != NULL)
    thread->attr = *attr;
  else
    {
      pthread_attr_init (&thread->attr);
      pthread_attr_setdetachstate (&thread->attr, PTHREAD_CREATE_DETACHED);
    }

  thread->exists = 0;
  INIT_LIST_HEAD (&thread->timer_queue);
  pthread_cond_init (&thread->cond, 0);
  thread->current_timer = 0;
  thread->captured = pthread_self ();
  thread->clock_id = clock_id;
}


/* Initialize the global lists, and acquire global resources.  Error
   reporting is done by storing a non-zero value to the global variable
   timer_init_failed.  */
static void
init_module (void)
{
  int i;

  INIT_LIST_HEAD (&timer_free_list);
  INIT_LIST_HEAD (&thread_free_list);
  INIT_LIST_HEAD (&thread_active_list);

  for (i = 0; i < TIMER_MAX; ++i)
    {
      list_append (&timer_free_list, &__timer_array[i].links);
      __timer_array[i].inuse = TIMER_FREE;
    }

  for (i = 0; i < THREAD_MAXNODES; ++i)
    list_append (&thread_free_list, &thread_array[i].links);

  thread_init (&__timer_signal_thread_rclk, 0, CLOCK_REALTIME);
}


/* This is a handler executed in a child process after a fork()
   occurs.  It reinitializes the module, resetting all of the data
   structures to their initial state.  The mutex is initialized in
   case it was locked in the parent process.  */
static void
reinit_after_fork (void)
{
  init_module ();
  pthread_mutex_init (&__timer_mutex, 0);
}


/* Called once form pthread_once in timer_init. This initializes the
   module and ensures that reinit_after_fork will be executed in any
   child process.  */
void
__timer_init_once (void)
{
  init_module ();
  pthread_atfork (0, 0, reinit_after_fork);
}


/* Deinitialize a thread that is about to be deallocated.  */
static void
thread_deinit (struct thread_node *thread)
{
  assert (list_isempty (&thread->timer_queue));
  pthread_cond_destroy (&thread->cond);
}


/* Allocate a thread structure from the global free list.  Global
   mutex lock must be held by caller.  The thread is moved to
   the active list. */
struct thread_node *
__timer_thread_alloc (const pthread_attr_t *desired_attr, clockid_t clock_id)
{
  struct list_head *node = list_first (&thread_free_list);

  if (node != list_null (&thread_free_list))
    {
      struct thread_node *thread = thread_links2ptr (node);
      list_unlink (node);
      thread_init (thread, desired_attr, clock_id);
      list_append (&thread_active_list, node);
      return thread;
    }

  return 0;
}


/* Return a thread structure to the global free list.  Global lock
   must be held by caller.  */
void
__timer_thread_dealloc (struct thread_node *thread)
{
  thread_deinit (thread);
  list_unlink (&thread->links);
  list_append (&thread_free_list, &thread->links);
}


/* Each of our threads which terminates executes this cleanup
   handler. We never terminate threads ourselves; if a thread gets here
   it means that the evil application has killed it.  If the thread has
   timers, these require servicing and so we must hire a replacement
   thread right away.  We must also unblock another thread that may
   have been waiting for this thread to finish servicing a timer (see
   timer_delete()).  */

static void
thread_cleanup (void *val)
{
  if (val != NULL)
    {
      struct thread_node *thread = val;

      /* How did the signal thread get killed?  */
      assert (thread != &__timer_signal_thread_rclk);

      pthread_mutex_lock (&__timer_mutex);

      thread->exists = 0;

      /* We are no longer processing a timer event.  */
      thread->current_timer = 0;

      if (list_isempty (&thread->timer_queue))
	__timer_thread_dealloc (thread);
      else
	(void) __timer_thread_start (thread);

      pthread_mutex_unlock (&__timer_mutex);

      /* Unblock potentially blocked timer_delete().  */
      pthread_cond_broadcast (&thread->cond);
    }
}


/* Handle a timer which is supposed to go off now.  */
static void
thread_expire_timer (struct thread_node *self, struct timer_node *timer)
{
  self->current_timer = timer; /* Lets timer_delete know timer is running. */

  pthread_mutex_unlock (&__timer_mutex);

  switch (__builtin_expect (timer->event.sigev_notify, SIGEV_SIGNAL))
    {
    case SIGEV_NONE:
      break;

    case SIGEV_SIGNAL:
#ifdef __NR_rt_sigqueueinfo
      {
	siginfo_t info;

	/* First, clear the siginfo_t structure, so that we don't pass our
	   stack content to other tasks.  */
	memset (&info, 0, sizeof (siginfo_t));
	/* We must pass the information about the data in a siginfo_t
           value.  */
	info.si_signo = timer->event.sigev_signo;
	info.si_code = SI_TIMER;
	info.si_pid = timer->creator_pid;
	info.si_uid = getuid ();
	info.si_value = timer->event.sigev_value;

	INLINE_SYSCALL (rt_sigqueueinfo, 3, info.si_pid, info.si_signo, &info);
      }
#else
      if (pthread_kill (self->captured, timer->event.sigev_signo) != 0)
	{
	  if (pthread_kill (self->id, timer->event.sigev_signo) != 0)
	    abort ();
        }
#endif
      break;

    case SIGEV_THREAD:
      timer->event.sigev_notify_function (timer->event.sigev_value);
      break;

    default:
      assert (! "unknown event");
      break;
    }

  pthread_mutex_lock (&__timer_mutex);

  self->current_timer = 0;

  pthread_cond_broadcast (&self->cond);
}


/* Thread function; executed by each timer thread. The job of this
   function is to wait on the thread's timer queue and expire the
   timers in chronological order as close to their scheduled time as
   possible.  */
static void
__attribute__ ((noreturn))
thread_func (void *arg)
{
  struct thread_node *self = arg;

  /* Register cleanup handler, in case rogue application terminates
     this thread.  (This cannot happen to __timer_signal_thread, which
     doesn't invoke application callbacks). */

  pthread_cleanup_push (thread_cleanup, self);

  pthread_mutex_lock (&__timer_mutex);

  while (1)
    {
      struct list_head *first;
      struct timer_node *timer = NULL;

      /* While the timer queue is not empty, inspect the first node.  */
      first = list_first (&self->timer_queue);
      if (first != list_null (&self->timer_queue))
	{
	  struct timespec now;

	  timer = timer_links2ptr (first);

	  /* This assumes that the elements of the list of one thread
	     are all for the same clock.  */
	  __clock_gettime (timer->clock, &now);

	  while (1)
	    {
	      /* If the timer is due or overdue, remove it from the queue.
		 If it's a periodic timer, re-compute its new time and
		 requeue it.  Either way, perform the timer expiry. */
	      if (timespec_compare (&now, &timer->expirytime) < 0)
		break;

	      list_unlink_ip (first);

	      if (__builtin_expect (timer->value.it_interval.tv_sec, 0) != 0
		  || timer->value.it_interval.tv_nsec != 0)
		{
		  timer->overrun_count = 0;
		  timespec_add (&timer->expirytime, &timer->expirytime,
				&timer->value.it_interval);
		  while (timespec_compare (&timer->expirytime, &now) < 0)
		    {
		      timespec_add (&timer->expirytime, &timer->expirytime,
				    &timer->value.it_interval);
		      if (timer->overrun_count < DELAYTIMER_MAX)
			++timer->overrun_count;
		    }
		  __timer_thread_queue_timer (self, timer);
		}

	      thread_expire_timer (self, timer);

	      first = list_first (&self->timer_queue);
	      if (first == list_null (&self->timer_queue))
		break;

	      timer = timer_links2ptr (first);
	    }
	}

      /* If the queue is not empty, wait until the expiry time of the
	 first node.  Otherwise wait indefinitely.  Insertions at the
	 head of the queue must wake up the thread by broadcasting
	 this condition variable.  */
      if (timer != NULL)
	pthread_cond_timedwait (&self->cond, &__timer_mutex,
				&timer->expirytime);
      else
	pthread_cond_wait (&self->cond, &__timer_mutex);
    }
  /* This macro will never be executed since the while loop loops
     forever - but we have to add it for proper nesting.  */
  pthread_cleanup_pop (1);
}


/* Enqueue a timer in wakeup order in the thread's timer queue.
   Returns 1 if the timer was inserted at the head of the queue,
   causing the queue's next wakeup time to change. */

int
__timer_thread_queue_timer (struct thread_node *thread,
			    struct timer_node *insert)
{
  struct list_head *iter;
  int athead = 1;

  for (iter = list_first (&thread->timer_queue);
       iter != list_null (&thread->timer_queue);
        iter = list_next (iter))
    {
      struct timer_node *timer = timer_links2ptr (iter);

      if (timespec_compare (&insert->expirytime, &timer->expirytime) < 0)
	  break;
      athead = 0;
    }

  list_insbefore (iter, &insert->links);
  return athead;
}


/* Start a thread and associate it with the given thread node.  Global
   lock must be held by caller.  */
int
__timer_thread_start (struct thread_node *thread)
{
  int retval = 1;
  sigset_t set, oset;

  assert (!thread->exists);
  thread->exists = 1;

  sigfillset (&set);
  pthread_sigmask (SIG_SETMASK, &set, &oset);

  if (pthread_create (&thread->id, &thread->attr,
		      (void *(*) (void *)) thread_func, thread) != 0)
    {
      thread->exists = 0;
      retval = -1;
    }

  pthread_sigmask (SIG_SETMASK, &oset, NULL);

  return retval;
}


void
__timer_thread_wakeup (struct thread_node *thread)
{
  pthread_cond_broadcast (&thread->cond);
}



/* Search the list of active threads and find one which has matching
   attributes.  Global mutex lock must be held by caller.  */
struct thread_node *
__timer_thread_find_matching (const pthread_attr_t *desired_attr,
			      clockid_t desired_clock_id)
{
  struct list_head *iter = list_first (&thread_active_list);

  while (iter != list_null (&thread_active_list))
    {
      struct thread_node *candidate = thread_links2ptr (iter);

      if (thread_attr_compare (desired_attr, &candidate->attr)
	  && desired_clock_id == candidate->clock_id)
	return candidate;

      iter = list_next (iter);
    }

  return NULL;
}


/* Grab a free timer structure from the global free list.  The global
   lock must be held by the caller.  */
struct timer_node *
__timer_alloc (void)
{
  struct list_head *node = list_first (&timer_free_list);

  if (node != list_null (&timer_free_list))
    {
      struct timer_node *timer = timer_links2ptr (node);
      list_unlink_ip (node);
      timer->inuse = TIMER_INUSE;
      timer->refcount = 1;
      return timer;
    }

  return NULL;
}


/* Return a timer structure to the global free list.  The global lock
   must be held by the caller.  */
void
__timer_dealloc (struct timer_node *timer)
{
  assert (timer->refcount == 0);
  timer->thread = NULL;	/* Break association between timer and thread.  */
  timer->inuse = TIMER_FREE;
  list_append (&timer_free_list, &timer->links);
}


/* Thread cancellation handler which unlocks a mutex.  */
void
__timer_mutex_cancel_handler (void *arg)
{
  pthread_mutex_unlock (arg);
}

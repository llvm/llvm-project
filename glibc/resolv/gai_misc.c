/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

#include <gai_misc.h>

#if !PTHREAD_IN_LIBC
/* The available function names differ outside of libc.  (In libc, we
   need to use hidden aliases to avoid the PLT.)  */
#define __pthread_attr_init pthread_attr_init
#define __pthread_attr_setdetachstate pthread_attr_setdetachstate
#define __pthread_cond_signal pthread_cond_signal
#define __pthread_cond_timedwait pthread_cond_timedwait
#define __pthread_create pthread_create
#define __pthread_exit pthread_exit
#endif

#ifndef gai_create_helper_thread
# define gai_create_helper_thread __gai_create_helper_thread

extern inline int
__gai_create_helper_thread (pthread_t *threadp, void *(*tf) (void *),
			    void *arg)
{
  pthread_attr_t attr;

  /* Make sure the thread is created detached.  */
  __pthread_attr_init (&attr);
  __pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  int ret = __pthread_create (threadp, &attr, tf, arg);

  (void) __pthread_attr_destroy (&attr);
  return ret;
}
#endif


/* Pool of request list entries.  */
static struct requestlist **pool;

/* Number of total and allocated pool entries.  */
static size_t pool_max_size;
static size_t pool_size;

/* We implement a two dimensional array but allocate each row separately.
   The macro below determines how many entries should be used per row.
   It should better be a power of two.  */
#define ENTRIES_PER_ROW	32

/* How many rows we allocate at once.  */
#define ROWS_STEP	8

/* List of available entries.  */
static struct requestlist *freelist;

/* Structure list of all currently processed requests.  */
static struct requestlist *requests;
static struct requestlist *requests_tail;

/* Number of threads currently running.  */
static int nthreads;

/* Number of threads waiting for work to arrive. */
static int idle_thread_count;


/* These are the values used for optimization.  We will probably
   create a funcion to set these values.  */
static struct gaiinit optim =
{
  20,	/* int gai_threads;	Maximal number of threads.  */
  64,	/* int gai_num;		Number of expected simultanious requests. */
  0,
  0,
  0,
  0,
  1,
  0
};


/* Since the list is global we need a mutex protecting it.  */
pthread_mutex_t __gai_requests_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

/* When you add a request to the list and there are idle threads present,
   you signal this condition variable. When a thread finishes work, it waits
   on this condition variable for a time before it actually exits. */
pthread_cond_t __gai_new_request_notification = PTHREAD_COND_INITIALIZER;


/* Functions to handle request list pool.  */
static struct requestlist *
get_elem (void)
{
  struct requestlist *result;

  if (freelist == NULL)
    {
      struct requestlist *new_row;
      int cnt;

      if (pool_size + 1 >= pool_max_size)
	{
	  size_t new_max_size = pool_max_size + ROWS_STEP;
	  struct requestlist **new_tab;

	  new_tab = (struct requestlist **)
	    realloc (pool, new_max_size * sizeof (struct requestlist *));

	  if (new_tab == NULL)
	    return NULL;

	  pool_max_size = new_max_size;
	  pool = new_tab;
	}

      /* Allocate the new row.  */
      cnt = pool_size == 0 ? optim.gai_num : ENTRIES_PER_ROW;
      new_row = (struct requestlist *) calloc (cnt,
					       sizeof (struct requestlist));
      if (new_row == NULL)
	return NULL;

      pool[pool_size++] = new_row;

      /* Put all the new entries in the freelist.  */
      do
	{
	  new_row->next = freelist;
	  freelist = new_row++;
	}
      while (--cnt > 0);
    }

  result = freelist;
  freelist = freelist->next;

  return result;
}


struct requestlist *
__gai_find_request (const struct gaicb *gaicbp)
{
  struct requestlist *runp;

  runp = requests;
  while (runp != NULL)
    if (runp->gaicbp == gaicbp)
      return runp;
    else
      runp = runp->next;

  return NULL;
}


int
__gai_remove_request (struct gaicb *gaicbp)
{
  struct requestlist *runp;
  struct requestlist *lastp;

  runp = requests;
  lastp = NULL;
  while (runp != NULL)
    if (runp->gaicbp == gaicbp)
      break;
    else
      {
	lastp = runp;
	runp = runp->next;
      }

  if (runp == NULL)
    /* Not known.  */
    return -1;
  if (runp->running != 0)
    /* Currently handled.  */
    return 1;

  /* Dequeue the request.  */
  if (lastp == NULL)
    requests = runp->next;
  else
    lastp->next = runp->next;
  if (runp == requests_tail)
    requests_tail = lastp;

  return 0;
}


/* The thread handler.  */
static void *handle_requests (void *arg);


/* The main function of the async I/O handling.  It enqueues requests
   and if necessary starts and handles threads.  */
struct requestlist *
__gai_enqueue_request (struct gaicb *gaicbp)
{
  struct requestlist *newp;
  struct requestlist *lastp;

  /* Get the mutex.  */
  __pthread_mutex_lock (&__gai_requests_mutex);

  /* Get a new element for the waiting list.  */
  newp = get_elem ();
  if (newp == NULL)
    {
      __pthread_mutex_unlock (&__gai_requests_mutex);
      __set_errno (EAGAIN);
      return NULL;
    }
  newp->running = 0;
  newp->gaicbp = gaicbp;
  newp->waiting = NULL;
  newp->next = NULL;

  lastp = requests_tail;
  if (requests_tail == NULL)
    requests = requests_tail = newp;
  else
    {
      requests_tail->next = newp;
      requests_tail = newp;
    }

  gaicbp->__return = EAI_INPROGRESS;

  /* See if we need to and are able to create a thread.  */
  if (nthreads < optim.gai_threads && idle_thread_count == 0)
    {
      pthread_t thid;

      newp->running = 1;

      /* Now try to start a thread.  */
      if (gai_create_helper_thread (&thid, handle_requests, newp) == 0)
	/* We managed to enqueue the request.  All errors which can
	   happen now can be recognized by calls to `gai_error'.  */
	++nthreads;
      else
	{
	  if (nthreads == 0)
	    {
	      /* We cannot create a thread in the moment and there is
		 also no thread running.  This is a problem.  `errno' is
		 set to EAGAIN if this is only a temporary problem.  */
	      assert (requests == newp || lastp->next == newp);
	      if (lastp != NULL)
		lastp->next = NULL;
	      else
		requests = NULL;
	      requests_tail = lastp;

	      newp->next = freelist;
	      freelist = newp;

	      newp = NULL;
	    }
	  else
	    /* We are not handling the request after all.  */
	    newp->running = 0;
	}
    }

  /* Enqueue the request in the request queue.  */
  if (newp != NULL)
    {
      /* If there is a thread waiting for work, then let it know that we
	 have just given it something to do. */
      if (idle_thread_count > 0)
	__pthread_cond_signal (&__gai_new_request_notification);
    }

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__gai_requests_mutex);

  return newp;
}


static void *
__attribute__ ((noreturn))
handle_requests (void *arg)
{
  struct requestlist *runp = (struct requestlist *) arg;

  do
    {
      /* If runp is NULL, then we were created to service the work queue
	 in general, not to handle any particular request. In that case we
	 skip the "do work" stuff on the first pass, and go directly to the
	 "get work off the work queue" part of this loop, which is near the
	 end. */
      if (runp == NULL)
	__pthread_mutex_lock (&__gai_requests_mutex);
      else
	{
	  /* Make the request.  */
	  struct gaicb *req = runp->gaicbp;
	  struct requestlist *srchp;
	  struct requestlist *lastp;

	  req->__return = getaddrinfo (req->ar_name, req->ar_service,
				       req->ar_request, &req->ar_result);

	  /* Get the mutex.  */
	  __pthread_mutex_lock (&__gai_requests_mutex);

	  /* Send the signal to notify about finished processing of the
	     request.  */
	  __gai_notify (runp);

	  /* Now dequeue the current request.  */
	  lastp = NULL;
	  srchp = requests;
	  while (srchp != runp)
	    {
	      lastp = srchp;
	      srchp = srchp->next;
	    }
	  assert (runp->running == 1);

	  if (requests_tail == runp)
	    requests_tail = lastp;
	  if (lastp == NULL)
	    requests = requests->next;
	  else
	    lastp->next = runp->next;

	  /* Free the old element.  */
	  runp->next = freelist;
	  freelist = runp;
	}

      runp = requests;
      while (runp != NULL && runp->running != 0)
	runp = runp->next;

      /* If the runlist is empty, then we sleep for a while, waiting for
	 something to arrive in it. */
      if (runp == NULL && optim.gai_idle_time >= 0)
	{
	  struct timespec now;
	  struct timespec wakeup_time;

	  ++idle_thread_count;
          __clock_gettime (CLOCK_REALTIME, &now);
	  wakeup_time.tv_sec = now.tv_sec + optim.gai_idle_time;
	  wakeup_time.tv_nsec = now.tv_nsec;
	  if (wakeup_time.tv_nsec >= 1000000000)
	    {
	      wakeup_time.tv_nsec -= 1000000000;
	      ++wakeup_time.tv_sec;
	    }
	  __pthread_cond_timedwait (&__gai_new_request_notification,
				    &__gai_requests_mutex, &wakeup_time);
	  --idle_thread_count;
	  runp = requests;
	  while (runp != NULL && runp->running != 0)
	    runp = runp->next;
	}

      if (runp == NULL)
	--nthreads;
      else
	{
	  /* Mark the request as being worked on.  */
	  assert (runp->running == 0);
	  runp->running = 1;

	  /* If we have a request to process, and there's still another in
	     the run list, then we need to either wake up or create a new
	     thread to service the request that is still in the run list. */
	  if (requests != NULL)
	    {
	      /* There are at least two items in the work queue to work on.
		 If there are other idle threads, then we should wake them
		 up for these other work elements; otherwise, we should try
		 to create a new thread. */
	      if (idle_thread_count > 0)
		__pthread_cond_signal (&__gai_new_request_notification);
	      else if (nthreads < optim.gai_threads)
		{
		  pthread_t thid;
		  pthread_attr_t attr;

		  /* Make sure the thread is created detached.  */
		  __pthread_attr_init (&attr);
		  __pthread_attr_setdetachstate (&attr,
						 PTHREAD_CREATE_DETACHED);

		  /* Now try to start a thread. If we fail, no big deal,
		     because we know that there is at least one thread (us)
		     that is working on lookup operations. */
		  if (__pthread_create (&thid, &attr, handle_requests, NULL)
		      == 0)
		    ++nthreads;
		}
	    }
	}

      /* Release the mutex.  */
      __pthread_mutex_unlock (&__gai_requests_mutex);
    }
  while (runp != NULL);

  __pthread_exit (NULL);
}


/* Free allocated resources.  */
libc_freeres_fn (free_res)
{
  size_t row;

  for (row = 0; row < pool_max_size; ++row)
    free (pool[row]);

  free (pool);
}

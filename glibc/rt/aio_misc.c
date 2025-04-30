/* Handle general operations.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <aio.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <pthreadP.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <aio_misc.h>

#if !PTHREAD_IN_LIBC
/* The available function names differ outside of libc.  (In libc, we
   need to use hidden aliases to avoid the PLT.)  */
# define __pread __libc_pread
# define __pthread_attr_destroy pthread_attr_destroy
# define __pthread_attr_init pthread_attr_init
# define __pthread_attr_setdetachstate pthread_attr_setdetachstate
# define __pthread_cond_signal pthread_cond_signal
# define __pthread_cond_timedwait pthread_cond_timedwait
# define __pthread_getschedparam pthread_getschedparam
# define __pthread_setschedparam pthread_setschedparam
# define __pwrite __libc_pwrite
#endif

#ifndef aio_create_helper_thread
# define aio_create_helper_thread __aio_create_helper_thread

extern inline int
__aio_create_helper_thread (pthread_t *threadp, void *(*tf) (void *), void *arg)
{
  pthread_attr_t attr;

  /* Make sure the thread is created detached.  */
  __pthread_attr_init (&attr);
  __pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  int ret = __pthread_create (threadp, &attr, tf, arg);

  __pthread_attr_destroy (&attr);
  return ret;
}
#endif

static void add_request_to_runlist (struct requestlist *newrequest);

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

/* List of request waiting to be processed.  */
static struct requestlist *runlist;

/* Structure list of all currently processed requests.  */
static struct requestlist *requests;

/* Number of threads currently running.  */
static int nthreads;

/* Number of threads waiting for work to arrive. */
static int idle_thread_count;


/* These are the values used to optimize the use of AIO.  The user can
   overwrite them by using the `aio_init' function.  */
static struct aioinit optim =
{
  20,	/* int aio_threads;	Maximal number of threads.  */
  64,	/* int aio_num;		Number of expected simultaneous requests. */
  0,
  0,
  0,
  0,
  1,
  0
};


/* Since the list is global we need a mutex protecting it.  */
pthread_mutex_t __aio_requests_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

/* When you add a request to the list and there are idle threads present,
   you signal this condition variable. When a thread finishes work, it waits
   on this condition variable for a time before it actually exits. */
pthread_cond_t __aio_new_request_notification = PTHREAD_COND_INITIALIZER;


/* Functions to handle request list pool.  */
static struct requestlist *
get_elem (void)
{
  struct requestlist *result;

  if (freelist == NULL)
    {
      struct requestlist *new_row;
      int cnt;

      assert (sizeof (struct aiocb) == sizeof (struct aiocb64));

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
      cnt = pool_size == 0 ? optim.aio_num : ENTRIES_PER_ROW;
      new_row = (struct requestlist *) calloc (cnt,
					       sizeof (struct requestlist));
      if (new_row == NULL)
	return NULL;

      pool[pool_size++] = new_row;

      /* Put all the new entries in the freelist.  */
      do
	{
	  new_row->next_prio = freelist;
	  freelist = new_row++;
	}
      while (--cnt > 0);
    }

  result = freelist;
  freelist = freelist->next_prio;

  return result;
}


void
__aio_free_request (struct requestlist *elem)
{
  elem->running = no;
  elem->next_prio = freelist;
  freelist = elem;
}


struct requestlist *
__aio_find_req (aiocb_union *elem)
{
  struct requestlist *runp = requests;
  int fildes = elem->aiocb.aio_fildes;

  while (runp != NULL && runp->aiocbp->aiocb.aio_fildes < fildes)
    runp = runp->next_fd;

  if (runp != NULL)
    {
      if (runp->aiocbp->aiocb.aio_fildes != fildes)
	runp = NULL;
      else
	while (runp != NULL && runp->aiocbp != elem)
	  runp = runp->next_prio;
    }

  return runp;
}


struct requestlist *
__aio_find_req_fd (int fildes)
{
  struct requestlist *runp = requests;

  while (runp != NULL && runp->aiocbp->aiocb.aio_fildes < fildes)
    runp = runp->next_fd;

  return (runp != NULL && runp->aiocbp->aiocb.aio_fildes == fildes
	  ? runp : NULL);
}


void
__aio_remove_request (struct requestlist *last, struct requestlist *req,
		      int all)
{
  assert (req->running == yes || req->running == queued
	  || req->running == done);

  if (last != NULL)
    last->next_prio = all ? NULL : req->next_prio;
  else
    {
      if (all || req->next_prio == NULL)
	{
	  if (req->last_fd != NULL)
	    req->last_fd->next_fd = req->next_fd;
	  else
	    requests = req->next_fd;
	  if (req->next_fd != NULL)
	    req->next_fd->last_fd = req->last_fd;
	}
      else
	{
	  if (req->last_fd != NULL)
	    req->last_fd->next_fd = req->next_prio;
	  else
	    requests = req->next_prio;

	  if (req->next_fd != NULL)
	    req->next_fd->last_fd = req->next_prio;

	  req->next_prio->last_fd = req->last_fd;
	  req->next_prio->next_fd = req->next_fd;

	  /* Mark this entry as runnable.  */
	  req->next_prio->running = yes;
	}

      if (req->running == yes)
	{
	  struct requestlist *runp = runlist;

	  last = NULL;
	  while (runp != NULL)
	    {
	      if (runp == req)
		{
		  if (last == NULL)
		    runlist = runp->next_run;
		  else
		    last->next_run = runp->next_run;
		  break;
		}
	      last = runp;
	      runp = runp->next_run;
	    }
	}
    }
}


/* The thread handler.  */
static void *handle_fildes_io (void *arg);


/* User optimization.  */
void
__aio_init (const struct aioinit *init)
{
  /* Get the mutex.  */
  __pthread_mutex_lock (&__aio_requests_mutex);

  /* Only allow writing new values if the table is not yet allocated.  */
  if (pool == NULL)
    {
      optim.aio_threads = init->aio_threads < 1 ? 1 : init->aio_threads;
      assert (powerof2 (ENTRIES_PER_ROW));
      optim.aio_num = (init->aio_num < ENTRIES_PER_ROW
		       ? ENTRIES_PER_ROW
		       : init->aio_num & ~(ENTRIES_PER_ROW - 1));
    }

  if (init->aio_idle_time != 0)
    optim.aio_idle_time = init->aio_idle_time;

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__aio_requests_mutex);
}


/* The main function of the async I/O handling.  It enqueues requests
   and if necessary starts and handles threads.  */
struct requestlist *
__aio_enqueue_request (aiocb_union *aiocbp, int operation)
{
  int result = 0;
  int policy, prio;
  struct sched_param param;
  struct requestlist *last, *runp, *newp;
  int running = no;

  if (operation == LIO_SYNC || operation == LIO_DSYNC)
    aiocbp->aiocb.aio_reqprio = 0;
  else if (aiocbp->aiocb.aio_reqprio < 0
#ifdef AIO_PRIO_DELTA_MAX
	   || aiocbp->aiocb.aio_reqprio > AIO_PRIO_DELTA_MAX
#endif
	   )
    {
      /* Invalid priority value.  */
      __set_errno (EINVAL);
      aiocbp->aiocb.__error_code = EINVAL;
      aiocbp->aiocb.__return_value = -1;
      return NULL;
    }

  /* Compute priority for this request.  */
  __pthread_getschedparam (__pthread_self (), &policy, &param);
  prio = param.sched_priority - aiocbp->aiocb.aio_reqprio;

  /* Get the mutex.  */
  __pthread_mutex_lock (&__aio_requests_mutex);

  last = NULL;
  runp = requests;
  /* First look whether the current file descriptor is currently
     worked with.  */
  while (runp != NULL
	 && runp->aiocbp->aiocb.aio_fildes < aiocbp->aiocb.aio_fildes)
    {
      last = runp;
      runp = runp->next_fd;
    }

  /* Get a new element for the waiting list.  */
  newp = get_elem ();
  if (newp == NULL)
    {
      __pthread_mutex_unlock (&__aio_requests_mutex);
      __set_errno (EAGAIN);
      return NULL;
    }
  newp->aiocbp = aiocbp;
  newp->waiting = NULL;

  aiocbp->aiocb.__abs_prio = prio;
  aiocbp->aiocb.__policy = policy;
  aiocbp->aiocb.aio_lio_opcode = operation;
  aiocbp->aiocb.__error_code = EINPROGRESS;
  aiocbp->aiocb.__return_value = 0;

  if (runp != NULL
      && runp->aiocbp->aiocb.aio_fildes == aiocbp->aiocb.aio_fildes)
    {
      /* The current file descriptor is worked on.  It makes no sense
	 to start another thread since this new thread would fight
	 with the running thread for the resources.  But we also cannot
	 say that the thread processing this desriptor shall immediately
	 after finishing the current job process this request if there
	 are other threads in the running queue which have a higher
	 priority.  */

      /* Simply enqueue it after the running one according to the
	 priority.  */
      last = NULL;
      while (runp->next_prio != NULL
	     && runp->next_prio->aiocbp->aiocb.__abs_prio >= prio)
	{
	  last = runp;
	  runp = runp->next_prio;
	}

      newp->next_prio = runp->next_prio;
      runp->next_prio = newp;

      running = queued;
    }
  else
    {
      running = yes;
      /* Enqueue this request for a new descriptor.  */
      if (last == NULL)
	{
	  newp->last_fd = NULL;
	  newp->next_fd = requests;
	  if (requests != NULL)
	    requests->last_fd = newp;
	  requests = newp;
	}
      else
	{
	  newp->next_fd = last->next_fd;
	  newp->last_fd = last;
	  last->next_fd = newp;
	  if (newp->next_fd != NULL)
	    newp->next_fd->last_fd = newp;
	}

      newp->next_prio = NULL;
      last = NULL;
    }

  if (running == yes)
    {
      /* We try to create a new thread for this file descriptor.  The
	 function which gets called will handle all available requests
	 for this descriptor and when all are processed it will
	 terminate.

	 If no new thread can be created or if the specified limit of
	 threads for AIO is reached we queue the request.  */

      /* See if we need to and are able to create a thread.  */
      if (nthreads < optim.aio_threads && idle_thread_count == 0)
	{
	  pthread_t thid;

	  running = newp->running = allocated;

	  /* Now try to start a thread.  */
	  result = aio_create_helper_thread (&thid, handle_fildes_io, newp);
	  if (result == 0)
	    /* We managed to enqueue the request.  All errors which can
	       happen now can be recognized by calls to `aio_return' and
	       `aio_error'.  */
	    ++nthreads;
	  else
	    {
	      /* Reset the running flag.  The new request is not running.  */
	      running = newp->running = yes;

	      if (nthreads == 0)
		{
		  /* We cannot create a thread in the moment and there is
		     also no thread running.  This is a problem.  `errno' is
		     set to EAGAIN if this is only a temporary problem.  */
		  __aio_remove_request (last, newp, 0);
		}
	      else
		result = 0;
	    }
	}
    }

  /* Enqueue the request in the run queue if it is not yet running.  */
  if (running == yes && result == 0)
    {
      add_request_to_runlist (newp);

      /* If there is a thread waiting for work, then let it know that we
	 have just given it something to do. */
      if (idle_thread_count > 0)
	__pthread_cond_signal (&__aio_new_request_notification);
    }

  if (result == 0)
    newp->running = running;
  else
    {
      /* Something went wrong.  */
      __aio_free_request (newp);
      aiocbp->aiocb.__error_code = result;
      __set_errno (result);
      newp = NULL;
    }

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__aio_requests_mutex);

  return newp;
}


static void *
handle_fildes_io (void *arg)
{
  pthread_t self = __pthread_self ();
  struct sched_param param;
  struct requestlist *runp = (struct requestlist *) arg;
  aiocb_union *aiocbp;
  int policy;
  int fildes;

  __pthread_getschedparam (self, &policy, &param);

  do
    {
      /* If runp is NULL, then we were created to service the work queue
	 in general, not to handle any particular request. In that case we
	 skip the "do work" stuff on the first pass, and go directly to the
	 "get work off the work queue" part of this loop, which is near the
	 end. */
      if (runp == NULL)
	__pthread_mutex_lock (&__aio_requests_mutex);
      else
	{
	  /* Hopefully this request is marked as running.  */
	  assert (runp->running == allocated);

	  /* Update our variables.  */
	  aiocbp = runp->aiocbp;
	  fildes = aiocbp->aiocb.aio_fildes;

	  /* Change the priority to the requested value (if necessary).  */
	  if (aiocbp->aiocb.__abs_prio != param.sched_priority
	      || aiocbp->aiocb.__policy != policy)
	    {
	      param.sched_priority = aiocbp->aiocb.__abs_prio;
	      policy = aiocbp->aiocb.__policy;
	      __pthread_setschedparam (self, policy, &param);
	    }

	  /* Process request pointed to by RUNP.  We must not be disturbed
	     by signals.  */
	  if ((aiocbp->aiocb.aio_lio_opcode & 127) == LIO_READ)
	    {
	      if (sizeof (off_t) != sizeof (off64_t)
		  && aiocbp->aiocb.aio_lio_opcode & 128)
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (__pread64 (fildes, (void *)
						 aiocbp->aiocb64.aio_buf,
						 aiocbp->aiocb64.aio_nbytes,
						 aiocbp->aiocb64.aio_offset));
	      else
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (__pread (fildes,
					       (void *)
					       aiocbp->aiocb.aio_buf,
					       aiocbp->aiocb.aio_nbytes,
					       aiocbp->aiocb.aio_offset));

	      if (aiocbp->aiocb.__return_value == -1 && errno == ESPIPE)
		/* The Linux kernel is different from others.  It returns
		   ESPIPE if using pread on a socket.  Other platforms
		   simply ignore the offset parameter and behave like
		   read.  */
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (read (fildes,
					    (void *) aiocbp->aiocb64.aio_buf,
					    aiocbp->aiocb64.aio_nbytes));
	    }
	  else if ((aiocbp->aiocb.aio_lio_opcode & 127) == LIO_WRITE)
	    {
	      if (sizeof (off_t) != sizeof (off64_t)
		  && aiocbp->aiocb.aio_lio_opcode & 128)
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (__pwrite64 (fildes, (const void *)
						  aiocbp->aiocb64.aio_buf,
						  aiocbp->aiocb64.aio_nbytes,
						  aiocbp->aiocb64.aio_offset));
	      else
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (__pwrite (fildes, (const void *)
						aiocbp->aiocb.aio_buf,
						aiocbp->aiocb.aio_nbytes,
						aiocbp->aiocb.aio_offset));

	      if (aiocbp->aiocb.__return_value == -1 && errno == ESPIPE)
		/* The Linux kernel is different from others.  It returns
		   ESPIPE if using pwrite on a socket.  Other platforms
		   simply ignore the offset parameter and behave like
		   write.  */
		aiocbp->aiocb.__return_value =
		  TEMP_FAILURE_RETRY (write (fildes,
					     (void *) aiocbp->aiocb64.aio_buf,
					     aiocbp->aiocb64.aio_nbytes));
	    }
	  else if (aiocbp->aiocb.aio_lio_opcode == LIO_DSYNC)
	    aiocbp->aiocb.__return_value =
	      TEMP_FAILURE_RETRY (fdatasync (fildes));
	  else if (aiocbp->aiocb.aio_lio_opcode == LIO_SYNC)
	    aiocbp->aiocb.__return_value =
	      TEMP_FAILURE_RETRY (fsync (fildes));
	  else
	    {
	      /* This is an invalid opcode.  */
	      aiocbp->aiocb.__return_value = -1;
	      __set_errno (EINVAL);
	    }

	  /* Get the mutex.  */
	  __pthread_mutex_lock (&__aio_requests_mutex);

	  if (aiocbp->aiocb.__return_value == -1)
	    aiocbp->aiocb.__error_code = errno;
	  else
	    aiocbp->aiocb.__error_code = 0;

	  /* Send the signal to notify about finished processing of the
	     request.  */
	  __aio_notify (runp);

	  /* For debugging purposes we reset the running flag of the
	     finished request.  */
	  assert (runp->running == allocated);
	  runp->running = done;

	  /* Now dequeue the current request.  */
	  __aio_remove_request (NULL, runp, 0);
	  if (runp->next_prio != NULL)
	    add_request_to_runlist (runp->next_prio);

	  /* Free the old element.  */
	  __aio_free_request (runp);
	}

      runp = runlist;

      /* If the runlist is empty, then we sleep for a while, waiting for
	 something to arrive in it. */
      if (runp == NULL && optim.aio_idle_time >= 0)
	{
	  struct timespec now;
	  struct timespec wakeup_time;

	  ++idle_thread_count;
	  __clock_gettime (CLOCK_REALTIME, &now);
	  wakeup_time.tv_sec = now.tv_sec + optim.aio_idle_time;
	  wakeup_time.tv_nsec = now.tv_nsec;
	  if (wakeup_time.tv_nsec >= 1000000000)
	    {
	      wakeup_time.tv_nsec -= 1000000000;
	      ++wakeup_time.tv_sec;
	    }
	  __pthread_cond_timedwait (&__aio_new_request_notification,
				    &__aio_requests_mutex,
				    &wakeup_time);
	  --idle_thread_count;
	  runp = runlist;
	}

      if (runp == NULL)
	--nthreads;
      else
	{
	  assert (runp->running == yes);
	  runp->running = allocated;
	  runlist = runp->next_run;

	  /* If we have a request to process, and there's still another in
	     the run list, then we need to either wake up or create a new
	     thread to service the request that is still in the run list. */
	  if (runlist != NULL)
	    {
	      /* There are at least two items in the work queue to work on.
		 If there are other idle threads, then we should wake them
		 up for these other work elements; otherwise, we should try
		 to create a new thread. */
	      if (idle_thread_count > 0)
		__pthread_cond_signal (&__aio_new_request_notification);
	      else if (nthreads < optim.aio_threads)
		{
		  pthread_t thid;
		  pthread_attr_t attr;

		  /* Make sure the thread is created detached.  */
		  __pthread_attr_init (&attr);
		  __pthread_attr_setdetachstate (&attr,
						 PTHREAD_CREATE_DETACHED);

		  /* Now try to start a thread. If we fail, no big deal,
		     because we know that there is at least one thread (us)
		     that is working on AIO operations. */
		  if (__pthread_create (&thid, &attr, handle_fildes_io, NULL)
		      == 0)
		    ++nthreads;
		}
	    }
	}

      /* Release the mutex.  */
      __pthread_mutex_unlock (&__aio_requests_mutex);
    }
  while (runp != NULL);

  return NULL;
}


/* Free allocated resources.  */
libc_freeres_fn (free_res)
{
  size_t row;

  for (row = 0; row < pool_max_size; ++row)
    free (pool[row]);

  free (pool);
}


/* Add newrequest to the runlist. The __abs_prio flag of newrequest must
   be correctly set to do this. Also, you had better set newrequest's
   "running" flag to "yes" before you release your lock or you'll throw an
   assertion. */
static void
add_request_to_runlist (struct requestlist *newrequest)
{
  int prio = newrequest->aiocbp->aiocb.__abs_prio;
  struct requestlist *runp;

  if (runlist == NULL || runlist->aiocbp->aiocb.__abs_prio < prio)
    {
      newrequest->next_run = runlist;
      runlist = newrequest;
    }
  else
    {
      runp = runlist;

      while (runp->next_run != NULL
	     && runp->next_run->aiocbp->aiocb.__abs_prio >= prio)
	runp = runp->next_run;

      newrequest->next_run = runp->next_run;
      runp->next_run = newrequest;
    }
}

#if PTHREAD_IN_LIBC
versioned_symbol (libc, __aio_init, aio_init, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
compat_symbol (librt, __aio_init, aio_init, GLIBC_2_1);
# endif
#else /* !PTHREAD_IN_LIBC */
weak_alias (__aio_init, aio_init)
#endif /* !PTHREAD_IN_LIBC */

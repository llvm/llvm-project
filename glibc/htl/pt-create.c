/* Thread creation.
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

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <resolv.h>

#include <atomic.h>
#include <hurd/resource.h>
#include <sys/single_threaded.h>

#include <pt-internal.h>
#include <pthreadP.h>

#if IS_IN (libpthread)
# include <ctype.h>
#endif
#ifdef HAVE_USELOCALE
# include <locale.h>
#endif

/* The total number of pthreads currently active.  This is defined
   here since it would be really stupid to have a threads-using
   program that doesn't call `pthread_create'.  */
unsigned int __pthread_total;


/* The entry-point for new threads.  */
static void
entry_point (struct __pthread *self, void *(*start_routine) (void *), void *arg)
{
  int err;

  ___pthread_self = self;
  __resp = &self->res_state;

#if IS_IN (libpthread)
  /* Initialize pointers to locale data.  */
  __ctype_init ();
#endif
#ifdef HAVE_USELOCALE
  /* A fresh thread needs to be bound to the global locale.  */
  uselocale (LC_GLOBAL_LOCALE);
#endif

  __pthread_startup ();

  /* We can now unleash signals.  */
  err = __pthread_sigstate (self, SIG_SETMASK, &self->init_sigset, 0, 0);
  assert_perror (err);

  if (self->c11)
    {
      /* The function pointer of the c11 thread start is cast to an incorrect
         type on __pthread_create call, however it is casted back to correct
         one so the call behavior is well-defined (it is assumed that pointers
         to void are able to represent all values of int).  */
      int (*start)(void*) = (int (*) (void*)) start_routine;
      __pthread_exit ((void*) (uintptr_t) start (arg));
    }
  else
    __pthread_exit (start_routine (arg));
}

/* Create a thread with attributes given by ATTR, executing
   START_ROUTINE with argument ARG.  */
int
__pthread_create (pthread_t * thread, const pthread_attr_t * attr,
		  void *(*start_routine) (void *), void *arg)
{
  int err;
  struct __pthread *pthread;

  err = __pthread_create_internal (&pthread, attr, start_routine, arg);
  if (!err)
    *thread = pthread->thread;
  else if (err == ENOMEM)
    err = EAGAIN;

  return err;
}
weak_alias (__pthread_create, pthread_create)
hidden_def (__pthread_create)

/* Internal version of pthread_create.  See comment in
   pt-internal.h.  */
int
__pthread_create_internal (struct __pthread **thread,
			   const pthread_attr_t * attr,
			   void *(*start_routine) (void *), void *arg)
{
  int err;
  struct __pthread *pthread;
  const struct __pthread_attr *setup;
  sigset_t sigset;
  size_t stacksize;

  /* Avoid a data race in the multi-threaded case.  */
  if (__libc_single_threaded)
    __libc_single_threaded = 0;

  /* Allocate a new thread structure.  */
  err = __pthread_alloc (&pthread);
  if (err)
    goto failed;

  if (attr == ATTR_C11_THREAD)
    {
      attr = NULL;
      pthread->c11 = true;
    }
  else
    pthread->c11 = false;

  /* Use the default attributes if ATTR is NULL.  */
  setup = attr ? attr : &__pthread_default_attr;

  stacksize = setup->__stacksize;
  if (stacksize == 0)
    {
      struct rlimit rlim;
      __getrlimit (RLIMIT_STACK, &rlim);
      if (rlim.rlim_cur != RLIM_INFINITY)
	stacksize = rlim.rlim_cur;
      if (stacksize == 0)
	stacksize = PTHREAD_STACK_DEFAULT;
    }

  /* Initialize the thread state.  */
  pthread->state = (setup->__detachstate == PTHREAD_CREATE_DETACHED
		    ? PTHREAD_DETACHED : PTHREAD_JOINABLE);

  if (setup->__stackaddr)
    {
      pthread->stackaddr = setup->__stackaddr;

      /* If the user supplied a stack, it is not our responsibility to
         setup a stack guard.  */
      pthread->guardsize = 0;
      pthread->stack = 0;
    }
  else
    {
      /* Allocate a stack.  */
      err = __pthread_stack_alloc (&pthread->stackaddr,
				   ((setup->__guardsize + __vm_page_size - 1)
				    / __vm_page_size) * __vm_page_size
				   + stacksize);
      if (err)
	goto failed_stack_alloc;

      pthread->guardsize = setup->__guardsize;
      pthread->stack = 1;
    }

  pthread->stacksize = stacksize;

  /* Allocate the kernel thread and other required resources.  */
  err = __pthread_thread_alloc (pthread);
  if (err)
    goto failed_thread_alloc;

  pthread->tcb = _dl_allocate_tls (NULL);
  if (pthread->tcb == NULL)
    {
      err = ENOMEM;
      goto failed_thread_tls_alloc;
    }
  pthread->tcb->tcb = pthread->tcb;

  /* And initialize the rest of the machine context.  This may include
     additional machine- and system-specific initializations that
     prove convenient.  */
  err = __pthread_setup (pthread, entry_point, start_routine, arg);
  if (err)
    goto failed_setup;

  /* Initialize the system-specific signal state for the new
     thread.  */
  err = __pthread_sigstate_init (pthread);
  if (err)
    goto failed_sigstate;

  /* If the new thread is joinable, add a reference for the caller.  */
  if (pthread->state == PTHREAD_JOINABLE)
    pthread->nr_refs++;

  /* Set the new thread's signal mask and set the pending signals to
     empty.  POSIX says: "The signal mask shall be inherited from the
     creating thread.  The set of signals pending for the new thread
     shall be empty."  If the currnet thread is not a pthread then we
     just inherit the process' sigmask.  */
  if (__pthread_num_threads == 1)
    err = __sigprocmask (0, 0, &pthread->init_sigset);
  else
    err = __pthread_sigstate (_pthread_self (), 0, 0, &pthread->init_sigset, 0);
  assert_perror (err);

  /* But block the signals for now, until the thread is fully initialized.  */
  __sigfillset (&sigset);
  err = __pthread_sigstate (pthread, SIG_SETMASK, &sigset, 0, 1);
  assert_perror (err);

  /* Increase the total number of threads.  We do this before actually
     starting the new thread, since the new thread might immediately
     call `pthread_exit' which decreases the number of threads and
     calls `exit' if the number of threads reaches zero.  Increasing
     the number of threads from within the new thread isn't an option
     since this thread might return and call `pthread_exit' before the
     new thread runs.  */
  atomic_increment (&__pthread_total);

  /* Store a pointer to this thread in the thread ID lookup table.  We
     could use __thread_setid, however, we only lock for reading as no
     other thread should be using this entry (we also assume that the
     store is atomic).  */
  __pthread_rwlock_rdlock (&__pthread_threads_lock);
  __pthread_threads[pthread->thread - 1] = pthread;
  __pthread_rwlock_unlock (&__pthread_threads_lock);

  /* At this point it is possible to guess our pthread ID.  We have to
     make sure that all functions taking a pthread_t argument can
     handle the fact that this thread isn't really running yet.  Since
     the new thread might be passed its ID through pthread_create (to
     avoid calling pthread_self), read it before starting the thread.  */
  *thread = pthread;

  /* Schedule the new thread.  */
  err = __pthread_thread_start (pthread);
  if (err)
    goto failed_starting;


  return 0;

failed_starting:
  /* If joinable, a reference was added for the caller.  */
  if (pthread->state == PTHREAD_JOINABLE)
    __pthread_dealloc (pthread);

  __pthread_setid (pthread->thread, NULL);
  atomic_decrement (&__pthread_total);
failed_sigstate:
  __pthread_sigstate_destroy (pthread);
failed_setup:
  _dl_deallocate_tls (pthread->tcb, 1);
  pthread->tcb = NULL;
failed_thread_tls_alloc:
  __pthread_thread_terminate (pthread);

  /* __pthread_thread_terminate has taken care of deallocating the stack and
     the thread structure.  */
  goto failed;
failed_thread_alloc:
  if (pthread->stack)
    __pthread_stack_dealloc (pthread->stackaddr,
			     ((setup->__guardsize + __vm_page_size - 1)
			      / __vm_page_size) * __vm_page_size + stacksize);
failed_stack_alloc:
  __pthread_dealloc (pthread);
failed:
  return err;
}

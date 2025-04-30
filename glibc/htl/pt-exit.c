/* Thread termination.
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
#include <stdlib.h>

#include <pt-internal.h>
#include <pthreadP.h>

#include <atomic.h>


/* Terminate the current thread and make STATUS available to any
   thread that might join it.  */
void
__pthread_exit (void *status)
{
  struct __pthread *self = _pthread_self ();
  struct __pthread_cancelation_handler **handlers;
  int oldstate;

  /* Run any cancelation handlers.  According to POSIX, the
     cancellation cleanup handlers should be called with cancellation
     disabled.  */
  __pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &oldstate);

  for (handlers = __pthread_get_cleanup_stack ();
       *handlers != NULL;
       *handlers = (*handlers)->__next)
    (*handlers)->__handler ((*handlers)->__arg);

  __pthread_setcancelstate (oldstate, &oldstate);

  /* Decrease the number of threads.  We use an atomic operation to
     make sure that only the last thread calls `exit'.  */
  if (atomic_decrement_and_test (&__pthread_total))
    /* We are the last thread.  */
    exit (0);

  /* Note that after this point the process can be terminated at any
     point if another thread calls `pthread_exit' and happens to be
     the last thread.  */

  __pthread_mutex_lock (&self->state_lock);

  if (self->cancel_state == PTHREAD_CANCEL_ENABLE && self->cancel_pending)
    status = PTHREAD_CANCELED;

  switch (self->state)
    {
    default:
      assert (!"Consistency error: unexpected self->state");
      abort ();
      break;

    case PTHREAD_DETACHED:
      __pthread_mutex_unlock (&self->state_lock);

      break;

    case PTHREAD_JOINABLE:
      /* We need to stay around for a while since another thread
         might want to join us.  */
      self->state = PTHREAD_EXITED;

      /* We need to remember the exit status.  A thread joining us
         might ask for it.  */
      self->status = status;

      /* Broadcast the condition.  This will wake up threads that are
         waiting to join us.  */
      __pthread_cond_broadcast (&self->state_cond);
      __pthread_mutex_unlock (&self->state_lock);

      break;
    }

  /* Destroy any thread specific data.  */
  __pthread_destroy_specific (self);

  /* Destroy any signal state.  */
  __pthread_sigstate_destroy (self);

  /* Self terminating requires TLS, so defer the release of the TCB until
     the thread structure is reused.  */

  /* Release kernel resources, including the kernel thread and the stack,
     and drop the self reference.  */
  __pthread_thread_terminate (self);

  /* NOTREACHED */
  abort ();
}

weak_alias (__pthread_exit, pthread_exit);

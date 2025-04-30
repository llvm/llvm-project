/* Wait on a condition.  Generic version.
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

#include <pthread.h>

#include <pt-internal.h>
#include <pthreadP.h>
#include <time.h>

extern int __pthread_cond_timedwait_internal (pthread_cond_t *cond,
					      pthread_mutex_t *mutex,
					      clockid_t clockid,
					      const struct timespec *abstime);

int
__pthread_cond_timedwait (pthread_cond_t *cond,
			  pthread_mutex_t *mutex,
			  const struct timespec *abstime)
{
  return __pthread_cond_timedwait_internal (cond, mutex, -1, abstime);
}

weak_alias (__pthread_cond_timedwait, pthread_cond_timedwait);

int
__pthread_cond_clockwait (pthread_cond_t *cond,
			  pthread_mutex_t *mutex,
			  clockid_t clockid,
			  const struct timespec *abstime)
{
  return __pthread_cond_timedwait_internal (cond, mutex, clockid, abstime);
}

weak_alias (__pthread_cond_clockwait, pthread_cond_clockwait);

struct cancel_ctx
{
  struct __pthread *wakeup;
  pthread_cond_t *cond;
};

static void
cancel_hook (void *arg)
{
  struct cancel_ctx *ctx = arg;
  struct __pthread *wakeup = ctx->wakeup;
  pthread_cond_t *cond = ctx->cond;
  int unblock;

  __pthread_spin_wait (&cond->__lock);
  /* The thread only needs to be awaken if it's blocking or about to block.
     If it was already unblocked, it's not queued any more.  */
  unblock = wakeup->prevp != NULL;
  if (unblock)
    __pthread_dequeue (wakeup);
  __pthread_spin_unlock (&cond->__lock);

  if (unblock)
    __pthread_wakeup (wakeup);
}

/* Block on condition variable COND until ABSTIME.  As a GNU
   extension, if ABSTIME is NULL, then wait forever.  MUTEX should be
   held by the calling thread.  On return, MUTEX will be held by the
   calling thread.  */
int
__pthread_cond_timedwait_internal (pthread_cond_t *cond,
				   pthread_mutex_t *mutex,
				   clockid_t clockid,
				   const struct timespec *abstime)
{
  error_t err;
  int cancelled, oldtype, drain;
  clockid_t clock_id;

  if (clockid != -1)
    clock_id = clockid;
  else
    clock_id = __pthread_default_condattr.__clock;

  if (abstime && ! valid_nanoseconds (abstime->tv_nsec))
    return EINVAL;

  err = __pthread_mutex_checklocked (mutex);
  if (err)
    return err;

  struct __pthread *self = _pthread_self ();
  struct cancel_ctx ctx;
  ctx.wakeup = self;
  ctx.cond = cond;

  /* Test for a pending cancellation request, switch to deferred mode for
     safer resource handling, and prepare the hook to call in case we're
     cancelled while blocking.  Once CANCEL_LOCK is released, the cancellation
     hook can be called by another thread at any time.  Whatever happens,
     this function must exit with MUTEX locked.

     This function contains inline implementations of pthread_testcancel and
     pthread_setcanceltype to reduce locking overhead.  */
  __pthread_mutex_lock (&self->cancel_lock);
  cancelled = (self->cancel_state == PTHREAD_CANCEL_ENABLE)
      && self->cancel_pending;

  if (cancelled)
    {
      __pthread_mutex_unlock (&self->cancel_lock);
      __pthread_exit (PTHREAD_CANCELED);
    }

  self->cancel_hook = cancel_hook;
  self->cancel_hook_arg = &ctx;
  oldtype = self->cancel_type;

  if (oldtype != PTHREAD_CANCEL_DEFERRED)
    self->cancel_type = PTHREAD_CANCEL_DEFERRED;

  /* Add ourselves to the list of waiters.  This is done while setting
     the cancellation hook to simplify the cancellation procedure, i.e.
     if the thread is queued, it can be cancelled, otherwise it is
     already unblocked, progressing on the return path.  */
  __pthread_spin_wait (&cond->__lock);
  __pthread_enqueue (&cond->__queue, self);
  if (cond->__attr != NULL && clockid == -1)
    clock_id = cond->__attr->__clock;
  __pthread_spin_unlock (&cond->__lock);

  __pthread_mutex_unlock (&self->cancel_lock);

  /* Release MUTEX before blocking.  */
  __pthread_mutex_unlock (mutex);

  /* Increase the waiter reference count.  Relaxed MO is sufficient because
     we only need to synchronize when decrementing the reference count.  */
  atomic_fetch_add_relaxed (&cond->__wrefs, 2);

  /* Block the thread.  */
  if (abstime != NULL)
    err = __pthread_timedblock (self, abstime, clock_id);
  else
    {
      err = 0;
      __pthread_block (self);
    }

  __pthread_spin_wait (&cond->__lock);
  if (self->prevp == NULL)
    {
      /* Another thread removed us from the list of waiters, which means a
         wakeup message has been sent.  It was either consumed while we were
         blocking, or queued after we timed out and before we acquired the
         condition lock, in which case the message queue must be drained.  */
      if (!err)
	drain = 0;
      else
	{
	  assert (err == ETIMEDOUT);
	  drain = 1;
	}
    }
  else
    {
      /* We're still in the list of waiters.  Noone attempted to wake us up,
         i.e. we timed out.  */
      assert (err == ETIMEDOUT);
      __pthread_dequeue (self);
      drain = 0;
    }
  __pthread_spin_unlock (&cond->__lock);

  /* If destruction is pending (i.e., the wake-request flag is nonzero) and we
     are the last waiter (prior value of __wrefs was 1 << 1), then wake any
     threads waiting in pthread_cond_destroy.  Release MO to synchronize with
     these threads.  Don't bother clearing the wake-up request flag.  */
  if ((atomic_fetch_add_release (&cond->__wrefs, -2)) == 3)
    __gsync_wake (__mach_task_self (), (vm_offset_t) &cond->__wrefs, 0, 0);

  if (drain)
    __pthread_block (self);

  /* We're almost done.  Remove the unblock hook, restore the previous
     cancellation type, and check for a pending cancellation request.  */
  __pthread_mutex_lock (&self->cancel_lock);
  self->cancel_hook = NULL;
  self->cancel_hook_arg = NULL;
  self->cancel_type = oldtype;
  cancelled = (self->cancel_state == PTHREAD_CANCEL_ENABLE)
      && self->cancel_pending;
  __pthread_mutex_unlock (&self->cancel_lock);

  /* Reacquire MUTEX before returning/cancelling.  */
  __pthread_mutex_lock (mutex);

  if (cancelled)
    __pthread_exit (PTHREAD_CANCELED);

  return err;
}

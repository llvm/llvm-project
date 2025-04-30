/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stddef.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <hurd.h>
#include <hurd/signal.h>
#include <hurd/sigpreempt.h>
#include <hurd/msg_request.h>
#include <mach.h>
#include <mach/message.h>

/* XXX Temporary cheezoid implementation of ITIMER_REAL/SIGALRM.  */

spin_lock_t _hurd_itimer_lock = SPIN_LOCK_INITIALIZER;
struct itimerval _hurd_itimerval; /* Current state of the timer.  */
mach_port_t _hurd_itimer_port;	/* Port the timer thread blocks on.  */
thread_t _hurd_itimer_thread;	/* Thread waiting for timeout.  */
int _hurd_itimer_thread_suspended; /* Nonzero if that thread is suspended.  */
vm_address_t _hurd_itimer_thread_stack_base; /* Base of its stack.  */
vm_size_t _hurd_itimer_thread_stack_size; /* Size of its stack.  */
struct timeval _hurd_itimer_started; /* Time the thread started waiting.  */

static void
quantize_timeval (struct timeval *tv)
{
  static time_t quantum = -1;

  if (quantum == -1)
    quantum = 1000000 / __getclktck ();

  tv->tv_usec = ((tv->tv_usec + (quantum - 1)) / quantum) * quantum;
  if (tv->tv_usec >= 1000000)
    {
      ++tv->tv_sec;
      tv->tv_usec -= 1000000;
    }
}

static inline void
subtract_timeval (struct timeval *from, const struct timeval *subtract)
{
  from->tv_usec -= subtract->tv_usec;
  from->tv_sec -= subtract->tv_sec;
  while (from->tv_usec < 0)
    {
      --from->tv_sec;
      from->tv_usec += 1000000;
    }
}

/* Function run by the itimer thread.
   This code must be very careful not ever to require a MiG reply port.  */

static void
timer_thread (void)
{
  while (1)
    {
      error_t err;
      /* The only message we ever expect to receive is the reply from the
         signal thread to a sig_post call we did.  We never examine the
	 contents.  */
      struct
	{
	  mach_msg_header_t header;
	  mach_msg_type_t return_code_type;
	  error_t return_code;
	} msg;

      /* Wait for a message on a port that noone sends to.  The purpose is
	 the receive timeout.  Notice interrupts so that if we are
	 thread_abort'd, we will loop around and fetch new values from
	 _hurd_itimerval.  */
      err = __mach_msg (&msg.header,
			MACH_RCV_MSG|MACH_RCV_TIMEOUT|MACH_RCV_INTERRUPT,
			0, sizeof(msg), _hurd_itimer_port,
			_hurd_itimerval.it_value.tv_sec * 1000
			+ _hurd_itimerval.it_value.tv_usec / 1000,
			MACH_PORT_NULL);
      switch (err)
	{
	case MACH_RCV_TIMED_OUT:
	  /* We got the expected timeout.  Send a message to the signal
	     thread to tell it to post a SIGALRM signal.  We use
	     _hurd_itimer_port as the reply port just so we will block until
	     the signal thread has frobnicated things to reload the itimer or
	     has terminated this thread.  */
	  __msg_sig_post_request (_hurd_msgport,
				  _hurd_itimer_port,
				  MACH_MSG_TYPE_MAKE_SEND_ONCE,
				  SIGALRM, SI_TIMER, __mach_task_self ());
	  break;

	case MACH_RCV_INTERRUPTED:
	  /* We were thread_abort'd.  This is to tell us that
	     _hurd_itimerval has changed and we need to reexamine it
	     and start waiting with the new timeout value.  */
	  break;

	case MACH_MSG_SUCCESS:
	  /* We got the reply message from the sig_post_request above.
	     Ignore it and reexamine the timer value.  */
	  __mach_msg_destroy (&msg.header); /* Just in case.  */
	  break;

	default:
	  /* Unexpected lossage.  Oh well, keep trying.  */
	  break;
	}
    }
}


/* Forward declaration.  */
static int setitimer_locked (const struct itimerval *new,
			     struct itimerval *old, void *crit,
			     int hurd_siglocked);

static sighandler_t
restart_itimer (struct hurd_signal_preemptor *preemptor,
		struct hurd_sigstate *ss,
		int *signo, struct hurd_signal_detail *detail)
{
  /* This function gets called in the signal thread
     each time a SIGALRM is arriving (even if blocked).  */
  struct itimerval it;

  /* Either reload or disable the itimer.  */
  __spin_lock (&_hurd_itimer_lock);
  it.it_value = it.it_interval = _hurd_itimerval.it_interval;
  setitimer_locked (&it, NULL, NULL, 1);

  /* Continue with normal delivery (or hold, etc.) of SIGALRM.  */
  return SIG_ERR;
}


/* Called before any normal SIGALRM signal is delivered.
   Reload the itimer, or disable the itimer.  */

static int
setitimer_locked (const struct itimerval *new, struct itimerval *old,
		  void *crit, int hurd_siglocked)
{
  struct itimerval newval;
  struct timeval now, remaining, elapsed;
  struct timeval old_interval;
  error_t err;

  inline void kill_itimer_thread (void)
    {
      __thread_terminate (_hurd_itimer_thread);
      __vm_deallocate (__mach_task_self (),
		       _hurd_itimer_thread_stack_base,
		       _hurd_itimer_thread_stack_size);
      _hurd_itimer_thread = MACH_PORT_NULL;
    }

  if (!new)
    {
      /* Just return the current value in OLD without changing anything.
	 This is what BSD does, even though it's not documented. */
      if (old)
	*old = _hurd_itimerval;
      spin_unlock (&_hurd_itimer_lock);
      _hurd_critical_section_unlock (crit);
      return 0;
    }

  newval = *new;
  quantize_timeval (&newval.it_interval);
  quantize_timeval (&newval.it_value);
  if ((newval.it_value.tv_sec | newval.it_value.tv_usec) != 0)
    {
      /* Make sure the itimer thread is set up.  */

      /* Set up a signal preemptor global for all threads to
	 run `restart_itimer' each time a SIGALRM would arrive.  */
      static struct hurd_signal_preemptor preemptor =
	{
	  __sigmask (SIGALRM), SI_TIMER, SI_TIMER,
	  &restart_itimer,
	};
      if (!hurd_siglocked)
	__mutex_lock (&_hurd_siglock);
      if (! preemptor.next && _hurdsig_preemptors != &preemptor)
	{
	  preemptor.next = _hurdsig_preemptors;
	  _hurdsig_preemptors = &preemptor;
	  _hurdsig_preempted_set |= preemptor.signals;
	}
      if (!hurd_siglocked)
	__mutex_unlock (&_hurd_siglock);

      if (_hurd_itimer_port == MACH_PORT_NULL)
	{
	  /* Allocate a receive right that the itimer thread will
	     block waiting for a message on.  */
	  if (err = __mach_port_allocate (__mach_task_self (),
					  MACH_PORT_RIGHT_RECEIVE,
					  &_hurd_itimer_port))
	    goto out;
	}

      if (_hurd_itimer_thread == MACH_PORT_NULL)
	{
	  /* Start up the itimer thread running `timer_thread' (below).  */
	  if (err = __thread_create (__mach_task_self (),
				     &_hurd_itimer_thread))
	    goto out;
	  _hurd_itimer_thread_stack_base = 0; /* Anywhere.  */
	  _hurd_itimer_thread_stack_size = __vm_page_size; /* Small stack.  */
	  if ((err = __mach_setup_thread (__mach_task_self (),
					 _hurd_itimer_thread,
					 &timer_thread,
					 &_hurd_itimer_thread_stack_base,
					 &_hurd_itimer_thread_stack_size))
	      || (err = __mach_setup_tls(_hurd_itimer_thread)))
	    {
	      __thread_terminate (_hurd_itimer_thread);
	      _hurd_itimer_thread = MACH_PORT_NULL;
	      goto out;
	    }
	  _hurd_itimer_thread_suspended = 1;
	}
    }

  if ((newval.it_value.tv_sec | newval.it_value.tv_usec) != 0 || old != NULL)
    {
      /* Calculate how much time is remaining for the pending alarm.  */
      {
	time_value_t tv;
	__host_get_time (__mach_host_self (), &tv);
	now.tv_sec = tv.seconds;
	now.tv_usec = tv.microseconds;
      }
      elapsed = now;
      subtract_timeval (&elapsed, &_hurd_itimer_started);
      remaining = _hurd_itimerval.it_value;
      if (timercmp (&remaining, &elapsed, <))
	{
	  /* Hmm.  The timer should have just gone off, but has not been reset.
	     This is a possible timing glitch.  The alarm will signal soon. */
	  /* XXX wrong */
	  remaining.tv_sec = 0;
	  remaining.tv_usec = 0;
	}
      else
	subtract_timeval (&remaining, &elapsed);

      /* Remember the old reload interval before changing it.  */
      old_interval = _hurd_itimerval.it_interval;

      /* Record the starting time that the timer interval relates to.  */
      _hurd_itimer_started = now;
    }

  /* Load the new itimer value.  */
  _hurd_itimerval = newval;

  if ((newval.it_value.tv_sec | newval.it_value.tv_usec) == 0)
    {
      /* Disable the itimer.  */
      if (_hurd_itimer_thread && !_hurd_itimer_thread_suspended)
	{
	  /* Suspend the itimer thread so it does nothing.  Then abort its
	     kernel context so that when the thread is resumed, mach_msg
	     will return to timer_thread (below) and it will fetch new
	     values from _hurd_itimerval.  */
	  if ((err = __thread_suspend (_hurd_itimer_thread))
	      || (err = __thread_abort (_hurd_itimer_thread)))
	    /* If we can't save it for later, nuke it.  */
	    kill_itimer_thread ();
	  else
	    _hurd_itimer_thread_suspended = 1;
	}
    }
  /* See if the timeout changed.  If so, we must alert the itimer thread.  */
  else if (remaining.tv_sec != newval.it_value.tv_sec
	   || remaining.tv_usec != newval.it_value.tv_usec)
    {
      /* The timeout value is changing.  Tell the itimer thread to
	 reexamine it and start counting down.  If the itimer thread is
	 marked as suspended, either we just created it, or it was
	 suspended and thread_abort'd last time the itimer was disabled;
	 either way it will wake up and start waiting for the new timeout
	 value when we resume it.  If it is not suspended, the itimer
	 thread is waiting to deliver a pending alarm that we will override
	 (since it would come later than the new alarm being set);
	 thread_abort will make mach_msg return MACH_RCV_INTERRUPTED, so it
	 will loop around and use the new timeout value.  */
      if (err = (_hurd_itimer_thread_suspended
		 ? __thread_resume : __thread_abort) (_hurd_itimer_thread))
	{
	  kill_itimer_thread ();
	  goto out;
	}
      _hurd_itimer_thread_suspended = 0;
    }

  __spin_unlock (&_hurd_itimer_lock);
  _hurd_critical_section_unlock (crit);

  if (old != NULL)
    {
      old->it_value = remaining;
      old->it_interval = old_interval;
    }
  return 0;

 out:
  __spin_unlock (&_hurd_itimer_lock);
  _hurd_critical_section_unlock (crit);
  return __hurd_fail (err);
}

/* Set the timer WHICH to *NEW.  If OLD is not NULL,
   set *OLD to the old value of timer WHICH.
   Returns 0 on success, -1 on errors.  */
int
__setitimer (enum __itimer_which which, const struct itimerval *new,
	     struct itimerval *old)
{
  void *crit;
  int ret;

  switch (which)
    {
    default:
      return __hurd_fail (EINVAL);

    case ITIMER_VIRTUAL:
    case ITIMER_PROF:
      return __hurd_fail (ENOSYS);

    case ITIMER_REAL:
      break;
    }

retry:
  crit = _hurd_critical_section_lock ();
  __spin_lock (&_hurd_itimer_lock);
  ret = setitimer_locked (new, old, crit, 0);
  if (ret == -1 && errno == EINTR)
    /* Got a signal while inside an RPC of the critical section, retry again */
    goto retry;

  return ret;
}

static void
fork_itimer (void)
{
  /* We must restart the itimer in the child.  */

  struct itimerval it;

  __spin_lock (&_hurd_itimer_lock);
  _hurd_itimer_thread = MACH_PORT_NULL;
  it = _hurd_itimerval;
  it.it_value = it.it_interval;

  setitimer_locked (&it, NULL, NULL, 0);

  (void) &fork_itimer;		/* Avoid gcc optimizing out the function.  */
}
text_set_element (_hurd_fork_child_hook, fork_itimer);

weak_alias (__setitimer, setitimer)

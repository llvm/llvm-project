/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lock-intern.h>	/* For `struct mutex'.  */
#include <pthreadP.h>
#include <mach.h>
#include <mach/thread_switch.h>
#include <mach/mig_support.h>
#include <mach/vm_param.h>

#include <hurd.h>
#include <hurd/id.h>
#include <hurd/signal.h>

#include "hurdfault.h"
#include "hurdmalloc.h"		/* XXX */
#include "../locale/localeinfo.h"

#include <libc-diag.h>

const char *_hurdsig_getenv (const char *);

struct mutex _hurd_siglock;
int _hurd_stopped;

/* Port that receives signals and other miscellaneous messages.  */
mach_port_t _hurd_msgport;

/* Thread listening on it.  */
thread_t _hurd_msgport_thread;

/* These are set up by _hurdsig_init.  */
unsigned long int __hurd_sigthread_stack_base;
unsigned long int __hurd_sigthread_stack_end;

/* Linked-list of per-thread signal state.  */
struct hurd_sigstate *_hurd_sigstates;

/* Sigstate for the task-global signals.  */
struct hurd_sigstate *_hurd_global_sigstate;

/* Timeout for RPC's after interrupt_operation. */
mach_msg_timeout_t _hurd_interrupted_rpc_timeout = 60000;

static void
default_sigaction (struct sigaction actions[NSIG])
{
  int signo;

  __sigemptyset (&actions[0].sa_mask);
  actions[0].sa_flags = SA_RESTART;
  actions[0].sa_handler = SIG_DFL;

  for (signo = 1; signo < NSIG; ++signo)
    actions[signo] = actions[0];
}

struct hurd_sigstate *
_hurd_thread_sigstate (thread_t thread)
{
  struct hurd_sigstate *ss;
  __mutex_lock (&_hurd_siglock);
  for (ss = _hurd_sigstates; ss != NULL; ss = ss->next)
    if (ss->thread == thread)
       break;
  if (ss == NULL)
    {
      ss = malloc (sizeof (*ss));
      if (ss == NULL)
	__libc_fatal ("hurd: Can't allocate sigstate\n");
      __spin_lock_init (&ss->critical_section_lock);
      __spin_lock_init (&ss->lock);
      ss->thread = thread;

      /* Initialize default state.  */
      __sigemptyset (&ss->blocked);
      __sigemptyset (&ss->pending);
      memset (&ss->sigaltstack, 0, sizeof (ss->sigaltstack));
      ss->sigaltstack.ss_flags |= SS_DISABLE;
      ss->preemptors = NULL;
      ss->suspended = MACH_PORT_NULL;
      ss->intr_port = MACH_PORT_NULL;
      ss->context = NULL;
      ss->active_resources = NULL;
      ss->cancel = 0;
      ss->cancel_hook = NULL;

      if (thread == MACH_PORT_NULL)
	{
	  /* Process-wide sigstate, use the system defaults.  */
	  default_sigaction (ss->actions);

	  /* The global sigstate is not added to the _hurd_sigstates list.
	     It is created with _hurd_thread_sigstate (MACH_PORT_NULL)
	     but should be accessed through _hurd_global_sigstate.  */
	}
      else
	{
	  error_t err;

	  /* Use the global actions as a default for new threads.  */
	  struct hurd_sigstate *s = _hurd_global_sigstate;
	  if (s)
	    {
	      __spin_lock (&s->lock);
	      memcpy (ss->actions, s->actions, sizeof (s->actions));
	      __spin_unlock (&s->lock);
	    }
	  else
	    default_sigaction (ss->actions);

	  ss->next = _hurd_sigstates;
	  _hurd_sigstates = ss;

	  err = __mach_port_mod_refs (__mach_task_self (), thread,
				      MACH_PORT_RIGHT_SEND, 1);
	  if (err)
	    __libc_fatal ("hurd: Can't add reference on Mach thread\n");
	}
    }
  __mutex_unlock (&_hurd_siglock);
  return ss;
}
libc_hidden_def (_hurd_thread_sigstate)

/* Destroy a sigstate structure.  Called by libpthread just before the
 * corresponding thread is terminated.  */
void
_hurd_sigstate_delete (thread_t thread)
{
  struct hurd_sigstate **ssp, *ss;

  __mutex_lock (&_hurd_siglock);
  for (ssp = &_hurd_sigstates; *ssp; ssp = &(*ssp)->next)
    if ((*ssp)->thread == thread)
      break;

  ss = *ssp;
  if (ss)
    *ssp = ss->next;

  __mutex_unlock (&_hurd_siglock);
  if (ss)
    {
      if (ss->thread != MACH_PORT_NULL)
	__mach_port_deallocate (__mach_task_self (), ss->thread);

      free (ss);
    }
}

/* Make SS a global receiver, with pthread signal semantics.  */
void
_hurd_sigstate_set_global_rcv (struct hurd_sigstate *ss)
{
  assert (ss->thread != MACH_PORT_NULL);
  ss->actions[0].sa_handler = SIG_IGN;
}

/* Check whether SS is a global receiver.  */
static int
sigstate_is_global_rcv (const struct hurd_sigstate *ss)
{
  return (_hurd_global_sigstate != NULL)
	 && (ss->actions[0].sa_handler == SIG_IGN);
}
libc_hidden_def (_hurd_sigstate_delete)

/* Lock/unlock a hurd_sigstate structure.  If the accessors below require
   it, the global sigstate will be locked as well.  */
void
_hurd_sigstate_lock (struct hurd_sigstate *ss)
{
  if (sigstate_is_global_rcv (ss))
    __spin_lock (&_hurd_global_sigstate->lock);
  __spin_lock (&ss->lock);
}
void
_hurd_sigstate_unlock (struct hurd_sigstate *ss)
{
  __spin_unlock (&ss->lock);
  if (sigstate_is_global_rcv (ss))
    __spin_unlock (&_hurd_global_sigstate->lock);
}
libc_hidden_def (_hurd_sigstate_set_global_rcv)

/* Retreive a thread's full set of pending signals, including the global
   ones if appropriate.  SS must be locked.  */
sigset_t
_hurd_sigstate_pending (const struct hurd_sigstate *ss)
{
  sigset_t pending = ss->pending;
  if (sigstate_is_global_rcv (ss))
    __sigorset (&pending, &pending, &_hurd_global_sigstate->pending);
  return pending;
}

/* Clear a pending signal and return the associated detailed
   signal information. SS must be locked, and must have signal SIGNO
   pending, either directly or through the global sigstate.  */
static struct hurd_signal_detail
sigstate_clear_pending (struct hurd_sigstate *ss, int signo)
{
  if (sigstate_is_global_rcv (ss)
      && __sigismember (&_hurd_global_sigstate->pending, signo))
    {
      __sigdelset (&_hurd_global_sigstate->pending, signo);
      return _hurd_global_sigstate->pending_data[signo];
    }

  assert (__sigismember (&ss->pending, signo));
  __sigdelset (&ss->pending, signo);
  return ss->pending_data[signo];
}
libc_hidden_def (_hurd_sigstate_lock)
libc_hidden_def (_hurd_sigstate_unlock)

/* Retreive a thread's action vector.  SS must be locked.  */
struct sigaction *
_hurd_sigstate_actions (struct hurd_sigstate *ss)
{
  if (sigstate_is_global_rcv (ss))
    return _hurd_global_sigstate->actions;
  else
    return ss->actions;
}
libc_hidden_def (_hurd_sigstate_pending)


/* Signal delivery itself is on this page.  */

#include <hurd/fd.h>
#include <hurd/crash.h>
#include <hurd/resource.h>
#include <hurd/paths.h>
#include <setjmp.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <thread_state.h>
#include <hurd/msg_server.h>
#include <hurd/msg_reply.h>	/* For __msg_sig_post_reply.  */
#include <hurd/interrupt.h>
#include <assert.h>
#include <unistd.h>


/* Call the crash dump server to mummify us before we die.
   Returns nonzero if a core file was written.  */
static int
write_corefile (int signo, const struct hurd_signal_detail *detail)
{
  error_t err;
  mach_port_t coreserver;
  file_t file, coredir;
  const char *name;

  /* Don't bother locking since we just read the one word.  */
  rlim_t corelimit = _hurd_rlimits[RLIMIT_CORE].rlim_cur;

  if (corelimit == 0)
    /* No core dumping, thank you very much.  Note that this makes
       `ulimit -c 0' prevent crash-suspension too, which is probably
       what the user wanted.  */
    return 0;

  /* XXX RLIMIT_CORE:
     When we have a protocol to make the server return an error
     for RLIMIT_FSIZE, then tell the corefile fs server the RLIMIT_CORE
     value in place of the RLIMIT_FSIZE value.  */

  /* First get a port to the core dumping server.  */
  coreserver = MACH_PORT_NULL;
  name = _hurdsig_getenv ("CRASHSERVER");
  if (name != NULL)
    coreserver = __file_name_lookup (name, 0, 0);
  if (coreserver == MACH_PORT_NULL)
    coreserver = __file_name_lookup (_SERVERS_CRASH, 0, 0);
  if (coreserver == MACH_PORT_NULL)
    return 0;

  /* Get a port to the directory where the new core file will reside.  */
  file = MACH_PORT_NULL;
  name = _hurdsig_getenv ("COREFILE");
  if (name == NULL)
    name = "core";
  coredir = __file_name_split (name, (char **) &name);
  if (coredir != MACH_PORT_NULL)
    /* Create the new file, but don't link it into the directory yet.  */
    __dir_mkfile (coredir, O_WRONLY|O_CREAT,
		  0600 & ~_hurd_umask, /* XXX ? */
		  &file);

  /* Call the core dumping server to write the core file.  */
  err = __crash_dump_task (coreserver,
			   __mach_task_self (),
			   file,
			   signo, detail->code, detail->error,
			   detail->exc, detail->exc_code, detail->exc_subcode,
			   _hurd_ports[INIT_PORT_CTTYID].port,
			   MACH_MSG_TYPE_COPY_SEND);
  __mach_port_deallocate (__mach_task_self (), coreserver);

  if (! err && file != MACH_PORT_NULL)
    /* The core dump into FILE succeeded, so now link it into the
       directory.  */
    err = __dir_link (coredir, file, name, 1);
  __mach_port_deallocate (__mach_task_self (), file);
  __mach_port_deallocate (__mach_task_self (), coredir);
  return !err && file != MACH_PORT_NULL;
}


/* The lowest-numbered thread state flavor value is 1,
   so we use bit 0 in machine_thread_all_state.set to
   record whether we have done thread_abort.  */
#define THREAD_ABORTED 1

/* SS->thread is suspended.  Abort the thread and get its basic state.  */
static void
abort_thread (struct hurd_sigstate *ss, struct machine_thread_all_state *state,
	      void (*reply) (void))
{
  assert (ss->thread != MACH_PORT_NULL);

  if (!(state->set & THREAD_ABORTED))
    {
      error_t err = __thread_abort (ss->thread);
      assert_perror (err);
      /* Clear all thread state flavor set bits, because thread_abort may
	 have changed the state.  */
      state->set = THREAD_ABORTED;
    }

  if (reply)
    (*reply) ();

  machine_get_basic_state (ss->thread, state);
}

/* Find the location of the MiG reply port cell in use by the thread whose
   state is described by THREAD_STATE.  If SIGTHREAD is nonzero, make sure
   that this location can be set without faulting, or else return NULL.  */

static mach_port_t *
interrupted_reply_port_location (thread_t thread,
				 struct machine_thread_all_state *thread_state,
				 int sigthread)
{
  mach_port_t *portloc = &THREAD_TCB(thread, thread_state)->reply_port;

  if (sigthread && _hurdsig_catch_memory_fault (portloc))
    /* Faulted trying to read the TCB.  */
    return NULL;

  DIAG_PUSH_NEEDS_COMMENT;
  /* GCC 6 and before seem to be confused by the setjmp call inside
     _hurdsig_catch_memory_fault and think that we may be returning a second
     time to here with portloc uninitialized (but we never do). */
  DIAG_IGNORE_NEEDS_COMMENT (6, "-Wmaybe-uninitialized");
  /* Fault now if this pointer is bogus.  */
  *(volatile mach_port_t *) portloc = *portloc;
  DIAG_POP_NEEDS_COMMENT;

  if (sigthread)
    _hurdsig_end_catch_fault ();

  return portloc;
}

#include <hurd/sigpreempt.h>
#include <intr-msg.h>

/* Timeout on interrupt_operation calls.  */
mach_msg_timeout_t _hurdsig_interrupt_timeout = 1000;

/* SS->thread is suspended.

   Abort any interruptible RPC operation the thread is doing.

   This uses only the constant member SS->thread and the unlocked, atomically
   set member SS->intr_port, so no locking is needed.

   If successfully sent an interrupt_operation and therefore the thread should
   wait for its pending RPC to return (possibly EINTR) before taking the
   incoming signal, returns the reply port to be received on.  Otherwise
   returns MACH_PORT_NULL.

   SIGNO is used to find the applicable SA_RESTART bit.  If SIGNO is zero,
   the RPC fails with EINTR instead of restarting (thread_cancel).

   *STATE_CHANGE is set nonzero if STATE->basic was modified and should
   be applied back to the thread if it might ever run again, else zero.  */

mach_port_t
_hurdsig_abort_rpcs (struct hurd_sigstate *ss, int signo, int sigthread,
		     struct machine_thread_all_state *state, int *state_change,
		     void (*reply) (void))
{
  extern const void _hurd_intr_rpc_msg_about_to;
  extern const void _hurd_intr_rpc_msg_in_trap;
  mach_port_t rcv_port = MACH_PORT_NULL;
  mach_port_t intr_port;

  *state_change = 0;

  intr_port = ss->intr_port;
  if (intr_port == MACH_PORT_NULL)
    /* No interruption needs done.  */
    return MACH_PORT_NULL;

  /* Abort the thread's kernel context, so any pending message send or
     receive completes immediately or aborts.  */
  abort_thread (ss, state, reply);

  if (state->basic.PC >= (natural_t) &_hurd_intr_rpc_msg_about_to
      && state->basic.PC < (natural_t) &_hurd_intr_rpc_msg_in_trap)
    {
      /* The thread is about to do the RPC, but hasn't yet entered
	 mach_msg.  Mutate the thread's state so it knows not to try
	 the RPC.  */
      INTR_MSG_BACK_OUT (&state->basic);
      MACHINE_THREAD_STATE_SET_PC (&state->basic,
				   &_hurd_intr_rpc_msg_in_trap);
      state->basic.SYSRETURN = MACH_SEND_INTERRUPTED;
      *state_change = 1;
    }
  else if (state->basic.PC == (natural_t) &_hurd_intr_rpc_msg_in_trap
	   /* The thread was blocked in the system call.  After thread_abort,
	      the return value register indicates what state the RPC was in
	      when interrupted.  */
	   && state->basic.SYSRETURN == MACH_RCV_INTERRUPTED)
      {
	/* The RPC request message was sent and the thread was waiting for
	   the reply message; now the message receive has been aborted, so
	   the mach_msg call will return MACH_RCV_INTERRUPTED.  We must tell
	   the server to interrupt the pending operation.  The thread must
	   wait for the reply message before running the signal handler (to
	   guarantee that the operation has finished being interrupted), so
	   our nonzero return tells the trampoline code to finish the message
	   receive operation before running the handler.  */

	mach_port_t *reply = interrupted_reply_port_location (ss->thread,
							      state,
							      sigthread);
	error_t err = __interrupt_operation (intr_port, _hurdsig_interrupt_timeout);

	if (err)
	  {
	    if (reply)
	      {
		/* The interrupt didn't work.
		   Destroy the receive right the thread is blocked on.  */
		__mach_port_destroy (__mach_task_self (), *reply);
		*reply = MACH_PORT_NULL;
	      }

	    /* The system call return value register now contains
	       MACH_RCV_INTERRUPTED; when mach_msg resumes, it will retry the
	       call.  Since we have just destroyed the receive right, the
	       retry will fail with MACH_RCV_INVALID_NAME.  Instead, just
	       change the return value here to EINTR so mach_msg will not
	       retry and the EINTR error code will propagate up.  */
	    state->basic.SYSRETURN = EINTR;
	    *state_change = 1;
	  }
	else if (reply)
	  rcv_port = *reply;

	/* All threads whose RPCs were interrupted by the interrupt_operation
	   call above will retry their RPCs unless we clear SS->intr_port.
	   So we clear it for the thread taking a signal when SA_RESTART is
	   clear, so that its call returns EINTR.  */
	if (! signo || !(_hurd_sigstate_actions (ss) [signo].sa_flags & SA_RESTART))
	  ss->intr_port = MACH_PORT_NULL;
      }

  return rcv_port;
}


/* Abort the RPCs being run by all threads but this one;
   all other threads should be suspended.  If LIVE is nonzero, those
   threads may run again, so they should be adjusted as necessary to be
   happy when resumed.  STATE is clobbered as a scratch area; its initial
   contents are ignored, and its contents on return are not useful.  */

static void
abort_all_rpcs (int signo, struct machine_thread_all_state *state, int live)
{
  /* We can just loop over the sigstates.  Any thread doing something
     interruptible must have one.  We needn't bother locking because all
     other threads are stopped.  */

  struct hurd_sigstate *ss;
  size_t nthreads;
  mach_port_t *reply_ports;

  /* First loop over the sigstates to count them.
     We need to know how big a vector we will need for REPLY_PORTS.  */
  nthreads = 0;
  for (ss = _hurd_sigstates; ss != NULL; ss = ss->next)
    ++nthreads;

  reply_ports = alloca (nthreads * sizeof *reply_ports);

  nthreads = 0;
  for (ss = _hurd_sigstates; ss != NULL; ss = ss->next, ++nthreads)
    if (ss->thread == _hurd_msgport_thread)
      reply_ports[nthreads] = MACH_PORT_NULL;
    else
      {
	int state_changed;
	state->set = 0;		/* Reset scratch area.  */

	/* Abort any operation in progress with interrupt_operation.
	   Record the reply port the thread is waiting on.
	   We will wait for all the replies below.  */
	reply_ports[nthreads] = _hurdsig_abort_rpcs (ss, signo, 1,
						     state, &state_changed,
						     NULL);
	if (live)
	  {
	    if (reply_ports[nthreads] != MACH_PORT_NULL)
	      {
		/* We will wait for the reply to this RPC below, so the
		   thread must issue a new RPC rather than waiting for the
		   reply to the one it sent.  */
		state->basic.SYSRETURN = EINTR;
		state_changed = 1;
	      }
	    if (state_changed)
	      /* Aborting the RPC needed to change this thread's state,
		 and it might ever run again.  So write back its state.  */
	      __thread_set_state (ss->thread, MACHINE_THREAD_STATE_FLAVOR,
				  (natural_t *) &state->basic,
				  MACHINE_THREAD_STATE_COUNT);
	  }
      }

  /* Wait for replies from all the successfully interrupted RPCs.  */
  while (nthreads-- > 0)
    if (reply_ports[nthreads] != MACH_PORT_NULL)
      {
	error_t err;
	mach_msg_header_t head;
	err = __mach_msg (&head, MACH_RCV_MSG|MACH_RCV_TIMEOUT, 0, sizeof head,
			  reply_ports[nthreads],
			  _hurd_interrupted_rpc_timeout, MACH_PORT_NULL);
	switch (err)
	  {
	  case MACH_RCV_TIMED_OUT:
	  case MACH_RCV_TOO_LARGE:
	    break;

	  default:
	    assert_perror (err);
	  }
      }
}

/* Wake up any sigsuspend or pselect call that is blocking SS->thread.  SS must
   be locked.  */
static void
wake_sigsuspend (struct hurd_sigstate *ss)
{
  error_t err;
  mach_msg_header_t msg;

  if (ss->suspended == MACH_PORT_NULL)
    return;

  /* There is a sigsuspend waiting.  Tell it to wake up.  */
  msg.msgh_bits = MACH_MSGH_BITS (MACH_MSG_TYPE_MAKE_SEND, 0);
  msg.msgh_remote_port = ss->suspended;
  msg.msgh_local_port = MACH_PORT_NULL;
  /* These values do not matter.  */
  msg.msgh_id = 8675309; /* Jenny, Jenny.  */
  ss->suspended = MACH_PORT_NULL;
  err = __mach_msg (&msg, MACH_SEND_MSG, sizeof msg, 0,
      MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE,
      MACH_PORT_NULL);
  assert_perror (err);
}

struct hurd_signal_preemptor *_hurdsig_preemptors = 0;
sigset_t _hurdsig_preempted_set;

/* XXX temporary to deal with spelling fix */
weak_alias (_hurdsig_preemptors, _hurdsig_preempters)

/* Mask of stop signals.  */
#define STOPSIGS (__sigmask (SIGTTIN) | __sigmask (SIGTTOU) \
		  | __sigmask (SIGSTOP) | __sigmask (SIGTSTP))

/* Actual delivery of a single signal.  Called with SS unlocked.  When
   the signal is delivered, return SS, locked (or, if SS was originally
   _hurd_global_sigstate, the sigstate of the actual thread the signal
   was delivered to).  If the signal is being traced, return NULL with
   SS unlocked.   */
static struct hurd_sigstate *
post_signal (struct hurd_sigstate *ss,
	     int signo, struct hurd_signal_detail *detail,
	     int untraced, void (*reply) (void))
{
  struct machine_thread_all_state thread_state;
  enum { stop, ignore, core, term, handle } act;
  int ss_suspended;

  /* sigaction for preemptors */
  struct sigaction preempt_sigaction = {
    .sa_flags = SA_RESTART
  };

  struct sigaction *action;

  /* Mark the signal as pending.  */
  void mark_pending (void)
    {
      __sigaddset (&ss->pending, signo);
      /* Save the details to be given to the handler when SIGNO is
	 unblocked.  */
      ss->pending_data[signo] = *detail;
    }

  /* Suspend the process with SIGNO.  */
  void suspend (void)
    {
      /* Stop all other threads and mark ourselves stopped.  */
      __USEPORT (PROC,
		 ({
		   /* Hold the siglock while stopping other threads to be
		      sure it is not held by another thread afterwards.  */
		   __mutex_lock (&_hurd_siglock);
		   __proc_dostop (port, _hurd_msgport_thread);
		   __mutex_unlock (&_hurd_siglock);
		   abort_all_rpcs (signo, &thread_state, 1);
		   reply ();
		   __proc_mark_stop (port, signo, detail->code);
		 }));
      _hurd_stopped = 1;
    }
  /* Resume the process after a suspension.  */
  void resume (void)
    {
      /* Resume the process from being stopped.  */
      thread_t *threads;
      mach_msg_type_number_t nthreads, i;
      error_t err;

      if (! _hurd_stopped)
	return;

      /* Tell the proc server we are continuing.  */
      __USEPORT (PROC, __proc_mark_cont (port));
      /* Fetch ports to all our threads and resume them.  */
      err = __task_threads (__mach_task_self (), &threads, &nthreads);
      assert_perror (err);
      for (i = 0; i < nthreads; ++i)
	{
	  if (act == handle && threads[i] == ss->thread)
	    {
	      /* The thread that will run the handler is kept suspended.  */
	      ss_suspended = 1;
	    }
	  else if (threads[i] != _hurd_msgport_thread)
	    {
	      err = __thread_resume (threads[i]);
	      assert_perror (err);
	    }
	  err = __mach_port_deallocate (__mach_task_self (),
					threads[i]);
	  assert_perror (err);
	}
      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) threads,
		       nthreads * sizeof *threads);
      _hurd_stopped = 0;
    }

  error_t err;
  sighandler_t handler;

  if (signo == 0)
    {
      if (untraced)
	{
	  /* This is PTRACE_CONTINUE.  */
	  act = ignore;
	  resume ();
	}

      /* This call is just to check for pending signals.  */
      _hurd_sigstate_lock (ss);
      return ss;
    }

  thread_state.set = 0;		/* We know nothing.  */

  _hurd_sigstate_lock (ss);

  /* If this is a global signal, try to find a thread ready to accept
     it right away.  This is especially important for untraced signals,
     since going through the global pending mask would de-untrace them.  */
  if (ss->thread == MACH_PORT_NULL)
  {
    struct hurd_sigstate *rss;

    __mutex_lock (&_hurd_siglock);
    for (rss = _hurd_sigstates; rss != NULL; rss = rss->next)
      {
	if (! sigstate_is_global_rcv (rss))
	  continue;

	/* The global sigstate is already locked.  */
	__spin_lock (&rss->lock);
	if (! __sigismember (&rss->blocked, signo))
	  {
	    ss = rss;
	    break;
	  }
	__spin_unlock (&rss->lock);
      }
    __mutex_unlock (&_hurd_siglock);
  }

  /* We want the preemptors to be able to update the blocking mask
     without affecting the delivery of this signal, so we save the
     current value to test against later.  */
  sigset_t blocked = ss->blocked;

  /* Check for a preempted signal.  Preempted signals can arrive during
     critical sections.  */
  {
    inline sighandler_t try_preemptor (struct hurd_signal_preemptor *pe)
      {				/* PE cannot be null.  */
	do
	  {
	    if (HURD_PREEMPT_SIGNAL_P (pe, signo, detail->exc_subcode))
	      {
		if (pe->preemptor)
		  {
		    sighandler_t handler = (*pe->preemptor) (pe, ss,
							     &signo, detail);
		    if (handler != SIG_ERR)
		      return handler;
		  }
		else
		  return pe->handler;
	      }
	    pe = pe->next;
	  } while (pe != 0);
	return SIG_ERR;
      }

    handler = ss->preemptors ? try_preemptor (ss->preemptors) : SIG_ERR;

    /* If no thread-specific preemptor, check for a global one.  */
    if (handler == SIG_ERR && __sigismember (&_hurdsig_preempted_set, signo))
      {
	__mutex_lock (&_hurd_siglock);
	handler = try_preemptor (_hurdsig_preemptors);
	__mutex_unlock (&_hurd_siglock);
      }
  }

  ss_suspended = 0;

  if (handler == SIG_IGN)
    /* Ignore the signal altogether.  */
    act = ignore;
  else if (handler != SIG_ERR)
    {
      /* Run the preemption-provided handler.  */
      action = &preempt_sigaction;
      act = handle;
    }
  else
    {
      /* No preemption.  Do normal handling.  */

      action = & _hurd_sigstate_actions (ss) [signo];

      if (!untraced && __sigismember (&_hurdsig_traced, signo))
	{
	  /* We are being traced.  Stop to tell the debugger of the signal.  */
	  if (_hurd_stopped)
	    /* Already stopped.  Mark the signal as pending;
	       when resumed, we will notice it and stop again.  */
	    mark_pending ();
	  else
	    suspend ();
	  _hurd_sigstate_unlock (ss);
	  reply ();
	  return NULL;
	}

      handler = action->sa_handler;

      if (handler == SIG_DFL)
	/* Figure out the default action for this signal.  */
	switch (signo)
	  {
	  case 0:
	    /* A sig_post msg with SIGNO==0 is sent to
	       tell us to check for pending signals.  */
	    act = ignore;
	    break;

	  case SIGTTIN:
	  case SIGTTOU:
	  case SIGSTOP:
	  case SIGTSTP:
	    act = stop;
	    break;

	  case SIGCONT:
	  case SIGIO:
	  case SIGURG:
	  case SIGCHLD:
	  case SIGWINCH:
	    act = ignore;
	    break;

	  case SIGQUIT:
	  case SIGILL:
	  case SIGTRAP:
	  case SIGIOT:
	  case SIGEMT:
	  case SIGFPE:
	  case SIGBUS:
	  case SIGSEGV:
	  case SIGSYS:
	    act = core;
	    break;

	  case SIGINFO:
	    if (_hurd_pgrp == _hurd_pid)
	      {
		/* We are the process group leader.  Since there is no
		   user-specified handler for SIGINFO, we use a default one
		   which prints something interesting.  We use the normal
		   handler mechanism instead of just doing it here to avoid
		   the signal thread faulting or blocking in this
		   potentially hairy operation.  */
		act = handle;
		handler = _hurd_siginfo_handler;
	      }
	    else
	      act = ignore;
	    break;

	  default:
	    act = term;
	    break;
	  }
      else if (handler == SIG_IGN)
	act = ignore;
      else
	act = handle;

      if (__sigmask (signo) & STOPSIGS)
	/* Stop signals clear a pending SIGCONT even if they
	   are handled or ignored (but not if preempted).  */
	__sigdelset (&ss->pending, SIGCONT);
      else
	{
	  if (signo == SIGCONT)
	    /* Even if handled or ignored (but not preempted), SIGCONT clears
	       stop signals and resumes the process.  */
	    ss->pending &= ~STOPSIGS;

	  if (_hurd_stopped && act != stop && (untraced || signo == SIGCONT))
	    resume ();
	}
    }

  if (_hurd_orphaned && act == stop
      && (__sigmask (signo) & (__sigmask (SIGTTIN) | __sigmask (SIGTTOU)
			       | __sigmask (SIGTSTP))))
    {
      /* If we would ordinarily stop for a job control signal, but we are
	 orphaned so noone would ever notice and continue us again, we just
	 quietly die, alone and in the dark.  */
      detail->code = signo;
      signo = SIGKILL;
      act = term;
    }

  /* Handle receipt of a blocked signal, or any signal while stopped.  */
  if (__sigismember (&blocked, signo) || (signo != SIGKILL && _hurd_stopped))
    {
      mark_pending ();
      act = ignore;
    }

  /* Perform the chosen action for the signal.  */
  switch (act)
    {
    case stop:
      if (_hurd_stopped)
	{
	  /* We are already stopped, but receiving an untraced stop
	     signal.  Instead of resuming and suspending again, just
	     notify the proc server of the new stop signal.  */
	  error_t err = __USEPORT (PROC, __proc_mark_stop
				   (port, signo, detail->code));
	  assert_perror (err);
	}
      else
	/* Suspend the process.  */
	suspend ();
      break;

    case ignore:
      if (detail->exc)
	/* Blocking or ignoring a machine exception is fatal.
	   Otherwise we could just spin on the faulting instruction.  */
	goto fatal;

      /* Nobody cares about this signal.  If there was a call to resume
	 above in SIGCONT processing and we've left a thread suspended,
	 now's the time to set it going. */
      if (ss_suspended)
	{
	  assert (ss->thread != MACH_PORT_NULL);
	  err = __thread_resume (ss->thread);
	  assert_perror (err);
	  ss_suspended = 0;
	}
      break;

    sigbomb:
      /* We got a fault setting up the stack frame for the handler.
	 Nothing to do but die; BSD gets SIGILL in this case.  */
      detail->code = signo;	/* XXX ? */
      signo = SIGILL;

    fatal:
      act = core;
      /* FALLTHROUGH */

    case term:			/* Time to die.  */
    case core:			/* And leave a rotting corpse.  */
      /* Have the proc server stop all other threads in our task.  */
      err = __USEPORT (PROC, __proc_dostop (port, _hurd_msgport_thread));
      assert_perror (err);
      /* No more user instructions will be executed.
	 The signal can now be considered delivered.  */
      reply ();
      /* Abort all server operations now in progress.  */
      abort_all_rpcs (signo, &thread_state, 0);

      {
	int status = W_EXITCODE (0, signo);
	/* Do a core dump if desired.  Only set the wait status bit saying we
	   in fact dumped core if the operation was actually successful.  */
	if (act == core && write_corefile (signo, detail))
	  status |= WCOREFLAG;
	/* Tell proc how we died and then stick the saber in the gut.  */
	_hurd_exit (status);
	/* NOTREACHED */
      }

    case handle:
      /* Call a handler for this signal.  */
      {
	struct sigcontext *scp, ocontext;
	int wait_for_reply, state_changed;

	assert (ss->thread != MACH_PORT_NULL);

	/* Stop the thread and abort its pending RPC operations.  */
	if (! ss_suspended)
	  {
	    err = __thread_suspend (ss->thread);
	    assert_perror (err);
	  }

	/* Abort the thread's kernel context, so any pending message send
	   or receive completes immediately or aborts.  If an interruptible
	   RPC is in progress, abort_rpcs will do this.  But we must always
	   do it before fetching the thread's state, because
	   thread_get_state is never kosher before thread_abort.  */
	abort_thread (ss, &thread_state, NULL);

	if (ss->context)
	  {
	    /* We have a previous sigcontext that sigreturn was about
	       to restore when another signal arrived.  */

	    mach_port_t *loc;

	    if (_hurdsig_catch_memory_fault (ss->context))
	      {
		/* We faulted reading the thread's stack.  Forget that
		   context and pretend it wasn't there.  It almost
		   certainly crash if this handler returns, but that's it's
		   problem.  */
		ss->context = NULL;
	      }
	    else
	      {
		/* Copy the context from the thread's stack before
		   we start diddling the stack to set up the handler.  */
		ocontext = *ss->context;
		ss->context = &ocontext;
	      }
	    _hurdsig_end_catch_fault ();

	    if (! machine_get_basic_state (ss->thread, &thread_state))
	      goto sigbomb;
	    loc = interrupted_reply_port_location (ss->thread,
						   &thread_state, 1);
	    if (loc && *loc != MACH_PORT_NULL)
	      /* This is the reply port for the context which called
		 sigreturn.  Since we are abandoning that context entirely
		 and restoring SS->context instead, destroy this port.  */
	      __mach_port_destroy (__mach_task_self (), *loc);

	    /* The thread was in sigreturn, not in any interruptible RPC.  */
	    wait_for_reply = 0;

	    assert (! __spin_lock_locked (&ss->critical_section_lock));
	  }
	else
	  {
	    int crit = __spin_lock_locked (&ss->critical_section_lock);

	    wait_for_reply
	      = (_hurdsig_abort_rpcs (ss,
				      /* In a critical section, any RPC
					 should be cancelled instead of
					 restarted, regardless of
					 SA_RESTART, so the entire
					 "atomic" operation can be aborted
					 as a unit.  */
				      crit ? 0 : signo, 1,
				      &thread_state, &state_changed,
				      reply)
		 != MACH_PORT_NULL);

	    if (crit)
	      {
		/* The thread is in a critical section.  Mark the signal as
		   pending.  When it finishes the critical section, it will
		   check for pending signals.  */
		mark_pending ();
		if (state_changed)
		  /* Some cases of interrupting an RPC must change the
		     thread state to back out the call.  Normally this
		     change is rolled into the warping to the handler and
		     sigreturn, but we are not running the handler now
		     because the thread is in a critical section.  Instead,
		     mutate the thread right away for the RPC interruption
		     and resume it; the RPC will return early so the
		     critical section can end soon.  */
		  __thread_set_state (ss->thread, MACHINE_THREAD_STATE_FLAVOR,
				      (natural_t *) &thread_state.basic,
				      MACHINE_THREAD_STATE_COUNT);
		/* */
		ss->intr_port = MACH_PORT_NULL;
		__thread_resume (ss->thread);
		break;
	      }
	  }

	/* Call the machine-dependent function to set the thread up
	   to run the signal handler, and preserve its old context.  */
	scp = _hurd_setup_sighandler (ss, action, handler, signo, detail,
				      wait_for_reply, &thread_state);
	if (scp == NULL)
	  goto sigbomb;

	/* Set the machine-independent parts of the signal context.  */

	{
	  /* Fetch the thread variable for the MiG reply port,
	     and set it to MACH_PORT_NULL.  */
	  mach_port_t *loc = interrupted_reply_port_location (ss->thread,
							      &thread_state,
							      1);
	  if (loc)
	    {
	      scp->sc_reply_port = *loc;
	      *loc = MACH_PORT_NULL;
	    }
	  else
	    scp->sc_reply_port = MACH_PORT_NULL;

	  /* Save the intr_port in use by the interrupted code,
	     and clear the cell before running the trampoline.  */
	  scp->sc_intr_port = ss->intr_port;
	  ss->intr_port = MACH_PORT_NULL;

	  if (ss->context)
	    {
	      /* After the handler runs we will restore to the state in
		 SS->context, not the state of the thread now.  So restore
		 that context's reply port and intr port.  */

	      scp->sc_reply_port = ss->context->sc_reply_port;
	      scp->sc_intr_port = ss->context->sc_intr_port;

	      ss->context = NULL;
	    }
	}

	/* Backdoor extra argument to signal handler.  */
	scp->sc_error = detail->error;

	/* Block requested signals while running the handler.  */
	scp->sc_mask = ss->blocked;
	__sigorset (&ss->blocked, &ss->blocked, &action->sa_mask);

	/* Also block SIGNO unless we're asked not to.  */
	if (! (action->sa_flags & (SA_RESETHAND | SA_NODEFER)))
	  __sigaddset (&ss->blocked, signo);

	/* Reset to SIG_DFL if requested.  SIGILL and SIGTRAP cannot
           be automatically reset when delivered; the system silently
           enforces this restriction.  */
	if (action->sa_flags & SA_RESETHAND
	    && signo != SIGILL && signo != SIGTRAP)
	  action->sa_handler = SIG_DFL;

	/* Any sigsuspend call must return after the handler does.  */
	wake_sigsuspend (ss);

	/* Start the thread running the handler (or possibly waiting for an
	   RPC reply before running the handler).  */
	err = __thread_set_state (ss->thread, MACHINE_THREAD_STATE_FLAVOR,
				  (natural_t *) &thread_state.basic,
				  MACHINE_THREAD_STATE_COUNT);
	assert_perror (err);
	err = __thread_resume (ss->thread);
	assert_perror (err);
	thread_state.set = 0;	/* Everything we know is now wrong.  */
	break;
      }
    }

  return ss;
}

/* Return the set of pending signals in SS which should be delivered. */
static sigset_t
pending_signals (struct hurd_sigstate *ss)
{
  /* We don't worry about any pending signals if we are stopped, nor if
     SS is in a critical section.  We are guaranteed to get a sig_post
     message before any of them become deliverable: either the SIGCONT
     signal, or a sig_post with SIGNO==0 as an explicit poll when the
     thread finishes its critical section.  */
  if (_hurd_stopped || __spin_lock_locked (&ss->critical_section_lock))
    return 0;

  return _hurd_sigstate_pending (ss) & ~ss->blocked;
}

/* Post the specified pending signals in SS and return 1.  If one of
   them is traced, abort immediately and return 0.  SS must be locked on
   entry and will be unlocked in all cases.  */
static int
post_pending (struct hurd_sigstate *ss, sigset_t pending, void (*reply) (void))
{
  int signo;
  struct hurd_signal_detail detail;

  /* Make sure SS corresponds to an actual thread, since we assume it won't
     change in post_signal. */
  assert (ss->thread != MACH_PORT_NULL);

  for (signo = 1; signo < NSIG; ++signo)
    if (__sigismember (&pending, signo))
      {
	detail = sigstate_clear_pending (ss, signo);
	_hurd_sigstate_unlock (ss);

	/* Will reacquire the lock, except if the signal is traced.  */
	if (! post_signal (ss, signo, &detail, 0, reply))
	  return 0;
      }

  /* No more signals pending; SS->lock is still locked.  */
  _hurd_sigstate_unlock (ss);

  return 1;
}

/* Post all the pending signals of all threads and return 1.  If a traced
   signal is encountered, abort immediately and return 0.  */
static int
post_all_pending_signals (void (*reply) (void))
{
  struct hurd_sigstate *ss;
  sigset_t pending = 0;

  for (;;)
    {
      __mutex_lock (&_hurd_siglock);
      for (ss = _hurd_sigstates; ss != NULL; ss = ss->next)
        {
	  _hurd_sigstate_lock (ss);

	  pending = pending_signals (ss);
	  if (pending)
	    /* post_pending() below will unlock SS. */
	    break;

	  _hurd_sigstate_unlock (ss);
	}
      __mutex_unlock (&_hurd_siglock);

      if (! pending)
	return 1;
      if (! post_pending (ss, pending, reply))
	return 0;
    }
}

/* Deliver a signal.  SS is not locked.  */
void
_hurd_internal_post_signal (struct hurd_sigstate *ss,
			    int signo, struct hurd_signal_detail *detail,
			    mach_port_t reply_port,
			    mach_msg_type_name_t reply_port_type,
			    int untraced)
{
  /* Reply to this sig_post message.  */
  __typeof (__msg_sig_post_reply) *reply_rpc
    = (untraced ? __msg_sig_post_untraced_reply : __msg_sig_post_reply);
  void reply (void)
    {
      error_t err;
      if (reply_port == MACH_PORT_NULL)
	return;
      err = (*reply_rpc) (reply_port, reply_port_type, 0);
      reply_port = MACH_PORT_NULL;
      if (err != MACH_SEND_INVALID_DEST) /* Ignore dead reply port.  */
	assert_perror (err);
    }

  ss = post_signal (ss, signo, detail, untraced, reply);
  if (! ss)
    return;

  /* The signal was neither fatal nor traced.  We still hold SS->lock.  */
  if (signo != 0 && ss->thread != MACH_PORT_NULL)
    {
      /* The signal has either been ignored or is now being handled.  We can
	 consider it delivered and reply to the killer.  */
      reply ();

      /* Post any pending signals for this thread.  */
      if (! post_pending (ss, pending_signals (ss), reply))
	return;
    }
  else
    {
      /* If this was a process-wide signal or a poll request, we need
	 to check for pending signals for all threads.  */
      _hurd_sigstate_unlock (ss);
      if (! post_all_pending_signals (reply))
	return;

      /* All pending signals delivered to all threads.
	 Now we can send the reply message even for signal 0.  */
      reply ();
    }
}

/* Decide whether REFPORT enables the sender to send us a SIGNO signal.
   Returns zero if so, otherwise the error code to return to the sender.  */

static error_t
signal_allowed (int signo, mach_port_t refport)
{
  if (signo < 0 || signo >= NSIG)
    return EINVAL;

  if (refport == __mach_task_self ())
    /* Can send any signal.  */
    goto win;

  /* Avoid needing to check for this below.  */
  if (refport == MACH_PORT_NULL)
    return EPERM;

  switch (signo)
    {
    case SIGINT:
    case SIGQUIT:
    case SIGTSTP:
    case SIGHUP:
    case SIGINFO:
    case SIGTTIN:
    case SIGTTOU:
    case SIGWINCH:
      /* Job control signals can be sent by the controlling terminal.  */
      if (__USEPORT (CTTYID, port == refport))
	goto win;
      break;

    case SIGCONT:
      {
	/* A continue signal can be sent by anyone in the session.  */
	mach_port_t sessport;
	if (! __USEPORT (PROC, __proc_getsidport (port, &sessport)))
	  {
	    __mach_port_deallocate (__mach_task_self (), sessport);
	    if (refport == sessport)
	      goto win;
	  }
      }
      break;

    case SIGIO:
    case SIGURG:
      {
	/* Any io object a file descriptor refers to might send us
	   one of these signals using its async ID port for REFPORT.

	   This is pretty wide open; it is not unlikely that some random
	   process can at least open for reading something we have open,
	   get its async ID port, and send us a spurious SIGIO or SIGURG
	   signal.  But BSD is actually wider open than that!--you can set
	   the owner of an io object to any process or process group
	   whatsoever and send them gratuitous signals.

	   Someday we could implement some reasonable scheme for
	   authorizing SIGIO and SIGURG signals properly.  */

	int d;
	int lucky = 0;		/* True if we find a match for REFPORT.  */
	__mutex_lock (&_hurd_dtable_lock);
	for (d = 0; !lucky && (unsigned) d < (unsigned) _hurd_dtablesize; ++d)
	  {
	    struct hurd_userlink ulink;
	    io_t port;
	    mach_port_t asyncid;
	    if (_hurd_dtable[d] == NULL)
	      continue;
	    port = _hurd_port_get (&_hurd_dtable[d]->port, &ulink);
	    if (! __io_get_icky_async_id (port, &asyncid))
	      {
		if (refport == asyncid)
		  /* Break out of the loop on the next iteration.  */
		  lucky = 1;
		__mach_port_deallocate (__mach_task_self (), asyncid);
	      }
	    _hurd_port_free (&_hurd_dtable[d]->port, &ulink, port);
	  }
	__mutex_unlock (&_hurd_dtable_lock);
	/* If we found a lucky winner, we've set D to -1 in the loop.  */
	if (lucky)
	  goto win;
      }
    }

  /* If this signal is legit, we have done `goto win' by now.
     When we return the error, mig deallocates REFPORT.  */
  return EPERM;

 win:
  /* Deallocate the REFPORT send right; we are done with it.  */
  __mach_port_deallocate (__mach_task_self (), refport);

  return 0;
}

/* Implement the sig_post RPC from <hurd/msg.defs>;
   sent when someone wants us to get a signal.  */
kern_return_t
_S_msg_sig_post (mach_port_t me,
		 mach_port_t reply_port, mach_msg_type_name_t reply_port_type,
		 int signo, natural_t sigcode,
		 mach_port_t refport)
{
  error_t err;
  struct hurd_signal_detail d;

  if (err = signal_allowed (signo, refport))
    return err;

  d.code = d.exc_subcode = sigcode;
  d.exc = 0;

  /* Post the signal to a global receiver thread (or mark it pending in
     the global sigstate).  This will reply when the signal can be
     considered delivered.  */
  _hurd_internal_post_signal (_hurd_global_sigstate,
			      signo, &d, reply_port, reply_port_type,
			      0); /* Stop if traced.  */

  return MIG_NO_REPLY;		/* Already replied.  */
}

/* Implement the sig_post_untraced RPC from <hurd/msg.defs>;
   sent when the debugger wants us to really get a signal
   even if we are traced.  */
kern_return_t
_S_msg_sig_post_untraced (mach_port_t me,
			  mach_port_t reply_port,
			  mach_msg_type_name_t reply_port_type,
			  int signo, natural_t sigcode,
			  mach_port_t refport)
{
  error_t err;
  struct hurd_signal_detail d;

  if (err = signal_allowed (signo, refport))
    return err;

  d.code = d.exc_subcode = sigcode;
  d.exc = 0;

  /* Post the signal to the designated signal-receiving thread.  This will
     reply when the signal can be considered delivered.  */
  _hurd_internal_post_signal (_hurd_global_sigstate,
			      signo, &d, reply_port, reply_port_type,
			      1); /* Untraced flag. */

  return MIG_NO_REPLY;		/* Already replied.  */
}

extern void __mig_init (void *);

#include <mach/task_special_ports.h>

/* Initialize the message port, _hurd_global_sigstate, and start the
   signal thread.  */

void
_hurdsig_init (const int *intarray, size_t intarraysize)
{
  error_t err;
  vm_size_t stacksize;
  struct hurd_sigstate *ss;

  __mutex_init (&_hurd_siglock);

  err = __mach_port_allocate (__mach_task_self (),
			      MACH_PORT_RIGHT_RECEIVE,
			      &_hurd_msgport);
  assert_perror (err);

  /* Make a send right to the signal port.  */
  err = __mach_port_insert_right (__mach_task_self (),
				  _hurd_msgport,
				  _hurd_msgport,
				  MACH_MSG_TYPE_MAKE_SEND);
  assert_perror (err);

  /* Initialize the global signal state.  */
  _hurd_global_sigstate = _hurd_thread_sigstate (MACH_PORT_NULL);

  /* We block all signals, and let actual threads pull them from the
     pending mask.  */
  __sigfillset(& _hurd_global_sigstate->blocked);

  /* Initialize the main thread's signal state.  */
  ss = _hurd_self_sigstate ();

  /* Mark it as a process-wide signal receiver.  Threads in this set use
     the common action vector in _hurd_global_sigstate.  */
  _hurd_sigstate_set_global_rcv (ss);

  /* Copy inherited signal settings from our parent (or pre-exec process
     state) */
  if (intarraysize > INIT_SIGMASK)
    ss->blocked = intarray[INIT_SIGMASK];
  if (intarraysize > INIT_SIGPENDING)
    _hurd_global_sigstate->pending = intarray[INIT_SIGPENDING];
  if (intarraysize > INIT_SIGIGN && intarray[INIT_SIGIGN] != 0)
    {
      int signo;
      for (signo = 1; signo < NSIG; ++signo)
	if (intarray[INIT_SIGIGN] & __sigmask(signo))
	  _hurd_global_sigstate->actions[signo].sa_handler = SIG_IGN;
    }

  /* Start the signal thread listening on the message port.  */

#pragma weak __pthread_create
  if (!__pthread_create)
    {
      err = __thread_create (__mach_task_self (), &_hurd_msgport_thread);
      assert_perror (err);

      stacksize = __vm_page_size * 8; /* Small stack for signal thread.  */
      err = __mach_setup_thread (__mach_task_self (), _hurd_msgport_thread,
				 _hurd_msgport_receive,
				 (vm_address_t *) &__hurd_sigthread_stack_base,
				 &stacksize);
      assert_perror (err);
      err = __mach_setup_tls (_hurd_msgport_thread);
      assert_perror (err);

      __hurd_sigthread_stack_end = __hurd_sigthread_stack_base + stacksize;

      /* Reinitialize the MiG support routines so they will use a per-thread
	 variable for the cached reply port.  */
      __mig_init ((void *) __hurd_sigthread_stack_base);

      err = __thread_resume (_hurd_msgport_thread);
      assert_perror (err);
    }
  else
    {
      pthread_t thread;
      pthread_attr_t attr;
      void *addr;
      size_t size;

      /* When pthread is being used, we need to make the signal thread a
         proper pthread.  Otherwise it cannot use mutex_lock et al, which
         will be the pthread versions.  Various of the message port RPC
         handlers need to take locks, so we need to be able to call into
         pthread code and meet its assumptions about how our thread and
         its stack are arranged.  Since pthread puts it there anyway,
         we'll let the signal thread's per-thread variables be found as for
         any normal pthread, and just leave the magic __hurd_sigthread_*
         values all zero so they'll be ignored.  */

#pragma weak __pthread_detach
#pragma weak __pthread_getattr_np
#pragma weak __pthread_attr_getstack
      __pthread_create(&thread, NULL, &_hurd_msgport_receive, NULL);

      /* Record signal thread stack layout for fork() */
      __pthread_getattr_np (thread, &attr);
      __pthread_attr_getstack (&attr, &addr, &size);
      __hurd_sigthread_stack_base = (uintptr_t) addr;
      __hurd_sigthread_stack_end = __hurd_sigthread_stack_base + size;

      __pthread_detach(thread);

      /* XXX We need the thread port for the signal thread further on
         in this thread (see hurdfault.c:_hurdsigfault_init).
         Therefore we block until _hurd_msgport_thread is initialized
         by the newly created thread.  This really shouldn't be
         necessary; we should be able to fetch the thread port for a
         pthread from here.  */
      while (_hurd_msgport_thread == 0)
	__swtch_pri (0);
    }

  /* Receive exceptions on the signal port.  */
#ifdef TASK_EXCEPTION_PORT
  __task_set_special_port (__mach_task_self (),
			   TASK_EXCEPTION_PORT, _hurd_msgport);
#elif defined (EXC_MASK_ALL)
  __task_set_exception_ports (__mach_task_self (),
			      EXC_MASK_ALL & ~(EXC_MASK_SYSCALL
					       | EXC_MASK_MACH_SYSCALL
					       | EXC_MASK_RPC_ALERT),
			      _hurd_msgport,
			      EXCEPTION_DEFAULT, MACHINE_THREAD_STATE);
#else
# error task_set_exception_port?
#endif

  /* Sanity check.  Any pending, unblocked signals should have been
     taken by our predecessor incarnation (i.e. parent or pre-exec state)
     before packing up our init ints.  This assert is last (not above)
     so that signal handling is all set up to handle the abort.  */
  assert ((ss->pending &~ ss->blocked) == 0);
}
				/* XXXX */
/* Reauthenticate with the proc server.  */

static void
reauth_proc (mach_port_t new)
{
  mach_port_t ref, ignore;

  ref = __mach_reply_port ();
  if (! HURD_PORT_USE (&_hurd_ports[INIT_PORT_PROC],
		       __proc_reauthenticate (port, ref,
					      MACH_MSG_TYPE_MAKE_SEND)
		       || __auth_user_authenticate (new, ref,
						    MACH_MSG_TYPE_MAKE_SEND,
						    &ignore))
      && ignore != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), ignore);
  __mach_port_destroy (__mach_task_self (), ref);

  /* Set the owner of the process here too. */
  __mutex_lock (&_hurd_id.lock);
  if (!_hurd_check_ids ())
    HURD_PORT_USE (&_hurd_ports[INIT_PORT_PROC],
		   __proc_setowner (port,
				    (_hurd_id.gen.nuids
				     ? _hurd_id.gen.uids[0] : 0),
				    !_hurd_id.gen.nuids));
  __mutex_unlock (&_hurd_id.lock);

  (void) &reauth_proc;		/* Silence compiler warning.  */
}
text_set_element (_hurd_reauth_hook, reauth_proc);

/* Like `getenv', but safe for the signal thread to run.
   If the environment is trashed, this will just return NULL.  */

const char *
_hurdsig_getenv (const char *variable)
{
  if (__libc_enable_secure)
    return NULL;

  if (_hurdsig_catch_memory_fault (__environ))
    /* We bombed in getenv.  */
    return NULL;
  else
    {
      const size_t len = strlen (variable);
      char *value = NULL;
      char *volatile *ep = __environ;
      while (*ep)
	{
	  const char *p = *ep;
	  _hurdsig_fault_preemptor.first = (long int) p;
	  _hurdsig_fault_preemptor.last = VM_MAX_ADDRESS;
	  if (! strncmp (p, variable, len) && p[len] == '=')
	    {
	      size_t valuelen;
	      p += len + 1;
	      valuelen = strlen (p);
	      _hurdsig_fault_preemptor.last = (long int) (p + valuelen);
	      value = malloc (++valuelen);
	      if (value)
		memcpy (value, p, valuelen);
	      break;
	    }
	  _hurdsig_fault_preemptor.first = (long int) ++ep;
	  _hurdsig_fault_preemptor.last = (long int) (ep + 1);
	}
      _hurdsig_end_catch_fault ();
      return value;
    }
}

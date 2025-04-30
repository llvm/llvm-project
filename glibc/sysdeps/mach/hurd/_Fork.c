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

#include <errno.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/signal.h>
#include <hurd/threadvar.h>
#include <setjmp.h>
#include <thread_state.h>
#include <sysdep.h>		/* For stack growth direction.  */
#include "set-hooks.h"
#include <assert.h>
#include "hurdmalloc.h"		/* XXX */
#include <tls.h>
#include <malloc/malloc-internal.h>
#include <nss/nss_database.h>
#include <unwind-link.h>
#include <register-atfork.h>

#undef __fork


/* Things that want to be locked while forking.  */
symbol_set_declare (_hurd_fork_locks)

/* Things that want to be called before we fork, to prepare the parent for
   task_create, when the new child task will inherit our address space.  */
DEFINE_HOOK (_hurd_fork_prepare_hook, (void));

/* Things that want to be called when we are forking, with the above all
   locked.  They are passed the task port of the child.  The child process
   is all set up except for doing proc_child, and has no threads yet.  */
DEFINE_HOOK (_hurd_fork_setup_hook, (void));

/* Things to be run in the child fork.  */
DEFINE_HOOK (_hurd_fork_child_hook, (void));

/* Things to be run in the parent fork.  */
DEFINE_HOOK (_hurd_fork_parent_hook, (void));


/* Clone the calling process, creating an exact copy.
   Return -1 for errors, 0 to the new process,
   and the process ID of the new process to the old process.  */
pid_t
_Fork (void)
{
  jmp_buf env;
  pid_t pid;
  size_t i;
  error_t err;
  struct hurd_sigstate *volatile ss;

  ss = _hurd_self_sigstate ();
retry:
  __spin_lock (&ss->critical_section_lock);

#undef	LOSE
#define LOSE do { assert_perror (err); goto lose; } while (0) /* XXX */

  if (! setjmp (env))
    {
      process_t newproc;
      task_t newtask;
      thread_t thread, sigthread;
      mach_port_urefs_t thread_refs, sigthread_refs;
      struct machine_thread_state state;
      mach_msg_type_number_t statecount;
      mach_port_t *portnames = NULL;
      mach_msg_type_number_t nportnames = 0;
      mach_port_type_t *porttypes = NULL;
      mach_msg_type_number_t nporttypes = 0;
      thread_t *threads = NULL;
      mach_msg_type_number_t nthreads = 0;
      int ports_locked = 0, stopped = 0;

      void resume_threads (void)
	{
	  if (! stopped)
	    return;

	  assert (threads);

	  for (i = 0; i < nthreads; ++i)
	    if (threads[i] != ss->thread)
	      __thread_resume (threads[i]);
	  stopped = 0;
	}

      /* Run things that prepare for forking before we create the task.  */
      RUN_HOOK (_hurd_fork_prepare_hook, ());

      /* Lock things that want to be locked before we fork.  */
      {
	void *const *p;
	for (p = symbol_set_first_element (_hurd_fork_locks);
	     ! symbol_set_end_p (_hurd_fork_locks, p);
	     ++p)
	  __mutex_lock (*p);
      }
      __mutex_lock (&_hurd_siglock);

      /* Acquire malloc locks.  This needs to come last because fork
	 handlers may use malloc, and the libio list lock has an
	 indirect malloc dependency as well (via the getdelim
	 function).  */
      _hurd_malloc_fork_prepare ();

      newtask = MACH_PORT_NULL;
      thread = sigthread = MACH_PORT_NULL;
      newproc = MACH_PORT_NULL;

      /* Lock all the port cells for the standard ports while we copy the
	 address space.  We want to insert all the send rights into the
	 child with the same names.  */
      for (i = 0; i < _hurd_nports; ++i)
	__spin_lock (&_hurd_ports[i].lock);
      ports_locked = 1;


      /* Keep our SS locked while stopping other threads, so they don't get a
         chance to have it locked in the copied space.  */
      __spin_lock (&ss->lock);
      /* Stop all other threads while copying the address space,
	 so nothing changes.  */
      err = __proc_dostop (_hurd_ports[INIT_PORT_PROC].port, ss->thread);
      __spin_unlock (&ss->lock);
      if (!err)
	{
	  stopped = 1;

#define XXX_KERNEL_PAGE_FAULT_BUG /* XXX work around page fault bug in mk */

#ifdef XXX_KERNEL_PAGE_FAULT_BUG
	  /* Gag me with a pitchfork.
	     The bug scenario is this:

	     - The page containing __mach_task_self_ is paged out.
	     - The signal thread was faulting on that page when we
	       suspended it via proc_dostop.  It holds some lock, or set
	       some busy bit, or somesuch.
	     - Now this thread faults on that same page.
	     - GRATUIOUS DEADLOCK

	     We can break the deadlock by aborting the thread that faulted
	     first, which if the bug happened was the signal thread because
	     it is the only other thread and we just suspended it.
	     */
	  __thread_abort (_hurd_msgport_thread);
#endif
	  /* Create the child task.  It will inherit a copy of our memory.  */
	  err = __task_create (__mach_task_self (),
#ifdef KERN_INVALID_LEDGER
			       NULL, 0,	/* OSF Mach */
#endif
			       1, &newtask);
	}

      /* Unlock the global signal state lock, so we do not
	 block the signal thread any longer than necessary.  */
      __mutex_unlock (&_hurd_siglock);

      if (err)
	LOSE;

      /* Fetch the names of all ports used in this task.  */
      if (err = __mach_port_names (__mach_task_self (),
				   &portnames, &nportnames,
				   &porttypes, &nporttypes))
	LOSE;
      if (nportnames != nporttypes)
	{
	  err = EGRATUITOUS;
	  LOSE;
	}

      /* Get send rights for all the threads in this task.
	 We want to avoid giving these rights to the child.  */
      if (err = __task_threads (__mach_task_self (), &threads, &nthreads))
	LOSE;

      /* Get the child process's proc server port.  We will insert it into
	 the child with the same name as we use for our own proc server
	 port; and we will need it to set the child's message port.  */
      if (err = __proc_task2proc (_hurd_ports[INIT_PORT_PROC].port,
				  newtask, &newproc))
	LOSE;

      /* Insert all our port rights into the child task.  */
      thread_refs = sigthread_refs = 0;
      for (i = 0; i < nportnames; ++i)
	{
	  if (porttypes[i] & MACH_PORT_TYPE_RECEIVE)
	    {
	      /* This is a receive right.  We want to give the child task
		 its own new receive right under the same name.  */
	      if (err = __mach_port_allocate_name (newtask,
						   MACH_PORT_RIGHT_RECEIVE,
						   portnames[i]))
		LOSE;
	      if (porttypes[i] & MACH_PORT_TYPE_SEND)
		{
		  /* Give the child as many send rights for its receive
		     right as we have for ours.  */
		  mach_port_urefs_t refs;
		  mach_port_t port;
		  mach_msg_type_name_t poly;
		  if (err = __mach_port_get_refs (__mach_task_self (),
						  portnames[i],
						  MACH_PORT_RIGHT_SEND,
						  &refs))
		    LOSE;
		  if (err = __mach_port_extract_right (newtask,
						       portnames[i],
						       MACH_MSG_TYPE_MAKE_SEND,
						       &port, &poly))
		    LOSE;
		  if (portnames[i] == _hurd_msgport)
		    {
		      /* We just created a receive right for the child's
			 message port and are about to insert send rights
			 for it.  Now, while we happen to have a send right
			 for it, give it to the proc server.  */
		      mach_port_t old;
		      if (err = __proc_setmsgport (newproc, port, &old))
			LOSE;
		      if (old != MACH_PORT_NULL)
			/* XXX what to do here? */
			__mach_port_deallocate (__mach_task_self (), old);
		      /* The new task will receive its own exceptions
			 on its message port.  */
		      if (err =
#ifdef TASK_EXCEPTION_PORT
			  __task_set_special_port (newtask,
						   TASK_EXCEPTION_PORT,
						   port)
#elif defined (EXC_MASK_ALL)
			  __task_set_exception_ports
			  (newtask, EXC_MASK_ALL & ~(EXC_MASK_SYSCALL
						     | EXC_MASK_MACH_SYSCALL
						     | EXC_MASK_RPC_ALERT),
			   port, EXCEPTION_DEFAULT, MACHINE_THREAD_STATE)
#else
# error task_set_exception_port?
#endif
			  )
			LOSE;
		    }
		  if (err = __mach_port_insert_right (newtask,
						      portnames[i],
						      port,
						      MACH_MSG_TYPE_MOVE_SEND))
		    LOSE;
		  if (refs > 1
		      && (err = __mach_port_mod_refs (newtask,
						      portnames[i],
						      MACH_PORT_RIGHT_SEND,
						      refs - 1)))
		    LOSE;
		}
	      if (porttypes[i] & MACH_PORT_TYPE_SEND_ONCE)
		{
		  /* Give the child a send-once right for its receive right,
		     since we have one for ours.  */
		  mach_port_t port;
		  mach_msg_type_name_t poly;
		  if (err = __mach_port_extract_right
		      (newtask,
		       portnames[i],
		       MACH_MSG_TYPE_MAKE_SEND_ONCE,
		       &port, &poly))
		    LOSE;
		  if (err = __mach_port_insert_right
		      (newtask,
		       portnames[i], port,
		       MACH_MSG_TYPE_MOVE_SEND_ONCE))
		    LOSE;
		}
	    }
	  else if (porttypes[i]
		   & (MACH_PORT_TYPE_SEND|MACH_PORT_TYPE_DEAD_NAME))
	    {
	      /* This is a send right or a dead name.
		 Give the child as many references for it as we have.  */
	      mach_port_urefs_t refs = 0, *record_refs = NULL;
	      mach_port_t insert;
	      mach_msg_type_name_t insert_type = MACH_MSG_TYPE_COPY_SEND;
	      if (portnames[i] == newtask || portnames[i] == newproc)
		/* Skip the name we use for the child's task or proc ports.  */
		continue;
	      if (portnames[i] == __mach_task_self ())
		/* For the name we use for our own task port,
		   insert the child's task port instead.  */
		insert = newtask;
	      else if (portnames[i] == _hurd_ports[INIT_PORT_PROC].port)
		{
		  /* Use the proc server port for the new task.  */
		  insert = newproc;
		  insert_type = MACH_MSG_TYPE_COPY_SEND;
		}
	      else if (portnames[i] == ss->thread)
		{
		  /* For the name we use for our own thread port, we will
		     insert the thread port for the child main user thread
		     after we create it.  */
		  insert = MACH_PORT_NULL;
		  record_refs = &thread_refs;
		  /* Allocate a dead name right for this name as a
		     placeholder, so the kernel will not chose this name
		     for any other new port (it might use it for one of the
		     rights created when a thread is created).  */
		  if (err = __mach_port_allocate_name
		      (newtask, MACH_PORT_RIGHT_DEAD_NAME, portnames[i]))
		    LOSE;
		}
	      else if (portnames[i] == _hurd_msgport_thread)
		/* For the name we use for our signal thread's thread port,
		   we will insert the thread port for the child's signal
		   thread after we create it.  */
		{
		  insert = MACH_PORT_NULL;
		  record_refs = &sigthread_refs;
		  /* Allocate a dead name right as a placeholder.  */
		  if (err = __mach_port_allocate_name
		      (newtask, MACH_PORT_RIGHT_DEAD_NAME, portnames[i]))
		    LOSE;
		}
	      else
		{
		  /* Skip the name we use for any of our own thread ports.  */
		  mach_msg_type_number_t j;
		  for (j = 0; j < nthreads; ++j)
		    if (portnames[i] == threads[j])
		      break;
		  if (j < nthreads)
		    continue;

		  /* Copy our own send right.  */
		  insert = portnames[i];
		}
	      /* Find out how many user references we have for
		 the send right with this name.  */
	      if (err = __mach_port_get_refs (__mach_task_self (),
					      portnames[i],
					      MACH_PORT_RIGHT_SEND,
					      record_refs ?: &refs))
		LOSE;
	      if (insert == MACH_PORT_NULL)
		continue;
	      if (insert == portnames[i]
		  && (porttypes[i] & MACH_PORT_TYPE_DEAD_NAME))
		/* This is a dead name; allocate another dead name
		   with the same name in the child.  */
	      allocate_dead_name:
		err = __mach_port_allocate_name (newtask,
						 MACH_PORT_RIGHT_DEAD_NAME,
						 portnames[i]);
	      else
		/* Insert the chosen send right into the child.  */
		err = __mach_port_insert_right (newtask,
						portnames[i],
						insert, insert_type);
	      switch (err)
		{
		case KERN_NAME_EXISTS:
		  {
		    /* It already has a send right under this name (?!).
		       Well, it starts out with a send right for its task
		       port, and inherits the bootstrap and exception ports
		       from us.  */
		    mach_port_t childport;
		    mach_msg_type_name_t poly;
		    assert (__mach_port_extract_right (newtask, portnames[i],
						       MACH_MSG_TYPE_COPY_SEND,
						       &childport,
						       &poly) == 0
			    && childport == insert
			    && __mach_port_deallocate (__mach_task_self (),
						       childport) == 0);
		    break;
		  }

		case KERN_INVALID_CAPABILITY:
		  /* The port just died.  It was a send right,
		     and now it's a dead name.  */
		  goto allocate_dead_name;

		default:
		  LOSE;
		  break;

		case KERN_SUCCESS:
		  /* Give the child as many user references as we have.  */
		  if (refs > 1
		      && (err = __mach_port_mod_refs (newtask,
						      portnames[i],
						      MACH_PORT_RIGHT_SEND,
						      refs - 1)))
		    LOSE;
		}
	    }
	}

      /* Unlock the standard port cells.  The child must unlock its own
	 copies too.  */
      for (i = 0; i < _hurd_nports; ++i)
	__spin_unlock (&_hurd_ports[i].lock);
      ports_locked = 0;

      /* All state has now been copied from the parent.  It is safe to
	 resume other parent threads.  */
      resume_threads ();

      /* Create the child main user thread and signal thread.  */
      if ((err = __thread_create (newtask, &thread))
	  || (err = __thread_create (newtask, &sigthread)))
	LOSE;

      /* Insert send rights for those threads.  We previously allocated
	 dead name rights with the names we want to give the thread ports
	 in the child as placeholders.  Now deallocate them so we can use
	 the names.  */
      if ((err = __mach_port_deallocate (newtask, ss->thread))
	  || (err = __mach_port_insert_right (newtask, ss->thread,
					      thread,
					      MACH_MSG_TYPE_COPY_SEND)))
	LOSE;
      /* XXX consumed? (_hurd_sigthread is no more) */
      if (thread_refs > 1
	  && (err = __mach_port_mod_refs (newtask, ss->thread,
					  MACH_PORT_RIGHT_SEND,
					  thread_refs - 1)))
	LOSE;
      if ((_hurd_msgport_thread != MACH_PORT_NULL) /* Let user have none.  */
	  && ((err = __mach_port_deallocate (newtask, _hurd_msgport_thread))
	      || (err = __mach_port_insert_right (newtask,
						  _hurd_msgport_thread,
						  sigthread,
						  MACH_MSG_TYPE_COPY_SEND))))
	LOSE;
      if (sigthread_refs > 1
	  && (err = __mach_port_mod_refs (newtask, _hurd_msgport_thread,
					  MACH_PORT_RIGHT_SEND,
					  sigthread_refs - 1)))
	LOSE;

      /* This seems like a convenient juncture to copy the proc server's
	 idea of what addresses our argv and envp are found at from the
	 parent into the child.  Since we happen to know that the child
	 shares our memory image, it is we who should do this copying.  */
      {
	vm_address_t argv, envp;
	err = (__USEPORT (PROC, __proc_get_arg_locations (port, &argv, &envp))
	       ?: __proc_set_arg_locations (newproc, argv, envp));
	if (err)
	  LOSE;
      }

      /* Set the child signal thread up to run the msgport server function
	 using the same signal thread stack copied from our address space.
	 We fetch the state before longjmp'ing it so that miscellaneous
	 registers not affected by longjmp (such as i386 segment registers)
	 are in their normal default state.  */
      statecount = MACHINE_THREAD_STATE_COUNT;
      if (err = __thread_get_state (_hurd_msgport_thread,
				    MACHINE_THREAD_STATE_FLAVOR,
				    (natural_t *) &state, &statecount))
	LOSE;
#ifdef STACK_GROWTH_UP
      if (__hurd_sigthread_stack_base == 0)
	{
	  state.SP &= __hurd_threadvar_stack_mask;
	  state.SP += __hurd_threadvar_stack_offset;
	}
      else
	state.SP = __hurd_sigthread_stack_base;
#else
      if (__hurd_sigthread_stack_end == 0)
	{
	  /* The signal thread has a stack assigned by pthread.
	     The threadvar_stack variables conveniently tell us how
	     to get to the highest address in the stack, just below
	     the per-thread variables.  */
	  state.SP &= __hurd_threadvar_stack_mask;
	  state.SP += __hurd_threadvar_stack_offset;
	}
      else
	state.SP = __hurd_sigthread_stack_end;
#endif
      MACHINE_THREAD_STATE_SET_PC (&state,
				   (unsigned long int) _hurd_msgport_receive);

      /* Do special signal thread setup for TLS if needed.  */
      if (err = _hurd_tls_fork (sigthread, _hurd_msgport_thread, &state))
	LOSE;

      if (err = __thread_set_state (sigthread, MACHINE_THREAD_STATE_FLAVOR,
				    (natural_t *) &state, statecount))
	LOSE;
      /* We do not thread_resume SIGTHREAD here because the child
	 fork needs to do more setup before it can take signals.  */

      /* Set the child user thread up to return 1 from the setjmp above.  */
      _hurd_longjmp_thread_state (&state, env, 1);

      /* Do special thread setup for TLS if needed.  */
      if (err = _hurd_tls_fork (thread, ss->thread, &state))
	LOSE;

      if (err = __thread_set_state (thread, MACHINE_THREAD_STATE_FLAVOR,
				    (natural_t *) &state, statecount))
	LOSE;

      /* Get the PID of the child from the proc server.  We must do this
	 before calling proc_child below, because at that point any
	 authorized POSIX.1 process may kill the child task with SIGKILL.  */
      if (err = __USEPORT (PROC, __proc_task2pid (port, newtask, &pid)))
	LOSE;

      /* Register the child with the proc server.  It is important that
	 this be that last thing we do before starting the child thread
	 running.  Once proc_child has been done for the task, it appears
	 as a POSIX.1 process.  Any errors we get must be detected before
	 this point, and the child must have a message port so it responds
	 to POSIX.1 signals.  */
      if (err = __USEPORT (PROC, __proc_child (port, newtask)))
	LOSE;

      /* This must be the absolutely last thing we do; we can't assume that
	 the child will remain alive for even a moment once we do this.  We
	 ignore errors because we have committed to the fork and are not
	 allowed to return them after the process becomes visible to
	 POSIX.1 (which happened right above when we called proc_child).  */
      (void) __thread_resume (thread);

    lose:
      if (ports_locked)
	for (i = 0; i < _hurd_nports; ++i)
	  __spin_unlock (&_hurd_ports[i].lock);

      resume_threads ();

      if (newtask != MACH_PORT_NULL)
	{
	  if (err)
	    __task_terminate (newtask);
	  __mach_port_deallocate (__mach_task_self (), newtask);
	}
      if (thread != MACH_PORT_NULL)
	__mach_port_deallocate (__mach_task_self (), thread);
      if (sigthread != MACH_PORT_NULL)
	__mach_port_deallocate (__mach_task_self (), sigthread);
      if (newproc != MACH_PORT_NULL)
	__mach_port_deallocate (__mach_task_self (), newproc);

      if (portnames)
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) portnames,
			 nportnames * sizeof (*portnames));
      if (porttypes)
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) porttypes,
			 nporttypes * sizeof (*porttypes));
      if (threads)
	{
	  for (i = 0; i < nthreads; ++i)
	    __mach_port_deallocate (__mach_task_self (), threads[i]);
	  __vm_deallocate (__mach_task_self (),
			   (vm_address_t) threads,
			   nthreads * sizeof (*threads));
	}

      /* Release malloc locks.  */
      _hurd_malloc_fork_parent ();

      /* Run things that want to run in the parent to restore it to
	 normality.  Usually prepare hooks and parent hooks are
	 symmetrical: the prepare hook arrests state in some way for the
	 fork, and the parent hook restores the state for the parent to
	 continue executing normally.  */
      RUN_HOOK (_hurd_fork_parent_hook, ());
    }
  else
    {
      struct hurd_sigstate *oldstates;

      /* We are the child task.  Unlock the standard port cells, which were
	 locked in the parent when we copied its memory.  The parent has
	 inserted send rights with the names that were in the cells then.  */
      for (i = 0; i < _hurd_nports; ++i)
	__spin_unlock (&_hurd_ports[i].lock);

      /* Claim our sigstate structure and unchain the rest: the
	 threads existed in the parent task but don't exist in this
	 task (the child process).  Delay freeing them until later
	 because some of the further setup and unlocking might be
	 required for free to work.  Before we finish cleaning up,
	 we will reclaim the signal thread's sigstate structure (if
	 it had one).  */
      oldstates = _hurd_sigstates;
      if (oldstates == ss)
	oldstates = ss->next;
      else
	{
	  while (_hurd_sigstates->next != ss)
	    _hurd_sigstates = _hurd_sigstates->next;
	  _hurd_sigstates->next = ss->next;
	}
      ss->next = NULL;
      _hurd_sigstates = ss;
      __mutex_unlock (&_hurd_siglock);
      /* Earlier on, the global sigstate may have been tainted and now needs to
         be reinitialized.  Nobody is interested in its present state anymore:
         we're not, the signal thread will be restarted, and there are no other
         threads.

         We can't simply allocate a fresh global sigstate here, as
         _hurd_thread_sigstate will call malloc and that will deadlock trying
         to determine the current thread's sigstate.  */
#if 0
      _hurd_thread_sigstate_init (_hurd_global_sigstate, MACH_PORT_NULL);
#else
      /* Only reinitialize the lock -- otherwise we might have to do additional
         setup as done in hurdsig.c:_hurdsig_init.  */
      __spin_lock_init (&_hurd_global_sigstate->lock);
#endif

      /* We are one of the (exactly) two threads in this new task, we
	 will take the task-global signals.  */
      _hurd_sigstate_set_global_rcv (ss);

      /* Fetch our new process IDs from the proc server.  No need to
	 refetch our pgrp; it is always inherited from the parent (so
	 _hurd_pgrp is already correct), and the proc server will send us a
	 proc_newids notification when it changes.  */
      err = __USEPORT (PROC, __proc_getpids (port, &_hurd_pid, &_hurd_ppid,
					     &_hurd_orphaned));

      /* Forking clears the trace flag and pending masks.  */
      __sigemptyset (&_hurdsig_traced);
      __sigemptyset (&_hurd_global_sigstate->pending);
      __sigemptyset (&ss->pending);

      __libc_unwind_link_after_fork ();

      /* Release malloc locks.  */
      _hurd_malloc_fork_child ();
      call_function_static_weak (__malloc_fork_unlock_child);

      /* Run things that want to run in the child task to set up.  */
      RUN_HOOK (_hurd_fork_child_hook, ());

      /* Set up proc server-assisted fault recovery for the signal thread.  */
      _hurdsig_fault_init ();

      /* Start the signal thread listening on the message port.  */
      if (!err)
	err = __thread_resume (_hurd_msgport_thread);

      /* Reclaim the signal thread's sigstate structure and free the
	 other old sigstate structures.  */
      while (oldstates != NULL)
	{
	  struct hurd_sigstate *next = oldstates->next;

	  if (oldstates->thread == _hurd_msgport_thread)
	    {
	      /* If we have a second signal state structure then we
		 must have been through here before--not good.  */
	      assert (_hurd_sigstates->next == 0);
	      _hurd_sigstates->next = oldstates;
	      oldstates->next = 0;
	    }
	  else
	    free (oldstates);

	  oldstates = next;
	}

      /* XXX what to do if we have any errors here? */

      pid = 0;
    }

  /* Unlock things we locked before creating the child task.
     They are locked in both the parent and child tasks.  */
  {
    void *const *p;
    for (p = symbol_set_first_element (_hurd_fork_locks);
	 ! symbol_set_end_p (_hurd_fork_locks, p);
	 ++p)
      __mutex_unlock (*p);
  }

  _hurd_critical_section_unlock (ss);
  if (err == EINTR)
    /* Got a signal while inside an RPC of the critical section, retry again */
    goto retry;

  return err ? __hurd_fail (err) : pid;
}
libc_hidden_def (_Fork)

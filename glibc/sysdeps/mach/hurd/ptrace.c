/* Process tracing interface `ptrace' for GNU Hurd.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/ptrace.h>
#include <sys/types.h>
#include <stdarg.h>
#include <hurd.h>
#include <hurd/signal.h>
#include <hurd/msg.h>
#include <thread_state.h>

/* Perform process tracing functions.  REQUEST is one of the values
   in <sys/ptrace.h>, and determines the action to be taken.
   For all requests except PTRACE_TRACEME, PID specifies the process to be
   traced.

   PID and the other arguments described above for the various requests should
   appear (those that are used for the particular request) as:
     pid_t PID, void *ADDR, int DATA, void *ADDR2
   after PID.  */
int
ptrace (enum __ptrace_request request, ... )
{
  pid_t pid;
  void *addr, *addr2;
  natural_t data;
  va_list ap;

  /* Read data from PID's address space, from ADDR for DATA bytes.  */
  error_t read_data (task_t task, vm_address_t *ourpage, vm_size_t *size)
    {
      /* Read the pages containing the addressed range.  */
      error_t err;
      *size = round_page (addr + data) - trunc_page (addr);
      err = __vm_read (task, trunc_page (addr), *size, ourpage, size);
      return err;
    }

  /* Fetch the thread port for PID's user thread.  */
  error_t fetch_user_thread (task_t task, thread_t *thread)
    {
      thread_t threadbuf[3], *threads = threadbuf;
      mach_msg_type_number_t nthreads = 3, i;
      error_t err = __task_threads (task, &threads, &nthreads);
      if (err)
	return err;
      if (nthreads == 0)
	return EINVAL;
      *thread = threads[0];	/* Assume user thread is first.  */
      for (i = 1; i < nthreads; ++i)
	__mach_port_deallocate (__mach_task_self (), threads[i]);
      if (threads != threadbuf)
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) threads, nthreads * sizeof threads[0]);
      return 0;
    }

  /* Fetch a thread state structure from PID and store it at ADDR.  */
  int get_regs (int flavor, mach_msg_type_number_t count)
    {
      error_t err;
      task_t task = __pid2task (pid);
      thread_t thread;
      if (task == MACH_PORT_NULL)
	return -1;
      err = fetch_user_thread (task, &thread);
      __mach_port_deallocate (__mach_task_self (), task);
      if (!err)
	err = __thread_get_state (thread, flavor, addr, &count);
      __mach_port_deallocate (__mach_task_self (), thread);
      return err ? __hurd_fail (err) : 0;
    }


  switch (request)
    {
    case PTRACE_TRACEME:
      /* Make this process be traced.  */
      __sigfillset (&_hurdsig_traced);
      __USEPORT (PROC, __proc_mark_traced (port));
      break;

    case PTRACE_CONT:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      data = va_arg (ap, int);
      va_end (ap);
      {
	/* Send a DATA signal to PID, telling it to take the signal
	   normally even if it's traced.  */
	error_t err;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	if (data == SIGKILL)
	  err = __task_terminate (task);
	else
	  {
	    if (addr != (void *) 1)
	      {
		/* Move the user thread's PC to ADDR.  */
		thread_t thread;
		err = fetch_user_thread (task, &thread);
		if (!err)
		  {
		    struct machine_thread_state state;
		    mach_msg_type_number_t count = MACHINE_THREAD_STATE_COUNT;
		    err = __thread_get_state (thread,
					      MACHINE_THREAD_STATE_FLAVOR,
					      (natural_t *) &state, &count);
		    if (!err)
		      {
			MACHINE_THREAD_STATE_SET_PC (&state, addr);
			err = __thread_set_state (thread,
						  MACHINE_THREAD_STATE_FLAVOR,
						  (natural_t *) &state, count);
		      }

		  }
		__mach_port_deallocate (__mach_task_self (), thread);
	      }
	    else
	      err = 0;

	    if (! err)
	      /* Tell the process to take the signal (or just resume if 0).  */
	      err = HURD_MSGPORT_RPC
		(__USEPORT (PROC, __proc_getmsgport (port, pid, &msgport)),
		 0, 0, __msg_sig_post_untraced (msgport, data, 0, task));
	  }
	__mach_port_deallocate (__mach_task_self (), task);
	return err ? __hurd_fail (err) : 0;
      }

    case PTRACE_KILL:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      va_end (ap);
      /* SIGKILL always just terminates the task,
	 so normal kill is just the same when traced.  */
      return __kill (pid, SIGKILL);

    case PTRACE_SINGLESTEP:
      /* This is a machine-dependent kernel RPC on
	 machines that support it.  Punt.  */
      return __hurd_fail (EOPNOTSUPP);

    case PTRACE_ATTACH:
    case PTRACE_DETACH:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      va_end (ap);
      {
	/* Tell PID to set or clear its trace bit.  */
	error_t err;
	mach_port_t msgport;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	err = __USEPORT (PROC, __proc_getmsgport (port, pid, &msgport));
	if (! err)
	  {
	    err = __msg_set_init_int (msgport, task, INIT_TRACEMASK,
				      request == PTRACE_DETACH ? 0
				      : ~(sigset_t) 0);
	    if (! err)
	      {
		if (request == PTRACE_ATTACH)
		  /* Now stop the process.  */
		  err = __msg_sig_post (msgport, SIGSTOP, 0, task);
		else
		  /* Resume the process from tracing stop.  */
		  err = __msg_sig_post_untraced (msgport, 0, 0, task);
	      }
	    __mach_port_deallocate (__mach_task_self (), msgport);
	  }
	__mach_port_deallocate (__mach_task_self (), task);
	return err ? __hurd_fail (err) : 0;
      }

    case PTRACE_PEEKTEXT:
    case PTRACE_PEEKDATA:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      va_end (ap);
      {
	/* Read the page (or two pages, if the word lies on a boundary)
	   containing the addressed word.  */
	error_t err;
	vm_address_t ourpage;
	vm_size_t size;
	natural_t word;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	data = sizeof word;
	ourpage = 0;
	size = 0;
	err = read_data (task, &ourpage, &size);
	__mach_port_deallocate (__mach_task_self (), task);
	if (err)
	  return __hurd_fail (err);
	word = *(natural_t *) ((vm_address_t) addr - trunc_page (addr)
			       + ourpage);
	__vm_deallocate (__mach_task_self (), ourpage, size);
	return word;
      }

    case PTRACE_PEEKUSER:
    case PTRACE_POKEUSER:
      /* U area, what's that?  */
      return __hurd_fail (EOPNOTSUPP);

    case PTRACE_GETREGS:
    case PTRACE_SETREGS:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      va_end (ap);
      return get_regs (MACHINE_THREAD_STATE_FLAVOR,
		       MACHINE_THREAD_STATE_COUNT);

    case PTRACE_GETFPREGS:
    case PTRACE_SETFPREGS:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      va_end (ap);
#ifdef MACHINE_THREAD_FLOAT_STATE_FLAVOR
      return get_regs (MACHINE_THREAD_FLOAT_STATE_FLAVOR,
		       MACHINE_THREAD_FLOAT_STATE_COUNT);
#else
      return __hurd_fail (EOPNOTSUPP);
#endif

    case PTRACE_GETFPAREGS:
    case PTRACE_SETFPAREGS:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      va_end (ap);
#ifdef MACHINE_THREAD_FPA_STATE_FLAVOR
      return get_regs (MACHINE_THREAD_FPA_STATE_FLAVOR,
		       MACHINE_THREAD_FPA_STATE_COUNT);
#else
      return __hurd_fail (EOPNOTSUPP);
#endif

    case PTRACE_POKETEXT:
    case PTRACE_POKEDATA:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      data = va_arg (ap, int);
      va_end (ap);
      {
	/* Read the page (or two pages, if the word lies on a boundary)
	   containing the addressed word.  */
	error_t err;
	vm_address_t ourpage;
	vm_size_t size;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	data = sizeof (natural_t);
	ourpage = 0;
	size = 0;
	err = read_data (task, &ourpage, &size);

	if (!err)
	  {
	    /* Now modify the specified word and write the page back.  */
	    *(natural_t *) ((vm_address_t) addr - trunc_page (addr)
			    + ourpage) = data;
	    err = __vm_write (task, trunc_page (addr), ourpage, size);
	    __vm_deallocate (__mach_task_self (), ourpage, size);
	  }

	__mach_port_deallocate (__mach_task_self (), task);
	return err ? __hurd_fail (err) : 0;
      }

    case PTRACE_READDATA:
    case PTRACE_READTEXT:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      data = va_arg (ap, int);
      addr2 = va_arg (ap, void *);
      va_end (ap);
      {
	error_t err;
	vm_address_t ourpage;
	vm_size_t size;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	if (((vm_address_t) addr2 + data) % __vm_page_size == 0)
	  {
	    /* Perhaps we can write directly to the user's buffer.  */
	    ourpage = (vm_address_t) addr2;
	    size = data;
	  }
	else
	  {
	    ourpage = 0;
	    size = 0;
	  }
	err = read_data (task, &ourpage, &size);
	__mach_port_deallocate (__mach_task_self (), task);
	if (!err && ourpage != (vm_address_t) addr2)
	  {
	    memcpy (addr2, (void *) ourpage, data);
	    __vm_deallocate (__mach_task_self (), ourpage, size);
	  }
	return err ? __hurd_fail (err) : 0;
      }

    case PTRACE_WRITEDATA:
    case PTRACE_WRITETEXT:
      va_start (ap, request);
      pid = va_arg (ap, pid_t);
      addr = va_arg (ap, void *);
      data = va_arg (ap, int);
      addr2 = va_arg (ap, void *);
      va_end (ap);
      {
	error_t err;
	vm_address_t ourpage;
	vm_size_t size;
	task_t task = __pid2task (pid);
	if (task == MACH_PORT_NULL)
	  return -1;
	if ((vm_address_t) addr % __vm_page_size == 0
	    && (vm_address_t) data % __vm_page_size == 0)
	  {
	    /* Writing whole pages; can go directly from the user's buffer.  */
	    ourpage = (vm_address_t) addr2;
	    size = data;
	    err = 0;
	  }
	else
	  {
	    /* Read the task's pages and modify our own copy.  */
	    ourpage = 0;
	    size = 0;
	    err = read_data (task, &ourpage, &size);
	    if (!err)
	      memcpy ((void *) ((vm_address_t) addr - trunc_page (addr)
				+ ourpage),
		      addr2,
		      data);
	  }
	if (!err)
	  /* Write back the modified pages.  */
	  err = __vm_write (task, trunc_page (addr), ourpage, size);
	__mach_port_deallocate (__mach_task_self (), task);
	return err ? __hurd_fail (err) : 0;
      }

    default:
      errno = EINVAL;
      return -1;
    }

  return 0;
}

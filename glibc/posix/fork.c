/* fork - create a child process.
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

#include <fork.h>
#include <libio/libioP.h>
#include <ldsodefs.h>
#include <malloc/malloc-internal.h>
#include <nss/nss_database.h>
#include <register-atfork.h>
#include <stdio-lock.h>
#include <sys/single_threaded.h>
#include <unwind-link.h>

static void
fresetlockfiles (void)
{
  _IO_ITER i;

  for (i = _IO_iter_begin(); i != _IO_iter_end(); i = _IO_iter_next(i))
    if ((_IO_iter_file (i)->_flags & _IO_USER_LOCK) == 0)
      _IO_lock_init (*((_IO_lock_t *) _IO_iter_file(i)->_lock));
}

pid_t
__libc_fork (void)
{
  /* Determine if we are running multiple threads.  We skip some fork
     handlers in the single-thread case, to make fork safer to use in
     signal handlers.  Although POSIX has dropped async-signal-safe
     requirement for fork (Austin Group tracker issue #62) this is
     best effort to make is async-signal-safe at least for single-thread
     case.  */
  bool multiple_threads = __libc_single_threaded == 0;

  __run_fork_handlers (atfork_run_prepare, multiple_threads);

  struct nss_database_data nss_database_data;

  /* If we are not running multiple threads, we do not have to
     preserve lock state.  If fork runs from a signal handler, only
     async-signal-safe functions can be used in the child.  These data
     structures are only used by unsafe functions, so their state does
     not matter if fork was called from a signal handler.  */
  if (multiple_threads)
    {
      call_function_static_weak (__nss_database_fork_prepare_parent,
				 &nss_database_data);

      _IO_list_lock ();

      /* Acquire malloc locks.  This needs to come last because fork
	 handlers may use malloc, and the libio list lock has an
	 indirect malloc dependency as well (via the getdelim
	 function).  */
      call_function_static_weak (__malloc_fork_lock_parent);
    }

  pid_t pid = _Fork ();

  if (pid == 0)
    {
      fork_system_setup ();

      /* Reset the lock state in the multi-threaded case.  */
      if (multiple_threads)
	{
	  __libc_unwind_link_after_fork ();

	  fork_system_setup_after_fork ();

	  /* Release malloc locks.  */
	  call_function_static_weak (__malloc_fork_unlock_child);

	  /* Reset the file list.  These are recursive mutexes.  */
	  fresetlockfiles ();

	  /* Reset locks in the I/O code.  */
	  _IO_list_resetlock ();

	  call_function_static_weak (__nss_database_fork_subprocess,
				     &nss_database_data);
	}

      /* Reset the lock the dynamic loader uses to protect its data.  */
      __rtld_lock_initialize (GL(dl_load_lock));

      reclaim_stacks ();

      /* Run the handlers registered for the child.  */
      __run_fork_handlers (atfork_run_child, multiple_threads);
    }
  else
    {
      /* If _Fork failed, preserve its errno value.  */
      int save_errno = errno;

      /* Release acquired locks in the multi-threaded case.  */
      if (multiple_threads)
	{
	  /* Release malloc locks, parent process variant.  */
	  call_function_static_weak (__malloc_fork_unlock_parent);

	  /* We execute this even if the 'fork' call failed.  */
	  _IO_list_unlock ();
	}

      /* Run the handlers registered for the parent.  */
      __run_fork_handlers (atfork_run_parent, multiple_threads);

      if (pid < 0)
	__set_errno (save_errno);
    }

  return pid;
}
weak_alias (__libc_fork, __fork)
libc_hidden_def (__fork)
weak_alias (__libc_fork, fork)

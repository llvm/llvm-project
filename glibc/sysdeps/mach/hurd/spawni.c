/* spawn a new process running an executable.  Hurd version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <fcntl.h>
#include <paths.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/signal.h>
#include <hurd/fd.h>
#include <hurd/id.h>
#include <hurd/lookup.h>
#include <hurd/resource.h>
#include <assert.h>
#include <argz.h>
#include "spawn_int.h"

/* Spawn a new process executing PATH with the attributes describes in *ATTRP.
   Before running the process perform the actions described in FILE-ACTIONS. */
int
__spawni (pid_t *pid, const char *file,
	  const posix_spawn_file_actions_t *file_actions,
	  const posix_spawnattr_t *attrp,
	  char *const argv[], char *const envp[],
	  int xflags)
{
  pid_t new_pid;
  char *path, *p, *name;
  char *concat_name = NULL;
  const char *relpath, *abspath;
  int res;
  size_t len;
  size_t pathlen;
  short int flags;

  /* The generic POSIX.1 implementation of posix_spawn uses fork and exec.
     In traditional POSIX systems (Unix, Linux, etc), the only way to
     create a new process is by fork, which also copies all the things from
     the parent process that will be immediately wiped and replaced by the
     exec.

     This Hurd implementation works by doing an exec on a fresh task,
     without ever doing all the work of fork.  The only work done by fork
     that remains visible after an exec is registration with the proc
     server, and the inheritance of various values and ports.  All those
     inherited values and ports are what get collected up and passed in the
     file_exec_paths RPC by an exec call.  So we do the proc server
     registration here, following the model of fork (see fork.c).  We then
     collect up the inherited values and ports from this (parent) process
     following the model of exec (see hurd/hurdexec.c), modify or replace each
     value that fork would (plus the specific changes demanded by ATTRP and
     FILE_ACTIONS), and make the file_exec_paths RPC on the requested
     executable file with the child process's task port rather than our own.
     This should be indistinguishable from the fork + exec implementation,
     except that all errors will be detected here (in the parent process)
     and return proper errno codes rather than the child dying with 127.

     XXX The one exception to this supposed indistinguishableness is that
     when posix_spawn_file_actions_addopen has been used, the parent
     process can do various filesystem RPCs on the child's behalf, rather
     than the child process doing it.  If these block due to a broken or
     malicious filesystem server or just a blocked network fs or a serial
     port waiting for carrier detect (!!), the parent's posix_spawn call
     can block arbitrarily rather than just the child blocking.  Possible
     solutions include:
     * punt to plain fork + exec implementation if addopen was used
     ** easy to do
     ** gives up all benefits of this implementation in that case
     * if addopen was used, don't do any file actions at all here;
       instead, exec an installed helper program e.g.:
	/libexec/spawn-helper close 3 dup2 1 2 open 0 /file 0x123 0666 exec /bin/foo foo a1 a2
     ** extra exec might be more or less overhead than fork
     * could do some weird half-fork thing where the child would inherit
       our vm and run some code here, but not do the full work of fork

     XXX Actually, the parent opens the executable file on behalf of
     the child, and that has all the same issues.

     I am favoring the half-fork solution.  That is, we do task_create with
     vm inheritance, and we setjmp/longjmp the child like fork does.  But
     rather than all the fork hair, the parent just packs up init/dtable
     ports and does a single IPC to a receive right inserted in the child.  */

  error_t err;
  task_t task;
  file_t execfile;
  process_t proc;
  auth_t auth;
  int ints[INIT_INT_MAX];
  file_t *dtable;
  unsigned int dtablesize, orig_dtablesize, i;
  struct hurd_port **dtable_cells;
  char *dtable_cloexec;
  struct hurd_userlink *ulink_dtable = NULL;
  struct hurd_sigstate *ss;

  /* Child current working dir */
  file_t ccwdir = MACH_PORT_NULL;

  /* For POSIX_SPAWN_RESETIDS, this reauthenticates our root/current
     directory ports with the new AUTH port.  */
  file_t rcrdir = MACH_PORT_NULL, rcwdir = MACH_PORT_NULL;
  error_t reauthenticate (int which, file_t *result)
    {
      error_t err;
      mach_port_t ref;
      if (*result != MACH_PORT_NULL)
	return 0;
      ref = __mach_reply_port ();
      if (which == INIT_PORT_CWDIR && ccwdir != MACH_PORT_NULL)
	{
	  err = __io_reauthenticate (ccwdir, ref, MACH_MSG_TYPE_MAKE_SEND);
	  if (!err)
	    err = __auth_user_authenticate (auth,
					    ref, MACH_MSG_TYPE_MAKE_SEND,
					    result);
	}
      else
	err = HURD_PORT_USE
	  (&_hurd_ports[which],
	   ({
	     err = __io_reauthenticate (port, ref, MACH_MSG_TYPE_MAKE_SEND);
	     if (!err)
	       err = __auth_user_authenticate (auth,
					       ref, MACH_MSG_TYPE_MAKE_SEND,
					       result);
	     err;
	   }));
      __mach_port_destroy (__mach_task_self (), ref);
      return err;
    }

  /* Reauthenticate one of our file descriptors for the child.  A null
     element of DTABLE_CELLS indicates a descriptor that was already
     reauthenticated, or was newly opened on behalf of the child.  */
  error_t reauthenticate_fd (int fd)
    {
      if (dtable_cells[fd] != NULL)
	{
	  file_t newfile;
	  mach_port_t ref = __mach_reply_port ();
	  error_t err = __io_reauthenticate (dtable[fd],
					     ref, MACH_MSG_TYPE_MAKE_SEND);
	  if (!err)
	    err = __auth_user_authenticate (auth,
					    ref, MACH_MSG_TYPE_MAKE_SEND,
					    &newfile);
	  __mach_port_destroy (__mach_task_self (), ref);
	  if (err)
	    return err;
	  _hurd_port_free (dtable_cells[fd], &ulink_dtable[fd], dtable[fd]);
	  dtable_cells[fd] = NULL;
	  dtable[fd] = newfile;
	}
      return 0;
    }

  /* These callbacks are for looking up file names on behalf of the child.  */
  error_t child_init_port (int which, error_t (*operate) (mach_port_t))
    {
      if (flags & POSIX_SPAWN_RESETIDS)
	switch (which)
	  {
	  case INIT_PORT_AUTH:
	    return (*operate) (auth);
	  case INIT_PORT_CRDIR:
	    return (reauthenticate (INIT_PORT_CRDIR, &rcrdir)
		    ?: (*operate) (rcrdir));
	  case INIT_PORT_CWDIR:
	    return (reauthenticate (INIT_PORT_CWDIR, &rcwdir)
		    ?: (*operate) (rcwdir));
	  }
      else
	switch (which)
	  {
	  case INIT_PORT_CWDIR:
	    if (ccwdir != MACH_PORT_NULL)
	      return (*operate) (ccwdir);
	    break;
	  }
      assert (which != INIT_PORT_PROC);
      return _hurd_ports_use (which, operate);
    }
  file_t child_fd (int fd)
    {
      if ((unsigned int) fd < dtablesize && dtable[fd] != MACH_PORT_NULL)
	{
	  if (flags & POSIX_SPAWN_RESETIDS)
	    {
	      /* Reauthenticate this descriptor right now,
		 since it is going to be used on behalf of the child.  */
	      errno = reauthenticate_fd (fd);
	      if (errno)
		return MACH_PORT_NULL;
	    }
	  __mach_port_mod_refs (__mach_task_self (), dtable[fd],
				MACH_PORT_RIGHT_SEND, +1);
	  return dtable[fd];
	}
      errno = EBADF;
      return MACH_PORT_NULL;
    }
  inline error_t child_lookup (const char *file, int oflag, mode_t mode,
			       file_t *result)
    {
      return __hurd_file_name_lookup (&child_init_port, &child_fd, 0,
				      file, oflag, mode, result);
    }
  auto error_t child_chdir (const char *name)
    {
      file_t new_ccwdir;

      /* Append trailing "/." to directory name to force ENOTDIR if
	 it's not a directory and EACCES if we don't have search
	 permission.  */
      len = strlen (name);
      const char *lookup = name;
      if (len >= 2 && name[len - 2] == '/' && name[len - 1] == '.')
	lookup = name;
      else if (len == 0)
	/* Special-case empty file name according to POSIX.  */
	return __hurd_fail (ENOENT);
      else
	{
	  char *n = alloca (len + 3);
	  memcpy (n, name, len);
	  n[len] = '/';
	  n[len + 1] = '.';
	  n[len + 2] = '\0';
	  lookup = n;
	}

      error_t err = child_lookup (lookup, 0, 0, &new_ccwdir);
      if (!err)
	{
	  if (ccwdir != MACH_PORT_NULL)
	    __mach_port_deallocate (__mach_task_self (), ccwdir);
	  ccwdir = new_ccwdir;
	}

      return err;
    }
  inline error_t child_lookup_under (file_t startdir, const char *file,
				     int oflag, mode_t mode, file_t *result)
    {
      error_t use_init_port (int which, error_t (*operate) (mach_port_t))
	{
	  return (which == INIT_PORT_CWDIR ? (*operate) (startdir)
		  : child_init_port (which, operate));
	}

      return __hurd_file_name_lookup (&use_init_port, &child_fd, 0,
				      file, oflag, mode, result);
    }
  auto error_t child_fchdir (int fd)
    {
      file_t new_ccwdir;
      error_t err;

      if ((unsigned int)fd >= dtablesize
	  || dtable[fd] == MACH_PORT_NULL)
	return EBADF;

      /* We look up "." to force ENOTDIR if it's not a directory and EACCES if
         we don't have search permission.  */
      if (dtable_cells[fd] != NULL)
	  err = HURD_PORT_USE (dtable_cells[fd],
		    ({
		      child_lookup_under (port, ".", O_NOTRANS, 0, &new_ccwdir);
		     }));
      else
	  err = child_lookup_under (dtable[fd], ".", O_NOTRANS, 0, &new_ccwdir);

      if (!err)
	{
	  if (ccwdir != MACH_PORT_NULL)
	    __mach_port_deallocate (__mach_task_self (), ccwdir);
	  ccwdir = new_ccwdir;
	}

      return err;
    }


  /* Do this once.  */
  flags = attrp == NULL ? 0 : attrp->__flags;

  /* Generate the new process.  We create a task that does not inherit our
     memory, and then register it as our child like fork does.  See fork.c
     for comments about the sequencing of these proc operations.  */

  err = __task_create (__mach_task_self (),
#ifdef KERN_INVALID_LEDGER
		       NULL, 0,	/* OSF Mach */
#endif
		       0, &task);
  if (err)
    return __hurd_fail (err);
  // From here down we must deallocate TASK and PROC before returning.
  proc = MACH_PORT_NULL;
  auth = MACH_PORT_NULL;
  err = __USEPORT (PROC, __proc_task2pid (port, task, &new_pid));
  if (!err)
    err = __USEPORT (PROC, __proc_task2proc (port, task, &proc));
  if (!err)
    err = __USEPORT (PROC, __proc_child (port, task));
  if (err)
    goto out;

  /* Load up the ints to give the new program.  */
  memset (ints, 0, sizeof ints);
  ints[INIT_UMASK] = _hurd_umask;
  ints[INIT_TRACEMASK] = _hurdsig_traced;

  ss = _hurd_self_sigstate ();

retry:
  assert (! __spin_lock_locked (&ss->critical_section_lock));
  __spin_lock (&ss->critical_section_lock);

  _hurd_sigstate_lock (ss);
  ints[INIT_SIGMASK] = ss->blocked;
  ints[INIT_SIGPENDING] = 0;
  ints[INIT_SIGIGN] = 0;
  /* Unless we were asked to reset all handlers to SIG_DFL,
     pass down the set of signals that were set to SIG_IGN.  */
  {
    struct sigaction *actions = _hurd_sigstate_actions (ss);
    if ((flags & POSIX_SPAWN_SETSIGDEF) == 0)
      for (i = 1; i < NSIG; ++i)
	if (actions[i].sa_handler == SIG_IGN)
	  ints[INIT_SIGIGN] |= __sigmask (i);
  }

  /* We hold the critical section lock until the exec has failed so that no
     signal can arrive between when we pack the blocked and ignored signals,
     and when the exec actually happens.  A signal handler could change what
     signals are blocked and ignored.  Either the change will be reflected
     in the exec, or the signal will never be delivered.  Setting the
     critical section flag avoids anything we call trying to acquire the
     sigstate lock.  */

  _hurd_sigstate_unlock (ss);

  /* Set signal mask.  */
  if ((flags & POSIX_SPAWN_SETSIGMASK) != 0)
    ints[INIT_SIGMASK] = attrp->__ss;

#ifdef _POSIX_PRIORITY_SCHEDULING
  /* Set the scheduling algorithm and parameters.  */
# error implement me
  if ((flags & (POSIX_SPAWN_SETSCHEDPARAM | POSIX_SPAWN_SETSCHEDULER))
      == POSIX_SPAWN_SETSCHEDPARAM)
    {
      if (__sched_setparam (0, &attrp->__sp) == -1)
	_exit (SPAWN_ERROR);
    }
  else if ((flags & POSIX_SPAWN_SETSCHEDULER) != 0)
    {
      if (__sched_setscheduler (0, attrp->__policy,
				(flags & POSIX_SPAWN_SETSCHEDPARAM) != 0
				? &attrp->__sp : NULL) == -1)
	_exit (SPAWN_ERROR);
    }
#endif

  if (!err && (flags & POSIX_SPAWN_SETSID) != 0)
    err = __proc_setsid (proc);

  /* Set the process group ID.  */
  if (!err && (flags & POSIX_SPAWN_SETPGROUP) != 0)
    err = __proc_setpgrp (proc, new_pid, attrp->__pgrp);

  /* Set the effective user and group IDs.  */
  if (!err && (flags & POSIX_SPAWN_RESETIDS) != 0)
    {
      /* We need a different auth port for the child.  */

      __mutex_lock (&_hurd_id.lock);
      err = _hurd_check_ids (); /* Get _hurd_id up to date.  */
      if (!err && _hurd_id.rid_auth == MACH_PORT_NULL)
	{
	  /* Set up _hurd_id.rid_auth.  This is a special auth server port
	     which uses the real uid and gid (the first aux uid and gid) as
	     the only effective uid and gid.  */

	  if (_hurd_id.aux.nuids < 1 || _hurd_id.aux.ngids < 1)
	    /* We do not have a real UID and GID.  Lose, lose, lose!  */
	    err = EGRATUITOUS;

	  /* Create a new auth port using our real UID and GID (the first
	     auxiliary UID and GID) as the only effective IDs.  */
	  if (!err)
	    err = __USEPORT (AUTH,
			     __auth_makeauth (port,
					      NULL, MACH_MSG_TYPE_COPY_SEND, 0,
					      _hurd_id.aux.uids, 1,
					      _hurd_id.aux.uids,
					      _hurd_id.aux.nuids,
					      _hurd_id.aux.gids, 1,
					      _hurd_id.aux.gids,
					      _hurd_id.aux.ngids,
					      &_hurd_id.rid_auth));
	}
      if (!err)
	{
	  /* Use the real-ID auth port in place of the normal one.  */
	  assert (_hurd_id.rid_auth != MACH_PORT_NULL);
	  auth = _hurd_id.rid_auth;
	  __mach_port_mod_refs (__mach_task_self (), auth,
				MACH_PORT_RIGHT_SEND, +1);
	}
      __mutex_unlock (&_hurd_id.lock);
    }
  else
    /* Copy our existing auth port.  */
    err = __USEPORT (AUTH, __mach_port_mod_refs (__mach_task_self (),
						 (auth = port),
						 MACH_PORT_RIGHT_SEND, +1));

  if (err)
    {
      _hurd_critical_section_unlock (ss);

      if (err == EINTR)
	{
	  /* Got a signal while inside an RPC of the critical section, retry again */
	  __mach_port_deallocate (__mach_task_self (), auth);
	  auth = MACH_PORT_NULL;
	  goto retry;
	}

      goto out;
    }

  /* Pack up the descriptor table to give the new program.
     These descriptors will need to be reauthenticated below
     if POSIX_SPAWN_RESETIDS is set.  */
  __mutex_lock (&_hurd_dtable_lock);
  dtablesize = _hurd_dtablesize;
  orig_dtablesize = _hurd_dtablesize;
  dtable = __alloca (dtablesize * sizeof (dtable[0]));
  ulink_dtable = __alloca (dtablesize * sizeof (ulink_dtable[0]));
  dtable_cells = __alloca (dtablesize * sizeof (dtable_cells[0]));
  dtable_cloexec = __alloca (orig_dtablesize);
  for (i = 0; i < dtablesize; ++i)
    {
      struct hurd_fd *const d = _hurd_dtable[i];
      if (d == NULL)
	{
	  dtable[i] = MACH_PORT_NULL;
	  dtable_cells[i] = NULL;
	  continue;
	}
      /* Note that this might return MACH_PORT_NULL.  */
      dtable[i] = _hurd_port_get (&d->port, &ulink_dtable[i]);
      dtable_cells[i] = &d->port;
      dtable_cloexec[i] = (d->flags & FD_CLOEXEC) != 0;
    }
  __mutex_unlock (&_hurd_dtable_lock);

  /* Safe to let signals happen now.  */
  _hurd_critical_section_unlock (ss);

  /* Execute the file actions.  */
  if (file_actions != NULL)
    for (i = 0; i < file_actions->__used; ++i)
      {
	/* Close a file descriptor in the child.  */
	error_t do_close (int fd)
	  {
	    if ((unsigned int)fd < dtablesize
		&& dtable[fd] != MACH_PORT_NULL)
	      {
		if (dtable_cells[fd] == NULL)
		  __mach_port_deallocate (__mach_task_self (), dtable[fd]);
		else
		  {
		    _hurd_port_free (dtable_cells[fd],
				     &ulink_dtable[fd], dtable[fd]);
		  }
		dtable_cells[fd] = NULL;
		dtable[fd] = MACH_PORT_NULL;
		return 0;
	      }
	    return EBADF;
	  }

	/* Close file descriptors in the child.  */
	error_t do_closefrom (int lowfd)
	  {
	    while ((unsigned int) lowfd < dtablesize)
	      {
		error_t err = do_close (lowfd);
		if (err != 0 && err != EBADF)
		  return err;
		lowfd++;
	      }
	    return 0;
	  }

	/* Make sure the dtable can hold NEWFD.  */
#define EXPAND_DTABLE(newfd)						      \
	({								      \
	  if ((unsigned int)newfd >= dtablesize				      \
	      && newfd < _hurd_rlimits[RLIMIT_OFILE].rlim_cur)		      \
	    {								      \
	      /* We need to expand the dtable for the child.  */	      \
	      NEW_TABLE (dtable, newfd);				      \
	      NEW_ULINK_TABLE (ulink_dtable, newfd);			      \
	      NEW_TABLE (dtable_cells, newfd);				      \
	      dtablesize = newfd + 1;					      \
	    }								      \
	  ((unsigned int)newfd < dtablesize ? 0 : EMFILE);		      \
	})
#define NEW_TABLE(x, newfd) \
  do { __typeof (x) new_##x = __alloca ((newfd + 1) * sizeof (x[0]));	      \
  memcpy (new_##x, x, dtablesize * sizeof (x[0]));			      \
  memset (&new_##x[dtablesize], 0, (newfd + 1 - dtablesize) * sizeof (x[0])); \
  x = new_##x; } while (0)
#define NEW_ULINK_TABLE(x, newfd) \
  do { __typeof (x) new_##x = __alloca ((newfd + 1) * sizeof (x[0]));	      \
  unsigned i;								      \
  for (i = 0; i < dtablesize; i++)					      \
    if (dtable_cells[i] != NULL)					      \
      _hurd_port_move (dtable_cells[i], &new_##x[i], &x[i]);		      \
    else								      \
      memset (&new_##x[i], 0, sizeof (new_##x[i]));			      \
  memset (&new_##x[dtablesize], 0, (newfd + 1 - dtablesize) * sizeof (x[0])); \
  x = new_##x; } while (0)

	struct __spawn_action *action = &file_actions->__actions[i];

	switch (action->tag)
	  {
	  case spawn_do_close:
	    err = do_close (action->action.close_action.fd);
	    break;

	  case spawn_do_dup2:
	    if ((unsigned int)action->action.dup2_action.fd < dtablesize
		&& dtable[action->action.dup2_action.fd] != MACH_PORT_NULL)
	      {
		const int fd = action->action.dup2_action.fd;
		const int newfd = action->action.dup2_action.newfd;
		// dup2 always clears any old FD_CLOEXEC flag on the new fd.
		if (newfd < orig_dtablesize)
		  dtable_cloexec[newfd] = 0;
		if (fd == newfd)
		  // Same is same as same was.
		  break;
		err = EXPAND_DTABLE (newfd);
		if (!err)
		  {
		    /* Close the old NEWFD and replace it with FD's
		       contents, which can be either an original
		       descriptor (DTABLE_CELLS[FD] != 0) or a new
		       right that we acquired in this function.  */
		    do_close (newfd);
		    dtable_cells[newfd] = dtable_cells[fd];
		    if (dtable_cells[newfd] != NULL)
		      dtable[newfd] = _hurd_port_get (dtable_cells[newfd],
						      &ulink_dtable[newfd]);
		    else
		      {
			dtable[newfd] = dtable[fd];
			err = __mach_port_mod_refs (__mach_task_self (),
						    dtable[fd],
						    MACH_PORT_RIGHT_SEND, +1);
		      }
		  }
	      }
	    else
	      // The old FD specified was bogus.
	      err = EBADF;
	    break;

	  case spawn_do_open:
	    /* Open a file on behalf of the child.

	       XXX note that this can subject the parent to arbitrary
	       delays waiting for the files to open.  I don't know what the
	       spec says about this.  If it's not permissible, then this
	       whole forkless implementation is probably untenable.  */
	    {
	      const int fd = action->action.open_action.fd;

	      do_close (fd);
	      if (fd < orig_dtablesize)
		dtable_cloexec[fd] = 0;
	      err = EXPAND_DTABLE (fd);
	      if (err)
		break;

	      err = child_lookup (action->action.open_action.path,
				  action->action.open_action.oflag,
				  action->action.open_action.mode,
				  &dtable[fd]);
	      dtable_cells[fd] = NULL;
	      break;
	    }

	  case spawn_do_chdir:
	    err = child_chdir (action->action.chdir_action.path);
	    break;

	  case spawn_do_fchdir:
	    err = child_fchdir (action->action.fchdir_action.fd);
	    break;

	  case spawn_do_closefrom:
	    err = do_closefrom (action->action.closefrom_action.from);
	    break;
	  }

	if (err)
	  goto out;
      }

  /* Only now can we perform FD_CLOEXEC.  We had to leave the descriptors
     unmolested for the file actions to use.  Note that the DTABLE_CLOEXEC
     array is never expanded by file actions, so it might now have fewer
     than DTABLESIZE elements.  */
  for (i = 0; i < orig_dtablesize; ++i)
    if (dtable[i] != MACH_PORT_NULL && dtable_cloexec[i])
      {
	assert (dtable_cells[i] != NULL);
	_hurd_port_free (dtable_cells[i], &ulink_dtable[i], dtable[i]);
	dtable[i] = MACH_PORT_NULL;
      }

  /* Prune trailing null ports from the descriptor table.  */
  while (dtablesize > 0 && dtable[dtablesize - 1] == MACH_PORT_NULL)
    --dtablesize;

  if (flags & POSIX_SPAWN_RESETIDS)
    {
      /* Reauthenticate all the child's ports with its new auth handle.  */

      mach_port_t ref;
      process_t newproc;

      /* Reauthenticate with the proc server.  */
      ref = __mach_reply_port ();
      err = __proc_reauthenticate (proc, ref, MACH_MSG_TYPE_MAKE_SEND);
      if (!err)
	err = __auth_user_authenticate (auth,
					ref, MACH_MSG_TYPE_MAKE_SEND,
					&newproc);
      __mach_port_destroy (__mach_task_self (), ref);
      if (!err)
	{
	  __mach_port_deallocate (__mach_task_self (), proc);
	  proc = newproc;
	}

      if (!err)
	err = reauthenticate (INIT_PORT_CRDIR, &rcrdir);
      if (!err)
	err = reauthenticate (INIT_PORT_CWDIR, &rcwdir);

      /* We must reauthenticate all the fds except those that came from
	 `spawn_do_open' file actions, which were opened using the child's
	 auth port to begin with.  */
      for (i = 0; !err && i < dtablesize; ++i)
	err = reauthenticate_fd (i);
    }
  if (err)
    goto out;

  /* Now we are ready to open the executable file using the child's ports.
     We do this after performing all the file actions so the order of
     events is the same as for a fork, exec sequence.  This affects things
     like the meaning of a /dev/fd file name, as well as which error
     conditions are diagnosed first and what side effects (file creation,
     etc) can be observed before what errors.  */

  if ((xflags & SPAWN_XFLAGS_USE_PATH) == 0 || strchr (file, '/') != NULL)
    /* The FILE parameter is actually a path.  */
    err = child_lookup (relpath = file, O_EXEC, 0, &execfile);
  else
    {
      /* We have to search for FILE on the path.  */
      path = getenv ("PATH");
      if (path == NULL)
	{
	  /* There is no `PATH' in the environment.
	     The default search path is the current directory
	     followed by the path `confstr' returns for `_CS_PATH'.  */
	  len = __confstr (_CS_PATH, (char *) NULL, 0);
	  path = (char *) __alloca (1 + len);
	  path[0] = ':';
	  (void) __confstr (_CS_PATH, path + 1, len);
	}

      len = strlen (file) + 1;
      pathlen = strlen (path);
      name = __alloca (pathlen + len + 1);
      /* Copy the file name at the top.  */
      name = (char *) memcpy (name + pathlen + 1, file, len);
      /* And add the slash.  */
      *--name = '/';

      p = path;
      do
	{
	  char *startp;

	  path = p;
	  p = __strchrnul (path, ':');

	  if (p == path)
	    /* Two adjacent colons, or a colon at the beginning or the end
	       of `PATH' means to search the current directory.  */
	    startp = name + 1;
	  else
	    startp = (char *) memcpy (name - (p - path), path, p - path);

	  /* Try to open this file name.  */
	  err = child_lookup (startp, O_EXEC, 0, &execfile);
	  switch (err)
	    {
	    case EACCES:
	    case ENOENT:
	    case ESTALE:
	    case ENOTDIR:
	      /* Those errors indicate the file is missing or not executable
		 by us, in which case we want to just try the next path
		 directory.  */
	      continue;

	    case 0:		/* Success! */
	    default:
	      /* Some other error means we found an executable file, but
		 something went wrong executing it; return the error to our
		 caller.  */
	      break;
	    }

	  // We only get here when we are done looking for the file.
	  relpath = startp;
	  break;
	}
      while (*p++ != '\0');
    }
  if (err)
    goto out;

  if (relpath[0] == '/')
    {
      /* Already an absolute path */
      abspath = relpath;
    }
  else
    {
      /* Relative path */
      char *cwd = __getcwd (NULL, 0);
      if (cwd == NULL)
	goto out;

      res = __asprintf (&concat_name, "%s/%s", cwd, relpath);
      free (cwd);
      if (res == -1)
	goto out;

      abspath = concat_name;
    }

  /* Almost there!  */
  {
    mach_port_t ports[_hurd_nports];
    struct hurd_userlink ulink_ports[_hurd_nports];
    char *args = NULL, *env = NULL;
    size_t argslen = 0, envlen = 0;

    inline error_t exec (file_t file)
      {
	error_t err = __file_exec_paths
	  (file, task,
	   __sigismember (&_hurdsig_traced, SIGKILL) ? EXEC_SIGTRAP : 0,
	   relpath, abspath, args, argslen, env, envlen,
	   dtable, MACH_MSG_TYPE_COPY_SEND, dtablesize,
	   ports, MACH_MSG_TYPE_COPY_SEND, _hurd_nports,
	   ints, INIT_INT_MAX,
	   NULL, 0, NULL, 0);

	/* Fallback for backwards compatibility.  This can just be removed
	   when __file_exec goes away.  */
	if (err == MIG_BAD_ID)
	  return __file_exec (file, task,
			      (__sigismember (&_hurdsig_traced, SIGKILL)
			      ? EXEC_SIGTRAP : 0),
			      args, argslen, env, envlen,
			      dtable, MACH_MSG_TYPE_COPY_SEND, dtablesize,
			      ports, MACH_MSG_TYPE_COPY_SEND, _hurd_nports,
			      ints, INIT_INT_MAX,
			      NULL, 0, NULL, 0);

	return err;
      }

    /* Now we are out of things that can fail before the file_exec RPC,
       for which everything else must be prepared.  The only thing left
       to do is packing up the argument and environment strings,
       and the array of init ports.  */

    if (argv != NULL)
      err = __argz_create (argv, &args, &argslen);
    if (!err && envp != NULL)
      err = __argz_create (envp, &env, &envlen);

    /* Load up the ports to give to the new program.
       Note the loop/switch below must parallel exactly to release refs.  */
    for (i = 0; i < _hurd_nports; ++i)
      {
	switch (i)
	  {
	  case INIT_PORT_AUTH:
	    ports[i] = auth;
	    continue;
	  case INIT_PORT_PROC:
	    ports[i] = proc;
	    continue;
	  case INIT_PORT_CRDIR:
	    if (flags & POSIX_SPAWN_RESETIDS)
	      {
		ports[i] = rcrdir;
		continue;
	      }
	    break;
	  case INIT_PORT_CWDIR:
	    if (flags & POSIX_SPAWN_RESETIDS)
	      {
		ports[i] = rcwdir;
		continue;
	      }
	    if (ccwdir != MACH_PORT_NULL)
	      {
		ports[i] = ccwdir;
		continue;
	      }
	    break;
	  }
	ports[i] = _hurd_port_get (&_hurd_ports[i], &ulink_ports[i]);
      }

    /* Finally, try executing the file we opened.  */
    if (!err)
      err = exec (execfile);
    __mach_port_deallocate (__mach_task_self (), execfile);

    if ((err == ENOEXEC) && (xflags & SPAWN_XFLAGS_TRY_SHELL) != 0)
      {
	/* The file is accessible but it is not an executable file.
	   Invoke the shell to interpret it as a script.  */
	err = 0;
	if (!argslen)
	  err = __argz_insert (&args, &argslen, args, relpath);
	if (!err)
	  err = __argz_insert (&args, &argslen, args, _PATH_BSHELL);
	if (!err)
	  err = child_lookup (_PATH_BSHELL, O_EXEC, 0, &execfile);
	if (!err)
	  {
	    err = exec (execfile);
	    __mach_port_deallocate (__mach_task_self (), execfile);
	  }
      }

    /* Release the references just packed up in PORTS.
       This switch must always parallel the one above that fills PORTS.  */
    for (i = 0; i < _hurd_nports; ++i)
      {
	switch (i)
	  {
	  case INIT_PORT_AUTH:
	  case INIT_PORT_PROC:
	    continue;
	  case INIT_PORT_CRDIR:
	    if (flags & POSIX_SPAWN_RESETIDS)
	      continue;
	    break;
	  case INIT_PORT_CWDIR:
	    if (flags & POSIX_SPAWN_RESETIDS)
	      continue;
	    if (ccwdir != MACH_PORT_NULL)
	      continue;
	    break;
	  }
	_hurd_port_free (&_hurd_ports[i], &ulink_ports[i], ports[i]);
      }

    free (args);
    free (env);
  }

  /* We did it!  We have a child!  */
  if (pid != NULL)
    *pid = new_pid;

 out:
  /* Clean up all the references we are now holding.  */

  if (task != MACH_PORT_NULL)
    {
      if (err)
	/* We failed after creating the task, so kill it.  */
	__task_terminate (task);
      __mach_port_deallocate (__mach_task_self (), task);
    }
  __mach_port_deallocate (__mach_task_self (), auth);
  __mach_port_deallocate (__mach_task_self (), proc);
  if (ccwdir != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), ccwdir);
  if (rcrdir != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), rcrdir);
  if (rcwdir != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), rcwdir);

  if (ulink_dtable)
    /* Release references to the file descriptor ports.  */
    for (i = 0; i < dtablesize; ++i)
      if (dtable[i] != MACH_PORT_NULL)
	{
	  if (dtable_cells[i] == NULL)
	    __mach_port_deallocate (__mach_task_self (), dtable[i]);
	  else
	    _hurd_port_free (dtable_cells[i], &ulink_dtable[i], dtable[i]);
	}

  free (concat_name);

  if (err)
    /* This hack canonicalizes the error code that we return.  */
    err = (__hurd_fail (err), errno);

  return err;
}

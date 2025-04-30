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

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/signal.h>
#include <hurd/id.h>
#include <assert.h>
#include <argz.h>

/* Overlay TASK, executing FILE with arguments ARGV and environment ENVP.
   If TASK == mach_task_self (), some ports are dealloc'd by the exec server.
   ARGV and ENVP are terminated by NULL pointers.
   Deprecated: use _hurd_exec_paths instead.  */
error_t
_hurd_exec (task_t task, file_t file,
	    char *const argv[], char *const envp[])
{
  return _hurd_exec_paths (task, file, NULL, NULL, argv, envp);
}

link_warning (_hurd_exec,
	      "_hurd_exec is deprecated, use _hurd_exec_paths instead");

/* Overlay TASK, executing FILE with arguments ARGV and environment ENVP.
   If TASK == mach_task_self (), some ports are dealloc'd by the exec server.
   ARGV and ENVP are terminated by NULL pointers.  PATH is the relative path to
   FILE and ABSPATH is the absolute path to FILE. Passing NULL, though possible,
   should be avoided, since then the exec server may not know the path to
   FILE if FILE is a script, and will then pass /dev/fd/N to the
   interpreter.  */
error_t
_hurd_exec_paths (task_t task, file_t file,
		   const char *path, const char *abspath,
		   char *const argv[], char *const envp[])
{
  error_t err;
  char *args, *env;
  size_t argslen, envlen;
  int ints[INIT_INT_MAX];
  mach_port_t ports[_hurd_nports];
  struct hurd_userlink ulink_ports[_hurd_nports];
  inline void free_port (unsigned int i)
    {
      _hurd_port_free (&_hurd_ports[i], &ulink_ports[i], ports[i]);
    }
  file_t *dtable;
  unsigned int dtablesize, i;
  struct hurd_port **dtable_cells;
  struct hurd_userlink *ulink_dtable;
  struct hurd_sigstate *ss;
  mach_port_t *please_dealloc, *pdp;
  int reauth = 0;

  /* XXX needs to be hurdmalloc XXX */
  if (argv == NULL)
    args = NULL, argslen = 0;
  else if (err = __argz_create (argv, &args, &argslen))
    return err;
  if (envp == NULL)
    env = NULL, envlen = 0;
  else if (err = __argz_create (envp, &env, &envlen))
    goto outargs;

  /* Load up the ports to give to the new program.  */
  for (i = 0; i < _hurd_nports; ++i)
    if (i == INIT_PORT_PROC && task != __mach_task_self ())
      {
	/* This is another task, so we need to ask the proc server
	   for the right proc server port for it.  */
	if (err = __USEPORT (PROC, __proc_task2proc (port, task, &ports[i])))
	  {
	    while (--i > 0)
	      free_port (i);
	    goto outenv;
	  }
      }
    else
      ports[i] = _hurd_port_get (&_hurd_ports[i], &ulink_ports[i]);


  /* Load up the ints to give the new program.  */
  for (i = 0; i < INIT_INT_MAX; ++i)
    switch (i)
      {
      case INIT_UMASK:
	ints[i] = _hurd_umask;
	break;

      case INIT_SIGMASK:
      case INIT_SIGIGN:
      case INIT_SIGPENDING:
	/* We will set these all below.  */
	break;

      case INIT_TRACEMASK:
	ints[i] = _hurdsig_traced;
	break;

      default:
	ints[i] = 0;
      }

  ss = _hurd_self_sigstate ();

retry:
  assert (! __spin_lock_locked (&ss->critical_section_lock));
  __spin_lock (&ss->critical_section_lock);

  _hurd_sigstate_lock (ss);
  struct sigaction *actions = _hurd_sigstate_actions (ss);
  ints[INIT_SIGMASK] = ss->blocked;
  ints[INIT_SIGPENDING] = _hurd_sigstate_pending (ss);
  ints[INIT_SIGIGN] = 0;
  for (i = 1; i < NSIG; ++i)
    if (actions[i].sa_handler == SIG_IGN)
      ints[INIT_SIGIGN] |= __sigmask (i);

  /* We hold the sigstate lock until the exec has failed so that no signal
     can arrive between when we pack the blocked and ignored signals, and
     when the exec actually happens.  A signal handler could change what
     signals are blocked and ignored.  Either the change will be reflected
     in the exec, or the signal will never be delivered.  Setting the
     critical section flag avoids anything we call trying to acquire the
     sigstate lock.  */

  _hurd_sigstate_unlock (ss);

  /* Pack up the descriptor table to give the new program.  */
  __mutex_lock (&_hurd_dtable_lock);

  dtablesize = _hurd_dtable ? _hurd_dtablesize : _hurd_init_dtablesize;

  if (task == __mach_task_self ())
    /* Request the exec server to deallocate some ports from us if the exec
       succeeds.  The init ports and descriptor ports will arrive in the
       new program's exec_startup message.  If we failed to deallocate
       them, the new program would have duplicate user references for them.
       But we cannot deallocate them ourselves, because we must still have
       them after a failed exec call.  */
    please_dealloc = __alloca ((_hurd_nports + 3 + (3 * dtablesize))
				* sizeof (mach_port_t));
  else
    please_dealloc = NULL;
  pdp = please_dealloc;

  if (_hurd_dtable != NULL)
    {
      dtable = __alloca (dtablesize * sizeof (dtable[0]));
      ulink_dtable = __alloca (dtablesize * sizeof (ulink_dtable[0]));
      dtable_cells = __alloca (dtablesize * sizeof (dtable_cells[0]));
      for (i = 0; i < dtablesize; ++i)
	{
	  struct hurd_fd *const d = _hurd_dtable[i];
	  if (d == NULL)
	    {
	      dtable[i] = MACH_PORT_NULL;
	      continue;
	    }
	  __spin_lock (&d->port.lock);
	  if (d->flags & FD_CLOEXEC)
	    {
	      /* This descriptor is marked to be closed on exec.
		 So don't pass it to the new program.  */
	      dtable[i] = MACH_PORT_NULL;
	      if (pdp && d->port.port != MACH_PORT_NULL)
		{
		  /* We still need to deallocate the ports.  */
		  *pdp++ = d->port.port;
		  if (d->ctty.port != MACH_PORT_NULL)
		    *pdp++ = d->ctty.port;
		}
	      __spin_unlock (&d->port.lock);
	    }
	  else
	    {
	      if (pdp && d->ctty.port != MACH_PORT_NULL)
		/* All the elements of DTABLE are added to PLEASE_DEALLOC
		   below, so we needn't add the port itself.
		   But we must deallocate the ctty port as well as
		   the normal port that got installed in DTABLE[I].  */
		*pdp++ = d->ctty.port;
	      dtable[i] = _hurd_port_locked_get (&d->port, &ulink_dtable[i]);
	      dtable_cells[i] = &d->port;
	    }
	}
    }
  else
    {
      dtable = _hurd_init_dtable;
      ulink_dtable = NULL;
      dtable_cells = NULL;
    }

  /* Prune trailing null ports from the descriptor table.  */
  while (dtablesize > 0 && dtable[dtablesize - 1] == MACH_PORT_NULL)
    --dtablesize;

  /* See if we need to diddle the auth port of the new program.
     The purpose of this is to get the effect setting the saved-set UID and
     GID to the respective effective IDs after the exec, as POSIX.1 requires.
     Note that we don't reauthenticate with the proc server; that would be a
     no-op since it only keeps track of the effective UIDs, and if it did
     keep track of the available IDs we would have the problem that we'd be
     changing the IDs before the exec and have to change them back after a
     failure.  Arguably we could skip all the reauthentications because the
     available IDs have no bearing on any filesystem.  But the conservative
     approach is to reauthenticate all the io ports so that no state anywhere
     reflects that our whole ID set differs from what we've set it to.  */
  __mutex_lock (&_hurd_id.lock);
  err = _hurd_check_ids ();
  if (err == 0 && ((_hurd_id.aux.nuids >= 2 && _hurd_id.gen.nuids >= 1
		    && _hurd_id.aux.uids[1] != _hurd_id.gen.uids[0])
		   || (_hurd_id.aux.ngids >= 2 && _hurd_id.gen.ngids >= 1
		       && _hurd_id.aux.gids[1] != _hurd_id.gen.gids[0])))
    {
      /* We have euid != svuid or egid != svgid.  POSIX.1 says that exec
	 sets svuid = euid and svgid = egid.  So we must get a new auth
	 port and reauthenticate everything with it.  We'll pass the new
	 ports in file_exec_paths instead of our own ports.  */

      auth_t newauth;

      _hurd_id.aux.uids[1] = _hurd_id.gen.uids[0];
      _hurd_id.aux.gids[1] = _hurd_id.gen.gids[0];
      _hurd_id.valid = 0;
      if (_hurd_id.rid_auth != MACH_PORT_NULL)
	{
	  __mach_port_deallocate (__mach_task_self (), _hurd_id.rid_auth);
	  _hurd_id.rid_auth = MACH_PORT_NULL;
	}

      err = __auth_makeauth (ports[INIT_PORT_AUTH],
			     NULL, MACH_MSG_TYPE_COPY_SEND, 0,
			     _hurd_id.gen.uids, _hurd_id.gen.nuids,
			     _hurd_id.aux.uids, _hurd_id.aux.nuids,
			     _hurd_id.gen.gids, _hurd_id.gen.ngids,
			     _hurd_id.aux.gids, _hurd_id.aux.ngids,
			     &newauth);
      if (err == 0)
	{
	  /* Now we have to reauthenticate the ports with this new ID.
	   */

	  inline error_t reauth_io (io_t port, io_t *newport)
	    {
	      mach_port_t ref = __mach_reply_port ();
	      *newport = MACH_PORT_NULL;
	      error_t err = __io_reauthenticate (port,
						 ref, MACH_MSG_TYPE_MAKE_SEND);
	      if (!err)
		err = __auth_user_authenticate (newauth,
						ref, MACH_MSG_TYPE_MAKE_SEND,
						newport);
	      __mach_port_destroy (__mach_task_self (), ref);
	      return err;
	    }
	  inline void reauth_port (unsigned int idx)
	    {
	      io_t newport;
	      err = reauth_io (ports[idx], &newport) ?: err;
	      if (pdp)
		*pdp++ = ports[idx]; /* XXX presumed still in _hurd_ports */
	      free_port (idx);
	      ports[idx] = newport;
	    }

	  if (pdp)
	    *pdp++ = ports[INIT_PORT_AUTH];
	  free_port (INIT_PORT_AUTH);
	  ports[INIT_PORT_AUTH] = newauth;

	  reauth_port (INIT_PORT_CRDIR);
	  reauth_port (INIT_PORT_CWDIR);

	  if (!err)
	    {
	      /* Now we'll reauthenticate each file descriptor.  */
	      if (ulink_dtable == NULL)
		{
		  assert (dtable == _hurd_init_dtable);
		  dtable = __alloca (dtablesize * sizeof (dtable[0]));
		  for (i = 0; i < dtablesize; ++i)
		    if (_hurd_init_dtable[i] != MACH_PORT_NULL)
		      {
			if (pdp)
			  *pdp++ = _hurd_init_dtable[i];
			err = reauth_io (_hurd_init_dtable[i], &dtable[i]);
			if (err)
			  {
			    while (++i < dtablesize)
			      dtable[i] = MACH_PORT_NULL;
			    break;
			  }
		      }
		    else
		      dtable[i] = MACH_PORT_NULL;
		}
	      else
		{
		  if (pdp)
		    {
		      /* Ask to deallocate all the old fd ports,
			 since we will have new ones in DTABLE.  */
		      memcpy (pdp, dtable, dtablesize * sizeof pdp[0]);
		      pdp += dtablesize;
		    }
		  for (i = 0; i < dtablesize; ++i)
		    if (dtable[i] != MACH_PORT_NULL)
		      {
			io_t newport;
			err = reauth_io (dtable[i], &newport);
			_hurd_port_free (dtable_cells[i], &ulink_dtable[i],
					 dtable[i]);
			dtable[i] = newport;
			if (err)
			  {
			    while (++i < dtablesize)
			      _hurd_port_free (dtable_cells[i],
					       &ulink_dtable[i], dtable[i]);
			    break;
			  }
		      }
		  ulink_dtable = NULL;
		  dtable_cells = NULL;
		}
	    }
	}

      reauth = 1;
    }
  __mutex_unlock (&_hurd_id.lock);

  /* The information is all set up now.  Try to exec the file.  */
  if (!err)
    {
      int flags;

      if (pdp)
	{
	  /* Request the exec server to deallocate some ports from us if
	     the exec succeeds.  The init ports and descriptor ports will
	     arrive in the new program's exec_startup message.  If we
	     failed to deallocate them, the new program would have
	     duplicate user references for them.  But we cannot deallocate
	     them ourselves, because we must still have them after a failed
	     exec call.  */

	  for (i = 0; i < _hurd_nports; ++i)
	    *pdp++ = ports[i];
	  for (i = 0; i < dtablesize; ++i)
	    *pdp++ = dtable[i];
	}

      flags = 0;
#ifdef EXEC_SIGTRAP
      /* PTRACE_TRACEME sets all bits in _hurdsig_traced, which is
	 propagated through exec by INIT_TRACEMASK, so this checks if
	 PTRACE_TRACEME has been called in this process in any of its
	 current or prior lives.  */
      if (__sigismember (&_hurdsig_traced, SIGKILL))
	flags |= EXEC_SIGTRAP;
#endif
      err = __file_exec_paths (file, task, flags,
			       path ? path : "",
			       abspath ? abspath : "",
			       args, argslen, env, envlen,
			       dtable, MACH_MSG_TYPE_COPY_SEND, dtablesize,
			       ports, MACH_MSG_TYPE_COPY_SEND,
			       _hurd_nports,
			       ints, INIT_INT_MAX,
			       please_dealloc, pdp - please_dealloc,
			       &_hurd_msgport,
			       task == __mach_task_self () ? 1 : 0);
      /* Fall back for backwards compatibility.  This can just be removed
         when __file_exec goes away.  */
      if (err == MIG_BAD_ID)
	err = __file_exec (file, task, flags,
			   args, argslen, env, envlen,
			   dtable, MACH_MSG_TYPE_COPY_SEND, dtablesize,
			   ports, MACH_MSG_TYPE_COPY_SEND, _hurd_nports,
			   ints, INIT_INT_MAX,
			   please_dealloc, pdp - please_dealloc,
			   &_hurd_msgport,
			   task == __mach_task_self () ? 1 : 0);
    }

  /* Release references to the standard ports.  */
  for (i = 0; i < _hurd_nports; ++i)
    if ((i == INIT_PORT_PROC && task != __mach_task_self ())
	|| (reauth && (i == INIT_PORT_AUTH
		       || i == INIT_PORT_CRDIR || i == INIT_PORT_CWDIR)))
      __mach_port_deallocate (__mach_task_self (), ports[i]);
    else
      free_port (i);

  /* Release references to the file descriptor ports.  */
  if (ulink_dtable != NULL)
    {
      for (i = 0; i < dtablesize; ++i)
	if (dtable[i] != MACH_PORT_NULL)
	  _hurd_port_free (dtable_cells[i], &ulink_dtable[i], dtable[i]);
    }
  else if (dtable && dtable != _hurd_init_dtable)
    for (i = 0; i < dtablesize; ++i)
      __mach_port_deallocate (__mach_task_self (), dtable[i]);

  /* Release lock on the file descriptor table. */
  __mutex_unlock (&_hurd_dtable_lock);

  /* Safe to let signals happen now.  */
  _hurd_critical_section_unlock (ss);
  if (err == EINTR)
    /* Got a signal while inside an RPC of the critical section, retry again */
    goto retry;

 outargs:
  free (args);
 outenv:
  free (env);
  return err;
}
libc_hidden_def (_hurd_exec_paths)

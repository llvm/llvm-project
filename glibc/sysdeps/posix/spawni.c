/* Guts of POSIX spawn interface.  Generic POSIX.1 version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <spawn.h>
#include <assert.h>
#include <fcntl.h>
#include <paths.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <not-cancel.h>
#include <local-setxid.h>
#include <shlib-compat.h>
#include <pthreadP.h>
#include <dl-sysdep.h>
#include <libc-pointer-arith.h>
#include <ldsodefs.h>
#include "spawn_int.h"


/* The Unix standard contains a long explanation of the way to signal
   an error after the fork() was successful.  Since no new wait status
   was wanted there is no way to signal an error using one of the
   available methods.  The committee chose to signal an error by a
   normal program exit with the exit code 127.  */
#define SPAWN_ERROR	127

struct posix_spawn_args
{
  sigset_t oldmask;
  const char *file;
  int (*exec) (const char *, char *const *, char *const *);
  const posix_spawn_file_actions_t *fa;
  const posix_spawnattr_t *restrict attr;
  char *const *argv;
  ptrdiff_t argc;
  char *const *envp;
  int xflags;
  int pipe[2];
};

/* Older version requires that shell script without shebang definition
   to be called explicitly using /bin/sh (_PATH_BSHELL).  */
static void
maybe_script_execute (struct posix_spawn_args *args)
{
  if (SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_15)
      && (args->xflags & SPAWN_XFLAGS_TRY_SHELL) && errno == ENOEXEC)
    {
      char *const *argv = args->argv;
      ptrdiff_t argc = args->argc;

      /* Construct an argument list for the shell.  */
      char *new_argv[argc + 2];
      new_argv[0] = (char *) _PATH_BSHELL;
      new_argv[1] = (char *) args->file;
      if (argc > 1)
	memcpy (new_argv + 2, argv + 1, argc * sizeof (char *));
      else
	new_argv[2] = NULL;

      /* Execute the shell.  */
      args->exec (new_argv[0], new_argv, args->envp);
    }
}

/* Function used in the clone call to setup the signals mask, posix_spawn
   attributes, and file actions.  */
static int
__spawni_child (void *arguments)
{
  struct posix_spawn_args *args = arguments;
  const posix_spawnattr_t *restrict attr = args->attr;
  const posix_spawn_file_actions_t *file_actions = args->fa;
  int ret;

  __close (args->pipe[0]);

  /* Set signal default action.  */
  if ((attr->__flags & POSIX_SPAWN_SETSIGDEF) != 0)
    {
      /* We have to iterate over all signals.  This could possibly be
	 done better but it requires system specific solutions since
	 the sigset_t data type can be very different on different
	 architectures.  */
      int sig;
      struct sigaction sa;

      memset (&sa, '\0', sizeof (sa));
      sa.sa_handler = SIG_DFL;

      for (sig = 1; sig <= _NSIG; ++sig)
	if (__sigismember (&attr->__sd, sig) != 0
	    && __sigaction (sig, &sa, NULL) != 0)
	  goto fail;
    }

#ifdef _POSIX_PRIORITY_SCHEDULING
  /* Set the scheduling algorithm and parameters.  */
  if ((attr->__flags & (POSIX_SPAWN_SETSCHEDPARAM | POSIX_SPAWN_SETSCHEDULER))
      == POSIX_SPAWN_SETSCHEDPARAM)
    {
      if (__sched_setparam (0, &attr->__sp) == -1)
	goto fail;
    }
  else if ((attr->__flags & POSIX_SPAWN_SETSCHEDULER) != 0)
    {
      if (__sched_setscheduler (0, attr->__policy, &attr->__sp) == -1)
	goto fail;
    }
#endif

  /* Set the process session ID.  */
  if ((attr->__flags & POSIX_SPAWN_SETSID) != 0
      && __setsid () < 0)
    goto fail;

  /* Set the process group ID.  */
  if ((attr->__flags & POSIX_SPAWN_SETPGROUP) != 0
      && __setpgid (0, attr->__pgrp) != 0)
    goto fail;

  /* Set the effective user and group IDs.  */
  if ((attr->__flags & POSIX_SPAWN_RESETIDS) != 0
      && (local_seteuid (__getuid ()) != 0
	  || local_setegid (__getgid ())) != 0)
    goto fail;

  /* Execute the file actions.  */
  if (file_actions != NULL)
    {
      int cnt;
      struct rlimit64 fdlimit;
      bool have_fdlimit = false;

      for (cnt = 0; cnt < file_actions->__used; ++cnt)
	{
	  struct __spawn_action *action = &file_actions->__actions[cnt];

	  switch (action->tag)
	    {
	    case spawn_do_close:
	      if (__close_nocancel (action->action.close_action.fd) != 0)
		{
		  if (have_fdlimit == 0)
		    {
		      __getrlimit64 (RLIMIT_NOFILE, &fdlimit);
		      have_fdlimit = true;
		    }

		  /* Only signal errors for file descriptors out of range.  */
		  if (action->action.close_action.fd < 0
		      || action->action.close_action.fd >= fdlimit.rlim_cur)
		    goto fail;
		}
	      break;

	    case spawn_do_open:
	      {
		/* POSIX states that if fildes was already an open file descriptor,
		   it shall be closed before the new file is opened.  This avoid
		   pontential issues when posix_spawn plus addopen action is called
		   with the process already at maximum number of file descriptor
		   opened and also for multiple actions on single-open special
		   paths (like /dev/watchdog).  */
		__close_nocancel (action->action.open_action.fd);

		int new_fd = __open_nocancel (action->action.open_action.path,
					      action->action.open_action.oflag
					      | O_LARGEFILE,
					      action->action.open_action.mode);

		if (new_fd == -1)
		  goto fail;

		/* Make sure the desired file descriptor is used.  */
		if (new_fd != action->action.open_action.fd)
		  {
		    if (__dup2 (new_fd, action->action.open_action.fd)
			!= action->action.open_action.fd)
		      goto fail;

		    if (__close_nocancel (new_fd) != 0)
		      goto fail;
		  }
	      }
	      break;

	    case spawn_do_dup2:
	      /* Austin Group issue #411 requires adddup2 action with source
		 and destination being equal to remove close-on-exec flag.  */
	      if (action->action.dup2_action.fd
		  == action->action.dup2_action.newfd)
		{
		  int fd = action->action.dup2_action.newfd;
		  int flags = __fcntl (fd, F_GETFD, 0);
		  if (flags == -1)
		    goto fail;
		  if (__fcntl (fd, F_SETFD, flags & ~FD_CLOEXEC) == -1)
		    goto fail;
		}
	      else if (__dup2 (action->action.dup2_action.fd,
			       action->action.dup2_action.newfd)
		       != action->action.dup2_action.newfd)
		goto fail;
	      break;

	    case spawn_do_chdir:
	      if (__chdir (action->action.chdir_action.path) != 0)
		goto fail;
	      break;

	    case spawn_do_fchdir:
	      if (__fchdir (action->action.fchdir_action.fd) != 0)
		goto fail;
	      break;

	    case spawn_do_closefrom:
	      __set_errno (EINVAL);
	      goto fail;
	    }
	}
    }

  /* Set the initial signal mask of the child if POSIX_SPAWN_SETSIGMASK
     is set, otherwise restore the previous one.  */
  __sigprocmask (SIG_SETMASK, (attr->__flags & POSIX_SPAWN_SETSIGMASK)
		 ? &attr->__ss : &args->oldmask, 0);

  args->exec (args->file, args->argv, args->envp);

  /* This is compatibility function required to enable posix_spawn run
     script without shebang definition for older posix_spawn versions
     (2.15).  */
  maybe_script_execute (args);

fail:
  /* errno should have an appropriate non-zero value; otherwise,
     there's a bug in glibc or the kernel.  For lack of an error code
     (EINTERNALBUG) describing that, use ECHILD.  Another option would
     be to set args->err to some negative sentinel and have the parent
     abort(), but that seems needlessly harsh.  */
  ret = errno ? : ECHILD;
  if (ret)
    /* Since sizeof errno < PIPE_BUF, the write is atomic. */
    while (__write_nocancel (args->pipe[1], &ret, sizeof (ret)) < 0);

  _exit (SPAWN_ERROR);
}

/* Spawn a new process executing PATH with the attributes describes in *ATTRP.
   Before running the process perform the actions described in FILE-ACTIONS. */
int
__spawnix (pid_t *pid, const char *file,
	   const posix_spawn_file_actions_t *file_actions,
	   const posix_spawnattr_t *attrp, char *const argv[],
	   char *const envp[], int xflags,
	   int (*exec) (const char *, char *const *, char *const *))
{
  struct posix_spawn_args args;
  int ec;

  if (__pipe2 (args.pipe, O_CLOEXEC))
    return errno;

  /* Disable asynchronous cancellation.  */
  int state;
  __libc_ptf_call (__pthread_setcancelstate,
                   (PTHREAD_CANCEL_DISABLE, &state), 0);

  ptrdiff_t argc = 0;
  ptrdiff_t limit = INT_MAX - 1;
  while (argv[argc++] != NULL)
    if (argc == limit)
      {
	errno = E2BIG;
	return errno;
      }

  args.file = file;
  args.exec = exec;
  args.fa = file_actions;
  args.attr = attrp ? attrp : &(const posix_spawnattr_t) { 0 };
  args.argv = argv;
  args.argc = argc;
  args.envp = envp;
  args.xflags = xflags;

  /* Generate the new process.  */
  pid_t new_pid = __fork ();

  if (new_pid == 0)
    __spawni_child (&args);
  else if (new_pid > 0)
    {
      __close (args.pipe[1]);

      if (__read (args.pipe[0], &ec, sizeof ec) != sizeof ec)
	ec = 0;
      else
	__waitpid (new_pid, &(int) { 0 }, 0);
    }
  else
    ec = errno;

  __close (args.pipe[0]);

  if ((ec == 0) && (pid != NULL))
    *pid = new_pid;

  __libc_ptf_call (__pthread_setcancelstate, (state, NULL), 0);

  return ec;
}

int
__spawni (pid_t * pid, const char *file,
	  const posix_spawn_file_actions_t * acts,
	  const posix_spawnattr_t * attrp, char *const argv[],
	  char *const envp[], int xflags)
{
  /* It uses __execvpex to avoid run ENOEXEC in non compatibility mode (it
     will be handled by maybe_script_execute).  */
  return __spawnix (pid, file, acts, attrp, argv, envp, xflags,
		    xflags & SPAWN_XFLAGS_USE_PATH ? __execvpex : __execve);
}

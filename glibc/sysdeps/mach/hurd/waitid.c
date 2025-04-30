/* Implementation of waitid.  Hurd version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.phys.columbia.edu>, 1997.

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
#include <sys/types.h>
#include <sys/wait.h>
#include <stddef.h>
#include <hurd.h>
#include <hurd/port.h>
#include <hurd/version.h>
#include <sysdep-cancel.h>

int
__waitid (idtype_t idtype, id_t id, siginfo_t *infop, int options)
{
  struct rusage ignored;
  error_t err;
  pid_t pid, child;
  int sigcode;
  int status;
  int cancel_oldtype;

  switch (idtype)
    {
    case P_PID:
      if (id <= 0)
	goto invalid;
      pid = (pid_t) id;
      break;
    case P_PGID:
      if (id < 0 || id == 1)
	goto invalid;
      pid = (pid_t) -id;
      break;
    case P_ALL:
      pid = -1;
      break;
    default:
    invalid:
      __set_errno (EINVAL);
      return -1;
    }

  /* Technically we're supposed to return EFAULT if infop is bogus,
     but that would involve mucking with signals, which is
     too much hassle.  User will have to deal with SIGSEGV/SIGBUS.
     We just check for a null pointer. */

  if (infop == NULL)
    {
      __set_errno (EFAULT);
      return -1;
    }

  cancel_oldtype = LIBC_CANCEL_ASYNC();
#if HURD_INTERFACE_VERSION >= 20201227
  err = __USEPORT_CANCEL (PROC, __proc_waitid (port, pid, options,
					       &status, &sigcode,
					       &ignored, &child));
  if (err == MIG_BAD_ID || err == EOPNOTSUPP)
#endif
    err = __USEPORT_CANCEL (PROC, __proc_wait (port, pid, options,
					       &status, &sigcode,
					       &ignored, &child));
  LIBC_CANCEL_RESET (cancel_oldtype);

  if (err == EAGAIN)
    {
      /* POSIX.1-2008, Technical Corrigendum 1 XSH/TC1-2008/0713 [153] states
	 that if waitid returns because WNOHANG was specified and status is
	 not available for any process specified by idtype and id, then the
	 si_signo and si_pid members of the structure pointed to by infop
	 shall be set to zero.  */
      infop->si_signo = 0;
      infop->si_code = 0;
      return 0;
    }

  if (err != 0)
    return __hurd_fail (err);

  /* Decode the status field and set infop members... */
  infop->si_signo = SIGCHLD;
  infop->si_pid = child;
  infop->si_errno = 0;

  if (WIFEXITED (status))
    {
      infop->si_code = CLD_EXITED;
      infop->si_status = WEXITSTATUS (status);
    }
  else if (WIFSIGNALED (status))
    {
      infop->si_code = WCOREDUMP (status) ? CLD_DUMPED : CLD_KILLED;
      infop->si_status = WTERMSIG (status);
    }
  else if (WIFSTOPPED (status))
    {
      infop->si_code = CLD_STOPPED;
      infop->si_status = WSTOPSIG (status);
    }
  else if (WIFCONTINUED (status))
    {
      infop->si_code = CLD_CONTINUED;
      infop->si_status = SIGCONT;
    }

  return 0;
}
weak_alias (__waitid, waitid)
strong_alias (__waitid, __libc_waitid)

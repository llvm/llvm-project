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

#include <hurd.h>
#include <hurd/resource.h>

/* Set the priority of all processes specified by WHICH and WHO
   to PRIO.  Returns 0 on success, -1 on errors.  */
int
__setpriority (enum __priority_which which, id_t who, int prio)
{
  error_t err;
  error_t pidloser, priloser;
  unsigned int npids, ntasks, nwin, nperm, nacces;

  error_t setonepriority (pid_t pid, struct procinfo *pi)
    {
      task_t task;
      error_t piderr = __USEPORT (PROC, __proc_pid2task (port, pid, &task));
      if (piderr == EPERM)
	++nperm;
      if (piderr != ESRCH)
	{
	  ++npids;
	  if (piderr && piderr != EPERM)
	    pidloser = piderr;
	}
      if (! piderr)
	{
	  error_t prierr;
	  ++ntasks;
#ifdef POLICY_TIMESHARE_BASE_COUNT
	  {
	    /* XXX This assumes timeshare policy.  */
	    struct policy_timeshare_base base
	      = { NICE_TO_MACH_PRIORITY (prio) };
	    prierr = __task_policy (task, POLICY_TIMESHARE,
				    (policy_base_t) &base,
				    POLICY_TIMESHARE_BASE_COUNT,
				    0, 1);
	  }
#else
	  prierr = __task_priority (task, NICE_TO_MACH_PRIORITY (prio), 1);
#endif
	  __mach_port_deallocate (__mach_task_self (), task);
	  switch (prierr)
	    {
	    case KERN_FAILURE:
	      ++nacces;
	      break;
	    case KERN_SUCCESS:
	      ++nwin;
	      break;
	    case KERN_INVALID_ARGUMENT: /* Task died.  */
	      --npids;
	      --ntasks;
	      break;
	    default:
	      priloser = prierr;
	    }
	}
      return 0;
    }

  npids = ntasks = nwin = nperm = nacces = 0;
  pidloser = priloser = 0;
  err = _hurd_priority_which_map (which, who, setonepriority, 0);

  if (!err && npids == 0)
    /* No error, but no pids found.  */
    err = ESRCH;
  else if (nperm == npids)
    /* Got EPERM from proc_task2pid for every process.  */
    err = EPERM;
  else if (nacces == ntasks)
    /* Got KERN_FAILURE from task_priority for every task.  */
    err = EACCES;
  else if (nwin == 0)
    err = pidloser ?: priloser;

  return err ? __hurd_fail (err) : 0;
}
libc_hidden_def (__setpriority)
weak_alias (__setpriority, setpriority)

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

#include <limits.h>
#include <hurd.h>
#include <hurd/resource.h>

/* Return the highest priority of any process specified by WHICH and WHO
   (see <sys/resource.h>); if WHO is zero, the current process, process group,
   or user (as specified by WHO) is used.  A lower priority number means higher
   priority.  Priorities range from PRIO_MIN to PRIO_MAX.  */
int
__getpriority (enum __priority_which which, id_t who)
{
  error_t err, onerr;
  int maxpri = INT_MIN;
  struct procinfo *pip;		/* Just for sizeof.  */
  int pibuf[sizeof *pip + 2 * sizeof (pip->threadinfos[0])], *pi = pibuf;
  size_t pisize = sizeof pibuf / sizeof pibuf[0];

  error_t getonepriority (pid_t pid, struct procinfo *pip)
    {
      if (pip)
	onerr = 0;
      else
	{
	  int *oldpi = pi;
	  size_t oldpisize = pisize;
	  char *tw = 0;
	  size_t twsz = 0;
	  int flags = PI_FETCH_TASKINFO;
	  onerr = __USEPORT (PROC, __proc_getprocinfo (port, pid, &flags,
						       &pi, &pisize,
						       &tw, &twsz));
	  if (twsz)
	    __vm_deallocate (__mach_task_self (), (vm_address_t) tw, twsz);
	  if (pi != oldpi && oldpi != pibuf)
	    /* Old buffer from last call was not reused; free it.  */
	    __vm_deallocate (__mach_task_self (),
			     (vm_address_t) oldpi, oldpisize * sizeof pi[0]);
	  pip = (struct procinfo *) pi;
	}
#ifdef TASK_SCHED_TIMESHARE_INFO
      if (!onerr && pip->timeshare_base_info.base_priority > maxpri)
	maxpri = pip->timeshare_base_info.base_priority;
#else
      if (!onerr && pip->taskinfo.base_priority > maxpri)
	maxpri = pip->taskinfo.base_priority;
#endif
      return 0;
    }

  onerr = 0;
  err = _hurd_priority_which_map (which, who,
				  getonepriority, PI_FETCH_TASKINFO);

  if (pi != pibuf)
    __vm_deallocate (__mach_task_self (),
		     (vm_address_t) pi, pisize * sizeof pi[0]);

  if (!err && maxpri == INT_MIN)
    /* No error, but no pids found.  */
    err = onerr ?: ESRCH;

  if (err)
    return __hurd_fail (err);

  return MACH_PRIORITY_TO_NICE (maxpri);
}
libc_hidden_def (__getpriority)
weak_alias (__getpriority, getpriority)

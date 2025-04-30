/* Support code for dealing with priorities in the Hurd.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <sys/mman.h>
#include <unistd.h>

error_t
_hurd_priority_which_map (enum __priority_which which, int who,
			  error_t (*function) (pid_t, struct procinfo *),
			  int pi_flags)
{
  mach_msg_type_number_t npids = 64, i;
  pid_t pidbuf[npids], *pids = pidbuf;
  error_t err;
  struct procinfo *pip;
  int pibuf[sizeof *pip + 5 * sizeof (pip->threadinfos[0])], *pi = pibuf;
  mach_msg_type_number_t pisize = sizeof (pibuf) / sizeof (int);

  switch (which)
    {
    default:
      return EINVAL;

    case PRIO_PROCESS:
      err = (*function) (who ?: getpid (), 0); /* XXX special-case self? */
      break;

    case PRIO_PGRP:
      err = __USEPORT (PROC, __proc_getpgrppids (port, who, &pids, &npids));
      for (i = 0; !err && i < npids; ++i)
	err = (*function) (pids[i], 0);
      break;

    case PRIO_USER:
      if (who == 0)
	who = __geteuid ();
      err = __USEPORT (PROC, __proc_getallpids (port, &pids, &npids));
      for (i = 0; !err && i < npids; ++i)
	{
	  /* Get procinfo to check the owner.  */
	  int *oldpi = pi;
	  mach_msg_type_number_t oldpisize = pisize;
	  char *tw = 0;
	  size_t twsz = 0;
	  err = __USEPORT (PROC, __proc_getprocinfo (port, pids[i],
						     &pi_flags,
						     &pi, &pisize,
						     &tw, &twsz));
	  if (!err)
	    {
	      if (twsz)		/* Gratuitous.  */
		__munmap (tw, twsz);
	      if (pi != oldpi && oldpi != pibuf)
		/* Old buffer from last call was not reused; free it.  */
		__munmap (oldpi, oldpisize * sizeof pi[0]);

	      pip = (struct procinfo *) pi;
	      if (pip->owner == (uid_t) who)
		err = (*function) (pids[i], pip);
	    }
	}
      break;
    }

  if (pids != pidbuf)
    __munmap (pids, npids * sizeof pids[0]);
  if (pi != pibuf)
    __munmap (pi, pisize * sizeof pi[0]);

  return err;
}

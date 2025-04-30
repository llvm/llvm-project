/* Get file-specific information about a file.  Linux version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sysdep.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/param.h>
#include <not-cancel.h>
#include <ldsodefs.h>
#include <sysconf-sigstksz.h>

/* Legacy value of ARG_MAX.  The macro is now not defined since the
   actual value varies based on the stack size.  */
#define legacy_ARG_MAX 131072

/* Newer kernels (4.13) limit the maximum command line arguments lengths to
   6MiB.  */
#define maximum_ARG_MAX (6 * 1024 * 1024)

static long int posix_sysconf (int name);


/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  const char *procfname = NULL;

  switch (name)
    {
    case _SC_MONOTONIC_CLOCK:
    case _SC_CPUTIME:
    case _SC_THREAD_CPUTIME:
      return _POSIX_VERSION;

    case _SC_ARG_MAX:
      {
        struct rlimit rlimit;
        /* Use getrlimit to get the stack limit.  */
        if (__getrlimit (RLIMIT_STACK, &rlimit) == 0)
	  {
	    const long int limit = MAX (legacy_ARG_MAX, rlimit.rlim_cur / 4);
	    return MIN (limit, maximum_ARG_MAX);
	  }

        return legacy_ARG_MAX;
      }

    case _SC_NGROUPS_MAX:
      /* Try to read the information from the /proc/sys/kernel/ngroups_max
	 file.  */
      procfname = "/proc/sys/kernel/ngroups_max";
      break;

    case _SC_SIGQUEUE_MAX:
      {
        struct rlimit rlimit;
        if (__getrlimit (RLIMIT_SIGPENDING, &rlimit) == 0)
	  return rlimit.rlim_cur;

        /* The /proc/sys/kernel/rtsig-max file contains the answer.  */
        procfname = "/proc/sys/kernel/rtsig-max";
      }
      break;

    case _SC_MINSIGSTKSZ:
      assert (GLRO(dl_minsigstacksize) != 0);
      return GLRO(dl_minsigstacksize);

    case _SC_SIGSTKSZ:
      return sysconf_sigstksz ();

    default:
      break;
    }

  if (procfname != NULL)
    {
      int fd = __open_nocancel (procfname, O_RDONLY | O_CLOEXEC);
      if (fd != -1)
	{
	  /* This is more than enough, the file contains a single integer.  */
	  char buf[32];
	  ssize_t n;
	  n = TEMP_FAILURE_RETRY (__read_nocancel (fd, buf, sizeof (buf) - 1));
	  __close_nocancel_nostatus (fd);

	  if (n > 0)
	    {
	      /* Terminate the string.  */
	      buf[n] = '\0';

	      char *endp;
	      long int res = strtol (buf, &endp, 10);
	      if (endp != buf && (*endp == '\0' || *endp == '\n'))
		return res;
	    }
	}
    }

  return posix_sysconf (name);
}

/* Now the POSIX version.  */
#undef __sysconf
#define __sysconf static posix_sysconf
#include <sysdeps/posix/sysconf.c>

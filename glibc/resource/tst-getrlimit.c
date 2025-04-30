/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <stdbool.h>
#include <stdio.h>
#include <sys/resource.h>


static struct
{
  const char *name;
  int resource;
  bool required;
} tests[] =
  {
    /* The following 7 limits are part of POSIX and must exist.  */
    { "RLIMIT_CORE", RLIMIT_CORE, true },
    { "RLIMIT_CPU", RLIMIT_CPU, true },
    { "RLIMIT_DATA", RLIMIT_DATA, true },
    { "RLIMIT_FSIZE", RLIMIT_FSIZE, true },
    { "RLIMIT_NOFILE", RLIMIT_NOFILE, true },
    { "RLIMIT_STACK", RLIMIT_STACK, true },
    { "RLIMIT_AS", RLIMIT_AS, true },
    /* The following are traditional Unix limits which are also
       expected (by us).  */
    { "RLIMIT_RSS", RLIMIT_RSS, true },
    { "RLIMIT_NPROC", RLIMIT_NPROC, true },
    /* The following are extensions.  */
#ifdef RLIMIT_MEMLOCK
    { "RLIMIT_MEMLOCK", RLIMIT_MEMLOCK, false },
#endif
#ifdef RLIMIT_LOCKS
    { "RLIMIT_LOCKS", RLIMIT_LOCKS, false },
#endif
#ifdef RLIMIT_SIGPENDING
    { "RLIMIT_SIGPENDING", RLIMIT_SIGPENDING, false },
#endif
#ifdef RLIMIT_MSGQUEUE
    { "RLIMIT_MSGQUEUE", RLIMIT_MSGQUEUE, false },
#endif
#ifdef RLIMIT_NICE
    { "RLIMIT_NICE", RLIMIT_NICE, false },
#endif
#ifdef RLIMIT_RTPRIO
    { "RLIMIT_RTPRIO", RLIMIT_RTPRIO, false },
#endif
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  int status = 0;

  for (int i = 0; i < ntests; ++i)
    {
      bool this_ok = true;

      struct rlimit r;
      int res = getrlimit (tests[i].resource, &r);
      if (res == -1)
	{
	  if (errno == EINVAL)
	    {
	      if (tests[i].required)
		{
		  printf ("limit %s expectedly not available for getrlimit\n",
			  tests[i].name);
		  status = 1;
		  this_ok = false;
		}
	    }
	  else
	    {
	      printf ("getrlimit for %s returned unexpected error: %m\n",
		      tests[i].name);
	      status = 1;
	      this_ok = false;
	    }
	}

      struct rlimit64 r64;
      res = getrlimit64 (tests[i].resource, &r64);
      if (res == -1)
	{
	  if (errno == EINVAL)
	    {
	      if (tests[i].required)
		{
		  printf ("limit %s expectedly not available for getrlimit64"
			  "\n", tests[i].name);
		  status = 1;
		  this_ok = false;
		}
	    }
	  else
	    {
	      printf ("getrlimit64 for %s returned unexpected error: %m\n",
		      tests[i].name);
	      status = 1;
	      this_ok = false;
	    }
	}

      if (this_ok)
	printf ("limit %s OK\n", tests[i].name);
    }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

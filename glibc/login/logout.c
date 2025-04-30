/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <string.h>
#include <utmp.h>
#include <time.h>
#include <sys/time.h>
#include <shlib-compat.h>

int
__logout (const char *line)
{
  struct utmp tmp, utbuf;
  struct utmp *ut;
  int result = 0;

  /* Tell that we want to use the UTMP file.  */
  if (__utmpname (_PATH_UTMP) == -1)
    return 0;

  /* Open UTMP file.  */
  __setutent ();

  /* Fill in search information.  */
  tmp.ut_type = USER_PROCESS;
  strncpy (tmp.ut_line, line, sizeof tmp.ut_line);

  /* Read the record.  */
  if (__getutline_r (&tmp, &utbuf, &ut) >= 0)
    {
      /* Clear information about who & from where.  */
      memset (ut->ut_name, '\0', sizeof ut->ut_name);
      memset (ut->ut_host, '\0', sizeof ut->ut_host);

      struct __timespec64 ts;
      __clock_gettime64 (CLOCK_REALTIME, &ts);
      TIMESPEC_TO_TIMEVAL (&ut->ut_tv, &ts);
      ut->ut_type = DEAD_PROCESS;

      if (__pututline (ut) != NULL)
	result = 1;
    }

  /* Close UTMP file.  */
  __endutent ();

  return result;
}
versioned_symbol (libc, __logout, logout, GLIBC_2_34);
libc_hidden_ver (__logout, logout)

#if OTHER_SHLIB_COMPAT (libutil, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libutil, __logout, logout, GLIBC_2_0);
#endif

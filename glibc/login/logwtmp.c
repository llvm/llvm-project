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

#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <utmp.h>
#include <struct___timespec64.h>
#include <shlib-compat.h>

void
__logwtmp (const char *line, const char *name, const char *host)
{
  struct utmp ut;

  /* Set information in new entry.  */
  memset (&ut, 0, sizeof (ut));
  ut.ut_pid = getpid ();
  ut.ut_type = name[0] ? USER_PROCESS : DEAD_PROCESS;
  strncpy (ut.ut_line, line, sizeof ut.ut_line);
  strncpy (ut.ut_name, name, sizeof ut.ut_name);
  strncpy (ut.ut_host, host, sizeof ut.ut_host);

  struct __timespec64 ts;
  __clock_gettime64 (CLOCK_REALTIME, &ts);
  TIMESPEC_TO_TIMEVAL (&ut.ut_tv, &ts);

  __updwtmp (_PATH_WTMP, &ut);
}
versioned_symbol (libc, __logwtmp, logwtmp, GLIBC_2_34);
libc_hidden_ver (__logwtmp, logwtmp)

#if OTHER_SHLIB_COMPAT (libutil, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libutil, __logwtmp, logwtmp, GLIBC_2_0);
#endif

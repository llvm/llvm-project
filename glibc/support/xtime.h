/* Support functionality for using time.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_TIME_H
#define SUPPORT_TIME_H

#include <time.h>

__BEGIN_DECLS

/* Name of the env variable, which indicates if it is possible to
   adjust time on target machine.  */
#define SETTIME_ENV_NAME "GLIBC_TEST_ALLOW_TIME_SETTING"

/* The following functions call the corresponding libc functions and
   terminate the process on error.  */

#ifndef __USE_TIME_BITS64
void xclock_gettime (clockid_t clock, struct timespec *ts);
void xclock_settime (clockid_t clock, const struct timespec *ts);
#else
void __REDIRECT (xclock_gettime, (clockid_t clock, struct timespec *ts),
		 xclock_gettime_time64);
void __REDIRECT (xclock_settime, (clockid_t clock, const struct timespec *ts),
		 xclock_settime_time64);
#endif

/* This helper can often simplify tests by avoiding an explicit
   variable declaration or allowing that declaration to be const. */

static inline struct timespec xclock_now (clockid_t clock)
{
  struct timespec ts;
  xclock_gettime (clock, &ts);
  return ts;
}

__END_DECLS

#endif /* SUPPORT_TIME_H */

/* Compatibility definitions for 'struct timeval' with 32-bit time_t.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#ifndef _TV32_COMPAT_H
#define _TV32_COMPAT_H 1

#include <bits/types/time_t.h>
#include <sys/resource.h>

/* Structures containing 'struct timeval' with 32-bit time_t.  */
struct __itimerval32
{
  struct __timeval32 it_interval;
  struct __timeval32 it_value;
};

struct __rusage32
{
  struct __timeval32 ru_utime;	/* user time used */
  struct __timeval32 ru_stime;	/* system time used */
  long ru_maxrss;		/* maximum resident set size */
  long ru_ixrss;		/* integral shared memory size */
  long ru_idrss;		/* integral unshared data size */
  long ru_isrss;		/* integral unshared stack size */
  long ru_minflt;		/* page reclaims */
  long ru_majflt;		/* page faults */
  long ru_nswap;		/* swaps */
  long ru_inblock;		/* block input operations */
  long ru_oublock;		/* block output operations */
  long ru_msgsnd;		/* messages sent */
  long ru_msgrcv;		/* messages received */
  long ru_nsignals;		/* signals received */
  long ru_nvcsw;		/* voluntary context switches */
  long ru_nivcsw;		/* involuntary " */
};

static inline void
rusage32_to_rusage64 (const struct __rusage32 *restrict r32,
                    struct __rusage64 *restrict r64)
{
  /* Make sure the entire output structure is cleared, including
     padding and reserved fields.  */
  memset (r64, 0, sizeof *r64);

  r64->ru_utime    = valid_timeval32_to_timeval64 (r32->ru_utime);
  r64->ru_stime    = valid_timeval32_to_timeval64 (r32->ru_stime);
  r64->ru_maxrss   = r32->ru_maxrss;
  r64->ru_ixrss    = r32->ru_ixrss;
  r64->ru_idrss    = r32->ru_idrss;
  r64->ru_isrss    = r32->ru_isrss;
  r64->ru_minflt   = r32->ru_minflt;
  r64->ru_majflt   = r32->ru_majflt;
  r64->ru_nswap    = r32->ru_nswap;
  r64->ru_inblock  = r32->ru_inblock;
  r64->ru_oublock  = r32->ru_oublock;
  r64->ru_msgsnd   = r32->ru_msgsnd;
  r64->ru_msgrcv   = r32->ru_msgrcv;
  r64->ru_nsignals = r32->ru_nsignals;
  r64->ru_nvcsw    = r32->ru_nvcsw;
  r64->ru_nivcsw   = r32->ru_nivcsw;
}

static inline void
rusage64_to_rusage32 (const struct __rusage64 *restrict r64,
                    struct __rusage32 *restrict r32)
{
  /* Make sure the entire output structure is cleared, including
     padding and reserved fields.  */
  memset (r32, 0, sizeof *r32);

  r32->ru_utime    = valid_timeval64_to_timeval32 (r64->ru_utime);
  r32->ru_stime    = valid_timeval64_to_timeval32 (r64->ru_stime);
  r32->ru_maxrss   = r64->ru_maxrss;
  r32->ru_ixrss    = r64->ru_ixrss;
  r32->ru_idrss    = r64->ru_idrss;
  r32->ru_isrss    = r64->ru_isrss;
  r32->ru_minflt   = r64->ru_minflt;
  r32->ru_majflt   = r64->ru_majflt;
  r32->ru_nswap    = r64->ru_nswap;
  r32->ru_inblock  = r64->ru_inblock;
  r32->ru_oublock  = r64->ru_oublock;
  r32->ru_msgsnd   = r64->ru_msgsnd;
  r32->ru_msgrcv   = r64->ru_msgrcv;
  r32->ru_nsignals = r64->ru_nsignals;
  r32->ru_nvcsw    = r64->ru_nvcsw;
  r32->ru_nivcsw   = r64->ru_nivcsw;
}

#endif /* tv32-compat.h */

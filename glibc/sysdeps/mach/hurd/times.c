/* Return CPU and real time used by process and its children.  Hurd version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <stddef.h>
#include <sys/resource.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mach.h>
#include <mach/task_info.h>
#include <hurd.h>

static inline clock_t
clock_from_time_value (const time_value_t *t)
{
  return t->seconds * 1000000 + t->microseconds;
}

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t
__times (struct tms *tms)
{
  struct task_basic_info bi;
  struct task_thread_times_info tti;
  mach_msg_type_number_t count;
  time_value_t now;
  error_t err;

  count = TASK_BASIC_INFO_COUNT;
  err = __task_info (__mach_task_self (), TASK_BASIC_INFO,
		     (task_info_t) &bi, &count);
  if (err)
    return __hurd_fail (err);

  count = TASK_THREAD_TIMES_INFO_COUNT;
  err = __task_info (__mach_task_self (), TASK_THREAD_TIMES_INFO,
		     (task_info_t) &tti, &count);
  if (err)
    return __hurd_fail (err);

  tms->tms_utime = (clock_from_time_value (&bi.user_time)
		    + clock_from_time_value (&tti.user_time));
  tms->tms_stime = (clock_from_time_value (&bi.system_time)
		    + clock_from_time_value (&tti.system_time));

  /* XXX This can't be implemented until getrusage(RUSAGE_CHILDREN) can be.  */
  tms->tms_cutime = tms->tms_cstime = 0;

  __host_get_time (__mach_host_self (), &now);

  return (clock_from_time_value (&now)
	  - clock_from_time_value (&bi.creation_time));
}
weak_alias (__times, times)

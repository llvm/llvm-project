/* Return the CPU time used by the program so far.  Hurd version.
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

#include <time.h>
#include <sys/time.h>
#include <mach.h>
#include <mach/task_info.h>
#include <hurd.h>

/* Return the time used by the program so far (user time + system time).  */
clock_t
clock (void)
{
  struct task_basic_info bi;
  struct task_thread_times_info tti;
  mach_msg_type_number_t count;
  clock_t total;
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

  total = bi.user_time.seconds * 1000000 + bi.user_time.microseconds;
  total += tti.user_time.seconds * 1000000 + tti.user_time.microseconds;
  total += bi.system_time.seconds * 1000000 + bi.system_time.microseconds;
  total += tti.system_time.seconds * 1000000 + tti.system_time.microseconds;

  return total;
}

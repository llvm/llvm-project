/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <time.h>
#include <mach.h>
#include <assert.h>
#include <shlib-compat.h>

/* Get the current time of day, putting it into *TS.
   Returns 0 on success, -1 on errors.  */
int
__clock_gettime (clockid_t clock_id, struct timespec *ts)
{
  mach_msg_type_number_t count;
  error_t err;

  switch (clock_id) {

    case CLOCK_REALTIME:
      {
	/* __host_get_time can only fail if passed an invalid host_t.
	   __mach_host_self could theoretically fail (producing an
	   invalid host_t) due to resource exhaustion, but we assume
	   this will never happen.  */
	time_value_t tv;
	__host_get_time (__mach_host_self (), &tv);
	TIME_VALUE_TO_TIMESPEC (&tv, ts);
	return 0;
      }

    case CLOCK_PROCESS_CPUTIME_ID:
      {
	struct time_value t = { .seconds = 0, .microseconds = 0 };
	struct task_basic_info bi;
	struct task_thread_times_info tti;

	/* Dead threads CPU time.  */
	count = TASK_BASIC_INFO_COUNT;
	err = __task_info (__mach_task_self (), TASK_BASIC_INFO,
			   (task_info_t) &bi, &count);
	if (err)
	  {
	    __set_errno(err);
	    return -1;
	  }
	time_value_add (&t, &bi.user_time);
	time_value_add (&t, &bi.system_time);

	/* Live threads CPU time.  */
	count = TASK_EVENTS_INFO_COUNT;
	err = __task_info (__mach_task_self (), TASK_THREAD_TIMES_INFO,
			   (task_info_t) &tti, &count);
	if (err)
	  {
	    __set_errno(err);
	    return -1;
	  }
	time_value_add (&t, &tti.user_time);
	time_value_add (&t, &tti.system_time);

	TIME_VALUE_TO_TIMESPEC(&t, ts);
	return 0;
      }

    case CLOCK_THREAD_CPUTIME_ID:
      {
	struct thread_basic_info bi;
	mach_port_t self = __mach_thread_self ();

	count = THREAD_BASIC_INFO_COUNT;
	err = __thread_info (self, THREAD_BASIC_INFO,
			     (thread_info_t) &bi, &count);
	__mach_port_deallocate (__mach_task_self (), self);
	if (err)
	  {
	    __set_errno(err);
	    return -1;
	  }
	time_value_add (&bi.user_time, &bi.system_time);

	TIME_VALUE_TO_TIMESPEC(&bi.user_time, ts);
	return 0;
      }
  }

  errno = EINVAL;
  return -1;
}
libc_hidden_def (__clock_gettime)

versioned_symbol (libc, __clock_gettime, clock_gettime, GLIBC_2_17);
/* clock_gettime moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
strong_alias (__clock_gettime, __clock_gettime_2);
compat_symbol (libc, __clock_gettime_2, clock_gettime, GLIBC_2_2);
#endif

int
__clock_gettime64 (clockid_t clock_id, struct __timespec64 *ts64)
{
  struct timespec ts;
  int ret;

  ret = __clock_gettime (clock_id, &ts);
  if (ret == 0)
    *ts64 = valid_timespec_to_timespec64 (ts);

  return ret;
}
libc_hidden_def (__clock_gettime64)

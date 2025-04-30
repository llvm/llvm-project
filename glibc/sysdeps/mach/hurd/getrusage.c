/* getrusage -- Get resource usage information about processes.  Hurd version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <sys/resource.h>
#include <mach.h>
#include <mach/task_info.h>
#include <hurd.h>

/* Return resource usage information on process indicated by WHO
   and put it in *USAGE.  Returns 0 for success, -1 for failure.  */
int
__getrusage (enum __rusage_who who, struct rusage *usage)
{
  struct task_basic_info bi;
  struct task_events_info ei;
  struct task_thread_times_info tti;
  mach_msg_type_number_t count;
  error_t err;

  switch (who)
    {
    case RUSAGE_SELF:
      count = TASK_BASIC_INFO_COUNT;
      err = __task_info (__mach_task_self (), TASK_BASIC_INFO,
			 (task_info_t) &bi, &count);
      if (err)
	return __hurd_fail (err);

      count = TASK_EVENTS_INFO_COUNT;
      err = __task_info (__mach_task_self (), TASK_EVENTS_INFO,
			 (task_info_t) &ei, &count);
      if (err == KERN_INVALID_ARGUMENT)	/* microkernel doesn't implement it */
	memset (&ei, 0, sizeof ei);
      else if (err)
	return __hurd_fail (err);

      count = TASK_THREAD_TIMES_INFO_COUNT;
      err = __task_info (__mach_task_self (), TASK_THREAD_TIMES_INFO,
			 (task_info_t) &tti, &count);
      if (err)
	return __hurd_fail (err);

      time_value_add (&bi.user_time, &tti.user_time);
      time_value_add (&bi.system_time, &tti.system_time);

      memset (usage, 0, sizeof (struct rusage));

      usage->ru_utime.tv_sec = bi.user_time.seconds;
      usage->ru_utime.tv_usec = bi.user_time.microseconds;
      usage->ru_stime.tv_sec = bi.system_time.seconds;
      usage->ru_stime.tv_usec = bi.system_time.microseconds;

      /* These statistics map only approximately.  */
      usage->ru_majflt = ei.pageins;
      usage->ru_minflt = ei.faults - ei.pageins;
      usage->ru_msgsnd = ei.messages_sent; /* Mach IPC, not SysV IPC */
      usage->ru_msgrcv = ei.messages_received; /* Mach IPC, not SysV IPC */
      break;

    case RUSAGE_CHILDREN:
      /* XXX Not implemented yet.  However, zero out USAGE to be
         consistent with the wait3 and wait4 functions.  */
      memset (usage, 0, sizeof (struct rusage));

      break;

    default:
      return EINVAL;
    }

  return 0;
}

weak_alias (__getrusage, getrusage)

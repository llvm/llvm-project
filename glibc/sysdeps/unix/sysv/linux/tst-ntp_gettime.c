/* Test for ntp_gettime.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <sys/timex.h>
#include <support/check.h>
#include <support/xtime.h>

#ifndef NTP_GETTIME_SYSCALL
# define NTP_GETTIME_SYSCALL ntp_gettime
#endif

#define STR(__s) #__s

static int
do_test (void)
{
  struct timespec tv_before_ntp, tv_after_ntp;
  struct ntptimeval ntv;

  /* To prevent seconds rollover (which is very unlikely though),
     loop until we do match seconds values before and after
     call to ntp_gettime.  */
  do
    {
      tv_before_ntp = xclock_now (CLOCK_REALTIME);

      int ret = NTP_GETTIME_SYSCALL (&ntv);
      if (ret == -1)
        FAIL_EXIT1 (STR(NTP_GETTIME_SYSCALL)" failed: %m\n");

      tv_after_ntp = xclock_now (CLOCK_REALTIME);
    }
  while (tv_after_ntp.tv_sec != tv_before_ntp.tv_sec);

  TEST_COMPARE (tv_after_ntp.tv_sec, ntv.time.tv_sec);
  return 0;
}

#include <support/test-driver.c>

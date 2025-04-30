/* Tests for POSIX timer implementation.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kaz Kylheku <kaz@ashi.footprints.net>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>


static void
notify_func1 (union sigval sigval)
{
  puts ("notify_func1");
}


static void
notify_func2 (union sigval sigval)
{
  puts ("notify_func2");
}


static void
signal_func (int sig)
{
  static const char text[] = "signal_func\n";
  signal (sig, signal_func);
  write (STDOUT_FILENO, text, sizeof text - 1);
}

static void
intr_sleep (int sec)
{
  struct timespec ts;

  ts.tv_sec = sec;
  ts.tv_nsec = 0;

  while (nanosleep (&ts, &ts) == -1 && errno == EINTR)
    ;
}

#define ZSIGALRM 14


int
main (void)
{
  struct timespec ts;
  timer_t timer_sig, timer_thr1, timer_thr2;
  int retval;
  struct sigevent sigev1 =
  {
    .sigev_notify = SIGEV_SIGNAL,
    .sigev_signo = ZSIGALRM
  };
  struct sigevent sigev2;
  struct itimerspec itimer1 = { { 0, 200000000 }, { 0, 200000000 } };
  struct itimerspec itimer2 = { { 0, 100000000 }, { 0, 500000000 } };
  struct itimerspec itimer3 = { { 0, 150000000 }, { 0, 300000000 } };
  struct itimerspec old;

  retval = clock_gettime (CLOCK_REALTIME, &ts);

  sigev2.sigev_notify = SIGEV_THREAD;
  sigev2.sigev_notify_function = notify_func1;
  sigev2.sigev_notify_attributes = NULL;
  /* It is unnecessary to do the following but to set a good example
     we do it anyhow.  */
  sigev2.sigev_value.sival_ptr = NULL;

  setvbuf (stdout, 0, _IOLBF, 0);

  printf ("clock_gettime returned %d, timespec = { %jd, %jd }\n",
	  retval, (intmax_t) ts.tv_sec, (intmax_t) ts.tv_nsec);

  retval = clock_getres (CLOCK_REALTIME, &ts);

  printf ("clock_getres returned %d, timespec = { %jd, %jd }\n",
	  retval, (intmax_t) ts.tv_sec, (intmax_t) ts.tv_nsec);

  if (timer_create (CLOCK_REALTIME, &sigev1, &timer_sig) != 0)
    {
      printf ("timer_create for timer_sig failed: %m\n");
      exit (1);
    }
  if (timer_create (CLOCK_REALTIME, &sigev2, &timer_thr1) != 0)
    {
      printf ("timer_create for timer_thr1 failed: %m\n");
      exit (1);
    }
  sigev2.sigev_notify_function = notify_func2;
  if (timer_create (CLOCK_REALTIME, &sigev2, &timer_thr2) != 0)
    {
      printf ("timer_create for timer_thr2 failed: %m\n");
      exit (1);
    }

  if (timer_settime (timer_thr1, 0, &itimer2, &old) != 0)
    {
      printf ("timer_settime for timer_thr1 failed: %m\n");
      exit (1);
    }
  if (timer_settime (timer_thr2, 0, &itimer3, &old) != 0)
    {
      printf ("timer_settime for timer_thr2 failed: %m\n");
      exit (1);
    }

  signal (ZSIGALRM, signal_func);

  if (timer_settime (timer_sig, 0, &itimer1, &old) != 0)
    {
      printf ("timer_settime for timer_sig failed: %m\n");
      exit (1);
    }

  intr_sleep (3);

  if (timer_delete (timer_sig) != 0)
    {
      printf ("timer_delete for timer_sig failed: %m\n");
      exit (1);
    }
  if (timer_delete (timer_thr1) != 0)
    {
      printf ("timer_delete for timer_thr1 failed: %m\n");
      exit (1);
    }

  intr_sleep (3);

  if (timer_delete (timer_thr2) != 0)
    {
      printf ("timer_delete for timer_thr2 failed: %m\n");
      exit (1);
    }

  return 0;
}

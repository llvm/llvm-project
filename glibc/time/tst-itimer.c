/* Basic tests for getitimer and setitimer.
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

#include <array_length.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <unistd.h>
#include <time.h>

static sig_atomic_t cnt;

static void
alrm_handler (int sig)
{
  if (++cnt > 3)
    cnt = 3;
}

static void
intr_sleep (int sec)
{
  struct timespec ts = { .tv_sec = sec, .tv_nsec = 0 };
  while (nanosleep (&ts, &ts) == -1 && errno == EINTR)
    ;
}

static int
do_test (void)
{
  struct itimerval it, it_old;
  const int timers[] = { ITIMER_REAL, ITIMER_VIRTUAL, ITIMER_PROF };
  for (int i = 0; i < array_length (timers); i++)
    {
      TEST_COMPARE (getitimer (timers[i], &it), 0);

      /* No timer set, all value should be 0.  */
      TEST_COMPARE (it.it_interval.tv_sec, 0);
      TEST_COMPARE (it.it_interval.tv_usec, 0);
      TEST_COMPARE (it.it_value.tv_sec, 0);
      TEST_COMPARE (it.it_value.tv_usec, 0);

      it.it_interval.tv_sec = 10;
      it.it_interval.tv_usec = 20;
      TEST_COMPARE (setitimer (timers[i], &it, NULL), 0);

      TEST_COMPARE (setitimer (timers[i], &(struct itimerval) { 0 }, &it_old),
		    0);
      /* ITIMER_REAL returns { 0, 0 } for single-shot timers, while
	 other timers returns setitimer value.  */
      if (timers[i] == ITIMER_REAL)
	{
	  TEST_COMPARE (it_old.it_interval.tv_sec, 0);
	  TEST_COMPARE (it_old.it_interval.tv_usec, 0);
	}
      else
	{
	  TEST_COMPARE (it_old.it_interval.tv_sec, 10);
	  TEST_COMPARE (it_old.it_interval.tv_usec, 20);
	}

      /* Create a periodic timer and check if the return value is the one
	 previously set.  */
      it.it_interval.tv_sec = 10;
      it.it_interval.tv_usec = 20;
      it.it_value.tv_sec = 30;
      it.it_value.tv_usec = 40;
      TEST_COMPARE (setitimer (timers[i], &it, NULL), 0);

      TEST_COMPARE (setitimer (timers[i], &(struct itimerval) { 0 }, &it_old),
		    0);
      TEST_COMPARE (it.it_interval.tv_sec, it_old.it_interval.tv_sec);
      TEST_COMPARE (it.it_interval.tv_usec, it_old.it_interval.tv_usec);

      if (sizeof (time_t) == 4)
	continue;

      /* Same as before, but with a 64 bit time_t value.  */
      it.it_interval.tv_sec = (time_t) 0x1ffffffffull;
      it.it_interval.tv_usec = 20;
      it.it_value.tv_sec = 0;
      it.it_value.tv_usec = 0;

      /* Linux does not provide 64 bit time_t support for getitimer and
	 setitimer on architectures with 32 bit time_t support.  */
      if (sizeof (__time_t) == 8)
	{
	  TEST_COMPARE (setitimer (timers[i], &it, NULL), 0);
	  TEST_COMPARE (setitimer (timers[i], &(struct itimerval) { 0 },
				   &it_old),
			0);
	  /* ITIMER_REAL returns { 0, 0 } for single-sort timers, while other
	     timers returns setitimer value.  */
	  if (timers[i] == ITIMER_REAL)
	    {
	      TEST_COMPARE (it_old.it_interval.tv_sec, 0ull);
	      TEST_COMPARE (it_old.it_interval.tv_usec, 0);
	    }
	  else
	    {
	      TEST_COMPARE (it_old.it_interval.tv_sec, 0x1ffffffffull);
	      TEST_COMPARE (it_old.it_interval.tv_usec, 20);
	    }
	}
      else
	{
	  TEST_COMPARE (setitimer (timers[i], &it, NULL), -1);
	  TEST_COMPARE (errno, EOVERFLOW);
	}

      /* Create a periodic timer and check if the return value is the one
	 previously set.  */
      it.it_interval.tv_sec = (time_t) 0x1ffffffffull;
      it.it_interval.tv_usec = 20;
      it.it_value.tv_sec = 30;
      it.it_value.tv_usec = 40;
      if (sizeof (__time_t) == 8)
	{
	  TEST_COMPARE (setitimer (timers[i], &it, NULL), 0);

	  TEST_COMPARE (setitimer (timers[i], &(struct itimerval) { 0 },
				   &it_old),
			0);
	  TEST_COMPARE (it.it_interval.tv_sec, it_old.it_interval.tv_sec);
	  TEST_COMPARE (it.it_interval.tv_usec, it_old.it_interval.tv_usec);
	}
      else
	{
	  TEST_COMPARE (setitimer (timers[i], &it, NULL), -1);
	  TEST_COMPARE (errno, EOVERFLOW);
	}
  }

  {
    struct sigaction sa = { .sa_handler = alrm_handler, .sa_flags = 0 };
    sigemptyset (&sa.sa_mask);
    xsigaction (SIGALRM, &sa, NULL);
  }

  /* Setup a timer to 0.1s and sleep for 1s and check to 3 signal handler
     execution.  */
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = 100000;
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 100000;

  /* Check ITIMER_VIRTUAL and ITIMER_PROF would require to generate load
     and be subject to system load.  */
  cnt = 0;
  TEST_COMPARE (setitimer (ITIMER_REAL, &it, NULL), 0);
  intr_sleep (1);
  TEST_COMPARE (cnt, 3);
  TEST_COMPARE (setitimer (ITIMER_REAL, &(struct itimerval) { 0 }, NULL), 0);

  return 0;
}

#include <support/test-driver.c>

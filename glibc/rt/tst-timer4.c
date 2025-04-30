/* Tests for POSIX timer implementation.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004

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
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#if _POSIX_THREADS && defined SA_SIGINFO
# include <pthread.h>

# ifndef TEST_CLOCK
#  define TEST_CLOCK		CLOCK_REALTIME
# endif

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

timer_t timer_none, timer_sig1, timer_sig2, timer_thr1, timer_thr2;

int thr1_cnt, thr1_err;
union sigval thr1_sigval;
struct timespec thr1_ts;

static void
thr1 (union sigval sigval)
{
  pthread_mutex_lock (&lock);
  thr1_err = clock_gettime (TEST_CLOCK, &thr1_ts);
  if (thr1_cnt >= 5)
    {
      struct itimerspec it = { };
      thr1_err |= timer_settime (timer_thr1, 0, &it, NULL);
    }
  thr1_sigval = sigval;
  ++thr1_cnt;
  pthread_cond_signal (&cond);
  pthread_mutex_unlock (&lock);
}

int thr2_cnt, thr2_err;
union sigval thr2_sigval;
size_t thr2_guardsize;
struct timespec thr2_ts;

static void
thr2 (union sigval sigval)
{
  pthread_attr_t nattr;
  int err = 0;
  size_t guardsize = -1;
  int ret = pthread_getattr_np (pthread_self (), &nattr);
  if (ret)
    {
      errno = ret;
      printf ("*** pthread_getattr_np failed: %m\n");
      err = 1;
    }
  else
    {
      ret = pthread_attr_getguardsize (&nattr, &guardsize);
      if (ret)
        {
          errno = ret;
          printf ("*** pthread_attr_getguardsize failed: %m\n");
          err = 1;
        }
      if (pthread_attr_destroy (&nattr) != 0)
        {
          puts ("*** pthread_attr_destroy failed");
          err = 1;
        }
    }
  pthread_mutex_lock (&lock);
  thr2_err = clock_gettime (TEST_CLOCK, &thr2_ts) | err;
  if (thr2_cnt >= 5)
    {
      struct itimerspec it = { };
      thr2_err |= timer_settime (timer_thr2, 0, &it, NULL);
    }
  thr2_sigval = sigval;
  ++thr2_cnt;
  thr2_guardsize = guardsize;
  pthread_cond_signal (&cond);
  pthread_mutex_unlock (&lock);
}

volatile int sig1_cnt, sig1_err;
volatile union sigval sig1_sigval;
struct timespec sig1_ts;

static void
sig1_handler (int sig, siginfo_t *info, void *ctx)
{
  int err = 0;
  if (sig != SIGRTMIN) err |= 1 << 0;
  if (info->si_signo != SIGRTMIN) err |= 1 << 1;
  if (info->si_code != SI_TIMER) err |= 1 << 2;
  if (clock_gettime (TEST_CLOCK, &sig1_ts) != 0)
    err |= 1 << 3;
  if (sig1_cnt >= 5)
    {
      struct itimerspec it = { };
      if (timer_settime (timer_sig1, 0, &it, NULL))
	err |= 1 << 4;
    }
  sig1_err |= err;
  sig1_sigval = info->si_value;
  ++sig1_cnt;
}

volatile int sig2_cnt, sig2_err;
volatile union sigval sig2_sigval;
struct timespec sig2_ts;

static void
sig2_handler (int sig, siginfo_t *info, void *ctx)
{
  int err = 0;
  if (sig != SIGRTMIN + 1) err |= 1 << 0;
  if (info->si_signo != SIGRTMIN + 1) err |= 1 << 1;
  if (info->si_code != SI_TIMER) err |= 1 << 2;
  if (clock_gettime (TEST_CLOCK, &sig2_ts) != 0)
    err |= 1 << 3;
  if (sig2_cnt >= 5)
    {
      struct itimerspec it = { };
      if (timer_settime (timer_sig2, 0, &it, NULL))
	err |= 1 << 4;
    }
  sig2_err |= err;
  sig2_sigval = info->si_value;
  ++sig2_cnt;
}

/* Check if end is later or equal to start + nsec.  */
static int
check_ts (const char *name, const struct timespec *start,
	  const struct timespec *end, long msec)
{
  struct timespec ts = *start;

  ts.tv_sec += msec / 1000000;
  ts.tv_nsec += (msec % 1000000) * 1000;
  if (ts.tv_nsec >= 1000000000)
    {
      ++ts.tv_sec;
      ts.tv_nsec -= 1000000000;
    }
  if (end->tv_sec < ts.tv_sec
      || (end->tv_sec == ts.tv_sec && end->tv_nsec < ts.tv_nsec))
    {
      printf ("\
*** timer %s invoked too soon: %ld.%09jd instead of expected %ld.%09jd\n",
	      name, (long) end->tv_sec, (intmax_t) end->tv_nsec,
	      (long) ts.tv_sec, (intmax_t) ts.tv_nsec);
      return 1;
    }
  else
    return 0;
}

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

#ifdef TEST_CLOCK_MISSING
  const char *missing = TEST_CLOCK_MISSING (TEST_CLOCK);
  if (missing != NULL)
    {
      printf ("%s missing, skipping test\n", missing);
      return 0;
    }
#endif

  struct timespec ts;
  if (clock_gettime (TEST_CLOCK, &ts) != 0)
    {
      printf ("*** clock_gettime failed: %m\n");
      result = 1;
    }
  else
    printf ("clock_gettime returned timespec = { %ld, %jd }\n",
	    (long) ts.tv_sec, (intmax_t) ts.tv_nsec);

  if (clock_getres (TEST_CLOCK, &ts) != 0)
    {
      printf ("*** clock_getres failed: %m\n");
      result = 1;
    }
  else
    printf ("clock_getres returned timespec = { %ld, %jd }\n",
	    (long) ts.tv_sec, (intmax_t) ts.tv_nsec);

  struct sigevent ev;
  memset (&ev, 0x11, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (timer_create (TEST_CLOCK, &ev, &timer_none) != 0)
    {
      printf ("*** timer_create for timer_none failed: %m\n");
      return 1;
    }

  struct sigaction sa = { .sa_sigaction = sig1_handler,
			  .sa_flags = SA_SIGINFO };
  sigemptyset (&sa.sa_mask);
  sigaction (SIGRTMIN, &sa, NULL);
  sa.sa_sigaction = sig2_handler;
  sigaction (SIGRTMIN + 1, &sa, NULL);

  memset (&ev, 0x22, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = &ev;
  if (timer_create (TEST_CLOCK, &ev, &timer_sig1) != 0)
    {
      printf ("*** timer_create for timer_sig1 failed: %m\n");
      return 1;
    }

  memset (&ev, 0x33, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN + 1;
  ev.sigev_value.sival_int = 163;
  if (timer_create (TEST_CLOCK, &ev, &timer_sig2) != 0)
    {
      printf ("*** timer_create for timer_sig2 failed: %m\n");
      return 1;
    }

  memset (&ev, 0x44, sizeof (ev));
  ev.sigev_notify = SIGEV_THREAD;
  ev.sigev_notify_function = thr1;
  ev.sigev_notify_attributes = NULL;
  ev.sigev_value.sival_ptr = &ev;
  if (timer_create (TEST_CLOCK, &ev, &timer_thr1) != 0)
    {
      printf ("*** timer_create for timer_thr1 failed: %m\n");
      return 1;
    }

  pthread_attr_t nattr;
  if (pthread_attr_init (&nattr)
      || pthread_attr_setguardsize (&nattr, 0))
    {
      puts ("*** pthread_attr_t setup failed");
      result = 1;
    }

  memset (&ev, 0x55, sizeof (ev));
  ev.sigev_notify = SIGEV_THREAD;
  ev.sigev_notify_function = thr2;
  ev.sigev_notify_attributes = &nattr;
  ev.sigev_value.sival_int = 111;
  if (timer_create (TEST_CLOCK, &ev, &timer_thr2) != 0)
    {
      printf ("*** timer_create for timer_thr2 failed: %m\n");
      return 1;
    }

  int ret = timer_getoverrun (timer_thr1);
  if (ret != 0)
    {
      if (ret == -1)
	printf ("*** timer_getoverrun failed: %m\n");
      else
	printf ("*** timer_getoverrun returned %d != 0\n", ret);
      result = 1;
    }

  struct itimerspec it;
  it.it_value.tv_sec = 0;
  it.it_value.tv_nsec = -26;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_nsec = 0;
  if (timer_settime (timer_sig1, 0, &it, NULL) == 0)
    {
      puts ("*** timer_settime with negative tv_nsec unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("*** timer_settime with negative tv_nsec did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 100000;
  it.it_interval.tv_nsec = 1000000000;
  if (timer_settime (timer_sig2, 0, &it, NULL) == 0)
    {
      puts ("\
*** timer_settime with tv_nsec 1000000000 unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("*** timer_settime with tv_nsec 1000000000 did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

#if 0
  it.it_value.tv_nsec = 0;
  it.it_interval.tv_nsec = -26;
  if (timer_settime (timer_thr1, 0, &it, NULL) != 0)
    {
      printf ("\
!!! timer_settime with it_value 0 it_interval invalid failed: %m\n");
      /* FIXME: is this mandated by POSIX?
      result = 1; */
    }

  it.it_interval.tv_nsec = 3000000000;
  if (timer_settime (timer_thr2, 0, &it, NULL) != 0)
    {
      printf ("\
!!! timer_settime with it_value 0 it_interval invalid failed: %m\n");
      /* FIXME: is this mandated by POSIX?
      result = 1; */
    }
#endif

  struct timespec startts;
  if (clock_gettime (TEST_CLOCK, &startts) != 0)
    {
      printf ("*** clock_gettime failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 100000000;
  it.it_interval.tv_nsec = 0;
  if (timer_settime (timer_none, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_none failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 200000000;
  if (timer_settime (timer_thr1, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_thr1 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 300000000;
  if (timer_settime (timer_thr2, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_thr2 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 400000000;
  if (timer_settime (timer_sig1, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_sig1 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 500000000;
  if (TEMP_FAILURE_RETRY (timer_settime (timer_sig2, 0, &it, NULL)) != 0)
    {
      printf ("*** timer_settime timer_sig2 failed: %m\n");
      result = 1;
    }

  pthread_mutex_lock (&lock);
  while (thr1_cnt == 0 || thr2_cnt == 0)
    pthread_cond_wait (&cond, &lock);
  pthread_mutex_unlock (&lock);

  while (sig1_cnt == 0 || sig2_cnt == 0)
    {
      ts.tv_sec = 0;
      ts.tv_nsec = 100000000;
      nanosleep (&ts, NULL);
    }

  pthread_mutex_lock (&lock);

  if (thr1_cnt != 1)
    {
      printf ("*** thr1 not called exactly once, but %d times\n", thr1_cnt);
      result = 1;
    }
  else if (thr1_err)
    {
      puts ("*** an error occurred in thr1");
      result = 1;
    }
  else if (thr1_sigval.sival_ptr != &ev)
    {
      printf ("*** thr1_sigval.sival_ptr %p != %p\n",
	      thr1_sigval.sival_ptr, &ev);
      result = 1;
    }
  else if (check_ts ("thr1", &startts, &thr1_ts, 200000))
    result = 1;

  if (thr2_cnt != 1)
    {
      printf ("*** thr2 not called exactly once, but %d times\n", thr2_cnt);
      result = 1;
    }
  else if (thr2_err)
    {
      puts ("*** an error occurred in thr2");
      result = 1;
    }
  else if (thr2_sigval.sival_int != 111)
    {
      printf ("*** thr2_sigval.sival_ptr %d != 111\n", thr2_sigval.sival_int);
      result = 1;
    }
  else if (check_ts ("thr2", &startts, &thr2_ts, 300000))
    result = 1;
  else if (thr2_guardsize != 0)
    {
      printf ("*** thr2 guardsize %zd != 0\n", thr2_guardsize);
      result = 1;
    }

  pthread_mutex_unlock (&lock);

  if (sig1_cnt != 1)
    {
      printf ("*** sig1 not called exactly once, but %d times\n", sig1_cnt);
      result = 1;
    }
  else if (sig1_err)
    {
      printf ("*** errors occurred in sig1 handler %x\n", sig1_err);
      result = 1;
    }
  else if (sig1_sigval.sival_ptr != &ev)
    {
      printf ("*** sig1_sigval.sival_ptr %p != %p\n",
	      sig1_sigval.sival_ptr, &ev);
      result = 1;
    }
  else if (check_ts ("sig1", &startts, &sig1_ts, 400000))
    result = 1;

  if (sig2_cnt != 1)
    {
      printf ("*** sig2 not called exactly once, but %d times\n", sig2_cnt);
      result = 1;
    }
  else if (sig2_err)
    {
      printf ("*** errors occurred in sig2 handler %x\n", sig2_err);
      result = 1;
    }
  else if (sig2_sigval.sival_int != 163)
    {
      printf ("*** sig2_sigval.sival_ptr %d != 163\n", sig2_sigval.sival_int);
      result = 1;
    }
  else if (check_ts ("sig2", &startts, &sig2_ts, 500000))
    result = 1;

  if (timer_gettime (timer_none, &it) != 0)
    {
      printf ("*** timer_gettime timer_none failed: %m\n");
      result = 1;
    }
  else if (it.it_value.tv_sec || it.it_value.tv_nsec
	   || it.it_interval.tv_sec || it.it_interval.tv_nsec)
    {
      printf ("\
*** timer_gettime timer_none returned { %ld.%09jd, %ld.%09jd }\n",
	      (long) it.it_value.tv_sec, (intmax_t) it.it_value.tv_nsec,
	      (long) it.it_interval.tv_sec, (intmax_t) it.it_interval.tv_nsec);
      result = 1;
    }

  if (clock_gettime (TEST_CLOCK, &startts) != 0)
    {
      printf ("*** clock_gettime failed: %m\n");
      result = 1;
    }

  it.it_value.tv_sec = 1;
  it.it_value.tv_nsec = 0;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_nsec = 100000000;
  if (timer_settime (timer_none, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_none failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 100000000;
  it.it_interval.tv_nsec = 200000000;
  if (timer_settime (timer_thr1, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_thr1 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 200000000;
  it.it_interval.tv_nsec = 300000000;
  if (timer_settime (timer_thr2, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_thr2 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 300000000;
  it.it_interval.tv_nsec = 400000000;
  if (timer_settime (timer_sig1, 0, &it, NULL) != 0)
    {
      printf ("*** timer_settime timer_sig1 failed: %m\n");
      result = 1;
    }

  it.it_value.tv_nsec = 400000000;
  it.it_interval.tv_nsec = 500000000;
  if (TEMP_FAILURE_RETRY (timer_settime (timer_sig2, 0, &it, NULL)) != 0)
    {
      printf ("*** timer_settime timer_sig2 failed: %m\n");
      result = 1;
    }

  pthread_mutex_lock (&lock);
  while (thr1_cnt < 6 || thr2_cnt < 6)
    pthread_cond_wait (&cond, &lock);
  pthread_mutex_unlock (&lock);

  while (sig1_cnt < 6 || sig2_cnt < 6)
    {
      ts.tv_sec = 0;
      ts.tv_nsec = 100000000;
      nanosleep (&ts, NULL);
    }

  pthread_mutex_lock (&lock);

  if (thr1_err)
    {
      puts ("*** an error occurred in thr1");
      result = 1;
    }
  else if (check_ts ("thr1", &startts, &thr1_ts, 1100000 + 4 * 200000))
    result = 1;

  if (thr2_err)
    {
      puts ("*** an error occurred in thr2");
      result = 1;
    }
  else if (check_ts ("thr2", &startts, &thr2_ts, 1200000 + 4 * 300000))
    result = 1;
  else if (thr2_guardsize != 0)
    {
      printf ("*** thr2 guardsize %zd != 0\n", thr2_guardsize);
      result = 1;
    }

  pthread_mutex_unlock (&lock);

  if (sig1_err)
    {
      printf ("*** errors occurred in sig1 handler %x\n", sig1_err);
      result = 1;
    }
  else if (check_ts ("sig1", &startts, &sig1_ts, 1300000 + 4 * 400000))
    result = 1;

  if (sig2_err)
    {
      printf ("*** errors occurred in sig2 handler %x\n", sig2_err);
      result = 1;
    }
  else if (check_ts ("sig2", &startts, &sig2_ts, 1400000 + 4 * 500000))
    result = 1;

  if (timer_gettime (timer_none, &it) != 0)
    {
      printf ("*** timer_gettime timer_none failed: %m\n");
      result = 1;
    }
  else if (it.it_interval.tv_sec || it.it_interval.tv_nsec != 100000000)
    {
      printf ("\
!!! second timer_gettime timer_none returned it_interval %ld.%09jd\n",
	      (long) it.it_interval.tv_sec, (intmax_t) it.it_interval.tv_nsec);
      /* FIXME: For now disabled.
      result = 1; */
    }

  if (timer_delete (timer_none) != 0)
    {
      printf ("*** timer_delete for timer_none failed: %m\n");
      result = 1;
    }

  if (timer_delete (timer_sig1) != 0)
    {
      printf ("*** timer_delete for timer_sig1 failed: %m\n");
      result = 1;
    }

  if (timer_delete (timer_sig2) != 0)
    {
      printf ("*** timer_delete for timer_sig2 failed: %m\n");
      result = 1;
    }

  if (timer_delete (timer_thr1) != 0)
    {
      printf ("*** timer_delete for timer_thr1 failed: %m\n");
      result = 1;
    }

  if (timer_delete (timer_thr2) != 0)
    {
      printf ("*** timer_delete for timer_thr2 failed: %m\n");
      result = 1;
    }
  return result;
}

#elif defined TEST_CLOCK_MISSING
/* This just ensures that any functions called in TEST_CLOCK_MISSING
   are not diagnosed as unused.  */
# define TEST_FUNCTION (TEST_CLOCK_MISSING (TEST_CLOCK), 0)
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"

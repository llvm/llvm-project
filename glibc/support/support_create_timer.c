/* Create a periodic timer.
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

#include <support/check.h>
#include <support/support.h>
#include <support/xsignal.h>
#include <time.h>

static void
dummy_alrm_handler (int sig)
{
}

timer_t
support_create_timer (uint64_t sec, long int nsec, bool repeat,
		      void (*callback)(int))
{
  struct sigaction sa;
  sa.sa_handler = callback != NULL ? callback : dummy_alrm_handler;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  xsigaction (SIGALRM, &sa, NULL);

  struct sigevent ev = {
    .sigev_notify = SIGEV_SIGNAL,
    .sigev_signo = SIGALRM
  };
  timer_t timerid;
  int r = timer_create (CLOCK_REALTIME, &ev, &timerid);
  if (r == -1)
    FAIL_EXIT1 ("timer_create: %m");

  /* Single timer with 0.1s.  */
  struct itimerspec its =
    {
      { .tv_sec = repeat ? sec : 0, .tv_nsec = repeat ? nsec : 0 },
      { .tv_sec = sec, .tv_nsec = nsec }
    };
  r = timer_settime (timerid, 0, &its, NULL);
  if (r == -1)
    FAIL_EXIT1 ("timer_settime: %m");

  return timerid;
}

/* Disable the timer TIMER.  */
void
support_delete_timer (timer_t timer)
{
  int r = timer_delete (timer);
  if (r == -1)
    FAIL_EXIT1 ("timer_delete: %m");
  xsignal (SIGALRM, SIG_DFL);
}

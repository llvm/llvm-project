/* Tests for computing deadlines for timeouts.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <inet/net-internal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <support/check.h>

/* Find the maximum value which can be represented in a time_t.  */
static time_t
time_t_max (void)
{
  _Static_assert (0 > (time_t) -1, "time_t is signed");
  uintmax_t current = 1;
  while (true)
    {
      uintmax_t next = current * 2;
      /* This cannot happen because time_t is signed.  */
      TEST_VERIFY_EXIT (next > current);
      ++next;
      if ((time_t) next < 0 || next != (uintmax_t) (time_t) next)
        /* Value cannot be represented in time_t.  Return the previous
           value. */
        return current;
      current = next;
    }
}

static int
do_test (void)
{
  {
    struct deadline_current_time current_time = __deadline_current_time ();
    TEST_VERIFY (current_time.current.tv_sec >= 0);
    current_time = __deadline_current_time ();
    /* Due to CLOCK_MONOTONIC, either seconds or nanoseconds are
       greater than zero.  This is also true for the gettimeofday
       fallback.  */
    TEST_VERIFY (current_time.current.tv_sec >= 0);
    TEST_VERIFY (current_time.current.tv_sec > 0
                 || current_time.current.tv_nsec > 0);
  }

  /* Check basic computations of deadlines.  */
  struct deadline_current_time current_time = { { 1, 123456789 } };
  struct deadline deadline = __deadline_from_timeval
    (current_time, (struct timeval) { 0, 1 });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 123457789);
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1);

  deadline = __deadline_from_timeval
    (current_time, ((struct timeval) { 0, 2 }));
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 123458789);
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1);

  deadline = __deadline_from_timeval
    (current_time, ((struct timeval) { 1, 0 }));
  TEST_VERIFY (deadline.absolute.tv_sec == 2);
  TEST_VERIFY (deadline.absolute.tv_nsec == 123456789);
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1000);

  /* Check if timeouts are correctly rounded up to the next
     millisecond.  */
  for (int i = 0; i < 999999; ++i)
    {
      ++current_time.current.tv_nsec;
      TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1000);
    }

  /* A full millisecond has elapsed, so the time to the deadline is
     now less than 1000.  */
  ++current_time.current.tv_nsec;
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 999);

  /* Check __deadline_to_ms carry-over.  */
  current_time = (struct deadline_current_time) { { 9, 123456789 } };
  deadline = (struct deadline) { { 10, 122456789 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 999);
  deadline = (struct deadline) { { 10, 122456790 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1000);
  deadline = (struct deadline) { { 10, 123456788 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1000);
  deadline = (struct deadline) { { 10, 123456789 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 1000);

  /* Check __deadline_to_ms overflow.  */
  deadline = (struct deadline) { { INT_MAX - 1, 1 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == INT_MAX);

  /* Check __deadline_to_ms for elapsed deadlines.  */
  current_time = (struct deadline_current_time) { { 9, 123456789 } };
  deadline.absolute = current_time.current;
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 0);
  current_time = (struct deadline_current_time) { { 9, 123456790 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 0);
  current_time = (struct deadline_current_time) { { 10, 0 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 0);
  current_time = (struct deadline_current_time) { { 10, 123456788 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 0);
  current_time = (struct deadline_current_time) { { 10, 123456789 } };
  TEST_VERIFY (__deadline_to_ms (current_time, deadline) == 0);

  /* Check carry-over in __deadline_from_timeval.  */
  current_time = (struct deadline_current_time) { { 9, 998000001 } };
  for (int i = 0; i < 2000; ++i)
    {
      deadline = __deadline_from_timeval
        (current_time, (struct timeval) { 1, i });
      TEST_VERIFY (deadline.absolute.tv_sec == 10);
      TEST_VERIFY (deadline.absolute.tv_nsec == 998000001 + i * 1000);
    }
  for (int i = 2000; i < 3000; ++i)
    {
      deadline = __deadline_from_timeval
        (current_time, (struct timeval) { 2, i });
      TEST_VERIFY (deadline.absolute.tv_sec == 12);
      TEST_VERIFY (deadline.absolute.tv_nsec == 1 + (i - 2000) * 1000);
    }

  /* Check infinite deadlines.  */
  deadline = __deadline_from_timeval
    ((struct deadline_current_time) { { 0, 1000 * 1000 * 1000 - 1000 } },
     (struct timeval) { time_t_max (), 1 });
  TEST_VERIFY (__deadline_is_infinite (deadline));
  deadline = __deadline_from_timeval
    ((struct deadline_current_time) { { 0, 1000 * 1000 * 1000 - 1001 } },
     (struct timeval) { time_t_max (), 1 });
  TEST_VERIFY (!__deadline_is_infinite (deadline));
  deadline = __deadline_from_timeval
    ((struct deadline_current_time)
       { { time_t_max (), 1000 * 1000 * 1000 - 1000 } },
     (struct timeval) { 0, 1 });
  TEST_VERIFY (__deadline_is_infinite (deadline));
  deadline = __deadline_from_timeval
    ((struct deadline_current_time)
       { { time_t_max () / 2 + 1, 0 } },
     (struct timeval) { time_t_max () / 2 + 1, 0 });
  TEST_VERIFY (__deadline_is_infinite (deadline));

  /* Check __deadline_first behavior.  */
  deadline = __deadline_first
    ((struct deadline) { { 1, 2 } },
     (struct deadline) { { 1, 3 } });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 2);
  deadline = __deadline_first
    ((struct deadline) { { 1, 3 } },
     (struct deadline) { { 1, 2 } });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 2);
  deadline = __deadline_first
    ((struct deadline) { { 1, 2 } },
     (struct deadline) { { 2, 1 } });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 2);
  deadline = __deadline_first
    ((struct deadline) { { 1, 2 } },
     (struct deadline) { { 2, 4 } });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 2);
  deadline = __deadline_first
    ((struct deadline) { { 2, 4 } },
     (struct deadline) { { 1, 2 } });
  TEST_VERIFY (deadline.absolute.tv_sec == 1);
  TEST_VERIFY (deadline.absolute.tv_nsec == 2);

  return 0;
}

#include <support/test-driver.c>

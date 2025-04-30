/* Computing deadlines for timeouts.
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

#include <net-internal.h>

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

struct deadline_current_time
__deadline_current_time (void)
{
  struct deadline_current_time result;
  if (__clock_gettime64 (CLOCK_MONOTONIC, &result.current) != 0)
    __clock_gettime64 (CLOCK_REALTIME, &result.current);
  assert (result.current.tv_sec >= 0);
  return result;
}

/* A special deadline value for which __deadline_is_infinite is
   true.  */
static inline struct deadline
infinite_deadline (void)
{
  return (struct deadline) { { -1, -1 } };
}

struct deadline
__deadline_from_timeval (struct deadline_current_time current,
                         struct timeval tv)
{
  assert (__is_timeval_valid_timeout (tv));

  /* Compute second-based deadline.  Perform the addition in
     uintmax_t, which is unsigned, to simply overflow detection.  */
  uintmax_t sec = current.current.tv_sec;
  sec += tv.tv_sec;
  if (sec < (uintmax_t) tv.tv_sec)
    return infinite_deadline ();

  /* Compute nanosecond deadline.  */
  int nsec = current.current.tv_nsec + tv.tv_usec * 1000;
  if (nsec >= 1000 * 1000 * 1000)
    {
      /* Carry nanosecond overflow to seconds.  */
      nsec -= 1000 * 1000 * 1000;
      if (sec + 1 < sec)
        return infinite_deadline ();
      ++sec;
    }
  /* This uses a GCC extension, otherwise these casts for detecting
     overflow would not be defined.  */
  if ((time_t) sec < 0 || sec != (uintmax_t) (time_t) sec)
    return infinite_deadline ();

  return (struct deadline) { { sec, nsec } };
}

int
__deadline_to_ms (struct deadline_current_time current,
                  struct deadline deadline)
{
  if (__deadline_is_infinite (deadline))
    return INT_MAX;

  if (current.current.tv_sec > deadline.absolute.tv_sec
      || (current.current.tv_sec == deadline.absolute.tv_sec
          && current.current.tv_nsec >= deadline.absolute.tv_nsec))
    return 0;
  time_t sec = deadline.absolute.tv_sec - current.current.tv_sec;
  if (sec >= INT_MAX)
    /* This value will overflow below.  */
    return INT_MAX;
  int nsec = deadline.absolute.tv_nsec - current.current.tv_nsec;
  if (nsec < 0)
    {
      /* Borrow from the seconds field.  */
      assert (sec > 0);
      --sec;
      nsec += 1000 * 1000 * 1000;
    }

  /* Prepare for rounding up to milliseconds.  */
  nsec += 999999;
  if (nsec > 1000 * 1000 * 1000)
    {
      assert (sec < INT_MAX);
      ++sec;
      nsec -= 1000 * 1000 * 1000;
    }

  unsigned int msec = nsec / (1000 * 1000);
  if (sec > INT_MAX / 1000)
    return INT_MAX;
  msec += sec * 1000;
  if (msec > INT_MAX)
    return INT_MAX;
  return msec;
}

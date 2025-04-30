/* Support code for timespec checks.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <support/timespec.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <intprops.h>

void
test_timespec_before_impl (const char *file, int line, struct timespec left,
			   struct timespec right)
{
  if (left.tv_sec > right.tv_sec
      || (left.tv_sec == right.tv_sec
	  && left.tv_nsec > right.tv_nsec)) {
    support_record_failure ();
    const struct timespec diff = timespec_sub (left, right);
    printf ("%s:%d: %jd.%09jds not before %jd.%09jds "
	    "(difference %jd.%09jds)\n",
	    file, line,
	    (intmax_t) left.tv_sec, (intmax_t) left.tv_nsec,
	    (intmax_t) right.tv_sec, (intmax_t) right.tv_nsec,
	    (intmax_t) diff.tv_sec, (intmax_t) diff.tv_nsec);
  }
}

void
test_timespec_equal_or_after_impl (const char *file, int line,
				   struct timespec left,
				   struct timespec right)
{
  if (left.tv_sec < right.tv_sec
      || (left.tv_sec == right.tv_sec
	  && left.tv_nsec < right.tv_nsec)) {
    support_record_failure ();
    const struct timespec diff = timespec_sub (right, left);
    printf ("%s:%d: %jd.%09jds not after %jd.%09jds "
	    "(difference %jd.%09jds)\n",
	    file, line,
	    (intmax_t) left.tv_sec, (intmax_t) left.tv_nsec,
	    (intmax_t) right.tv_sec, (intmax_t) right.tv_nsec,
	    (intmax_t) diff.tv_sec, (intmax_t) diff.tv_nsec);
  }
}

/* Convert TIME to nanoseconds stored in a time_t.
   Returns time_t maximum or minimum if the conversion overflows
   or underflows, respectively.  */
time_t
support_timespec_ns (struct timespec time)
{
  time_t time_ns;
  if (INT_MULTIPLY_WRAPV(time.tv_sec, TIMESPEC_HZ, &time_ns))
    return time.tv_sec < 0 ? TYPE_MINIMUM(time_t) : TYPE_MAXIMUM(time_t);
  if (INT_ADD_WRAPV(time_ns, time.tv_nsec, &time_ns))
    return time.tv_nsec < 0 ? TYPE_MINIMUM(time_t) : TYPE_MAXIMUM(time_t);
  return time_ns;
}

/* Returns time normalized timespec with .tv_nsec < TIMESPEC_HZ
   and the whole seconds  added to .tv_sec. If an overflow or
   underflow occurs the values are clamped to its maximum or
   minimum respectively.  */
struct timespec
support_timespec_normalize (struct timespec time)
{
  struct timespec norm;
  if (INT_ADD_WRAPV (time.tv_sec, (time.tv_nsec / TIMESPEC_HZ), &norm.tv_sec))
   {
     norm.tv_sec = (time.tv_nsec < 0) ? TYPE_MINIMUM (time_t): TYPE_MAXIMUM (time_t);
     norm.tv_nsec = (time.tv_nsec < 0) ? -1 * (TIMESPEC_HZ - 1) : TIMESPEC_HZ - 1;
     return norm;
   }
  norm.tv_nsec = time.tv_nsec % TIMESPEC_HZ;
  return norm;
}

/* Returns TRUE if the observed time is within the given percentage
   bounds of the expected time, and FALSE otherwise.
   For example the call

   support_timespec_check_in_range(expected, observed, 0.5, 1.2);

   will check if

   0.5 of expected <= observed <= 1.2 of expected

   In other words it will check if observed time is within 50% to
   120% of the expected time.  */
int
support_timespec_check_in_range (struct timespec expected, struct timespec observed,
			      double lower_bound, double upper_bound)
{
  assert (upper_bound >= lower_bound);
  time_t expected_norm, observed_norm;
  expected_norm = support_timespec_ns (expected);
  /* Don't divide by zero  */
  assert(expected_norm != 0);
  observed_norm = support_timespec_ns (observed);
  double ratio = (double)observed_norm / expected_norm;
  return (lower_bound <= ratio && ratio <= upper_bound);
}

/* Helpers for utimes/utimens conversions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <hurd/hurd_types.h>
#include <stddef.h>
#include <sys/time.h>

/* Initializes atime/mtime timespec structures from an array of timeval.  */
static inline void
utime_ts_from_tval (const struct timeval tvp[2],
                    struct timespec *atime, struct timespec *mtime)
{
  if (tvp == NULL)
    {
      /* Setting the number of nanoseconds to UTIME_NOW tells the
         underlying filesystems to use the current time.  */
      atime->tv_sec = 0;
      atime->tv_nsec = UTIME_NOW;
      mtime->tv_sec = 0;
      mtime->tv_nsec = UTIME_NOW;
    }
  else
    {
      TIMEVAL_TO_TIMESPEC (&tvp[0], atime);
      TIMEVAL_TO_TIMESPEC (&tvp[1], mtime);
    }
}

/* Initializes atime/mtime time_value_t structures from an array of timeval.  */
static inline void
utime_tvalue_from_tval (const struct timeval tvp[2],
                        time_value_t *atime, time_value_t *mtime)
{
  if (tvp == NULL)
    /* Setting the number of microseconds to `-1' tells the
       underlying filesystems to use the current time.  */
    atime->microseconds = mtime->microseconds = -1;
  else
    {
      atime->seconds = tvp[0].tv_sec;
      atime->microseconds = tvp[0].tv_usec;
      mtime->seconds = tvp[1].tv_sec;
      mtime->microseconds = tvp[1].tv_usec;
    }
}

/* Changes the access time of the file behind PORT using a timeval array.  */
static inline error_t
hurd_futimes (const file_t port, const struct timeval tvp[2])
{
  error_t err;
  struct timespec atime, mtime;

  utime_ts_from_tval (tvp, &atime, &mtime);

  err = __file_utimens (port, atime, mtime);

  if (err == MIG_BAD_ID || err == EOPNOTSUPP)
    {
      time_value_t atim, mtim;

      utime_tvalue_from_tval (tvp, &atim, &mtim);

      err = __file_utimes (port, atim, mtim);
    }

  return err;
}

/* Initializes atime/mtime timespec structures from an array of timespec.  */
static inline void
utime_ts_from_tspec (const struct timespec tsp[2],
                     struct timespec *atime, struct timespec *mtime)
{
  if (tsp == NULL)
    {
      /* Setting the number of nanoseconds to UTIME_NOW tells the
         underlying filesystems to use the current time.  */
      atime->tv_sec = 0;
      atime->tv_nsec = UTIME_NOW;
      mtime->tv_sec = 0;
      mtime->tv_nsec = UTIME_NOW;
    }
  else
    {
      *atime = tsp[0];
      *mtime = tsp[1];
    }
}

/* Initializes atime/mtime time_value_t structures from an array of timespec.  */
static inline void
utime_tvalue_from_tspec (const struct timespec tsp[2],
                         time_value_t *atime, time_value_t *mtime)
{
  if (tsp == NULL)
    /* Setting the number of microseconds to `-1' tells the
       underlying filesystems to use the current time.  */
    atime->microseconds = mtime->microseconds = -1;
  else
    {
      if (tsp[0].tv_nsec == UTIME_NOW)
	atime->microseconds = -1;
      else if (tsp[0].tv_nsec == UTIME_OMIT)
	atime->microseconds = -2;
      else
	TIMESPEC_TO_TIME_VALUE (atime, &(tsp[0]));
      if (tsp[1].tv_nsec == UTIME_NOW)
	mtime->microseconds = -1;
      else if (tsp[1].tv_nsec == UTIME_OMIT)
	mtime->microseconds = -2;
      else
	TIMESPEC_TO_TIME_VALUE (mtime, &(tsp[1]));
    }
}

/* Changes the access time of the file behind PORT using a timespec array.  */
static inline error_t
hurd_futimens (const file_t port, const struct timespec tsp[2])
{
  error_t err;
  struct timespec atime, mtime;

  utime_ts_from_tspec (tsp, &atime, &mtime);

  err = __file_utimens (port, atime, mtime);

  if (err == MIG_BAD_ID || err == EOPNOTSUPP)
    {
      time_value_t atim, mtim;

      utime_tvalue_from_tspec (tsp, &atim, &mtim);

      err = __file_utimes (port, atim, mtim);
    }

  return err;
}

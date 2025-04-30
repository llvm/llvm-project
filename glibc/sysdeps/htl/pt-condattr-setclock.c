/* pthread_condattr_setclock.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <pt-internal.h>

int
pthread_condattr_setclock (pthread_condattr_t *attr, clockid_t clock)
{
  /* Only a few clocks are allowed.  CLOCK_REALTIME is always allowed.
     CLOCK_MONOTONIC only if the kernel has the necessary support.  */
  if (clock == CLOCK_MONOTONIC)
    {
      /* Check whether the clock is available.  */
      static int avail;

      if (avail == 0)
	{
	  struct timespec ts;
	  int res;

	  res = clock_gettime (CLOCK_MONOTONIC, &ts);
	  avail = res < 0 ? -1 : 1;
	}

      if (avail < 0)
	/* Not available.  */
	return EINVAL;
    }
  else if (clock != CLOCK_REALTIME)
    return EINVAL;

  attr->__clock = clock;

  return 0;
}

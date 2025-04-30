/* Return the time used by the program so far (user time + system time).
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <sys/times.h>
#include <time.h>
#include <unistd.h>

clock_t
clock (void)
{
  struct __timespec64 ts;

  _Static_assert (CLOCKS_PER_SEC == 1000000,
		  "CLOCKS_PER_SEC should be 1000000");

  if (__glibc_unlikely (__clock_gettime64 (CLOCK_PROCESS_CPUTIME_ID, &ts) != 0))
    return (clock_t) -1;

  return (ts.tv_sec * CLOCKS_PER_SEC
	  + ts.tv_nsec / (1000000000 / CLOCKS_PER_SEC));
}

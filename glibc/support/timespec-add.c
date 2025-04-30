/* Add two struct timespec values.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library and is also part of gnulib.
   Patches to this file should be submitted to both projects.

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

/* Return the sum of two timespec values A and B.  On overflow, return
   an extremal value.  This assumes 0 <= tv_nsec < TIMESPEC_HZ.  */

#include <config.h>
#include "timespec.h"

#include "intprops.h"

struct timespec
timespec_add (struct timespec a, struct timespec b)
{
  time_t rs = a.tv_sec;
  time_t bs = b.tv_sec;
  int ns = a.tv_nsec + b.tv_nsec;
  int nsd = ns - TIMESPEC_HZ;
  int rns = ns;

  if (0 <= nsd)
    {
      rns = nsd;
      time_t bs1;
      if (!INT_ADD_WRAPV (bs, 1, &bs1))
        bs = bs1;
      else if (rs < 0)
        rs++;
      else
        goto high_overflow;
    }

  if (INT_ADD_WRAPV (rs, bs, &rs))
    {
      if (bs < 0)
        {
          rs = TYPE_MINIMUM (time_t);
          rns = 0;
        }
      else
        {
        high_overflow:
          rs = TYPE_MAXIMUM (time_t);
          rns = TIMESPEC_HZ - 1;
        }
    }

  return (struct timespec) { .tv_sec = rs, .tv_nsec = rns };
}

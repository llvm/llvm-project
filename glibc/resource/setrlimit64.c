/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/resource.h>
#include <sys/types.h>

/* Set the soft and hard limits for RESOURCE to *RLIMITS.
   Only the super-user can increase hard limits.
   Return 0 if successful, -1 if not (and sets errno).  */
int
setrlimit64 (enum __rlimit_resource resource, const struct rlimit64 *rlimits)
{
  struct rlimit rlimits32;

  if (rlimits->rlim_cur >= RLIM_INFINITY)
    rlimits32.rlim_cur = RLIM_INFINITY;
  else
    rlimits32.rlim_cur = rlimits->rlim_cur;
  if (rlimits->rlim_max >= RLIM_INFINITY)
    rlimits32.rlim_max = RLIM_INFINITY;
  else
    rlimits32.rlim_max = rlimits->rlim_max;

  return __setrlimit (resource, &rlimits32);
}

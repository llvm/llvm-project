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

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/resource.h>

/* Set the soft and hard limits for RESOURCE to *RLIMITS.
   Only the super-user can increase hard limits.
   Return 0 if successful, -1 if not (and sets errno).  */
int
__setrlimit (enum __rlimit_resource resource, const struct rlimit *rlimits)
{
  struct rlimit lim;

  if (rlimits == NULL || (unsigned int) resource >= RLIMIT_NLIMITS)
    {
      errno = EINVAL;
      return -1;
    }

  lim = *rlimits;

  /* Even though most limits do nothing, there is no inheritance, and hard
     limits are not really hard, we just let any old call succeed to make
     life easier for programs that expect normal behavior.  */

  if (lim.rlim_cur > lim.rlim_max)
    lim.rlim_cur = lim.rlim_max;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_rlimit_lock);
  _hurd_rlimits[resource] = lim;
  __mutex_unlock (&_hurd_rlimit_lock);
  HURD_CRITICAL_END;

  return 0;
}

libc_hidden_def (__setrlimit)
weak_alias (__setrlimit, setrlimit)

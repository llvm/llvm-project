/* Get POSIX {CHILD_MAX} run-time limit value.  Unix version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>

#ifndef CHILD_MAX
long int
__get_child_max (void)
{
# ifdef RLIMIT_NPROC
  struct rlimit limit;
  if (__getrlimit (RLIMIT_NPROC, &limit) == 0
      && limit.rlim_cur != RLIM_INFINITY)
    return limit.rlim_cur;
# endif

  return -1;
}
#endif

/* Threshold at which to diagnose ELOOP.  Hurd version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _ELOOP_THRESHOLD_H
#define _ELOOP_THRESHOLD_H      1

/* Return the maximum number of symlink traversals to permit
   before diagnosing ELOOP.

   In the Hurd version, here we are actually setting the only policy
   there is on the system.  We use a literal number here rather than
   defining SYMLOOP_MAX so that programs don't compile in a number
   but instead use sysconf and the number can be changed here to
   affect sysconf's result.  */

static inline unsigned int __attribute__ ((const))
__eloop_threshold (void)
{
  return 32;
}

#endif  /* eloop-threshold.h */

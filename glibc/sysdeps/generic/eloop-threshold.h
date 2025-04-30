/* Threshold at which to diagnose ELOOP.  Generic version.
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

#include <limits.h>
#include <sys/param.h>

/* POSIX specifies SYMLOOP_MAX as the "Maximum number of symbolic
   links that can be reliably traversed in the resolution of a
   pathname in the absence of a loop."  This makes it a minimum that
   we should certainly accept.  But it leaves open the possibility
   that more might sometimes work--just not "reliably".

   For example, Linux implements a complex policy whereby there is a
   small limit on the number of direct symlink traversals (a symlink
   to a symlink to a symlink), but larger limit on the total number of
   symlink traversals overall.  Hence the SYMLOOP_MAX number should be
   the small one, but the limit library functions enforce on users
   should be the larger one.

   So, we use the larger of the reported SYMLOOP_MAX (if any) and our
   own constant MIN_ELOOP_THRESHOLD, below.  This constant should be
   large enough that it never rules out a file name and directory tree
   that the underlying system (i.e. calls to 'open' et al) would
   resolve successfully.  It should be small enough that actual loops
   are detected without a huge number of iterations.  */

#ifndef MIN_ELOOP_THRESHOLD
# define MIN_ELOOP_THRESHOLD    40
#endif

/* Return the maximum number of symlink traversals to permit
   before diagnosing ELOOP.  */
static inline unsigned int __attribute__ ((const))
__eloop_threshold (void)
{
#ifdef SYMLOOP_MAX
  const int symloop_max = SYMLOOP_MAX;
#else
  /* The function is marked 'const' even though we use memory and
     call a function, because sysconf is required to return the
     same value in every call and so it must always be safe to
     call __eloop_threshold exactly once and reuse the value.  */
  static long int sysconf_symloop_max;
  if (sysconf_symloop_max == 0)
    sysconf_symloop_max = __sysconf (_SC_SYMLOOP_MAX);
  const unsigned int symloop_max = (sysconf_symloop_max <= 0
                                    ? _POSIX_SYMLOOP_MAX
                                    : sysconf_symloop_max);
#endif

  return MAX (symloop_max, MIN_ELOOP_THRESHOLD);
}

#endif  /* eloop-threshold.h */

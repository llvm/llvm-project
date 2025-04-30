/* Convert between lowlevel sigmask and libc representation of sigset_t.
   Generic version.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Joe Keane <jgk@jgk.org>.

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

/* Convert between an old-style 32-bit signal mask and a POSIX sigset_t.  */

#include <sigsetops.h>

/* Perform *SET = MASK.  Unused bits of *SET are set to 0.
   Returns zero for success or -1 for errors (from sigaddset/sigemptyset).  */
static inline int __attribute__ ((unused))
sigset_set_old_mask (sigset_t *set, int mask)
{
  if (sizeof (__sigset_t) == sizeof (unsigned int))
    *set = (unsigned int) mask;
  else
    {
      unsigned int __sig;

      if (__sigemptyset (set) < 0)
	return -1;

      for (__sig = 1; __sig < NSIG && __sig <= sizeof (mask) * 8; __sig++)
	if (mask & __sigmask (__sig))
	  if (__sigaddset (set, __sig) < 0)
	    return -1;
    }
  return 0;
}

/* Return the sigmask corresponding to *SET.
   Unused bits of *SET are thrown away.  */
static inline int __attribute__ ((unused))
sigset_get_old_mask (const sigset_t *set)
{
  if (sizeof (sigset_t) == sizeof (unsigned int))
    return (unsigned int) *set;
  else
    {
      unsigned int mask = 0;
      unsigned int sig;

      for (sig = 1; sig < NSIG && sig <= sizeof (mask) * 8; sig++)
	if (__sigismember (set, sig))
	  mask |= __sigmask (sig);

      return mask;
    }
}

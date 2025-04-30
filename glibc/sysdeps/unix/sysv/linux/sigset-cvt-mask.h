/* Convert between lowlevel sigmask and libc representation of sigset_t.
   Linux version.
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

static inline int __attribute__ ((unused))
sigset_set_old_mask (sigset_t *set, int mask)
{
  unsigned long int *ptr;
  int cnt;

  ptr = &set->__val[0];

  *ptr++ = (unsigned int) mask;

  cnt = _SIGSET_NWORDS - 2;
  do
    *ptr++ = 0ul;
  while (--cnt >= 0);

  return 0;
}

static inline int __attribute__ ((unused))
sigset_get_old_mask (const sigset_t *set)
{
  return (unsigned int) set->__val[0];
}

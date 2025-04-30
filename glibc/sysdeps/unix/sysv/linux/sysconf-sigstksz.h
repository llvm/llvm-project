/* sysconf_sigstksz ().  Linux version.
   Copyright (C) 2020 Free Software Foundation, Inc.
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

/* Return sysconf (_SC_SIGSTKSZ).  */

static long int
sysconf_sigstksz (void)
{
  long int minsigstacksize = GLRO(dl_minsigstacksize);
  assert (minsigstacksize != 0);
  _Static_assert (__builtin_constant_p (MINSIGSTKSZ),
		  "MINSIGSTKSZ is constant");
  if (minsigstacksize < MINSIGSTKSZ)
    minsigstacksize = MINSIGSTKSZ;
  /* MAX (MINSIGSTKSZ, sysconf (_SC_MINSIGSTKSZ)) * 4.  */
  long int sigstacksize = minsigstacksize * 4;
  /* Return MAX (SIGSTKSZ, sigstacksize).  */
  _Static_assert (__builtin_constant_p (SIGSTKSZ),
		  "SIGSTKSZ is constant");
  if (sigstacksize < SIGSTKSZ)
    sigstacksize = SIGSTKSZ;
  return sigstacksize;
}

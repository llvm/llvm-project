/* Tune kernel clock.  Linux specific syscall.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <sys/timex.h>
#include <sysdep.h>

int
___adjtimex64 (struct __timex64 *tx64)
{
  return __clock_adjtime64 (CLOCK_REALTIME, tx64);
}

#if __TIMESIZE != 64
libc_hidden_def (___adjtimex64)

int
___adjtimex (struct timex *tx)
{
  struct __timex64 tx64;
  int retval;

  tx64 = valid_timex_to_timex64 (*tx);
  retval = ___adjtimex64 (&tx64);
  *tx = valid_timex64_to_timex (tx64);

  return retval;
}
#endif

#ifdef VERSION_adjtimex
weak_alias (___adjtimex, __wadjtimex);
weak_alias (___adjtimex, __wnadjtime);
default_symbol_version (___adjtimex,  __adjtimex, VERSION_adjtimex);
default_symbol_version (__wadjtimex,    adjtimex, VERSION_adjtimex);
default_symbol_version (__wnadjtime, ntp_adjtime, VERSION_adjtimex);
libc_hidden_ver (___adjtimex, __adjtimex);
#else
strong_alias (___adjtimex, __adjtimex)
weak_alias (___adjtimex, adjtimex)
weak_alias (___adjtimex, ntp_adjtime)
libc_hidden_def (__adjtimex)
#endif

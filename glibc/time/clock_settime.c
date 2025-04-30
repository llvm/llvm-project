/* Set a clock to a given value.  Stub version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
#include <time.h>
#include <shlib-compat.h>

/* Set CLOCK to value TP.  */
int
__clock_settime (clockid_t clock_id, const struct timespec *tp)
{
  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__clock_settime)

versioned_symbol (libc, __clock_settime, clock_settime, GLIBC_2_17);
/* clock_settime moved to libc in version 2.17;
   old binaries may expect the symbol version it had in librt.  */
#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_17)
compat_symbol (libc, __clock_settime, clock_settime, GLIBC_2_2);
#endif

stub_warning (clock_settime)

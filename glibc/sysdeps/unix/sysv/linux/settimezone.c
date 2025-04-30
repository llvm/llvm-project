/* Obsolete set system time.  Linux version.
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

#include <errno.h>
#include <sys/time.h>
#include <stddef.h>
#include <sysdep.h>

/* Set the system-wide timezone.
   This call is restricted to the super-user.
   This operation is considered obsolete, kernel support may not be
   available on all architectures.  */
int
__settimezone (const struct timezone *tz)
{
#ifdef __NR_settimeofday
  return INLINE_SYSCALL_CALL (settimeofday, NULL, tz);
#else
  __set_errno (ENOSYS);
  return -1;
#endif
}

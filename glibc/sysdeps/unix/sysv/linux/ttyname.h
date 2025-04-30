/* Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <unistd.h>
#include <stdbool.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/stat.h>

/* Return true if this is a UNIX98 pty device, as defined in
   linux/Documentation/devices.txt (on linux < 4.10) or
   linux/Documentation/admin-guide/devices.txt (on linux >= 4.10).  */
static inline bool
is_pty (struct __stat64_t64 *sb)
{
  int m = __gnu_dev_major (sb->st_rdev);
  return (136 <= m && m <= 143);
}

static inline bool
is_mytty (const struct __stat64_t64 *mytty, const struct __stat64_t64 *maybe)
{
  return (maybe->st_ino == mytty->st_ino
	  && maybe->st_dev == mytty->st_dev
	  && S_ISCHR (maybe->st_mode)
	  && maybe->st_rdev == mytty->st_rdev
	  );
}

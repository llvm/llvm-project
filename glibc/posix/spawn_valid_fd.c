/* File descriptor validity check for posix_spawn file actions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include "spawn_int.h"

#include <unistd.h>

bool
__spawn_valid_fd (int fd)
{
  long maxfd = __sysconf (_SC_OPEN_MAX);
  return __glibc_likely (fd >= 0)
    && (__glibc_unlikely (maxfd < 0) /* No limit set.  */
	|| __glibc_likely (fd < maxfd));
}

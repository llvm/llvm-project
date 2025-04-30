/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <spawn.h>
#include <string.h>

#define ALL_FLAGS (POSIX_SPAWN_RESETIDS					      \
		   | POSIX_SPAWN_SETPGROUP				      \
		   | POSIX_SPAWN_SETSIGDEF				      \
		   | POSIX_SPAWN_SETSIGMASK				      \
		   | POSIX_SPAWN_SETSCHEDPARAM				      \
		   | POSIX_SPAWN_SETSCHEDULER				      \
		   | POSIX_SPAWN_SETSID					      \
		   | POSIX_SPAWN_USEVFORK)

/* Store flags in the attribute structure.  */
int
__posix_spawnattr_setflags (posix_spawnattr_t *attr, short int flags)
{
  /* Check no invalid bits are set.  */
  if (flags & ~ALL_FLAGS)
    return EINVAL;

  /* Store the flag word.  */
  attr->__flags = flags;

  return 0;
}
weak_alias (__posix_spawnattr_setflags, posix_spawnattr_setflags)

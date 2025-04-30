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

#include <spawn.h>
#include <string.h>

/* Set signal mask for signals with default handling in ATTR to SIGDEFAULT.  */
int
__posix_spawnattr_setsigdefault (posix_spawnattr_t *attr,
				 const sigset_t *sigdefault)
{
  /* Copy the sigset_t data to the user buffer.  */
  memcpy (&attr->__sd, sigdefault, sizeof (sigset_t));

  return 0;
}
weak_alias (__posix_spawnattr_setsigdefault, posix_spawnattr_setsigdefault)

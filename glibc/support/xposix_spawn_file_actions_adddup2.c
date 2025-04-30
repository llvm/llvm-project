/* xposix_spawn_file_actions_adddup2 implementation.
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
   <https://www.gnu.org/licenses/>.  */

#include <support/xspawn.h>
#include <support/check.h>

int
xposix_spawn_file_actions_adddup2 (posix_spawn_file_actions_t *fa, int fd,
				   int newfd)
{
  int status = posix_spawn_file_actions_adddup2 (fa, fd, newfd);
  if (status == -1)
    FAIL_EXIT1 ("posix_spawn_file_actions_adddup2 failed: %m\n");
  return status;
}

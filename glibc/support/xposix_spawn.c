/* xposix_spawn implementation.
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

pid_t
xposix_spawn (const char *file, const posix_spawn_file_actions_t *fa,
	      const posix_spawnattr_t *attr, char *const args[],
	      char *const envp[])
{
  pid_t pid;
  int status = posix_spawn (&pid, file, fa, attr, args, envp);
  if (status != 0)
    FAIL_EXIT1 ("posix_spawn to %s file failed: %m", file);
  return pid;
}

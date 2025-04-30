/* Guts of POSIX spawn interface.  Stub version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include "spawn_int.h"


/* The Unix standard contains a long explanation of the way to signal
   an error after the fork() was successful.  Since no new wait status
   was wanted there is no way to signal an error using one of the
   available methods.  The committee chose to signal an error by a
   normal program exit with the exit code 127.  */
#define SPAWN_ERROR	127


/* Spawn a new process executing PATH with the attributes describes in *ATTRP.
   Before running the process perform the actions described in FILE-ACTIONS. */
int
__spawni (pid_t *pid, const char *file,
	  const posix_spawn_file_actions_t *file_actions,
	  const posix_spawnattr_t *attrp, char *const argv[],
	  char *const envp[], int xflags)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (__spawni)

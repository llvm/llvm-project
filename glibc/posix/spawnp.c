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
#include "spawn_int.h"
#include <shlib-compat.h>

/* Spawn a new process executing FILE with the attributes describes in *ATTRP.
   Before running the process perform the actions described in FILE-ACTIONS. */
int
__posix_spawnp (pid_t *pid, const char *file,
		const posix_spawn_file_actions_t *file_actions,
		const posix_spawnattr_t *attrp, char *const argv[],
		char *const envp[])
{
  return __spawni (pid, file, file_actions, attrp, argv, envp,
		   SPAWN_XFLAGS_USE_PATH);
}
versioned_symbol (libc, __posix_spawnp, posix_spawnp, GLIBC_2_15);


#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_15)
int
attribute_compat_text_section
__posix_spawnp_compat (pid_t *pid, const char *file,
		       const posix_spawn_file_actions_t *file_actions,
		       const posix_spawnattr_t *attrp, char *const argv[],
		       char *const envp[])
{
  return __spawni (pid, file, file_actions, attrp, argv, envp,
		   SPAWN_XFLAGS_USE_PATH | SPAWN_XFLAGS_TRY_SHELL);
}
compat_symbol (libc, __posix_spawnp_compat, posix_spawnp, GLIBC_2_2);
#endif

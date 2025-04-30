/* posix_spawn with support checks.
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

#ifndef SUPPORT_XSPAWN_H
#define SUPPORT_XSPAWN_H

#include <spawn.h>

__BEGIN_DECLS

int xposix_spawn_file_actions_addclose (posix_spawn_file_actions_t *, int);
int xposix_spawn_file_actions_adddup2 (posix_spawn_file_actions_t *, int, int);

pid_t xposix_spawn (const char *, const posix_spawn_file_actions_t *,
		    const posix_spawnattr_t *, char *const [], char *const []);

__END_DECLS

#endif

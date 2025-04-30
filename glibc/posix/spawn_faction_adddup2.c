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
#include <unistd.h>

#include "spawn_int.h"

/* Add an action to FILE-ACTIONS which tells the implementation to call
   `dup2' for the given file descriptors during the `spawn' call.  */
int
__posix_spawn_file_actions_adddup2 (posix_spawn_file_actions_t *file_actions,
				    int fd, int newfd)
{
  struct __spawn_action *rec;

  if (!__spawn_valid_fd (fd) || !__spawn_valid_fd (newfd))
    return EBADF;

  /* Allocate more memory if needed.  */
  if (file_actions->__used == file_actions->__allocated
      && __posix_spawn_file_actions_realloc (file_actions) != 0)
    /* This can only mean we ran out of memory.  */
    return ENOMEM;

  /* Add the new value.  */
  rec = &file_actions->__actions[file_actions->__used];
  rec->tag = spawn_do_dup2;
  rec->action.dup2_action.fd = fd;
  rec->action.dup2_action.newfd = newfd;

  /* Account for the new entry.  */
  ++file_actions->__used;

  return 0;
}
weak_alias (__posix_spawn_file_actions_adddup2,
	    posix_spawn_file_actions_adddup2)

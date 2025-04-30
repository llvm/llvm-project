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
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "spawn_int.h"

/* Add an action to FILE-ACTIONS which tells the implementation to call
   `open' for the given file during the `spawn' call.  */
int
__posix_spawn_file_actions_addopen (posix_spawn_file_actions_t *file_actions,
				    int fd, const char *path, int oflag,
				    mode_t mode)
{
  struct __spawn_action *rec;

  if (!__spawn_valid_fd (fd))
    return EBADF;

  char *path_copy = __strdup (path);
  if (path_copy == NULL)
    return ENOMEM;

  /* Allocate more memory if needed.  */
  if (file_actions->__used == file_actions->__allocated
      && __posix_spawn_file_actions_realloc (file_actions) != 0)
    {
      /* This can only mean we ran out of memory.  */
      free (path_copy);
      return ENOMEM;
    }

  /* Add the new value.  */
  rec = &file_actions->__actions[file_actions->__used];
  rec->tag = spawn_do_open;
  rec->action.open_action.fd = fd;
  rec->action.open_action.path = path_copy;
  rec->action.open_action.oflag = oflag;
  rec->action.open_action.mode = mode;

  /* Account for the new entry.  */
  ++file_actions->__used;

  return 0;
}
weak_alias (__posix_spawn_file_actions_addopen,
	    posix_spawn_file_actions_addopen)

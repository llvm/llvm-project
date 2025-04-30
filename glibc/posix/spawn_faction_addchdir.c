/* Add a directory change to a file action list for posix_spawn.
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

#include <errno.h>
#include <spawn.h>
#include <string.h>
#include <stdlib.h>
#include "spawn_int.h"

int
posix_spawn_file_actions_addchdir_np (posix_spawn_file_actions_t *file_actions,
                                      const char *path)
{
  struct __spawn_action *rec;

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
  rec->tag = spawn_do_chdir;
  rec->action.chdir_action.path = path_copy;

  /* Account for the new entry.  */
  ++file_actions->__used;

  return 0;
}

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

#include "spawn_int.h"

int
posix_spawn_file_actions_addfchdir_np (posix_spawn_file_actions_t *actions,
                                       int fd)
{
  struct __spawn_action *rec;

  /* Allocate more memory if needed.  */
  if (actions->__used == actions->__allocated
      && __posix_spawn_file_actions_realloc (actions) != 0)
    /* This can only mean we ran out of memory.  */
    return ENOMEM;

  /* Add the new value.  */
  rec = &actions->__actions[actions->__used];
  rec->tag = spawn_do_fchdir;
  rec->action.fchdir_action.fd = fd;

  /* Account for the new entry.  */
  ++actions->__used;

  return 0;
}

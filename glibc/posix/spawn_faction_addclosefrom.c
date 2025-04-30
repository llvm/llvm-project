/* Add a closefrom to a file action list for posix_spawn.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
#include <spawn_int.h>

int
__posix_spawn_file_actions_addclosefrom (posix_spawn_file_actions_t
					 *file_actions, int from)
{
#if __SPAWN_SUPPORT_CLOSEFROM
  struct __spawn_action *rec;

  if (!__spawn_valid_fd (from))
    return EBADF;

  /* Allocate more memory if needed.  */
  if (file_actions->__used == file_actions->__allocated
      && __posix_spawn_file_actions_realloc (file_actions) != 0)
    /* This can only mean we ran out of memory.  */
    return ENOMEM;

  /* Add the new value.  */
  rec = &file_actions->__actions[file_actions->__used];
  rec->tag = spawn_do_closefrom;
  rec->action.closefrom_action.from = from;

  /* Account for the new entry.  */
  ++file_actions->__used;

  return 0;
#else
  return EINVAL;
#endif
}
weak_alias (__posix_spawn_file_actions_addclosefrom,
	    posix_spawn_file_actions_addclosefrom_np)
#if !__SPAWN_SUPPORT_CLOSEFROM
stub_warning (posix_spawn_file_actions_addclosefrom_np)
#endif

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
#include <stdlib.h>

#include "spawn_int.h"

/* Deallocate the file actions.  */
int
__posix_spawn_file_actions_destroy (posix_spawn_file_actions_t *file_actions)
{
  /* Free the paths in the open actions.  */
  for (int i = 0; i < file_actions->__used; ++i)
    {
      struct __spawn_action *sa = &file_actions->__actions[i];
      switch (sa->tag)
	{
	case spawn_do_open:
	  free (sa->action.open_action.path);
	  break;
	case spawn_do_chdir:
	  free (sa->action.chdir_action.path);
	  break;
	case spawn_do_close:
	case spawn_do_dup2:
	case spawn_do_fchdir:
	case spawn_do_closefrom:
	  /* No cleanup required.  */
	  break;
	}
    }

  /* Free the array of actions.  */
  free (file_actions->__actions);
  return 0;
}
weak_alias (__posix_spawn_file_actions_destroy,
	    posix_spawn_file_actions_destroy)

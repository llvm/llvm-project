/* Internal definitions for posix_spawn functionality.
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

#ifndef _SPAWN_INT_H
#define _SPAWN_INT_H

#include <spawn.h>
#include <spawn_int_def.h>
#include <stdbool.h>

/* Data structure to contain the action information.  */
struct __spawn_action
{
  enum
  {
    spawn_do_close,
    spawn_do_dup2,
    spawn_do_open,
    spawn_do_chdir,
    spawn_do_fchdir,
    spawn_do_closefrom,
  } tag;

  union
  {
    struct
    {
      int fd;
    } close_action;
    struct
    {
      int fd;
      int newfd;
    } dup2_action;
    struct
    {
      int fd;
      char *path;
      int oflag;
      mode_t mode;
    } open_action;
    struct
    {
      char *path;
    } chdir_action;
    struct
    {
      int fd;
    } fchdir_action;
    struct
    {
      int from;
    } closefrom_action;
  } action;
};

#define SPAWN_XFLAGS_USE_PATH	0x1
#define SPAWN_XFLAGS_TRY_SHELL	0x2

extern int __posix_spawn_file_actions_realloc (posix_spawn_file_actions_t *
					       file_actions)
     attribute_hidden;

extern int __spawni (pid_t *pid, const char *path,
		     const posix_spawn_file_actions_t *file_actions,
		     const posix_spawnattr_t *attrp, char *const argv[],
		     char *const envp[], int xflags) attribute_hidden;

/* Return true if FD falls into the range valid for file descriptors.
   The check in this form is mandated by POSIX.  */
bool __spawn_valid_fd (int fd) attribute_hidden;

#endif /* _SPAWN_INT_H */

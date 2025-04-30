/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <getcwd.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

/* Replace the current process, executing FILE_NAME with arguments ARGV and
   environment ENVP.  ARGV and ENVP are terminated by NULL pointers.  */
int
__execveat (int dirfd, const char *file_name, char *const argv[],
            char *const envp[], int flags)
{
  error_t err;
  char *concat_name = NULL;
  const char *abs_path;

  file_t file = __file_name_lookup_at (dirfd, flags, file_name, O_EXEC, 0);
  if (file == MACH_PORT_NULL)
    return -1;

  if (file_name[0] == '/')
    {
      /* Already an absolute path */
      abs_path = file_name;
    }
  else
    {
      /* Relative path */
      char *cwd;
      if (dirfd == AT_FDCWD)
	{
	  cwd = __getcwd (NULL, 0);
	  if (cwd == NULL)
	    {
	      __mach_port_deallocate (__mach_task_self (), file);
	      return -1;
	    }
	}
      else
	{
	  err = HURD_DPORT_USE (dirfd,
	    (cwd = __hurd_canonicalize_directory_name_internal (port, NULL, 0),
	     cwd == NULL ? errno : 0));
	  if (err)
	    {
	      __mach_port_deallocate (__mach_task_self (), file);
	      return __hurd_fail (err);
	    }
	}

      int res = __asprintf (&concat_name, "%s/%s", cwd, file_name);
      free (cwd);
      if (res == -1)
	{
	  __mach_port_deallocate (__mach_task_self (), file);
	  return -1;
	}

      abs_path = concat_name;
    }

  /* Hopefully this will not return.  */
  err = _hurd_exec_paths (__mach_task_self (), file,
			  file_name, abs_path, argv, envp);

  /* Oh well.  Might as well be tidy.  */
  __mach_port_deallocate (__mach_task_self (), file);
  free (concat_name);

  return __hurd_fail (err);
}

weak_alias (__execveat, execveat)

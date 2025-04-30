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
#include <hurd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

/* Replace the current process, executing FILE_NAME with arguments ARGV and
   environment ENVP.  ARGV and ENVP are terminated by NULL pointers.  */
int
__execve (const char *file_name, char *const argv[], char *const envp[])
{
  return __execveat (AT_FDCWD, file_name, argv, envp, 0);
}

weak_alias (__execve, execve)

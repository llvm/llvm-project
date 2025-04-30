/* ANSI C `remove' function to delete a file or directory.  POSIX.1 version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <unistd.h>


#ifndef IS_NO_DIRECTORY_ERROR
# define IS_NO_DIRECTORY_ERROR errno != EPERM
#endif


int
remove (const char *file)
{
  /* First try to unlink since this is more frequently the necessary action. */
  if (__unlink (file) != 0
      /* If it is indeed a directory...  */
      && (IS_NO_DIRECTORY_ERROR
	  /* ...try to remove it.  */
	  || __rmdir (file) != 0))
    /* Cannot remove the object for whatever reason.  */
    return -1;

  return 0;
}
libc_hidden_def (remove)

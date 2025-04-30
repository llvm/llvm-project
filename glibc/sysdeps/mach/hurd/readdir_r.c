/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <hurd.h>
#include <hurd/fd.h>
#include "dirstream.h"


/* Read a directory entry from DIRP.  */
int
__readdir_r (DIR *dirp, struct dirent *entry, struct dirent **result)
{
  if (sizeof (struct dirent64) == sizeof (struct dirent))
    /* We should in fact just be an alias to readdir64_r on this machine.  */
    return __readdir64_r (dirp,
			  (struct dirent64 *) entry,
			  (struct dirent64 **) result);

  struct dirent64 *result64;
  union
  {
    struct dirent64 d;
    char b[offsetof (struct dirent64, d_name) + UCHAR_MAX + 1];
  } u;
  int err;

  err = __readdir64_r (dirp, &u.d, &result64);
  if (result64)
    {
      entry->d_fileno = result64->d_fileno;
      entry->d_reclen = result64->d_reclen;
      entry->d_type = result64->d_type;
      entry->d_namlen = result64->d_namlen;
      memcpy (entry->d_name, result64->d_name, result64->d_namlen + 1);
      *result = entry;
    }
  else
    *result = NULL;

  return err;
}

weak_alias (__readdir_r, readdir_r)

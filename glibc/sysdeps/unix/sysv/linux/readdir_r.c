/* Read a directory in reentrant mode.  Linux no-LFS version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <dirent.h>

#if !_DIRENT_MATCHES_DIRENT64
/* Read a directory entry from DIRP.  */
int
__readdir_r (DIR *dirp, struct dirent *entry, struct dirent **result)
{
  struct dirent *dp;
  size_t reclen;

  __libc_lock_lock (dirp->lock);

  while (1)
    {
      dp = __readdir_unlocked (dirp);
      if (dp == NULL)
	break;

      reclen = dp->d_reclen;
      if (reclen <= offsetof (struct dirent, d_name) + NAME_MAX + 1)
	break;

      /* The record is very long.  It could still fit into the caller-supplied
	 buffer if we can skip padding at the end.  */
      size_t namelen = _D_EXACT_NAMLEN (dp);
      if (namelen <= NAME_MAX)
	{
	  reclen = offsetof (struct dirent, d_name) + namelen + 1;
	  break;
	}

      /* The name is too long.  Ignore this file.  */
      dirp->errcode = ENAMETOOLONG;
    }

  if (dp != NULL)
    {
      *result = memcpy (entry, dp, reclen);
      entry->d_reclen = reclen;
    }
  else
    *result = NULL;

  __libc_lock_unlock (dirp->lock);

  return dp != NULL ? 0 : dirp->errcode;
}

weak_alias (__readdir_r, readdir_r)
#endif /* _DIRENT_MATCHES_DIRENT64  */

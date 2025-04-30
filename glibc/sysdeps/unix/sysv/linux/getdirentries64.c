/* Get directory entries in a filesystem-independent format.  LFS version.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#define getdirentries __no_getdirentries_decl
#include <dirent.h>
#undef getdirentries
#include <unistd.h>

ssize_t
getdirentries64 (int fd, char *buf, size_t nbytes, off64_t *basep)
{
  off64_t base = __lseek64 (fd, (off_t) 0, SEEK_CUR);

  ssize_t result = __getdents64 (fd, buf, nbytes);

  if (result != -1)
    *basep = base;

  return result;
}

#if _DIRENT_MATCHES_DIRENT64
weak_alias (getdirentries64, getdirentries)
#endif

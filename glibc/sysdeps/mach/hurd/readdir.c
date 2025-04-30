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
#include <stddef.h>
#include <dirent.h>
#include <unistd.h>
#include <endian.h>
#include <assert.h>

/* Read a directory entry from DIRP.  */
struct dirent *
__readdir (DIR *dirp)
{
  struct dirent64 *entry64 = __readdir64 (dirp);

  if (sizeof (struct dirent64) == sizeof (struct dirent))
    /* We should in fact just be an alias to readdir64 on this machine.  */
    return (struct dirent *) entry64;

  /* These are all compile-time constants.  We know that d_ino is the first
     member and that the layout of the following members matches exactly in
     both structures.  */
  assert (offsetof (struct dirent, d_ino) == 0);
  assert (offsetof (struct dirent64, d_ino) == 0);
# define MATCH(memb)							      \
  assert (offsetof (struct dirent64, memb) - sizeof (entry64->d_ino)	      \
	  == offsetof (struct dirent, memb) - sizeof (ino_t))
  MATCH (d_reclen);
  MATCH (d_type);
  MATCH (d_namlen);
# undef MATCH

  if (entry64 == NULL)
    return NULL;

  struct dirent *const entry = ((void *) (&entry64->d_ino + 1)
				- sizeof entry->d_ino);
  const ino_t d_ino = entry64->d_ino;
  if (d_ino != entry64->d_ino)
    {
      __set_errno (EOVERFLOW);
      return NULL;
    }
# if BYTE_ORDER != BIG_ENDIAN	/* We just skipped over the zero high word.  */
  entry->d_ino = d_ino;	/* ... or the nonzero low word, swap it.  */
# endif
  entry->d_reclen -= sizeof entry64->d_ino - sizeof entry->d_ino;
  return entry;
}

weak_alias (__readdir, readdir)

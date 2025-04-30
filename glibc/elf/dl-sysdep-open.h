/* System-specific call to open a shared object by name.  Stub version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _DL_SYSDEP_OPEN_H
#define _DL_SYSDEP_OPEN_H	1

#include <assert.h>
#include <stddef.h>

/* NAME is a name without slashes, as it appears in a DT_NEEDED entry
   or a dlopen call's argument or suchlike.  NAMELEN is (strlen (NAME) + 1).

   Find NAME in an OS-dependent fashion, and return its "real" name.
   Optionally fill in *FD with a file descriptor open on that file (or
   else leave its initial value of -1).  The return value is a new
   malloc'd string, which will be free'd by the caller.  If NAME is
   resolved to an actual file that can be opened, then the return
   value should name that file (and if *FD was not set, then a normal
   __open call on that string will be made).  If *FD was set by some
   other means than a normal open and there is no "real" name to use,
   then __strdup (NAME) is fine (modulo error checking).  */

static inline char *
_dl_sysdep_open_object (const char *name, size_t namelen, int *fd)
{
  assert (*fd == -1);
  return NULL;
}

#endif  /* dl-sysdep-open.h */

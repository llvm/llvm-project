/* Header for directory for shm/sem files.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _SHM_DIRECTORY_H

#include <limits.h>
#include <paths.h>
#include <stdbool.h>

/* The directory that contains shared POSIX objects.  */
#define SHMDIR _PATH_DEV "shm/"

struct shmdir_name
{
  /* The combined prefix/name.  The sizeof includes the terminating
     NUL byte.  4 bytes are needed for the optional "sem." prefix.  */
  char name[sizeof (SHMDIR) + 4 + NAME_MAX];
};

/* Sets RESULT->name to the constructed name and returns 0 on success,
   or -1 on failure.  Includes the "sem." prefix in the name if
   SEM_PREFIX is true.  */
int __shm_get_name (struct shmdir_name *result, const char *name,
		    bool sem_prefix);
libc_hidden_proto (__shm_get_name)

#endif  /* shm-directory.h */

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
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>
#include "dirstream.h"

/* Close the directory stream DIRP.
   Return 0 if successful, -1 if not.  */
int
__closedir (DIR *dirp)
{
  error_t err;

  if (dirp == NULL)
    {
      errno = EINVAL;
      return -1;
    }

  __libc_lock_lock (dirp->__lock);
  err = __vm_deallocate (__mach_task_self (),
			 (vm_address_t) dirp->__data, dirp->__allocation);
  dirp->__data = NULL;
  err = _hurd_fd_close (dirp->__fd);

  if (err)
    {
      /* Unlock the DIR.  A failing closedir can be repeated (and may fail
	 again, but shouldn't deadlock).  */
      __libc_lock_unlock (dirp->__lock);
      return __hurd_fail (err);
    }

  /* Clean up the lock and free the structure.  */
  __libc_lock_fini (dirp->__lock);
  free (dirp);

  return 0;
}
weak_alias (__closedir, closedir)

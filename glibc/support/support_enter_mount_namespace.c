/* Enter a mount namespace.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <support/namespace.h>

#include <sched.h>
#include <stdio.h>
#ifdef CLONE_NEWNS
# include <sys/mount.h>
#endif /* CLONE_NEWNS */

bool
support_enter_mount_namespace (void)
{
#ifdef CLONE_NEWNS
  if (unshare (CLONE_NEWNS) == 0)
    {
      /* On some systems, / is marked as MS_SHARED, which means that
         mounts within the namespace leak to the rest of the system,
         which is not what we want.  */
      if (mount ("none", "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
        {
          printf ("warning: making the mount namespace private failed: %m\n");
          return false;
        }
      return true;
    }
  else
    printf ("warning: unshare (CLONE_NEWNS) failed: %m\n");
#endif /* CLONE_NEWNS */
  return false;
}

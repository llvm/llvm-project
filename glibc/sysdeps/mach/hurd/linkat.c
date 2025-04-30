/* Make a link between file names relative to open directories.  Hurd version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>


/* Make a link to FROM relative to FROMFD called TO relative to TOFD.  */
int
linkat (int fromfd, const char *from, int tofd, const char *to, int flags)
{
  error_t err;
  file_t oldfile, linknode, todir;
  char *toname;

  /* POSIX says linkat doesn't follow symlinks by default, so pass
     O_NOLINK.  That can be overridden by AT_SYMLINK_FOLLOW in FLAGS.  */
  oldfile = __file_name_lookup_at (fromfd, flags, from, O_NOLINK, 0);
  if (oldfile == MACH_PORT_NULL)
    return -1;

  /* The file_getlinknode RPC returns the port that should be passed to
     the receiving filesystem (the one containing TODIR) in dir_link.  */

  err = __file_getlinknode (oldfile, &linknode);
  __mach_port_deallocate (__mach_task_self (), oldfile);
  if (err)
    return __hurd_fail (err);

  todir = __file_name_split_at (tofd, to, &toname);
  if (todir != MACH_PORT_NULL)
    {
      err = __dir_link (todir, linknode, toname, 1);
      __mach_port_deallocate (__mach_task_self (), todir);
    }
  __mach_port_deallocate (__mach_task_self (), linknode);
  if (todir == MACH_PORT_NULL)
    return -1;

  if (err)
    return __hurd_fail (err);
  return 0;
}

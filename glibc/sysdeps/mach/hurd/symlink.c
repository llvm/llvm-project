/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/paths.h>
#include <fcntl.h>

/* Make a link to FROM called TO.  */
int
__symlink (const char *from, const char *to)
{
  error_t err;
  file_t dir, node;
  char *name;
  const size_t len = strlen (from) + 1;
  char buf[sizeof (_HURD_SYMLINK) + len];

  /* A symlink is a file whose translator is "/hurd/symlink\0target\0".  */

  memcpy (buf, _HURD_SYMLINK, sizeof (_HURD_SYMLINK));
  memcpy (&buf[sizeof (_HURD_SYMLINK)], from, len);

  dir = __file_name_split (to, &name);
  if (dir == MACH_PORT_NULL)
    return -1;

  /* Create a new, unlinked node in the target directory.  */
  err = __dir_mkfile (dir, O_WRITE, 0777 & ~_hurd_umask, &node);

  if (! err)
    {
      /* Set the node's translator to make it a symlink.  */
      err = __file_set_translator (node,
                                   FS_TRANS_EXCL|FS_TRANS_SET,
                                   FS_TRANS_EXCL|FS_TRANS_SET, 0,
                                   buf, sizeof (_HURD_SYMLINK) + len,
                                   MACH_PORT_NULL, MACH_MSG_TYPE_COPY_SEND);

      if (! err)
        /* Link the node, now a valid symlink, into the target directory.  */
        err = __dir_link (dir, node, name, 1);

      __mach_port_deallocate (__mach_task_self (), node);
    }

  __mach_port_deallocate (__mach_task_self (), dir);

  if (err)
    return __hurd_fail (err);
  return 0;
}

weak_alias (__symlink, symlink)

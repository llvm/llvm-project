/* Create a device file relative to an open directory.  Hurd version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/stat.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/paths.h>
#include <fcntl.h>
#include <_itoa.h>
#include <string.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <shlib-compat.h>

/* Create a device file named PATH relative to FD, with permission and
   special bits MODE and device number DEV (which can be constructed
   from major and minor device numbers with the `makedev' macro
   above).  */
int
__mknodat (int fd, const char *path, mode_t mode, dev_t dev)
{
  error_t errnode, err;
  file_t dir, node;
  char *name;
  char buf[100], *bp;
  const char *translator;
  size_t len;

  if (S_ISCHR (mode))
    {
      translator = _HURD_CHRDEV;
      len = sizeof (_HURD_CHRDEV);
    }
  else if (S_ISBLK (mode))
    {
      translator = _HURD_BLKDEV;
      len = sizeof (_HURD_BLKDEV);
    }
  else if (S_ISFIFO (mode))
    {
      translator = _HURD_FIFO;
      len = sizeof (_HURD_FIFO);
    }
  else if (S_ISREG (mode))
    {
      translator = NULL;
      len = 0;
    }
  else
    {
      errno = EINVAL;
      return -1;
    }

  if (translator != NULL && ! S_ISFIFO (mode))
    {
      /* We set the translator to "ifmt\0major\0minor\0", where IFMT
	 depends on the S_IFMT bits of our MODE argument, and MAJOR and
	 MINOR are ASCII decimal (octal or hex would do as well)
	 representations of our arguments.  Thus the convention is that
	 CHRDEV and BLKDEV translators are invoked with two non-switch
	 arguments, giving the major and minor device numbers in %i format. */

      bp = buf + sizeof (buf);
      *--bp = '\0';
      bp = _itoa (__gnu_dev_minor (dev), bp, 10, 0);
      *--bp = '\0';
      bp = _itoa (__gnu_dev_major (dev), bp, 10, 0);
      memcpy (bp - len, translator, len);
      translator = bp - len;
      len = buf + sizeof (buf) - translator;
    }

  dir = __file_name_split_at (fd, path, &name);
  if (dir == MACH_PORT_NULL)
    return -1;

  /* Create a new, unlinked node in the target directory.  */
  errnode = err = __dir_mkfile (dir, O_WRITE, (mode & ~S_IFMT) & ~_hurd_umask, &node);

  if (! err && translator != NULL)
    /* Set the node's translator to make it a device.  */
    err = __file_set_translator (node,
				 FS_TRANS_EXCL | FS_TRANS_SET,
				 FS_TRANS_EXCL | FS_TRANS_SET, 0,
				 translator, len,
				 MACH_PORT_NULL, MACH_MSG_TYPE_COPY_SEND);

  if (! err)
    /* Link the node, now a valid device, into the target directory.  */
    err = __dir_link (dir, node, name, 1);

  __mach_port_deallocate (__mach_task_self (), dir);
  if (! errnode)
    __mach_port_deallocate (__mach_task_self (), node);

  if (err)
    return __hurd_fail (err);
  return 0;
}
libc_hidden_def (__mknodat)
weak_alias (__mknodat, mknodat)

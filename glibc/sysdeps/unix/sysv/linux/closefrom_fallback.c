/* Close a range of file descriptors.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <arch-fd_to_filename.h>
#include <dirent.h>
#include <not-cancel.h>
#include <stdbool.h>

/* Fallback code: iterates over /proc/self/fd, closing each file descriptor
   that fall on the criteria.  If DIRFD_FALLBACK is set, a failure on
   /proc/self/fd open will trigger a fallback that tries to close a file
   descriptor before proceed.  */
_Bool
__closefrom_fallback (int from, _Bool dirfd_fallback)
{
  bool ret = false;

  int dirfd = __open_nocancel (FD_TO_FILENAME_PREFIX, O_RDONLY | O_DIRECTORY,
                               0);
  if (dirfd == -1)
    {
      /* The closefrom should work even when process can't open new files.  */
      if (errno == ENOENT || !dirfd_fallback)
        goto err;

      for (int i = from; i < INT_MAX; i++)
        {
          int r = __close_nocancel (i);
          if (r == 0 || (r == -1 && errno != EBADF))
            break;
        }

      dirfd = __open_nocancel (FD_TO_FILENAME_PREFIX, O_RDONLY | O_DIRECTORY,
                               0);
      if (dirfd == -1)
        goto err;
    }

  char buffer[1024];
  while (true)
    {
      ssize_t ret = __getdents64 (dirfd, buffer, sizeof (buffer));
      if (ret == -1)
        goto err;
      else if (ret == 0)
        break;

      /* If any file descriptor is closed it resets the /proc/self position
         read again from the start (to obtain any possible kernel update).  */
      bool closed = false;
      char *begin = buffer, *end = buffer + ret;
      while (begin != end)
        {
          unsigned short int d_reclen;
          memcpy (&d_reclen, begin + offsetof (struct dirent64, d_reclen),
                  sizeof (d_reclen));
          const char *dname = begin + offsetof (struct dirent64, d_name);
          begin += d_reclen;

          if (dname[0] == '.')
            continue;

          int fd = 0;
          for (const char *s = dname; (unsigned int) (*s) - '0' < 10; s++)
            fd = 10 * fd + (*s - '0');

          if (fd == dirfd || fd < from)
            continue;

          /* We ignore close errors because EBADF, EINTR, and EIO means the
             descriptor has been released.  */
          __close_nocancel (fd);
          closed = true;
        }

      if (closed && __lseek (dirfd, 0, SEEK_SET) < 0)
        goto err;
    }

  ret = true;
err:
  __close_nocancel (dirfd);
  return ret;
}

/* Change the protections of file relative to open directory.  Linux version.
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
#include <fd_to_filename.h>
#include <not-cancel.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sysdep.h>
#include <unistd.h>

int
fchmodat (int fd, const char *file, mode_t mode, int flag)
{
  if (flag == 0)
    return INLINE_SYSCALL (fchmodat, 3, fd, file, mode);
  else if (flag != AT_SYMLINK_NOFOLLOW)
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
  else
    {
      /* The kernel system call does not have a mode argument.
	 However, we can create an O_PATH descriptor and change that
	 via /proc (which does not resolve symbolic links).  */

      int pathfd = __openat_nocancel (fd, file,
				      O_PATH | O_NOFOLLOW | O_CLOEXEC);
      if (pathfd < 0)
	/* This may report errors such as ENFILE and EMFILE.  The
	   caller can treat them as temporary if necessary.  */
	return pathfd;

      /* Use fstatat because fstat does not work on O_PATH descriptors
	 before Linux 3.6.  */
      struct stat64 st;
      if (__fstatat64 (pathfd, "", &st, AT_EMPTY_PATH) != 0)
	{
	  __close_nocancel (pathfd);
	  return -1;
	}

      /* Some Linux versions with some file systems can actually
	 change symbolic link permissions via /proc, but this is not
	 intentional, and it gives inconsistent results (e.g., error
	 return despite mode change).  The expected behavior is that
	 symbolic link modes cannot be changed at all, and this check
	 enforces that.  */
      if (S_ISLNK (st.st_mode))
	{
	  __close_nocancel (pathfd);
	  __set_errno (EOPNOTSUPP);
	  return -1;
	}

      /* For most file systems, fchmod does not operate on O_PATH
	 descriptors, so go through /proc.  */
      struct fd_to_filename filename;
      int ret = __chmod (__fd_to_filename (pathfd, &filename), mode);
      if (ret != 0)
	{
	  if (errno == ENOENT)
	    /* /proc has not been mounted.  Without /proc, there is no
	       way to upgrade the O_PATH descriptor to a full
	       descriptor.  It is also not possible to re-open the
	       file without O_PATH because the file name may refer to
	       another file, and opening that without O_PATH may have
	       side effects (such as blocking, device rewinding, or
	       releasing POSIX locks).  */
	    __set_errno (EOPNOTSUPP);
	}
      __close_nocancel (pathfd);
      return ret;
    }
}
libc_hidden_def (fchmodat)

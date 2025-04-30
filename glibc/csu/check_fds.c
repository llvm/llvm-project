/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <paths.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

/* Try to get a machine dependent instruction which will make the
   program crash.  This is used in case everything else fails.  */
#include <abort-instr.h>
#ifndef ABORT_INSTRUCTION
/* No such instruction is available.  */
# define ABORT_INSTRUCTION
#endif

#include <device-nrs.h>
#include <not-cancel.h>


/* Should other OSes (e.g., Hurd) have different versions which can
   be written in a better way?  */
static void
check_one_fd (int fd, int mode)
{
  if (__builtin_expect (__fcntl64_nocancel (fd, F_GETFD), 0) == -1
      && errno == EBADF)
    {
      const char *name;
      dev_t dev;

      /* For writable descriptors we use /dev/full.  */
      if ((mode & O_ACCMODE) == O_WRONLY)
	{
	  name = _PATH_DEV "full";
	  dev = __gnu_dev_makedev (DEV_FULL_MAJOR, DEV_FULL_MINOR);
	}
      else
	{
	  name = _PATH_DEVNULL;
	  dev = __gnu_dev_makedev (DEV_NULL_MAJOR, DEV_NULL_MINOR);
	}

      /* Something is wrong with this descriptor, it's probably not
	 opened.  Open /dev/null so that the SUID program we are
	 about to start does not accidentally use this descriptor.  */
      int nullfd = __open_nocancel (name, mode, 0);

      /* We are very paranoid here.  With all means we try to ensure
	 that we are actually opening the /dev/null device and nothing
	 else.

	 Note that the following code assumes that STDIN_FILENO,
	 STDOUT_FILENO, STDERR_FILENO are the three lowest file
	 decsriptor numbers, in this order.  */
      struct __stat64_t64 st;
      if (__glibc_unlikely (nullfd != fd)
	  || __glibc_likely (__fstat64_time64 (fd, &st) != 0)
	  || __glibc_unlikely (S_ISCHR (st.st_mode) == 0)
	  || st.st_rdev != dev)
	/* We cannot even give an error message here since it would
	   run into the same problems.  */
	while (1)
	  /* Try for ever and ever.  */
	  ABORT_INSTRUCTION;
    }
}


void
__libc_check_standard_fds (void)
{
  /* Check all three standard file descriptors.  The O_NOFOLLOW flag
     is really paranoid but some people actually are.  If /dev/null
     should happen to be a symlink to somewhere else and not the
     device commonly known as "/dev/null" we bail out.  */
  check_one_fd (STDIN_FILENO, O_WRONLY | O_NOFOLLOW);
  check_one_fd (STDOUT_FILENO, O_RDONLY | O_NOFOLLOW);
  check_one_fd (STDERR_FILENO, O_RDONLY | O_NOFOLLOW);
}

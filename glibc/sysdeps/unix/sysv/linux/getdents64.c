/* Get directory entries.  Linux LFS version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>

/* The kernel struct linux_dirent64 matches the 'struct dirent64' type.  */
ssize_t
__getdents64 (int fd, void *buf, size_t nbytes)
{
  /* The system call takes an unsigned int argument, and some length
     checks in the kernel use an int type.  */
  if (nbytes > INT_MAX)
    nbytes = INT_MAX;
  return INLINE_SYSCALL_CALL (getdents64, fd, buf, nbytes);
}
libc_hidden_def (__getdents64)
weak_alias (__getdents64, getdents64)

#if _DIRENT_MATCHES_DIRENT64
strong_alias (__getdents64, __getdents)
#else
# include <shlib-compat.h>

# if SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_2)
#  include <olddirent.h>
#  include <unistd.h>

static ssize_t
handle_overflow (int fd, __off64_t offset, ssize_t count)
{
  /* If this is the first entry in the buffer, we can report the
     error.  */
  if (offset == 0)
    {
      __set_errno (EOVERFLOW);
      return -1;
    }

  /* Otherwise, seek to the overflowing entry, so that the next call
     will report the error, and return the data read so far.  */
  if (__lseek64 (fd, offset, SEEK_SET) != 0)
    return -1;
  return count;
}

ssize_t
__old_getdents64 (int fd, char *buf, size_t nbytes)
{
  /* We do not move the individual directory entries.  This is only
     possible if the target type (struct __old_dirent64) is smaller
     than the source type.  */
  _Static_assert (offsetof (struct __old_dirent64, d_name)
		  <= offsetof (struct dirent64, d_name),
		  "__old_dirent64 is larger than dirent64");
  _Static_assert (__alignof__ (struct __old_dirent64)
		  <= __alignof__ (struct dirent64),
		  "alignment of __old_dirent64 is larger than dirent64");

  ssize_t retval = INLINE_SYSCALL_CALL (getdents64, fd, buf, nbytes);
  if (retval > 0)
    {
      /* This is the marker for the first entry.  Offset 0 is reserved
	 for the first entry (see rewinddir).  Here, we use it as a
	 marker for the first entry in the buffer.  We never actually
	 seek to offset 0 because handle_overflow reports the error
	 directly, so it does not matter that the offset is incorrect
	 if entries have been read from the descriptor before (so that
	 the descriptor is not actually at offset 0).  */
      __off64_t previous_offset = 0;

      char *p = buf;
      char *end = buf + retval;
      while (p < end)
	{
	  struct dirent64 *source = (struct dirent64 *) p;

	  /* Copy out the fixed-size data.  */
	  __ino_t ino = source->d_ino;
	  __off64_t offset = source->d_off;
	  unsigned int reclen = source->d_reclen;
	  unsigned char type = source->d_type;

	  /* Check for ino_t overflow.  */
	  if (__glibc_unlikely (ino != source->d_ino))
	    return handle_overflow (fd, previous_offset, p - buf);

	  /* Convert to the target layout.  Use a separate struct and
	     memcpy to side-step aliasing issues.  */
	  struct __old_dirent64 result;
	  result.d_ino = ino;
	  result.d_off = offset;
	  result.d_reclen = reclen;
	  result.d_type = type;

	  /* Write the fixed-sized part of the result to the
	     buffer.  */
	  size_t result_name_offset = offsetof (struct __old_dirent64, d_name);
	  memcpy (p, &result, result_name_offset);

	  /* Adjust the position of the name if necessary.  Copy
	     everything until the end of the record, including the
	     terminating NUL byte.  */
	  if (result_name_offset != offsetof (struct dirent64, d_name))
	    memmove (p + result_name_offset, source->d_name,
		     reclen - offsetof (struct dirent64, d_name));

	  p += reclen;
	  previous_offset = offset;
	}
     }
  return retval;
}
# endif /* SHLIB_COMPAT(libc, GLIBC_2_1, GLIBC_2_2)  */
#endif /* _DIRENT_MATCHES_DIRENT64  */

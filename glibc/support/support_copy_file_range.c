/* Simplified copy_file_range with cross-device copy.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <inttypes.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <support/support.h>

ssize_t
support_copy_file_range (int infd, __off64_t *pinoff,
			 int outfd, __off64_t *poutoff,
			 size_t length, unsigned int flags)
{
  if (flags != 0)
    {
      errno = EINVAL;
      return -1;
    }

  struct stat64 instat;
  struct stat64 outstat;
  if (fstat64 (infd, &instat) != 0 || fstat64 (outfd, &outstat) != 0)
    return -1;
  if (S_ISDIR (instat.st_mode) || S_ISDIR (outstat.st_mode))
    {
      errno = EISDIR;
      return -1;
    }
  if (!S_ISREG (instat.st_mode) || !S_ISREG (outstat.st_mode))
    {
      /* We need a regular input file so that the we can seek
	 backwards in case of a write failure.  */
      errno = EINVAL;
      return -1;
    }

  /* The output descriptor must not have O_APPEND set.  */
  if (fcntl (outfd, F_GETFL) & O_APPEND)
    {
      errno = EBADF;
      return -1;
    }

  /* Avoid an overflow in the result.  */
  if (length > SSIZE_MAX)
    length = SSIZE_MAX;

  /* Main copying loop.  The buffer size is arbitrary and is a
     trade-off between stack size consumption, cache usage, and
     amortization of system call overhead.  */
  size_t copied = 0;
  char buf[8192];
  while (length > 0)
    {
      size_t to_read = length;
      if (to_read > sizeof (buf))
	to_read = sizeof (buf);

      /* Fill the buffer.  */
      ssize_t read_count;
      if (pinoff == NULL)
	read_count = read (infd, buf, to_read);
      else
	read_count = pread64 (infd, buf, to_read, *pinoff);
      if (read_count == 0)
	/* End of file reached prematurely.  */
	return copied;
      if (read_count < 0)
	{
	  if (copied > 0)
	    /* Report the number of bytes copied so far.  */
	    return copied;
	  return -1;
	}
      if (pinoff != NULL)
	*pinoff += read_count;

      /* Write the buffer part which was read to the destination.  */
      char *end = buf + read_count;
      for (char *p = buf; p < end; )
	{
	  ssize_t write_count;
	  if (poutoff == NULL)
	    write_count = write (outfd, p, end - p);
	  else
	    write_count = pwrite64 (outfd, p, end - p, *poutoff);
	  if (write_count < 0)
	    {
	      /* Adjust the input read position to match what we have
		 written, so that the caller can pick up after the
		 error.  */
	      size_t written = p - buf;
	      /* NB: This needs to be signed so that we can form the
		 negative value below.  */
	      ssize_t overread = read_count - written;
	      if (pinoff == NULL)
		{
		  if (overread > 0)
		    {
		      /* We are on an error recovery path, so we
			 cannot deal with failure here.  */
		      int save_errno = errno;
		      (void) lseek64 (infd, -overread, SEEK_CUR);
		      errno = save_errno;
		    }
		}
	      else /* pinoff != NULL */
		*pinoff -= overread;

	      if (copied + written > 0)
		/* Report the number of bytes copied so far.  */
		return copied + written;
	      return -1;
	    }
	  p += write_count;
	  if (poutoff != NULL)
	    *poutoff += write_count;
	} /* Write loop.  */

      copied += read_count;
      length -= read_count;
    }
  return copied;
}

/* sendfile -- copy data directly from one file descriptor to another
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/sendfile.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <sys/mman.h>

/* Send COUNT bytes from file associated with IN_FD starting at OFFSET to
   descriptor OUT_FD.  */
ssize_t
__sendfile64 (int out_fd, int in_fd, off64_t *offset, size_t count)
{
  /* We just do a vanilla io_read followed by a vanilla io_write here.
     In theory the IN_FD filesystem can return us out-of-line data that
     we then send out-of-line to the OUT_FD filesystem and no copying
     takes place until those pages need to be flushed or packaged by
     that filesystem (e.g. packetized by a network socket).  However,
     we momentarily consume COUNT bytes of our local address space,
     which might blow if it's huge or address space is real tight.  */

  char *data = 0;
  size_t datalen = 0;
  error_t err = HURD_DPORT_USE (in_fd,
				__io_read (port, &data, &datalen,
					   offset ? *offset : (off_t) -1,
					   count));
  if (err == 0)
    {
      size_t nwrote;
      if (datalen == 0)
	return 0;
      err = HURD_DPORT_USE (out_fd, __io_write (port, data, datalen,
						(off_t) -1, &nwrote));
      __munmap (data, datalen);
      if (err == 0)
	{
	  if (offset)
	    *offset += datalen;
	  return nwrote;
	}
    }
  return __hurd_fail (err);
}
strong_alias (__sendfile64, sendfile64)

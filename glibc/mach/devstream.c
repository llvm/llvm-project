/* stdio on a Mach device port.
   Translates \n to \r\n on output, echos and translates \r to \n on input.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <mach.h>
#include <device/device.h>
#include <errno.h>
#include <string.h>
#include <libioP.h>


static ssize_t
devstream_write (void *cookie, const char *buffer, size_t n)
{
  const device_t dev = (device_t) cookie;

  int write_some (const char *p, size_t to_write)
    {
      kern_return_t err;
      int wrote;
      int thiswrite;

      while (to_write > 0)
	{
	  thiswrite = to_write;
	  if (thiswrite > IO_INBAND_MAX)
	    thiswrite = IO_INBAND_MAX;

	  if (err = device_write_inband (dev, 0, 0, p, thiswrite, &wrote))
	    {
	      errno = err;
	      return 0;
	    }
	  p += wrote;
	  to_write -= wrote;
	}
      return 1;
    }
  int write_crlf (void)
    {
      static const char crlf[] = "\r\n";
      return write_some (crlf, 2);
    }

  /* Search for newlines (LFs) in the buffer.  */

  const char *start = buffer, *p;
  while ((p = memchr (start, '\n', n)) != NULL)
    {
      /* Found one.  Write out through the preceding character,
	 and then write a CR/LF pair.  */

      if ((p > start && !write_some (start, p - start))
	  || !write_crlf ())
	return (start - buffer) ?: -1;

      n -= p + 1 - start;
      start = p + 1;
    }

  /* Write the remainder of the buffer.  */
  if (write_some (start, n))
    start += n;
  return (start - buffer) ?: -1;
}

static ssize_t
devstream_read (void *cookie, char *buffer, size_t to_read)
{
  const device_t dev = (device_t) cookie;

  kern_return_t err;
  mach_msg_type_number_t nread = to_read;

  err = device_read_inband (dev, 0, 0, to_read, buffer, &nread);
  if (err)
    {
      errno = err;
      return -1;
    }

  /* Translate CR to LF.  */
  {
    char *p;
    for (p = memchr (buffer, '\r', nread); p;
	 p = memchr (p + 1, '\r', (buffer + nread) - (p + 1)))
      *p = '\n';
  }

  /* Echo back what we read.  */
  (void) devstream_write (cookie, buffer, nread);

  return nread;
}

static int
dealloc_ref (void *cookie)
{
  if (__mach_port_deallocate (mach_task_self (), (mach_port_t) cookie))
    {
      errno = EINVAL;
      return -1;
    }
  return 0;
}

FILE *
mach_open_devstream (mach_port_t dev, const char *mode)
{
  FILE *stream;

  if (mach_port_mod_refs (mach_task_self (), dev, MACH_PORT_RIGHT_SEND, 1))
    {
      errno = EINVAL;
      return NULL;
    }

  stream = _IO_fopencookie ((void *) dev, mode,
			    (cookie_io_functions_t) { write: devstream_write,
						      read: devstream_read,
						      close: dealloc_ref });
  if (stream == NULL)
    {
      __mach_port_deallocate (mach_task_self (), dev);
      return NULL;
    }

  return stream;
}

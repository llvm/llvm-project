/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>

/* Read up to N chars into BUF from COOKIE.
   Return how many chars were read, 0 for EOF or -1 for error.  */
static ssize_t
readio (void *cookie, char *buf, size_t n)
{
  mach_msg_type_number_t nread;
  error_t err;
  char *bufp = buf;

  nread = n;
  if (err = __io_read ((io_t) cookie, &bufp, &nread, -1, n))
    return __hurd_fail (err);

  if (bufp != buf)
    {
      memcpy (buf, bufp, nread);
      __vm_deallocate (__mach_task_self (),
		       (vm_address_t) bufp, (vm_size_t) nread);
    }

  return nread;
}

/* Write up to N chars from BUF to COOKIE.
   Return how many chars were written or -1 for error.  */
static ssize_t
writeio (void *cookie, const char *buf, size_t n)
{
  mach_msg_type_number_t wrote;
  error_t err;

  if (err = __io_write ((io_t) cookie, buf, n, -1, &wrote))
    return __hurd_fail (err);

  return wrote;
}

/* Move COOKIE's file position *POS bytes, according to WHENCE.
   The current file position is stored in *POS.
   Returns zero if successful, nonzero if not.  */
static int
seekio (void *cookie,
	off64_t *pos,
	int whence)
{
  error_t err = __io_seek ((file_t) cookie, *pos, whence, pos);
  return err ? __hurd_fail (err) : 0;
}

/* Close the file associated with COOKIE.
   Return 0 for success or -1 for failure.  */
static int
closeio (void *cookie)
{
  error_t error = __mach_port_deallocate (__mach_task_self (),
					  (mach_port_t) cookie);
  if (error)
    return __hurd_fail (error);
  return 0;
}

#include "../libio/libioP.h"
#define fopencookie _IO_fopencookie
static const cookie_io_functions_t funcsio =
{ readio, writeio, seekio, closeio };


/* Open a stream on PORT.  MODE is as for fopen.  */

FILE *
__fopenport (mach_port_t port, const char *mode)
{
  int pflags;
  int needflags;
  error_t err;

  const char *m = mode;

  switch (*m++)
    {
    case 'r':
      needflags = O_READ;
      break;
    case 'w':
      needflags = O_WRITE;
      break;
    case 'a':
      needflags = O_WRITE|O_APPEND;
      break;
    default:
      return NULL;
  }
  if (m[0] == '+' || (m[0] == 'b' && m[1] == '+'))
    needflags |= O_RDWR;

  /* Verify the PORT is valid allows the access MODE specifies.  */

  if (err = __io_get_openmodes (port, &pflags))
    return __hurd_fail (err), NULL;

  /* Check the access mode.  */
  if ((pflags & needflags) != needflags)
    {
      errno = EBADF;
      return NULL;
    }

  return fopencookie ((void *) port, mode, funcsio);
}
weak_alias (__fopenport, fopenport)

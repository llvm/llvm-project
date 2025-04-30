/* fmemopen implementation.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* fmemopen() from 2.22 and forward works as defined by POSIX.  It also
   provides an older symbol, version 2.2.5, that behaves different regarding
   SEEK_END (libio/oldfmemopen.c).  */


#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include "libioP.h"


typedef struct fmemopen_cookie_struct fmemopen_cookie_t;
struct fmemopen_cookie_struct
{
  char        *buffer;   /* memory buffer.  */
  int         mybuffer;  /* allocated my buffer?  */
  int         append;    /* buffer open for append?  */
  size_t      size;      /* buffer length in bytes.  */
  off64_t     pos;       /* current position at the buffer.  */
  size_t      maxpos;    /* max position in buffer.  */
};


static ssize_t
fmemopen_read (void *cookie, char *b, size_t s)
{
  fmemopen_cookie_t *c = (fmemopen_cookie_t *) cookie;

  if (c->pos + s > c->maxpos)
    {
      s = c->maxpos - c->pos;
      if ((size_t) c->pos > c->maxpos)
	s = 0;
    }

  memcpy (b, &(c->buffer[c->pos]), s);

  c->pos += s;

  return s;
}


static ssize_t
fmemopen_write (void *cookie, const char *b, size_t s)
{
  fmemopen_cookie_t *c = (fmemopen_cookie_t *) cookie;;
  off64_t pos = c->append ? c->maxpos : c->pos;
  int addnullc = (s == 0 || b[s - 1] != '\0');

  if (pos + s > c->size)
    {
      if ((size_t) (c->pos + addnullc) >= c->size)
	{
	  __set_errno (ENOSPC);
	  return 0;
	}
      s = c->size - pos;
    }

  memcpy (&(c->buffer[pos]), b, s);

  c->pos = pos + s;
  if ((size_t) c->pos > c->maxpos)
    {
      c->maxpos = c->pos;
      if (c->maxpos < c->size && addnullc)
	c->buffer[c->maxpos] = '\0';
      /* A null byte is written in a stream open for update iff it fits.  */
      else if (c->append == 0 && addnullc != 0)
	c->buffer[c->size-1] = '\0';
    }

  return s;
}


static int
fmemopen_seek (void *cookie, off64_t *p, int w)
{
  off64_t np;
  fmemopen_cookie_t *c = (fmemopen_cookie_t *) cookie;

  switch (w)
    {
    case SEEK_SET:
      np = *p;
      break;

    case SEEK_CUR:
      np = c->pos + *p;
      break;

    case SEEK_END:
      np = c->maxpos + *p;
      break;

    default:
      return -1;
    }

  if (np < 0 || (size_t) np > c->size)
    {
      __set_errno (EINVAL);
      return -1;
    }

  *p = c->pos = np;

  return 0;
}


static int
fmemopen_close (void *cookie)
{
  fmemopen_cookie_t *c = (fmemopen_cookie_t *) cookie;

  if (c->mybuffer)
    free (c->buffer);
  free (c);

  return 0;
}


FILE *
__fmemopen (void *buf, size_t len, const char *mode)
{
  cookie_io_functions_t iof;
  fmemopen_cookie_t *c;
  FILE *result;

  c = (fmemopen_cookie_t *) calloc (sizeof (fmemopen_cookie_t), 1);
  if (c == NULL)
    return NULL;

  c->mybuffer = (buf == NULL);

  if (c->mybuffer)
    {
      c->buffer = (char *) malloc (len);
      if (c->buffer == NULL)
	{
	  free (c);
	  return NULL;
	}
      c->buffer[0] = '\0';
    }
  else
    {
      if (__glibc_unlikely ((uintptr_t) len > -(uintptr_t) buf))
	{
	  free (c);
	  __set_errno (EINVAL);
	  return NULL;
	}

      c->buffer = buf;

      /* POSIX states that w+ mode should truncate the buffer.  */
      if (mode[0] == 'w' && mode[1] == '+')
	c->buffer[0] = '\0';

      if (mode[0] == 'a')
        c->maxpos = strnlen (c->buffer, len);
    }


  /* Mode   |  starting position (cookie::pos) |          size (cookie::size)
     ------ |----------------------------------|-----------------------------
     read   |          beginning of the buffer |                size argument
     write  |          beginning of the buffer |                         zero
     append |    first null or size buffer + 1 |  first null or size argument
   */

  c->size = len;

  if (mode[0] == 'r')
    c->maxpos = len;

  c->append = mode[0] == 'a';
  if (c->append)
    c->pos = c->maxpos;
  else
    c->pos = 0;

  iof.read = fmemopen_read;
  iof.write = fmemopen_write;
  iof.seek = fmemopen_seek;
  iof.close = fmemopen_close;

  result = _IO_fopencookie (c, mode, iof);
  if (__glibc_unlikely (result == NULL))
    {
      if (c->mybuffer)
	free (c->buffer);

      free (c);
    }

  return result;
}
libc_hidden_def (__fmemopen)
versioned_symbol (libc, __fmemopen, fmemopen, GLIBC_2_22);

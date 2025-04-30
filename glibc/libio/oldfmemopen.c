/* Fmemopen implementation.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Hanno Mueller, kontakt@hanno.de, 2000.

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

/*
 * fmemopen() - "my" version of a string stream
 * Hanno Mueller, kontakt@hanno.de
 *
 *
 * I needed fmemopen() for an application that I currently work on,
 * but couldn't find it in libio. The following snippet of code is an
 * attempt to implement what glibc's documentation describes.
 *
 *
 *
 * I already see some potential problems:
 *
 * - I never used the "original" fmemopen(). I am sure that "my"
 *   fmemopen() behaves differently than the original version.
 *
 * - The documentation doesn't say wether a string stream allows
 *   seeks. I checked the old fmemopen implementation in glibc's stdio
 *   directory, wasn't quite able to see what is going on in that
 *   source, but as far as I understand there was no seek there. For
 *   my application, I needed fseek() and ftell(), so it's here.
 *
 * - "append" mode and fseek(p, SEEK_END) have two different ideas
 *   about the "end" of the stream.
 *
 *   As described in the documentation, when opening the file in
 *   "append" mode, the position pointer will be set to the first null
 *   character of the string buffer (yet the buffer may already
 *   contain more data). For fseek(), the last byte of the buffer is
 *   used as the end of the stream.
 *
 * - It is unclear to me what the documentation tries to say when it
 *   explains what happens when you use fmemopen with a NULL
 *   buffer.
 *
 *   Quote: "fmemopen [then] allocates an array SIZE bytes long. This
 *   is really only useful if you are going to write things to the
 *   buffer and then read them back in again."
 *
 *   What does that mean if the original fmemopen() did not allow
 *   seeking? How do you read what you just wrote without seeking back
 *   to the beginning of the stream?
 *
 * - I think there should be a second version of fmemopen() that does
 *   not add null characters for each write. (At least in my
 *   application, I am not actually using strings but binary data and
 *   so I don't need the stream to add null characters on its own.)
 */

#include "libioP.h"

#if SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_22)

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>


typedef struct fmemopen_cookie_struct fmemopen_cookie_t;
struct fmemopen_cookie_struct
{
  char *buffer;
  int mybuffer;
  int binmode;
  size_t size;
  off64_t pos;
  size_t maxpos;
};


static ssize_t
fmemopen_read (void *cookie, char *b, size_t s)
{
  fmemopen_cookie_t *c;

  c = (fmemopen_cookie_t *) cookie;

  if (c->pos + s > c->size)
    {
      if ((size_t) c->pos == c->size)
	return 0;
      s = c->size - c->pos;
    }

  memcpy (b, &(c->buffer[c->pos]), s);

  c->pos += s;
  if ((size_t) c->pos > c->maxpos)
    c->maxpos = c->pos;

  return s;
}


static ssize_t
fmemopen_write (void *cookie, const char *b, size_t s)
{
  fmemopen_cookie_t *c;
  int addnullc;

  c = (fmemopen_cookie_t *) cookie;

  addnullc = c->binmode == 0 && (s == 0 || b[s - 1] != '\0');

  if (c->pos + s + addnullc > c->size)
    {
      if ((size_t) (c->pos + addnullc) >= c->size)
	{
	  __set_errno (ENOSPC);
	  return 0;
	}
      s = c->size - c->pos - addnullc;
    }

  memcpy (&(c->buffer[c->pos]), b, s);

  c->pos += s;
  if ((size_t) c->pos > c->maxpos)
    {
      c->maxpos = c->pos;
      if (addnullc)
	c->buffer[c->maxpos] = '\0';
    }

  return s;
}


static int
fmemopen_seek (void *cookie, off64_t *p, int w)
{
  off64_t np;
  fmemopen_cookie_t *c;

  c = (fmemopen_cookie_t *) cookie;

  switch (w)
    {
    case SEEK_SET:
      np = *p;
      break;

    case SEEK_CUR:
      np = c->pos + *p;
      break;

    case SEEK_END:
      np = (c->binmode ? c->size : c->maxpos) - *p;
      break;

    default:
      return -1;
    }

  if (np < 0 || (size_t) np > c->size)
    return -1;

  *p = c->pos = np;

  return 0;
}


static int
fmemopen_close (void *cookie)
{
  fmemopen_cookie_t *c;

  c = (fmemopen_cookie_t *) cookie;

  if (c->mybuffer)
    free (c->buffer);
  free (c);

  return 0;
}


FILE *
__old_fmemopen (void *buf, size_t len, const char *mode)
{
  cookie_io_functions_t iof;
  fmemopen_cookie_t *c;
  FILE *result;

  if (__glibc_unlikely (len == 0))
    {
    einval:
      __set_errno (EINVAL);
      return NULL;
    }

  c = (fmemopen_cookie_t *) malloc (sizeof (fmemopen_cookie_t));
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
      c->maxpos = 0;
    }
  else
    {
      if (__glibc_unlikely ((uintptr_t) len > -(uintptr_t) buf))
	{
	  free (c);
	  goto einval;
	}

      c->buffer = buf;

      if (mode[0] == 'w')
	c->buffer[0] = '\0';

      c->maxpos = strnlen (c->buffer, len);
    }

  c->size = len;

  if (mode[0] == 'a')
    c->pos = c->maxpos;
  else
    c->pos = 0;

  c->binmode = mode[0] != '\0' && mode[1] == 'b';

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
compat_symbol (libc, __old_fmemopen, fmemopen, GLIBC_2_2);
#endif

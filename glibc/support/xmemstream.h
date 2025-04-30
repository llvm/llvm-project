/* Error-checking wrappers for memstream functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_XMEMSTREAM_H
#define SUPPORT_XMEMSTREAM_H

#include <stdio.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Wrappers for other libc functions.  */
struct xmemstream
{
  FILE *out;
  char *buffer;
  size_t length;
};

/* Create a new in-memory stream.  Initializes *STREAM.  After this
   function returns, STREAM->out is a file descriptor open for
   writing.  errno is preserved, so that the %m format specifier can
   be used for writing to STREAM->out.  */
void xopen_memstream (struct xmemstream *stream);

/* Closes STREAM->OUT.  After this function returns, STREAM->buffer
   and STREAM->length denote a memory range which contains the bytes
   written to the output stream.  The caller should free
   STREAM->buffer.  */
void xfclose_memstream (struct xmemstream *stream);

__END_DECLS

#endif /* SUPPORT_XMEMSTREAM_H */

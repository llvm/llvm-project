/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

#include "libioP.h"
#include <string.h>

size_t
_IO_getline (FILE *fp, char *buf, size_t n, int delim,
	     int extract_delim)
{
  return _IO_getline_info (fp, buf, n, delim, extract_delim, (int *) 0);
}
libc_hidden_def (_IO_getline)

/* Algorithm based on that used by Berkeley pre-4.4 fgets implementation.

   Read chars into buf (of size n), until delim is seen.
   Return number of chars read (at most n).
   Does not put a terminating '\0' in buf.
   If extract_delim < 0, leave delimiter unread.
   If extract_delim > 0, insert delim in output. */

size_t
_IO_getline_info (FILE *fp, char *buf, size_t n, int delim,
		  int extract_delim, int *eof)
{
  char *ptr = buf;
  if (eof != NULL)
    *eof = 0;
  if (__builtin_expect (fp->_mode, -1) == 0)
    _IO_fwide (fp, -1);
  while (n != 0)
    {
      ssize_t len = fp->_IO_read_end - fp->_IO_read_ptr;
      if (len <= 0)
	{
	  int c = __uflow (fp);
	  if (c == EOF)
	    {
	      if (eof)
		*eof = c;
	      break;
	    }
	  if (c == delim)
	    {
 	      if (extract_delim > 0)
		*ptr++ = c;
	      else if (extract_delim < 0)
		_IO_sputbackc (fp, c);
	      if (extract_delim > 0)
		++len;
	      return ptr - buf;
	    }
	  *ptr++ = c;
	  n--;
	}
      else
	{
	  char *t;
	  if ((size_t) len >= n)
	    len = n;
	  t = (char *) memchr ((void *) fp->_IO_read_ptr, delim, len);
	  if (t != NULL)
	    {
	      size_t old_len = ptr-buf;
	      len = t - fp->_IO_read_ptr;
	      if (extract_delim >= 0)
		{
		  ++t;
		  if (extract_delim > 0)
		    ++len;
		}
	      memcpy ((void *) ptr, (void *) fp->_IO_read_ptr, len);
	      fp->_IO_read_ptr = t;
	      return old_len + len;
	    }
	  memcpy ((void *) ptr, (void *) fp->_IO_read_ptr, len);
	  fp->_IO_read_ptr += len;
	  ptr += len;
	  n -= len;
	}
    }
  return ptr - buf;
}
libc_hidden_def (_IO_getline_info)

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
   <https://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

#include <stdlib.h>
#include "libioP.h"
#include <string.h>
#include <errno.h>
#include <limits.h>

/* Read up to (and including) a TERMINATOR from FP into *LINEPTR
   (and null-terminate it).  *LINEPTR is a pointer returned from malloc (or
   NULL), pointing to *N characters of space.  It is realloc'ed as
   necessary.  Returns the number of characters read (not including the
   null terminator), or -1 on error or EOF.  */

ssize_t
__getdelim (char **lineptr, size_t *n, int delimiter, FILE *fp)
{
  ssize_t result;
  ssize_t cur_len = 0;
  ssize_t len;

  if (lineptr == NULL || n == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }
  CHECK_FILE (fp, -1);
  _IO_acquire_lock (fp);
  if (_IO_ferror_unlocked (fp))
    {
      result = -1;
      goto unlock_return;
    }

  if (*lineptr == NULL || *n == 0)
    {
      *n = 120;
      *lineptr = (char *) malloc (*n);
      if (*lineptr == NULL)
	{
	  result = -1;
	  goto unlock_return;
	}
    }

  len = fp->_IO_read_end - fp->_IO_read_ptr;
  if (len <= 0)
    {
      if (__underflow (fp) == EOF)
	{
	  result = -1;
	  goto unlock_return;
	}
      len = fp->_IO_read_end - fp->_IO_read_ptr;
    }

  for (;;)
    {
      size_t needed;
      char *t;
      t = (char *) memchr ((void *) fp->_IO_read_ptr, delimiter, len);
      if (t != NULL)
	len = (t - fp->_IO_read_ptr) + 1;
      if (__glibc_unlikely (len >= SSIZE_MAX - cur_len))
	{
	  __set_errno (EOVERFLOW);
	  result = -1;
	  goto unlock_return;
	}
      /* Make enough space for len+1 (for final NUL) bytes.  */
      needed = cur_len + len + 1;
      if (needed > *n)
	{
	  char *new_lineptr;

	  if (needed < 2 * *n)
	    needed = 2 * *n;  /* Be generous. */
	  new_lineptr = (char *) realloc (*lineptr, needed);
	  if (new_lineptr == NULL)
	    {
	      result = -1;
	      goto unlock_return;
	    }
	  *lineptr = new_lineptr;
	  *n = needed;
	}
      memcpy (*lineptr + cur_len, (void *) fp->_IO_read_ptr, len);
      fp->_IO_read_ptr += len;
      cur_len += len;
      if (t != NULL || __underflow (fp) == EOF)
	break;
      len = fp->_IO_read_end - fp->_IO_read_ptr;
    }
  (*lineptr)[cur_len] = '\0';
  result = cur_len;

unlock_return:
  _IO_release_lock (fp);
  return result;
}
libc_hidden_def (__getdelim)
weak_alias (__getdelim, getdelim)

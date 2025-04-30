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
#include <wchar.h>

wchar_t *
fgetws_unlocked (wchar_t *buf, int n, FILE *fp)
{
  size_t count;
  wchar_t *result;
  int old_error;
  CHECK_FILE (fp, NULL);
  if (n <= 0)
    return NULL;
  if (__glibc_unlikely (n == 1))
    {
      /* Another irregular case: since we have to store a NUL byte and
	 there is only room for exactly one byte, we don't have to
	 read anything.  */
      buf[0] = L'\0';
      return buf;
    }
  /* This is very tricky since a file descriptor may be in the
     non-blocking mode. The error flag doesn't mean much in this
     case. We return an error only when there is a new error. */
  old_error = fp->_flags & _IO_ERR_SEEN;
  fp->_flags &= ~_IO_ERR_SEEN;
  count = _IO_getwline (fp, buf, n - 1, L'\n', 1);
  /* If we read in some bytes and errno is EAGAIN, that error will
     be reported for next read. */
  if (count == 0 || ((fp->_flags & _IO_ERR_SEEN) && errno != EAGAIN))
    result = NULL;
  else
    {
      buf[count] = '\0';
      result = buf;
    }
  fp->_flags |= old_error;
  return result;
}

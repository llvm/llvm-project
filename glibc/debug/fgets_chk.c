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
#include <stdio.h>
#include <sys/param.h>

char *
__fgets_chk (char *buf, size_t size, int n, FILE *fp)
{
  size_t count;
  char *result;
  CHECK_FILE (fp, NULL);
  if (n <= 0)
    return NULL;
  _IO_acquire_lock (fp);
  /* This is very tricky since a file descriptor may be in the
     non-blocking mode. The error flag doesn't mean much in this
     case. We return an error only when there is a new error. */
  int old_error = fp->_flags & _IO_ERR_SEEN;
  fp->_flags &= ~_IO_ERR_SEEN;
  count = _IO_getline (fp, buf, MIN ((size_t) n - 1, size), '\n', 1);
  /* If we read in some bytes and errno is EAGAIN, that error will
     be reported for next read. */
  if (count == 0 || ((fp->_flags & _IO_ERR_SEEN) && errno != EAGAIN))
    result = NULL;
  else if (count >= size)
    __chk_fail ();
  else
    {
      buf[count] = '\0';
      result = buf;
    }
  fp->_flags |= old_error;
  _IO_release_lock (fp);
  return result;
}

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

#include "../libio/libioP.h"
#include <limits.h>

char *
__gets_chk (char *buf, size_t size)
{
  size_t count;
  int ch;
  char *retval;

  if (size == 0)
    __chk_fail ();

  _IO_acquire_lock (stdin);
  ch = _IO_getc_unlocked (stdin);
  if (ch == EOF)
    {
      retval = NULL;
      goto unlock_return;
    }
  if (ch == '\n')
    count = 0;
  else
    {
      /* This is very tricky since a file descriptor may be in the
	 non-blocking mode. The error flag doesn't mean much in this
	 case. We return an error only when there is a new error. */
      int old_error = stdin->_flags & _IO_ERR_SEEN;
      stdin->_flags &= ~_IO_ERR_SEEN;
      buf[0] = (char) ch;
      count = _IO_getline (stdin, buf + 1, size - 1, '\n', 0) + 1;
      if (stdin->_flags & _IO_ERR_SEEN)
	{
	  retval = NULL;
	  goto unlock_return;
	}
      else
	stdin->_flags |= old_error;
    }
  if (count >= size)
    __chk_fail ();
  buf[count] = 0;
  retval = buf;
unlock_return:
  _IO_release_lock (stdin);
  return retval;
}

link_warning (__gets_chk, "the `gets' function is dangerous and should not be used.")

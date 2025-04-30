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

#define _IOFBF 0 /* Fully buffered. */
#define _IOLBF 1 /* Line buffered. */
#define _IONBF 2 /* No buffering. */

int
_IO_setvbuf (FILE *fp, char *buf, int mode, size_t size)
{
  int result;
  CHECK_FILE (fp, EOF);
  _IO_acquire_lock (fp);
  switch (mode)
    {
    case _IOFBF:
      fp->_flags &= ~(_IO_LINE_BUF|_IO_UNBUFFERED);
      if (buf == NULL)
	{
	  if (fp->_IO_buf_base == NULL)
	    {
	      /* There is no flag to distinguish between "fully buffered
		 mode has been explicitly set" as opposed to "line
		 buffering has not been explicitly set".  In both
		 cases, _IO_LINE_BUF is off.  If this is a tty, and
		 _IO_filedoalloc later gets called, it cannot know if
		 it should set the _IO_LINE_BUF flag (because that is
		 the default), or not (because we have explicitly asked
		 for fully buffered mode).  So we make sure a buffer
		 gets allocated now, and explicitly turn off line
		 buffering.

		 A possibly cleaner alternative would be to add an
		 extra flag, but then flags are a finite resource.  */
	      if (_IO_DOALLOCATE (fp) < 0)
		{
		  result = EOF;
		  goto unlock_return;
		}
	      fp->_flags &= ~_IO_LINE_BUF;
	    }
	  result = 0;
	  goto unlock_return;
	}
      break;
    case _IOLBF:
      fp->_flags &= ~_IO_UNBUFFERED;
      fp->_flags |= _IO_LINE_BUF;
      if (buf == NULL)
	{
	  result = 0;
	  goto unlock_return;
	}
      break;
    case _IONBF:
      fp->_flags &= ~_IO_LINE_BUF;
      fp->_flags |= _IO_UNBUFFERED;
      buf = NULL;
      size = 0;
      break;
    default:
      result = EOF;
      goto unlock_return;
    }
  result = _IO_SETBUF (fp, buf, size) == NULL ? EOF : 0;

unlock_return:
  _IO_release_lock (fp);
  return result;
}
libc_hidden_def (_IO_setvbuf)

weak_alias (_IO_setvbuf, setvbuf)

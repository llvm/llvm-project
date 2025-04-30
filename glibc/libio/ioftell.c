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
#include <stdlib.h>
#include <errno.h>
/* ANSI explicily requires setting errno to a positive value on failure. */

long int
_IO_ftell (FILE *fp)
{
  off64_t pos;
  CHECK_FILE (fp, -1L);
  _IO_acquire_lock (fp);
  pos = _IO_seekoff_unlocked (fp, 0, _IO_seek_cur, 0);
  if (_IO_in_backup (fp) && pos != _IO_pos_BAD)
    {
      if (_IO_vtable_offset (fp) != 0 || fp->_mode <= 0)
	pos -= fp->_IO_save_end - fp->_IO_save_base;
    }
  _IO_release_lock (fp);
  if (pos == _IO_pos_BAD)
    {
      if (errno == 0)
	__set_errno (EIO);
      return -1L;
    }
  if ((off64_t) (long int) pos != pos)
    {
      __set_errno (EOVERFLOW);
      return -1L;
    }
  return pos;
}
libc_hidden_def (_IO_ftell)

weak_alias (_IO_ftell, ftell)

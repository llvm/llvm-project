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

#undef fwrite_unlocked

size_t
fwrite_unlocked (const void *buf, size_t size, size_t count, FILE *fp)
{
  size_t request = size * count;
  size_t written = 0;
  CHECK_FILE (fp, 0);
  if (request == 0)
    return 0;
  if (_IO_fwide (fp, -1) == -1)
    {
      written = _IO_sputn (fp, (const char *) buf, request);
      /* We have written all of the input in case the return value indicates
	 this or EOF is returned.  The latter is a special case where we
	 simply did not manage to flush the buffer.  But the data is in the
	 buffer and therefore written as far as fwrite is concerned.  */
      if (written == request || written == EOF)
	return count;
    }

  return written / size;
}
libc_hidden_def (fwrite_unlocked)

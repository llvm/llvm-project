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

int
__fileno (FILE *fp)
{
  CHECK_FILE (fp, EOF);

  if (!(fp->_flags & _IO_IS_FILEBUF) || _IO_fileno (fp) < 0)
    {
      __set_errno (EBADF);
      return -1;
    }

  return _IO_fileno (fp);
}
libc_hidden_def (__fileno)
weak_alias (__fileno, fileno)
libc_hidden_weak (fileno)

/* The fileno implementation for libio does not require locking because
   it only accesses once a single variable and this is already atomic
   (at least at thread level).  Therefore we don't test _IO_MTSAFE_IO here.  */

weak_alias (__fileno, fileno_unlocked)

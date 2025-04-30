/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include "libioP.h"
#include "stdio.h"

#undef _IO_putc

int
_IO_putc (int c, FILE *fp)
{
  int result;
  CHECK_FILE (fp, EOF);
  if (!_IO_need_lock (fp))
    return _IO_putc_unlocked (c, fp);
  _IO_acquire_lock (fp);
  result = _IO_putc_unlocked (c, fp);
  _IO_release_lock (fp);
  return result;
}
libc_hidden_def (_IO_putc)

#undef putc

weak_alias (_IO_putc, putc)

#ifndef _IO_MTSAFE_IO
#undef putc_unlocked
weak_alias (_IO_putc, putc_unlocked)
#endif

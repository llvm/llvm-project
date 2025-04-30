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

int
_IO_fputs (const char *str, FILE *fp)
{
  size_t len = strlen (str);
  int result = EOF;
  CHECK_FILE (fp, EOF);
  _IO_acquire_lock (fp);
  if ((_IO_vtable_offset (fp) != 0 || _IO_fwide (fp, -1) == -1)
      && _IO_sputn (fp, str, len) == len)
    result = 1;
  _IO_release_lock (fp);
  return result;
}
libc_hidden_def (_IO_fputs)

weak_alias (_IO_fputs, fputs)
libc_hidden_weak (fputs)

# ifndef _IO_MTSAFE_IO
strong_alias (_IO_fputs, __fputs_unlocked)
libc_hidden_def (__fputs_unlocked)
weak_alias (_IO_fputs, fputs_unlocked)
libc_hidden_ver (_IO_fputs, fputs_unlocked)
# endif

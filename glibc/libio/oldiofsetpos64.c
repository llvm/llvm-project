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

#include <libioP.h>
#include <errno.h>

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_2)

int
attribute_compat_text_section
_IO_old_fsetpos64 (FILE *fp, const __fpos64_t *posp)
{
  int result;
  CHECK_FILE (fp, EOF);
  _IO_acquire_lock (fp);
  if (_IO_seekpos_unlocked (fp, posp->__pos, _IOS_INPUT|_IOS_OUTPUT)
      == _IO_pos_BAD)
    {
      /* ANSI explicitly requires setting errno to a positive value on
	 failure.  */
      if (errno == 0)
	__set_errno (EIO);
      result = EOF;
    }
  else
    result = 0;
  _IO_release_lock (fp);
  return result;
}

compat_symbol (libc, _IO_old_fsetpos64, _IO_fsetpos64, GLIBC_2_1);
strong_alias (_IO_old_fsetpos64, __old_fsetpos64)
compat_symbol (libc, __old_fsetpos64, fsetpos64, GLIBC_2_1);

#endif

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

/* We need to avoid the header declarations of these, because
   the types don't match _IO_fsetpos and then the compiler will
   complain about the mismatch when we do the alias below.  */
#define _IO_new_fsetpos64 __renamed__IO_new_fsetpos64
#define _IO_fsetpos64 __renamed__IO_fsetpos64
#define fsetpos64 __renamed_fsetpos64

#include <libioP.h>

#undef _IO_new_fsetpos64
#undef _IO_fsetpos64
#undef fsetpos64

#include <errno.h>
#include <shlib-compat.h>

int
_IO_new_fsetpos (FILE *fp, const __fpos_t *posp)
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
    {
      result = 0;
      if (fp->_mode > 0 && __libio_codecvt_encoding (fp->_codecvt) < 0)
	/* This is a stateful encoding, restore the state.  */
	fp->_wide_data->_IO_state = posp->__state;
    }
  _IO_release_lock (fp);
  return result;
}

strong_alias (_IO_new_fsetpos, __new_fsetpos)
versioned_symbol (libc, _IO_new_fsetpos, _IO_fsetpos, GLIBC_2_2);
versioned_symbol (libc, __new_fsetpos, fsetpos, GLIBC_2_2);

#ifdef __OFF_T_MATCHES_OFF64_T
strong_alias (_IO_new_fsetpos, _IO_new_fsetpos64)
strong_alias (_IO_new_fsetpos64, __new_fsetpos64)
versioned_symbol (libc, __new_fsetpos64, fsetpos64, GLIBC_2_2);
versioned_symbol (libc, _IO_new_fsetpos64, _IO_fsetpos64, GLIBC_2_2);
#endif

/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include "strfile.h"


static wint_t _IO_wstrn_overflow (FILE *fp, wint_t c) __THROW;

static wint_t
_IO_wstrn_overflow (FILE *fp, wint_t c)
{
  /* When we come to here this means the user supplied buffer is
     filled.  But since we must return the number of characters which
     would have been written in total we must provide a buffer for
     further use.  We can do this by writing on and on in the overflow
     buffer in the _IO_wstrnfile structure.  */
  _IO_wstrnfile *snf = (_IO_wstrnfile *) fp;

  if (fp->_wide_data->_IO_buf_base != snf->overflow_buf)
    {
      _IO_wsetb (fp, snf->overflow_buf,
		 snf->overflow_buf + (sizeof (snf->overflow_buf)
				      / sizeof (wchar_t)), 0);

      fp->_wide_data->_IO_write_base = snf->overflow_buf;
      fp->_wide_data->_IO_read_base = snf->overflow_buf;
      fp->_wide_data->_IO_read_ptr = snf->overflow_buf;
      fp->_wide_data->_IO_read_end = (snf->overflow_buf
				      + (sizeof (snf->overflow_buf)
					 / sizeof (wchar_t)));
    }

  fp->_wide_data->_IO_write_ptr = snf->overflow_buf;
  fp->_wide_data->_IO_write_end = snf->overflow_buf;

  /* Since we are not really interested in storing the characters
     which do not fit in the buffer we simply ignore it.  */
  return c;
}


const struct _IO_jump_t _IO_wstrn_jumps libio_vtable attribute_hidden =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_wstr_finish),
  JUMP_INIT(overflow, (_IO_overflow_t) _IO_wstrn_overflow),
  JUMP_INIT(underflow, (_IO_underflow_t) _IO_wstr_underflow),
  JUMP_INIT(uflow, (_IO_underflow_t) _IO_wdefault_uflow),
  JUMP_INIT(pbackfail, (_IO_pbackfail_t) _IO_wstr_pbackfail),
  JUMP_INIT(xsputn, _IO_wdefault_xsputn),
  JUMP_INIT(xsgetn, _IO_wdefault_xsgetn),
  JUMP_INIT(seekoff, _IO_wstr_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_default_setbuf),
  JUMP_INIT(sync, _IO_default_sync),
  JUMP_INIT(doallocate, _IO_wdefault_doallocate),
  JUMP_INIT(read, _IO_default_read),
  JUMP_INIT(write, _IO_default_write),
  JUMP_INIT(seek, _IO_default_seek),
  JUMP_INIT(close, _IO_default_close),
  JUMP_INIT(stat, _IO_default_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};


int
__vswprintf_internal (wchar_t *string, size_t maxlen, const wchar_t *format,
		      va_list args, unsigned int mode_flags)
{
  _IO_wstrnfile sf;
  int ret;
  struct _IO_wide_data wd;
#ifdef _IO_MTSAFE_IO
  sf.f._sbf._f._lock = NULL;
#endif

  if (maxlen == 0)
    /* Since we have to write at least the terminating L'\0' a buffer
       length of zero always makes the function fail.  */
    return -1;

  _IO_no_init (&sf.f._sbf._f, _IO_USER_LOCK, 0, &wd, &_IO_wstrn_jumps);
  _IO_fwide (&sf.f._sbf._f, 1);
  string[0] = L'\0';
  _IO_wstr_init_static (&sf.f._sbf._f, string, maxlen - 1, string);
  ret = __vfwprintf_internal ((FILE *) &sf.f._sbf, format, args, mode_flags);

  if (sf.f._sbf._f._wide_data->_IO_buf_base == sf.overflow_buf)
    /* ISO C99 requires swprintf/vswprintf to return an error if the
       output does not fit in the provided buffer.  */
    return -1;

  /* Terminate the string.  */
  *sf.f._sbf._f._wide_data->_IO_write_ptr = '\0';

  return ret;
}

int
__vswprintf (wchar_t *string, size_t maxlen, const wchar_t *format,
	     va_list args)
{
  return __vswprintf_internal (string, maxlen, format, args, 0);
}
ldbl_weak_alias (__vswprintf, vswprintf)

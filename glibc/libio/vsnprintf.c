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

static int _IO_strn_overflow (FILE *fp, int c) __THROW;

static int
_IO_strn_overflow (FILE *fp, int c)
{
  /* When we come to here this means the user supplied buffer is
     filled.  But since we must return the number of characters which
     would have been written in total we must provide a buffer for
     further use.  We can do this by writing on and on in the overflow
     buffer in the _IO_strnfile structure.  */
  _IO_strnfile *snf = (_IO_strnfile *) fp;

  if (fp->_IO_buf_base != snf->overflow_buf)
    {
      /* Terminate the string.  We know that there is room for at
	 least one more character since we initialized the stream with
	 a size to make this possible.  */
      *fp->_IO_write_ptr = '\0';

      _IO_setb (fp, snf->overflow_buf,
		snf->overflow_buf + sizeof (snf->overflow_buf), 0);

      fp->_IO_write_base = snf->overflow_buf;
      fp->_IO_read_base = snf->overflow_buf;
      fp->_IO_read_ptr = snf->overflow_buf;
      fp->_IO_read_end = snf->overflow_buf + sizeof (snf->overflow_buf);
    }

  fp->_IO_write_ptr = snf->overflow_buf;
  fp->_IO_write_end = snf->overflow_buf;

  /* Since we are not really interested in storing the characters
     which do not fit in the buffer we simply ignore it.  */
  return c;
}


const struct _IO_jump_t _IO_strn_jumps libio_vtable attribute_hidden =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_str_finish),
  JUMP_INIT(overflow, _IO_strn_overflow),
  JUMP_INIT(underflow, _IO_str_underflow),
  JUMP_INIT(uflow, _IO_default_uflow),
  JUMP_INIT(pbackfail, _IO_str_pbackfail),
  JUMP_INIT(xsputn, _IO_default_xsputn),
  JUMP_INIT(xsgetn, _IO_default_xsgetn),
  JUMP_INIT(seekoff, _IO_str_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_default_setbuf),
  JUMP_INIT(sync, _IO_default_sync),
  JUMP_INIT(doallocate, _IO_default_doallocate),
  JUMP_INIT(read, _IO_default_read),
  JUMP_INIT(write, _IO_default_write),
  JUMP_INIT(seek, _IO_default_seek),
  JUMP_INIT(close, _IO_default_close),
  JUMP_INIT(stat, _IO_default_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};


int
__vsnprintf_internal (char *string, size_t maxlen, const char *format,
		      va_list args, unsigned int mode_flags)
{
  _IO_strnfile sf;
  int ret;
#ifdef _IO_MTSAFE_IO
  sf.f._sbf._f._lock = NULL;
#endif

  /* We need to handle the special case where MAXLEN is 0.  Use the
     overflow buffer right from the start.  */
  if (maxlen == 0)
    {
      string = sf.overflow_buf;
      maxlen = sizeof (sf.overflow_buf);
    }

  _IO_no_init (&sf.f._sbf._f, _IO_USER_LOCK, -1, NULL, NULL);
  _IO_JUMPS (&sf.f._sbf) = &_IO_strn_jumps;
  string[0] = '\0';
  _IO_str_init_static_internal (&sf.f, string, maxlen - 1, string);
  ret = __vfprintf_internal (&sf.f._sbf._f, format, args, mode_flags);

  if (sf.f._sbf._f._IO_buf_base != sf.overflow_buf)
    *sf.f._sbf._f._IO_write_ptr = '\0';
  return ret;
}

int
___vsnprintf (char *string, size_t maxlen, const char *format, va_list args)
{
  return __vsnprintf_internal (string, maxlen, format, args, 0);
}
ldbl_weak_alias (___vsnprintf, __vsnprintf)
ldbl_weak_alias (___vsnprintf, vsnprintf)

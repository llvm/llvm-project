/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <stdlib.h>
#include <strfile.h>

int
__vasprintf_internal (char **result_ptr, const char *format, va_list args,
		      unsigned int mode_flags)
{
  /* Initial size of the buffer to be used.  Will be doubled each time an
     overflow occurs.  */
  const size_t init_string_size = 100;
  char *string;
  _IO_strfile sf;
  int ret;
  size_t needed;
  size_t allocated;
  /* No need to clear the memory here (unlike for open_memstream) since
     we know we will never seek on the stream.  */
  string = (char *) malloc (init_string_size);
  if (string == NULL)
    return -1;
#ifdef _IO_MTSAFE_IO
  sf._sbf._f._lock = NULL;
#endif
  _IO_no_init (&sf._sbf._f, _IO_USER_LOCK, -1, NULL, NULL);
  _IO_JUMPS (&sf._sbf) = &_IO_str_jumps;
  _IO_str_init_static_internal (&sf, string, init_string_size, string);
  sf._sbf._f._flags &= ~_IO_USER_BUF;
  sf._s._allocate_buffer_unused = (_IO_alloc_type) malloc;
  sf._s._free_buffer_unused = (_IO_free_type) free;
  ret = __vfprintf_internal (&sf._sbf._f, format, args, mode_flags);
  if (ret < 0)
    {
      free (sf._sbf._f._IO_buf_base);
      return ret;
    }
  /* Only use realloc if the size we need is of the same (binary)
     order of magnitude then the memory we allocated.  */
  needed = sf._sbf._f._IO_write_ptr - sf._sbf._f._IO_write_base + 1;
  allocated = sf._sbf._f._IO_write_end - sf._sbf._f._IO_write_base;
  if ((allocated >> 1) <= needed)
    *result_ptr = (char *) realloc (sf._sbf._f._IO_buf_base, needed);
  else
    {
      *result_ptr = (char *) malloc (needed);
      if (*result_ptr != NULL)
	{
	  memcpy (*result_ptr, sf._sbf._f._IO_buf_base, needed - 1);
	  free (sf._sbf._f._IO_buf_base);
	}
      else
	/* We have no choice, use the buffer we already have.  */
	*result_ptr = (char *) realloc (sf._sbf._f._IO_buf_base, needed);
    }
  if (*result_ptr == NULL)
    *result_ptr = sf._sbf._f._IO_buf_base;
  (*result_ptr)[needed - 1] = '\0';
  return ret;
}

int
__vasprintf (char **result_ptr, const char *format, va_list args)
{
  return __vasprintf_internal (result_ptr, format, args, 0);
}
ldbl_weak_alias (__vasprintf, vasprintf)

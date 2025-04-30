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
#include <fcntl.h>
#include <stdlib.h>
#include <stddef.h>
#include <shlib-compat.h>

FILE *
__fopen_maybe_mmap (FILE *fp)
{
#if _G_HAVE_MMAP
  if ((fp->_flags2 & _IO_FLAGS2_MMAP) && (fp->_flags & _IO_NO_WRITES))
    {
      /* Since this is read-only, we might be able to mmap the contents
	 directly.  We delay the decision until the first read attempt by
	 giving it a jump table containing functions that choose mmap or
	 vanilla file operations and reset the jump table accordingly.  */

      if (fp->_mode <= 0)
	_IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps_maybe_mmap;
      else
	_IO_JUMPS_FILE_plus (fp) = &_IO_wfile_jumps_maybe_mmap;
      fp->_wide_data->_wide_vtable = &_IO_wfile_jumps_maybe_mmap;
    }
#endif
  return fp;
}


FILE *
__fopen_internal (const char *filename, const char *mode, int is32)
{
  struct locked_FILE
  {
    struct _IO_FILE_plus fp;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
    struct _IO_wide_data wd;
  } *new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));

  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fp.file._lock = &new_f->lock;
#endif
  _IO_no_init (&new_f->fp.file, 0, 0, &new_f->wd, &_IO_wfile_jumps);
  _IO_JUMPS (&new_f->fp) = &_IO_file_jumps;
  _IO_new_file_init_internal (&new_f->fp);
  if (_IO_file_fopen ((FILE *) new_f, filename, mode, is32) != NULL)
    return __fopen_maybe_mmap (&new_f->fp.file);

  _IO_un_link (&new_f->fp);
  free (new_f);
  return NULL;
}

FILE *
_IO_new_fopen (const char *filename, const char *mode)
{
  return __fopen_internal (filename, mode, 1);
}

strong_alias (_IO_new_fopen, __new_fopen)
versioned_symbol (libc, _IO_new_fopen, _IO_fopen, GLIBC_2_1);
versioned_symbol (libc, __new_fopen, fopen, GLIBC_2_1);

# if !defined O_LARGEFILE || O_LARGEFILE == 0
weak_alias (_IO_new_fopen, _IO_fopen64)
weak_alias (_IO_new_fopen, fopen64)
# endif

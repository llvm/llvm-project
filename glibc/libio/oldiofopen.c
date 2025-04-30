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

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#define _IO_USE_OLD_IO_FILE
#include "libioP.h"
#include <stdlib.h>


FILE *
attribute_compat_text_section
_IO_old_fopen (const char *filename, const char *mode)
{
  struct locked_FILE
  {
    struct _IO_FILE_complete_plus fp;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
  } *new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));

  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fp.file._file._lock = &new_f->lock;
#endif
  _IO_old_init (&new_f->fp.file._file, 0);
  _IO_JUMPS_FILE_plus (&new_f->fp) = &_IO_old_file_jumps;
  _IO_old_file_init_internal ((struct _IO_FILE_plus *) &new_f->fp);
  if (_IO_old_file_fopen ((FILE *) &new_f->fp, filename, mode) != NULL)
    return (FILE *) &new_f->fp;
  _IO_un_link ((struct _IO_FILE_plus *) &new_f->fp);
  free (new_f);
  return NULL;
}

strong_alias (_IO_old_fopen, __old_fopen)
compat_symbol (libc, _IO_old_fopen, _IO_fopen, GLIBC_2_0);
compat_symbol (libc, __old_fopen, fopen, GLIBC_2_0);
#endif

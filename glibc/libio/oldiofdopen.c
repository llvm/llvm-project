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
#include <stdlib.h>
#include "libioP.h"
#include <fcntl.h>

FILE *
attribute_compat_text_section
_IO_old_fdopen (int fd, const char *mode)
{
  int read_write;
  int posix_mode = 0;
  struct locked_FILE
  {
    struct _IO_FILE_complete_plus fp;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
  } *new_f;
  int fd_flags;

  switch (*mode++)
    {
    case 'r':
      read_write = _IO_NO_WRITES;
      break;
    case 'w':
      read_write = _IO_NO_READS;
      break;
    case 'a':
      posix_mode = O_APPEND;
      read_write = _IO_NO_READS|_IO_IS_APPENDING;
      break;
    default:
      __set_errno (EINVAL);
      return NULL;
  }
  if (mode[0] == '+' || (mode[0] == 'b' && mode[1] == '+'))
    read_write &= _IO_IS_APPENDING;
  fd_flags = __fcntl (fd, F_GETFL);
  if (fd_flags == -1
      || ((fd_flags & O_ACCMODE) == O_RDONLY && !(read_write & _IO_NO_WRITES))
      || ((fd_flags & O_ACCMODE) == O_WRONLY && !(read_write & _IO_NO_READS)))
    return NULL;

  /* The May 93 draft of P1003.4/D14.1 (redesignated as 1003.1b)
     [System Application Program Interface (API) Amendment 1:
     Realtime Extensions], Rationale B.8.3.3
     Open a Stream on a File Descriptor says:

         Although not explicitly required by POSIX.1, a good
         implementation of append ("a") mode would cause the
         O_APPEND flag to be set.

     (Historical implementations [such as Solaris2] do a one-time
     seek in fdopen.)

     However, we do not turn O_APPEND off if the mode is "w" (even
     though that would seem consistent) because that would be more
     likely to break historical programs.
     */
  if ((posix_mode & O_APPEND) && !(fd_flags & O_APPEND))
    {
      if (__fcntl (fd, F_SETFL, fd_flags | O_APPEND) == -1)
	return NULL;
    }

  new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));
  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fp.file._file._lock = &new_f->lock;
#endif
  _IO_old_init (&new_f->fp.file._file, 0);
  _IO_JUMPS_FILE_plus (&new_f->fp) = &_IO_old_file_jumps;
  _IO_old_file_init_internal ((struct _IO_FILE_plus *) &new_f->fp);
  if (_IO_old_file_attach (&new_f->fp.file._file, fd) == NULL)
    {
      _IO_un_link ((struct _IO_FILE_plus *) &new_f->fp);
      free (new_f);
      return NULL;
    }
  new_f->fp.file._file._flags &= ~_IO_DELETE_DONT_CLOSE;

  _IO_mask_flags (&new_f->fp.file._file, read_write,
		  _IO_NO_READS+_IO_NO_WRITES+_IO_IS_APPENDING);

  return (FILE *) &new_f->fp;
}

strong_alias (_IO_old_fdopen, __old_fdopen)
compat_symbol (libc, _IO_old_fdopen, _IO_fdopen, GLIBC_2_0);
compat_symbol (libc, __old_fdopen, fdopen, GLIBC_2_0);

#endif

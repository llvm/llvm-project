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

#include <stdlib.h>
#include "libioP.h"
#include <fcntl.h>

#include <shlib-compat.h>

FILE *
_IO_new_fdopen (int fd, const char *mode)
{
  int read_write;
  struct locked_FILE
  {
    struct _IO_FILE_plus fp;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
    struct _IO_wide_data wd;
  } *new_f;
  int i;
  int use_mmap = 0;

  /* Decide whether we modify the offset of the file we attach to and seek to
     the end of file.  We only do this if the mode is 'a' and if the file
     descriptor did not have O_APPEND in its flags already.  */
  bool do_seek = false;

  switch (*mode)
    {
    case 'r':
      read_write = _IO_NO_WRITES;
      break;
    case 'w':
      read_write = _IO_NO_READS;
      break;
    case 'a':
      read_write = _IO_NO_READS|_IO_IS_APPENDING;
      break;
    default:
      __set_errno (EINVAL);
      return NULL;
  }
  for (i = 1; i < 5; ++i)
    {
      switch (*++mode)
	{
	case '\0':
	  break;
	case '+':
	  read_write &= _IO_IS_APPENDING;
	  break;
	case 'm':
	  use_mmap = 1;
	  continue;
	case 'x':
	case 'b':
	default:
	  /* Ignore */
	  continue;
	}
      break;
    }
  int fd_flags = __fcntl (fd, F_GETFL);
  if (fd_flags == -1)
    return NULL;

  if (((fd_flags & O_ACCMODE) == O_RDONLY && !(read_write & _IO_NO_WRITES))
      || ((fd_flags & O_ACCMODE) == O_WRONLY && !(read_write & _IO_NO_READS)))
    {
      __set_errno (EINVAL);
      return NULL;
    }

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
  if ((read_write & _IO_IS_APPENDING) && !(fd_flags & O_APPEND))
    {
      do_seek = true;
      if (__fcntl (fd, F_SETFL, fd_flags | O_APPEND) == -1)
	return NULL;
    }

  new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));
  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fp.file._lock = &new_f->lock;
#endif
  _IO_no_init (&new_f->fp.file, 0, 0, &new_f->wd,
#if _G_HAVE_MMAP
	       (use_mmap && (read_write & _IO_NO_WRITES))
	       ? &_IO_wfile_jumps_maybe_mmap :
#endif
	       &_IO_wfile_jumps);
  _IO_JUMPS (&new_f->fp) =
#if _G_HAVE_MMAP
    (use_mmap && (read_write & _IO_NO_WRITES)) ? &_IO_file_jumps_maybe_mmap :
#endif
      &_IO_file_jumps;
  _IO_new_file_init_internal (&new_f->fp);
  /* We only need to record the fd because _IO_file_init_internal will
     have unset the offset.  It is important to unset the cached
     offset because the real offset in the file could change between
     now and when the handle is activated and we would then mislead
     ftell into believing that we have a valid offset.  */
  new_f->fp.file._fileno = fd;
  new_f->fp.file._flags &= ~_IO_DELETE_DONT_CLOSE;

  _IO_mask_flags (&new_f->fp.file, read_write,
		  _IO_NO_READS+_IO_NO_WRITES+_IO_IS_APPENDING);

  /* For append mode, set the file offset to the end of the file if we added
     O_APPEND to the file descriptor flags.  Don't update the offset cache
     though, since the file handle is not active.  */
  if (do_seek && ((read_write & (_IO_IS_APPENDING | _IO_NO_READS))
		  == (_IO_IS_APPENDING | _IO_NO_READS)))
    {
      off64_t new_pos = _IO_SYSSEEK (&new_f->fp.file, 0, _IO_seek_end);
      if (new_pos == _IO_pos_BAD && errno != ESPIPE)
	return NULL;
    }
  return &new_f->fp.file;
}
libc_hidden_ver (_IO_new_fdopen, _IO_fdopen)

strong_alias (_IO_new_fdopen, __new_fdopen)
versioned_symbol (libc, _IO_new_fdopen, _IO_fdopen, GLIBC_2_1);
versioned_symbol (libc, __new_fdopen, fdopen, GLIBC_2_1);

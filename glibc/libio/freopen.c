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

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <libioP.h>
#include <fd_to_filename.h>
#include <shlib-compat.h>

FILE *
freopen (const char *filename, const char *mode, FILE *fp)
{
  FILE *result = NULL;
  struct fd_to_filename fdfilename;

  CHECK_FILE (fp, NULL);

  _IO_acquire_lock (fp);
  /* First flush the stream (failure should be ignored).  */
  _IO_SYNC (fp);

  if (!(fp->_flags & _IO_IS_FILEBUF))
    goto end;

  int fd = _IO_fileno (fp);
  const char *gfilename
    = filename != NULL ? filename : __fd_to_filename (fd, &fdfilename);

  fp->_flags2 |= _IO_FLAGS2_NOCLOSE;
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
  if (&_IO_stdin_used == NULL)
    {
      /* If the shared C library is used by the application binary which
	 was linked against the older version of libio, we just use the
	 older one even for internal use to avoid trouble since a pointer
	 to the old libio may be passed into shared C library and wind
	 up here. */
      _IO_old_file_close_it (fp);
      _IO_JUMPS_FUNC_UPDATE (fp, &_IO_old_file_jumps);
      result = _IO_old_file_fopen (fp, gfilename, mode);
    }
  else
#endif
    {
      _IO_file_close_it (fp);
      _IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps;
      if (_IO_vtable_offset (fp) == 0 && fp->_wide_data != NULL)
	fp->_wide_data->_wide_vtable = &_IO_wfile_jumps;
      result = _IO_file_fopen (fp, gfilename, mode, 1);
      if (result != NULL)
	result = __fopen_maybe_mmap (result);
    }
  fp->_flags2 &= ~_IO_FLAGS2_NOCLOSE;
  if (result != NULL)
    {
      /* unbound stream orientation */
      result->_mode = 0;

      if (fd != -1 && _IO_fileno (result) != fd)
	{
	  /* At this point we have both file descriptors already allocated,
	     so __dup3 will not fail with EBADF, EINVAL, or EMFILE.  But
	     we still need to check for EINVAL and, due Linux internal
	     implementation, EBUSY.  It is because on how it internally opens
	     the file by splitting the buffer allocation operation and VFS
	     opening (a dup operation may run when a file is still pending
	     'install' on VFS).  */
	  if (__dup3 (_IO_fileno (result), fd,
		      (result->_flags2 & _IO_FLAGS2_CLOEXEC) != 0
		      ? O_CLOEXEC : 0) == -1)
	    {
	      _IO_file_close_it (result);
	      result = NULL;
	      goto end;
	    }
	  __close (_IO_fileno (result));
	  _IO_fileno (result) = fd;
	}
    }
  else if (fd != -1)
    __close (fd);

end:
  _IO_release_lock (fp);
  return result;
}

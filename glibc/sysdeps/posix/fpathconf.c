/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>


/* Get file-specific information about descriptor FD.  */
long int
__fpathconf (int fd, int name)
{
  if (fd < 0)
    {
      __set_errno (EBADF);
      return -1;
    }

  switch (name)
    {
    default:
      __set_errno (EINVAL);
      return -1;

    case _PC_LINK_MAX:
#ifdef	LINK_MAX
      return LINK_MAX;
#else
      return -1;
#endif

    case _PC_MAX_CANON:
#ifdef	MAX_CANON
      return MAX_CANON;
#else
      return -1;
#endif

    case _PC_MAX_INPUT:
#ifdef	MAX_INPUT
      return MAX_INPUT;
#else
      return -1;
#endif

    case _PC_NAME_MAX:
#ifdef	NAME_MAX
      {
	struct statvfs64 sv;
	int save_errno = errno;

	if (__fstatvfs64 (fd, &sv) < 0)
	  {
	    if (errno == ENOSYS)
	      {
		__set_errno (save_errno);
		return NAME_MAX;
	      }
	    else if (errno == ENODEV)
	      __set_errno (EINVAL);

	    return -1;
	  }
	else
	  {
	    return sv.f_namemax;
	  }
      }
#else
      return -1;
#endif

    case _PC_PATH_MAX:
#ifdef	PATH_MAX
      return PATH_MAX;
#else
      return -1;
#endif

    case _PC_PIPE_BUF:
#ifdef	PIPE_BUF
      return PIPE_BUF;
#else
      return -1;
#endif

    case _PC_CHOWN_RESTRICTED:
#if _POSIX_CHOWN_RESTRICTED == -1
# error "Invalid value for _POSIX_CHOWN_RESTRICTED"
#endif
      return _POSIX_CHOWN_RESTRICTED;

    case _PC_NO_TRUNC:
#if _POSIX_NO_TRUNC == -1
# error "Invalid value for _POSIX_NO_TRUNC"
#endif
      return _POSIX_NO_TRUNC;

    case _PC_VDISABLE:
#if _POSIX_VDISABLE == -1
# error "Invalid value for _POSIX_VDISABLE"
#endif
      return _POSIX_VDISABLE;

    case _PC_SYNC_IO:
#ifdef	_POSIX_SYNC_IO
      return _POSIX_SYNC_IO;
#else
      return -1;
#endif

    case _PC_ASYNC_IO:
#ifdef	_POSIX_ASYNC_IO
      {
	/* AIO is only allowed on regular files and block devices.  */
	struct stat64 st;

	if (__fstat64 (fd, &st) < 0
	    || (! S_ISREG (st.st_mode) && ! S_ISBLK (st.st_mode)))
	  return -1;
	else
	  return 1;
      }
#else
      return -1;
#endif

    case _PC_PRIO_IO:
#ifdef	_POSIX_PRIO_IO
      return _POSIX_PRIO_IO;
#else
      return -1;
#endif

    case _PC_SOCK_MAXBUF:
#ifdef	SOCK_MAXBUF
      return SOCK_MAXBUF;
#else
      return -1;
#endif

    case _PC_FILESIZEBITS:
#ifdef FILESIZEBITS
      return FILESIZEBITS;
#else
      /* We let platforms with larger file sizes overwrite this value.  */
      return 32;
#endif

    case _PC_REC_INCR_XFER_SIZE:
      /* XXX It is not entirely clear what the limit is supposed to do.
	 What is incremented?  */
      return -1;

    case _PC_REC_MAX_XFER_SIZE:
      /* XXX It is not entirely clear what the limit is supposed to do.
	 In general there is no top limit of the number of bytes which
	 case be transported at once.  */
      return -1;

    case _PC_REC_MIN_XFER_SIZE:
      {
	/* XXX It is not entirely clear what the limit is supposed to do.
	   I assume this is the block size of the filesystem.  */
	struct statvfs64 sv;

	if (__fstatvfs64 (fd, &sv) < 0)
	  return -1;
	return sv.f_bsize;
      }

    case _PC_REC_XFER_ALIGN:
      {
	/* XXX It is not entirely clear what the limit is supposed to do.
	   I assume that the number should reflect the minimal block
	   alignment.  */
	struct statvfs64 sv;

	if (__fstatvfs64 (fd, &sv) < 0)
	  return -1;
	return sv.f_frsize;
      }

    case _PC_ALLOC_SIZE_MIN:
      {
	/* XXX It is not entirely clear what the limit is supposed to do.
	   I assume that the number should reflect the minimal block
	   alignment.  */
	struct statvfs64 sv;

	if (__fstatvfs64 (fd, &sv) < 0)
	  return -1;
	return sv.f_frsize;
      }

    case _PC_SYMLINK_MAX:
      /* In general there are no limits.  If a system has one it should
	 overwrite this case.  */
      return -1;

    case _PC_2_SYMLINKS:
      /* Unix systems generally have symlinks.  */
      return 1;
    }
}

#undef __fpathconf
weak_alias (__fpathconf, fpathconf)

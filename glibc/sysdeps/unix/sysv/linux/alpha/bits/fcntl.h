/* O_*, F_*, FD_* bit values for Linux.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef	_FCNTL_H
# error "Never use <bits/fcntl.h> directly; include <fcntl.h> instead."
#endif

#define O_CREAT		 01000	/* not fcntl */
#define O_TRUNC		 02000	/* not fcntl */
#define O_EXCL		 04000	/* not fcntl */
#define O_NOCTTY	010000	/* not fcntl */

#define O_NONBLOCK	 00004
#define O_APPEND	 00010
#define O_SYNC		020040000

#define __O_DIRECTORY	0100000	/* Must be a directory.  */
#define __O_NOFOLLOW	0200000	/* Do not follow links.  */
#define __O_CLOEXEC	010000000 /* Set close_on_exec.  */

#define __O_DIRECT	02000000 /* Direct disk access.  */
#define __O_NOATIME	04000000 /* Do not set atime.  */
#define __O_PATH	040000000 /* Resolve pathname but do not open file.  */
#define __O_TMPFILE	0100100000 /* Atomically create nameless file.  */

/* Not necessary, files are always with 64bit off_t.  */
#define __O_LARGEFILE	0

#define __O_DSYNC	040000	/* Synchronize data.  */

#define F_GETLK		7	/* Get record locking info.  */
#define F_SETLK		8	/* Set record locking info (non-blocking).  */
#define F_SETLKW	9	/* Set record locking info (blocking).  */
#define F_GETLK64	F_GETLK	/* Get record locking info.  */
#define F_SETLK64	F_SETLK	/* Set record locking info (non-blocking).  */
#define F_SETLKW64	F_SETLKW /* Set record locking info (blocking).  */

#define __F_SETOWN	5	/* Get owner of socket (receiver of SIGIO).  */
#define __F_GETOWN	6	/* Set owner of socket (receiver of SIGIO).  */

/* For posix fcntl() and `l_type' field of a `struct flock' for lockf() */
#define F_RDLCK		1	/* Read lock.  */
#define F_WRLCK		2	/* Write lock.  */
#define F_UNLCK		8	/* Remove lock.  */

/* for old implementation of bsd flock () */
#define F_EXLCK		16	/* or 3 */
#define F_SHLCK		32	/* or 4 */

/* We don't need to support __USE_FILE_OFFSET64.  */
struct flock
  {
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
    __off_t l_start;	/* Offset where the lock begins.  */
    __off_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
  };

#ifdef __USE_LARGEFILE64
struct flock64
  {
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
    __off64_t l_start;	/* Offset where the lock begins.  */
    __off64_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
  };
#endif

/* Include generic Linux declarations.  */
#include <bits/fcntl-linux.h>

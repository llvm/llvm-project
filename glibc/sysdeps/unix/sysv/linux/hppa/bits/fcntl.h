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

#define O_CREAT		00000400 /* not fcntl */
#define O_EXCL		00002000 /* not fcntl */
#define O_NOCTTY	00400000 /* not fcntl */
#define O_APPEND	00000010
#define O_NONBLOCK	00200000
#define __O_DSYNC	01000000
#define __O_SYNC	00100000
#define O_SYNC		(__O_SYNC|__O_DSYNC)

#define __O_DIRECTORY	000010000 /* Must be a directory.  */
#define __O_NOFOLLOW	000000200 /* Do not follow links.  */
#define __O_CLOEXEC	010000000 /* Set close_on_exec.  */
#define __O_NOATIME	004000000 /* Do not set atime.  */
#define __O_PATH        020000000
#define __O_TMPFILE     040010000 /* Atomically create nameless file. */

#define __O_LARGEFILE	00004000

#define F_GETLK64	8	/* Get record locking info.  */
#define F_SETLK64	9	/* Set record locking info (non-blocking).  */
#define F_SETLKW64	10	/* Set record locking info (blocking).  */

#define __F_GETOWN	11	/* Get owner of socket (receiver of SIGIO).  */
#define __F_SETOWN	12	/* Set owner of socket (receiver of SIGIO).  */

#define __F_SETSIG	13	/* Set number of signal to be sent.  */
#define __F_GETSIG	14	/* Get number of signal to be sent.  */

/* For posix fcntl() and `l_type' field of a `struct flock' for lockf().  */
#define F_RDLCK		1	/* Read lock.  */
#define F_WRLCK		2	/* Write lock.  */
#define F_UNLCK		3	/* Remove lock.  */

struct flock
  {
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.	*/
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
#ifndef __USE_FILE_OFFSET64
    __off_t l_start;	/* Offset where the lock begins.  */
    __off_t l_len;	/* Size of the locked area; zero means until EOF.  */
#else
    __off64_t l_start;	/* Offset where the lock begins.  */
    __off64_t l_len;	/* Size of the locked area; zero means until EOF.  */
#endif
    __pid_t l_pid;	/* Process holding the lock.  */
  };

#ifdef __USE_LARGEFILE64
struct flock64
  {
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.	*/
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
    __off64_t l_start;	/* Offset where the lock begins.  */
    __off64_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
  };
#endif

/* Include generic Linux declarations.  */
#include <bits/fcntl-linux.h>

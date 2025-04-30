/* O_*, F_*, FD_* bit values. 4.4BSD/Generic version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef	_FCNTL_H
#error "Never use <bits/fcntl.h> directly; include <fcntl.h> instead."
#endif


/* File access modes for `open' and `fcntl'.  */
#define	O_RDONLY	0	/* Open read-only.  */
#define	O_WRONLY	1	/* Open write-only.  */
#define	O_RDWR		2	/* Open read/write.  */


/* Bits OR'd into the second argument to open.  */
#define	O_CREAT		0x0200	/* Create file if it doesn't exist.  */
#define	O_EXCL		0x0800	/* Fail if file already exists.  */
#define	O_TRUNC		0x0400	/* Truncate file to zero length.  */
#define	O_NOCTTY	0x8000	/* Don't assign a controlling terminal.  */
#define	O_ASYNC		0x0040	/* Send SIGIO to owner when data is ready.  */
#define	O_FSYNC		0x0080	/* Synchronous writes.  */
#define	O_SYNC		O_FSYNC
#ifdef	__USE_MISC
#define	O_SHLOCK	0x0010	/* Open with shared file lock.  */
#define	O_EXLOCK	0x0020	/* Open with shared exclusive lock.  */
#endif
#ifdef __USE_XOPEN2K8
# define O_DIRECTORY	0x00200000	/* Must be a directory.	 */
# define O_NOFOLLOW	0x00000100	/* Do not follow links.	 */
# define O_CLOEXEC	0x00400000      /* Set close_on_exec.  */
#endif
#if defined __USE_POSIX199309 || defined __USE_UNIX98
# define O_DSYNC	0x00010000	/* Synchronize data.  */
# define O_RSYNC	0x00020000	/* Synchronize read operations.	 */
#endif

/* All opens support large file sizes, so there is no flag bit for this.  */
#ifdef __USE_LARGEFILE64
# define O_LARGEFILE	0
#endif

/* File status flags for `open' and `fcntl'.  */
#define	O_APPEND	0x0008	/* Writes append to the file.  */
#define	O_NONBLOCK	0x0004	/* Non-blocking I/O.  */

#ifdef __USE_MISC
# define O_NDELAY	O_NONBLOCK
#endif

#ifdef __USE_MISC
/* Flags for TIOCFLUSH.  */
# define FREAD		1
# define FWRITE		2

/* Traditional BSD names the O_* bits.  */
# define FASYNC		O_ASYNC
# define FFSYNC		O_FSYNC
# define FSYNC		O_SYNC
# define FAPPEND	O_APPEND
# define FNDELAY	O_NDELAY
#endif

/* Mask for file access modes.  This is system-dependent in case
   some system ever wants to define some other flavor of access.  */
#define	O_ACCMODE	(O_RDONLY|O_WRONLY|O_RDWR)

/* Values for the second argument to `fcntl'.  */
#define	F_DUPFD	  	0	/* Duplicate file descriptor.  */
#define	F_GETFD		1	/* Get file descriptor flags.  */
#define	F_SETFD		2	/* Set file descriptor flags.  */
#define	F_GETFL		3	/* Get file status flags.  */
#define	F_SETFL		4	/* Set file status flags.  */
#if defined __USE_UNIX98 || defined __USE_XOPEN2K8
#define	F_GETOWN	5	/* Get owner (receiver of SIGIO).  */
#define	F_SETOWN	6	/* Set owner (receiver of SIGIO).  */
#endif
#define	F_GETLK		7	/* Get record locking info.  */
#define	F_SETLK		8	/* Set record locking info (non-blocking).  */
#define	F_SETLKW	9	/* Set record locking info (blocking).  */
/* Not necessary, we always have 64-bit offsets.  */
#define F_GETLK64	F_GETLK	/* Get record locking info.  */
#define F_SETLK64	F_SETLK	/* Set record locking info (non-blocking).  */
#define F_SETLKW64	F_SETLKW/* Set record locking info (blocking).	*/
#ifdef __USE_XOPEN2K8
# define F_DUPFD_CLOEXEC 12	/* Duplicate file descriptor with
				   close-on-exit set.  */
#endif

/* File descriptor flags used with F_GETFD and F_SETFD.  */
#define	FD_CLOEXEC	1	/* Close on exec.  */


#include <bits/types.h>

/* The structure describing an advisory lock.  This is the type of the third
   argument to `fcntl' for the F_GETLK, F_SETLK, and F_SETLKW requests.  */
struct flock
  {
    __off_t l_start;	/* Offset where the lock begins.  */
    __off_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
  };

#ifdef __USE_LARGEFILE64
/* Note this matches struct flock exactly.  */
struct flock64
  {
    __off_t l_start;	/* Offset where the lock begins.  */
    __off_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
    short int l_type;	/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    short int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
  };
#endif

/* Values for the `l_type' field of a `struct flock'.  */
#define	F_RDLCK	1	/* Read lock.  */
#define	F_WRLCK	2	/* Write lock.  */
#define	F_UNLCK	3	/* Remove lock.  */

/* Advise to `posix_fadvise'.  */
#ifdef __USE_XOPEN2K
# define POSIX_FADV_NORMAL	0 /* No further special treatment.  */
# define POSIX_FADV_RANDOM	1 /* Expect random page references.  */
# define POSIX_FADV_SEQUENTIAL	2 /* Expect sequential page references.  */
# define POSIX_FADV_WILLNEED	3 /* Will need these pages.  */
# define POSIX_FADV_DONTNEED	4 /* Don't need these pages.  */
# define POSIX_FADV_NOREUSE	5 /* Data will be accessed once.  */
#endif

/* O_*, F_*, FD_* bit values for GNU.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef _FCNTL_H
# error "Never use <bits/fcntl.h> directly; include <fcntl.h> instead."
#endif

#include <sys/types.h>

/* File access modes.  These are understood by io servers; they can be
   passed in `dir_lookup', and are returned by `io_get_openmodes'.
   Consequently they can be passed to `open', `hurd_file_name_lookup', and
   `file_name_lookup'; and are returned by `fcntl' with the F_GETFL
   command.  */

/* In GNU, read and write are bits (unlike BSD).  */
#ifdef __USE_GNU
# define O_READ		O_RDONLY /* Open for reading.  */
# define O_WRITE	O_WRONLY /* Open for writing.  */
# define O_EXEC		0x0004	/* Open for execution.  */
# define O_NORW		0	/* Open without R/W access.  */
#endif
/* POSIX.1 standard names.  */
#define	O_RDONLY	0x0001	/* Open read-only.  */
#define	O_WRONLY	0x0002	/* Open write-only.  */
#define	O_RDWR		(O_RDONLY|O_WRONLY) /* Open for reading and writing. */
#define	O_ACCMODE	O_RDWR	/* Mask for file access modes.  */

#define O_LARGEFILE	0


/* File name translation flags.  These are understood by io servers;
   they can be passed in `dir_lookup', and consequently to `open',
   `hurd_file_name_lookup', and `file_name_lookup'.  */

#define	O_CREAT		0x0010	/* Create file if it doesn't exist.  */
#define	O_EXCL		0x0020	/* Fail if file already exists.  */
#ifdef __USE_GNU
# define O_NOLINK	0x0040	/* No name mappings on final component.  */
# define O_NOTRANS	0x0080	/* No translator on final component. */
#endif

#ifdef __USE_XOPEN2K8
# define O_NOFOLLOW	0x00100000 /* Produce ENOENT if file is a symlink.  */
# define O_DIRECTORY	0x00200000 /* Produce ENOTDIR if not a directory.  */
#endif


/* I/O operating modes.  These are understood by io servers; they can be
   passed in `dir_lookup' and set or fetched with `io_*_openmodes'.
   Consequently they can be passed to `open', `hurd_file_name_lookup',
   `file_name_lookup', and `fcntl' with the F_SETFL command; and are
   returned by `fcntl' with the F_GETFL command.  */

#define	O_APPEND	0x0100	/* Writes always append to the file.  */
#define O_ASYNC		0x0200	/* Send SIGIO to owner when data is ready.  */
#define O_FSYNC		0x0400	/* Synchronous writes.  */
#define O_SYNC		O_FSYNC
#ifdef __USE_GNU
# define O_NOATIME	0x0800	/* Don't set access time on read (owner).  */
#endif
#ifdef	__USE_MISC
# define O_SHLOCK	0x00020000 /* Open with shared file lock.  */
# define O_EXLOCK	0x00040000 /* Open with exclusive file lock.  */
#endif

/* These are lesser flavors of partial synchronization that are
   implied by our one flag (O_FSYNC).  */
#if defined __USE_POSIX199309 || defined __USE_UNIX98
# define O_DSYNC	O_SYNC	/* Synchronize data.  */
# define O_RSYNC	O_SYNC	/* Synchronize read operations.	 */
#endif


/* The name O_NONBLOCK is unfortunately overloaded; it is both a file name
   translation flag and an I/O operating mode.  O_NDELAY is the deprecated
   BSD name for the same flag, overloaded in the same way.

   When used in `dir_lookup' (and consequently `open', `hurd_file_name_lookup',
   or `file_name_lookup'), O_NONBLOCK says the open should return immediately
   instead of blocking for any significant length of time (e.g., to wait
   for carrier detect on a serial line).  It is also saved as an I/O
   operating mode, and after open has the following meaning.

   When used in `io_*_openmodes' (and consequently `fcntl' with the F_SETFL
   command), the O_NONBLOCK flag means to do nonblocking i/o: any i/o
   operation that would block for any significant length of time will instead
   fail with EAGAIN.  */

#define	O_NONBLOCK	0x0008	/* Non-blocking open or non-blocking I/O.  */
#ifdef __USE_MISC
# define O_NDELAY	O_NONBLOCK /* Deprecated.  */
#endif


#ifdef __USE_GNU
/* Mask of bits which are understood by io servers.  */
# define O_HURD		(0xffff | O_EXLOCK | O_SHLOCK)
#endif


/* Open-time action flags.  These are understood by `hurd_file_name_lookup'
   and consequently by `open' and `file_name_lookup'.  They are not preserved
   once the file has been opened.  */

#define	O_TRUNC		0x00010000 /* Truncate file to zero length.  */
#ifdef __USE_XOPEN2K8
# define O_CLOEXEC	0x00400000 /* Set FD_CLOEXEC.  */
#endif


/* Controlling terminal flags.  These are understood only by `open',
   and are not preserved once the file has been opened.  */

#ifdef __USE_GNU
# define O_IGNORE_CTTY	0x00080000 /* Don't do any ctty magic at all.  */
#endif
/* `open' never assigns a controlling terminal in GNU.  */
#define	O_NOCTTY	0	/* Don't assign a controlling terminal.  */


#ifdef __USE_MISC
/* Flags for TIOCFLUSH.  */
# define FREAD		O_RDONLY
# define FWRITE		O_WRONLY

/* Traditional BSD names the O_* bits.  */
# define FASYNC		O_ASYNC
# define FCREAT		O_CREAT
# define FEXCL		O_EXCL
# define FTRUNC		O_TRUNC
# define FNOCTTY	O_NOCTTY
# define FFSYNC		O_FSYNC
# define FSYNC		O_SYNC
# define FAPPEND	O_APPEND
# define FNONBLOCK	O_NONBLOCK
# define FNDELAY	O_NDELAY
#endif


/* Values for the second argument to `fcntl'.  */
#define	F_DUPFD	  	0	/* Duplicate file descriptor.  */
#define	F_GETFD		1	/* Get file descriptor flags.  */
#define	F_SETFD		2	/* Set file descriptor flags.  */
#define	F_GETFL		3	/* Get file status flags.  */
#define	F_SETFL		4	/* Set file status flags.  */
#if defined __USE_UNIX98 || defined __USE_XOPEN2K8
# define F_GETOWN	5	/* Get owner (receiver of SIGIO).  */
# define F_SETOWN	6	/* Set owner (receiver of SIGIO).  */
#endif
#ifdef __USE_FILE_OFFSET64
# define	F_GETLK		F_GETLK64
# define	F_SETLK		F_SETLK64
# define	F_SETLKW	F_SETLKW64
#else
# define	F_GETLK		7	/* Get record locking info.  */
# define	F_SETLK		8	/* Set record locking info (non-blocking).  */
# define	F_SETLKW	9	/* Set record locking info (blocking).  */
#endif
#define	F_GETLK64	10	/* Get record locking info.  */
#define	F_SETLK64	11	/* Set record locking info (non-blocking).  */
#define	F_SETLKW64	12	/* Set record locking info (blocking).  */

#ifdef __USE_XOPEN2K8
# define F_DUPFD_CLOEXEC 1030	/* Duplicate, set FD_CLOEXEC on new one.  */
#endif


/* File descriptor flags used with F_GETFD and F_SETFD.  */
#define	FD_CLOEXEC	1	/* Close on exec.  */


#include <bits/types.h>

/* The structure describing an advisory lock.  This is the type of the third
   argument to `fcntl' for the F_GETLK, F_SETLK, and F_SETLKW requests.  */
struct flock
  {
    int l_type;		/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
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
    int l_type;		/* Type of lock: F_RDLCK, F_WRLCK, or F_UNLCK.  */
    int l_whence;	/* Where `l_start' is relative to (like `lseek').  */
    __off64_t l_start;	/* Offset where the lock begins.  */
    __off64_t l_len;	/* Size of the locked area; zero means until EOF.  */
    __pid_t l_pid;	/* Process holding the lock.  */
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

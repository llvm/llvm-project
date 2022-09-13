//===-- Definition of macros from fcntl.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H

// File creation flags
#define O_CLOEXEC 02000000
#define O_CREAT 00000100
#define O_PATH 010000000

#ifdef __aarch64__
#define O_DIRECTORY 040000
#else
#define O_DIRECTORY 00200000
#endif

#define O_EXCL 00000200
#define O_NOCTTY 00000400

#ifdef __aarch64__
#define O_NOFOLLOW 0100000
#else
#define O_NOFOLLOW 00400000
#endif

#define O_TRUNC 00001000
#define O_TMPFILE (020000000 | O_DIRECTORY)

// File status flags
#define O_APPEND 00002000
#define O_DSYNC 00010000
#define O_NONBLOCK 00004000
#define O_SYNC 04000000 | O_DSYNC

// File access mode mask
#define O_ACCMODE 00000003

// File access mode flags
#define O_RDONLY 00000000
#define O_RDWR 00000002
#define O_WRONLY 00000001

// File mode flags
#define S_IRWXU 0700
#define S_IRUSR 0400
#define S_IWUSR 0200
#define S_IXUSR 0100
#define S_IRWXG 070
#define S_IRGRP 040
#define S_IWGRP 020
#define S_IXGRP 010
#define S_IRWXO 07
#define S_IROTH 04
#define S_IWOTH 02
#define S_IXOTH 01
#define S_ISUID 04000
#define S_ISGID 02000

// Special directory FD to indicate that the path argument to
// openat is relative to the current directory.
#define AT_FDCWD -100

// Special flag to the function unlinkat to indicate that it
// has to perform the equivalent of "rmdir" on the path argument.
#define AT_REMOVEDIR 0x200

// Values of SYS_fcntl commands.
#define F_DUPFD 0
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4

#endif // __LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H

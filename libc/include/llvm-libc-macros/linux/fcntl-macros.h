//===-- Definition of macros from fcntl.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H

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

// Special directory FD to indicate that the path argument to
// openat is relative to the current directory.
#define AT_FDCWD -100

// Special flag to the function unlinkat to indicate that it
// has to perform the equivalent of "rmdir" on the path argument.
#define AT_REMOVEDIR 0x200

// Special flag for functions like lstat to convey that symlinks
// should not be followed.
#define AT_SYMLINK_NOFOLLOW 0x100

// Allow empty relative pathname.
#define AT_EMPTY_PATH 0x1000

// Values of SYS_fcntl commands.
#define F_DUPFD 0
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4

#endif // LLVM_LIBC_MACROS_LINUX_FCNTL_MACROS_H

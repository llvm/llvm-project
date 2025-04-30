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
   <https://www.gnu.org/licenses/>.  */

#ifndef	_DIRSTREAM_H
#define	_DIRSTREAM_H	1

#include <sys/types.h>

#include <libc-lock.h>

/* Directory stream type.

   The miscellaneous Unix `readdir' implementations read directory data
   into a buffer and return `struct dirent *' pointers into it.  */

struct __dirstream
  {
    int fd;			/* File descriptor.  */

    __libc_lock_define (, lock) /* Mutex lock for this structure.  */

    size_t allocation;		/* Space allocated for the block.  */
    size_t size;		/* Total valid data in the block.  */
    size_t offset;		/* Current offset into the block.  */

    off_t filepos;		/* Position of next entry to read.  */

    int errcode;		/* Delayed error code.  */

    /* Directory block.  We must make sure that this block starts
       at an address that is aligned adequately enough to store
       dirent entries.  Using the alignment of "void *" is not
       sufficient because dirents on 32-bit platforms can require
       64-bit alignment.  We use "long double" here to be consistent
       with what malloc uses.  */
    char data[0] __attribute__ ((aligned (__alignof__ (long double))));
  };

#define _DIR_dirfd(dirp)	((dirp)->fd)

#endif	/* dirstream.h */

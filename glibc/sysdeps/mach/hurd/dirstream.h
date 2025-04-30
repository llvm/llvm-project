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

#include <libc-lock.h>

/* Directory stream type.

   The Hurd directory format is the same as `struct dirent', so `readdir'
   returns a pointer into the buffer we read directory data into.  */

struct __dirstream
  {
    void *__fd;			/* `struct hurd_fd' pointer for descriptor.  */
    char *__data;		/* Directory block.  */
    int __entry_data;		/* Entry number `__data' corresponds to.  */
    char *__ptr;		/* Current pointer into the block.  */
    int __entry_ptr;		/* Entry number `__ptr' corresponds to.  */
    size_t __allocation;	/* Space allocated for the block.  */
    size_t __size;		/* Total valid data in the block.  */
    __libc_lock_define (, __lock) /* Mutex lock for this structure.  */
  };

#endif	/* dirstream.h */

/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef __OLD_DIRENT_H
#define __OLD_DIRENT_H 1

#include <dirent.h>

struct __old_dirent64
  {
    __ino_t d_ino;
    __off64_t d_off;
    unsigned short int d_reclen;
    unsigned char d_type;
    char d_name[256];		/* We must not include limits.h! */
  };

/* Now define the internal interfaces.  */
extern struct __old_dirent64 *__old_readdir64 (DIR *__dirp);
libc_hidden_proto (__old_readdir64);
extern int __old_readdir64_r (DIR *__dirp, struct __old_dirent64 *__entry,
			  struct __old_dirent64 **__result);
extern __ssize_t __old_getdents64 (int __fd, char *__buf, size_t __nbytes)
	attribute_hidden;
int __old_scandir64 (const char * __dir,
		     struct __old_dirent64 *** __namelist,
		     int (*__selector) (const struct __old_dirent64 *),
		     int (*__cmp) (const struct __old_dirent64 **,
				   const struct __old_dirent64 **));

#endif

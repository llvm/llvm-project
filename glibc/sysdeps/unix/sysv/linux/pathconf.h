/* Common parts of Linux implementation of pathconf and fpathconf.
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

#include <errno.h>
#include <unistd.h>
#include <sys/statfs.h>


/* Used like: return __statfs_link_max (__statfs (name, &buf), &buf,
					name, -1); */
extern long int __statfs_link_max (int result, const struct statfs *fsbuf,
				   const char *file, int fd)
     attribute_hidden;


/* Used like: return __statfs_filesize_max (__statfs (name, &buf), &buf); */
extern long int __statfs_filesize_max (int result, const struct statfs *fsbuf)
     attribute_hidden;


/* Used like: return __statfs_link_max (__statfs (name, &buf), &buf); */
extern long int __statfs_symlinks (int result, const struct statfs *fsbuf)
     attribute_hidden;


/* Used like: return __statfs_chown_restricted (__statfs (name, &buf), &buf);*/
extern long int __statfs_chown_restricted (int result,
					   const struct statfs *fsbuf)
     attribute_hidden;

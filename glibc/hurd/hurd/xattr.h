/* Access to extended attributes on files for GNU/Hurd.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#ifndef	_HURD_XATTR_H
#define	_HURD_XATTR_H	1

#include <sys/xattr.h>		/* This defines the XATTR_* flags.  */

/* These are the internal versions of getxattr/setxattr/listxattr.  */
extern error_t _hurd_xattr_get (io_t port, const char *name,
				void *value, size_t *size);
extern error_t _hurd_xattr_set (io_t port, const char *name,
				const void *value, size_t size, int flags);
extern error_t _hurd_xattr_remove (io_t port, const char *name);
extern error_t _hurd_xattr_list (io_t port, void *buffer, size_t *size);



#endif	/* hurd/xattr.h */

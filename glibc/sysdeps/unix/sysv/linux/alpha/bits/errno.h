/* Error constants.  Linux/Alpha specific version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _BITS_ERRNO_H
#define _BITS_ERRNO_H 1

#if !defined _ERRNO_H
# error "Never include <bits/errno.h> directly; use <errno.h> instead."
#endif

# include <linux/errno.h>

/* Older Linux headers do not define these constants.  */
# ifndef ENOTSUP
#  define ENOTSUP		EOPNOTSUPP
# endif

# ifndef ECANCELED
#  define ECANCELED		131
# endif

# ifndef EOWNERDEAD
#  define EOWNERDEAD		136
# endif

# ifndef ENOTRECOVERABLE
#  define ENOTRECOVERABLE	137
# endif

# ifndef ERFKILL
#  define ERFKILL		138
# endif

# ifndef EHWPOISON
#  define EHWPOISON		139
# endif

#endif /* bits/errno.h.  */

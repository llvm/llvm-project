/* Error constants.  Generic version.
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

/* This file defines the errno constants.  */

#ifndef _BITS_ERRNO_H
#define _BITS_ERRNO_H 1

#if !defined _ERRNO_H
# error "Never include <bits/errno.h> directly; use <errno.h> instead."
#endif

#error "Generic bits/errno.h included -- port is incomplete."

/* Authors of new ports of the GNU C Library must override this file
   with their own bits/errno.h in an appropriate subdirectory of
   sysdeps/.  Its function is to define all of the error constants
   from C2011 and POSIX.1-2008, with values appropriate to the
   operating system, and any additional OS-specific error constants.

   C2011 requires all error constants to be object-like macros that
   expand to "integer constant expressions with type int, positive
   values, and suitable for use in #if directives".  Moreover, all of
   their names must begin with a capital E, followed immediately by
   either another capital letter, or a digit.  It is OK to define
   macros that are not error constants, but only in the implementation
   namespace.

   errno.h is sometimes included from assembly language.  Therefore,
   when __ASSEMBLER__ is defined, bits/errno.h may only define macros;
   it may not make any other kind of C declaration or definition.
   Also, the error constants should, if at all possible, expand to
   simple decimal or hexadecimal numbers.  */

#endif /* bits/errno.h.  */

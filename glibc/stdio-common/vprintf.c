/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <stdarg.h>
#undef	__OPTIMIZE__	/* Avoid inline `vprintf' function.  */
#include <stdio.h>
#include <libioP.h>

#undef	vprintf

/* Write formatted output to stdout according to the
   format string FORMAT, using the argument list in ARG.  */
int
__vprintf (const char *format, va_list ap)
{
  return __vfprintf_internal (stdout, format, ap, 0);
}

ldbl_strong_alias (__vprintf, vprintf)

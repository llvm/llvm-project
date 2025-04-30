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

#include <libio/libioP.h>

extern int
__vfprintf (FILE *fp, const char *format, va_list ap)
{
  return __vfprintf_internal (fp, format, ap, 0);
}
ldbl_strong_alias (__vfprintf, _IO_vfprintf);
ldbl_strong_alias (__vfprintf, vfprintf);
ldbl_hidden_def (__vfprintf, vfprintf)

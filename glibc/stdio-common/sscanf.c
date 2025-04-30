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

/* This file defines one of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include <stdarg.h>
#include <libio/strfile.h>

/* Read formatted input from S, according to the format string FORMAT.  */

int
__sscanf (const char *s, const char *format, ...)
{
  va_list arg;
  int done;
  _IO_strfile sf;
  FILE *f = _IO_strfile_read (&sf, s);

  va_start (arg, format);
  done = __vfscanf_internal (f, format, arg, 0);
  va_end (arg);

  return done;
}
ldbl_strong_alias (__sscanf, sscanf)
ldbl_strong_alias (__sscanf, _IO_sscanf)

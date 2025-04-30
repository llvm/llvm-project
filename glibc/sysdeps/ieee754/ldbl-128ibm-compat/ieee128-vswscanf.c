/* Wrapper for vswscanf.  IEEE128 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <libioP.h>
#include <wchar.h>
#include <strfile.h>

extern int
___ieee128_vswscanf (const wchar_t *string, const wchar_t *format,
		     va_list ap)
{
  _IO_strfile sf;
  struct _IO_wide_data wd;
  FILE *fp = _IO_strfile_readw (&sf, &wd, string);
  return __vfwscanf_internal (fp, format, ap, SCANF_LDBL_USES_FLOAT128);
}
strong_alias (___ieee128_vswscanf, __vswscanfieee128)

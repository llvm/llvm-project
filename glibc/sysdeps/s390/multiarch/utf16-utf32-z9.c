/* Conversion between UTF-16 and UTF-32 BE/internal - multiarch s390 version.

   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <sysdeps/s390/utf16-utf32-z9.c>
#include <ifunc-resolve.h>

#undef FROM_LOOP
#define FROM_LOOP	__from_utf16_loop
#undef TO_LOOP
#define TO_LOOP		__to_utf16_loop

#define _SINGLE_NAME(NAME) NAME##_single
#define SINGLE_NAME(NAME) _SINGLE_NAME(NAME)
strong_alias (SINGLE_NAME (FROM_LOOP_DEFAULT), SINGLE_NAME (FROM_LOOP))
strong_alias (SINGLE_NAME (TO_LOOP_DEFAULT), SINGLE_NAME (TO_LOOP))

/* Generate ifunc'ed loop functions for FROM/TO_LOOP.  */
s390_libc_ifunc_expr (FROM_LOOP_DEFAULT, FROM_LOOP,
		      (HAVE_FROM_VX_CU && (hwcap & HWCAP_S390_VXE))
		      ? FROM_LOOP_VX_CU
		      : (HAVE_FROM_VX && (hwcap & HWCAP_S390_VX))
			? FROM_LOOP_VX
			: FROM_LOOP_DEFAULT);

s390_libc_ifunc_expr (TO_LOOP_DEFAULT, TO_LOOP,
		      (HAVE_TO_VX_CU && (hwcap & HWCAP_S390_VXE))
		      ? TO_LOOP_VX_CU
		      : (HAVE_TO_VX && (hwcap & HWCAP_S390_VX))
			? TO_LOOP_VX
			: TO_LOOP_DEFAULT);

#include <iconv/skeleton.c>

/* Optimized strcasestr implementation for POWER8.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#define STRCASESTR __strcasestr_power8

#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(name)

/* The following definitions are used in strcasestr optimization.  */

/* strlen is used to calculate len of r4.  */
#define STRLEN __strlen_power8
/* strnlen is used to check if len of r3 is more than r4.  */
#define STRNLEN __strnlen_power8
/* strchr is used to check if first char of r4 is present in r3.  */
#define STRCHR __strchr_power8

#include <sysdeps/powerpc/powerpc64/power8/strcasestr.S>

/* Multiple versions of stpcpy.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <ifunc-stpcpy.h>

#if HAVE_STPCPY_IFUNC
# define stpcpy __redirect_stpcpy
# define __stpcpy __redirect___stpcpy
/* Omit the stpcpy inline definitions because it would redefine stpcpy.  */
# define __NO_STRING_INLINES
# define NO_MEMPCPY_STPCPY_REDIRECT
# include <string.h>
# undef stpcpy
# undef __stpcpy
# include <ifunc-resolve.h>

# if HAVE_STPCPY_C
extern __typeof (__redirect_stpcpy) STPCPY_C attribute_hidden;
# endif

# if HAVE_STPCPY_Z13
extern __typeof (__redirect_stpcpy) STPCPY_Z13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect___stpcpy, __stpcpy,
		      (HAVE_STPCPY_Z13 && (hwcap & HWCAP_S390_VX))
		      ? STPCPY_Z13
		      : STPCPY_DEFAULT
		      )
weak_alias (__stpcpy, stpcpy)
#endif

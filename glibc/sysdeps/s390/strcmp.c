/* Multiple versions of strcmp.
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

#include <ifunc-strcmp.h>

#if HAVE_STRCMP_IFUNC
# define strcmp __redirect_strcmp
/* Omit the strcmp inline definitions because it would redefine strcmp.  */
# define __NO_STRING_INLINES
# include <string.h>
# include <ifunc-resolve.h>
# undef strcmp

# if HAVE_STRCMP_Z900_G5
extern __typeof (__redirect_strcmp) STRCMP_Z900_G5 attribute_hidden;
# endif

# if HAVE_STRCMP_Z13
extern __typeof (__redirect_strcmp) STRCMP_Z13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_strcmp, strcmp,
		      (HAVE_STRCMP_Z13 && (hwcap & HWCAP_S390_VX))
		      ? STRCMP_Z13
		      : STRCMP_DEFAULT
		      )
#endif

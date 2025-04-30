/* Multiple versions of wcsspn.
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

#include <ifunc-wcsspn.h>

#if HAVE_WCSSPN_IFUNC
# define wcsspn __redirect_wcsspn
# include <wchar.h>
# undef wcsspn
# include <ifunc-resolve.h>
# if HAVE_WCSSPN_C
extern __typeof (__redirect_wcsspn) WCSSPN_C attribute_hidden;
# endif

# if HAVE_WCSSPN_Z13
extern __typeof (__redirect_wcsspn) WCSSPN_Z13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_wcsspn, wcsspn,
		      (HAVE_WCSSPN_Z13 && (hwcap & HWCAP_S390_VX))
		      ? WCSSPN_Z13
		      : WCSSPN_DEFAULT
		      )
#endif

/* Multiple versions of wcschr.
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

#include <ifunc-wcschr.h>

#if HAVE_WCSCHR_IFUNC
# define wcschr __redirect_wcschr
# define __wcschr __redirect___wcschr
# include <wchar.h>
# undef wcschr
# undef __wcschr
# include <ifunc-resolve.h>

# if HAVE_WCSCHR_C
extern __typeof (__redirect___wcschr) WCSCHR_C attribute_hidden;
# endif

# if HAVE_WCSCHR_Z13
extern __typeof (__redirect___wcschr) WCSCHR_Z13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect___wcschr, __wcschr,
		      (HAVE_WCSCHR_Z13 && (hwcap & HWCAP_S390_VX))
		      ? WCSCHR_Z13
		      : WCSCHR_DEFAULT
		      )
weak_alias (__wcschr, wcschr)
#endif

/* Multiple versions of strstr.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <ifunc-strstr.h>

#if HAVE_STRSTR_IFUNC
# define strstr __redirect_strstr
# include <string.h>
# include <ifunc-resolve.h>
# undef strstr

# if HAVE_STRSTR_C
extern __typeof (__redirect_strstr) STRSTR_C attribute_hidden;
# endif

# if HAVE_STRSTR_Z13
extern __typeof (__redirect_strstr) STRSTR_Z13 attribute_hidden;
# endif

# if HAVE_STRSTR_ARCH13
extern __typeof (__redirect_strstr) STRSTR_ARCH13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_strstr, strstr,
		      (HAVE_STRSTR_ARCH13 && (hwcap & HWCAP_S390_VXRS_EXT2))
		      ? STRSTR_ARCH13
		      : (HAVE_STRSTR_Z13 && (hwcap & HWCAP_S390_VX))
		      ? STRSTR_Z13
		      : STRSTR_DEFAULT
		      )
#endif

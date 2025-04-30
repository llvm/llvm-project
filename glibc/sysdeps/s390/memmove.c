/* Multiple versions of memmove.
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

#include <ifunc-memcpy.h>

#if HAVE_MEMMOVE_IFUNC
/* If we don't use ifunc, an alias is defined for memmove
   in sysdeps/s390/memmove-c.c or sysdeps/s390/memcpy.S
   depending on the used default implementation.  */
# undef memmove
# define memmove __redirect_memmove
# include <string.h>
# include <ifunc-resolve.h>
# undef memmove

# if HAVE_MEMMOVE_C
extern __typeof (__redirect_memmove) MEMMOVE_C attribute_hidden;
# endif

# if HAVE_MEMMOVE_Z13
extern __typeof (__redirect_memmove) MEMMOVE_Z13 attribute_hidden;
# endif

# if HAVE_MEMMOVE_ARCH13
extern __typeof (__redirect_memmove) MEMMOVE_ARCH13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_memmove, memmove,
		      ({
			s390_libc_ifunc_expr_stfle_init ();
			(HAVE_MEMMOVE_ARCH13 && (hwcap & HWCAP_S390_VXRS_EXT2)
			 && S390_IS_ARCH13_MIE3 (stfle_bits))
			  ? MEMMOVE_ARCH13
			  : (HAVE_MEMMOVE_Z13 && (hwcap & HWCAP_S390_VX))
			  ? MEMMOVE_Z13
			  : MEMMOVE_DEFAULT;
		      })
		      )
#endif

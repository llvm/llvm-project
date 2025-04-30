/* Multiple versions of memcmp.
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

#include <ifunc-memcmp.h>
#if HAVE_MEMCMP_IFUNC
# define memcmp __redirect_memcmp
# include <string.h>
# undef memcmp
# include <ifunc-resolve.h>

# if HAVE_MEMCMP_Z900_G5
extern __typeof (__redirect_memcmp) MEMCMP_Z900_G5 attribute_hidden;
# endif

# if HAVE_MEMCMP_Z10
extern __typeof (__redirect_memcmp) MEMCMP_Z10 attribute_hidden;
# endif

# if HAVE_MEMCMP_Z196
extern __typeof (__redirect_memcmp) MEMCMP_Z196 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_memcmp, memcmp,
		      ({
			s390_libc_ifunc_expr_stfle_init ();
			(HAVE_MEMCMP_Z196 && S390_IS_Z196 (stfle_bits))
			  ? MEMCMP_Z196
			  : (HAVE_MEMCMP_Z10 && S390_IS_Z10 (stfle_bits))
			  ? MEMCMP_Z10
			  : MEMCMP_DEFAULT;
		      })
		      )
weak_alias (memcmp, bcmp);
#endif

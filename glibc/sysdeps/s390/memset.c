/* Multiple versions of memset.
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

#include <ifunc-memset.h>
#if HAVE_MEMSET_IFUNC
# define memset __redirect_memset
# include <string.h>
# undef memset
# include <ifunc-resolve.h>

# if HAVE_MEMSET_Z900_G5
extern __typeof (__redirect_memset) MEMSET_Z900_G5 attribute_hidden;
# endif

# if HAVE_MEMSET_Z10
extern __typeof (__redirect_memset) MEMSET_Z10 attribute_hidden;
# endif

# if HAVE_MEMSET_Z196
extern __typeof (__redirect_memset) MEMSET_Z196 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_memset, memset,
		      ({
			s390_libc_ifunc_expr_stfle_init ();
			(HAVE_MEMSET_Z196 && S390_IS_Z196 (stfle_bits))
			  ? MEMSET_Z196
			  : (HAVE_MEMSET_Z10 && S390_IS_Z10 (stfle_bits))
			  ? MEMSET_Z10
			  : MEMSET_DEFAULT;
		      })
		      )
#endif

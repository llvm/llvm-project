/* Multiple versions of mempcpy.
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

#if HAVE_MEMCPY_IFUNC
# define mempcpy __redirect_mempcpy
# define __mempcpy __redirect___mempcpy
# define __NO_STRING_INLINES
# define NO_MEMPCPY_STPCPY_REDIRECT
# include <string.h>
# undef mempcpy
# undef __mempcpy
# include <ifunc-resolve.h>

# if HAVE_MEMCPY_Z900_G5
extern __typeof (__redirect___mempcpy) MEMPCPY_Z900_G5 attribute_hidden;
# endif

# if HAVE_MEMCPY_Z10
extern __typeof (__redirect___mempcpy) MEMPCPY_Z10 attribute_hidden;
# endif

# if HAVE_MEMCPY_Z196
extern __typeof (__redirect___mempcpy) MEMPCPY_Z196 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect___mempcpy, __mempcpy,
		      ({
			s390_libc_ifunc_expr_stfle_init ();
			(HAVE_MEMCPY_Z196 && S390_IS_Z196 (stfle_bits))
			  ? MEMPCPY_Z196
			  : (HAVE_MEMCPY_Z10 && S390_IS_Z10 (stfle_bits))
			  ? MEMPCPY_Z10
			  : MEMPCPY_DEFAULT;
		      })
		      )
weak_alias (__mempcpy, mempcpy);
#endif

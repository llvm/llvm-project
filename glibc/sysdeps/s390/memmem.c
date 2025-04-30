/* Multiple versions of memmem.
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

#include <ifunc-memmem.h>

#if HAVE_MEMMEM_IFUNC
# define memmem __redirect_memmem
# define __memmem __redirect___memmem
# include <string.h>
# include <ifunc-resolve.h>
# undef memmem
# undef __memmem

# if HAVE_MEMMEM_C
extern __typeof (__redirect_memmem) MEMMEM_C attribute_hidden;
# endif

# if HAVE_MEMMEM_Z13
extern __typeof (__redirect_memmem) MEMMEM_Z13 attribute_hidden;
# endif

# if HAVE_MEMMEM_ARCH13
extern __typeof (__redirect_memmem) MEMMEM_ARCH13 attribute_hidden;
# endif

s390_libc_ifunc_expr (__redirect_memmem, __memmem,
		      (HAVE_MEMMEM_ARCH13 && (hwcap & HWCAP_S390_VXRS_EXT2))
		      ? MEMMEM_ARCH13
		      : (HAVE_MEMMEM_Z13 && (hwcap & HWCAP_S390_VX))
		      ? MEMMEM_Z13
		      : MEMMEM_DEFAULT
		      )
weak_alias (__memmem, memmem)
#endif

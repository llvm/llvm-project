/* Optimized bzero implementation for PowerPC32/PPC32.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Luis Machado <luisgpm@br.ibm.com>.
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

#include <sysdep.h>

/* memset ifunc selector is not built for static and memset@local
   for shared builds makes the linker point the call to the ifunc
   selector.  */
#ifdef SHARED
# define MEMSET __memset_ppc
#else
# define MEMSET memset
#endif

ENTRY (__bzero_ppc)
        mr      r5,r4
        li      r4,0
        b       MEMSET@local
END (__bzero_ppc)

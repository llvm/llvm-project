/* Machine-dependent definitions for profiling support.  Generic GCC 2 version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

/* GCC version 2 gives us a perfect magical function to get
   just the information we need:
     void *__builtin_return_address (unsigned int N)
   returns the return address of the frame N frames up.  */

/* Be warned that GCC cannot usefully compile __builtin_return_address(N)
   for N != 0 on all machines.  In this case, you may have to write
   your own version of _mcount().  */

#if __GNUC__ < 2
 #error "This file uses __builtin_return_address, a GCC 2 extension."
#endif

#include <sysdep.h>
/* The canonical name for the function is `_mcount' in both C and asm,
   but some old asm code might assume it's `mcount'.  */
void _mcount (void);
weak_alias (_mcount, mcount)

static void mcount_internal (u_long frompc, u_long selfpc);

#define _MCOUNT_DECL(frompc, selfpc) \
static inline void mcount_internal (u_long frompc, u_long selfpc)

#define MCOUNT \
void _mcount (void)							      \
{									      \
  mcount_internal ((u_long) RETURN_ADDRESS (1), (u_long) RETURN_ADDRESS (0)); \
}

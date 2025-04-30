/* sparc-specific implementation of profiling support.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>, 2008

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

/* We must not pollute the global namespace.  */
#define mcount_internal __mcount_internal

extern void mcount_internal (u_long frompc, u_long selfpc);

#define _MCOUNT_DECL(frompc, selfpc) \
void mcount_internal (u_long frompc, u_long selfpc)

/* Define MCOUNT as empty since we have the implementation in another
   file.  */
#define MCOUNT

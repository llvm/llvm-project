/* AArch64 definitions for profiling support.
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

/* Accept 'frompc' address as argument from the function that calls
   __mcount for profiling.  Use  __builtin_return_address (0)
   for the 'selfpc' address.  */

#include <sysdep.h>

static void mcount_internal (u_long frompc, u_long selfpc);

#define _MCOUNT_DECL(frompc, selfpc) \
static inline void mcount_internal (u_long frompc, u_long selfpc)

/* Note: strip_pac is needed for frompc because of gcc PR target/94791.  */
#define MCOUNT                                                    \
void __mcount (void *frompc)                                      \
{                                                                 \
  mcount_internal ((u_long) strip_pac (frompc), (u_long) RETURN_ADDRESS (0)); \
}

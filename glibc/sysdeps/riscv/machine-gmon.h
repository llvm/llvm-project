/* RISC-V definitions for profiling support.
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
   _mcount for profiling.  Use  __builtin_return_address (0)
   for the 'selfpc' address.  */

#include <sysdep.h>

static void mcount_internal (unsigned long int frompc,
			     unsigned long int selfpc);

#define _MCOUNT_DECL(frompc, selfpc) \
static inline void mcount_internal (unsigned long int frompc, \
unsigned long int selfpc)

#define MCOUNT								\
void _mcount (void *frompc)						\
{									\
  mcount_internal ((unsigned long int) frompc,				\
		   (unsigned long int) RETURN_ADDRESS (0));		\
}

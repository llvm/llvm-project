/* Machine-dependent definitions for profiling support.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

#define _MCOUNT_DECL(frompc, selfpc)					\
static void								\
__mcount_internal (unsigned long int frompc, unsigned long int selfpc)

/* This is very simple as gcc does all the heavy lifting at _mcount call site
    - sets up caller's blink in r0, so frompc is setup correctly
    - preserve argument registers for original call.  */

#define MCOUNT								\
void									\
_mcount (void *frompc)							\
{									\
  __mcount_internal ((unsigned long int) frompc,			\
		     (unsigned long int) __builtin_return_address (0));	\
}

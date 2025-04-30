/* i386-specific implementation of profiling support.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

/* We need a special version of the `mcount' function since for ix86 it
   must not clobber any register.  This has several reasons:
     - there is a bug in gcc as of version 2.7.2.2 which prohibits the
       use of profiling together with nested functions
     - the ELF `fixup' function uses GCC's regparm feature
     - some (future) systems might want to pass parameters in registers.  */

/* We must not pollute the global namespace.  */
#define mcount_internal __mcount_internal

extern void mcount_internal (u_long frompc, u_long selfpc)
  __attribute__ ((regparm (2))) attribute_hidden;

#define _MCOUNT_DECL(frompc, selfpc)                \
  __attribute__ ((regparm (2)))			    \
void mcount_internal (u_long frompc, u_long selfpc)


/* Define MCOUNT as empty since we have the implementation in another
   file.  */
#define MCOUNT

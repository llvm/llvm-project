/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#ifndef _TLS_H

/* Additional definitions for <tls.h> on i686 and up.  */


/* Macros to load from and store into segment registers.  We can use
   the 32-bit instructions.  */
#define TLS_GET_GS() \
  ({ int __seg; __asm ("movl %%gs, %0" : "=q" (__seg)); __seg; })
#define TLS_SET_GS(val) \
  __asm ("movl %0, %%gs" :: "q" (val))


/* Get the full set of definitions.  */
#include_next <tls.h>

#endif	/* tls.h */

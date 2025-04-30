/* Examine __jmp_buf for unwinding frames.  HPPA version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <setjmp.h>
#include <jmpbuf-offsets.h>

/* Test if longjmp to JMPBUF would unwind the frame containing a local
   variable at ADDRESS.  */
#define _JMPBUF_UNWINDS(_jmpbuf, _address, _demangle)			\
  ((void *) (_address)							\
   > (void *) _demangle ((((unsigned long *) _jmpbuf)[JB_SP])))

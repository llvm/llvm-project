/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

/* Default stack size.  */
#define ARCH_STACK_DEFAULT_SIZE	(2 * 1024 * 1024)

/* Minimum guard size.  */
#define ARCH_MIN_GUARD_SIZE 0

/* Required stack pointer alignment at beginning.  SSE requires 16
   bytes.  */
#define STACK_ALIGN		16

/* Minimal stack size after allocating thread descriptor and guard size.  */
#define MINIMAL_REST_STACK	2048

/* Alignment requirement for TCB.  */
#define TCB_ALIGNMENT		16


/* Location of current stack frame.

   __builtin_frame_address (0) returns the value of the hard frame
   pointer, which will point at the location of the saved PC on the
   stack.  Below this in memory is the remainder of the linkage info,
   occupying 12 bytes.  Therefore in order to address from
   CURRENT_STACK_FRAME using "struct layout", we need to have the macro
   return the hard FP minus 12.  Of course, this makes no sense
   without the obsolete APCS stack layout...  */
#define CURRENT_STACK_FRAME	(__builtin_frame_address (0) - 12)

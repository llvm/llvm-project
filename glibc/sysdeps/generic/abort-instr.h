/* Magic instruction to crash quickly and reliably.  Generic/stub version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _ABORT_INSTR_H
#define _ABORT_INSTR_H  1

/* If the compiler provides the generic way to generate the right
   instruction, we can use that without any machine-specific knowledge.  */
#if HAVE_BUILTIN_TRAP
# define ABORT_INSTRUCTION      __builtin_trap ()
#else
/* We cannot give any generic instruction to crash the program.
   abort will have to make sure it never returns.  */
#endif

#endif  /* abort-instr.h */

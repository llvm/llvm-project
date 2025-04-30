/* Crash the process immediately, without possibility of deadlock.  Generic.
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

#ifndef _SAFE_FATAL_H
#define _SAFE_FATAL_H   1

#include <abort-instr.h>

static inline void
__safe_fatal (void)
{
#ifdef ABORT_INSTRUCTION
  /* This is not guaranteed to be free from the possibility of deadlock,
     since it might generate a signal that can be caught.  But it's better
     than nothing.  */
  ABORT_INSTRUCTION;
#else
# error Need an OS-specific or machine-specific safe-fatal.h
#endif
}

#endif  /* safe-fatal.h */

/* Private macros for accessing __jmp_buf contents.  Nios II version.
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

/* Save offsets within __jmp_buf.  */
#define JB_R16 0
#define JB_R17 1
#define JB_R18 2
#define JB_R19 3
#define JB_R20 4
#define JB_R21 5
#define JB_R22 6
#define JB_FP  7
#define JB_RA  8
#define JB_SP  9

#ifndef  __ASSEMBLER__
#include <setjmp.h>
#include <stdint.h>
#include <sysdep.h>

static inline uintptr_t __attribute__ ((unused))
_jmpbuf_sp (__jmp_buf jmpbuf)
{
  uintptr_t sp = jmpbuf[JB_SP];
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (sp);
#endif
  return sp;
}
#endif

/* Helper for generic ____longjmp_chk(). */
#define JB_FRAME_ADDRESS(buf) ((void *) _jmpbuf_sp (buf))

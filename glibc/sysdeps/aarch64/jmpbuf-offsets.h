/* Copyright (C) 2006-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define JB_X19            0
#define JB_X20            1
#define JB_X21            2
#define JB_X22            3
#define JB_X23            4
#define JB_X24            5
#define JB_X25            6
#define JB_X26            7
#define JB_X27            8
#define JB_X28            9
#define JB_X29           10
#define JB_LR            11
#define JB_SP		 13

#define JB_D8		 14
#define JB_D9		 15
#define JB_D10		 16
#define JB_D11		 17
#define JB_D12		 18
#define JB_D13		 19
#define JB_D14		 20
#define JB_D15		 21

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
#define JB_FRAME_ADDRESS(buf) \
  ((void *) _jmpbuf_sp (buf))

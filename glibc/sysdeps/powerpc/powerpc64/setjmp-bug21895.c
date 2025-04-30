/* Shared object part of test for setjmp interoperability with static
   dlopen BZ #21895.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <setjmp.h>

/* Copy r1 adress to a local variable.  */
#define GET_STACK_POINTER(sp)	  \
  ({				  \
    asm volatile ("mr %0, 1\n\t"  \
		  : "=r" (sp));	  \
  })

jmp_buf jb;
void (*bar)(jmp_buf, unsigned long);

void
lbar (unsigned long sp)
{
  bar(jb, sp);
  for(;;);
}

void
foo (void)
{
  unsigned long sp;
  /* Copy r1 (stack pointer) to sp. It will be use later to get
     TOC area.  */
  GET_STACK_POINTER(sp);
  setjmp(jb);
  lbar(sp);

  for(;;);
}

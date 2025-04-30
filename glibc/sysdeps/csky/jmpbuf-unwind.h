/* Examine __jmp_buf for unwinding frames.  C-SkY version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <unwind.h>
#include <sysdep.h>

/* Test if longjmp to JMPBUF would unwind the frame
   containing a local variable at ADDRESS.  */
#define _JMPBUF_UNWINDS(jmpbuf, address, demangle)		\
  ((void *) (address) < (void *) demangle ((jmpbuf)[0].__sp))

#define _JMPBUF_CFA_UNWINDS_ADJ(_jmpbuf, _context, _adj)		      \
  _JMPBUF_UNWINDS_ADJ (_jmpbuf,						      \
		       (void *) (_Unwind_Ptr) _Unwind_GetCFA (_context),      \
		       _adj)

static inline uintptr_t __attribute__ ((unused))
_jmpbuf_sp (__jmp_buf regs)
{
  uintptr_t sp = (uintptr_t) regs[0].__sp;
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (sp);
#endif
  return sp;
}

#define _JMPBUF_UNWINDS_ADJ(_jmpbuf, _address, _adj) \
  ((uintptr_t) (_address) - (_adj) < _jmpbuf_sp (_jmpbuf) - (_adj))

/* We use the normal longjmp for unwinding.  */
#define __libc_unwind_longjmp(buf, val) __libc_longjmp (buf, val)

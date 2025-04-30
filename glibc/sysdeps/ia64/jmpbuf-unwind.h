/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <setjmp.h>
#include <stdint.h>
#include <unwind.h>

/* Test if longjmp to JMPBUF would unwind the frame containing a local
   variable at ADDRESS.  */
#define _JMPBUF_UNWINDS(_jmpbuf, _address, _demangle) \
  ((void *) (_address) < (void *) (((long int *) _jmpbuf)[0]))

#define _JMPBUF_CFA_UNWINDS_ADJ(_jmpbuf, _context, _adj) \
  ({ uintptr_t _cfa = (uintptr_t) _Unwind_GetCFA (_context) - (_adj);	\
     (_cfa < (uintptr_t)(((long *)(_jmpbuf))[0]) - (_adj)		\
      || (_cfa == (uintptr_t)(((long *)(_jmpbuf))[0]) - (_adj)		\
	  && (uintptr_t) _Unwind_GetBSP (_context) - (_adj)		\
	     >= (uintptr_t)(((long *)(_jmpbuf))[17]) - (_adj)));	\
  })

#define _JMPBUF_UNWINDS_ADJ(_jmpbuf, _address, _adj) \
  ((uintptr_t)(_address) - (_adj) < (uintptr_t)(((long *)_jmpbuf)[0]) - (_adj))

/* We use a longjmp() which can cross from the alternate signal-stack
   to the normal stack.  */
extern void __libc_unwind_longjmp (sigjmp_buf env, int val)
          __attribute__ ((noreturn));
hidden_proto (__libc_unwind_longjmp)

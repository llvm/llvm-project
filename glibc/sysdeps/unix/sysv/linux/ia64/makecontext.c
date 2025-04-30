/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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

#include <libintl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>
#include <sys/rse.h>
#include <link.h>
#include <dl-fptr.h>


#define PUSH(val)				\
do {						\
  if (ia64_rse_is_rnat_slot (rbs))		\
    *rbs++ = 0;					\
  *rbs++ = (val);				\
} while (0)


/* This implementation can handle an ARGC value of at most 8 and
   values can be passed only in integer registers (r32-r39).  */

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  mcontext_t *sc = &ucp->uc_mcontext;
  extern void __start_context (ucontext_t *link, long gp, ...);
  unsigned long stack_start, stack_end;
  va_list ap;
  unsigned long *rbs;
  int i;

  stack_start = (long) sc->sc_stack.ss_sp;
  stack_end = (long) sc->sc_stack.ss_sp + sc->sc_stack.ss_size;

  stack_start = (stack_start + 7) & -8;
  stack_end = stack_end & -16;

  if (argc > 8)
    {
      fprintf (stderr, _("\
makecontext: does not know how to handle more than 8 arguments\n"));
      exit (-1);
    }

  /* set the entry point and global pointer: */
  sc->sc_br[0] = ELF_PTR_TO_FDESC (&__start_context)->ip;
  sc->sc_br[1] = ELF_PTR_TO_FDESC (func)->ip;
  sc->sc_gr[1] = ELF_PTR_TO_FDESC (func)->gp;

  /* set up the call frame: */
  sc->sc_ar_pfs = ((sc->sc_ar_pfs & ~0x3fffffffffUL)
		   | (argc + 2) | ((argc + 2) << 7));
  rbs = (unsigned long *) stack_start;
  PUSH((long) ucp->uc_link);
  PUSH(ELF_PTR_TO_FDESC (&__start_context)->gp);
  va_start (ap, argc);
  for (i = 0; i < argc; ++i)
    PUSH(va_arg (ap, long));
  va_end (ap);

  /* set the memory and register stack pointers: */
  sc->sc_ar_bsp = (long) rbs;
  sc->sc_gr[12] = stack_end - 16;

  /* clear the NaT bits for r1 and r12: */
  sc->sc_nat &= ~((1 << 1) | (1 << 12));
  sc->sc_ar_rnat = 0;
}

weak_alias (__makecontext, makecontext)

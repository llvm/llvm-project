/* Create new context.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Helge Deller <deller@gmx.de>, 2008.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <libintl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sysdep.h>
#include <ucontext.h>

/* POSIX only supports integer arguments.  */

/* Stack must be 64-byte aligned at all times.  */
#define STACK_ALIGN 64
/* Size of frame marker in unsigned long words.  */
#define FRAME_SIZE_UL 8
/* Size of frame marker in bytes.  */
#define FRAME_SIZE_BYTES (8 * sizeof (unsigned long))
/* Size of X arguments in bytes.  */
#define ARGS(x) (x * sizeof (unsigned long))

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  unsigned long *sp, *osp;
  va_list ap;
  int i;

  /* Create a 64-byte aligned frame to store args. Use ss_sp if
     it is available, otherwise be robust and use the currently
     saved stack pointer.  */
  if (ucp->uc_stack.ss_sp && ucp->uc_stack.ss_size)
    osp = (unsigned long *)ucp->uc_stack.ss_sp;
  else
    osp = (unsigned long *)ucp->uc_mcontext.sc_gr[30];

  sp = (unsigned long *)((((unsigned long) osp)
			   + FRAME_SIZE_BYTES + ARGS(argc) + STACK_ALIGN)
			 & ~(STACK_ALIGN - 1));

  /* Use new frame.  */
  ucp->uc_mcontext.sc_gr[30] = ((unsigned long) sp);

  /* Finish frame setup.  */
  if (ucp->uc_link)
    {
      /* Returning to the next context and next frame.  */
      sp[-4 / sizeof (unsigned long)] = ucp->uc_link->uc_mcontext.sc_gr[30];
      sp[-20 / sizeof (unsigned long)] = ucp->uc_link->uc_mcontext.sc_gr[2];
    }
  else
    {
      /* This is the main context. No frame marker, and no return address.  */
      sp[-4 / sizeof (unsigned long)] = 0x0;
      sp[-20 / sizeof (unsigned long)] = 0x0;
    }

  /* Store address to jump to.  */
  ucp->uc_mcontext.sc_gr[2] = (unsigned long) func;

  /* Process arguments.  */
  va_start (ap, argc);
  for (i = 0; i < argc; ++i)
    {
      if (i < 4)
	{
	  ucp->uc_mcontext.sc_gr[26-i] = va_arg (ap, int);
	  continue;
	}

      if ((i < 8) && (sizeof (unsigned long) == 8))
	{
	  /* 64bit: r19-r22 are arg7-arg4.  */
	  ucp->uc_mcontext.sc_gr[22+4-i] = va_arg (ap, int);
	  continue;
	}

      /* All other arguments go on the stack.  */
      sp[-1 * (FRAME_SIZE_UL + 1 + i)] = va_arg (ap, int);
    }
  va_end (ap);
}
weak_alias(__makecontext, makecontext)

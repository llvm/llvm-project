/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

/* The public __longjmp() implementation is limited to jumping within
   the same stack.  That is, in general it is not possible to use this
   __longjmp() implementation to cross from one stack to another.
   In contrast, the __sigstack_longjmp() implemented here allows
   crossing from the alternate signal stack to the normal stack
   as a special case.  */

#include <assert.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>

#include <sysdep.h>
#include <sys/rse.h>

#define JB_SP	0
#define JB_BSP	17

struct rbs_flush_values
  {
    unsigned long bsp;
    unsigned long rsc;
    unsigned long rnat;
  };

extern struct rbs_flush_values __ia64_flush_rbs (void);
extern void __ia64_longjmp (__jmp_buf buf, int val, long rnat, long rsc)
     __attribute__ ((__noreturn__));

static void
copy_rbs (unsigned long *dst, unsigned long *dst_end, unsigned long dst_rnat,
	  unsigned long *src, unsigned long *src_end,
	  unsigned long current_rnat)
{
  unsigned long dst_slot, src_rnat = 0, src_slot, *src_rnat_addr, nat_bit;
  int first_time = 1;

  while (dst < dst_end)
    {
      dst_slot = ia64_rse_slot_num (dst);
      if (dst_slot == 63)
	{
	  *dst++ = dst_rnat;
	  dst_rnat = 0;
	}
      else
	{
	  /* read source value, including NaT bit: */
	  src_slot = ia64_rse_slot_num (src);
	  if (src_slot == 63)
	    {
	      /* skip src RNaT slot */
	      ++src;
	      src_slot = 0;
	    }
	  if (first_time || src_slot == 0)
	    {
	      first_time = 0;
	      src_rnat_addr = ia64_rse_rnat_addr (src);
	      if (src_rnat_addr < src_end)
		src_rnat = *src_rnat_addr;
	      else
		src_rnat = current_rnat;
	    }
	  nat_bit = (src_rnat >> src_slot) & 1;

	  assert (src < src_end);

	  *dst++ = *src++;
	  if (nat_bit)
	    dst_rnat |=  (1UL << dst_slot);
	  else
	    dst_rnat &= ~(1UL << dst_slot);
	}
    }
  dst_slot = ia64_rse_slot_num (dst);
  if (dst_slot > 0)
    *ia64_rse_rnat_addr (dst) = dst_rnat;
}

void
__sigstack_longjmp (__jmp_buf buf, int val)
{
  unsigned long *rbs_base, *bsp, *bspstore, *jb_bsp, jb_sp, ss_sp;
  unsigned long ndirty, rnat, load_rnat, *jb_rnat_addr;
  struct sigcontext *sc;
  stack_t stk;
  struct rbs_flush_values c;

  /* put RSE into enforced-lazy mode and return current bsp/rsc/rnat: */
  c = __ia64_flush_rbs ();

  jb_sp  = ((unsigned long *)  buf)[JB_SP];
  jb_bsp = ((unsigned long **) buf)[JB_BSP];

  INTERNAL_SYSCALL_CALL (sigaltstack, NULL, &stk);

  ss_sp = (unsigned long) stk.ss_sp;
  jb_rnat_addr = ia64_rse_rnat_addr (jb_bsp);

  if ((stk.ss_flags & SS_ONSTACK) == 0 || jb_sp - ss_sp < stk.ss_size)
    /* Normal non-stack-crossing longjmp; if the RNaT slot for the bsp
       saved in the jump-buffer is the same as the one for the current
       BSP, use the current AR.RNAT value, otherwise, load it from the
       jump-buffer's RNaT-slot.  */
    load_rnat = (ia64_rse_rnat_addr ((unsigned long *) c.bsp) != jb_rnat_addr);
  else
    {
      /* If we are on the alternate signal-stack and the jump-buffer
	 lies outside the signal-stack, we may need to copy back the
	 dirty partition which was torn off and saved on the
	 signal-stack when the signal was delivered.

	 Caveat: we assume that the top of the alternate signal-stack
		 stores the sigcontext structure of the signal that
		 caused the switch to the signal-stack.	 This should
		 be a fairly safe assumption but the kernel _could_
		 do things differently.. */
      sc = ((struct sigcontext *) ((ss_sp + stk.ss_size) & -16) - 1);

      /* As a sanity-check, verify that the register-backing-store base
	 of the alternate signal-stack is where we expect it.  */
      rbs_base = (unsigned long *)
	((ss_sp + sizeof (long) - 1) & -sizeof (long));

      assert ((unsigned long) rbs_base == sc->sc_rbs_base);

      ndirty = ia64_rse_num_regs (rbs_base, rbs_base + (sc->sc_loadrs >> 19));
      bsp = (unsigned long *) sc->sc_ar_bsp;
      bspstore = ia64_rse_skip_regs (bsp, -ndirty);

      if (bspstore < jb_bsp)
	/* AR.BSPSTORE at the time of the signal was below the value
	   of AR.BSP saved in the jump-buffer => copy the missing
	   portion from the torn off dirty partition which got saved
	   on the alternate signal-stack.  */
	copy_rbs (bspstore, jb_bsp, sc->sc_ar_rnat,
		  rbs_base, (unsigned long *) c.bsp, c.rnat);

      load_rnat = 1;
    }
  if (load_rnat)
    rnat = *jb_rnat_addr;
  else
    rnat = c.rnat;
  __ia64_longjmp (buf, val, rnat, c.rsc);
}

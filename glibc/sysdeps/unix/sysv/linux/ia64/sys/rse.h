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

#ifndef _ia64_rse_h
#define _ia64_rse_h

#include <features.h>

/* Register stack engine related helper functions.  This file may be
   used in applications, so be careful about the name-space and give
   some consideration to non-GNU C compilers (though __inline is
   fine). */

static __inline unsigned long
ia64_rse_slot_num (unsigned long *addr)
{
	return (((unsigned long) addr) >> 3) & 0x3f;
}

/* Return TRUE if ADDR is the address of an RNAT slot.  */

static __inline unsigned long
ia64_rse_is_rnat_slot (unsigned long *addr)
{
	return ia64_rse_slot_num (addr) == 0x3f;
}

/* Returns the address of the RNAT slot that covers the slot at
   address SLOT_ADDR.  */

static __inline unsigned long *
ia64_rse_rnat_addr (unsigned long *slot_addr)
{
	return (unsigned long *) ((unsigned long) slot_addr | (0x3f << 3));
}

/* Calcuate the number of registers in the dirty partition starting at
   BSPSTORE with a size of DIRTY bytes.  This isn't simply DIRTY
   divided by eight because the 64th slot is used to store ar.rnat.  */

static __inline unsigned long
ia64_rse_num_regs (unsigned long *bspstore, unsigned long *bsp)
{
	unsigned long slots = (bsp - bspstore);

	return slots - (ia64_rse_slot_num(bspstore) + slots)/0x40;
}

/* The inverse of the above: given bspstore and the number of
   registers, calculate ar.bsp.  */

static __inline unsigned long *
ia64_rse_skip_regs (unsigned long *addr, long num_regs)
{
	long delta = ia64_rse_slot_num(addr) + num_regs;

	if (num_regs < 0)
		delta -= 0x3e;
	return addr + num_regs + delta/0x3f;
}

#endif /* _ia64_rse_h */

/* PLT fixups.  Sparc 32-bit version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _DL_PLT_H
#define _DL_PLT_H

/* Some SPARC opcodes we need to use for self-modifying code.  */
#define OPCODE_NOP	0x01000000 /* nop */
#define OPCODE_CALL	0x40000000 /* call ?; add PC-rel word address */
#define OPCODE_SETHI_G1	0x03000000 /* sethi ?, %g1; add value>>10 */
#define OPCODE_JMP_G1	0x81c06000 /* jmp %g1+?; add lo 10 bits of value */
#define OPCODE_SAVE_SP	0x9de3bfa8 /* save %sp, -(16+6)*4, %sp */
#define OPCODE_BA	0x30800000 /* b,a ?; add PC-rel word address */
#define OPCODE_BA_PT	0x30480000 /* ba,a,pt %icc, ?; add PC-rel word address */

static inline __attribute__ ((always_inline)) Elf32_Addr
sparc_fixup_plt (const Elf32_Rela *reloc, Elf32_Addr *reloc_addr,
		 Elf32_Addr value, int t, int do_flush)
{
  Elf32_Sword disp;

  /* 't' is '0' if we are resolving this PLT entry for RTLD bootstrap,
     in which case we'll be resolving all PLT entries and thus can
     optimize by overwriting instructions starting at the first PLT entry
     instruction and we need not be mindful of thread safety.

     Otherwise, 't' is '1'.  */
  reloc_addr += t;
  disp = value - (Elf32_Addr) reloc_addr;

  if (disp >= -0x800000 && disp < 0x800000)
    {
      unsigned int insn = OPCODE_BA | ((disp >> 2) & 0x3fffff);

#ifdef __sparc_v9__
      /* On V9 we can do even better by using a branch with
	 prediction if we fit into the even smaller 19-bit
	 displacement field.  */
      if (disp >= -0x100000 && disp < 0x100000)
	insn = OPCODE_BA_PT | ((disp >> 2) & 0x07ffff);
#endif

      /* Even if we are writing just a single branch, we must not
	 ignore the 't' offset.  Consider a case where we have some
	 PLT slots which can be optimized into a single branch and
	 some which cannot.  Then we can end up with a PLT which looks
	 like:

		PLT4.0: sethi	%(PLT_4_INDEX), %g1
			sethi	%(fully_resolved_sym_4), %g1
			jmp	%g1 + %lo(fully_resolved_sym_4)
		PLT5.0:	ba,a	fully_resolved_sym_5
			ba,a	PLT0.0
			...

	  The delay slot of that jmp must always be either a sethi to
	  %g1 or a nop.  But if we try to place this displacement
	  branch there, PLT4.0 will jump to fully_resolved_sym_4 for 1
	  instruction and then go immediately to
	  fully_resolved_sym_5.  */

      reloc_addr[0] = insn;
      if (do_flush)
	__asm __volatile ("flush %0" : : "r"(reloc_addr));
    }
  else
    {
      /* For thread safety, write the instructions from the bottom and
	 flush before we overwrite the critical "b,a".  This of course
	 need not be done during bootstrapping, since there are no threads.
	 But we also can't tell if we _can_ use flush, so don't. */

      reloc_addr[1] = OPCODE_JMP_G1 | (value & 0x3ff);
      if (do_flush)
	__asm __volatile ("flush %0+4" : : "r"(reloc_addr));

      reloc_addr[0] = OPCODE_SETHI_G1 | (value >> 10);
      if (do_flush)
	__asm __volatile ("flush %0" : : "r"(reloc_addr));
    }

  return value;
}

#endif /* dl-plt.h */

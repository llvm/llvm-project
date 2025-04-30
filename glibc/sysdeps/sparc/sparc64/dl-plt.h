/* PLT fixups.  Sparc 64-bit version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* We have 4 cases to handle.  And we code different code sequences
   for each one.  I love V9 code models...  */
static inline void __attribute__ ((always_inline))
sparc64_fixup_plt (struct link_map *map, const Elf64_Rela *reloc,
		   Elf64_Addr *reloc_addr, Elf64_Addr value,
		   Elf64_Addr high, int t)
{
  unsigned int *insns = (unsigned int *) reloc_addr;
  Elf64_Addr plt_vaddr = (Elf64_Addr) reloc_addr;
  Elf64_Sxword disp = value - plt_vaddr;

  /* 't' is '0' if we are resolving this PLT entry for RTLD bootstrap,
     in which case we'll be resolving all PLT entries and thus can
     optimize by overwriting instructions starting at the first PLT entry
     instruction and we need not be mindful of thread safety.

     Otherwise, 't' is '1'.

     Now move plt_vaddr up to the call instruction.  */
  plt_vaddr += ((t + 1) * 4);

  /* PLT entries .PLT32768 and above look always the same.  */
  if (__builtin_expect (high, 0) != 0)
    {
      *reloc_addr = value - map->l_addr;
    }
  /* Near destination.  */
  else if (disp >= -0x800000 && disp < 0x800000)
    {
      unsigned int insn;

      /* ba,a */
      insn = 0x30800000 | ((disp >> 2) & 0x3fffff);

      if (disp >= -0x100000 && disp < 0x100000)
	{
	  /* ba,a,pt %icc */
	  insn = 0x30480000  | ((disp >> 2) & 0x07ffff);
	}

      /* As this is just one instruction, it is thread safe and so we
	 can avoid the unnecessary sethi FOO, %g1.  Each 64-bit PLT
	 entry is 8 instructions long, so we can't run into the 'jmp'
	 delay slot problems 32-bit PLTs can.  */
      insns[0] = insn;
      __asm __volatile ("flush %0" : : "r" (insns));
    }
  /* 32-bit Sparc style, the target is in the lower 32-bits of
     address space.  */
  else if (insns += t, (value >> 32) == 0)
    {
      /* sethi	%hi(target), %g1
	 jmpl	%g1 + %lo(target), %g0  */

      insns[1] = 0x81c06000 | (value & 0x3ff);
      __asm __volatile ("flush %0 + 4" : : "r" (insns));

      insns[0] = 0x03000000 | ((unsigned int)(value >> 10));
      __asm __volatile ("flush %0" : : "r" (insns));
    }
  /* We can also get somewhat simple sequences if the distance between
     the target and the PLT entry is within +/- 2GB.  */
  else if ((plt_vaddr > value
	    && ((plt_vaddr - value) >> 31) == 0)
	   || (value > plt_vaddr
	       && ((value - plt_vaddr) >> 31) == 0))
    {
      unsigned int displacement;

      if (plt_vaddr > value)
	displacement = (0 - (plt_vaddr - value));
      else
	displacement = value - plt_vaddr;

      /* mov	%o7, %g1
	 call	displacement
	  mov	%g1, %o7  */

      insns[2] = 0x9e100001;
      __asm __volatile ("flush %0 + 8" : : "r" (insns));

      insns[1] = 0x40000000 | (displacement >> 2);
      __asm __volatile ("flush %0 + 4" : : "r" (insns));

      insns[0] = 0x8210000f;
      __asm __volatile ("flush %0" : : "r" (insns));
    }
  /* Worst case, ho hum...  */
  else
    {
      unsigned int high32 = (value >> 32);
      unsigned int low32 = (unsigned int) value;

      /* ??? Some tricks can be stolen from the sparc64 egcs backend
	     constant formation code I wrote.  -DaveM  */

      if (__glibc_unlikely (high32 & 0x3ff))
	{
	  /* sethi	%hh(value), %g1
	     sethi	%lm(value), %g5
	     or		%g1, %hm(value), %g1
	     or		%g5, %lo(value), %g5
	     sllx	%g1, 32, %g1
	     jmpl	%g1 + %g5, %g0
	      nop  */

	  insns[5] = 0x81c04005;
	  __asm __volatile ("flush %0 + 20" : : "r" (insns));

	  insns[4] = 0x83287020;
	  __asm __volatile ("flush %0 + 16" : : "r" (insns));

	  insns[3] = 0x8a116000 | (low32 & 0x3ff);
	  __asm __volatile ("flush %0 + 12" : : "r" (insns));

	  insns[2] = 0x82106000 | (high32 & 0x3ff);
	}
      else
	{
	  /* sethi	%hh(value), %g1
	     sethi	%lm(value), %g5
	     sllx	%g1, 32, %g1
	     or		%g5, %lo(value), %g5
	     jmpl	%g1 + %g5, %g0
	      nop  */

	  insns[4] = 0x81c04005;
	  __asm __volatile ("flush %0 + 16" : : "r" (insns));

	  insns[3] = 0x8a116000 | (low32 & 0x3ff);
	  __asm __volatile ("flush %0 + 12" : : "r" (insns));

	  insns[2] = 0x83287020;
	}

      __asm __volatile ("flush %0 + 8" : : "r" (insns));

      insns[1] = 0x0b000000 | (low32 >> 10);
      __asm __volatile ("flush %0 + 4" : : "r" (insns));

      insns[0] = 0x03000000 | (high32 >> 10);
      __asm __volatile ("flush %0" : : "r" (insns));
    }
}

#endif /* dl-plt.h */

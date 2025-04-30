/* PLT trampolines.  ia64 version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#undef ret

/*
   This code is used in dl-runtime.c to call the `_dl_fixup' function
   and then redirect to the address it returns. `_dl_fixup()' takes two
   arguments, however _dl_profile_fixup() takes five.

   The ABI specifies that we will never see more than 8 input
   registers to a function call, thus it is safe to simply allocate
   those, and simpler than playing stack games.  */

/* Used to save and restore 8 incoming fp registers */
#define RESOLVE_FRAME_SIZE (16*8)

ENTRY(_dl_runtime_resolve)
	{ .mmi
	  .prologue
	  .save ar.pfs, r40
	  alloc loc0 = ar.pfs, 8, 6, 2, 0
	  /* Use the 16 byte scratch area. r2 will start at f8 and
	     r3 will start at f9.  */
	  adds r2 = -(RESOLVE_FRAME_SIZE - 16), r12
	  adds r3 = -(RESOLVE_FRAME_SIZE - 32), r12
	}
	{ .mii
	  .fframe RESOLVE_FRAME_SIZE
	  adds r12 = -RESOLVE_FRAME_SIZE, r12
	  .save rp, loc1
	  mov loc1 = b0
	  .body
	  mov loc2 = r8		/* preserve struct value register */
	  ;;
	}
	{ .mii
	  mov loc3 = r9		/* preserve language specific register */
	  mov loc4 = r10	/* preserve language specific register */
	  mov loc5 = r11	/* preserve language specific register */
	}
	{ .mmi
	  stf.spill [r2] = f8, 32
	  stf.spill [r3] = f9, 32
	  mov out0 = r16
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f10, 32
	  stf.spill [r3] = f11, 32
	  shl out1 = r15, 4
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f12, 32
	  stf.spill [r3] = f13, 32
	  /* Relocation record is 24 byte. */
	  shladd out1 = r15, 3, out1
	  ;;
	}
	{ .mmb
	  stf.spill [r2] = f14
	  stf.spill [r3] = f15
	  br.call.sptk.many b0 = _dl_fixup
	}
	{ .mii
	  /* Skip the 16byte scratch area.  */
	  adds r2 = 16, r12
	  adds r3 = 32, r12
	  mov b6 = ret0
	  ;;
	}
	{ .mmi
	  ldf.fill f8 = [r2], 32
	  ldf.fill f9 = [r3], 32
	  mov b0 = loc1
	  ;;
	}
	{ .mmi
	  ldf.fill f10 = [r2], 32
	  ldf.fill f11 = [r3], 32
	  mov gp = ret1
	  ;;
	}
	{ .mmi
	  ldf.fill f12 = [r2], 32
	  ldf.fill f13 = [r3], 32
	  mov ar.pfs = loc0
	  ;;
	}
	{ .mmi
	  ldf.fill f14 = [r2], 32
	  ldf.fill f15 = [r3], 32
	  .restore sp		/* pop the unwind frame state */
	  adds r12 = RESOLVE_FRAME_SIZE, r12
	  ;;
	}
	{ .mii
	  mov r9 = loc3		/* restore language specific register */
	  mov r10 = loc4	/* restore language specific register */
	  mov r11 = loc5	/* restore language specific register */
	}
	{ .mii
	  mov r8 = loc2		/* restore struct value register */
	  ;;
	}
	/* An alloc is needed for the break system call to work.
	   We don't care about the old value of the pfs register.  */
	{ .mmb
	  .prologue
	  .body
	  alloc r2 = ar.pfs, 0, 0, 8, 0
	  br.sptk.many b6
	  ;;
	}
END(_dl_runtime_resolve)


/* The fourth argument to _dl_profile_fixup and the third one to
   _dl_call_pltexit are a pointer to La_ia64_regs:

   8byte r8
   8byte r9
   8byte r10
   8byte r11
   8byte in0
   8byte in1
   8byte in2
   8byte in3
   8byte in4
   8byte in5
   8byte in6
   8byte in7
   16byte f8
   16byte f9
   16byte f10
   16byte f11
   16byte f12
   16byte f13
   16byte f14
   16byte f15
   8byte ar.unat
   8byte sp

   The fifth argument to _dl_profile_fixup is a pointer to long int.
   The fourth argument to _dl_call_pltexit is a pointer to
   La_ia64_retval:

   8byte r8
   8byte r9
   8byte r10
   8byte r11
   16byte f8
   16byte f9
   16byte f10
   16byte f11
   16byte f12
   16byte f13
   16byte f14
   16byte f15

  Since stack has to be 16 byte aligned, the stack allocation is in
  16byte increment. Before calling _dl_profile_fixup, the stack will
  look like

  psp	new frame_size
  +16	La_ia64_regs
  sp	scratch

 */

#define PLTENTER_FRAME_SIZE (4*8 + 8*8 + 8*16 + 2*8 + 16)
#define PLTEXIT_FRAME_SIZE (PLTENTER_FRAME_SIZE + 4*8 + 8*16)

#ifndef PROF
ENTRY(_dl_runtime_profile)
	{ .mii
	  .prologue
	  .save ar.pfs, r40
	  alloc loc0 = ar.pfs, 8, 12, 8, 0
	  .vframe loc10
	  mov loc10 = r12
	  .save rp, loc1
	  mov loc1 = b0
	}
	{ .mii
	  .save ar.unat, r17
	  mov r17 = ar.unat
	  .save ar.lc, loc6
	  mov loc6 = ar.lc
	  mov loc11 = gp
	}
	{ .mii
	  .body
	  /* There is a 16 byte scratch area. r2 will start at r8 and
	     r3 will start at r9 for La_ia64_regs.  */
	  adds r2 = -(PLTENTER_FRAME_SIZE - 16), r12
	  adds r3 = -(PLTENTER_FRAME_SIZE - 24), r12
	  adds r12 = -PLTENTER_FRAME_SIZE, r12
	  ;;
	}
	{ .mmi
	  st8 [r2] = r8, 16;
	  st8 [r3] = r9, 16;
	  mov out2 = b0		/* needed by _dl_fixup_profile */
	  ;;
	}
	{ .mmi
	  st8 [r2] = r10, 16;
	  st8 [r3] = r11, 16;
	  adds out3 = 16, r12	/* pointer to La_ia64_regs */
	  ;;
	}
	{ .mmi
	  .mem.offset 0, 0
	  st8.spill [r2] = in0, 16
	  .mem.offset 8, 0
	  st8.spill [r3] = in1, 16
	  mov out4 = loc10	/* pointer to new frame size  */
	  ;;
	}
	{ .mmi
	  .mem.offset 0, 0
	  st8.spill [r2] = in2, 16
	  .mem.offset 8, 0
	  st8.spill [r3] = in3, 16
	  mov loc2 = r8		/* preserve struct value register */
	  ;;
	}
	{ .mmi
	  .mem.offset 0, 0
	  st8.spill [r2] = in4, 16
	  .mem.offset 8, 0
	  st8.spill [r3] = in5, 16
	  mov loc3 = r9		/* preserve language specific register */
	  ;;
	}
	{ .mmi
	  .mem.offset 0, 0
	  st8 [r2] = in6, 16
	  .mem.offset 8, 0
	  st8 [r3] = in7, 24	/* adjust for f9 */
	  mov loc4 = r10	/* preserve language specific register */
	  ;;
	}
	{ .mii
	  mov r18 = ar.unat	/* save it in La_ia64_regs */
	  mov loc7 = out3	/* save it for _dl_call_pltexit */
	  mov loc5 = r11	/* preserve language specific register */
	}
	{ .mmi
	  stf.spill [r2] = f8, 32
	  stf.spill [r3] = f9, 32
	  mov out0 = r16	/* needed by _dl_fixup_profile */
	  ;;
	}
	{ .mii
	  mov ar.unat = r17	/* restore it for function call */
	  mov loc8 = r16	/* save it for _dl_call_pltexit */
	  nop.i 0x0
	}
	{ .mmi
	  stf.spill [r2] = f10, 32
	  stf.spill [r3] = f11, 32
	  shl out1 = r15, 4
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f12, 32
	  stf.spill [r3] = f13, 32
	  /* Relocation record is 24 byte. */
	  shladd out1 = r15, 3, out1
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f14, 32
	  stf.spill [r3] = f15, 24
	  mov loc9 = out1	/* save it for _dl_call_pltexit */
	  ;;
	}
	{ .mmb
	  st8 [r2] = r18	/* store ar.unat */
	  st8 [r3] = loc10	/* store sp */
	  br.call.sptk.many b0 = _dl_profile_fixup
	}
	{ .mii
	  /* Skip the 16byte scratch area, 4 language specific GRs and
	     8 incoming GRs to restore incoming fp registers.  */
	  adds r2 = (4*8 + 8*8 + 16), r12
	  adds r3 = (4*8 + 8*8 + 32), r12
	  mov b6 = ret0
	  ;;
	}
	{ .mmi
	  ldf.fill f8 = [r2], 32
	  ldf.fill f9 = [r3], 32
	  mov gp = ret1
	  ;;
	}
	{ .mmi
	  ldf.fill f10 = [r2], 32
	  ldf.fill f11 = [r3], 32
	  mov r8 = loc2		/* restore struct value register */
	  ;;
	}
	{ .mmi
	  ldf.fill f12 = [r2], 32
	  ldf.fill f13 = [r3], 32
	  mov r9 = loc3		/* restore language specific register */
	  ;;
	}
	{ .mmi
	  ldf.fill f14 = [r2], 32
	  ldf.fill f15 = [r3], 32
	  mov r10 = loc4	/* restore language specific register */
	  ;;
	}
	{ .mii
	  ld8 r15 = [loc10]	/* load the new frame size */
	  mov r11 = loc5	/* restore language specific register */
	  ;;
	  cmp.eq p6, p7 = -1, r15
	  ;;
	}
	{ .mii
(p7)	  cmp.eq p8, p9 = 0, r15
(p6)	  mov b0 = loc1
(p6)	  mov ar.lc = loc6
	}
	{ .mib
	  nop.m 0x0
(p6)	  mov ar.pfs = loc0
(p6)	  br.cond.dptk.many .Lresolved
	  ;;
	}

	/* At this point, the stack looks like

	  +psp	free
	  +16	La_ia64_regs
	  sp	scratch

	  We need to keep the current stack and call the resolved
	  function by copying the r15 byte from sp + PLTENTER_FRAME_SIZE
	  + 16 (scratch area) to sp + 16 (scratch area). Since stack
	  has to be 16byte aligned, we around r15 up to 16byte.  */

	{ .mbb
(p9)	  adds r15 = 15, r15
(p8)	  br.cond.dptk.many .Lno_new_frame
	  nop.b 0x0
	  ;;
	}
	{ .mmi
	  and r15 = -16, r15
	  ;;
	  /* We don't copy the 16byte scatch area. Prepare r16/r17 as
	     destination.  */
	  sub r16 = r12, r15
	  sub r17 = r12, r15
	  ;;
	}
	{ .mii
	  adds r16 = 16, r16
	  adds r17 = 24, r17
	  sub r12 = r12, r15		/* Adjust stack  */
	  ;;
	}
	{ .mii
	  nop.m 0x0
	  shr r15 = r15, 4
	  ;;
	  adds r15 = -1, r15
	  ;;
	}
	{ .mii
	  /* Skip the 16byte scatch area. Prepare r2/r3 as source.  */
	  adds r2 = 16, loc10
	  adds r3 = 24, loc10
	  mov ar.lc = r15
	  ;;
	}
.Lcopy:
	{ .mmi
	  ld8 r18 = [r2], 16
	  ld8 r19 = [r3], 16
	  nop.i 0x0
	  ;;
	}
	{ .mmb
	  st8 [r16] = r18, 16
	  st8 [r17] = r19, 16
	  br.cloop.sptk.few .Lcopy
	}
.Lno_new_frame:
	{ .mii
	  mov out0 = in0
	  mov out1 = in1
	  mov out2 = in2
	}
	{ .mii
	  mov out3 = in3
	  mov out4 = in4
	  mov out5 = in5
	}
	{ .mib
	  mov out6 = in6
	  mov out7 = in7
	  /* Call the resolved function  */
	  br.call.sptk.many b0 = b6
	}
	{ .mii
	  /* Prepare stack for _dl_call_pltexit. Loc10 has the original
	     stack pointer.  */
	  adds r12 = -PLTEXIT_FRAME_SIZE, loc10
	  adds r2 = -(PLTEXIT_FRAME_SIZE - 16), loc10
	  adds r3 = -(PLTEXIT_FRAME_SIZE - 24), loc10
	  ;;
	}
	{ .mmi
	  /* Load all possible return values into buffer.  */
	  st8 [r2] = r8, 16
	  st8 [r3] = r9, 16
	  mov out0 = loc8
	  ;;
	}
	{ .mmi
	  st8 [r2] = r10, 16
	  st8 [r3] = r11, 24
	  mov out1 = loc9
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f8, 32
	  stf.spill [r3] = f9, 32
	  mov out2 = loc7		/* Pointer to La_ia64_regs */
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f10, 32
	  stf.spill [r3] = f11, 32
	  adds out3 = 16, r12		/* Pointer to La_ia64_retval */
	  ;;
	}
	{ .mmi
	  stf.spill [r2] = f12, 32
	  stf.spill [r3] = f13, 32
	  /* We need to restore gp for _dl_call_pltexit. */
	  mov gp = loc11
	  ;;
	}
	{ .mmb
	  stf.spill [r2] = f14
	  stf.spill [r3] = f15
	  br.call.sptk.many b0 = _dl_call_pltexit
	}
	{ .mmi
	  /* Load all the non-floating and floating return values. Skip
	     the 16byte scratch area.  */
	  adds r2 = 16, r12
	  adds r3 = 24, r12
	  nop.i 0x0
	  ;;
	}
	{ .mmi
	  ld8 r8 = [r2], 16
	  ld8 r9 = [r3], 16
	  nop.i 0x0
	  ;;
	}
	{ .mmi
	  ld8 r10 = [r2], 16
	  ld8 r11 = [r3], 24
	  nop.i 0x0
	  ;;
	}
	{ .mmi
	  ldf.fill f8 = [r2], 32
	  ldf.fill f9 = [r3], 32
	  mov ar.lc = loc6
	  ;;
	}
	{ .mmi
	  ldf.fill f10 = [r2], 32
	  ldf.fill f11 = [r3], 32
	  mov ar.pfs = loc0
	  ;;
	}
	{ .mmi
	  ldf.fill f12 = [r2], 32
	  ldf.fill f13 = [r3], 32
	  mov b0 = loc1
	  ;;
	}
	{ .mmi
	  ldf.fill f14 = [r2]
	  ldf.fill f15 = [r3]
	  /* We know that the previous stack pointer, loc10, isn't 0.
	     We use it to reload p7.  */
	  cmp.ne p7, p0 = 0, loc10
	  ;;
	}
.Lresolved:
	{ .mmb
	  .restore sp
	  mov r12 = loc10
(p7)	  br.ret.sptk.many b0
	  ;;
	}
	/* An alloc is needed for the break system call to work. We
	   don't care about the old value of the pfs register. After
	   this alloc, we can't use any rotating registers. Otherwise
	   assembler won't be happy. This has to be at the end.  */
	{ .mmb
	  .prologue
	  .body
	  alloc r2 = ar.pfs, 0, 0, 8, 0
	  br.sptk.many b6
	  ;;
	}
END(_dl_runtime_profile)
#endif

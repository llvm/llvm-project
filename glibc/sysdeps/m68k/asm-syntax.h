/* Definitions for 68k syntax variations.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.  Its master source is NOT part of
   the C library, however.  The master source lives in the GNU MP Library.

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

/* For ELF we need to prefix register names and local labels.  */
#define R_(r) %##r
#define R(r) R_(r)
#define L(label) .##label

#ifdef MIT_SYNTAX
#define MEM(base)R(base)@
#define MEM_DISP(base,displacement)R(base)@(displacement)
#define MEM_INDX(base,idx,size_suffix)R(base)@(R(idx):size_suffix)
#define MEM_INDX1(base,idx,size_suffix,scale)R(base)@(R(idx):size_suffix:scale)
#define MEM_PREDEC(memory_base)R(memory_base)@-
#define MEM_POSTINC(memory_base)R(memory_base)@+
#define TEXT .text
/* Use variable sized opcodes.  */
#define bcc jcc
#define bcs jcs
#define bls jls
#define beq jeq
#define bne jne
#define bra jra
#endif

#ifdef MOTOROLA_SYNTAX
#define MEM(base)(R(base))
#define MEM_DISP(base,displacement)(displacement,R(base))
#define MEM_PREDEC(memory_base)-(R(memory_base))
#define MEM_POSTINC(memory_base)(R(memory_base))+
#define MEM_INDX_(base,idx,size_suffix)(R(base),R(idx##.##size_suffix))
#define MEM_INDX(base,idx,size_suffix)MEM_INDX_(base,idx,size_suffix)
#define MEM_INDX1_(base,idx,size_suffix,scale)(R(base),R(idx##.##size_suffix*scale))
#define MEM_INDX1(base,idx,size_suffix,scale)MEM_INDX1_(base,idx,size_suffix,scale)
#define TEXT .text
#define bcc jbcc
#define bcs jbcs
#define bls jbls
#define beq jbeq
#define bne jbne
#define bra jbra
#define movel move.l
#define moveml movem.l
#define moveql moveq.l
#define cmpl cmp.l
#define orl or.l
#define clrl clr.l
#define andw and.w
#define eorw eor.w
#define andl and.l
#define lsrl lsr.l
#define lsll lsl.l
#define roxrl roxr.l
#define roxll roxl.l
#define addl add.l
#define addxl addx.l
#define addql addq.l
#define subl sub.l
#define subxl subx.l
#define subqw subq.w
#define subql subq.l
#define negl neg.l
#define mulul mulu.l
#define tstw tst.w
#define tstl tst.l
#endif

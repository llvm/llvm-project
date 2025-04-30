/* Dump registers.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <sys/uio.h>
#include <_itoa.h>

/* We will print the register dump in this format:

  R0: XXXXXXXX   R1: XXXXXXXX   R2: XXXXXXXX   R3: XXXXXXXX
  R4: XXXXXXXX   R5: XXXXXXXX   R6: XXXXXXXX   R7: XXXXXXXX
  R8: XXXXXXXX   R9: XXXXXXXX  R10: XXXXXXXX  R11: XXXXXXXX
 R12: XXXXXXXX  R13: XXXXXXXX  R14: XXXXXXXX  R15: XXXXXXXX

MACL: XXXXXXXX MACH: XXXXXXXX

  PC: XXXXXXXX   PR: XXXXXXXX  GBR: XXXXXXXX   SR: XXXXXXXX

 FR0: XXXXXXXX  FR1: XXXXXXXX  FR2: XXXXXXXX  FR3: XXXXXXXX
 FR4: XXXXXXXX  FR5: XXXXXXXX  FR6: XXXXXXXX  FR7: XXXXXXXX
 FR8: XXXXXXXX  FR9: XXXXXXXX FR10: XXXXXXXX FR11: XXXXXXXX
FR12: XXXXXXXX FR13: XXXXXXXX FR14: XXXXXXXX FR15: XXXXXXXX

 XR0: XXXXXXXX  XR1: XXXXXXXX  XR2: XXXXXXXX  XR3: XXXXXXXX
 XR4: XXXXXXXX  XR5: XXXXXXXX  XR6: XXXXXXXX  XR7: XXXXXXXX
 XR8: XXXXXXXX  XR9: XXXXXXXX XR10: XXXXXXXX XR11: XXXXXXXX
XR12: XXXXXXXX XR13: XXXXXXXX XR14: XXXXXXXX XR15: XXXXXXXX

FPSCR: XXXXXXXX FPUL: XXXXXXXX

 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
register_dump (int fd, struct ucontext_t *ctx)
{
  char regs[22][8];
  struct iovec iov[22 * 2 + 34 * 2 + 2];
  size_t nr = 0;

#define ADD_STRING(str) \
  iov[nr].iov_base = (char *) str;					      \
  iov[nr].iov_len = strlen (str);					      \
  ++nr
#define ADD_MEM(str, len) \
  iov[nr].iov_base = str;						      \
  iov[nr].iov_len = len;						      \
  ++nr

  /* Generate strings of register contents.  */
  hexvalue (ctx->uc_mcontext.gregs[REG_R0], regs[0], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R1], regs[1], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R2], regs[2], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R3], regs[3], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R4], regs[4], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R5], regs[5], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R6], regs[6], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R7], regs[7], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R8], regs[8], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R9], regs[9], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R10], regs[10], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R11], regs[11], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R12], regs[12], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R13], regs[13], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R14], regs[14], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_R15], regs[15], 8);
  hexvalue (ctx->uc_mcontext.macl, regs[16], 8);
  hexvalue (ctx->uc_mcontext.mach, regs[17], 8);
  hexvalue (ctx->uc_mcontext.pc, regs[18], 8);
  hexvalue (ctx->uc_mcontext.pr, regs[19], 8);
  hexvalue (ctx->uc_mcontext.gbr, regs[20], 8);
  hexvalue (ctx->uc_mcontext.sr, regs[21], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n  R0: ");
  ADD_MEM (regs[0], 8);
  ADD_STRING ("   R1: ");
  ADD_MEM (regs[1], 8);
  ADD_STRING ("   R2: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("   R3: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n  R4: ");
  ADD_MEM (regs[4], 8);
  ADD_STRING ("   R5: ");
  ADD_MEM (regs[5], 8);
  ADD_STRING ("   R6: ");
  ADD_MEM (regs[6], 8);
  ADD_STRING ("   R7: ");
  ADD_MEM (regs[7], 8);
  ADD_STRING ("\n  R8: ");
  ADD_MEM (regs[8], 8);
  ADD_STRING ("   R9: ");
  ADD_MEM (regs[9], 8);
  ADD_STRING ("  R10: ");
  ADD_MEM (regs[10], 8);
  ADD_STRING ("  R11: ");
  ADD_MEM (regs[11], 8);
  ADD_STRING ("\n R12: ");
  ADD_MEM (regs[12], 8);
  ADD_STRING ("  R13: ");
  ADD_MEM (regs[13], 8);
  ADD_STRING ("  R14: ");
  ADD_MEM (regs[14], 8);
  ADD_STRING ("  R15: ");
  ADD_MEM (regs[15], 8);

  ADD_STRING ("\n\nMACL: ");
  ADD_MEM (regs[16], 8);
  ADD_STRING (" MACH: ");
  ADD_MEM (regs[17], 8);

  ADD_STRING ("\n\n  PC: ");
  ADD_MEM (regs[18], 8);
  ADD_STRING ("   PR: ");
  ADD_MEM (regs[19], 8);
  ADD_STRING ("  GBR: ");
  ADD_MEM (regs[20], 8);
  ADD_STRING ("   SR: ");
  ADD_MEM (regs[21], 8);

  ADD_STRING ("\n");

#ifdef __SH_FPU_ANY__
  char fpregs[34][8];
  if (ctx->uc_mcontext.ownedfp != 0)
    {
      hexvalue (ctx->uc_mcontext.fpregs[0], fpregs[0], 8);
      hexvalue (ctx->uc_mcontext.fpregs[1], fpregs[1], 8);
      hexvalue (ctx->uc_mcontext.fpregs[2], fpregs[2], 8);
      hexvalue (ctx->uc_mcontext.fpregs[3], fpregs[3], 8);
      hexvalue (ctx->uc_mcontext.fpregs[4], fpregs[4], 8);
      hexvalue (ctx->uc_mcontext.fpregs[5], fpregs[5], 8);
      hexvalue (ctx->uc_mcontext.fpregs[6], fpregs[6], 8);
      hexvalue (ctx->uc_mcontext.fpregs[7], fpregs[7], 8);
      hexvalue (ctx->uc_mcontext.fpregs[8], fpregs[8], 8);
      hexvalue (ctx->uc_mcontext.fpregs[9], fpregs[9], 8);
      hexvalue (ctx->uc_mcontext.fpregs[10], fpregs[10], 8);
      hexvalue (ctx->uc_mcontext.fpregs[11], fpregs[11], 8);
      hexvalue (ctx->uc_mcontext.fpregs[12], fpregs[12], 8);
      hexvalue (ctx->uc_mcontext.fpregs[13], fpregs[13], 8);
      hexvalue (ctx->uc_mcontext.fpregs[14], fpregs[14], 8);
      hexvalue (ctx->uc_mcontext.fpregs[15], fpregs[15], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[0], fpregs[16], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[1], fpregs[17], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[2], fpregs[18], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[3], fpregs[19], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[4], fpregs[20], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[5], fpregs[21], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[6], fpregs[22], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[7], fpregs[23], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[8], fpregs[24], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[9], fpregs[25], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[10], fpregs[26], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[11], fpregs[27], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[12], fpregs[28], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[13], fpregs[29], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[14], fpregs[30], 8);
      hexvalue (ctx->uc_mcontext.xfpregs[15], fpregs[31], 8);
      hexvalue (ctx->uc_mcontext.fpscr, fpregs[32], 8);
      hexvalue (ctx->uc_mcontext.fpul, fpregs[33], 8);

      ADD_STRING ("\n\n FR0: ");
      ADD_MEM (fpregs[0], 8);
      ADD_STRING ("  FR1: ");
      ADD_MEM (fpregs[1], 8);
      ADD_STRING ("  FR2: ");
      ADD_MEM (fpregs[2], 8);
      ADD_STRING ("  FR3: ");
      ADD_MEM (fpregs[3], 8);
      ADD_STRING ("\n FR4: ");
      ADD_MEM (fpregs[4], 8);
      ADD_STRING ("  FR5: ");
      ADD_MEM (fpregs[5], 8);
      ADD_STRING ("  FR6: ");
      ADD_MEM (fpregs[6], 8);
      ADD_STRING ("  FR7: ");
      ADD_MEM (fpregs[7], 8);
      ADD_STRING ("\n FR8: ");
      ADD_MEM (fpregs[8], 8);
      ADD_STRING ("  FR9: ");
      ADD_MEM (fpregs[9], 8);
      ADD_STRING (" FR10: ");
      ADD_MEM (fpregs[10], 8);
      ADD_STRING (" FR11: ");
      ADD_MEM (fpregs[11], 8);
      ADD_STRING ("\nFR12: ");
      ADD_MEM (fpregs[12], 8);
      ADD_STRING (" FR13: ");
      ADD_MEM (fpregs[13], 8);
      ADD_STRING (" FR14: ");
      ADD_MEM (fpregs[14], 8);
      ADD_STRING (" FR15: ");
      ADD_MEM (fpregs[15], 8);
      ADD_STRING ("\n\n XR0: ");
      ADD_MEM (fpregs[16], 8);
      ADD_STRING ("  XR1: ");
      ADD_MEM (fpregs[17], 8);
      ADD_STRING ("  XR2: ");
      ADD_MEM (fpregs[18], 8);
      ADD_STRING ("  XR3: ");
      ADD_MEM (fpregs[19], 8);
      ADD_STRING ("\n XR4: ");
      ADD_MEM (fpregs[20], 8);
      ADD_STRING ("  XR5: ");
      ADD_MEM (fpregs[21], 8);
      ADD_STRING ("  XR6: ");
      ADD_MEM (fpregs[22], 8);
      ADD_STRING ("  XR7: ");
      ADD_MEM (fpregs[23], 8);
      ADD_STRING ("\n XR8: ");
      ADD_MEM (fpregs[24], 8);
      ADD_STRING ("  XR9: ");
      ADD_MEM (fpregs[25], 8);
      ADD_STRING (" XR10: ");
      ADD_MEM (fpregs[26], 8);
      ADD_STRING (" XR11: ");
      ADD_MEM (fpregs[27], 8);
      ADD_STRING ("\nXR12: ");
      ADD_MEM (fpregs[28], 8);
      ADD_STRING (" XR13: ");
      ADD_MEM (fpregs[29], 8);
      ADD_STRING (" XR14: ");
      ADD_MEM (fpregs[30], 8);
      ADD_STRING (" XR15: ");
      ADD_MEM (fpregs[31], 8);

      ADD_STRING ("\n\nFPSCR: ");
      ADD_MEM (fpregs[32], 8);
      ADD_STRING (" FPUL: ");
      ADD_MEM (fpregs[33], 8);

      ADD_STRING ("\n");
    }
#endif /* __SH_FPU_ANY__  */

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

/* Dump registers.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

 RAX: XXXXXXXXXXXXXXXX   RBX: XXXXXXXXXXXXXXXX  RCX: XXXXXXXXXXXXXXXX
 RDX: XXXXXXXXXXXXXXXX   RSI: XXXXXXXXXXXXXXXX  RDI: XXXXXXXXXXXXXXXX
 RBP: XXXXXXXXXXXXXXXX   R8 : XXXXXXXXXXXXXXXX  R9 : XXXXXXXXXXXXXXXX
 R10: XXXXXXXXXXXXXXXX   R11: XXXXXXXXXXXXXXXX  R12: XXXXXXXXXXXXXXXX
 R13: XXXXXXXXXXXXXXXX   R14: XXXXXXXXXXXXXXXX  R15: XXXXXXXXXXXXXXXX
 RSP: XXXXXXXXXXXXXXXX

 RIP: XXXXXXXXXXXXXXXX   EFLAGS: XXXXXXXX

 CS:  XXXX   DS: XXXX   ES: XXXX   FS: XXXX   GS: XXXX

 Trap:  XXXXXXXX   Error: XXXXXXXX   OldMask: XXXXXXXX
 RSP/SIGNAL: XXXXXXXXXXXXXXXX  CR2: XXXXXXXX

 FPUCW: XXXXXXXX   FPUSW: XXXXXXXX   TAG: XXXXXXXX
 IPOFF: XXXXXXXX   CSSEL: XXXX   DATAOFF: XXXXXXXX   DATASEL: XXXX

 ST(0) XXXX XXXXXXXXXXXXXXXX   ST(1) XXXX XXXXXXXXXXXXXXXX
 ST(2) XXXX XXXXXXXXXXXXXXXX   ST(3) XXXX XXXXXXXXXXXXXXXX
 ST(4) XXXX XXXXXXXXXXXXXXXX   ST(5) XXXX XXXXXXXXXXXXXXXX
 ST(6) XXXX XXXXXXXXXXXXXXXX   ST(7) XXXX XXXXXXXXXXXXXXXX

 mxcsr: XXXX
 XMM0 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM1 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM2 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM3 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM4 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM5 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM6 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM7 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM8 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM9 : XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM10: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM11: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM12: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM13: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 XMM14: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XMM15: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
register_dump (int fd, ucontext_t *ctx)
{
  char regs[25][16];
  char fpregs[30][8];
  char xmmregs[16][32];
  struct iovec iov[147];
  size_t nr = 0;
  int i;

#define ADD_STRING(str) \
  iov[nr].iov_base = (char *) str;					      \
  iov[nr].iov_len = strlen (str);					      \
  ++nr
#define ADD_MEM(str, len) \
  iov[nr].iov_base = str;						      \
  iov[nr].iov_len = len;						      \
  ++nr

  /* Generate strings of register contents.  */
  hexvalue (ctx->uc_mcontext.gregs[REG_RAX], regs[0], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RBX], regs[1], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RCX], regs[2], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RDX], regs[3], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RSI], regs[4], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RDI], regs[5], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RBP], regs[6], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R8], regs[7], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R9], regs[8], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R10], regs[9], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R11], regs[10], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R12], regs[11], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R13], regs[12], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R14], regs[13], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_R15], regs[14], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RSP], regs[15], 16);
  hexvalue (ctx->uc_mcontext.gregs[REG_RIP], regs[16], 16);

  hexvalue (ctx->uc_mcontext.gregs[REG_EFL], regs[17], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_CSGSFS] & 0xffff, regs[18], 4);
  hexvalue ((ctx->uc_mcontext.gregs[REG_CSGSFS] >> 16) & 0xffff, regs[19], 4);
  hexvalue ((ctx->uc_mcontext.gregs[REG_CSGSFS] >> 32) & 0xffff, regs[20], 4);
  /* hexvalue (ctx->ss, regs[23], 4); */
  hexvalue (ctx->uc_mcontext.gregs[REG_TRAPNO], regs[21], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_ERR], regs[22], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_OLDMASK], regs[23], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_CR2], regs[24], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n RAX: ");
  ADD_MEM (regs[0], 16);
  ADD_STRING ("   RBX: ");
  ADD_MEM (regs[1], 16);
  ADD_STRING ("   RCX: ");
  ADD_MEM (regs[2], 16);
  ADD_STRING ("\n RDX: ");
  ADD_MEM (regs[3], 16);
  ADD_STRING ("   RSI: ");
  ADD_MEM (regs[4], 16);
  ADD_STRING ("   RDI: ");
  ADD_MEM (regs[5], 16);
  ADD_STRING ("\n RBP: ");
  ADD_MEM (regs[6], 16);
  ADD_STRING ("   R8 : ");
  ADD_MEM (regs[7], 16);
  ADD_STRING ("   R9 : ");
  ADD_MEM (regs[8], 16);
  ADD_STRING ("\n R10: ");
  ADD_MEM (regs[9], 16);
  ADD_STRING ("   R11: ");
  ADD_MEM (regs[10], 16);
  ADD_STRING ("   R12: ");
  ADD_MEM (regs[11], 16);
  ADD_STRING ("\n R13: ");
  ADD_MEM (regs[12], 16);
  ADD_STRING ("   R14: ");
  ADD_MEM (regs[13], 16);
  ADD_STRING ("   R15: ");
  ADD_MEM (regs[14], 16);
  ADD_STRING ("\n RSP: ");
  ADD_MEM (regs[15], 16);
  ADD_STRING ("\n\n RIP: ");
  ADD_MEM (regs[16], 16);
  ADD_STRING ("   EFLAGS: ");
  ADD_MEM (regs[17], 8);
  ADD_STRING ("\n\n CS: ");
  ADD_MEM (regs[18], 4);
  ADD_STRING ("   FS: ");
  ADD_MEM (regs[19], 4);
  ADD_STRING ("   GS: ");
  ADD_MEM (regs[20], 4);
  /*
  ADD_STRING ("   SS: ");
  ADD_MEM (regs[23], 4);
  */
  ADD_STRING ("\n\n Trap: ");
  ADD_MEM (regs[21], 8);
  ADD_STRING ("   Error: ");
  ADD_MEM (regs[22], 8);
  ADD_STRING ("   OldMask: ");
  ADD_MEM (regs[23], 8);
  ADD_STRING ("   CR2: ");
  ADD_MEM (regs[24], 8);

  if (ctx->uc_mcontext.fpregs != NULL)
    {

      /* Generate output for the FPU control/status registers.  */
      hexvalue (ctx->uc_mcontext.fpregs->cwd, fpregs[0], 8);
      hexvalue (ctx->uc_mcontext.fpregs->swd, fpregs[1], 8);
      hexvalue (ctx->uc_mcontext.fpregs->ftw, fpregs[2], 8);
      hexvalue (ctx->uc_mcontext.fpregs->rip, fpregs[3], 8);
      hexvalue (ctx->uc_mcontext.fpregs->rdp, fpregs[4], 8);

      ADD_STRING ("\n\n FPUCW: ");
      ADD_MEM (fpregs[0], 8);
      ADD_STRING ("   FPUSW: ");
      ADD_MEM (fpregs[1], 8);
      ADD_STRING ("   TAG: ");
      ADD_MEM (fpregs[2], 8);
      ADD_STRING ("\n RIP: ");
      ADD_MEM (fpregs[3], 8);
      ADD_STRING ("   RDP: ");
      ADD_MEM (fpregs[4], 8);

      /* Now the real FPU registers.  */
      hexvalue (ctx->uc_mcontext.fpregs->_st[0].exponent, fpregs[5], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[0].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[0].significand[2], fpregs[6],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[0].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[0].significand[0], fpregs[7],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[1].exponent, fpregs[8], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[1].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[1].significand[2], fpregs[9],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[1].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[1].significand[0], fpregs[10],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[2].exponent, fpregs[11], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[2].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[2].significand[2], fpregs[12],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[2].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[2].significand[0], fpregs[13],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[3].exponent, fpregs[14], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[3].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[3].significand[2], fpregs[15],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[3].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[3].significand[0], fpregs[16],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[4].exponent, fpregs[17], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[4].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[4].significand[2], fpregs[18],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[4].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[4].significand[0], fpregs[19],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[5].exponent, fpregs[20], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[5].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[5].significand[2], fpregs[21],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[5].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[5].significand[0], fpregs[22],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[6].exponent, fpregs[23], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[6].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[6].significand[2], fpregs[24],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[6].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[6].significand[0], fpregs[25],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[7].exponent, fpregs[26], 8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[7].significand[3] << 16
		| ctx->uc_mcontext.fpregs->_st[7].significand[2], fpregs[27],
		8);
      hexvalue (ctx->uc_mcontext.fpregs->_st[7].significand[1] << 16
		| ctx->uc_mcontext.fpregs->_st[7].significand[0], fpregs[28],
		8);

      hexvalue (ctx->uc_mcontext.fpregs->mxcsr, fpregs[29], 4);

      for (i = 0; i < 16; i++)
	hexvalue (ctx->uc_mcontext.fpregs->_xmm[i].element[3] << 24
		  | ctx->uc_mcontext.fpregs->_xmm[i].element[2] << 16
		  | ctx->uc_mcontext.fpregs->_xmm[i].element[1] << 8
		  | ctx->uc_mcontext.fpregs->_xmm[i].element[0], xmmregs[i],
		  32);


      ADD_STRING ("\n\n ST(0) ");
      ADD_MEM (fpregs[5], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[6], 8);
      ADD_MEM (fpregs[7], 8);
      ADD_STRING ("   ST(1) ");
      ADD_MEM (fpregs[8], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[9], 8);
      ADD_MEM (fpregs[10], 8);
      ADD_STRING ("\n ST(2) ");
      ADD_MEM (fpregs[11], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[12], 8);
      ADD_MEM (fpregs[13], 8);
      ADD_STRING ("   ST(3) ");
      ADD_MEM (fpregs[14], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[15], 8);
      ADD_MEM (fpregs[16], 8);
      ADD_STRING ("\n ST(4) ");
      ADD_MEM (fpregs[17], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[18], 8);
      ADD_MEM (fpregs[19], 8);
      ADD_STRING ("   ST(5) ");
      ADD_MEM (fpregs[20], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[21], 8);
      ADD_MEM (fpregs[22], 8);
      ADD_STRING ("\n ST(6) ");
      ADD_MEM (fpregs[23], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[24], 8);
      ADD_MEM (fpregs[25], 8);
      ADD_STRING ("   ST(7) ");
      ADD_MEM (fpregs[27], 4);
      ADD_STRING (" ");
      ADD_MEM (fpregs[27], 8);
      ADD_MEM (fpregs[28], 8);

      ADD_STRING ("\n mxcsr: ");
      ADD_MEM (fpregs[29], 4);

      ADD_STRING ("\n XMM0:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM1:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM2:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM3:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM4:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM5:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM6:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM7:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM8:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM9:  ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM10: ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM11: ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM12: ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM13: ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING ("\n XMM14: ");
      ADD_MEM (xmmregs[0], 32);
      ADD_STRING (" XMM15: ");
      ADD_MEM (xmmregs[0], 32);

    }

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

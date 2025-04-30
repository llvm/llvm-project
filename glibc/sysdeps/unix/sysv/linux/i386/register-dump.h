/* Dump registers.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

 EAX: XXXXXXXX   EBX: XXXXXXXX   ECX: XXXXXXXX   EDX: XXXXXXXX
 ESI: XXXXXXXX   EDI: XXXXXXXX   EBP: XXXXXXXX   ESP: XXXXXXXX

 EIP: XXXXXXXX   EFLAGS: XXXXXXXX

 CS:  XXXX   DS: XXXX   ES: XXXX   FS: XXXX   GS: XXXX   SS: XXXX

 Trap:  XXXXXXXX   Error: XXXXXXXX   OldMask: XXXXXXXX
 ESP/SIGNAL: XXXXXXXX   CR2: XXXXXXXX

 FPUCW: XXXXXXXX   FPUSW: XXXXXXXX   TAG: XXXXXXXX
 IPOFF: XXXXXXXX   CSSEL: XXXX   DATAOFF: XXXXXXXX   DATASEL: XXXX

 ST(0) XXXX XXXXXXXXXXXXXXXX   ST(1) XXXX XXXXXXXXXXXXXXXX
 ST(2) XXXX XXXXXXXXXXXXXXXX   ST(3) XXXX XXXXXXXXXXXXXXXX
 ST(4) XXXX XXXXXXXXXXXXXXXX   ST(5) XXXX XXXXXXXXXXXXXXXX
 ST(6) XXXX XXXXXXXXXXXXXXXX   ST(7) XXXX XXXXXXXXXXXXXXXX

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
  char regs[21][8];
  char fpregs[31][8];
  struct iovec iov[97];
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
  hexvalue (ctx->uc_mcontext.gregs[REG_EAX], regs[0], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_EBX], regs[1], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_ECX], regs[2], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_EDX], regs[3], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_ESI], regs[4], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_EDI], regs[5], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_EBP], regs[6], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_ESP], regs[7], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_EIP], regs[8], 8);
  hexvalue (ctx->uc_flags, regs[9], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_CS], regs[10], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_DS], regs[11], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_ES], regs[12], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_FS], regs[13], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_GS], regs[14], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_SS], regs[15], 4);
  hexvalue (ctx->uc_mcontext.gregs[REG_TRAPNO], regs[16], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_ERR], regs[17], 8);
  hexvalue (ctx->uc_mcontext.oldmask, regs[18], 8);
  hexvalue (ctx->uc_mcontext.gregs[REG_UESP], regs[19], 8);
  hexvalue (ctx->uc_mcontext.cr2, regs[20], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n EAX: ");
  ADD_MEM (regs[0], 8);
  ADD_STRING ("   EBX: ");
  ADD_MEM (regs[1], 8);
  ADD_STRING ("   ECX: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("   EDX: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n ESI: ");
  ADD_MEM (regs[4], 8);
  ADD_STRING ("   EDI: ");
  ADD_MEM (regs[5], 8);
  ADD_STRING ("   EBP: ");
  ADD_MEM (regs[6], 8);
  ADD_STRING ("   ESP: ");
  ADD_MEM (regs[7], 8);
  ADD_STRING ("\n\n EIP: ");
  ADD_MEM (regs[8], 8);
  ADD_STRING ("   EFLAGS: ");
  ADD_MEM (regs[9], 8);
  ADD_STRING ("\n\n CS: ");
  ADD_MEM (regs[10], 4);
  ADD_STRING ("   DS: ");
  ADD_MEM (regs[11], 4);
  ADD_STRING ("   ES: ");
  ADD_MEM (regs[12], 4);
  ADD_STRING ("   FS: ");
  ADD_MEM (regs[13], 4);
  ADD_STRING ("   GS: ");
  ADD_MEM (regs[14], 4);
  ADD_STRING ("   SS: ");
  ADD_MEM (regs[15], 4);
  ADD_STRING ("\n\n Trap: ");
  ADD_MEM (regs[16], 8);
  ADD_STRING ("   Error: ");
  ADD_MEM (regs[17], 8);
  ADD_STRING ("   OldMask: ");
  ADD_MEM (regs[18], 8);
  ADD_STRING ("\n ESP/signal: ");
  ADD_MEM (regs[19], 8);
  ADD_STRING ("   CR2: ");
  ADD_MEM (regs[20], 8);

  /* Generate output for the FPU control/status registers.  */
  hexvalue (ctx->__fpregs_mem.cw, fpregs[0], 8);
  hexvalue (ctx->__fpregs_mem.sw, fpregs[1], 8);
  hexvalue (ctx->__fpregs_mem.tag, fpregs[2], 8);
  hexvalue (ctx->__fpregs_mem.ipoff, fpregs[3], 8);
  hexvalue (ctx->__fpregs_mem.cssel, fpregs[4], 4);
  hexvalue (ctx->__fpregs_mem.dataoff, fpregs[5], 8);
  hexvalue (ctx->__fpregs_mem.datasel, fpregs[6], 4);

  ADD_STRING ("\n\n FPUCW: ");
  ADD_MEM (fpregs[0], 8);
  ADD_STRING ("   FPUSW: ");
  ADD_MEM (fpregs[1], 8);
  ADD_STRING ("   TAG: ");
  ADD_MEM (fpregs[2], 8);
  ADD_STRING ("\n IPOFF: ");
  ADD_MEM (fpregs[3], 8);
  ADD_STRING ("   CSSEL: ");
  ADD_MEM (fpregs[4], 4);
  ADD_STRING ("   DATAOFF: ");
  ADD_MEM (fpregs[5], 8);
  ADD_STRING ("   DATASEL: ");
  ADD_MEM (fpregs[6], 4);

  /* Now the real FPU registers.  */
  hexvalue (ctx->__fpregs_mem._st[0].exponent, fpregs[7], 8);
  hexvalue (ctx->__fpregs_mem._st[0].significand[3] << 16
		| ctx->__fpregs_mem._st[0].significand[2], fpregs[8], 8);
  hexvalue (ctx->__fpregs_mem._st[0].significand[1] << 16
		| ctx->__fpregs_mem._st[0].significand[0], fpregs[9], 8);
  hexvalue (ctx->__fpregs_mem._st[1].exponent, fpregs[10], 8);
  hexvalue (ctx->__fpregs_mem._st[1].significand[3] << 16
		| ctx->__fpregs_mem._st[1].significand[2], fpregs[11], 8);
  hexvalue (ctx->__fpregs_mem._st[1].significand[1] << 16
		| ctx->__fpregs_mem._st[1].significand[0], fpregs[12], 8);
  hexvalue (ctx->__fpregs_mem._st[2].exponent, fpregs[13], 8);
  hexvalue (ctx->__fpregs_mem._st[2].significand[3] << 16
		| ctx->__fpregs_mem._st[2].significand[2], fpregs[14], 8);
  hexvalue (ctx->__fpregs_mem._st[2].significand[1] << 16
		| ctx->__fpregs_mem._st[2].significand[0], fpregs[15], 8);
  hexvalue (ctx->__fpregs_mem._st[3].exponent, fpregs[16], 8);
  hexvalue (ctx->__fpregs_mem._st[3].significand[3] << 16
		| ctx->__fpregs_mem._st[3].significand[2], fpregs[17], 8);
  hexvalue (ctx->__fpregs_mem._st[3].significand[1] << 16
		| ctx->__fpregs_mem._st[3].significand[0], fpregs[18], 8);
  hexvalue (ctx->__fpregs_mem._st[4].exponent, fpregs[19], 8);
  hexvalue (ctx->__fpregs_mem._st[4].significand[3] << 16
		| ctx->__fpregs_mem._st[4].significand[2], fpregs[20], 8);
  hexvalue (ctx->__fpregs_mem._st[4].significand[1] << 16
		| ctx->__fpregs_mem._st[4].significand[0], fpregs[21], 8);
  hexvalue (ctx->__fpregs_mem._st[5].exponent, fpregs[22], 8);
  hexvalue (ctx->__fpregs_mem._st[5].significand[3] << 16
		| ctx->__fpregs_mem._st[5].significand[2], fpregs[23], 8);
  hexvalue (ctx->__fpregs_mem._st[5].significand[1] << 16
		| ctx->__fpregs_mem._st[5].significand[0], fpregs[24], 8);
  hexvalue (ctx->__fpregs_mem._st[6].exponent, fpregs[25], 8);
  hexvalue (ctx->__fpregs_mem._st[6].significand[3] << 16
		| ctx->__fpregs_mem._st[6].significand[2], fpregs[26], 8);
  hexvalue (ctx->__fpregs_mem._st[6].significand[1] << 16
		| ctx->__fpregs_mem._st[6].significand[0], fpregs[27], 8);
  hexvalue (ctx->__fpregs_mem._st[7].exponent, fpregs[28], 8);
  hexvalue (ctx->__fpregs_mem._st[7].significand[3] << 16
		| ctx->__fpregs_mem._st[7].significand[2], fpregs[29], 8);
  hexvalue (ctx->__fpregs_mem._st[7].significand[1] << 16
		| ctx->__fpregs_mem._st[7].significand[0], fpregs[30], 8);

  ADD_STRING ("\n\n ST(0) ");
  ADD_MEM (fpregs[7], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[8], 8);
  ADD_MEM (fpregs[9], 8);
  ADD_STRING ("   ST(1) ");
  ADD_MEM (fpregs[10], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[11], 8);
  ADD_MEM (fpregs[12], 8);
  ADD_STRING ("\n ST(2) ");
  ADD_MEM (fpregs[13], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[14], 8);
  ADD_MEM (fpregs[15], 8);
  ADD_STRING ("   ST(3) ");
  ADD_MEM (fpregs[16], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[17], 8);
  ADD_MEM (fpregs[18], 8);
  ADD_STRING ("\n ST(4) ");
  ADD_MEM (fpregs[19], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[20], 8);
  ADD_MEM (fpregs[21], 8);
  ADD_STRING ("   ST(5) ");
  ADD_MEM (fpregs[22], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[23], 8);
  ADD_MEM (fpregs[24], 8);
  ADD_STRING ("\n ST(6) ");
  ADD_MEM (fpregs[25], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[26], 8);
  ADD_MEM (fpregs[27], 8);
  ADD_STRING ("   ST(7) ");
  ADD_MEM (fpregs[28], 4);
  ADD_STRING (" ");
  ADD_MEM (fpregs[29], 8);
  ADD_MEM (fpregs[30], 8);

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

/* Dump registers.  64 bit S/390 version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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

 GPR0: XXXXXXXXXXXXXXXX  GPR1: XXXXXXXXXXXXXXXX
 GPR2: XXXXXXXXXXXXXXXX  GPR3: XXXXXXXXXXXXXXXX
 GPR4: XXXXXXXXXXXXXXXX  GPR5: XXXXXXXXXXXXXXXX
 GPR6: XXXXXXXXXXXXXXXX  GPR7: XXXXXXXXXXXXXXXX
 GPR8: XXXXXXXXXXXXXXXX  GPR9: XXXXXXXXXXXXXXXX
 GPRA: XXXXXXXXXXXXXXXX  GPRB: XXXXXXXXXXXXXXXX
 GPRC: XXXXXXXXXXXXXXXX  GPRD: XXXXXXXXXXXXXXXX
 GPRE: XXXXXXXXXXXXXXXX  GPRF: XXXXXXXXXXXXXXXX

 PSW.MASK: XXXXXXXXXXXXXXXX   PSW.ADDR: XXXXXXXXXXXXXXXX

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
  char regs[19][16];
  struct iovec iov[40];
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
  hexvalue (ctx->uc_mcontext.gregs[0], regs[0], 16);
  hexvalue (ctx->uc_mcontext.gregs[1], regs[1], 16);
  hexvalue (ctx->uc_mcontext.gregs[2], regs[2], 16);
  hexvalue (ctx->uc_mcontext.gregs[3], regs[3], 16);
  hexvalue (ctx->uc_mcontext.gregs[4], regs[4], 16);
  hexvalue (ctx->uc_mcontext.gregs[5], regs[5], 16);
  hexvalue (ctx->uc_mcontext.gregs[6], regs[6], 16);
  hexvalue (ctx->uc_mcontext.gregs[7], regs[7], 16);
  hexvalue (ctx->uc_mcontext.gregs[8], regs[8], 16);
  hexvalue (ctx->uc_mcontext.gregs[9], regs[9], 16);
  hexvalue (ctx->uc_mcontext.gregs[10], regs[10], 16);
  hexvalue (ctx->uc_mcontext.gregs[11], regs[11], 16);
  hexvalue (ctx->uc_mcontext.gregs[12], regs[12], 16);
  hexvalue (ctx->uc_mcontext.gregs[13], regs[13], 16);
  hexvalue (ctx->uc_mcontext.gregs[14], regs[14], 16);
  hexvalue (ctx->uc_mcontext.gregs[15], regs[15], 16);
  hexvalue (ctx->uc_mcontext.psw.mask, regs[16], 16);
  hexvalue (ctx->uc_mcontext.psw.addr, regs[17], 16);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n GPR0: ");
  ADD_MEM (regs[0], 16);
  ADD_STRING ("  GPR1: ");
  ADD_MEM (regs[1], 16);
  ADD_STRING ("  GPR2: ");
  ADD_MEM (regs[2], 16);
  ADD_STRING ("  GPR3: ");
  ADD_MEM (regs[3], 16);
  ADD_STRING ("\n GPR4: ");
  ADD_MEM (regs[4], 16);
  ADD_STRING ("  GPR5: ");
  ADD_MEM (regs[5], 16);
  ADD_STRING ("  GPR6: ");
  ADD_MEM (regs[6], 16);
  ADD_STRING ("  GPR7: ");
  ADD_MEM (regs[7], 16);
  ADD_STRING ("\n GPR8: ");
  ADD_MEM (regs[8], 16);
  ADD_STRING ("  GPR9: ");
  ADD_MEM (regs[9], 16);
  ADD_STRING ("  GPRA: ");
  ADD_MEM (regs[10], 16);
  ADD_STRING ("  GPRB: ");
  ADD_MEM (regs[11], 16);
  ADD_STRING ("\n GPRC: ");
  ADD_MEM (regs[12], 16);
  ADD_STRING ("  GPRD: ");
  ADD_MEM (regs[13], 16);
  ADD_STRING ("  GPRE: ");
  ADD_MEM (regs[14], 16);
  ADD_STRING ("  GPRF: ");
  ADD_MEM (regs[15], 16);
  ADD_STRING ("\n\n PSW.MASK: ");
  ADD_MEM (regs[16], 16);
  ADD_STRING ("  PSW.ADDR: ");
  ADD_MEM (regs[17], 16);
  ADD_STRING ("  TRAP: ");
  ADD_MEM (regs[18], 4);
  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

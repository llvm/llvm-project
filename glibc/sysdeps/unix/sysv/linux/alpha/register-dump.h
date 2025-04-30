/* Dump registers.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <ucontext.h>
#include <sys/uio.h>
#include <_itoa.h>

/* We will print the register dump in this format:

    V0: XXXXXXXXXXXXXXXX    T0: XXXXXXXXXXXXXXXX    T1: XXXXXXXXXXXXXXXX
    T2: XXXXXXXXXXXXXXXX    T3: XXXXXXXXXXXXXXXX    T4: XXXXXXXXXXXXXXXX
    T5: XXXXXXXXXXXXXXXX    T6: XXXXXXXXXXXXXXXX    T7: XXXXXXXXXXXXXXXX
    S0: XXXXXXXXXXXXXXXX    S1: XXXXXXXXXXXXXXXX    S2: XXXXXXXXXXXXXXXX
    S3: XXXXXXXXXXXXXXXX    S4: XXXXXXXXXXXXXXXX    S5: XXXXXXXXXXXXXXXX
    S6: XXXXXXXXXXXXXXXX    A0: XXXXXXXXXXXXXXXX    A1: XXXXXXXXXXXXXXXX
    A2: XXXXXXXXXXXXXXXX    A3: XXXXXXXXXXXXXXXX    A4: XXXXXXXXXXXXXXXX
    A5: XXXXXXXXXXXXXXXX    T8: XXXXXXXXXXXXXXXX    T9: XXXXXXXXXXXXXXXX
   T10: XXXXXXXXXXXXXXXX   T11: XXXXXXXXXXXXXXXX    RA: XXXXXXXXXXXXXXXX
   T12: XXXXXXXXXXXXXXXX    AT: XXXXXXXXXXXXXXXX    GP: XXXXXXXXXXXXXXXX
    SP: XXXXXXXXXXXXXXXX    PC: XXXXXXXXXXXXXXXX

   FP0: XXXXXXXXXXXXXXXX   FP1: XXXXXXXXXXXXXXXX   FP2: XXXXXXXXXXXXXXXX
   FP3: XXXXXXXXXXXXXXXX   FP4: XXXXXXXXXXXXXXXX   FP5: XXXXXXXXXXXXXXXX
   FP6: XXXXXXXXXXXXXXXX   FP7: XXXXXXXXXXXXXXXX   FP8: XXXXXXXXXXXXXXXX
   FP9: XXXXXXXXXXXXXXXX  FP10: XXXXXXXXXXXXXXXX  FP11: XXXXXXXXXXXXXXXX
  FP12: XXXXXXXXXXXXXXXX  FP13: XXXXXXXXXXXXXXXX  FP14: XXXXXXXXXXXXXXXX
  FP15: XXXXXXXXXXXXXXXX  FP16: XXXXXXXXXXXXXXXX  FP17: XXXXXXXXXXXXXXXX
  FP18: XXXXXXXXXXXXXXXX  FP19: XXXXXXXXXXXXXXXX  FP20: XXXXXXXXXXXXXXXX
  FP21: XXXXXXXXXXXXXXXX  FP22: XXXXXXXXXXXXXXXX  FP23: XXXXXXXXXXXXXXXX
  FP24: XXXXXXXXXXXXXXXX  FP25: XXXXXXXXXXXXXXXX  FP26: XXXXXXXXXXXXXXXX
  FP27: XXXXXXXXXXXXXXXX  FP28: XXXXXXXXXXXXXXXX  FP29: XXXXXXXXXXXXXXXX
  FP30: XXXXXXXXXXXXXXXX  FPCR: XXXXXXXXXXXXXXXX

   TA0: XXXXXXXXXXXXXXXX   TA1: XXXXXXXXXXXXXXXX   TA2: XXXXXXXXXXXXXXXX
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
  struct iovec iov[31 * 2 + 2    /* REGS + PC.  */
                   + 31 * 2 + 2  /* FREGS + FPCR.  */
                   + (3 * 2)     /* TA0, TA1, TA3.  */
                   + 1           /* '\n'.  */];
  size_t nr = 0;

#define ADD_STRING(str) \
  iov[nr].iov_base = (char *) str;					      \
  iov[nr].iov_len = strlen (str);					      \
  ++nr
#define ADD_MEM(str, len) \
  iov[nr].iov_base = str;						      \
  iov[nr].iov_len = len;						      \
  ++nr

  char regs[31][16];
  char pc[16];
  for (int i = 0; i < 31; i++)
    hexvalue (ctx->uc_mcontext.sc_regs[i], regs[i], 16);
  hexvalue (ctx->uc_mcontext.sc_pc, pc, 16);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n    V0: ");
  ADD_MEM (regs[0], 16);
  ADD_STRING ("    T0: ");
  ADD_MEM (regs[1], 16);
  ADD_STRING ("    T1: ");
  ADD_MEM (regs[2], 16);
  ADD_STRING ("\n    T2: ");
  ADD_MEM (regs[3], 16);
  ADD_STRING ("    T3: ");
  ADD_MEM (regs[4], 16);
  ADD_STRING ("    T4: ");
  ADD_MEM (regs[5], 16);
  ADD_STRING ("\n    T5: ");
  ADD_MEM (regs[6], 16);
  ADD_STRING ("    T6: ");
  ADD_MEM (regs[7], 16);
  ADD_STRING ("    T7: ");
  ADD_MEM (regs[8], 16);
  ADD_STRING ("\n    S0: ");
  ADD_MEM (regs[9], 16);
  ADD_STRING ("    S1: ");
  ADD_MEM (regs[10], 16);
  ADD_STRING ("    S2: ");
  ADD_MEM (regs[11], 16);
  ADD_STRING ("\n    S3: ");
  ADD_MEM (regs[12], 16);
  ADD_STRING ("    S4: ");
  ADD_MEM (regs[13], 16);
  ADD_STRING ("    S5: ");
  ADD_MEM (regs[14], 16);
  ADD_STRING ("\n    S6: ");
  ADD_MEM (regs[15], 16);
  ADD_STRING ("    A0: ");
  ADD_MEM (regs[16], 16);
  ADD_STRING ("    A1: ");
  ADD_MEM (regs[17], 16);
  ADD_STRING ("\n    A2: ");
  ADD_MEM (regs[18], 16);
  ADD_STRING ("    A3: ");
  ADD_MEM (regs[19], 16);
  ADD_STRING ("    A4: ");
  ADD_MEM (regs[20], 16);
  ADD_STRING ("\n    A5: ");
  ADD_MEM (regs[21], 16);
  ADD_STRING ("    T8: ");
  ADD_MEM (regs[22], 16);
  ADD_STRING ("    T9: ");
  ADD_MEM (regs[23], 16);
  ADD_STRING ("\n   T10: ");
  ADD_MEM (regs[24], 16);
  ADD_STRING ("   T11: ");
  ADD_MEM (regs[25], 16);
  ADD_STRING ("    RA: ");
  ADD_MEM (regs[26], 16);
  ADD_STRING ("\n   T12: ");
  ADD_MEM (regs[27], 16);
  ADD_STRING ("    AT: ");
  ADD_MEM (regs[28], 16);
  ADD_STRING ("    GP: ");
  ADD_MEM (regs[29], 16);
  ADD_STRING ("\n    SP: ");
  ADD_MEM (regs[30], 16);
  ADD_STRING ("    PC: ");
  ADD_MEM (pc, 16);

  char fpregs[31][16];
  char fpcr[16];
  for (int i = 0; i < 31; i++)
    hexvalue (ctx->uc_mcontext.sc_fpregs[i], fpregs[i], 16);
  hexvalue (ctx->uc_mcontext.sc_fpcr, fpcr, 16);

  ADD_STRING ("\n\n   FP0: ");
  ADD_MEM (fpregs[0], 16);
  ADD_STRING ("   FP1: ");
  ADD_MEM (fpregs[1], 16);
  ADD_STRING ("   FP2: ");
  ADD_MEM (fpregs[2], 16);
  ADD_STRING ("\n   FP3: ");
  ADD_MEM (fpregs[3], 16);
  ADD_STRING ("   FP4: ");
  ADD_MEM (fpregs[4], 16);
  ADD_STRING ("   FP5: ");
  ADD_MEM (fpregs[5], 16);
  ADD_STRING ("\n   FP6: ");
  ADD_MEM (fpregs[6], 16);
  ADD_STRING ("   FP7: ");
  ADD_MEM (fpregs[7], 16);
  ADD_STRING ("   FP8: ");
  ADD_MEM (fpregs[8], 16);
  ADD_STRING ("\n   FP9: ");
  ADD_MEM (fpregs[9], 16);
  ADD_STRING ("  FP10: ");
  ADD_MEM (fpregs[10], 16);
  ADD_STRING ("  FP11: ");
  ADD_MEM (fpregs[11], 16);
  ADD_STRING ("\n  FP12: ");
  ADD_MEM (fpregs[12], 16);
  ADD_STRING ("  FP13: ");
  ADD_MEM (fpregs[13], 16);
  ADD_STRING ("  FP14: ");
  ADD_MEM (fpregs[14], 16);
  ADD_STRING ("\n  FP15: ");
  ADD_MEM (fpregs[15], 16);
  ADD_STRING ("  FP16: ");
  ADD_MEM (fpregs[16], 16);
  ADD_STRING ("  FP17: ");
  ADD_MEM (fpregs[17], 16);
  ADD_STRING ("\n  FP18: ");
  ADD_MEM (fpregs[18], 16);
  ADD_STRING ("  FP19: ");
  ADD_MEM (fpregs[19], 16);
  ADD_STRING ("  FP20: ");
  ADD_MEM (fpregs[20], 16);
  ADD_STRING ("\n  FP21: ");
  ADD_MEM (fpregs[21], 16);
  ADD_STRING ("  FP22: ");
  ADD_MEM (fpregs[22], 16);
  ADD_STRING ("  FP23: ");
  ADD_MEM (fpregs[23], 16);
  ADD_STRING ("\n  FP24: ");
  ADD_MEM (fpregs[24], 16);
  ADD_STRING ("  FP25: ");
  ADD_MEM (fpregs[25], 16);
  ADD_STRING ("  FP26: ");
  ADD_MEM (fpregs[26], 16);
  ADD_STRING ("\n  FP27: ");
  ADD_MEM (fpregs[27], 16);
  ADD_STRING ("  FP28: ");
  ADD_MEM (fpregs[28], 16);
  ADD_STRING ("  FP29: ");
  ADD_MEM (fpregs[29], 16);
  ADD_STRING ("\n  FP30: ");
  ADD_MEM (fpregs[30], 16);
  ADD_STRING ("  FPCR: ");
  ADD_MEM (fpcr, 16);

  char traparg[3][16];
  hexvalue (ctx->uc_mcontext.sc_traparg_a0, traparg[0], 16);
  hexvalue (ctx->uc_mcontext.sc_traparg_a1, traparg[1], 16);
  hexvalue (ctx->uc_mcontext.sc_traparg_a2, traparg[2], 16);
  ADD_STRING ("\n\n   TA0: ");
  ADD_MEM (traparg[0], 16);
  ADD_STRING ("   TA1: ");
  ADD_MEM (traparg[1], 16);
  ADD_STRING ("   TA2: ");
  ADD_MEM (traparg[2], 16);

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}

#define REGISTER_DUMP register_dump (fd, ctx)

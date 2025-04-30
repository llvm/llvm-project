/* Dump registers.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <sys/uio.h>
#include <_itoa.h>
#include <bits/sigcontext.h>
#include <sys/ucontext.h>

/* abiv1 register dump in this format:

 PSR: XXXXXXXX  PC: XXXXXXXX   SP: XXXXXXXX   LR: XXXXXXXX
 MASK: XXXXXXXX

 A0: XXXXXXXX   A1: XXXXXXXX   A2: XXXXXXXX   A3: XXXXXXXX
 R6: XXXXXXXX   R7: XXXXXXXX   R8: XXXXXXXX   R9: XXXXXXXX
 R10: XXXXXXXX  R11: XXXXXXXX  R12: XXXXXXXX  R13: XXXXXXXX
 R14: XXXXXXXX  R1: XXXXXXXX

 abiv2 register dump in this format:

 PSR: XXXXXXXX  PC: XXXXXXXX   SP: XXXXXXXX   LR: XXXXXXXX
 MASK: XXXXXXXX

 A0: XXXXXXXX   A1: XXXXXXXX   A2: XXXXXXXX   A3: XXXXXXXX
 R4: XXXXXXXX   R5: XXXXXXXX   R6: XXXXXXXX   R7: XXXXXXXX
 R8: XXXXXXXX   R9: XXXXXXXX   R10: XXXXXXXX  R11: XXXXXXXX
 R12: XXXXXXXX  R13: XXXXXXXX  R14: XXXXXXXX  R15: XXXXXXXX
 R16: XXXXXXXX  R17: XXXXXXXX  R18: XXXXXXXX  R19: XXXXXXXX
 R20: XXXXXXXX  R21: XXXXXXXX  R22: XXXXXXXX  R23: XXXXXXXX
 R24: XXXXXXXX  R25: XXXXXXXX  R26: XXXXXXXX  R27: XXXXXXXX
 R28: XXXXXXXX  R29: XXXXXXXX  R30: XXXXXXXX  R31: XXXXXXXX

 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
register_dump (int fd, const struct ucontext_t *ctx)
{
  char regs[35][8];
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
  hexvalue (ctx->uc_mcontext.__gregs.__sr, regs[0], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__pc, regs[1], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__usp, regs[2], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__lr, regs[3], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__a0, regs[4], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__a1, regs[5], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__a2, regs[6], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__a3, regs[7], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[0], regs[8], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[1], regs[9], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[2], regs[10], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[3], regs[11], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[4], regs[12], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[5], regs[13], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[6], regs[14], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[7], regs[15], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[8], regs[16], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__regs[9], regs[17], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[0], regs[18], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[1], regs[19], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[2], regs[20], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[3], regs[21], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[4], regs[22], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[5], regs[23], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[6], regs[24], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[7], regs[25], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[8], regs[26], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[9], regs[27], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[10], regs[28], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[11], regs[29], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[12], regs[30], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[13], regs[31], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__exregs[14], regs[32], 8);
  hexvalue (ctx->uc_mcontext.__gregs.__tls, regs[33], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n PSR: ");
  ADD_MEM (regs[0], 8);
  ADD_STRING ("  PC: ");
  ADD_MEM (regs[1], 8);
  ADD_STRING ("   SP: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("   LR: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n\n A0: ");
  ADD_MEM (regs[4], 8);
  ADD_STRING ("   A1: ");
  ADD_MEM (regs[5], 8);
  ADD_STRING ("   A2: ");
  ADD_MEM (regs[6], 8);
  ADD_STRING ("   A3: ");
  ADD_MEM (regs[7], 8);
  ADD_STRING ("\n R4: ");
  ADD_MEM (regs[8], 8);
  ADD_STRING ("   R5: ");
  ADD_MEM (regs[9], 8);
  ADD_STRING ("   R6: ");
  ADD_MEM (regs[10], 8);
  ADD_STRING ("   R7: ");
  ADD_MEM (regs[11], 8);
  ADD_STRING ("\n R8: ");
  ADD_MEM (regs[12], 8);
  ADD_STRING ("   R9: ");
  ADD_MEM (regs[13], 8);
  ADD_STRING ("   R10: ");
  ADD_MEM (regs[14], 8);
  ADD_STRING ("  R11: ");
  ADD_MEM (regs[15], 8);
  ADD_STRING ("\n R12: ");
  ADD_MEM (regs[16], 8);
  ADD_STRING ("  R13: ");
  ADD_MEM (regs[17], 8);
  ADD_STRING ("  R14: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("  R15: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n R16: ");
  ADD_MEM (regs[18], 8);
  ADD_STRING ("  R17: ");
  ADD_MEM (regs[19], 8);
  ADD_STRING ("  R18: ");
  ADD_MEM (regs[20], 8);
  ADD_STRING ("  R19: ");
  ADD_MEM (regs[21], 8);
  ADD_STRING ("\n R20: ");
  ADD_MEM (regs[22], 8);
  ADD_STRING ("  R21: ");
  ADD_MEM (regs[23], 8);
  ADD_STRING ("  R22: ");
  ADD_MEM (regs[24], 8);
  ADD_STRING ("  R23: ");
  ADD_MEM (regs[25], 8);
  ADD_STRING ("\n R24: ");
  ADD_MEM (regs[26], 8);
  ADD_STRING ("  R25: ");
  ADD_MEM (regs[27], 8);
  ADD_STRING ("  R26: ");
  ADD_MEM (regs[28], 8);
  ADD_STRING ("  R27: ");
  ADD_MEM (regs[29], 8);
  ADD_STRING ("\n R28: ");
  ADD_MEM (regs[30], 8);
  ADD_STRING ("  R29: ");
  ADD_MEM (regs[31], 8);
  ADD_STRING ("  R30: ");
  ADD_MEM (regs[32], 8);
  ADD_STRING ("  R31: ");
  ADD_MEM (regs[33], 8);

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}

#define REGISTER_DUMP register_dump (fd, ctx)

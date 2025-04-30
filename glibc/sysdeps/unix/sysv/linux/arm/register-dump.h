/* Dump registers.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <pb@nexus.co.uk>, 1998.

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
#include <sys/ucontext.h>

/* We will print the register dump in this format:

 R0: XXXXXXXX   R1: XXXXXXXX   R2: XXXXXXXX   R3: XXXXXXXX
 R4: XXXXXXXX   R5: XXXXXXXX   R6: XXXXXXXX   R7: XXXXXXXX
 R8: XXXXXXXX   R9: XXXXXXXX   SL: XXXXXXXX   FP: XXXXXXXX
 IP: XXXXXXXX   SP: XXXXXXXX   LR: XXXXXXXX   PC: XXXXXXXX

 CPSR: XXXXXXXX

 Trap: XXXXXXXX   Error: XXXXXXXX   OldMask: XXXXXXXX
 Addr: XXXXXXXX

 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
register_dump (int fd, const ucontext_t *ctx)
{
  char regs[21][8];
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
  hexvalue (ctx->uc_mcontext.arm_r0, regs[0], 8);
  hexvalue (ctx->uc_mcontext.arm_r1, regs[1], 8);
  hexvalue (ctx->uc_mcontext.arm_r2, regs[2], 8);
  hexvalue (ctx->uc_mcontext.arm_r3, regs[3], 8);
  hexvalue (ctx->uc_mcontext.arm_r4, regs[4], 8);
  hexvalue (ctx->uc_mcontext.arm_r5, regs[5], 8);
  hexvalue (ctx->uc_mcontext.arm_r6, regs[6], 8);
  hexvalue (ctx->uc_mcontext.arm_r7, regs[7], 8);
  hexvalue (ctx->uc_mcontext.arm_r8, regs[8], 8);
  hexvalue (ctx->uc_mcontext.arm_r9, regs[9], 8);
  hexvalue (ctx->uc_mcontext.arm_r10, regs[10], 8);
  hexvalue (ctx->uc_mcontext.arm_fp, regs[11], 8);
  hexvalue (ctx->uc_mcontext.arm_ip, regs[12], 8);
  hexvalue (ctx->uc_mcontext.arm_sp, regs[13], 8);
  hexvalue (ctx->uc_mcontext.arm_lr, regs[14], 8);
  hexvalue (ctx->uc_mcontext.arm_pc, regs[15], 8);
  hexvalue (ctx->uc_mcontext.arm_cpsr, regs[16], 8);
  hexvalue (ctx->uc_mcontext.trap_no, regs[17], 8);
  hexvalue (ctx->uc_mcontext.error_code, regs[18], 8);
  hexvalue (ctx->uc_mcontext.oldmask, regs[19], 8);
  hexvalue (ctx->uc_mcontext.fault_address, regs[20], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n R0: ");
  ADD_MEM (regs[0], 8);
  ADD_STRING ("   R1: ");
  ADD_MEM (regs[1], 8);
  ADD_STRING ("   R2: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("   R3: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n R4: ");
  ADD_MEM (regs[4], 8);
  ADD_STRING ("   R5: ");
  ADD_MEM (regs[5], 8);
  ADD_STRING ("   R6: ");
  ADD_MEM (regs[6], 8);
  ADD_STRING ("   R7: ");
  ADD_MEM (regs[7], 8);
  ADD_STRING ("\n R8: ");
  ADD_MEM (regs[8], 8);
  ADD_STRING ("   R9: ");
  ADD_MEM (regs[9], 8);
  ADD_STRING ("   SL: ");
  ADD_MEM (regs[10], 8);
  ADD_STRING ("   FP: ");
  ADD_MEM (regs[11], 8);
  ADD_STRING ("\n IP: ");
  ADD_MEM (regs[12], 8);
  ADD_STRING ("   SP: ");
  ADD_MEM (regs[13], 8);
  ADD_STRING ("   LR: ");
  ADD_MEM (regs[14], 8);
  ADD_STRING ("   PC: ");
  ADD_MEM (regs[15], 8);
  ADD_STRING ("\n\n CPSR: ");
  ADD_MEM (regs[16], 8);
  ADD_STRING ("\n\n Trap: ");
  ADD_MEM (regs[17], 8);
  ADD_STRING ("   Error: ");
  ADD_MEM (regs[18], 8);
  ADD_STRING ("   OldMask: ");
  ADD_MEM (regs[19], 8);
  ADD_STRING ("\n Addr: ");
  ADD_MEM (regs[20], 8);

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

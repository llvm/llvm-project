/* Dump registers.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>, 1999.

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

 TSTATE: XXXXXXXXXXXXXXXX TPC: XXXXXXXXXXXXXXXX TNPC: XXXXXXXXXXXXXXXX
 Y: XXXXXXXX
 g0: 0000000000000000  g1: XXXXXXXXXXXXXXXX  g2: XXXXXXXXXXXXXXXX
 g3: XXXXXXXXXXXXXXXX  g4: XXXXXXXXXXXXXXXX  g5: XXXXXXXXXXXXXXXX
 g6: XXXXXXXXXXXXXXXX  g7: XXXXXXXXXXXXXXXX
 o0: XXXXXXXXXXXXXXXX  o1: XXXXXXXXXXXXXXXX  o2: XXXXXXXXXXXXXXXX
 o3: XXXXXXXXXXXXXXXX  o4: XXXXXXXXXXXXXXXX  o5: XXXXXXXXXXXXXXXX
 sp: XXXXXXXXXXXXXXXX  o7: XXXXXXXXXXXXXXXX
 l0: XXXXXXXXXXXXXXXX  l1: XXXXXXXXXXXXXXXX  l2: XXXXXXXXXXXXXXXX
 l3: XXXXXXXXXXXXXXXX  l4: XXXXXXXXXXXXXXXX  l5: XXXXXXXXXXXXXXXX
 l6: XXXXXXXXXXXXXXXX  l7: XXXXXXXXXXXXXXXX
 i0: XXXXXXXXXXXXXXXX  i1: XXXXXXXXXXXXXXXX  i2: XXXXXXXXXXXXXXXX
 i3: XXXXXXXXXXXXXXXX  i4: XXXXXXXXXXXXXXXX  i5: XXXXXXXXXXXXXXXX
 fp: XXXXXXXXXXXXXXXX  i7: XXXXXXXXXXXXXXXX

 Mask: XXXXXXXXXXXXXXXX XFSR: XXXXXXXXXXXXXXXX GSR: XX FPRS: X
  f0: XXXXXXXXXXXXXXXX   f2: XXXXXXXXXXXXXXXX   f4: XXXXXXXXXXXXXXXX
  f6: XXXXXXXXXXXXXXXX   f8: XXXXXXXXXXXXXXXX  f10: XXXXXXXXXXXXXXXX
 f12: XXXXXXXXXXXXXXXX  f14: XXXXXXXXXXXXXXXX  f16: XXXXXXXXXXXXXXXX
 f18: XXXXXXXXXXXXXXXX  f20: XXXXXXXXXXXXXXXX  f22: XXXXXXXXXXXXXXXX
 f24: XXXXXXXXXXXXXXXX  f26: XXXXXXXXXXXXXXXX  f28: XXXXXXXXXXXXXXXX
 f30: XXXXXXXXXXXXXXXX  f32: XXXXXXXXXXXXXXXX  f34: XXXXXXXXXXXXXXXX
 f36: XXXXXXXXXXXXXXXX  f38: XXXXXXXXXXXXXXXX  f40: XXXXXXXXXXXXXXXX
 f42: XXXXXXXXXXXXXXXX  f44: XXXXXXXXXXXXXXXX  f46: XXXXXXXXXXXXXXXX
 f48: XXXXXXXXXXXXXXXX  f50: XXXXXXXXXXXXXXXX  f52: XXXXXXXXXXXXXXXX
 f54: XXXXXXXXXXXXXXXX  f56: XXXXXXXXXXXXXXXX  f58: XXXXXXXXXXXXXXXX
 f60: XXXXXXXXXXXXXXXX  f62: XXXXXXXXXXXXXXXX

 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

/* The sparc64 kernel signal frame for SA_SIGINFO is defined as:

   struct rt_signal_frame
     {
       struct sparc_stackf ss;
       siginfo_t info;
       struct pt_regs regs;          <- void *ctx
       __siginfo_fpu_t *fpu_save;
       stack_t stack;
       sigset_t mask;
       __siginfo_rwin_t *rwin_save;
     };

  Unlike other architectures, sparc32 passes pt_regs32 REGS pointers as
  the third argument to a sa_sigaction handler with SA_SIGINFO enabled.  */

static void
register_dump (int fd, void *ctx)
{
  char regs[36][16];
  char fregs[68][8];
  struct iovec iov[150];
  size_t nr = 0;
  int i;
  struct pt_regs *ptregs = (struct pt_regs*) ((siginfo_t *)ctx + 1);
  unsigned long *r = (unsigned long *) (ptregs->u_regs[14] + STACK_BIAS);
  __siginfo_fpu_t *f = (__siginfo_fpu_t *)(ptregs + 1);
  struct kernel_sigset_t {
    unsigned long sig[1];
  } *mask = (struct kernel_sigset_t *)((stack_t *)(f + 1) + 1);

#define ADD_STRING(str) \
  iov[nr].iov_base = (char *) str;					      \
  iov[nr].iov_len = strlen (str);					      \
  ++nr
#define ADD_MEM(str, len) \
  iov[nr].iov_base = str;						      \
  iov[nr].iov_len = len;						      \
  ++nr

  /* Generate strings of register contents.  */
  hexvalue (ptregs->tstate, regs[0], 16);
  hexvalue (ptregs->tpc, regs[1], 16);
  hexvalue (ptregs->tnpc, regs[2], 16);
  hexvalue (ptregs->y, regs[3], 8);
  for (i = 1; i <= 15; i++)
    hexvalue (ptregs->u_regs[i], regs[3+i], 16);
  for (i = 0; i <= 15; i++)
    hexvalue (r[i], regs[19+i], 16);
  hexvalue (mask->sig[0], regs[35], 16);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n TSTATE: ");
  ADD_MEM (regs[0], 16);
  ADD_STRING (" TPC: ");
  ADD_MEM (regs[1], 16);
  ADD_STRING (" TNPC: ");
  ADD_MEM (regs[2], 16);
  ADD_STRING ("\n Y: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n  g0: 0000000000000000   g1: ");
  ADD_MEM (regs[4], 16);
  ADD_STRING ("  g2: ");
  ADD_MEM (regs[5], 16);
  ADD_STRING ("\n g3: ");
  ADD_MEM (regs[6], 16);
  ADD_STRING ("  g4: ");
  ADD_MEM (regs[7], 16);
  ADD_STRING ("  g5: ");
  ADD_MEM (regs[8], 16);
  ADD_STRING ("\n g6: ");
  ADD_MEM (regs[9], 16);
  ADD_STRING ("  g7: ");
  ADD_MEM (regs[10], 16);
  ADD_STRING ("\n o0: ");
  ADD_MEM (regs[11], 16);
  ADD_STRING ("  o1: ");
  ADD_MEM (regs[12], 16);
  ADD_STRING ("  o2: ");
  ADD_MEM (regs[13], 16);
  ADD_STRING ("\n o3: ");
  ADD_MEM (regs[14], 16);
  ADD_STRING ("  o4: ");
  ADD_MEM (regs[15], 16);
  ADD_STRING ("  o5: ");
  ADD_MEM (regs[16], 16);
  ADD_STRING ("\n sp: ");
  ADD_MEM (regs[17], 16);
  ADD_STRING ("  o7: ");
  ADD_MEM (regs[18], 16);
  ADD_STRING ("\n l0: ");
  ADD_MEM (regs[19], 16);
  ADD_STRING ("  l1: ");
  ADD_MEM (regs[20], 16);
  ADD_STRING ("  l2: ");
  ADD_MEM (regs[21], 16);
  ADD_STRING ("\n l3: ");
  ADD_MEM (regs[22], 16);
  ADD_STRING ("  l4: ");
  ADD_MEM (regs[23], 16);
  ADD_STRING ("  l5: ");
  ADD_MEM (regs[24], 16);
  ADD_STRING ("\n l6: ");
  ADD_MEM (regs[25], 16);
  ADD_STRING ("  l7: ");
  ADD_MEM (regs[26], 16);
  ADD_STRING ("\n i0: ");
  ADD_MEM (regs[27], 16);
  ADD_STRING ("  i1: ");
  ADD_MEM (regs[28], 16);
  ADD_STRING ("  i2: ");
  ADD_MEM (regs[29], 16);
  ADD_STRING ("\n i3: ");
  ADD_MEM (regs[30], 16);
  ADD_STRING ("  i4: ");
  ADD_MEM (regs[31], 16);
  ADD_STRING ("  i5: ");
  ADD_MEM (regs[32], 16);
  ADD_STRING ("\n fp: ");
  ADD_MEM (regs[33], 16);
  ADD_STRING ("  i7: ");
  ADD_MEM (regs[34], 16);
  ADD_STRING ("\n\n Mask: ");
  ADD_MEM (regs[35], 16);

  if (f != NULL)
    {
      for (i = 0; i < 64; i++)
	hexvalue (f->si_float_regs[i], fregs[i], 8);
      hexvalue (f->si_fsr, fregs[64], 16);
      hexvalue (f->si_gsr, fregs[66], 2);
      hexvalue (f->si_fprs, fregs[67], 1);
      ADD_STRING (" XFSR: ");
      ADD_MEM (fregs[64], 16);
      ADD_STRING (" GSR: ");
      ADD_MEM (fregs[66], 2);
      ADD_STRING (" FPRS: ");
      ADD_MEM (fregs[67], 1);
      ADD_STRING ("\n  f0: ");
      ADD_MEM (fregs[0], 16);
      ADD_STRING ("   f2: ");
      ADD_MEM (fregs[2], 16);
      ADD_STRING ("   f4: ");
      ADD_MEM (fregs[4], 16);
      ADD_STRING ("\n  f6: ");
      ADD_MEM (fregs[6], 16);
      ADD_STRING ("   f8: ");
      ADD_MEM (fregs[8], 16);
      ADD_STRING ("  f10: ");
      ADD_MEM (fregs[10], 16);
      ADD_STRING ("\n f12: ");
      ADD_MEM (fregs[12], 16);
      ADD_STRING ("  f14: ");
      ADD_MEM (fregs[14], 16);
      ADD_STRING ("  f16: ");
      ADD_MEM (fregs[16], 16);
      ADD_STRING ("\n f18: ");
      ADD_MEM (fregs[18], 16);
      ADD_STRING ("  f20: ");
      ADD_MEM (fregs[20], 16);
      ADD_STRING ("  f22: ");
      ADD_MEM (fregs[22], 16);
      ADD_STRING ("\n f24: ");
      ADD_MEM (fregs[24], 16);
      ADD_STRING ("  f26: ");
      ADD_MEM (fregs[26], 16);
      ADD_STRING ("  f28: ");
      ADD_MEM (fregs[28], 16);
      ADD_STRING ("\n f30: ");
      ADD_MEM (fregs[30], 16);
      ADD_STRING ("  f32: ");
      ADD_MEM (fregs[32], 16);
      ADD_STRING ("  f34: ");
      ADD_MEM (fregs[34], 16);
      ADD_STRING ("\n f36: ");
      ADD_MEM (fregs[36], 16);
      ADD_STRING ("  f38: ");
      ADD_MEM (fregs[38], 16);
      ADD_STRING ("  f40: ");
      ADD_MEM (fregs[40], 16);
      ADD_STRING ("\n f42: ");
      ADD_MEM (fregs[42], 16);
      ADD_STRING ("  f44: ");
      ADD_MEM (fregs[44], 16);
      ADD_STRING ("  f46: ");
      ADD_MEM (fregs[46], 16);
      ADD_STRING ("\n f48: ");
      ADD_MEM (fregs[48], 16);
      ADD_STRING ("  f50: ");
      ADD_MEM (fregs[50], 16);
      ADD_STRING ("  f52: ");
      ADD_MEM (fregs[52], 16);
      ADD_STRING ("\n f54: ");
      ADD_MEM (fregs[54], 16);
      ADD_STRING ("  f56: ");
      ADD_MEM (fregs[56], 16);
      ADD_STRING ("  f58: ");
      ADD_MEM (fregs[58], 16);
      ADD_STRING ("\n f60: ");
      ADD_MEM (fregs[60], 16);
      ADD_STRING ("  f62: ");
      ADD_MEM (fregs[62], 16);
    }

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

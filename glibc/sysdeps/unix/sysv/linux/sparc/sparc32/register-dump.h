/* Dump registers.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 1999.

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

 PSR: XXXXXXXX PC: XXXXXXXX NPC: XXXXXXXX   Y: XXXXXXXX
 g0: 00000000  g1: XXXXXXXX  g2: XXXXXXXX  g3: XXXXXXXX
 g4: XXXXXXXX  g5: XXXXXXXX  g6: XXXXXXXX  g7: XXXXXXXX
 o0: XXXXXXXX  o1: XXXXXXXX  o2: XXXXXXXX  o3: XXXXXXXX
 o4: XXXXXXXX  o5: XXXXXXXX  sp: XXXXXXXX  o7: XXXXXXXX
 l0: XXXXXXXX  l1: XXXXXXXX  l2: XXXXXXXX  l3: XXXXXXXX
 l4: XXXXXXXX  l5: XXXXXXXX  l6: XXXXXXXX  l7: XXXXXXXX
 i0: XXXXXXXX  i1: XXXXXXXX  i2: XXXXXXXX  i3: XXXXXXXX
 i4: XXXXXXXX  i5: XXXXXXXX  fp: XXXXXXXX  i7: XXXXXXXX

 followed on sun4, sun4c, sun4d, sun4m by:

 Old mask: XXXXXXXX FSR: XXXXXXXX FPQ: XXXXXXXX
  f0: XXXXXXXXXXXXXXXX   f2: XXXXXXXXXXXXXXXX   f4: XXXXXXXXXXXXXXXX
  f6: XXXXXXXXXXXXXXXX   f8: XXXXXXXXXXXXXXXX  f10: XXXXXXXXXXXXXXXX
 f12: XXXXXXXXXXXXXXXX  f14: XXXXXXXXXXXXXXXX  f16: XXXXXXXXXXXXXXXX
 f18: XXXXXXXXXXXXXXXX  f20: XXXXXXXXXXXXXXXX  f22: XXXXXXXXXXXXXXXX
 f24: XXXXXXXXXXXXXXXX  f26: XXXXXXXXXXXXXXXX  f28: XXXXXXXXXXXXXXXX
 f30: XXXXXXXXXXXXXXXX

 and on sun4u by:

 Old mask: XXXXXXXX XFSR: XXXXXXXXXXXXXXXX GSR: XX FPRS: X
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

struct __siginfo_sparc32_fpu
{
  unsigned int si_float_regs[32];
  unsigned int si_fsr;
  unsigned int si_fpq;
};
struct __siginfo_sparc64_fpu
{
  unsigned int si_float_regs[64];
  unsigned int si_xfsr;
  unsigned int si_fsr;
  unsigned int _pad1;
  unsigned int si_gsr;
  unsigned int _pad2;
  unsigned int si_fprs;
};

/* Unlike other architectures, sparc32 passes pt_regs32 REGS pointer as
   the third argument to a sa_sigaction handler with SA_SIGINFO enabled.  */
static void
register_dump (int fd, void *ctx)
{
  char regs[36][8];
  char fregs[68][8];
  struct iovec iov[150];
  size_t nr = 0;
  int i;
  struct pt_regs32 *ptregs = (struct pt_regs32 *) ctx;
  struct compat_sigset_t
  {
    unsigned int sig[2];
  };
  struct compat_sigset_t *mask = (struct compat_sigset_t *)(ptregs + 1);
  unsigned int *r = (unsigned int *) ptregs->u_regs[14];

#define ADD_STRING(str) \
  iov[nr].iov_base = (char *) str;					      \
  iov[nr].iov_len = strlen (str);					      \
  ++nr
#define ADD_MEM(str, len) \
  iov[nr].iov_base = str;						      \
  iov[nr].iov_len = len;						      \
  ++nr

  /* Generate strings of register contents.  */
  hexvalue (ptregs->psr, regs[0], 8);
  hexvalue (ptregs->pc, regs[1], 8);
  hexvalue (ptregs->npc, regs[2], 8);
  hexvalue (ptregs->y, regs[3], 8);
  for (i = 1; i <= 15; i++)
    hexvalue (ptregs->u_regs[i], regs[3+i], 8);
  for (i = 0; i <= 15; i++)
    hexvalue (r[i], regs[19+i], 8);

  hexvalue (mask->sig[0], regs[35], 8);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n PSR: ");
  ADD_MEM (regs[0], 8);
  ADD_STRING (" PC: ");
  ADD_MEM (regs[1], 8);
  ADD_STRING (" NPC: ");
  ADD_MEM (regs[2], 8);
  ADD_STRING ("   Y: ");
  ADD_MEM (regs[3], 8);
  ADD_STRING ("\n g0: 00000000 g1: ");
  ADD_MEM (regs[4], 8);
  ADD_STRING ("  g2: ");
  ADD_MEM (regs[5], 8);
  ADD_STRING ("  g3: ");
  ADD_MEM (regs[6], 8);
  ADD_STRING ("\n g4: ");
  ADD_MEM (regs[7], 8);
  ADD_STRING ("  g5: ");
  ADD_MEM (regs[8], 8);
  ADD_STRING ("  g6: ");
  ADD_MEM (regs[9], 8);
  ADD_STRING ("  g7: ");
  ADD_MEM (regs[10], 8);
  ADD_STRING ("\n o0: ");
  ADD_MEM (regs[11], 8);
  ADD_STRING ("  o1: ");
  ADD_MEM (regs[12], 8);
  ADD_STRING ("  o2: ");
  ADD_MEM (regs[13], 8);
  ADD_STRING ("  o3: ");
  ADD_MEM (regs[14], 8);
  ADD_STRING ("\n o4: ");
  ADD_MEM (regs[15], 8);
  ADD_STRING ("  o5: ");
  ADD_MEM (regs[16], 8);
  ADD_STRING ("  sp: ");
  ADD_MEM (regs[17], 8);
  ADD_STRING ("  o7: ");
  ADD_MEM (regs[18], 8);
  ADD_STRING ("\n l0: ");
  ADD_MEM (regs[19], 8);
  ADD_STRING ("  l1: ");
  ADD_MEM (regs[20], 8);
  ADD_STRING ("  l2: ");
  ADD_MEM (regs[21], 8);
  ADD_STRING ("  l3: ");
  ADD_MEM (regs[22], 8);
  ADD_STRING ("\n l4: ");
  ADD_MEM (regs[23], 8);
  ADD_STRING ("  l5: ");
  ADD_MEM (regs[24], 8);
  ADD_STRING ("  l6: ");
  ADD_MEM (regs[25], 8);
  ADD_STRING ("  l7: ");
  ADD_MEM (regs[26], 8);
  ADD_STRING ("\n i0: ");
  ADD_MEM (regs[27], 8);
  ADD_STRING ("  i1: ");
  ADD_MEM (regs[28], 8);
  ADD_STRING ("  i2: ");
  ADD_MEM (regs[29], 8);
  ADD_STRING ("  i3: ");
  ADD_MEM (regs[30], 8);
  ADD_STRING ("\n i4: ");
  ADD_MEM (regs[31], 8);
  ADD_STRING ("  i5: ");
  ADD_MEM (regs[32], 8);
  ADD_STRING ("  fp: ");
  ADD_MEM (regs[33], 8);
  ADD_STRING ("  i7: ");
  ADD_MEM (regs[34], 8);
  ADD_STRING ("\n\n Old mask: ");
  ADD_MEM (regs[35], 8);

  if ((ptregs->psr & 0xff000000) == 0xff000000)
    {
      struct __siginfo_sparc64_fpu *f = *(struct __siginfo_sparc64_fpu **)
	(mask + 1);

      if (f != NULL)
	{
	  for (i = 0; i < 64; i++)
	    hexvalue (f->si_float_regs[i], fregs[i], 8);
	  hexvalue (f->si_xfsr, fregs[64], 8);
	  hexvalue (f->si_fsr, fregs[65], 8);
	  hexvalue (f->si_gsr, fregs[66], 2);
	  hexvalue (f->si_fprs, fregs[67], 1);
	  ADD_STRING (" XFSR: ");
	  ADD_MEM (fregs[64], 8);
	  ADD_MEM (fregs[65], 8);
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
    }
  else
    {
      struct __siginfo_sparc32_fpu *f = *(struct __siginfo_sparc32_fpu **)
	(mask + 1);

      if (f != NULL)
	{
	  for (i = 0; i < 32; i++)
	    hexvalue (f->si_float_regs[i], fregs[i], 8);
	  hexvalue (f->si_fsr, fregs[64], 8);
	  hexvalue (f->si_fpq, fregs[65], 8);
	  ADD_STRING (" FSR: ");
	  ADD_MEM (fregs[64], 8);
	  ADD_STRING (" FPQ: ");
	  ADD_MEM (fregs[65], 8);
	  ADD_STRING ("\n  f0: ");
	  ADD_MEM (fregs[0], 16);
	  ADD_STRING ("  f2: ");
	  ADD_MEM (fregs[2], 16);
	  ADD_STRING ("  f4: ");
	  ADD_MEM (fregs[4], 16);
	  ADD_STRING ("\n  f6: ");
	  ADD_MEM (fregs[6], 16);
	  ADD_STRING ("   f8: ");
	  ADD_MEM (fregs[8], 16);
	  ADD_STRING ("  f10: ");
	  ADD_MEM (fregs[10], 16);
	  ADD_STRING ("\n  f12: ");
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
	}
    }

  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

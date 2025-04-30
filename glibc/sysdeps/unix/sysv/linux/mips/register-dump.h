/* Dump registers.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2000.

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

#include <sgidefs.h>
#include <sys/uio.h>
#include <_itoa.h>

#if _MIPS_SIM == _ABIO32
# define CTX_TYPE	struct sigcontext *
# define CTX_REG(ctx, i)	((ctx)->sc_regs[(i)])
# define CTX_PC(ctx)	((ctx)->sc_pc)
# define CTX_MDHI(ctx)	((ctx)->sc_mdhi)
# define CTX_MDLO(ctx)	((ctx)->sc_mdlo)
# define REG_HEX_SIZE	8
#else
# define CTX_TYPE	ucontext_t *
# define CTX_REG(ctx, i)	((ctx)->uc_mcontext.gregs[(i)])
# define CTX_PC(ctx)	((ctx)->uc_mcontext.pc)
# define CTX_MDHI(ctx)	((ctx)->uc_mcontext.mdhi)
# define CTX_MDLO(ctx)	((ctx)->uc_mcontext.mdhi)
# define REG_HEX_SIZE	16
#endif

/* We will print the register dump in this format:

 R0   XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX
 R8   XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX
 R16  XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX
 R24  XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX
            pc       lo       hi
      XXXXXXXX XXXXXXXX XXXXXXXX
 The FPU registers will not be printed.
*/

static void
hexvalue (_ITOA_WORD_TYPE value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
register_dump (int fd, CTX_TYPE ctx)
{
  char regs[38][REG_HEX_SIZE];
  struct iovec iov[38 * 2 + 10];
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
  for (i = 0; i < 32; i++)
    hexvalue (CTX_REG (ctx, i), regs[i], REG_HEX_SIZE);
  hexvalue (CTX_PC (ctx), regs[32], REG_HEX_SIZE);
  hexvalue (CTX_MDHI (ctx), regs[33], REG_HEX_SIZE);
  hexvalue (CTX_MDLO (ctx), regs[34], REG_HEX_SIZE);

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n R0   ");
  for (i = 0; i < 8; i++)
    {
      ADD_MEM (regs[i], REG_HEX_SIZE);
      ADD_STRING (" ");
    }
  ADD_STRING ("\n R8   ");
  for (i = 8; i < 16; i++)
    {
      ADD_MEM (regs[i], REG_HEX_SIZE);
      ADD_STRING (" ");
    }
  ADD_STRING ("\n R16  ");
  for (i = 16; i < 24; i++)
    {
      ADD_MEM (regs[i], REG_HEX_SIZE);
      ADD_STRING (" ");
    }
  ADD_STRING ("\n R24  ");
  for (i = 24; i < 32; i++)
    {
      ADD_MEM (regs[i], REG_HEX_SIZE);
      ADD_STRING (" ");
    }
  ADD_STRING ("\n            pc       lo       hi\n      ");
  for (i = 32; i < 35; i++)
    {
      ADD_MEM (regs[i], REG_HEX_SIZE);
      ADD_STRING (" ");
    }
  ADD_STRING ("\n");

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

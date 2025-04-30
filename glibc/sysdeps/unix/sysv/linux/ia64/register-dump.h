/* Dump registers.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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

#include <string.h>
#include <sys/uio.h>
#include <_itoa.h>

/* We will print the register dump in this format:

 GP:   XXXXXXXXXXXXXXXX R2:   XXXXXXXXXXXXXXXX R3:   XXXXXXXXXXXXXXXX
 R8:   XXXXXXXXXXXXXXXX R9:   XXXXXXXXXXXXXXXX R10:  XXXXXXXXXXXXXXXX
 R11:  XXXXXXXXXXXXXXXX SP:   XXXXXXXXXXXXXXXX TP:   XXXXXXXXXXXXXXXX
 R14:  XXXXXXXXXXXXXXXX R15:  XXXXXXXXXXXXXXXX R16:  XXXXXXXXXXXXXXXX
 R17:  XXXXXXXXXXXXXXXX R18:  XXXXXXXXXXXXXXXX R19:  XXXXXXXXXXXXXXXX
 R20:  XXXXXXXXXXXXXXXX R21:  XXXXXXXXXXXXXXXX R22:  XXXXXXXXXXXXXXXX
 R23:  XXXXXXXXXXXXXXXX R24:  XXXXXXXXXXXXXXXX R25:  XXXXXXXXXXXXXXXX
 R26:  XXXXXXXXXXXXXXXX R27:  XXXXXXXXXXXXXXXX R28:  XXXXXXXXXXXXXXXX
 R29:  XXXXXXXXXXXXXXXX R30:  XXXXXXXXXXXXXXXX R31:  XXXXXXXXXXXXXXXX

 RP:   XXXXXXXXXXXXXXXX B6:   XXXXXXXXXXXXXXXX B7:   XXXXXXXXXXXXXXXX

 IP:   XXXXXXXXXXXXXXXX RSC:  XXXXXXXXXXXXXXXX PR:   XXXXXXXXXXXXXXXX
 PFS:  XXXXXXXXXXXXXXXX UNAT: XXXXXXXXXXXXXXXX CFM:  XXXXXXXXXXXXXXXX
 CCV:  XXXXXXXXXXXXXXXX FPSR: XXXXXXXXXXXXXXXX

 F32:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX F33:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 F34:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX F35:  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
...
 F124: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX F125: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 F126: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX F127: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 */

static void
hexvalue (unsigned long int value, char *buf, size_t len)
{
  char *cp = _itoa_word (value, buf + len, 16, 0);
  while (cp > buf)
    *--cp = '0';
}

static void
regvalue (unsigned long int *value, char letter, int regno, char *buf)
{
  int n = regno >= 100 ? 3 : regno >= 10 ? 2 : 1;
  buf[0] = ' ';
  buf[1] = letter;
  _itoa_word (regno, buf + 2 + n, 10, 0);
  buf[2 + n] = ':';
  for (++n; n <= 4; ++n)
    buf[2 + n] = ' ';
  hexvalue (value[0], buf + 7, 16);
  if (letter == 'F')
    {
      hexvalue (value[1], buf + 7 + 16, 16);
      buf[7 + 32] = '\n';
    }
  else
    buf[7 + 16] = '\n';
}

static void
register_dump (int fd, struct sigcontext *ctx)
{
  char gpregs[32 - 5][8 + 16];
  char fpregs[128 - 32][8 + 32];
  char bpregs[3][8 + 16];
  char spregs[8][16];
  struct iovec iov[146];
  size_t nr = 0;
  int i;

#define ADD_STRING(str) \
  do									      \
    {									      \
      iov[nr].iov_base = (char *) str;					      \
      iov[nr].iov_len = strlen (str);					      \
      ++nr;								      \
    }									      \
  while (0)
#define ADD_MEM(str, len) \
  do									      \
    {									      \
      iov[nr].iov_base = str;						      \
      iov[nr].iov_len = len;						      \
      ++nr;								      \
    }									      \
  while (0)

  /* Generate strings of register contents.  */
  for (i = 1; i < 4; ++i)
    {
      regvalue (&ctx->sc_gr[i], 'R', i, gpregs[i - 1]);
      if (ctx->sc_nat & (1L << i))
        memcpy (gpregs[i - 1] + 7, "NaT             ", 16);
    }
  for (i = 8; i < 32; ++i)
    {
      regvalue (&ctx->sc_gr[i], 'R', i, gpregs[i - 5]);
      if (ctx->sc_nat & (1L << i))
        memcpy (gpregs[i - 1] + 7, "NaT             ", 16);
    }
  memcpy (gpregs[0] + 1, "GP:", 3);
  memcpy (gpregs[7] + 1, "SP: ", 4);
  memcpy (gpregs[8] + 1, "TP: ", 4);

  regvalue (&ctx->sc_br[0], 'B', 0, bpregs[0]);
  regvalue (&ctx->sc_br[6], 'B', 6, bpregs[1]);
  regvalue (&ctx->sc_br[7], 'B', 7, bpregs[2]);
  memcpy (bpregs[0] + 1, "RP:", 3);

  if (ctx->sc_flags & IA64_SC_FLAG_FPH_VALID)
    for (i = 32; i < 128; ++i)
      regvalue (&ctx->sc_fr[i].u.bits[0], 'F', i, fpregs[i - 32]);

  hexvalue (ctx->sc_ip, spregs[0], sizeof (spregs[0]));
  hexvalue (ctx->sc_ar_rsc, spregs[1], sizeof (spregs[1]));
  hexvalue (ctx->sc_pr, spregs[2], sizeof (spregs[2]));
  hexvalue (ctx->sc_ar_pfs, spregs[3], sizeof (spregs[3]));
  hexvalue (ctx->sc_ar_unat, spregs[4], sizeof (spregs[4]));
  hexvalue (ctx->sc_cfm, spregs[5], sizeof (spregs[5]));
  hexvalue (ctx->sc_ar_ccv, spregs[6], sizeof (spregs[6]));
  hexvalue (ctx->sc_ar_fpsr, spregs[7], sizeof (spregs[7]));

  /* Generate the output.  */
  ADD_STRING ("Register dump:\n\n");

  for (i = 0; i < 32 - 5; ++i)
    ADD_MEM (gpregs[i], sizeof (gpregs[0]) - 1 + ((i % 3) == 2));
  ADD_STRING ("\n");

  for (i = 0; i < 3; ++i)
    ADD_MEM (bpregs[i], sizeof (bpregs[0]) - 1);

  ADD_STRING ("\n\n IP:   ");
  ADD_MEM (spregs[0], sizeof (spregs[0]));
  ADD_STRING (" RSC:  ");
  ADD_MEM (spregs[1], sizeof (spregs[0]));
  ADD_STRING (" PR:   ");
  ADD_MEM (spregs[2], sizeof (spregs[0]));
  ADD_STRING ("\n PFS:  ");
  ADD_MEM (spregs[3], sizeof (spregs[0]));
  ADD_STRING (" UNAT: ");
  ADD_MEM (spregs[4], sizeof (spregs[0]));
  ADD_STRING (" CFM:  ");
  ADD_MEM (spregs[5], sizeof (spregs[0]));
  ADD_STRING ("\n CCV:  ");
  ADD_MEM (spregs[6], sizeof (spregs[0]));
  ADD_STRING (" FPSR: ");
  ADD_MEM (spregs[7], sizeof (spregs[0]));
  ADD_STRING ("\n");

  if (ctx->sc_flags & IA64_SC_FLAG_FPH_VALID)
    {
      ADD_STRING ("\n");

      for (i = 0; i < 128 - 32; ++i)
        ADD_MEM (fpregs[i], sizeof (fpregs[0]) - 1 + (i & 1));
    }

  /* Write the stuff out.  */
  writev (fd, iov, nr);
}


#define REGISTER_DUMP register_dump (fd, ctx)

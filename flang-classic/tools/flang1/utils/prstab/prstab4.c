/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR parser (part 4)
 *
 */

#include "lrutils.h"
#include "prstab.h"

INT
zprntk(void)
{
  INT i_1, i_2;
  INT i, j, ch, ie;
  INT iptr;
  INT istart;
  FILE *filptr;
  DECL_LINE(80);

  filptr = fopen("tokdf.h", "wb");
  if (!filptr) {
    error("unable to open tokname file", 27, 2, 0, 27);
    return 0;
  }
  fprintf(filptr, "%s", "static const char *tokname[] = { \" \",\n"); /* } */
  istart = 4;
  ie = istart;
  i_1 = istart;
  for (i = 1; i <= i_1; ++i) {
    line[i - 1] = 32;
  }

  i_1 = g_1.nvoc;
  for (i = 1; i <= i_1; ++i) {
    ++ie;
    line[ie - 1] = 34;
    iptr = g_1.vocab[i - 1];
    i_2 = s1_1.sthead[iptr] - 1;
    for (j = s1_1.sthead[iptr - 1]; j <= i_2; ++j) {
      ch = s1_1.sstore[j - 1] & 255;
      if ((ch == 34) || (ch == 92)) {
        ++ie;
        line[ie - 1] = 92;
      }
      ++ie;
      line[ie - 1] = ch;
    }
    ++ie;
    line[ie - 1] = 34;
    ++ie;
    line[ie - 1] = 44;
    if (ie > 60) {
      ++ie;
      line[ie - 1] = 32;
    } else {
      wtline(filptr, line, ie);
      ie = istart;
    }
  }
  if (ie > istart) {
    wtline(filptr, line, ie);
  }
  line[0] = 125;
  line[1] = 59;
  wtline(filptr, line, 2);
  fclose(filptr);

  /*      non-terminal defines */

  /*    call a4tos1("grammarnt.h", line, 11) */
  filptr = fopen("gramnt.h", "wb");
  if (!filptr) {
    error("unable to open grammarnt file", 29, 2, 0, 29);
    return 0;
  }

  a4tos1("#define NT_", line, 11);
  i_1 = g_1.nvoc;
  for (i = g_1.nterms + 1; i <= i_1; ++i) {
    ie = 11;
    for (j = ie + 1; j <= 80; ++j) {
      line[j - 1] = 32;
    }
    iptr = g_1.vocab[i - 1];
    /*      get non-terminal name ignoring beginning < and ending > */
    i_2 = s1_1.sthead[iptr] - 2;
    for (j = s1_1.sthead[iptr - 1] + 1; j <= i_2; ++j) {
      ch = s1_1.sstore[j - 1] & 255;
      if (ch == 32) {
        ch = 95;
      }
      ++ie;
      line[ie - 1] = ch;
    }
    i32tos(&i, &line[ie + 1], 4, 0, 10, 1);
    i_2 = ie + 5;
    wtline(filptr, line, i_2);
  }
  fclose(filptr);

  return 0;
} /* zprntk */

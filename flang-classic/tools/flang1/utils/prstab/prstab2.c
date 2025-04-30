/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR parser (part 2)
 *
 */

#include "lrutils.h"
#include "prstab.h"

INT
gentab(void)
{
  /* Initialized data */
  static const char *msg5 =
      "d   i   m   e   n   s   i   o   n       p   r   o   "
      "d   (                       )   ,       l   e   f   t   s   "
      "i   d   e   (               )   ,       l   e   n   r   e   "
      "d   (               )   ";

  static const char *msg6 =
      "d   i   m   e   n   s   i   o   n       l   o   o   "
      "k   s   e   t   (           ,                       )   ";

  static const char *msg7 =
      "d   a   t   a   (   v   o   c   a   b   l   r   y   "
      "(               )   =   ";

  static const char *msg8 =
      "d   a   t   a   (   t   r   a   n   s   (           "
      "            )   =   ";

  static const char *msg9 =
      "d   a   t   a   (   f   r   d   t   r   a   n   s   "
      "(                       )   =   ";

  static const char *msg10 =
      "d   a   t   a   (   e   n   t   r   a   n   c   e   "
      "(                       )   =   ";

  static const char *msg11 =
      "d   a   t   a   (   f   l   o   o   k   a   h   d   "
      "(                       )   =   ";

  static const char *msg12 =
      "d   a   t   a   (   n   l   o   o   k   s   e   t   "
      "(                       )   =   ";

  static const char *msg13 =
      "d   a   t   a   (   p   r   o   d   (               "
      "        )   =   ";

  static const char *msg14 =
      "d   a   t   a   (   l   e   f   t   s   i   d   e   "
      "(               )   =   ";

  static const char *msg15 =
      "d   a   t   a   (   l   e   n   r   e   d   (       "
      "        )   =   ";

  static const char *msg16 =
      "d   a   t   a   (   l   o   o   k   s   e   t   (   "
      "        ,                       )   =   ";

  static const char *digit = "0   1   2   3   4   5   6   7   ";

  static const char *msga =
      "p   a   r   a   m   e   t   e   r   (   n   t   e   "
      "r   m   s   =               )   ";

  static const char *msg0 =
      "p   a   r   a   m   e   t   e   r   (   i   f   i   "
      "n   s   t   a   t   =                       )   ";

  static const char *msg1 =
      "b   y   t   e       t   r   a   n   s   (           "
      ")   ,       f   r   d   t   r   a   n   s   (           )   "
      ",       e   n   t   r   a   n   c   e   (           )   ,   "
      "    f   l   o   o   k   a   h   d   (           )   ";

  static const char *msg2 =
      "b   y   t   e       n   l   o   o   k   s   e   t   "
      "(           )   ,       p   r   o   d   (           )   ,   "
      "    l   e   f   t   s   i   d   e   (           )   ,       "
      "l   e   n   r   e   d   (       6   )   ";

  static const char *msg3 =
      "d   i   m   e   n   s   i   o   n       v   o   c   "
      "a   b   l   r   y   (               )   ,       t   r   a   "
      "n   s   (                       )   ,       f   r   d   t   "
      "r   a   n   s   (                       )   ";

  static const char *msg4 =
      "d   i   m   e   n   s   i   o   n       e   n   t   "
      "r   a   n   c   e   (                       )   ,       f   "
      "l   o   o   k   a   h   d   (                       )   ,   "
      "    n   l   o   o   k   s   e   t   (                       "
      ")   ";

  INT i_1, i_2, i_3, i_4;
  INT i, j, k, l, kk, lv, kke, kks, iend, jend, kend;
  INT ient;
  INT iptr, kptr, kitem, itemp;
  INT jstart, kstart;
  DECL_LINE(132);

  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  for (i = 1; i <= 25; ++i) {
    line[i + 5] = msg0[i - 1];
  }
  condec(s_1.ifinal, line, 26, 30);
  output(files.dbgfil, line, 31);

  for (i = 1; i <= 21; ++i) {
    line[i + 5] = msga[i - 1];
  }
  condec(g_1.nterms, line, 24, 26);
  output(files.dbgfil, line, 27);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  for (i = 1; i <= 56; ++i) {
    line[i + 5] = msg1[i - 1];
  }
  i_1 = log76(&s_1.nstate);
  condec(i_1, line, 18, 19);
  i_2 = s_1.nxttrn - 1;
  i_1 = log76(&i_2);
  condec(i_1, line, 32, 33);
  i_1 = log76(&g_1.nvoc);
  condec(i_1, line, 46, 47);
  i_2 = (s_1.nxtred - 1) / 2;
  i_1 = log76(&i_2);
  condec(i_1, line, 60, 61);
  output(files.dbgfil, line, 62);

  for (i = 1; i <= 53; ++i) {
    line[i + 5] = msg2[i - 1];
  }
  i_1 = log76(&s_1.ncsets);
  condec(i_1, line, 21, 22);
  i_1 = log76(&g_1.numprd);
  condec(i_1, line, 31, 32);
  i_1 = log76(&g_1.nvoc);
  condec(i_1, line, 45, 46);
  output(files.dbgfil, line, 59);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  for (i = 1; i <= 54; ++i) {
    line[i + 5] = msg3[i - 1];
  }
  condec(g_1.nvoc, line, 26, 28);
  i_1 = s_1.nxttrn - 1;
  condec(i_1, line, 38, 42);
  i_1 = s_1.nstate + 1;
  condec(i_1, line, 55, 59);
  output(files.dbgfil, line, 60);

  for (i = 1; i <= 59; ++i) {
    line[i + 5] = msg4[i - 1];
  }
  condec(s_1.nstate, line, 26, 30);
  i_1 = s_1.nstate + 1;
  condec(i_1, line, 43, 47);
  i_1 = (s_1.nxtred - 1) / 2;
  condec(i_1, line, 60, 64);
  output(files.dbgfil, line, 65);

  for (i = 1; i <= 49; ++i) {
    line[i + 5] = msg5[i - 1];
  }
  i_1 = (s_1.nxtred - 1) / 2;
  condec(i_1, line, 22, 26);
  condec(g_1.numprd, line, 39, 41);
  condec(g_1.numprd, line, 52, 54);
  output(files.dbgfil, line, 55);

  for (i = 1; i <= 27; ++i) {
    line[i + 5] = msg6[i - 1];
  }
  i_1 = (g_1.nterms + 59) / 60;
  condec(i_1, line, 25, 26);
  condec(s_1.ncsets, line, 28, 32);
  output(files.dbgfil, line, 33);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /* output the vocabulary array. */

  i = 1;
  while (1) {
    for (j = 1; j <= 19; ++j) {
      line[j + 5] = msg7[j - 1];
    }
    condec(i, line, 21, 23);
    l = 26;

    do {
      lv = length(&g_1.vocab[i - 1]);
      if (lv > 10) {
        lv = 10;
      }
      if (l + lv + 4 >= 73) {
        line[l - 2] = ')';
        i_1 = l - 1;
        output(files.dbgfil, line, i_1);
        break;
      }
      i_1 = l + 1;
      condec(lv, line, l, i_1);
      line[l + 1] = 'H';
      l += 3;
      i_1 = l + 9;
      movstr(g_1.vocab[i - 1], line, &l, i_1);
      line[l - 1] = ',';
      ++l;
      ++i;
    } while (i > g_1.nvoc);

    if (i <= g_1.nvoc)
      break;
  }

  line[l - 2] = ')';
  i_1 = l - 1;
  output(files.dbgfil, line, i_1);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /* output the transition array. */

  iend = s_1.nxttrn - 1;
  i_1 = iend;
  for (i = 1; i <= i_1; i += 8) {
    for (j = 1; j <= 18; ++j) {
      line[j + 5] = msg8[j - 1];
    }
    condec(i, line, 18, 22);
    l = 25;
    jstart = i;
    jend = i + 7;
    if (jend > iend) {
      jend = iend;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      i_3 = l + 4;
      condec(s_1.basis[s_1.tran[j - 1] - 1], line, l, i_3);
      line[l + 4] = ',';
      l += 6;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the first read transition array. */

  k = -2;
  i_1 = s_1.nstate;
  for (i = 1; i <= i_1; i += 7) {
    for (j = 1; j <= 21; ++j) {
      line[j + 5] = msg9[j - 1];
    }
    condec(i, line, 21, 25);
    l = 28;
    jstart = i;
    jend = i + 6;
    if (jend > s_1.nstate) {
      jend = s_1.nstate;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      k = k + s_1.basis[k + 3] * 3 + 8;
      i_3 = l + 4;
      condec(s_1.basis[k - 1], line, l, i_3);
      line[l + 4] = ',';
      l += 6;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  for (j = 1; j <= 21; ++j) {
    line[j + 5] = msg9[j - 1];
  }
  i_1 = s_1.nstate + 1;
  condec(i_1, line, 21, 25);
  i_1 = s_1.basis[k - 1] + s_1.basis[k - 2];
  condec(i_1, line, 28, 32);
  line[32] = ')';
  output(files.dbgfil, line, 33);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the entrance symbol array. */

  k = 5;
  i_1 = s_1.nstate;
  for (i = 1; i <= i_1; i += 11) {
    for (j = 1; j <= 21; ++j) {
      line[j + 5] = msg10[j - 1];
    }
    condec(i, line, 21, 25);
    l = 28;
    jstart = i;
    jend = i + 10;
    if (jend > s_1.nstate) {
      jend = s_1.nstate;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      ient = g_1.prodcn[g_1.prdind[s_1.basis[k - 1] - 1] + s_1.basis[k] - 1];
      k = k + s_1.basis[k - 4] * 3 + 8;
      i_3 = l + 2;
      condec(ient, line, l, i_3);
      line[l + 2] = ',';
      l += 4;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the first lookahead array. */

  k = 0;
  i_1 = s_1.nstate;
  for (i = 1; i <= i_1; i += 7) {
    for (j = 1; j <= 21; ++j) {
      line[j + 5] = msg11[j - 1];
    }
    condec(i, line, 21, 25);
    l = 28;
    jstart = i;
    jend = i + 6;
    if (jend > s_1.nstate) {
      jend = s_1.nstate;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      k = k + s_1.basis[k + 1] * 3 + 8;
      i_3 = (s_1.basis[k - 1] + 1) / 2;
      i_4 = l + 4;
      condec(i_3, line, l, i_4);
      line[l + 4] = ',';
      l += 6;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  for (i = 1; i <= 21; ++i) {
    line[i + 5] = msg11[i - 1];
  }
  i_1 = s_1.nstate + 1;
  condec(i_1, line, 21, 25);
  i_1 = ((s_1.basis[k - 2] << 1) + s_1.basis[k - 1] + 1) / 2;
  condec(i_1, line, 28, 32);
  line[32] = ')';
  output(files.dbgfil, line, 33);
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the look ahead set number array. */

  iend = s_1.nxtred - 1;
  i_1 = iend;
  for (i = 2; i <= i_1; i += 14) {
    for (j = 1; j <= 21; ++j) {
      line[j + 5] = msg12[j - 1];
    }
    i_2 = i / 2;
    condec(i_2, line, 21, 25);
    l = 28;
    jstart = i;
    jend = i + 12;
    if (jend > iend) {
      jend = iend;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; j += 2) {
      i_3 = l + 4;
      condec(s4.item[s_1.red[j - 1] - 1], line, l, i_3);
      line[l + 4] = ',';
      l += 6;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the production array. */

  iend = s_1.nxtred - 1;
  i_1 = iend;
  for (i = 1; i <= i_1; i += 16) {
    for (j = 1; j <= 17; ++j) {
      line[j + 5] = msg13[j - 1];
    }
    i_2 = (i + 1) / 2;
    condec(i_2, line, 17, 21);
    l = 24;
    jstart = i;
    jend = i + 14;
    if (jend > iend) {
      jend = iend;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; j += 2) {
      i_3 = l + 4;
      condec(s_1.red[j - 1], line, l, i_3);
      line[l + 4] = ',';
      l += 6;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the left side array. */

  k = 1;
  i_1 = g_1.numprd;
  for (i = 1; i <= i_1; i += 11) {
    for (j = 1; j <= 19; ++j) {
      line[j + 5] = msg14[j - 1];
    }
    condec(i, line, 21, 23);
    l = 26;
    jstart = i;
    jend = i + 10;
    if (jend > g_1.numprd) {
      jend = g_1.numprd;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      i_3 = l + 2;
      condec(g_1.prodcn[k - 1], line, l, i_3);
      line[l + 2] = ',';
      l += 4;
      k = k + g_1.prodcn[k] + 2;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the right side length array. */

  k = 2;
  i_1 = g_1.numprd;
  for (i = 1; i <= i_1; i += 12) {
    for (j = 1; j <= 17; ++j) {
      line[j + 5] = msg15[j - 1];
    }
    condec(i, line, 19, 21);
    l = 24;
    jstart = i;
    jend = i + 11;
    if (jend > g_1.numprd) {
      jend = g_1.numprd;
    }
    i_2 = jend;
    for (j = jstart; j <= i_2; ++j) {
      i_3 = l + 2;
      condec(g_1.prodcn[k - 1], line, l, i_3);
      line[l + 2] = ',';
      l += 4;
      k = k + g_1.prodcn[k - 1] + 2;
    }
    line[l - 2] = ')';
    i_2 = l - 1;
    output(files.dbgfil, line, i_2);
  }
  line[0] = 'C';
  output(files.dbgfil, line, 1);

  /*     output the context sets. */

  iptr = s_1.listcs;
  i_1 = s_1.ncsets;
  for (i = 1; i <= i_1; ++i) {
    kptr = s4.nextt[s4.item[iptr - 1] - 1];
    kitem = s4.item[kptr - 1];
    i_2 = g_1.nterms;
    for (j = 1; j <= i_2; j += 60) {
      for (k = 1; k <= 23; ++k) {
        line[k + 5] = msg16[k - 1];
      }
      i_3 = (j + 59) / 60;
      condec(i_3, line, 20, 21);
      condec(i, line, 23, 27);
      l = 30;
      kstart = j;
      kend = j + 59;
      i_3 = kend;
      for (k = kstart; k <= i_3; k += 3) {
        itemp = 0;
        kks = k;
        kke = k + 2;
        i_4 = kke;
        for (kk = kks; kk <= i_4; ++kk) {
          itemp <<= 1;
          if (kk != kitem) {
            continue;
          }
          ++itemp;
          kptr = s4.nextt[kptr - 1];
          if (kptr != 0) {
            kitem = s4.item[kptr - 1];
          }
        }
        line[l - 1] = digit[itemp];
        ++l;
      }
      line[l - 1] = 'B';
      line[l] = ')';
      i_3 = l + 1;
      output(files.dbgfil, line, i_3);
    }
    iptr = s4.nextt[iptr - 1];
  }
  return 0;
} /* gentab */

INT
genthd(void)
{
  static const char *hdg =
      "t   h   e       t   h   e   a   d       s   e   t   "
      "s   ";

  INT i_1, i_2;
  INT i, j, k;
  INT iend, ichn, kend;
  INT iptr;
#define reset (scrtch + 199)
  INT ichnge;
  INT istart, kstart, jstart;
  DECL_LINE(132);

  /*     the theads for a terminal symbol is that symbol. */
  /*     Fix from Al Shannon: zero out thedpt */
  /*     J.R.L. - 8 dec 78 */

  for (i = 1; i <= MAXSHD; ++i) {
    s_1.thedpt[i - 1] = 0;
  }
  i_1 = g_1.nterms;
  for (i = 1; i <= i_1; ++i) {
    additl(&i, &s_1.thedpt[i - 1], &ichn);
  }

  do {
    ichnge = 0;
    i_1 = g_1.numprd;
    for (i = 1; i <= i_1; ++i) {
      j = g_1.prdind[i - 1];
      kstart = j + 2;
      kend = kstart + g_1.prodcn[j] - 1;
      if (kstart > kend) {
        continue;
      }
      i_2 = kend;
      for (k = kstart; k <= i_2; ++k) {
        if (g_1.prodcn[k - 1] != g_1.prodcn[j - 1]) {
          addltl(&s_1.thedpt[g_1.prodcn[k - 1] - 1],
                 &s_1.thedpt[g_1.prodcn[j - 1] - 1], &ichn);
          if (ichn != 0) {
            ichnge = 1;
          }
        }

        if (s_1.nullnt[g_1.prodcn[k - 1] - 1] == 0) {
          break;
        }
      } /* k */
    }   /* i */
  } while (ichnge != 0);

  /*     now pack up the sets. if two sets are identical then they will be */
  /*     shared. this is okay because the thead sets never change once
   * generated. */

  istart = g_1.nterms + 1;
  i_1 = g_1.nvoc;
  for (i = istart; i <= i_1; ++i) {
    reset[i - 1] = 0;
  }
  iend = g_1.nvoc - 1;
  i_1 = iend;
  for (i = istart; i <= i_1; ++i) {
    if (reset[i - 1] != 0) {
      continue;
    }
    jstart = i + 1;
    i_2 = g_1.nvoc;
    for (j = jstart; j <= i_2; ++j) {
      if (reset[j - 1] == 0) {
        if (lcompr(&s_1.thedpt[i - 1], &s_1.thedpt[j - 1]) != 0 &&
            s_1.thedpt[j - 1] != 0) {
          rel(&s_1.thedpt[j - 1]);
          s_1.thedpt[j - 1] = s_1.thedpt[i - 1];
          reset[j - 1] = 1;
        }
      }
    }
  } /* i */

  if (!switches.dbgcsw) {
    return 0;
  }

  for (i = 1; i <= 14; ++i) {
    line[i] = hdg[i - 1];
  }
  output(files.dbgfil, line, 15);
  kstart = g_1.nterms + 1;
  i_1 = g_1.nvoc;
  for (k = kstart; k <= i_1; ++k) {
    i = 4;
    movstr(g_1.vocab[k - 1], line, &i, 120);
    i += 2;
    iptr = s_1.thedpt[k - 1];

    while (1) {
      if (iptr <= 0) {
        i_2 = i - 1;
        output(files.dbgfil, line, i_2);
        break;
      }
      if (length(&g_1.vocab[s4.item[iptr - 1] - 1]) + i > 120) {
        i_2 = i - 1;
        output(files.dbgfil, line, i_2);
        i = 10;
      }

      movstr(g_1.vocab[s4.item[iptr - 1] - 1], line, &i, 120);
      ++i;
      iptr = s4.nextt[iptr - 1];
    }
  } /* k */

  return 0;
} /* genthd */

#undef reset

INT
ground(void)
{
  INT i_1, i_2, i_3;

  INT i, j, k, jend, kend;
  INT flag_1 = 0, flag_2 = 0;
  INT ibase;
  INT change;
#define grnded scrtch
  INT istart;

  /*     first mark all the terminals grounded: */
  i_1 = g_1.nterms;
  for (i = 1; i <= i_1; ++i) {
    grnded[i - 1] = 1;
  }
  istart = g_1.nterms + 1;
  i_1 = g_1.nvoc;
  for (i = istart; i <= i_1; ++i) {
    grnded[i - 1] = 0;
  }

  do {
    change = 0;
    i_1 = g_1.nvoc;
    for (i = istart; i <= i_1; ++i) {
      if (grnded[i - 1] != 0) {
        continue;
      }
      ibase = g_1.prdind[g_1.frsprd[i - 1] - 1] + 1;
      jend = g_1.nprods[i - 1];
      i_2 = jend;
      for (j = 1; j <= i_2; ++j) {
        if (g_1.prodcn[ibase - 1] == 0) {
          flag_1 = 1;
          break;
        }
        kend = g_1.prodcn[ibase - 1];
        i_3 = kend;
        for (k = 1; k <= i_3; ++k) {
          if (grnded[g_1.prodcn[ibase + k - 1] - 1] == 0) {
            flag_2 = 1;
            break;
          }
        }

        if (flag_2) {
          flag_2 = 0;
          ibase = ibase + g_1.prodcn[ibase - 1] + 2;
        } else {
          flag_1 = 1;
          break;
        }
      }

      if (flag_1) {
        grnded[i - 1] = 1;
        change = 1;
      } else {
        flag_1 = 0;
      }
    } /* i */
  } while (change != 0);

  /*     print out the un-grounded symbols. */

  i_1 = g_1.nvoc;
  for (i = istart; i <= i_1; ++i) {
    if (grnded[i - 1] == 0) {
      error("not grounded", 12, 1, g_1.vocab[i - 1], 12);
    }
  }
  return 0;
} /* ground */

#undef grnded

INT
hashof(INT *i__)
{
  INT ret_val;
  INT itemp, itemp1, itemp2;

  itemp1 = s1_1.sstore[s1_1.sthead[*i__ - 1] - 1] & 255;
  itemp2 = s1_1.sstore[s1_1.sthead[*i__] - 2] & 255;
  itemp = itemp1 * itemp2;
  /*     make sure we return a positive value from 0 to 511 */
  /*     J.R.L. - 13 dec 78 */
  if (itemp < 0) {
    itemp = -itemp;
  }
  ret_val = itemp - (MAXHASH * (itemp / MAXHASH));
  return ret_val;
} /* hashof */

INT
hepify(INT iptr, INT imax)
{
  /* Local variables */
  INT i, j, i0, i1, i2;
  INT itemp;

  i = iptr;

  while (i << 1 <= imax) {
    i0 = g_1.vocab[i - 1];
    i1 = g_1.vocab[(i << 1) - 1];
    j = i << 1;
    if ((i << 1) + 1 <= imax) {
      i2 = g_1.vocab[i * 2];
      if (less(&i1, &i2) != 0) {
        i1 = i2;
        j = (i << 1) + 1;
      }
    }

    if (less(&i0, &i1) == 0) {
      break;
    }
    itemp = g_1.vocab[i - 1];
    g_1.vocab[i - 1] = g_1.vocab[j - 1];
    g_1.vocab[j - 1] = itemp;
    i = j;
  }

  return 0;
} /* hepify */

INT
hlddmp(INT *hold, INT *hldidx)
{
  INT i_1, i_2, i_3;
  INT i, line[55];

  /* Parameter adjustments */
  --hold;

  /* Function Body */
  i_1 = *hldidx;
  for (i = 1; i <= i_1; ++i) {
    i_2 = i * 5 - 4;
    i_3 = i * 5;
    condec(hold[i], line, i_2, i_3);
  }
  i_1 = *hldidx * 5;
  output(files.dbgfil, line, i_1);
  return 0;
} /* hlddmp */

INT
imtrcs(INT *ibasis, INT *iptr)
{
  INT i_1;
  INT i, ip, ich, iend;
  INT istart;

  /*     generates context sets for immediate transitions. */
  /*     the immediate transition context set is a set the theads of the */
  /*     substring of symbols in the production following the symbol after */
  /*     the dot unioned with the context set of the production if that */
  /*     substring is null or potentially null. */
  i = g_1.prdind[scrtch[*ibasis - 1] - 1];
  /*     if there is no substring just return the context set */
  /*     of the production. */
  if (scrtch[*ibasis] + 1 > g_1.prodcn[i]) {
    *iptr = scrtch[*ibasis + 1];
    ++s4.item[*iptr - 1];
    return 0;
  }

  istart = scrtch[*ibasis] + i + 2;
  iend = g_1.prodcn[i] + i + 1;
  copyl(&s_1.thedpt[g_1.prodcn[istart - 1] - 1], &ip);
  if (s_1.nullnt[g_1.prodcn[istart - 1] - 1] == 0) {
    newcs(&ip, iptr);
    return 0;
  }
  if (istart == iend) {
    addltl(&s4.nextt[scrtch[*ibasis + 1] - 1], &ip, &ich);
    newcs(&ip, iptr);
    return 0;
  }
  ++istart;
  i_1 = iend;
  for (i = istart; i <= i_1; ++i) {
    addltl(&s_1.thedpt[g_1.prodcn[i - 1] - 1], &ip, &ich);
    if (s_1.nullnt[g_1.prodcn[i - 1] - 1] == 0) {
      newcs(&ip, iptr);
      return 0;
    }
  }

  addltl(&s4.nextt[scrtch[*ibasis + 1] - 1], &ip, &ich);
  newcs(&ip, iptr);
  return 0;
} /* imtrcs */

void
lowerFilename(CHAR *filename, INT len)
{
  INT i;
  for (i = 0; i < len; i++) {
    filename[i] = tolower(filename[i]);
  }
}

void
formatFilename(INT *filename, INT len)
{
  intArrayToCharArray(filename, linech, len);
  lowerFilename(linech, len);
  SNPRINTF(filnambuf, sizeof(CHAR) * (len + 1), "%s", linech);
  filnambuf[len] = '\0';
}

INT
init(void)
{
  INT i, flag = 0;
  INT len, ptr, srcp, pptr;
#define swtab scrtch
  INT swcnt, swpos;
  INT opntyp, swtype;
  INT *s;

  /* initialize */
  scrtch = (INT *)malloc(sizeof(INT) * MAXSCR);
  memset(scrtch, 0, sizeof(INT) * MAXSCR);
  s = scrtch + 1500;

  linech = (CHAR *)malloc(sizeof(CHAR) * MAXSCR);
  filnambuf = (char *)malloc(sizeof(CHAR) * MAXLEN);

  hashpt = (INT *)malloc(sizeof(INT) * MAXHASH);

  s4.item = (INT *)malloc(sizeof(INT) * MAXLST);
  s4.nextt = (INT *)malloc(sizeof(INT) * MAXLST);

  s1_1.sstore = (INT *)malloc(sizeof(INT) * MAXSST);
  s1_1.sthead = (INT *)malloc(sizeof(INT) * MAXSHDP1);

  g_1.lftuse = (INT *)malloc(sizeof(INT) * MAXSHD);
  g_1.rgtuse = (INT *)malloc(sizeof(INT) * MAXSHD);
  g_1.frsprd = (INT *)malloc(sizeof(INT) * MAXSHD);
  g_1.nprods = (INT *)malloc(sizeof(INT) * MAXSHD);
  g_1.prodcn = (INT *)malloc(sizeof(INT) * MAXPRDC);
  g_1.prdind = (INT *)malloc(sizeof(INT) * MAXPROD);
  g_1.vocab = (INT *)malloc(sizeof(INT) * MAXSHD);

  s_1.thedpt = (INT *)malloc(sizeof(INT) * MAXSHD);
  s_1.nullnt = (INT *)malloc(sizeof(INT) * MAXSHD);
  s_1.basis = (INT *)malloc(sizeof(INT) * MAXBAS);
  s_1.tran = (INT *)malloc(sizeof(INT) * MAXTRN);
  s_1.red = (INT *)malloc(sizeof(INT) * MAXRED);

  /*     set default option values, read command line, open files, */
  /*     initialize variables: */
  switches.listsw = FALSE;
  switches.runosw = FALSE;
  switches.toksw = FALSE;
  switches.semsw = FALSE;
  switches.datasw = FALSE;
  switches.lrltsw = FALSE;
  switches.dbgsw = FALSE;
  switches.dbgasw = FALSE;
  switches.dbgbsw = FALSE;
  switches.dbgcsw = FALSE;
  switches.dbgdsw = FALSE;
  switches.dbgesw = FALSE;
  /*     initialize switch table input to CLI: */
  i = 1;
  ptr = 1;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 5;
  a4tos1("L.IST", &swtab[ptr + 1], 5);
  ptr += 7;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 6;
  a4tos1("RUNOFF", &swtab[ptr + 1], 6);
  ptr += 8;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 7;
  a4tos1("TOK.ENS", &swtab[ptr + 1], 7);
  ptr += 9;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 7;
  a4tos1("SEM.ANT", &swtab[ptr + 1], 7);
  ptr += 9;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DATA", &swtab[ptr + 1], 4);
  ptr += 6;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 7;
  a4tos1("LRLTRAN", &swtab[ptr + 1], 7);
  ptr += 9;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DBGA", &swtab[ptr + 1], 4);
  ptr += 6;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DBGB", &swtab[ptr + 1], 4);
  ptr += 6;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DBGC", &swtab[ptr + 1], 4);
  ptr += 6;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DBGD", &swtab[ptr + 1], 4);
  ptr += 6;
  swtab[ptr - 1] = i;
  ++i;
  swtab[ptr] = 4;
  a4tos1("DBGE", &swtab[ptr + 1], 4);
  ptr += 6;
  /*     process command line: */
  cli(swtab, s, 1500, 0);
  srcp = 0;
  swpos = 1;

  while (s[swpos - 1] != 0) {
    swtype = s[swpos - 1];
    if (swtype < -2) {
      error("illegal command line", 20, 2, 0, 20);
    }

    /*     number of switch values for this switch */
    swcnt = s[swpos];
    swpos += 2;
    /*     source file name */
    if (swtype == -1) {
      srcp = swpos;
      swpos = swpos + s[swpos - 1] + 1;
      continue;
    }

    if (swtype == -2) {
      error("unrecognized option", 19, 2, 0, 19);
    }
    if (swcnt != 0) {
      error("switch values not allowed", 25, 2, 0, 25);
    }
    switch (swtype) {
    case 1:
      switches.listsw = TRUE;
      switches.xrefsw = TRUE;
      continue;
    case 2:
      switches.runosw = TRUE;
      continue;
    case 3:
      switches.toksw = TRUE;
      continue;
    case 4:
      switches.semsw = TRUE;
      continue;
    case 5:
      switches.datasw = TRUE;
      continue;
    case 6:
      switches.lrltsw = TRUE;
      continue;
    case 7:
      switches.dbgasw = TRUE;
      continue;
    case 8:
      switches.dbgbsw = TRUE;
      continue;
    case 9:
      switches.dbgcsw = TRUE;
      continue;
    case 10:
      switches.dbgdsw = TRUE;
      continue;
    case 11:
      switches.dbgesw = TRUE;
      continue;
    }
  } /* while */

  switches.dbgsw = switches.dbgasw || switches.dbgbsw || switches.dbgcsw ||
                   switches.dbgdsw || switches.dbgesw;
  /*     open files: */
  if (srcp == 0) {
    error("input file not specified", 24, 2, 0, 24);
  }
  len = s[srcp - 1];
  ptr = srcp + len;

  while (ptr > srcp) {
    if (s[ptr - 1] == 46) {
      /*     pointer to beginning of suffix */
      pptr = ptr + 1;
      flag = 1;
      break;
    }
    /*     add default suffix for input file: */
    --ptr;
  }
  if (!flag) {
    pptr = srcp + len + 2;
    len += 4;
    a4tos1(".txt", &s[pptr - 2], 4);
  }

  ++srcp;
  formatFilename(&s[srcp - 1], len);
  files.gramin = fopen(filnambuf, "r");
  if (!files.gramin) {
    error("unable to open input file", 25, 2, 0, 25);
  }
  files.infile = files.gramin;
  len = pptr - srcp + 3;
  if (switches.listsw) {
    /*     open listing file: */
    a4tos1("lis", &s[pptr - 1], 3);
    opntyp = 13;
    if (switches.runosw) {
      opntyp = 12;
    }
    formatFilename(&s[srcp - 1], len);
    files.lstfil = fopen(filnambuf, "wb");
    if (!files.lstfil) {
      error("unable to open listing file", 27, 2, 0, 27);
    }
  }

  if (switches.dbgsw || switches.lrltsw) {
    a4tos1("dbg", &s[pptr - 1], 3);
    formatFilename(&s[srcp - 1], len);
    files.dbgfil = fopen(filnambuf, "wb");
    if (!files.dbgfil) {
      error("unable to open debug file", 25, 2, 0, 25);
    }
  }

  if (switches.datasw) {
    a4tos1("df.h", &s[pptr - 2], 4);
    formatFilename(&s[srcp - 1], len);
    files.datfil = fopen(filnambuf, "wb");
    if (!files.datfil) {
      error("unable to open df.h file", 24, 2, 0, 24);
    }
  }

  if (switches.toksw) {
    a4tos1(".tki", &s[pptr - 2], 4);
    formatFilename(&s[srcp - 1], len);
    files.tokin = fopen(filnambuf, "r");
    a4tos1("tk.h", &s[pptr - 2], 4);
    formatFilename(&s[srcp - 1], len);
    files.tokout = fopen(filnambuf, "wb");
    if (!files.tokin || !files.tokout) {
      error("unable to open token file", 25, 2, 0, 25);
    }
  }

  if (switches.semsw) {
    a4tos1("sm.h", &s[pptr - 2], 4);
    formatFilename(&s[srcp - 1], len);
    files.semfil = fopen(filnambuf, "wb");
    if (!files.semfil) {
      error("unable to open sm.h file", 24, 2, 0, 24);
    }
  }

  /*     initialize variables: */
  adqcom.adequt = 1;
  string_1.shdptr = 0;
  string_1.sstptr = 0;
  lstcom.garbag = 0;
  lstcom.lstptr = 0;
  s_1.nstate = 0;
  s_1.indbas = 1;
  s_1.nxttrn = 1;
  s_1.nxtred = 1;
  s_1.ncsets = 0;
  qcom.qhead = 0;
  qcom.qtail = 0;
  for (i = 0; i < MAXHASH; ++i) {
    hashpt[i] = 0;
  }
  for (i = 0; i < MAXSHD; ++i) {
    g_1.lftuse[i] = 0;
    g_1.rgtuse[i] = 0;
  }
  /*     initialize scanning: */
  readcm.lineno = 0;
  a4tos1("<SYSTEM GOAL SYMBOL>::=END END END", readcm.linbuf, 34);
  readcm.linbuf[34] = 258;
  readcm.curchr = 1;
  /*     production number for semantic definitions file */
  ptsem.pnum = 0;
  /*     break flag = false */
  ptsem.brkflg = 0;
  /*     first number to use for semantic break macro def */
  ptsem.semnum = 2;
  return 0;
} /* init */

#undef swtab

INT
lcompr(INT *iptr1, INT *iptr2)
{
  INT ret_val;
  INT i1, i2;

  ret_val = 1;
  if (*iptr1 == *iptr2) {
    return ret_val;
  }
  i1 = *iptr1;
  i2 = *iptr2;

  while ((i1 != 0) && (i2 != 0)) {
    if (s4.item[i1 - 1] != s4.item[i2 - 1]) {
      ret_val = 0;
    }
    i1 = s4.nextt[i1 - 1];
    i2 = s4.nextt[i2 - 1];
  }

  if (i1 != i2) {
    ret_val = 0;
  }

  return ret_val;
} /* lcompr */

INT
length(INT *iptr)
{
  INT ret_val;

  ret_val = s1_1.sthead[*iptr] - s1_1.sthead[*iptr - 1];
  return ret_val;
} /* length */

INT
lenlst(INT *ihead)
{
  INT ret_val;
  INT icnt, iptr;

  /*     returns the length of the list pointed to by ihead. */
  iptr = *ihead;
  icnt = 0;

  while (iptr > 0) {
    ++icnt;
    iptr = s4.nextt[iptr - 1];
  }

  ret_val = icnt;
  return ret_val;
} /* lenlst */

INT
less(INT *iptr1, INT *iptr2)
{
  INT ret_val;
  INT irslt;

  ret_val = 0;
  if (g_1.lftuse[*iptr1 - 1] == g_1.lftuse[*iptr2 - 1]) {
    strcomp(iptr1, iptr2, &irslt);
    if (irslt < 0) {
      ret_val = 1;
    }
    return ret_val;
  }

  if (g_1.lftuse[*iptr1 - 1] == 0) {
    ret_val = 1;
  }
  return ret_val;
} /* less */

INT
lint(INT *iptr1, INT *iptr2)
{
  INT ret_val;
  INT i1, i2;

  /*     returns 1 if there is a common item in the two lists, 0 otherwise. */
  i1 = *iptr1;
  i2 = *iptr2;
  ret_val = 0;

  while ((i1 != 0) && (i2 != 0)) {
    if (s4.item[i1 - 1] == s4.item[i2 - 1]) {
      ret_val = 1;
      break;
    }
    if (s4.item[i1 - 1] < s4.item[i2 - 1]) {
      i1 = s4.nextt[i1 - 1];
      continue;
    }

    i2 = s4.nextt[i2 - 1];
  }

  return ret_val;
} /* lint */

INT
log76(INT *iarg)
{
  INT ret_val;

  if (*iarg <= 7) {
    ret_val = 3;
  } else if (*iarg <= 15) {
    ret_val = 4;
  } else if (*iarg <= 31) {
    ret_val = 5;
  } else if (*iarg <= 63) {
    ret_val = 6;
  } else if (*iarg <= 1023) {
    ret_val = 10;
  } else if (*iarg > 4095) {
    ret_val = 15;
  } else {
    ret_val = 12;
  }

  return ret_val;
} /* log76 */

INT
merge(INT *ibasis, INT *ires, INT *ichng)
{
  INT i1, i2;
  INT i, j, ich, iend, ient, flag = 0;
  INT iptr;

  /*     find the entrance symbol to the config. set and then search the */
  /*     list of configs. with that entrance symbol. */
  ient = g_1.prodcn[g_1.prdind[s_1.basis[*ibasis + 3] - 1] +
                    s_1.basis[*ibasis + 4] - 1];
  iptr = g_1.lftuse[ient - 1];

  while (1) {
    if (iptr == 0) {
      *ichng = 1;
      *ires = *ibasis;
      s_1.basis[*ibasis + 2] = g_1.lftuse[ient - 1];
      g_1.lftuse[ient - 1] = *ibasis;
      return 0;
    }
    /*     does this config. set have the same number of basis productions? */
    if (s_1.basis[iptr] != s_1.basis[*ibasis]) {
      iptr = s_1.basis[iptr + 2];
      continue;
    }
    /*     compare the basis configurations. */
    iend = s_1.basis[iptr] * 3 + 3;
    i1 = iend;
    for (i = 4; i <= i1; i += 3) {
      if (s_1.basis[iptr + i - 1] != s_1.basis[*ibasis + i - 1]) {
        flag = 1;
        break;
      }
      if (s_1.basis[iptr + i] != s_1.basis[*ibasis + i]) {
        flag = 1;
        break;
      }
    }

    if (flag) {
      iptr = s_1.basis[iptr + 2];
      flag = 0;
      continue;
    }

    /*     are the  config. sets compatible? they are not if there would be */
    /*     two intersecting context sets created by the merge where there was */
    /*     no intersection before. */

    iend += 2;
    i1 = iend;
    for (i = 6; i <= i1 && !flag; i += 3) {
      i2 = iend;
      for (j = 6; j <= i2; j += 3) {
        if (i == j) {
          continue;
        }
        if (lint(&s4.nextt[s_1.basis[iptr + i - 1] - 1],
                 &s4.nextt[s_1.basis[*ibasis + j - 1] - 1]) == 0) {
          continue;
        }
        if (lint(&s4.nextt[s_1.basis[iptr + i - 1] - 1],
                 &s4.nextt[s_1.basis[iptr + j - 1] - 1]) != 0) {
          continue;
        }
        if (lint(&s4.nextt[s_1.basis[*ibasis + i - 1] - 1],
                 &s4.nextt[s_1.basis[*ibasis + j - 1] - 1]) == 0) {
          flag = 1;
          break;
        }
      } /* j */
    }   /* i */

    if (flag) {
      iptr = s_1.basis[iptr + 2];
    } else {
      break;
    }
  } /* while */

  /*     the sets are compatible. merge them by unioning the context sets. */
  /*     if the sets are equal then no change will occur when the */
  /*     then delete the context sets in the trial basis. */
  /*     context sets are merged. return the old basis and delete the trial. */
  *ichng = 0;
  i1 = iend;
  for (i = 6; i <= i1; i += 3) {
    csun(&s_1.basis[*ibasis + i - 1], &s_1.basis[iptr + i - 1], &ich);
    if (ich != 0) {
      *ichng = 1;
    }
    delcs(&s_1.basis[*ibasis + i - 1]);
  }

  /*     the trial basis is always constructed as the last basis in the basis */
  /*     array. since we are merging it into another basis we can now delete */
  /*     it. the context sets for the trial basis have already been released. */
  /*     all that remains to do is to reset the pointer into the basis array */
  /*     and release the state number. */
  s_1.indbas = *ibasis;
  --s_1.nstate;
  *ires = iptr;

  return 0;
} /* merge */

INT
movstr(INT iptr, INT *line, INT *istart, INT iend)
{
  INT i1;
  INT i, j, ie;

  /* Parameter adjustments */
  --line;

  ie = s1_1.sthead[iptr] - s1_1.sthead[iptr - 1] + *istart - 1;
  if (iend < ie) {
    ie = iend;
  }
  j = s1_1.sthead[iptr - 1];
  i1 = ie;
  for (i = *istart; i <= i1; ++i) {
    line[i] = s1_1.sstore[j - 1];
    ++j;
  }
  *istart = ie + 1;
  return 0;
} /* movstr */

INT new (INT *iptr)
{
  if (lstcom.garbag == 0) {
    if (lstcom.lstptr >= MAXLST) {
      error("list space overflow.", 20, 2, 0, 20);
      s4.item[*iptr - 1] = 0;
      s4.nextt[*iptr - 1] = 0;
      return 0;
    }
    ++lstcom.lstptr;
    *iptr = lstcom.lstptr;
    s4.item[*iptr - 1] = 0;
    s4.nextt[*iptr - 1] = 0;
    return 0;
  }

  if (s4.item[lstcom.garbag - 1] == 0) {
    *iptr = lstcom.garbag;
    lstcom.garbag = s4.nextt[lstcom.garbag - 1];
    s4.item[*iptr - 1] = 0;
    s4.nextt[*iptr - 1] = 0;
    return 0;
  }

  *iptr = s4.item[lstcom.garbag - 1];
  s4.item[lstcom.garbag - 1] = s4.nextt[*iptr - 1];

  s4.item[*iptr - 1] = 0;
  s4.nextt[*iptr - 1] = 0;
  return 0;
} /* new */

INT
newbas(INT *index)
{
  if (s_1.indbas + 10 > MAXBAS) {
    error("basis area overflow", 19, 2, 0, 19);
  }
  ++s_1.nstate;
  s_1.basis[s_1.indbas - 1] = s_1.nstate;
  s_1.basis[s_1.indbas] = 0;
  s_1.basis[s_1.indbas + 1] = -1;
  s_1.basis[s_1.indbas + 2] = 0;
  *index = s_1.indbas;
  s_1.indbas += 4;
  return 0;
} /* newbas */

INT
newcs(INT *is, INT *iptr)
{
  INT i, i0000;
  INT ihead;

  ihead = g_1.rgtuse[s4.item[*is - 1] - 1];
  i = ihead;

  while (i != 0) {
    if (lcompr(&s4.nextt[s4.item[i - 1] - 1], is) != 0) {
      *iptr = s4.item[i - 1];
      ++s4.item[*iptr - 1];
      rel(is);
      return 0;
    }
    i = s4.nextt[i - 1];
  }

  new (iptr);
  s4.item[*iptr - 1] = 1;
  s4.nextt[*iptr - 1] = *is;
  new (&i);
  s4.item[i - 1] = *iptr;
  s4.nextt[i - 1] = ihead;
  i0000 = s4.item[*is - 1];
  g_1.rgtuse[i0000 - 1] = i;

  return 0;
} /* newcs */

INT
newred(INT *ibasis, INT *imax)
{
  INT i;

  /*     set imax to the maximum number of reductions there is space for. */
  i = *ibasis + s_1.basis[*ibasis] * 3 + 6;
  *imax = s_1.basis[i - 1];
  /*     if space was previously allocated for the reductions for this basis */
  /*     set then reuse that space. the length required will never change */
  /*     since the reductions are based on the completed basis and the basis */
  /*     is never changed, only its context sets. */
  if (s_1.basis[i] == 0) {
    *imax = MAXRED - s_1.nxtred + 1;
    s_1.basis[i] = s_1.nxtred;
  }

  s_1.basis[i - 1] = 0;
  return 0;
} /* newred */

INT
newtrn(INT *ibasis, INT *imax)
{
  INT i;

  /*     set imax to the maximum number of transitions there is space for. */
  i = *ibasis + s_1.basis[*ibasis] * 3 + 4;
  *imax = s_1.basis[i - 1];
  /*     if space was previously allocated for the transitions for this basis */
  /*     set then reuse that space. the length required will never change */
  /*     since the transitions are based on the completed basis and the basis */
  /*     is never changed, only its context sets. */
  if (s_1.basis[i] == 0) {
    *imax = MAXTRN - s_1.nxttrn + 1;
    s_1.basis[i] = s_1.nxttrn;
  }

  s_1.basis[i - 1] = 0;
  return 0;
} /* newtrn */

INT
output(FILE *unit, INT *line, INT nchars)
{
  INT i1;
  INT i;

  /* Parameter adjustments */
  --line;

  /* Function Body */
  a1tos1(&line[1], &line[1], nchars);
  wtline(unit, &line[1], nchars);
  i1 = nchars;
  for (i = 1; i <= i1; ++i) {
    line[i] = ' ';
  }
  return 0;
} /* output */

INT
pntbas(INT iprod, INT idot, INT iptr)
{
  INT i1, i2;

  INT i, j, l, ip, iend;
  INT ibase;
  INT istart;
  DECL_LINE(132);

  /*     first print the left side of the production. */
  l = 2;
  i1 = l + 5;
  condec(iprod, line, l, i1);
  l = 9;
  ibase = g_1.prdind[iprod - 1];
  movstr(g_1.vocab[g_1.prodcn[ibase - 1] - 1], line, &l, 120);
  line[l] = ':';
  line[l + 1] = ':';
  line[l + 2] = '=';
  /*     print out the right side of the production. insert the dot before */
  /*     the idot - th right hand side symbol. */
  l += 5;
  istart = ibase + 2;
  iend = g_1.prodcn[ibase] + ibase + 1;
  j = 1;
  i1 = iend;
  for (i = istart; i <= i1; ++i) {
    if (length(&g_1.vocab[g_1.prodcn[i - 1] - 1]) + l > 118) {
      i2 = l - 1;
      output(files.dbgfil, line, i2);
      l = 17;
    }
    if (j == idot) {
      line[l - 1] = '.';
      l += 2;
    }

    movstr(g_1.vocab[g_1.prodcn[i - 1] - 1], line, &l, 120);
    ++l;
    ++j;
  }
  if (j == idot) {
    line[l - 1] = '.';
    l += 2;
  }

  i1 = l - 1;
  output(files.dbgfil, line, i1);
  if (iptr == 0) {
    return 0;
  }
  /*     now print the associated context set. */
  ip = iptr;
  l = 11;

  do {
    if (length(&g_1.vocab[s4.item[ip - 1] - 1]) + l >= 121) {
      i1 = l - 1;
      output(files.dbgfil, line, i1);
      l = 12;
    }

    movstr(g_1.vocab[s4.item[ip - 1] - 1], line, &l, 120);
    ++l;
    ip = s4.nextt[ip - 1];
  } while (ip != 0);

  i1 = l - 1;
  output(files.dbgfil, line, i1);

  return 0;
} /* pntbas */

INT
pntset(void)
{
  static const char *hdg1 =
      "    c   o   n   f   i   g   u   r   a   t   i   o  "
      " n       s   e   t   :   ";

  static const char *hdg2 =
      "    t   h   e       t   r   a   n   s   i   t   i  "
      " o   n   s   :   ";

  static const char *hdg3 =
      "    t   h   e       r   e   d   u   c   t   i   o  "
      " n   s   :   ";

  static const char *msg1 =
      "    *   *   *   i   n   t   e   r   s   e   c   t  "
      " i   o   n       w   i   t   h       t   r   a   n   s   i  "
      " t   i   o   n       t   o   ";

  static const char *msg2 =
      "    *   *   *   i   n   t   e   r   s   e   c   t  "
      " i   o   n       w   i   t   h       r   e   d   u   c   t  "
      " i   o   n   ";

  INT i1, i2, i3;

  INT i, j, k, l, n, ii;
  INT iend, jend, kend, lend;
  INT iptr;
  INT itemp;
  INT lstrt;
  INT jstart, kstart;
  DECL_LINE(132);

  new (&itemp);
  for (i = 1; i <= 120; ++i) {
    line[i - 1] = ' ';
  }
  i = 1;
  n = 1;
  line[0] = '1';

  do {
    if (switches.dbgasw) {
      output(files.dbgfil, line, 1);
      output(files.dbgfil, line, 1);
      for (j = 1; j <= 19; ++j) {
        line[j - 1] = hdg1[j - 1];
      }
      l = 20;
      i1 = l + 5;
      condec(s_1.basis[i - 1], line, l, i1);
      output(files.dbgfil, line, 25);
    }

    iend = s_1.basis[i];
    i += 4;
    if (s_1.basis[i - 1] == 1) {
      if (s_1.basis[i] > 3) {
        s_1.ifinal = s_1.basis[i - 5];
      }
    }

    i1 = iend;
    for (j = 1; j <= i1; ++j) {
      if (switches.dbgasw) {
        pntbas(s_1.basis[i - 1], s_1.basis[i], s4.nextt[s_1.basis[i + 1] - 1]);
      }
      delcs(&s_1.basis[i + 1]);
      i += 3;
    }
    if (switches.dbgasw) {
      /*     print the transitions ... */
      jstart = s_1.basis[i];
      jend = jstart + s_1.basis[i - 1] - 1;
      if (jstart <= jend) {
        for (j = 3; j <= 19; ++j) {
          line[j - 1] = hdg2[j - 3];
        }
        output(files.dbgfil, line, 19);
        l = 10;
        i1 = jend;
        for (j = jstart; j <= i1; j += 18) {
          kstart = j;
          kend = j + 17;
          if (kend > jend) {
            kend = jend;
          }
          i2 = kend;
          for (k = kstart; k <= i2; ++k) {
            i3 = l + 5;
            condec(s_1.basis[s_1.tran[k - 1] - 1], line, l, i3);
            l += 6;
          }
          i2 = l - 1;
          output(files.dbgfil, line, i2);
          l = 10;
        } /* j */
      }
    }

    i += 2;
    jstart = s_1.basis[i];
    jend = jstart + (s_1.basis[i - 1] << 1) - 1;
    if (jstart > jend) {
      i += 2;
      ++n;
      continue; /* break the outmost while */
    }
    if (switches.dbgasw) {
      for (j = 3; j <= 18; ++j) {
        line[j - 1] = hdg3[j - 3];
      }
      output(files.dbgfil, line, 18);
    }

    i1 = jend;
    for (j = jstart; j <= i1; j += 2) {
      if (switches.dbgasw) {
        l = 10;
        i2 = l + 5;
        condec(s_1.red[j - 1], line, l, i2);
        l += 8;

        /*     print the context set for this reduction. */

        iptr = s4.nextt[s_1.red[j] - 1];

        do {
          if (length(&g_1.vocab[s4.item[iptr - 1] - 1]) + l >= 121) {
            i2 = l - 1;
            output(files.dbgfil, line, i2);
            l = 20;
          }
          movstr(g_1.vocab[s4.item[iptr - 1] - 1], line, &l, 120);
          ++l;
          iptr = s4.nextt[iptr - 1];
        } while (iptr != 0);

        i2 = l - 1;
        output(files.dbgfil, line, i2);
      }

      /*     test for conflicts in the state. */
      lstrt = s_1.basis[i - 2];
      lend = lstrt + s_1.basis[i - 3] - 1;
      if (lstrt <= lend) {
        i2 = lend;
        for (l = lstrt; l <= i2; ++l) {
          s4.item[itemp - 1] =
              g_1.prodcn[g_1.prdind[s_1.basis[s_1.tran[l - 1] + 3] - 1] +
                         s_1.basis[s_1.tran[l - 1] + 4] - 1];
          if (lint(&itemp, &s4.nextt[s_1.red[j] - 1]) == 0) {
            continue;
          }
          adqcom.adequt = 0;
          if (!switches.dbgasw) {
            continue;
          }
          for (ii = 1; ii <= 35; ++ii) {
            line[ii - 1] = msg1[ii - 1];
          }
          condec(s_1.basis[s_1.tran[l - 1] - 1], line, 36, 41);
          output(files.dbgfil, line, 41);
        } /* l */
      }

      if (j == jstart) {
        continue; /* j loop */
      }
      lend = j - 2;
      i2 = lend;
      for (l = jstart; l <= i2; l += 2) {
        if (lint(&s4.nextt[s_1.red[l] - 1], &s4.nextt[s_1.red[j] - 1]) == 0) {
          continue;
        }
        adqcom.adequt = 0;
        if (!switches.dbgasw) {
          continue;
        }
        for (ii = 1; ii <= 31; ++ii) {
          line[ii - 1] = msg2[ii - 1];
        }
        condec(s_1.red[l - 1], line, 32, 37);
        output(files.dbgfil, line, 37);
      } /* l */
    }   /* j */

    i += 2;
    ++n;
  } while (n <= s_1.nstate);

  rel(&itemp);
  if (adqcom.adequt == 0) {
    error("this grammar is not lr(1)", 25, 1, 0, 25);
  }
  return 0;
} /* pntset */

INT
prntgm(void)
{
  INT i1, i2;
  INT i, j, k, l, m, np, lhs, iend, kend;
  INT ibase;
  INT istart;
  DECL_LINE(132);

  /*     generate the heading for the terminals and non - terminals */

  if (switches.runosw) {
    a4tos1(".GS 2 \"TERMINAL AND NON-TERMINAL SYMBOLS\"", line, 41);
    wtline(files.lstfil, line, 41);
    a4tos1(".CS", line, 3);
    wtline(files.lstfil, line, 3);
    /*     blank line */
    wtline(files.lstfil, line, 0);
  } else {
    wtpage(files.lstfil);
    a4tos1("    GRAMMAR CROSS REFERENCE LISTING", line, 35);
    wtline(files.lstfil, line, 35);
    /*     blank line */
    wtline(files.lstfil, line, 0);
    a4tos1("    TERMINAL AND NON-TERMINAL SYMBOLS", line, 37);
    wtline(files.lstfil, line, 37);
    /*     blank line */
    wtline(files.lstfil, line, 0);
  }

  /*     print out the terminals and non - terminals */

  for (i = 1; i <= 120; ++i) {
    line[i - 1] = ' ';
  }
  /* Computing MAX */
  i1 = g_1.nterms, i2 = g_1.nvoc - g_1.nterms;
  iend = max(i1, i2);
  i1 = iend;
  for (i = 1; i <= i1; ++i) {
    /*     starting location in listing file */
    j = 7;
    if (i <= g_1.nterms) {
      condec(i, line, 2, 5);
      movstr(g_1.vocab[i - 1], line, &j, 120);
      ++j;
    }

    if (g_1.nterms + i <= g_1.nvoc) {
      if (j < 37) {
        j = 37;
      }
      i2 = g_1.nterms + i;
      condec(i2, line, j, 40);
      j += 5;
      movstr(g_1.vocab[g_1.nterms + i - 1], line, &j, 120);
    }

    i2 = j - 1;
    output(files.lstfil, line, i2);
  }

  /*     print the productions heading */

  if (switches.runosw) {
    a4tos1(".CE", line, 3);
    wtline(files.lstfil, line, 3);
    a4tos1(".bp", line, 3);
    wtline(files.lstfil, line, 3);
    a4tos1(".GS 2 \"THE PRODUCTIONS\"", line, 23);
    wtline(files.lstfil, line, 23);
    a4tos1(".CS", line, 3);
    wtline(files.lstfil, line, 3);
  } else {
    wtpage(files.lstfil);
    a4tos1("    THE PRODUCTIONS", line, 19);
    wtline(files.lstfil, line, 19);
  }

  /*     print out the productions */
  for (i = 1; i <= 120; ++i) {
    line[i - 1] = ' ';
  }
  i = 1;

  do {
    wtline(files.lstfil, line, 0);
    j = 7;
    lhs = g_1.prodcn[g_1.prdind[i - 1] - 1];
    movstr(g_1.vocab[lhs - 1], line, &j, 120);
    line[j] = ':';
    line[j + 1] = ':';
    istart = j + 3;
    m = '=';
    kend = g_1.nprods[lhs - 1];
    i1 = kend;
    for (k = 1; k <= i1; ++k) {
      j = 2;
      i2 = i - 1;
      condec(i2, line, j, 5);
      line[istart - 1] = m;
      m = '|';
      j = istart + 2;
      ibase = g_1.prdind[i - 1] + 1;
      np = g_1.prodcn[ibase - 1];
      if (np == 0) {
        i2 = j - 1;
        output(files.lstfil, line, i2);
        ++i;
        continue;
      }
      l = 1;

      while (1) {
        movstr(g_1.vocab[g_1.prodcn[ibase + l - 1] - 1], line, &j, 120);
        ++j;
        ++l;
        if (l > np) {
          break;
        }
        if (length(&g_1.vocab[g_1.prodcn[ibase + l - 1] - 1]) + j <= 72) {
          continue;
        }
        i2 = j - 1;
        output(files.lstfil, line, i2);
        j = istart + 5;
      }

      i2 = j - 1;
      output(files.lstfil, line, i2);
      ++i;
    } /* k */
  } while (i <= g_1.numprd);

  if (switches.runosw) {
    a4tos1(".CE", line, 3);
    wtline(files.lstfil, line, 3);
  }

  return 0;
} /* prntgm */

INT
putsem(INT curlhs, INT *curprd)
{
  INT i1;
  INT c, i, ii;
#define SM_LINESZ 50
  DECL_LINE(80);
  /*
   * emit #define NAME <value>
   * <value> will be presented as a 4 digit int
   * the max length of NAME is ~(SM_LINESZ-13) (~37 chars)
   */
  ++ptsem.pnum;
  /*  60 character in the following string  */
  a4tos1("#define                                                     ", line,
         SM_LINESZ - 1);
  ii = 8;
  movstr(curlhs, line, &ii, SM_LINESZ - 6);
  i1 = ii - 8;
  a1tos1(&line[7], &line[7], i1);
  line[7] = 32;
  i1 = ii;
  for (i = 9; i <= i1; ++i) {
    c = line[i - 1];
    if (c == 32) {
      line[i - 1] = 95;
      continue;
    }

    if (c == 47) {
      line[i - 1] = 95;
      continue;
    }

    if (c == 61) {
      line[i - 1] = 95;
      continue;
    }

    if (c < 97) {
      continue;
    }
    if (c <= 122) {
      line[i - 1] = c - 32;
    }
  } /* i */

  i32tos(&g_1.nprods[curlhs - 1], &line[ii - 2], 3, 1, 10, 0);
  i1 = ptsem.pnum - 1;
  i32tos(&i1, &line[SM_LINESZ - 4], 4, 1, 10, 1);
  wtline(files.semfil, line, SM_LINESZ);
  if (ptsem.brkflg != 1) {
    return 0;
  }
  /*     a semantic break directive was just processed by readln: */
  ptsem.brkflg = 0;
  a4tos1("#define SEM       ", line, 18);
  /*     was line(13) */
  i32tos(&ptsem.semnum, &line[11], 1, 0, 10, 1);
  i32tos(curprd, &line[14], 4, 0, 10, 1);
  wtline(files.semfil, line, 18);
  ++ptsem.semnum;
  /*     pnum = 0 1/22/91 don't reset case values */

  return 0;
} /* putsem */

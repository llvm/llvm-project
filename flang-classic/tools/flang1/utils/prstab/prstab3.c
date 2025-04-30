/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR parser (part 3)
 *
 */

#include "lrutils.h"
#include "prstab.h"

INT
rdgram(void)
{
  INT i1;
  INT i, k;
  INT temp, token;
  INT curlhs, curprd, prdptr;

  /*     current production number */
  curprd = 1;
  /*     avail pointer into prodcn array */
  prdptr = 1;

  while (1) {
    /*     read left hand side symbol and process it: */
    /*     left hand side symbol */
    scan(&curlhs);
    if (curlhs == -4) {
      break;
    }
    if (curlhs <= 0) {
      error("left hand side symbol expected", 30, 12, 0, 30);
    }
    if (g_1.lftuse[curlhs - 1] != 0) {
      error("used in two sets of definitions", 31, 12, curlhs, 31);
    }
    g_1.lftuse[curlhs - 1] = 1;
    g_1.frsprd[curlhs - 1] = curprd;
    g_1.nprods[curlhs - 1] = 0;
    /*     next symbol must be '::=': */
    scan(&temp);
    if (temp != -1) {
      error("::= symbol expected", 19, 12, 0, 19);
    }

    /*     loop for each right-hand side */
    while (1) {
      if (curprd > MAXPROD) {
        error("too many productions", 20, 12, 0, 20);
      }
      g_1.prdind[curprd - 1] = prdptr;
      ++g_1.nprods[curlhs - 1];
      g_1.prodcn[prdptr - 1] = curlhs;
      /*     count symbols in this rhs */
      k = 0;

      while (1) {
        scan(&token);
        if (token <= 0) {
          break;
        }
        /* add token to rhs: */
        g_1.rgtuse[token - 1] = 1;
        ++k;
        if (k + prdptr + 1 > MAXPRDC) {
          error("grammar too large", 17, 12, 0, 17);
        }
        g_1.prodcn[prdptr + 1 + k - 1] = token;
      }

      if (token == -1) {
        error("misplaced ::=", 13, 12, 0, 13);
      }
      if (token == -4) {
        error("unexpected eof", 14, 2, 0, 14);
      }
      /*     finish processing of right-hand side: */
      g_1.prodcn[prdptr] = k;
      prdptr = prdptr + 2 + k;
      if (switches.semsw) {
        putsem(curlhs, &curprd);
      }
      ++curprd;
      if (token == -3) {
        break;
      }
      if (token != -2) {
        error("rdgram - illegal token", 22, 12, 0, 22);
      }
    } /* rhs while */
  }   /* while */

  /* count the terminal symbols: */
  g_1.nterms = 0;
  i1 = string_1.shdptr;
  for (i = 1; i <= i1; ++i) {
    if (g_1.lftuse[i - 1] != 0) {
      continue;
    }
    if (g_1.rgtuse[i - 1] != 0) {
      ++g_1.nterms;
    }
  }
  g_1.nvoc = string_1.shdptr;
  g_1.numprd = curprd - 1;
  return 0;
} /* rdgram */

INT
readln(void)
{
  INT status;

  do {
    while (1) {
      status = rdline(files.infile, readcm.linbuf, 81);
      ++readcm.lineno;
      readcm.curchr = 1;
      if (readcm.linbuf[0] == '#') {
          /* ignore the comment line in .txt and .tki file */
          continue;
      }
      if (status == -1) {
        readcm.linbuf[0] = 257;
        status = 1;
        break;
      }

      if (status == 81) {
        status = 80;
        error("line length exceeds 80 characters", 33, 11, 0, 33);
        break;
      }

      if (readcm.linbuf[0] != 46) {
        break;
      }
      if (readcm.linbuf[1] != 66) {
        break;
      }
      /*     process semantic break directive */
      ptsem.brkflg = 1;
    } /* while */

    while (1) {
      if (readcm.linbuf[status - 1] != 32) {
        if (readcm.linbuf[status - 1] != 9) {
          break;
        }
      }

      if (status <= 0) {
        break;
      }
      --status;
    }

  } while (status <= 0);

  readcm.linbuf[status] = 258;
  return 0;
} /* readln */

INT
rel(INT *iptr)
{
  s4.item[*iptr - 1] = s4.nextt[*iptr - 1];
  s4.nextt[*iptr - 1] = lstcom.garbag;
  lstcom.garbag = *iptr;
  return 0;
} /* rel */

INT
scan(INT *token)
{
  INT i1;
  INT c;

  /*     function called to enter symbol in symbol table */
  while (1) {
    c = readcm.linbuf[readcm.curchr - 1];
    readcm.fstchr = readcm.curchr;
    ++readcm.curchr;
    if (c == 32) {
      continue;
    }
    if (c == 9) {
      continue;
    }

    if (c == 60) {
      while (1) {
        if (readcm.linbuf[readcm.curchr - 1] == 62) {
          break;
        }
        if (readcm.linbuf[readcm.curchr - 1] == 258) {
          break;
        }
        ++readcm.curchr;
      }

      if (readcm.linbuf[readcm.curchr - 1] == 258) {
        error("unmatched left angle bracket", 28, 12, 0, 28);
      }
      ++readcm.curchr;
      i1 = readcm.curchr - readcm.fstchr;
      *token = enter(&readcm.linbuf[readcm.fstchr - 1], &i1);
      return 0;
    }

    if ((c == 58) && (readcm.linbuf[readcm.curchr - 1] == 58) &&
        (readcm.linbuf[readcm.curchr] == 61)) {
      readcm.curchr += 2;
      *token = -1;
      return 0;
    }

    if (c == 124) {
      *token = -2;
      if (readcm.linbuf[readcm.curchr - 1] == 258) {
        readln();
      }
      return 0;
    }

    if (c == 39) {
      while ((readcm.linbuf[readcm.curchr - 1] != 39) &&
             (readcm.linbuf[readcm.curchr - 1] != 258)) {
        ++readcm.curchr;
      }

      if (readcm.linbuf[readcm.curchr - 1] == 258) {
        error("unmatched single quote", 22, 12, 0, 22);
      }

      ++readcm.curchr;
      i1 = readcm.curchr - readcm.fstchr;
      *token = enter(&readcm.linbuf[readcm.fstchr - 1], &i1);
      return 0;
    }

    if (c == 258) {
      readln();
      *token = -3;
      return 0;
    }

    if (c == 257) {
      *token = -4;
      return 0;
    }

    /*     continuation line */
    if (c != 95) {
      break;
    }
    readln();
  } /* while */

  if (c == 62) {
    error("isolated right angle bracket", 28, 12, 0, 28);
    return 0;
  }

  while ((readcm.linbuf[readcm.curchr - 1] != 32) &&
         (readcm.linbuf[readcm.curchr - 1] != 9) &&
         (readcm.linbuf[readcm.curchr - 1] != 258)) {
    ++readcm.curchr;
  }

  i1 = readcm.curchr - readcm.fstchr;
  *token = enter(&readcm.linbuf[readcm.fstchr - 1], &i1);

  return 0;
} /* scan */

INT
sortcg(INT *nsets)
{
  INT i1, i2;
  INT i, j, k, iend, itemp;
  INT jstart;

  /*     bubble sort the configuration sets in the scrtch array. */
  iend = *nsets - 3;
  /*     fix from al shannon: return if iend is zero */
  /*     J.R.L. -  8 dec 78 */
  if (iend == 0) {
    return 0;
  }

  i1 = iend;
  for (i = 1; i <= i1; i += 3) {
    jstart = i + 3;
    i2 = *nsets;
    for (j = jstart; j <= i2; j += 3) {

      /*     if either both configs are reduce configs or the */
      /*     second one is then no exchange should take place. */
      /*     a config is a reduce config if the dot is at the */
      /*     end of the production. */

      if (scrtch[j] > g_1.prodcn[g_1.prdind[scrtch[j - 1] - 1]]) {
        continue;
      }

      /*     if the first config is a reduce and the second isn't, */
      /*     exchange. */

      if (scrtch[i] <= g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1]]) {
        /*     otherwise compare the symbols after the dot. */
        if (g_1.prodcn[g_1.prdind[scrtch[j - 1] - 1] + scrtch[j]] >
            g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1] + scrtch[i]]) {
          continue;
        }
      }

      for (k = 1; k <= 3; ++k) {
        itemp = scrtch[i + k - 2];
        scrtch[i + k - 2] = scrtch[j + k - 2];
        scrtch[j + k - 2] = itemp;
      }
    }
  }

  return 0;
} /* sortcg */

INT
sortgm(void)
{
  INT i1, i2;
  INT i, j, k, m1, n1, n2, m2, i0000, iend;
#define moved (scrtch + 2000)
#define xlate (scrtch)
  INT itemp;
  INT istart;

  /*     the vocabulary items are sorted into terminal non - terminal order, */
  /*     alphabetically within each group, using a heap sort. */
  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    g_1.vocab[i - 1] = i;
  }
  bildhp();
  iend = g_1.nvoc - 1;
  i1 = iend;
  for (i = 1; i <= i1; ++i) {
    itemp = g_1.vocab[g_1.nvoc - i];
    g_1.vocab[g_1.nvoc - i] = g_1.vocab[0];
    g_1.vocab[0] = itemp;
    i2 = g_1.nvoc - i;
    hepify(1, i2);
  }

  /*     now that the vocabulary is sorted build a translation table and */
  /*     translate from the original token numbers to the sorted numbers. */

  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    i0000 = g_1.vocab[i - 1];
    xlate[i0000 - 1] = i;
    moved[i - 1] = 0;
  }

  i = 1;
  k = 0;

  do {
    g_1.prodcn[i - 1] = xlate[g_1.prodcn[i - 1] - 1];
    istart = i + 2;
    iend = g_1.prodcn[i] + i + 1;
    if (iend >= istart) {
      i1 = iend;
      for (j = istart; j <= i1; ++j) {
        g_1.prodcn[j - 1] = xlate[g_1.prodcn[j - 1] - 1];
      }
    }

    i = i + g_1.prodcn[i] + 2;
    ++k;
  } while (k < g_1.numprd);

  g_1.goal = xlate[g_1.goal - 1];

  /*     now reorder arrays indexed by token number. */

  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    j = i;
    n1 = g_1.frsprd[j - 1];
    n2 = g_1.nprods[j - 1];

    while (moved[j - 1] != 1) {
      k = xlate[j - 1];
      m1 = g_1.frsprd[k - 1];
      m2 = g_1.nprods[k - 1];
      g_1.frsprd[k - 1] = n1;
      g_1.nprods[k - 1] = n2;
      moved[j - 1] = 1;
      n1 = m1;
      n2 = m2;
      j = k;
    }
  }
  return 0;
} /* sortgm */

#undef xlate
#undef moved

INT
strcomp(INT *iptr1, INT *iptr2, INT *irslt)
{
  INT i1, i2, i3;
  INT i, i_1, i_2, len1, len2, iend, jrslt;

  *irslt = 0;
  i_1 = s1_1.sthead[*iptr1 - 1] - 1;
  i_2 = s1_1.sthead[*iptr2 - 1] - 1;
  len1 = s1_1.sthead[*iptr1] - i_1 - 1;
  len2 = s1_1.sthead[*iptr2] - i_2 - 1;
  iend = len1;
  if (len2 < iend) {
    iend = len2;
  }
  i1 = iend;
  for (i = 1; i <= i1; ++i) {
    i2 = i_1 + i;
    i3 = i_2 + i;
    chrcmp(&i2, &i3, &jrslt);
    if (jrslt < 0) {
      *irslt = -1;
      return 0;
    }
    if (jrslt != 0) {
      *irslt = 1;
      return 0;
    }
  }

  if (len1 == len2) {
    return 0;
  }
  if (len1 <= len2) {
    *irslt = -1;
    return 0;
  }

  *irslt = 1;
  return 0;
} /* strcomp */

INT
tablea(void)
{
  INT i1, i2, i3, i4;
  INT i, j, k, l, iend, jend;
  INT ient, last, iptr, lptr;
  INT lngvcb;
  INT lngprd, jstart;
  INT const_1 = 1;
  DECL_LINE(80);

  /*     print header */

  a4tos1("/**************************************", line, 39);
  wtline(files.datfil, line, 39);
  a4tos1("* This file is produced by a utility, *", line, 39);
  wtline(files.datfil, line, 39);
  a4tos1("* do not modify it directly.          *", line, 39);
  wtline(files.datfil, line, 39);
  a4tos1("**************************************/", line, 39);
  wtline(files.datfil, line, 39);
  /*     blank line */
  wtline(files.datfil, line, 0);
  /*     blank line */
  wtline(files.datfil, line, 0);
  /* output #defines for all this stuff! */
  /*     the number of states = nstates */
  /*     the final state = ifinal */
  /*     the initial state = 1 (always, as it happens) */
  /*     the size of the vocabulary = nvoc */
  /*     the number of terminal symbols = nterms */
  /*     the length of the longest symbol (is computed) */
  /*     the number of lookahead set choices = (nxtred - 1)/2 */
  /*     the number of transitions = nxttrn - 1 */
  /*     the number of lookahead sets = ncsets */
  /*     the number of productions = numprd */
  /*     the length of the longest right hand side (is computed) */

  /*     all numbers are in i5 format */
  a4tos1("#define PRS_NSTATES ", line, 20);
  i32tos(&s_1.nstate, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_IFINAL  ", line, 20);
  i32tos(&s_1.ifinal, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_IINIT   ", line, 20);
  i32tos(&const_1, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NVOC    ", line, 20);
  i32tos(&g_1.nvoc, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NTERMS  ", line, 20);
  i32tos(&g_1.nterms, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_SYMLEN  ", line, 20);
  lngvcb = 0;
  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    /* Computing MAX */
    i2 = lngvcb, i3 = length(&g_1.vocab[i - 1]);
    lngvcb = max(i2, i3);
  }
  i32tos(&lngvcb, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NLOOKS  ", line, 20);
  i1 = (s_1.nxtred - 1) / 2;
  i32tos(&i1, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NTRANS  ", line, 20);
  i1 = s_1.nxttrn - 1;
  i32tos(&i1, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NCSETS  ", line, 20);
  i32tos(&s_1.ncsets, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_NPRODS  ", line, 20);
  i32tos(&g_1.numprd, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  a4tos1("#define PRS_RHSLEN  ", line, 20);
  lngprd = 0;
  k = 2;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; ++i) {
    i2 = lngprd, i3 = g_1.prodcn[k - 1];
    lngprd = max(i2, i3);
    k = k + g_1.prodcn[k - 1] + 2;
  }
  i32tos(&lngprd, &line[20], 5, 0, 10, 1);
  wtline(files.datfil, line, 25);

  /*     print TRAN array */

  a4tos1("static short int tran[     ] = {", line, 32);
  i1 = s_1.nxttrn - 1;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }

  iend = s_1.nxttrn - 1;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      i3 = s_1.basis[s_1.tran[j - 1] - 1] - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);

      if (j < iend) {
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      l += 4;
    } /* j */
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print FTRN array */

  a4tos1("static short int ftrn[     ] = {", line, 32);
  i1 = s_1.nstate + 1;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  iend = s_1.nstate + 1;
  k = -2;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      if (j < iend) {
        k = k + s_1.basis[k + 3] * 3 + 8;
        i3 = s_1.basis[k - 1] - 1;
        i4 = l + 4;
        condec(i3, line, l, i4);
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      i3 = s_1.basis[k - 1] + s_1.basis[k - 2] - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);
      l += 4;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print ENT array */

  a4tos1("static short int ent [     ] = {", line, 32);
  i32tos(&s_1.nstate, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  k = 5;
  i1 = s_1.nstate;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > s_1.nstate) {
      jend = s_1.nstate;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      ient = g_1.prodcn[g_1.prdind[s_1.basis[k - 1] - 1] + s_1.basis[k] - 1];
      k = k + s_1.basis[k - 4] * 3 + 8;
      i3 = l + 4;
      condec(ient, line, l, i3);
      if (j < s_1.nstate) {
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      l += 4;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print FRED array */

  a4tos1("static short int fred[     ] = {", line, 32);
  i1 = s_1.nstate + 1;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  iend = s_1.nstate + 1;
  k = 0;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      if (j < iend) {
        k = k + s_1.basis[k + 1] * 3 + 8;
        i3 = (s_1.basis[k - 1] + 1) / 2 - 1;
        i4 = l + 4;
        condec(i3, line, l, i4);
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      i3 = ((s_1.basis[k - 2] << 1) + s_1.basis[k - 1] + 1) / 2 - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);
      l += 4;
    }
    output(files.datfil, line, l);
  }

  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print NSET array */

  a4tos1("static short int nset[     ] = {", line, 32);
  i1 = (s_1.nxtred - 1) / 2;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  iend = s_1.nxtred - 1;
  i1 = iend;
  for (i = 2; i <= i1; i += 20) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 18;
    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; j += 2) {
      i3 = s4.item[s_1.red[j - 1] - 1] - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);
      if (j < iend) {
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      l += 4;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print PROD array */

  a4tos1("static short int prod[     ] = {", line, 32);
  i1 = (s_1.nxtred - 1) / 2;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  iend = s_1.nxtred - 1;
  i1 = iend;
  for (i = 1; i <= i1; i += 20) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 18;
    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; j += 2) {
      i3 = s_1.red[j - 1] - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);
      if (j < iend - 1) {
        line[l + 4] = ',';
        l += 7;
        continue;
      }

      l += 4;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print LHS array */

  a4tos1("static short int lhs [     ] = {", line, 32);
  i32tos(&g_1.numprd, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  k = 1;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > g_1.numprd) {
      jend = g_1.numprd;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      i3 = l + 4;
      condec(g_1.prodcn[k - 1], line, l, i3);
      if (j >= g_1.numprd) {
        l += 4;
      } else {
        line[l + 4] = ',';
        l += 7;
      }

      k = k + g_1.prodcn[k] + 2;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print LEN array */

  a4tos1("static short int len [     ] = {", line, 32);
  i32tos(&g_1.numprd, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  k = 2;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; i += 10) {
    /*     start index */
    l = 3;
    jstart = i;
    jend = i + 9;
    if (jend > g_1.numprd) {
      jend = g_1.numprd;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      i3 = l + 4;
      condec(g_1.prodcn[k - 1], line, l, i3);
      if (j >= g_1.numprd) {
        l += 4;
      } else {
        line[l + 4] = ',';
        l += 7;
      }

      k = k + g_1.prodcn[k - 1] + 2;
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print LSET array */

  a4tos1("static short int lset[     ] = {", line, 32);
  i1 = s_1.ncsets + 1;
  i32tos(&i1, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  condec(0, line, 3, 7);
  line[7] = ',';
  iptr = s_1.lencsl;
  if (s_1.ncsets > 9) {
    iend = s_1.ncsets + 1;
  } else {
    iend = s_1.ncsets;
  }

  last = 1;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    jstart = i;
    if (i != 1) {
      /*     start index */
      l = 3;
      jend = i + 9;
    } else {
      /*     next start index */
      l = 10;
      jend = i + 8;
    }

    if (jend > iend) {
      jend = iend;
    }
    i2 = jend;
    for (j = jstart; j <= i2; ++j) {
      last += s4.item[iptr - 1];
      i3 = last - 1;
      i4 = l + 4;
      condec(i3, line, l, i4);
      if (j >= iend) {
        l += 4;
      } else {
        line[l + 4] = ',';
        l += 7;
      }

      iptr = s4.nextt[iptr - 1];
    }
    output(files.datfil, line, l);
  }
  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  /*     blank line */
  wtline(files.datfil, line, 0);

  /*     print LS array */

  a4tos1("static short int ls  [     ] = {", line, 32);
  i32tos(&s_1.lsets, &line[22], 5, 0, 10, 1);
  wtline(files.datfil, line, 32);
  for (i = 1; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  lptr = s_1.listcs;
  iptr = s4.nextt[s4.item[lptr - 1] - 1];

  INT flag = 0;
  do {
    for (i = 1; i <= 10; ++i) {
      while (iptr <= 0) {
        lptr = s4.nextt[lptr - 1];
        if (lptr <= 0) {
          flag = 1;
          break;
        }
        iptr = s4.nextt[s4.item[lptr - 1] - 1];
      }

      if (flag)
        break;

      /*     start index */
      if (i == 1) {
        l = 3;
      }
      i1 = l + 4;
      condec(s4.item[iptr - 1], line, l, i1);
      iptr = s4.nextt[iptr - 1];
      if (iptr > 0 || s4.nextt[lptr - 1] > 0) {
        line[l + 4] = ',';
        l += 7;
      } else {
        l += 4;
      }
    }

    output(files.datfil, line, l);
  } while (lptr > 0);

  a4tos1("};", line, 2);
  wtline(files.datfil, line, 2);
  return 0;
} /* tablea */

INT
tableu(void)
{
  INT i1, i2, i3;
  INT i, j, k, l, iend, jend, hold[11], iptr;
  INT lngvcb, hldidx;
  INT lngprd, lenptr;
  DECL_LINE(80);

  for (i = 2; i <= 80; ++i) {
    line[i - 1] = ' ';
  }
  line[0] = '1';
  output(files.dbgfil, line, 1);
  /*     begin by printing the control parameters for the parser. */
  /*     presumably, these can all be used by the programmer or the */
  /*     parser to allocate appropriate space for the tables.  there */
  /*     will be 11 lines of them */

  /*     the number of states = nstates */
  /*     the final state = ifinal */
  /*     the initial state = 1 (always, as it happens) */
  /*     the size of the vocabulary = nvoc */
  /*     the number of terminal symbols = nterms */
  /*     the length of the longest symbol (is computed) */
  /*     the number of lookahead set choices = (nxtred - 1)/2 */
  /*     the number of transitions = nxttrn - 1 */
  /*     the number of lookahead sets = ncsets */
  /*     the number of productions = numprd */
  /*     the length of the longest right hand side (is computed) */

  /*     all numbers are in i5 format flush right on 5 character */
  /*     boundaries. */
  condec(s_1.nstate, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(s_1.ifinal, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(1, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(g_1.nvoc, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(g_1.nterms, line, 1, 5);
  output(files.dbgfil, line, 5);

  lngvcb = 0;
  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    /* Computing MAX */
    i2 = lngvcb, i3 = length(&g_1.vocab[i - 1]);
    lngvcb = max(i2, i3);
    /* L20: */
  }
  condec(lngvcb, line, 1, 5);
  output(files.dbgfil, line, 5);

  i1 = (s_1.nxtred - 1) / 2;
  condec(i1, line, 1, 5);
  output(files.dbgfil, line, 5);

  i1 = s_1.nxttrn - 1;
  condec(i1, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(s_1.ncsets, line, 1, 5);
  output(files.dbgfil, line, 5);

  condec(g_1.numprd, line, 1, 5);
  output(files.dbgfil, line, 5);

  lngprd = 0;
  k = 2;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; ++i) {
    /* Computing MAX */
    i2 = lngprd, i3 = g_1.prodcn[k - 1];
    lngprd = max(i2, i3);
    k = k + g_1.prodcn[k - 1] + 2;
  }
  condec(lngprd, line, 1, 5);
  output(files.dbgfil, line, 5);

  /*     the vocabulary items appear one per line.  they are flush */
  /*     left and take exactly as many characters as necessary. */

  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    l = 1;
    i2 = length(&g_1.vocab[i - 1]);
    movstr(g_1.vocab[i - 1], line, &l, i2);
    i2 = length(&g_1.vocab[i - 1]);
    output(files.dbgfil, line, i2);
  }

  /*     the entrance symbol array is output 10 items per line, as is */
  /*     the case with all other arrays. */

  k = 5;
  i1 = s_1.nstate;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = s_1.nstate, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] =
          g_1.prodcn[g_1.prdind[s_1.basis[k - 1] - 1] + s_1.basis[k] - 1];
      k = k + s_1.basis[k - 4] * 3 + 8;
    }
    hlddmp(hold, &hldidx);
  }

  /*     the first lookahead array.  the extra last element will be */
  /*     tacked onto the last row, making it possibly 11 elements long. */

  k = 0;
  i1 = s_1.nstate;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = s_1.nstate, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      k = k + s_1.basis[k + 1] * 3 + 8;
      hold[hldidx - 1] = (s_1.basis[k - 1] + 2) / 2;
    }
    if (jend == s_1.nstate) {
      ++hldidx;
      hold[hldidx - 1] = ((s_1.basis[k - 2] << 1) + s_1.basis[k - 1] + 1) / 2;
    }

    hlddmp(hold, &hldidx);
  }

  /*     the firsttransition array, also with a potintial tack on. */

  k = -2;
  i1 = s_1.nstate;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = s_1.nstate, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      k = k + s_1.basis[k + 3] * 3 + 8;
      hold[hldidx - 1] = s_1.basis[k - 1];
      /* L100: */
    }
    if (jend == s_1.nstate) {
      ++hldidx;
      hold[hldidx - 1] = s_1.basis[k - 1] + s_1.basis[k - 2];
    }

    hlddmp(hold, &hldidx);
  }

  /*     the transition vector with no tackon. */

  iend = s_1.nxttrn - 1;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = iend, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] = s_1.basis[s_1.tran[j - 1] - 1];
    }
    hlddmp(hold, &hldidx);
  }

  /*     the ptrlookahead vector, known in fortran as nlookset. */

  iend = (s_1.nxtred - 1) / 2;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = iend, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] = s4.item[s_1.red[(j << 1) - 1] - 1];
    }
    hlddmp(hold, &hldidx);
  }

  /*     vector choosereduction also known as prod */

  iend = (s_1.nxtred - 1) / 2;
  i1 = iend;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = iend, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] = s_1.red[(j << 1) - 2];
    }
    hlddmp(hold, &hldidx);
  }

  /*     the lookahead sets have a strange format.  each set starts a */
  /*     new line.  the first element is the number of terminals in the */
  /*     set and the rest is the terminal numbers from the set. */

  lenptr = s_1.lencsl;
  lstcom.lstptr = s_1.listcs;
  iptr = s4.nextt[s4.item[lstcom.lstptr - 1] - 1];

  while (lenptr > 0) {
    hldidx = 1;
    hold[0] = s4.item[lenptr - 1];
    INT flag1 = 0, flag2 = 0;

    while (!flag1) {
      if (iptr > 0) {
        flag2 = 1;
      }

      do {
        if (!flag2) {
          hlddmp(hold, &hldidx);
          hldidx = 0;
          if (iptr <= 0) {
            flag1 = 1;
            break;
          }
        }

        ++hldidx;
        hold[hldidx - 1] = s4.item[iptr - 1];
        iptr = s4.nextt[iptr - 1];
      } while (hldidx >= 10);

      flag2 = 0;
    }

    lenptr = s4.nextt[lenptr - 1];
    lstcom.lstptr = s4.nextt[lstcom.lstptr - 1];
    iptr = s4.nextt[s4.item[lstcom.lstptr - 1] - 1];
  }

  /*     vector redlength also known as lenred. */
  k = 2;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = g_1.numprd, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] = g_1.prodcn[k - 1];
      k = k + g_1.prodcn[k - 1] + 2;
    }
    hlddmp(hold, &hldidx);
  }

  /*     vector lefthandside also known as leftside */

  k = 1;
  i1 = g_1.numprd;
  for (i = 1; i <= i1; i += 10) {
    /* Computing MIN */
    i2 = g_1.numprd, i3 = i + 9;
    jend = min(i2, i3);
    hldidx = 0;
    i2 = jend;
    for (j = i; j <= i2; ++j) {
      ++hldidx;
      hold[hldidx - 1] = g_1.prodcn[k - 1];
      k = k + g_1.prodcn[k] + 2;
    }
    hlddmp(hold, &hldidx);
  }
  return 0;
} /* tableu */

INT
tokmac(void)
{
  INT cc, i;
  INT incnt;
#define xlate (scrtch)
  INT iterm;
  INT nutok;
  INT outval;
  DECL_LINE(80);

  /*     read token input file and write file of token macro definitions: */
  /*     this routine uses xlate table created in scratch area by sortgm */
  /*     and so must be called immediately after sortgm. */
  /*     local variables */
  nutok = g_1.nterms;
  incnt = 0;
  /*     switch input file */
  files.infile = files.tokin;
  readcm.lineno = 0;

  while (1) {
    readln();
    scan(&iterm);
    if (iterm == -4)
      break; /* exit */

    if (g_1.lftuse[iterm - 1] != 0 || iterm > g_1.nvoc) {
      error("is not a terminal symbol", 24, 11, iterm, 24);
      ++nutok;
      outval = nutok;
    } else {
      outval = xlate[iterm - 1];
    }
    ++incnt;

    while (1) {
      if (readcm.linbuf[readcm.curchr - 1] != 32) {
        if (readcm.linbuf[readcm.curchr - 1] != 9) {
          break;
        }
      }
      ++readcm.curchr;
    }

#define TK_LINESZ 50
    /*  60 character in the following string  */
    /*
     * emit #define NAME <value>
     * <value> will be presented as a 4 digit int
     * the max length of NAME is ~(TK_LINESZ-13) (~37 chars)
     */
    a4tos1("#define                                                     ", line,
           TK_LINESZ);
    i32tos(&outval, &line[TK_LINESZ - 4], 4, 0, 10, 1);
    for (i = 1; i <= TK_LINESZ - 13; ++i) {
      cc = readcm.linbuf[readcm.curchr + i - 2];
      if (cc == 258) {
        break;
      }
      if (cc >= 97) {
        if (cc <= 122) {
          line[i + 7] = cc - 32;
          continue;
        }
      }
      line[i + 7] = cc;
    } /* i */
    wtline(files.tokout, line, TK_LINESZ);
  } /* outer while */

  if (incnt > g_1.nterms) {
    error("more entries in token input file than tokens", 44, 1, 0, (INT)44);
  }
  if (incnt < g_1.nterms) {
    error("token input file is missing tokens", 34, 1, 0, 34);
  }
  return 0;
} /* tokmac */

#undef xlate

INT
trnred(INT *ibasis, INT *jmax)
{
  INT i1;
  INT i, nb, ich, lhs, maxr, maxt;
  INT flag = 0;
  INT ipath;
  INT nbasis;

  i = 1;
  newtrn(ibasis, &maxt);
  if (scrtch[i] <= g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1]]) {
    do {
      lhs = g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1] + scrtch[i]];
      newbas(&nbasis);

      do {
        i1 = scrtch[i] + 1;
        addbas(nbasis, scrtch[i - 1], i1, scrtch[i + 1]);
        delcs(&scrtch[i + 1]);
        i += 3;
        ipath = 0;
        if (i > *jmax) {
          flag = 1;
          break;
        }
        if (scrtch[i] > g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1]]) {
          flag = 1;
          break;
        }
      } while (lhs == g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1] + scrtch[i]]);

      if (!flag) {
        ipath = 1;
      }

      flag = 0; /* recover */
      endbas(&nbasis);
      merge(&nbasis, &nb, &ich);
      if (ich != 0) {
        enque(&nb);
      }
      addtrn(ibasis, &nb, &maxt);
    } while (ipath != 0);
  }

  endtrn(ibasis);
  newred(ibasis, &maxr);
  if (i <= *jmax) {
    do {
      addred(ibasis, &scrtch[i - 1], &scrtch[i + 1], &maxr);
      delcs(&scrtch[i + 1]);
      i += 3;
    } while (i < *jmax);
  }
  endred(ibasis);
  return 0;
} /* trnred */

INT
xref(void)
{
  INT i1, i2;
  INT i, j, k, l, n, i0000;
  INT kend;
  INT link;
#define lsthed (scrtch)
  INT lstart;
  DECL_LINE(132);

  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    lsthed[i - 1] = 0;
  }
  i = g_1.numprd;

  do {
    j = g_1.prdind[i - 1];
    new (&link);
    s4.item[link - 1] = -i + 1;
    s4.nextt[link - 1] = lsthed[g_1.prodcn[j - 1] - 1];
    i0000 = g_1.prodcn[j - 1];
    lsthed[i0000 - 1] = link;
    kend = g_1.prodcn[j];
    if (kend > 0) {
      i1 = kend;
      for (k = 1; k <= i1; ++k) {
        new (&link);
        s4.item[link - 1] = i - 1;
        s4.nextt[link - 1] = lsthed[g_1.prodcn[j + k] - 1];
        i0000 = g_1.prodcn[j + k];
        lsthed[i0000 - 1] = link;
      }
    }
    --i;
  } while (i > 0);

  /*     print the cross-reference lists header */

  if (!switches.runosw) {
    wtpage(files.lstfil);
    a4tos1("    CROSS REFERENCE OF SYMBOLS", line, 30);
    wtline(files.lstfil, line, 30);
    /*     blank line */
    wtline(files.lstfil, line, 0);
  }
  a4tos1(".CE", line, 3);
  wtline(files.lstfil, line, 3);
  a4tos1(".bp", line, 3);
  wtline(files.lstfil, line, 3);
  a4tos1(".GS 2 \"CROSS REFERENCE OF SYMBOLS\"", line, 34);
  wtline(files.lstfil, line, 34);
  a4tos1(".CS", line, 3);
  wtline(files.lstfil, line, 3);
  /*     blank line */
  wtline(files.lstfil, line, 0);

  /*     print the cross-reference lists */

  for (j = 1; j <= 120; ++j) {
    line[j - 1] = ' ';
  }
  i1 = g_1.nvoc;
  for (i = 1; i <= i1; ++i) {
    l = 2;
    movstr(g_1.vocab[i - 1], line, &l, 120);
    l = (l + 4) / 5 * 5;
    lstart = l;
    if (lstart > 20) {
      lstart = 20;
    }
    n = lsthed[i - 1];

    while (n != 0) {
      if (l >= 67) {
        i2 = l - 1;
        output(files.lstfil, line, i2);
        l = lstart;
      }

      i2 = l + 4;
      condec(s4.item[n - 1], line, l, i2);
      l += 5;
      n = s4.nextt[n - 1];
    }

    i2 = l - 1;
    output(files.lstfil, line, i2);
    rel(&lsthed[i - 1]);
  }

  if (switches.runosw) {
    a4tos1(".CE", line, 3);
    wtline(files.lstfil, line, 3);
  }

  return 0;
} /* xref */

#undef lsthed

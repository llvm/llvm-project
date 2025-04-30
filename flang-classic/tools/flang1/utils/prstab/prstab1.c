/**
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief LR parser (part 1)
 *
 */

#include "lrutils.h"
#include "prstab.h"

INT xargc;
char **xargv;

/* Global Declarations */

struct files_s files;

INT *scrtch;
INT *hashpt;
CHAR *linech;
char *filnambuf;

struct s4_s s4;

struct s1_1_s s1_1;

struct lstcom_s lstcom;

struct qcom_s qcom;

struct switches_s switches;

struct adqcom_s adqcom;

struct g_1_s g_1;

struct s_1_s s_1;

struct string_1_s string_1;

struct readcm_s readcm;

struct ptsem_s ptsem;

int
main(INT argc, char **argv)
{
  xargc = argc;
  xargv = argv;

  /* perform initializations (open files) */
  init();
  /* read in grammar */
  rdgram();
  /* find goal symbol */
  fndgol();
  /* sort symbols and build translation table */
  sortgm();
  /* write ratfor macro definitions for tokens */
  if (switches.toksw) {
    tokmac();
  }
  /* check that all symbols connected to goal */
  conect();
  /* check that all symbols grounded */
  ground();
  /* write listing of grammar */
  if (switches.listsw) {
    prntgm();
  }
  /* write xreference portion of listing */
  if (switches.xrefsw) {
    xref();
  }
  /* write token names and non-terminal defines */
  zprntk();

  analyz();
  /* print sets / check for LR(1) */
  pntset();

  chncsl();
  /* output tables in lrltran */
  if (switches.lrltsw) {
    gentab();
  }
  /* output data stmts and defn */
  if (switches.datasw) {
    tablea();
  }
  /* for parse tables. */
  /* output the tables unformatted */
  if (switches.dbgdsw) {
    tableu();
  }
  /* close files and exit */
  finish();
  return 0;
} /* MAIN */

INT
addbas(INT iptr, INT npr, INT ndot, INT nset)
{
  if (s_1.indbas + 2 > MAXBAS) {
    error("basis area overflow", 19, 2, 0, 19);
  }
  ++s_1.basis[iptr];
  s_1.basis[s_1.indbas - 1] = npr;
  s_1.basis[s_1.indbas] = ndot;
  s_1.basis[s_1.indbas + 1] = nset;
  ++s4.item[nset - 1];
  s_1.indbas += 3;
  return 0;
} /* addbas */

INT
additl(INT *iarg, INT *lptr, INT *ichnge)
{
  INT i;
  INT last, nptr;

  /* adds the single item iarg into the list pointed to by lptr. */
  *ichnge = 0;
  last = 0;
  i = *lptr;
  while (i > 0) {
    if (*iarg == s4.item[i - 1]) {
      return 0;
    }
    if (*iarg < s4.item[i - 1]) {
      break;
    }

    last = i;
    i = s4.nextt[i - 1];
  }

  *ichnge = 1;
  new (&nptr);
  s4.item[nptr - 1] = *iarg;
  s4.nextt[nptr - 1] = i;
  if (last > 0) {
    s4.nextt[last - 1] = nptr;
  } else {
    *lptr = nptr;
  }

  return 0;
} /* additl */

INT
addltl(INT *lptr1, INT *lptr2, INT *ichnge)
{
  static INT i, ichn;

  /* adds the items in the list pointed to by lptr1 to the list pointed */
  /* to by lptr2. */
  *ichnge = 0;
  i = *lptr1;

  while (i > 0) {
    additl(&s4.item[i - 1], lptr2, &ichn);
    if (ichn != 0) {
      *ichnge = 1;
    }

    i = s4.nextt[i - 1];
  }

  return 0;
} /* addltl */

INT
addred(INT *ibasis, INT *iprod, INT *icntxt, INT *maxr)
{
  INT i__, j;

  i__ = *ibasis + s_1.basis[*ibasis] * 3 + 6;
  if (s_1.basis[i__ - 1] >= *maxr) {
    error("reduction array overflow", 24, 2, 0, 24);
  }
  j = (s_1.basis[i__ - 1] << 1) + s_1.basis[i__];
  s_1.red[j - 1] = *iprod;
  s_1.red[j] = *icntxt;
  ++s_1.basis[i__ - 1];
  ++s4.item[*icntxt - 1];
  return 0;
} /* addred */

INT
addtrn(INT *ibasis, INT *itran, INT *imax)
{
  INT i, i0000;

  i = *ibasis + s_1.basis[*ibasis] * 3 + 4;
  if (s_1.basis[i - 1] >= *imax) {
    error("transition array overflow", 25, 2, 0, 25);
  }
  i0000 = s_1.basis[i] + s_1.basis[i - 1];
  s_1.tran[i0000 - 1] = *itran;
  ++s_1.basis[i - 1];
  return 0;
} /* addtrn */

INT
analyz(void)
{
  INT i_1;
  INT i, n;
  INT jmax, nptr;

  fndnul();
  genthd();
  i_1 = g_1.nterms;
  for (i = 1; i <= i_1; ++i) {
    g_1.rgtuse[i - 1] = 0;
  }
  newbas(&i);
  new (&nptr);
  s4.item[nptr - 1] = g_1.prodcn[g_1.prdind[0] + 1];
  newcs(&nptr, &n);
  addbas(i, 1, 1, n);
  endbas(&i);

  do {
    complt(&i, &jmax);
    sortcg(&jmax);
    trnred(&i, &jmax);
    deque(&i);
  } while (i > 0);

  return 0;
} /* analyz */

INT
bildhp(void)
{
  INT i_1;
  INT i, j, iend;

  iend = g_1.nvoc / 2;
  i_1 = iend;
  for (i = 1; i <= i_1; ++i) {
    j = iend - i + 1;
    hepify(j, g_1.nvoc);
  }
  return 0;
} /* bildhp */

INT
chncsl(void)
{
  INT i_1;
  INT i, i0000;
  LOGICAL flag;
  INT last, iptr, lptr, nptr;

  /* function returning length of list */
  /* find the first context set chain. */
  s_1.lsets = 0;
  s_1.ncsets = 0;
  i_1 = g_1.nterms;
  flag = 0;
  for (i = 1; i <= i_1; ++i) {
    if (g_1.rgtuse[i - 1] != 0) {
      flag = 1;
      break;
    }
  }
  if (flag == 0) {
    error("chncsl error", 12, 2, 0, 12);
  }

  new (&nptr);
  s_1.lencsl = nptr;
  s_1.listcs = g_1.rgtuse[i - 1];
  iptr = g_1.rgtuse[i - 1];

  while (1) {
    ++s_1.ncsets;
    i0000 = s4.item[iptr - 1];
    s4.item[i0000 - 1] = s_1.ncsets;
    s4.item[nptr - 1] = lenlst(&s4.nextt[s4.item[iptr - 1] - 1]);
    s_1.lsets += s4.item[nptr - 1];
    lptr = nptr;
    new (&s4.nextt[nptr - 1]);
    nptr = s4.nextt[nptr - 1];
    last = iptr;
    iptr = s4.nextt[iptr - 1];
    if (iptr != 0) {
      continue;
    }

    do {
      if (i >= g_1.nterms) {
        s4.nextt[lptr - 1] = 0;
        rel(&nptr);
        return 0;
      }
      ++i;
    } while (g_1.rgtuse[i - 1] == 0);

    s4.nextt[last - 1] = g_1.rgtuse[i - 1];
    iptr = g_1.rgtuse[i - 1];
  } /* while */
} /* chncsl */

INT
chrcmp(INT *iptr1, INT *iptr2, INT *irslt)
{
  INT indx1, indx2, ichar1, ichar2;

  ichar1 = s1_1.sstore[*iptr1 - 1];
  ichar2 = s1_1.sstore[*iptr2 - 1];
  indx1 = ichar1 & 255;
  indx2 = ichar2 & 255;
  *irslt = 0;
  if (indx1 > indx2) {
    *irslt = 1;
  }
  if (indx1 < indx2) {
    *irslt = -1;
  }
  return 0;
} /* chrcmp */

INT
complt(INT *istate, INT *maxset)
{
  INT i_1;
  INT i, j, k, i0000, ich, lhs, iend, kend, mark[MAXPROD];
  INT iptr;
  INT ichang, istart, kstart;

  /* first move the basis to scrtch. */
  istart = *istate + 4;
  iend = istart + s_1.basis[*istate] * 3 - 1;
  if (iend - istart + 1 > 3000) {
    error("configuration set too large", 27, 2, 0, 27);
    return 0;
  }
  j = 0;
  i_1 = iend;
  for (i = istart; i <= i_1; i += 3) {
    scrtch[j] = s_1.basis[i - 1];
    scrtch[j + 1] = s_1.basis[i];
    scrtch[j + 2] = s_1.basis[i + 1];
    i0000 = scrtch[j + 2];
    ++s4.item[i0000 - 1];
    j += 3;
  }

  /* set all the productions unmarked. */

  i_1 = g_1.numprd;
  for (i = 1; i <= i_1; ++i) {
    mark[i - 1] = 0;
  }

  do {
    ichang = 0;
    i = 1;

    do {
      /* if the dot is at the end of the production then there are no */
      /* immediate transitions from the production. */

      if (scrtch[i] > g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1]]) {
        i += 3;
        if (i < j) {
          continue;
        } else {
          break;
        }
      }
      lhs = g_1.prodcn[g_1.prdind[scrtch[i - 1] - 1] + scrtch[i]];

      /* if the dot is before a terminal symbol then there is */
      /* no immediate transition from the production. */

      if (lhs <= g_1.nterms) {
        i += 3;
        if (i < j) {
          continue;
        } else {
          break;
        }
      }

      /* generate the context set which is the same for all */
      /* the immediate transitions from this production. */

      imtrcs(&i, &iptr);

      /* add all the un-marked productions with the left */
      /* hand side equal to the non-terminal to the right */
      /* of the dot. union in the new context set if the */
      /* production has already been included. */

      kstart = g_1.frsprd[lhs - 1];
      kend = kstart + g_1.nprods[lhs - 1] - 1;
      i_1 = kend;
      for (k = kstart; k <= i_1; ++k) {
        if (mark[k - 1] == 0) {
          if (j + 3 > 3000) {
            error("configuration set too large", 27, 2, 0, 27);
            return 0;
          }
          mark[k - 1] = j + 3;
          scrtch[j] = k;
          scrtch[j + 1] = 1;
          scrtch[j + 2] = iptr;
          ++s4.item[iptr - 1];
          j += 3;
          ichang = 1;
          continue;
        }
        csun(&iptr, &scrtch[mark[k - 1] - 1], &ich);
        if (ich != 0) {
          ichang = 1;
        }
      }

      /* the call to delete the context set deletes the */
      /* "extra" reference to the set and deletes the set */
      /* completely from the list space if if it was never */
      /* referenced. remember that as a result of the call */
      /* to imtrcs, iptr had its ref count incremented. */

      delcs(&iptr);
      i += 3;
    } while (i < j);
  } while (ichang != 0);

  *maxset = j;

  return 0;
} /* complt */

INT
condec(INT number, INT *line, INT istart, INT iend)
{
  INT digit[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
  INT i_1;
  INT i, j, m, n;

  /* Parameter adjustments */
  --line;

  n = number;
  if (n < 0) {
    n = -n;
  }
  i = iend;

  do {
    if (i < istart) {
      i_1 = iend;
      for (j = istart; j <= i_1; ++j) {
        line[j] = '*';
      }
      return 0;
    }
    m = n - n / 10 * 10;
    n /= 10;
    line[i] = digit[m];
    --i;
  } while (n != 0);

  if (number < 0) {
    if (i < istart) {
      i_1 = iend;
      for (j = istart; j <= i_1; ++j) {
        line[j] = '*';
      }
      return 0;
    }
    line[i] = '-';
    --i;
  }
  if (i < istart) {
    return 0;
  }
  i_1 = i;
  for (j = istart; j <= i_1; ++j) {
    line[j] = ' ';
  }

  return 0;
} /* condec */

#undef digit

INT
conect(void)
{
  INT i_1, i_2;

  INT i, j, i0000, iend, jend;
  INT isym, jsym, ibase;
#define stack (scrtch)
#define cnectd (scrtch + MAXSHDP1)
  INT istkpt;

  i_1 = g_1.nvoc;
  for (i = 1; i <= i_1; ++i) {
    cnectd[i - 1] = 0;
  }
  istkpt = 1;
  stack[0] = g_1.prodcn[0];
  i0000 = g_1.prodcn[0];
  cnectd[i0000 - 1] = 1;

  do {
    isym = stack[istkpt - 1];
    --istkpt;
    if (isym <= g_1.nterms) {
      continue;
    }
    ibase = g_1.prdind[g_1.frsprd[isym - 1] - 1] + 1;
    iend = g_1.nprods[isym - 1];
    i_1 = iend;
    for (i = 1; i <= i_1; ++i) {
      if (g_1.prodcn[ibase - 1] == 0) {
        ibase = ibase + g_1.prodcn[ibase - 1] + 2;
        continue;
      }
      jend = g_1.prodcn[ibase - 1];
      i_2 = jend;
      for (j = 1; j <= i_2; ++j) {
        jsym = g_1.prodcn[ibase + j - 1];
        if (cnectd[jsym - 1] != 0) {
          continue;
        }
        cnectd[jsym - 1] = 1;
        ++istkpt;
        stack[istkpt - 1] = jsym;
      }

      ibase = ibase + g_1.prodcn[ibase - 1] + 2;
    }
  } while (istkpt > 0);

  i_1 = g_1.nvoc;
  for (i = 1; i <= i_1; ++i) {
    if (cnectd[i - 1] == 0) {
      error("not connected to goal symbol", 28, 1, g_1.vocab[i - 1], (INT)28);
    }
  }
  return 0;
} /* conect */

#undef cnectd
#undef stack

INT
copyl(INT *lptr1, INT *lptr2)
{
  INT i;
  INT last, nptr;

  /* copies the list pointed to by lptr1 to a list pointed to by lptr2. */
  *lptr2 = 0;
  if (*lptr1 == 0) {
    return 0;
  }
  new (&nptr);
  s4.item[nptr - 1] = s4.item[*lptr1 - 1];
  last = nptr;
  *lptr2 = nptr;
  i = s4.nextt[*lptr1 - 1];
  while (i > 0) {
    new (&nptr);
    s4.item[nptr - 1] = s4.item[i - 1];
    s4.nextt[last - 1] = nptr;
    last = nptr;
    i = s4.nextt[i - 1];
  }

  return 0;
} /* copyl */

INT
csun(INT *iptr1, INT *iptr2, INT *ich)
{
  static INT head;

  *ich = 0;
  if (*iptr1 == *iptr2) {
    return 0;
  }
  copyl(&s4.nextt[*iptr2 - 1], &head);
  addltl(&s4.nextt[*iptr1 - 1], &head, ich);
  delcs(iptr2);
  newcs(&head, iptr2);

  return 0;
} /* csun */

INT
delcs(INT *iptr)
{
  static INT i, i0000;
  static INT last;

  if (s4.item[*iptr - 1] > 1) {
    --s4.item[*iptr - 1];
    return 0;
  }

  i = g_1.rgtuse[s4.item[s4.nextt[*iptr - 1] - 1] - 1];
  last = 0;

  while (1) {
    if (i == 0) {
      rel(iptr);
      return 0;
    }
    if (s4.item[i - 1] == *iptr) {
      break;
    }
    last = i;
    i = s4.nextt[i - 1];
  }

  if (last == 0) {
    i0000 = s4.nextt[*iptr - 1];
    i0000 = s4.item[i0000 - 1];
    g_1.rgtuse[i0000 - 1] = s4.nextt[i - 1];
  } else {
    s4.nextt[last - 1] = s4.nextt[i - 1];
  }

  s4.nextt[i - 1] = 0;
  rel(&i);
  rel(iptr);

  return 0;
} /* delcs */

INT
deque(INT *iptr)
{
  /* Local variables */
  *iptr = qcom.qhead;
  if (qcom.qhead != 0) {
    qcom.qhead = s_1.basis[qcom.qhead + 1];
    s_1.basis[*iptr + 1] = -1;
  }

  return 0;
} /* deque */

INT
endbas(INT *iptr)
{
  static INT i;

  s_1.indbas += 4;
  if (s_1.indbas > MAXBAS) {
    error("configuration set overflow", 26, 2, 0, 26);
  }
  for (i = 1; i <= 4; ++i) {
    s_1.basis[s_1.indbas + i - 6] = 0;
  }
  return 0;
} /* endbas */

INT
endred(INT *ibasis)
{
  INT i, j;

  i = *ibasis + s_1.basis[*ibasis] * 3 + 6;
  j = (s_1.basis[i - 1] << 1) + s_1.basis[i];
  if (j > s_1.nxtred) {
    s_1.nxtred = j;
  }
  return 0;
} /* endred */

INT
endtrn(INT *ibasis)
{
  INT i, j;

  i = *ibasis + s_1.basis[*ibasis] * 3 + 4;
  j = s_1.basis[i - 1] + s_1.basis[i];
  if (j > s_1.nxttrn) {
    s_1.nxttrn = j;
  }
  return 0;
} /* endtrn */

INT
enque(INT *iptr)
{
  if (s_1.basis[*iptr + 1] != -1) {
    return 0;
  }
  s_1.basis[*iptr + 1] = 0;
  if (qcom.qhead == 0) {
    qcom.qhead = *iptr;
    qcom.qtail = *iptr;
  }
  s_1.basis[qcom.qtail + 1] = *iptr;
  qcom.qtail = *iptr;

  return 0;
} /* enque */

INT
enter(INT *buff, INT *len)
{
  INT ret_val = -1;
  INT i, j, k, iptr, icomp;

  /* temporarily enter string at end of string storage: */
  /* Parameter adjustments */
  --buff;

  /* Function Body */
  if (switches.dbgesw) {
    wtline(files.dbgfil, &buff[1], *len);
  }
  /* pointer into string area */
  iptr = string_1.sstptr + 1;
  /* increment avail pointer */
  string_1.sstptr += *len;
  if (string_1.sstptr > MAXSST) {
    error("string storage overflow", 24, 12, 0, 23);
  }
  s1toa1(&buff[1], &s1_1.sstore[iptr - 1], len);
  /* symbol pointer */
  ++string_1.shdptr;
  if (string_1.shdptr >= MAXSHD) {
    error("too many literal strings", 24, 12, 0, 24);
  }
  s1_1.sthead[string_1.shdptr - 1] = iptr;
  /* needed to compute length */
  s1_1.sthead[string_1.shdptr] = iptr + *len;
  /* use hashing algorithm to determine if symbol already exists: */
  i = hashof(&string_1.shdptr) + 1;
  if (hashpt[i - 1] != 0) {
    for (j = 1; j <= MAXHASH; ++j) {
      if (i + j - 1 > MAXHASH) {
        i += -MAXHASH;
      }
      k = i + j - 1;
      if (hashpt[k - 1] == 0) {
        hashpt[k - 1] = string_1.shdptr;
        g_1.vocab[string_1.shdptr - 1] = string_1.shdptr;
        ret_val = string_1.shdptr;
        return ret_val;
      }

      strcomp(&string_1.shdptr, &hashpt[k - 1], &icomp);
      if (icomp == 0) {
        --string_1.shdptr;
        /* restore sstptr */
        string_1.sstptr = iptr - 1;
        ret_val = hashpt[k - 1];
        return ret_val;
      }
    }

    /* hash table overflow. */
    error("hash table overflow", 19, 12, 0, 19);
    return ret_val;
  }

  hashpt[i - 1] = string_1.shdptr;
  g_1.vocab[string_1.shdptr - 1] = string_1.shdptr;
  ret_val = string_1.shdptr;
  return ret_val;
} /* enter */

void
intArrayToCharArray(INT *intarr, CHAR *charr, INT count)
{
  INT i;
  for (i = 0; i < count; i++) {
    charr[i] = intarr[i];
  }
  charr[i] = '\0';
}

INT
error(const char *msg, INT msglen, INT ibad, INT sym, INT alias)
{
  INT i_1;
  INT jbad, iend, ibase;
  INT istrt;
  DECL_LINE(132);

  jbad = ibad;
  if (jbad > 9) {
    jbad += -10;
  }
  a4tos1(" ** LR- ", line, 8);
  ibase = 8;
  if (jbad != 1) {
    if (jbad != 0) {
      a4tos1("fatal: ", &line[ibase], 7);
      ibase += 7;
    }
  } else {
    a4tos1("warning: ", &line[ibase], 9);
    ibase += 9;
  }

  /* move symbol name, if present, into message: */
  if (sym != 0) {
    ++ibase;
    istrt = ibase;
    movstr(sym, line, &ibase, 30);
    i_1 = ibase - istrt;
    a1tos1(&line[istrt - 1], &line[istrt - 1], i_1);
    line[ibase - 1] = 32;
  }

  /* move message text into line: */
  iend = min(60, msglen);
  a4tos1(msg, &line[ibase], iend);
  ibase = ibase + iend + 1;
  line[ibase - 1] = 32;
  if (ibad >= 10) {
    /* insert the line number and character position: */
    a4tos1("at line      char ", &line[ibase], 18);
    i32tos(&readcm.lineno, &line[ibase + 8], 4, 0, 10, 1);
    i32tos(&readcm.fstchr, &line[ibase + 18], 3, 0, 10, 1);
    ibase += 21;
  }

  intArrayToCharArray(line, linech, ibase);
  printf("%s\n", linech);

  if (switches.dbgsw) {
    wtline(files.dbgfil, line, ibase);
  }
  if (switches.listsw) {
    wtline(files.lstfil, line, ibase);
  }
  /* if terminal error, stop here: */
  if (jbad >= 2) {
    finish();
  }
  return 0;
} /* error */

INT
finish(void)
{
  free(scrtch);
  free(hashpt);
  free(s4.item);
  free(s4.nextt);
  free(s1_1.sstore);
  free(s1_1.sthead);
  free(g_1.lftuse);
  free(g_1.rgtuse);
  free(g_1.frsprd);
  free(g_1.nprods);
  free(g_1.prodcn);
  free(g_1.prdind);
  free(g_1.vocab);
  free(s_1.thedpt);
  free(s_1.nullnt);
  free(s_1.basis);
  free(s_1.tran);
  free(s_1.red);

#define FILES_CLOSE(f) \
  if (files.f)         \
  (void) fclose(files.f)
  FILES_CLOSE(gramin);
  FILES_CLOSE(lstfil);
  FILES_CLOSE(dbgfil);
  FILES_CLOSE(datfil);
  FILES_CLOSE(tokin);
  FILES_CLOSE(tokout);
  FILES_CLOSE(semfil);

  exitf(0);
  return 0;
} /* finish */

INT
fndgol(void)
{
  INT i_1;
  INT i;

  g_1.goal = 0;
  i_1 = g_1.nvoc;
  for (i = 3; i <= i_1; ++i) {
    if (g_1.lftuse[i - 1] == 0) {
      continue;
    }
    if (g_1.rgtuse[i - 1] == 1) {
      continue;
    }
    if (g_1.goal != 0) {
      error("is extra goal symbol", 20, 1, i, 12);
    } else {
      g_1.goal = i;
    }
  }

  if (g_1.goal == 0) {
    g_1.goal = g_1.prodcn[g_1.prdind[1] - 1];
  }
  g_1.prodcn[3] = g_1.goal;
  return 0;
} /* fndgol */

INT
fndnul(void)
{
  static struct {
    char e_1[128];
    INT e_2;
  } equiv_128 = {"1       p   o   t   e   n   t   i   a   l   l   y  "
                 "     n   u   l   l       n   o   n   -   t   e   r   m   i  "
                 " n   a   l   s   ",
                 0};

#define hdg ((INT *)&equiv_128)

  INT i_1, i_2, i_3;

  INT i, j, k, iend, jend, flag = 0;
  INT ichnge;
  INT istart, kstart, jstart;
  DECL_LINE(132);

  /* mark nullnt(i) if vocabulary(i) is potentially null. this really */
  /* applys just to non-terminals so the terminals will never be marked. */
  /* for each non-terminal not already marked check each of its right */
  /* hand sides to be null or composed of non-terminals that */
  /* are all marked. if so, mark the non-terminal, indicate a change and */
  /* check the next. continue this while there is a change. */

  /* Fix from Al Shannon: zero out nullnt */
  /* J.R.L. - 8 dec 78 */

  for (i = 1; i <= MAXSHD; ++i) {
    s_1.nullnt[i - 1] = 0;
  }

  do {
    ichnge = 0;
    kstart = g_1.nterms + 1;
    i_1 = g_1.nvoc;
    for (k = kstart; k <= i_1; ++k) {
      if (s_1.nullnt[k - 1] == 1) {
        continue;
      }
      istart = g_1.frsprd[k - 1];
      iend = istart + g_1.nprods[k - 1] - 1;
      i_2 = iend;
      for (i = istart; i <= i_2; ++i) {
        if (g_1.prodcn[g_1.prdind[i - 1]] == 0) {
          ichnge = 1;
          s_1.nullnt[k - 1] = 1;
          break;
        }
        jstart = g_1.prdind[i - 1] + 2;
        jend = g_1.prdind[i - 1] + g_1.prodcn[g_1.prdind[i - 1]] + 1;
        i_3 = jend;
        for (j = jstart; j <= i_3; ++j) {
          if (s_1.nullnt[g_1.prodcn[j - 1] - 1] == 0) {
            flag = 1;
            break;
          }
        }

        if (!flag) {
          ichnge = 1;
          s_1.nullnt[k - 1] = 1;
          break;
        } else { /* restore flag */
          flag = 0;
        }
      } /* i */
    }   /* k */
  } while (ichnge != 0);

  if (!switches.dbgcsw) {
    return 0;
  }
  for (i = 1; i <= 32; ++i) {
    line[i - 1] = hdg[i - 1];
  }
  output(files.dbgfil, line, 32);
  i = 4;
  i_1 = g_1.nvoc;
  for (k = kstart; k <= i_1; ++k) {
    if (s_1.nullnt[k - 1] == 0) {
      continue;
    }
    if (length(&g_1.vocab[k - 1]) + i > 120) {
      i_2 = i - 1;
      output(files.dbgfil, line, i_2);
      i = 4;
    }

    movstr(g_1.vocab[k - 1], line, &i, 120);
    ++i;
  }
  i_1 = i - 1;
  output(files.dbgfil, line, i_1);

  return 0;
} /* fndnul */

#undef hdg

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran module responsible for generating Cross Reference
           Listing.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"

typedef short int LINENOTYPE;
typedef short int USETYPE;

struct memitem {
  LINENOTYPE lineno;
  USETYPE use;
  INT link;
};

static FILE *fd;
static int xrefcnt;
static struct memitem *baseptr;

/* ---------------------------------------------------------------------- */

void
xrefinit(void)
{
  xrefcnt = 0;
  baseptr = NULL;

  /* create temporary file and open it for writing */
  if ((fd = tmpfile()) == NULL)
    errfatal(5);
}

/* ---------------------------------------------------------------------- */

/** \brief Write one reference record to Reference File:  */
void
xrefput(int symptr, int usage)
{
  struct memitem item;

  xrefcnt++;
  item.lineno = gbl.lineno;
  item.use = usage;
  item.link = symptr;

  if ((fwrite((char *)(&item), sizeof(struct memitem), 1, fd)) == 0)
    interr("xrefput fwrite failed", xrefcnt, 4);
}

/* ---------------------------------------------------------------------- */

/* write two lines of info for symbol elem to listing. */
static void
putentry(int elem)
{
  char buf[200];
  const char *ptr;
  const char *sc_p;
  int stype, dtype;
  char hyp;
  static const char *scs[] = {"n/a",    "local",  "static", "dummy",
                              "common", "extern", "based",  "seven"};

  stype = STYPEG(elem);

  ptr = SYMNAME(elem);
  if (stype == ST_LABEL)
    ptr += 2; /* delete ".L" from front of label name */

  dtype = 0;
  if (stype != ST_LABEL && stype != ST_CMBLK) {
    dtype = DTYPEG(elem);
  }

  sprintf(buf, "%-16.16s ", ptr);
  getdtype(dtype, &buf[17]);
  strcat(buf, " ");
  strcat(buf, stb.stypes[stype]);
  list_line(buf);

  if ((int)strlen(ptr) > 16) {
    ptr += 16;
    hyp = '-';
  } else {
    ptr = "";
    hyp = ' ';
  }

  switch (stype) {
  case ST_LABEL:
    sprintf(buf, "%c%-15.15s   addr: %" ISZ_PF "x", hyp, ptr, ADDRESSG(elem));
    break;
  case ST_PARAM:
    /* Changed to display ALL PARAMETER data types LDE */
    /* if (dtype <= DT_INT)
        sprintf(buf, "%c%-15.15s  value: %d", hyp, ptr, CONVAL1G(elem));
    else
        sprintf(buf, "%c%-15.15s", hyp, ptr);
    */
    sprintf(buf, "%c%-15.15s  value: %s", hyp, ptr, parmprint(elem));
    break;
  case ST_ENTRY:
    sprintf(buf, "%c%-15.15s   addr: %" ISZ_PF "x", hyp, ptr, ADDRESSG(elem));
    break;
  case ST_CMBLK:
    sprintf(buf, "%c%-15.15s   size: %" ISZ_PF "d bytes", hyp, ptr,
            SIZEG(elem));
    break;

  case ST_VAR:
  case ST_IDENT:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
    if (SCG(elem) == SC_CMBLK)
      sc_p = SYMNAME(CMBLKG(elem));
    else
      sc_p = scs[SCG(elem)];
    sprintf(buf, "%c%-15.15s  sc: %s   addr: %" ISZ_PF "x", hyp, ptr, sc_p,
            ADDRESSG(elem));
    break;

  case ST_PROC:
  case ST_STFUNC:
  case ST_USERGENERIC:
  case ST_GENERIC:
  case ST_INTRIN:
  default:
    if (hyp == '-')
      sprintf(buf, "%c%-15.15s", hyp, ptr);
    else
      hyp = 0;
  }

  if (hyp != 0)
    list_line(buf);

}

/* ---------------------------------------------------------------------- */

static void
printxref(int liststart)
{
  int sptr;
  INT refptr;
  int i;
  char buf[200];

  /* print header */
  list_line("Cross Reference Listing:");
  list_line("");

  /* for each ref'd element of symbol table with a name */
  for (sptr = liststart; sptr != 0; sptr = HASHLKG(sptr)) {
    assert((NMPTRG(sptr) != 0), "printxref bad record name", sptr, 3);
    assert((XREFLKG(sptr) != 0), "printxref bad record link", sptr, 3);

    /* print stuff about symbol table entry */
    putentry(sptr);

    refptr = XREFLKG(sptr);

#define ENDOFLINE 80
    /* put references info */
    i = 999; /* i is char position in line being printed */
    do {
      if (i > (ENDOFLINE - 5)) {
        if (i != 999)
          list_line(buf);
        sprintf(buf, "                ");
        i = 16;
      }
      sprintf(&buf[i], "%5d%c", (baseptr + refptr)->lineno,
              (baseptr + refptr)->use);
      refptr = (baseptr + refptr)->link;
      i += 6;
    } while (refptr != 0);
    list_line(buf);
  }

  /* print trailer */
  list_line("---------------------------------");
  list_page();
}

/* ---------------------------------------------------------------------- */

static int
insertsrt(int q, int r)
{
  int i, last, next;
  char *ptr;

  HASHLKP(0, 0);

  for (i = q; i <= r; i++) {
    /* skip over those records without ref lists */
    if (XREFLKG(i) == 0)
      continue;
    ptr = SYMNAME(i);
    last = 0;
    next = HASHLKG(0);
    while ((next != 0) && (strcmp(ptr, SYMNAME(next)) >= 0)) {
      last = next;
      next = HASHLKG(next);
    }
    HASHLKP(i, next);
    HASHLKP(last, i);
  }
  return (HASHLKG(0));
}

/* ---------------------------------------------------------------------- */

static int
merge(int q, int r)
{
  int i, j, k;

  k = 0;
  i = q;
  j = r;
  while ((i != 0) && (j != 0)) {
    if (strcmp(SYMNAME(i), SYMNAME(j)) <= 0) {
      HASHLKP(k, i);
      k = i;
      i = HASHLKG(i);
    } else {
      HASHLKP(k, j);
      k = j;
      j = HASHLKG(j);
    }
  }
  if (i == 0)
    HASHLKP(k, j);
  else
    HASHLKP(k, i);

  return (HASHLKG(0));
}

/* ---------------------------------------------------------------------- */

static int
mergesrt(int low, int high)
{
  int mid;
  int p, q, r;
#define SORTLIM 16

  if ((high - low) < SORTLIM)
    p = insertsrt(low, high);
  else {
    mid = (low + high) / 2;
    q = mergesrt(low, mid);
    r = mergesrt(mid + 1, high);
    p = merge(q, r);
  }
  return (p);
}

/* ---------------------------------------------------------------------- */

void
xref(void)
{
  struct memitem *iptr;
  int sptr;
  INT index;
  int i;
  int liststart, nsyms;

  if (!xrefcnt)
    return;

  /* rewind file */
  if (fseek(fd, 0L, 0) == EOF)
    interr("xref's fseek failed", 0, 4);

  /* allocate space for ref lists */
  NEW(baseptr, struct memitem, xrefcnt + 1);
  if (baseptr == NULL)
    interr("xref couldn't alloc enough space", 0, 4);

  /* fill symbol table's hash field with NULLs (sort links)    */
  /* fill w7 field in symbol table with NULLs  (ref list head) */
  nsyms = stb.stg_avail;
  for (i = 1; i <= nsyms; i++) {
    HASHLKP(i, 0);
    XREFLKP(i, 0);
  }

  /* read ref info into memory - 0 pos is wasted */
  if ((fread((char *)(baseptr + 1), sizeof(struct memitem), xrefcnt, fd)) == 0)
    interr("xref fread failed", xrefcnt, 4);

  /* process in reverse order so lists are in correct order */
  iptr = baseptr + xrefcnt;
  index = xrefcnt;
  while (iptr > baseptr) {
    sptr = iptr->link;
    iptr->link = XREFLKG(sptr);
    XREFLKP(sptr, index--);
    iptr--;
  }

  /* sort symbol table on SYMNAMES  */
  liststart = mergesrt(1, nsyms);

  printxref(liststart);
  fclose(fd);

}

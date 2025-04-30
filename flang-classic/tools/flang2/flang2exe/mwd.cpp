/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief mw's dump routines
 */

#include "mwd.h"
#include "error.h"
#include "machar.h"
#include "global.h"
#include "symtab.h"
#include "ilm.h"
#include "fih.h"
#include "ili.h"
#include "iliutil.h"
#include "dtypeutl.h"
#include "machreg.h"
#ifdef SOCPTRG
#include "soc.h"
#endif
#include "llutil.h"
#include "symfun.h"

static int putdtypex(DTYPE dtype, int len);
static void _printnme(int n);
static bool g_dout = true;

#ifdef _WIN64
#define vsnprintf _vsnprintf
#endif

#if DEBUG

static FILE *dfile;
static int linelen = 0;
#define BUFSIZE 10000
static char BUF[BUFSIZE];
static int longlines = 1, tight = 0, nexttight = 0;

/* for debug purpuse: test if the current
 * function is the one that func specifies */
int 
testcurrfunc(const char* func)
{
  if(strcmp(SYMNAME(GBL_CURRFUNC), func)==0)
    return true;
  else
    return false;
}

/*
 * 'full' is zero for a 'diff' dump, so things like symbol numbers,
 * ili numbers, etc., are left off; this makes ili trees and symbol dumps
 * that are for all intents and purposes the same look more identical.
 * 'full' is 2 for an 'important things' only dump; nmptr, hashlk left off
 * 'full' is 1 for full dump, everything
 */
static int full = 1;

void
dumplong(void)
{
  longlines = 1;
} /* dumplong */

void
dumpshort(void)
{
  longlines = 0;
} /* dumpshort */

void
dumpdiff(void)
{
  full = 0;
} /* dumpdiff */

void
dumpddiff(int v)
{
  full = v;
} /* dumpddiff */

static void
putit(void)
{
  int l = strlen(BUF);
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (linelen + l >= 78 && !longlines) {
    fprintf(dfile, "\n%s", BUF);
    linelen = l;
  } else if (linelen > 0 && nexttight) {
    fprintf(dfile, "%s", BUF);
    linelen += l + 1;
  } else if (linelen > 0 && tight) {
    fprintf(dfile, " %s", BUF);
    linelen += l + 1;
  } else if (linelen > 0) {
    fprintf(dfile, "  %s", BUF);
    linelen += l + 2;
  } else {
    fprintf(dfile, "%s", BUF);
    linelen = l;
  }
  nexttight = 0;
} /* putit */

static void
puttight(void)
{
  int l = strlen(BUF);
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "%s", BUF);
  linelen += l;
} /* puttight */

static void
putline(void)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (linelen)
    fprintf(dfile, "\n");
  linelen = 0;
} /* putline */

void
putmwline()
{
  putline();
} /* putmwline */

#include <stdarg.h>

static void
puterr(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  putline();
  strcpy(BUF, "*** ");
  vsnprintf(BUF + 4, BUFSIZE - 4, fmt, argptr);
  strcat(BUF, " ***");
  putit();
  putline();
} /* puterr */

static void
appendit(void)
{
  int l = strlen(BUF);
  if (g_dout)
    fprintf(dfile, "%s", BUF);
  linelen += l;
  nexttight = 0;
} /* appendit */

static void
putint(const char *s, int d)
{
  if (g_dout) {
    snprintf(BUF, BUFSIZE, "%s:%d", s, d);
    putit();
  }
} /* putint */

static void
putdouble(const char *s, double d)
{
  if (g_dout) {
    snprintf(BUF, BUFSIZE, "%s:%lg", s, d);
    putit();
  }
} /* putdouble */

static void
putbigint(const char *s, ISZ_T d)
{
  snprintf(BUF, BUFSIZE, "%s:%" ISZ_PF "d", s, d);
  putit();
} /* putbigint */

static void
putINT(const char *s, INT d)
{
  snprintf(BUF, BUFSIZE, "%s:%ld", s, (long)d);
  putit();
} /* putINT */

static void
putintarray(const char *s, int *x, int size)
{
  int i;
  if (x != NULL) {
    for (i = 0; i < size; ++i) {
      if (x[i] != 0) {
        snprintf(BUF, BUFSIZE, "%s[%d]:%8d %8x", s, i, x[i], x[i]);
        x[i] = 0;
        putit();
        putline();
      }
    }
  }
} /* putintarray */

static void
putnzint(const char *s, int d)
{
  if (d != 0) {
    snprintf(BUF, BUFSIZE, "%s:%d", s, d);
    putit();
  }
} /* putnzint */

static void
putnzbigint(const char *s, ISZ_T d)
{
  if (d != 0) {
    snprintf(BUF, BUFSIZE, "%s:%" ISZ_PF "d", s, d);
    putit();
  }
} /* putnzbigint */

static void
putnzINT(const char *s, INT d)
{
  if (d != 0) {
    snprintf(BUF, BUFSIZE, "%s:%ld", s, (long)d);
    putit();
  }
} /* putnzINT */

static void
putnzisz(const char *s, ISZ_T d)
{
  if (d != 0) {
    snprintf(BUF, BUFSIZE, "%s:%" ISZ_PF "d", s, d);
    putit();
  }
} /* putnzint */

static void
put2int(const char *s, int d1, int d2)
{
  snprintf(BUF, BUFSIZE, "%s(%d:%d)", s, d1, d2);
  putit();
} /* put2int */

static void
putpint(const char *s, int d)
{
  put2int(s, (int)(d & 0xff), (int)(d >> 8));
} /* putpint */

void
putint1(int d)
{
  snprintf(BUF, BUFSIZE, "%d", d);
  putit();
} /* putint1 */

static int
appendint1(int d)
{
  int r;
  if (!g_dout) {
    r = snprintf(BUF, BUFSIZE, "%d", d);
    sprintf(BUF, "%s%d", BUF, d);
    r = 0;
  } else
    r = sprintf(BUF, "%d", d);

  appendit();
  return r;
} /* appendint1 */

static int
appendbigint(ISZ_T d)
{
  int r;
  if (!g_dout) {
    r = snprintf(BUF, BUFSIZE, "%" ISZ_PF "d", d);
    sprintf(BUF, ("%s%" ISZ_PF "d"), BUF, d);
    r = 0;
  } else
    r = sprintf(BUF, ("%" ISZ_PF "d"), d);

  appendit();
  return r;
} /* appendbigint */

static void
appendhex1(int d)
{
  snprintf(BUF, BUFSIZE, "0x%x", d);
  appendit();
} /* appendhex1 */

static void
putbit(const char *s, int b)
{
  /* single space between flags */
  if (b) {
    int l = strlen(s);
    if (linelen + l >= 79 && !longlines) {
      fprintf(dfile, "\n%s", s);
      linelen = l;
    } else if (linelen > 0) {
      fprintf(dfile, " %s", s);
      linelen += l + 1;
    } else {
      fprintf(dfile, "%s", s);
      linelen = l;
    }
  }
} /* putbit */

static void
putstring(const char *s, const char *t)
{
  snprintf(BUF, BUFSIZE, "%s:%s", s, t);
  putit();
} /* putstring */

static void
putnstring(const char *s, const char *t)
{
  if (t != NULL) {
    snprintf(BUF, BUFSIZE, "%s:%s", s, t);
    putit();
  }
} /* putstring */

static void
putstringarray(const char *s, char **arr)
{
  int i;
  if (arr != NULL) {
    for (i = 0; arr[i] != NULL; ++i) {
      snprintf(BUF, BUFSIZE, "%s[%d]:%s", s, i, arr[i]);
      putit();
    }
  }
} /* putstringarray */

static void
putdefarray(const char *s, char **arr)
{
  int i;
  if (arr != NULL) {
    for (i = 0; arr[i] != NULL; i += 2) {
      if (arr[i + 1] == (const char *)1) {
        snprintf(BUF, BUFSIZE, "%s[%d]  pred:%s", s, i, arr[i]);
      } else if (arr[i + 1] == (const char *)0) {
        snprintf(BUF, BUFSIZE, "%s[%d] undef:%s", s, i, arr[i]);
      } else {
        snprintf(BUF, BUFSIZE, "%s[%d]   def:%s", s, i, arr[i]);
      }
      putit();
      putline();
    }
  }
} /* putdefarray */

static void
putstring1(const char *t)
{
  snprintf(BUF, BUFSIZE, "%s", t);
  putit();
} /* putstring1 */

static void
putstring1t(const char *t)
{
  snprintf(BUF, BUFSIZE, "%s", t);
  puttight();
  nexttight = 0;
} /* putstring1t */

static int
appendstring1(const char *t)
{
  int r;
  if (!g_dout) {
    strcat(BUF, t);
    r = 0;
  } else {
    r = snprintf(BUF, BUFSIZE, "%s", t);
  }
  appendit();
  return r;
} /* appendstring1 */

static void
putsym(const char *s, SPTR sptr)
{
  if (full) {
    if (sptr == NOSYM) {
      snprintf(BUF, BUFSIZE, "%s:%d=%s", s, sptr, "NOSYM");
    } else if (sptr > 0 && sptr < stb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%s:%d=%s%s", s, sptr, printname(sptr),
               ADDRTKNG(sptr) ? " (&)" : "");
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, sptr);
    }
  } else {
    if (sptr == NOSYM) {
      snprintf(BUF, BUFSIZE, "%s:%s", s, "NOSYM");
    } else if (sptr > 0 && sptr < stb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%s:%s%s", s, printname(sptr),
               ADDRTKNG(sptr) ? " (&)" : "");
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, sptr);
    }
  }

  putit();
} /* putsym */

static void
putnsym(const char *s, SPTR sptr)
{
  if (sptr != 0)
    putsym(s, sptr);
} /* putnsym */

static void
putsym1(int sptr)
{
  if (full) {
    if (sptr == NOSYM) {
      snprintf(BUF, BUFSIZE, "%d=%s", sptr, "NOSYM");
    } else if (sptr > 0 && sptr < stb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%d=%s", sptr, printname(sptr));
    } else {
      snprintf(BUF, BUFSIZE, "%d", sptr);
    }
  } else {
    if (sptr == NOSYM) {
      snprintf(BUF, BUFSIZE, "%s", "NOSYM");
    } else if (sptr > 0 && sptr < stb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%s", printname(sptr));
    } else {
      snprintf(BUF, BUFSIZE, "%d", sptr);
    }
  }
  putit();
} /* putsym1 */

static int
appendsym1(int sptr)
{
  int r;
  if (sptr == NOSYM) {
    r = snprintf(BUF, BUFSIZE, "%s", "NOSYM");
  } else if (sptr > 0 && sptr < stb.stg_avail) {
    r = snprintf(BUF, BUFSIZE, "%s", printname(sptr));
  } else {
    r = snprintf(BUF, BUFSIZE, "sym%d", sptr);
  }
  appendit();
  return r;
} /* appendsym1 */

static void
putsc(const char *s, int sc)
{
  if (full) {
    if (sc >= 0 && sc <= SC_MAX) {
      snprintf(BUF, BUFSIZE, "%s:%d=%s", s, sc, stb.scnames[sc] + 3);
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, sc);
    }
  } else {
    if (sc >= 0 && sc <= SC_MAX) {
      snprintf(BUF, BUFSIZE, "%s:%s", s, stb.scnames[sc] + 3);
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, sc);
    }
  }
  putit();
} /* putsc */

static void
putnsc(const char *s, int sc)
{
  if (sc != 0)
    putsc(s, sc);
} /* putnsc */

static void
putstype(const char *s, int stype)
{
  if (full) {
    if (stype >= 0 && stype <= ST_MAX) {
      snprintf(BUF, BUFSIZE, "%s:%d=%s", s, stype, stb.stypes[stype]);
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, stype);
    }
  } else {
    if (stype >= 0 && stype <= ST_MAX) {
      snprintf(BUF, BUFSIZE, "%s:%s", s, stb.stypes[stype]);
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d", s, stype);
    }
  }
  putit();
} /* putstype */

static void
putddtype(const char *s, DTYPE d)
{
  if (d) {
    snprintf(BUF, BUFSIZE, "%s:%d=", s, d);
    putit();
    putdtypex(d, 80);
  }
} /* putddtype */

static void
putnname(const char *s, int off)
{
  if (off) {
    putstring(s, stb.n_base + off);
  }
} /* putnname */

static void
putsymlk(const char *name, int list)
{
  int c = 0;
  if (list <= NOSYM)
    return;
  putline();
  putstring1(name);
  for (; list > NOSYM && c < 200; list = SYMLKG(list), ++c)
    putsym1(list);
  putline();
} /* putsymlk */

void
putnme(const char *s, int nme)
{
  if (full) {
    if (nme < 0 || nme >= nmeb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%s:%d=%s", s, nme, "NONE");
      putit();
    } else {
      snprintf(BUF, BUFSIZE, "%s:%d=", s, nme);
      putit();
      _printnme(nme);
    }
  } else {
    if (nme < 0 || nme >= nmeb.stg_avail) {
      snprintf(BUF, BUFSIZE, "%s:%d", s, nme);
      putit();
    } else {
      snprintf(BUF, BUFSIZE, "%s:", s);
      putit();
      _printnme(nme);
    }
  }
} /* putnme */

#ifdef DPDSCG
static void
putparam(int dpdsc, int paramct)
{
  if (paramct == 0)
    return;
  putline();
  putstring1("Parameters:");
  for (; paramct; ++dpdsc, --paramct) {
    int sptr = aux.dpdsc_base[dpdsc];
    if (sptr == 0) {
      putstring1("*");
    } else {
      putsym1(sptr);
    }
  }
} /* putparam */
#endif /* DPDSCG */

#define SIZEOF(array) (sizeof(array) / sizeof(char *))

static void
putval1(int val, const char *values[], int sizeofvalues)
{
  if (val < 0 || val >= sizeofvalues) {
    snprintf(BUF, BUFSIZE, "%d", val);
    putit();
  } else {
    putstring1(values[val]);
  }
} /* putval1 */

#ifdef SOCPTRG
void
putsoc(int socptr)
{
  int p, q;

  if (socptr == 0)
    return;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (soc.base == NULL) {
    fprintf(dfile, "soc.base not allocated\n");
    return;
  }
  q = 0;
  putline();
  putstring1("Storage Overlap Chain:");

  for (p = socptr; p; p = SOC_NEXT(p)) {
    putsym1(SOC_SPTR(p));
    if (q == p) {
      putstring1(" >>> soc loop");
      break;
    }
    q = p;
  }
} /* putsoc */
#endif /* SOCPTRG */

#ifdef CUDAG
static void
putcuda(const char *s, int cu)
{
  if (cu) {
    strcpy(BUF, s);
    strcat(BUF, ":");
    if (cu & CUDA_HOST) {
      strcat(BUF, "host");
      cu &= ~CUDA_HOST;
      if (cu)
        strcat(BUF, "+");
    }
    if (cu & CUDA_DEVICE) {
      strcat(BUF, "device");
      cu &= ~CUDA_DEVICE;
      if (cu)
        strcat(BUF, "+");
    }
    if (cu & CUDA_GLOBAL) {
      strcat(BUF, "global");
      cu &= ~CUDA_GLOBAL;
      if (cu)
        strcat(BUF, "+");
    }
    if (cu & CUDA_BUILTIN) {
      strcat(BUF, "builtin");
      cu &= ~CUDA_BUILTIN;
      if (cu)
        strcat(BUF, "+");
    }
    if (cu & CUDA_CONSTRUCTOR) {
      strcat(BUF, "constructor");
      cu &= ~CUDA_CONSTRUCTOR;
      if (cu)
        strcat(BUF, "+");
    }
#ifdef CUDA_STUB
    if (cu & CUDA_STUB) {
      strcat(BUF, "stub");
      cu &= ~CUDA_STUB;
      if (cu)
        strcat(BUF, "+");
    }
#endif
    putit();
  }
} /* putcuda */
#endif

static void
check(const char *s, int v)
{
  if (v) {
    fprintf(dfile, "*** %s: %d %x\n", s, v, v);
  }
} /* check */

/* dump one symbol to gbl.dbgfil */
void
dsym(int sptr)
{
  SYMTYPE stype;
  DTYPE dtype;
  const char *np;
#ifdef SOCPTRG
  int socptr;
#endif
  int i;

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;

  if (sptr == 0 || sptr >= stb.stg_avail) {
    fprintf(dfile, "\nsymbol %d out of %d\n", sptr, stb.stg_avail - 1);
    return;
  }

  BCOPY(stb.stg_base, stb.stg_base + sptr, SYM, 1);

  stype = STYPEG(0);
  switch (stype) {
  case ST_BLOCK:
  case ST_CMBLK:
  case ST_LABEL:
  case ST_BASE:
  case ST_UNKNOWN:
    dtype = DT_NONE;
    break;
  default:
    dtype = DTYPEG(0);
    break;
  }
  np = printname(sptr);
  if (strlen(np) >= 30) {
    fprintf(dfile, "\n%s", np);
    np = " ";
  }
  fprintf(dfile, "\n%-30.30s ", np);
  if (dtype) {
    putdtypex(dtype, 50);
  }
  fprintf(dfile, "\n");

  linelen = 0;

  if (full) {
    putint("sptr", sptr);
    putnzint("dtype", DTYPEG(0));
  }
  if (full & 1)
    putnzint("nmptr", NMPTRG(0));
  putnsc("sc", SCG(0));
  putstype("stype", STYPEG(0));
  putline();
  STYPEP(0, ST_UNKNOWN);
  DTYPEP(0, DT_NONE);
  NMPTRP(0, 0);
  SCP(0, SC_NONE);

  if (full & 1)
    putnsym("hashlk", HASHLKG(0));
  putnsym("scope", SCOPEG(SPTR_NULL));
  putnsym("symlk", SYMLKG(0));
  putline();
  HASHLKP(0, SPTR_NULL);
  SCOPEP(0, 0);
  SYMLKP(0, SPTR_NULL);

  switch (stype) {
  default:
    break;
  case ST_ARRAY:
  case ST_IDENT:
  case ST_STRUCT:
  case ST_UNION:
  case ST_UNKNOWN:
  case ST_VAR:
    /* three lines: integers, symbols, bits */
    putnzint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
#ifdef SDSCG
    putnsym("sdsc", SDSCG(0));
    SDSCP(0, 0);
#endif
#ifdef ENCLFUNCG
    putnsym("enclfunc", ENCLFUNCG(0));
    ENCLFUNCP(0, 0);
#endif
#ifdef OMPACCDEVSYMG
      putbit("ompaccel", OMPACCDEVSYMG(0));
      OMPACCDEVSYMP(0, 0);
#endif
#ifdef OMPACCSHMEMG
      putbit("ompaccel-shared", OMPACCSHMEMG(0));
      OMPACCSHMEMP(0, 0);
#endif
#ifdef OMPACCSTRUCTG
      putbit("ompaccel-host", OMPACCSTRUCTG(0));
      OMPACCSTRUCTP(0, 0);
#endif
#ifdef OMPACCDEVSYMG
      putbit("ompaccel", OMPACCDEVSYMG(0));
      OMPACCDEVSYMP(0, 0);
#endif
#ifdef OMPACCSHMEMG
      putbit("ompaccel-shared", OMPACCSHMEMG(0));
      OMPACCSHMEMP(0, 0);
#endif
#ifdef OMPACCSTRUCTG
      putbit("ompaccel-host", OMPACCSTRUCTG(0));
      OMPACCSTRUCTP(0, 0);
#endif
#ifdef OMPACCLITERALG
    putbit("ompaccel-literal", OMPACCLITERALG(0));
    OMPACCLITERALP(0, 0);
#endif
#ifdef SOCPTRG
    socptr = SOCPTRG(0);
    putnzint("socptr", SOCPTRG(0));
    SOCPTRP(0, 0);
#endif
    putline();
    {
#ifdef MDG
      putmd("mdg", MDG(0));
      MDP(0, 0);
#endif
    }
    putnzint("b3", b3G(0));
    b3P(0, 0);
#ifdef ALIASG
    putnsym("alias", ALIASG(0));
    ALIASP(0, 0);
#endif
#ifdef ALTNAMEG
    putnsym("altname", ALTNAMEG(0));
    ALTNAMEP(0, 0);
#endif
#ifdef BASESYMG
    if (BASEADDRG(0)) {
      putnsym("basesym", BASESYMG(0));
      BASESYMP(0, 0);
    }
#endif
#ifdef CLENG
    putnsym("clen", CLENG(0));
    CLENP(0, 0);
#endif
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
#ifdef IPAINFOG
    if (IPANAMEG(0)) {
      putnzint("ipainfo", IPAINFOG(0));
      if (stb.n_base && IPANAMEG(0) && IPAINFOG(0) > 0 &&
          IPAINFOG(0) < stb.namavl) {
        putstring("ipaname", stb.n_base + IPAINFOG(0));
      }
      IPAINFOP(0, 0);
    }
#endif
#ifdef ORIGDIMG
    putnzint("origdim", ORIGDIMG(0));
    ORIGDIMP(0, 0);
#endif
#ifdef ORIGDUMMYG
    putnsym("origdummy", ORIGDUMMYG(0));
    ORIGDUMMYP(0, 0);
#endif
#ifdef PDALNG
    putnzint("pdaln", PDALNG(0));
    b4P(0, 0);
#endif
#ifdef PARAMVALG
    {
      putnzint("paramval", PARAMVALG(0));
      PARAMVALP(0, 0);
    }
#endif
#ifdef TPLNKG
    if (stype == ST_ARRAY) {
      putnsym("tplnk", TPLNKG(0));
      TPLNKP(0, 0);
    }
#endif
#ifdef XYPTRG
    {
      putxyptr("xyptr", XYPTRG(0));
      XYPTRP(0, 0);
    }
#endif
#ifdef XREFLKG
    {
      putnsym("xreflk", XREFLKG(0));
      XREFLKP(0, 0);
    }
#endif
#ifdef ELFSCNG
    putnsym("elfscn", ELFSCNG(0));
    ELFSCNP(0, 0);
    if (stype != ST_FUNC) {
      putnsym("elflkg", ELFLKG(0));
      ELFLKP(0, 0);
    }
#endif
    putline();
    putbit("addrtkn", ADDRTKNG(0));
    ADDRTKNP(0, 0);
#ifdef ACCINITDATAG
    putbit("accinitdata", ACCINITDATAG(0));
    ACCINITDATAP(0, 0);
#endif
#ifdef ADJARRG
    putbit("adjarr", ADJARRG(0));
    ADJARRP(0, 0);
#endif
#ifdef ALLOCG
    putbit("alloc", ALLOCG(0));
    ALLOCP(0, 0);
#endif
#ifdef ARG1PTRG
    putbit("arg1ptr", ARG1PTRG(0));
    ARG1PTRP(0, 0);
#endif
    putbit("assn", ASSNG(0));
    ASSNP(0, 0);
#ifdef ASSUMSHPG
    putbit("assumshp", ASSUMSHPG(0));
    ASSUMSHPP(0, 0);
#endif
#ifdef ASUMSZG
    putbit("asumsz", ASUMSZG(0));
    ASUMSZP(0, 0);
#endif
#ifdef AUTOBJG
    putbit("autobj", AUTOBJG(0));
    AUTOBJP(0, 0);
#endif
#ifdef BASEADDRG
    putbit("baseaddr", BASEADDRG(0));
    BASEADDRP(0, 0);
#endif
#ifdef CALLEDG
    putbit("called", CALLEDG(0));
    CALLEDP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef CLASSG
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
#ifdef CONSTANTG
    putbit("constant", CONSTANTG(0));
    CONSTANTP(0, 0);
#endif
#ifdef CONSTG
    putbit("const", CONSTG(0));
    CONSTP(0, 0);
#endif
#ifdef COPYPRMSG
    putbit("copyprms", COPYPRMSG(0));
    COPYPRMSP(0, 0);
#endif
#ifdef DCLDG
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#endif
    putbit("defd", DEFDG(0));
    DEFDP(0, 0);
#ifdef DESCARRAYG
    putbit("descarray", DESCARRAYG(0));
    DESCARRAYP(0, 0);
#endif
#ifdef DEVICEG
    putbit("device", DEVICEG(0));
    DEVICEP(0, 0);
#endif
#ifdef DEVICECOPYG
    putbit("devicecopy", DEVICECOPYG(0));
    DEVICECOPYP(0, 0);
#endif
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
#ifdef DVLG
    putbit("dvl", DVLG(0));
    DVLP(0, 0);
#endif
#ifdef ESCTYALIASG
    putbit("esctyalias", ESCTYALIASG(0));
    ESCTYALIASP(0, 0);
#endif
#ifdef FROMINLRG
    putbit("frominlr", FROMINLRG(0));
    FROMINLRP(0, 0);
#endif
    putbit("gscope", GSCOPEG(0));
    GSCOPEP(0, 0);
#ifdef HOMEDG
    putbit("homed", HOMEDG(0));
    HOMEDP(0, 0);
#endif
#ifdef INLING
    putbit("inlin", INLING(0));
    INLINP(0, 0);
#endif
#ifdef ALWAYSINLING
    putbit("alwaysinlin", ALWAYSINLING(0));
    ALWAYSINLINP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
#ifdef INLNARRG
    putbit("inlnarr", INLNARRG(0));
    INLNARRP(0, 0);
#endif
#ifdef LIBCG
    putbit("libc", LIBCG(0));
    LIBCP(0, 0);
#endif
#ifdef LIBMG
    putbit("libm", LIBMG(0));
    LIBMP(0, 0);
#endif
#ifdef LOCLIFETMG
    putbit("loclifetm", LOCLIFETMG(0));
    LOCLIFETMP(0, 0);
#endif
#ifdef LSCOPEG
    putbit("lscope", LSCOPEG(0));
    LSCOPEP(0, 0);
#endif
#ifdef LVALG
    putbit("lval", LVALG(0));
    LVALP(0, 0);
#endif
#ifdef MANAGEDG
    putbit("managed", MANAGEDG(0));
    MANAGEDP(0, 0);
#endif
    putbit("memarg", MEMARGG(0));
    MEMARGP(0, 0);
#ifdef MIRROREDG
    putbit("mirrored", MIRROREDG(0));
    MIRROREDP(0, 0);
#endif
#ifdef ACCCREATEG
    putbit("acccreate", ACCCREATEG(0));
    ACCCREATEP(0, 0);
#endif
#ifdef ACCCOPYING
    putbit("acccopyin", ACCCOPYING(0));
    ACCCOPYINP(0, 0);
#endif
#ifdef ACCRESIDENTG
    putbit("accresident", ACCRESIDENTG(0));
    ACCRESIDENTP(0, 0);
#endif
#ifdef ACCLINKG
    putbit("acclink", ACCLINKG(0));
    ACCLINKP(0, 0);
#endif
#ifdef MODESETG
    if (stype == ST_FUNC) {
      putbit("modeset", MODESETG(0));
      MODESETP(0, 0);
    }
#endif
#ifdef MSCALLG
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#endif
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
#ifdef NOCONFLICTG
    putbit("noconflict", NOCONFLICTG(0));
    NOCONFLICTP(0, 0);
#endif
#ifdef NOTEXTERNG
    putbit("notextern", NOTEXTERNG(0));
    NOTEXTERNP(0, 0);
#endif
#ifdef OPTARGG
    putbit("optarg", OPTARGG(0));
    OPTARGP(0, 0);
#endif
#ifdef OSXINITG
    if (stype == ST_FUNC) {
      putbit("osxinit", OSXINITG(0));
      OSXINITP(0, 0);
      putbit("osxterm", OSXTERMG(0));
      OSXTERMP(0, 0);
    }
#endif
#ifdef PARAMG
    putbit("param", PARAMG(0));
    PARAMP(0, 0);
#endif
#ifdef PASSBYVALG
    putbit("passbyval", PASSBYVALG(0));
    PASSBYVALP(0, 0);
#endif
#ifdef PASSBYREFG
    putbit("passbyref", PASSBYREFG(0));
    PASSBYREFP(0, 0);
#endif
#ifdef PINNEDG
    putbit("pinned", PINNEDG(0));
    PINNEDP(0, 0);
#endif
#ifdef POINTERG
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
#endif
#ifdef ALLOCATTRP
    putbit("allocattr", ALLOCATTRG(0));
    ALLOCATTRP(0, 0);
#endif
#ifdef PROTODCLG
    putbit("protodcl", PROTODCLG(0));
    PROTODCLP(0, 0);
#endif
#ifdef PTRSAFEG
    putbit("ptrsafe", PTRSAFEG(0));
    PTRSAFEP(0, 0);
#endif
    putbit("qaln", QALNG(0));
    QALNP(0, 0);
#ifdef REDUCG
    putbit("reduc", REDUCG(0));
    REDUCP(0, 0);
#endif
#ifdef REFG
    putbit("ref", REFG(0));
    REFP(0, 0);
#endif
#ifdef REFDG
    putbit("refd", REFDG(0));
    REFDP(0, 0);
#endif
#ifdef REFLECTEDG
    putbit("reflected", REFLECTEDG(0));
    REFLECTEDP(0, 0);
#endif
    putbit("regarg", REGARGG(0));
    REGARGP(0, 0);
#ifdef RESTRICTG
    putbit("restrict", RESTRICTG(0));
    RESTRICTP(0, 0);
#endif
#ifdef SAFEG
    putbit("safe", SAFEG(0));
    SAFEP(0, 0);
#endif
#ifdef SAVEG
    putbit("save", SAVEG(0));
    SAVEP(0, 0);
#endif
#ifdef SDSCCONTIGG
    putbit("sdsccontig", SDSCCONTIGG(0));
    SDSCCONTIGP(0, 0);
#endif
#ifdef SDSCS1G
    putbit("sdscs1", SDSCS1G(0));
    SDSCS1P(0, 0);
#endif
#ifdef SECTG
    putbit("sect", SECTG(0));
    SECTP(0, 0);
#endif
#ifdef SHAREDG
    putbit("shared", SHAREDG(0));
    SHAREDP(0, 0);
#endif
#ifdef TEXTUREG
    putbit("texture", TEXTUREG(0));
    TEXTUREP(0, 0);
#endif
#ifdef INTENTING
    putbit("intentin", INTENTING(0));
    INTENTINP(0, 0);
#endif
    putbit("thread", THREADG(0));
    THREADP(0, 0);
#ifdef UNSAFEG
    putbit("unsafe", UNSAFEG(0));
    UNSAFEP(0, 0);
#endif
#ifdef UPLEVELG
    putbit("uplevel", UPLEVELG(0));
    UPLEVELP(0, 0);
#endif
#ifdef INTERNREFG
    putbit("internref", INTERNREFG(0));
    INTERNREFP(0, 0);
#endif
#ifdef VARDSCG
    putbit("vardsc", VARDSCG(0));
    VARDSCP(0, 0);
#endif
#ifdef VLAG
    putbit("vla", VLAG(0));
    VLAP(0, 0);
#endif
#ifdef VLAIDXG
    putbit("vlaidx", VLAIDXG(0));
    VLAIDXP(0, 0);
#endif
    putbit("vol", VOLG(0));
    VOLP(0, 0);
#ifdef WEAKG
    putbit("weak", WEAKG(0));
    WEAKP(0, 0);
#endif
#ifdef PARREFG
    putbit("parref", PARREFG(0));
    PARREFP(0, 0);
#endif
#ifdef PARREFLOADG
    putbit("parrefload", PARREFLOADG(0));
    PARREFLOADP(0, 0);
#endif
#ifdef OMPTEAMPRIVATEG
    putbit("team-private", OMPTEAMPRIVATEG(0));
    OMPTEAMPRIVATEP(0, 0);
#endif
/*
        putbit( "#", #G(0) );		#P(0,0);
*/
#ifdef SOCPTRG
    if (socptr)
      putsoc(socptr);
#endif
    break;

  case ST_BLOCK:
    putint("startline", STARTLINEG(0));
    STARTLINEP(0, 0);
    putint("endline", ENDLINEG(0));
    ENDLINEP(0, 0);
#ifdef ENCLFUNCG
    putnsym("enclfunc", ENCLFUNCG(0));
    ENCLFUNCP(0, 0);
#endif
    putnsym("startlab", STARTLABG(0));
    STARTLABP(0, 0);
    putnsym("endlab", ENDLABG(0));
    ENDLABP(0, 0);
    putnsym("beginscopelab", BEGINSCOPELABG(0));
    BEGINSCOPELABP(0, 0);
    putnsym("endscopelab", ENDSCOPELABG(0));
    ENDSCOPELABP(0, 0);
#ifdef AUTOBJG
    putnzint("autobj", AUTOBJG(0));
    AUTOBJP(0, 0);
#endif
#ifdef AINITG
    putbit("ainit", AINITG(0));
    AINITP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef FROMINLRG
    putbit("frominlr", FROMINLRG(0));
    FROMINLRP(0, 0);
#endif
#ifdef REFDG
    putbit("refd", REFDG(0));
    REFDP(0, 0);
#endif
#ifdef PARUPLEVELG
    putint("paruplevel", PARUPLEVELG(0));
    PARUPLEVELP(0, 0);
#endif
#ifdef PARSYMSG
    putbit("parsyms", PARSYMSG(0));
    PARSYMSP(0, 0);
#endif
#ifdef PARSYMSCTG
    putbit("parsymsct", PARSYMSCTG(0));
    PARSYMSCTP(0, 0);
#endif
    break;

  case ST_BASE:
    break;

  case ST_CMBLK:
    putint("size", SIZEG(0));
    SIZEP(0, 0);
    putline();
    putsym("cmemf", CMEMFG(0));
    CMEMFP(0, 0);
    putsym("cmeml", CMEMLG(0));
    CMEMLP(0, 0);
#ifdef INMODULEG
    putnzint("inmodule", INMODULEG(0));
    INMODULEP(0, 0);
#endif
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
#ifdef ALTNAMEG
    putnsym("altname", ALTNAMEG(0));
    ALTNAMEP(0, 0);
#endif
#ifdef PDALNG
    putnzint("pdaln", PDALNG(0));
    PDALNP(0, 0);
#endif
    putline();
#ifdef ACCCREATEG
    putbit("acccreate", ACCCREATEG(0));
    ACCCREATEP(0, 0);
#endif
#ifdef ACCCOPYING
    putbit("acccopyin", ACCCOPYING(0));
    ACCCOPYINP(0, 0);
#endif
#ifdef ACCRESIDENTG
    putbit("accresident", ACCRESIDENTG(0));
    ACCRESIDENTP(0, 0);
#endif
#ifdef ACCLINKG
    putbit("acclink", ACCLINKG(0));
    ACCLINKP(0, 0);
#endif
    putbit("alloc", ALLOCG(0));
    ALLOCP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef CONSTANTG
    putbit("constant", CONSTANTG(0));
    CONSTANTP(0, 0);
#endif
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("defd", DEFDG(0));
    DEFDP(0, 0);
#ifdef DEVICEG
    putbit("device", DEVICEG(0));
    DEVICEP(0, 0);
#endif
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
#ifdef FROMINLRG
    putbit("frominlr", FROMINLRG(0));
    FROMINLRP(0, 0);
#endif
#ifdef FROMMODG
    putbit("frommod", FROMMODG(0));
    FROMMODP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
#ifdef MODCMNG
    putbit("modcmn", MODCMNG(0));
    MODCMNP(0, 0);
#endif
    putbit("qaln", QALNG(0));
    QALNP(0, 0);
    putbit("save", SAVEG(0));
    SAVEP(0, 0);
#ifdef THREADG
    putbit("thread", THREADG(0));
    THREADP(0, 0);
#endif
    putbit("vol", VOLG(0));
    VOLP(0, 0);
    putsymlk("Members:", CMEMFG(sptr));
    break;

  case ST_CONST:
    putnzint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    if (DTY(dtype) == TY_PTR) {
      putsym("pointee", STGetPointee(0));
      CONVAL1P(0, 0);
      putnzbigint("offset", ACONOFFG(0));
      ACONOFFP(0, 0);
    } else {
      putint("conval1g", CONVAL1G(0));
      CONVAL1P(0, 0);
      putbigint("conval2g", CONVAL2G(0));
      CONVAL2P(0, 0);
    }
    putline();
    break;

  case ST_ENTRY:
    putnzint("bihnum", BIHNUMG(0));
    BIHNUMP(0, 0);
#ifdef ARGSIZEG
    putnzint("argsize", ARGSIZEG(0));
    ARGSIZEP(0, 0);
#endif
    putint("dpdsc", DPDSCG(0));
    DPDSCP(0, 0);
    putint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
#ifdef DECLLINEG
    putnzint("declline", DECLLINEG(0));
    DECLLINEP(0, 0);
#endif
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
#ifdef ALTNAMEG
    putnsym("altname", ALTNAMEG(0));
    ALTNAMEP(0, 0);
#endif
    putline();
#ifdef INMODULEG
    putnsym("inmodule", INMODULEG(0));
    INMODULEP(0, 0);
#endif
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putnsym("fval", FVALG(0));
    FVALP(0, 0);
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
#ifdef ACCROUTG
    putnzint("accrout", ACCROUTG(0));
    ACCROUTP(0, 0);
#endif
#ifdef OMPACCFUNCDEVG
    putbit("ompaccel-device", OMPACCFUNCDEVG(0));
    OMPACCFUNCDEVP(0, 0);
#endif
#ifdef OMPACCFUNCKERNELG
    putbit("ompaccel-kernel", OMPACCFUNCKERNELG(0));
    OMPACCFUNCKERNELP(0, 0);
#endif
#ifdef OMPACCFUNCDEVG
    putbit("ompaccel-device", OMPACCFUNCDEVG(0));
    OMPACCFUNCDEVP(0, 0);
#endif
#ifdef OMPACCFUNCKERNELG
    putbit("ompaccel-kernel", OMPACCFUNCKERNELG(0));
    OMPACCFUNCKERNELP(0, 0);
#endif
#ifdef IPAINFOG
    putnzint("ipainfo", IPAINFOG(0));
    if (stb.n_base && IPANAMEG(0) && IPAINFOG(0) > 0 &&
        IPAINFOG(0) < stb.namavl) {
      putstring("ipaname", stb.n_base + IPAINFOG(0));
    }
    IPAINFOP(0, 0);
#endif
    putline();
    putbit("adjarr", ADJARRG(0));
    ADJARRP(0, 0);
    putbit("aftent", AFTENTG(0));
    AFTENTP(0, 0);
#ifdef CALLEDG
    putbit("called", CALLEDG(0));
    CALLEDP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef CONTAINEDG
    putbit("contained", CONTAINEDG(0));
    CONTAINEDP(0, 0);
#endif
#ifdef COPYPRMSG
    putbit("copyprms", COPYPRMSG(0));
    COPYPRMSP(0, 0);
#endif
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#ifdef DECORATEG
    putbit("decorate", DECORATEG(0));
    DECORATEP(0, 0);
#endif
#ifdef FROMINLRG
    putbit("frominlr", FROMINLRG(0));
    FROMINLRP(0, 0);
#endif
#ifdef MSCALLG
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#endif
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
#ifdef NOTEXTERNG
    putbit("notextern", NOTEXTERNG(0));
    NOTEXTERNP(0, 0);
#endif
    putbit("pure", PUREG(0));
    PUREP(0, 0);
#ifdef ELEMENTALG
    putbit("elemental", ELEMENTALG(0));
    ELEMENTALP(0, 0);
#endif
#ifdef RECURG
    putbit("recur", RECURG(0));
    RECURP(0, 0);
#endif
#ifdef STDCALLG
    putbit("stdcall", STDCALLG(0));
    STDCALLP(0, 0);
#endif
    putline();
    putparam(DPDSCG(sptr), PARAMCTG(sptr));
    break;

  case ST_GENERIC:
    putnsym("gcmplx", GCMPLXG(0));
    GCMPLXP(0, 0);
    putnsym("gdble", GDBLEG(0));
    GDBLEP(0, 0);
    putnsym("gdcmplx", GDCMPLXG(0));
    GDCMPLXP(0, 0);
    putnsym("gint", GINTG(0));
    GINTP(0, 0);
    putnsym("gint8", GINT8G(0));
    GINT8P(0, 0);
#ifdef GQCMPLXG
    putnsym("gqcmplx", GQCMPLXG(0));
    GQCMPLXP(0, 0);
#endif
#ifdef GQUADG
    putnsym("gquad", GQUADG(0));
    GQUADP(0, 0);
#endif
    putnsym("greal", GREALG(0));
    GREALP(0, 0);
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putnsym("gsint", GSINTG(0));
    GSINTP(0, 0);
    putnzint("gndsc", GNDSCG(0));
    GNDSCP(0, 0);
    putnzint("gncnt", GNCNTG(0));
    GNCNTP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    break;

  case ST_INTRIN:
    putddtype("argtyp", ARGTYPG(0));
    ARGTYPP(0, 0);
    putnzint("arrayf", ARRAYFG(0));
    ARRAYFP(0, 0);
    putnzint("ilm", ILMG(0));
    ILMP(0, 0);
    putddtype("inttyp", INTTYPG(0));
    INTTYPP(0, 0);
    putnname("pnmptr", PNMPTRG(0));
    PNMPTRP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("expst", EXPSTG(0));
    EXPSTP(0, 0);
    break;

  case ST_LABEL:
    putnzint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
#ifdef ENCLFUNCG
    putnsym("enclfunc", ENCLFUNCG(0));
    ENCLFUNCP(0, 0);
#endif
#ifdef FMTPTG
    putnsym("fmtpt", FMTPTG(0));
    FMTPTP(0, 0);
#endif
    putnzint("iliblk", ILIBLKG(0));
    ILIBLKP(0, 0);
#ifdef JSCOPEG
    putnzint("jscope", JSCOPEG(0));
    JSCOPEP(0, 0);
#endif
    putint("rfcnt", RFCNTG(0));
    RFCNTP(0, 0);
    putline();
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("defd", DEFDG(0));
    DEFDP(0, 0);
#ifdef INLING
    putbit("inlin", INLING(0));
    INLINP(0, 0);
#endif
#ifdef ALWAYSINLING
    putbit("alwaysinlin", ALWAYSINLING(0));
    ALWAYSINLINP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
#ifdef LSCOPEG
    putbit("lscope", LSCOPEG(0));
    LSCOPEP(0, 0);
#endif
    putbit("qaln", QALNG(0));
    QALNP(0, 0);
#ifdef REFDG
    putbit("refd", REFDG(0));
    REFDP(0, 0);
#endif
    putbit("vol", VOLG(0));
    VOLP(0, 0);
    putbit("beginscope", BEGINSCOPEG(0));
    BEGINSCOPEP(0, 0);
    putbit("endscope", ENDSCOPEG(0));
    ENDSCOPEP(0, 0);
    break;

  case ST_MEMBER:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
#ifdef BITOFFG
    putnzint("bitoff", BITOFFG(0));
    BITOFFP(0, 0);
#endif
#ifdef FLDSZG
    putnzint("fldsz", FLDSZG(0));
    FLDSZP(0, 0);
#endif
#ifdef LDSIZEG
    putnzint("ldsize", LDSIZEG(0));
    LDSIZEP(0, 0);
#endif
    putsym("psmem", PSMEMG(0));
    PSMEMP(0, 0);
#ifdef VARIANTG
    putsym("variant", VARIANTG(0));
    VARIANTP(0, 0);
#endif
#ifdef INDTYPEG
    putnzint("indtype", INDTYPEG(0));
    INDTYPEP(0, 0);
#endif
#ifdef SDSCG
    putnsym("sdsc", SDSCG(0));
    SDSCP(0, 0);
#endif
    putline();
#ifdef ALLOCATTRG
    putbit("allocattr", ALLOCATTRG(0));
    ALLOCATTRP(0, 0);
#endif
#ifdef CLASSG
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef DCLDG
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#endif
#ifdef DESCARRAYG
    putbit("descarray", DESCARRAYG(0));
    DESCARRAYP(0, 0);
#endif
#ifdef FIELDG
    putbit("field", FIELDG(0));
    FIELDP(0, 0);
#endif
#ifdef INLING
    putbit("inlin", INLING(0));
    INLINP(0, 0);
#endif
#ifdef ALWAYSINLING
    putbit("alwaysinlin", ALWAYSINLING(0));
    ALWAYSINLINP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
#ifdef MSCALLG
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#endif
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
#ifdef NOCONFLICTG
    putbit("noconflict", NOCONFLICTG(0));
    NOCONFLICTP(0, 0);
#endif
#ifdef PLAING
    putbit("plain", PLAING(0));
    PLAINP(0, 0);
#endif
#ifdef POINTERG
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
#endif
#ifdef PTRSAFEG
    putbit("ptrsafe", PTRSAFEG(0));
    PTRSAFEP(0, 0);
#endif
#ifdef REFDG
    putbit("refd", REFDG(0));
    REFDP(0, 0);
#endif
#ifdef SDSCCONTIGG
    putbit("sdsccontig", SDSCCONTIGG(0));
    SDSCCONTIGP(0, 0);
#endif
#ifdef SDSCS1G
    putbit("sdscs1", SDSCS1G(0));
    SDSCS1P(0, 0);
#endif
    break;

  case ST_NML:
    putsym("plist", (SPTR) ADDRESSG(0)); // ???
    ADDRESSP(0, 0);
    putsym("cmemf", CMEMFG(0));
    CMEMFP(0, 0);
    putsym("cmeml", CMEMLG(0));
    CMEMLP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
    putline();
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i)) {
      putsym1(NML_SPTR(i));
    }
    break;

  case ST_PARAM:
    putint("conval1g", CONVAL1G(0));
    CONVAL1P(0, 0);
    putint("conval2g", CONVAL2G(0));
    CONVAL2P(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
    break;

  case ST_PD:
    putddtype("argtyp", ARGTYPG(0));
    ARGTYPP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putnzint("ilm", ILMG(0));
    ILMP(0, 0);
    putint("pdnum", PDNUMG(0));
    PDNUMP(0, 0);
    break;

  case ST_PLIST:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
#ifdef BASESYMG
    if (BASEADDRG(0)) {
      putnsym("basesym", BASESYMG(0));
      BASESYMP(0, 0);
    }
#endif
    putnzint("deflab", DEFLABG(0));
    DEFLABP(0, 0);
    putint("pllen", PLLENG(0));
    PLLENP(0, 0);
    putnzint("swel", SWELG(0));
    SWELP(0, 0);
    putline();
    putbit("addrtkn", ADDRTKNG(0));
    ADDRTKNP(0, 0);

#ifdef BASEADDRG
    putbit("baseaddr", BASEADDRG(0));
    BASEADDRP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
#ifdef INLING
    putbit("inlin", INLING(0));
    INLINP(0, 0);
#endif
#ifdef ALWAYSINLING
    putbit("alwaysinlin", ALWAYSINLING(0));
    ALWAYSINLINP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
#ifdef LSCOPEG
    putbit("lscope", LSCOPEG(0));
    LSCOPEP(0, 0);
#endif
    putbit("ref", REFG(0));
    REFP(0, 0);
    break;

  case ST_PROC:
#ifdef INMODULEG
    putnsym("inmodule", INMODULEG(0));
    INMODULEP(0, 0);
#endif
#ifdef ARGSIZEG
    putnzint("argsize", ARGSIZEG(0));
    ARGSIZEP(0, 0);
#endif
    putnzint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
#ifdef IPAINFOG
    putnzint("ipainfo", IPAINFOG(0));
    if (stb.n_base && IPANAMEG(0) && IPAINFOG(0) > 0 &&
        IPAINFOG(0) < stb.namavl) {
      putstring("ipaname", stb.n_base + IPAINFOG(0));
    }
    IPAINFOP(0, 0);
#endif
#ifdef ALTNAMEG
    putnsym("altname", ALTNAMEG(0));
    ALTNAMEP(0, 0);
#endif
#ifdef ACCROUTG
    putnzint("accrout", ACCROUTG(0));
    ACCROUTP(0, 0);
#endif
#ifdef ARG1PTRG
    putbit("arg1ptr", ARG1PTRG(0));
    ARG1PTRP(0, 0);
#endif
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
    putline();
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef COPYPRMSG
    putbit("copyprms", COPYPRMSG(0));
    COPYPRMSP(0, 0);
#endif
#ifdef CONTAINEDG
    putbit("contained", CONTAINEDG(0));
    CONTAINEDP(0, 0);
#endif
#ifdef DECORATEG
    putbit("decorate", DECORATEG(0));
    DECORATEP(0, 0);
#endif
#ifdef MSCALLG
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#endif
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
#ifdef NEEDMODG
    putbit("needmod", NEEDMODG(0));
    NEEDMODP(0, 0);
#endif
    putbit("nopad", NOPADG(0));
    NOPADP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("func", FUNCG(0));
    FUNCP(0, 0);
#ifdef FROMINLRG
    putbit("frominlr", FROMINLRG(0));
    FROMINLRP(0, 0);
#endif
#ifdef LIBMG
    putbit("libm", LIBMG(0));
    LIBMP(0, 0);
#endif
    putbit("memarg", MEMARGG(0));
    MEMARGP(0, 0);
#ifdef PASSBYVALG
    putbit("passbyval", PASSBYVALG(0));
    PASSBYVALP(0, 0);
#endif
#ifdef PASSBYREFG
    putbit("passbyref", PASSBYREFG(0));
    PASSBYREFP(0, 0);
#endif
    putbit("pure", PUREG(0));
    PUREP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
#ifdef SDSCSAFEG
    putbit("sdscsafe", SDSCSAFEG(0));
    SDSCSAFEP(0, 0);
#endif
#ifdef STDCALLG
    putbit("stdcall", STDCALLG(0));
    STDCALLP(0, 0);
#endif
#ifdef UNIFIEDG
    putbit("unified", UNIFIEDG(0));
    UNIFIEDP(0, 0);
#endif
    putline();
    putparam(DPDSCG(sptr), PARAMCTG(sptr));
    break;

  case ST_STFUNC:
    putint("excvlen", EXCVLENG(0));
    EXCVLENP(0, 0);
    putint("sfdsc", SFDSCG(0));
    SFDSCP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    break;

  case ST_TYPEDEF:
#ifdef ENCLFUNCG
    putnsym("enclfunc", ENCLFUNCG(0));
    ENCLFUNCP(0, 0);
#endif
#ifdef PDALNG
    putnzint("pdaln", PDALNG(0));
    PDALNP(0, 0);
#endif
#ifdef DCLDG
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#endif
#ifdef FROMMODG
    putbit("frommod", FROMMODG(0));
    FROMMODP(0, 0);
#endif
#ifdef MSCALLG
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#endif
#ifdef CFUNCG
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#endif
#ifdef CLASSG
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
#ifdef PLAING
    putbit("plain", PLAING(0));
    PLAINP(0, 0);
#endif
    break;

  case ST_STAG:
#ifdef ENCLFUNCG
    putnsym("enclfunc", ENCLFUNCG(0));
    ENCLFUNCP(0, 0);
#endif
#ifdef DCLDG
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#endif
#ifdef INLNG
    putbit("inln", INLNG(0));
    INLNP(0, 0);
#endif
    break;

  } /* switch(stype) */
  putline();

  check("b3", stb.stg_base[0].b3);
  check("b4", stb.stg_base[0].b4);
  check("f1", stb.stg_base[0].f1);
  check("f2", stb.stg_base[0].f2);
  check("f3", stb.stg_base[0].f3);
  check("f4", stb.stg_base[0].f4);
  check("f5", stb.stg_base[0].f5);
  check("f6", stb.stg_base[0].f6);
  check("f7", stb.stg_base[0].f7);
  check("f8", stb.stg_base[0].f8);
  check("f9", stb.stg_base[0].f9);
  check("f10", stb.stg_base[0].f10);
  check("f11", stb.stg_base[0].f11);
  check("f12", stb.stg_base[0].f12);
  check("f13", stb.stg_base[0].f13);
  check("f14", stb.stg_base[0].f14);
  check("f15", stb.stg_base[0].f15);
  check("f16", stb.stg_base[0].f16);
  check("f17", stb.stg_base[0].f17);
  check("f18", stb.stg_base[0].f18);
  check("f19", stb.stg_base[0].f19);
  check("f20", stb.stg_base[0].f20);
  check("f21", stb.stg_base[0].f21);
  check("f22", stb.stg_base[0].f22);
  check("f23", stb.stg_base[0].f23);
  check("f24", stb.stg_base[0].f24);
  check("f25", stb.stg_base[0].f25);
  check("f26", stb.stg_base[0].f26);
  check("f27", stb.stg_base[0].f27);
  check("f28", stb.stg_base[0].f28);
  check("f29", stb.stg_base[0].f29);
  check("f30", stb.stg_base[0].f30);
  check("f31", stb.stg_base[0].f31);
  check("f32", stb.stg_base[0].f32);
  check("f33", stb.stg_base[0].f33);
  check("f34", stb.stg_base[0].f34);
  check("f35", stb.stg_base[0].f35);
  check("f36", stb.stg_base[0].f36);
  check("f37", stb.stg_base[0].f37);
  check("f38", stb.stg_base[0].f38);
  check("f39", stb.stg_base[0].f39);
  check("f40", stb.stg_base[0].f40);
  check("f41", stb.stg_base[0].f41);
  check("f42", stb.stg_base[0].f42);
  check("f43", stb.stg_base[0].f43);
  check("f44", stb.stg_base[0].f44);
  check("f45", stb.stg_base[0].f45);
  check("f46", stb.stg_base[0].f46);
  check("f47", stb.stg_base[0].f47);
  check("f48", stb.stg_base[0].f48);
  check("f50", stb.stg_base[0].f50);
  check("f51", stb.stg_base[0].f51);
  check("f52", stb.stg_base[0].f52);
  check("f53", stb.stg_base[0].f53);
  check("f54", stb.stg_base[0].f54);
  check("f55", stb.stg_base[0].f55);
  check("f56", stb.stg_base[0].f56);
  check("f57", stb.stg_base[0].f57);
  check("f58", stb.stg_base[0].f58);
  check("f59", stb.stg_base[0].f59);
  check("f60", stb.stg_base[0].f60);
  check("f61", stb.stg_base[0].f61);
  check("f62", stb.stg_base[0].f62);
  check("f63", stb.stg_base[0].f63);
  check("f64", stb.stg_base[0].f64);
  check("f65", stb.stg_base[0].f65);
  check("f66", stb.stg_base[0].f66);
  check("f67", stb.stg_base[0].f67);
  check("f68", stb.stg_base[0].f68);
  check("f69", stb.stg_base[0].f69);
  check("f70", stb.stg_base[0].f70);
  check("f71", stb.stg_base[0].f71);
  check("f72", stb.stg_base[0].f72);
  check("f73", stb.stg_base[0].f73);
  check("f74", stb.stg_base[0].f74);
  check("f75", stb.stg_base[0].f75);
  check("f76", stb.stg_base[0].f76);
  check("f77", stb.stg_base[0].f77);
  check("f78", stb.stg_base[0].f78);
  check("f79", stb.stg_base[0].f79);
  check("f80", stb.stg_base[0].f80);
  check("f81", stb.stg_base[0].f81);
  check("f82", stb.stg_base[0].f82);
  check("f83", stb.stg_base[0].f83);
  check("f84", stb.stg_base[0].f84);
  check("f85", stb.stg_base[0].f85);
  check("f86", stb.stg_base[0].f86);
  check("f87", stb.stg_base[0].f87);
  check("f88", stb.stg_base[0].f88);
  check("f89", stb.stg_base[0].f89);
  check("f90", stb.stg_base[0].f90);
  check("f91", stb.stg_base[0].f91);
  check("f92", stb.stg_base[0].f92);
  check("f93", stb.stg_base[0].f93);
  check("f94", stb.stg_base[0].f94);
  check("f95", stb.stg_base[0].f95);
  check("f96", stb.stg_base[0].f96);
  check("f110", stb.stg_base[0].f110);
  check("f111", stb.stg_base[0].f111);
  check("w8", stb.stg_base[0].w8);
  check("w9", stb.stg_base[0].w9);
  check("w10", stb.stg_base[0].w10);
  check("w11", stb.stg_base[0].w11);
  check("w12", stb.stg_base[0].w12);
  check("w13", stb.stg_base[0].w13);
  check("w14", stb.stg_base[0].w14);
  check("w15", stb.stg_base[0].w15);
  check("w16", stb.stg_base[0].w16);
  check("w17", stb.stg_base[0].w17);
  check("w18", stb.stg_base[0].w18);
  check("w20", stb.stg_base[0].w20);
} /* dsym */

void
dsyms(int l, int u)
{
  int i;
  if (l <= 0)
    l = stb.firstusym;
  if (u <= 0)
    u = stb.stg_avail - 1;
  if (u >= stb.stg_avail)
    u = stb.stg_avail - 1;
  for (i = l; i <= u; ++i) {
    dsym(i);
  }
  fprintf(dfile, "\n");
} /* dsyms */

void
ds(int sptr)
{
  dsym(sptr);
} /* ds */

void
dsa(void)
{
  dsyms(0, 0);
} /* dsa */

void
dss(int l, int u)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** SYMBOL TABLE **********\n");
  dsyms(l, u);
} /* dss */

void
dgbl(void)
{
  GBL mbl;
  int *ff;
  int i, mblsize;
  memcpy(&mbl, &gbl, sizeof(gbl));
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  putsym("gbl.currsub", mbl.currsub);
  mbl.currsub = SPTR_NULL;
  putnstring("gbl.datetime", mbl.datetime);
  memset(mbl.datetime, 0, sizeof(mbl.datetime));
  putline();
  putnzint("gbl.maxsev", mbl.maxsev);
  mbl.maxsev = 0;
  putnzint("gbl.findex", mbl.findex);
  mbl.findex = 0;
  putsymlk("gbl.entries=", mbl.entries);
  mbl.entries = SPTR_NULL;
  putsymlk("gbl.cmblks=", mbl.cmblks);
  mbl.cmblks = SPTR_NULL;
  putsymlk("gbl.externs=", mbl.externs);
  mbl.externs = SPTR_NULL;
  putsymlk("gbl.consts=", mbl.consts);
  mbl.consts = SPTR_NULL;
  putsymlk("gbl.locals=", mbl.locals);
  mbl.locals = SPTR_NULL;
  putsymlk("gbl.statics=", mbl.statics);
  mbl.statics = SPTR_NULL;
  putsymlk("gbl.bssvars=", mbl.bssvars);
  mbl.bssvars = SPTR_NULL;
  putsymlk("gbl.locals=", mbl.locals);
  mbl.locals = SPTR_NULL;
  putsymlk("gbl.basevars=", mbl.basevars);
  mbl.basevars = SPTR_NULL;
  putsymlk("gbl.asgnlbls=", mbl.asgnlbls);
  mbl.asgnlbls = SPTR_NULL;
  putsymlk("gbl.autobj=", mbl.autobj);
  mbl.autobj = 0;
  putsymlk("gbl.typedescs=", mbl.typedescs);
  mbl.typedescs = SPTR_NULL;
  putline();
  putnsym("gbl.outersub", mbl.outersub);
  mbl.outersub = SPTR_NULL;
  putline();
  putnzint("gbl.vfrets", mbl.vfrets);
  mbl.vfrets = 0;
  putnzint("gbl.func_count", mbl.func_count);
  mbl.func_count = 0;
  putnzint("gbl.rutype=", mbl.rutype);
  mbl.rutype = (RUTYPE)0; // no 0 value defined
  putnzint("gbl.funcline=", mbl.funcline);
  mbl.funcline = 0;
  putnzint("gbl.threadprivate=", mbl.threadprivate);
  mbl.threadprivate = SPTR_NULL;
  putnzint("gbl.nofperror=", mbl.nofperror);
  mbl.nofperror = 0;
  putnzint("gbl.fperror_status=", mbl.fperror_status);
  mbl.fperror_status = 0;
  putnzint("gbl.entbih", mbl.entbih);
  mbl.entbih = 0;
  putnzint("gbl.lineno", mbl.lineno);
  mbl.lineno = 0;
  mbl.multiversion = 0;
  mbl.multi_func_count = 0;
  mbl.numversions = 0;
  putnzint("gbl.pgfi_avail", mbl.pgfi_avail);
  mbl.pgfi_avail = 0;
  putnzint("gbl.ec_avail", mbl.ec_avail);
  mbl.ec_avail = 0;
  putnzint("gbl.cuda_constructor", mbl.cuda_constructor);
  mbl.cuda_constructor = 0;
  putnzint("gbl.cudaemu", mbl.cudaemu);
  mbl.cudaemu = 0;
  putnzint("gbl.ftn_true", mbl.ftn_true);
  mbl.ftn_true = 0;
  putnzint("gbl.in_include", mbl.in_include);
  mbl.in_include = 0;
  putnzint("gbl.denorm", mbl.denorm);
  mbl.denorm = 0;
  putnzint("gbl.nowarn", mbl.nowarn);
  mbl.nowarn = 0;
  putnzint("gbl.internal", mbl.internal);
  mbl.internal = 0;
  putnzisz("gbl.caddr", mbl.caddr);
  mbl.caddr = 0;
  putnzisz("gbl.locaddr", mbl.locaddr);
  mbl.locaddr = 0;
  putnzisz("gbl.saddr", mbl.saddr);
  mbl.saddr = 0;
  putnzisz("gbl.bss_addr", mbl.bss_addr);
  mbl.bss_addr = 0;
  putnzisz("gbl.paddr", mbl.paddr);
  mbl.paddr = 0;
  putline();
  putnsym("gbl.prvt_sym_sz", (SPTR) mbl.prvt_sym_sz); // ???
  mbl.prvt_sym_sz = 0;
  putnsym("gbl.stk_sym_sz", (SPTR) mbl.stk_sym_sz); // ???
  mbl.stk_sym_sz = 0;
  putline();
  putnstring("gbl.src_file", mbl.src_file);
  mbl.src_file = NULL;
  putnstring("gbl.file_name", mbl.file_name);
  mbl.file_name = NULL;
  putnstring("gbl.curr_file", mbl.curr_file);
  mbl.curr_file = NULL;
  putnstring("gbl.module", mbl.module);
  mbl.module = NULL;
  mbl.srcfil = NULL;
  mbl.cppfil = NULL;
  mbl.dbgfil = NULL;
  mbl.ilmfil = NULL;
  mbl.objfil = NULL;
  mbl.asmfil = NULL;
  putline();
  ff = (int *)(&mbl);
  mblsize = sizeof(mbl) / sizeof(int);
  for (i = 0; i < mblsize; ++i) {
    if (ff[i] != 0) {
      fprintf(dfile, "*** gbl[%d] = %d 0x%x\n", i, ff[i], ff[i]);
    }
  }
} /* dgbl */

void
dflg(void)
{
  FLG mlg;
  int *ff;
  int i, mlgsize;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  memcpy(&mlg, &flg, sizeof(flg));
  putnzint("flg.asmcode", mlg.asmcode);
  mlg.asmcode = 0;
  putnzint("flg.list", mlg.list);
  mlg.list = 0;
  putnzint("flg.object", mlg.object);
  mlg.object = 0;
  putnzint("flg.xref", mlg.xref);
  mlg.xref = 0;
  putnzint("flg.code", mlg.code);
  mlg.code = 0;
  putnzint("flg.include", mlg.include);
  mlg.include = 0;
  putnzint("flg.debug", mlg.debug);
  mlg.debug = 0;
  putnzint("flg.opt", mlg.opt);
  mlg.opt = 0;
  putnzint("flg.depchk", mlg.depchk);
  mlg.depchk = 0;
  putnzint("flg.depwarn", mlg.depwarn);
  mlg.depwarn = 0;
  putnzint("flg.dclchk", mlg.dclchk);
  mlg.dclchk = 0;
  putnzint("flg.locchk", mlg.locchk);
  mlg.locchk = 0;
  putnzint("flg.onetrip", mlg.onetrip);
  mlg.onetrip = 0;
  putnzint("flg.save", mlg.save);
  mlg.save = 0;
  putnzint("flg.inform", mlg.inform);
  mlg.inform = 0;
  putnzINT("flg.xoff", mlg.xoff);
  mlg.xoff = 0;
  putnzINT("flg.xon", mlg.xon);
  mlg.xon = 0;
  putnzint("flg.ucase", mlg.ucase);
  mlg.ucase = 0;
  putnzint("flg.dlines", mlg.dlines);
  mlg.dlines = 0;
  putnzint("flg.extend_source", mlg.extend_source);
  mlg.extend_source = 0;
  putnzint("flg.i4", mlg.i4);
  mlg.i4 = 0;
  putnzint("flg.line", mlg.line);
  mlg.line = 0;
  putnzint("flg.symbol", mlg.symbol);
  mlg.symbol = 0;
  putnzint("flg.profile", mlg.profile);
  mlg.profile = 0;
  putnzint("flg.standard", mlg.standard);
  mlg.profile = 0;
  putnzint("flg.dalign", mlg.dalign);
  mlg.dalign = 0;
  putnzint("flg.astype", mlg.astype);
  mlg.astype = 0;
  putnzint("flg.recursive", mlg.recursive);
  mlg.recursive = 0;
  putnzint("flg.ieee", mlg.ieee);
  mlg.ieee = 0;
  putnzint("flg.inliner", mlg.inliner);
  mlg.inliner = 0;
  putnzint("flg.autoinline", mlg.autoinline);
  mlg.autoinline = 0;
  putnzint("flg.vect", mlg.vect);
  mlg.vect = 0;
  putnzint("flg.endian", mlg.endian);
  mlg.endian = 0;
  putnzint("flg.terse", mlg.terse);
  mlg.terse = 0;
  putnzint("flg.dollar", mlg.dollar);
  mlg.dollar = 0;
  putnzint("flg.quad", mlg.quad);
  mlg.quad = 0;
  putnzint("flg.anno", mlg.anno);
  mlg.anno = 0;
  putnzint("flg.qa", mlg.qa);
  mlg.qa = 0;
  putnzint("flg.es", mlg.es);
  mlg.es = 0;
  putnzint("flg.p", mlg.p);
  mlg.p = 0;
  putnzint("flg.smp", mlg.smp);
  mlg.smp = 0;
  putnzint("flg.errorlimit", mlg.errorlimit);
  mlg.errorlimit = 0;
  putnzint("flg.trans_inv", mlg.trans_inv);
  mlg.trans_inv = 0;
  putnzint("flg.tpcount", mlg.tpcount);
  mlg.tpcount = 0;
  if (mlg.stdinc == (char *)0) {
    putint("flg.stdinc", 0);
    mlg.stdinc = NULL;
  } else if (mlg.stdinc == (char *)1) {
    putint("flg.stdinc", 1);
    mlg.stdinc = NULL;
  } else {
    putline();
    putstring("flg.stdinc", mlg.stdinc);
    mlg.stdinc = NULL;
  }
  putline();
  putdefarray("flg.def", mlg.def);
  mlg.def = NULL;
  putstringarray("flg.idir", mlg.idir);
  mlg.idir = NULL;
  putline();
  putintarray("flg.tpvalue", mlg.tpvalue, sizeof(mlg.tpvalue) / sizeof(int));
  putintarray("flg.dbg", mlg.dbg, sizeof(mlg.dbg) / sizeof(int));
  putintarray("flg.x", mlg.x, sizeof(mlg.x) / sizeof(int));
  putline();
  ff = (int *)(&mlg);
  mlgsize = sizeof(mlg) / sizeof(int);
  for (i = 0; i < mlgsize; ++i) {
    if (ff[i] != 0) {
      fprintf(dfile, "*** flg[%d] = %d %x\n", i, ff[i], ff[i]);
    }
  }
} /* dflg */

static bool
simpledtype(DTYPE dtype)
{
  if (dtype < DT_NONE || ((int)dtype) >= stb.dt.stg_avail)
    return false;
  if (DTY(dtype) < TY_NONE || DTY(dtype) > TY_MAX)
    return false;
  if (dlen(DTY(dtype)) == 1)
    return true;
  if (DTY(dtype) == TY_PTR)
    return true;
  return false;
} /* simpledtype */

int
putdty(TY_KIND dty)
{
  int r;
  switch (dty) {
  case TY_NONE:
    r = appendstring1("none");
    break;
  case TY_WORD:
    r = appendstring1("word");
    break;
  case TY_DWORD:
    r = appendstring1("dword");
    break;
  case TY_HOLL:
    r = appendstring1("hollerith");
    break;
  case TY_BINT:
    r = appendstring1("int*1");
    break;
  case TY_UBINT:
    r = appendstring1("uint*1");
    break;
  case TY_SINT:
    r = appendstring1("short int");
    break;
  case TY_USINT:
    r = appendstring1("unsigned short");
    break;
  case TY_INT:
    r = appendstring1("int");
    break;
  case TY_UINT:
    r = appendstring1("unsigned int");
    break;
  case TY_INT8:
    r = appendstring1("int*8");
    break;
  case TY_UINT8:
    r = appendstring1("unsigned int*8");
    break;
  case TY_INT128:
    r = appendstring1("int128");
    break;
  case TY_UINT128:
    r = appendstring1("uint128");
    break;
  case TY_128:
    r = appendstring1("ty128");
    break;
  case TY_256:
    r = appendstring1("ty256");
    break;
  case TY_512:
    r = appendstring1("ty512");
    break;
  case TY_REAL:
    r = appendstring1("real");
    break;
  case TY_FLOAT128:
    r = appendstring1("float128");
    break;
  case TY_DBLE:
    r = appendstring1("double");
    break;
  case TY_QUAD:
    r = appendstring1("quad");
    break;
  case TY_CMPLX:
    r = appendstring1("complex");
    break;
  case TY_DCMPLX:
    r = appendstring1("double complex");
    break;
  case TY_CMPLX128:
    r = appendstring1("cmplx128");
    break;
  case TY_BLOG:
    r = appendstring1("byte logical");
    break;
  case TY_SLOG:
    r = appendstring1("short logical");
    break;
  case TY_LOG:
    r = appendstring1("logical");
    break;
  case TY_LOG8:
    r = appendstring1("logical*8");
    break;
  case TY_CHAR:
    r = appendstring1("character");
    break;
  case TY_NCHAR:
    r = appendstring1("ncharacter");
    break;
  case TY_PTR:
    r = appendstring1("pointer");
    break;
  case TY_ARRAY:
    r = appendstring1("array");
    break;
  case TY_STRUCT:
    r = appendstring1("struct");
    break;
  case TY_UNION:
    r = appendstring1("union");
    break;
  case TY_NUMERIC:
    r = appendstring1("numeric");
    break;
  case TY_ANY:
    r = appendstring1("any");
    break;
  case TY_PROC:
    r = appendstring1("proc");
    break;
  case TY_VECT:
    r = appendstring1("vect");
    break;
  case TY_PFUNC:
    r = appendstring1("prototype func");
    break;
  case TY_PARAM:
    r = appendstring1("parameter");
    break;
  default:
    // Don't use a case label for TY_FLOAT, because it might alias TY_REAL.
    if (dty == TY_FLOAT) {
      r = appendstring1("float");
      break;
    }
    r = appendstring1("dty:");
    r += appendint1(dty);
    r = 0;
    break;
  }
  return r;
} /* putdty */

void
_putdtype(DTYPE dtype, int structdepth)
{
  TY_KIND dty;
  ADSC *ad;
  int numdim;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (dtype >= stb.dt.stg_avail) {
    fprintf(dfile, "\ndtype %d out of %d\n", dtype, stb.dt.stg_avail - 1);
    return;
  }
  dty = DTY(dtype);
  switch (dty) {
  default:
    putdty(dty);
    break;
  case TY_CHAR:
    appendstring1("char*");
    appendint1(DTyCharLength(dtype));
    break;
  case TY_ARRAY:
    _putdtype(DTySeqTyElement(dtype), structdepth);
    ad = AD_DPTR(dtype);
    numdim = AD_NUMDIM(ad);
    appendstring1("(");
    if (numdim >= 1 && numdim <= 7) {
      int i;
      for (i = 0; i < numdim; ++i) {
        if (i)
          appendstring1(",");
        appendsym1(AD_LWBD(ad, i));
        appendstring1(":");
        appendsym1(AD_UPBD(ad, i));
      }
    }
    appendstring1(")");
    break;
  case TY_PTR:
    if (simpledtype(DTySeqTyElement(dtype))) {
      appendstring1("*");
      _putdtype(DTySeqTyElement(dtype), structdepth);
    } else {
      appendstring1("*(");
      _putdtype(DTySeqTyElement(dtype), structdepth);
      appendstring1(")");
    }
    break;
  case TY_PARAM:
    appendstring1("(");
    _putdtype(DTyArgType(dtype), structdepth);
    if (DTyArgSym(dtype)) {
      appendstring1(" ");
      appendsym1(DTyArgSym(dtype));
    }
    if (DTyArgNext(dtype)) {
      appendstring1(", next=");
      appendint1(DTyArgNext(dtype));
    }
    appendstring1(")");
    break;
  case TY_STRUCT:
  case TY_UNION:
    if (dty == TY_STRUCT)
      appendstring1("struct");
    if (dty == TY_UNION)
      appendstring1("union");
    DTySet(dtype, -dty);
    if (DTyAlgTyTag(dtype)) {
      appendstring1(" ");
      appendsym1(DTyAlgTyTag(dtype));
    }
    if (DTyAlgTyTag(dtype) == SPTR_NULL || structdepth == 0) {
      appendstring1("{");
      if (DTyAlgTyMember(dtype)) {
        int member;
        for (member = DTyAlgTyMember(dtype); member > NOSYM && member < stb.stg_avail;) {
          _putdtype(DTYPEG(member), structdepth + 1);
          appendstring1(" ");
          appendsym1(member);
          member = SYMLKG(member);
          appendstring1(";");
        }
      }
      appendstring1("}");
    }
    DTySet(dtype, dty);
    break;
  case -TY_STRUCT:
  case -TY_UNION:
    if (dty == -TY_STRUCT)
      appendstring1("struct");
    if (dty == -TY_UNION)
      appendstring1("union");
    if (DTyAlgTyTagNeg(dtype)) {
      appendstring1(" ");
      appendsym1(DTyAlgTyTagNeg(dtype));
    } else {
      appendstring1(" ");
      appendint1(dtype);
    }
    break;
  }

} /* _putdtype */

void
putdtype(DTYPE dtype)
{
  _putdtype(dtype, 0);
} /* putdtype */

static int
putdtypex(DTYPE dtype, int len)
{
  TY_KIND dty;
  int r = 0;
  ADSC *ad;
  int numdim;
  if (len < 0)
    return 0;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (dtype >= stb.dt.stg_avail) {
    fprintf(dfile, "\ndtype %d out of %d\n", dtype, stb.dt.stg_avail - 1);
    return 0;
  }
  dty = DTY(dtype);
  switch (dty) {
  default:
    r += putdty(dty);
    break;
  case TY_CHAR:
    r += appendstring1("char*");
    r += appendint1(DTyCharLength(dtype));
    break;
  case TY_ARRAY:
    r += putdtypex(DTySeqTyElement(dtype), len - r);
    ad = AD_DPTR(dtype);
    numdim = AD_NUMDIM(ad);
    r += appendstring1("(");
    if (numdim >= 1 && numdim <= 7) {
      int i;
      for (i = 0; i < numdim && r < len; ++i) {
        if (i)
          r += appendstring1(",");
        r += appendsym1(AD_LWBD(ad, i));
        r += appendstring1(":");
        r += appendsym1(AD_UPBD(ad, i));
      }
    }
    r += appendstring1(")");
    break;
  case TY_PTR:
    if (simpledtype(DTySeqTyElement(dtype))) {
      r += appendstring1("*");
      r += putdtypex(DTySeqTyElement(dtype), len - 4);
    } else {
      r += appendstring1("*(");
      r += putdtypex(DTySeqTyElement(dtype), len - 4);
      r += appendstring1(")");
    }
    break;
  case TY_PARAM:
    r += appendstring1("(");
    r += putdtypex(DTyArgType(dtype), len - 4);
    if (DTyArgSym(dtype)) {
      r += appendstring1(" ");
      r += appendsym1(DTyArgSym(dtype));
    }
    if (DTyArgNext(dtype)) {
      r += appendstring1(", next=");
      r += appendint1(DTyArgNext(dtype));
    }
    r += appendstring1(")");
    break;
  case TY_STRUCT:
  case TY_UNION:
    if (dty == TY_STRUCT)
      r += appendstring1("struct");
    if (dty == TY_UNION)
      r += appendstring1("union");
    DTySet(dtype, -dty);
    if (DTyAlgTyTag(dtype)) {
      r += appendstring1(" ");
      r += appendsym1(DTyAlgTyTag(dtype));
    }
    r += appendstring1("{");
    if (DTyAlgTyMember(dtype)) {
      int member;
      for (member = DTyAlgTyMember(dtype);
           member > NOSYM && member < stb.stg_avail && r < len;) {
        r += putdtypex(DTYPEG(member), len - 4);
        r += appendstring1(" ");
        r += appendsym1(member);
        member = SYMLKG(member);
        r += appendstring1(";");
      }
    }
    r += appendstring1("}");
    DTySet(dtype, dty);
    break;
  case -TY_STRUCT:
  case -TY_UNION:
    if (dty == -TY_STRUCT)
      r += appendstring1("struct");
    if (dty == -TY_UNION)
      r += appendstring1("union");
    if (DTyAlgTyTagNeg(dtype)) {
      r += appendstring1(" ");
      r += appendsym1(DTyAlgTyTagNeg(dtype));
    } else {
      r += appendstring1(" ");
      r += appendint1(dtype);
    }
    break;
  }
  return r;
} /* putdtypex */

void
dumpdtype(DTYPE dtype)
{
  ADSC *ad;
  int numdim;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n");
  putint("dtype", dtype);
  if (dtype <= 0 || dtype >= stb.dt.stg_avail) {
    fprintf(dfile, "\ndtype %d out of %d\n", dtype, stb.dt.stg_avail - 1);
    return;
  }
  appendstring1(" ");
  putdty(DTY(dtype));
  switch (DTY(dtype)) {
  case TY_ARRAY:
    putint("dtype", DTySeqTyElement(dtype));
    ad = AD_DPTR(dtype);
    numdim = AD_NUMDIM(ad);
    putint("numdim", numdim);
    putnzint("scheck", AD_SCHECK(ad));
    putnsym("zbase", (SPTR) AD_ZBASE(ad)); // ???
    putnsym("numelm", AD_NUMELM(ad));
    putnsym("sdsc", AD_SDSC(ad));
    if (numdim >= 1 && numdim <= 7) {
      int i;
      for (i = 0; i < numdim; ++i) {
        putline();
        putint("dim", i);
        putint("mlpyr", AD_MLPYR(ad, i));
        putint("lwbd", AD_LWBD(ad, i));
        putint("upbd", AD_UPBD(ad, i));
      }
    }
    break;
  case TY_CHAR:
    putint("len", DTyCharLength(dtype));
    break;
  case TY_PARAM:
    putint("dtype", DTyArgType(dtype));
    putnsym("sptr", DTyArgSym(dtype));
    putint("next", DTyArgNext(dtype));
    break;
  case TY_PTR:
    putint("dtype", DTySeqTyElement(dtype));
    break;
  case TY_STRUCT:
  case TY_UNION:
    putsym("member", DTyAlgTyMember(dtype));
    putint("size", DTyAlgTySize(dtype));
    putnsym("tag", DTyAlgTyTag(dtype));
    putint("align", DTyAlgTyAlign(dtype));
    break;
  case TY_VECT:
    fprintf(dfile, "<%lu x ", DTyVecLength(dtype));
    putdtype(DTySeqTyElement(dtype));
    fputc('>', dfile);
    FLANG_FALLTHROUGH;
  default:
    /* simple datatypes, just the one line of info */
    putline();
    return;
  }
  putline();
  putdtype(dtype);
  putline();
} /* dumpdtype */

void
ddtype(DTYPE dtype)
{
  dumpdtype(dtype);
} /* ddtype */

void
dumpdtypes(void)
{
  DTYPE dtype;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** DATATYPE TABLE **********\n");
  for (dtype = (DTYPE)1; ((int)dtype) < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
    dumpdtype(dtype);
  }
  fprintf(dfile, "\n");

} /* dumpdtypes */

void
dumpnewdtypes(int olddtavail)
{
  DTYPE dtype;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** DATATYPE TABLE **********\n");
  for (dtype = (DTYPE)olddtavail; ((int)dtype) < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
    dumpdtype(dtype);
  }
  fprintf(dfile, "\n");
} /* dumpnewdtypes */

void
ddtypes(void)
{
  dumpdtypes();
} /* ddtypes */

static char prefix[1500];

static char *
smsz(int m)
{
  const char *msz = NULL;
  static char B[15];
  switch (m) {
  case MSZ_SBYTE:
    msz = "sb";
    break;
  case MSZ_SHWORD:
    msz = "sh";
    break;
  case MSZ_WORD:
    msz = "wd";
    break;
  case MSZ_SLWORD:
    msz = "sl";
    break;
  case MSZ_BYTE:
    msz = "bt";
    break;
  case MSZ_UHWORD:
    msz = "uh";
    break;
  case MSZ_PTR:
    msz = "pt";
    break;
  case MSZ_ULWORD:
    msz = "ul";
    break;
  case MSZ_F4:
    msz = "fl";
    break;
  case MSZ_F8:
    msz = "db";
    break;
#ifdef MSZ_I8
  case MSZ_I8:
    msz = "i8";
    break;
#endif
#ifdef MSZ_F10
  case MSZ_F10:
    msz = "ep";
    break;
#endif
  default:
    break;
  }
  if (msz ) {
    snprintf(B, 15, "%s", msz);
  } else {
    snprintf(B, 15, "%d", m);
  }
  return B;
} /* smsz */

static void
putstc(ILI_OP opc, int opnum, int opnd)
{
  switch (ilstckind(opc, opnum)) {
  case 1:
    putstring("cond", scond(opnd));
    break;
  case 2:
    putstring("msz", smsz(opnd));
    break;
  default:
    putint("stc", opnd);
    break;
  }
} /* putstc */

#define OT_UNARY 1
#define OT_BINARY 2
#define OT_LEAF 3

static int
optype(int opc)
{
  switch (opc) {
  case IL_INEG:
  case IL_UINEG:
  case IL_KNEG:
  case IL_UKNEG:
  case IL_SCMPLXNEG:
  case IL_DCMPLXNEG:
  case IL_FNEG:
  case IL_DNEG:
    return OT_UNARY;

  case IL_LD:
  case IL_LDKR:
  case IL_LDSP:
  case IL_LDDP:
  case IL_LDA:
  case IL_ICON:
  case IL_KCON:
  case IL_DCON:
  case IL_FCON:
  case IL_ACON:
    return OT_LEAF;
  }
  return OT_BINARY;
} /* optype */

static void _printili(int i);

static void
appendtarget(int sptr)
{
  if (sptr > 0 && sptr < stb.stg_avail) {
    appendstring1("[bih");
    appendint1(ILIBLKG(sptr));
    appendstring1("]");
  }
} /* appendtarget */

static void
_printili(int i)
{
  int n, k, j, noprs;
  ILI_OP opc;
  int o, typ;
  const char *opval;
  static const char *ccval[] = {"??",  "==",  "!=", "<",   ">=",  "<=", ">",
                          "!==", "!!=", "!<", "!>=", "!<=", "!>"};
  static const char *ccvalzero[] = {"??",   "==0",  "!=0",  "<0",   ">=0",
                              "<=0",  ">0",   "!==0", "!!=0", "!<0",
                              "!>=0", "!<=0", "!>0"};
#define NONE 0
#define UNOP 1
#define postUNOP 2
#define BINOP 3
#define INTRINSIC 4
#define MVREG 5
#define DFREG 6
#define PSCOMM 7
#define ENC_N_OP 8

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (g_dout && (i <= 0 || i >= ilib.stg_size)) {
    fprintf(dfile, "ili %d out of %d", i, ilib.stg_size - 1);
    return;
  }
  opc = ILI_OPC(i);
  if (opc <= 0 || opc >= N_ILI) {
    appendstring1("illegalopc(");
    appendint1(opc);
    appendstring1(")");
    return;
  }
  noprs = ilis[opc].oprs;
  typ = NONE;

  switch (opc) {
  case IL_IADD:
  case IL_KADD:
  case IL_UKADD:
  case IL_FADD:
  case IL_DADD:
  case IL_UIADD:
  case IL_AADD:
    opval = "+";
    typ = BINOP;
    break;
  case IL_ISUB:
  case IL_KSUB:
  case IL_UKSUB:
  case IL_FSUB:
  case IL_DSUB:
  case IL_UISUB:
  case IL_ASUB:
    opval = "-";
    typ = BINOP;
    break;
  case IL_IMUL:
  case IL_KMUL:
  case IL_UKMUL:
  case IL_FMUL:
  case IL_DMUL:
  case IL_UIMUL:
    opval = "*";
    typ = BINOP;
    break;
  case IL_DDIV:
  case IL_KDIV:
  case IL_UKDIV:
  case IL_FDIV:
  case IL_IDIV:
    opval = "/";
    typ = BINOP;
    break;
  case IL_KAND:
  case IL_AND:
    opval = "&";
    typ = BINOP;
    break;
  case IL_KOR:
  case IL_OR:
    opval = "|";
    typ = BINOP;
    break;
  case IL_KXOR:
  case IL_XOR:
    opval = "^";
    typ = BINOP;
    break;
  case IL_KMOD:
  case IL_KUMOD:
  case IL_MOD:
  case IL_UIMOD:
    opval = "%";
    typ = BINOP;
    break;
  case IL_LSHIFT:
  case IL_ULSHIFT:
    opval = "<<";
    typ = BINOP;
    break;
  case IL_RSHIFT:
  case IL_URSHIFT:
    opval = ">>";
    typ = BINOP;
    break;
  case IL_ARSHIFT:
  case IL_KARSHIFT:
    opval = "a>>";
    typ = BINOP;
    break;
  case IL_KCMP:
  case IL_UKCMP:
  case IL_ICMP:
  case IL_FCMP:
  case IL_SCMPLXCMP:
  case IL_DCMPLXCMP:
  case IL_DCMP:
  case IL_ACMP:
  case IL_UICMP:
    opval = ccval[ILI_OPND(i, 3)];
    typ = BINOP;
    break;

  case IL_INEG:
  case IL_KNEG:
  case IL_UKNEG:
  case IL_DNEG:
  case IL_UINEG:
  case IL_FNEG:
  case IL_SCMPLXNEG:
  case IL_DCMPLXNEG:
    opval = "-";
    typ = UNOP;
    break;
  case IL_NOT:
  case IL_UNOT:
  case IL_KNOT:
  case IL_UKNOT:
    opval = "!";
    typ = UNOP;
    break;
  case IL_ICMPZ:
  case IL_KCMPZ:
  case IL_UKCMPZ:
  case IL_FCMPZ:
  case IL_DCMPZ:
  case IL_ACMPZ:
  case IL_UICMPZ:
    opval = ccvalzero[ILI_OPND(i, 2)];
    typ = postUNOP;
    break;

  case IL_FMAX:
  case IL_DMAX:
  case IL_KMAX:
  case IL_UKMAX:
  case IL_IMAX:
    n = 2;
    opval = "max";
    typ = INTRINSIC;
    break;
  case IL_FMIN:
  case IL_DMIN:
  case IL_KMIN:
  case IL_UKMIN:
  case IL_IMIN:
    n = 2;
    opval = "min";
    typ = INTRINSIC;
    break;
  case IL_DBLE:
    n = 1;
    opval = "dble";
    typ = INTRINSIC;
    break;
  case IL_SNGL:
    n = 1;
    opval = "sngl";
    typ = INTRINSIC;
    break;

  case IL_FIX:
  case IL_FIXK:
  case IL_FIXUK:
    n = 1;
    opval = "fix";
    typ = INTRINSIC;
    break;
  case IL_DFIXK:
  case IL_DFIXUK:
    n = 1;
    opval = "dfix";
    typ = INTRINSIC;
    break;
  case IL_UFIX:
    n = 1;
    opval = "fix";
    typ = INTRINSIC;
    break;
  case IL_DFIX:
  case IL_DFIXU:
    n = 1;
    opval = "dfix";
    typ = INTRINSIC;
    break;
  case IL_FLOAT:
  case IL_FLOATU:
    n = 1;
    opval = "float";
    typ = INTRINSIC;
    break;
  case IL_DFLOAT:
  case IL_DFLOATU:
    n = 1;
    opval = "dfloat";
    typ = INTRINSIC;
    break;
  case IL_DNEWT:
  case IL_FNEWT:
    n = 1;
    opval = "recip";
    typ = INTRINSIC;
    break;
  case IL_DABS:
    n = 1;
    opval = "abs";
    typ = INTRINSIC;
    break;
  case IL_FABS:
    n = 1;
    opval = "abs";
    typ = INTRINSIC;
    break;
  case IL_KABS:
    n = 1;
    opval = "abs";
    typ = INTRINSIC;
    break;
  case IL_IABS:
    n = 1;
    opval = "abs";
    typ = INTRINSIC;
    break;
  case IL_FSQRT:
    n = 1;
    opval = "sqrt";
    typ = INTRINSIC;
    break;
  case IL_DSQRT:
    n = 1;
    opval = "dsqrt";
    typ = INTRINSIC;
    break;

  case IL_KCJMP:
  case IL_UKCJMP:
  case IL_ICJMP:
  case IL_FCJMP:
  case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
#endif
  case IL_ACJMP:
  case IL_UICJMP:
    _printili(ILI_OPND(i, 1));
    appendstring1(" ");
    appendstring1(ccval[ILI_OPND(i, 3)]);
    appendstring1(" ");
    _printili(ILI_OPND(i, 2));
    appendstring1(" goto ");
    if (full) {
      appendint1(ILI_OPND(i, 4));
      appendstring1("=");
    }
    appendsym1(ILI_OPND(i, 4));
    appendtarget(ILI_OPND(i, 4));
    break;
  case IL_KCJMPZ:
  case IL_UKCJMPZ:
  case IL_ICJMPZ:
  case IL_FCJMPZ:
  case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
#endif
  case IL_ACJMPZ:
  case IL_UICJMPZ:
    _printili(ILI_OPND(i, 1));
    appendstring1(" ");
    appendstring1(ccval[ILI_OPND(i, 2)]);
    appendstring1(" 0 ");
    appendstring1(" goto ");
    if (full) {
      appendint1(ILI_OPND(i, 3));
      appendstring1("=");
    }
    appendsym1(ILI_OPND(i, 3));
    appendtarget(ILI_OPND(i, 3));
    break;

  case IL_JMP:
    appendstring1("goto ");
    if (full) {
      appendint1(ILI_OPND(i, 1));
      appendstring1("=");
    }
    appendsym1(ILI_OPND(i, 1));
    appendtarget(ILI_OPND(i, 1));
    break;

  case IL_DFRKR:
  case IL_DFRIR:
  case IL_DFRSP:
  case IL_DFRDP:
  case IL_DFRCS:
  case IL_DFRCD:
  case IL_DFRAR:
    _printili(ILI_OPND(i, 1));
    break;

  case IL_QJSR:
  case IL_JSR:
    appendstring1(printname(ILI_OPND(i, 1)));
    appendstring1("(");
    j = ILI_OPND(i, 2);
    k = 0;
    while (ILI_OPC(j) != 0) {
      if (k)
        appendstring1(", ");
      switch (ILI_OPC(j)) {
      case IL_DAKR:
      case IL_DAAR:
      case IL_DADP:
#ifdef IL_DA128
      case IL_DA128:
#endif
#ifdef IL_DA256
      case IL_DA256:
#endif
      case IL_DASP:
      case IL_DAIR:
        _printili(ILI_OPND(j, 1));
        j = ILI_OPND(j, 3);
        break;
      case IL_ARGKR:
      case IL_ARGIR:
      case IL_ARGSP:
      case IL_ARGDP:
      case IL_ARGAR:
        _printili(ILI_OPND(j, 1));
        j = ILI_OPND(j, 2);
        break;
#ifdef IL_ARGRSRV
      case IL_ARGRSRV:
        appendstring1("rsrv(");
        appendint1(ILI_OPND(j, 1));
        appendstring1(")");
        j = ILI_OPND(j, 2);
        break;
#endif
      default:
        goto done;
      }
      ++k;
    }
  done:
    appendstring1(")");
    break;

  case IL_MVKR:
    opval = "MVKR";
    appendstring1(opval);
    appendstring1("(");
    appendint1(KR_MSH(ILI_OPND(i, 2)));
    appendstring1(",");
    appendint1(KR_LSH(ILI_OPND(i, 2)));
    appendstring1(")");
    _printili(ILI_OPND(i, 1));
    break;
  case IL_MVIR:
    opval = "MVIR";
    typ = MVREG;
    break;
  case IL_MVSP:
    opval = "MVSP";
    typ = MVREG;
    break;
  case IL_MVDP:
    opval = "MVDP";
    typ = MVREG;
    break;
  case IL_MVAR:
    opval = "MVAR";
    typ = MVREG;
    break;
  case IL_KRDF:
    opval = "KRDF";
    appendstring1(opval);
    appendstring1("(");
    appendint1(KR_MSH(ILI_OPND(i, 1)));
    appendstring1(",");
    appendint1(KR_LSH(ILI_OPND(i, 1)));
    appendstring1(")");
    break;
  case IL_IRDF:
    opval = "IRDF";
    typ = DFREG;
    break;
  case IL_SPDF:
    opval = "SPDF";
    typ = DFREG;
    break;
  case IL_DPDF:
    opval = "DPDF";
    typ = DFREG;
    break;
  case IL_ARDF:
    opval = "ARDF";
    typ = DFREG;
    break;
  case IL_IAMV:
  case IL_AIMV:
  case IL_KAMV:
  case IL_AKMV:
    _printili(ILI_OPND(i, 1));
    break;
  case IL_KIMV:
    appendstring1("_K2I(");
    _printili(ILI_OPND(i, 1));
    appendstring1(")");
    break;
  case IL_IKMV:
    appendstring1("_I2K(");
    _printili(ILI_OPND(i, 1));
    appendstring1(")");
    break;
  case IL_UIKMV:
    appendstring1("_UI2K(");
    _printili(ILI_OPND(i, 1));
    appendstring1(")");
    break;

  case IL_CSE:
  case IL_CSEKR:
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSEAR:
  case IL_CSECS:
  case IL_CSECD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
    appendstring1("#<");
    _printili(ILI_OPND(i, 1));
    appendstring1(">#");
    break;
  case IL_FREEKR:
    opval = "FREEKR";
    typ = PSCOMM;
    break;
  case IL_FREEDP:
    opval = "FREEDP";
    typ = PSCOMM;
    break;
  case IL_FREECS:
    opval = "FREECS";
    typ = PSCOMM;
    break;
  case IL_FREECD:
    opval = "FREECD";
    typ = PSCOMM;
    break;
  case IL_FREESP:
    opval = "FREESP";
    typ = PSCOMM;
    break;
  case IL_FREEAR:
    opval = "FREEAR";
    typ = PSCOMM;
    break;
  case IL_FREEIR:
    opval = "FREEIR";
    typ = PSCOMM;
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FREE:
    opval = "FLOAT128FREE";
    typ = PSCOMM;
    break;
#endif

  case IL_KCON:
  case IL_ICON:
  case IL_FCON:
  case IL_DCON:
    appendstring1(printname(ILI_OPND(i, 1)));
    break;

  case IL_ACON:
    j = ILI_OPND(i, 1);
    appendstring1("&");
    if (ACONOFFG(j)) {
      appendstring1("(");
    }
    if (CONVAL1G(j)) {
      appendstring1(printname(CONVAL1G(j)));
      if (CONVAL1G(j) > NOSYM && CONVAL1G(j) < stb.stg_avail &&
          SCG(CONVAL1G(j)) == SC_PRIVATE)
        appendstring1("'");
    } else {
      appendint1(CONVAL1G(j));
    }
    if (ACONOFFG(j) > 0) {
      appendstring1("+");
      appendbigint(ACONOFFG(j));
      appendstring1(")");
    } else if (ACONOFFG(j) < 0) {
      appendstring1("-");
      appendbigint(-ACONOFFG(j));
      appendstring1(")");
    }
    break;

  case IL_LD:
  case IL_LDSP:
  case IL_LDDP:
  case IL_LDKR:
  case IL_LDA:
    _printnme(ILI_OPND(i, 2));
    if (DBGBIT(10, 4096)) {
      appendstring1("<*");
      _printili(ILI_OPND(i, 1));
      appendstring1("*>");
    }
    break;

  case IL_STKR:
  case IL_ST:
  case IL_STDP:
  case IL_STSP:
  case IL_SSTS_SCALAR:
  case IL_DSTS_SCALAR:
  case IL_STA:
    _printnme(ILI_OPND(i, 3));
    if (DBGBIT(10, 4096)) {
      appendstring1("<*");
      _printili(ILI_OPND(i, 2));
      appendstring1("*>");
    }
    appendstring1(" = ");
    _printili(ILI_OPND(i, 1));
    appendstring1(";");
    break;

  case IL_LABEL: {
    int label = ILI_OPND(i, 1);
    appendstring1("label ");
    appendsym1(label);
    if (BEGINSCOPEG(label)) {
      appendstring1(" beginscope(");
      appendsym1(ENCLFUNCG(label));
      appendstring1(")");
    }
    if (ENDSCOPEG(label)) {
      appendstring1(" endscope(");
      appendsym1(ENCLFUNCG(label));
      appendstring1(")");
    }
    break;
  }

  case IL_NULL:
    if (noprs == 1 && ILI_OPND(i, 1) == 0) {
      /* expected case, print nothing else */
      appendstring1("NULL");
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    appendstring1(ilis[opc].name);
    if (noprs) {
      int j;
      appendstring1("(");
      for (j = 1; j <= noprs; ++j) {
        if (j > 1)
          appendstring1(",");
        switch (IL_OPRFLAG(opc, j)) {
#ifdef ILIO_NULL
        case ILIO_NULL:
          appendstring1("null=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_SYM
        case ILIO_SYM:
          if (full) {
            appendint1(ILI_OPND(i, j));
            appendstring1("=");
          }
          appendsym1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_OFF
        case ILIO_OFF:
          appendstring1("off=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_NME
        case ILIO_NME:
          appendstring1("nme=");
          if (full) {
            appendint1(ILI_OPND(i, j));
            appendstring1("=");
          }
          _printnme(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_IR
        case ILIO_IR:
          appendstring1("ir=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_SP
        case ILIO_SP:
          appendstring1("sp=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_DR
        case ILIO_DR:
          appendstring1("dr=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_AR
        case ILIO_AR:
          appendstring1("ar=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_KR
        case ILIO_KR:
          appendstring1("kr=");
          appendint1(ILI_OPND(i, j));
          break;
#endif
#ifdef ILIO_XMM
        case ILIO_XMM:
          /*
             bits 0:23 of the operand represent the virtual register
             number, and the value of the top byte is 1 for 'ymm'
             register, otherwise for 'xmm' register.
          */
          if (ILI_OPND(i, j) >> 24 == 1)
            appendstring1("ymm=");
          else
            appendstring1("xmm=");
          appendint1(ILI_OPND(i, j) & 0xFFFFFF);
          break;
#endif
#ifdef ILIO_LNK
        case ILIO_LNK:
#endif
#ifdef ILIO_IRLNK
        case ILIO_IRLNK:
#endif
#ifdef ILIO_SPLNK
        case ILIO_SPLNK:
#endif
#ifdef ILIO_DPLNK
        case ILIO_DPLNK:
#endif
#ifdef ILIO_ARLNK
        case ILIO_ARLNK:
#endif
#ifdef ILIO_KRLNK
        case ILIO_KRLNK:
#endif
#ifdef ILIO_CSLNK
        case ILIO_CSLNK:
#endif
#ifdef ILIO_CDLNK
        case ILIO_CDLNK:
#endif
#ifdef ILIO_QPLNK
        case ILIO_QPLNK:
#endif
#ifdef ILIO_CQLNK
        case ILIO_CQLNK:
#endif
#ifdef ILIO_128LNK
        case ILIO_128LNK:
#endif
#ifdef ILIO_256LNK
        case ILIO_256LNK:
#endif
#ifdef ILIO_512LNK
        case ILIO_512LNK:
#endif
#ifdef ILIO_X87LNK
        case ILIO_X87LNK:
#endif
#ifdef ILIO_DOUBLEDOUBLELNK
        case ILIO_DOUBLEDOUBLELNK:
#endif
          _printili(ILI_OPND(i, j));
          break;
        default:
          appendstring1("op=");
          appendint1(ILI_OPND(i, j));
          break;
        }
      }
      appendstring1(")");
    }
    break;
  }

  switch (typ) {
  case BINOP:
    o = optype(ILI_OPC(ILI_OPND(i, 1)));
    if (o != OT_UNARY && o != OT_LEAF) {
      appendstring1("(");
      _printili(ILI_OPND(i, 1));
      appendstring1(")");
    } else {
      _printili(ILI_OPND(i, 1));
    }
    appendstring1(opval);
    o = optype(ILI_OPC(ILI_OPND(i, 2)));
    if (o != OT_UNARY && o != OT_LEAF) {
      appendstring1("(");
      _printili(ILI_OPND(i, 2));
      appendstring1(")");
    } else {
      _printili(ILI_OPND(i, 2));
    }
    break;
  case UNOP:
    appendstring1(opval);
    o = optype(ILI_OPC(ILI_OPND(i, 1)));
    if (o != OT_UNARY && o != OT_LEAF) {
      appendstring1("(");
      _printili(ILI_OPND(i, 1));
      appendstring1(")");
    } else {
      _printili(ILI_OPND(i, 1));
    }
    break;
  case postUNOP:
    o = optype(ILI_OPC(ILI_OPND(i, 1)));
    if (o != OT_UNARY && o != OT_LEAF) {
      appendstring1("(");
      _printili(ILI_OPND(i, 1));
      appendstring1(")");
    } else {
      _printili(ILI_OPND(i, 1));
    }
    appendstring1(opval);
    break;
  case INTRINSIC:
    appendstring1(opval);
    appendstring1("(");
    for (j = 1; j <= n; ++j) {
      _printili(ILI_OPND(i, j));
      if (j != n)
        appendstring1(",");
    }
    appendstring1(")");
    break;
  case MVREG:
    appendstring1(opval);
    appendstring1(".");
    appendint1(ILI_OPND(i, 2));
    appendstring1("=");
    _printili(ILI_OPND(i, 1));
    break;
  case DFREG:
    appendstring1(opval);
    appendstring1("(");
    appendint1(ILI_OPND(i, 1));
    appendstring1(")");
    break;
  case PSCOMM:
    appendstring1(opval);
    appendstring1(" = ");
    _printili(ILI_OPND(i, 1));
    appendstring1(";");
    break;
  case ENC_N_OP:
    appendstring1(opval);
    appendstring1("#0x");
    appendhex1(ILI_OPND(i, n + 1));
    appendstring1("(");
    for (j = 1; j <= n; ++j) {
      _printili(ILI_OPND(i, j));
      if (j != n)
        appendstring1(",");
    }
    appendstring1(")");
    break;
  default:
    break;
  }
} /* _printili */

/*
 * call _printili with linelen = 0, so no prefix blanks are added
 */
void
printili(int i)
{
  linelen = 0;
  _printili(i);
  linelen = 0;
} /* printili */

/**
 * call _printilt with linelen = 0, so no prefix blanks are added
 */
void
printilt(int i)
{
  linelen = 0;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  snprintf(BUF, BUFSIZE, "ilt:%-4d", i);
  putit();
  if (iltb.stg_base && i > 0 && i < iltb.stg_size && ILT_ILIP(i)) {
    snprintf(BUF, BUFSIZE, "lineno:%-4d ili:%-4d  ", ILT_LINENO(i),
             ILT_ILIP(i));
    putit();
    _printili(ILT_ILIP(i));
  }
  putline();
  linelen = 0;
} /* printilt */

void
putili(const char *name, int ilix)
{
  if (ilix <= 0)
    return;
  if (full) {
    snprintf(BUF, BUFSIZE, "%s:%d=", name, ilix);
  } else {
    snprintf(BUF, BUFSIZE, "%s=", name);
  }
  putit();
  _printili(ilix);
} /* putili */

void
printblock(int block)
{
  int ilt;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  db(block);
  for (ilt = BIH_ILTFIRST(block); ilt; ilt = ILT_NEXT(ilt)) {
    if (full) {
      snprintf(BUF, BUFSIZE, "ilt:%d", ilt);
      putit();
    }
    if (ilt >= 0 && ilt < iltb.stg_size) {
      putint("lineno", ILT_LINENO(ilt));
      putili("ili", ILT_ILIP(ilt));
      putline();
    }
  }
} /* printblock */

void
printblockline(int block)
{
  int ilt;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  db(block);
  for (ilt = BIH_ILTFIRST(block); ilt; ilt = ILT_NEXT(ilt)) {
    if (full) {
      snprintf(BUF, BUFSIZE, "ilt:%d", ilt);
      putit();
    }
    if (ilt >= 0 && ilt < iltb.stg_size) {
      putint("lineno", ILT_LINENO(ilt));
      putint("findex", ILT_FINDEX(ilt));
      putili("ili", ILT_ILIP(ilt));
      putline();
    }
  }
} /* printblockline */

void
printblocks(void)
{
  int block;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  if (full) {
    fprintf(dfile, "func_count=%d, curr_func=%d=%s\n", gbl.func_count,
            GBL_CURRFUNC, GBL_CURRFUNC > 0 ? SYMNAME(GBL_CURRFUNC) : "");
  } else {
    fprintf(dfile, "func_count=%d, curr_func=%s\n", gbl.func_count,
            GBL_CURRFUNC > 0 ? SYMNAME(GBL_CURRFUNC) : "");
  }
#ifdef CUDAG
  if (GBL_CURRFUNC > 0)
    putcuda("cuda", CUDAG(GBL_CURRFUNC));
  fprintf(dfile, "\n");
#endif
  block = BIHNUMG(GBL_CURRFUNC);
  for (; block; block = BIH_NEXT(block)) {
    printblock(block);
    if (BIH_LAST(block))
      break;
    fprintf(dfile, "\n");
  }
} /* printblocks */

void
printblockt(int firstblock, int lastblock)
{
  int block, limit = 1000, b;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  fprintf(dfile, "func_count=%d, curr_func=%d=%s, blocks=%d:%d\n",
          gbl.func_count, GBL_CURRFUNC,
          GBL_CURRFUNC > 0 ? SYMNAME(GBL_CURRFUNC) : "", firstblock, lastblock);
  block = BIHNUMG(GBL_CURRFUNC);
  for (; block; block = BIH_NEXT(block)) {
    if (block == firstblock)
      break;
  }
  if (block != firstblock) {
    fprintf(dfile, "block:%d not found\n", firstblock);
    for (b = 0, block = firstblock; block && b < limit;
         block = BIH_NEXT(block), ++b) {
      printblock(block);
      if (BIH_LAST(block) || block == lastblock)
        break;
      fprintf(dfile, "\n");
    }
    if (block != lastblock)
      fprintf(dfile, "block:%d not found\n", lastblock);
  } else {
    for (b = 0; block && b < limit; block = BIH_NEXT(block), ++b) {
      printblock(block);
      if (BIH_LAST(block) || block == lastblock)
        break;
      fprintf(dfile, "\n");
    }
    if (block != lastblock)
      fprintf(dfile, "block:%d not found\n", lastblock);
  }
} /* printblockt */

void
printblocktt(int firstblock, int lastblock)
{
  int block;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  fprintf(dfile, "func_count=%d, curr_func=%d=%s, blocks=%d:%d\n",
          gbl.func_count, GBL_CURRFUNC,
          GBL_CURRFUNC > 0 ? SYMNAME(GBL_CURRFUNC) : "", firstblock, lastblock);
  for (block = firstblock; block; block = BIH_NEXT(block)) {
    printblock(block);
    if (BIH_LAST(block) || block == lastblock)
      break;
    fprintf(dfile, "\n");
  }
  if (block != lastblock) {
    fprintf(dfile, "block:%d not found\n", lastblock);
  }
} /* printblocktt */

void
printblocksline(void)
{
  int block;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  fprintf(dfile, "func_count=%d, curr_func=%d=%s\n", gbl.func_count,
          GBL_CURRFUNC, GBL_CURRFUNC > 0 ? SYMNAME(GBL_CURRFUNC) : "");
  block = BIHNUMG(GBL_CURRFUNC);
  for (; block; block = BIH_NEXT(block)) {
    printblockline(block);
    if (BIH_LAST(block))
      break;
    fprintf(dfile, "\n");
  }
} /* printblocksline */

void
dili(int ilix)
{
  ILI_OP opc;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;

  if (full)
    putint("ili", ilix);
  if (ilix <= 0 || ilix >= ilib.stg_size) {
    putstring1("out of bounds");
    putline();
    return;
  }

  opc = ILI_OPC(ilix);
  if (opc <= 0 || opc >= N_ILI) {
    putint("illegalopc", opc);
  } else {
    int noprs, j;
    static const char *iltypes[] = {"(null)",   "(arth)", "(branch)", "(cons)",
                              "(define)", "(load)", "(move)",   "(other)",
                              "(proc)",   "(store)"};
    putstring("opc", IL_NAME(opc));
    putval1(IL_TYPE(opc), iltypes, SIZEOF(iltypes));
    noprs = IL_OPRS(opc);
    for (j = 1; j <= noprs; ++j) {
      int opnd;
      opnd = ILI_OPND(ilix, j);
      switch (IL_OPRFLAG(opc, j)) {
      case ILIO_SYM:
        putsym("sym", (SPTR)opnd);
        if (opc == IL_ACON) {
          putnsym("base", SymConval1((SPTR)opnd));
          putnzbigint("offset", ACONOFFG(opnd));
        }
        break;
      case ILIO_OFF:
        putsym("sym", (SPTR)opnd);
        break;
      case ILIO_NME:
        putnme("nme", opnd);
        break;
      case ILIO_STC:
        putstc(opc, j, opnd);
        break;
      case ILIO_LNK:
        if (full) {
          putint("lnk", opnd);
        } else {
          putstring1("lnk");
        }
        break;
      case ILIO_IRLNK:
        if (full) {
          putint("irlnk", opnd);
        } else {
          putstring1("irlnk");
        }
        break;
      case ILIO_KRLNK:
        if (full) {
          putint("krlnk", opnd);
        } else {
          putstring1("krlnk");
        }
        break;
      case ILIO_ARLNK:
        if (full) {
          putint("arlnk", opnd);
        } else {
          putstring1("arlnk");
        }
        break;
      case ILIO_SPLNK:
        if (full) {
          putint("splnk", opnd);
        } else {
          putstring1("splnk");
        }
        break;
      case ILIO_DPLNK:
        if (full) {
          putint("dplnk", opnd);
        } else {
          putstring1("dplnk");
        }
        break;
#ifdef ILIO_CSLNK
      case ILIO_CSLNK:
        if (full) {
          putint("cslnk", opnd);
        } else {
          putstring1("cslnk");
        }
        break;
      case ILIO_QPLNK:
        if (full) {
          putint("qplnk", opnd);
        } else {
          putstring1("qplnk");
        }
        break;
      case ILIO_CDLNK:
        if (full) {
          putint("cdlnk", opnd);
        } else {
          putstring1("cdlnk");
        }
        break;
      case ILIO_CQLNK:
        if (full) {
          putint("cqlnk", opnd);
        } else {
          putstring1("cqlnk");
        }
        break;
      case ILIO_128LNK:
        if (full) {
          putint("128lnk", opnd);
        } else {
          putstring1("128lnk");
        }
        break;
      case ILIO_256LNK:
        if (full) {
          putint("256lnk", opnd);
        } else {
          putstring1("256lnk");
        }
        break;
      case ILIO_512LNK:
        if (full) {
          putint("512lnk", opnd);
        } else {
          putstring1("512lnk");
        }
        break;
#ifdef LONG_DOUBLE_FLOAT128
      case ILIO_FLOAT128LNK:
        if (full) {
          putint("float128lnk", opnd);
        } else {
          putstring1("float128lnk");
        }
        break;
#endif
#endif /* ILIO_CSLNK */
#ifdef ILIO_PPLNK
      case ILIO_PPLNK:
        if (full) {
          putint("pplnk", opnd);
        } else {
          putstring1("pplnk");
        }
        break;
#endif
      case ILIO_IR:
        putint("ir", opnd);
        break;
#ifdef ILIO_KR
      case ILIO_KR:
        putpint("kr", opnd);
        break;
#endif
      case ILIO_AR:
        putint("ar", opnd);
        break;
      case ILIO_SP:
        putint("sp", opnd);
        break;
      case ILIO_DP:
        putint("dp", opnd);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      /* just for debug to dump ili */
      case ILIO_QP:
        putint("qp", opnd);
        break;
#endif
      default:
        put2int("Unknown", IL_OPRFLAG(opc, j), opnd);
        break;
      }
    }
  }
  if (full) {
    putnzint("alt", ILI_ALT(ilix));
  } else {
    if (ILI_ALT(ilix)) {
      putstring1("alt");
    }
  }
  putnzint("count/rat/repl", ILI_COUNT(ilix));
  if (full)
    putnzint("hshlnk", ILI_HSHLNK(ilix));
  putnzint("visit", ILI_VISIT(ilix));
  if (full)
    putnzint("vlist", ILI_VLIST(ilix));
  putline();
} /* dili */

static void
dilitreex(int ilix, int l, int notlast)
{
  ILI_OP opc;
  int noprs, j, jj, nlinks;
  int nshift = 0;

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "%s", prefix);
  dili(ilix);
  if (ilix <= 0 || ilix >= ilib.stg_size)
    return;
  if (l) {
    if (notlast) {
      strcpy(prefix + l - 4, "|   ");
    } else {
      strcpy(prefix + l - 4, "    ");
    }
  }
  opc = ILI_OPC(ilix);
  if (opc >= 0 && opc < N_ILI) {
    noprs = IL_OPRS(opc);
  } else {
    noprs = 0;
  }
  nlinks = 0;
  for (j = 1; j <= noprs; ++j) {
    if (IL_ISLINK(opc, j)) {
      ++nlinks;
    }
  }
  if (ILI_ALT(ilix))
    ++nlinks;
  switch (opc) {
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSECS:
  case IL_CSECD:
  case IL_CSEAR:
  case IL_CSEKR:
  case IL_CSE:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
    /* don't recurse unless we're at the top level */
    if (l > 4)
      nlinks = 0;
    break;
  case IL_ACCCOPY:
  case IL_ACCCOPYIN:
  case IL_ACCCOPYOUT:
  case IL_ACCLOCAL:
  case IL_ACCCREATE:
  case IL_ACCDELETE:
  case IL_ACCPRESENT:
  case IL_ACCPCOPY:
  case IL_ACCPCOPYIN:
  case IL_ACCPCOPYOUT:
  case IL_ACCPCREATE:
  case IL_ACCPNOT:
  case IL_ACCTRIPLE:
    nshift = 1;
    break;
  default :
    break;
  }
  if (nlinks) {
    for (jj = 1; jj <= noprs; ++jj) {
      j = jj;
      if (nshift) {
        j += nshift;
        if (j > noprs)
          j -= noprs;
      }
      if (IL_ISLINK(opc, j)) {
        int opnd;
        opnd = ILI_OPND(ilix, j);
        if (ILI_OPC(opnd) != IL_NULL) {
          strcpy(prefix + l, "+-- ");
          dilitreex(opnd, l + 4, --nlinks);
        }
        prefix[l] = '\0';
      }
    }
    if (ILI_ALT(ilix) && ILI_ALT(ilix) != ilix &&
        ILI_OPC(ILI_ALT(ilix)) != IL_NULL) {
      int opnd;
      opnd = ILI_ALT(ilix);
      strcpy(prefix + l, "+-- ");
      dilitreex(opnd, l + 4, --nlinks);
      prefix[l] = '\0';
    }
  }
} /* dilitreex */

void
dilitre(int ilix)
{
  prefix[0] = ' ';
  prefix[1] = ' ';
  prefix[2] = ' ';
  prefix[3] = ' ';
  prefix[4] = '\0';
  dilitreex(ilix, 4, 0);
  prefix[0] = '\0';
} /* dilitre */

void
dilt(int ilt)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (iltb.stg_base == NULL) {
    fprintf(dfile, "iltb.stg_base not allocated\n");
    return;
  }
  if (full) {
    putint("ilt", ilt);
  } else {
    putstring1("ilt:");
  }
  if (ilt <= 0 || ilt >= iltb.stg_size) {
    fprintf(dfile, "\nilt %d out of %d\n", ilt, iltb.stg_size - 1);
    return;
  }
  if (full) {
    putnzint("ilip", ILT_ILIP(ilt));
    putnzint("prev", ILT_PREV(ilt));
    putnzint("next", ILT_NEXT(ilt));
#ifdef ILT_GUARD
    putnzint("guard", ILT_GUARD(ilt));
#endif
  }
  putnzint("lineno", ILT_LINENO(ilt));
  putnzint("findex", ILT_FINDEX(ilt));
#ifdef ILT_EXSDSCUNSAFE
  putbit("sdscunsafe", ILT_EXSDSCUNSAFE(ilt));
#endif
  putbit("st", ILT_ST(ilt));
  putbit("br", ILT_BR(ilt));
  putbit("can_throw", ILT_CAN_THROW(ilt));
  putbit("dbgline", ILT_DBGLINE(ilt));
  putbit("delete", ILT_DELETE(ilt));
  putbit("ex", ILT_EX(ilt));
  putbit("free", ILT_FREE(ilt));
  putbit("ignore", ILT_IGNORE(ilt));
  putbit("split", ILT_SPLIT(ilt));
  putbit("cplx", ILT_CPLX(ilt));
  putbit("keep", ILT_KEEP(ilt));
  putbit("mcache", ILT_MCACHE(ilt));
  putbit("nodel", ILT_NODEL(ilt));
#ifdef ILT_DELEBB
  putbit("delebb", ILT_DELEBB(ilt));
#endif
#ifdef ILT_EQASRT
  putbit("eqasrt", ILT_EQASRT(ilt));
#endif
#ifdef ILT_PREDC
  putbit("predc", ILT_PREDC(ilt));
#endif
  putline();
} /* dilt */

void
dumpilt(int ilt)
{
  dilt(ilt);
  if (ilt >= 0 && ilt < iltb.stg_size)
    dilitre(ILT_ILIP(ilt));
} /* dumpilt */

void
dumpilts()
{
  int bihx, iltx;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  bihx = BIHNUMG(gbl.currsub);
  for (; bihx; bihx = BIH_NEXT(bihx)) {
    db(bihx);
    for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = ILT_NEXT(iltx)) {
      dilt(iltx);
    }
    if (BIH_LAST(bihx))
      break;
    fprintf(dfile, "\n");
  }
} /* dumpilts */

void
db(int block)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  if (full) {
    putint("block", block);
  } else {
    putstring1("block:");
  }
  if (block <= 0 || block >= bihb.stg_size) {
    fprintf(dfile, "\nblock %d out of %d\n", block, bihb.stg_size - 1);
    return;
  }
  putnzint("lineno", BIH_LINENO(block));
  if (full)
    putnzint("iltfirst", BIH_ILTFIRST(block));
  if (full)
    putnzint("iltlast", BIH_ILTLAST(block));
  if (full)
    putnzint("prev", BIH_PREV(block));
  if (full)
    putnzint("next", BIH_NEXT(block));
  putnsym("label", BIH_LABEL(block));
  if (BIH_LABEL(block)) {
    putnzint("rfcnt", RFCNTG(BIH_LABEL(block)));
    putbit("vol", VOLG(BIH_LABEL(block)));
  }
  putnzint("assn", BIH_ASSN(block));
  putnzint("rgset", BIH_RGSET(block));
#ifdef BIH_ASM
  putbit("asm", BIH_ASM(block));
#endif
  putbit("rd", BIH_RD(block));
  putbit("ft", BIH_FT(block));
  putbit("en", BIH_EN(block));
  putbit("ex", BIH_EX(block));
#ifdef BIH_EXSDSCUNSAFE
  putbit("sdscunsafe", BIH_EXSDSCUNSAFE(block));
#endif
  putbit("last", BIH_LAST(block));
  putbit("xt", BIH_XT(block));
  putbit("pl", BIH_PL(block));
  putbit("ztrp", BIH_ZTRP(block));
  putbit("guardee", BIH_GUARDEE(block));
  putbit("guarder", BIH_GUARDER(block));
  putbit("smove", BIH_SMOVE(block));
  putbit("nobla", BIH_NOBLA(block));
  putbit("nomerge", BIH_NOMERGE(block));
  putbit("qjsr", BIH_QJSR(block));
  putbit("head", BIH_HEAD(block));
  putbit("tail", BIH_TAIL(block));
  putbit("innermost", BIH_INNERMOST(block));
  putbit("mexits", BIH_MEXITS(block));
  putbit("ozcr", BIH_OZCR(block));
  putbit("par", BIH_PAR(block));
  putbit("cs", BIH_CS(block));
  putbit("streg", BIH_STREG(block));
  putbit("vpar", BIH_VPAR(block));
  putbit("enlab", BIH_ENLAB(block));
  putbit("mark", BIH_MARK(block));
  putbit("mark2", BIH_MARK2(block));
  putbit("mark3", BIH_MARK3(block));
  putbit("parloop", BIH_PARLOOP(block));
  putbit("parsect", BIH_PARSECT(block));
  putbit("resid", BIH_RESID(block));
  putbit("ujres", BIH_UJRES(block));
  putbit("simd", BIH_SIMD(block));
  putbit("nosimd", BIH_NOSIMD(block));
  putbit("unroll", BIH_UNROLL(block));
  putbit("unroll_count", BIH_UNROLL_COUNT(block));
  putbit("nounroll", BIH_NOUNROLL(block));
  putbit("ldvol", BIH_LDVOL(block));
  putbit("stvol", BIH_STVOL(block));
  putbit("task", BIH_TASK(block));
  putbit("paraln", BIH_PARALN(block));
  putbit("invif", BIH_INVIF(block));
  putbit("noinvif", BIH_NOINVIF(block));
  putbit("combst", BIH_COMBST(block));
  putbit("deletable", BIH_DELETABLE(block));
  putbit("vcand", BIH_VCAND(block));
  putbit("accel", BIH_ACCEL(block));
  putbit("endaccel", BIH_ENDACCEL(block));
  putbit("accdata", BIH_ACCDATA(block));
  putbit("endaccdata", BIH_ENDACCDATA(block));
  putbit("kernel", BIH_KERNEL(block));
  putbit("endkernel", BIH_ENDKERNEL(block));
  putbit("midiom", BIH_MIDIOM(block));
  putbit("nodepchk", BIH_NODEPCHK(block));
  putbit("doconc", BIH_DOCONC(block));
  putline();
#ifdef BIH_FINDEX
  if (BIH_FINDEX(block) || BIH_FTAG(block)) {
    putint("findex", BIH_FINDEX(block));
    putint("ftag", BIH_FTAG(block));
    /* The casting from double to int may cause an overflow in int.
     * Just take a short-cut here for the ease of debugging. Will need
     * to create a new function to accommodate the non-int types.
     */
    if (BIH_BLKCNT(block) != -1.0)
      putdouble("blkCnt", BIH_BLKCNT(block));
    if (BIH_FINDEX(block) > 0 && BIH_FINDEX(block) < fihb.stg_avail) {
      if (FIH_DIRNAME(BIH_FINDEX(block))) {
        putstring1(FIH_DIRNAME(BIH_FINDEX(block)));
        putstring1t("/");
        putstring1t(FIH_FILENAME(BIH_FINDEX(block)));
      } else {
        putstring1(FIH_FILENAME(BIH_FINDEX(block)));
      }
      if (FIH_FUNCNAME(BIH_FINDEX(block)) != NULL) {
        putstring1(FIH_FUNCNAME(BIH_FINDEX(block)));
      }
    } else if (BIH_FINDEX(block) < 0 || BIH_FINDEX(block) >= fihb.stg_avail) {
      puterr("bad findex value");
    }
    putline();
  }
#endif
} /* db */

void
dumpblock(int block)
{
  int ilt;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  db(block);
  for (ilt = BIH_ILTFIRST(block); ilt; ilt = ILT_NEXT(ilt)) {
    dilt(ilt);
    if (ilt >= 0 && ilt < iltb.stg_size)
      dilitre(ILT_ILIP(ilt));
  }
} /* dumpblock */

void
dumptblock(const char *title, int block)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** dump block %d %s **********\n", block, title);
  dumpblock(block);
} /* dumptblock */

void
dbih(void)
{
  int block;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  block = BIHNUMG(gbl.currsub);
  for (; block; block = BIH_NEXT(block)) {
    dumpblock(block);
    if (BIH_LAST(block))
      break;
    fprintf(dfile, "\n");
  }
} /* dbih */

void
dbihonly(void)
{
  int block;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (bihb.stg_base == NULL) {
    fprintf(dfile, "bihb.stg_base not allocated\n");
    return;
  }
  block = BIHNUMG(gbl.currsub);
  for (; block; block = BIH_NEXT(block)) {
    db(block);
    if (BIH_LAST(block))
      break;
    fprintf(dfile, "\n");
  }
} /* dbihonly */

void
dumpblocksonly(void)
{
  dbihonly();
} /* dumpblocksonly */

void
dumpblocks(const char *title)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** BLOCK INFORMATION HEADER TABLE **********\n");
  fprintf(dfile, "%s called\n", title);
  dbih();
  fprintf(dfile, "%s done\n**********\n\n", title);
} /* dumpblocks */

#ifdef FIH_FULLNAME
void
dfih(int f)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (fihb.stg_base == NULL) {
    fprintf(dfile, "fihb.stg_base not allocated\n");
    return;
  }
  if (full) {
    putint("fih", f);
  } else {
    putstring1("fih:");
  }
  if (f <= 0 || f >= fihb.stg_size) {
    fprintf(dfile, "\nfile %d out of %d\n", f, fihb.stg_size - 1);
    return;
  }
  putstring("fullname", FIH_FULLNAME(f));
  if (FIH_FUNCNAME(f) != NULL && FIH_FUNCNAME(f)[0] != '\0') {
    putstring("funcname", FIH_FUNCNAME(f));
  }
  putint("functag", FIH_FUNCTAG(f));
  putint("srcline", FIH_SRCLINE(f));
  putnzint("level", FIH_LEVEL(f));
  putnzint("parent", FIH_PARENT(f));
  putnzint("lineno", FIH_LINENO(f));
  putnzint("next", FIH_NEXT(f));
  putnzint("child", FIH_CHILD(f));
  putbit("included", FIH_INC(f));
  putbit("inlined", FIH_INL(f));
  putbit("ipainlined", FIH_IPAINL(f));
  putbit("ccff", FIH_FLAGS(f) & FIH_CCFF);
  putbit("ccffinfo", (FIH_CCFFINFO(f) != NULL));
  putline();
} /* dfih */

void
dumpfile(int f)
{
  dfih(f);
} /* dumpfile */

void
dumpfiles(void)
{
  int f;
  for (f = 1; f < fihb.stg_avail; ++f) {
    dfih(f);
  }
} /* dumpfiles */
#endif

#ifdef NME_PTE
void
putptelist(int pte)
{
  for (; pte > 0; pte = (PTE_NEXT(pte) == pte ? -1 : PTE_NEXT(pte))) {
    switch (PTE_TYPE(pte)) {
    case PT_UNK:
      putstring1("unk");
      break;
    case PT_PSYM:
      putsym("psym", PTE_SPTR(pte));
      break;
    case PT_ISYM:
      putsym("isym", PTE_SPTR(pte));
      break;
    case PT_ANON:
      putint("anon", PTE_VAL(pte));
      break;
    case PT_GDYN:
      putint("gdyn", PTE_VAL(pte));
      break;
    case PT_LDYN:
      putint("ldyn", PTE_VAL(pte));
      break;
    case PT_NLOC:
      putstring1("nonlocal");
      break;
    default:
      put2int("???", PTE_TYPE(pte), PTE_VAL(pte));
      break;
    }
  }
} /* putptelist */
#endif

static void
_printnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (n < 0 || n >= nmeb.stg_avail) {
    putint("nme", n);
    return;
  }
  switch (NME_TYPE(n)) {
  case NT_VAR:
    appendstring1(printname(NME_SYM(n)));
    if (NME_SYM(n) > NOSYM && NME_SYM(n) < stb.stg_avail &&
        SCG(NME_SYM(n)) == SC_PRIVATE)
      appendstring1("'");
    break;
  case NT_MEM:
    if (NME_NM(n)) {
      _printnme(NME_NM(n));
    } else {
      appendstring1("Unknown");
    }
    if (NME_SYM(n) == 0) {
      appendstring1(".real");
    } else if (NME_SYM(n) == 1) {
      appendstring1(".imag");
    } else {
      appendstring1(".");
      appendstring1(printname(NME_SYM(n)));
    }
    break;
  case NT_ARR:
    if (NME_NM(n)) {
      _printnme(NME_NM(n));
    } else {
      appendstring1("Unknown");
    }
    if (NME_SYM(n) == 0) {
      appendstring1("[");
      appendint1(NME_CNST(n));
      appendstring1("]");
    } else if (NME_SUB(n) != 0) {
      appendstring1("[");
      _printili(NME_SUB(n));
      appendstring1("]");
    } else {
      appendstring1("[?]");
    }
    break;
  case NT_IND:
    appendstring1("*(");
    if (NME_NM(n)) {
      _printnme(NME_NM(n));
    } else {
      appendstring1("Unknown");
    }
    if (NME_SYM(n) == NME_NULL) {
    } else if (NME_SYM(n) == 0) {
      if (NME_CNST(n)) {
        appendstring1("+");
        appendint1(NME_CNST(n));
      }
    } else {
      appendstring1("+?");
    }
    appendstring1(")");
    if (NME_SUB(n) != 0) {
      appendstring1("[");
      _printili(NME_SUB(n));
      appendstring1("]");
    }
    break;
  case NT_UNK:
    if (NME_SYM(n) == 0) {
      appendstring1("unknown");
    } else if (NME_SYM(n) == 1) {
      appendstring1("volatile");
    } else {
      appendstring1("unknown:");
      appendint1(NME_SYM(n));
    }
    break;
  default:
    appendstring1("nme(");
    appendint1(n);
    appendstring1(":");
    appendint1(NME_TYPE(n));
    appendstring1(")");
    break;
  }
} /* _printnme */

void
pprintnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  putstring1("");
  _printnme(n);
} /* pprintnme */

void
printnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  linelen = 0;
  _printnme(n);
} /* printnme */

static const char *nmetypes[] = {"unknown ", "indirect", "variable",
                           "member  ", "element ", "safe    "};

void
_dumpnme(int n, bool dumpdefsuses)
{
  int pte;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (n <= 0 || n >= nmeb.stg_avail) {
    fprintf(dfile, "\nNME %d out of %d\n", n, nmeb.stg_avail - 1);
    return;
  }
  if (nmeb.stg_base == NULL) {
    fprintf(dfile, "nmeb.stg_base not allocated\n");
    return;
  }
  if (full) {
    putint("nme", n);
  } else {
    putstring1("nme");
  }
  putval1(NME_TYPE(n), nmetypes, SIZEOF(nmetypes));
  switch (NME_TYPE(n)) {
  case NT_VAR:
    putsym("var", NME_SYM(n));
    break;
  case NT_MEM:
    pprintnme(n);
    if (NME_SYM(n) == 0) {
      putstring1(".real");
    } else if (NME_SYM(n) == 1) {
      putstring1(".imag");
      break;
    } else {
      putsym("member", NME_SYM(n));
    }
    break;
  case NT_ARR:
  case NT_IND:
    pprintnme(n);
    if (NME_SYM(n) == NME_NULL) {
      putint("sym", -1);
    } else if (NME_SYM(n) == 0) {
    } else {
      if (NME_SYM(n) > 0 && NME_SYM(n) < stb.stg_avail) {
        putsym("sym", NME_SYM(n));
      } else {
        putnzint("sym", NME_SYM(n));
      }
    }
    break;
  case NT_UNK:
    pprintnme(n);
    break;
  default:
    if (NME_SYM(n) > 0 && NME_SYM(n) < stb.stg_avail) {
      putsym("sym", NME_SYM(n));
    } else {
      putnzint("sym", NME_SYM(n));
    }
    break;
  }
  putnzint("nm", NME_NM(n));
#ifdef NME_BASE
  putnzint("base", NME_BASE(n));
#endif
  putnzint("cnst", NME_CNST(n));
  putnzint("cnt", NME_CNT(n));
  if (full & 1)
    putnzint("hashlink", NME_HSHLNK(n));
  putnzint("inlarr", NME_INLARR(n));
  putnzint("rat/rfptr", NME_RAT(n));
  putnzint("stl", NME_STL(n));
  putnzint("sub", NME_SUB(n));
  putnzint("mask", NME_MASK(n));
#ifdef NME_PTE
  pte = NME_PTE(n);
  if (dumpdefsuses && pte) {
    putline();
    putstring1("pointer targets:");
    putptelist(pte);
  }
#endif
} /* _dumpnme */

void
dumpnnme(int n)
{
  linelen = 0;
  _dumpnme(n, false);
  putline();
} /* dumpnme */

void
dumpnme(int n)
{
  linelen = 0;
  _dumpnme(n, true);
  putline();
} /* dumpnme */

void
dumpnmes(void)
{
  int n;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (nmeb.stg_base == NULL) {
    fprintf(dfile, "nmeb.stg_base not allocated\n");
    return;
  }
  fprintf(dfile, "\n********** NME TABLE **********\n");
  for (n = 1; n < nmeb.stg_avail; ++n) {
    fprintf(dfile, "\n");
    dumpnme(n);
  }
  fprintf(dfile, "\n");
} /* dumpnmes */

#endif

char *
printname(int sptr)
{
  extern void cprintf(char *s, const char *format, INT *val);
  static char b[200];
  double dd;
  union {
    float ff;
    ISZ_T ww;
  } xx;

  if (sptr <= 0 || sptr >= stb.stg_avail) {
    snprintf(b, 200, "symbol %d out of %d", sptr, stb.stg_avail - 1);
    return b;
  }

  if (STYPEG(sptr) == ST_CONST) {
    INT num[2], cons1, cons2;
    int pointee;
    char *bb, *ee;
    switch (DTY(DTYPEG(sptr))) {
    case TY_INT8:
    case TY_UINT8:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      ui64toax(num, b, 22, 0, 10);
      break;
    case TY_INT:
      snprintf(b, 200, "%10d", CONVAL2G(sptr));
      break;

    case TY_FLOAT:
      xx.ww = CONVAL2G(sptr);
      if ((xx.ww & 0x7f800000) == 0x7f800000) {
        /* Infinity or NaN */
        snprintf(b, 200, ("(float)(0x%8.8" ISZ_PF "x)"), xx.ww);
      } else {
        dd = xx.ff;
        snprintf(b, 200, "%.8ef", dd);
      }
      break;

    case TY_DBLE:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      if ((num[0] & 0x7ff00000) == 0x7ff00000) {
        /* Infinity or NaN */
        snprintf(b, 200, "(double)(0x%8.8x%8.8xLL)", num[0], num[1]);
      } else {
        cprintf(b, "%.17le", num);
      }
      break;

    case TY_CMPLX:
      xx.ww = CONVAL1G(sptr);
      if ((xx.ww & 0x7f800000) == 0x7f800000) {
        /* Infinity or NaN */
        int len;
        len = snprintf(b, 200, ("(0x%8.8" ISZ_PF "x, "), xx.ww);
        bb = b + len;
      } else {
        b[0] = '(';
        sprintf(&b[1], "%17.10e", xx.ff);
        b[18] = ',';
        b[19] = ' ';
        bb = &b[20];
      }
      xx.ww = CONVAL2G(sptr);
      if ((xx.ww & 0x7f800000) == 0x7f800000) {
        snprintf(bb, 200, ("(0x%8.8" ISZ_PF "x, "), xx.ww);
      } else {
        sprintf(bb, "%17.10e", xx.ff);
        bb += 17;
        *bb++ = ')';
        *bb = '\0';
      }
      break;

    case TY_DCMPLX:
      cons1 = CONVAL1G(sptr);
      cons2 = CONVAL2G(sptr);
      num[0] = CONVAL1G(cons1);
      num[1] = CONVAL2G(cons1);
      if ((num[0] & 0x7ff00000) == 0x7ff00000) {
        /* Infinity or NaN */
        int len;
        len = snprintf(b, 200, "(0x%8.8x%8.8xLL, ", num[0], num[1]);
        bb = b + len;

      } else {
        b[0] = '(';
        cprintf(&b[1], "%24.17le", num);
        b[25] = ',';
        b[26] = ' ';
        bb = &b[27];
      }

      num[0] = CONVAL1G(cons2);
      num[1] = CONVAL2G(cons2);
      if ((num[0] & 0x7ff00000) == 0x7ff00000) {
        /* Infinity or NaN */
        snprintf(bb, 200, "0x%8.8x%8.8xLL", num[0], num[1]);
      } else {
        cprintf(bb, "%24.17le", num);
        bb += 24;
        *bb++ = ')';
        *bb = '\0';
      }

      break;

    case TY_QUAD:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      cprintf(b, "%.17le", num);
      break;

    case TY_PTR:
      pointee = CONVAL1G(sptr);
      if (pointee > 0 && pointee < stb.stg_avail && STYPEG(pointee) != ST_CONST
          ) {
        if (ACONOFFG(sptr) == 0) {
          snprintf(b, 200, "*%s", SYMNAME(pointee));
        } else {
          snprintf(b, 200, "*%s+%" ISZ_PF "d", SYMNAME(pointee),
                   ACONOFFG(sptr));
        }
      } else {
        if (ACONOFFG(sptr) == 0) {
          snprintf(b, 200, "*(sym %d)", pointee);
        } else {
          snprintf(b, 200, "*(sym %d)+%" ISZ_PF "d", pointee, ACONOFFG(sptr));
        }
      }
      break;

    case TY_WORD:
      snprintf(b, 200, "%10" ISZ_PF "d", ACONOFFG(sptr));
      break;

    case TY_CHAR:
      return stb.n_base + CONVAL1G(sptr);
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      snprintf(b, 200, "%10d", CONVAL2G(sptr));
      break;
    case TY_LOG8:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      ui64toax(num, b, 22, 0, 10);
      break;

    default:
      snprintf(b, 200, "unknown constant %d dty %d", sptr,
               DTY(DTYPEG(sptr)));
      break;
    }
    for (bb = b; *bb == ' '; ++bb)
      ;
    for (ee = bb; *ee; ++ee)
      ; /* go to end of string */
    for (; ee > bb && *(ee - 1) == ' '; --ee)
      *ee = '\0';
    return bb;
  }
  /* default case */
  if (strncmp(SYMNAME(sptr), "..inline", 8) == 0) {
    /* append symbol number to distinguish */
    snprintf(b, 200, "%s_%d", SYMNAME(sptr), sptr);
    return b;
  }
  return SYMNAME(sptr);
} /* printname */

#if DEBUG

/*
 * dump the DVL structure
 */
void
dumpdvl(int d)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (aux.dvl_base == NULL) {
    fprintf(dfile, "aux.dvl_base not allocated\n");
    return;
  }
  if (d < 0 || d >= aux.dvl_avl) {
    fprintf(dfile, "\ndvl %d out of %d\n", d, aux.dvl_avl - 1);
    return;
  }
  putint("dvl", d);
  putsym("sym", (SPTR) DVL_SPTR(d)); // ???
  putINT("conval", DVL_CONVAL(d));
  putline();
} /* dumpdvl */

void
dumpdvls()
{
  int d;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (aux.dvl_base == NULL) {
    fprintf(dfile, "aux.dvl_base not allocated\n");
    return;
  }
  for (d = 0; d < aux.dvl_avl; ++d) {
    fprintf(dfile, "\n");
    dumpdvl(d);
  }
} /* dumpdvls */

/*
 * dump variables which are kept on the stack
 */
void
stackvars()
{
  int sptr;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "Local variables:\n");
  for (sptr = gbl.locals; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    dsym(sptr);
  }
} /* stackvars */

/*
 * diagnose what stack locations are used
 */
void
stackcheck()
{
  long maxstack, minstack, i, j, addr, size, totused, totfree;
  int sptr, lastclash;
  int *stack, *stackmem;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  minstack = 0;
  maxstack = -1;
  for (sptr = gbl.locals; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    size = size_of(DTYPEG(sptr));
    if (ADDRESSG(sptr) - size < minstack)
      minstack = ADDRESSG(sptr) - size;
    if (ADDRESSG(sptr) + size > maxstack)
      maxstack = ADDRESSG(sptr) + size;
  }
  fprintf(dfile, "Stack for subprogram %d:%s\n%8ld:%-8ld\n", gbl.func_count,
          SYMNAME(gbl.currsub), minstack, maxstack);
  stackmem = (int *)malloc((maxstack - minstack + 1) * sizeof(int));
  stack = stackmem - minstack; /* minstack is <= 0 */
  for (i = minstack; i <= maxstack; ++i)
    stack[i] = 0;
  for (sptr = gbl.locals; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    addr = ADDRESSG(sptr);
    size = size_of(DTYPEG(sptr));
    lastclash = 0;
    for (i = addr; i < addr + size; ++i) {
      if (stack[i] != 0) {
        if (stack[i] != lastclash)
          fprintf(dfile, "sptr %d:%s and %d:%s clash at memory %ld\n", sptr,
                  SYMNAME(sptr), stack[i], SYMNAME(stack[i]), i);
        lastclash = stack[i];
      }
      stack[i] = sptr;
    }
  }
  sptr = -1;
  totfree = 0;
  totused = 0;
  for (i = minstack; i <= maxstack; ++i) {
    if (stack[i] == 0)
      ++totfree;
    else
      ++totused;
    if (stack[i] != sptr) {
      sptr = stack[i];
      for (j = i; j < maxstack && stack[j + 1] == sptr; ++j)
        ;
      if (sptr == 0) {
        fprintf(dfile, "%8ld:%-8ld           ---free (%ld)\n", i, j, j - i + 1);
      } else {
        size = size_of(DTYPEG(sptr));
        fprintf(dfile, "%8ld:%-8ld %8ld(%%rsp)  %5d:%s (%ld) ", i, j,
                i + 8 - minstack, sptr, SYMNAME(sptr), size);
        putdtypex(DTYPEG(sptr), 1000);
        fprintf(dfile, "\n");
      }
    }
  }
  fprintf(dfile, "%8ld used\n%8ld free\n", totused, totfree);
  free(stackmem);
} /* stackcheck */

#endif /* DEBUG */

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
      \file dump.c
      \brief dump routines for compiler's internal data structures.
      This file contains dump routines for the compiler's various internal
      data structures (e.g., symbol tables, dtype records, etc.).
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "dpm_out.h"
#include "hlvect.h"
#include "soc.h"
#include "fih.h"
#include "symutl.h"

#if DEBUG
static FILE *dfile;
static int linelen = 0;
static char BUF[900];

static int longlines = 1;

static void
putit()
{
  int l = strlen(BUF);
  if (linelen + l >= 78 && !longlines) {
    fprintf(dfile, "\n%s", BUF);
    linelen = l;
  } else if (linelen > 0) {
    fprintf(dfile, "  %s", BUF);
    linelen += l + 2;
  } else {
    fprintf(dfile, "%s", BUF);
    linelen = l;
  }
} /* putit */

static void
appendit()
{
  int l = strlen(BUF);
  fprintf(dfile, "%s", BUF);
  linelen += l;
} /* appendit */

static void
putline()
{
  if (linelen)
    fprintf(dfile, "\n");
  linelen = 0;
} /* putline */

static void
putint(const char *s, int d)
{
  sprintf(BUF, "%s:%d", s, d);
  putit();
} /* putint */

#ifdef FLANG_DUMP_UNUSED
static void
putintstar(char *s, int d, int star, char *starstring)
{
  if (star) {
    sprintf(BUF, "%s:%d%s", s, d, starstring);
  } else {
    sprintf(BUF, "%s:%d", s, d);
  }
  putit();
} /* putintstar */
#endif

static void
put2int(const char *s, int d1, int d2)
{
  sprintf(BUF, "%s(%d:%d)", s, d1, d2);
  putit();
} /* put2int */

static void
putint1(int d)
{
  sprintf(BUF, "%d", d);
  putit();
} /* putint1 */

static void
appendint1(int d)
{
  sprintf(BUF, "%d", d);
  appendit();
} /* appendint1 */

static void
put2int1(int d1, int d2)
{
  sprintf(BUF, "%d:%d", d1, d2);
  putit();
} /* put2int1 */

static void
putint1star(int d, int star, const char *starstring)
{
  if (star) {
    sprintf(BUF, "%d%s", d, starstring);
  } else {
    sprintf(BUF, "%d", d);
  }
  putit();
} /* putint1star */

static void
putnzint(const char *s, int d)
{
  if (d != 0) {
    sprintf(BUF, "%s:%d", s, d);
    putit();
  }
} /* putnzint */

static void
putnzhex(const char *s, int d)
{
  if (d != 0) {
    sprintf(BUF, "%s:0x%x", s, d);
    putit();
  }
} /* putnzhex */

static void
putnzbits(const char *s, int d)
{
  if (d != 0) {
    char *b;
    int any = 0;
    sprintf(BUF, "%s:", s);
    b = BUF + strlen(BUF);
    while (d) {
      if (d & 0x8000000) {
        *b = '1';
        ++b;
        any = 1;
      } else if (any) {
        *b = '0';
        ++b;
      }
      d = d << 1;
    }
    *b = '\0';
    putit();
  }
} /* putnzbits */

static void
putcharacter(const char *s, char c)
{
  sprintf(BUF, "%s:%c", s, c);
  putit();
} /* putchar */

static void
putedge(int d1, int d2)
{
  sprintf(BUF, "%d-->%d", d1, d2);
  putit();
} /* putedge */

static void
putsym(const char *s, int sptr)
{
  if (sptr == NOSYM) {
    sprintf(BUF, "%s:%d=%s", s, sptr, "NOSYM");
  } else if (sptr > 0 && sptr < stb.stg_avail) {
    sprintf(BUF, "%s:%d=%s", s, sptr, SYMNAME(sptr));
  } else {
    sprintf(BUF, "%s:%d", s, sptr);
  }
  putit();
} /* putsym */

static void
putsym1(int sptr)
{
  if (sptr == NOSYM) {
    sprintf(BUF, "%d=%s", sptr, "NOSYM");
  } else if (sptr > 0 && sptr < stb.stg_avail) {
    sprintf(BUF, "%d=%s", sptr, SYMNAME(sptr));
  } else {
    sprintf(BUF, "%d", sptr);
  }
  putit();
} /* putsym1 */

#ifdef FLANG_DUMP_UNUSED
static void
putintsym1(int d, int sptr)
{
  if (sptr == NOSYM) {
    sprintf(BUF, "%d:%d=%s", d, sptr, "NOSYM");
  } else if (sptr > 0 && sptr < stb.stg_avail) {
    sprintf(BUF, "%d:%d=%s", d, sptr, SYMNAME(sptr));
  } else {
    sprintf(BUF, "%d:%d", d, sptr);
  }
  putit();
} /* putintsym1 */
#endif

static void
putsymilist(int symi)
{
  for (; symi; symi = SYMI_NEXT(symi)) {
    putsym1(SYMI_SPTR(symi));
  }
} /* putsymilist */

void
dumpsymi(int symi)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (symi <= 0 || symi >= aux.symi_avl) {
    fprintf(dfile, "\nsymi %d out of %d\n", symi, aux.symi_avl);
    return;
  }
  putint("symi", symi);
  putsymilist(symi);
  putline();
} /* dumpsymi */

static void
putnsym(const char *s, int sptr)
{
  if (sptr == NOSYM) {
    sprintf(BUF, "%s:%d=%s", s, sptr, "NOSYM");
    putit();
  } else if (sptr > 0 && sptr < stb.stg_avail) {
    if (STYPEG(sptr) == ST_CONST) {
      if (DTY(DTYPEG(sptr)) == TY_CHAR) {
        sprintf(BUF, "%s:%d=%s", s, sptr, stb.n_base + CONVAL1G(sptr));
      } else {
        sprintf(BUF, "%s:%d", s, sptr);
      }
    } else {
      sprintf(BUF, "%s:%d=%s", s, sptr, SYMNAME(sptr));
    }
    putit();
  } else {
    sprintf(BUF, "%s:%d", s, sptr);
    putit();
  }
} /* putnsym */

static void
putstring(const char *s, const char *t)
{
  sprintf(BUF, "%s:%s", s, t);
  putit();
} /* putstring */

static void
putstring1(const char *t)
{
  sprintf(BUF, "%s", t);
  putit();
} /* putstring1 */

static void
appendstring1(const char *t)
{
  sprintf(BUF, "%s", t);
  appendit();
} /* appendstring1 */

static void
putsc(const char *s, int sc)
{
  if (sc >= 0 && sc <= SC_MAX) {
    sprintf(BUF, "%s:%d=%s", s, sc, stb.scnames[sc] + 3);
  } else {
    sprintf(BUF, "%s:%d", s, sc);
  }
  putit();
} /* putsc */

static void
putstype(const char *s, int stype)
{
  if (stype >= 0 && stype <= ST_MAX) {
    sprintf(BUF, "%s:%d=%s", s, stype, stb.stypes[stype]);
  } else {
    sprintf(BUF, "%s:%d", s, stype);
  }
  putit();
} /* putstype */

static void
putstype1(int stype)
{
  if (stype >= 0 && stype <= ST_MAX) {
    sprintf(BUF, "%s", stb.stypes[stype]);
  } else {
    sprintf(BUF, "stype=%d", stype);
  }
  appendit();
} /* putstype1 */

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
    if (cu & CUDA_GRID) {
      strcat(BUF, "grid");
      cu &= ~CUDA_GRID;
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
    putit();
  }
} /* putcuda */
#endif

static void
putintent(const char *s, int intent)
{
  switch (intent) {
  case 0:
    break;
  case 1:
    sprintf(BUF, "%s:in", s);
    putit();
    break;
  case 2:
    sprintf(BUF, "%s:out", s);
    putit();
    break;
  case 3:
    sprintf(BUF, "%s:inout", s);
    putit();
    break;
  default:
    sprintf(BUF, "%s:%d", s, intent);
    putit();
    break;
  }
} /* putintent */

static void
putinkind(const char *s, int k)
{
  switch (k) {
  case IK_ELEMENTAL:
    putstring(s, "elemental");
    break;
  case IK_INQUIRY:
    putstring(s, "inquiry");
    break;
  case IK_TRANSFORM:
    putstring(s, "transform");
    break;
  case IK_SUBROUTINE:
    putstring(s, "subroutine");
    break;
  default:
    putint(s, k);
    break;
  }
} /* putinkind */

static void
put_inkwd(const char *s, int k)
{
  if (k) {
    putstring(s, intrinsic_kwd[k]);
  }
} /* put_inkwd */

static void
putnname(const char *s, int off)
{
  if (off) {
    putstring(s, stb.n_base + off);
  }
} /* putnname */

static void
putdtype(const char *s, int d)
{
  if (d) {
    char typebuff[4096];
    getdtype(d, typebuff);
    sprintf(BUF, "%s:%d:%s", s, d, typebuff);
    putit();
  }
} /* putdtype */

static void
putdty(const char *s, int dty)
{
  if (dty < 0 || dty > TY_MAX || stb.tynames[dty] == NULL) {
    sprintf(BUF, "%s:<%d>", s, dty);
  } else {
    sprintf(BUF, "%s:%s", s, stb.tynames[dty]);
  }
  putit();
} /* putdty */

static void
putparam(int dpdsc, int paramct)
{
  if (paramct == 0)
    return;
  putline();
  putstring1("Parameters:");
  for (; dpdsc && paramct; ++dpdsc, --paramct) {
    int sptr = aux.dpdsc_base[dpdsc];
    if (sptr == 0) {
      putstring1("*");
    } else {
      putsym1(sptr);
    }
  }
} /* putparam */

void
putsymlk(const char *name, int list)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (list <= NOSYM)
    return;
  putline();
  putstring1(name);
  for (; list > NOSYM; list = SYMLKG(list)) {
    putsym1(list);
  }
} /* putsymlk */

void
putslnk1(int list)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (list <= NOSYM)
    return;
  for (; list > NOSYM; list = SLNKG(list)) {
    putsym1(list);
  }
} /* putslnk1 */

void
dumplists()
{
  int stype;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  for (stype = 0; stype <= ST_MAX; ++stype) {
    if (aux.list[stype] > NOSYM) {
      putstring1("aux.list[");
      putstype1(stype);
      appendstring1("] =");
      putslnk1(aux.list[stype]);
      putline();
    }
  }
} /* dumplists */

#ifdef FLANG_DUMP_UNUSED
static void
putsymflags()
{
#undef GETBIT
#define GETBIT(flg, f) flg = (flg << 1) | stb.stg_base->f
  int flag1, flag2;
  flag1 = 0;
  GETBIT(flag1, f31);
  GETBIT(flag1, f30);
  GETBIT(flag1, f29);
  GETBIT(flag1, f28);
  GETBIT(flag1, f27);
  GETBIT(flag1, f26);
  GETBIT(flag1, f25);
  GETBIT(flag1, f24);
  GETBIT(flag1, f23);
  GETBIT(flag1, f22);
  GETBIT(flag1, f21);
  GETBIT(flag1, f20);
  GETBIT(flag1, f19);
  GETBIT(flag1, f18);
  GETBIT(flag1, f17);
  GETBIT(flag1, f16);
  GETBIT(flag1, f15);
  GETBIT(flag1, f14);
  GETBIT(flag1, f13);
  GETBIT(flag1, f12);
  GETBIT(flag1, f11);
  GETBIT(flag1, f10);
  GETBIT(flag1, f9);
  GETBIT(flag1, f8);
  GETBIT(flag1, f7);
  GETBIT(flag1, f6);
  GETBIT(flag1, f5);
  GETBIT(flag1, f4);
  GETBIT(flag1, f3);
  GETBIT(flag1, f2);
  GETBIT(flag1, f1);
  flag2 = 0;
  GETBIT(flag2, f64);
  GETBIT(flag2, f63);
  GETBIT(flag2, f62);
  GETBIT(flag2, f61);
  GETBIT(flag2, f60);
  GETBIT(flag2, f59);
  GETBIT(flag2, f58);
  GETBIT(flag2, f57);
  GETBIT(flag2, f56);
  GETBIT(flag2, f55);
  GETBIT(flag2, f54);
  GETBIT(flag2, f53);
  GETBIT(flag2, f52);
  GETBIT(flag2, f51);
  GETBIT(flag2, f50);
  GETBIT(flag2, f49);
  GETBIT(flag2, f48);
  GETBIT(flag2, f47);
  GETBIT(flag2, f46);
  GETBIT(flag2, f45);
  GETBIT(flag2, f44);
  GETBIT(flag2, f43);
  GETBIT(flag2, f42);
  GETBIT(flag2, f41);
  GETBIT(flag2, f40);
  GETBIT(flag2, f39);
  GETBIT(flag2, f38);
  GETBIT(flag2, f37);
  GETBIT(flag2, f36);
  GETBIT(flag2, f35);
  GETBIT(flag2, f34);
  GETBIT(flag2, f33);
  GETBIT(flag2, f32);
#undef GETBIT
  if (flag1 || flag2) {
    sprintf(BUF, "flags=%8.8x %8.8x ", flag1, flag2);
    putit();
  }
} /* putsymflags */
#endif

static void
putbit(const char *s, int b)
{
  /* single space between flags */
  if (b) {
    int l = strlen(s);
    if (linelen + l >= 78 && !longlines) {
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
check(const char *s, int v)
{
  if (v) {
    fprintf(dfile, "*** %s: %d 0x%x\n", s, v, v);
  }
} /* check */

#ifdef FLANG_DUMP_UNUSED
static void
putmap(char *s, int m)
{
  /* single space between flags */
  switch (m) {
  case PRESCRIPTIVE:
    putstring(s, "prescriptive");
    break;
  case DESCRIPTIVE:
    putstring(s, "descriptive");
    break;
  case TRANSCRIPTIVE:
    putstring(s, "transcriptive");
    break;
  default:
    sprintf(BUF, "%s:(%d)", s, m);
    putit();
  }
} /* putmap */
#endif

void
putasttype(const char *s, int opc)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (opc < 0 || opc > AST_MAX) {
    sprintf(BUF, "%s:%d", s, opc);
  } else {
    sprintf(BUF, "%s:%d:%s", s, opc, astb.atypes[opc]);
  }
  putit();
} /* putasttype */

static void
putast(const char *s, int a)
{
  if (a == 0)
    return;
  if (a < 0 || a >= astb.stg_avail) {
    if (s && *s) {
      sprintf(BUF, "(%s=%d)", s, a);
    } else {
      sprintf(BUF, "(%d)", a);
    }
    putit();
    return;
  }
  if (linelen + 2 >= 78 && !longlines) {
    fprintf(dfile, "\n");
    linelen = 1;
  } else if (linelen > 0) {
    fprintf(dfile, " ");
    linelen += 1;
  }
  if (s && *s) {
    sprintf(BUF, "%s:%d=", s, a);
    putit();
  }
  printast(a);
} /* putast */

static void
putval(const char *s, int val, const char *values[], int sizeofvalues)
{
  if (val < 0 || val >= sizeofvalues) {
    sprintf(BUF, "%s:%d", s, val);
    putit();
  } else {
    putstring(s, values[val]);
  }
} /* putval */

#define SIZEOF(array) (sizeof(array) / sizeof(char *))

static void
putoptype(const char *s, int optype)
{
  static const char *opt[] = {
      "neg", "add",   "sub",    "mul",   "div",    "xtoi", "xtox",
      "cmp", "aif",   "ld",     "st",    "func",   "con",  "cat",
      "log", "leqv",  "lneqv",  "lor",   "land",   "eq",   "ge",
      "gt",  "le",    "lt",     "ne",    "lnot",   "loc",  "ref",
      "val", "scand", "scalar", "array", "derived"};
  putval(s, optype, opt, SIZEOF(opt));
} /* putoptype */

void
dastli(int astli)
{
  int a;
  for (a = astli; a; a = ASTLI_NEXT(a)) {
    if (a <= 0 || a > astb.astli.stg_avail) {
      sprintf(BUF, "badastli:%d", a);
      putit();
    } else {
      sprintf(BUF, "%d:%d:%d:0x%4.4x", a, ASTLI_SPTR(a), ASTLI_TRIPLE(a),
              ASTLI_FLAGS(a));
    }
  }
} /* dastli */

void
dast(int astx)
{
  int atype, dtype, asdx, j, argcnt, args, astli;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (astx < 0 || astx >= astb.stg_avail) {
    fprintf(dfile, "\nast %d out of range [0:%d)\n", astx, astb.stg_avail);
    return;
  }
  putint("ast", astx);

  BCOPY(astb.stg_base, astb.stg_base + astx, AST, 1);

  atype = A_TYPEG(0);
  putasttype("atype", atype);
  putint("hashlk/std", A_HSHLKG(0));
  putint("shape", A_SHAPEG(0));

  switch (atype) {
  case A_ID:
  case A_CNST:
  case A_BINOP:
  case A_UNOP:
  case A_CMPLXC:
  case A_CONV:
  case A_PAREN:
  case A_MEM:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_FUNC:
  case A_INTR:
  case A_INIT:
  case A_ASN:
    dtype = A_DTYPEG(0);
    putdtype("dtype", dtype);
    break;
  }
  switch (atype) {
  case A_ID:
  case A_BINOP:
  case A_UNOP:
  case A_CMPLXC:
  case A_CONV:
  case A_PAREN:
  case A_SUBSTR:
  case A_FUNC:
  case A_INTR:
    putnzint("alias", A_ALIASG(0));
    A_ALIASP(0, 0);
    putbit("callfg", A_CALLFGG(0));
    A_CALLFGP(0, 0);
    break;
  }
  putnzint("opt1", A_OPT1G(0));
  A_OPT1P(0, 0);
  putnzint("opt2", A_OPT2G(0));
  A_OPT2P(0, 0);
  switch (atype) {
  case A_NULL:
    break;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    putsym("sptr", A_SPTRG(0));
    A_SPTRP(0, 0);
    break;
  case A_CNST:
    putsym("sptr", A_SPTRG(0));
    A_SPTRP(0, 0);
    break;
  case A_BINOP:
    putoptype("optype", A_OPTYPEG(0));
    A_OPTYPEP(0, 0);
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("rop", A_ROPG(astx));
    A_LOPP(0, 0);
    break;
  case A_UNOP:
    putoptype("optype", A_OPTYPEG(astx));
    A_OPTYPEP(0, 0);
    putint("lop", A_LOPG(astx));
    A_LOPP(0, 0);
    putbit("ptr0", astx == astb.ptr0);
    putbit("ptr1", astx == astb.ptr1);
    putbit("ptr0c", astx == astb.ptr0c);
    break;
  case A_CMPLXC:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("rop", A_ROPG(astx));
    A_LOPP(0, 0);
    break;
  case A_CONV:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_PAREN:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_MEM:
    putint("parent", A_PARENTG(0));
    A_PARENTP(0, 0);
    putint("mem", A_MEMG(0));
    A_MEMP(0, 0);
    putnzint("alias", A_ALIASG(0));
    A_ALIASP(0, 0);
    break;
  case A_SUBSCR:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    asdx = A_ASDG(0);
    putint("asd", A_ASDG(0));
    A_ASDP(0, 0);
    putnzint("alias", A_ALIASG(0));
    A_ALIASP(0, 0);
    for (j = 0; j < ASD_NDIM(asdx); ++j) {
      put2int1(j, ASD_SUBS(asdx, j));
    }
    break;
  case A_SUBSTR:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("left", A_LEFTG(0));
    A_LEFTP(0, 0);
    putint("right", A_RIGHTG(0));
    A_RIGHTP(0, 0);
    break;
  case A_TRIPLE:
    putint("lbd", A_LBDG(0));
    A_LBDP(0, 0);
    putint("upbd", A_UPBDG(0));
    A_UPBDP(0, 0);
    putint("stride", A_STRIDEG(0));
    A_STRIDEP(0, 0);
    break;
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    argcnt = A_ARGCNTG(0);
    putint("argcnt", A_ARGCNTG(0));
    A_ARGCNTP(0, 0);
    args = A_ARGSG(0);
    putint("args", A_ARGSG(0));
    A_ARGSP(0, 0);
    if (atype == A_INTR || atype == A_ICALL || atype == A_INIT) {
      putoptype("optype", A_OPTYPEG(0));
      A_OPTYPEP(0, 0);
    }
    for (j = 0; j < argcnt; ++j) {
      put2int1(j, ARGT_ARG(args, j));
    }
    break;
  case A_ASN:
  case A_ASNGOTO:
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    break;
  case A_IF:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    putint("ifstmt", A_IFSTMTG(0));
    A_IFSTMTP(0, 0);
    break;
  case A_IFTHEN:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    break;
  case A_ELSE:
    break;
  case A_ELSEIF:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    break;
  case A_ENDIF:
    break;
  case A_AIF:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    putint("l1g", A_L1G(0));
    A_L1P(0, 0);
    putint("l2g", A_L2G(0));
    A_L2P(0, 0);
    putint("l3g", A_L3G(0));
    A_L3P(0, 0);
    break;
  case A_GOTO:
    putint("l1g", A_L1G(0));
    A_L1P(0, 0);
    break;
  case A_CGOTO:
  case A_AGOTO:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    astli = A_LISTG(0);
    putint("list", A_LISTG(0));
    A_LISTP(0, 0);
    dastli(astli);
    break;
  case A_DO:
    putint("dolab", A_DOLABG(0));
    A_DOLABP(0, 0);
    putint("dovar", A_DOVARG(0));
    A_DOVARP(0, 0);
    putint("m1g", A_M1G(0));
    A_M1P(0, 0);
    putint("m2g", A_M2G(0));
    A_M2P(0, 0);
    putint("m3g", A_M3G(0));
    A_M3P(0, 0);
    putint("m4g", A_M4G(0));
    A_M4P(0, 0);
    break;
  case A_DOWHILE:
    putint("dolab", A_DOLABG(0));
    A_DOLABP(0, 0);
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    break;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_ALLOC:
    putint("tok", A_TKNG(0));
    A_TKNP(0, 0);
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("m3", A_M3G(0));
    A_M3P(0, 0);
    putint("start", A_STARTG(0));
    A_STARTP(0, 0);
    putint("dtype", A_DTYPEG(0));
    A_DTYPEP(0, 0);
    putint("devsrc", A_DEVSRCG(0));
    A_DEVSRCP(0, 0);
    putint("align", A_ALIGNG(0));
    A_ALIGNP(0, 0);
    break;
  case A_WHERE:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    putint("ifstmt", A_IFSTMTG(0));
    A_IFSTMTP(0, 0);
    break;
  case A_FORALL:
    putint("ifexpr", A_IFEXPRG(0));
    A_IFEXPRP(0, 0);
    putint("ifstmt", A_IFSTMTG(0));
    A_IFSTMTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    astli = A_LISTG(0);
    putint("list", A_LISTG(0));
    A_LISTP(0, 0);
    dastli(astli);
    break;
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ELSEFORALL:
    break;
  case A_REDIM:
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    break;
  case A_COMMENT:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_INIT:
    putint("left", A_LEFTG(0));
    A_LEFTP(0, 0);
    putint("right", A_RIGHTG(0));
    A_RIGHTP(0, 0);
    putsym("sptr", A_SPTRG(0));
    A_SPTRP(0, 0);
    putoptype("optype", A_OPTYPEG(0));
    A_OPTYPEP(0, 0);
    break;
  case A_COMSTR:
    putint("comptr", A_COMPTRG(0));
    A_COMPTRP(0, 0);
    putstring1(COMSTR(astx));
    break;
  case A_REALIGN:
    putint("alndsc", A_DTYPEG(0));
    A_DTYPEP(0, 0);
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_REDISTRIBUTE:
    putint("dstdsc", A_DTYPEG(0));
    A_DTYPEP(0, 0);
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_HLOCALIZEBNDS:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("itriple", A_ITRIPLEG(0));
    A_ITRIPLEP(0, 0);
    putint("otriple", A_OTRIPLEG(0));
    A_OTRIPLEP(0, 0);
    putint("dim", A_DIMG(0));
    A_DIMP(0, 0);
    break;
  case A_HALLOBNDS:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_HCYCLICLP:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("itriple", A_ITRIPLEG(0));
    A_ITRIPLEP(0, 0);
    putint("otriple", A_OTRIPLEG(0));
    A_OTRIPLEP(0, 0);
    putint("otriple1", A_OTRIPLE1G(0));
    A_OTRIPLE1P(0, 0);
    putint("dim", A_DIMG(0));
    A_DIMP(0, 0);
    break;
  case A_HOFFSET:
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("rop", A_ROPG(astx));
    A_LOPP(0, 0);
    break;
  case A_HSECT:
    putint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("bvect", A_BVECTG(0));
    A_BVECTP(0, 0);
    break;
  case A_HCOPYSECT:
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putint("ddesc", A_DDESCG(0));
    A_DDESCP(0, 0);
    putint("sdesc", A_SDESCG(0));
    A_SDESCP(0, 0);
    break;
  case A_HPERMUTESECT:
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putint("ddesc", A_DDESCG(0));
    A_DDESCP(0, 0);
    putint("sdesc", A_SDESCG(0));
    A_SDESCP(0, 0);
    putint("bvect", A_BVECTG(0));
    A_BVECTP(0, 0);
    break;
  case A_HOVLPSHIFT:
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putint("sdesc", A_SDESCG(0));
    A_SDESCP(0, 0);
    break;
  case A_HGETSCLR:
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_HGATHER:
  case A_HSCATTER:
    putint("vsub", A_VSUBG(0));
    A_VSUBP(0, 0);
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    putint("ddesc", A_DDESCG(0));
    A_DDESCP(0, 0);
    putint("mdesc", A_MDESCG(0));
    A_MDESCP(0, 0);
    putint("sdesc", A_SDESCG(0));
    A_SDESCP(0, 0);
    putint("bvect", A_BVECTG(0));
    A_BVECTP(0, 0);
    break;
  case A_HCSTART:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putint("dest", A_DESTG(0));
    A_DESTP(0, 0);
    putint("src", A_SRCG(0));
    A_SRCP(0, 0);
    break;
  case A_HCFINISH:
  case A_HCFREE:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_MASTER:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_ENDMASTER:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    argcnt = A_ARGCNTG(0);
    putint("argcnt", A_ARGCNTG(0));
    A_ARGCNTP(0, 0);
    args = A_ARGSG(0);
    putint("args", A_ARGSG(0));
    A_ARGSP(0, 0);
    for (j = 0; j < argcnt; ++j) {
      put2int1(j, ARGT_ARG(args, j));
    }
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_SECTIONS:
  case A_MP_CANCELLATIONPOINT:
    putint("endlab", A_ENDLABG(0));
    A_ENDLABP(0, 0);
    break;
  case A_MP_CANCEL:
    putnzint("ifcancel", A_IFPARG(0));
    A_IFPARP(0, 0);
    putint("endlab", A_ENDLABG(0));
    A_ENDLABP(0, 0);
    break;
  case A_MP_PARALLEL:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("ifpar", A_IFPARG(0));
    A_IFPARP(0, 0);
    putnzint("endlab", A_ENDLABG(0));
    A_ENDLABP(0, 0);
    putnzint("procbind", A_PROCBINDG(0));
    A_PROCBINDP(0, 0);
    putnzint("num_threads", A_NPARG(0));
    A_NPARP(0, 0);
    break;
  case A_MP_TEAMS:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("nteams", A_NTEAMSG(0));
    A_NTEAMSP(0, 0);
    putnzint("thrlimit", A_THRLIMITG(0));
    A_THRLIMITP(0, 0);
    break;
  case A_MP_BMPSCOPE:
    putnzint("stblk", A_STBLKG(0));
    A_STBLKP(0, 0);
    break;
  case A_MP_TASKFIRSTPRIV:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("rop", A_ROPG(0));
    A_ROPP(0, 0);
    break;
  case A_MP_TASK:
  case A_MP_TASKLOOP:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("ifpar", A_IFPARG(0));
    A_IFPARP(0, 0);
    putnzint("final", A_FINALPARG(0));
    A_FINALPARP(0, 0);
    putbit("untied", A_UNTIEDG(0));
    A_UNTIEDP(0, 0);
    putbit("mergeable", A_MERGEABLEG(0));
    A_MERGEABLEP(0, 0);
    putbit("exeimm", A_EXEIMMG(0));
    A_EXEIMMP(0, 0);
    if (atype == A_MP_TASKLOOP) {
      putnzint("priority", A_PRIORITYG(0));
      A_PRIORITYP(0, 0);
      putbit("nogroup", A_NOGROUPG(0));
      A_NOGROUPP(0, 0);
      putbit("grainsize", A_GRAINSIZEG(0));
      A_GRAINSIZEP(0, 0);
      putbit("num_tasks", A_NUM_TASKSG(0));
      A_NUM_TASKSP(0, 0);
    }
    putnzint("endlab", A_ENDLABG(0));
    A_ENDLABP(0, 0);
    break;
  case A_MP_TARGET:
  case A_MP_TARGETDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETUPDATE:
    putnzint("if", A_IFPARG(0));
    A_IFPARP(0, 0);
    break;
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_ENDSECTIONS:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    break;
  case A_MP_PDO:
    putint("dolab", A_DOLABG(0));
    A_DOLABP(0, 0);
    putint("dovar", A_DOVARG(0));
    A_DOVARP(0, 0);
    putint("lastvar", A_LASTVALG(0));
    A_LASTVALP(0, 0);
    putint("m1g", A_M1G(0));
    A_M1P(0, 0);
    putint("m2g", A_M2G(0));
    A_M2P(0, 0);
    putint("m3g", A_M3G(0));
    A_M3P(0, 0);
    putint("m3g", A_CHUNKG(0));
    A_CHUNKP(0, 0);
    putint("sched_Type", A_SCHED_TYPEG(0));
    A_SCHED_TYPEP(0, 0);
    putbit("ordered", A_ORDEREDG(0));
    A_ORDEREDP(0, 0);
    putint("endlab", A_ENDLABG(0));
    A_ENDLABP(0, 0);
    break;
  case A_MP_TASKLOOPREG:
    putint("m1g", A_M1G(0));
    A_M1P(0, 0);
    putint("m2g", A_M2G(0));
    A_M2P(0, 0);
    putint("m3g", A_M3G(0));
    A_M3P(0, 0);
    putint("m3g", A_CHUNKG(0));
    break;
  case A_MP_ATOMICREAD:
    putnzint("src", A_SRCG(0));
    A_ROPP(0, 0);
    putbit("mem_order", A_MEM_ORDERG(0));
    A_MEM_ORDERP(0, 0);
    break;
  case A_MP_ATOMICWRITE:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("rop", A_ROPG(0));
    A_ROPP(0, 0);
    putbit("mem_order", A_MEM_ORDERG(0));
    A_MEM_ORDERP(0, 0);
    break;
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("rop", A_ROPG(0));
    A_ROPP(0, 0);
    putoptype("optype", A_OPTYPEG(0));
    A_OPTYPEP(0, 0);
    putbit("mem_order", A_MEM_ORDERG(0));
    A_MEM_ORDERP(0, 0);
    break;
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_BARRIER:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_ENDPDO:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_BCOPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_FLUSH:
  case A_MP_ENDTARGET:
  case A_MP_ENDTEAMS:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_DISTRIBUTE:
    break;
  case A_MP_PRE_TLS_COPY:
  case A_MP_COPYIN:
  case A_MP_COPYPRIVATE:
    putsym("sptr", A_SPTRG(0));
    A_SPTRP(0, 0);
    putnzint("rop", A_ROPG(0));
    A_ROPP(0, 0);
    break;
  case A_PREFETCH:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putoptype("optype", A_OPTYPEG(0));
    A_OPTYPEP(0, 0);
    break;
  case A_PRAGMA:
    putnzint("lop", A_LOPG(0));
    A_LOPP(0, 0);
    putnzint("pragmatype", A_PRAGMATYPEG(0));
    A_PRAGMATYPEP(0, 0);
    putnzint("pragmascope", A_PRAGMASCOPEG(0));
    A_PRAGMASCOPEP(0, 0);
    break;
  default:
    putbit("unknown", 1);
    break;
  }
  putline();
} /* dast */

void
dumpasts()
{
  int astx;
  for (astx = 1; astx < astb.stg_avail; ++astx) {
    dast(astx);
  }
} /* dumpasts */

void
dumpshape(int shd)
{
  int nd, ii;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (shd < 0 || shd >= astb.shd.stg_avail) {
    fprintf(dfile, "\nshd %d out of range [0:%d)\n", shd, astb.shd.stg_avail);
    return;
  }
  putint("shd", shd);
  nd = SHD_NDIM(shd);
  putint("ndim", nd);
  for (ii = 0; ii < nd; ++ii) {
    putline();
    putint("dim", ii);
    putast("lwb", SHD_LWB(shd, ii));
    putast("upb", SHD_UPB(shd, ii));
    putast("stride", SHD_STRIDE(shd, ii));
  }
  putline();
} /* dumpshape */

void
dumpshapes()
{
  int shd;
  for (shd = 1; shd < astb.shd.stg_avail;) {
    if (shd > 1) {
      fprintf(dfile, "\n");
    }
    dumpshape(shd);
    shd += 1 + SHD_NDIM(shd);
  }
} /* dumpshapes */

static char prefix[500];

static void
dastreex(int astx, int l, int notlast)
{
  int asdx, ndim, j, args, argcnt;
  if (astx == 0)
    return;
  if (l > 4)
    strcpy(prefix + l - 4, "+-- ");
  fprintf(dfile, "%s", prefix);
  dast(astx);
  if (astx <= 0 || astx >= astb.stg_avail)
    return;
  if (l) {
    if (notlast) {
      strcpy(prefix + l - 4, "|   ");
    } else {
      strcpy(prefix + l - 4, "    ");
    }
  }
  switch (A_TYPEG(astx)) {
  case A_NULL:
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_CNST:
  case A_CMPLXC:
  case A_GOTO:
  case A_CGOTO:
  case A_AGOTO:
    break;
  case A_BINOP:
    dastreex(A_LOPG(astx), l + 4, 1);
    dastreex(A_ROPG(astx), l + 4, 0);
    break;
  case A_MEM:
    dastreex(A_MEMG(astx), l + 4, 1);
    dastreex(A_PARENTG(astx), l + 4, 0);
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    dastreex(A_LOPG(astx), l + 4, 0);
    break;
  case A_SUBSCR:
    asdx = A_ASDG(astx);
    ndim = ASD_NDIM(asdx);
    dastreex(A_LOPG(astx), l + 4, ndim > 0);
    for (j = 0; j < ndim; ++j) {
      dastreex(ASD_SUBS(asdx, j), l + 4, ndim - j - 1);
    }
    break;
  case A_SUBSTR:
    dastreex(A_LEFTG(astx), l + 4, 1);
    dastreex(A_RIGHTG(astx), l + 4, 1);
    dastreex(A_LOPG(astx), l + 4, 0);
    break;
  case A_INIT:
    dastreex(A_LEFTG(astx), l + 4, 1);
    dastreex(A_RIGHTG(astx), l + 4, 0);
    break;
  case A_TRIPLE:
    dastreex(A_LBDG(astx), l + 4, 1);
    dastreex(A_UPBDG(astx), l + 4, 1);
    dastreex(A_STRIDEG(astx), l + 4, 0);
    break;
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
    argcnt = A_ARGCNTG(astx);
    args = A_ARGSG(astx);
    dastreex(A_LOPG(astx), l + 4, argcnt > 0);
    for (j = 0; j < argcnt; ++j) {
      dastreex(ARGT_ARG(args, j), l + 4, argcnt - j - 1);
    }
    break;
  case A_ASN:
  case A_ASNGOTO:
    dastreex(A_DESTG(astx), l + 4, 1);
    dastreex(A_SRCG(astx), l + 4, 0);
    break;
  case A_IF:
    dastreex(A_IFEXPRG(astx), l + 4, 1);
    dastreex(A_IFSTMTG(astx), l + 4, 0);
    break;
  case A_IFTHEN:
    dastreex(A_IFEXPRG(astx), l + 4, 0);
    break;
  case A_ELSE:
    break;
  case A_ELSEIF:
    dastreex(A_IFEXPRG(astx), l + 4, 0);
    break;
  case A_ENDIF:
    break;
  case A_AIF:
    dastreex(A_IFEXPRG(astx), l + 4, 0);
    break;
  case A_DO:
    dastreex(A_M1G(astx), l + 4, 1);
    dastreex(A_M2G(astx), l + 4, 1);
    dastreex(A_M3G(astx), l + 4, 0);
    dastreex(A_M4G(astx), l + 4, 0);
    break;
  case A_DOWHILE:
    dastreex(A_IFEXPRG(astx), l + 4, 0);
    break;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    dastreex(A_LOPG(astx), l + 4, 0);
    break;
  case A_ALLOC:
    dastreex(A_LOPG(astx), l + 4, 0);
    dastreex(A_SRCG(astx), l + 4, 0);
    dastreex(A_DESTG(astx), l + 4, 0);
    dastreex(A_M3G(astx), l + 4, 0);
    dastreex(A_STARTG(astx), l + 4, 0);
    dastreex(A_DTYPEG(astx), l + 4, 0);
    break;
  case A_WHERE:
    dastreex(A_IFEXPRG(astx), l + 4, 1);
    dastreex(A_IFSTMTG(astx), l + 4, 0);
    break;
  case A_FORALL:
    break;
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ELSEFORALL:
    break;
  case A_REDIM:
    break;
  case A_COMMENT:
  case A_COMSTR:
    break;
  case A_REALIGN:
  case A_REDISTRIBUTE:
    break;
  case A_HLOCALIZEBNDS:
    break;
  case A_HALLOBNDS:
    break;
  case A_HCYCLICLP:
    break;
  case A_HOFFSET:
    break;
  case A_HSECT:
    break;
  case A_HCOPYSECT:
    break;
  case A_HPERMUTESECT:
    break;
  case A_HOVLPSHIFT:
    break;
  case A_HGETSCLR:
    dastreex(A_DESTG(astx), l + 4, 1);
    dastreex(A_SRCG(astx), l + 4, A_LOPG(astx));
    if (A_LOPG(astx)) {
      dastreex(A_LOPG(astx), l + 4, 0);
    }
    break;
  case A_HGATHER:
  case A_HSCATTER:
    break;
  case A_HCSTART:
    break;
  case A_HCFINISH:
  case A_HCFREE:
    break;
  case A_MASTER:
    break;
  case A_ENDMASTER:
    argcnt = A_ARGCNTG(astx);
    args = A_ARGSG(astx);
    for (j = 0; j < argcnt; ++j) {
      dastreex(ARGT_ARG(args, j), l + 4, argcnt - j - 1);
    }
    break;
  case A_ATOMIC:
  case A_PREFETCH:
  case A_PRAGMA:
    dastreex(A_LOPG(astx), l + 4, 0);
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_TARGET:
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_PARALLEL:
    dastreex(A_IFPARG(astx), l + 4, 0);
    break;
  case A_MP_TEAMS:
    dastreex(A_NTEAMSG(astx), l + 4, 0);
    dastreex(A_THRLIMITG(astx), l + 4, 0);
    break;
  case A_MP_BMPSCOPE:
    dastreex(A_STBLKG(astx), l + 4, 0);
    break;
  case A_MP_TASK:
  case A_MP_TASKLOOP:
    dastreex(A_IFPARG(astx), l + 4, 0);
    break;
  case A_MP_TASKFIRSTPRIV:
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_BARRIER:
  case A_MP_ETASKDUP:
  case A_MP_ETASKLOOPREG:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_EMPSCOPE:
  case A_MP_ENDTARGET:
  case A_MP_ENDTEAMS:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_ENDTARGETDATA:
  case A_MP_TASKDUP:
    break;
  case A_MP_ATOMICREAD:
    dastreex(A_SRCG(astx), l + 4, 0);
    FLANG_FALLTHROUGH;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    dastreex(A_LOPG(astx), l + 4, 0);
    dastreex(A_ROPG(astx), l + 4, 0);
    break;
  case A_MP_TASKREG:
    dastreex(A_ENDLABG(astx), l + 4, 0);
    break;
  case A_MP_TASKLOOPREG:
    dastreex(A_M1G(astx), l + 4, 1);
    dastreex(A_M2G(astx), l + 4, 1);
    dastreex(A_M3G(astx), l + 4, 1);
    break;
  case A_MP_CANCEL:
    dastreex(A_IFPARG(astx), l + 4, 0);
    FLANG_FALLTHROUGH;
  case A_MP_SECTIONS:
  case A_MP_CANCELLATIONPOINT:
    dastreex(A_ENDLABG(astx), l + 4, 0);
    break;
  case A_MP_PDO:
    dastreex(A_M1G(astx), l + 4, 1);
    dastreex(A_M2G(astx), l + 4, 1);
    dastreex(A_M3G(astx), l + 4, 1);
    dastreex(A_CHUNKG(astx), l + 4, 0);
    break;
  case A_MP_ENDPDO:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_MAP:
  case A_MP_EMAP:
  case A_MP_TARGETLOOPTRIPCOUNT:
  case A_MP_DISTRIBUTE:
  case A_MP_EREDUCTION:
  case A_MP_BREDUCTION:
  case A_MP_REDUCTIONITEM:
    break;
  default:
    fprintf(gbl.dbgfil, "NO DUMP AVL");
    break;
  }
  prefix[l - 4] = '\0';
} /* dastreex */

void
dastree(int astx)
{
  int savelonglines;
  savelonglines = longlines;
  longlines = 1;
  prefix[0] = ' ';
  prefix[1] = ' ';
  prefix[2] = ' ';
  prefix[3] = ' ';
  prefix[4] = '\0';
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  dastreex(astx, 4, 0);
  prefix[0] = '\0';
  longlines = savelonglines;
} /* dastree */

void
past(int astx)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  printast(astx);
  fprintf(dfile, "\n");
} /* past */

static void
putflags(const char *s, int flags)
{
  if (flags) {
    sprintf(BUF, "%s=%8.8x ", s, flags);
    putit();
  }
} /* putflags */

void
dumpfnode(int v)
{
  PSI_P succ, pred;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (opt.fgb.stg_base == NULL) {
    fprintf(dfile, "OPT.FGB.STG not allocated\n");
    return;
  }
  if (v < 0 || v > opt.num_nodes) {
    fprintf(dfile, "Flow graph node %d out of %d\n", v, opt.num_nodes);
    return;
  }
  putint("fnode", v);
  putint("lineno", FG_LINENO(v));
  putnsym("label", FG_LABEL(v));
  putnzint("lprev", FG_LPREV(v));
  putnzint("lnext", FG_LNEXT(v));
  putnzint("first", FG_STDFIRST(v));
  putnzint("last", FG_STDLAST(v));
  putnzint("atomic", FG_ATOMIC(v));
  putnzint("dfn", FG_DFN(v));
  putnzint("dom", FG_DOM(v));
  putnzint("pdom", FG_PDOM(v));
  putnzint("loop", FG_LOOP(v));
  putnzint("next", FG_NEXT(v));
  putnzint("natnxt", FG_NATNXT(v));
  putnzint("par", FG_PAR(v));
  putnzint("parloop", FG_PARLOOP(v));
  putline();
  pred = FG_PRED(v);
  if (pred == NULL) {
    putstring("pred", "none");
  } else {
    putstring1("pred:");
    for (; pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
      putint1star(PSI_NODE(pred), PSI_FT(pred), "(ft)");
    }
  }
  succ = FG_SUCC(v);
  if (succ == NULL) {
    putstring("succ", "none");
  } else {
    putstring1("succ:");
    for (; succ != PSI_P_NULL; succ = PSI_NEXT(succ)) {
      putint1star(PSI_NODE(succ), PSI_FT(succ), "(ft)");
    }
  }
  putline();
  putbit("call", FG_EX(v));
  putbit("cs", FG_CS(v));
  putbit("ctlequiv", FG_CTLEQUIV(v));
  putbit("entry", FG_EN(v));
  putbit("exit", FG_XT(v));
  putbit("fallthru", FG_FT(v));
  putbit("head", FG_HEAD(v));
  putbit("innermost", FG_INNERMOST(v));
  putbit("jumptable", FG_JMP_TBL(v));
  putbit("master", FG_MASTER(v));
  putbit("mexits", FG_MEXITS(v));
  putbit("parsect", FG_PARSECT(v));
  putbit("task", FG_TASK(v));
  putbit("ptrstore", FG_PTR_STORE(v));
  putbit("tail", FG_TAIL(v));
  putbit("zerotrip", FG_ZTRP(v));
  putline();
} /* dumpfnode */

void
dumpfg(int v)
{
  dumpfnode(v);
} /* dumpfg */

void
dfg(int v)
{
  dumpfnode(v);
} /* dfg */

void
dumpfgraph()
{
  int v;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (opt.fgb.stg_base == NULL) {
    fprintf(dfile, "OPT.FGB.STG not allocated\n");
    return;
  }
  for (v = 1; v <= opt.num_nodes; ++v) {
    if (v > 1)
      fprintf(dfile, "\n");
    dumpfnode(v);
  }

  fprintf(dfile, "\nDepth First Order:\n");
  for (v = 1; v <= opt.dfn; ++v) {
    putint1(VTX_NODE(v));
  }
  putline();

  fprintf(dfile, "\nRetreating Edges:\n");
  for (v = 0; v < opt.rteb.stg_avail; ++v) {
    putedge(EDGE_PRED(v), EDGE_SUCC(v));
  }
  putline();
} /* dumpfgraph */

void
putnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (n < 0 || n >= nmeb.stg_avail) {
    putint("nme", n);
    return;
  }
  switch (NME_TYPE(n)) {
  case NT_VAR:
    putstring1(getprint(NME_SYM(n)));
    break;
  case NT_MEM:
    putnme(NME_NM(n));
    if (NME_SYM(n) == 0) {
      appendstring1(".real");
      break;
    }
    if (NME_SYM(n) == 1) {
      appendstring1(".imag");
      break;
    }
    appendstring1(".");
    appendstring1(getprint(NME_SYM(n)));
    break;
  default:
    put2int("nme", n, NME_TYPE(n));
    break;
  }
} /* putnme */

void
putnmetype(const char *s, int t)
{
  switch (t) {
  case NT_ADD:
    putstring(s, "add");
    break;
  case NT_UNK: /* unknown */
    putstring(s, "unk");
    break;
  case NT_IND: /* Indirect ref e.g. *p      */
    putstring(s, "ind");
    break;
  case NT_VAR: /* Variable ref. (struct, array or scalar) */
    putstring(s, "var");
    break;
  case NT_MEM: /* Structure member ref. */
    putstring(s, "mem");
    break;
  case NT_ARR: /* Array element ref. */
    putstring(s, "arr");
    break;
  case NT_SAFE: /* special names; does not conflict with preceding refs */
    putstring(s, "safe");
    break;
  default:
    putint(s, t);
    break;
  }
} /* putnmetype */

char *
printname(int sptr)
{
  static char b[200];

  if (sptr <= 0 || sptr >= stb.stg_avail) {
    sprintf(b, "symbol %d out of %d", sptr, stb.stg_avail - 1);
    return b;
  }

  if (STYPEG(sptr) == ST_CONST) {
    INT num[2];
    int pointee;
    char *bb;
    switch (DTY(DTYPEG(sptr))) {
    case TY_INT8:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      ui64toax(num, b, 22, 0, 10);
      break;
    case TY_INT:
      sprintf(b, "%10d", CONVAL2G(sptr));
      break;

    case TY_FLOAT:
      num[0] = CONVAL2G(sptr);
      cprintf(b, "%.8e", num);
      break;

    case TY_QUAD:
    case TY_DBLE:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      cprintf(b, "%.17le", num);
      break;

    case TY_PTR:
      pointee = CONVAL1G(sptr);
      if (pointee > 0 && pointee < stb.stg_avail && STYPEG(pointee) != ST_CONST) {
        if (CONVAL2G(sptr) == 0) {
          sprintf(b, "*%s", SYMNAME(pointee));
        } else {
          sprintf(b, "*%s+%d", SYMNAME(pointee), CONVAL2G(sptr));
        }
      } else {
        if (CONVAL2G(sptr) == 0) {
          sprintf(b, "*(sym %d)", pointee);
        } else {
          sprintf(b, "*(sym %d)+%d", pointee, CONVAL2G(sptr));
        }
      }
      break;

    case TY_WORD:
      sprintf(b, "%10d", CONVAL2G(sptr));
      break;

    case TY_CHAR:
      return stb.n_base + CONVAL1G(sptr);
      break;

    default:
      sprintf(b, "unknown constant %d dty %" ISZ_PF "d", sptr,
              DTY(DTYPEG(sptr)));
      break;
    }
    for (bb = b; *bb == ' '; ++bb)
      ;
    return bb;
  }
  /* default case */
  if (strncmp(SYMNAME(sptr), "..inline", 8) == 0) {
    /* append symbol number to distinguish */
    sprintf(b, "%s_%d", SYMNAME(sptr), sptr);
    return b;
  }
  return SYMNAME(sptr);
} /* printname */

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
    } else
    {
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
      int asdx, i, ndim;
      appendstring1("[");
      asdx = NME_SUB(n);
      ndim = ASD_NDIM(asdx);
      for (i = 0; i < ndim; ++i) {
        int d;
        d = ASD_SUBS(n, i);
        printast(d);
        if (i)
          appendstring1(",");
      }
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
printnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  putstring1("");
  _printnme(n);
} /* printnme */

void
dumpnme(int n)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (nmeb.stg_base == NULL) {
    fprintf(dfile, "NME not allocated\n");
    return;
  }
  if (n < 0 || n > nmeb.stg_avail) {
    fprintf(dfile, "NME %d out of %d\n", n, nmeb.stg_avail);
    return;
  }
  putint("nme", n);
  putnmetype("type", NME_TYPE(n));
  putnzint("inlarr", NME_INLARR(n));
  putnsym("sym", NME_SYM(n));
  putnzint("nm", NME_NM(n));
  putnzint("hshlnk", NME_HSHLNK(n));
  putnzint("rfptr", NME_RFPTR(n));
  putnzint("cnst", NME_CNST(n));
  putnzint("sub", NME_SUB(n));
  putnzint("stl", NME_STL(n));
  putnzint("cnt", NME_CNT(n));
  putline();
  if (NME_DEF(n)) {
    int d;
    putstring1(" defs:");
    for (d = NME_DEF(n); d; d = DEF_NEXT(d)) {
      putint1(d);
    }
    putline();
  }
} /* dumpnme */

void
dumpnmes()
{
  int n;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (nmeb.stg_base == NULL) {
    fprintf(dfile, "NME not allocated\n");
    return;
  }
  for (n = 0; n < nmeb.stg_avail; ++n) {
    if (n)
      fprintf(dfile, "\n");
    dumpnme(n);
  }
} /* dumpnmes */

void
dumploop(int l)
{
  PSI_P p;
  Q_ITEM *q;
  int v;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (opt.lpb.stg_base == NULL) {
    fprintf(dfile, "opt.lpb not allocated\n");
    return;
  }
  if (l < 0 || l > opt.nloops) {
    fprintf(dfile, "Loop %d out of %d\n", l, opt.nloops);
    return;
  }
  putint("loop", l);
  putint("level", LP_LEVEL(l));
  putnzint("parent", LP_PARENT(l));
  putint("head", LP_HEAD(l));
  putint("tail", LP_TAIL(l));
  putnzint("child", LP_CHILD(l));
  putnzint("sibling", LP_SIBLING(l));
  put2int("lines", BIH_LINENO(FG_TO_BIH(LP_HEAD(l))),
          BIH_LINENO(FG_TO_BIH(LP_TAIL(l))));
  putnzint("parloop", LP_PARLOOP(l));
  putline();
  putstring1("nodes:");
  for (v = LP_FG(l); v; v = FG_NEXT(v)) {
    putint1(v);
  }
  p = LP_EXITS(l);
  if (p == NULL) {
    putstring("exits", "none");
  } else {
    putline();
    putstring1(" exits:");
    for (; p != PSI_P_NULL; p = PSI_NEXT(p)) {
      putint1(PSI_NODE(p));
    }
  }
  q = LP_STL_PAR(l);
  if (q) {
    putline();
    putstring1(" stl_par:");
    for (; q; q = q->next) {
      putnme(q->info);
    }
  }

  putline();
  putbit("callfg", LP_CALLFG(l));
  putbit("callinternal", LP_CALLINTERNAL(l));
#ifdef LP_CNCALL
  putbit("cncall", LP_CNCALL(l));
#endif
  putbit("cs", LP_CS(l));
  putbit("csect", LP_CSECT(l));
#ifdef LP_EXT_STORE
  putbit("ext_store", LP_EXT_STORE(l));
#endif
  putbit("forall", LP_FORALL(l));
  putbit("innermost", LP_INNERMOST(l));
#ifdef LP_INVARIF
  putbit("invarif", LP_INVARIF(l));
#endif
  putbit("jmp_tbl", LP_JMP_TBL(l));
  putbit("mark", LP_MARK(l));
  putbit("master", LP_MASTER(l));
  putbit("mexits", LP_MEXITS(l));
  putbit("nobla", LP_NOBLA(l));
  putbit("parregn", LP_PARREGN(l));
  putbit("parsect", LP_PARSECT(l));
#ifdef LP_PTR_LOAD
  putbit("ptr_load", LP_PTR_LOAD(l));
#endif
#ifdef LP_PTR_STORE
  putbit("ptr_store", LP_PTR_STORE(l));
#endif
  putbit("qjsr", LP_QJSR(l));
#ifdef LP_SMOVE
  putbit("smove", LP_SMOVE(l));
#endif
#ifdef LP_XTNDRNG
  putbit("xtndrng", LP_XTNDRNG(l));
#endif
#ifdef LP_ZEROTRIP
  putbit("zerotrip", LP_ZEROTRIP(l));
#endif
  putline();
} /* dumploop */

void
dumploops()
{
  int l;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (opt.lpb.stg_base == NULL) {
    fprintf(dfile, "opt.lpb not allocated\n");
    return;
  }
  dumploop(0);
  fprintf(dfile, "\n");
  for (l = opt.nloops; l; --l) {
    dumploop(LP_LOOP(l));
    fprintf(dfile, "\n");
  }
} /* dumploops */

void
dstd(int stdx)
{
  int astx;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (stdx < 0 || stdx >= astb.std.stg_avail) {
    fprintf(dfile, "\nstd %d out of range [0:%d)\n", stdx, astb.std.stg_avail);
    return;
  }
  astx = STD_AST(stdx);
  putint("std", stdx);
  putint("ast", astx);
  putnzint("lineno", STD_LINENO(stdx));
  putnsym("label", STD_LABEL(stdx));
  if (STD_BLKSYM(stdx) != SPTR_NULL)
    putnsym("blksym", STD_BLKSYM(stdx));
  putint("prev", STD_PREV(stdx));
  putint("next", STD_NEXT(stdx));
#ifdef STD_TAG
  putnzint("tag", STD_TAG(stdx));
#endif
  putnzint("fg", STD_FG(stdx));
  putflags("flags", STD_FLAGS(stdx));
  putbit("ex", STD_EX(stdx));
  putbit("st", STD_ST(stdx));
  putbit("br", STD_BR(stdx));
  putbit("delete", STD_DELETE(stdx));
  putbit("ignore", STD_IGNORE(stdx));
  putbit("split/moved", STD_SPLIT(stdx));
  putbit("minfo", STD_MINFO(stdx));
  putbit("local", STD_LOCAL(stdx));
  putbit("pure", STD_PURE(stdx));
  putbit("par", STD_PAR(stdx));
  putbit("cs", STD_CS(stdx));
  putbit("accel", STD_ACCEL(stdx));
  putbit("rescope", STD_RESCOPE(stdx));
  putbit("indiv", STD_INDIVISIBLE(stdx));
  putbit("atomic", STD_ATOMIC(stdx));
  putbit("kernel", STD_KERNEL(stdx));
  putbit("task", STD_TASK(stdx));
  putbit("orig", STD_ORIG(stdx));
  putbit("parsect", STD_PARSECT(stdx));
  putbit("mark", STD_MARK(stdx));
  putbit("ast.callfg", A_CALLFGG(astx));
  if (astx) {
    putstring("atype", astb.atypes[A_TYPEG(astx)]);
  }
  putline();
  if (STD_LABEL(stdx)) {
    putstring(SYMNAME(STD_LABEL(stdx)), "");
    putline();
  }
  dbg_print_ast(astx, dfile);
} /* dstd */

void
dsstd(int stdx)
{
  int astx;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (stdx < 0 || stdx >= astb.std.stg_avail) {
    fprintf(dfile, "\nstd %d out of range [0:%d)\n", stdx, astb.std.stg_avail);
    return;
  }
  astx = STD_AST(stdx);
  fprintf(dfile, "std:%-4d ast:%-5d lineno:%-5d ", stdx, astx,
          STD_LINENO(stdx));
  dbg_print_ast(astx, dfile);
} /* dsstd */

void
dstdp(int stdx)
{
  dstd(stdx);
  if (STD_PTA(stdx)) {
    putstdpta(stdx);
  }
  putstdassigns(stdx);
} /* dstdp */

void
dstdtree(int stdx)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (stdx < 0 || stdx >= astb.std.stg_avail) {
    fprintf(dfile, "\nstd %d out of range [0:%d)\n", stdx, astb.std.stg_avail);
    return;
  }
  dstd(stdx);
  if (STD_AST(stdx)) {
    dastree(STD_AST(stdx));
  }
} /* dstdtree */

void
dstds(int std1, int std2)
{
  int stdx;
  if (std1 <= 0 || std1 >= astb.std.stg_avail)
    std1 = STD_NEXT(0);
  if (std2 <= 0 || std2 >= astb.std.stg_avail)
    std2 = STD_PREV(0);
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "subprogram %d %s:\n", gbl.func_count, SYMNAME(gbl.currsub));
  for (stdx = std1; stdx != 0; stdx = STD_NEXT(stdx)) {
    dstd(stdx);
    if (stdx == std2)
      break;
  }
} /* dstds */

void
dstdps(int std1, int std2)
{
  int stdx;
  if (std1 <= 0 || std1 >= astb.std.stg_avail)
    std1 = STD_NEXT(0);
  if (std2 <= 0 || std2 >= astb.std.stg_avail)
    std2 = STD_PREV(0);
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "subprogram %d %s:\n", gbl.func_count, SYMNAME(gbl.currsub));
  for (stdx = std1; stdx != 0; stdx = STD_NEXT(stdx)) {
    dstdp(stdx);
    if (stdx == std2)
      break;
  }
} /* dstdps */

/*
 * for simpler output
 */
void
dsstds(int std1, int std2)
{
  int stdx;
  if (std1 <= 0 || std1 >= astb.std.stg_avail)
    std1 = STD_NEXT(0);
  if (std2 <= 0 || std2 >= astb.std.stg_avail)
    std2 = STD_PREV(0);
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "subprogram %d %s:\n", gbl.func_count, SYMNAME(gbl.currsub));
  for (stdx = std1; stdx != 0; stdx = STD_NEXT(stdx)) {
    dsstd(stdx);
    if (stdx == std2)
      break;
  }
} /* dsstds */

/* dump 'count' before and after stdx */
void
dstdr(int stdx, int count)
{
  int s1, s2, c;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (stdx < 0 || stdx >= astb.std.stg_avail) {
    fprintf(dfile, "\nstd %d out of range [0:%d)\n", stdx, astb.std.stg_avail);
    return;
  }
  /* go backwards for 'count' */
  for (s1 = stdx, c = count; c > 0 && s1 > 0 && s1 <= astb.std.stg_avail;
       s1 = STD_PREV(s1), --c)
    ;
  /* go forwards for 'count' */
  for (s2 = stdx, c = count; c > 0 && s2 > 0 && s2 <= astb.std.stg_avail;
       s2 = STD_NEXT(s2), --c)
    ;
  dstds(s1, s2);
} /* dstdr */

void
dstda()
{
  dstds(0, 0);
} /* dstda */

void
dstdpa()
{
  putstdassigns(0);
  if (STD_PTA(0)) {
    putstdpta(0);
  }
  dstdps(0, 0);
} /* dstdpa */

void
dumpstdtrees()
{
  int stdx, std1;
  std1 = STD_NEXT(0);
  for (stdx = STD_NEXT(0); stdx != 0; stdx = STD_NEXT(stdx)) {
    dstdtree(stdx);
    if (stdx == STD_PREV(0))
      break;
    fprintf(dfile, "\n");
  }
} /* dumpstdtrees */

void
dsocptr(int sochead)
{
  int socptr;
  if (sochead == 0)
    return;
  if (sochead < 0 || sochead >= soc.avail) {
    putline();
    fprintf(dfile, "SOCPTR:%d out of bounds", sochead);
    return;
  }
  putline();
  putstring1("symbol overlap list:");
  for (socptr = sochead; socptr; socptr = SOC_NEXT(socptr)) {
    putsym1(SOC_SPTR(socptr));
  }
} /* dsocptr */

void
puttmpltype(const char *s, int t)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  switch (t) {
  case REPLICATED:
    sprintf(BUF, "%s:replicated", s);
    break;
  case DISTRIBUTED:
    sprintf(BUF, "%s:distributed", s);
    break;
  case ALIGNED:
    sprintf(BUF, "%s:aligned", s);
    break;
  case INHERITED:
    sprintf(BUF, "%s:inherited", s);
    break;
  default:
    sprintf(BUF, "%s:%d", s, t);
    break;
  }
  putit();
} /* puttmpltype */

void
puttmplsc(const char *s, int t)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  switch (t) {
  case NONE_SC:
    sprintf(BUF, "%s:none", s);
    break;
  case ALLOC_SC:
    sprintf(BUF, "%s:alloc", s);
    break;
  case DUMMY_SC:
    sprintf(BUF, "%s:dummy", s);
    break;
  case STATIC_SC:
    sprintf(BUF, "%s:static", s);
    break;
  case COMMON_SC:
    sprintf(BUF, "%s:common", s);
    break;
  default:
    sprintf(BUF, "%s:%d", s, t);
    break;
  }
  putit();
} /* puttmplsc */

void
puttmplflag(char *s, int f)
{
  /* single space between flags */
  if (f) {
    int l = strlen(s);
    if (linelen + l >= 79) {
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
} /* puttmplflag */

#ifdef IGNORE_TKRG
static void
put_ignore(const char *s, int tkr)
{
  if (tkr) {
    strcpy(BUF, s);
    strcat(BUF, ":");
    if ((tkr & IGNORE_TKR_ALL) == IGNORE_TKR_ALL) {
      strcat(BUF, "all");
      if (tkr & IGNORE_D)
        strcat(BUF, "+D");
    } else {
      if (tkr & IGNORE_T)
        strcat(BUF, "T");
      if (tkr & IGNORE_K)
        strcat(BUF, "K");
      if (tkr & IGNORE_R)
        strcat(BUF, "R");
      if (tkr & IGNORE_D)
        strcat(BUF, "D");
      if (tkr & IGNORE_M)
        strcat(BUF, "M");
      if (tkr & IGNORE_C)
        strcat(BUF, "C");
    }
    putit();
  }
} /* put_ignore */
#endif

void
dalnd(int alnd)
{
  int i, f;
  if (alnd == 0)
    return;
  if (dtb.base == NULL)
    return;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (alnd < 0 || alnd >= dtb.avl) {
    putline();
    fprintf(dfile, "ALND:%d out of bounds 1:%d\n", alnd, dtb.avl);
    return;
  }
  putline();
  putint("alnd", alnd);
  puttmpltype("type", TMPL_TYPE(alnd));
  puttmplsc("alignee_sc", TMPL_ALIGNEE_SC(alnd));
  puttmplsc("target_sc", TMPL_TARGET_SC(alnd));
  putint("rank", TMPL_RANK(alnd));
  f = TMPL_FLAG(alnd);
  putbit("sequential", f & __SEQUENTIAL);
  f &= ~__SEQUENTIAL;
  putbit("assumedsize", f & __ASSUMED_SIZE);
  f &= ~__ASSUMED_SIZE;
  putbit("assumedshape", f & __ASSUMED_SHAPE);
  f &= ~__ASSUMED_SHAPE;
  putbit("save", f & __SAVE);
  f &= ~__SAVE;
  putbit("inherit", f & __INHERIT);
  f &= ~__INHERIT;
  putbit("dynamic", f & __DYNAMIC);
  f &= ~__DYNAMIC;
  putbit("pointer", f & __POINTER);
  f &= ~__POINTER;
  putbit("local", f & __LOCAL);
  f &= ~__LOCAL;
  putbit("off_template", f & __OFF_TEMPLATE);
  f &= ~__OFF_TEMPLATE;
  putbit("identify_map", f & __IDENTITY_MAP);
  f &= ~__IDENTITY_MAP;
  putbit("no_overlaps", f & __NO_OVERLAPS);
  f &= ~__NO_OVERLAPS;
  putbit("in", f & __INTENT_IN);
  f &= ~__INTENT_IN;
  putbit("out", f & __INTENT_OUT);
  f &= ~__INTENT_OUT;

  if ((f & __TRANSCRIPTIVE_DIST_TARGET) == __TRANSCRIPTIVE_DIST_TARGET) {
    putstring("disttarget", "transcriptive");
  } else if (f & __PRESCRIPTIVE_DIST_TARGET) {
    putstring("disttarget", "prescriptive");
  } else if (f & __DESCRIPTIVE_DIST_TARGET) {
    putstring("disttarget", "descriptive");
  } else {
    putstring("disttarget", "omitted");
  }
  f &= ~__TRANSCRIPTIVE_DIST_TARGET;
  if ((f & __TRANSCRIPTIVE_DIST_FORMAT) == __TRANSCRIPTIVE_DIST_FORMAT) {
    putstring("format", "transcriptive");
  } else if (f & __PRESCRIPTIVE_DIST_FORMAT) {
    putstring("format", "prescriptive");
  } else if (f & __DESCRIPTIVE_DIST_FORMAT) {
    putstring("format", "descriptive");
  } else {
    putstring("format", "omitted");
  }
  f &= ~__TRANSCRIPTIVE_DIST_FORMAT;
  if (f & __PRESCRIPTIVE_ALIGN_TARGET) {
    putstring("aligntarget", "prescriptive");
  }
  f &= ~__PRESCRIPTIVE_ALIGN_TARGET;
  if (f & __DESCRIPTIVE_ALIGN_TARGET) {
    putstring("aligntarget", "descriptive");
  }
  f &= ~__DESCRIPTIVE_ALIGN_TARGET;

  if (f) {
    putnzhex("flag", f);
  }
  putnzbits("isstar", TMPL_ISSTAR(alnd));
  putnzbits("collapse", TMPL_COLLAPSE(alnd));
  putline();
  putnsym("descr", TMPL_DESCR(alnd));
  putnsym("align_target", TMPL_ALIGN_TARGET(alnd));
  putnsym("target_descr", TMPL_TARGET_DESCR(alnd));
  putnsym("dist_target", TMPL_DIST_TARGET(alnd));
  putnsym("dist_target_descr", TMPL_DIST_TARGET_DESCR(alnd));
  putline();
  for (i = 0; i < TMPL_RANK(alnd); ++i) {
    putint("dim", i);
    putast("lb", TMPL_LB(alnd, i));
    putast("ub", TMPL_UB(alnd, i));
    putline();
  }
} /* dalnd */

void
dsecd(int secd)
{
  if (secd == 0)
    return;
  if (dtb.base == NULL)
    return;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (secd < 0 || secd >= dtb.avl) {
    putline();
    fprintf(dfile, "SECD:%d out of bounds 1:%d\n", secd, dtb.avl);
    return;
  }
  putline();
  putint("secd", secd);
  putint("rank", INS_RANK(secd));
  putnsym("descr", INS_DESCR(secd));
  putnsym("template", INS_TEMPLATE(secd));
  putdtype("dtype", INS_DTYPE(secd));
  putline();
} /* dsecd */

/* dump one symbol */
void
dsym(int sptr)
{
  static char namebuff[210], typebuff[300];
  int stype, dtype, alnd, secd, scope;
  int i;

  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;

  if (sptr < 0 || sptr >= stb.stg_avail) {
    fprintf(dfile, "\nsymbol %d out of %d\n", sptr, stb.stg_avail);
    return;
  }

  BCOPY(stb.stg_base, stb.stg_base + sptr, SYM, 1);

  strcpy(namebuff, getprint(sptr));
  stype = STYPEG(0);
  dtype = DTYPEG(0);
  typebuff[0] = '\0';
  if (dtype)
    getdtype(dtype, typebuff);

  fprintf(dfile, "\n%-30.30s %s\n", namebuff, typebuff);
  linelen = 0;

  putint("sptr", sptr);
  putint("dtype", DTYPEG(0));
  putsc("sc", SCG(0));
  putstype("stype", STYPEG(0));
  if (UNAMEG(0)) {
    putstring("uname", stb.n_base + UNAMEG(0));
  }

  putline();
  DTYPEP(0, 0);
  NMPTRP(0, 0);
  SCP(0, 0);
  UNAMEP(0, 0);

  putsym("enclfunc", ENCLFUNCG(0));
  putsym("hashlk", HASHLKG(0));
  scope = SCOPEG(0);
  if (scope == 0 || scope >= 200) // 200 is largely arbitrary
    putsym("scope", SCOPEG(0));
  else if (scope == 1)
    putint("[constant]scope", SCOPEG(0));
  // scope == 2 might have special meaning
  else
    putint("[unnamed]scope", SCOPEG(0)); // interface and parallel scopes
  putsym("symlk", SYMLKG(0));
  putline();
  ENCLFUNCP(0, 0);
  HASHLKP(0, 0);
  SCOPEP(0, 0);
  SYMLKP(0, 0);

  switch (stype) {
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_IDENT:
  case ST_STRUCT:
  case ST_UNION:
  case ST_UNKNOWN:
  case ST_VAR:
    /* three lines: integers, symbols, bits */
    if (SCG(sptr) != SC_DUMMY) {
      putint("address", ADDRESSG(0));
      ADDRESSP(0, 0);
    }
    putintent("intent", INTENTG(0));
    INTENTP(0, 0);
    putnzint("paramval", PARAMVALG(0));
    PARAMVALP(0, 0);
#ifdef PDALNG
    putnzint("pdaln", PDALNG(0));
    b4P(0, 0);
#endif
    putnzint("socptr", SOCPTRG(0));
    SOCPTRP(0, 0);
    putline();
    putnsym("autobj", AUTOBJG(0));
    AUTOBJP(0, 0);
#ifdef BYTELENG
    if (DESCARRAYG(0)) {
      putnzint("bytelen", BYTELENG(0));
      BYTELENP(0, 0);
    }
#endif
    putnsym("cmblk", CMBLKG(0));
    CMBLKP(0, 0);
    putnsym("cvlen", CVLENG(0));
    CVLENP(0, 0);
    putnsym("adjstrlk", ADJSTRLKG(0));
    ADJSTRLKP(0, 0);
    putnsym("descr", DESCRG(0));
    DESCRP(0, 0);
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
    putnsym("newarg", NEWARGG(0));
    NEWARGP(0, 0);
    if (SCG(sptr) == SC_DUMMY) {
      putnsym("newdsc", NEWDSCG(0));
      NEWDSCP(0, 0);
    }
#ifdef DEVICECOPYG
    if (DEVICECOPYG(sptr)) {
      putsym("devcopy(of)", DEVCOPYG(0));
      DEVCOPYP(0, 0);
    } else {
      putnsym("devcopy", DEVCOPYG(0));
      DEVCOPYP(0, 0);
    }
#endif
#ifdef NMCNSTG
    putnsym("nmcnst", NMCNSTG(0));
    NMCNSTP(0, 0);
#endif
    putnsym("ptroff", PTROFFG(0));
    PTROFFP(0, 0);
    putnsym("sdsc", SDSCG(0));
    SDSCP(0, 0);
#ifdef IGNORE_TKRG
    put_ignore("ignore", IGNORE_TKRG(0));
    IGNORE_TKRP(0, 0);
#endif
    putnsym("slnk", SLNKG(0));
    SLNKP(0, 0);
    putline();
    putbit("addrtkn", ADDRTKNG(0));
    ADDRTKNP(0, 0);
    putbit("adjarr", ADJARRG(0));
    ADJARRP(0, 0);
    putbit("adjlen", ADJLENG(0));
    ADJLENP(0, 0);
    putbit("aftent", AFTENTG(0));
    AFTENTP(0, 0);
    putbit("alloc", ALLOCG(0));
    ALLOCP(0, 0);
#ifdef ALLOCATTRG
    putbit("allocattr", ALLOCATTRG(0));
    ALLOCATTRP(0, 0);
#endif
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("assn", ASSNG(0));
    ASSNP(0, 0);
    putbit("assumlen", ASSUMLENG(0));
    ASSUMLENP(0, 0);
    putbit("assumshp", ASSUMSHPG(0));
    ASSUMSHPP(0, 0);
    putbit("assumrank", ASSUMRANKG(0));
    ASSUMRANKP(0, 0);
    putbit("asumsz", ASUMSZG(0));
    ASUMSZP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
#ifdef CONSTANTG
    putbit("constant", CONSTANTG(0));
    CONSTANTP(0, 0);
#endif
    if (CONSTRUCTSYMG(0)) {
      putbit("constructsym", CONSTRUCTSYMG(0));
      CONSTRUCTSYMP(0, 0);
    }
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("descarray", DESCARRAYG(0));
    DESCARRAYP(0, 0);
    putbit("descused", DESCUSEDG(0));
    DESCUSEDP(0, 0);
#ifdef DEVICEG
    putbit("device", DEVICEG(0));
    DEVICEP(0, 0);
#endif
#ifdef DEVICECOPYG
    putbit("devicecopy", DEVICECOPYG(0));
    DEVICECOPYP(0, 0);
    putbit("devicesd", DEVICESDG(0));
    DEVICESDP(0, 0);
#endif
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("dovar", DOVARG(0));
    DOVARP(0, 0);
#ifdef DYNAMICG
    putbit("dynamic", DYNAMICG(0));
    DYNAMICP(0, 0);
#endif
    putbit("eqv", EQVG(0));
    EQVP(0, 0);
#ifdef EARLYSPECG
    putbit("earlyspec", EARLYSPECG(0));
    EARLYSPECP(0, 0);
#endif
    putbit("f90pointer", F90POINTERG(0));
    F90POINTERP(0, 0);
    putbit("forallndx", FORALLNDXG(0));
    FORALLNDXP(0, 0);
    putbit("func", FUNCG(0));
    FUNCP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
#ifdef INHERITP
    putbit("inherit", INHERITG(0));
    INHERITP(0, 0);
#endif
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
#ifdef LCLMEMP
    putbit("lclmem", LCLMEMG(0));
    LCLMEMP(0, 0);
#endif
    putbit("lnrzd", LNRZDG(0));
    LNRZDP(0, 0);
#ifdef MANAGEDG
    putbit("managed", MANAGEDG(0));
    MANAGEDP(0, 0);
#endif
    putbit("mdalloc", MDALLOCG(0));
    MDALLOCP(0, 0);
#ifdef MIRROREDG
    putbit("mirrored", MIRROREDG(0));
    MIRROREDP(0, 0);
    putbit("acccreate", ACCCREATEG(0));
    ACCCREATEP(0, 0);
    putbit("acccopyin", ACCCOPYING(0));
    ACCCOPYINP(0, 0);
    putbit("accresident", ACCRESIDENTG(0));
    ACCRESIDENTP(0, 0);
    putbit("acclink", ACCLINKG(0));
    ACCLINKP(0, 0);
#endif
#ifdef MUSTDECLP
    putbit("mustdecl", MUSTDECLG(0));
    MUSTDECLP(0, 0);
#endif
    putbit("nml", NMLG(0));
    NMLP(0, 0);
    putbit("nomdcom", NOMDCOMG(0));
    NOMDCOMP(0, 0);
    putbit("nodesc", NODESCG(0));
    NODESCP(0, 0);
    putbit("optarg", OPTARGG(0));
    OPTARGP(0, 0);
    putbit("param", PARAMG(0));
    PARAMP(0, 0);
#ifdef PASSBYVALP
    putbit("passbyval", PASSBYVALG(0));
    PASSBYVALP(0, 0);
#endif
#ifdef PE_RESIDENTP
    putbit("pe_resident", PE_RESIDENTG(0));
    PE_RESIDENTP(0, 0);
#endif
#ifdef PE_PRIVATEP
    putbit("pe_private", PE_PRIVATEG(0));
    PE_PRIVATEP(0, 0);
#endif
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
#ifdef PTRRHSG
    putbit("ptrrhs", PTRRHSG(0));
    PTRRHSP(0, 0);
#endif
    putbit("ptrv", PTRVG(0));
    PTRVP(0, 0);
    putbit("pure", PUREG(0));
    PUREP(0, 0);
    putbit("impure", IMPUREG(0));
    IMPUREP(0, 0);
    putbit("qaln", QALNG(0));
    QALNP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
#ifdef REFLECTEDG
    putbit("reflected", REFLECTEDG(0));
    REFLECTEDP(0, 0);
#endif
    putbit("result", RESULTG(0));
    RESULTP(0, 0);
#ifdef RUNTIMEP
    putbit("runtime", RUNTIMEG(0));
    RUNTIMEP(0, 0);
#endif
    putbit("save", SAVEG(0));
    SAVEP(0, 0);
#ifdef SCFXP
    putbit("scfx", SCFXG(0));
    SCFXP(0, 0);
#endif
#ifdef SDSCINITP
    putbit("sdscinit", SDSCINITG(0));
    SDSCINITP(0, 0);
#endif
#ifdef SDSCCONTIGP
    putbit("sdsccontig", SDSCCONTIGG(0));
    SDSCCONTIGP(0, 0);
#endif
#ifdef SDSCS1P
    putbit("sdscs1", SDSCS1G(0));
    SDSCS1P(0, 0);
#endif
    putbit("seq", SEQG(0));
    SEQP(0, 0);
#ifdef SHAREDP
    putbit("shared", SHAREDG(0));
    SHAREDP(0, 0);
#endif
#ifdef SYMMETRICP
    putbit("symmetric", SYMMETRICG(0));
    SYMMETRICP(0, 0);
#endif
    putbit("target", TARGETG(0));
    TARGETP(0, 0);
#ifdef TEXTUREP
    putbit("texture", TEXTUREG(0));
    TEXTUREP(0, 0);
#endif
    putbit("thread", THREADG(0));
    THREADP(0, 0);
#ifdef TQALNP
    putbit("tqaln", TQALNG(0));
    TQALNP(0, 0);
#endif
    putbit("typ8", TYP8G(0));
    TYP8P(0, 0);
    putbit("vcsym", VCSYMG(0));
    VCSYMP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putbit("vol", VOLG(0));
    VOLP(0, 0);
#ifdef PARREFP
    putbit("parref", PARREFG(0));
    PARREFP(0, 0);
#endif
#ifdef CLASSP
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
    /*
            putbit( "#", #G(0) );		#P(0,0);
    */
    if (SOCPTRG(sptr)) {
      dsocptr(SOCPTRG(sptr));
    }
    break;

  case ST_ALIAS:
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
    putbit("recur", RECURG(0));
    RECURP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("pure", PUREG(0));
    PUREP(0, 0);
    putbit("impure", IMPUREG(0));
    IMPUREP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_ARRDSC:
    alnd = ALNDG(0);
    putint("alnd", ALNDG(0));
    ALNDP(0, 0);
    secd = SECDG(0);
    putint("secd", SECDG(0));
    SECDP(0, 0);
    putline();
    putnsym("array", ARRAYG(0));
    ARRAYP(0, 0);
    putnsym("descr", DESCRG(0));
    DESCRP(0, 0);
    putnsym("rename", RENAMEG(0));
    RENAMEP(0, 0);
    putnsym("secdsc", SECDSCG(0));
    SECDSCP(0, 0);
    putnsym("slnk", SLNKG(0));
    SLNKP(0, 0);
    putline();
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    dsecd(secd);
    break;

  case ST_BLOCK:
    putint("startline", STARTLINEG(0));
    STARTLINEP(0, 0);
    putint("endline", ENDLINEG(0));
    ENDLINEP(0, 0);
    putint("rfcnt", RFCNTG(0));
    RFCNTP(0, 0);
    putsym("startlab", STARTLABG(0));
    STARTLABP(0, 0);
    putsym("endlab", ENDLABG(0));
    ENDLABP(0, 0);
    putint("entstd", ENTSTDG(0));
    ENTSTDP(0, 0);
#ifdef PARUPLEVELG
    putsym("paruplevel", PARUPLEVELG(0));
    PARUPLEVELP(0, 0);
#endif
    putsym("parsyms", PARSYMSG(0));
    PARSYMSP(0, 0);
    putsym("parsymsct", PARSYMSCTG(0));
    PARSYMSCTP(0, 0);
    putline();
    putbit("defd", DEFDG(0));
    DEFDP(0, 0);
    break;

  case ST_CMBLK:
    putint("size", SIZEG(0));
    SIZEP(0, 0);
    putline();
    putnsym("array", ARRAYG(0));
    ARRAYP(0, 0);
    putsym("cmemf", CMEMFG(0));
    CMEMFP(0, 0);
    putsym("cmeml", CMEMLG(0));
    CMEMLP(0, 0);
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
#ifdef PDALNG
    putnzint("pdaln", PDALNG(0));
    PDALNP(0, 0);
#endif
#ifdef DEVCOPYG
    putnsym("devcopy", DEVCOPYG(0));
    DEVCOPYP(0, 0);
#endif
    putline();
#ifdef ACCCREATEG
    putbit("acccreate", ACCCREATEG(0));
    ACCCREATEP(0, 0);
    putbit("acccopyin", ACCCOPYING(0));
    ACCCOPYINP(0, 0);
    putbit("accresident", ACCRESIDENTG(0));
    ACCRESIDENTP(0, 0);
    putbit("acclink", ACCLINKG(0));
    ACCLINKP(0, 0);
#endif
    putbit("alloc", ALLOCG(0));
    ALLOCP(0, 0);
    putbit("blankc", BLANKCG(0));
    BLANKCP(0, 0);
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
#ifdef CONSTANTG
    putbit("constant", CONSTANTG(0));
    CONSTANTP(0, 0);
#endif
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
#ifdef DEVICEG
    putbit("device", DEVICEG(0));
    DEVICEP(0, 0);
#endif
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
#ifdef HOISTEDG
    putbit("hoisted", HOISTEDG(0));
    HOISTEDP(0, 0);
#endif
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
#ifdef MAPPEDG
    putbit("mapped", MAPPEDG(0));
    MAPPEDP(0, 0);
#endif
#ifdef MUSTDECLP
    putbit("mustdecl", MUSTDECLG(0));
    MUSTDECLP(0, 0);
#endif
    putbit("nodesc", NODESCG(0));
    NODESCP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("qaln", QALNG(0));
    QALNP(0, 0);
#ifdef REWRITTENG
    putbit("rewritten", REWRITTENG(0));
    REWRITTENP(0, 0);
#endif
    putbit("save", SAVEG(0));
    SAVEP(0, 0);
    putbit("seq", SEQG(0));
    SEQP(0, 0);
    putbit("stdcall", STDCALLG(0));
    STDCALLP(0, 0);
    putbit("thread", THREADG(0));
    THREADP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putbit("vol", VOLG(0));
    VOLP(0, 0);
    putbit("frommod", FROMMODG(0));
    FROMMODP(0, 0);
    putsymlk("Members:", CMEMFG(sptr));
    break;

  case ST_CONST:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    putint("conval1g", CONVAL1G(0));
    CONVAL1P(0, 0);
    putint("conval2g", CONVAL2G(0));
    CONVAL2P(0, 0);
    putint("conval3g", CONVAL3G(0));
    CONVAL3P(0, 0);
    putint("conval4g", CONVAL4G(0));
    CONVAL4P(0, 0);
    putline();
    putbit("holl", HOLLG(0));
    HOLLP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("seq", SEQG(0));
    SEQP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_CONSTRUCT:
    putint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_CRAY:
  case ST_PD:
    putinkind("inkind", INKINDG(0));
    INKINDP(0, 0);
    putnzint("intast", INTASTG(0));
    INTASTP(0, 0);
    putdtype("inttyp", INTTYPG(0));
    INTTYPP(0, 0);
    put_inkwd("kwdarg", KWDARGG(0));
    KWDARGP(0, 0);
    putint("kwdcnt", KWDCNTG(0));
    KWDCNTP(0, 0);
    putint("pdnum", PDNUMG(0));
    PDNUMP(0, 0);
    putnname("pnmptr", PNMPTRG(0));
    PNMPTRP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("expst", EXPSTG(0));
    EXPSTP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("native", NATIVEG(0));
    NATIVEP(0, 0);
    putbit("typ8", TYP8G(0));
    TYP8P(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_ENTRY:
    putnzint("bihnum", BIHNUMG(0));
    BIHNUMP(0, 0);
    putint("dpdsc", DPDSCG(0));
    DPDSCP(0, 0);
    putnzint("endline", ENDLINEG(0));
    ENDLINEP(0, 0);
    putint("entnum", ENTNUMG(0));
    ENTNUMP(0, 0);
    putint("entstd", ENTSTDG(0));
    ENTSTDP(0, 0);
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
    putint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putline();
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putnsym("fval", FVALG(0));
    FVALP(0, 0);
#ifdef ACCROUTG
    putnzint("accrout", ACCROUTG(0));
    ACCROUTP(0, 0);
#endif
    putnsym("slnk", SLNKG(0));
    SLNKP(0, 0);
    putline();
    putbit("adjarr", ADJARRG(0));
    ADJARRP(0, 0);
    putbit("adjlen", ADJLENG(0));
    ADJLENP(0, 0);
    putbit("aftent", AFTENTG(0));
    AFTENTP(0, 0);
    putbit("assumlen", ASSUMLENG(0));
    ASSUMLENP(0, 0);
    putbit("assumshp", ASSUMSHPG(0));
    ASSUMSHPP(0, 0);
    putbit("asumsz", ASUMSZG(0));
    ASUMSZP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("decorate", DECORATEG(0));
    DECORATEP(0, 0);
    putbit("elemental", ELEMENTALG(0));
    ELEMENTALP(0, 0);
    putbit("f90pointer", F90POINTERG(0));
    F90POINTERP(0, 0);
    putbit("func", FUNCG(0));
    FUNCP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("inmodule", INMODULEG(0));
    INMODULEP(0, 0);
    putbit("ancestor", ANCESTORG(0));
    ANCESTORP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("ptrarg", PTRARGG(0));
    PTRARGP(0, 0);
    putbit("pure", PUREG(0));
    PUREP(0, 0);
    putbit("impure", IMPUREG(0));
    IMPUREP(0, 0);
    putbit("recur", RECURG(0));
    RECURP(0, 0);
    putbit("result", RESULTG(0));
    RESULTP(0, 0);
    putbit("seq", SEQG(0));
    SEQP(0, 0);
    putbit("stdcall", STDCALLG(0));
    STDCALLP(0, 0);
    putbit("typ8", TYP8G(0));
    TYP8P(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putline();
    putparam(DPDSCG(sptr), PARAMCTG(sptr));
    break;

  case ST_GENERIC:
    putinkind("inkind", INKINDG(0));
    INKINDP(0, 0);
    putnzint("intast", INTASTG(0));
    INTASTP(0, 0);
    putint("kindpos", KINDPOSG(0));
    KINDPOSP(0, 0);
    putint("kwdarg", KWDARGG(0));
    KWDARGP(0, 0);
    putint("kwdcnt", KWDCNTG(0));
    KWDCNTP(0, 0);
    putline();
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
    putnsym("gqcmplx", GQCMPLXG(0));
    GQCMPLXP(0, 0);
    putnsym("gquad", GQUADG(0));
    GQUADP(0, 0);
    putnsym("greal", GREALG(0));
    GREALP(0, 0);
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putnsym("gsint", GSINTG(0));
    GSINTP(0, 0);
    putline();
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("expst", EXPSTG(0));
    EXPSTP(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_INTRIN:
    putdtype("argtyp", ARGTYPG(0));
    ARGTYPP(0, 0);
    putnzint("arrayf", ARRAYFG(0));
    ARRAYFP(0, 0);
    putnzint("ilm", ILMG(0));
    ILMP(0, 0);
    putinkind("inkind", INKINDG(0));
    INKINDP(0, 0);
    putnzint("intast", INTASTG(0));
    INTASTP(0, 0);
    putdtype("inttyp", INTTYPG(0));
    INTTYPP(0, 0);
    put_inkwd("kwdarg", KWDARGG(0));
    KWDARGP(0, 0);
    putint("kwdcnt", KWDCNTG(0));
    KWDCNTP(0, 0);
    putnname("pnmptr", PNMPTRG(0));
    PNMPTRP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putline();
    putbit("addrtkn", ADDRTKNG(0));
    ADDRTKNP(0, 0);
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("expst", EXPSTG(0));
    EXPSTP(0, 0);
    putbit("native", NATIVEG(0));
    NATIVEP(0, 0);
#ifdef PTRRHSG
    putbit("ptrrhs", PTRRHSG(0));
    PTRRHSP(0, 0);
#endif
    putbit("target", TARGETG(0));
    TARGETP(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_MODULE:
    putint("base", CMEMFG(0));
    CMEMFP(0, 0);
    putnzint("endline", ENDLINEG(0));
    ENDLINEP(0, 0);
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
    putint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("needmod", NEEDMODG(0));
    NEEDMODP(0, 0);
    putbit("ancestor", ANCESTORG(0));
    ANCESTORP(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_PROC:
    putnzint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    putint("dpdsc", DPDSCG(0));
    DPDSCP(0, 0);
    putnzint("endline", ENDLINEG(0));
    ENDLINEP(0, 0);
#ifdef CUDAG
    putcuda("cuda", CUDAG(0));
    CUDAP(0, 0);
#endif
    putnzint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
    putintent("intent", INTENTG(0));
    INTENTP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putline();
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putnsym("fval", FVALG(0));
    FVALP(0, 0);
#ifdef ACCROUTG
    putnzint("accrout", ACCROUTG(0));
    ACCROUTP(0, 0);
#endif
    putnsym("altname", ALTNAMEG(0));
    ALTNAMEP(0, 0);
    putnsym("slnk", SLNKG(0));
    SLNKP(0, 0);
#ifdef VTOFFP
    putnzint("vtoff", VTOFFG(0));
    VTOFFP(0, 0);
#endif
#ifdef TBPLNKP
    putnzint("tbplnk", TBPLNKG(0));
    TBPLNKP(0, 0);
#endif
#ifdef INVOBJP
    putnzint("invobj", INVOBJG(0));
    INVOBJP(0, 0);
#endif
    putline();
    putbit("adjarr", ADJARRG(0));
    ADJARRP(0, 0);
    putbit("adjlen", ADJLENG(0));
    ADJLENP(0, 0);
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("asumsz", ASUMSZG(0));
    ASUMSZP(0, 0);
    putbit("assumlen", ASSUMLENG(0));
    ASSUMLENP(0, 0);
    putbit("assumshp", ASSUMSHPG(0));
    ASSUMSHPP(0, 0);
    putbit("cfunc", CFUNCG(0));
    CFUNCP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("decorate", DECORATEG(0));
    DECORATEP(0, 0);
    putbit("func", FUNCG(0));
    FUNCP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("indep", INDEPG(0));
    INDEPP(0, 0);
    putbit("inmodule", INMODULEG(0));
    INMODULEP(0, 0);
    putbit("ancestor", ANCESTORG(0));
    ANCESTORP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
#ifdef L3FG
    putbit("l3f", L3FG(0));
    L3FP(0, 0);
#endif
#ifdef LIBMP
    putbit("libm", LIBMG(0));
    LIBMP(0, 0);
#endif
    putbit("mscall", MSCALLG(0));
    MSCALLP(0, 0);
#ifdef MUSTDECLP
    putbit("mustdecl", MUSTDECLG(0));
    MUSTDECLP(0, 0);
#endif
    putbit("nocomm", NOCOMMG(0));
    NOCOMMP(0, 0);
    putbit("nodesc", NODESCG(0));
    NODESCP(0, 0);
    putbit("optarg", OPTARGG(0));
    OPTARGP(0, 0);
#ifdef PASSBYREFP
    putbit("passbyref", PASSBYREFG(0));
    PASSBYREFP(0, 0);
#endif
#ifdef PASSBYVALP
    putbit("passbyval", PASSBYVALG(0));
    PASSBYVALP(0, 0);
#endif
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("ptrarg", PTRARGG(0));
    PTRARGP(0, 0);
    putbit("pure", PUREG(0));
    PUREP(0, 0);
    putbit("impure", IMPUREG(0));
    IMPUREP(0, 0);
    putbit("recur", RECURG(0));
    RECURP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
    putbit("result", RESULTG(0));
    RESULTP(0, 0);
    putbit("sdscsafe", SDSCSAFEG(0));
    SDSCSAFEP(0, 0);
    putbit("sequent", SEQUENTG(0));
    SEQUENTP(0, 0);
    putbit("stdcall", STDCALLG(0));
    STDCALLP(0, 0);
    putbit("typ8", TYP8G(0));
    TYP8P(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
#ifdef CLASSP
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
    putparam(DPDSCG(sptr), PARAMCTG(sptr));
    break;

  case ST_LABEL:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    putnzint("iliblk", ILIBLKG(0));
    ILIBLKP(0, 0);
    if (LABSTDG(0)) {
      putint("labstd", LABSTDG(0));
      LABSTDP(0, 0);
    }
    putnzint("fmtpt", FMTPTG(0));
    FMTPTP(0, 0);
    putint("rfcnt", RFCNTG(0));
    RFCNTP(0, 0);
    putline();
    putbit("assn", ASSNG(0));
    ASSNP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("defd", DEFDG(0));
    DEFDP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("target", TARGETG(0));
    TARGETP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putbit("vol", VOLG(0));
    VOLP(0, 0);
    break;

  case ST_MEMBER:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
#ifdef BYTELENG
    if (DESCARRAYG(0)) {
      putnzint("bytelen", BYTELENG(0));
      BYTELENP(0, 0);
    }
#endif
    putnzint("encldtype", ENCLDTYPEG(0));
    ENCLDTYPEP(0, 0);
    putline();
    putnsym("descr", DESCRG(0));
    DESCRP(0, 0);
    putnsym("midnum", MIDNUMG(0));
    MIDNUMP(0, 0);
    putsym("psmem", PSMEMG(0));
    PSMEMP(0, 0);
    putnsym("ptroff", PTROFFG(0));
    PTROFFP(0, 0);
    putnsym("sdsc", SDSCG(0));
    SDSCP(0, 0);
    putsym("variant", VARIANTG(0));
    VARIANTP(0, 0);
#ifdef BINDP
    putnsym("bind", BINDG(0));
    BINDP(0, 0);
#endif
#ifdef VTABLEP
    putnsym("vtable", VTABLEG(0));
    VTABLEP(0, 0);
#endif
#ifdef PASSP
    putnsym("pass", PASSG(0));
    PASSP(0, 0);
#endif
    putline();
    /*putbit( "adjarr", ADJARRG(0) );		ADJARRP(0,0);*/
    putbit("alloc", ALLOCG(0));
    ALLOCP(0, 0);
#ifdef ALLOCATTRG
    putbit("allocattr", ALLOCATTRG(0));
    ALLOCATTRP(0, 0);
#endif
    putbit("arg", ARGG(0));
    ARGP(0, 0);
    putbit("assn", ASSNG(0));
    ASSNP(0, 0);
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("descarray", DESCARRAYG(0));
    DESCARRAYP(0, 0);
    putbit("descused", DESCUSEDG(0));
    DESCUSEDP(0, 0);
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("end", ENDG(0));
    ENDP(0, 0);
    putbit("f90pointer", F90POINTERG(0));
    F90POINTERP(0, 0);
    putbit("fnml", FNMLG(0));
    FNMLP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
#ifdef INHERITP
    putbit("inherit", INHERITG(0));
    INHERITP(0, 0);
#endif
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("lnrzd", LNRZDG(0));
    LNRZDP(0, 0);
    putbit("nodesc", NODESCG(0));
    NODESCP(0, 0);
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("ptrv", PTRVG(0));
    PTRVP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
#ifdef SDSCCONTIGP
    putbit("sdsccontig", SDSCCONTIGG(0));
    SDSCCONTIGP(0, 0);
#endif
#ifdef SDSCS1P
    putbit("sdscs1", SDSCS1G(0));
    SDSCS1P(0, 0);
#endif
#ifdef CLASSP
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_MODPROC:
    putint("funcline", FUNCLINEG(0));
    FUNCLINEP(0, 0);
    putint("symi", SYMIG(0));
    SYMIP(0, 0);
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putline();
    putsymilist(SYMIG(sptr));
    break;

  case ST_NML:
    putsym("plist", ADDRESSG(0));
    ADDRESSP(0, 0);
    putsym("cmemf", CMEMFG(0));
    CMEMFP(0, 0);
    putsym("cmeml", CMEMLG(0));
    CMEMLP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putline();
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i)) {
      putsym1(NML_SPTR(i));
    }
    break;

  case ST_OPERATOR:
    putnzint("gncnt", GNCNTG(0));
    GNCNTP(0, 0);
    putnzint("gndsc", GNDSCG(0));
    GNDSCP(0, 0);
    putinkind("inkind", INKINDG(0));
    INKINDP(0, 0);
    putnname("pdnum", PDNUMG(0));
    PDNUMP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putline();
    putsymilist(GNDSCG(sptr));
    break;

  case ST_PARAM:
    putint("conval1g", CONVAL1G(0));
    CONVAL1P(0, 0);
    putint("conval2g", CONVAL2G(0));
    CONVAL2P(0, 0);
    putint("paramval", PARAMVALG(0));
    PARAMVALP(0, 0);
    putnsym("slnk", SLNKG(0));
    SLNKP(0, 0);
    putline();
#ifdef CONSTANTG
    putbit("constant", CONSTANTG(0));
    CONSTANTP(0, 0);
#endif
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("end", ENDG(0));
    ENDP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("nodesc", NODESCG(0));
    NODESCP(0, 0);
    putbit("param", PARAMG(0));
    PARAMP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
#ifdef RUNTIMEP
    putbit("runtime", RUNTIMEG(0));
    RUNTIMEP(0, 0);
#endif
    putbit("typd", TYPDG(0));
    TYPDP(0, 0);
    putbit("vax", VAXG(0));
    VAXP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_PLIST:
    putint("address", ADDRESSG(0));
    ADDRESSP(0, 0);
    putnzint("deflab", DEFLABG(0));
    DEFLABP(0, 0);
    putint("pllen", PLLENG(0));
    PLLENP(0, 0);
    putnzint("swel", SWELG(0));
    SWELP(0, 0);
    putnsym("cmblk", CMBLKG(0));
    CMBLKP(0, 0);
    putline();
    putbit("ccsym", CCSYMG(0));
    CCSYMP(0, 0);
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("hccsym", HCCSYMG(0));
    HCCSYMP(0, 0);
    putbit("hidden", HIDDENG(0));
    HIDDENP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("ref", REFG(0));
    REFP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_STAG:
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("nest", NESTG(0));
    NESTP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_STFUNC:
    putint("excvlen", EXCVLENG(0));
    EXCVLENP(0, 0);
    putint("paramct", PARAMCTG(0));
    PARAMCTP(0, 0);
    putint("sfast", SFASTG(0));
    SFASTP(0, 0);
    putint("sfdsc", SFDSCG(0));
    SFDSCP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("typ8", TYP8G(0));
    TYP8P(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    break;

  case ST_TYPEDEF:
    putnsym("typdef_init", TYPDEF_INITG(0));
    TYPDEF_INITP(0, 0);
#ifdef PARENTG
    putnsym("parent", PARENTG(0));
    PARENTP(0, 0);
#endif
#ifdef VTOFFP
    putnzint("vtoff", VTOFFG(0));
    VTOFFP(0, 0);
#endif
    putnsym("sdsc", SDSCG(0));
    SDSCP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("dinit", DINITG(0));
    DINITP(0, 0);
    putbit("distmem", DISTMEMG(0));
    DISTMEMP(0, 0);
    putbit("ignore", IGNOREG(0));
    IGNOREP(0, 0);
    putbit("internal", INTERNALG(0));
    INTERNALP(0, 0);
    putbit("pointer", POINTERG(0));
    POINTERP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("seq", SEQG(0));
    SEQP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putbit("frommod", FROMMODG(0));
    FROMMODP(0, 0);
#ifdef ALLOCFLDG
    putbit("allocfld", ALLOCFLDG(0));
    ALLOCFLDP(0, 0);
#endif
#ifdef CLASSP
    putbit("class", CLASSG(0));
    CLASSP(0, 0);
#endif
    putbit("descused", DESCUSEDG(0));
    DESCUSEDP(0, 0);
    break;

  case ST_USERGENERIC:
    putnzint("gncnt", GNCNTG(0));
    GNCNTP(0, 0);
    putnzint("gndsc", GNDSCG(0));
    GNDSCP(0, 0);
    putnsym("gsame", GSAMEG(0));
    GSAMEP(0, 0);
    putline();
    putbit("dcld", DCLDG(0));
    DCLDP(0, 0);
    putbit("expst", EXPSTG(0));
    EXPSTP(0, 0);
    putbit("private", PRIVATEG(0));
    PRIVATEP(0, 0);
    putbit("visit", VISITG(0));
    VISITP(0, 0);
    putbit("visit2", VISIT2G(0));
    VISIT2P(0, 0);
    putline();
    putsymilist(GNDSCG(sptr));
    break;

  } /* switch(stype) */
  putline();

  STYPEP(0, 0);
  check("b3", stb.stg_base[0].b3);
  check("b4", stb.stg_base[0].b4);
  check("f1", stb.stg_base[0].f1);
  check("f1", stb.stg_base[0].f2);
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
  check("f49", stb.stg_base[0].f49);
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
  check("w19", stb.stg_base[0].w19);
  check("w20", stb.stg_base[0].w20);
  check("w21", stb.stg_base[0].w21);
  check("w22", stb.stg_base[0].w22);
  check("w23", stb.stg_base[0].w23);
  check("w24", stb.stg_base[0].w24);
  check("w25", stb.stg_base[0].w25);
  check("w26", stb.stg_base[0].w26);
  check("w27", stb.stg_base[0].w27);
  check("w28", stb.stg_base[0].w28);
  check("uname", stb.stg_base[0].uname);
  check("w30", stb.stg_base[0].w30);
  check("w31", stb.stg_base[0].w31);
  check("w32", stb.stg_base[0].w32);
  check("w34", stb.stg_base[0].w34);
  check("w35", stb.stg_base[0].w35);
  check("w36", stb.stg_base[0].w36);
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
} /* dsyms */

void
ds(int sptr)
{
  dsym(sptr);
} /* ds */

void
dsa()
{
  dsyms(0, 0);
} /* dsa */

void
dss(int l, int u)
{
  dsyms(l, u);
} /* dss */

void
dumphashlk(int sptr)
{
  int s, n;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "hash link for %d:%s\n", sptr, SYMNAME(sptr));
  n = NMPTRG(sptr);
  for (s = first_hash(sptr); s > 1; s = HASHLKG(s)) {
    if (NMPTRG(s) == n) {
      putline();
    }
    putsym1(s);
  }
  putline();
} /* dumphashlk */

void
dcommon(int c)
{
  int m;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (c <= NOSYM || c >= stb.stg_avail) {
    fprintf(dfile, "\ncommon block symbol %d out of %d\n", c, stb.stg_avail);
    return;
  }
  fprintf(dfile, "common/%d:%s/ ", c, SYMNAME(c));
  for (m = CMEMFG(c); m > NOSYM; m = SYMLKG(m)) {
    if (m != CMEMFG(c))
      fprintf(dfile, ", ");
    fprintf(dfile, "%d:%s", m, SYMNAME(m));
  }
  fprintf(dfile, "\n");
} /* dcommon */

void
dcommons()
{
  int cmn, any;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (gbl.cmblks <= NOSYM) {
    fprintf(dfile, "Common Block List is Empty\n");
  } else {
    fprintf(dfile, "Common Block List:\n");
    for (cmn = gbl.cmblks; cmn > NOSYM; cmn = SYMLKG(cmn)) {
      dcommon(cmn);
    }
  }
  any = 0;
  for (cmn = 2; cmn < stb.stg_avail; ++cmn) {
    if (STYPEG(cmn) == ST_CMBLK) {
      /* find it on the list */
      int c;
      for (c = gbl.cmblks; c != cmn && c > NOSYM; c = SYMLKG(c))
        ;
      if (c != cmn) {
        /* not found */
        if (!any) {
          fprintf(dfile, "Not on Common Block List:\n");
        }
        dcommon(c);
        ++any;
      }
    }
  }
} /* dcommons */

void
dumpdt(int dt)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (dt <= 0 || dt >= stb.dt.stg_avail) {
    fprintf(dfile, "\ndatatype %d out of %d\n", dt, stb.dt.stg_avail);
    return;
  }
  putint("dtype", dt);
  putdty("ty", DTY(dt));
  putline();

  switch (DTY(dt)) {
  case TY_CHAR:
  case TY_NCHAR:
    putast("len", DTY(dt + 1));
    break;
  case TY_PTR:
    putint("pointee", DTY(dt + 1));
    break;
  case TY_ARRAY:
    putint("element", DTY(dt + 1));
    putint("desc", DTY(dt + 2));
    if (DTY(dt + 2)) {
      int numdim, i;
      numdim = ADD_NUMDIM(dt);
      putint("numdim", numdim);
      putbit("adjarr", ADD_ADJARR(dt));
      putbit("assumedsize", ADD_ASSUMSZ(dt));
      putbit("assumedshape", ADD_ASSUMSHP(dt));
      putbit("assumedrank", ADD_ASSUMRANK(dt));
      putbit("defer", ADD_DEFER(dt));
      putbit("nobounds", ADD_NOBOUNDS(dt));
      putast("zbase", ADD_ZBASE(dt));
      putast("numelm", ADD_NUMELM(dt));
      if (numdim < 1 || numdim > 7) {
        numdim = 0;
      }
      for (i = 0; i < numdim; ++i) {
        putline();
        putint("dim", i);
        putast("lwbd", ADD_LWBD(dt, i));
        putast("upbd", ADD_UPBD(dt, i));
        putast("lwast", ADD_LWAST(dt, i));
        putast("upast", ADD_UPAST(dt, i));
        putast("extntast", ADD_EXTNTAST(dt, i));
        putast("mlpyr", ADD_MLPYR(dt, i));
      }
    }
    break;
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    putsym("sptr", DTY(dt + 1));
    putint("size", DTY(dt + 2));
    putsym("tag", DTY(dt + 3));
    putint("align", DTY(dt + 4));
    putint("ict", DTY(dt + 5));
    putline();
    putsymlk("members", DTY(dt + 1));
    break;
  }
  putline();
  if (dt > 0 && dt < stb.dt.stg_avail && DTY(dt) > 0)
    putdtype("type", dt);
  putline();
} /* dumpdt */

void
dumpdts()
{
  int dt;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** DATATYPE TABLE **********\n");
  for (dt = 1; dt < stb.dt.stg_avail; dt += dlen(DTY(dt))) {
    if (dt) {
      fprintf(dfile, "\n");
    }
    dumpdt(dt);
  }
} /* dumpdts */

void
ddt(int dt)
{
  dumpdt(dt);
} /* ddt */

void
dumpdtypes()
{
  dumpdts();
}

/* hlvect.h */
char *
getdddir(DDEDGE *dd)
{
  static char buf[99];
  char *p;
  int i, dir;
  p = buf;
  dir = DD_DIRVEC(dd);
  for (i = MAX_LOOPS - 1; i >= 0; --i) {
    DIRVEC t;
    if (p != buf)
      *p++ = ',';
    t = DIRV_ENTRYG(dir, i);
    if (t == DIRV_LT) {
      *p++ = '<';
    } else if (t == DIRV_EQ) {
      *p++ = '=';
    } else if (t == DIRV_GT) {
      *p++ = '>';
    } else if (t == (DIRV_LT | DIRV_EQ)) {
      *p++ = '<';
      *p++ = '=';
    } else if (t == (DIRV_GT | DIRV_EQ)) {
      *p++ = '>';
      *p++ = '=';
    } else if (t == (DIRV_LT | DIRV_GT)) {
      *p++ = '<';
      *p++ = '>';
    } else if (t == (DIRV_LT | DIRV_EQ | DIRV_GT)) {
      *p++ = '*';
    }
    if (t & DIRV_RD)
      *p++ = 'R';
  }
  if (dir & DIRV_ALLEQ) {
    *p++ = ',';
    *p++ = '=';
    *p++ = '=';
  }
  *p++ = '\0';
  return buf;
} /* getdddir */

void
putdd(DDEDGE *dd)
{
  static const char *types[] = {"flow", "anti", "output", "?3",
                                "?4",   "?5",   "?6",     "?7"};
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (dd == NULL) {
    fprintf(dfile, "null-dd-edge");
    return;
  }

  sprintf(BUF, "%s (%s) %d", types[DD_TYPE(dd)], getdddir(dd), DD_SINK(dd));
  putit();
} /* putdd */

void
putddlist(const char *s, DDEDGE *ddlist)
{
  DDEDGE *dd;
  if (ddlist == NULL)
    return;
  putstring1(s);
  for (dd = ddlist; dd; dd = DD_NEXT(dd)) {
    putdd(dd);
  }
  putline();
} /* putddlist */

void
dumpvind(int i)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.indbase == NULL) {
    fprintf(dfile, "HLV.INDBASE not allocated\n");
    return;
  }
  if (i <= 0 || i >= hlv.indavail) {
    fprintf(dfile, "\nvind %d out of %d\n", i, hlv.indavail);
    return;
  }
  putint("vind", i);
  putint("nme", VIND_NM(i));
  putnme(VIND_NM(i));
  putast("load", VIND_LOAD(i));
  putast("init", VIND_INIT(i));
  putast("originit", VIND_ORIGINIT(i));
  putast("skip", VIND_SKIP(i));
  putast("totskip", VIND_TOTSKIP(i));
  putasttype("opc", VIND_OPC(i));
  putline();
  putbit("ptr", VIND_PTR(i));
  putbit("delete", VIND_DELETE(i));
  putbit("niu", VIND_NIU(i));
  putbit("midf", VIND_MIDF(i));
  putbit("alias", VIND_ALIAS(i));
  putbit("noinv", VIND_NOINV(i));
  putbit("omit", VIND_OMIT(i));
  putbit("visit", VIND_VISIT(i));
  putbit("alast", VIND_ALAST(i));
  putbit("save", VIND_SAVE(i));
  putline();
} /* dumpvind */

void
dumpvinds(int start, int end)
{
  int i;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.indbase == NULL) {
    fprintf(dfile, "HLV.INDBASE not allocated\n");
    return;
  }
  for (i = start; i <= end; ++i) {
    dumpvind(i);
  }
} /* dumpvinds */

void
dumpmr(int mr)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.mrbase == NULL) {
    fprintf(dfile, "HLV.MRBASE not allocated\n");
    return;
  }
  if (mr <= 0 || mr >= hlv.mravail) {
    fprintf(dfile, "memref %d should be between 1 and %d\n", mr,
            hlv.mravail - 1);
    return;
  }
  putint("memref", mr);
  putcharacter("type", MR_TYPE(mr));
  putast("ast", MR_ILI(mr));
  putint("nme", MR_NME(mr));
  putnme(MR_NME(mr));
  putint("std", MR_ILT(mr));
  putint("fg", MR_FG(mr));
  putnzint("loop", MR_LOOP(mr));
  putnzint("stmt", MR_STMT(mr));
  putint("nest", MR_NEST(mr));
  putint("subst", MR_SUBST(mr));
  putnzint("subcnt", MR_SUBCNT(mr));
  putnzint("fsubs", MR_FSUBS(mr));
  putnzint("cseili", MR_CSEILI(mr));
  putnzint("rg", MR_RG(mr));
  putnzint("prev", MR_PREV(mr));
  putnzint("next", MR_NEXT(mr));
  if (MR_IVUSE(mr)) {
    putnzint("iv", MR_IV(mr));
  } else if (MR_SCALR(mr)) {
    putnzint("sclr", MR_SCLR(mr));
  } else {
    putnzint("exparr", MR_EXPARR(mr));
  }
  putast("rewr", MR_REWR(mr));
  putline();
  putbit("array", MR_ARRAY(mr));
  putbit("based", MR_BASED(mr));
  putbit("exp", MR_EXP(mr));
  putbit("indir", MR_INDIR(mr));
  putbit("induc", MR_INDUC(mr));
  putbit("init", MR_INIT(mr));
  putbit("inval", MR_INVAL(mr));
  putbit("invar", MR_INVAR(mr));
  putbit("invec", MR_INVEC(mr));
  putbit("ivuse", MR_IVUSE(mr));
  putbit("scalr", MR_SCALR(mr));
  putline();
  putddlist("succ:", MR_SUCC(mr));
} /* dumpmr */

void
dumpmemref(int mr)
{
  dumpmr(mr);
} /* dumpmemref */

void
dumpmemrefs(int m1, int m2)
{
  int mr;
  for (mr = m1; mr <= m2; ++mr) {
    dumpmr(mr);
  }
} /* dumpmemrefs */

void
putmrlist(const char *s, int ivlist)
{
  if (ivlist) {
    int i;
    putstring1(s);
    for (i = ivlist; i; i = MR_NEXT(i)) {
      putint1(i);
    }
    putline();
  }
} /* putmrlist */

void
putsclist(const char *s, int sclist)
{
  if (sclist) {
    int sc;
    putstring1(s);
    for (sc = sclist; sc; sc = SCLR_NEXT(sc)) {
      putint("sclr", sc);
      putnme(SCLR_NME(sc));
      putbit("flag", SCLR_FLAG(sc));
      putbit("span", SCLR_SPAN(sc));
      putnsym("arrsym", SCLR_ARRSYM(sc));
      putnsym("masksym", SCLR_MASKSYM(sc));
    }
    putline();
  }
} /* putsclist */

void
dumpvloop(int l)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.lpbase == NULL) {
    fprintf(dfile, "VL not allocated\n");
    return;
  }
  if (l <= 0 || l >= hlv.lpavail) {
    fprintf(dfile, "\nvloop %d out of %d\n", l, hlv.lpavail);
    return;
  }
  putint("vloop", l);
  putint("nest", VL_NEST(l));
  putnzint("sibling", VL_SIBLING(l));
  putnzint("mrstart", VL_MRSTART(l));
  putnzint("mrcnt", VL_MRCNT(l));
  putnzint("mrecnt", VL_MRECNT(l));
  putnzint("istart", VL_ISTART(l));
  putnzint("icnt", VL_ICNT(l));
  putnzint("lbnd", VL_LBND(l));
  putnzint("ubnd", VL_UBND(l));
  putnzint("ilbnd", VL_ILBND(l));
  putnzint("albnd", VL_ALBND(l));
  putnzint("iubnd", VL_IUBND(l));
  putnzint("aubnd", VL_AUBND(l));
  putnzint("sclist", VL_SCLIST(l));
  putnzint("ealist", VL_EALIST(l));
  putnzint("prebih", VL_PREBIH(l));
  putast("lpcnt", VL_LPCNT(l));
  putline();
  putbit("assoc", VLP_ASSOC(l));
  putbit("cand", VL_CAND(l));
  putbit("del", VL_DEL(l));
  putbit("depchk", VLP_DEPCHK(l));
  putbit("eqvchk", VLP_EQVCHK(l));
  putbit("lastval", VLP_LSTVAL(l));
  putbit("ldstsplit", VLP_LDSTSPLIT(l));
  putbit("mincnt", VLP_MINCNT(l));
  putbit("opsplit", VLP_OPSPLIT(l));
  putbit("perf", VL_PERF(l));
  putbit("recog", VLP_RECOG(l));
  putbit("safecall", VLP_SAFECALL(l));
  putbit("shortlp", VLP_SHORTLP(l));
  putbit("trans", VLP_TRANS(l));
  putbit("ztrip", VL_ZTRIP(l));
  putline();
  dumpvinds(VL_ISTART(l), VL_ISTART(l) + VL_ICNT(l) - 1);
  dumpmemrefs(VL_MRSTART(l), VL_MRSTART(l) + VL_MRCNT(l) - 1);
  putmrlist("ivlist: ", VL_IVLIST(l));
  putsclist("sclist: ", VL_SCLIST(l));
} /* dumpvloop */

void
dumpvnest(int l)
{
  int inner;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.lpbase == NULL) {
    fprintf(dfile, "VL not allocated\n");
    return;
  }
  dumpvloop(l);
  fprintf(dfile, "\n");
  for (inner = VL_CHILD(l); inner > 0; inner = VL_SIBLING(inner)) {
    dumpvnest(inner);
  }
} /* dumpvnest */

void
dumpvloops()
{
  int l;
  if (hlv.lpbase == NULL) {
    fprintf(dfile, "VL not allocated\n");
    return;
  }
  for (l = hlv.looplist; l > 0; l = VL_SIBLING(l)) {
    dumpvnest(l);
  }
} /* dumpvloops */

void
dumpvnest2(int l)
{
  int inner;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (hlv.lpbase == NULL) {
    fprintf(dfile, "VL not allocated\n");
    return;
  }
  if (opt.lpb.stg_base == NULL) {
    fprintf(dfile, "opt.lpb not allocated\n");
    return;
  }
  dumploop(l);
  dumpvloop(l);
  fprintf(dfile, "\n");
  for (inner = VL_CHILD(l); inner > 0; inner = VL_SIBLING(inner)) {
    dumpvnest2(inner);
  }
} /* dumpvnest2 */

void
dumpvloops2()
{
  int l;
  if (hlv.lpbase == NULL) {
    fprintf(dfile, "VL not allocated\n");
    return;
  }
  if (opt.lpb.stg_base == NULL) {
    fprintf(dfile, "opt.lpb not allocated\n");
    return;
  }
  for (l = hlv.looplist; l > 0; l = VL_SIBLING(l)) {
    dumpvnest2(l);
  }
} /* dumpvloops2 */

void
dumpdef(int def)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (def <= 0 || def > opt.ndefs) {
    fprintf(dfile, "\ndef %d out of %d\n", def, opt.ndefs);
    return;
  }
  putint("def", def);
  putnzint("fg", DEF_FG(def));
  putnzint("std", DEF_STD(def));
  putnzint("next", DEF_NEXT(def));
  putnzint("lnext", DEF_LNEXT(def));
  putint("nme", DEF_NM(def));
  putint("lhs", DEF_LHS(def));
  putint("rhs", DEF_RHS(def));
  putint("addr", DEF_ADDR(def));
  if (DEF_NM(def))
    printnme(DEF_NM(def));
  putbit("arg", DEF_ARG(def));
  putbit("confl", DEF_CONFL(def));
  putbit("const", DEF_CONST(def));
  putbit("delete", DEF_DELETE(def));
  putbit("doinit", DEF_DOINIT(def));
  putbit("doend", DEF_DOEND(def));
  putbit("gen", DEF_GEN(def));
  putbit("other", DEF_OTHER(def));
  putbit("precise", DEF_PRECISE(def));
  putbit("self", DEF_SELF(def));
#ifdef DEF_LOOPENTRY
  putbit("loopentry", DEF_LOOPENTRY(def));
#endif
  putline();
  if (DEF_DU(def)) {
    DU *du;
    putstring1(" uses:");
    for (du = DEF_DU(def); du != NULL; du = du->next) {
      putint1(du->use);
    }
    putline();
  }
  if (DEF_CSEL(def)) {
    DU *du;
    putstring1(" cse:");
    for (du = DEF_CSEL(def); du != NULL; du = du->next) {
      putint1(du->use);
    }
    putline();
  }
} /* dumpdef */

void
dumpdefs(void)
{
  int def;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** DEF TABLE **********\n");
  for (def = FIRST_DEF; def <= opt.ndefs; ++def) {
    dumpdef(def);
    fprintf(dfile, "\n");
  }
  fprintf(dfile, "\n");
} /* dumpdefs */

void
dumpdeflist(int d)
{
  int def;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  for (def = d; def <= opt.ndefs && def > 0; def = DEF_NEXT(def)) {
    dumpdef(def);
    fprintf(dfile, "\n");
  }
  fprintf(dfile, "\n");
} /* dumpdeflist */

void
dumpuse(int use)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (use <= 0 || use > opt.useb.stg_avail) {
    fprintf(dfile, "use %d out of %d\n", use, opt.useb.stg_avail - 1);
    return;
  }
  putint("use", use);
  putnzint("fg", USE_FG(use));
  putnzint("std", USE_STD(use));
  putnzint("ast", USE_AST(use));
  putnzint("nme", USE_NM(use));
  if (USE_NM(use))
    printnme(USE_NM(use));
  putint("addr", USE_ADDR(use));
  putbit("arg", USE_ARG(use));
  putbit("cse", USE_CSE(use));
  putbit("doinit", USE_DOINIT(use));
  putbit("exposed", USE_EXPOSED(use));
  putbit("precise", USE_PRECISE(use));
#ifdef USE_MARK1
  putbit("mark1", USE_MARK1(use));
#endif
#ifdef USE_MARK2
  putbit("mark2", USE_MARK2(use));
#endif
#ifdef USE_LOOPENTRY
  putbit("loopentry", USE_LOOPENTRY(use));
#endif
  putline();
  if (USE_UD(use) != NULL) {
    UD *ud;
    putstring1(" defs:");
    for (ud = USE_UD(use); ud != NULL; ud = ud->next) {
      putint1(ud->def);
    }
    putline();
  }
} /* dumpuse */

void
dumpuses(void)
{
  int use;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "\n********** USE TABLE **********\n");
  for (use = 1; use < opt.useb.stg_avail; ++use) {
    dumpuse(use);
    fprintf(dfile, "\n");
  }
  fprintf(dfile, "\n");
} /* dumpuses */

#ifdef FIH_NAME
void
dfih(int f)
{
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (fihb.stg_base == NULL) {
    fprintf(dfile, "fihb.stg_base not allocated\n");
    return;
  }
  putint("fih", f);
  if (f <= 0 || f >= fihb.stg_size) {
    fprintf(dfile, "\nfile %d out of %d\n", f, fihb.stg_size - 1);
    return;
  }
  putstring("name", FIH_NAME(f));
  if (FIH_FUNCNAME(f) != NULL) {
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

#endif

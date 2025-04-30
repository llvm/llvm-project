/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - high level vectorization utils
 */

#include "gbldefs.h"
#include "error.h"
#ifndef NOVECTORIZE
#include "global.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "hlvect.h"
#include "fdirect.h"
#include "ccffinfo.h"
#include "symutl.h"
#include "induc.h"

#if DEBUG
#include <stdarg.h>

#ifdef FLANG_HLVECT_UNUSED
#define Trace(a) TraceOutput a
#define STrace(a) STraceOutput a
#endif

#ifdef FLANG_HLVECT_UNUSED
/* print a message, continue */
static void
TraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);

  if (DBGBIT(37, 2)) {
    if (gbl.dbgfil) {
      vfprintf(gbl.dbgfil, fmt, argptr);
      fprintf(gbl.dbgfil, "\n");
    } else {
      fprintf(stderr, "Trace: ");
      vfprintf(stderr, fmt, argptr);
      fprintf(stderr, "\n");
    }
  }
  va_end(argptr);
} /* TraceOutput */
#endif

#ifdef FLANG_HLVECT_UNUSED
/* print a message, continue */
static void
STraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);

  if (DBGBIT(37, 4)) {
    if (gbl.dbgfil) {
      vfprintf(gbl.dbgfil, fmt, argptr);
      fprintf(gbl.dbgfil, "\n");
    } else {
      fprintf(stderr, "Trace: ");
      vfprintf(stderr, fmt, argptr);
      fprintf(stderr, "\n");
    }
  }
  va_end(argptr);
} /* STraceOutput */
#endif

extern void dumpnme(int n);
extern void dumploop(int l);

#else

/* eliminate the trace output */
#define Trace(a)
#define STrace(a)
#endif

void unmark_vinduc(int loop);
void mark_vinduc(int loop);
int hlv_getsym(int phase, int type);
void vdel_bih(int bihx);
LOGICAL is_parent_loop(int lpParent, int lpChild);

#ifdef FLANG_HLVECT_UNUSED
static void analyze_subs(int loop);
static void compute_subs(int mr, int loop);
static LOGICAL ilt_preceeds(int ilt1, int ilt2);
static LOGICAL succ_search(int fg1, int fg2, int loop);
static void get_memrefs(int ili);
static void hlv_buildlp(void);
static void hlv_dovect(int loop);
static void hlv_vectall(void);
static void hlv_parall(void);
static void process_loop(int loop);
static void process_lpinfo(int loop, int removedo);
static void process_scalars(int loop);
static void find_scalars(int loop);
static void prune(int loop);
static void report_cause(int ln, int cause);
static void trans_calls(int lp);
static void trans_ifgoto(int lp);
static void remove_goto(int lp);
static void trans_minmax(int lp);
static void save_induc(int loop);
static void save_subs(int mr);
static void find_ztrip(void);
static int check_ztrip(int loop, int ili);
static void find_init(void);

static void vunlnk_bih(void);
static void delbih(int fgx);

static void hlv_syminitfunc(void);
#endif

#if DEBUG
void dump_memrefs(int start, int cnt);
#ifdef FLANG_HLVECT_UNUSED
static void dump_subs(void);
#endif
void dump_vinduc(int start, int cnt);
void dump_vloops(int first, int base);
#endif

HLVECT hlv = {0};

/* NOTE: the dtype of the hlv integer temps needs to be DT_INT.  Since DT_INT
 * is not a constant, the dtype field in the VTMP structures for integer
 * temps is initialized to DT_INT in hlv_syminitfile();
 */
VTMP hlv_temps[VT_PHASES][VT_MAX] = {
    /* Phase 0 -- initial values */
    {
        {0, 0, 0, "ivi", DT_INT4}, /* integer initial values */
        {0, 0, 0, "ivp", DT_CPTR}, /* pointer initial values */
                                   /* unused */
    },
    /* Phase 1 -- induction temps */
    {
        {0, 0, 0, "ndi", DT_INT4},  /* induction integers */
        {0, 0, 0, "ndp", DT_CPTR},  /* induction pointers */
        {0, 0, 0, "nds", DT_REAL4}, /* single precision */
        {0, 0, 0, "ndd", DT_REAL8}, /* double precision */
    },
};
VTMP hlv_vtemps = {
    0, 0, 0, "vt", DT_INT4,
};

/* Causes of loops not being vectorized. See messages in report_cause(). */
enum {
  NO_CAUSE,
  NEST_CAUSE,
  CALL_CAUSE,
  MEXIT_CAUSE,
  PRAGMA_CAUSE,
  COUNTABLE_CAUSE,
  COUNT_CAUSE,
  STMT_CAUSE,
  IN_MASTER,
  IN_CRITICAL
};

static int vdel_lst; /* list of blocks to be deleted at end of processing */

#ifdef FLANG_HLVECT_UNUSED
/* Init for function */
static void
hlv_init(void)
{
  optshrd_init();
  induction_init();
  optshrd_finit();
  vdel_lst = 0;
  hlv.looplist = 0;

  hlv.mrsize = 100;
  NEW(hlv.mrbase, MEMREF, hlv.mrsize);
  hlv.mravail = 1;
  hlv.indsize = 100;
  NEW(hlv.indbase, VIND, hlv.indsize);
  hlv.indavail = 1;
  hlv.subsize = 100;
  NEW(hlv.subbase, SUBS, hlv.subsize);
#if DEBUG
  memset(hlv.subbase, -1, sizeof(SUBS) * hlv.subsize);
#endif
  hlv.subavail = 1;

  hlv.rgsize = 100;
  NEW(hlv.rgbase, REGION, hlv.rgsize);
  hlv.rgavail = 1;
  hlv.stsize = 100;
  NEW(hlv.stbase, STMT, hlv.stsize);
  hlv.stavail = 1;
  hlv.scsize = 100;
  NEW(hlv.scbase, SCALAR, hlv.scsize);
  hlv.scavail = 1;
  hlv.easize = 100;
  NEW(hlv.eabase, EXPARR, hlv.easize);
  hlv.eaavail = 1;

  hlv_syminitfunc();
}
#endif

#ifdef FLANG_HLVECT_UNUSED
/* end for function */
static void
hlv_end(void)
{
  vunlnk_bih();
  optshrd_fend();
  optshrd_end();
  induction_end();
  FREE(hlv.subbase);
  FREE(hlv.indbase);
  FREE(hlv.mrbase);
  FREE(hlv.lpbase);
  FREE(hlv.rgbase);
  FREE(hlv.stbase);
  FREE(hlv.scbase);
  FREE(hlv.eabase);
}
#endif

/************************************************************************
 ************************************************************************/

void
mark_vinduc(int loop)
{
  /* mark all induction NMEs */
  int i, end;

  end = VL_ISTART(loop) + VL_ICNT(loop);
  for (i = VL_ISTART(loop); i < end; ++i)
    NME_RFPTR(VIND_NM(i)) = i;
}

void
unmark_vinduc(int loop)
{
  /* clear all induction NMEs */
  int i, end;

  end = VL_ISTART(loop) + VL_ICNT(loop);
  for (i = VL_ISTART(loop); i < end; ++i)
    NME_RFPTR(VIND_NM(i)) = 0;
}

#ifdef FLANG_HLVECT_UNUSED
static void
hlv_syminitfunc(void)
{
  int i, j;

  for (i = 0; i < VT_PHASES; ++i)
    for (j = 0; j < VT_MAX; ++j)
      hlv_temps[i][j].iavl = hlv_temps[i][j].iavl_base =
          hlv_temps[i][j].iavl_max;
  hlv_vtemps.iavl = hlv_vtemps.iavl_base = hlv_vtemps.iavl_max;
}
#endif

int
hlv_getsym(int phase, int type)
{
  int sptr;
  VTMP *p;

  p = &hlv_temps[phase][type];
  sptr = sym_get_scalar(p->pfx, "vt", p->dtype);
  HCCSYMP(sptr, TRUE);
  VCSYMP(sptr, TRUE);
  return (sptr);
}

int
hlv_getvsym(int baseDtype)
{
  int sptr = getsymf("pgf_%s%04d", hlv_vtemps.pfx, hlv_vtemps.iavl++);
  int dtype;
  ADSC *ad;

  if (hlv_vtemps.iavl > hlv_vtemps.iavl_max)
    hlv_vtemps.iavl_max = hlv_vtemps.iavl;
  STYPEP(sptr, ST_ARRAY);
  dtype = get_array_dtype(1, baseDtype);
  DCLDP(sptr, TRUE);
  DTYPEP(sptr, dtype);
  ad = AD_DPTR(dtype);
  AD_NUMDIM(ad) = 1;
  AD_LWBD(ad, 0) = AD_LWAST(ad, 0) = astb.i0;
  AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = AD_EXTNTAST(ad, 0) =
      mk_cval(STRIPSIZE - 1, DT_INT);
  AD_MLPYR(ad, 0) = astb.i1;
  AD_NUMELM(ad) = mk_cval(STRIPSIZE, DT_INT);

  /* needs to be static for VPU */
  /* switchable? */
  SCP(sptr, SC_STATIC);
  return sptr;
}

/* Add bihx to the list of blocks that need to be deleted at
 * the end of processing by hlvect. */
void
vdel_bih(int bihx)
{
  /* Use the BIH_RGSET field to construct a singly-linked list of
   * deleted blocks. */
  BIH_RGSET(bihx) = vdel_lst;
  vdel_lst = bihx;
}

#ifdef FLANG_HLVECT_UNUSED
/* Delete all bih's within the vdel_lst list, as well as all ilt's within
 * those bih's. */
static void
vunlnk_bih(void)
{
  int bihx, nextbih;

  for (bihx = vdel_lst; bihx; bihx = nextbih) {
    nextbih = BIH_RGSET(bihx);
    BIH_RGSET(bihx) = 0;
    rdilts(bihx);
    while (STD_NEXT(0) != 0)
      delilt(STD_NEXT(0));
    wrilts(bihx);
    delbih(bihx);
  }
}
#endif

#ifdef FLANG_HLVECT_UNUSED
static void
delbih(int fgx)
{
  FG_LNEXT(FG_LPREV(fgx)) = FG_LNEXT(fgx);
  FG_LPREV(FG_LNEXT(fgx)) = FG_LPREV(fgx);
}
#endif

#if DEBUG
#ifdef FLANG_HLVECT_UNUSED
/* DEBUG DUMP ROUTINES */
static void
dump_subs(void)
{
  int i;
  int j;

  for (i = 1; i < hlv.subavail; ++i) {
    fprintf(gbl.dbgfil, "*** %d ***; base: ", i);
    printast(SB_BASE(i));
    fprintf(gbl.dbgfil, "\n");
    for (j = 0; j < MAX_LOOPS; ++j) {
      if (SB_STRIDE(i)[j] == 0)
        break;
      fprintf(gbl.dbgfil, "   bases[%d] = ", j);
      printast(SB_BASES(i)[j + 1]);
      fprintf(gbl.dbgfil, "   stride[%d] = ", j);
      printast(SB_STRIDE(i)[j]);
      fprintf(gbl.dbgfil, "\n");
    }
  }
}
#endif

void
dump_memref_hdr(void)
{
  fprintf(gbl.dbgfil,
          "      fg ast std nme sst sct typ nst  lp  rg fsb flags\n");
}

void
dump_one_memref(int i)
{
  fprintf(gbl.dbgfil, "%3d: %3d %3d %3d %3d %3d %3d %3c %3d %3d %3d %3d <", i,
          MR_FG(i), MR_ILI(i), MR_ILT(i), MR_NME(i), MR_SUBST(i), MR_SUBCNT(i),
          MR_TYPE(i), MR_NEST(i), MR_LOOP(i), MR_RG(i), MR_FSUBS(i));
  /* flags here */
  if (MR_IVUSE(i))
    fprintf(gbl.dbgfil, "iu ");
  if (MR_ARRAY(i))
    fprintf(gbl.dbgfil, "ar ");
  if (MR_INDIR(i))
    fprintf(gbl.dbgfil, "*  ");
  if (MR_SCALR(i))
    fprintf(gbl.dbgfil, "sc ");
  if (MR_BASED(i))
    fprintf(gbl.dbgfil, "ba ");

  if (MR_INVAL(i))
    fprintf(gbl.dbgfil, "is ");
  if (MR_EXP(i))
    fprintf(gbl.dbgfil, "ex ");
  if (MR_INDUC(i))
    fprintf(gbl.dbgfil, "iv ");
  if (MR_INVAR(i))
    fprintf(gbl.dbgfil, "nv ");
  if (MR_INVEC(i))
    fprintf(gbl.dbgfil, "vc ");
  fprintf(gbl.dbgfil, ">");
  if (MR_NME(i))
    dumpname(MR_NME(i));
  if (MR_ILI(i)) {
    fprintf(gbl.dbgfil, " ");
    printast(MR_ILI(i));
  }
  fprintf(gbl.dbgfil, "\n");
  dump_dd(MR_SUCC(i));
}

void
dump_memrefs(int start, int cnt)
{
  int i, end;

  end = start + cnt;

  dump_memref_hdr();
  for (i = start; i < end; ++i) {
    dump_one_memref(i);
  }
}

void
dump_memref_list(int base)
{
  dump_memref_hdr();
  for (; base != 0; base = MR_NEXT(base))
    dump_one_memref(base);
}
void
dump_one_vloop(int i, int level)
{
  int j;

  for (j = 0; j < level; ++j)
    fprintf(gbl.dbgfil, "  ");
  fprintf(gbl.dbgfil,
          "%3d: child %d sibling %d nest %d mrstart %d mrcnt: %d mrecnt %d\n",
          i, VL_CHILD(i), VL_SIBLING(i), VL_NEST(i), VL_MRSTART(i), VL_MRCNT(i),
          VL_MRECNT(i));
  for (j = 0; j < level; ++j)
    fprintf(gbl.dbgfil, "  ");
  fprintf(gbl.dbgfil, "     flags: <");
  if (VL_CAND(i))
    fprintf(gbl.dbgfil, "cand ");
  if (VL_PERF(i))
    fprintf(gbl.dbgfil, "perf ");
  if (VL_ZTRIP(i))
    fprintf(gbl.dbgfil, "ztrip ");
  fprintf(gbl.dbgfil, "> istart %d icnt %d lpcnt %d ubnd %d lbnd %d\n",
          VL_ISTART(i), VL_ICNT(i), VL_LPCNT(i), VL_UBND(i), VL_LBND(i));
}

void
dump_vloops(int first, int base)
{
  int i;

  for (i = first; i != 0; i = VL_SIBLING(i)) {
    dump_one_vloop(i, base);
    dump_vloops(VL_CHILD(i), base + 1);
  }
}

void
dump_vinduc(int start, int cnt)
{
  int i, end;

  end = start + cnt;
  for (i = start; i < end; ++i) {
    fprintf(gbl.dbgfil, "%3d: %s, nme %d, load: %d, opc: %d, flags:<", i,
            getprint(basesym_of(VIND_NM(i))), VIND_NM(i), VIND_LOAD(i),
            VIND_OPC(i));
    if (VIND_PTR(i))
      fprintf(gbl.dbgfil, "ptr ");
    if (VIND_DELETE(i))
      fprintf(gbl.dbgfil, "del ");
    if (VIND_NIU(i))
      fprintf(gbl.dbgfil, "niu ");
    if (VIND_MIDF(i))
      fprintf(gbl.dbgfil, "mid ");
    if (VIND_ALIAS(i))
      fprintf(gbl.dbgfil, "al ");
    fprintf(gbl.dbgfil, ">\n");
    fprintf(gbl.dbgfil, "	  init:\t");
    printast(VIND_INIT(i));
    if (DBGBIT(10, 512))
      fprintf(gbl.dbgfil, "\n");
    fprintf(gbl.dbgfil, "	  totskip:\t");
    printast(VIND_TOTSKIP(i));
    if (DBGBIT(10, 512))
      fprintf(gbl.dbgfil, "\n");
    fprintf(gbl.dbgfil, "	  skip:\t");
    printast(VIND_SKIP(i));
    if (DBGBIT(10, 512))
      fprintf(gbl.dbgfil, "\n");
  }
}

#endif
#endif

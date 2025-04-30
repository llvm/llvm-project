/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Data dependence framework.
 */

#include "gbldefs.h"
#include "error.h"
#ifndef NOVECTORIZE
#include "global.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "hlvect.h"

#include "induc.h"

#include "soc.h"
#include "fdirect.h"
#include "extern.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

#if DEBUG
#define TRACE0(s)    \
  if (DBGBIT(36, 2)) \
  fprintf(gbl.dbgfil, s)
#define TRACE1(s, a1) \
  if (DBGBIT(36, 2))  \
  fprintf(gbl.dbgfil, s, a1)
#define TRACE2(s, a1, a2) \
  if (DBGBIT(36, 2))      \
  fprintf(gbl.dbgfil, s, a1, a2)
#define TRACE3(s, a1, a2, a3) \
  if (DBGBIT(36, 2))          \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#define TRACE4(s, a1, a2, a3, a4) \
  if (DBGBIT(36, 2))              \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4)
#define TRACE5(s, a1, a2, a3, a4, a5) \
  if (DBGBIT(36, 2))                  \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4, a5)

#define STRACE0(s)   \
  if (DBGBIT(36, 4)) \
  fprintf(gbl.dbgfil, s)
#define STRACE1(s, a1) \
  if (DBGBIT(36, 4))   \
  fprintf(gbl.dbgfil, s, a1)
#define STRACE2(s, a1, a2) \
  if (DBGBIT(36, 4))       \
  fprintf(gbl.dbgfil, s, a1, a2)
#define STRACE3(s, a1, a2, a3) \
  if (DBGBIT(36, 4))           \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#define STRACE4(s, a1, a2, a3, a4) \
  if (DBGBIT(36, 4))               \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4)
#define STRACE5(s, a1, a2, a3, a4, a5) \
  if (DBGBIT(36, 4))                   \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4, a5)

#define DTRACE0(s)    \
  if (DBGBIT(36, 16)) \
  fprintf(gbl.dbgfil, s)
#define DTRACE1(s, a1) \
  if (DBGBIT(36, 16))  \
  fprintf(gbl.dbgfil, s, a1)
#define DTRACE2(s, a1, a2) \
  if (DBGBIT(36, 16))      \
  fprintf(gbl.dbgfil, s, a1, a2)
#define DTRACE3(s, a1, a2, a3) \
  if (DBGBIT(36, 16))          \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#define DTRACE4(s, a1, a2, a3, a4) \
  if (DBGBIT(36, 16))              \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4)
#define DTRACE5(s, a1, a2, a3, a4, a5) \
  if (DBGBIT(36, 16))                  \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4, a5)

#define BTRACE0(s)    \
  if (DBGBIT(36, 64)) \
  fprintf(gbl.dbgfil, s)
#define BTRACE1(s, a1) \
  if (DBGBIT(36, 64))  \
  fprintf(gbl.dbgfil, s, a1)
#define BTRACE2(s, a1, a2) \
  if (DBGBIT(36, 64))      \
  fprintf(gbl.dbgfil, s, a1, a2)
#define BTRACE3(s, a1, a2, a3) \
  if (DBGBIT(36, 64))          \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#define BTRACE4(s, a1, a2, a3, a4) \
  if (DBGBIT(36, 64))              \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4)
#define BTRACE5(s, a1, a2, a3, a4, a5) \
  if (DBGBIT(36, 64))                  \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4, a5)

#else

#define TRACE0(s)
#define TRACE1(s, a1)
#define TRACE2(s, a1, a2)
#define TRACE3(s, a1, a2, a3)
#define TRACE4(s, a1, a2, a3, a4)
#define TRACE5(s, a1, a2, a3, a4, a5)
#define STRACE0(s)
#define STRACE1(s, a1)
#define STRACE2(s, a1, a2)
#define STRACE3(s, a1, a2, a3)
#define STRACE4(s, a1, a2, a3, a4)
#define STRACE5(s, a1, a2, a3, a4, a5)
#define DTRACE0(s)
#define DTRACE1(s, a1)
#define DTRACE2(s, a1, a2)
#define DTRACE3(s, a1, a2, a3)
#define DTRACE4(s, a1, a2, a3, a4)
#define DTRACE5(s, a1, a2, a3, a4, a5)
#define BTRACE0(s)
#define BTRACE1(s, a1)
#define BTRACE2(s, a1, a2)
#define BTRACE3(s, a1, a2, a3)
#define BTRACE4(s, a1, a2, a3, a4)
#define BTRACE5(s, a1, a2, a3, a4, a5)
#endif

#if DEBUG
ISZ_T
DBGcnst(int s)
{
  ISZ_T yy;
  yy = get_isz_cval(s);
  printf("CNSTG: %ld\n", yy);
  return yy;
}
#endif
#define ad_icon(i) (mk_isz_cval(i, astb.bnd.dtype))
#define prilitree(a) (printast(a))
#define ILT_NEXT(s) (STD_NEXT(s))
#define IS_CNST(a) (A_TYPEG(a) == A_CNST)
#define CNSTG(a) (get_isz_cval(A_SPTRG(a)))
#define ZZZCNSTG(a) (DBGcnst(A_SPTRG(a)))
#define RESG(a) (A_DTYPEG(a))
#define IS_IRES(a) (DT_ISINT(RESG(a)))
#define IS_ARES(a) (RESG(a) == DT_ADDR)
#define ILI_VISIT(a) (A_VISITG(a))
#define ILI_REPL(a) (A_REPLG(a))
/* arbitrary opcode numbers */
#define IL_INEG 1
#define IL_IADD 2
#define IL_ISUB 3
#define IL_IMUL 4
#define IL_IDIV 5
#define IL_MOD 6
#define IL_AADD 7
#define IL_ASUB 8

typedef struct DV {
  DIRVEC vec;
  struct DV *next;
} DV;

#define MAX_DD (2 * MAX_LOOPS)
#define MAX_N MAX_DD
#define MAXNMESIZ 30
#define MAX_S MAXNMESIZ

typedef struct bound {
  int bnd[MAX_N + 1];
  int mplyr; /* multiplier of lhs */
  int gcd;   /* gcd of lhs terms */
  struct bound *next;
} BOUND;

static void build_loop_dd(int loop);
static void dd_compute(void);
void dd_edge(int src, int sink, DIRVEC vec);
static void dd_succ(int loop);

static void resolve_vv(void);
static void resolve_pv(void);
static void resolve_pp(void);
static void resolve_uv(void);
static void do_subscript(int nsubs);
static void add_dep(DIRVEC vec);
static void hierarchy(DIRVEC dir, int lev, DIRVEC veco);
static void unpack_loops(void);
static void chkref(int i, int j);
LOGICAL dd_array_conflict(int astliTriples, int astArrSrc, int astArrSink,
                          int bSinkAfterSrc);
/*static DIRVEC dirv_permute();*/
static int symbolic_mul(int a, int b);
static LOGICAL symbolic_divide(int num, int den, int *quot);
static void cln_visit(void);
static int ili_symbolic(int ili);
int dd_symbolic(int il);
#if DEBUG
static void dump_one_bound(BOUND *p, int k, LOGICAL btype);
static void dump_two_bound(BOUND *p, BOUND *q, int k, LOGICAL btype);
#endif
static int ad1ili(int opc, int ast1);
static int ad2ili(int opc, int ast1, int ast2);
static int ILI_OPC(int astx);
static int ILI_OPND(int astx, int opnd);

static struct {
  int mr1, mr2;         /* current two mem refs */
  int subs1, subs2;     /* current two subscripts */
  int basenm1, basenm2; /* current two base names */
  int outer_loop;       /* outermost loop */
  DIRVEC vec;           /* temp */
  DV *dvlist;           /* list of direction vectors */
  int unknown;          /* dependence not proven */
  int n1, n2;           /* number of non-common loops */
  int n;                /* number of common loops */
  int common;           /* innermost common loop */
  short lps[MAX_DD];    /* unpacked loops */
  VPRAGMAS pragmas;
} ddinfo;

/*
 * Data dependence framework:
 * mr1 and mr2 are enclosed in n common loops, with n1 loops enclosing mr1
 * but not mr2, and n2 loops enclosing mr2 but not mr1.  common is the loop
 * index of the innermost common loop; the nesting depth of common is n;
 * the nesting depth of mr1 is n1+n; the nesting depth of mr2 is n2+n.
 *
 * L[1]	    DO			ddinfo.outer_loop
 *		...			    (ddinfo.n loops)
 * L[n]		DO			ddinfo.common
 * L1[1]	    DO
 *			...			(ddinfo.n1 loops)
 * L1[n1]		DO		    MR_LOOP(ddinfo.mr1)
 *			    mr1
 *			ENDDO
 *			...
 *		    ENDDO
 * L2[1]	    DO
 *			...			(ddinfo.n2 loops)
 * L2[n2]		DO		    MR_LOOP(ddinfo.mr2)
 *			    mr2
 *			ENDDO
 *			...
 *		    ENDDO
 *		ENDDO
 *		...
 *	    ENDDO
 *
 * The loops are unpacked into lps as follows:
 *
 * L1[n1],...,L1[1], L2[n2],...,L2[1], L[n],...,L[1]
 */

/* local data */
typedef BV *BVP;
static BVP *fgsucc;
static BV *fgbase;

/** \brief Build the data dependency graph. */
void
build_dd(int loop)
{
  int i;

  /* go through load/stores & build data dependency graph */
  ddinfo.n = 0;
  ddinfo.outer_loop = loop;
/*adjloops(loop);*/
#if DEBUG
  if (DBGBIT(36, 1)) {
    fprintf(gbl.dbgfil, "\n------- dump before build_loop_dd, loop %d ------\n",
            loop);
    dump_one_vloop(loop, 0);
    for (i = VL_CHILD(loop); i != 0; i = VL_SIBLING(i))
      dump_vloops(i, 1);
  }
#endif
  build_loop_dd(loop);
  cln_visit();
}

static void
build_nat_loop(int loop)
{
  int i;

  for (i = LP_FG(loop); i != 0; i = FG_NEXT(i)) {
    ++hlv.fgn;
    FG_NATNXT(i) = hlv.natural_loop;
    hlv.natural_loop = i;
  }
  for (i = VL_CHILD(loop); i != 0; i = VL_SIBLING(i)) {
    build_nat_loop(i);
  }
}

static void
dd_succ(int loop)
{
  /*
   * We need to know, for two blocks b1 and b2, if b1 can execute before b2
   * in the same iteration of the loop; if b1 cannot execute before b2,
   * then there can be no dependence with an equals direction from a memory
   * reference in b1 to a memory reference in b2.  To do this, we create,
   * for each flowgraph node, a bit vector giving all the successors of
   * that node, except that successors of the loop tail are not included.
   * Then, a transitive closure is performed on those bit vectors.
   */
  int i;
  int nbits;
  int nnodes;
  int ntotal;
  int offs;
  int change;
  PSI_P p;
  BV *temp_set;

  /*
   * number of bits needed -- count the number of flowgraph nodes in the
   * loop; for each of those, need a bit for each flowgraph node in the
   * entire flowgraph.
   */
  hlv.fgn = 0;
  hlv.natural_loop = 0;
  build_nat_loop(loop);

  nnodes = hlv.fgn;
  nbits = opt.num_nodes + 1;
  nbits = (nbits + BV_BITS - 1) / BV_BITS;
  ntotal = nbits * (nnodes + 1);
  NEW(fgsucc, BVP, opt.num_nodes + 1);
  NEW(fgbase, BV, ntotal);

  /* go through and initialize sets */
  offs = 0;
  for (i = hlv.natural_loop; i != 0; i = FG_NATNXT(i)) {
    if (i == LP_TAIL(loop))
      continue;
    fgsucc[i] = fgbase + offs;
    assert(offs < ntotal - nbits, "dd_succ: bad offs", loop, 4);
    offs += nbits;
    BZERO(fgsucc[i], BV, nbits);
    /* initialize successors */
    for (p = FG_SUCC(i); p != 0; p = PSI_NEXT(p))
      bv_set(fgsucc[i], PSI_NODE(p));
  }
  /* tail has no successors */
  fgsucc[LP_TAIL(loop)] = fgbase + offs;
  assert(offs < ntotal - nbits, "dd_succ: bad offs", loop, 4);
  offs += nbits;
  BZERO(fgsucc[LP_TAIL(loop)], BV, nbits);

  /* temporary set */
  temp_set = fgbase + offs;
  assert(offs <= ntotal - nbits, "dd_succ: bad offs", loop, 4);
  offs += nbits;

  /* now perform transitive closure */
  do {
    change = 0;
    for (i = hlv.natural_loop; i != 0; i = FG_NATNXT(i)) {
      if (i == LP_TAIL(loop))
        continue;
      bv_copy(temp_set, fgsucc[i], nbits);
      /*
       * union successor set of this node with successor sets of this
       * node's successor
       */
      for (p = FG_SUCC(i); p != 0; p = PSI_NEXT(p)) {
        bv_union(temp_set, fgsucc[PSI_NODE(p)], nbits);
      }
      if (bv_notequal(temp_set, fgsucc[i], nbits)) {
        change = TRUE;
        bv_copy(fgsucc[i], temp_set, nbits);
      }
    }
  } while (change);
#if DEBUG
  if (DBGBIT(36, 32)) {
    fprintf(gbl.dbgfil, "Successor bit sets\n");
    for (i = hlv.natural_loop; i != 0; i = FG_NATNXT(i)) {
      fprintf(gbl.dbgfil, "%3d:", i);
      bv_print(fgsucc[i], opt.num_nodes + 1);
    }
  }
#endif
}

static void
build_loop_dd(int loop)
{
  int i, j;
  int l1, l2;
  int end, end1;

  ++ddinfo.n;
  TRACE2("build_loop_dd: loop %d lev %d\n", loop, ddinfo.n);
  /* do inner loops first */
  for (i = VL_CHILD(loop); i != 0; i = VL_SIBLING(i)) {
    build_loop_dd(i);
  }
  ddinfo.common = loop;

  /* build successor bit vector */
  dd_succ(loop);

  /* go through all memory references */

  /* use pragmas for the common loop */
  ddinfo.pragmas = VL_PRAGMAS(loop);

  /* first, this loop */
  end = VL_MRSTART(loop) + VL_MRCNT(loop);
  for (i = VL_MRSTART(loop); i < end; ++i)
    for (j = VL_MRSTART(loop); j <= i; ++j)
      chkref(i, j);

  /* next, this loop against inner loops */
  end1 = VL_MRSTART(loop) + VL_MRECNT(loop);
  for (i = VL_MRSTART(loop); i < end; ++i)
    for (j = end; j < end1; ++j)
      chkref(i, j);

  /* finally, inner loops against other inner loops */
  for (l1 = VL_CHILD(loop); l1 != 0; l1 = VL_SIBLING(l1))
    for (l2 = VL_CHILD(loop); l2 != l1; l2 = VL_SIBLING(l2)) {
      end = VL_MRSTART(l1) + VL_MRECNT(l1);
      end1 = VL_MRSTART(l2) + VL_MRECNT(l2);
      for (i = VL_MRSTART(l1); i < end; ++i)
        for (j = VL_MRSTART(l2); j < end1; ++j)
          chkref(i, j);
    }

  FREE(fgsucc);
  FREE(fgbase);
  --ddinfo.n;
#if DEBUG
  if (DBGBIT(36, 8)) {
    fprintf(gbl.dbgfil, "----Mem ref after build_dd loop %d----\n", loop);
    dump_memrefs(VL_MRSTART(loop), VL_MRECNT(loop));
  }
#endif
}

/* assume mr1, mr2 in same fg node; return TRUE if mr1 comes before mr2 */
static LOGICAL
mr_precedes(int mr1, int mr2)
{
  int ilt1, ilt2;

  if (mr1 == mr2)
    return FALSE;
  ilt1 = MR_ILT(mr1);
  ilt2 = MR_ILT(mr2);
  if (ilt1 == ilt2) {
    /* if this is the same ilt, don't allow the store to precede the
     * load */
    if (MR_TYPE(mr1) != 'l') {
      assert(MR_TYPE(mr2) == 'l', "mr_precedes: too may st/br", mr2, 4);
      return FALSE;
    }
    return TRUE;
  }
  while (ilt1 != 0) {
    if (ilt1 == ilt2)
      return TRUE;
    ilt1 = ILT_NEXT(ilt1);
  }
  return FALSE;
}

static void
chkref(int i, int j)
{
  DIRVEC exo_dirvec_ji;
  DIRVEC exo_dirvec_ij;
  int fg1, fg2;
  DIRVEC vec;
  DV *p;

  if ((MR_TYPE(i) != 'l' && MR_TYPE(i) != 's') ||
      (MR_TYPE(j) != 'l' && MR_TYPE(j) != 's'))
    return;
  if (MR_TYPE(i) != 's' && MR_TYPE(j) != 's')
    return;

  STRACE2("mem ref j %d i %d\n", j, i);

  /*
   * now compute data dependence vector indicating when a
   * dependence can exist between memory references i and j
   */
  /* compute all possible dependence vectors */
  ddinfo.dvlist = NULL;
  ddinfo.mr1 = j;
  ddinfo.mr2 = i;
  dd_compute();

  /** compute execution orders from j to i */
  fg1 = MR_FG(j);
  fg2 = MR_FG(i);
  if (fg1 == fg2) {
    if (mr_precedes(j, i))
      exo_dirvec_ji = dirv_exo(ddinfo.n, TRUE);
    else
      exo_dirvec_ji = dirv_exo(ddinfo.n, FALSE);
  } else if (bv_mem(fgsucc[fg1], fg2))
    exo_dirvec_ji = dirv_exo(ddinfo.n, TRUE);
  else
    exo_dirvec_ji = dirv_exo(ddinfo.n, FALSE);

  /** compute execution orders from i to j */
  if (fg1 == fg2) {
    if (mr_precedes(i, j))
      exo_dirvec_ij = dirv_exo(ddinfo.n, TRUE);
    else
      exo_dirvec_ij = dirv_exo(ddinfo.n, FALSE);
  } else if (bv_mem(fgsucc[fg2], fg1))
    exo_dirvec_ij = dirv_exo(ddinfo.n, TRUE);
  else
    exo_dirvec_ij = dirv_exo(ddinfo.n, FALSE);

  /*
   * intersect the execution order direction vector with the
   * dependence direction vectors to determine true dependence
   * direction vector
   */

  for (p = ddinfo.dvlist; p != 0; p = p->next) {
    /* j precedes i */
    vec = p->vec & exo_dirvec_ji;
    if (!dirv_chkzero(vec, ddinfo.n))
      dd_edge(j, i, vec);

    /* i precedes j; must invert dependence vector */
    vec = dirv_inverse(p->vec) & exo_dirvec_ij;
    if (!dirv_chkzero(vec, ddinfo.n))
      dd_edge(i, j, vec);
  }
  freearea(HLV_AREA1);
}

static void
edge_func(DIRVEC vec)
{
  ddinfo.vec |= vec;
}

void
dd_edge(int src, int sink, DIRVEC vec)
{
  int type; /* type of dependence */
  DDEDGE *p;

  /* figure out type of dependence */
  /* dependence goes from src to sink */
  ddinfo.vec = 0;
  dirv_gen(vec, (int *)0, 0, DIRV_BIGPOS, edge_func);
  vec = ddinfo.vec;
  if (vec == 0)
    return;
  TRACE3("dd_edge: src %d sink %d vec %s \n", src, sink, dirv_print(vec));
  if (MR_TYPE(src) == 's' && MR_TYPE(sink) == 's')
    type = DIRV_FOUT;
  else if (MR_TYPE(src) == 'l' && MR_TYPE(sink) == 's') {
    type = DIRV_FANTI;
    /* under certain circumstances we can ignore anti dependences.
     * If they are from the load of a scalar to the store of the same
     * scalar, and the store has a loop-invariant RHS.  There must
     * also be only one assignment of the variable in the loop, and
     * the memory refs must be in the same ilt.
     * The theory is that it doesn't matter if we do the load before the
     * store, since the same value is always stored.
     */
    if (MR_NME(src) == MR_NME(sink) && MR_INVAL(sink) &&
        MR_ILT(src) == MR_ILT(sink)) {
      int def;
      int lp;
      int inloop;

      inloop = 0;
      for (def = NME_DEF(MR_NME(src)); def != 0; def = DEF_NEXT(def)) {
        /* is it in the loop? */
        for (lp = FG_LOOP(DEF_FG(def)); lp != 0; lp = LP_PARENT(lp))
          if (lp == ddinfo.common)
            goto in_loop;
        continue; /* not in loop */
      in_loop:
        if (inloop++)
          goto skip;
      }
      TRACE0("  Ignoring edge because inv scalar\n");
      return;
    }
  skip:;
  } else if (MR_TYPE(src) == 's' && MR_TYPE(sink) == 'l')
    type = DIRV_FFLOW;
  else {
    interr("unknown load/load dep in dd_edge", 0, 3);
    type = 3;
  }
  /* add to list for this mem ref */
  for (p = MR_SUCC(src); p != 0; p = DD_NEXT(p)) {
    if (DD_SINK(p) == sink && DD_TYPE(p) == type) {
      DD_DIRVEC(p) |= vec;
      return;
    }
  }
  /* add a new one */
  p = (DDEDGE *)getitem(HLV_AREA, sizeof(DDEDGE));
  DD_TYPE(p) = type;
  DD_DIRVEC(p) = vec;
  DD_SINK(p) = sink;
  DD_NEXT(p) = MR_SUCC(src);
  MR_SUCC(src) = p;
}

static void
add_dep(DIRVEC vec)
{
  DV *p;

  p = (DV *)getitem(HLV_AREA1, sizeof(DV));
  p->vec = vec;
  p->next = ddinfo.dvlist;
  ddinfo.dvlist = p;
}

static void
dd_compute(void)
{
  int nm1, nm2;
  int ui, loop, mr;

  STRACE4("dd_compute: mr1 %d mr2 %d common %d n %d\n", ddinfo.mr1, ddinfo.mr2,
          ddinfo.common, ddinfo.n);

  /* I believe that 'yuck' is the appropriate word here.  All this
   * deals with is initial values that the vectorizer added for induction
   * vars with non-invariant initial values.  Since the uses haven't
   * yet been replaced, artificial checks must be done for a dependence
   * that will be added.  While ugly, the code is probably correct and
   * reasonably efficient.
   */
  ui = FALSE;
  if (MR_IVUSE(ddinfo.mr1) && MR_INIT(ddinfo.mr2)) {
    loop = MR_LOOP(ddinfo.mr1);
    mr = ddinfo.mr2;
    if (VL_PREBIH(loop) == FG_TO_BIH(MR_FG(ddinfo.mr2)))
      ui = TRUE;
  } else if (MR_IVUSE(ddinfo.mr2) && MR_INIT(ddinfo.mr1)) {
    loop = MR_LOOP(ddinfo.mr2);
    mr = ddinfo.mr1;
    if (VL_PREBIH(loop) == FG_TO_BIH(MR_FG(ddinfo.mr1)))
      ui = TRUE;
  }
  if (ui) {
    /* initial value-use situation */
    int i;

    for (i = VL_IVLIST(loop); i != 0; i = MR_NEXT(i)) {
      if (MR_IV(i) == MR_IV(mr)) {
        add_dep(dirv_fulldep(ddinfo.n));
        return;
      }
    }
  }

  nm1 = MR_NME(ddinfo.mr1);
  while (NME_TYPE(nm1) == NT_ARR || NME_TYPE(nm1) == NT_MEM)
    nm1 = NME_NM(nm1);
  ddinfo.basenm1 = nm1;

  nm2 = MR_NME(ddinfo.mr2);
  while (NME_TYPE(nm2) == NT_ARR || NME_TYPE(nm2) == NT_MEM)
    nm2 = NME_NM(nm2);
  ddinfo.basenm2 = nm2;

  if (NME_TYPE(nm1) == NT_VAR && NME_TYPE(nm2) == NT_VAR) {
    if (nm1 != nm2) {
      int i;
      /*
       * for FORTRAN equivalence stmts...  return TRUE if one symbol is
       * in storage overlap chain of the other. Eventually we need to
       * be more clever here.
       */
      int s1 = basesym_of(nm1);
      int s2 = basesym_of(nm2);
      if (!VP_DEPCHK(ddinfo.pragmas) || !VP_EQVCHK(ddinfo.pragmas))
        return;
      if (SOCPTRG(s1)) {
        for (i = SOCPTRG(s1); i; i = SOC_NEXT(i))
          if (SOC_SPTR(i) == s2) {
            add_dep(dirv_fulldep(ddinfo.n));
            return;
          }
      }
      if ((POINTERG(s1) || POINTERG(s2)) &&
          expr_dependent(MR_ILI(ddinfo.mr2), MR_ILI(ddinfo.mr1),
                         MR_ILT(ddinfo.mr2), MR_ILT(ddinfo.mr1))) {
        add_dep(dirv_fulldep(ddinfo.n));
        return;
      }
      return;
    }
    resolve_vv();
  } else if ((NME_TYPE(nm1) == NT_IND && NME_TYPE(nm2) == NT_VAR) ||
             (NME_TYPE(nm1) == NT_VAR && NME_TYPE(nm2) == NT_IND))
    resolve_pv();
  else if (NME_TYPE(nm1) == NT_IND && NME_TYPE(nm2) == NT_IND)
    resolve_pp();
  else if ((NME_TYPE(nm1) == NT_UNK && NME_TYPE(nm2) == NT_IND) ||
           (NME_TYPE(nm1) == NT_IND && NME_TYPE(nm2) == NT_UNK) ||
           (NME_TYPE(nm1) == NT_UNK && NME_TYPE(nm2) == NT_UNK)) {
    if (!VP_DEPCHK(ddinfo.pragmas))
      return;
    add_dep(dirv_fulldep(ddinfo.n));
  } else if ((NME_TYPE(nm1) == NT_UNK && NME_TYPE(nm2) == NT_VAR) ||
             (NME_TYPE(nm1) == NT_VAR && NME_TYPE(nm2) == NT_UNK))
    resolve_uv();
  else
    assert(0, "dd_compute: can't happen", 0, 4);
}

static void
resolve_pp(void)
{
  /* for pointer-pointer, best we can do is the -x switches */
  open_pragma(BIH_LINENO(FG_TO_BIH(LP_HEAD(ddinfo.common))));
  if (is_ptr_safe(ddinfo.basenm1))
    return;
  if (is_ptr_safe(ddinfo.basenm2))
    return;
  close_pragma();
  if (!VP_DEPCHK(ddinfo.pragmas))
    return;
  if (ddinfo.basenm1 == ddinfo.basenm2 && MR_SUBST(ddinfo.mr1) &&
      MR_SUBST(ddinfo.mr2)) {
    /* do data dependence analysis */
    unpack_loops();
    do_subscript(1);
  } else
    add_dep(dirv_fulldep(ddinfo.n));
}

static void
resolve_pv(void)
{
  int ptr, var;

  /* use a combination of optimizer utilities & stolen optimizer code */
  if (NME_TYPE(ddinfo.basenm1) == NT_IND) {
    ptr = ddinfo.basenm1;
    var = ddinfo.basenm2;
  } else {
    ptr = ddinfo.basenm2;
    var = ddinfo.basenm1;
  }
  assert(NME_TYPE(ptr) == NT_IND && NME_TYPE(var) == NT_VAR,
         "resolve_pv: bad ptr/var", 0, 4);
  open_pragma(BIH_LINENO(FG_TO_BIH(LP_HEAD(ddinfo.common))));
  if (is_ptr_safe(ptr))
    return;
  if (is_sym_optsafe(var, ddinfo.common))
    return;
  close_pragma();
  add_dep(dirv_fulldep(ddinfo.n));
}

static void
resolve_uv(void)
{
  add_dep(dirv_fulldep(ddinfo.n));
}

static void
unpack_loops(void)
{
  int i, j;
  /* unpack the loops */
  i = 0;
  ddinfo.n1 = 0;
  for (j = MR_LOOP(ddinfo.mr1); j != ddinfo.common; j = LP_PARENT(j)) {
    ddinfo.lps[i++] = j;
    ++ddinfo.n1;
  }
  ddinfo.n2 = 0;
  for (j = MR_LOOP(ddinfo.mr2); j != ddinfo.common; j = LP_PARENT(j)) {
    ddinfo.lps[i++] = j;
    ++ddinfo.n2;
  }
  j = ddinfo.common;
  for (;;) {
    ddinfo.lps[i++] = j;
    if (j == ddinfo.outer_loop)
      break;
    j = LP_PARENT(j);
  }
  assert(i <= MAX_DD, "resolve_vv: bad i", ddinfo.common, 4);
  assert(i == ddinfo.n1 + ddinfo.n2 + ddinfo.n, "resolve_vv: bad iA",
         ddinfo.common, 4);
  ddinfo.subs1 = MR_SUBST(ddinfo.mr1) + MR_SUBCNT(ddinfo.mr1);
  ddinfo.subs2 = MR_SUBST(ddinfo.mr2) + MR_SUBCNT(ddinfo.mr2);
  DTRACE2("resolve_vv: subs1 %d subs2 %d\n", ddinfo.subs1, ddinfo.subs2);
}

static void
resolve_vv(void)
{
  int i1, i2;
  int nm1, nm2;
  DIRVEC dir;
  int buf1[MAXNMESIZ], buf2[MAXNMESIZ];
  int i;
  int j;

  /* check for both scalar */
  if (MR_SCALR(ddinfo.mr1) && MR_SCALR(ddinfo.mr2)) {
    /* complex have different nmes */
    assert(ddinfo.basenm1 == ddinfo.basenm2, "resolve_vv: mr1 != mr2",
           ddinfo.mr1, 4);
    /* expanded scalars are different */
    for (i = VL_SCLIST(ddinfo.common); i != 0; i = SCLR_NEXT(i))
      if (MR_NME(ddinfo.mr1) == SCLR_NME(i))
        goto found;
    add_dep(dirv_fulldep(ddinfo.n));
    return;
  found:
    /* equivalent to equals direction in last place */
    dir = DIRV_ALLEQ;
    for (i = 1; i < ddinfo.n; ++i)
      DIRV_ENTRYP(dir, i, DIRV_STAR);
    DIRV_ENTRYP(dir, 0, DIRV_EQ);
    add_dep(dir);
    return;
  }
  /* unpack names info into buf1, buf2 */
  i1 = i2 = MAXNMESIZ;
  nm1 = MR_NME(ddinfo.mr1);
  while (i1 > 0) {
    if (NME_TYPE(nm1) == NT_VAR)
      break;
    buf1[--i1] = nm1;
    nm1 = NME_NM(nm1);
  }
  /* too big? */
  if (NME_TYPE(nm1) != NT_VAR) {
    add_dep(dirv_fulldep(ddinfo.n));
    return;
  }

  nm2 = MR_NME(ddinfo.mr2);
  while (i2 > 0) {
    if (NME_TYPE(nm2) == NT_VAR)
      break;
    buf2[--i2] = nm2;
    nm2 = NME_NM(nm2);
  }
  /* too big? */
  if (NME_TYPE(nm2) != NT_VAR) {
    add_dep(dirv_fulldep(ddinfo.n));
    return;
  }

  unpack_loops();

  /* see if we can avoid any checking */
  i = 0;
  for (;;) {
    /* invariant: we have a common prefix of buf1 and buf2 */
    /* for example: a.b.c[...] */
    if (i1 + i >= MAXNMESIZ || i2 + i >= MAXNMESIZ)
      break;
    if (NME_TYPE(buf1[i1 + i]) == NT_MEM && NME_TYPE(buf2[i2 + i]) == NT_MEM) {
      if (NME_SYM(buf1[i1 + i]) != NME_SYM(buf2[i2 + i])) {
        ddinfo.dvlist = NULL;
        return;
      }
    }
    ++i;
  }
  i = 0;
  j = 0; /* count subscripts */
  for (;;) {
    /* all member references match */
    if (i1 + i >= MAXNMESIZ || i2 + i >= MAXNMESIZ)
      break;
    if (NME_TYPE(buf1[i1 + i]) == NT_MEM && NME_TYPE(buf2[i2 + i]) == NT_MEM) {
      assert(NME_SYM(buf1[i1 + i]) == NME_SYM(buf2[i2 + i]),
             "resolve_vv: diff MEM symbols", 0, 4);
      ++i;
    } else if (NME_TYPE(buf1[i1 + i]) == NT_ARR &&
               NME_TYPE(buf2[i2 + i]) == NT_ARR) {
      while (i1 + i < MAXNMESIZ && NME_TYPE(buf1[i1 + i]) == NT_ARR) {
        assert(NME_TYPE(buf2[i2 + i]) == NT_ARR, "resolve_vv: not both arrays",
               0, 4);
        assert(ASD_NDIM(NME_SUB(buf1[i1 + i])) ==
                   ASD_NDIM(NME_SUB(buf1[i1 + i])),
               "resolve_vv: ndims not equal", 0, 4);
        j += ASD_NDIM(NME_SUB(buf1[i1 + i]));
        ++i;
      }
    } else
      assert(0, "resolve_vv: can't happen", 0, 4);
  }
  do_subscript(j);
  ddinfo.subs1 -= j;
  ddinfo.subs2 -= j;
}

static int nvars;
static int neqns;
static int nfree;

#define UD xUD
static int UD[MAX_N][MAX_S + MAX_N];
static int T[MAX_N];
static int C[MAX_S];
static int Dist[MAX_LOOPS];
static int TU[MAX_S][MAX_N];

#define BOUND_LEN (2 * (MAX_N + 1))
static BOUND *Bound[MAX_N + 1][2];
static BOUND *SaveBound[MAX_N + 1][2];

#if DEBUG
static void
prmat(int col, int row)
{
  int i, j;

  fprintf(gbl.dbgfil, "Matrix Dump: %d, %d-----\n", col, row);
  for (i = 0; i < nvars; ++i) {
    for (j = 0; j < nvars + neqns; ++j) {
      prilitree(UD[i][j]);
      fprintf(gbl.dbgfil, "\t");
    }
    fprintf(gbl.dbgfil, "\n");
  }
}
#endif

static INT
gcd(INT a, INT b)
{
  INT u, v;
  INT r;

  u = a > 0 ? a : -a;
  v = b > 0 ? b : -b;
  while (v != 0) {
    r = u % v;
    u = v;
    v = r;
  }
  return u;
}

static int
ceilg(int t, int g)
{
  int q;
  int a;

  a = t;
  if (t < 0)
    a = -t;
  /* return ceil(t/g), g is positive */
  q = a / g;
  if (g * q == a) {
    if (t < 0)
      return -q;
    return q;
  }
  /* q is floor(a/g) */
  ddinfo.unknown = 1;
  if (t > 0)
    return q + 1;
  return -q;
}

static int
floorg(int t, int g)
{
  int q;
  int a;

  a = t;
  if (t < 0)
    a = -t;
  /* return floor(t/g), g is positive */
  q = a / g;
  if (g * q == a) {
    if (t < 0)
      return -q;
    return q;
  }
  /* q is floor(a/g) */
  ddinfo.unknown = 1;
  if (t > 0)
    return q;
  return -q - 1;
}

static LOGICAL
compare_one_bound(BOUND *low, BOUND *up, int var)
{
  int i, j;
  int tmp[MAX_N + 1];
  LOGICAL btype;
  int t, g, t1;
  BOUND b, *p;
  int icon0 = ad_icon(0);

#if DEBUG
  if (DBGBIT(36, 64)) {
    fprintf(gbl.dbgfil, "compare_one_bound: ");
    dump_two_bound(low, up, var, FALSE);
  }
#endif
  /* compute difference of bounds */
  for (j = 1; j < var; ++j) {
    tmp[j] = ili_symbolic(
        ad2ili(IL_ISUB, ad2ili(IL_IMUL, up->bnd[j], ad_icon(low->mplyr)),
               ad2ili(IL_IMUL, low->bnd[j], ad_icon(up->mplyr))));
  }
  /* compute difference of lower bounds */
  tmp[0] = ili_symbolic(ad2ili(
      IL_ISUB, ad2ili(IL_IMUL, up->bnd[0], ad_icon(low->mplyr * low->gcd)),
      ad2ili(IL_IMUL, low->bnd[0], ad_icon(up->mplyr * up->gcd))));
#if DEBUG
  if (DBGBIT(36, 64)) {
    fprintf(gbl.dbgfil, "compare_one_bound: up-low: ");
    for (j = 0; j < var; ++j) {
      prilitree(tmp[j]);
      fprintf(gbl.dbgfil, ", ");
    }
    fprintf(gbl.dbgfil, "\n");
  }
#endif
  /* tmp[0] is divided by product of gcds */
  /* find highest non-zero coefficient */
  for (j = var - 1; j > 0; --j)
    if (tmp[j] != icon0)
      break;
  if (j == 0) {
    /* check this one immediately */
    if (IS_CNST(tmp[0])) {
      if (CNSTG(tmp[0]) < 0) {
        DTRACE0(
            "compare_one_bound finds constant inconsistency, returns nodep\n");
        return TRUE;
      }
      return FALSE; /* constant consistent */
    }
    /* variable inconsistent, unknown data dep */
    ddinfo.unknown = 1;
    return FALSE;
  }
  /* This coefficient must be constant or we'll ignore this bound */
  /* since we don't know if it is upper or lower */
  if (!IS_CNST(tmp[j])) {
    DTRACE1("compare_one_bound ignores bound for %d\n", var);
    ddinfo.unknown = 1;
    return FALSE;
  }
  /* decide whether this is a lower or upper bound */
  t = CNSTG(tmp[j]);
  if (t > 0) {
    btype = FALSE; /* lower */
    for (i = 0; i < j; ++i)
      tmp[i] = ili_symbolic(ad1ili(IL_INEG, tmp[i]));
  } else {
    btype = TRUE;
    tmp[j] = ad1ili(IL_INEG, tmp[j]);
    t = -t;
  }
  /* compute gcd if possible */
  g = 1;
  if (t != 1) {
    g = 0;
    for (i = 1; i <= j; ++i) {
      if (!IS_CNST(tmp[i]))
        break;
      g = gcd(g, CNSTG(tmp[i]));
    }
    if (g != 0) {
      /* divide thru */
      t /= g;
      for (i = 1; i <= j; ++i) {
        tmp[i] = ad_icon(CNSTG(tmp[i]) / g);
      }
      /* try to handle 0th element */
      if (IS_CNST(tmp[0])) {
        t1 = CNSTG(tmp[0]);
        /* compute floor/ceil ( t1 / g) */
        if (btype)
          t1 = floorg(t1, g);
        else
          t1 = ceilg(t1, g);
        tmp[0] = ad_icon(t1);
        g = 1;
      }
    } else
      g = 1;
  }
  b.gcd = up->gcd * low->gcd * g;
  b.mplyr = t;
  for (i = 0; i < j; ++i)
    b.bnd[i] = tmp[i];

#if DEBUG
  if (DBGBIT(36, 64)) {
    fprintf(gbl.dbgfil, "compare_one_bound: new bound: ");
    dump_one_bound(&b, j, btype);
  }
#endif
  /* add this one */
  for (p = Bound[j][btype]; p != 0; p = p->next) {
    if (b.mplyr == p->mplyr && b.gcd == p->gcd) {
      for (i = 0; i < j; ++i)
        if (p->bnd[i] != tmp[i])
          goto cont;
      goto found;
    }
  cont:;
  }
  p = (BOUND *)getitem(HLV_AREA1, sizeof(BOUND));
  *p = b;
  p->next = Bound[j][btype];
  Bound[j][btype] = p;
#if DEBUG
  if (DBGBIT(36, 16)) {
    fprintf(gbl.dbgfil, "Add bound---:");
    dump_one_bound(p, j, btype);
  }
#endif
found:
  return FALSE;
}

/* ibound = bound in terms of index variables */
/* upflag = true if upper bound */
/* var = index var for which this is a bound */
static int
bound_add(int *ibound, LOGICAL upflag, int var)
{
  BOUND b1;
  BOUND b2;
  int j, k;
  int icon0 = ad_icon(0);

  /* express the bound in terms of the free variables */
  /* create a linear combination of bounds on free variables */
  BTRACE4("bound_add: nvars: %d, nfree: %d, upflag: %d, var: %d\n", nvars,
          nfree, upflag, var);
  for (j = 0; j <= nfree; ++j) {
    b1.bnd[j] = icon0;
  }
  b1.bnd[0] = ibound[0];
  for (j = 1; j <= nvars; ++j) {
    /* b1.bnd += bound[j] * TU(*,j) */
    for (k = 0; k <= nfree; ++k) {
      b1.bnd[k] = ili_symbolic(
          ad2ili(IL_IADD, b1.bnd[k], ad2ili(IL_IMUL, ibound[j], TU[k][j - 1])));
    }
  }
  /* construct b2 */
  for (k = 0; k <= nfree; ++k)
    b2.bnd[k] = TU[k][var];
  b1.gcd = b2.gcd = 1;
  b1.mplyr = b2.mplyr = 1;
  if (upflag) {
    /* TU(*,var) <= tmp1 */
    return compare_one_bound(&b2, &b1, nfree + 1);
  } else
    return compare_one_bound(&b1, &b2, nfree + 1);
}

static LOGICAL
check_bounds(void)
{
  int k;
  BOUND *p, *q;

  /* compare bounds from nfree down */
  for (k = nfree; k >= 1; --k) {
    for (p = Bound[k][1]; p != 0; p = p->next)
      for (q = Bound[k][0]; q != 0; q = q->next) {
        /* compare p and q , q <= p */
        if (compare_one_bound(q, p, k))
          return TRUE;
      }
  }
  return FALSE;
}

static LOGICAL
bnd_le(int k, LOGICAL eqflag)
{
  int i;
  BOUND b1, b2;

  for (i = 0; i <= nfree; ++i) /* ik */
    b1.bnd[i] = TU[i][2 * k];
  if (!eqflag) /* strict inequality */
    b1.bnd[0] = ili_symbolic(ad2ili(IL_IADD, b1.bnd[0], ad_icon(1)));
  b1.gcd = b1.mplyr = 1;
  for (i = 0; i <= nfree; ++i) /* jk */
    b2.bnd[i] = TU[i][2 * k + 1];
  b2.gcd = b2.mplyr = 1;
  return compare_one_bound(&b1, &b2, nfree + 1);
}

static LOGICAL
bnd_ge(int k, LOGICAL eqflag)
{
  int i;
  BOUND b1, b2;

  for (i = 0; i <= nfree; ++i) /* ik */
    b1.bnd[i] = TU[i][2 * k];
  b1.gcd = b1.mplyr = 1;
  for (i = 0; i <= nfree; ++i) /* jk */
    b2.bnd[i] = TU[i][2 * k + 1];
  if (!eqflag)
    b2.bnd[0] = ili_symbolic(ad2ili(IL_IADD, b2.bnd[0], ad_icon(1)));
  b2.gcd = b2.mplyr = 1;
  return compare_one_bound(&b2, &b1, nfree + 1);
}

/* check for dependence with direction vector dir */
static LOGICAL
check_new_bound(DIRVEC dir, int lev)
{
  int k;
  DIRVEC d;

  BCOPY(Bound, SaveBound, BOUND *, BOUND_LEN);
  /* create equations describing this direction vector */
  for (k = 0; k < lev; ++k) {
    d = DIRV_ENTRYG(dir, lev - k - 1);
    switch (d) {
    case DIRV_STAR:
      break;
    case DIRV_LT:
      /* ik < jk --> ik+1 <= jk */
      if (bnd_le(k, FALSE))
        return TRUE;
      break;
    case DIRV_EQ:
      /* simulate with <=, >= */
      /* ik >= jk and ik <= jk */
      if (bnd_le(k, TRUE))
        return TRUE;
      if (bnd_ge(k, TRUE))
        return TRUE;
      break;
    case DIRV_GT:
      /* ik > jk --> ik >= jk+1 */
      if (bnd_ge(k, FALSE))
        return TRUE;
      break;
    }
  }
  return check_bounds();
}

static void
do_subscript(int nsubs)
{
  /*
   * this function is the heart of the data dependence analysis. subs1 and
   * subs2 point to subscript entries for the same array. do_subscript
   * determines a direction vector under which those subscripts will
   * intersect.
   */
  int i, j, k, k1, bnd, n1;
  int q, d1, d2, sgn, sc, ili;
  int d;
  int t;
  DIRVEC vec, vec1;
  int icon0, icon1, iconneg1;
  int sub;
  int invar1, invar2;
  int lp;
  int tmp[MAX_N + 1];

  if (nsubs <= 0)
    goto give_up;
  icon0 = ad_icon(0L);
  icon1 = ad_icon(1L);
  iconneg1 = ad_icon(-1L);
  /* quick check for non-intersecting loop-invariant subscripts */
  /* also find out how many outer loops to remove */
  /* Set d to number of outer loops to consider based on
   * how far we were able to analyze both subscripts. */
  d = 0x7FFFFFFF;
  for (sub = 1; sub <= nsubs; ++sub) {
    k = ddinfo.n1 + ddinfo.n;
    for (i = 0; i < k; ++i) {
      if (SB_STRIDE(ddinfo.subs1 - sub)[i] == 0) {
        break;
      }
      if (SB_STRIDE(ddinfo.subs1 - sub)[i] < 0 ||
          SB_STRIDE(ddinfo.subs1 - sub)[i] >= astb.stg_avail) {
        fprintf(stderr, "sub=%d,ddinfo.subs1=%d,i=%d,SB_STRICE(%d)[%d]=%d\n",
                sub, ddinfo.subs1, i, ddinfo.subs1 - sub, i,
                SB_STRIDE(ddinfo.subs1 - sub)[i]);
      }
    }
    if (d > i - ddinfo.n1)
      d = i - ddinfo.n1;
    k = ddinfo.n2 + ddinfo.n;
    for (i = 0; i < k; ++i) {
      if (SB_STRIDE(ddinfo.subs2 - sub)[i] == 0) {
        break;
      }
      if (SB_STRIDE(ddinfo.subs2 - sub)[i] < 0 ||
          SB_STRIDE(ddinfo.subs2 - sub)[i] >= astb.stg_avail) {
        fprintf(stderr, "sub=%d,ddinfo.subs2=%d,i=%d,SB_STRICE(%d)[%d]=%d\n",
                sub, ddinfo.subs2, i, ddinfo.subs2 - sub, i,
                SB_STRIDE(ddinfo.subs2 - sub)[i]);
      }
    }
    if (d > i - ddinfo.n2)
      d = i - ddinfo.n2;
  }
  if (d <= 0)
    goto give_up;
  for (sub = 1; sub <= nsubs; ++sub) {
    invar1 = TRUE;
    k = ddinfo.n1 + d;
    for (i = 0; i < k; ++i)
      if (SB_STRIDE(ddinfo.subs1 - sub)[i] != icon0) {
        invar1 = FALSE;
        break;
      }
    invar2 = TRUE;
    k = ddinfo.n2 + d;
    for (i = 0; i < k; ++i)
      if (SB_STRIDE(ddinfo.subs2 - sub)[i] != icon0) {
        invar2 = FALSE;
        break;
      }
    if (invar1 && invar2) {
      d1 = SB_BASES(ddinfo.subs1 - sub)[ddinfo.n1 + d];
      d2 = SB_BASES(ddinfo.subs2 - sub)[ddinfo.n2 + d];
      if (A_TYPEG(d1) == A_TRIPLE || A_DTYPEG(d2) == A_TRIPLE) {
        /* get lower bound from d1, d2 */
        int d1low, d1high, d2low, d2high;
        if (A_TYPEG(d1) != A_TRIPLE) {
          d1low = d1high = d1;
        } else {
          d1low = A_LBDG(d1);
          d1high = A_UPBDG(d1);
          if (A_STRIDEG(d1)) {
            int d1str;
            d1str = ili_symbolic(A_STRIDEG(d1));
            if (!IS_CNST(d1str)) {
              goto give_up;
            } else if (CNSTG(d1str) < 0) {
              int t;
              t = d1low;
              d1low = d1high;
              d1high = t;
            }
          }
        }
        if (A_TYPEG(d2) != A_TRIPLE) {
          d2low = d2high = d2;
        } else {
          d2low = A_LBDG(d2);
          d2high = A_UPBDG(d2);
          if (A_STRIDEG(d2)) {
            int d2str;
            d2str = ili_symbolic(A_STRIDEG(d2));
            if (!IS_CNST(d2str)) {
              goto give_up;
            } else if (CNSTG(d2str) < 0) {
              int t;
              t = d2low;
              d2low = d2high;
              d2high = t;
            }
          }
        }
        if (IS_CNST(d1low) && IS_CNST(d2high) && CNSTG(d1low) > CNSTG(d2high)) {
          goto no_dep;
        }
        if (IS_CNST(d2low) && IS_CNST(d1high) && CNSTG(d2low) > CNSTG(d1high)) {
          goto no_dep;
        }
        continue;
      }
      if (RESG(d1) != RESG(d2))
        goto no_dep;
      if (IS_IRES(d1))
        t = ad2ili(IL_ISUB, d1, d2);
      else if (IS_ARES(d1))
        t = ad2ili(IL_ASUB, d1, d2);
      else
        interr("do_subscript: unknown base type for strides", ddinfo.common, 4);
      t = ili_symbolic(t);
      if (IS_CNST(t) && CNSTG(t) != 0L) {
        goto no_dep;
      }
    }
  }

  DTRACE2("do_subscript: %d subs, # common loops = %d\n", nsubs, d);
  /* setup the equations */
  /* there are 2*d + ddinfo.n1 + ddinfo.n2 variables */
  nvars = 2 * d + ddinfo.n1 + ddinfo.n2;
  neqns = nsubs;
  for (sub = 1; sub <= nsubs; ++sub) {
    /* first set up coefficients for common loops */
    j = 0;
    for (i = 0; i < d; ++i) {
      /* subscript for A */
      UD[j][sub + nvars - 1] =
          SB_STRIDE(ddinfo.subs1 - sub)[ddinfo.n1 + d - i - 1];
      ++j;
      UD[j][sub + nvars - 1] =
          ad1ili(IL_INEG, SB_STRIDE(ddinfo.subs2 - sub)[ddinfo.n2 + d - i - 1]);
      ++j;
    }
    /* now set up coefficients for first loop */
    for (i = 0; i < ddinfo.n1; ++i) {
      UD[j][sub + nvars - 1] = SB_STRIDE(ddinfo.subs1 - sub)[ddinfo.n1 - i - 1];
      ++j;
    }
    /* now set up coefficients for second loop */
    for (i = 0; i < ddinfo.n2; ++i) {
      UD[j][sub + nvars - 1] =
          ad1ili(IL_INEG, SB_STRIDE(ddinfo.subs2 - sub)[ddinfo.n2 - i - 1]);
      ++j;
    }
    assert(j == nvars, "do_subscript: not enough vars", 0, 4);
  }
  /* set up identity matrix */
  for (i = 0; i < nvars; ++i) {
    for (j = 0; j < nvars; ++j) {
      UD[i][j] = icon0;
    }
    UD[i][i] = icon1;
  }
  /* set up rhs */
  for (sub = 1; sub <= nsubs; ++sub) {
    d1 = SB_BASES(ddinfo.subs1 - sub)[ddinfo.n1 + d];
    d2 = SB_BASES(ddinfo.subs2 - sub)[ddinfo.n2 + d];
    if (RESG(d1) != RESG(d2))
      goto no_dep;
    if (A_TYPEG(d1) == A_TRIPLE || A_DTYPEG(d2) == A_TRIPLE) {
      continue;
    }
    if (IS_IRES(d1))
      t = ad2ili(IL_ISUB, d2, d1);
    else if (IS_ARES(d1))
      t = ad2ili(IL_ASUB, d2, d1);
    else
      interr("do_subscript: unknown base type for constants", ddinfo.common, 4);
    C[sub - 1] = ili_symbolic(t);
  }
#if DEBUG
  if (DBGBIT(36, 64)) {
    prmat(0, 0);
    fprintf(gbl.dbgfil, "RHS: ");
    for (i = 0; i < neqns; ++i) {
      fprintf(gbl.dbgfil, "\t%d: ", i);
      prilitree(C[i]);
      fprintf(gbl.dbgfil, "\n");
    }
  }
#endif

  /* Gaussian elimination on the matrix */
  bnd = neqns;
  if (bnd > nvars)
    bnd = nvars;
  n1 = neqns;
  j = 0;
  k1 = 0;
  while (k1 < bnd) {
    i = nvars - 1;
    while (i > k1) {
      if (UD[i][j + nvars] == icon0) {
        --i;
        continue;
      }
      /* work on rows i, i-1 */
      d1 = UD[i][j + nvars];
      d2 = UD[i - 1][j + nvars];
      /* check if d1 == d2 or d1 == -d2 */
      t = ad2ili(IL_ISUB, d1, d2);
      t = ili_symbolic(t);
      if (t == icon0) {
        /* d1 == d2, so sgn == 1, so q == 1 */
        q = icon1;
      } else {
        t = ad2ili(IL_IADD, d1, d2);
        t = ili_symbolic(t);
        if (t == icon0) {
          /* d1 == -d2, so sgn == -1, so q = -1 */
          q = ad_icon((INT)-1);
        } else {
          /* they had better both be constants */
          if (!IS_CNST(d1) || !IS_CNST(d2)) {
            if (!symbolic_divide(d2, d1, &q))
              goto give_up;
          } else {
            d1 = CNSTG(d1);
            d2 = CNSTG(d2);
            sgn = d1 * d2;
            if (d1 < 0)
              d1 = -d1;
            if (d2 < 0)
              d2 = -d2;
            q = d2 / d1;
            if (sgn < 0)
              q = -q;
            q = ad_icon(q);
          }
        }
      }

      if (q != icon0) {
        /* saxpy */
        for (k = 0; k < nvars + neqns; ++k)
          UD[i - 1][k] = ili_symbolic(
              ad2ili(IL_ISUB, UD[i - 1][k], symbolic_mul(q, UD[i][k])));
      }
      /* interchange */
      for (k = 0; k < neqns + nvars; ++k) {
        q = UD[i - 1][k];
        UD[i - 1][k] = UD[i][k];
        UD[i][k] = q;
      }
    }
    if (UD[k1][j + nvars] == icon0) {
      /* This column is linear combination of prev. cols */
      ++j;
      --n1;
      if (bnd > n1)
        bnd = n1;
      continue;
    }
    ++j;
    ++k1;
  }
#if DEBUG
  if (DBGBIT(36, 64))
    prmat(j, i);
#endif

  /* eliminate linear dependent columns (0 on the diagonal) from D */
  bnd = neqns;
  j = 0;
  while (j < bnd) {
    if (UD[j][nvars + j] != icon0) {
      j++;
      continue;
    }
    /* 0 found in D[j][j] */
    bnd--; /* decrement # of columns of D */
    for (i = j; i < bnd; i++) {
      for (k = 0; k < nvars; k++)
        /* copy D[k][(i+1)..(bnd+1)] over D[k][i..bnd] */
        UD[k][nvars + i] = UD[k][nvars + i + 1];
      C[i] = C[i + 1];
    }
  }
  neqns = bnd;

  /* solve tD=C and get distances */
  bnd = nvars;
  if (bnd > neqns)
    bnd = neqns;
  /* solve TD = C */
  for (j = 0; j < bnd; j++) {
    int djj;
    sc = icon0;
    /* sc = T[0..(j-1)] * D[0..(j-1)][j] */
    for (i = 0; i < j; i++) {
      ili = ad2ili(IL_IMUL, T[i], UD[i][nvars + j]);
      ili = ad2ili(IL_IADD, sc, ili);
      sc = ili_symbolic(ili);
    }
    /* t = C[j] - sc */
    ili = ad2ili(IL_ISUB, C[j], sc);
    t = ili_symbolic(ili);
    djj = UD[j][j + nvars];
    /* T[j] = t / D[j][j] */
    if (djj == icon1)
      T[j] = t;
    else if (djj == iconneg1)
      T[j] = ad1ili(IL_INEG, t);
    else if (t == icon0 && (XBIT(2, 0x200) || IS_CNST(djj)))
      /* 0 / x == 0 */
      T[j] = icon0;
    else if (t == djj && (XBIT(2, 0x200) || IS_CNST(djj)))
      /* x / x == 1 */
      T[j] = icon1;
    else if (IS_CNST(djj) && IS_CNST(t)) {
      assert(djj != icon0, "do_subscript: linear combo not removed", 0, 4);
      d1 = CNSTG(djj);
      d2 = CNSTG(t);
      if (d2 % d1 != 0) {
        DTRACE0("do_subscript: inconsistent eqns, no dep\n");
        goto no_dep;
      }
      T[j] = ad_icon(d2 / d1);
    } else
      goto give_up;
  }
  /* if system is overdetermined, check for consistency */
  for (j = nvars; j < neqns; j++) {
    sc = icon0;
    /* sc = T[0..(nvars-1)] * D[0..(nvars-1)][j] */
    for (i = 0; i < nvars; i++) {
      ili = ad2ili(IL_IMUL, T[i], UD[i][nvars + j]);
      ili = ad2ili(IL_IADD, sc, ili);
      sc = ili_symbolic(ili);
    }
    /* t = sc - C[j] */
    ili = ad2ili(IL_ISUB, sc, C[j]);
    t = ili_symbolic(ili);
    if (IS_CNST(t) && t != icon0) {
      DTRACE0("do_subscript: inconsistent overdetermined eqns, no dep\n");
      goto no_dep;
    }
  }
  if (neqns > nvars)
    /* system is consistent; ignore the last neqns - nvars columns of UD */
    neqns = nvars;

#if DEBUG
  if (DBGBIT(36, 64)) {
    prmat(0, 0);
    fprintf(gbl.dbgfil, "Solution:");
    for (j = 0; j < bnd; ++j) {
      fprintf(gbl.dbgfil, "\t%d: ", j);
      prilitree(T[j]);
      fprintf(gbl.dbgfil, "\n");
    }
  }
#endif

  /* check for constant dependence distances */
  /* actually check only in common loops */
  for (j = 0; j < ddinfo.n; ++j)
    Dist[j] = 0;
  /* actually, only check in the common loops
   * that we are counting, that is, up to nest 'd' */
  for (j = 0; j < d; ++j) {
    for (k = neqns; k < nvars; ++k) {
      if (UD[k][2 * j + 1] != UD[k][2 * j])
        goto skip;
    }
    /* distance is constant for this loop */
    sc = icon0;
    for (k = 0; k < neqns; ++k) {
      sc = ili_symbolic(ad2ili(
          IL_IADD, sc, ad2ili(IL_IMUL, T[k], ad2ili(IL_ISUB, UD[k][2 * j + 1],
                                                    UD[k][2 * j]))));
    }
    Dist[j] = sc;
  skip:;
  }
#if DEBUG
  if (DBGBIT(36, 16)) {
    fprintf(gbl.dbgfil, "Distance:\n");
    for (j = 0; j < ddinfo.n; ++j) {
      fprintf(gbl.dbgfil, "%d: ", j);
      if (Dist[j] == 0)
        fprintf(gbl.dbgfil, "<undef>\n");
      else {
        prilitree(Dist[j]);
        fprintf(gbl.dbgfil, "\n");
      }
    }
  }
#endif
  nfree = nvars - neqns;
  assert(nfree >= 0, "do_subscript: negative nfree", 0, 4);

  /* Extended GCD test is done.  Now: */
  /* 1. Express index variables in terms of free variables by
   *    multiplying tU.  This involves creating a matrix with one
   *    column for each index variable and one row for each free
   *    variable, plus one extra row to hold the constant coefficient
   *    derived from the already solved variables in T
   */
  for (i = 0; i < nvars; ++i) {
    /* do const part */
    sc = icon0;
    for (j = 0; j < neqns; ++j)
      sc = ili_symbolic(ad2ili(IL_IADD, sc, ad2ili(IL_IMUL, T[j], UD[j][i])));
    TU[0][i] = sc;
    /* do var part */
    for (j = neqns + 1; j <= nvars; ++j)
      TU[j - neqns][i] = UD[j - 1][i];
  }
#if DEBUG
  if (DBGBIT(36, 64)) {
    fprintf(gbl.dbgfil, "Solution of index vars:\n");
    for (i = 0; i < nvars; ++i) {
      fprintf(gbl.dbgfil, "I%d= ", i);
      for (j = 0; j <= nfree; ++j) {
        fprintf(gbl.dbgfil, "\t");
        prilitree(TU[j][i]);
        if (j) {
          fprintf(gbl.dbgfil, "*h%d", j);
        }
      }
      fprintf(gbl.dbgfil, "\n");
    }
  }
#endif
  /* 2. Derive a list of upper and lower bounds from the loop limits.
   *    Express the loop bounds in terms of the index variables, then
   *    impose the bounds on the appropriate column, then simplify.
   */
  /* A bound is a linear combination of index variables */
  for (i = 0; i <= nfree; ++i) {
    Bound[i][0] = Bound[i][1] = 0;
  }
/* common loops */
#define BOUND_ADD(t, l, i) \
  if (bound_add(t, l, i))  \
    goto no_dep;           \
  else
  for (i = 0; i < d; i++) {
    /* get the loop */
    lp = ddinfo.lps[ddinfo.n1 + ddinfo.n2 + d - i - 1];
    /* this loop derives bounds for variable 2*i and 2*i+1 */
    /* same as column 2*i and column 2*i+1 */
    /* see if we have to ignore some bounds */
    for (j = 0; j < i; ++j)
      if (SB_STRIDE(VL_LBND(lp))[j] == 0) {
        BTRACE1("Ignoring lower bound for loop %d\n", lp);
        goto skipl1;
      }

    /*----- get lower bound for I */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_LBND(lp))[i]; /* const part */
    for (j = 0; j < i; ++j) {
      /* express the bound in terms of the index variables */
      tmp[2 * j + 1] = SB_STRIDE(VL_LBND(lp))[i - j - 1];
    }
    BOUND_ADD(tmp, FALSE, 2 * i);

    /*----- get lower bound for J */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_LBND(lp))[i]; /* const part */
    for (j = 0; j < i; ++j) {
      /* express the bound in terms of the index variables */
      tmp[2 * j + 2] = SB_STRIDE(VL_LBND(lp))[i - j - 1];
    }
    BOUND_ADD(tmp, FALSE, 2 * i + 1);

  skipl1:
    for (j = 0; j < i; ++j)
      if (SB_STRIDE(VL_UBND(lp))[j] == 0) {
        BTRACE1("Ignoring upper bound for loop %d\n", lp);
        goto skipu1;
      }
    /*----- get upper bound for I */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_UBND(lp))[i]; /* const part */
    for (j = 0; j < i; ++j) {
      /* express the bound in terms of the index variables */
      tmp[2 * j + 1] = SB_STRIDE(VL_UBND(lp))[i - j - 1];
    }
    BOUND_ADD(tmp, TRUE, 2 * i);

    /*----- get upper bound for J */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_UBND(lp))[i]; /* const part */
    for (j = 0; j < i; ++j) {
      /* express the bound in terms of the index variables */
      tmp[2 * j + 2] = SB_STRIDE(VL_UBND(lp))[i - j - 1];
    }
    BOUND_ADD(tmp, TRUE, 2 * i + 1);
  skipu1:;
  }

  for (i = 0; i < ddinfo.n1; ++i) {
    /* get the loop */
    lp = ddinfo.lps[ddinfo.n1 - i - 1];
    /* this loop derives bounds for variable 2*d+i */
    for (j = 0; j < i + d; ++j)
      if (SB_STRIDE(VL_LBND(lp))[j] == 0) {
        BTRACE1("Ignoring lower bound for loop %d\n", lp);
        goto skipl2;
      }

    /*----- get lower bound for I */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_LBND(lp))[d + i];
    for (j = 0; j < d; ++j)
      tmp[2 * j + 1] = SB_STRIDE(VL_LBND(lp))[d + i - j - 1];
    for (j = d; j < d + i; ++j)
      tmp[d + j + 1] = SB_STRIDE(VL_LBND(lp))[d + i - j - 1];
    BOUND_ADD(tmp, FALSE, 2 * d + i);

  skipl2:
    for (j = 0; j < i + d; ++j)
      if (SB_STRIDE(VL_UBND(lp))[j] == 0) {
        BTRACE1("Ignoring upper bound for loop %d\n", lp);
        goto skipu2;
      }
    /*----- get upper bound for I */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_UBND(lp))[d + i];
    for (j = 0; j < d; ++j)
      tmp[2 * j + 1] = SB_STRIDE(VL_UBND(lp))[d + i - j - 1];
    for (j = d; j < d + i; ++j)
      tmp[d + j + 1] = SB_STRIDE(VL_UBND(lp))[d + i - j - 1];
    BOUND_ADD(tmp, TRUE, 2 * d + i);

  skipu2:;
  }
  for (i = 0; i < ddinfo.n2; ++i) {
    /* get the loop */
    lp = ddinfo.lps[ddinfo.n1 + ddinfo.n2 - i - 1];
    /* this loop derives bounds for variable 2*d+ddinfo.n1+i */

    for (j = 0; j < i + d; ++j)
      if (SB_STRIDE(VL_LBND(lp))[j] == 0) {
        BTRACE1("Ignoring lower bound for loop %d\n", lp);
        goto skipl3;
      }
    /*----- get lower bound for J */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_LBND(lp))[d + i];
    for (j = 0; j < d; ++j)
      tmp[2 * j + 2] = SB_STRIDE(VL_LBND(lp))[d + i - j - 1];
    for (j = d; j < d + i; ++j)
      tmp[d + ddinfo.n1 + j + 1] = SB_STRIDE(VL_LBND(lp))[d + i - j - 1];
    BOUND_ADD(tmp, FALSE, 2 * d + ddinfo.n1 + i);

  skipl3:
    for (j = 0; j < i + d; ++j)
      if (SB_STRIDE(VL_UBND(lp))[j] == 0) {
        BTRACE1("Ignoring upper bound for loop %d\n", lp);
        goto skipu3;
      }
    /*----- get upper bound for J */
    for (j = 1; j <= nvars; ++j)
      tmp[j] = icon0;
    tmp[0] = SB_BASES(VL_UBND(lp))[d + i];
    for (j = 0; j < d; ++j)
      tmp[2 * j + 1] = SB_STRIDE(VL_UBND(lp))[d + i - j - 1];
    for (j = d; j < d + i; ++j)
      tmp[d + ddinfo.n1 + j + 1] = SB_STRIDE(VL_UBND(lp))[d + i - j - 1];
    BOUND_ADD(tmp, TRUE, 2 * d + ddinfo.n1 + i);

  skipu3:;
  }

  /* 4. Direction vector hierarchy if dependent */
  vec = dirv_fulldep(ddinfo.n) & ~DIRV_ALLEQ;
  vec1 = 0;
  for (i = d; i < ddinfo.n; ++i)
    DIRV_ENTRYP(vec1, i, DIRV_STAR);
  hierarchy(vec, d, vec1);
  return;

give_up:
  if (VP_DEPCHK(ddinfo.pragmas))
    add_dep(dirv_fulldep(ddinfo.n));
no_dep:
  return;
}

static void
hierarchy(DIRVEC dir, int lev, DIRVEC veco)
{
  int i;
  int dir2;
  LOGICAL top;

  DTRACE2("hierarchy: dir 0x%lx -- %s\n", dir, dirv_print(dir));

  for (i = 0; i < lev; ++i)
    if (DIRV_ENTRYG(dir, i) != DIRV_STAR)
      break;
  if (i == lev)
    /* all * */
    top = TRUE;
  else
    top = FALSE;
  for (i = 0; i < lev; ++i)
    if (DIRV_ENTRYG(dir, i) == DIRV_STAR)
      goto test;
  goto bottom;

test:
  /*
   * test dependence at this level; if no dependence, don't need to go any
   * further: if (no dependence with direction vector dir) return 0;
   */
  if (top) {
    if (check_bounds() == TRUE)
      return;
    BCOPY(SaveBound, Bound, BOUND *, BOUND_LEN);
  } else {
    if (check_new_bound(dir, lev) == TRUE)
      return;
  }

  /* '*' entry, refine it further */
  dir2 = dir;
  DIRV_ENTRYC(dir2, i);
  DIRV_ENTRYP(dir2, i, DIRV_LT);
  hierarchy(dir2, lev, veco);

  dir2 = dir;
  DIRV_ENTRYC(dir2, i);
  DIRV_ENTRYP(dir2, i, DIRV_EQ);
  hierarchy(dir2, lev, veco);

  dir2 = dir;
  DIRV_ENTRYC(dir2, i);
  DIRV_ENTRYP(dir2, i, DIRV_GT);
  hierarchy(dir2, lev, veco);

  return;

bottom:
  /*
   * we're at the bottom level.  Test dependence with this direction vector
   * & return 0 or this direction vector.  In addition, need to set the
   * all-equals bit if this direction vector is all equals
   */
  ddinfo.unknown = 0;
  if (check_new_bound(dir, lev) == TRUE)
    return;
  if (ddinfo.unknown && !VP_DEPCHK(ddinfo.pragmas))
    return;
  for (i = 0; i < lev; ++i)
    if (DIRV_ENTRYG(dir, i) != DIRV_EQ) {
      add_dep(veco | dir);
      return;
    }
  add_dep(veco | dir | DIRV_ALLEQ);
}

/** \brief Invert a direction vector */
DIRVEC
dirv_inverse(DIRVEC vec)
{
  static DIRVEC revtab[8] = {
      /* >=< */    /* >=< */
      /* 000 */ 0, /* 000 */
      /* 001 */ 4, /* 100 */
      /* 010 */ 2, /* 010 */
      /* 011 */ 6, /* 110 */
      /* 100 */ 1, /* 001 */
      /* 101 */ 5, /* 101 */
      /* 110 */ 3, /* 011 */
      /* 111 */ 7, /* 111 */
  };

  int i;
  int t;
  DIRVEC vec1;

  /* count number of entrys */
  for (i = 0; DIRV_ENTRYG(vec, i) != 0 && i < MAX_LOOPS; ++i)
    ;
  /* extract information portion */
  vec1 = DIRV_INFOPART(vec);
  for (--i; i >= 0; --i) {
    t = DIRV_ENTRYG(vec, i);
    DIRV_ENTRYP(vec1, i, revtab[t]);
  }
  return vec1;
}

/** \brief Generate full dependence at a given level */
DIRVEC
dirv_fulldep(int level)
{
  int i;
  DIRVEC vec;

  vec = 0;
  for (i = 0; i < level; ++i) {
    vec <<= DIRV_ENTSIZ;
    vec |= DIRV_STAR;
  }
  vec |= DIRV_ALLEQ;
  return vec;
}

/** \brief Generate legal execution order direction vectors.
    \param level loop nesting level
    \param flag  says whether all '=' allowed
 */
DIRVEC
dirv_exo(int level, int flag)
{
  int i;
  DIRVEC vec;

  /* outermost loop: legal vectors are '<='; for rest they are '*' */
  /* alleq is only possible if flag is set */
  vec = 0;
  DIRV_ENTRYP(vec, level - 1, DIRV_EQ | DIRV_LT);
  i = level - 1;
  while (i > 0) {
    DIRV_ENTRYP(vec, i - 1, DIRV_STAR);
    --i;
  }
  if (flag)
    vec |= DIRV_ALLEQ;
  return vec;
}

/*
 * Return the direction vector that corresponds to the input
 * direction vector dir under the mapping m.  If m is NULL, the
 * trivial mapping is used.  A mapping is a permutation of the integers
 * 0..nest-1 representing the order of the loops as permuted from
 * their original order, from inner to outer.
 * The trivial mapping is thus nest-1, ..., 0
 */
static DIRVEC
dirv_permute(DIRVEC dir, int *m, int nest)
{
  int i, j, k;
  DIRVEC e;
  int seen_lt;
  DIRVEC rdir;
  int base;

  for (i = 0; DIRV_ENTRYG(dir, i) != 0 && i < MAX_LOOPS; ++i)
    ;
  base = nest - i;
#if DEBUG
  assert(m == 0 || base >= 0, "dirv_permute: nest is wrong", nest, 4);
#endif
  seen_lt = 0;
  rdir = 0;
  for (j = 0; j < i; ++j) {
    k = m ? m[j + base] : i - j - 1;
    e = DIRV_ENTRYG(dir, k);
    rdir <<= DIRV_ENTSIZ;
    if (!seen_lt)
      rdir |= (e & ~DIRV_GT);
    else
      rdir |= e;
    if (e & DIRV_LT)
      seen_lt = 1;
  }
  if (rdir == 0)
    return 0;
  /* check alleq case */
  for (j = 0; j < i; ++j) {
    if (DIRV_ENTRYG(rdir, j) != DIRV_EQ)
      goto skip;
  }
  /* all are equal */
  if (!(dir & DIRV_ALLEQ))
    return 0;

skip:
  rdir |= DIRV_INFOPART(dir);
  return rdir;
}

static void
dovec(DIRVEC dir, DIRVEC dir1, int pos, int ig, void (*f)(DIRVEC), int alleq,
      int seenlt)
{
  DIRVEC dir2;
  DIRVEC e;

  if (pos < 0) {
    if (dir1 != 0 && !alleq)
      (*f)(dir1);
    else if (dir1 != 0 && alleq && (dir & DIRV_ALLEQ))
      (*f)(dir1 | DIRV_ALLEQ);
    return;
  }
  e = DIRV_ENTRYG(dir, pos);
  if (e & DIRV_LT) {
    dir2 = dir1 | (DIRV_LT << DIRV_ENTSIZ * pos);
    if (pos <= ig)
      dovec(dir, dir2, pos - 1, ig, f, 0, 1);
  }
  if (e & DIRV_EQ) {
    dir2 = dir1 | (DIRV_EQ << DIRV_ENTSIZ * pos);
    dovec(dir, dir2, pos - 1, ig, f, alleq, seenlt);
  }
  if (e & DIRV_RD) {
    dir2 = dir1 | (DIRV_RD << DIRV_ENTSIZ * pos);
    dovec(dir, dir2, pos - 1, ig, f, 0, seenlt);
  }
  if (e & DIRV_GT) {
    dir2 = dir1 | (DIRV_GT << DIRV_ENTSIZ * pos);
    if (seenlt)
      dovec(dir, dir2, pos - 1, ig, f, 0, 1);
  }
}

void
dirv_gen(DIRVEC dir, int *map, int nest, int ig, void (*f)(DIRVEC))
{
  /*
   * if a position to the left of ig contains '<', then all dirvecs gen'd
   * from that can be ignored
   */
  int i;
  DIRVEC dir1;

  dir1 = dirv_permute(dir, map, nest);
  for (i = 0; DIRV_ENTRYG(dir, i) != 0 && i < MAX_LOOPS; ++i)
    ;
  dovec(dir1, 0, i - 1, ig, f, 1, 0);
}

static LOGICAL is_linear(int);

/* Initialize subscript struct sub with information from expression
 * ast under multiplier astmpyr. The loop index of the ith outer loop
 * is the FORALL-index in the ith outer triplet within astliTriples.
 * astmpyr is invariant with respect to the FORALL-indices.
 * Return TRUE if ast is a linear expression. */
static LOGICAL
mkSub(int astliTriples, int sub, int astmpyr, int ast)
{
  int i;
  int astli;
  int aststride, astbase;
  LOGICAL bLinear;

  switch (A_TYPEG(ast)) {
  case A_CONV:
    /*
     * when the convert case did not exist, mkSub() returned false
     * indicating the subscript expression is non-linear. The convert
     * could appear when mixing integer types or the use of -Mlarge_arrays
     * and default integer*4. For specaccel palm -Mlarge_arrays, a false
     * dependency was returned for some array assignment in an openacc
     * kernel; the assignment was not parallelized, but the acc CG
     * generated incorrect scalar code.
     * We could try pushing the convert into its operand; but all I'm
     * going to do is check if its operand is linear, and if so, I will
     * treat it as an ID (a single term)
     */
    if (!is_linear(A_LOPG(ast)))
      return FALSE;
    FLANG_FALLTHROUGH;
  case A_ID:
    i = 0;
    for (astli = astliTriples; astli; astli = ASTLI_NEXT(astli), i++)
      if (A_SPTRG(ast) == ASTLI_SPTR(astli)) {
        aststride = A_STRIDEG(ASTLI_TRIPLE(astli));
        if (!aststride)
          aststride = astb.bnd.one;
        aststride = mk_binop(OP_MUL, aststride, astmpyr, astb.bnd.dtype);
        aststride =
            mk_binop(OP_ADD, aststride, SB_STRIDE(sub)[i], astb.bnd.dtype);
        SB_STRIDE(sub)[i] = aststride;
        astbase = A_LBDG(ASTLI_TRIPLE(astli));
        if (A_DTYPEG(astbase) != astb.bnd.dtype) {
          astbase = mk_convert(astbase, astb.bnd.dtype);
        }
        astbase = mk_binop(OP_MUL, astbase, astmpyr, astb.bnd.dtype);
        astbase = mk_binop(OP_ADD, astbase, SB_BASE(sub), astb.bnd.dtype);
        SB_BASE(sub) = astbase;
        return TRUE;
      }
    astbase = mk_binop(OP_MUL, astmpyr, ast, astb.bnd.dtype);
    SB_BASE(sub) = mk_binop(OP_ADD, astbase, SB_BASE(sub), astb.bnd.dtype);
    return TRUE;
  case A_CNST:
    astbase = mk_binop(OP_MUL, astmpyr, ast, astb.bnd.dtype);
    SB_BASE(sub) = mk_binop(OP_ADD, astbase, SB_BASE(sub), astb.bnd.dtype);
    return TRUE;
  case A_UNOP:
    if (A_OPTYPEG(ast) != OP_SUB)
      return FALSE;
    astmpyr = mk_unop(OP_SUB, astmpyr, astb.bnd.dtype);
    bLinear = mkSub(astliTriples, sub, astmpyr, A_LOPG(ast));
    return bLinear;
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      bLinear = (mkSub(astliTriples, sub, astmpyr, A_LOPG(ast)) &&
                 mkSub(astliTriples, sub, astmpyr, A_ROPG(ast)));
      return bLinear;
    case OP_SUB:
      bLinear = mkSub(astliTriples, sub, astmpyr, A_LOPG(ast));
      if (!bLinear)
        return FALSE;
      astmpyr = mk_unop(OP_SUB, astmpyr, astb.bnd.dtype);
      bLinear = mkSub(astliTriples, sub, astmpyr, A_ROPG(ast));
      return bLinear;
    case OP_MUL:
      astmpyr = mk_binop(OP_MUL, A_LOPG(ast), astmpyr, astb.bnd.dtype);
      bLinear = mkSub(astliTriples, sub, astmpyr, A_ROPG(ast));
      if (!bLinear)
        return FALSE;
      astmpyr = mk_binop(OP_MUL, A_ROPG(ast), astmpyr, astb.bnd.dtype);
      bLinear = mkSub(astliTriples, sub, astmpyr, A_LOPG(ast));
      return bLinear;
    default:
      return FALSE;
    }
  default:
    return FALSE;
  }
}

static LOGICAL
is_linear(int ast)
{
  if (!IS_IRES(ast))
    return FALSE;
  switch (A_TYPEG(ast)) {
  case A_CONV:
    return is_linear(A_LOPG(ast));
  case A_ID:
  case A_CNST:
    return TRUE;
  case A_UNOP:
    if (A_OPTYPEG(ast) != OP_SUB)
      return FALSE;
    return is_linear(A_LOPG(ast));
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
      if (!is_linear(A_LOPG(ast)))
        return FALSE;
      return is_linear(A_ROPG(ast));
    default:
      return FALSE;
    }
  default:
    return FALSE;
  }
}

/* Local storage for fwd_func() callback. */
static struct {
  int nloops;   /* number of loops to check */
  LOGICAL bFwd; /* TRUE if forward dependence on entry idv */
} fwd;

static void
fwd_func(DIRVEC dv)
{
  int i;

  for (i = 0; i < fwd.nloops; i++)
    fwd.bFwd |= ((DIRV_ENTRYG(dv, i) & DIRV_LT) != 0);
}

/*
 * fill SB_ data structures for any subscripts found in the reference.
 * return the number of subscripts, or -1 if a nonlinear subscript is found.
 */
static int
fill_subscripts(int astRef, int mr, int subStart, int ntriples,
                int astliTriples)
{
  int n, asd, ndim, sub;
  switch (A_TYPEG(astRef)) {
  case A_ID:
    n = 0;
    break;
  case A_MEM:
    n = fill_subscripts(A_PARENTG(astRef), mr, subStart, ntriples,
                        astliTriples);
    break;
  case A_SUBSTR:
    n = fill_subscripts(A_LOPG(astRef), mr, subStart, ntriples, astliTriples);
    break;
  case A_SUBSCR:
    n = fill_subscripts(A_LOPG(astRef), mr, subStart, ntriples, astliTriples);
    if (n < 0)
      return n;
    asd = A_ASDG(astRef);
    ndim = ASD_NDIM(asd);
    for (sub = 0; sub < ndim; ++sub) {
      LOGICAL bLinear;
      int astSub, i;
      ++(MR_SUBCNT(mr));
      ++hlv.subavail;
      NEED(hlv.subavail, hlv.subbase, SUBS, hlv.subsize, hlv.subsize + 100);
      BZERO(&hlv.subbase[subStart + n], SUBS, 1);
      SB_BASE(subStart + n) = astb.bnd.zero;
      for (i = 0; i < ntriples; i++)
        SB_STRIDE(subStart + n)[i] = astb.bnd.zero;
      astSub = ASD_SUBS(asd, sub);
      astSub = ili_symbolic(astSub);
      bLinear = mkSub(astliTriples, subStart + n, astb.bnd.one, astSub);
      if (!bLinear)
        return -1;
      SB_BASES(subStart + n)[ntriples] = SB_BASE(subStart + n);
      ++n;
    }
    break;
  default:
    interr("fill_subscripts: unexpected AST type", astRef, 3);
    n = 0;
  }
  return n;
} /* fill_subscripts */

/** \brief Return TRUE if there is a forward dependence from \p astArrSrc
    (the source) to \p astArrSink (the sink) within an iteration space
    described by a triplet list beginning at \p astliTriples.

    \p bSinkAfterSrc should be:
      +  1 if the sink is lexically after the source.
      +  0 if the sink is lexically before the source.
      + -1 if this is a FORALL test
 */
LOGICAL
dd_array_conflict(int astliTriples, int astArrSrc, int astArrSink,
                  int bSinkAfterSrc)
{
  int i, ntriples;
  int astli;
  int lp, lpOuter;
  int sub, nsubsSrc, nsubsSink, subStart, nsubs;
  int astTriple;
  int asdSrc, asdSink, aSrc, aSink, nSrc, nSink;
  int mrSink, mrSrc;

  assert(astliTriples, "dd_array_conflict: empty triplet list", astArrSrc, 4);

  ntriples = 0;
  for (astli = astliTriples; astli; astli = ASTLI_NEXT(astli))
    ntriples++;
  if (ntriples >= MAX_LOOPS)
    /* Assume dependence if too many triples. */
    return TRUE;

  nsubsSrc = nsubsSink = 0;
  for (aSrc = astArrSrc; aSrc && A_TYPEG(aSrc) != A_ID;) {
    switch (A_TYPEG(aSrc)) {
    case A_MEM:
      aSrc = A_PARENTG(aSrc);
      break;
    case A_SUBSTR:
      aSrc = A_LOPG(aSrc);
      break;
    case A_SUBSCR:
      asdSrc = A_ASDG(aSrc);
      nsubsSrc += ASD_NDIM(asdSrc);
      aSrc = A_LOPG(aSrc);
      break;
    default:
      interr("dd_array_conflict: unexpected AST in source", aSrc, 3);
      aSrc = 0;
      break;
    }
  }
  for (aSink = astArrSink; A_TYPEG(aSink) != A_ID;) {
    switch (A_TYPEG(aSink)) {
    case A_MEM:
      aSink = A_PARENTG(aSink);
      break;
    case A_SUBSTR:
      aSink = A_LOPG(aSink);
      break;
    case A_SUBSCR:
      asdSink = A_ASDG(aSink);
      nsubsSink += ASD_NDIM(asdSink);
      aSink = A_LOPG(aSink);
      break;
    default:
      interr("dd_array_conflict: unexpected AST in sink", aSink, 3);
      aSrc = 0;
      break;
    }
  }

  /* Initialize vectorizer's memory. */
  hlv.mrsize = 100;
  NEW(hlv.mrbase, MEMREF, hlv.mrsize);
  hlv.mravail = 1;
  hlv.lpsize = 100;
  NEW(hlv.lpbase, VLOOP, hlv.lpsize);
  hlv.lpavail = 1;
  hlv.subsize = 100;
  NEW(hlv.subbase, SUBS, hlv.subsize);
  hlv.subavail = 1;

  /* Allocate ntriples vectorizer loops. */
  lpOuter = hlv.lpavail;
  hlv.lpavail += ntriples;
  NEED(hlv.lpavail, hlv.lpbase, VLOOP, hlv.lpsize, hlv.lpsize + 100);
  BZERO(&hlv.lpbase[lpOuter], VLOOP, ntriples);

  /* Initialize the lower and upper bound subscripts of all loops. */
  astli = astliTriples;
  for (lp = hlv.lpavail - 1; lp >= lpOuter; --lp, astli = ASTLI_NEXT(astli)) {
    int aststride;
    astTriple = ASTLI_TRIPLE(astli);
    assert(A_TYPEG(astTriple) == A_TRIPLE,
           "dd_array_conflict: wrong triplet type", astliTriples, 4);
    sub = hlv.subavail++;
    NEED(hlv.subavail, hlv.subbase, SUBS, hlv.subsize, hlv.subsize + 100);
    BZERO(&hlv.subbase[sub], SUBS, 1);
    SB_BASE(sub) = astb.bnd.zero;
    for (i = 0; i < hlv.lpavail - 1 - lp; i++) {
      SB_STRIDE(sub)[i] = astb.bnd.zero;
      SB_BASES(sub)[i + 1] = astb.bnd.zero;
    }
    VL_LBND(lp) = sub;
    sub = hlv.subavail++;
    NEED(hlv.subavail, hlv.subbase, SUBS, hlv.subsize, hlv.subsize + 100);
    BZERO(&hlv.subbase[sub], SUBS, 1);
    SB_BASE(sub) =
        mk_binop(OP_SUB, A_UPBDG(astTriple), A_LBDG(astTriple), astb.bnd.dtype);
    aststride = A_STRIDEG(astTriple);
    if (aststride && IS_CNST(aststride)) {
      ISZ_T val;
      val = CNSTG(aststride);
      if (val < 0) {
        SB_BASE(sub) = mk_binop(OP_SUB, A_LBDG(astTriple), A_UPBDG(astTriple),
                                astb.bnd.dtype);
      }
    }
    for (i = 0; i < hlv.lpavail - 1 - lp; i++) {
      SB_STRIDE(sub)[i] = astb.bnd.zero;
      SB_BASES(sub)[i + 1] = SB_BASE(sub);
    }
    VL_UBND(lp) = sub;
  }

  /* Create memory reference structures for astArrSrc & astArrSink. */
  mrSink = hlv.mravail++;
  NEED(hlv.mravail, hlv.mrbase, MEMREF, hlv.mrsize, hlv.mrsize + 100);
  BZERO(&hlv.mrbase[mrSink], MEMREF, 1);
  MR_ILI(mrSink) = astArrSink;
  MR_TYPE(mrSink) = 'l';

  mrSrc = hlv.mravail++;
  NEED(hlv.mravail, hlv.mrbase, MEMREF, hlv.mrsize, hlv.mrsize + 100);
  BZERO(&hlv.mrbase[mrSrc], MEMREF, 1);
  MR_ILI(mrSrc) = astArrSrc;
  MR_TYPE(mrSrc) = 's';

  /* Create subscript structures for mrSink. */
  subStart = hlv.subavail;
  MR_SUBST(mrSink) = subStart;
  MR_SUBCNT(mrSink) = 0;
  nSrc = fill_subscripts(astArrSink, mrSink, subStart, ntriples, astliTriples);
  if (nSrc < 0) {
    fwd.bFwd = TRUE;
    goto nonlinear;
  }

  /* Create subscript structures for mrSrc. */
  subStart = hlv.subavail;
  MR_SUBST(mrSrc) = subStart;
  MR_SUBCNT(mrSrc) = 0;
  nSink = fill_subscripts(astArrSrc, mrSrc, subStart, ntriples, astliTriples);
  if (nSink < 0) {
    fwd.bFwd = TRUE;
    goto nonlinear;
  }

  /* Fill in ddinfo for dependence analysis. */
  ddinfo.mr1 = mrSrc;
  ddinfo.mr2 = mrSink;
  ddinfo.subs1 = MR_SUBST(mrSrc) + MR_SUBCNT(mrSrc);
  ddinfo.subs2 = MR_SUBST(mrSink) + MR_SUBCNT(mrSink);

  ddinfo.n = ntriples; /* the array references are in same loop nest. */
  ddinfo.n1 = ddinfo.n2 = 0;

  for (i = 0; i < ntriples; i++)
    ddinfo.lps[i] = (lpOuter + ntriples - 1) - i;

  ddinfo.common = ddinfo.lps[0];
  ddinfo.outer_loop = lpOuter;

  VP_DEPCHK(ddinfo.pragmas) = TRUE;

  ddinfo.dvlist = NULL;

  /* Do dependence analysis. */
  nsubs = nsubsSrc < nsubsSink ? nsubsSrc : nsubsSink;
  do_subscript(nsubs);

  if (bSinkAfterSrc < 0) {
    DV *pdv;
    /* ANY loop-carried dependence is a forward dependence */
    fwd.bFwd = FALSE;
    for (pdv = ddinfo.dvlist; pdv; pdv = pdv->next) {
      DIRVEC dv;
      int i;
      dv = pdv->vec;
      for (i = 0; i < MAX_LOOPS; ++i) {
        int t;
        t = DIRV_ENTRYG(dv, i);
        if (t == 0)
          break; /* done */
        if (t & (DIRV_GT | DIRV_LT)) {
          /* found a loop-carried dependence */
          fwd.bFwd = TRUE;
          break;
        }
      }
    }
  } else {
    /* Generate direction vectors. */
    DIRVEC dvexo;
    DDEDGE *dd;
    DV *pdv;
    dvexo = dirv_exo(ntriples, bSinkAfterSrc);
    for (pdv = ddinfo.dvlist; pdv; pdv = pdv->next) {
      DIRVEC dv;
      dv = pdv->vec & dvexo;
      if (!dirv_chkzero(dv, ntriples))
        dd_edge(mrSrc, mrSink, dv);
    }

    /* Determine if there is a loop-carried dependence. */
    fwd.bFwd = FALSE;
    fwd.nloops = ntriples;
    for (dd = MR_SUCC(mrSrc); dd; dd = DD_NEXT(dd)) {
      dirv_gen(dd->dirvec, 0, ntriples, ntriples, fwd_func);
      if (fwd.bFwd)
        break;
    }
  }

nonlinear:
  /* Clean up allocated memory. */
  cln_visit();
  FREE(hlv.lpbase);
  FREE(hlv.mrbase);
  FREE(hlv.subbase);
  freearea(HLV_AREA1);

  return fwd.bFwd;
}

static int
symbolic_mul(int a, int b)
{
  int flag;
  int c;

  flag = 1;
  if (ILI_OPC(a) == IL_INEG) {
    a = ILI_OPND(a, 1);
    flag = -1 * flag;
  }
  if (ILI_OPC(b) == IL_INEG) {
    b = ILI_OPND(b, 1);
    flag = -1 * flag;
  }
  c = ad2ili(IL_IMUL, a, b);
  if (flag < 0)
    c = ad1ili(IL_INEG, c);
  return c;
}

/* num/den = ili */
/* quot = symbolic quotient */
static LOGICAL
symbolic_divide(int num, int den, int *quot)
{
  int sign = 1;
  int q = num;
  int icon1, iconm1, icon0;

  if (num == (icon0 = ad_icon((INT)0))) {
    /* 0 / x */
    q = icon0;
    goto ret;
  }
  /* x / +-1 */
  if (den == (icon1 = ad_icon((INT)1)))
    goto ret;
  if (den == (iconm1 = ad_icon((INT)-1))) {
    sign = -1;
    goto ret;
  }
  /* we'll assume -1 / sym and 1 / sym is 0; this
   * is o.k. since we'll just interchange rows & try
   * again
   */
  if (num == icon1 || num == iconm1) {
    q = ad_icon((INT)0);
    goto ret;
  }

  if (ILI_OPC(num) == IL_INEG) {
    num = ILI_OPND(num, 1);
    sign = -1 * sign;
  }
  if (ILI_OPC(den) == IL_INEG) {
    den = ILI_OPND(den, 1);
    sign = -1 * sign;
  }
  if (ILI_OPC(num) == IL_IMUL) {
    if (ILI_OPND(num, 1) == den) {
      q = ILI_OPND(num, 2);
      goto ret;
    }
    if (ILI_OPND(num, 2) == den) {
      q = ILI_OPND(num, 1);
      goto ret;
    }
  }
  return FALSE;
ret:
  if (sign < 0)
    *quot = ad1ili(IL_INEG, q);
  else
    *quot = q;
  return TRUE;
}

static LOGICAL rdc;              /* TRUE if a reduction has occurred */
static int visit_chain = 0;      /* head of chain of ili with ILI_REPL set */
static LOGICAL use_visit = TRUE; /* TRUE if ILI_REPL to be used */

typedef struct arith_term {
  long confact;            /* constant multiplier */
  int varfact;             /* variable factor */
  struct arith_term *next; /* next term */
} ARITH_TERM, *ARITH_LIST; /* arithmetic terms */

/* Clear the ILI_REPL & ILI_VISIT fields of all ili in the chain beginning
 * at visit_chain. */
static void
cln_visit(void)
{
  int il = visit_chain;
  int ilnext;

  for (il = visit_chain; il; il = ilnext) {
    ilnext = ILI_VISIT(il);
    ILI_VISIT(il) = ILI_REPL(il) = 0;
  }
  visit_chain = 0;
}

/* Append list l2 to the end of l1, and return the resulting list. */
static ARITH_LIST
apnd(ARITH_LIST l1, ARITH_LIST l2)
{
  ARITH_LIST l;

  if (l1 == NULL)
    return l2;
  if (l2 == NULL)
    return l1;
  /* Set l to the last cell of l1's list */
  for (l = l1; l->next; l = l->next)
    ;
  l->next = l2;
  return l1;
}

/* Extract constant factors from the varfact member of arithmetic term atp.
 * Rules are:
 *	r(<c,d>) = <c*d,1>, where d is a constant.
 *	r(<c,x*y) = <c*d*e,x'*y'>, where:
 *			<d,x'> = r(<1,x>)
 *			<e,y'> = r(<1,y>).
 *	r(<c,-x>) = <-c*d,y>, where
 *			<d,y> = r(<1,x>).
 *	r(<c,x>) = <c,x>, otherwise.
 */
static void
refactor(ARITH_LIST atp)
{
  ARITH_TERM at1, at2;
  int opc;
  int icon1 = ad_icon(1L);

  at1.next = at2.next = NULL;

  if (IS_CNST(atp->varfact)) {
    atp->confact *= CNSTG(atp->varfact);
    atp->varfact = icon1;
    return;
  }

  opc = ILI_OPC(atp->varfact);
  switch (opc) {
  case IL_IMUL:
    at1.confact = at2.confact = 1L;
    at1.varfact = ILI_OPND(atp->varfact, 1);
    at2.varfact = ILI_OPND(atp->varfact, 2);
    refactor(&at1);
    refactor(&at2);
    atp->confact *= at1.confact * at2.confact;
    if (at1.varfact == icon1)
      atp->varfact = at2.varfact;
    else if (at2.varfact == icon1)
      atp->varfact = at1.varfact;
    else
      atp->varfact = ad2ili(IL_IMUL, at1.varfact, at2.varfact);
    break;
  case IL_INEG:
    at1.confact = 1L;
    at1.varfact = ILI_OPND(atp->varfact, 1);
    refactor(&at1);
    atp->confact *= -at1.confact;
    atp->varfact = at1.varfact;
    break;
  default:
    break;
  }
}

/* Extract constant factors of each term in arithmetic term list lst. */
static void
refactor_list(ARITH_LIST lst)
{
  ARITH_LIST l;

  for (l = lst; l; l = l->next)
    refactor(l);
}

/* Create a sum of all terms in list lst. */
static int
sum(ARITH_LIST lst)
{
  int ilcon, ilterm, ilsum;
  ARITH_LIST l;

  ilsum = ad_icon(0L);
  for (l = lst; l; l = l->next) {
    ilcon = ad_icon(l->confact);
    ilterm = ad2ili(IL_IMUL, ilcon, l->varfact);
    ilsum = ad2ili(IL_IADD, ilterm, ilsum);
  }
  return ilsum;
}

/* Return an arithmetic term list whose sum is equivalent to mpyr * ili,
 * where mpyr is distributed across terms in the resulting list. All confact
 * fields in the term list are set to 1.
 * Rules are:
 *	distrib(x*y, m) = [distrib(x, yi) | yi <- distrib(y, m)].
 *	distrib(x+y, m) = distrib(x, m) apnd distrib(y, m).
 *	distrib(x-y, m) = distrib(x, m) apnd distrib(y, -m).
 *	distrib(-x, m) = distrib(x, -m).
 *	distrib(x, m + n) = distrib(m, x) apnd distrib(n, x); if x is a load.
 *	distrib(x, m) = [x * m]; otherwise.
 */
static ARITH_LIST
distrib(int ili, int mpyr)
{
  int ili2, ilitmp;
  int opc, opc1;
  ARITH_LIST l1, l2, l;

  assert(mpyr, "distrib: invalid multiplier", ili, 4);
  assert(ili, "distrib: invalid ili", ili, 4);

  opc = ILI_OPC(ili);
  switch (opc) {
  case IL_INEG:
    ilitmp = ad1ili(IL_INEG, mpyr);
    l = distrib(ILI_OPND(ili, 1), ilitmp);
    return l;
  case IL_IADD:
    l1 = distrib(ILI_OPND(ili, 1), mpyr);
    l2 = distrib(ILI_OPND(ili, 2), mpyr);
    l = apnd(l1, l2);
    return l;
  case IL_ISUB:
    l1 = distrib(ILI_OPND(ili, 1), mpyr);
    ilitmp = ad1ili(IL_INEG, mpyr);
    l2 = distrib(ILI_OPND(ili, 2), ilitmp);
    l = apnd(l1, l2);
    return l;
  case IL_IMUL:
    l2 = distrib(ILI_OPND(ili, 2), mpyr);
    l = NULL;
    for (; l2; l2 = l2->next) {
      assert(l2->confact == 1L, "distrib: non-unit constant", ili, 4);
      l1 = distrib(ILI_OPND(ili, 1), l2->varfact);
      l = apnd(l1, l);
    }
    return l;
  case IL_IDIV:
  case IL_MOD:
    ilitmp = ili_symbolic(ILI_OPND(ili, 1));
    if (ilitmp != ili) {
      ilitmp = ad2ili(opc, ilitmp, ILI_OPND(ili, 2));
      ilitmp = ad2ili(IL_IMUL, ilitmp, mpyr);
      l = (ARITH_LIST)getitem(HLV_AREA1, sizeof(ARITH_TERM));
      l->confact = 1;
      l->varfact = ilitmp;
      l->next = NULL;
      return l;
    }
    FLANG_FALLTHROUGH;
  default:
    opc1 = ILI_OPC(mpyr);
    if (opc1 == IL_IADD) {
      l1 = distrib(ILI_OPND(mpyr, 1), ili);
      l2 = distrib(ILI_OPND(mpyr, 2), ili);
      l = apnd(l1, l2);
      return l;
    }
    if (opc1 == IL_ISUB) {
      l1 = distrib(ILI_OPND(mpyr, 1), ili);
      ili2 = ad1ili(IL_INEG, ILI_OPND(mpyr, 2));
      l2 = distrib(ili2, ili);
      l = apnd(l1, l2);
      return l;
    }
    if (IS_IRES(ili))
      ilitmp = ad2ili(IL_IMUL, ili, mpyr);
    else
      ilitmp = ili;
    l = (ARITH_LIST)getitem(HLV_AREA1, sizeof(ARITH_TERM));
    l->confact = 1;
    l->varfact = ilitmp;
    l->next = NULL;
    return l;
  }
}

/* Return TRUE if arithmetic term list lst contains a term whose varfact
 * member is equivalent to trm.
 * Rules are:
 *	c_c_f([], x) = FALSE.
 *	c_c_f([<c,x>|l], x) = TRUE.
 *	c_c_f([x|l], y) = c_c_f(l, y), otherwise.
 */
static LOGICAL
contains_common_factor(ARITH_LIST lst, int trm)
{
  ARITH_LIST l;

  assert(trm, "contains_common_factor: invalid term", trm, 4);
  for (l = lst; l; l = l->next)
    if (l->varfact == trm)
      return TRUE;
  return FALSE;
}

/* Sum the confact members of all terms in arithmetic term list lst
 * whose varfact members are all equal to trm. Return the sum.
 * Rules are:
 *	c_t([], x) = 0.
 *	c_t([<c,x>| l], x) = c + c_t(l, x).
 *	c_t([x|l], y) = c_t(l, y); otherwise.
 */
static long
combine_terms(ARITH_LIST lst, int trm)
{
  ARITH_LIST l;
  long sum = 0L;

  for (l = lst; l; l = l->next)
    if (l->varfact == trm)
      sum += l->confact;
  return sum;
}

/* Remove all arithmetic terms from arithmetic term list lst whose varfact
 * members are equivalent to trm.
 * Rules are:
 *	r_t([], x) = [].
 *	r_t([<c,x>|l], x) = l.
 *	r_t([x|l], y) = [x | r_t(l,y)], otherwise
 */
static ARITH_LIST
remove_terms(ARITH_LIST lst, int trm)
{
  ARITH_LIST lcur, lprev, lhead;

  assert(trm, "remove_terms: invalid term", trm, 4);

  lhead = lst;
  lprev = NULL;
  for (lcur = lst; lcur; lcur = lcur->next)
    if (lcur->varfact == trm)
      /* remove the current term from the list */
      if (lprev == NULL)
        lhead = lcur->next;
      else
        lprev->next = lcur->next;
    else
      lprev = lcur;
  return lhead;
}

/* Combine terms from arithmetic term list lst that have common factors, and
 * return the result.
 * Rules are:
 *	elim([]) = [].
 *	elim([<c,x>|l]) = if c_c_f(l, x) then
 *			     [<c_t(l, x)+c,x> | elim(r_t(l,x))]
 *			  else
 *			     [<c,x> | elim(l)].
 * When a reduction takes place, set global rdc to TRUE.
 */
static ARITH_LIST
elim(ARITH_LIST lst)
{
  ARITH_LIST lprev, lcur, lhead;
  long sum;

  lhead = lst;
  lprev = NULL;
  for (lcur = lst; lcur; lcur = lcur->next) {
    if (contains_common_factor(lcur->next, lcur->varfact)) {
      sum = combine_terms(lcur, lcur->varfact);
      lcur->next = remove_terms(lcur->next, lcur->varfact);
      if (sum == 0L)
        if (lprev)
          lprev->next = lcur->next;
        else
          lhead = lcur->next;
      else {
        lcur->confact = sum;
        lprev = lcur;
      }
      rdc = TRUE;
    } else
      lprev = lcur;
  }
  return lhead;
}

static int
ili_symbolic(int ili)
{
  /* perform symbolic manipulation on ili to simplify its form */
  int opc;
  int icon1 = ad_icon(1L);
  ARITH_LIST lst1, lst2;
  int ilires;

  if (use_visit && ILI_REPL(ili))
    return ILI_REPL(ili);

  if (IS_CNST(ili))
    return ili;
  if (A_TYPEG(ili) == A_ID)
    return ili;

  opc = ILI_OPC(ili);
  if (IS_IRES(ili)) {
    /* symbolic manipulation on integer expressions only */
    rdc = FALSE;
    lst1 = distrib(ili, icon1); /* create list of ili */
    refactor_list(lst1);        /* simplify terms */
    lst2 = elim(lst1);          /* combine terms */
    if (rdc) {
      ilires = sum(lst2); /* create sum of terms */
#if DEBUG
      if (DBGBIT(36, 128)) {
        fprintf(gbl.dbgfil, "Term reduced: ");
        prilitree(ili);
        fprintf(gbl.dbgfil, " -> ");
        prilitree(ilires);
        fprintf(gbl.dbgfil, "\n");
      }
#endif
    } else
      ilires = ili;
  } else
    ilires = ili;

  if (!use_visit)
    return ilires;
  ILI_REPL(ili) = ilires;
  ILI_VISIT(ili) = visit_chain;
  visit_chain = ili;
  return ilires;
}

/** \brief Perform symbolic algebraic simplification of term il.

  \b NOTE: Call `freearea(HLV_AREA1)` after calling this function.
 */
int
dd_symbolic(int il)
{
  int ilres;

  use_visit = FALSE;
  ilres = ili_symbolic(il);
  use_visit = TRUE;
  return ilres;
}

int
dirv_chkzero(DIRVEC dir, int n)
{
  int i;

  for (i = 0; i < n; ++i)
    if (DIRV_ENTRYG(dir, i) == 0)
      return 1;
  return 0;
}

/* opc = a unary arithmetic ili opcode */
static int
ad1ili(int opc, int ast1)
{
  int astx;

  if (opc != IL_INEG)
    interr("ad1ili: unidentified opcode", opc, 4);
  if (A_DTYPEG(ast1) != astb.bnd.dtype)
    ast1 = mk_convert(ast1, astb.bnd.dtype);
  astx = mk_unop(OP_SUB, ast1, astb.bnd.dtype);
  return astx;
}

/* opc = a binary arithmetic ili opcode */
static int
ad2ili(int opc, int ast1, int ast2)
{
  int optype = 0;
  int dtype = 0;
  int astx;

  switch (opc) {
  case IL_IADD:
    optype = OP_ADD;
    dtype = astb.bnd.dtype;
    break;
  case IL_ISUB:
    optype = OP_SUB;
    dtype = astb.bnd.dtype;
    break;
  case IL_IMUL:
    optype = OP_MUL;
    dtype = astb.bnd.dtype;
    break;
  case IL_IDIV:
    optype = OP_DIV;
    dtype = astb.bnd.dtype;
    break;
  case IL_AADD:
    optype = OP_ADD;
    dtype = DT_ADDR;
    break;
  case IL_ASUB:
    optype = OP_SUB;
    dtype = DT_ADDR;
    break;
  default:
    interr("ad2ili: unidentified opcode", opc, 4);
  }

  if (A_DTYPEG(ast1) != dtype)
    ast1 = mk_convert(ast1, dtype);
  astx = mk_binop(optype, ast1, ast2, dtype);
  return astx;
}

static int
ILI_OPC(int astx)
{
  if (A_TYPEG(astx) == A_BINOP)
    switch (A_OPTYPEG(astx)) {
    case OP_ADD:
      if (A_DTYPEG(astx) == DT_ADDR)
        return IL_AADD;
      if (DT_ISBASIC(A_DTYPEG(astx)))
        return IL_IADD;
      interr("ILI_OPC: unknown dtype for ADD", astx, 4);
      return 0;
    case OP_SUB:
      if (A_DTYPEG(astx) == DT_ADDR)
        return IL_ASUB;
      if (DT_ISBASIC(A_DTYPEG(astx)))
        return IL_ISUB;
      interr("ILI_OPC: unknown dtype for SUB", astx, 4);
      return 0;
    case OP_MUL:
      if (DT_ISBASIC(A_DTYPEG(astx)))
        return IL_IMUL;
      interr("ILI_OPC: unknown dtype for MUL", astx, 4);
      return 0;
    case OP_DIV:
      if (DT_ISBASIC(A_DTYPEG(astx)))
        return IL_IDIV;
      interr("ILI_OPC: unknown dtype for DIV", astx, 4);
      return 0;
    default:
      return 0; /* opcode unknown */
    }
  if (A_TYPEG(astx) == A_UNOP && A_OPTYPEG(astx) == OP_SUB &&
      DT_ISBASIC(A_DTYPEG(astx)))
    return IL_INEG;
  return 0; /* opcode unknown */
}

static int
ILI_OPND(int astx, int opnd)
{
  if (A_TYPEG(astx) == A_UNOP) {
    if (opnd == 1)
      return A_LOPG(astx);
    else
      interr("ILI_OPND: UNOP operand # out of range", astx, 4);
  }
  if (A_TYPEG(astx) == A_BINOP) {
    if (opnd == 1)
      return A_LOPG(astx);
    else if (opnd == 2)
      return A_ROPG(astx);
    else
      interr("ILI_OPND: BINOP operand # out of range", astx, 4);
  }
  interr("ILI_OPND: unknown operand type", astx, 4);
  return 0;
}

#if DEBUG
char *
dirv_print(DIRVEC dir)
{
  static char xyz[32];
  char *p = xyz;
  DIRVEC t;
  int i;

  for (i = 0; DIRV_ENTRYG(dir, i) != 0 && i < MAX_LOOPS; ++i)
    ;
  if (dir & DIRV_ALLEQ)
    *p++ = '#';
  else
    *p++ = ' ';
  for (--i; i >= 0; --i) {
    t = DIRV_ENTRYG(dir, i);
    if (t & DIRV_LT)
      *p++ = '<';
    else
      *p++ = '.';
    if (t & DIRV_EQ)
      *p++ = '=';
    else
      *p++ = '.';
    if (t & DIRV_GT)
      *p++ = '>';
    else
      *p++ = '.';
    if (t & DIRV_RD)
      *p++ = 'R';
    else
      *p++ = ' ';
  }
  *p++ = 0;
  return xyz;
}

void
dump_dd(DDEDGE *p)
{
  static const char *types[] = {"flow", "anti", "outp", "????"};

  for (; p != 0; p = DD_NEXT(p)) {
    fprintf(gbl.dbgfil, "     %4d %4s %-32s\n", DD_SINK(p), types[DD_TYPE(p)],
            dirv_print(DD_DIRVEC(p)));
  }
}

static void
dump_one_bound(BOUND *p, int k, LOGICAL btype)
{
  int j;
  BOUND b;

  b = *p;
  if (b.mplyr != 1)
    fprintf(gbl.dbgfil, "%d*T[%d]", b.mplyr, k);
  else
    fprintf(gbl.dbgfil, "T[%d]", k);
  fprintf(gbl.dbgfil, btype ? "<=" : ">=");
  /* dump 0 term */
  if (b.gcd == 1 || IS_CNST(b.bnd[0]))
    prilitree(b.bnd[0]);
  else {
    fprintf(gbl.dbgfil, "(");
    prilitree(b.bnd[0]);
    fprintf(gbl.dbgfil, ")/%d", b.gcd);
  }
  for (j = 1; j < k; ++j) {
    if (IS_CNST(b.bnd[j]) && CNSTG(b.bnd[j]) == 1L)
      fprintf(gbl.dbgfil, "+");
    else if (IS_CNST(b.bnd[j]) && CNSTG(b.bnd[j]) == 0L)
      continue;
    else {
      fprintf(gbl.dbgfil, "+");
      prilitree(b.bnd[j]);
      fprintf(gbl.dbgfil, "*");
    }
    fprintf(gbl.dbgfil, "T[%d]", j);
  }
  fprintf(gbl.dbgfil, "\n");
}

static void
dump_two_bound(BOUND *p, BOUND *q, int k, LOGICAL btype)
{
  int j;
  if (p->mplyr != 1)
    fprintf(gbl.dbgfil, "(");
  /* dump 0 term */
  if (p->gcd == 1 || IS_CNST(p->bnd[0]))
    prilitree(p->bnd[0]);
  else {
    fprintf(gbl.dbgfil, "(");
    prilitree(p->bnd[0]);
    fprintf(gbl.dbgfil, ")");
  }
  for (j = 1; j < k; ++j) {
    if (IS_CNST(p->bnd[j]) && CNSTG(p->bnd[j]) == 1L)
      fprintf(gbl.dbgfil, "+");
    else if (IS_CNST(p->bnd[j]) && CNSTG(p->bnd[j]) == 0L)
      continue;
    else {
      fprintf(gbl.dbgfil, "+");
      prilitree(p->bnd[j]);
      fprintf(gbl.dbgfil, "*");
    }
    fprintf(gbl.dbgfil, "T[%d]", j);
  }
  if (p->mplyr != 1)
    fprintf(gbl.dbgfil, ")/%d", p->mplyr);
  fprintf(gbl.dbgfil, btype ? ">=" : "<=");
  if (q->mplyr != 1)
    fprintf(gbl.dbgfil, "(");
  /* dump 0 term */
  if (q->gcd == 1 || IS_CNST(q->bnd[0]))
    prilitree(q->bnd[0]);
  else {
    fprintf(gbl.dbgfil, "(");
    prilitree(q->bnd[0]);
    fprintf(gbl.dbgfil, ")");
  }
  for (j = 1; j < k; ++j) {
    if (IS_CNST(q->bnd[j]) && CNSTG(q->bnd[j]) == 1L)
      fprintf(gbl.dbgfil, "+");
    else if (IS_CNST(q->bnd[j]) && CNSTG(q->bnd[j]) == 0L)
      continue;
    else {
      fprintf(gbl.dbgfil, "+");
      prilitree(q->bnd[j]);
      fprintf(gbl.dbgfil, "*");
    }
    fprintf(gbl.dbgfil, "T[%d]", j);
  }
  if (q->mplyr != 1)
    fprintf(gbl.dbgfil, ")/%d", q->mplyr);

  fprintf(gbl.dbgfil, "\n");
}

#if DEBUG
/* Dump a list of arithmetic terms. */
void
dump_termlist(ARITH_LIST lst)
{
  ARITH_LIST l;

  if (lst == NULL) {
    fprintf(gbl.dbgfil, "[]");
    return;
  }
  fprintf(gbl.dbgfil, "[");
  for (l = lst; l->next; l = l->next) {
    fprintf(gbl.dbgfil, "%ld*", l->confact);
    prilitree(l->varfact);
    fprintf(gbl.dbgfil, ", ");
  }
  fprintf(gbl.dbgfil, "%ld*", l->confact);
  prilitree(l->varfact);
  fprintf(gbl.dbgfil, "]");
}
#endif

#endif

#endif

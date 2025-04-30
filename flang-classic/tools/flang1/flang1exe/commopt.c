/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran communications optimizer module
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "comm.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "transfrm.h"
#include "extern.h"
#include "commopt.h"
#include "ccffinfo.h"
#include "optimize.h"
#include "nme.h"
#include "rte.h"
#include "hlvect.h"
#include "symutl.h" /* for is_impure */

extern int rewrite_opfields;

static void commopt(void);
static void comm_optimize_init(void);
static LOGICAL same_forall_bnds(int, int, int);
static LOGICAL is_fusable(int, int, int);
static LOGICAL smp_conflict(int, int);
static LOGICAL is_in_block(int, int);
static LOGICAL Conflict(int, int, int, LOGICAL, int, int);
static LOGICAL is_branch_between(int, int);
static LOGICAL is_contains_ast_between(int, int, int);
static LOGICAL is_same_align_shape(int, int, int, int);

static void comm_optimize_end(void);
static void eliminate_rt(int);
static void eliminate_bnds(int, int, int, int);
static void eliminate_alloc(int, int, int, int);
static void eliminate_sect(int, int, int, int);
static void eliminate_copy(int, int, int, int);
static void eliminate_gather(int, int, int, int);
static void eliminate_shift(int, int, int, int);
static void eliminate_start(int, int, int, int);
static void eliminate_get_scalar(void);
static void fuse_forall(int);
static LOGICAL is_same_descr_for_bnds(int, int);
static LOGICAL Conflict_(int);
static LOGICAL is_same_idx(int, int);
static LOGICAL is_dominator_fg(int, int);
static LOGICAL must_follow(int, int, int, int);
#if DEBUG
int find_lp(int);
#endif
static void init_optsum(void);
#if DEBUG
static void optsummary(void);
#endif
static void alloc2ast(void);
static void opt_allocate(void);
#if DEBUG
LOGICAL is_same_def(int, int);
#endif
static LOGICAL is_safe_copy(int);
static LOGICAL is_allocatable_assign(int ast);
static int propagate_bound(LITEMF *defs_to_propagate, int bound);
static void rewrite_all_shape(LITEMF *);
static void decompose_expression(int, int[], int, int *, int *);
static LOGICAL is_avail_expr_with_list(int, int, int, int, int, int[], int);
static void put_forall_fg(void);
static LOGICAL independent_commtype(int, int);
static LOGICAL is_olap_conflict(int, int);
static LOGICAL is_pcalls(int, int);
static void forall_make_same_idx(int);

INT *lpsort;
FTB ftb;
OPTSUM optsum;

static void
selection_sort(void)
{
  int i, j;
  int hd, hd1;
  int dom;
  int t;
  int ast, ast1;
  int std, std1;

  lpsort = (INT *)getitem(FORALL_AREA, (opt.nloops + 1) * sizeof(INT));
  for (i = 1; i <= opt.nloops; ++i)
    lpsort[i] = i;

  for (i = 1; i <= opt.nloops - 1; ++i) {
    dom = i;
    for (j = i + 1; j <= opt.nloops; ++j) {
      hd = LP_HEAD(lpsort[dom]);
      hd1 = LP_HEAD(lpsort[j]);
      std = FG_STDFIRST(hd);
      ast = STD_AST(std);
      std1 = FG_STDFIRST(hd1);
      ast1 = STD_AST(std1);
      if (is_dominator(hd1, hd))
        dom = j;
    }
    t = lpsort[dom];
    lpsort[dom] = lpsort[i];
    lpsort[i] = t;
  }
}

void
comm_optimize_post(void)
{
  alloc2ast();
  comm_optimize_init();
  flowgraph(); /* build the flowgraph for the function */
#if DEBUG
  if (DBGBIT(35, 1))
    dump_flowgraph();
#endif

  findloop(HLOPT_ALL); /* find the loops */

  rewrite_opfields = 0x3; /* copy opt1 and opt2 fields */
  flow();                 /* do flow analysis on the loops  */
  rewrite_opfields = 0;   /* reset */

#if DEBUG
  if (DBGBIT(35, 4)) {
    dump_flowgraph();
    dump_loops();
  }
#endif

  commopt();

#if DEBUG
  if (DBGBIT(59, 1))
    optsummary();
#endif
  comm_optimize_end();
}

void
comm_optimize_pre(void)
{

  comm_optimize_init();
  flowgraph(); /* build the flowgraph for the function */
#if DEBUG
  if (DBGBIT(35, 1))
    dump_flowgraph();
#endif

  findloop(HLOPT_ALL); /* find the loops */

  flow(); /* do flow analysis on the loops  */

#if DEBUG
  if (DBGBIT(35, 4)) {
    dump_flowgraph();
    dump_loops();
  }
#endif
  selection_sort();
  fuse_forall(0);
  comm_optimize_end();
}

void
forall_init(void)
{
  int std;
  int ast;
  SPTR current_block = 0;
  int parallel_depth;
  int task_depth;

  init_ftb();
  init_dtb();
  init_optsum();
  init_bnd();

  for (std = STD_NEXT(0); std;) {
    ast = STD_AST(std);
    if (A_TYPEG(ast) == A_FORALL) {
      forall_opt1(ast);
    }
    std = STD_NEXT(std);
  }
  /* put calls into calls table */
  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std;) {
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_CONTINUE: {
      // Set current_block to the current ST_BLOCK sym.
      SPTR lab = STD_LABEL(std);
      SPTR block_sptr;
      if (!lab)
        break;
      block_sptr = ENCLFUNCG(lab);
      if (!block_sptr || STYPEG(block_sptr) != ST_BLOCK)
        break;
      if (lab == STARTLABG(block_sptr)) {
        current_block = block_sptr;
      } else if (lab == ENDLABG(block_sptr)) {
        block_sptr = ENCLFUNCG(block_sptr);
        current_block =
          block_sptr && STYPEG(block_sptr) == ST_BLOCK ? block_sptr : 0;
      }
      break;
    }
    case A_FORALL:
      put_forall_pcalls(std);
      forall_make_same_idx(std);
      if (!STD_BLKSYM(std))
        // Mark the loop with the current block, for use in fuse_forall.
        STD_BLKSYM(std) = current_block;
      break;
    case A_MP_PARALLEL:
      ++parallel_depth;
      /*symutl.sc = SC_PRIVATE;*/
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      if (parallel_depth == 0 && task_depth == 0) {
        /*symutl.sc = SC_LOCAL;*/
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      ++task_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDTASK:
    case A_MP_ETASKLOOP:
      --task_depth;
      if (parallel_depth == 0 && task_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    default:
      break;
    }
    std = STD_NEXT(std);
  }
}

static void
init_optsum(void)
{
  optsum.fuse = 0;
  optsum.bnd = 0;
  optsum.alloc = 0;
  optsum.sect = 0;
  optsum.copysection = 0;
  optsum.gatherx = 0;
  optsum.scatterx = 0;
  optsum.shift = 0;
  optsum.start = 0;
}

#if DEBUG
static void
optsummary(void)
{

  static char msg0[] = "===== comm opt summary for %s ====\n";
  static char msg1[] = "%5d loops fused\n";
  static char msg2[] = "%5d pghpf_localize_bnd calls eliminated\n";
  static char msg3[] = "%5d allocates eliminated\n";
  static char msg4[] = "%5d pghpf_sect calls eliminated\n";
  static char msg5[] = "%5d pghpf_comm_copy calls eliminated\n";
  static char msg6[] = "%5d pghpf_comm_gatherx calls eliminated\n";
  static char msg7[] = "%5d pghpf_comm_scatterx calls eliminated\n";
  static char msg8[] = "%5d pghpf_comm_shift calls eliminated\n";
  static char msg9[] = "%5d pghpf_comm_start calls eliminated\n";
  FILE *fil;

  fil = gbl.dbgfil;
  if (fil == NULL)
    fil = stderr;
  fprintf(fil, msg0, SYMNAME(gbl.currsub));
  fprintf(fil, msg1, optsum.fuse);
  fprintf(fil, msg2, optsum.bnd);
  fprintf(fil, msg3, optsum.alloc);
  fprintf(fil, msg4, optsum.sect);
  fprintf(fil, msg5, optsum.copysection);
  fprintf(fil, msg6, optsum.gatherx);
  fprintf(fil, msg7, optsum.scatterx);
  fprintf(fil, msg8, optsum.shift);
  fprintf(fil, msg9, optsum.start);
}
#endif

static void
commopt(void)
{
  selection_sort();
  put_forall_fg();

  if (!XBIT(47, 0x2000))
    eliminate_rt(1);
  if (!XBIT(47, 0x4000))
    eliminate_rt(2);
  if (!XBIT(47, 0x8000))
    eliminate_rt(3);
  if (!XBIT(47, 0x10000))
    eliminate_rt(4);
  if (!XBIT(47, 0x20000))
    eliminate_get_scalar();
}

static void
fuse_forall(int nested)
{
  int i, j;
  int std, std1;
  int forall, forall1;
  int nd, nd1;
  int hd, hd1;
  int ii, jj;
  int fg, fg1;
  int number_of_try;

  for (ii = 1; ii <= opt.nloops; ++ii) {
    i = lpsort[ii];
    if (!LP_FORALL(i))
      continue;
    hd = LP_HEAD(i); /*the flow graph node which is the loop head*/
    fg = LP_FG(i);
    std = FG_STDFIRST(hd);
    forall = STD_AST(std);
    nd = A_OPT1G(forall);
    if (FT_NFUSE(nd, nested) >= MAXFUSE)
      continue;
    if (FT_FUSED(nd))
      continue;
    number_of_try = 0;
    for (jj = ii + 1; jj <= opt.nloops; ++jj) {
      j = lpsort[jj];
      if (LP_PARENT(i) != LP_PARENT(j))
        continue;
      if (FT_NFUSE(nd, nested) >= MAXFUSE)
        break;
      if (number_of_try > 25)
        continue;
      if (LP_FORALL(j)) {
        /*the flow graph node which is the loop head */
        hd1 = LP_HEAD(j);
        fg1 = LP_FG(j);
        if (!is_dominator(hd, hd1))
          continue;
        std1 = FG_STDFIRST(hd1);
        if (STD_BLKSYM(std) != STD_BLKSYM(std1))
          continue;
        if (STD_ACCEL(std) != STD_ACCEL(std1))
          continue;
        forall1 = STD_AST(std1);
        nd1 = A_OPT1G(forall1);
        if (FT_FUSED(nd1))
          continue;
        if (!same_forall_bnds(i, j, nested))
          continue;
        if (!is_same_idx(std, std1))
          continue;
        number_of_try++;
        if (!is_fusable(i, j, nested))
          continue;

        FT_FUSELP(nd, nested, FT_NFUSE(nd, nested)) = j;
        FT_FUSEDSTD(nd, nested, FT_NFUSE(nd, nested)) = std1;
        FT_NFUSE(nd, nested)++;
        FT_FUSED(nd1) = 1;
        FT_HEADER(nd1) = FT_HEADER(nd);
        FT_FUSED(nd) = 1;
        optsum.fuse++;
      }
    }
  }
}

static LOGICAL
is_same_idx(int std, int std1)
{
  int idx[7], idx1[7];
  int list, list1, listp;
  int forall, forall1;
  int nidx, nidx1;
  int isptr, isptr1;
  int i;

  forall = STD_AST(std);
  list = A_LISTG(forall);
  nidx = 0;
  for (listp = list; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx[nidx] = listp;
    nidx++;
  }

  forall1 = STD_AST(std1);
  list1 = A_LISTG(forall1);
  nidx1 = 0;
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx1[nidx1] = listp;
    nidx1++;
  }

  if (nidx != nidx1)
    return FALSE;

  for (i = 0; i < nidx; i++) {
    isptr = ASTLI_SPTR(idx[i]);
    isptr1 = ASTLI_SPTR(idx1[i]);
    if (isptr != isptr1)
      return FALSE;
  }
  return TRUE;
}

LOGICAL
same_forall_size(int lp1, int lp2, int nested)
{
  int idx1[7], idx2[7];
  int itriple1, itriple2;
  int lb1, ub1, st1;
  int lb2, ub2, st2;
  int std1, std2;
  int fg1, fg2;
  int hd1, hd2;
  int list1, list2, listp;
  int forall1, forall2;
  int nidx1, nidx2;
  int i;
  int lhs1, lhs2;
  LOGICAL same = FALSE;

  hd1 = LP_HEAD(lp1); /*the flow graph node which is the loop head*/
  fg1 = LP_FG(lp1);
  std1 = FG_STDFIRST(hd1);
  forall1 = STD_AST(std1);
  lhs1 = A_DESTG(A_IFSTMTG(forall1));
  list1 = A_LISTG(forall1);
  nidx1 = 0;
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx1[nidx1] = listp;
    ++nidx1;
  }

  hd2 = LP_HEAD(lp2);
  fg2 = LP_FG(lp2);
  if (!is_dominator(hd1, hd2))
    return FALSE;
  std2 = FG_STDFIRST(hd2);
  forall2 = STD_AST(std2);
  lhs2 = A_DESTG(A_IFSTMTG(forall2));
  list2 = A_LISTG(forall2);
  nidx2 = 0;
  for (listp = list2; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx2[nidx2] = listp;
    nidx2++;
  }

  if (nidx1 != nidx2)
    return FALSE;

  for (i = 0; i < nidx1 - nested; i++) {
    itriple1 = ASTLI_TRIPLE(idx1[i]);
    lb1 = A_LBDG(itriple1);
    ub1 = A_UPBDG(itriple1);
    st1 = A_STRIDEG(itriple1);
    if (st1 == 0)
      st1 = astb.i1;
    itriple2 = ASTLI_TRIPLE(idx2[i]);
    lb2 = A_LBDG(itriple2);
    ub2 = A_UPBDG(itriple2);
    st2 = A_STRIDEG(itriple2);
    if (st2 == 0)
      st2 = astb.i1;
    same = FALSE;

    /*is_same_size(lb1,lb2, ub1, ub2, st1, st2) */

    if ((lb1 == lb2 && ub1 == ub2 && st1 == st2) ||
        ((mk_binop(OP_SUB, ub1, lb1, astb.bnd.dtype) ==
          mk_binop(OP_SUB, ub2, lb2, astb.bnd.dtype)) &&
         st1 == st2)) {
      if ((is_avail_expr(lb1, std1, fg1, std2, fg2) &&
           is_avail_expr(ub1, std1, fg1, std2, fg2) &&
           is_avail_expr(st1, std1, fg1, std2, fg2)))
        same = TRUE;
    }
    if (!same)
      return FALSE;
  }

  if (!same)
    return FALSE;
  return TRUE;
}

static LOGICAL
same_forall_bnds(int lp1, int lp2, int nested)
{
  int idx1[7], idx2[7];
  int itriple1, itriple2;
  int lb1, ub1, st1;
  int lb2, ub2, st2;
  int std1, std2;
  int fg1, fg2;
  int hd1, hd2;
  int list1, list2, listp;
  int forall1, forall2;
  int nidx1, nidx2;
  int i, k;
  int asd1, asd2;
  int ndim1, ndim2;
  int order2[MAXDIMS];
  int no;
  int lhs1, lhs2, newlhs2, l, l2;
  int sptr1, sptr2;
  int isptr1, isptr2;
  int dim1, dim2;
  int newast, oldast;
  LOGICAL same;

  hd1 = LP_HEAD(lp1); /*the flow graph node which is the loop head*/
  fg1 = LP_FG(lp1);
  std1 = FG_STDFIRST(hd1);
  forall1 = STD_AST(std1);
  lhs1 = A_DESTG(A_IFSTMTG(forall1));
  list1 = A_LISTG(forall1);
  nidx1 = 0;
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx1[nidx1] = listp;
    ++nidx1;
  }

  hd2 = LP_HEAD(lp2);
  fg2 = LP_FG(lp2);
  if (!is_dominator(hd1, hd2))
    return FALSE;
  std2 = FG_STDFIRST(hd2);
  forall2 = STD_AST(std2);
  lhs2 = A_DESTG(A_IFSTMTG(forall2));
  list2 = A_LISTG(forall2);
  nidx2 = 0;
  for (listp = list2; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx2[nidx2] = listp;
    nidx2++;
  }

  if (nidx1 != nidx2)
    return FALSE;

  for (i = 0; i < nidx1 - nested; i++) {
    itriple1 = ASTLI_TRIPLE(idx1[i]);
    lb1 = A_LBDG(itriple1);
    ub1 = A_UPBDG(itriple1);
    st1 = A_STRIDEG(itriple1);
    if (st1 == 0)
      st1 = astb.i1;
    itriple2 = ASTLI_TRIPLE(idx2[i]);
    lb2 = A_LBDG(itriple2);
    ub2 = A_UPBDG(itriple2);
    st2 = A_STRIDEG(itriple2);
    if (st2 == 0)
      st2 = astb.i1;
    same = FALSE;
    if (lb1 == lb2 && ub1 == ub2 && st1 == st2) {
      if ((is_avail_expr(lb1, std1, fg1, std2, fg2) &&
           is_avail_expr(ub1, std1, fg1, std2, fg2) &&
           is_avail_expr(st1, std1, fg1, std2, fg2)))
        same = TRUE;
    }
    if (!same)
      return FALSE;
  }

  if (is_duplicate(lhs1, list1))
    return FALSE;
  if (is_duplicate(lhs2, list2))
    return FALSE;

  ast_visit(1, 1);
  for (i = 0; i < nidx1; i++) {
    isptr1 = ASTLI_SPTR(idx1[i]);
    newast = mk_id(isptr1);
    isptr2 = ASTLI_SPTR(idx2[i]);
    oldast = mk_id(isptr2);
    ast_replace(oldast, newast);
  }
  newlhs2 = ast_rewrite(lhs2);
  ast_unvisit();
  if (!is_ordered(lhs1, newlhs2, list1, order2, &no))
    return FALSE;

  /* has to have same distribution */
  l = left_subscript_ast(lhs1);
  sptr1 = left_array_symbol(lhs1);
  l2 = left_subscript_ast(newlhs2);
  sptr2 = left_array_symbol(newlhs2);

  assert(A_TYPEG(l) == A_SUBSCR, "same_forall_bnds: not a subscript", l, 4);
  asd1 = A_ASDG(l);
  ndim1 = ASD_NDIM(asd1);
  assert(A_TYPEG(l2) == A_SUBSCR, "same_forall_bnds: not a subscript", l, 4);
  asd2 = A_ASDG(l2);
  ndim2 = ASD_NDIM(asd2);

  for (i = 0; i < nidx1; i++) {
    isptr1 = ASTLI_SPTR(idx1[i]);
    dim1 = -1;
    dim2 = -1;
    for (k = 0; k < ndim1; k++)
      if (is_name_in_expr(ASD_SUBS(asd1, k), isptr1))
        dim1 = k;
    for (k = 0; k < ndim2; k++)
      if (is_name_in_expr(ASD_SUBS(asd2, k), isptr1))
        dim2 = k;
    if (dim1 == -1 || dim2 == -1) {
      /* index does not appear in any distributed dimensions;
       * until I figure out just what this is doing, be safe */
      return FALSE;
    }
    if (!is_same_align_shape(sptr1, dim1, sptr2, dim2))
      return FALSE;
    if (ASD_SUBS(asd1, dim1) != ASD_SUBS(asd2, dim2))
      return FALSE;
  }

  return TRUE;
}

static struct {
  int otherlhs;
  int src;
  int sink;
  int list;
  int after;
  int order;
  int forcomm;
} conf;

/* Return TRUE if there is conflict for loop fusion.
 * order is just for swap for src and sink at Conflict_.
 * forcomm is that there is Isno_comm TRUE, no need to test iff forcomm set. */
static LOGICAL
Conflict(int list, int src, int sink, int after, int order, int forcomm)
{
  conf.src = src;
  conf.sink = sink;
  conf.list = list;
  conf.after = after;
  conf.order = order;
  conf.forcomm = forcomm;
  return Conflict_(sink);
}

/* This routine will return TRUE iff,
 * lhs array appears at sink and their subscripts are different.
 */
static LOGICAL
Conflict_(int sink)
{
  LOGICAL l, r;
  int argt;
  int nargs;
  int i;
  int result;
  int src, sptr, sptr1;

  if (sink == 0)
    return FALSE;
  switch (A_TYPEG(sink)) {
  case A_CMPLXC:
  case A_CNST:
  case A_SUBSTR:
    return FALSE;
  case A_MEM:
    if (Conflict_(A_PARENTG(sink))) {
      /* see if this 'member' appears in the 'conf.src' tree */
      int a, p, member;
      a = conf.src;
      member = A_SPTRG(A_MEMG(sink));
      while (1) {
        switch (A_TYPEG(a)) {
        case A_SUBSCR:
          a = A_LOPG(a);
          break;
        case A_ID:
          return TRUE;
          break;
        case A_MEM:
          p = A_PARENTG(a);
          if (DDTG(A_DTYPEG(p)) == ENCLDTYPEG(member)) {
            /* same member? different? */
            if (A_MEMG(a) == A_MEMG(sink))
              return TRUE;
            return FALSE;
          }
          a = p;
          break;
        default:
          interr("Conflict_: unexpected AST in member tree", a, 3);
          return FALSE;
        }
      }
    }
    return FALSE;
  case A_ID:
    if (A_SPTRG(sink) == sym_of_ast(conf.src))
      return TRUE;
    else
      return FALSE;
  case A_BINOP:
    l = Conflict_(A_LOPG(sink));
    if (l)
      return TRUE;
    r = Conflict_(A_ROPG(sink));
    if (r)
      return TRUE;
    return FALSE;
  case A_UNOP:
    return Conflict_(A_LOPG(sink));
  case A_PAREN:
  case A_CONV:
    return Conflict_(A_LOPG(sink));
  case A_SUBSCR:
    if (Conflict_(A_LOPG(sink))) {
      if (conf.order)
        result = dd_array_conflict(conf.list, sink, conf.src, conf.after);
      else
        result = dd_array_conflict(conf.list, conf.src, sink, conf.after);
      return result;
    }
    return FALSE;
  case A_TRIPLE:
    l = Conflict_(A_LBDG(sink));
    if (l)
      return TRUE;
    r = Conflict_(A_UPBDG(sink));
    if (r)
      return TRUE;
    return Conflict_(A_STRIDEG(sink));
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(sink);
    argt = A_ARGSG(sink);
    /* dd_array_conflict does not work if the constructed section is
     * out of bound. a(10);forall(i=1:10) b(i) = cshift(a(i+2))
     * here a(3:12) is not in bounds.
     */
    if (A_OPTYPEG(sink) == I_CSHIFT || A_OPTYPEG(sink) == I_EOSHIFT) {
      src = ARGT_ARG(argt, 0);
      sptr = sym_of_ast(src);
      sptr1 = sym_of_ast(conf.src);
      if (sptr == sptr1)
        return TRUE;
    }
    for (i = 0; i < nargs; ++i) {
      l = Conflict_(ARGT_ARG(argt, i));
      if (l)
        return TRUE;
    }
    return FALSE;
  case A_LABEL:
  default:
    interr("Conflict_: unexpected ast", sink, 2);
    return FALSE;
  }
}

static LOGICAL
_olap_conflict(int expr, int expr1)
{
  LOGICAL l, r;
  int argt;
  int nargs;
  int i;

  if (expr == 0)
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_SUBSTR:
  case A_MEM:
    return FALSE;
  case A_BINOP:
    l = _olap_conflict(A_LOPG(expr), expr1);
    if (l)
      return TRUE;
    r = _olap_conflict(A_ROPG(expr), expr1);
    if (r)
      return TRUE;
    return FALSE;
  case A_UNOP:
    return _olap_conflict(A_LOPG(expr), expr1);
  case A_PAREN:
  case A_CONV:
    return _olap_conflict(A_LOPG(expr), expr1);
  case A_SUBSCR:
    return FALSE;
  case A_TRIPLE:
    l = _olap_conflict(A_LBDG(expr), expr1);
    if (l)
      return TRUE;
    r = _olap_conflict(A_UPBDG(expr), expr1);
    if (r)
      return TRUE;
    return _olap_conflict(A_STRIDEG(expr), expr1);
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      l = _olap_conflict(ARGT_ARG(argt, i), expr1);
      if (l)
        return TRUE;
    }
    if (A_OPTYPEG(expr) == I_CSHIFT || A_OPTYPEG(expr) == I_EOSHIFT)
      return is_shift_conflict(expr, argt, expr1);
    return FALSE;
  case A_LABEL:
  default:
    interr("_olap_conflict: unexpected ast", expr, 2);
    return FALSE;
  }
}

static LOGICAL
is_olap_conflict(int forall, int forall1)
{
  int expr, rhs;
  int expr1, rhs1;

  expr = A_IFEXPRG(forall);
  rhs = A_SRCG(A_IFSTMTG(forall));
  expr1 = A_IFEXPRG(forall1);
  rhs1 = A_SRCG(A_IFSTMTG(forall1));

  if (_olap_conflict(expr, expr1))
    return TRUE;
  if (_olap_conflict(expr, rhs1))
    return TRUE;
  if (_olap_conflict(expr1, expr))
    return TRUE;
  if (_olap_conflict(expr1, rhs))
    return TRUE;

  if (_olap_conflict(rhs, expr1))
    return TRUE;
  if (_olap_conflict(rhs, rhs1))
    return TRUE;
  if (_olap_conflict(rhs1, expr))
    return TRUE;
  if (_olap_conflict(rhs1, rhs))
    return TRUE;
  return FALSE;
}

/**
   \brief ...

   <pre>
     S1: A(i) = ...
     S2: ...  = ... A(f(i)) ...
   </pre>

   The loops cannot be fused if either of the following calls to
   dd_array_conflict() return TRUE:
   <pre>
     dd_array_conflict(tripletList, A(f(i)), A(i), FALSE)
     dd_array_conflict(tripletList, A(i), A(f(i)), TRUE)
   </pre>

   Alternatively, the statements within the original loops may be of the
   following form:

   <pre>
     S1: ...  = ... A(f(i))
     S2: A(i) = ...
   </pre>

   The loops cannot be fused if the following call to dd_array_conflict()
   returns TRUE.
   <pre>
     dd_array_conflict(tripletList, A(i), A(f(i)), FALSE)
   </pre>

   The back end recognizes loops with 0 RHS and transforms these into calls
   to _mzero, so don't fuse assignments with just 0 on the RHS.
 */
static LOGICAL
is_fusable(int lp, int lp1, int nested)
{

  int k;
  int std, std1;
  int forall, forall1;
  int nd, nd1;
  int hd, hd1;
  int expr, expr1;
  int asn, asn1;
  int rhs, rhs1;
  int lhs, lhs_array, lhs1, lhs1_array;
  int lhs_sptr, lhs_sptr1;
  int list, list1, listp;
  int isptr;
  int fg, fg1;
  LOGICAL fuseable;
  int fuselp;
  int triple;
  int idx[7];
  int cnt;
  LOGICAL fuse_cnst_rhs;

  if (XBIT(47, 0x80000000))
    return FALSE;

  hd = LP_HEAD(lp); /*the flow graph node which is the loop head*/
  fg = LP_FG(lp);
  std = FG_STDFIRST(hd);
  forall = STD_AST(std);
  list = A_LISTG(forall);
  /*
   * NOTES:
   * XBIT(47,0x4000000) - fuse even if constant RHS
   * XBIT(8,0x8000000)  - inhibit mem idioms
   */
  fuse_cnst_rhs = FALSE;
  if (XBIT(47, 0x4000000))
    fuse_cnst_rhs = TRUE;
  else if (XBIT(8, 0x8000000))
    fuse_cnst_rhs = TRUE;
  else {
  }

  expr = A_IFEXPRG(forall);
  asn = A_IFSTMTG(forall);
  rhs = A_SRCG(asn);
  lhs = A_DESTG(asn);
  for (lhs_array = lhs; A_TYPEG(lhs_array) == A_MEM;
       lhs_array = A_PARENTG(lhs_array))
    ;
  lhs_sptr = sym_of_ast(lhs);
  nd = A_OPT1G(forall);

  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return FALSE;

  if (A_TYPEG(rhs) == A_CONV)
    rhs = A_LOPG(rhs);
  if (flg.opt >= 2 && !fuse_cnst_rhs && A_TYPEG(rhs) == A_CNST) {
    /*
     * prefer calling a tuned mzero/mem rather fusing.
     */
    return FALSE;
  }
  /*the flow graph node which is the loop head */
  hd1 = LP_HEAD(lp1);
  fg1 = LP_FG(lp1);
  if (!is_dominator(hd, hd1))
    return FALSE;
  if (smp_conflict(hd, hd1))
    return FALSE;
  std1 = FG_STDFIRST(hd1);
  forall1 = STD_AST(std1);
  nd1 = A_OPT1G(forall1);
  list1 = A_LISTG(forall1);
  expr1 = A_IFEXPRG(forall1);
  asn1 = A_IFSTMTG(forall1);
  rhs1 = A_SRCG(asn1);
  lhs1 = A_DESTG(asn1);
  for (lhs1_array = lhs1; A_TYPEG(lhs1_array) == A_MEM;
       lhs1_array = A_PARENTG(lhs1_array))
    ;
  lhs_sptr1 = sym_of_ast(lhs1);
  if (A_TYPEG(rhs1) == A_CONV)
    rhs1 = A_LOPG(rhs1);
  if (flg.opt >= 2 && !fuse_cnst_rhs && A_TYPEG(rhs1) == A_CNST) {
    /*
     * prefer calling a tuned mzero/mem rather fusing.
     */
    return FALSE;
  }

  /* forall1 values should not changed between std and std1 */
  /* Except that if they are in block, nothing change them */
  if (!is_in_block(std, std1)) {
    cnt = 0;
    for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
      isptr = ASTLI_SPTR(listp);
      idx[cnt] = mk_id(isptr);
      cnt++;
    }
    if (!is_avail_expr_with_list(expr1, std, fg, std1, fg1, idx, cnt) ||
        !is_avail_expr_with_list(rhs1, std, fg, std1, fg1, idx, cnt) ||
        !is_avail_expr_with_list(lhs1, std, fg, std1, fg1, idx, cnt))
      return FALSE;
  }

  /* dependency */
  if (subscr_dependent(rhs1, lhs, std1, std))
    return FALSE;
  if (rhs1 && A_LOPG(rhs1) && A_TYPEG(rhs1) == A_SUBSCR &&
      A_TYPEG(A_LOPG(rhs1)) == A_MEM)
    if (subscr_dependent(A_LOPG(rhs1), lhs, std1, std))
      return FALSE;
  if (subscr_dependent(expr1, lhs, std1, std))
    return FALSE;
  if (expr1 && A_LOPG(expr1) && A_TYPEG(expr1) == A_SUBSCR &&
      A_TYPEG(A_LOPG(expr1)) == A_MEM)
    if (subscr_dependent(A_LOPG(expr1), lhs, std1, std))
      return FALSE;
  if (subscr_dependent(lhs1, lhs, std1, std))
    return FALSE;
  if (lhs1 && A_LOPG(lhs1) && A_TYPEG(lhs1) == A_SUBSCR &&
      A_TYPEG(A_LOPG(lhs1)) == A_MEM)
    if (subscr_dependent(A_LOPG(lhs1), lhs, std1, std))
      return FALSE;
  if (rhs && lhs1 && A_TYPEG(rhs) == A_SUBSCR && A_LOPG(lhs1) &&
      A_TYPEG(lhs1) == A_SUBSCR)
    if (subscr_dependent(rhs, A_LOPG(lhs1), std1, std))
      return FALSE;

  /* don't let LHSs appears at FORALL list */
  /* forall(i=x(1):x(2)) x(i) =   */
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    triple = ASTLI_TRIPLE(listp);
    if (A_TYPEG(lhs_array) == A_SUBSCR &&
        contains_ast(triple, A_LOPG(lhs_array)))
      return FALSE;
    if (A_TYPEG(lhs1_array) == A_SUBSCR &&
        contains_ast(triple, A_LOPG(lhs1_array)))
      return FALSE;
  }

  /* because of using mask pghpf_vsub_gather	*/
  /* this can be optimized, iff expr1 and rhs1
   * does not have indirections */
  if (A_TYPEG(lhs_array) == A_SUBSCR && contains_ast(expr1, A_LOPG(lhs_array)))
    return FALSE;

  if (expr1)
    if (Conflict(A_LISTG(forall), lhs, expr1, FALSE, 1, 0))
      return FALSE;
  if (Conflict(A_LISTG(forall), lhs, rhs1, FALSE, 1, 0))
    return FALSE;

  /* for communication */
  conf.otherlhs = lhs1;
  if (expr1)
    if (Conflict(A_LISTG(forall), lhs, expr1, TRUE, 0, 1))
      return FALSE;
  if (Conflict(A_LISTG(forall), lhs, rhs1, TRUE, 0, 1))
    return FALSE;

  if (expr)
    if (Conflict(A_LISTG(forall), lhs1, expr, FALSE, 0, 0))
      return FALSE;
  if (Conflict(A_LISTG(forall), lhs1, rhs, FALSE, 0, 0))
    return FALSE;

  if (is_branch_between(lp, lp1))
    return FALSE;
  if (!is_in_block(std, std1))
    if (A_TYPEG(lhs1_array) == A_SUBSCR &&
        is_contains_ast_between(lp, lp1, A_LOPG(lhs1_array)))
      return FALSE;

  if (is_olap_conflict(forall, forall1))
    return FALSE;
  fuseable = TRUE;
  for (k = 0; k < FT_NFUSE(nd, nested); k++) {
    fuselp = FT_FUSELP(nd, nested, k);
    if (!is_fusable(fuselp, lp1, nested))
      fuseable = FALSE;
  }
  if (!fuseable)
    return FALSE;

  return TRUE;
}

/*
 * Determine if two flowgraph nodes conflict with respect to smp
 * execution given that fg1 dominates fg2.
 */
static LOGICAL
smp_conflict(int fg1, int fg2)
{
  int fg;
  int std;
  int ast;

  if (!flg.smp)
    return FALSE;
  if (fg1 == fg2)
    /* no conflict if the same node */
    return FALSE;
  if ((FG_PAR(fg1) && !FG_PAR(fg2)) || (!FG_PAR(fg1) && FG_PAR(fg2)))
    /* fg1 is serial & fg2 is parallel, or vice versa */
    return TRUE;
  if (FG_PAR(fg2)) {
    /* both are within a parallel region; determine if it's the
     * same region.
     */
    for (fg = fg1; fg != fg2; fg = FG_LNEXT(fg)) {
      rdilts(fg);
      for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
        ast = STD_AST(std);
        if (A_TYPEG(ast) == A_MP_ENDPARALLEL) {
          /* endparallel was seen in a node before fg2 - it must
           * be a different parallel region.
           */
          wrilts(fg);
          return TRUE;
        }
      }
      wrilts(fg);
    }
  }
  if (FG_CS(fg1) ^ FG_CS(fg2))
    /* fg1 is in a critical section & fg2 is not, or vice versa */
    return TRUE;
  if (FG_CS(fg2)) {
    /* both are within a critical section; determine if it's the
     * same critical section.
     */
    for (fg = fg1; fg != fg2; fg = FG_LNEXT(fg)) {
      rdilts(fg);
      for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
        ast = STD_AST(std);
        if (A_TYPEG(ast) == A_MP_ENDCRITICAL) {
          wrilts(fg);
          return TRUE;
        }
      }
      wrilts(fg);
    }
  }
  if (FG_PARSECT(fg1) ^ FG_PARSECT(fg2))
    /* fg1 is in a parallel section (master, single, sections) & fg2 is
     * not, or vice versa
     */
    return TRUE;
  if (FG_PARSECT(fg2)) {
    /* both are within a parallel section; determine if it's the
     * same parallel section.
     */
    for (fg = fg1; fg != fg2; fg = FG_LNEXT(fg)) {
      rdilts(fg);
      for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
        ast = STD_AST(std);
        switch (A_TYPEG(ast)) {
        case A_MP_ENDMASTER:
        case A_MP_ENDSINGLE:
        case A_MP_ENDSECTIONS:
          wrilts(fg);
          return TRUE;
        }
      }
      wrilts(fg);
    }
  }

  return FALSE;
}

static LOGICAL
is_in_block(int std, int std1)
{
  int forall, forall1, forallh;
  int header, nextstd, fusedstd;
  int nd, k;

  forall = STD_AST(std);
  forall1 = STD_AST(std1);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  forallh = STD_AST(header);
  assert(A_TYPEG(forallh) == A_FORALL, "is_in_block: expecting forall", forallh,
         3);
  nd = A_OPT1G(forallh);
  assert(nd, "is_in_block: nd is 0", forallh, 3);
  for (k = 0; k < FT_NFUSE(nd, 0); k++) {
    fusedstd = FT_FUSEDSTD(nd, 0, k);
    nextstd = STD_NEXT(header);
    while (A_TYPEG(STD_AST(nextstd)) == A_CONTINUE)
      nextstd = STD_NEXT(nextstd);
    header = nextstd;
    if (nextstd == fusedstd)
      continue;
    return FALSE;
  }
  nextstd = STD_NEXT(header);
  while (A_TYPEG(STD_AST(nextstd)) == A_CONTINUE)
    nextstd = STD_NEXT(nextstd);
  return (nextstd == std1);
}

static LOGICAL
is_branch_between(int lp, int lp1)
{
  int ast, std;
  int hd, fg;
  int hd1, fg1;
  int type;

  hd = LP_HEAD(lp);
  fg = LP_FG(lp);
  hd1 = LP_HEAD(lp1);
  fg1 = LP_FG(lp1);

  if (lp == lp1)
    return FALSE;

  while (TRUE) {
    fg = FG_LNEXT(fg);
    if (fg == fg1)
      return FALSE;
    rdilts(fg);
    for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
      ast = STD_AST(std);
      type = A_TYPEG(ast);
      if (type != A_FORALL && type != A_DO && type != A_ENDDO) {
        if (STD_BR(std)) {
          wrilts(fg);
          return TRUE;
        }
      }
    }
    wrilts(fg);
  }
}

static LOGICAL
is_contains_ast_between(int lp, int lp1, int a)
{
  int ast, std;
  int hd, fg;
  int hd1, fg1;

  hd = LP_HEAD(lp);
  fg = LP_FG(lp);
  hd1 = LP_HEAD(lp1);
  fg1 = LP_FG(lp1);

  while (TRUE) {
    fg = FG_LNEXT(fg);
    if (fg == fg1)
      return FALSE;
    rdilts(fg);
    for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
      ast = STD_AST(std);
      if (contains_ast(ast, a)) {
        wrilts(fg);
        return TRUE;
      }
    }
    wrilts(fg);
  }
}

/* This routine is same as is_avail_expr except that
 * it will look at the cnt number of variables given by idx.
 * idx has variable A_ID ast. it will decompose_expression
 * and eliminate idx variable from list.
 * it helps forall triplet variables ignored.
 */
static LOGICAL
is_avail_expr_with_list(int expr, int std, int fg, int std1, int fg1, int idx[],
                        int cnt)
{
  int lst[100], lst1[100];
  int size, nvar;
  int ele;
  int found;
  int i, j;
  int nvar1;

  if (!expr)
    return TRUE;
  size = 100;
  nvar = 0;

  decompose_expression(expr, lst, size, &nvar, NULL);
  if (nvar > size)
    return FALSE;

  /* eliminate duplicate variables from the lst */
  nvar1 = 0;
  for (i = 0; i < nvar; i++) {
    found = 0;
    for (j = 0; j < nvar1; j++)
      if (lst[i] == lst1[j])
        found = 1;
    if (found)
      continue;
    lst1[nvar1] = lst[i];
    nvar1++;
  }

  for (i = 0; i < nvar1; i++) {
    ele = lst1[i];
    found = 0;
    for (j = 0; j < cnt; j++)
      if (idx[j] == ele)
        found = 1;
    if (found)
      continue;
    if (!is_avail_expr(ele, std, fg, std1, fg1))
      return FALSE;
  }
  return TRUE;
}

static void
put_forall_fg(void)
{
  int i, ii;
  int std, hd;
  int forall;
  int nd;
  for (ii = 1; ii <= opt.nloops; ++ii) {
    i = lpsort[ii];
    if (LP_FORALL(i)) {
      hd = LP_HEAD(i); /*the flow graph node which is the loop head*/
      std = FG_STDFIRST(hd);
      forall = STD_AST(std);
      nd = A_OPT1G(forall);
      FT_FG(nd) = hd;
    }
  }
}

static void
eliminate_rt(int type0)
{
  int i, j, k, l;
  int std, std1;
  int forall, forall1;
  int nd, nd1;
  int rt, rt1;
  int rt_std, rt1_std;
  int hd, hd1;
  int type, type1;
  int ii, jj;
  int header, header1;
  int header_forall, header_forall1;
  int header_fg, header_fg1;
  LITEMF *list_rt, *list_rt1;

  for (ii = 1; ii <= opt.nloops; ++ii) {
    i = lpsort[ii];
    if (LP_FORALL(i)) {
      hd = LP_HEAD(i); /*the flow graph node which is the loop head*/
      std = FG_STDFIRST(hd);
      forall = STD_AST(std);
      nd = A_OPT1G(forall);
      if (FT_NRT(nd) == 0)
        continue;
      for (jj = ii; jj <= opt.nloops; ++jj) {
        j = lpsort[jj];
        if (LP_FORALL(j)) {
          /*the flow graph node which is the loop head */
          hd1 = LP_HEAD(j);
          if (!is_dominator(hd, hd1))
            continue;
          std1 = FG_STDFIRST(hd1);
          forall1 = STD_AST(std1);
          nd1 = A_OPT1G(forall1);
          if (FT_NRT(nd1) == 0)
            continue;

          /* when foralls are fused,
           *	first header should dominate the second  header
           */
          header = FT_HEADER(nd);
          header1 = FT_HEADER(nd1);
          header_forall = STD_AST(header);
          header_forall1 = STD_AST(header1);
          assert(A_TYPEG(header_forall) == A_FORALL,
                 "eliminate_rt:expecting forall", header_forall, 2);
          assert(A_TYPEG(header_forall1) == A_FORALL,
                 "eliminate_rt:expecting forall1", header_forall1, 2);
          header_fg = FT_FG(A_OPT1G(header_forall));
          header_fg1 = FT_FG(A_OPT1G(header_forall1));
          if (header != header1)
            if (!is_dominator(header_fg, header_fg1))
              continue;

          list_rt = FT_RTL(nd);
          for (k = 0; k < FT_NRT(nd); k++) {
            rt_std = list_rt->item;
            list_rt = list_rt->next;
            rt = STD_AST(rt_std);
            if (STD_DELETE(rt_std))
              continue;
            list_rt1 = FT_RTL(nd1);
            for (l = 0; l < FT_NRT(nd1); l++) {
              rt1_std = list_rt1->item;
              list_rt1 = list_rt1->next;

              rt1 = STD_AST(rt1_std);
              type = A_TYPEG(rt);
              if (type == A_ASN)
                type = A_TYPEG(A_SRCG(rt));
              type1 = A_TYPEG(rt1);
              if (type1 == A_ASN)
                type1 = A_TYPEG(A_SRCG(rt1));
              if (STD_DELETE(rt1_std))
                continue;
              if (rt == rt1)
                continue;
              if (type != type1)
                continue;
              if (i == j)
                if (k > l)
                  continue;
              if (type == A_HCYCLICLP)
                type = A_HLOCALIZEBNDS;
              /* to share alloc after sharing comm_start
               * since comm_start can change alloc
               */

              if (!independent_commtype(type, type0))
                continue;

              /* all run-tme calls require some freeing
               * except localize bounds
               * a = b
               * if(i.eq.0) then
               *	    a =b
               * endif
               */
              /* You need post_dominator here for much better
               * optimization
               */
              if (type != A_HLOCALIZEBNDS && type != A_HCYCLICLP &&
                  is_branch_between(i, j))
                continue;

              switch (type) {
              case A_HLOCALIZEBNDS: /* type0==4 */
              case A_HCYCLICLP:     /* type0==4 */
                eliminate_bnds(i, j, rt_std, rt1_std);
                break;
              case A_HALLOBNDS: /* type0==3 */
                eliminate_alloc(i, j, rt_std, rt1_std);
                break;
              case A_HSECT: /* type0==4 */
                eliminate_sect(i, j, rt_std, rt1_std);
                break;
              case A_HCOPYSECT: /* type0==1 */
                eliminate_copy(i, j, rt_std, rt1_std);
                break;
              case A_HGATHER:  /* type0==1 */
              case A_HSCATTER: /* type0==1 */
                eliminate_gather(i, j, rt_std, rt1_std);
                break;
              case A_HOVLPSHIFT: /* type0==1 */
                eliminate_shift(i, j, rt_std, rt1_std);
                break;
              case A_HCSTART: /* type0==2 */
                eliminate_start(i, j, rt_std, rt1_std);
                break;
              default:
                break;
              }
            }
          }
        }
      }
    }
  }
}

/* This routine is used by eliminate_rt.
 * It is designed to eliminate independent rt at the same
 * call of eliminate_rt to speed up compilation.
 */
static LOGICAL
independent_commtype(int type, int type0)
{
  if (type0 == 1) {
    if (type == A_HGATHER || type == A_HSCATTER || type == A_HCOPYSECT ||
        type == A_HOVLPSHIFT)
      return TRUE;
    else
      return FALSE;
  } else if (type0 == 2) {
    if (type == A_HCSTART)
      return TRUE;
    else
      return FALSE;
  } else if (type0 == 3) {
    if (type == A_HALLOBNDS)
      return TRUE;
    else
      return FALSE;
  } else if (type0 == 4) {
    if (type == A_HSECT || type == A_HLOCALIZEBNDS)
      return TRUE;
    else
      return FALSE;
  } else
    assert(0, "merged_commtype:wrong-type", type0, 2);
  return FALSE;
}

static LOGICAL
is_same_descr_for_bnds(int rt, int rt1)
{
  int lhs, lhs1;
  int nd, nd1;
  int sptr, sptr1;
  int dim, dim1;

  nd = A_OPT1G(rt);
  nd1 = A_OPT1G(rt1);

  lhs = FT_BND_LHS(nd);
  sptr = left_array_symbol(lhs);
  dim = A_DIMG(rt);
  dim = get_int_cval(A_SPTRG(A_ALIASG(dim))) - 1;

  lhs1 = FT_BND_LHS(nd1);
  sptr1 = left_array_symbol(lhs1);
  dim1 = A_DIMG(rt1);
  dim1 = get_int_cval(A_SPTRG(A_ALIASG(dim1))) - 1;
  if (POINTERG(sptr) || POINTERG(sptr1))
    return FALSE;
  return is_same_align_shape(sptr, dim, sptr1, dim1);
}

static LOGICAL
is_same_align_shape(int sptr, int dim, int sptr1, int dim1)
{
  return TRUE;
}

static void
eliminate_bnds(int lp, int lp1, int rt_std, int rt1_std)
{
  int itriple, itriple1;
  int lb, ub, st;
  int lb1, ub1, st1;
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int hd, hd1;

  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_bnds: forall LP not set", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_bnds: forall1 LP not set", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  if (!is_same_descr_for_bnds(rt, rt1))
    return;

  nd = A_OPT1G(rt);
  nd1 = A_OPT1G(rt1);
  itriple = A_ITRIPLEG(rt);
  lb = A_LBDG(itriple);
  ub = A_UPBDG(itriple);
  st = A_STRIDEG(itriple);
  if (st == 0)
    st = astb.i1;

  itriple1 = A_ITRIPLEG(rt1);
  lb1 = A_LBDG(itriple1);
  ub1 = A_UPBDG(itriple1);
  st1 = A_STRIDEG(itriple1);
  if (st1 == 0)
    st1 = astb.i1;
  if (lb == lb1 && ub == ub1 && st == st1) {
    if (is_avail_expr(lb, std, fg, std1, fg1) &&
        is_avail_expr(ub, std, fg, std1, fg1) &&
        is_avail_expr(st, std, fg, std1, fg1)) {
      FT_BND_SAME(nd1) = rt;
      STD_DELETE(rt1_std) = 1;
      optsum.bnd++;
    }
  }
}

static LOGICAL
is_same_array_bounds(int sub, int sub1, int std, int std1, int fg, int fg1)
{
  int ndim, ndim1;
  int asd, asd1;
  int count;
  int lb, ub, st;
  int lb1, ub1, st1;
  int i;
  int itriple, itriple1;

  while (1) {
    if (A_TYPEG(sub) != A_TYPEG(sub1))
      return FALSE;
    switch (A_TYPEG(sub)) {
    case A_ID:
      return TRUE;
    case A_MEM:
      sub = A_PARENTG(sub);
      sub1 = A_PARENTG(sub1);
      break;
    case A_SUBSTR:
      sub = A_LOPG(sub);
      sub1 = A_LOPG(sub1);
      break;
    case A_SUBSCR:
      asd = A_ASDG(sub);
      ndim = ASD_NDIM(asd);
      asd1 = A_ASDG(sub1);
      ndim1 = ASD_NDIM(asd1);
      if (ndim != ndim1)
        return FALSE;
      count = 0;
      for (i = 0; i < ndim; i++) {
        itriple = ASD_SUBS(asd, i);
        if (A_TYPEG(itriple) == A_TRIPLE) {
          lb = A_LBDG(itriple);
          ub = A_UPBDG(itriple);
          st = A_STRIDEG(itriple);
          if (st == 0)
            st = astb.i1;
        } else {
          lb = itriple;
          ub = astb.i1;
          st = astb.i1;
        }

        itriple1 = ASD_SUBS(asd1, i);
        if (A_TYPEG(itriple1) == A_TRIPLE) {
          lb1 = A_LOPG(itriple1);
          ub1 = A_UPBDG(itriple1);
          st1 = A_STRIDEG(itriple1);
          if (st1 == 0)
            st1 = astb.i1;
        } else {
          lb1 = itriple1;
          ub1 = astb.i1;
          st1 = astb.i1;
        }

        if (lb == lb1 && ub == ub1 && st == st1) {
          if (is_avail_expr(lb, std, fg, std1, fg1) &&
              is_avail_expr(ub, std, fg, std1, fg1) &&
              is_avail_expr(st, std, fg, std1, fg1))
            count++;
        }
      }
      if (count != ndim)
        return FALSE;
      sub = A_LOPG(sub);
      sub1 = A_LOPG(sub1);
      break;
    default:
      interr("is_same_array_bounds: unexpected AST type ", sub, 0);
      return FALSE;
    }
  }
}

/* This routine checks whether sub and sub1 arrays have same subscripts
 * it can allows that they can have different subscripts iff
 *    1-if subscript is scalar and collapsed.
 *    2-if trailing subscripts are scalars and they are collapsed.
 * Also, each array dimension extent must be the same, up to the
 * last nonscalar/nondistributed dimension.
 */
static LOGICAL
is_same_array_bounds_for_schedule(int sub, int sub1, int std, int std1, int fg,
                                  int fg1)
{
  return TRUE;
}

LOGICAL
is_same_array_alignment(int sptr, int sptr1)
{
  return TRUE;
}

/* This routine is used for schedule elimination.
 * It checks if the arrays are aligned to the same template
 * with the same number of distribution and
 * aligned to the same template axis.
 * And also overlap area has to be the same.
 */
static LOGICAL
is_same_array_alignment_for_schedule(int sptr, int sptr1)
{
  return TRUE;
}

#if DEBUG
int
find_lp(int std)
{
  int i;
  int hd1;
  int std1;

  for (i = 1; i <= opt.nloops; ++i) {
    hd1 = LP_HEAD(i);
    std1 = FG_STDFIRST(hd1);
    if (std == std1)
      return i;
  }
  assert(0, "find_lp: loop not found", std, 3);
  return 0;
}
#endif

static void
eliminate_alloc(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int sub;
  int sub1;
  int sptr;
  int sptr1;
  int hd, hd1;
  int freestd, freestd1;

  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return;
  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_alloc: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_alloc: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  /* has to have same distribution */
  nd = A_OPT1G(rt);
  nd1 = A_OPT1G(rt1);
  if (FT_ALLOC_USED(nd1))
    return;
  if (FT_ALLOC_USED(nd))
    return;

  /* has to be disjoint live time */
  /* first free has to dominate the second alloc */
  if (lp == lp1)
    return;
  freestd = FT_ALLOC_FREE(nd);
  if (freestd == std1)
    return;
  if (!is_dominator_fg(freestd, std1))
    return;

  freestd1 = FT_ALLOC_FREE(nd1);
  if (freestd == freestd1)
    return;
  if (!is_dominator_fg(freestd, freestd1))
    return;

  sptr = FT_ALLOC_SPTR(nd);
  sptr1 = FT_ALLOC_SPTR(nd1);
  if (!is_same_array_alignment(sptr, sptr1))
    return;

  /* has to have the same type */
  if (DTY(DTYPEG(sptr) + 1) != DTY(DTYPEG(sptr1) + 1))
    return;

  /* can not be re-used at the same loop twice */
  if (FT_ALLOC_FREE(nd) == FT_ALLOC_FREE(nd1))
    return;

  /* has to have the same alloc bounds */
  sub = A_LOPG(rt);
  sub1 = A_LOPG(rt1);
  if (is_same_array_bounds(sub, sub1, std, std1, fg, fg1)) {
    FT_ALLOC_SAME(nd1) = rt;
    FT_ALLOC_OUT(nd1) = FT_ALLOC_OUT(nd);
    if (is_dominator_fg(FT_ALLOC_FREE(nd), FT_ALLOC_FREE(nd1)))
      FT_ALLOC_FREE(nd) = FT_ALLOC_FREE(nd1);
    FT_ALLOC_REUSE(nd) = 1;
    STD_DELETE(rt1_std) = 1;
    optsum.alloc++;
  }
}

LOGICAL
is_same_array_shape(int sptr, int sptr1)
{
  LOGICAL result;
  ADSC *ad, *ad1;
  int ndim, ndim1;
  int lb, lb1;
  int ub, ub1;
  int i;

  result = TRUE;
  if (sptr == sptr1)
    return TRUE;

  if (ALLOCG(sptr) || ALLOCG(sptr1))
    result = FALSE;

  ad = AD_DPTR(DTYPEG(sptr));
  ad1 = AD_DPTR(DTYPEG(sptr1));
  ndim = rank_of_sym(sptr);
  ndim1 = rank_of_sym(sptr1);

  if (ndim != ndim1)
    result = FALSE;

  for (i = 0; i < ndim; i++) {
    lb = AD_LWAST(ad, i);
    lb1 = AD_LWAST(ad1, i);
    if (lb != lb1)
      result = FALSE;
    ub = AD_UPAST(ad, i);
    ub1 = AD_UPAST(ad1, i);
    if (ub != ub1)
      result = FALSE;
  }

  return result;
}

static void
eliminate_sect(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int sub;
  int sub1;
  int sptr;
  int sptr1;
  int hd, hd1;
  int sect, sect1;
  int allocstd, allocstd1;
  int alloc, alloc1;
  int nd3;
  int arr, arr1;
  int sectflag, sectflag1;
  int bogus, bogus1;

  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return;
  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_sect: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_sect: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  sect = A_SRCG(rt);
  sect1 = A_SRCG(rt1);
  nd = A_OPT1G(sect);
  nd1 = A_OPT1G(sect1);
  arr = A_LOPG(sect);
  arr1 = A_LOPG(sect1);

  /* has to have same flag */
  sectflag = FT_SECT_FLAG(nd);
  sectflag1 = FT_SECT_FLAG(nd1);
  if (sectflag != sectflag1)
    return;

  allocstd = FT_SECT_ALLOC(nd);
  if (allocstd) {
    alloc = STD_AST(allocstd);
    nd3 = A_OPT1G(alloc);
    sptr = FT_ALLOC_OUT(nd3);
    FT_SECT_SPTR(nd) = sptr;
    bogus = getbit(sectflag, 8);
    if (is_whole_array(arr) && !bogus) {
      DESCUSEDP(sptr, 1);
      FT_SECT_OUT(nd) = DESCRG(sptr);
    }
  }

  allocstd1 = FT_SECT_ALLOC(nd1);
  if (allocstd1) {
    alloc1 = STD_AST(allocstd1);
    nd3 = A_OPT1G(alloc1);
    sptr = FT_ALLOC_OUT(nd3);
    FT_SECT_SPTR(nd1) = sptr;
    bogus1 = getbit(sectflag1, 8);
    if (is_whole_array(arr1) && !bogus1) {
      DESCUSEDP(sptr, 1);
      FT_SECT_OUT(nd1) = DESCRG(sptr);
    }
  }

  /* has to have same distribution */
  sptr = FT_SECT_SPTR(nd);
  if (STYPEG(sptr) == ST_MEMBER)
    return;
  sptr1 = FT_SECT_SPTR(nd1);
  if (!is_same_array_alignment(sptr, sptr1))
    return;

  /* has to have the same type */
  if (DTY(DTYPEG(sptr) + 1) != DTY(DTYPEG(sptr1) + 1))
    return;

  if (!is_same_array_shape(sptr, sptr1))
    return;

  /* has to have the same section bounds */
  sub = A_LOPG(sect);
  sub1 = A_LOPG(sect1);
  if (is_same_array_bounds(sub, sub1, std, std1, fg, fg1)) {
    FT_SECT_SAME(nd1) = rt;
    FT_SECT_OUT(nd1) = FT_SECT_OUT(nd);
    if (is_dominator_fg(FT_SECT_FREE(nd), FT_SECT_FREE(nd1)))
      FT_SECT_FREE(nd) = FT_SECT_FREE(nd1);
    FT_SECT_REUSE(nd) = 1;
    STD_DELETE(rt1_std) = 1;
    optsum.sect++;
  }
}

static LOGICAL
is_dominator_fg(int std, int std1)
{
  int hd, hd1;

  hd = STD_FG(std);
  hd1 = STD_FG(std1);
  if (is_dominator(hd, hd1))
    return TRUE;
  return FALSE;
}

/*
 * if fg1 == fg2, then return TRUE if std1 precedes std2 and FALSE otherwise
 * if fg1 != fg2, then return TRUE if fg1 dominates fg2,
 *  or the postdominator of fg1 dominates fg2,
 *  or some postdominator of fg1 dominates fg2, and so on
 */
static LOGICAL
must_follow(int fg1, int std1, int fg2, int std2)
{
  int std, found1, fg;
  if (fg1 == fg2) {
    rdilts(fg1);
    found1 = FALSE;
    for (std = FG_STDFIRST(fg1); std; std = STD_NEXT(std)) {
      if (std == std1) {
        found1 = TRUE;
      }
      if (std == std2) {
        wrilts(fg1);
        return found1;
      }
    }
    wrilts(fg1);
    /* didn't find std2 */
  } else {
    for (fg = fg1; fg > 0; fg = FG_PDOM(fg)) {
      if (is_dominator(fg, fg2))
        return TRUE;
    }
  }
  return FALSE;
} /* must_follow */

static void
eliminate_copy(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1, nd2;
  int sub;
  int sub1;
  int sptr;
  int sptr1;
  int hd, hd1;
  int copy, copy1;
  int lhs, lhs1;
  int rhs, rhs1;
  int sect;

  if (XBIT(47, 0x40000))
    return;
  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return;
  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_copy: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_copy: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  copy = A_SRCG(rt);
  copy1 = A_SRCG(rt1);
  nd = A_OPT1G(copy);
  nd1 = A_OPT1G(copy1);

  /* has to have same left-hand-side distribution */
  lhs = FT_CCOPY_LHS(nd);
  lhs1 = FT_CCOPY_LHS(nd1);
  sptr = left_array_symbol(lhs);
  sptr1 = left_array_symbol(lhs1);
  if (!is_same_array_alignment_for_schedule(sptr, sptr1))
    return;

  /* has to have same temp distribution */
  sptr = FT_CCOPY_TSPTR(nd);
  sptr1 = FT_CCOPY_TSPTR(nd1);
  if (!is_same_array_alignment_for_schedule(sptr, sptr1))
    return;

  /* has to have same right-hand-side distribution */
  rhs = FT_CCOPY_RHS(nd);
  rhs1 = FT_CCOPY_RHS(nd1);
  sptr = left_array_symbol(rhs);
  sptr1 = left_array_symbol(rhs1);
  if (!is_same_array_alignment_for_schedule(sptr, sptr1))
    return;

  /* has to have the same tmps bounds */
  sub = A_DESTG(copy);
  sub1 = A_DESTG(copy1);
  if (!is_same_array_bounds_for_schedule(sub, sub1, std, std1, fg, fg1))
    return;

  /* has to have the same rhs bounds */
  sub = A_SRCG(copy);
  sub1 = A_SRCG(copy1);
  if (!is_same_array_bounds_for_schedule(sub, sub1, std, std1, fg, fg1))
    return;

  /* neither source or destination sections can have bogus flags */
  sect = FT_CCOPY_SECTL(nd);
  sub = STD_AST(sect);
  nd2 = A_OPT1G(A_SRCG(sub));
  if (FT_SECT_FLAG(nd2) & BOGUSFLAG)
    return;
  sect = FT_CCOPY_SECTR(nd);
  sub = STD_AST(sect);
  nd2 = A_OPT1G(A_SRCG(sub));
  if (FT_SECT_FLAG(nd2) & BOGUSFLAG)
    return;
  sect = FT_CCOPY_SECTL(nd1);
  sub = STD_AST(sect);
  nd2 = A_OPT1G(A_SRCG(sub));
  if (FT_SECT_FLAG(nd2) & BOGUSFLAG)
    return;
  sect = FT_CCOPY_SECTR(nd1);
  sub = STD_AST(sect);
  nd2 = A_OPT1G(A_SRCG(sub));
  if (FT_SECT_FLAG(nd2) & BOGUSFLAG)
    return;

  /* has to have the same lhs bounds */
  sub = FT_CCOPY_LHSSEC(nd);
  sub1 = FT_CCOPY_LHSSEC(nd1);
  if (sub && sub1 &&
      !is_same_array_bounds_for_schedule(sub, sub1, std, std1, fg, fg1)) {
    FT_CCOPY_NOTLHS(nd) = 1;
    FT_CCOPY_NOTLHS(nd1) = 1;
  }

  if (FT_CCOPY_NOTLHS(nd) || FT_CCOPY_NOTLHS(nd1)) {
    FT_CCOPY_NOTLHS(nd) = 1;
    FT_CCOPY_NOTLHS(nd1) = 1;
  }

  FT_CCOPY_SAME(nd1) = rt;
  FT_CCOPY_OUT(nd1) = FT_CCOPY_OUT(nd);
  if (is_dominator_fg(FT_CCOPY_FREE(nd), FT_CCOPY_FREE(nd1)))
    FT_CCOPY_FREE(nd) = FT_CCOPY_FREE(nd1);
  FT_CCOPY_REUSE(nd) = 1;
  STD_DELETE(rt1_std) = 1;
  optsum.copysection++;
}

static void
eliminate_gather(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int i;
  int vsub, ndim, asd;
  int vsub1, ndim1, asd1;
  int hd, hd1;
  int gather, gather1;
  int nvsub, nvsub1;
  int mask, mask1;
  int sptr, sptr1;
  int v, v1;
  int per, per1;
  int sub, sub1;

  if (XBIT(47, 0x80000))
    return;
  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return;
  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_gather: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_gather: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  gather = A_SRCG(rt);
  gather1 = A_SRCG(rt1);
  nd = A_OPT1G(gather);
  nd1 = A_OPT1G(gather1);

  /* flags hast to be same */
  if (FT_CGATHER_TYPE(nd) != FT_CGATHER_TYPE(nd1))
    return;
  if (FT_CGATHER_VFLAG(nd) != FT_CGATHER_VFLAG(nd1))
    return;
  if (FT_CGATHER_PFLAG(nd) != FT_CGATHER_PFLAG(nd1))
    return;

  /* vsub has to have same distribution */
  vsub = FT_CGATHER_VSUB(nd);
  vsub1 = FT_CGATHER_VSUB(nd1);
  sptr = left_array_symbol(vsub);
  sptr1 = left_array_symbol(vsub1);
  if (!is_same_array_alignment_for_schedule(sptr, sptr1))
    return;

  /* nvsub has to have same distribution */
  nvsub = FT_CGATHER_NVSUB(nd);
  nvsub1 = FT_CGATHER_NVSUB(nd1);
  sptr = left_array_symbol(nvsub);
  sptr1 = left_array_symbol(nvsub1);
  if (!is_same_array_alignment_for_schedule(sptr, sptr1))
    return;

  /* has to have the same vsub bounds */
  if (!is_same_array_bounds_for_schedule(vsub, vsub1, std, std1, fg, fg1))
    return;

  /* has to have the same nvsub bounds */
  if (!is_same_array_bounds_for_schedule(nvsub, nvsub1, std, std1, fg, fg1))
    return;

  /* masks have to have contents */
  mask = FT_CGATHER_MASK(nd);
  mask1 = FT_CGATHER_MASK(nd1);
  if (mask != mask1)
    return;
  if (mask) {
    if (!is_avail_expr(mask, std, fg, std1, fg1))
      return;
  }

  asd = A_ASDG(left_subscript_ast(vsub));
  ndim = ASD_NDIM(asd);

  asd1 = A_ASDG(left_subscript_ast(vsub1));
  ndim1 = ASD_NDIM(asd1);

  if (ndim != ndim1)
    return;

  for (i = 0; i < ndim; i++) {
    v = FT_CGATHER_V(nd, i);
    v1 = FT_CGATHER_V(nd1, i);
    if (v != v1)
      return;
    if (v) {
      if (!is_avail_expr(v, std, fg, std1, fg1))
        return;
    }
    /* has to have same permute */
    per = FT_CGATHER_PERMUTE(nd, i);
    per1 = FT_CGATHER_PERMUTE(nd1, i);
    if (per != per1)
      return;
  }

  /* has to have the same lhs bounds */
  sub = FT_CGATHER_LHSSEC(nd);
  sub1 = FT_CGATHER_LHSSEC(nd1);
  if (sub && sub1 &&
      !is_same_array_bounds_for_schedule(sub, sub1, std, std1, fg, fg1)) {
    FT_CGATHER_NOTLHS(nd) = 1;
    FT_CGATHER_NOTLHS(nd1) = 1;
  }

  if (FT_CGATHER_NOTLHS(nd) || FT_CGATHER_NOTLHS(nd1)) {
    FT_CGATHER_NOTLHS(nd) = 1;
    FT_CGATHER_NOTLHS(nd1) = 1;
  }

  FT_CGATHER_SAME(nd1) = rt;
  FT_CGATHER_OUT(nd1) = FT_CGATHER_OUT(nd);
  FT_CGATHER_FREE(nd) = FT_CGATHER_FREE(nd1);
  FT_CGATHER_REUSE(nd) = 1;
  STD_DELETE(rt1_std) = 1;
  if (FT_CGATHER_TYPE(nd) == A_HGATHER)
    optsum.gatherx++;
  else
    optsum.scatterx++;
}

static void
eliminate_shift(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int i;
  int ndim, asd;
  int ndim1, asd1;
  int src, src1, srcl;
  int hd, hd1;
  int shift, shift1;
  int sptr, sptr1;
  int v, v1, cv, cv1, ns, ns1;
  int nmax, pmax;
  int sub[7];
  int new;

  if (XBIT(47, 0x100000))
    return;
  if (LP_PARENT(lp) != LP_PARENT(lp1))
    return;
  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_shift: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_shift: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  shift = A_SRCG(rt);
  shift1 = A_SRCG(rt1);
  nd = A_OPT1G(shift);
  nd1 = A_OPT1G(shift1);

  if (FT_SHIFT_TYPE(nd1) != FT_SHIFT_TYPE(nd))
    return;
  if (FT_SHIFT_BOUNDARY(nd1) != FT_SHIFT_BOUNDARY(nd))
    return;
  if (FT_SHIFT_BOUNDARY(nd) &&
      !is_avail_expr(FT_SHIFT_BOUNDARY(nd), std, fg, std1, fg1))
    return;
  src = A_SRCG(shift);
  sptr = left_array_symbol(src);

  src1 = A_SRCG(shift1);
  sptr1 = left_array_symbol(src1);

  if (!is_same_array_alignment(sptr, sptr1))
    return;

  if (!is_same_array_shape(sptr, sptr1))
    return;

  /* has to have the same shift values */
  /* second shift has to be less than equal to first shift
   * at all dimension negative and positive direction
   */

  srcl = left_subscript_ast(src);
  asd = A_ASDG(srcl);
  asd1 = A_ASDG(left_subscript_ast(src1));
  ndim = ASD_NDIM(asd);
  ndim1 = ASD_NDIM(asd1);
  if (ndim != ndim1)
    return;
  for (i = 0; i < ndim; i++) {
    v = ASD_SUBS(asd, i);
    v1 = ASD_SUBS(asd1, i);
    assert(A_TYPEG(v) == A_TRIPLE, "eliminate_shift: expecting triple", v, 3);
    assert(A_TYPEG(v1) == A_TRIPLE, "eliminate_shift: expecting triple1", v1,
           3);
    ns = A_LBDG(v);
    ns1 = A_LBDG(v1);
    assert(A_TYPEG(ns) == A_CNST, "eliminate_shift:expecting constant", v, 3);
    assert(A_TYPEG(ns1) == A_CNST, "eliminate_shift:expecting constant1", v1,
           3);
    cv = get_int_cval(A_SPTRG(A_ALIASG(ns)));
    cv1 = get_int_cval(A_SPTRG(A_ALIASG(ns1)));
    nmax = ns;
    if (cv < cv1)
      nmax = ns1;

    ns = A_UPBDG(v);
    ns1 = A_UPBDG(v1);
    assert(A_TYPEG(ns) == A_CNST, "eliminate_shift:expecting constant2", v, 3);
    assert(A_TYPEG(ns1) == A_CNST, "eliminate_shift:expecting constant3", v, 3);
    cv = get_int_cval(A_SPTRG(A_ALIASG(ns)));
    cv1 = get_int_cval(A_SPTRG(A_ALIASG(ns1)));
    pmax = ns;
    if (cv < cv1)
      pmax = ns1;
    sub[i] = mk_triple(nmax, pmax, 0);
  }

  new = mk_subscr(A_LOPG(srcl), sub, ndim, DTY(DTYPEG(sptr) + 1));
  new = replace_ast_subtree(src, srcl, new);
  A_SRCP(shift, new);

  FT_SHIFT_SAME(nd1) = rt;
  FT_SHIFT_OUT(nd1) = FT_SHIFT_OUT(nd);
  if (is_dominator_fg(FT_SHIFT_FREE(nd), FT_SHIFT_FREE(nd1)))
    FT_SHIFT_FREE(nd) = FT_SHIFT_FREE(nd1);
  FT_SHIFT_REUSE(nd) = 1;
  STD_DELETE(rt1_std) = 1;
  optsum.shift++;
}

static void
eliminate_start(int lp, int lp1, int rt_std, int rt1_std)
{
  int std, std1;
  int fg, fg1;
  int rt, rt1;
  int nd, nd1;
  int sub;
  int sub1;
  int sptr, sptr1;
  int ast, ast1;
  int hd, hd1;
  int start, start1;
  int cp, cp1;
  int rhs;
  int comm, commstd;
  int comm1, commstd1;
  int asn, asn1;
  int nd3, nd4;
  int stype, stype1;
  int alloc_std, alloc;
  int nd5;
  int src, src1;
  int dest, dest1;

  rt = STD_AST(rt_std);
  rt1 = STD_AST(rt1_std);
  assert(LP_FORALL(lp), "eliminate_start: expecting forall", lp, 2);
  hd = LP_HEAD(lp);
  std = FG_STDFIRST(hd);
  fg = LP_FG(lp);

  assert(LP_FORALL(lp1), "eliminate_start: expecting forall1", lp1, 2);
  hd1 = LP_HEAD(lp1);
  std1 = FG_STDFIRST(hd1);
  fg1 = LP_FG(lp1);

  start = A_SRCG(rt);
  start1 = A_SRCG(rt1);
  nd = A_OPT1G(start);
  nd1 = A_OPT1G(start1);

  stype = FT_CSTART_TYPE(nd);
  stype1 = FT_CSTART_TYPE(nd1);
  if (stype != stype1)
    return;

  /* has to have same cp */
  commstd = FT_CSTART_COMM(nd);
  asn = STD_AST(commstd);
  comm = A_SRCG(asn);
  nd3 = A_OPT1G(comm);
  if (stype == A_HCOPYSECT)
    cp = FT_CCOPY_OUT(nd3);
  if (stype == A_HOVLPSHIFT)
    cp = FT_SHIFT_OUT(nd3);
  if (stype == A_HGATHER)
    cp = FT_CGATHER_OUT(nd3);
  if (stype == A_HSCATTER)
    cp = FT_CGATHER_OUT(nd3);

  commstd1 = FT_CSTART_COMM(nd1);
  asn1 = STD_AST(commstd1);
  comm1 = A_SRCG(asn1);
  nd4 = A_OPT1G(comm1);
  if (stype1 == A_HCOPYSECT)
    cp1 = FT_CCOPY_OUT(nd4);
  if (stype1 == A_HOVLPSHIFT)
    cp1 = FT_SHIFT_OUT(nd4);
  if (stype1 == A_HGATHER)
    cp1 = FT_CGATHER_OUT(nd4);
  if (stype1 == A_HSCATTER)
    cp1 = FT_CGATHER_OUT(nd4);

  if (cp != cp1)
    return;

  /* has to have the same source array */
  sub = A_SRCG(start);
  sub1 = A_SRCG(start1);
  ast = left_subscript_ast(sub);
  ast1 = left_subscript_ast(sub1);
  if (A_LOPG(ast) != A_LOPG(ast1))
    return;

  rhs = A_LOPG(FT_CSTART_RHS(nd1));
  if (std != std1)
    if (!is_avail_expr(rhs, std, fg, std1, fg1))
      return;

  /* scatterx needs also destination is same */
  if (stype1 == A_HSCATTER) {
    sub = A_DESTG(start);
    sub1 = A_DESTG(start1);
    ast = left_subscript_ast(sub);
    ast1 = left_subscript_ast(sub1);
    if (A_LOPG(ast) != A_LOPG(ast1))
      return;
  }

  src = A_SRCG(start);
  src1 = A_SRCG(start1);

  dest = A_DESTG(start);
  dest1 = A_DESTG(start1);

  if (stype == A_HCOPYSECT || stype == A_HGATHER || stype == A_HSCATTER) {

    /* has to have same destination distribution */
    sptr = left_array_symbol(dest);
    sptr1 = left_array_symbol(dest1);
    if (!is_same_array_alignment(sptr, sptr1))
      return;

    /* has to have same src distribution */
    sptr = left_array_symbol(src);
    sptr1 = left_array_symbol(src1);
    if (!is_same_array_alignment(sptr, sptr1))
      return;

    /* has to have the same dest bounds */
    if (!is_same_array_bounds(dest, dest1, std, std1, fg, fg1))
      return;

    /* has to have the same src bounds */
    if (!is_same_array_bounds(src, src1, std, std1, fg, fg1))
      return;
  }

  FT_CSTART_SAME(nd1) = rt;
  FT_CSTART_OUT(nd1) = FT_CSTART_OUT(nd);
  if (FT_CSTART_ALLOC(nd1) && FT_CSTART_ALLOC(nd)) {
    FT_CSTART_ALLOC(nd1) = alloc_std = FT_CSTART_ALLOC(nd);
    alloc = STD_AST(alloc_std);
    nd5 = A_OPT1G(alloc);
    if (is_dominator_fg(FT_ALLOC_FREE(nd5), FT_CSTART_FREE(nd1)))
      FT_ALLOC_FREE(nd5) = FT_CSTART_FREE(nd1);
    FT_ALLOC_USED(nd5) = 1;
  }
  if (is_dominator_fg(FT_CSTART_FREE(nd), FT_CSTART_FREE(nd1)))
    FT_CSTART_FREE(nd) = FT_CSTART_FREE(nd1);
  FT_CSTART_REUSE(nd) = 1;
  STD_DELETE(rt1_std) = 1;
  optsum.start++;
}

static void
eliminate_get_scalar(void)
{
  int i, j;
  int src, src1;
  int fg, fg1;
  int commstd, commstd1;
  int rt, rt1;
  int nd, nd1;

  init_gstbl();
  find_get_scalar();

  for (i = 0; i < gstbl.avl; i++) {
    commstd = gstbl.base[i].f1;
    if (STD_DELETE(commstd))
      continue;
    rt = STD_AST(commstd);
    nd = A_OPT1G(rt);
    assert(A_TYPEG(rt) == A_HGETSCLR, "generate_get_scalar: wrong ast type", 2,
           rt);
    assert(nd, "generate_get_scalar: nd is 0", 2, rt);
    for (j = i + 1; j < gstbl.avl; j++) {
      commstd1 = gstbl.base[j].f1;
      if (STD_DELETE(commstd1))
        continue;
      rt1 = STD_AST(commstd1);
      src = A_SRCG(rt);
      src1 = A_SRCG(rt1);
      if (src != src1)
        continue;

      nd1 = A_OPT1G(rt1);
      fg = STD_FG(commstd);
      fg1 = STD_FG(commstd1);
      if (!is_dominator(fg, fg1))
        continue;
      if (!is_avail_expr(src, commstd, fg, commstd1, fg1))
        continue;
      FT_GETSCLR_SAME(nd1) = rt;
      FT_GETSCLR_REUSE(nd) = 1;
      STD_DELETE(commstd1) = 1;
    }
  }
  free_gstbl();
}

void
init_gstbl(void)
{
  gstbl.size = 200;
  NEW(gstbl.base, TABLE, gstbl.size);
  gstbl.avl = 0;
}

void
free_gstbl(void)
{

  FREE(gstbl.base);
}

int
get_gstbl(void)
{
  int nd;

  nd = gstbl.avl++;
  NEED(gstbl.avl, gstbl.base, TABLE, gstbl.size, gstbl.size + 100);
  if (nd > SPTR_MAX || gstbl.base == NULL)
    errfatal(7);
  return nd;
}

void
init_brtbl(void)
{
  brtbl.size = 200;
  NEW(brtbl.base, TABLE, brtbl.size);
  brtbl.avl = 0;
}

void
free_brtbl(void)
{
  FREE(brtbl.base);
}

int
get_brtbl(void)
{
  int nd;

  nd = brtbl.avl++;
  NEED(brtbl.avl, brtbl.base, TABLE, brtbl.size, brtbl.size + 100);
  if (nd > SPTR_MAX || brtbl.base == NULL)
    errfatal(7);
  return nd;
}

static void
comm_optimize_init(void)
{
  optshrd_init();
  induction_init();
  optshrd_finit();
}

static void
comm_optimize_end(void)
{
  optshrd_fend();
  optshrd_end();
  induction_end();
}

/* optimization table */

void
init_ftb(void)
{
  ftb.size = 240;
  NEW(ftb.base, FT, ftb.size);
  ftb.avl = 1;
}

int
mk_ftb(void)
{
  int nd;

  nd = ftb.avl++;
  NEED(ftb.avl, ftb.base, FT, ftb.size, ftb.size + 240);
  if (ftb.base == NULL)
    errfatal(7);
  return nd;
}

LITEMF *
clist(void)
{
  LITEMF *list;

  list = (LITEMF *)getitem(FORALL_AREA, sizeof(LITEMF));
  list->nitem = 0;
  list->next = 0;
  return list;
}

void
plist(LITEMF *list, int item)
{
  LITEMF *listp, *last;
  assert(list, "plist: list is NULL", 0, 3);
  if (list->nitem == 0) {
    list->item = item;
    list->next = 0;
    list->nitem = 1;
    return;
  }
  for (listp = list; listp != 0; listp = listp->next)
    last = listp;
  listp = (LITEMF *)getitem(FORALL_AREA, sizeof(LITEMF));
  listp->item = item;
  listp->next = 0;
  last->next = listp;
  list->nitem++;
}

int
glist(LITEMF *list, int n)
{
  LITEMF *listp;
  int i;

  assert(list->nitem > n, "glist: nitem not >", n, 0);
  listp = list;
  for (i = 0; i < list->nitem; i++) {
    if (i == n)
      return listp->item;
    listp = listp->next;
  }
  return 0;
}

/* Is this item in the list? */
LOGICAL
inlist(LITEMF *list, int item)
{
  LITEMF *p;
  if (list->nitem == 0)
    return FALSE;  /* list->item is invalid */
  for (p = list; p != 0; p = p->next) {
    if (p->item == item)
      return TRUE;
  }
  return FALSE;
}

/* Dump this list of ints. */
void
dlist(LITEMF *list)
{
  LITEMF *p;
  FILE *dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  fprintf(dfile, "List of %d items:", list->nitem);
  if (list->nitem > 0) {
    for (p = list; p != 0; p = p->next) {
      fprintf(dfile, " %d", p->item);
    }
  }
  fprintf(dfile, "\n");
}

#ifdef FLANG_COMMOPT_UNUSED
static int
common_compute_point(LITEMF *nm_list, int fg, int std)
{
  LITEMF *fg_list, *std_list;
  int i, j, n;
  int nme;
  int use;
  DU *du;
  int def;
  int fg1, fg2;
  int count;

  fg_list = clist();
  std_list = clist();

  n = nm_list->nitem;
  for (i = 0; i < n; i++) {
    nme = A_NMEG(glist(nm_list, i));
    def = NME_DEF(nme);
    if (def)
      for (du = DEF_DU(def); du != 0; du = du->next) {
        use = du->use;
        if (USE_STD(use) == std) {
          if (only_one_ud(use)) {
            plist(fg_list, DEF_FG(def));
            plist(std_list, DEF_STD(def));
          } else {
            plist(fg_list, fg);
            plist(fg_list, std);
          }
        }
      }
    else {
      plist(fg_list, 0);
      plist(std_list, 0);
    }
  }

  count = 0;
  for (i = 0; i < n; i++) {
    fg1 = glist(fg_list, i);
    count = 0;
    for (j = 0; j < n; j++) {
      fg2 = glist(fg_list, j);
      if (is_dominator(fg2, fg1))
        count++;
    }
    if (count == n)
      break;
  }
  if (glist(std_list, i) != std)
    return glist(std_list, i);
  else
    return STD_PREV(std);
}
#endif

/* This routine is to change all allocate statements into
 * allocate ast which is defined for hpf communication ast.
 * By this way, any allocate may benefit of comm_invar.
 */
static void
alloc2ast(void)
{
  int start, i;
  int std, stdnext;
  int allocast, newallocast;
  int deallocast, ast;
  int nd;
  LOGICAL found;
  int sptr;

  start = ftb.avl;
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    if (STD_IGNORE(std))
      continue;
    ast = STD_AST(std);
    if (A_TYPEG(ast) == A_ALLOC) {
      if (A_TKNG(ast) == TK_ALLOCATE) {
        allocast = STD_AST(std);
        ast = A_SRCG(allocast);
        if (A_TYPEG(ast) != A_SUBSCR)
          continue;
        if (A_TYPEG(A_LOPG(ast)) != A_ID)
          continue;
        sptr = A_SPTRG(A_LOPG(ast));
        if (!HCCSYMG(sptr))
          continue;
        if (ADJLENG(sptr))
          continue;
        newallocast = new_node(A_HALLOBNDS);
        A_LOPP(newallocast, ast);

        nd = mk_ftb();
        FT_STD(nd) = 0;
        FT_FORALL(nd) = 0;
        FT_ALLOC_SPTR(nd) = sptr;
        FT_ALLOC_FREE(nd) = 0;
        FT_ALLOC_SAME(nd) = 0;
        FT_ALLOC_REUSE(nd) = 0;
        FT_ALLOC_USED(nd) = 0;
        FT_ALLOC_OUT(nd) = sptr;
        A_OPT1P(newallocast, nd);
        STD_AST(std) = newallocast;
        A_STDP(newallocast, std);
        /*
           add_stmt_before(newallocast, std);
           delete_stmt(std);
         */
      } else {
        assert(A_TKNG(ast) == TK_DEALLOCATE, "alloc2ast: bad dealloc", std, 4);
        deallocast = STD_AST(std);
        ast = A_SRCG(deallocast);
        if (A_TYPEG(ast) != A_ID)
          continue;
        sptr = A_SPTRG(ast);
        if (!HCCSYMG(sptr) || STYPEG(sptr) != ST_ARRAY)
          continue;
        if (ADJLENG(sptr))
          continue;
        found = FALSE;
        for (i = start; i < ftb.avl; i++) {
          if (FT_ALLOC_FREE(i))
            continue;
          if (FT_ALLOC_SPTR(i) == sptr) {
            FT_ALLOC_FREE(i) = STD_PREV(std);
            FT_ALLOC_PTASGN(i) = STD_PTASGN(std);
            delete_stmt(std);
            found = TRUE;
          }
        }
        assert(found, "alloc2ast: missing allocate", std, 2);
      }
    }
  }
}

/** \brief Try to optimize allocate statements.

    If they have same bounds, give them the same bound variable.
    This will help the compiler to reduce # of communication
    for allocatable variables.
 */
void
optimize_alloc(void)
{
  comm_optimize_init();
  flowgraph();      /* build the flowgraph for the function */
  postdominators(); /* need these as well */
#if DEBUG
  if (DBGBIT(35, 1))
    dump_flowgraph();
#endif

  findloop(HLOPT_ALL); /* find the loops */

  flow(); /* do flow analysis on the loops  */

#if DEBUG
  if (DBGBIT(35, 4)) {
    dump_flowgraph();
    dump_loops();
  }
#endif
  opt_allocate();
  comm_optimize_end();
}

static void
opt_allocate(void)
{
  ADSC *ad;
  int allocast;
  int ast;
  int std;
  int sptr;
  int i;
  int ndim;
  int stdnext;
  int sub[MAXSUBS];
  LITEMF *defs_to_propagate = clist();
  LITEMF *shape_exceptions = clist(); /* don't propagate into these shapes */

  for (std = STD_NEXT(0); std; std = stdnext) {
    LOGICAL changed;
    int asd;
    stdnext = STD_NEXT(std);
    allocast = STD_AST(std);
    if (STD_PAR(std))
      continue;
    if (is_allocatable_assign(allocast)) {
      /* don't propagate shape bounds -- may be changed in transform() */
      int shape = A_SHAPEG(A_DESTG(allocast));
      plist(shape_exceptions, shape);
      continue;
    }
    if (A_TYPEG(allocast) != A_ALLOC)
      continue;
    if (A_TKNG(allocast) != TK_ALLOCATE)
      continue;
    ast = A_SRCG(allocast);
    if (A_TYPEG(ast) != A_SUBSCR)
      continue;
    /* member is busted -- lfm */
    if (A_TYPEG(A_LOPG(ast)) != A_ID)
      continue;
    sptr = A_SPTRG(A_LOPG(ast));
    /* pointer lb, ub is not A_ID,
       it may array static_descriptor */
    if (NOALLOOPTG(sptr))
      continue;
    if (POINTERG(sptr))
      continue;
    if (CMBLKG(sptr))
      continue;
    if (MDALLOCG(sptr))
      continue;
    if (SAVEG(sptr))
      continue;

    /* a SAVEd allocatable will not appear itself in a common block,
     * but its pointer offset variable will. */
    if (PTROFFG(sptr) && SCG(PTROFFG(sptr)) == SC_CMBLK)
      continue;

    /* put bounds into NME table */
    ad = AD_DPTR(DTYPEG(sptr));
    ndim = rank_of_sym(sptr);
    asd = A_ASDG(ast);

    changed = FALSE;  /* did any bounds change? */
    for (i = 0; i < ndim; i++) {
      int lw, up, lw2, up2;
      int ss = ASD_SUBS(asd, i);
      if (A_TYPEG(ss) == A_TRIPLE) {
        lw = A_LBDG(ss);
        up = A_UPBDG(ss);
      } else {
        lw = astb.i1;
        up = ss;
      }
      lw2 = propagate_bound(defs_to_propagate, lw);
      up2 = propagate_bound(defs_to_propagate, up);
      if (lw2 != lw || up2 != up) {
        sub[i] = mk_triple(lw2, up2, 0);
        changed = TRUE;
      } else {
        sub[i] = ss;
      }
    }

    /* change allocate too */
    if (changed) {
      int new = mk_subscr(mk_id(sptr), sub, ndim, DTY(DTYPEG(sptr) + 1));
      A_SRCP(allocast, new);
    }

    /* optimize allocatable alignment */
  }

  ast_visit(1, 1);
  for (i = 0; i < defs_to_propagate->nitem; i++) {
    int def = glist(defs_to_propagate, i);
    int stddef = DEF_STD(def);
    int astdef = STD_AST(stddef);
    int src, dest;
    assert(A_TYPEG(astdef) == A_ASN, "expecting ASN ast", astdef, ERR_Fatal);
    src = A_SRCG(astdef);
    dest = A_DESTG(astdef);
    ast_replace(dest, src);
  }

  /* change all shape*/
  rewrite_all_shape(shape_exceptions);
  ast_unvisit();
  freearea(FORALL_AREA);
}

/* Is this an assignment with F2003 allocatable semantics? */
static LOGICAL
is_allocatable_assign(int ast)
{
  int dest, src;
  LOGICAL dest_is_mem = FALSE;
  if (A_TYPEG(ast) != A_ASN) return FALSE;
  dest = A_DESTG(ast);
  src = A_SRCG(ast);
  while (A_TYPEG(dest) == A_MEM) {
    dest = A_MEMG(dest);
    dest_is_mem = TRUE;
  }
  if (!dest_is_mem && !XBIT(54, 0x1))
    return FALSE;
  while (A_TYPEG(src) == A_MEM) {
    src = A_MEMG(src);
  }
  if (A_TYPEG(dest) == A_ID && A_TYPEG(src) == A_ID) {
    int dest_sym = sym_of_ast(dest);
    if (ALLOCATTRG(dest_sym)) {
      return TRUE;
    }
  }
  return FALSE;
}

/* If possible, propagate the assignment to bound, add it to defs_to_propagate
 * and return the new value. */
static int
propagate_bound(LITEMF *defs_to_propagate, int bound)
{
  if (A_TYPEG(bound) == A_ID) {
    int nme = addnme(NT_VAR, A_SPTRG(bound), 0, (INT)0);
    int def = NME_DEF(nme);
    if (is_safe_copy(def)) {
      int std = DEF_STD(def);
      int ast = STD_AST(std);
      int sptr;
      assert(A_TYPEG(ast) == A_ASN, "expecting ASN ast", ast, ERR_Fatal);
      sptr = sym_of_ast(A_DESTG(ast));
      if (!GSCOPEG(sptr))
        plist(defs_to_propagate, def);
      return A_SRCG(ast);
    }
  }
  return bound;
}

/* this routine is to decide whether it is safe to
 * propagate the definition for an array bound variable to its use
 *  example, z_b_1=n*m
 * The list of definitions for the variable is passed as 'def'.
 * The requirements are:
 *  1-) All definitions are assignments, with the same RHS
 *  2-) All assignments to variables used in the RHS must
 *      be post-dominated by a dominator of these uses
 */
static LOGICAL
is_safe_copy(int deflist)
{
  int std, fg, ast, src, dest;
  int def, defstd, deffg, defast;
  int nvar, onlyvar, a[10];
  int v, defv;
  int stdv, fgv;
  int i;

  if (deflist == 0)
    return FALSE;
  std = DEF_STD(deflist);
  fg = DEF_FG(deflist);

  /* must be A_ASN */
  ast = STD_AST(std);
  if (A_TYPEG(ast) != A_ASN)
    return FALSE;
  src = A_SRCG(ast);
  dest = A_DESTG(ast);

  for (def = deflist; def; def = DEF_NEXT(def)) {
    defstd = DEF_STD(def);
    deffg = DEF_FG(def);
    defast = STD_AST(defstd);
    if (A_TYPEG(defast) != A_TYPEG(ast))
      return FALSE;
    if (A_SRCG(defast) != src)
      return FALSE;
    if (A_DESTG(defast) != dest)
      return FALSE;
  }

  /* at this point, all assignments have the same RHS */

  if (A_TYPEG(src) != A_CNST) {
    /* decompose the expression into the variables that comprise it */
    nvar = 0;
    onlyvar = 1;

    decompose_expression(src, a, 10, &nvar, &onlyvar);

    if (nvar > 10) /* too many variables in RHS? */
      return FALSE;
    if (!onlyvar) /* something complex in RHS? */
      return FALSE;
    for (i = 0; i < nvar; ++i) {
      v = a[i];
      v = addnme(NT_VAR, A_SPTRG(v), 0, (INT)0); /* find NME */
      /* go through defs of this variable;
       * all the uses here must be dominated by a postdominator
       * of the definition of the variable.
       * This allows:
       *    n = xxx
       *    if(..)then
       *      m = yyy
       *    endif
       *    z_b_1 = n * m
       *    allocate(foo(z_b_1))
       * but disallows:
       *    z_b_1 = n * m
       *    allocate(foo(z_b_1))
       *    n = xxx
       *    if(..)then
       *      m = yyy
       *    endif
       */

      for (defv = NME_DEF(v); defv; defv = DEF_NEXT(defv)) {
        stdv = DEF_STD(defv);
        fgv = DEF_FG(defv);
        for (def = deflist; def; def = DEF_NEXT(def)) {
          defstd = DEF_STD(def);
          deffg = DEF_FG(def);
          if (!must_follow(fgv, stdv, deffg, defstd))
            return FALSE;
        }
      }
    }
  }
  return TRUE;
} /* is_safe_copy */

#if DEBUG
LOGICAL
is_same_def(int def, int def1)
{
  int std, std1;
  int fg, fg1;
  int next, next1;
  int ast, ast1;
  int expr, expr1;

  if (def == 0 || def1 == 0)
    return FALSE;
  std = DEF_STD(def);
  std1 = DEF_STD(def1);
  fg = DEF_FG(def);
  fg1 = DEF_FG(def1);

  next = DEF_NEXT(def);
  next1 = DEF_NEXT(def1);

  /* there should be only one defs */
  if (next || next1)
    return FALSE;

  /* they have to be A_ASN and same A_SRC */
  ast = STD_AST(std);
  ast1 = STD_AST(std1);
  if (A_TYPEG(ast) != A_ASN || A_TYPEG(ast1) != A_ASN)
    return FALSE;
  expr = A_SRCG(ast);
  expr1 = A_SRCG(ast1);
  if (expr != expr1)
    return FALSE;

  /* value is not changed between them */
  if (!is_dominator(fg, fg1))
    return FALSE;
  if (!is_avail_expr(expr, std, fg, std1, fg1))
    return FALSE;

  return TRUE;
}
#endif

#ifdef FLANG_COMMOPT_UNUSED
/* This routine checks that def has only one definition and
 * src of that definition is a constant; if so, it returns
 * the ast of the difference, else it returns 'defaultval'
 */
static int
diff_def_cnst(int cnstAst, int def, int defaultval)
{
  int std;
  int fg;
  int next;
  int ast;
  int expr;
  int condef, conast;

  if (def == 0)
    return defaultval;
  std = DEF_STD(def);
  fg = DEF_FG(def);
  next = DEF_NEXT(def);
  /* there should be only one defs */
  if (next)
    return defaultval;

  /* they have to be A_ASN and same A_SRC */
  ast = STD_AST(std);
  if (A_TYPEG(ast) != A_ASN)
    return defaultval;
  expr = A_SRCG(ast);
  if (A_ALIASG(expr))
    expr = A_ALIASG(expr);
  if (A_TYPEG(expr) != A_CNST)
    return defaultval;
  if (A_DTYPEG(expr) != DT_INT)
    return defaultval;
  condef = A_SPTRG(expr);
  condef = CONVAL2G(condef);
  conast = A_SPTRG(cnstAst);
  conast = CONVAL2G(conast);
  condef = conast - condef;
  ast = mk_cval(condef, DT_INT);

  return ast;
}
#endif

static void
rewrite_all_shape(LITEMF *exceptions)
{
  int shape;
  int ndim;
  int ii, i;
  int old_lwb, old_upb, old_st;
  int new_lwb, new_upb, new_st;
  int old_sptr, new_sptr;

  for (ii = 1; ii < MAXDIMS; ii++) {
    ndim = ii;
    /* search the existing SHDs with the same number of dimensions
     */
    for (shape = astb.shd.hash[ndim - 1]; shape; shape = SHD_NEXT(shape)) {
      if (inlist(exceptions, shape))
        continue;
      for (i = 0; i < ndim; i++) {
        old_lwb = SHD_LWB(shape, i);
        old_upb = SHD_UPB(shape, i);
        old_st = SHD_STRIDE(shape, i);
        new_lwb = ast_rewrite(SHD_LWB(shape, i));
        new_upb = ast_rewrite(SHD_UPB(shape, i));
        new_st = ast_rewrite(SHD_STRIDE(shape, i));
        SHD_LWB(shape, i) = new_lwb;
        SHD_UPB(shape, i) = new_upb;
        SHD_STRIDE(shape, i) = new_st;
        if (flg.smp) {
          if (A_TYPEG(old_lwb) == A_ID) {
            old_sptr = sym_of_ast(old_lwb);
            new_sptr = 0;
            if (ast_is_sym(new_lwb)) {
              new_sptr = sym_of_ast(new_lwb);
            }
            if (new_sptr && new_sptr != old_sptr &&
                PARREFG(old_sptr) && STYPEG(new_sptr) != ST_CONST) {
              set_parref_flag2(new_sptr, old_sptr, 0);
            }
          }
          if (A_TYPEG(old_upb) == A_ID) {
            old_sptr = sym_of_ast(old_upb);
            new_sptr = 0;
            if (ast_is_sym(new_upb)) {
              new_sptr = sym_of_ast(new_upb);
            }
            if (new_sptr &&  new_sptr != old_sptr &&
                PARREFG(old_sptr) && STYPEG(new_sptr) != ST_CONST) {
              set_parref_flag2(new_sptr, old_sptr, 0);
            }
          }
          if (A_TYPEG(old_st) == A_ID) {
            old_sptr = sym_of_ast(old_st);
            new_sptr = 0;
            if (ast_is_sym(new_st)) {
              new_sptr = sym_of_ast(new_st);
            }
            if (new_sptr && new_sptr != old_sptr &&
                PARREFG(old_sptr) && STYPEG(new_sptr) != ST_CONST) {
              set_parref_flag2(new_sptr, old_sptr, 0);
            }
          }
        }
      }
    }
  }
}

static void
decompose_expression(int expr, int a[], int size, int *nvar, int *onlyvar)
{
  int i;
  int asd;
  int ndim, n;
  int arr;
  int argt;

  if (expr == 0)
    return;
  switch (A_TYPEG(expr)) {
  case A_ID:
    if (*nvar >= size) {
      *nvar = size + 1;
      return;
    }
    a[*nvar] = expr;
    (*nvar)++;
    return;
  case A_BINOP:
    decompose_expression(A_LOPG(expr), a, size, nvar, onlyvar);
    decompose_expression(A_ROPG(expr), a, size, nvar, onlyvar);
    return;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    decompose_expression(A_LOPG(expr), a, size, nvar, onlyvar);
    return;
  case A_LABEL:
  case A_CMPLXC:
  case A_CNST:
    return;
  case A_MEM:
    decompose_expression((int)A_PARENTG(expr), a, size, nvar, onlyvar);
    return;
  case A_SUBSTR:
    if (onlyvar) {
      *onlyvar = 0;
      return;
    }
    decompose_expression((int)A_LOPG(expr), a, size, nvar, onlyvar);
    decompose_expression((int)A_LEFTG(expr), a, size, nvar, onlyvar);
    decompose_expression((int)A_RIGHTG(expr), a, size, nvar, onlyvar);
    return;
  case A_ICALL:
  case A_INTR:
  case A_FUNC:
    if (onlyvar) {
      *onlyvar = 0;
      return;
    }
    argt = A_ARGSG(expr);
    n = A_ARGCNTG(expr);
    for (i = 0; i < n; ++i)
      decompose_expression(ARGT_ARG(argt, i), a, size, nvar, onlyvar);
    return;
  case A_TRIPLE:
    if (onlyvar) {
      *onlyvar = 0;
      return;
    }
    decompose_expression(A_LBDG(expr), a, size, nvar, onlyvar);
    decompose_expression(A_UPBDG(expr), a, size, nvar, onlyvar);
    decompose_expression(A_STRIDEG(expr), a, size, nvar, onlyvar);
    return;
  case A_SUBSCR:
    if (onlyvar) {
      *onlyvar = 0;
      return;
    }
    arr = A_LOPG(expr);
    decompose_expression(A_LOPG(expr), a, size, nvar, onlyvar);
    asd = A_ASDG(expr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++)
      decompose_expression(ASD_SUBS(asd, i), a, size, nvar, onlyvar);
    return;
  default:
    interr("decompose_expression: unknown type", expr, 3);
    return;
  }
}

/** \brief Put forall calls into forall tables.

    FT_MCALL will have calls used in the mask.<br>
    FT_SCALL will have calls used in the statement.<br>
    These later will be used to parallel calls.
 */
void
put_forall_pcalls(int fstd)
{
  int forall;
  int topstd;
  int i, j;
  int mask;
  int nd, nd1, nd2;
  int nargs, argt, arg;
  int pstd, past, psptr;
  int pstd1, past1;
  int asn;

  forall = STD_AST(fstd);
  assert(A_TYPEG(forall) == A_FORALL, "put_forall_pcalls: must be forall",
         forall, 4);

  mask = A_IFEXPRG(forall);
  asn = A_IFSTMTG(forall);
  nd = A_OPT1G(forall);
  topstd = A_SRCG(forall);
  if (!topstd)
    return;
  if (A_ARRASNG(forall))
    return;
  /* put statements calls */
  for (i = topstd; i != fstd; i = STD_NEXT(i)) {
    if (is_pcalls(i, fstd)) {
      plist(FT_PCALL(nd), i);
      STD_PURE(i) = TRUE;
      FT_NPCALL(nd)++;
    }
  }

  for (i = 0; i < FT_NPCALL(nd); i++) {
    pstd = glist(FT_PCALL(nd), i);
    past = STD_AST(pstd);
    nargs = A_ARGCNTG(past);
    argt = A_ARGSG(past);
    arg = ARGT_ARG(argt, 0);
    switch (A_TYPEG(arg)) { /* FS#18714 - skip over non-symbol arg */
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      break;
    default:
      continue;
    }
    psptr = sym_of_ast(arg);
    if (mask && expr_dependent(mask, arg, fstd, fstd)) {
      plist(FT_MCALL(nd), pstd);
      FT_NMCALL(nd)++;
    } else if (expr_dependent(A_SRCG(asn), arg, fstd, fstd) ||
               expr_dependent(A_DESTG(asn), arg, fstd, fstd)) {
      plist(FT_SCALL(nd), pstd);
      FT_NSCALL(nd)++;
    }
  }

  for (i = 0; i < FT_NPCALL(nd); i++) {
    pstd = glist(FT_PCALL(nd), i);
    past = STD_AST(pstd);
    nargs = A_ARGCNTG(past);
    argt = A_ARGSG(past);
    arg = ARGT_ARG(argt, 0);
    switch (A_TYPEG(arg)) { /* FS#18714 - skip over non-symbol arg */
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      break;
    default:
      continue;
    }
    psptr = sym_of_ast(arg);
    nd1 = mk_ftb();
    FT_CALL_SPTR(nd1) = psptr;
    /* don't distribute PURE result */
    FT_CALL_NCALL(nd1) = 0;
    FT_CALL_CALL(nd1) = clist();
    FT_CALL_POS(nd1) = 0;
    A_OPT1P(past, nd1);
  }

  for (i = 0; i < FT_NPCALL(nd); i++) {
    pstd = glist(FT_PCALL(nd), i);
    past = STD_AST(pstd);
    nd1 = A_OPT1G(past);
    if (nd1 == 0)
      continue;
    psptr = FT_CALL_SPTR(nd1);
    for (j = 0; j < FT_NPCALL(nd); j++) {
      if (i == j)
        continue;
      pstd1 = glist(FT_PCALL(nd), j);
      past1 = STD_AST(pstd1);
      nd2 = A_OPT1G(past1);
      if (contains_ast(past1, mk_id(psptr))) {
        plist(FT_CALL_CALL(nd2), pstd);
        FT_CALL_NCALL(nd2)++;
      }
    }
  }
}

/* This checks:
 * Whether std is call which may be pure
 */
static LOGICAL
is_pcalls(int std, int fstd)
{
  int ast;
  int sptr;
  ast = STD_AST(std);
  if (A_TYPEG(ast) == A_CALL) {
    sptr = A_SPTRG(A_LOPG(ast));
    if (is_impure(sptr))
      error(488, 4, STD_LINENO(fstd), "subprogram call in FORALL",
            SYMNAME(sptr));
    else
      return TRUE;
  }
  if (A_TYPEG(ast) == A_ICALL)
    return TRUE;
  return FALSE;
}

static void
forall_make_same_idx(int std)
{

  int idx[7];
  int list, listp;
  int forall;
  int nidx;
  int isptr, isptr1;
  int dtype;
  int i;
  int nd, nd1;
  int oldast, newast;
  int newforall;
  int af, st, nc, bd;
  int cnt;
  int ip, pstd, past;
  LITEMF *plist;

  forall = STD_AST(std);
  assert(A_TYPEG(forall) == A_FORALL, "make_same_idx: not forall", 2, forall);
  list = A_LISTG(forall);
  nidx = 0;
  for (listp = list; listp != 0; listp = ASTLI_NEXT(listp)) {
    idx[nidx] = listp;
    nidx++;
  }
  assert(nidx <= 7, "make_same_idx: illegal forall", 2, forall);

  /* if it is already changed, don't do any thing */
  cnt = 0;
  for (i = 0; i < nidx; i++) {
    isptr1 = ASTLI_SPTR(idx[i]);
    dtype = DTYPEG(isptr1);
    isptr = get_init_idx(i, dtype);
    if (isptr == isptr1)
      cnt++;
    if (STD_TASK(std) && SCG(isptr) == SC_PRIVATE) {
      TASKP(isptr, 1);  
    }
  }
  if (cnt == nidx)
    return;
  ast_visit(1, 1);

  /* change forall */
  for (i = 0; i < nidx; i++) {
    isptr = ASTLI_SPTR(idx[i]);
    dtype = DTYPEG(isptr);
    oldast = mk_id(isptr);
    isptr = get_init_idx(i, dtype);
    newast = mk_id(isptr);
    if (STD_TASK(std) && SCG(isptr) == SC_PRIVATE) {
      TASKP(isptr, 1);  
    }
    ast_replace(oldast, newast);
  }

  nd = A_OPT1G(forall);
  af = A_ARRASNG(forall);
  st = A_STARTG(forall);
  nc = A_NCOUNTG(forall);
  bd = A_CONSTBNDG(forall);
  newforall = ast_rewrite(forall);
  A_OPT1P(newforall, nd);
  A_ARRASNP(newforall, af);
  A_STARTP(newforall, st);
  A_NCOUNTP(newforall, nc);
  A_CONSTBNDP(newforall, bd);

  A_STDP(newforall, std);
  STD_AST(std) = newforall;

  /* change also pcalls */
  plist = FT_PCALL(nd);
  for (ip = 0; ip < FT_NPCALL(nd); ip++) {
    pstd = plist->item;
    plist = plist->next;
    past = STD_AST(pstd);
    nd1 = A_OPT1G(past);
    past = ast_rewrite(past);
    A_OPT1P(past, nd1);
    STD_AST(pstd) = past;
    A_STDP(past, pstd);
  }
  ast_unvisit();
}

/* SHARED MEMORY OPTIMIZATION START HERE */

/** \brief Put all get_scalar into table to easily process */
void
find_get_scalar(void)
{
  int std, stdnext;
  int ast;
  int i;
  int type;
  int nd;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    type = A_TYPEG(ast);
    if (type != A_HGETSCLR)
      continue;
    nd = A_OPT1G(ast);
    if (nd == 0) {
      nd = mk_ftb();
      FT_STD(nd) = 0;
      FT_GETSCLR_SAME(nd) = 0;
      FT_GETSCLR_REUSE(nd) = 0;
      FT_GETSCLR_NOMEM(nd) = 0;
      FT_GETSCLR_OMEMED(nd) = 0;
    }
    A_OPT1P(ast, nd);
    i = get_gstbl();
    gstbl.base[i].f1 = std;
  }
}

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Invariant communication optimization.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "machar.h"
#include "gramtk.h"
#include "optimize.h"
#include "induc.h"
#include "extern.h"
#include "commopt.h"
#include "ccffinfo.h"
#include "fdirect.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

/* Compiler switches:
    -x 49 0x20000:	Inhibit invariant communication hoisting
    -q 43 2:		Trace invariant communication hoisting
    -q 43 64:		Dump statements before and after hoisting
 */

/* local macros */
#define A_CONDG(a) A_OPT2G(a)
#define A_CONDP(a, v) A_OPT2P(a, v)

#if DEBUG
#define TRACE0(s)    \
  if (DBGBIT(43, 2)) \
  fprintf(gbl.dbgfil, s)
#define TRACE1(s, a1) \
  if (DBGBIT(43, 2))  \
  fprintf(gbl.dbgfil, s, a1)
#define TRACE2(s, a1, a2) \
  if (DBGBIT(43, 2))      \
  fprintf(gbl.dbgfil, s, a1, a2)
#define TRACE3(s, a1, a2, a3) \
  if (DBGBIT(43, 2))          \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#define TRACE4(s, a1, a2, a3, a4) \
  if (DBGBIT(43, 2))              \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4)
#define TRACE5(s, a1, a2, a3, a4, a5) \
  if (DBGBIT(43, 2))                  \
  fprintf(gbl.dbgfil, s, a1, a2, a3, a4, a5)
#else
#define TRACE0(s)
#define TRACE1(s, a1)
#define TRACE2(s, a1, a2)
#define TRACE3(s, a1, a2, a3)
#define TRACE4(s, a1, a2, a3, a4)
#define TRACE5(s, a1, a2, a3, a4, a5)
#endif

/* local functions */
static LOGICAL hoist_loop(int lp);
static void clean_flags(void);
static void get_loopcounts(void);
static LOGICAL contains_alloc(int lp);
static LOGICAL unique_allo(int allo_std, int allo_ft, int lp);
static void hoist_comms(int lp);
static void hoist_std(int, int, int);
static LOGICAL check_outer_defs(int std, int lp, int nme);
static LOGICAL check_inner_uses(int std, int lp, int nme);
static LOGICAL is_invariant_base(int nme);
static LOGICAL is_parent_loop(int lpParent, int lpChild);
static LOGICAL has_invar_subs(int ast);
static LOGICAL is_invariant_list(int n, int vast[]);
static LOGICAL check_moved_list(int n, int vstd[]);
static LOGICAL check_moved(int std);
static void move_invar_std(int std, int fg);
static void mk_loop_hdtl(int lp);
static int mk_count_test(int astCond, int astCount);
static int mk_dealloc_stub(int astCond);
static void mk_ztrips(int lp);
static int remove_cond(int astTarg, int astSrc);

/* global data */
static int fgPre;              /* pre-header of current loop */
static int fgPost;             /* trailer of current loop */
static LOGICAL bMovedComm;     /* TRUE when invariant comm statement moved */
static LOGICAL bMovedAssn;     /* TRUE when invariant assignment moved */
static int *pastCounts = NULL; /* loop counts */

/** \brief Hoist invariant communication expressions out of loops. */
void
comm_invar(void)
{
  int lp;

  hlopt_init(HLOPT_INDUC);

#if DEBUG
  if (DBGBIT(43, 64)) {
    fprintf(gbl.dbgfil, "----------- STDs before invariant hoisting\n");
    dump_std();
  }
#endif

  get_loopcounts();

  clean_flags();

  for (lp = LP_CHILD(0); lp; lp = LP_SIBLING(lp))
    hoist_loop(lp);

  mk_ztrips(0);
#if DEBUG
  if (DBGBIT(43, 64)) {
    fprintf(gbl.dbgfil, "----------- STDs after invariant hoisting\n");
    dump_std();
  }
#endif

  if (pastCounts) {
    FREE(pastCounts);
    pastCounts = NULL;
  }
  hlopt_end(HLOPT_INDUC, 0);
}

/* Clear flags & fields in STDs & ASTs that will be used for hoisting. */
static void
clean_flags(void)
{
  int std;

  for (std = STD_FIRST; std; std = STD_NEXT(std)) {
    STD_MOVED(std) = FALSE;
    A_CONDP(STD_AST(std), 0);
  }
}

/* For each loop lp, set pastCounts[lp] to its count. */
static void
get_loopcounts(void)
{
  int lp;

  NEW(pastCounts, int, opt.nloops + 1);

  for (lp = 1; lp <= opt.nloops; lp++) {
    /* Mark invariant ASTs within lp. */
    invariant_nomotion(lp);
    pastCounts[lp] = get_loop_count(lp);
    end_loop_count();
    invariant_unmarkv();
  }
}

/* Hoist invariant communication expressions out of loop #lp. Return TRUE
 * if lp is a candidate for hoisting. */
static LOGICAL
hoist_loop(int lp)
{
  int lpi;
  LOGICAL bHoistInner = TRUE;
  LOGICAL bx49;

  for (lpi = LP_CHILD(lp); lpi; lpi = LP_SIBLING(lpi))
    bHoistInner &= hoist_loop(lpi);

  if (!bHoistInner)
    /* Don't analyze if an inner loop can't be analyzed. */
    return FALSE;
  if (LP_MEXITS(lp))
    /* Don't analyze if lp contains multiple exits. */
    return TRUE;
  if (!pastCounts[lp])
    /* Loop not countable. */
    return TRUE;
  if (contains_alloc(lp))
    /* Don't analyze if user-defined arrays are allocated. */
    return FALSE;

  /* Check for a disabling directive. */
  open_pragma(FG_LINENO(LP_HEAD(lp)));
  bx49 = (XBIT(49, 0x20000) != 0);
  close_pragma();
  if (bx49)
    return TRUE;

  /* Mark invariant ASTs within lp. */
  invariant_nomotion(lp);

  fgPre = fgPost = 0;

  /* Hoist communication ASTs. */
  hoist_comms(lp);

  invariant_unmarkv();
  return TRUE;
}

/* Return TRUE if loop #lp contains an ALLOCATE/DEALLOCATE statement of
 * a user-defined array. */
static LOGICAL
contains_alloc(int lp)
{
  int fg;
  int std, stdend;
  int ast, sptr;

  for (fg = LP_FG(lp); fg; fg = FG_NEXT(fg)) {
    std = FG_STDFIRST(fg);
    if (!std)
      continue;
    stdend = STD_NEXT(FG_STDLAST(fg));
    for (; std != stdend; std = STD_NEXT(std)) {
      ast = STD_AST(std);
      if (A_TYPEG(ast) != A_ALLOC)
        continue;
      ast = A_SRCG(ast);
      if (A_TYPEG(ast) == A_SUBSCR)
        ast = A_LOPG(ast);
      switch (A_TYPEG(ast)) {
      case A_ID:
        sptr = A_SPTRG(ast);
        break;
      case A_MEM:
        sptr = A_SPTRG(A_MEMG(ast));
        break;
      default:
        interr("contains_alloc: ID operand not found", lp, 4);
      }
      if (!CCSYMG(sptr))
        return TRUE;
    }
  }
  return FALSE;
}

/* Hoist invariant communication ASTs within loop #lp into a
 * pre-header (and trailer) located at fgPre (and fgPost). */
static void
hoist_comms(int lp)
{
  int fg, fgtail, fgend;
  int std, stdend, stdnext;
  int par;

  bMovedComm = FALSE;
  bMovedAssn = FALSE;
  fgtail = LP_TAIL(lp);
  fgend = FG_LNEXT(fgtail);
  fg = LP_HEAD(lp);
  par = STD_PAR(FG_STDFIRST(fg));
  for (; fg != fgend; fg = FG_LNEXT(fg)) {
    if (FG_LOOP(fg) != lp)
      continue;
    if (!is_dominator(fg, fgtail))
      continue;
    std = FG_STDFIRST(fg);
    if (!std)
      continue;
    stdend = STD_NEXT(FG_STDLAST(fg));
    stdnext = STD_NEXT(std);
    for (; std != stdend; std = stdnext, stdnext = STD_NEXT(stdnext)) {
      if (par != STD_PAR(std))
        continue;
      if (!STD_DELETE(std))
        hoist_std(fg, std, lp);
    }
  }

  if (!bMovedComm && !bMovedAssn)
    return;

  /* Clear the flags. */
  std = FG_STDFIRST(fgPre);
  if (std) {
    stdend = STD_NEXT(FG_STDLAST(fgPre));
    for (; std != stdend; std = STD_NEXT(std))
      STD_MOVED(std) = FALSE;
  }

  if (bMovedComm && flg.hpf)
    ccff_info(MSGFTN, "FTN012", 1, FG_LINENO(LP_HEAD(lp)),
              "Invariant communication calls hoisted out of loop", NULL);
  if (bMovedAssn)
    ccff_info(MSGFTN, "FTN013", 1, FG_LINENO(LP_HEAD(lp)),
              "Invariant assignments hoisted out of loop", NULL);
}

/* Return TRUE if the A_HALLOBNDS at allo_std is unique in lp.  That is,
   no other A_HALLOBNDS in the loop allocates/deallocates the same symbol. */
static LOGICAL
unique_allo(int allo_std, int allo_ft, int lp)
{
  int fg;
  int std, stdend;
  int ast, ft;

  for (fg = LP_FG(lp); fg; fg = FG_NEXT(fg)) {
    std = FG_STDFIRST(fg);
    if (!std)
      continue;
    stdend = STD_NEXT(FG_STDLAST(fg));
    for (; std != stdend; std = STD_NEXT(std)) {
      if (std == allo_std)
        continue;
      ast = STD_AST(std);
      if (A_TYPEG(ast) != A_HALLOBNDS)
        continue;
      ft = A_OPT1G(ast);
      if (!ft)
        continue;
      if (FT_ALLOC_SPTR(ft) == FT_ALLOC_SPTR(allo_ft))
        return FALSE;
    }
  }
  return TRUE;
}

/* Hoist an invariant communication AST at STD std to fgPre. STD std occurs
 * within node #fg. Set bMovedComm or bMovedAssn to TRUE when hoisting
 * occurs. */
static void
hoist_std(int fg, int std, int lp)
{
  int ast, ast1, astcomm;
  int ft;
  int stdfree;
  int nme;
  int ndim, dim;
  int asd;

  if (!std)
    return;

  ast = STD_AST(std);

  if (A_CONDG(ast) && !is_invariant(A_CONDG(ast)))
    return;

  if (A_TYPEG(ast) == A_ASN && A_TYPEG(A_DESTG(ast)) == A_ID
      && STYPEG(A_SPTRG(A_DESTG(ast))) == ST_VAR) {
    /* Hoist unique scalar assignments with invariant right-hand sides,
     * which may be allocatable array bounds. */
    nme = A_NMEG(A_DESTG(ast));
    if (is_sym_invariant_safe(nme, lp) &&
        check_outer_defs(std, FG_LOOP(fg), nme) &&
        check_inner_uses(std, FG_LOOP(fg), nme) && is_invariant(A_SRCG(ast)) &&
        is_invariant_base(nme)) {
      invariant_mark(A_DESTG(ast), INV);
      NME_STL(nme) = 0;
      move_invar_std(std, fg);
      bMovedAssn = TRUE;
      return;
    }
  }

  if (A_TYPEG(ast) == A_ASN)
    astcomm = A_SRCG(ast);
  else
    astcomm = ast;
  ft = A_OPT1G(astcomm);

  switch (A_TYPEG(astcomm)) {
  case A_HALLOBNDS:
    if (!ft || !unique_allo(std, ft, lp) || !has_invar_subs(A_LOPG(astcomm)))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_ALLOC_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    return;
  case A_HSECT:
    if (!ft || !has_invar_subs(A_LOPG(astcomm)) ||
        !check_moved(FT_SECT_ALLOC(ft)))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_SECT_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    return;
  case A_HLOCALIZEBNDS:
  case A_HCYCLICLP:
    if (!ft || !is_invariant(A_ITRIPLEG(astcomm)))
      return;
    move_invar_std(std, fg);
    bMovedComm = TRUE;
    return;
  case A_HCOPYSECT:
    if (!ft || !has_invar_subs(A_SRCG(astcomm)) ||
        !check_moved(FT_CCOPY_ALLOC(ft)) || !check_moved(FT_CCOPY_SECTL(ft)) ||
        !check_moved(FT_CCOPY_SECTR(ft)))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_CCOPY_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    return;
  case A_HCSTART:
    assert(A_TYPEG(A_SRCG(astcomm)) == A_SUBSCR,
           "hoist_std: source operand not subscript", ast, 4);
    assert(A_TYPEG(A_DESTG(astcomm)) == A_SUBSCR,
           "hoist_std: dest operand not subscript", ast, 4);
    if (!ft || !is_invariant(A_SRCG(astcomm)) ||
        !is_invariant(A_DESTG(astcomm)) || !check_moved(FT_CSTART_COMM(ft)) ||
        !check_moved(FT_CSTART_SECTL(ft)) || !check_moved(FT_CSTART_SECTR(ft)))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_CSTART_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    FT_CSTART_INVMVD(ft) = 1;
    return;
  case A_HOVLPSHIFT:
    if (!ft || !has_invar_subs(A_SRCG(astcomm)) ||
        (FT_SHIFT_BOUNDARY(ft) && !is_invariant(FT_SHIFT_BOUNDARY(ft))))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_SHIFT_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    return;
  case A_HSCATTER:
  case A_HGATHER:
    if (!ft)
      return;
    ndim = ASD_NDIM(A_ASDG(FT_CGATHER_VSUB(ft)));
    if (!has_invar_subs(FT_CGATHER_VSUB(ft)) ||
        !has_invar_subs(FT_CGATHER_NVSUB(ft)) ||
        (FT_CGATHER_MASK(ft) && !is_invariant(FT_CGATHER_MASK(ft))) ||
        !is_invariant_list(ndim, &FT_CGATHER_V(ft, 0)) ||
        !check_moved(FT_CGATHER_SECTVSUB(ft)) ||
        !check_moved(FT_CGATHER_SECTNVSUB(ft)) ||
        !check_moved(FT_CGATHER_SECTM(ft)) ||
        !check_moved_list(ndim, &FT_CGATHER_SECTV(ft, 0)))
      return;
    move_invar_std(std, fg);
    stdfree = mk_dealloc_stub(A_CONDG(ast));
    FT_CGATHER_FREE(ft) = stdfree;
    bMovedComm = TRUE;
    return;
  case A_HGETSCLR:
    if (!is_invariant(A_SRCG(astcomm)))
      return;
    move_invar_std(std, fg);
    bMovedComm = TRUE;
    return;
  case A_HOWNERPROC:
    ast1 = A_LOPG(astcomm);
    assert(A_TYPEG(ast1) == A_SUBSCR, "hoist_std: missing array ref", astcomm,
           4);
    asd = A_ASDG(ast1);
    dim = CONVAL2G(A_SPTRG(A_DIMG(astcomm)));
    if (!is_invariant(ASD_SUBS(asd, dim)))
      return;
    move_invar_std(std, fg);
    bMovedComm = TRUE;
    return;
  default:
    return;
  }
}

/* Return TRUE if every definition of nme that is not at std occurs in
 * a parent of loop lp. */
static LOGICAL
check_outer_defs(int std, int lp, int nme)
{
  int def;

  for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
    if (DEF_STD(def) == std)
      continue;
    if (is_parent_loop(lp, FG_LOOP(DEF_FG(def))))
      return FALSE;
  }
  return TRUE;
}

/* Return TRUE if every use of nme within loop lp is not reached by
 * a def of nme other than the def at STD std. */
static LOGICAL
check_inner_uses(int std, int lp, int nme)
{
  int def;
  DU *du;
  int use;

  for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
    if (DEF_STD(def) == std)
      continue;
    for (du = DEF_DU(def); du; du = du->next) {
      use = du->use;
      if (is_parent_loop(lp, FG_LOOP(USE_FG(use))))
        return FALSE;
    }
  }
  return TRUE;
}

/*
 * return FALSE if this NME is for a BASED symbol, and the
 * base pointer (from MIDNUM) is not already marked invariant
 */
static LOGICAL
is_invariant_base(int nme)
{
  int sym;
  sym = NME_SYM(nme);
  if (SCG(sym) == SC_BASED && MIDNUMG(sym) > NOSYM &&
      !is_invariant(mk_id(MIDNUMG(sym))))
    return FALSE;
  return TRUE;
} /* is_invariant_base */

/* Return TRUE if lpParent is a parent loop of lpChild, or if they are
 * identical. */
static LOGICAL
is_parent_loop(int lpParent, int lpChild)
{
  int lpo;

  for (lpo = lpChild; lpo; lpo = LP_PARENT(lpo))
    if (lpo == lpParent)
      return TRUE;
  return FALSE;
}

/* Return TRUE if the subscript expressions of AST ast are all invariant. */
static LOGICAL
has_invar_subs(int ast)
{
  int asd;
  int ndims, sub;

  if (!ast)
    return TRUE;

  if (A_TYPEG(ast) == A_ID) {
    assert(A_SHAPEG(ast), "has_invar_subs: ast has no shape", ast, 4);
    return TRUE;
  }
  while (A_TYPEG(ast) != A_ID) {
    switch (A_TYPEG(ast)) {
    case A_SUBSCR:
      asd = A_ASDG(ast);
      ndims = ASD_NDIM(asd);
      for (sub = 0; sub < ndims; sub++) {
        if (!is_invariant(ASD_SUBS(asd, sub)))
          return FALSE;
      }
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      ast = A_PARENTG(ast);
      break;
    default:
      interr("reference not ID,SUBSCR,MEM", ast, 4);
    }
  }
  return TRUE;
}

/* Return TRUE if every AST in vast[0], ..., vast[n-1] is invariant. */
static LOGICAL
is_invariant_list(int n, int vast[])
{
  int i;

  for (i = 0; i < n; i++)
    if (vast[i] && !is_invariant(vast[i]))
      return FALSE;
  return TRUE;
}

/* Return TRUE if every STD in vstd[0], ..., vstd[n-1] has been moved. */
static LOGICAL
check_moved_list(int n, int vstd[])
{
  int i;

  for (i = 0; i < n; i++)
    if (!check_moved(vstd[i]))
      return FALSE;
  return TRUE;
}

/* Return TRUE if std has been moved. */
static LOGICAL
check_moved(int std)
{
  if (!std)
    return TRUE;
  if (STD_DELETE(std))
    return FALSE;
  return STD_MOVED(std) != 0;
}

/* Move the invariant AST at STD #std within node #fg to the last statement
 * in the preheader of fg's loop. Set the std's MOVED flag. */
static void
move_invar_std(int std, int fg)
{
  int stdprev;
  int astCont, astStmt, astCond;

  mk_loop_hdtl(FG_LOOP(fg));

  rdilts(fg);
  stdprev = STD_PREV(std);
  if (STD_LABEL(std)) {
    astCont = mk_stmt(A_CONTINUE, 0);
    stdprev = add_stmt_after(astCont, stdprev);
    STD_LABEL(stdprev) = STD_LABEL(std);
    STD_LABEL(std) = 0;
    STD_LINENO(stdprev) = STD_LINENO(std);
  }
  STD_NEXT(stdprev) = STD_NEXT(std);
  STD_PREV(STD_NEXT(std)) = stdprev;
  wrilts(fg);

  rdilts(fgPre);
  stdprev = STD_LAST;
  STD_NEXT(stdprev) = std;
  STD_PREV(std) = stdprev;
  STD_NEXT(std) = 0;
  STD_LAST = std;
  wrilts(fgPre);

  STD_MOVED(std) = TRUE;
  astStmt = STD_AST(std);
  astCond = mk_count_test(A_CONDG(astStmt), pastCounts[FG_LOOP(fg)]);
  A_CONDP(astStmt, astCond);
  /*
   * This is where fgPost needs be associated with astCond & saved
   * in a data structure for mk_ztrips(int lp).  Probably need to
   * repurpose A_COND.
fprintf(STDERR,
"XXX lp:%d, fgPre:%d, fgPost:%d, fgPost_stdfirst:%d\n",
FG_LOOP(fg), fgPre, fgPost, FG_STDFIRST(fgPost));
   */
}

/* If necessary create new preheader and trailer nodes for loop #lp,
 * and set fgPre & fgPost to these nodes. */
static void
mk_loop_hdtl(int lp)
{
  int ast;
  int std;

  if (fgPre)
    return;

  fgPre = add_fg(FG_LPREV(LP_HEAD(lp)));
  opt.pre_fg = fgPre;
  FG_FT(fgPre) = TRUE;
  std = FG_STDFIRST(fgPre);
  if (!std) {
    /* Node empty; add a CONTINUE statement. */
    rdilts(fgPre);
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, 0);
    wrilts(fgPre);
  }
  FG_LINENO(fgPre) = STD_LINENO(std) = FG_LINENO(LP_HEAD(lp));
  add_loop_preheader(lp);

  add_single_loop_exit(lp);
  fgPost = FG_LNEXT(LP_TAIL(lp));
}

/** \brief If necessary create new preheader and trailer nodes for loop \#lp,
    and set fgPre & fgPost to these nodes. */
void
add_loop_hd(int lp)
{
  int ast;
  int std;

  fgPre = add_fg(FG_LPREV(LP_HEAD(lp)));
  opt.pre_fg = fgPre;
  FG_FT(fgPre) = TRUE;
  std = FG_STDFIRST(fgPre);
  if (!std) {
    /* Node empty; add a CONTINUE statement. */
    rdilts(fgPre);
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, 0);
    wrilts(fgPre);
  }

  add_loop_preheader(lp);
}

/* Augment the current conjunction of count tests, astCond, with a new test
 * that the loop count, astCount, is greater than 0. */
static int
mk_count_test(int astCond, int astCount)
{
  int ast, astNewCond;

  assert(astCount, "mk_count_test: loop not countable", 0, 4);
  if (A_TYPEG(astCount) == A_CNST)
    return astCond;
  ast = mk_binop(OP_GT, astCount, astb.i0, DT_LOG);
  if (astCond)
    astNewCond = mk_binop(OP_LAND, astCond, ast, DT_LOG);
  else
    astNewCond = ast;
  return astNewCond;
}

/* Create a CONTINUE statement, with COND field astCond,
 * in node fgPost, and return the STD. */
static int
mk_dealloc_stub(int astCond)
{
  int ast = mk_stmt(A_CONTINUE, 0);
  int std;

  A_CONDP(ast, astCond);
  rdilts(fgPost);
  std = add_stmt_after(ast, 0);
  wrilts(fgPost);
  STD_LINENO(std) = FG_LINENO(fgPost);
  return std;
}

/*
 * Create zero-trip tests around hoisted communication statements within
 * loop lp.
 */
static void
mk_ztrips(int lp)
{
  int lpi;
  int fg;
  int std, stdend;
  int ast, astCond, astNewCond;
  int *pastConds = NULL;
  int maxconds = opt.nloops + 1;
  int nconds;

  for (lpi = LP_CHILD(lp); lpi; lpi = LP_SIBLING(lpi))
    mk_ztrips(lpi);

  NEW(pastConds, int, maxconds);
  for (fg = LP_FG(lp); fg; fg = FG_NEXT(fg)) {
    nconds = 0;
    std = FG_STDFIRST(fg);
    if (!std)
      continue;
    stdend = STD_NEXT(FG_STDLAST(fg));
    for (; std != stdend; std = STD_NEXT(std)) {
      astCond = A_CONDG(STD_AST(std));
      astNewCond = astCond;
      for (; nconds; nconds--) {
        if (astCond)
          astNewCond = remove_cond(astCond, pastConds[nconds - 1]);
        else
          astNewCond = astCond;
        if (astNewCond != astCond)
          /* ...new condition contained in old. */
          break;
        /* ...new condition not in old, so terminate old condition. */
        ast = mk_stmt(A_ENDIF, 0);
        add_stmt_before(ast, std);
      }
      if (!astCond || !astNewCond)
        continue;
      ast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ast, astNewCond);
      add_stmt_before(ast, std);
      pastConds[nconds++] = astCond;
      assert(nconds <= opt.nloops, "mk_ztrips: too many conditions", lp, 4);
    }
    /*
     * f20730 - hoisting of invariant assignments is suboptimal for
     * reaching defs.  Hoisting an invariant assignment from a loop
     * occurs as follows:
     *
     * the loop
     * ---------
     *   do __the_loop__
     *     XX = __invariant_expression__
     *     __work__ << uses XX>
     *   enddo
     *
     * the hoist
     * ---------
     *   if (zero-trip-test) then
     *     XX = __invariant_expression__
     *   endif
     *   do __the_loop__
     *     __work__ <<< uses XX>>>
     *   enddo
     *
     * this needs to be
     * ----------------
     *   if (zero-trip-test) then
     *     XX = __invariant_expression__
     *     do __the_loop__  !!! and somehow elide the backend's ZT
     *       __work__ << uses XX>
     *     enddo
     *   endif
     *
     * THE PROBLEM
     * -----------
     * IF there exists a def of XX prior to the original loop, that def
     * now reaches the XX within the transformed loop (the hoisted
     * assignment is now guarded).  Without the hoisting, XX was killed
     * by the def of XX in the loop.  This can lead to false uses of
     * XX -- for WRF, this was especially bad given that the XX was
     * considered a live-out of a loop, which inhibited vectorizing
     * that loop.
     *
     * To correctly place the ENDIF, move_invar_std() needs to associate
     * A_COND with fgPost (FG_STDFIRST(fgPost) would be the correct
     * insert point for the ENDIF.
     *
     * AT THIS TIME, it is not be worth the effort -- the backend will
     * still hoist the RHS within the correct zero-trip context.
     */
    for (; nconds; nconds--) {
      ast = mk_stmt(A_ENDIF, 0);
      add_stmt_before(ast, std);
    }
  }
  FREE(pastConds);
}

/* Remove conjunction astSrc from conjunction astTarg. If astSrc not in
 * astTarg, return astTarg. If astSrc = astTarg, return 0. */
static int
remove_cond(int astTarg, int astSrc)
{
  int astNewl, astNewr, astNew;

  if (astSrc == astTarg)
    return 0;
  if (A_TYPEG(astTarg) != A_BINOP || A_OPTYPEG(astTarg) != OP_LAND)
    return astTarg;
  astNewl = remove_cond(A_LOPG(astTarg), astSrc);
  if (!astNewl) {
    astNewr = remove_cond(A_ROPG(astTarg), astSrc);
    return astNewr;
  }
  astNewr = remove_cond(A_ROPG(astTarg), astSrc);
  if (!astNewr) {
    astNewl = remove_cond(A_LOPG(astTarg), astSrc);
    return astNewl;
  }
  if (astNewl == A_LOPG(astTarg) && astNewr == A_ROPG(astTarg))
    astNew = astTarg;
  else
    astNew = mk_binop(OP_LAND, astNewl, astNewr, DT_LOG);
  return astNew;
}

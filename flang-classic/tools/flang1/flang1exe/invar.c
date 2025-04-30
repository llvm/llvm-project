/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief optimizer submodule responsible for invariant analysis
   for a given loop.  Used by the optimizer and vectorizer.

   NOTE: During the detection phase of the analysis, stores are analyzed by
   traversing the store lists of a loop and its children.  A store is marked by
   recording the store item in the corresponding name's stl field.  The lifetime
   of the names' stl fields extends after the analysis phase so that the general
   query function, is_invariant, can be provided.  The stl fields (and ili
   marked during the analysis) are cleaned up in unmark/unmarkv.
 */

/* FIXME -- move these to the definition sites
    void invariant(int)	- do invariant analysis and code motion
    void invariant_nomotion(int) - just determine which expressions are
        invariant
    void invariant_mark(int, int) - mark ili as invariant or not invariant
    LOGICAL is_invariant(int) - analyze ili for invariancy
    void invariant_unmark() - unmark ili & nmes which have been marked by the
        invariant analysis when performing code motion.  The lifetime of
        marked ili is during invariant and other optimizations (such as
        induction analysis) which need to know what expressions are invariant.
        unmarking occurs after the info is no longer needed.
    void invariant_unmarkv() - unmark ili & nmes which have been marked by the
        invariant analysis when not performing code motion (i.e., by the
        vectorizer).
    static void invar_init(int)
    static void invar_end(int)
    static void initnames(*stl)
    static void cleannames(*stl)
    static LOGICAL invar_src(int)
    static void invar_arrnme(int)
    static void invar_motion(int)
    static void store_ili(int)
    LOGICAL is_sym_invariant_safe(int, int)

*/
#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "semant.h"

static LOGICAL invar_src(int);
static void invar_init(int);
static void invar_mark(int);
static void invar_end(int);
#ifdef FLANG_INVAR_UNUSED
static void invar_arrnme(int);
#endif
static void invar_motion(int);
static void initnames(STL *);
static void cleannames(STL *);

static int lpx, fgx, stdx;
static LOGICAL safe;        /* safe to classify an expression invariant */
static LOGICAL mark_value;  /* invariant mark (INV) or NOT_INV if in a
                             * critical section.
                             */
static LOGICAL mark_return; /* TRUE, if mark_value == INV */

/*
 * perform invariant analysis along with code motion.
 */
void
invariant(int lp)
{
  int stdx, fgx, fg_tail, fg_tailnext;
  invar_init(lp);

  invar_mark(lp);
  LP_HSTDF(lp) = 0;
  LP_DSTDF(lp) = 0;

  if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
    if (!LP_MEXITS(lp) && !LP_PARLOOP(lp) && !LP_CS(lp)) {
      fg_tail = LP_TAIL(lp);
      fg_tailnext = FG_LNEXT(fg_tail);
      for (fgx = LP_HEAD(lp); fgx && fgx != fg_tailnext; fgx = FG_LNEXT(fgx)) {
        for (stdx = FG_STDFIRST(fgx); stdx; stdx = STD_NEXT(stdx)) {
          if (STD_ACCEL(stdx) || STD_KERNEL(stdx)) {
          } else {
            invar_motion((int)stdx);
          }
          if (stdx == FG_STDLAST(fgx))
            break;
        }
        if (fgx == fg_tail)
          break;
      }
    }
  }

  invar_end(lp);
  LP_HSTDF(lp) = 0;
  LP_DSTDF(lp) = 0;

}

/*
 * perform invariant analysis only (no code motion); used by vectorizer
 * to determine which expressions are invariant in a loop.
 */
void
invariant_nomotion(int lp)
{
  invar_init(lp);

  invar_mark(lp);

  invar_end(lp);

}

void
invariant_mark(int ilix, int mark)
{
  int i;

  AST_INVP(ilix, mark);
  i = opt.invb.stg_avail++;
  OPT_NEED(opt.invb, int, 100);
  opt.invb.stg_base[i] = ilix;
}

/*
 * cleanup invariant table after it's not needed (typically during the last
 * step in induction analysis, must be used after invariant analysis when
 * it includes code motion.
 */
void
invariant_unmark(void)
{
  int i, ilix;
  STL *stl;

  for (i = opt.invb.stg_avail - 1; i; i--) {
    ilix = opt.invb.stg_base[i];
    if (!AST_ISINV_TEMP(ilix))
      AST_INVP(ilix, 0);
  }
  /*
   * clean up the names entries of the stores in the loop and in any nested
   * loops
   */
  stl = LP_STL(lpx);
  cleannames(stl);
}

/*
 * cleanup invariant table after it's not needed by the vectorize.
 */
void
invariant_unmarkv(void)
{
  int i, ilix;
  STL *stl;

  for (i = opt.invb.stg_avail - 1; i; i--) {
    ilix = opt.invb.stg_base[i];
    AST_INVP(ilix, 0);
  }
  /*
   * clean up the names entries of the stores in the loop and in any nested
   * loops
   */
  stl = LP_STL(lpx);
  cleannames(stl);
}

static void
invar_init(int lp)
{
  STL *stl;

  opt.invb.stg_avail = 1;
  lpx = lp;
  if (OPTDBG(9, 256))
    fprintf(gbl.dbgfil, "\n---------- Invariant trace of loop %d\n", lp);

  stl = LP_STL(lp);
  /*
   * initialize the names entries of the stores in the loop and in any
   * nested loops.  Cleanup occurs in unmark/unmarkv.
   */
  initnames(stl);
}

static void
invar_mark(int lp)
{
  /*
   * If a loop contains a critical section, first traverse the blocks which
   * have their FG_CS flags set; this ensures that any ili in the critical
   * sections will not be considered to be invariant.
   */
  if (LP_CS(lp)) {
    mark_value = NOT_INV;
    mark_return = FALSE;
    for (fgx = LP_FG(lp); fgx; fgx = FG_NEXT(fgx)) {
      safe = TRUE;
      if (FG_CS(fgx)) {
        if (OPTDBG(9, 256))
          fprintf(gbl.dbgfil, "   flow graph node %d (LP_CS)\n", fgx);
        for (stdx = FG_STDFIRST(fgx); stdx; stdx = STD_NEXT(stdx)) {
          if (OPTDBG(9, 256))
            fprintf(gbl.dbgfil, "      ilt %d\n", stdx);
          (void)invar_src((int)STD_AST(stdx));
          if (stdx == FG_STDLAST(fgx))
            break;
        }
      }
    }
  }
  mark_value = INV;
  mark_return = TRUE;

  /*
   * Next, traverse the blocks in the loop that are always executed.
   * i.e, blocks which dominate the tail of the loop.  Otherwise, could
   * first discover an unsafe invariant expression which appears later
   * in block that's always executed.
   */
  for (fgx = LP_FG(lp); fgx; fgx = FG_NEXT(fgx)) {
    safe = is_dominator(fgx, (int)LP_TAIL(lp));
    if (safe) {
      if (OPTDBG(9, 256))
        fprintf(gbl.dbgfil, "   flow graph node %d (%s)\n", fgx,
                safe ? "safe" : "unsafe");
      for (stdx = FG_STDFIRST(fgx); stdx; stdx = STD_NEXT(stdx)) {
        if (OPTDBG(9, 256))
          fprintf(gbl.dbgfil, "      std %d\n", stdx);
        (void)invar_src((int)STD_AST(stdx));
        if (stdx == FG_STDLAST(fgx))
          break;
      }
    }
  }
  /*
   * The last traversal is the blocks in the loop that are not always
   * executed.
   */
  for (fgx = LP_FG(lp); fgx; fgx = FG_NEXT(fgx)) {
    safe = is_dominator(fgx, (int)LP_TAIL(lp));
    if (!safe) {
      if (OPTDBG(9, 256))
        fprintf(gbl.dbgfil, "   flow graph node %d (%s)\n", fgx,
                safe ? "safe" : "unsafe");
      for (stdx = FG_STDFIRST(fgx); stdx; stdx = STD_NEXT(stdx)) {
        if (OPTDBG(9, 256))
          fprintf(gbl.dbgfil, "      std %d\n", stdx);
        (void)invar_src((int)STD_AST(stdx));
        if (stdx == FG_STDLAST(fgx))
          break;
      }
    }
  }

}

extern void restore_hoist_stmt(int lp);

static void
invar_end(int lp)
{
  restore_hoist_stmt(lp);
}

/*
 * initialize the names entries of the stores in the loop and in any nested
 * loops
 */
static void
initnames(STL *stlitem)
{
  STL *tmp;
  int i, j, nme;

  tmp = stlitem;
  for (i = tmp->store; i; i = STORE_NEXT(i)) {
    nme = STORE_NM(i);
    j = NME_STL(nme);
    if (j)
      STORE_TYPE(j) |= STORE_TYPE(i);
    else
      NME_STL(nme) = i;
  }
  /* init names (recursively) for nested loops too. */
  for (tmp = tmp->childlst; tmp != NULL; tmp = tmp->nextsibl) {
    initnames(tmp);
  }
}

/*
 * clean up the names entries of the stores in the loop and in any nested
 * loops
 */
static void
cleannames(STL *stlitem)
{
  STL *tmp;
  int i;

  tmp = stlitem;
  for (i = tmp->store; i; i = STORE_NEXT(i))
    NME_STL(STORE_NM(i)) = 0;

  /* clean up (recursively) names of any descendants, too */
  for (tmp = tmp->childlst; tmp != NULL; tmp = tmp->nextsibl) {
    cleannames(tmp);
  }
}

/*
 * Determine if given ili is invariant.  WARNING:  this assumes that
 * the regular analysis has been done; i.e., safe and lpx which are
 * static have been set.
 */
LOGICAL
is_invariant(int ilix)
{
  LOGICAL invf;
  if (OPTDBG(9, 256))
    fprintf(gbl.dbgfil, "is_invariant of ili %d called\n", ilix);
  invf = invar_src(ilix);
  if (OPTDBG(9, 256))
    fprintf(gbl.dbgfil, "is_invariant of ili %d returns %s\n", ilix,
            invf ? "TRUE" : "FALSE");
  return invf;
}

static void compute_invariant(int, int *);

static LOGICAL
invar_src(int ast)
{
  ast_visit(1, 1);
  ast_traverse(ast, NULL, compute_invariant, NULL);
  ast_unvisit();
  return AST_ISINV(ast);
}

static void
compute_invariant(int ast, int *dummy)
{
  int atype;
  int sym;
  int nme;
  int dtype;
  int i, asd;
  int argt;
  int cnt;
  LOGICAL invar_flag;

  /* already done
  if (AST_INVG(ast))
      return;
  */

  switch (atype = A_TYPEG(ast)) {
  case A_NULL:
  case A_CNST:
  case A_CMPLXC:
    break; /* mark invariant */
  case A_ID:
    sym = A_SPTRG(ast);
    if (ST_ISVAR(STYPEG(sym)) || (ST_ARRDSC == STYPEG(sym))) {
      nme = A_NMEG(ast);
      if (NME_STL(nme)) {
        if (OPTDBG(9, 256))
          fprintf(gbl.dbgfil,
                  "         ast %5d-%s %s is not invariant--ST_NME %d\n", ast,
                  astb.atypes[atype], SYMNAME(sym), nme);
        goto mark_variant;
      }
      if (NME_TYPE(nme) != NT_VAR)
        goto mark_variant;
      if (!is_sym_invariant_safe(nme, lpx)) {
        if (OPTDBG(9, 256)) {
          fprintf(gbl.dbgfil, "         ast %5d-%s not invariant - %s unsafe\n",
                  ast, astb.atypes[atype], SYMNAME(sym));
        }
        goto mark_variant;
      }
    }
    break;
  case A_MEM:
    if (!AST_ISINV(A_PARENTG(ast)))
      goto markd_variant;
    /* _ast_trav(A_MEMG(ast)) */
    break;
  case A_BINOP:
    invar_flag = AST_ISINV(A_LOPG(ast)) && AST_ISINV(A_ROPG(ast));
    if (!invar_flag)
      goto markd_variant;
    goto chk_fp_safe;
  case A_UNOP:
  case A_CONV:
  case A_PAREN:
    if (!AST_ISINV(A_LOPG(ast)))
      goto markd_variant;
    goto chk_fp_safe;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    if (!AST_ISINV(A_LOPG(ast)))
      goto markd_variant;
    for (i = 0; i < (int)ASD_NDIM(asd); i++)
      if (!AST_ISINV(ASD_SUBS(asd, i)))
        goto markd_variant;
    break;
  case A_SUBSTR:
    if (!AST_ISINV(A_LOPG(ast)))
      goto markd_variant;
    if (A_LEFTG(ast) && !AST_ISINV(A_LEFTG(ast)))
      goto markd_variant;
    if (A_RIGHTG(ast) && !AST_ISINV(A_RIGHTG(ast)))
      goto markd_variant;
    break;
  case A_TRIPLE:
    /* [lb]:[ub][:stride] */
    if (A_LBDG(ast) && !AST_ISINV(A_LBDG(ast)))
      goto markd_variant;
    if (A_UPBDG(ast) && !AST_ISINV(A_UPBDG(ast)))
      goto markd_variant;
    if (A_STRIDEG(ast) && !AST_ISINV(A_STRIDEG(ast)))
      goto markd_variant;
    break;
  case A_FUNC:
    goto markd_variant;
  case A_ALLOC:

/* currently enabled when STD_HSTBLE(stdx) is set , this flag is set
 * when the allocate/deallocate is for forall temp array.
 * might need to keep track of allo/deall.
 */
    if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
      if (STD_HSTBLE(stdx) && is_alloc_ast(ast)) {
        int srcast = A_SRCG(ast);
        if (A_TYPEG(srcast) == A_SUBSCR)
          goto markd_variant;
        compute_invariant(srcast, NULL);
        if (!AST_ISINV(srcast))
          goto markd_variant;
        if (A_SHAPEG(srcast)) {
          int asd, i, ndim, ss;
          asd = A_ASDG(srcast);
          ndim = ASD_NDIM(asd);
          for (i = 0; i < ndim; ++i) {
            ss = ASD_SUBS(asd, i);
            compute_invariant((ss), NULL);
            if (!AST_ISINV(ss))
              goto markd_variant;
          }
        }
      }
      if (STD_HSTBLE(stdx) && is_dealloc_ast(ast)) {
        int std = STD_HSTBLE(stdx);
        /* check if ast of std is this ast and it is hoistable and it is
         * invariant */

        if (A_TYPEG(A_SRCG(ast)) == A_SUBSCR || !STD_HSTBLE(std))
          goto markd_variant;
        if (AST_ISINV(A_SRCG(ast)))
          break;
      }
    }
    goto markd_variant;
    break;

  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_RAN:
    case I_RANDOM_NUMBER:
    case I_RANDOM_SEED:
      goto markd_variant;
    }
    cnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    if (cnt) {
      i = 0;
      while (TRUE) {
        /* lfm -- optional args */
        if (ARGT_ARG(argt, i) != 0 && !AST_ISINV(ARGT_ARG(argt, i)))
          goto markd_variant;
        if (++i >= cnt)
          break;
      }
    }
  chk_fp_safe:
    if (!safe) {
      dtype = DDTG(A_DTYPEG(ast));
      if (DT_ISREAL(dtype) || DT_ISCMPLX(dtype))
        goto unsafe;
    }
    break;
  default:
    AST_INVP(ast, NOT_INV);
    return; /* default; don't add to invb list */
  }

  invariant_mark(ast, mark_value);
  if (OPTDBG(9, 256)) {
    if (mark_value == INV || mark_value == T_INV) {
      fprintf(gbl.dbgfil, "         ast %5d-%s", ast, astb.atypes[atype]);
      if (A_TYPEG(ast) == A_ID)
        fprintf(gbl.dbgfil, " %s", SYMNAME(A_SPTRG(ast)));
      fprintf(gbl.dbgfil, " is invariant\n");
    } else
      fprintf(gbl.dbgfil, "         ast %5d-%s is not invariant -- csec\n", ast,
              astb.atypes[atype]);
  }
  return;

unsafe:
  if (OPTDBG(9, 256))
    fprintf(gbl.dbgfil, "         ast %5d-%s -- unsafe\n", ast,
            astb.atypes[atype]);
/* fall thru */
markd_variant:
  if (OPTDBG(9, 256))
    fprintf(gbl.dbgfil, "         ast %5d-%s is not invariant\n", ast,
            astb.atypes[atype]);
mark_variant:
  invariant_mark(ast, NOT_INV);

}

#ifdef FLANG_INVAR_UNUSED
/*
 * for any array nme's in nme, check their subscripts.  Note that this only
 * marks the ili if necessary.  Detecting an invariant subscript is not
 * enough for it to be moved; the subscript ili must actually appear in
 * a tree rooted by an ilt.  There are situations (e.g., a[i+1]) where
 * the subscript ili does not actually appear in the reference due to
 * algebraic simplification.
 */
static void
invar_arrnme(int nme)
{
  int anme;

  anme = nme;
  while (NME_TYPE(anme) == NT_MEM || NME_TYPE(anme) == NT_IND)
    anme = NME_NM(anme);
  while (NME_TYPE(anme) == NT_ARR) {
    if (NME_SUB(anme)) {
      if (OPTDBG(9, 256))
        fprintf(gbl.dbgfil, "         arrnme %d, subili %d\n", anme,
                NME_SUB(anme));
      (void)invar_src((int)NME_SUB(anme));
    }
    anme = NME_NM(anme);
  }
}
#endif

static LOGICAL is_std_hoistable(int, int);

/* allocate statement to move up, deallocate statement to move down */
static void
invar_motion(int std)
{
  int ast, astd;
  LOGICAL hstable = FALSE;
  /* only STD_HSTBLE for now */
  if (STD_HSTBLE(std)) {
    ast = STD_AST(std);
    if (is_alloc_ast(ast)) {
      hstable = is_std_hoistable(std, lpx);
      if (hstable) {
        hoist_stmt(std, LP_FG(lpx), lpx);
      } else {
        /* mark not hoistable so that the deallocate will not be hoist */
        astd = STD_HSTBLE(std);
        STD_HSTBLE(std) = 0;
        STD_HSTBLE(astd) = 0;
      }
    } else if (is_dealloc_ast(ast) && STD_HSTBLE(std)) {
      astd = STD_HSTBLE(std);
      if (STD_VISIT(astd)) {
        hoist_stmt(std, LP_FG(lpx), lpx);

        /* only do one level of hoisting for now */
        STD_HSTBLE(std) = 0;
        STD_HSTBLE(astd) = 0;
      }
    }
  }
}

static LOGICAL
def_in_innerlp(int lpx, int def_lp)
{
  int lp;
  for (lp = LP_CHILD(lpx); lp; lp = LP_SIBLING(lp)) {
    if (lp == def_lp)
      return TRUE;
  }
  return FALSE;
}

/* return a number of def in a loop
 * relies on LP_STL
 * for subscript x[3]:
 *    if there is a whole array ref (x) => no
 *    if there is a subscript but variable index (x[n]) => no
 */
static LOGICAL
has_def_inlp(int ast, int lp, int std)
{
  int lop, def, nme, def_fg, asd, def_addr, ndim;
  int def_count = 0;

  switch (A_TYPEG(ast)) {

  case A_SUBSCR:
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) != A_ID) /* don't handle derived type yet */
      return FALSE;
    nme = A_NMEG(ast);
    if (!nme)
      return FALSE;
    if (NME_TYPE(nme) != NT_ARR)
      return FALSE;
    nme = NME_NM(nme);

    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      def_fg = DEF_FG(def);
      def_addr = DEF_ADDR(def);
      asd = A_ASDG(def_addr);
      ndim = ASD_NDIM(asd);
      if (FG_LOOP(def_fg) != lp && !def_in_innerlp(lp, FG_LOOP(def_fg)))
        continue;
      if (DEF_STD(def) == std) {
        ++def_count;
      } else {
        if (def_addr == DEF_LHS(def)) /* whole array */
          ++def_count;
        else if (A_TYPEG(ASD_SUBS(asd, 0)) != A_CNST)
          ++def_count;            /* index is variable, don't do further */
        else if (def_addr == ast) /* same index */
          ++def_count;
      }
      if (def_count > 1)
        return FALSE;
    }
    break;
  case A_ID:
    nme = A_NMEG(ast);
    break;
  default:
    return FALSE;
    break;
  }

  if (def_count > 1)
    return FALSE;

  return TRUE;
}

static LOGICAL
    /* also do the hoist */
    is_hoistable(int ast, int std, int lp)
{
  int lop, sym, nme, sptr, def_addr, ndim, def_std, i;
  int asd, du_std;
  int use_count, def_count;
  int l, u, s, r, cnme, found_nme, def, def_fg, hoistme;
  STL *stl;
  DU *du;

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    r = is_hoistable(A_ROPG(ast), std, lp);
    l = is_hoistable(A_LOPG(ast), std, lp);
    if (l & r) {
      return TRUE;
    }
    break;
  case A_UNOP:
    l = is_hoistable(A_LOPG(ast), std, lp);
    if (l)
      return TRUE;
    break;
  case A_TRIPLE:
    l = is_hoistable(A_LBDG(ast), std, lp);
    u = is_hoistable(A_UPBDG(ast), std, lp);
    s = is_hoistable(A_STRIDEG(ast), std, lp);

    return (l & u & s);
    break;
  case A_CNST:
    return TRUE;
  case A_ID:
    return FALSE; /* for now */
    break;
  case A_SUBSCR:
    if (A_SHAPEG(ast)) /* currently don't do too many nested subscripts */
      return FALSE;
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      sym = A_SPTRG(lop);
      if (ADDRTKNG(sym) && (LP_CALLFG(lp) || LP_CALLINTERNAL(lp))) {
        /* to do list: if call is external we can check if this is passed
         * as an argument
         * NOTE: should I also check inner loop calls?
         */
        return FALSE;
      }
      cnme = A_NMEG(lop);
      if (NME_SYM(cnme) == -1)
        cnme = NME_NM(cnme);
      if (STYPEG(sym) == ST_DESCRIPTOR) {
        stl = LP_STL(lp);
        /* no need to use store list */
        found_nme = 0;
        for (i = stl->store; i; i = STORE_NEXT(i)) {
          nme = STORE_NM(i);
          if (NME_SYM(nme) == -1)
            nme = NME_NM(nme);
          sptr = NME_SYM(nme);
          if (cnme != nme)
            continue;
          if (NME_TYPE(nme) != NT_VAR)
            return FALSE;

          found_nme = 1;
          use_count = 0;
          def_count = 0;
          hoistme = 0;
          for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
            def_addr = DEF_ADDR(def);
            def_std = DEF_STD(def);

            /* check if this def is in a loop or inner loop */
            def_fg = DEF_FG(def);
            if (lp != FG_LOOP(def_fg)) {
              if (LP_CHILD(lp) && !def_in_innerlp(lp, FG_LOOP(def_fg)))
                continue;
            }
            if (def_addr && A_TYPEG(def_addr) == A_SUBSCR) {
              asd = A_ASDG(def_addr);
              ndim = ASD_NDIM(asd);
              if (ndim > 1)
                return FALSE;
              /* don't want to do more checking if subscript is not constant */
              if (A_TYPEG(ASD_SUBS(asd, 0)) != A_CNST)
                return FALSE;

              if (def_addr == ast) {

                /* handle with 1 def for now */
                ++def_count;
                if (def_count > 1)
                  return FALSE;

                for (du = DEF_DU(def); du; du = du->next) {
                  du_std = USE_STD(du->use);
                  if (USE_STD(du->use) == std) {
                    use_count++;

                    /* this check(def_std != std) is unnessary but we
                     * want to make sure that infinite recursive
                     * won't happen here.
                     */
                    if (def_std != std) {
                      if (!is_std_hoistable(def_std, lp)) {
                        return FALSE;
                      } else {
                        hoistme = 1;
                      }
                    }
                  }
                }
              }

            } else {
              return FALSE;
            }
            if (def_count > 1)
              return FALSE;

            /* do the hoist of std */
            if (hoistme) {
              hoist_stmt(def_std, LP_FG(lp), lp);
              return TRUE;
            }
          }

          if (found_nme) {
            return TRUE;
          }
          /* found nme */
          break;
        }
        if (found_nme == 0) { /* not found in store list */
          return TRUE;
        }
      }
    }
    break;
  default:
    break;
  }
  return FALSE;
}

/* currently handle temp alloc/dealloc and assignment of descriptors */
static LOGICAL
is_std_hoistable(int std, int lpx)
{
  LOGICAL canhoist = FALSE;
  LOGICAL l, r, s, u, sym, tmplog;
  int shape, n, i, astd;
  int ast = STD_AST(std);

  switch (A_TYPEG(ast)) {
  case A_ALLOC:
    /* assume temp alloc */
    if (is_alloc_ast(ast) && STD_HSTBLE(std)) {
      ast = A_SRCG(ast);
      switch (A_TYPEG(ast)) {
      case A_SUBSCR:
        shape = A_SHAPEG(ast);
        n = SHD_NDIM(shape);
        tmplog = FALSE;
        for (i = 0; i < n; ++i) {
          l = is_hoistable(SHD_LWB(shape, i), std, lpx);
          u = is_hoistable(SHD_UPB(shape, i), std, lpx);
          s = is_hoistable(SHD_STRIDE(shape, i), std, lpx);
          tmplog = l && u && s;
          if (tmplog) {
            int fast = A_FIRSTALLOCG(ast);
            if (fast) {
              if (!is_hoistable(fast, std, lpx))
                return FALSE;
            }
          } else {
            return FALSE;
          }
        }
        return tmplog;
        break;
      case A_ID:
        return FALSE; /* To do list */
        break;
      default:
        return FALSE;
      }
    } else if (is_dealloc_ast(ast) && STD_HSTBLE(std)) {
      /* this assume that allocate statement is already visited */
      astd = STD_HSTBLE(ast);
      if (STD_HSTBLE(astd) && STD_VISIT(astd))
        return TRUE;
    }
    return FALSE;
    break;

  case A_ASN:
    l = A_DESTG(ast);
    r = A_SRCG(ast);

    if (A_TYPEG(l) == A_SUBSCR) {
      int a = A_LOPG(l);
      if (A_TYPEG(a) != A_ID) {
        return FALSE;
      }
    } else if (A_TYPEG(l) == A_ID) {
      sym = A_SPTRG(ast);
      if (ADDRTKNG(sym)) {
        if (LP_CALLFG(lpx) || LP_CALLINTERNAL(lpx))
          return FALSE;
      }
    } else {
      return FALSE;
    }

    if (has_def_inlp(l, lpx, std) <= 1) {
      /* only hoist this std if the lhs has only one def,
       * address taken ok but no call
       */
      if (is_hoistable(r, std, lpx)) {
        return TRUE;
      } else {
        return FALSE;
      }
    }

    break;

  /* everything else is not hoistable for now */
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
  case A_ASNGOTO:
  default:
    return FALSE;
    break;
  }
  return canhoist;
}

#ifdef FLANG_INVAR_UNUSED
static void
store_ili(int ilix)
{
  int temp;

  /* assign temp; routine marks ILI with candidate entry */
}
#endif

#ifdef FLANG_INVAR_UNUSED
static LOGICAL
is_nme_loop_safe(int ldnme, int lpx)
{
  if (flg.opt < 2)
    return FALSE; /* copy from front end, why false? */

  /*
      int stl = LP_STL(lpx);
      if( stl ){
          int store;
          for( store = stl->store; store; store = STORE_NEXT(store) ){
              int stnme = STORE_NM(store);
              if( conflict( ldnme, stnme ) != NOCONFLICT ){
                  return FALSE;
              }
          }
      }
      if (!is_call_safe(ldnme)) {
              return FALSE;
      }

      for( inner = LP_CHILD(lpx); inner; inner = LP_SIBLING(inner) ){
          if( !is_nme_loop_safe( ldnme, inner ) )
              return FALSE;
      }

  */
  return TRUE;
}
#endif

LOGICAL
is_sym_invariant_safe(int nme, int lpx)
{
  int sym, sflag;
  sflag = 0;

  /*
   * a symbol is not safe (potential "side-effect" conflicts exist) if:
   * 1.  for c and fortran, the symbol is volatile
   *
   * 2.  for fortran, the symbol is equivalenced
   *
   * 3.  loop contains a call AND
   *     a.  sym's address has been taken, or
   *     b.  sym's storage class is not auto (it is static or extern).
   *     c.  or for pascal- sym is used in two or more nested scope levels
   *
   * 4.  loop contains a store via a pointer AND the sym could
   *     conflict with a pointer (see optutil.c:is_sym_ptrsafe()).
   *
   * 5.  the symbol is private and the loop is not a parallel region.
   *
   * 6.  the symbol is not a private variable and is stored while in a
   *     critical section in this loop or any enclosed loop, or in a
   *     parallel region.
   *
   * 7.  the symbol is not a private variable and the loop contains
   *     a parallel section.
   *
   * 8.  for Fortran, the loop contains a call to an internal subprogram
   *     and this is a noninternal variable.
   *
   * NOTE that this differs from is_sym_optsafe in that a load via a pointer
   * does not rule out loads of certain symbols as being invariant. This is
   * because the symbol cannot be redefined with a load via a pointer.
   */
  sym = NME_SYM(nme);
  if (VOLG(sym))
    return (FALSE);

  if (SOCPTRG(sym))
    return (FALSE);
  if (LP_CALLFG(lpx) && (!IS_LCL_OR_DUM(sym)
                             )) {
    return (FALSE);
  }

  if (SOCPTRG(sym))
    return (FALSE);

  if (LP_CALLINTERNAL(lpx) && !INTERNALG(sym))
    return FALSE;

  if (IS_PRIVATE(sym) && !LP_PARREGN(lpx))
    return (FALSE);

  if ((LP_CSECT(lpx) || LP_PARREGN(lpx)) && !IS_PRIVATE(sym)) {
    Q_ITEM *q;
    for (q = LP_STL_PAR(lpx); q != NULL; q = q->next)
      if (q->info == nme)
        return (FALSE);
  }

  if (LP_PARSECT(lpx) && !IS_PRIVATE(sym)) {
    return (FALSE);
  }

  if (!sflag && LP_PTR_STORE(lpx) && !is_sym_ptrsafe(sym))
    /* check ptr before argument */
    /*    if (flg.opt >= 2 && XBIT (xxx)) {
            is_sym_invar_inlp();
        }*/
    return (FALSE);

  if (!sflag && LP_CALLFG(lpx) && ADDRTKNG(sym)) {
    /* check argument in a call, if it is in then it is safe  */
    return (FALSE);
  }

  return (TRUE);
}

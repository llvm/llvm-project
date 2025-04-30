/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file bblock.c
    \brief Fortran front-end basic block module.
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "ast.h"
#include "semant.h"
#include "soc.h"
#include "transfrm.h"
#include "extern.h"
#include "dinit.h"
#include "fdirect.h"
#include "pd.h"
#include "rtlRtns.h"

static int entry_point;
static int par;     /* in OpenMp parallel region */
static int cs;      /* in OpenMp critical section */
static int parsect; /* in OpenMp parallel section */
static int task;    /* in OpenMp task */
static int atomic;  /* in atomic region */
static int kernel; /* in cuda kernel */

static void init_newargs(int);
static void assumshp_args(int);
static void adj_based_arrays(void);
static void gen_early_bnd_dependencies(int);
static void add_bound_assignments(int);
static void set_std_parflags(int);
static void gen_early_array_bnds(int sptr);
static void gen_early_str_len();

struct {
  int *base;
  int sz;
  int avl;
} erly_bnds_depd = {NULL, 0, 0};

/** \brief Called from semant_init().
 */
void
bblock_init()
{
}

/** \brief Return nonzero if there are any CUDA Fortran kernels
 */
int
bblock()
{
  int std;
  int ast;
  int sptr;
  int label;
  int ent;
  int next_std;
  int new_label;
  int new_ast;
  int new_std;
  int penul_std, last_std;
  int ret_label;
  int aret_id;
  int ret_cnt;
  int tmp;
  INT ent_cnt;
  int ent_select_id;
  int has_kernel = 0;
  ITEM *itemp;

  if (STD_NEXT(0) == STD_PREV(0)) { /* end only ? */
    /* add something for entry -- lfm */
    new_ast = mk_stmt(A_CONTINUE, 0);
    add_stmt_after(new_ast, 0);
  }

  erly_bnds_depd.base = NULL;
  erly_bnds_depd.sz = 10;
  erly_bnds_depd.avl = 0;
  NEW(erly_bnds_depd.base, int, erly_bnds_depd.sz);

  sem.temps_reset = TRUE;
  entry_point = 0;
  last_std = STD_LAST;

  if (gbl.arets) {
    /* for alternate returns, will use a compiler-created local symbol
     * which will contain the alternate return value
     */
    sptr = getsymbol("z__ret");
    STYPEP(sptr, ST_VAR);
    DTYPEP(sptr, DT_INT);
    DCLDP(sptr, 1);
    SCP(sptr, SC_LOCAL);
    aret_id = mk_id(sptr);
  }

  penul_std = STD_PREV(last_std);
  ret_label = 0;
  if ((label = STD_LABEL(last_std))) {
    /* if end statement is labeled, insert a continue before the end
     * and transfer the label to the continue.  This continue becomes
     * the common exit point for the subprogram.
     */
    ret_label = label;
    new_ast = mk_stmt(A_CONTINUE, 0);
    gbl.exitstd = penul_std = add_stmt_after(new_ast, penul_std);
    STD_LABEL(penul_std) = ret_label;
    STD_LABEL(last_std) = 0;
    STD_LINENO(penul_std) = STD_LINENO(last_std);
  } else /* if (gbl.arets) */ {
    /* if there are alternate returns, always create a new exit point
     * since aret_id contains the alternate return value. Will generate
     * return when A_END is seen.
     */
    /* LFM -- always create a new last statement, since we may need to
     * add stuff around the last statement.
     */
    ret_label = getlab();
    new_ast = mk_stmt(A_CONTINUE, 0);
    gbl.exitstd = penul_std = add_stmt_after(new_ast, penul_std);
    STD_LABEL(penul_std) = ret_label;
    STD_LINENO(penul_std) = STD_LINENO(last_std);
  }

  ret_cnt = 0;

  /* Use the ast visit routines to keep track of the id asts representing
   * the left-hand sides of any assignments added to the prologue which
   * assign values to adjustable bounds temporaries.  If the id ast has
   * been visited, then its assignment has been written.
   */
  ast_visit(1, 1);

  if (gbl.rutype != RU_PROG) {
    init_newargs(gbl.currsub);
    if (ASSUMSHPG(gbl.currsub)) /* must process before adjustable args */
      assumshp_args(gbl.currsub);
  }

  adj_based_arrays();
  gen_early_str_len();

  FREE(erly_bnds_depd.base);
  erly_bnds_depd.sz = 0;
  erly_bnds_depd.avl = 0;
  ast_unvisit(); /* clean up visit fields of the id asts of bound temps*/

  ent_cnt = ent_select_id = 0;
  ENTNUMP(gbl.currsub, 0);
  if (SYMLKG(gbl.currsub) > NOSYM) {
    gbl.ent_select = getsymbol("z__ent");
    STYPEP(gbl.ent_select, ST_VAR);
    DTYPEP(gbl.ent_select, DT_INT);
    DCLDP(gbl.ent_select, 1);
    SCP(gbl.ent_select, SC_LOCAL);
    HCCSYMP(gbl.ent_select, TRUE);
    ent_select_id = mk_id(gbl.ent_select);
    ast = mk_assn_stmt(ent_select_id, mk_cval(ent_cnt, DT_INT), DT_INT);
    entry_point = add_stmt_after(ast, entry_point);
  }
  ENTSTDP(gbl.currsub, entry_point);

  par = cs = parsect = atomic = 0;
  kernel = 0;

  for (std = STD_NEXT(entry_point); std; std = next_std) {
    next_std = STD_NEXT(std); /* 'cause insertions may alter STD_NEXT */
    gbl.lineno = STD_LINENO(std);
    label = STD_LABEL(std);
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_ENTRY:
      new_label = 0;
      if (A_TYPEG(STD_AST(STD_PREV(std))) != A_GOTO) {
        int astlab;
        new_label = getlab();
        DEFDP(new_label, 1);
        RFCNTI(new_label);
        new_ast = mk_stmt(A_GOTO, 0);
        astlab = mk_label(new_label);
        A_L1P(new_ast, astlab);
        (void)add_stmt_before(new_ast, std);
      }

      ent = A_SPTRG(ast);
      entry_point = std;

      ast_visit(1, 1);
      init_newargs(ent);
      if (ASSUMSHPG(ent)) /* must process before adjustable args */
        assumshp_args(ent);
      adj_based_arrays();

      ast_unvisit();

      ent_cnt++;
      ast = mk_assn_stmt(ent_select_id, mk_cval(ent_cnt, DT_INT), DT_INT);
      entry_point = add_stmt_after(ast, entry_point);
      ENTSTDP(ent, entry_point);
      ENTNUMP(ent, ent_cnt);

      new_ast = mk_stmt(A_CONTINUE, 0);
      new_std = add_stmt_after(new_ast, entry_point);
      if (new_label)
        STD_LABEL(new_std) = new_label;

      break;

    case A_RETURN:
      if (ret_cnt == 0 && penul_std == std) {
        /* the subprogram's only exit point is from the return which
         * is the penultimate statement. The subprogram doesn't not
         * contain alternate returns.
         */
        if (label) {
          /* since the return is labeled, insert a continue before
           * the return and transfer the label to the continue.
           * The continue becomes the subprogram's exit point.
           */
          new_ast = mk_stmt(A_CONTINUE, 0);
          new_std = add_stmt_before(new_ast, std);
          STD_LABEL(new_std) = label;
          STD_LABEL(std) = 0;
          gbl.exitstd = new_std;
        } else
          /* since the return isn't labeled, just use the statement
           * before the return as the subprogram's exit point.
           */
          gbl.exitstd = STD_PREV(std);
      } else {
        int astlab;
        /* change this return to a 'goto ret_label' */
        ret_cnt++;
        if (ret_label == 0)
          ret_label = getlab();
        tmp = A_LOPG(ast); /* just in case it's an alternate ret.*/
        A_LOPP(ast, 0);
        A_TYPEP(ast, A_GOTO);
        RFCNTI(ret_label);
        astlab = mk_label(ret_label);
        A_L1P(ast, astlab);
        if (gbl.arets) {
          if (tmp == 0)
            tmp = astb.i0;
          new_ast = mk_assn_stmt(aret_id, tmp, DT_INT);
          new_std = add_stmt_before(new_ast, std);
          if (label) {
            STD_LABEL(new_std) = label;
            STD_LABEL(std) = 0;
          }
        }
      }
      break;

    case A_END:
      if (gbl.arets) {
        /*  gbl.exitstd already set */
        new_ast = mk_stmt(A_RETURN, 0);
        A_LOPP(new_ast, aret_id);
        (void)add_stmt_before(new_ast, std);
      } else if (gbl.exitstd == 0) {
        if (ret_label) {
          new_ast = mk_stmt(A_CONTINUE, 0);
          gbl.exitstd = add_stmt_before(new_ast, std);
          STD_LABEL(gbl.exitstd) = ret_label;
        } else
          gbl.exitstd = STD_PREV(std);
      }
      break;

    case A_CONTINUE:
      if (penul_std != std && (label == 0 || RFCNTG(label) == 0)) {
        int s;
        s = STD_PREV(std);
        STD_PREV(next_std) = s;
        STD_NEXT(s) = next_std;
      }
      break;

    case A_MP_PARALLEL:
      par++;
      break;
    case A_MP_ENDPARALLEL:
      STD_PAR(std) = 1;
      par--;
      break;
    case A_MP_CRITICAL:
    case A_MP_ATOMIC:
      cs++;
      break;
    case A_MP_ENDATOMIC:
    case A_MP_ENDCRITICAL:
      STD_CS(std) = 1;
      cs--;
      break;
    case A_ATOMIC:
      atomic++;
      break;
    case A_ENDATOMIC:
      STD_ATOMIC(std) = 1;
      atomic--;
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      task++;
      break;
    case A_MP_ENDTASK:
    case A_MP_ETASKLOOP:
      STD_TASK(std) = 1;
      task--;
      break;
    case A_MP_MASTER:
    case A_MP_SINGLE:
    case A_MP_SECTIONS:
      parsect++;
      break;
    case A_MP_ENDMASTER:
    case A_MP_ENDSINGLE:
    case A_MP_ENDSECTIONS:
      STD_PARSECT(std) = 1;
      parsect--;
      break;

    default:
      if (label) {
        if (RFCNTG(label)) {
          new_ast = mk_stmt(A_CONTINUE, 0);
          new_std = add_stmt_before(new_ast, std);
          STD_LABEL(new_std) = label;
          set_std_parflags(new_std);
        }
        STD_LABEL(std) = 0;
      }
      break;
    }
    set_std_parflags(std);
  }

  if (gbl.arets)
    /* since the alternate return id must be initialized to 0, add
     * the assignment to the 'prologue' of each entry.
     */
    for (ent = gbl.currsub; ent != NOSYM; ent = SYMLKG(ent)) {
      entry_point = ENTSTDG(ent);
      new_ast = mk_assn_stmt(aret_id, astb.i0, DT_INT);
      entry_point = add_stmt_after(new_ast, entry_point);
      ENTSTDP(ent, entry_point);
    }

  if (sem.type_initialize) {
    int std2;
    for (std2 = ENTSTDG(gbl.currsub); STD_LINENO(std2) == 0;
         std2 = STD_NEXT(std2))
      ;
    std2 = STD_PREV(std2);
    for (itemp = sem.type_initialize; itemp; itemp = itemp->next) {
      int stdx = CONSTRUCTSYMG(itemp->t.sptr) ?
        BLOCK_ENTRY_STD(itemp->t.sptr) : std2;
      gen_type_initialize_for_sym(itemp->t.sptr, stdx, 0, 0);
    }
  }

  for (itemp = sem.alloc_mem_initialize; itemp; itemp = itemp->next) {
    int stdx = CONSTRUCTSYMG(itemp->t.sptr) ?
      BLOCK_ENTRY_STD(itemp->t.sptr) : ENTSTDG(gbl.currsub);
    gen_alloc_mem_initialize_for_sym(itemp->t.sptr, stdx);
  }

  for (itemp = sem.auto_dealloc; itemp; itemp = itemp->next) {
    int stdx = CONSTRUCTSYMG(itemp->t.sptr) ?
      STD_PREV(BLOCK_EXIT_STD(itemp->t.sptr)) : gbl.exitstd;
    gen_conditional_dealloc_for_sym(itemp->t.sptr, stdx);
  }

  for (itemp = sem.auto_finalize; itemp; itemp = itemp->next) {
    int stdx = CONSTRUCTSYMG(itemp->t.sptr) ?
      STD_PREV(BLOCK_EXIT_STD(itemp->t.sptr)) : gbl.exitstd;
    gen_finalization_for_sym(itemp->t.sptr, stdx, 0);
  }

#if DEBUG
  if (DBGBIT(10, 2))
    dump_std();
#endif
  return has_kernel;
}

/* This routine is used to initialize the dummies to newarg and newdsc
 * For example, call sub(a) will be later call sub(a$b, a$s).
 * this routine creates a$b and a$s and store them to the first and
 * second halves of address field of a
 */
static void
init_newargs(int ent)
{
  int dscptr;
  int arg, narg;
  int i;
  int newarg, newdsc;

  narg = PARAMCTG(ent);
  dscptr = DPDSCG(ent);
  for (i = 0; i < narg; i++) {
    arg = aux.dpdsc_base[dscptr + i];
    if (STYPEG(arg) == ST_ENTRY)
      error(480, 4, gbl.lineno, SYMNAME(arg), CNULL);
    if (STYPEG(arg) == ST_PROC)
      continue;
    if (POINTERG(arg) && STYPEG(arg) != ST_ARRAY &&
        (DDTG(DTYPEG(arg)) == DT_DEFERCHAR ||
         DDTG(DTYPEG(arg)) == DT_DEFERNCHAR)) {
    } else if (!ALLOCATTRG(arg) && STYPEG(arg) != ST_ARRAY &&
               DTY(DTYPEG(arg)) != TY_ARRAY) {
      continue;
    }
    if (is_bad_dtype(DTYPEG(arg)))
      continue;
    newarg = NEWARGG(arg);
    newdsc = NEWDSCG(arg);
    if (newarg != 0 && newdsc != 0)
      continue;
    if ((SEQG(arg) && !POINTERG(arg)) || F90POINTERG(arg)) {
      newarg = arg;
    } else if (XBIT(57, 0x80) || (XBIT(57, 0x10000) && ASSUMSHPG(arg)) ||
               XBIT(57, 0x80000)) {
      newarg = arg;
    } else {
      newarg = sym_get_formal(arg);
    }
    if (XBIT(57, 0x80000) && SDSCG(arg)) {
      newdsc = SDSCG(arg);
    } else {
      newdsc = sym_get_arg_sec(arg);
    }
    NEWARGP(arg, newarg);
    NEWDSCP(arg, newdsc);
  }
}

static void find_assumshp_dep(int);
static LOGICAL _find_assumshp_dep(int, LOGICAL *);
static void add_assumshp(int);

static struct {
  int *abase;
  int acnt;
  int sacnt;
} assumshp;

/*
 * process assumed-shape array arguments before any adjustable arrays.
 * the function result could be an adjustable array whose size depends
 * on the size of one of the dummy arguments.
 * Need to check their lower bounds for any dependencies on other assumed-shape
 * arguments; need to first process those arguments on which other arrays
 * depend.
 */
static void
assumshp_args(int ent)
{
  int cnt;
  int dscptr;
  int arg;
  int i;
  int any;

  cnt = PARAMCTG(ent);
  if (cnt == 0)
    return;
  /*
   * Create a table large enough to be hold two copies of the assumed-shape
   * arguments.  First, collect the assumed-shape arguments into the first
   * part of the table.
   */
  assumshp.acnt = 0;
  NEW(assumshp.abase, int, 2 * cnt + 1);
  dscptr = DPDSCG(ent);
  while (TRUE) {
    arg = aux.dpdsc_base[dscptr];
    if (arg && STYPEG(arg) != ST_PROC && DTY(DTYPEG(arg)) == TY_ARRAY &&
        ASSUMSHPG(arg)) {
      assumshp.abase[assumshp.acnt] = arg;
      assumshp.acnt++;
      VISITP(arg, 1);  /* is an assumed-shape arg to the function */
      VISIT2P(arg, 0); /* have not yet checked for dependencies */
    }
    if (--cnt == 0)
      break;
    dscptr++;
  }

  /*
   * Recursively check the lower bounds for any dependencies on other
   * assumed-shape arguments; the second part of the table will contain
   * the arguments in an order such that they will be processed before
   * any dependents.
   */
  any = 0;
  assumshp.sacnt = assumshp.acnt;
  if (assumshp.acnt > 1) {
    for (i = 0; i < assumshp.acnt; i++) {
      arg = assumshp.abase[i];
      find_assumshp_dep(arg);
      any = 1;
    }
  } else if (assumshp.acnt == 1)
    add_assumshp(assumshp.abase[0]);

  if (any) {
    ast_unvisit(); /* clean up the visit list */
                   /* repeat visit setup for keeping track of assignments to the
                    * adjustable bounds temporaries.
                    */
    ast_visit(1, 1);
  }

  ENTSTDP(ent, entry_point);
  for (i = assumshp.acnt; i < assumshp.sacnt; i++) {
    arg = assumshp.abase[i];
    VISITP(arg, 0);
    VISIT2P(arg, 0);
    set_assumed_bounds(arg, ent, 0);
  }
  entry_point = ENTSTDG(ent);
  FREE(assumshp.abase);
}

static void
find_assumshp_dep(int arg)
{
  int ndim;
  ADSC *ad;
  int d;
  int lb;

  if (VISIT2G(arg)) /* already checked for dependencies */
    return;
  VISIT2P(arg, 1);
  ad = AD_DPTR(DTYPEG(arg));
  ndim = AD_NUMDIM(ad);
  for (d = 0; d < ndim; d++) {
    lb = AD_LWBD(ad, d);
    if (!lb || A_ALIASG(lb)) {
      continue;
    }
    ast_traverse(lb, _find_assumshp_dep, NULL, NULL);
  }
  add_assumshp(arg);
}

static LOGICAL
_find_assumshp_dep(int ast, LOGICAL *rr)
{
  if (A_TYPEG(ast) == A_ID) {
    int sym;

    sym = A_SPTRG(ast);
    switch (STYPEG(sym)) {
    case ST_ARRAY:
    case ST_VAR:
      if (SCG(sym) == SC_DUMMY && DTY(DTYPEG(sym)) == TY_ARRAY &&
          ASSUMSHPG(sym) && VISITG(sym)) {
        find_assumshp_dep(sym);
      }
      break;
    default:;
    }
  }
  return FALSE;
}

static void
add_assumshp(int arg)
{
  int i;

  for (i = assumshp.acnt; i < assumshp.sacnt; i++) {
    if (arg == assumshp.abase[i])
      return;
  }
  assumshp.abase[assumshp.sacnt] = arg;
  assumshp.sacnt++;
}

static int
early_specification_stmt_needed(int ast)
{
  int argt;
  int argcnt;
  int i;
  int sptr;

  if (ast) {
    switch (A_TYPEG(ast)) {
    case A_BINOP:
      return (early_specification_stmt_needed(A_LOPG(ast)) ||
              early_specification_stmt_needed(A_ROPG(ast)));
    case A_UNOP:
    case A_CONV:
    case A_PAREN:
      return (early_specification_stmt_needed(A_LOPG(ast)));
    case A_INTR:
      sptr = A_SPTRG(A_LOPG(ast));
      switch (STYPEG(sptr)) {
      case ST_GENERIC:
      case ST_INTRIN:
        /*  simple */
        argt = A_ARGSG(ast);
        argcnt = ARGT_CNT(argt);
        for (i = 0; i < argcnt; i++) {
          if (early_specification_stmt_needed(ARGT_ARG(argt, i))) {
            return TRUE;
          }
        }
        break;
      default:
        return TRUE;
      }
      break;
    case A_FUNC:
      /* Part of the fix for FS1551.  Early specification stmt is
       * needed if any function call is found.
       */
      return TRUE;
    }
  }
  return FALSE;
}

static void
add_to_early_bnd_list(int ast)
{
  NEED(erly_bnds_depd.avl + 1, erly_bnds_depd.base, int, erly_bnds_depd.sz,
       erly_bnds_depd.sz + 10);
  erly_bnds_depd.base[erly_bnds_depd.avl++] = ast;
}

static void
gen_early_bnd_dependencies(int ast)
{
  int sptr;
  int std;
  ADSC *ad;
  int ndims;
  int i;
  int bndsptr;
  int argt;
  int argcnt;
  int dtype;
  LOGICAL early_spec_gend = FALSE;

  if (!ast)
    return;

  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = A_SPTRG(ast);
    /* insert dependencies before dependent bnds exprs */
    std = CONSTRUCTSYMG(sptr) ? BLOCK_ENTRY_STD(sptr) : ENTSTDG(gbl.currsub);
    if (STYPEG(sptr) == ST_ARRAY && ADJARRG(sptr) && !EARLYSPECG(sptr)) {
      ad = AD_DPTR(DTYPEG(sptr));
      ndims = AD_NUMDIM(ad);
      for (i = 0; i < ndims; i++) {
        if (A_TYPEG(AD_LWAST(ad, i)) != A_CNST) {
          bndsptr = A_SPTRG(AD_LWAST(ad, i));
          if (!EARLYSPECG(bndsptr)) {
            std = add_stmt_after(
                mk_assn_stmt(AD_LWAST(ad, i), AD_LWBD(ad, i), astb.bnd.dtype),
                std);
            EARLYSPECP(bndsptr, 1);
            gen_early_bnd_dependencies(AD_LWBD(ad, i));
          }
          early_spec_gend = TRUE;
        }
        if (A_TYPEG(AD_UPAST(ad, i)) != A_CNST) {
          bndsptr = A_SPTRG(AD_UPAST(ad, i));
          if (!EARLYSPECG(bndsptr)) {
            std = add_stmt_after(
                mk_assn_stmt(AD_UPAST(ad, i), AD_UPBD(ad, i), astb.bnd.dtype),
                std);
            EARLYSPECP(bndsptr, 1);
            gen_early_bnd_dependencies(AD_UPBD(ad, i));
          }
          early_spec_gend = TRUE;
        }
      }
    }
    if (ADJLENG(sptr)) {
      if (!EARLYSPECG(sptr)) {
        int rhs, cvlen;
        dtype = DDTG(DTYPEG(sptr));
        if (!CVLENG(sptr)) {
          CVLENP(sptr, sym_get_scalar(SYMNAME(sptr), "len", DT_INT));
        }
        cvlen = CVLENG(sptr);
        rhs = DTY(dtype + 1);
        rhs = mk_convert(rhs, DTYPEG(cvlen));
        rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));
        std = add_stmt_after(
            mk_assn_stmt(mk_id(CVLENG(sptr)), rhs, DTYPEG(cvlen)), std);
        add_to_early_bnd_list(rhs);
        EARLYSPECP(CVLENG(sptr), 1);
      }
      early_spec_gend = TRUE;
    }
    if (early_spec_gend) {
      EARLYSPECP(sptr, 1);
    }
    break;
  case A_FUNC:
  case A_INTR:
    argt = A_ARGSG(ast);
    argcnt = ARGT_CNT(argt);
    for (i = 0; i < argcnt; i++) {
      gen_early_bnd_dependencies(ARGT_ARG(argt, i));
    }
    break;
  case A_BINOP:
    gen_early_bnd_dependencies(A_LOPG(ast));
    gen_early_bnd_dependencies(A_ROPG(ast));
    break;
  case A_UNOP:
  case A_CONV:
  case A_PAREN:
    gen_early_bnd_dependencies(A_LOPG(ast));
    break;
  }
}

static void
gen_early_str_len()
{
  SPTR sptr;
  int dtype;
  int i;

  for (sptr = gbl.p_adjstr; sptr != NOSYM; sptr = ADJSTRLKG(sptr)) {
    if (IGNOREG(sptr)) {
      continue;
    }
    dtype = DDTG(DTYPEG(sptr));
    if (HCCSYMG(sptr) || CCSYMG(sptr)) {
      continue;
    }
    if (early_specification_stmt_needed(DTY(dtype + 1))) {
      int rhs, cvlen;
      if (!CVLENG(sptr)) {
        CVLENP(sptr, sym_get_scalar(SYMNAME(sptr), "len", DT_INT));
      }
      cvlen = CVLENG(sptr);
      rhs = DTY(dtype + 1);
      rhs = mk_convert(rhs, DTYPEG(cvlen));
      rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));
      if (CONSTRUCTSYMG(sptr))
        (void)add_stmt_before(
            mk_assn_stmt(mk_id(CVLENG(sptr)), rhs, DTYPEG(cvlen)),
              BLOCK_ENDPROLOG_STD(sptr));
      else
        entry_point = add_stmt_after(
            mk_assn_stmt(mk_id(CVLENG(sptr)), rhs, DTYPEG(cvlen)), entry_point);
      add_to_early_bnd_list(rhs);
      EARLYSPECP(sptr, 1);
      EARLYSPECP(CVLENG(sptr), 1);
    }
  }
  for (i = erly_bnds_depd.avl; i; --i) {
    gen_early_bnd_dependencies(erly_bnds_depd.base[i - 1]);
  }
  erly_bnds_depd.avl = 0;
}

static void
gen_early_array_bnds(int sptr)
{
  ADSC *ad;
  int ndims;
  int i;
  LOGICAL early_bnd_emitted = FALSE;

  ad = AD_DPTR(DTYPEG(sptr));
  ndims = AD_NUMDIM(ad);
  for (i = 0; i < ndims; i++) {
    int bndsptr;
    bndsptr = A_SPTRG(AD_LWAST(ad, i));
    if (early_specification_stmt_needed(AD_LWBD(ad, i))) {
      if (!EARLYSPECG(bndsptr)) {
        if (CONSTRUCTSYMG(sptr))
          (void)add_stmt_before(
              mk_assn_stmt(AD_LWAST(ad, i), AD_LWBD(ad, i), astb.bnd.dtype),
              BLOCK_ENDPROLOG_STD(sptr));
        else
          entry_point = add_stmt_after(
              mk_assn_stmt(AD_LWAST(ad, i), AD_LWBD(ad, i), astb.bnd.dtype),
              entry_point);
        add_to_early_bnd_list(AD_LWBD(ad, i));
        EARLYSPECP(bndsptr, 1);
      }
      AD_LWBD(ad, i) = AD_LWAST(ad, i);
      early_bnd_emitted = TRUE;
    }
    bndsptr = A_SPTRG(AD_UPAST(ad, i));
    if (early_specification_stmt_needed(AD_UPBD(ad, i))) {
      if (!EARLYSPECG(bndsptr)) {
        if (CONSTRUCTSYMG(sptr))
          (void)add_stmt_before(
              mk_assn_stmt(AD_UPAST(ad, i), AD_UPBD(ad, i), astb.bnd.dtype),
              BLOCK_ENDPROLOG_STD(sptr));
        else
          entry_point = add_stmt_after(
              mk_assn_stmt(AD_UPAST(ad, i), AD_UPBD(ad, i), astb.bnd.dtype),
              entry_point);
        add_to_early_bnd_list(AD_UPBD(ad, i));
        EARLYSPECP(bndsptr, 1);
      }
      AD_UPBD(ad, i) = AD_UPAST(ad, i);
      early_bnd_emitted = TRUE;
    }
  }
  if (early_bnd_emitted) {
    EARLYSPECP(sptr, 1);
  }
  for (i = erly_bnds_depd.avl; i; --i) {
    gen_early_bnd_dependencies(erly_bnds_depd.base[i - 1]);
  }
  /* MORE need to adjust other erly_bnds_depd flds */
  erly_bnds_depd.avl = 0;
}

/* pointer-based arrays with adjustable dimensions */
static void
adj_based_arrays(void)
{
  int sptr;

  for (sptr = gbl.p_adjarr; sptr != NOSYM; sptr = SYMLKG(sptr)) {
    if (!IGNOREG(sptr)) {
      if (RESULTG(sptr)) {
        if (SCG(sptr) == SC_DUMMY)
          gen_early_array_bnds(sptr);
      } else if (SCG(sptr) == SC_LOCAL || SCG(sptr) == SC_DUMMY) {
        gen_early_array_bnds(sptr);
      } else if (SCG(sptr) != SC_NONE) {
        add_bound_assignments(sptr);
      }
    }
  }
}

static void
add_bound_assignments(int sym)
{
  int dtype;
  ADSC *ad;
  int numdim;
  int i;
  int bnd;
  int ast;
  int tmp;
  int zbaseast;
  int save_entry_point = entry_point;

  if (CONSTRUCTSYMG(sym))
    entry_point = STD_PREV(BLOCK_ENDPROLOG_STD(sym));

  dtype = DTYPEG(sym);
  ad = AD_DPTR(dtype);
  numdim = AD_NUMDIM(ad);
  zbaseast = 0;
  /* NOTE: a bound is adjustable if its ast is non-zero and it is
   *       not a constant or aliased constant.
   */
  for (i = 0; i < numdim; i++) {
    bnd = AD_LWBD(ad, i);
    tmp = AD_LWAST(ad, i);
    if (bnd && A_ALIASG(tmp) == 0) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, bnd, astb.bnd.dtype);
        entry_point = add_stmt_after(ast, entry_point);
        ast_visit(tmp, tmp); /* mark id ast as visited */
      }
    }
    bnd = AD_UPBD(ad, i);
    tmp = AD_UPAST(ad, i);
    if (bnd && A_ALIASG(tmp) == 0) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, bnd, astb.bnd.dtype);
        entry_point = add_stmt_after(ast, entry_point);
        ast_visit(tmp, tmp); /* mark id ast as visited */
      }
    }
    {
      /* update the ZBASE ast tree */
      int nexttmp, ast;
      if (i == 0) {
        zbaseast = AD_LWAST(ad, i);
      } else if (A_ALIASG(AD_ZBASE(ad)) == 0) {
        int a;
        a = mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i), astb.bnd.dtype);
        zbaseast = mk_binop(OP_ADD, zbaseast, a, astb.bnd.dtype);
      }
      /* add assignment to multiplier temp for next dimension */
      tmp = AD_MLPYR(ad, i);
      nexttmp = AD_MLPYR(ad, i + 1);
      if (tmp && nexttmp && A_ALIASG(nexttmp) == 0 && A_VISITG(nexttmp) == 0) {
        if (AD_LWBD(ad, i) == astb.bnd.one)
          ast = astb.bnd.one;
        else
          ast = AD_LWAST(ad, i);
        ast = mk_mlpyr_expr(ast, AD_UPAST(ad, i), tmp);
        ast = mk_assn_stmt(nexttmp, ast, astb.bnd.dtype);
        entry_point = add_stmt_after(ast, entry_point);
        ast_visit(nexttmp, nexttmp); /* mark id ast as visited */
      }
    }
  }
  if (A_ALIASG(AD_ZBASE(ad)) == 0) {
    /* add assignment to zbase temp */
    tmp = AD_ZBASE(ad);
    if (A_VISITG(tmp) == 0) {
      ast = mk_assn_stmt(tmp, zbaseast, astb.bnd.dtype);
      entry_point = add_stmt_after(ast, entry_point);
      ast_visit(tmp, tmp); /* mark id ast as visited */
    }
  }

  if (CONSTRUCTSYMG(sym))
    entry_point = save_entry_point;
}

static void
set_std_parflags(int std)
{
  if (par)
    STD_PAR(std) = 1;
  if (cs)
    STD_CS(std) = 1;
  if (parsect)
    STD_PARSECT(std) = 1;
  if (task)
    STD_TASK(std) = 1;
  if (kernel)
    STD_KERNEL(std) = 1;
}

static void merge(int, int);
static int donetwice1, donetwice2; /* to prevent message coming out twice */

/** \brief Merge common blocks.

    If two modules declare the same common block, both sets of symbols
    appear in the program.  If either uses the same variable names,
    by Fortran 90 rules, the names may not be used, though they could be
    hidden by USE, ONLY clauses.  To generate correct output, we must
    merge the two definitions.  This routine checks that the distributed
    variables in the duplicate commons have the same type, distribution,
    offsets, and sizes.  It replaces all references to one of the commons
    by references to the other.  For nondistributed data, this routine
    generates EQUIVALENCE data.
 */
void
merge_commons()
{
  int cmn1, pcmn1, biggest1, cmn2, pcmn2;
  donetwice1 = 0;
  donetwice2 = 0;
  pcmn1 = 0;
  for (cmn1 = gbl.cmblks; cmn1 > NOSYM; cmn1 = SYMLKG(cmn1)) {
    int removed1, any;
    biggest1 = cmn1;
    any = 0;
    for (cmn2 = SYMLKG(cmn1); cmn2 > NOSYM; cmn2 = SYMLKG(cmn2)) {
      if (NMPTRG(cmn1) == NMPTRG(cmn2)) {
        any = 1;
        if (SIZEG(cmn2) > SIZEG(biggest1)) {
          biggest1 = cmn2;
        }
      }
    }
    removed1 = 0;
    if (any) {
      pcmn2 = pcmn1;
      for (cmn2 = cmn1; cmn2 > NOSYM; cmn2 = SYMLKG(cmn2)) {
        if (NMPTRG(biggest1) == NMPTRG(cmn2) && cmn2 != biggest1) {
          merge(biggest1, cmn2);
          /* remove cmn2 */
          if (pcmn2) {
            SYMLKP(pcmn2, SYMLKG(cmn2));
          } else {
            gbl.cmblks = SYMLKG(cmn2);
          }
          if (cmn2 == cmn1)
            removed1 = 0;
        } else {
          pcmn2 = cmn2;
        }
      }
    }
    if (!removed1)
      pcmn1 = cmn1;
  }
} /* merge_commons */

#ifdef FLANG_COMMON_BLOCK_ERROR_UNUSED
/* FIXME: This function is not being used, but it isn't clear why error 482
 * ("COMMON /$/ is declared differently in two subprograms - $") does not
 * need to be emitted.
 */
static void
puttwice(int cmn1, int cmn2)
{
  static char errmsg[256];
  int func1, func2;
  if (donetwice1 == cmn1 && donetwice2 == cmn2)
    return;
  donetwice1 = cmn1;
  donetwice2 = cmn2;
  func1 = ENCLFUNCG(cmn1);
  if (func1 == 0)
    func1 = gbl.currsub;
  func2 = ENCLFUNCG(cmn2);
  if (func2 == 0)
    func2 = gbl.currsub;
  sprintf(errmsg, "%s and %s", SYMNAME(func1), SYMNAME(func2));
  error(482, 3, FUNCLINEG(gbl.currsub), SYMNAME(cmn1), errmsg);
} /* puttwice */
#endif

static LOGICAL
same_datatype(int s1, int s2)
{
  if (POINTERG(s1) != POINTERG(s2))
    return FALSE;
  if (!eq_dtype(DTYPEG(s1), DTYPEG(s2)))
    return FALSE;
  return TRUE;
} /* same_datatype */

static void
rewrite_all_asts()
{
  int std, dtype, dim, numdim, ss, dd;
  /* rewrite all statement AST trees */
  for (std = STD_NEXT(0); std > 0; std = STD_NEXT(std)) {
    ss = ast_rewrite(STD_AST(std));
    STD_AST(std) = ss;
  }
  /* rewrite all array bounds */
  for (dtype = 0; dtype < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
    switch (DTY(dtype)) {
    case TY_ARRAY:
      if (DTY(dtype + 2)) {
        dd = ast_rewrite(ADD_ZBASE(dtype));
        ADD_ZBASE(dtype) = dd;
        dd = ast_rewrite(ADD_NUMELM(dtype));
        ADD_NUMELM(dtype) = dd;
        numdim = ADD_NUMDIM(dtype);
        for (dim = 0; dim < numdim; ++dim) {
          dd = ast_rewrite(ADD_MLPYR(dtype, dim));
          ADD_MLPYR(dtype, dim) = dd;
          dd = ast_rewrite(ADD_LWBD(dtype, dim));
          ADD_LWBD(dtype, dim) = dd;
          dd = ast_rewrite(ADD_UPBD(dtype, dim));
          ADD_UPBD(dtype, dim) = dd;
          dd = ast_rewrite(ADD_LWAST(dtype, dim));
          ADD_LWAST(dtype, dim) = dd;
          dd = ast_rewrite(ADD_UPAST(dtype, dim));
          ADD_UPAST(dtype, dim) = dd;
          dd = ast_rewrite(ADD_EXTNTAST(dtype, dim));
          ADD_EXTNTAST(dtype, dim) = dd;
        }
      }
      break;
    }
  }
} /* rewrite_all_asts */

/* this is a symbol in the base common block to which misaligned symbols
 * can be equivalenced */
static int merge_common_symbol;
static int merge_member;

static int
make_equiv_ss(ISZ_T pos, int dtype)
{
  int ss, numss, SIZE, i;
  ss = sem.eqv_ss_avail;
  numss = ADD_NUMDIM(dtype);
  sem.eqv_ss_avail += numss + 1;
  NEED(sem.eqv_ss_avail, sem.eqv_ss_base, int, sem.eqv_ss_size,
       sem.eqv_ss_size + 50);
  EQV_NUMSS(ss) = numss;
  SIZE = 1;
  for (i = 0; i < numss; ++i) {
    int lb, ub, lbval, ubval, p, sval;
    lb = ADD_LWBD(dtype, i);
    ub = ADD_UPBD(dtype, i);
    if (lb == 0) {
      lbval = 1;
    } else {
      lb = A_ALIASG(lb);
      if (lb == 0) {
        interr("merge_common: no lower bound", dtype, 3);
        lb = astb.i1;
      }
      lbval = CONVAL2G(A_SPTRG(lb));
    }
    if (pos == 0) {
      EQV_SS(ss, i) = mk_cval(lbval, DT_INT);
    } else {
      ub = A_ALIASG(ub);
      if (ub == 0) {
        interr("merge_common: no upper bound", dtype, 3);
        ub = lb;
      }
      ubval = CONVAL2G(A_SPTRG(ub));
      sval = (ubval - lbval + 1) * SIZE;
      p = pos % sval;
      pos = pos - p;
      p = p / SIZE;
      EQV_SS(ss, i) = mk_cval(lbval + p, DT_INT);
      SIZE *= (ubval - lbval + 1);
    }
  }
  return ss;
} /* make_equiv_ss */

static int
smallest_dtype(int cmn)
{
  /* find the member with the smallest datatype */
  int mem, smallest, smallest_size;
  int sz_dt_int = size_of(DT_INT);
  smallest = DT_INT;
  smallest_size = sz_dt_int;
  for (mem = CMEMFG(cmn); mem > NOSYM; mem = SYMLKG(mem)) {
    int dtype, size;
    dtype = DTYPEG(mem);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    size = size_of(dtype);
    if (DTY(dtype) == TY_CHAR && size % sz_dt_int != 0) {
      /* if the character*n is not a multiple of size_of(DT_INT), must use the
       * size_of(TY_CHAR) so add_equivalence will generated a
       * merge_common_symbol with a
       * length that will align the common block elements (FS 18720) */
      dtype = DT_CHAR;
      size = size_of(dtype);
    }
    if (size < smallest_size) {
      smallest = dtype;
      size = smallest_size;
    }
  }
  return smallest;
} /* smallest_dtype */

static void
add_equivalence(int cmn1, int cmn2, int mem1, int mem2)
{
  /* if the symbols are mutually aligned, insert an equivalence */
  ISZ_T off1, off2, off, size1, size2, size;
  int dty1, dtype1, dty2, dtype2, CASE;
  if (merge_member == mem2)
    return;
  off1 = ADDRESSG(mem1);
  off2 = ADDRESSG(mem2);
  dty1 = dtype1 = DTYPEG(mem1);
  dty2 = dtype2 = DTYPEG(mem2);
  if (DTY(dtype1) == TY_ARRAY)
    dty1 = DTY(dtype1 + 1);
  if (DTY(dtype2) == TY_ARRAY)
    dty2 = DTY(dtype2 + 1);
  size1 = size_of(dty1);
  size2 = size_of(dty2);
  if (size1 <= size2) {
    size = size1;
    off = off2 - off1;
    CASE = 1;
  } else {
    size = size2;
    off = off1 - off2;
    CASE = 2;
  }
  /* check alignment */
  if ((off % size) == 0 && (size1 % size) == 0 && (size2 % size) == 0) {
    ISZ_T o, s, m2, m1;
    int evp, ss;
    /* we want to declare EQUIVALENCE( mem1(lb1+m1), mem2(lb2+m2) )
     * but in general we don't know the offsets m1 and m2.
     * In the general case, we have a stupid diophantine equation:
     *	off1 + m1*size1 = off2 + m2*size2
     *	m1*size1 - m2*size2 = off2-off1 = off	[if case 1, size1==size]
     *	m2*size2 - m1*size1 = off1-off2 = off	[if case 2, size2==size]
     * we know 'size' divides size1 and size2, so divide by 'size'
     *	m1 - m2*(size2/size) = (off/size)	[if case 1, size1==size]
     *		m1 = (off/size) + m2*(size2/size)
     *	m2 - m1*(size1/size) = (off/size)	[if case 2, size2==size]
     *		m2 = (off/size) + m1*(size1/size)
     * in either case, we choose a positive value of the independent
     * variable that makes the dependent variable also positive.
     * In case 1, we have
     *	(off/size) + m2*(size2/size) > 0
     *	m2 > -(off/size)/(size2/size)
     *	m2 > -(off*size)/(size2*size)
     *	m2 > -off/size2
     * In case 2, we have
     *	(off/size) + m1*(size1/size) > 0
     *      m1 > -off/size1
     */
    if (CASE == 1) {
      s = size2 / size;
      o = off / size;
      if (o >= 0) {
        m2 = 0;
      } else {
        m2 = -off / size2;
      }
      m1 = o + m2 * s;
    } else {
      s = size1 / size;
      o = off / size;
      if (o >= 0) {
        m1 = 0;
      } else {
        m1 = -o / s;
      }
      m2 = o + m1 * s;
    }
#if DEBUG
    if (DBGBIT(10, 8))
      fprintf(gbl.dbgfil, "common/%s/ %s(%d) and /%s/ %s(%d)\n", SYMNAME(cmn1),
              SYMNAME(mem1), (int)m1, SYMNAME(cmn2), SYMNAME(mem2), (int)m2);
#endif
    if (m1 >= 0 && m2 >= 0 && (DTY(dtype1) == TY_ARRAY || m1 == 0) &&
        (DTY(dtype2) == TY_ARRAY || m2 == 0)) {
      /* Convert 'm1' into subscripts */
      if (DTY(dtype1) != TY_ARRAY) {
        ss = 0;
      } else {
        ss = make_equiv_ss(m1, dtype1);
      }
      evp = sem.eqv_avail;
      ++sem.eqv_avail;
      NEED(sem.eqv_avail + 1, sem.eqv_base, EQVV, sem.eqv_size,
           sem.eqv_size + 20);
      EQV(evp).sptr = mem1;
      EQV(evp).is_first = 1;
      EQV(evp).lineno = 0;
      EQV(evp).subscripts = ss;
      EQV(evp).substring = 0;
      EQV(evp).byte_offset = m1;
      EQV(evp).next = evp + 1;

      /* Convert 'm2' into subscripts */
      if (DTY(dtype2) != TY_ARRAY) {
        ss = 0;
      } else {
        ss = make_equiv_ss(m2, dtype2);
      }
      evp = sem.eqv_avail;
      ++sem.eqv_avail;
      EQV(evp).sptr = mem2;
      EQV(evp).is_first = 0;
      EQV(evp).lineno = 0;
      EQV(evp).subscripts = ss;
      EQV(evp).substring = 0;
      EQV(evp).byte_offset = m2;
      EQV(evp).next = sem.eqvlist;
      sem.eqvlist = evp - 1;
      merge_member = mem2;
      return;
    }
  }
  /* unaligned, need to add something */
  if (merge_common_symbol == 0) {
    int d1, d2, dty, size, sizeast;
    /* find smallest item in cmn1, cmn2 */
    d1 = smallest_dtype(cmn1);
    d2 = smallest_dtype(cmn2);
    if (size_of(d2) < size_of(d1)) {
      d1 = d2;
    }
    /* make it array of type d1 */
    dty = get_array_dtype(1, d1);
    /* how big? */
    size = SIZEG(cmn1);
    if (size < SIZEG(cmn2))
      size = SIZEG(cmn2);
    size = size / size_of(d1);
    sizeast = mk_cval(size, DT_INT);
    ADD_ZBASE(dty) = astb.i1;
    ADD_NUMELM(dty) = sizeast;
    ADD_LWBD(dty, 0) = astb.i1;
    ADD_LWAST(dty, 0) = astb.i1;
    ADD_UPBD(dty, 0) = sizeast;
    ADD_UPAST(dty, 0) = sizeast;
    ADD_EXTNTAST(dty, 0) = sizeast;
    merge_common_symbol = get_next_sym(SYMNAME(cmn1), "eqv");
    DTYPEP(merge_common_symbol, dty);
    SCOPEP(merge_common_symbol, stb.curr_scope);
    STYPEP(merge_common_symbol, ST_ARRAY);
    ADDRESSP(merge_common_symbol, 0);
    CMBLKP(merge_common_symbol, cmn1);
    EQVP(merge_common_symbol, 1);
    DCLDP(merge_common_symbol, 1);
    add_equivalence(cmn1, cmn1, mem1, merge_common_symbol);
    if (gbl.internal > 1)
      INTERNALP(merge_common_symbol, 1);
  }
  add_equivalence(cmn1, cmn2, merge_common_symbol, mem2);
  merge_member = mem2;
} /* add_equivalence */

static void
add_soc(int mem1, int mem2)
{
  if (soc.size == 0) {
    soc.size = 1000;
    NEW(soc.base, SOC_ITEM, soc.size);
    soc.base[0].sptr = 0;
    soc.base[0].next = 0;
  }
  NEED(soc.avail + 2, soc.base, SOC_ITEM, soc.size, soc.size + 100);
  SOC_SPTR(soc.avail) = mem2;
  SOC_NEXT(soc.avail) = SOCPTRG(mem1);
  SOCPTRP(mem1, soc.avail);
  SEQP(mem1, 1);
  ++soc.avail;
  SOC_SPTR(soc.avail) = mem1;
  SOC_NEXT(soc.avail) = SOCPTRG(mem2);
  SOCPTRP(mem2, soc.avail);
  SEQP(mem2, 1);
  ++soc.avail;
} /* add_soc */

static void
merge(int cmn1, int cmn2)
{
  int mem1, mem2, nmem1, nmem2;
  ISZ_T off1, off2, size1, size2;
  int unignore;
#if DEBUG
  if (DBGBIT(10, 8))
    fprintf(gbl.dbgfil, "common/%s/ at symbols %d and %d\n", SYMNAME(cmn1),
            cmn1, cmn2);
#endif
  unignore = 1;
  if (IGNOREG(cmn1) && IGNOREG(cmn2)) {
    unignore = 0;
  }
  if (unignore)
    IGNOREP(cmn1, 0);
  ast_visit(1, 1);
  merge_common_symbol = 0;
  merge_member = 0;
  mem1 = CMEMFG(cmn1);
  mem2 = CMEMFG(cmn2);
  while (mem1 > NOSYM && mem2 > NOSYM) {
    if (unignore) {
      IGNOREP(mem1, 0);
      IGNOREP(mem2, 0); /* this may get marked 'ignore' below */
    }
    nmem1 = SYMLKG(mem1);
    nmem2 = SYMLKG(mem2);
    /* if the offsets and datatypes are exactly equal,
     * replace one by the other.  This works even for distributed
     * variables */
    off1 = ADDRESSG(mem1);
    off2 = ADDRESSG(mem2);
    if (off1 == off2 && same_datatype(mem1, mem2)) {
      ast_replace(mk_id(mem2), mk_id(mem1));
      /* if it doesn't appear in any other 'equivalences', delete it */
      if (SOCPTRG(mem2) == 0) {
        IGNOREP(mem2, 1);
        mem1 = nmem1;
        mem2 = nmem2;
        continue;
      }
    }
    /* if the offsets are the same and the datatypes are the
     * same size, insert an equivalence.  This works only for
     * nondistributed variables */
    size1 = size_of(DTYPEG(mem1));
    size2 = size_of(DTYPEG(mem2));
    /* does mem1 completely come before mem2? */
    if (off1 + size1 <= off2) {
      mem1 = nmem1;
      /* else, does mem2 completely come before mem1? */
    } else if (off2 + size2 <= off1) {
      mem2 = nmem2;
    } else {

      /* add an equivalence statement */
      add_equivalence(cmn1, cmn2, mem1, mem2);
      /* add SOC record */
      add_soc(mem1, mem2);

      if (off1 + size1 < off2 + size2) {
        mem1 = nmem1;
      } else if (off1 + size1 > off2 + size2) {
        mem2 = nmem2;
      } else {
        mem1 = nmem1;
        mem2 = nmem2;
      }
    }
  }
  rewrite_all_asts();
  ast_unvisit();
  /* move all cmn2 members (that aren't ignored) to cmn1 */
  mem1 = CMEMLG(cmn1);
  for (mem2 = CMEMFG(cmn2); mem2 > NOSYM; mem2 = nmem2) {
    nmem2 = SYMLKG(mem2);
    if (!IGNOREG(mem2)) {
      CMBLKP(mem2, cmn1);
      SYMLKP(mem2, NOSYM);
      SYMLKP(mem1, mem2);
      EQVP(mem2, 1);
      mem1 = mem2;
    }
  }
  CMEMLP(cmn1, mem1);
  IGNOREP(cmn2, 1);
} /* merge */

/* for XBIT(57,0x100), renumber source lines */
static int renumber = 0;
void
renumber_lines()
{
  int std;
  for (std = STD_NEXT(0); std > 0; std = STD_NEXT(std)) {
    STD_LINENO(std) = ++renumber;
  }
} /* renumber_lines */

static void mark_symbol(int sptr, int limit);
static int eliminate_save_alignments;
static int eliminate_save_distributed;

static void
mark(int sptr, int limit)
{
  if (sptr <= 0 || sptr >= stb.stg_avail)
    return;
  if (VISITG(sptr))
    return;
  if (IGNOREG(sptr))
    return;
  VISITP(sptr, 1);
  if (sptr < limit)
    mark_symbol(sptr, limit);
} /* mark */

static void
mark_used_variable(int ast, int *plimit)
{
  int sptr, limit;
  limit = *plimit;
  switch (A_TYPEG(ast)) {
  case A_CNST:
  case A_ENTRY:
  case A_ID:
  case A_INIT:
  case A_LABEL:
  case A_MP_COPYIN:
  case A_MP_COPYPRIVATE:
    sptr = A_SPTRG(ast);
    mark(sptr, limit);
    break;
  }
} /* mark_used_variable */

static void
mark_ast(int ast, int limit)
{
  int nlimit;
  if (ast <= 0 || ast >= astb.stg_avail)
    return;
  nlimit = limit;
  ast_traverse(ast, NULL, mark_used_variable, &nlimit);
} /* mark_ast */

static void
mark_dtype(int dtype, int limit)
{
  int member, i, n;
  switch (DTY(dtype)) {
  case TY_PTR:
    mark_dtype(DTY(dtype + 1), limit);
    break;
  case TY_ARRAY:
    n = ADD_NUMDIM(dtype);
    for (i = 0; i < n; ++i) {
      mark_ast(ADD_LWBD(dtype, i), limit);
      mark_ast(ADD_UPBD(dtype, i), limit);
      mark_ast(ADD_LWAST(dtype, i), limit);
      mark_ast(ADD_UPAST(dtype, i), limit);
      mark_ast(ADD_EXTNTAST(dtype, i), limit);
      mark_ast(ADD_MLPYR(dtype, i), limit);
    }
    mark_ast(ADD_NUMELM(dtype), limit);
    mark_ast(ADD_ZBASE(dtype), limit);
    break;
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    for (member = DTY(dtype + 1); member > NOSYM; member = SYMLKG(member)) {
      mark_dtype(DTYPEG(member), limit);
    }
    break;
  }
} /* mark_dtype */

static void
mark_symbol(int sptr, int limit)
{
  /* go through data type looking for symbols and ASTs */
  mark_dtype(DTYPEG(sptr), limit);
  switch (STYPEG(sptr)) {
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_IDENT:
  case ST_VAR:
    /* go through other symbol links */
    mark(CVLENG(sptr), limit);
    mark(DESCRG(sptr), limit);
    mark(MIDNUMG(sptr), limit);
    mark(PTROFFG(sptr), limit);
    mark(SDSCG(sptr), limit);
    mark_ast(PARAMVALG(sptr), limit);
    break;
  case ST_MEMBER:
    mark(DESCRG(sptr), limit);
    mark(MIDNUMG(sptr), limit);
    mark(PTROFFG(sptr), limit);
    mark(SDSCG(sptr), limit);
    break;
  case ST_PROC:
    mark(FVALG(sptr), limit);
    break;
  case ST_ARRDSC:
    mark(SECDSCG(sptr), limit);
    break;
  default:;
  }
} /* mark_symbol */

static void
mark_used_in_independent(INDEP_INFO *indep)
{
  NEWVAR *list;
  REDUCVAR *redp, *locp;
  REDUC_JA *redjap;
  REDUC_JA_SPEC *specp;

  if (!indep)
    return;
  mark_ast(indep->onhome, 0);
  for (list = indep->new_list; list; list = list->next)
    mark_symbol(list->var, 0);

  for (list = indep->index_list; list; list = list->next)
    mark_symbol(list->var, 0);

  for (redp = indep->reduction_list; redp; redp = redp->next)
    mark_symbol(redp->var, 0);

  for (redjap = indep->reduction_ja_list; redjap; redjap = redjap->next) {
    for (specp = redjap->speclist; specp; specp = specp->next) {
      mark_symbol(specp->var, 0);
      for (locp = specp->locvar_list; locp; locp = locp->next) {
        mark_symbol(locp->var, 0);
      }
    }
  }
} /* mark_used_in_independent */

static void
check_used_in_data()
{
  int nw, lineno, fileno, ast;
  VAR *ivl, *tivl, *nivl;
  ACL *ict;
  /* were there any data statements? */
  if ((flg.ipa & 0x20) == 0)
    return;
  if (astb.df == NULL)
    return;
  nw = fseek(astb.df, 0L, 0);
  while (1) {
    nw = fread(&lineno, sizeof(lineno), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&fileno, sizeof(fileno), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&ivl, sizeof(ivl), 1, astb.df);
    if (nw == 0)
      break;
    nw = fread(&ict, sizeof(ict), 1, astb.df);
    for (tivl = ivl; tivl != NULL; tivl = nivl) {
      nivl = tivl->next;
      switch (tivl->id) {
      case Dostart:
        ast = tivl->u.dostart.indvar;
        mark_ast(ast, 0);
        break;
      case Varref:
        ast = tivl->u.varref.ptr;
        mark_ast(ast, 0);
        break;
      case Doend:
      default:
        break;
      }
    }
  }
} /* check_used_in_data */

static LOGICAL
must_mark(int sptr)
{
  if (NMLG(sptr) || EQVG(sptr) || DINITG(sptr) || REFG(sptr)) {
    return TRUE;
  }
  if (STYPEG(sptr) == ST_PROC &&
      (REFG(sptr) || (HCCSYMG(sptr) && TYPDG(sptr)))) {
    return TRUE;
  }
  if (SCG(sptr) == SC_DUMMY) {
    return TRUE;
  }
  if ((STYPEG(sptr) == ST_VAR || STYPEG(sptr) == ST_ARRAY)) {
    if (((MIDNUMG(sptr) && VISITG(MIDNUMG(sptr))) ||
         (PTROFFG(sptr) && VISITG(PTROFFG(sptr))) ||
         (SDSCG(sptr) && VISITG(SDSCG(sptr))))) {
      return TRUE;
    }
    if (eliminate_save_distributed && (ALIGNG(sptr) || DISTG(sptr)) &&
        !MDALLOCG(sptr) &&
        (SCG(sptr) == SC_CMBLK || (SCG(sptr) == SC_BASED && MIDNUMG(sptr) &&
                                   SCG(MIDNUMG(sptr)) == SC_CMBLK))) {
      /* save COMMON distributed symbols for allocation */
      return TRUE;
    }
  }
  return FALSE;
} /* must_mark */

/** \brief For XBIT(57,0x2000), eliminate declarations of any unused variables,
    unless they are in COMMON */
void
eliminate_unused_variables(int which)
{
  int sptr, std, limit, prevsptr, nextsptr, dir;

  /* don't eliminate variables in modules, block data, or host
   * subprograms containing internal subprograms */
  if (gbl.rutype == RU_BDATA) {
    /* 'undeclare' hpf_np$ */
    STYPEP(gbl.sym_nproc, ST_UNKNOWN);
    return;
  } else if (gbl.internal == 1)
    return;

  /* look at 'align' and 'dist' at 1st pass, not 2nd pass */
  eliminate_save_alignments = ((which & 1) ? 1 : 0);
  /* save any common distributed variables, for static initialization */
  eliminate_save_distributed = ((which & 2) ? 1 : 0);

  /* clear visit fields to start */
  for (sptr = 0; sptr < stb.stg_avail; ++sptr)
    VISITP(sptr, 0);

  /* go through all statements */
  ast_visit(1, 1);
  limit = 0;
  for (std = STD_NEXT(0); std > 0; std = STD_NEXT(std)) {
    int ast;
    ast = STD_AST(std);
    ast_traverse(ast, NULL, mark_used_variable, &limit);
  }

  /* go through all loop directives */
  for (dir = 1; dir < direct.lpg.avail; ++dir) {
    LPPRG *lpprg;
    lpprg = direct.lpg.stgb + dir;
    mark_used_in_independent(lpprg->indep);
  }
  /* go through all dynamic loop directives */
  for (dir = 1; dir < direct.dynlpg.avail; ++dir) {
    LPPRG *lpprg;
    lpprg = direct.dynlpg.stgb + dir;
    mark_used_in_independent(lpprg->indep);
  }

  /* some symbols must be visited anyway */
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (IGNOREG(sptr)) {
    } else if (STYPEG(sptr) == ST_STFUNC) {
      IGNOREP(sptr, 1);
    } else if (must_mark(sptr)) {
      mark(sptr, 0);
    }
  }
  /* look for symbols used as indices in data statements */
  check_used_in_data();
  for (sptr = gbl.entries; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    if (!IGNOREG(sptr))
      mark(sptr, 0);
  }
  for (sptr = gbl.externs; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    if (!IGNOREG(sptr))
      mark(sptr, 0);
  }
  /* go through all declarations */
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (VISITG(sptr)) {
      mark_symbol(sptr, sptr);
    }
  }
  /* mark all variables in a common block if anything in the common is used */
  for (sptr = gbl.cmblks; sptr > NOSYM; sptr = nextsptr) {
    int mbr;
    nextsptr = SYMLKG(sptr);
    for (mbr = CMEMFG(sptr); mbr > NOSYM; mbr = SYMLKG(mbr)) {
      if (VISITG(mbr))
        break;
      if (DINITG(mbr))
        break;
    }
    if (mbr > NOSYM) {
      prevsptr = sptr;
      for (mbr = CMEMFG(sptr); mbr > NOSYM; mbr = SYMLKG(mbr)) {
        mark(mbr, mbr + 1);
      }
    }
  }
  ast_unvisit();
  /* eliminate the unused variables or arrays */
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (VISITG(sptr))
      continue;
    if (IGNOREG(sptr))
      continue;
    switch (STYPEG(sptr)) {
    case ST_IDENT:
    case ST_VAR:
    case ST_DESCRIPTOR:
    case ST_ARRAY:
      /* do not eliminate common, dummy, extern */
      switch (SCG(sptr)) {
      case SC_BASED:
        if (MIDNUMG(sptr) && VISITG(MIDNUMG(sptr)))
          break;
        FLANG_FALLTHROUGH;
      case SC_LOCAL:
      case SC_NONE:
      case SC_PRIVATE:
      case SC_STATIC:
        /* mark unused */
        STYPEP(sptr, ST_UNKNOWN);
        SCP(sptr, SC_NONE);
        IGNOREP(sptr, 1);
        break;
      case SC_DUMMY:
      case SC_CMBLK:
      case SC_EXTERN:
        break;
      }
      break;
    case ST_PROC:
      /* mark unused */
      STYPEP(sptr, ST_UNKNOWN);
      SCP(sptr, SC_NONE);
      IGNOREP(sptr, 1);
      break;
    default:;
    }
  }
  /* eliminate any completely unused subprograms */
  prevsptr = 0;
  for (sptr = aux.list[ST_PROC]; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = SLNKG(sptr);
    if (VISITG(sptr)) {
      prevsptr = sptr;
    } else {
      if (prevsptr) {
        SLNKP(prevsptr, nextsptr);
      } else {
        aux.list[ST_PROC] = nextsptr;
      }
      STYPEP(sptr, ST_UNKNOWN);
      SCP(sptr, SC_NONE);
      IGNOREP(sptr, 1);
    }
  }
#if DEBUG
  /* aux.list[ST_PROC] must be terminated with NOSYM, not 0 */
  assert(sptr > 0, "eliminate_unused_variables: corrupted ST_PROC list", sptr, 
         3);
#endif
  /* eliminate any completely unused common blocks */
  if (which == 1) {
    prevsptr = 0;
    for (sptr = gbl.cmblks; sptr > NOSYM; sptr = nextsptr) {
      int mbr;
      nextsptr = SYMLKG(sptr);
      for (mbr = CMEMFG(sptr); mbr > NOSYM; mbr = SYMLKG(mbr)) {
        if (VISITG(mbr))
          break;
        if (DINITG(mbr))
          break;
      }
      if (mbr > NOSYM) {
        prevsptr = sptr;
      } else {
        /* none of the symbols were used or initialized */
        for (mbr = CMEMFG(sptr); mbr > NOSYM; mbr = SYMLKG(mbr)) {
          STYPEP(mbr, ST_UNKNOWN);
          SCP(mbr, SC_NONE);
          IGNOREP(mbr, 1);
        }
        if (prevsptr) {
          SYMLKP(prevsptr, nextsptr);
        } else {
          gbl.cmblks = nextsptr;
        }
        STYPEP(sptr, ST_UNKNOWN);
        SCP(sptr, SC_NONE);
        IGNOREP(sptr, 1);
      }
    }
  }
  /* removed eliminated arrays from gbl.tp_adjarr list */
  prevsptr = 0;
  for (sptr = gbl.tp_adjarr; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = AUTOBJG(sptr);
    if (VISITG(sptr)) {
      prevsptr = sptr;
    } else {
      if (prevsptr) {
        AUTOBJP(prevsptr, nextsptr);
      } else {
        gbl.tp_adjarr = nextsptr;
      }
    }
  }
  /* removed eliminated arrays from gbl.p_adjarr list */
  prevsptr = 0;
  for (sptr = gbl.p_adjarr; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = SYMLKG(sptr);
    if (VISITG(sptr)) {
      prevsptr = sptr;
    } else {
      if (prevsptr) {
        SYMLKP(prevsptr, nextsptr);
      } else {
        gbl.p_adjarr = nextsptr;
      }
    }
  }
  /* clear visit fields when done */
  for (sptr = 0; sptr < stb.stg_avail; ++sptr)
    VISITP(sptr, 0);
} /* eliminate_unused_variables */


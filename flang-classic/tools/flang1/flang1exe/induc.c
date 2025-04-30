/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - optimizer submodule responsible for induction analysis
 */

/* Contains:
 *  void induction_init()    - initialize for induction submodule. called at
 *                             the beginning of the optimizer.
 *  void induction_end()     - cleanup after all functions. called at the end
 *                             of the optimizer.
 *  void induction(int)      - perform induction optimizations on a loop.
 *
 *  static void ind_loop_init()  -  init induction analyis for a loop
 *  static void find_bivs()  - find basic induction variables
 *  static void new_ivs()    - process induction uses, create new ind vars
 *
 *  static void removest(STL *)
 *  static int def_in_out(int, int)  -  determine if a def is live out of a
 *      loop region.
 *  static int is_biv(int, int, int, int)
 *
 *  static void scan_ind_uses(int)
 *  static int  find_fam(int)
 *
 *  static void dump_ind()
 *
 *  *  ALTERNATE ENTRIES TO INDUCTION  *
 *
 *  int get_loop_count(int)  - determine if a loop is countable
 *  void compute_last_values(int) - compute last values and add stores to block
 *  void end_loop_count()    - cleanup
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "induc.h"

static void ind_loop_init(void);
static void find_bivs(void);
static void new_ivs(void);

static void removest(STL *);
static int add_ind(int, int, int, int, Q_ITEM *);
static int is_biv(int ind, int, int, int);
static Q_ITEM *add_biv_def(int, int);
static void check_skip(int, Q_ITEM *);
static void scan_ind_uses(int);
static void scan_def(int, int);

static int find_fam(int);
#ifdef FLANG_INDUC_UNUSED
static DU *get_du(int, int, int, int, int);
static void add_du(DU *, int);
static int add_def(int, int, int, DU *, DU *);
#endif
static void do_branch(void);
static int conv_to_int(int);

static LOGICAL is_biv_store(int, int);
static LOGICAL check_alias(int);
static int def_in_out(int, int);
static void dump_ind(void);

/*  Global Induction Common  */

INDUC induc;

/*  Local Induction Common */

static LOGICAL call_in_loop; /* if entry through induction, TRUE if current
                              * loop contains a call.
                              * if entry through get_loop_count:
                              *     TRUE if current loop contains a call
                              *       and -x 42 4 is not set
                              */
static int lpx;              /* loop being processed */

typedef struct ADU_TAG {/* structure to keep track of aliased iv's */
  int ind;              /* ind. table entry of var which is an alias
                         * KEY flags of an alias ind. entry:
                         * OMIT   - alias is not a biv
                         * ALIAS  - flag indicating have an alias
                         * DELETE - can delete its store (not live-out)
                         * FAMILY - alias of what biv
                         * INIT   - storing expr of def
                         * GONE   - alias def deleted from loop
                         */
  int def;              /* def entry of the alias */
  struct ADU_TAG *next; /* next aliased record */
} ADU;

typedef struct DFIV_TAG {/* struct to keep track of deferred iv uses */
  int ili;               /* deferred induction use */
  int ind;               /* iv upon which use use is based */
  int opn;               /* which operand is non-induction use */
  int use;               /* use item of the ili */
  int nme;               /* nme of load/store whose addr is iv use */
  struct DFIV_TAG *next;
} DFIV;

static struct {/* structure to aid searching */
  int ind;     /* induction table entry of iv being searched */
  int load;
  int def;
  int biv_ili;
  int use;
  int new_skip_opc;
  int new_skip;
  IUSE *usel;     /* queue of induction uses */
  IUSE *last_use; /* end of the induction use queue */
  struct {
    int ind;
    int ili; /* branch ili which might be optimized */
    int cse; /* use of induction variable is cse */
    int opn; /* operand # of x in <biv> <cmp> x */
  } branch;
  ADU *adu_queue; /* list of where alias du's are attached in the
                   * order in which they're discovered.
                   */
  ADU *lastadu;
} srch;

static int mark_useb; /* remember first available use table entry */
static int mark_defb; /* remember first available def table entry */
static int mark_invb; /* remember first available invariant tbl ent*/

static int lastval_cnt;         /* ili of loop count computed by induction
                                 * to be used to compute last values
                                 */
#ifdef FLANG_INDUC_UNUSED
static LOGICAL cr_lastval_uses; /* True if compute_last_values must record
                                 * uses of lastval_cnt
                                 */
#endif
static Q_ITEM *cr_lastval_q;    /* queue of lastval_cnt uses in preheader
                                    * or exit for which new uses (add_new_uses())
                                    * must be created.
                                    */
static int loop_cnt;            /* ili of loop count computed by induction */

/*      Initialize and wrapup for induction analysis performed     */
/*      by the optimizer on a function or subprogram unit          */

void
induction_init(void)
{

  OPT_ALLOC(induc.indb, IND, 100); /* induction table */
  induc.mark_du = NULL; /* ind-related du's reused for loops in func */

}

void
induction_end(void)
{

  OPT_FREE(induc.indb);

}

/*     Induction Analysis Controller                               */
/*     The following steps occur:                                  */
/*       1.  ind_loop_init - initialize for analyzing a loop       */
/*           potential induction variables are found in the loop   */
/*       2.  find_bivs - determines which of the potential ind.    */
/*           variables are basic induction variables               */
/*       3.  new_ivs - checks the uses of the induction var's      */
/*           and creates new induction variables                   */

void
induction(int lp)
{
  (void)get_loop_count(lp);

  if (OPTDBG(9, 2048)) {
    fprintf(gbl.dbgfil, "\n* * *  loop count %d  * * *\n", loop_cnt);
    if (loop_cnt)
      dbg_print_ast(loop_cnt, gbl.dbgfil);
  }

  end_loop_count();

}

static void
ind_loop_init(void)
{
  /* initialize for induction analysis on the loop -- setup the
   * storage needed and make a pass over all of the stores in the
   * loop including any nested loops
   */
  register STL *stl;
  register int nme, sym, i, ind;

  induc.indb.stg_avail = 1;

  /*
   * scan through the store lists for any nested blocks. The names
   * found cannot be induction variables
   */
  stl = LP_STL(lpx);
  if (stl->childlst != 0)
    removest(stl->childlst);

  /*
   * scan through the store lists for just the loop omitting certain
   * names
   */
  stl = LP_STL(lpx);
  for (i = stl->store; i; i = STORE_NEXT(i)) {
    nme = STORE_NM(i);
    if (NME_TYPE(nme) == NT_VAR && ISSCALAR(sym = NME_SYM(nme)) &&
        DT_ISINT(DTYPEG(sym)) && NME_RFPTR(nme) == 0) {
      /* the basic induction variable's family is itself. this
       * will be propagated to all new induction variables based
       * on the basic induction var.
       */
      ind = add_ind(nme, 0, 0, 0, NULL);
      if (!is_sym_optsafe(nme, lpx)) {
        IND_OMIT(ind) = 1;
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- omit nme %d - sym (%s) unsafe\n", nme,
                  getprint((int)sym));
      } else {
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- potential ind nme %d (%s)\n", nme,
                  getprint(sym));

        /*
         * we can only delete c variables which are automatic or
         * arguments or ftn variables which are local, and other
         * criteria.
         */
        if (XBIT(19, 0x2)) /* #pragma nolstval */
          IND_DELETE(ind) = 1;
        else
          IND_DELETE(ind) = !is_sym_imp_live(nme);
#ifndef ALLOW_COMPLEX
        if (DT_ISCMPLX(DTYPEG(sym)))
          /* TEMPORARY - disallow complex;  NOTE: global flow of
           * complex vars is not done (defs are not created). This
           * test needs to be removed when we allow complex.
           */
          IND_OMIT(ind) = 1;
#endif
      }
    }
  }

  /*
   * mark currently available locations of the use, def, and invariant
   * tables  -- the analysis will create new use and def items, and
   * perhaps invariant ili which are needed only for the loop.
   * The space will be re-used for the next loop.
   */
  mark_useb = opt.useb.stg_avail;
  mark_defb = opt.defb.stg_avail;
  mark_invb = opt.invb.stg_avail;

  lastval_cnt = 0;
  loop_cnt = 0;

  cr_lastval_q = NULL;

}

/*
 * scan through the store lists for any nested blocks. The names found
 * cannot be induction variables removest is passed a list of children to
 * (recursively) mark names as not valid induction variables.
 */
static void
removest(STL *stlist)
{
  STL *tmp;
  register int nme, i, ind;

  for (tmp = stlist; tmp != NULL; tmp = tmp->nextsibl) {
    for (i = tmp->store; i; i = STORE_NEXT(i)) {
      nme = STORE_NM(i);
      if (NME_TYPE(nme) == NT_VAR && ISSCALAR(NME_SYM(nme)) &&
          NME_RFPTR(nme) == 0) {
        ind = add_ind(nme, 0, 0, 0, NULL);
        IND_OMIT(ind) = 1;
        IND_RMST(ind) = 1;
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- omit nme %d (%s) - in nested loop\n", nme,
                  getprint((int)NME_SYM(nme)));
      }
    }
    if (tmp->childlst != 0)
      removest(tmp->childlst);
  }
}

static int
add_ind(int nme, int load, int tnode, int derived, Q_ITEM *astl)
{
  int i;

  i = induc.indb.stg_avail++;
  OPT_NEED(induc.indb, IND, 100);
  IND_NM(i) = nme;
  IND_LOAD(i) = load;
  IND_FLAGS(i) = 0;
  IND_BIVL(i) = NULL;
  if (derived) {
    IND_FAMILY(i) = IND_FAMILY(derived);
    IND_DERIVED(i) = derived;
  } else {
    IND_FAMILY(i) = i;
    IND_DERIVED(i) = 0;
  }
  IND_ASTL(i) = astl;
  IND_OPC(i) = IND_SKIP(i) = 0;
  IND_INITDEF(i) = 0;
  NME_RFPTR(nme) = i;
  return i;
}

/*     step 1.  determine which of the potential induction         */
/*              variables are biv's                                */

static void
find_bivs(void)
{
  register int fgx, def, nme, ind, ilix;
  int store_ili;
  int top;

  for (fgx = LP_FG(lpx); fgx; fgx = FG_NEXT(fgx)) {
    for (def = FG_FDEF(fgx); def; def = DEF_LNEXT(def)) {
      nme = DEF_NM(def);
      ind = NME_RFPTR(nme);
      if (NME_TYPE(nme) != NT_VAR || ind == 0)
        continue;
      if (IND_OMIT(ind))
        continue;
      if (DEF_CONFL(def)) {
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- find_bivs: omit nme %d (%s) - defconfl\n",
                  nme, getprint((int)NME_SYM(nme)));
        IND_OMIT(ind) = 1;
        continue;
      }
      store_ili = STD_AST(DEF_STD(def));
      switch (A_TYPEG(store_ili)) {

      case A_ENDDO:
        ilix = is_biv(ind, def, (int)DEF_LHS(def), (int)DEF_RHS(def));
        if (ilix == 0) {
          /* The rhs is not in the correct form which seems odd
           * for the def derived from a DO/ENDDO.  An example of
           * how this might occur is:
           *     m3 = 0
           *     do i = m1, m2, m3
           *        ...
           * The def for the enddo will be entered as
           *     i = m1
           * i.e., the increment is optimized away.
           */
          IND_OMIT(ind) = 1;
          IND_LOAD(ind) = DEF_LHS(def);
          break;
        }
        assert(ilix == DEF_LHS(def), "find_bivs():enddo load", ilix, 3);
        IND_LOAD(ind) = ilix;
        top = A_OPT2G(store_ili);
        IND_INIT(ind) = A_M1G(top);
        IND_PTR(ind) = 0;
        break;

      case A_ASN:
        ilix = is_biv(ind, def, (int)DEF_LHS(def), (int)DEF_RHS(def));
        if (ilix == 0) {
          IND_OMIT(ind) = 1;
          IND_LOAD(ind) = DEF_LHS(def);
        } else if (IND_LOAD(ind) == 0) {
          IND_INIT(ind) = IND_LOAD(ind) = ilix;
        } else if (IND_LOAD(ind) != ilix)
          IND_OMIT(ind) = 1;
        IND_PTR(ind) = 0;
        break;

      default:
        IND_OMIT(ind) = 1;
        break;
      }
      if (OPTDBG(9, 2048)) {
        if (IND_OMIT(ind))
          fprintf(gbl.dbgfil, "--- find_bivs: omit nme %d (%s)\n", nme,
                  getprint((int)NME_SYM(nme)));
        else
          fprintf(gbl.dbgfil, "--- find_bivs: ind nme %d (%s) is a biv\n", nme,
                  getprint((int)NME_SYM(nme)));
      }
    }
  }

  {
    /*
     * for those induction variables which are deletable due to their
     * storage class, determine if any of them are "live-out" simply due
     * to control flow -- this is for the case where the value of the
     * variable upon exit from the loop is used (later) upon entry to the
     * loop. Typically, live-in/live-out analysis is done to solve this
     * problem. Here, the solution is to check the OUT sets of the
     * predecessors not in the loop of the loop head for the induction
     * variables which are deletable.
     *
     * Also, for each basic induction variable, determine if there's
     * a single definition of the induction variable which reaches the
     * bottom of the predecessor of the loop head.  If this definition's
     * storing value is a constant and it occurs in a block which dominates
     * the predecessor, then we can use the constant as the induction
     * variable's initial value.
     * Otherwise, we'll just use its default (i.e., the load of
     * the variable).  NOTE that we don't care if the induction variable
     * cannot be deleted; this relies on the correctness of the reaching
     * def information (i.e., a block's KILL set is computed correctly
     * and call defs and ptr store defs are tracked correctly).
     */
    PSI_P pred;
    Q_ITEM *p;
    int rdef;

    for (ind = 1; ind < induc.indb.stg_avail; ind++) {
      if (IND_OMIT(ind))
        continue;
      p = IND_BIVL(ind);
      if (p == NULL) {
        IND_OMIT(ind) = 1;
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- find_bivs: omit nme %d (%s)\n",
                  (int)IND_NM(ind), getprint((int)NME_SYM(IND_NM(ind))));
        continue;
      }
      if (!IND_DELETE(ind))
        goto chk_reaching;
      for (; p != NULL; p = p->next) {
        def = p->info;
        pred = FG_PRED(LP_HEAD(lpx));
        for (; pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
          fgx = PSI_NODE(pred);
          if (FG_LOOP(fgx) == lpx)
            continue;
          if (def_in_out(fgx, def)) { /* is def in OUT of fgx */
            if (OPTDBG(9, 2048))
              fprintf(gbl.dbgfil, "--- def %d cannot be deleted (in OUT(%d))\n",
                      def, fgx);
            IND_DELETE(ind) = 0;
            goto chk_reaching;
          }
        }
      }
    chk_reaching:
      /*
       * First, find the loop head's predecessor; if we find more than
       * one, give up.
       */
      fgx = pred_of_loop(lpx);
      if (fgx == 0)
        continue;
      /*
       * find a single reaching definition of the induction variable
       * which reaches the end of the loop's predecessor.
       */
      nme = IND_NM(ind);
      rdef = find_rdef(nme, fgx, FALSE); /* FALSE ==> end of block */

      if (rdef && can_copy_def(rdef, fgx, FALSE)) {
        /*
         * it's safe to use the def's value as the init expression for
         * the induction variable.  If the def's value is a cse,
         * the cse is removed (use its operand) -- may want to
         * prune_cse.
         */
        int temp_ili;
        temp_ili = DEF_RHS(rdef);
        IND_INIT(ind) = temp_ili;
        IND_INITDEF(ind) = rdef;
        if (OPTDBG(9, 2048))
          fprintf(gbl.dbgfil, "--- single init %d for %s\n", (int)IND_INIT(ind),
                  getprint((int)NME_SYM(nme)));
      }
    }
  }

  induc.last_biv = induc.indb.stg_avail - 1;

  if (OPTDBG(9, 4096))
    dump_ind();
}

/* must recursively search because some blocks are added after flow
 * analysis and do not have OUT's defined.
 */
static int
def_in_out(int fgx, int def)
{
  int ret;
  int tempfgx;
  PSI_P pred;

  if (FG_OUT(fgx) != NULL) {
    return bv_mem(FG_OUT(fgx), def);
  }
  /* else we are dealing with optimizer created block */
  /* return true if def is found in any of this block's predecessors */
  FG_VISITED(fgx) = 1; /* to prevent looping */
  ret = 0;
  pred = FG_PRED(fgx);
  for (; pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
    tempfgx = PSI_NODE(pred);
    if (FG_LOOP(tempfgx) == lpx)
      continue;
    if (FG_VISITED(tempfgx) == 1)
      continue;
    if (def_in_out(tempfgx, def)) { /* recurse */
      ret = 1;
      break;
    }
  }
  FG_VISITED(fgx) = 0;
  return ret;
}

static int
is_biv(int ind, int def, int lhs, int rhs)
{
  register int load, add;
  register Q_ITEM *p;

  add = rhs;
  if (A_TYPEG(add) == A_BINOP && A_OPTYPEG(add) == OP_ADD) {
    load = A_LOPG(add);
    if (load == lhs) {
      if (DEF_DOEND(def) || IS_INVARIANT((int)A_ROPG(add))) {
        p = add_biv_def(ind, def);
        p->flag = (2 << BIVL_OPN_SHF) | '+';
        check_skip(ind, p);
        return (load);
      }
    } else {
      load = A_ROPG(add);
      if (load == lhs) {
        if (DEF_DOEND(def) || IS_INVARIANT((int)A_LOPG(add))) {
          p = add_biv_def(ind, def);
          p->flag = (1 << BIVL_OPN_SHF) | '+';
          check_skip(ind, p);
          return (load);
        }
      }
    }
  } else if (A_TYPEG(add) == A_BINOP && A_OPTYPEG(add) == OP_SUB) {
    load = A_LOPG(add);
    if (load == lhs) {
      if (DEF_DOEND(def) || IS_INVARIANT((int)A_ROPG(add))) {
        p = add_biv_def(ind, def);
        p->flag = (2 << BIVL_OPN_SHF) | '-';
        check_skip(ind, p);
        return (load);
      }
    }
  }
  return (0);
}

static Q_ITEM *
add_biv_def(int ind, int def)
{
  Q_ITEM *p;
  p = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
  p->info = def;
  p->next = IND_BIVL(ind);
  IND_BIVL(ind) = p;
  return p;
}

/*
 * for an induction definition just added, check its skip against what's
 * already there (if any)
 */
static void
check_skip(int ind, Q_ITEM *bivl)
{
  int def;
  int value, opc, skip;

  if (bivl->next)
    IND_MIDF(ind) = 1;
  def = bivl->info;
  value = DEF_RHS(def);
  if (((bivl->flag >> BIVL_OPN_SHF) & BIVL_OPN_MSK) == 1)
    skip = A_LOPG(value);
  else
    skip = A_ROPG(value);
  opc = A_OPTYPEG(value);
  if (bivl->next) {
    if (skip != IND_SKIP(ind) || opc != IND_OPC(ind))
      IND_SKIP(ind) = 0;
  } else {
    IND_SKIP(ind) = skip;
    IND_OPC(ind) = opc;
  }

}

/*     step 2.  check the uses of each biv                         */
/*     Scan the induction variables for their induction            */
/*     uses.  This step will create new induction variables        */

static void
new_ivs(void)
{
  register int ind;

  for (ind = 1; ind < induc.indb.stg_avail; ind++) {
    if (IND_OMIT(ind))
      continue;

    IND_USEL(ind) = srch.last_use = srch.usel =
        (IUSE *)getitem(IUSE_AREA, sizeof(IUSE));
    srch.last_use->use = 0;
    srch.last_use->next = NULL;
    srch.branch.ili = 0;

    if (OPTDBG(9, 2048))
      fprintf(gbl.dbgfil, "--- new_ivs: biv use trace of ind %d \"%s\"\n", ind,
              getprint((int)NME_SYM(IND_NM(ind))));
    scan_ind_uses(ind);
  }

  if (OPTDBG(9, 4096))
    dump_ind();
}

static void
scan_ind_uses(int ind)
{
  /*
   * the first thing to do is to scan all of the uses of the
   * induction definitions of the current induction variable
   * marking those which are induction expressions
   */
  Q_ITEM *p;
  ADU *adu;

  srch.ind = ind;

  srch.lastadu = srch.adu_queue = NULL;
  /*
   * process the uses of the definitions of the basic induction variable;
   * this will discover aliases which will be added to the 'adu' queue.
   */
  for (p = IND_BIVL(ind); p != NULL; p = p->next)
    scan_def((int)p->info, ind);
  /*
   * process any aliases found for this definition; 'aliases of aliases'
   * are discovered during this process, in which case the queue is
   * extended.
   */
  for (adu = srch.adu_queue; adu != NULL; adu = adu->next)
    scan_def(adu->def, adu->ind);
  /*
   * for all of the aliases found for the induction variable:
   * 1.  ensure that the delete flag is correct (if set, the alias
   *     definition can be deleted);
   * 2.  compute the initial value of the derived induction variables.
   */
  for (adu = srch.adu_queue; adu != NULL; adu = adu->next) {
    int drv;
    int ii;

    ii = adu->ind;
    drv = IND_DERIVED(ii);
    if (IND_NIU(drv))
      IND_DELETE(ii) = 0;
    ast_visit(1, 1);
    ast_replace((int)IND_LOAD(drv), (int)IND_INIT(drv));
    IND_INIT(ii) = ast_rewrite((int)DEF_RHS(adu->def));
    ast_unvisit();
  }

  /*
   * if a branch use was found in find_fam, it is determined if it can be
   * turned into a check of an iteration count (i.e. a countable loop)
   */
  if (srch.branch.ili && srch.branch.ind == ind)
    do_branch();

}

static void
scan_def(int def, int ind)
{
  DU *du;
  int use;
  int lpuse;

  srch.ind = ind;
  srch.def = def;
  srch.biv_ili = STD_AST(DEF_STD(def));
  if (OPTDBG(9, 2048))
    fprintf(gbl.dbgfil, "    def %d\n", def);
  for (du = DEF_DU(def); du != NULL; du = du->next) {
    srch.use = use = du->use;
    if (STD_DELETE(USE_STD(use))) {
      if (OPTDBG(9, 2048))
        fprintf(gbl.dbgfil, "       use %d deleted\n", use);
      goto next_du;
    }
    if ((lpuse = FG_LOOP(USE_FG(use))) != lpx) {
      lpuse = LP_PARENT(lpuse);
      for (; lpuse; lpuse = LP_PARENT(lpuse)) {
        if (lpuse == lpx) {
          IND_NIU(ind) = 1;
          if (OPTDBG(9, 2048))
            fprintf(gbl.dbgfil, "       use %d in contained loop %d\n", use,
                    lpuse);
          goto next_du;
        }
      }
      IND_DELETE(ind) = 0;
      if (OPTDBG(9, 2048))
        fprintf(gbl.dbgfil, "       use %d not in loop %d\n", use, lpuse);
    } else if (USE_DOINIT(use)) {
      /* The use was created when a doinit def was created.  Its 'ast'
       * actually appears in the next (FG_LNEXT) flowgraph node. This
       * node is not a member of the loop region denoted by lpx;
       * consequently, this use is skipped.
       */
      ;
    } else {
      srch.load = USE_AST(use);
      if (OPTDBG(9, 2048))
        fprintf(gbl.dbgfil, "       use %d, load %d \"%s\"\n", use, srch.load,
                astb.atypes[A_TYPEG(srch.load)]);
      srch.new_skip_opc = IND_OPC(ind);
      srch.new_skip = IND_SKIP(ind);
      (void)find_fam((int)STD_AST(USE_STD(use)));
    }
  next_du:;
  }
#if DEBUG
  if (OPTDBG(9, 2048))
    fprintf(gbl.dbgfil, "       def %d, non-biv uses: %d\n", def, IND_NIU(ind));
#endif
}

static LOGICAL _fam(int ilix, int *fm_p);

/*
 * recurse thru an ili tree searching for the "load" of the current
 * induction variable.  Find all induction uses of the induction variable
 * and create new induction variables (i.e., the current induction variable's
 * family).  Note that find_fam also is designed to work when we're
 * processing a new induction variable (i.e., the "load" is actually some
 * linear expression which was previously "replaced").
 *
 * This function returns 0 (not an induction use), or the induction table
 * entry of the induction variable replacing the current ili (ilix).
 */
static int
find_fam(int ilix)
{
  int fm;

  fm = 0;
  ast_traverse_all(ilix, _fam, NULL, &fm);
  return fm;
}

static LOGICAL
_fam(int ilix, int *fm_p)
{
  int opc, i;
  int i1, i2;
  int asd;
  int subflg[7];

  if (ilix == srch.load) {
    srch.new_skip = IND_SKIP(srch.ind);
    srch.new_skip_opc = IND_OPC(srch.ind);
    *fm_p = srch.ind;
    return TRUE;
  }

  i1 = i2 = 0;
  switch (opc = A_TYPEG(ilix)) {

  case A_DO:
    /* don't search A_DOVAR */
    ast_traverse_all((int)A_M1G(ilix), _fam, NULL, &i1);
    if (i1)
      goto not_iuse;
    i1 = 0;
    ast_traverse_all((int)A_M2G(ilix), _fam, NULL, &i1);
    if (i1)
      goto not_iuse;
    if (A_M3G(ilix)) {
      i1 = 0;
      ast_traverse_all((int)A_M3G(ilix), _fam, NULL, &i1);
      if (i1)
        goto not_iuse;
    }
    if (A_M4G(ilix)) {
      i1 = 0;
      ast_traverse_all((int)A_M4G(ilix), _fam, NULL, &i1);
      if (i1)
        goto not_iuse;
    }
    break;
  case A_ENDDO:
    if (OPTDBG(9, 2048))
      fprintf(gbl.dbgfil, " :  enddo %d\n", ilix);
    srch.branch.ind = srch.ind;
    srch.branch.ili = ilix;
    srch.branch.opn = 0;
    break;

  case A_BINOP:
    switch (A_OPTYPEG(ilix)) {
    case OP_ADD:
      ast_traverse_all((int)A_LOPG(ilix), _fam, NULL, &i1);
      if (i1) {
        int s1, o1, s2, o2, d;
        if (IS_INVARIANT(A_ROPG(ilix)))
          goto is_iuse;
        s1 = srch.new_skip;
        o1 = srch.new_skip_opc;
        ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i2);
        if (i2) {
          s2 = srch.new_skip;
          o2 = srch.new_skip_opc;
          d = A_DTYPEG(ilix);
          /* must add the two skips together */
          if (o1 == OP_ADD && o2 == OP_ADD) {
            srch.new_skip = mk_binop(OP_ADD, s1, s2, d);
            srch.new_skip_opc = OP_ADD;
          } else if (o1 == OP_ADD && o2 == OP_SUB) {
            srch.new_skip = mk_binop(OP_SUB, s1, s2, d);
            srch.new_skip_opc = OP_ADD;
          } else if (o1 == OP_SUB && o2 == OP_ADD) {
            srch.new_skip = mk_binop(OP_SUB, s2, s1, d);
            srch.new_skip_opc = OP_ADD;
          } else if (o1 == OP_SUB && o2 == OP_SUB) {
            srch.new_skip = mk_binop(OP_ADD, s1, s2, d);
            srch.new_skip_opc = OP_SUB;
          } else {
            goto not_iuse;
          }
          goto is_iuse;
        }
        goto not_iuse;
      }
      i1 = 0;
      ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i1);
      if (i1) {
        if (IS_INVARIANT(A_LOPG(ilix)))
          goto is_iuse;
        goto not_iuse;
      }
      break;
    case OP_SUB:
      ast_traverse_all((int)A_LOPG(ilix), _fam, NULL, &i1);
      if (i1) {
        int s1, o1, s2, o2, d;
        if (IS_INVARIANT(A_ROPG(ilix)))
          goto is_iuse;
        s1 = srch.new_skip;
        o1 = srch.new_skip_opc;
        ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i2);
        if (i2) {
          s2 = srch.new_skip;
          o2 = srch.new_skip_opc;
          d = A_DTYPEG(ilix);
          /* must add the two skips together */
          if (o1 == OP_ADD && o2 == OP_ADD) {
            srch.new_skip = mk_binop(OP_SUB, s1, s2, d);
            srch.new_skip_opc = OP_ADD;
          } else if (o1 == OP_ADD && o2 == OP_SUB) {
            srch.new_skip = mk_binop(OP_ADD, s1, s2, d);
            srch.new_skip_opc = OP_ADD;
          } else if (o1 == OP_SUB && o2 == OP_ADD) {
            srch.new_skip = mk_binop(OP_ADD, s1, s2, d);
            srch.new_skip_opc = OP_SUB;
          } else if (o1 == OP_SUB && o2 == OP_SUB) {
            srch.new_skip = mk_binop(OP_SUB, s2, s1, d);
            srch.new_skip_opc = OP_ADD;
          } else {
            goto not_iuse;
          }
          goto is_iuse;
        }
        goto not_iuse;
      }
      i1 = 0;
      ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i1);
      if (i1) {
        if (IS_INVARIANT(A_LOPG(ilix))) {
          if (srch.new_skip)
            srch.new_skip_opc = srch.new_skip_opc == OP_ADD ? OP_SUB : OP_ADD;
          goto is_iuse;
        }
        goto not_iuse;
      }
      break;
    case OP_MUL:
      ast_traverse_all((int)A_LOPG(ilix), _fam, NULL, &i1);
      if (i1) {
        if (IS_INVARIANT(A_ROPG(ilix))) {
          if (srch.new_skip)
            srch.new_skip = mk_binop(OP_MUL, srch.new_skip, (int)A_ROPG(ilix),
                                     (int)A_DTYPEG(ilix));
          goto is_iuse;
        }
        ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i2);
        if (i2) {
          if (srch.new_skip)
            srch.new_skip = mk_binop(OP_MUL, srch.new_skip, srch.new_skip,
                                     (int)A_DTYPEG(srch.new_skip));
          goto is_iuse;
        }
        goto not_iuse;
      }
      i1 = 0;
      ast_traverse_all((int)A_ROPG(ilix), _fam, NULL, &i1);
      if (i1) {
        if (IS_INVARIANT(A_LOPG(ilix))) {
          if (srch.new_skip)
            srch.new_skip = mk_binop(OP_MUL, srch.new_skip, (int)A_LOPG(ilix),
                                     (int)A_DTYPEG(ilix));
          goto is_iuse;
        }
        goto not_iuse;
      }
      break;
    default:
      goto check_operands;
    }
    break;

  case A_UNOP:
    if (A_OPTYPEG(ilix) == OP_SUB) {
      ast_traverse_all((int)A_LOPG(ilix), _fam, NULL, &i1);
      if (i1) {
        srch.new_skip_opc = srch.new_skip_opc == OP_ADD ? OP_SUB : OP_ADD;
        goto is_iuse;
      }
    }
    break;

  case A_SUBSCR:
    ast_traverse_all((int)A_LOPG(ilix), _fam, NULL, &i1);
    asd = A_ASDG(ilix);
    for (i = 0; i < (int)ASD_NDIM(asd); i++) {
      i2 = 0;
      ast_traverse_all((int)ASD_SUBS(asd, i), _fam, NULL, &i2);
      if (i2)
        subflg[i] = -1;
      else
        subflg[i] = IS_INVARIANT(ASD_SUBS(asd, i));
    }
    i1 = 0; /* other subscripts variant */
    i2 = 0; /* any induction uses */
    for (i = 0; i < (int)ASD_NDIM(asd); i++) {
      if (subflg[i] < 0)
        i2++;
      else if (!subflg[i])
        i1++;
    }
    if (i2 && i1)
      goto not_iuse;
    break;

  case A_ASN:
    if (is_biv_store(srch.ind, ilix)) {
      if (OPTDBG(9, 2048))
        fprintf(gbl.dbgfil, "   -- biv store %d^\n", ilix);
      break;
    }

    ast_traverse_all((int)A_DESTG(ilix), _fam, NULL, &i1);
    ast_traverse_all((int)A_SRCG(ilix), _fam, NULL, &i1);
    if (i1) {
      /*
       * if the value of an induction variable (basic or derived)
       * is stored, check if this is an alias candidate if the stored
       * iv is the current iv.  Otherwise, create a use item for
       * the expression and add it to the du chain of the (derived)
       * induction variable --- note that this defers the alias check
       * until the current iv is this induction variable.
       */
      if (check_alias(ilix))
        break;
      IND_NIU(i1) = 1;
      if (OPTDBG(9, 2048))
        fprintf(gbl.dbgfil, " :  other stuse %d^(%s), ind %d\n", ilix,
                astb.atypes[A_TYPEG(ilix)], i1);
    }
    break;

  default:
  check_operands:
    /* check the operands of the ast */
    i1 = 0;
    ast_trav_recurse(ilix, &i1);
    if (i1)
      goto not_iuse;
    break;
  }
  return TRUE; /* stop the traverse */

is_iuse:
  *fm_p = srch.ind;
  return TRUE; /* stop the traverse */

not_iuse:
  IND_NIU(srch.ind) = 1;
  if (OPTDBG(9, 2048))
    fprintf(gbl.dbgfil, " :  other use %d^(%s), ind %d\n", ilix,
            astb.atypes[A_TYPEG(ilix)], i1);
  return TRUE; /* stop the traverse */
}

#ifdef FLANG_INDUC_UNUSED
static DU *
get_du(int nme, int fgx, int iltx, int ilix, int cse)
{
  DU *du;
  int i;
  i = opt.useb.stg_avail++;
  OPT_NEED(opt.useb, USE, 100);
  USE_NM(i) = nme;
  USE_FG(i) = fgx;
  USE_STD(i) = iltx;
  USE_AST(i) = ilix;
  USE_CSE(i) = cse;
  GET_DU(du);
  du->use = i;
  return du;
}
#endif

#ifdef FLANG_INDUC_UNUSED
static void
add_du(DU *du, int ind)
{
  Q_ITEM *p;
  int def;
  int use;
  DU *dux;
  int searched;
  int usex;

  use = du->use;
  searched = 0;
  for (p = IND_BIVL(ind); p != NULL; p = p->next) {
    def = p->info;
    if (!searched) {
      for (dux = DEF_DU(def); dux != NULL; dux = dux->next) {
        usex = dux->use;
        if (USE_AST(usex) == USE_AST(use) && USE_STD(usex) == USE_STD(use) &&
            USE_CSE(usex) == USE_CSE(use))
          return;
      }
      searched = 1;
    }
    du->next = DEF_DU(def);
    DEF_DU(def) = du;
  }
}
#endif

#ifdef FLANG_INDUC_UNUSED
static int
add_def(int nme, int fgx, int iltx, DU *csel, DU *du)
{
  int i;

  i = opt.defb.stg_avail++;
  OPT_NEED(opt.defb, DEF, 100);
  DEF_NEXT(i) = NME_DEF(nme);
  DEF_FG(i) = fgx;
  DEF_STD(i) = iltx;
  DEF_NM(i) = nme;
  DEF_ALL(i) = 0;
  DEF_DU(i) = du;
  DEF_CSEL(i) = csel;
  NME_DEF(nme) = i;
  return i;
}
#endif

static void
do_branch(void)
{
  /*
   * if a branch use was found in find_fam, it is determined if it can be
   * turned into a check of an iteration count (i.e. a countable loop); the
   * conditions are:
   * 1. loop cannot contain calls (determined by find_fam).
   * 2. the branch is the last use in the tail (determined by find_fam).
   * 3. all induction variables are of the form
   *        i = i + c  OR  i = i - c,
   *    where c is a constant and is the same value in each definition.
   * 4. for each induction definition, the flow graph node containing the
   *    definition dominates the tail of the loop.
   * 5. the comparison involved is one of >=, >, <=, or <.
   *
   * NOTE that an additional optimization is to use the iteration count
   * in a target's "loop count" instruction.
   */
  int astx;
  int ind;

  ind = srch.branch.ind;
  /*
   * don't use loop instruction if this is a loop which has multiple
   * tails.
   */
  if (EDGE_NEXT(LP_EDGE(lpx)) >= 0)
    goto br_not_optz;
  /*
   * allow loop instruction if there are any non-induction uses unless
   * this restriction is requested.
   */
  if (XBIT(8, 0x80) && IND_NIU(ind))
    goto br_not_optz;

  /* look at the type of the branch */

  if (A_TYPEG(srch.branch.ili) == A_ENDDO) {
    int tt;
    astx = A_OPT2G(srch.branch.ili);
#if DEBUG
    assert(A_TYPEG(astx) == A_DO, "do_branch:mismatched do-enddo",
           srch.branch.ili, 3);
#endif
    opt.cntlp.cnt = conv_to_int(A_M2G(astx));
    tt = conv_to_int(A_M1G(astx));
    opt.cntlp.cnt = mk_binop(OP_SUB, opt.cntlp.cnt, tt, DT_INT);
    if ((tt = A_M3G(astx))) {
      tt = conv_to_int(tt);
      opt.cntlp.cnt = mk_binop(OP_ADD, opt.cntlp.cnt, tt, DT_INT);
      opt.cntlp.cnt = mk_binop(OP_DIV, opt.cntlp.cnt, tt, DT_INT);
    } else
      opt.cntlp.cnt = mk_binop(OP_ADD, opt.cntlp.cnt, astb.i1, DT_INT);
  } else
    goto br_not_optz;

  opt.cntlp.branch = srch.branch.ili;
  induc.branch_ind = ind;

  loop_cnt = opt.cntlp.cnt;
  /*
   * set the iteration count which is used for last value computation.
   * Also, this value indicates that the loop is countable.
   * A countable loop cannot have multiple exits.
   * NOTE:  even though we can't compute last values, it's still
   * possible (and important) to use the loop instruction.
   */
  if (LP_MEXITS(lpx)) {
    if (OPTDBG(9, 2048))
      fprintf(gbl.dbgfil, "---  no lastval loop count, multiple exits\n");
    lastval_cnt = 0;
  } else {
    lastval_cnt = opt.cntlp.cnt;
    if (OPTDBG(9, 2048))
      fprintf(gbl.dbgfil, "---  lastval loop count is ili %d\n", lastval_cnt);
  }

  goto br_complete;

br_not_optz:
  if (OPTDBG(9, 2048))
    fprintf(gbl.dbgfil, "---  branch %d(%s) not optimized\n", srch.branch.ili,
            astb.atypes[A_TYPEG(srch.branch.ili)]);
  opt.cntlp.cnt = 0;
  IND_NIU(ind) = 1;

br_complete:;
}

static int
conv_to_int(int expr)
{
  if (A_TYPEG(expr) == A_CONV)
    expr = ast_intr(I_INT, DT_INT, 1, A_LOPG(expr));
  return expr;
}

static LOGICAL
is_biv_store(int ind, int ilix)
{
  Q_ITEM *p;

  for (p = IND_BIVL(ind); p != NULL; p = p->next)
    if (ilix == STD_AST(DEF_STD(p->info)))
      return TRUE;
  return FALSE;
}

/* a store of an induction expression has been found. determine if the stored
 * variable can be treated as an alias for the induction expression.
 */
static LOGICAL
check_alias(int store)
{
  int nme, sym;
  int ind;
  int def;
  int i;
  int nuses;
  DU *du;
  ADU *adu;
  Q_ITEM *p;
  int use;
  int lpuse;

  nme = A_NMEG(A_DESTG(store));
  if (NME_TYPE(nme) != NT_VAR || !ISSCALAR(sym = NME_SYM(nme)) ||
      !DT_ISINT(DTYPEG(sym)) || (ind = NME_RFPTR(nme)) == 0 || IND_RMST(ind) ||
      DT_ISCMPLX(DTYPEG(sym)) ||
      !is_sym_optsafe(nme, lpx))
    return FALSE;

  if (IND_ALIAS(ind)) /* it's already an alias */
    return TRUE;

  /* Only one def of sym allowed in loop */

  def = 0;
  for (i = NME_DEF(nme); i; i = DEF_NEXT(i)) {
    if (FG_LOOP(DEF_FG(i)) == lpx) {
      if (def)
        return FALSE;
      def = i;
    }
  }
#if DEBUG
  assert(STD_AST(DEF_STD(def)) == store, "check_alias: def bad", def, 3);
#endif
  if (DEF_DELETE(def) && DEF_DU(def) == NULL) {
    /* can get here if delete_store (flow) deleted a def which does not
     * have any uses.  If the def contains a use of the induction variable,
     * it still shows up in the induction variable's du.  If we allow the
     * processing to continue, a "bug" could occur since we'll end up
     * deleting a def that has already been deleted (its ilt is removed
     * via unlnkilt more than once).
     */
    if (OPTDBG(9, 2048))
      fprintf(gbl.dbgfil, "      alias use, store %d^, %s, already deleted\n",
              store, SYMNAME(sym));
    return TRUE;
  }
  /*
   * Scan all uses of the def. For those uses in the loop, check if this is
   * the only definition that reaches the uses along with other criteria
   * (such as if the def contains any external references, if it is defined
   * upon entry and the use can be reached from the entry, etc.).
   */
  nuses = 0; /* number of uses not found in the loop or loop nest */
  for (du = DEF_DU(def); du != NULL; du = du->next) {
    use = du->use;
    if ((lpuse = FG_LOOP(USE_FG(use))) != lpx) {
      lpuse = LP_PARENT(lpuse);
      for (; lpuse; lpuse = LP_PARENT(lpuse)) {
        if (lpuse == lpx) {
          IND_NIU(ind) = 1;
          goto next_du;
        }
      }
      nuses++;
    } else if (!single_ud(use)) {
      return FALSE;
    }
  next_du:;
  }

  if (OPTDBG(9, 2048))
    fprintf(gbl.dbgfil, "      alias use, store %d^, %s, # out uses %d\n",
            store, SYMNAME(sym), nuses);
  /*
   * create a new adu record and initialize its fields; it needs to be added
   * to the end of the queue (i.e., it's a queue).
   */
  adu = (ADU *)getitem(Q_AREA, sizeof(ADU));
  adu->def = def;
  adu->ind = ind;
  adu->next = NULL;
  if (srch.adu_queue == NULL)
    srch.adu_queue = adu;
  else
    srch.lastadu->next = adu;
  srch.lastadu = adu;

  IND_ALIAS(ind) = 1;
  IND_INIT(ind) = DEF_RHS(def);
  IND_FAMILY(ind) = IND_FAMILY(srch.ind);
  IND_SKIP(ind) = srch.new_skip;
  IND_OPC(ind) = srch.new_skip_opc;
  IND_DERIVED(ind) = srch.ind;
  p = add_biv_def(ind, def); /* create a biv item so its def is recorded */
  p->flag = 'a';

  /* currently, we don't try to compute an alias' last value. */
  if (nuses || ADDRTKNG(sym) || is_sym_exit_live(nme)) {
    IND_DELETE(ind) = 0;
    IND_NIU(srch.ind) = 1;
  } else
    IND_DELETE(ind) = 1;

  return TRUE;
}

#ifdef FLANG_INDUC_UNUSED
/*     Compute last values for induction variables.                */
static void
last_values(void)
{
  extern void compute_last_values(int exit, int prehdr);

  if (XBIT(8, 0x40) || !LP_INNERMOST(lpx)) {
    lastval_cnt = 0;
    return;
  }

  cr_lastval_uses = TRUE;
  compute_last_values((int)FG_TO_BIH(opt.exit_fg), (int)FG_TO_BIH(opt.pre_fg));
  cr_lastval_uses = FALSE;

}
#endif

static void
dump_ind(void)
{
  int i, ilix;
  Q_ITEM *p;

  fprintf(gbl.dbgfil,
          "\n* * *  Induction Table for Loop %d, Function \"%s\"  * * *\n", lpx,
          getprint((int)BIH_LABEL(gbl.entbih)));
  for (i = 1; i < induc.indb.stg_avail; i++) {
    fprintf(gbl.dbgfil, "%5d.  nme: %-5u  init: %-5d  fam: %-5d \"%s\"", i,
            IND_NM(i), IND_INIT(i), IND_FAMILY(i),
            getprint((int)NME_SYM(IND_NM(i))));
    if (IND_PTR(i))
      fprintf(gbl.dbgfil, " <pt>");
    if (IND_DELETE(i))
      fprintf(gbl.dbgfil, " <del>");
    if (IND_NIU(i))
      fprintf(gbl.dbgfil, " <niu>");
    if (IND_GONE(i))
      fprintf(gbl.dbgfil, " <gone>");
    if (IND_ALIAS(i))
      fprintf(gbl.dbgfil, " <alias>");
    if (IND_OMIT(i)) {
      fprintf(gbl.dbgfil, " <omit>");
      if (!IND_ALIAS(i)) {
        fprintf(gbl.dbgfil, "\n");
        continue;
      }
    }
    fprintf(gbl.dbgfil, "\n");
    fprintf(gbl.dbgfil, "        load: %-5d initdef: %-5d derived: %-5d",
            IND_LOAD(i), IND_INITDEF(i), IND_DERIVED(i));
    fprintf(gbl.dbgfil, "\n");
    fprintf(gbl.dbgfil, "        opc: %d  skip: %d ", IND_OPC(i), IND_SKIP(i));
    if (IND_SKIP(i)) {
#if DEBUG
      printast(IND_SKIP(i));
#else
      if (A_TYPEG(IND_SKIP(i)) == A_CNST)
        fprintf(gbl.dbgfil, " <%d>", get_int_cval(A_SPTRG(IND_SKIP(i))));
      else
        fprintf(gbl.dbgfil, " <nconst>");
#endif
    } else
      fprintf(gbl.dbgfil, "<diff skips>");
    if (IND_MIDF(i))
      fprintf(gbl.dbgfil, " <midf>");
    fprintf(gbl.dbgfil, "\n");
    for (p = IND_BIVL(i); p != NULL; p = p->next) {
      fprintf(gbl.dbgfil, "        def: %-5d  incr: <idt %d, op# %d, %c>\n",
              p->info, (p->flag >> BIVL_IDT_SHF) & BIVL_IDT_MSK,
              (p->flag >> BIVL_OPN_SHF) & BIVL_OPN_MSK,
              (p->flag) & BIVL_OPC_MSK);
    }
    fprintf(gbl.dbgfil, "        repl:");
    for (p = IND_ASTL(i); p != NULL; p = p->next) {
      ilix = p->info;
      fprintf(gbl.dbgfil, " %d", ilix);
    }
    fprintf(gbl.dbgfil, "\n");
  }
  fflush(gbl.dbgfil);

}

/*           ALTERNATE ENTRIES FOR INDUCTION ANALYSIS              */
/*     Determine if loop is a countable loop.  Perform induction   */
/*     analysis but do not perform any "code" generation           */

int
get_loop_count(int lp)
{
  if (OPTDBG(9, 2048))
    fprintf(
        gbl.dbgfil,
        "\n* * *  get_loop_count Trace for Loop %d, Function \"%s\"  * * *\n",
        lp, getprint((int)BIH_LABEL(gbl.entbih)));

  lpx = lp;
  if (XBIT(42, 4))
    call_in_loop = FALSE;
  else
    call_in_loop = LP_CALLFG(lpx);

  ind_loop_init();

  /* extended range loops are not countable. Check must be performed
   * after ind_loop_init() so the end_loop_count() functions correctly.
   */
  if (!XBIT(19, 0x40000) && LP_XTNDRNG(lp))
    return 0;

  find_bivs();

  new_ivs();

  return (loop_cnt);
}

/*     Compute last values for induction variables.                */

/*
 * there are cases when exit and prehdr are the same block (i.e., when
 * the vectorizer calls compute_last_values).  When this occurs, stores
 * of temps are added to the beginning of the block (after the "last"
 * temp store) and last values are added to the end of the block.
 */
/* block to which stores of induction vars are added */
/* block to which stores of temps are added */
void
compute_last_values(int exit, int prehdr)
{
}

/*     do any cleanup after completion of induction analysis       */
/*     this is the same as last_step but without the               */
/*     assignments.                                                */

void
end_loop_count(void)
{
  register int i;

  /*
   * cleanup the rfptr fields of the names entries of the induction
   * variables -- this must be done here since the rfptr field is used for
   * locating register candidates of names entries.
   */
  for (i = 1; i < induc.indb.stg_avail; i++)
    NME_RFPTR(IND_NM(i)) = 0;

  /* cleanup any temp definitions created  */

  for (i = mark_defb; i < opt.defb.stg_avail; i++)
    NME_DEF(DEF_NM(i)) = 0;

  opt.useb.stg_avail = mark_useb;
  opt.defb.stg_avail = mark_defb;

  opt.cntlp.cnt = 0;

  freearea(Q_AREA);
  freearea(IUSE_AREA);

}

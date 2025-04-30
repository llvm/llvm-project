/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  optimize.c - main module of the optimize phase.
    portions of this module are used by the vectorizer.

    void optimize()  -  main controller for the optimizer
    void optshrd_init()  -  init for structures and submodules shared by
        the by optimizer and vectorizer
    void optshrd_finit() -  init for a function before its processed
        by the optimizer or vectorizer.
    void optshrd_fend()  -  cleanup after a function has been processed
        by the optimizer or vectorizer.
    void optshrd_end()  -  cleanup of shared structures and submodules
        shared by the optimizer and vectorizer.

    static void optimize_init()
    static void optimize_end()
    static void function_init()
    static void function_end()
    static void merge_blocks()
    static void loop_init(int)
    extern void add_loop_preheader(int)
    static void add_loop_exit(int)
    static void process_exit(int, int)
    static void replace_label(int, int, int)

*/
#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "machar.h"

#if DEBUG
extern void dmpnme(void);
#endif

extern void open_pragma(int);
extern void close_pragma(void);

static void br_to_br(void);
static void merge_blocks(void);
static void loop_init(int lp);
#ifdef FLANG_OPTIMIZE_UNUSED
static void process_exit(int lp, int s);
#endif
static void replace_label(int bihx, int label, int newlab);

/*   SHARED init and end routines for the vectorizer and optimizer */

/*
 * Initialize and free the data structures that are shared by the
 * vectorizer and optimizer.  These data structures represent the
 * information created by those modules that are shared by the vectorizer
 * and optimizer.  For C, this space is reused across functions in the
 * source file during a given pass (vectorizer and optimizer); for Fortran,
 * the space is allocated for each function.  At the end of each pass, the
 * space is freed.
 *
 * The shared modules are:
 *    flowgraph
 *    findloop
 *    flow
 *    invariant analysis (not code motion).
 *
 * Since it's possible for the vectorizer/optimizer to see more than one
 * function, it's necessary to initialize and free certain data structures
 * that are valid only for the lifetime of a function (for example,
 * various getitem areas).
 *
 * A skeleton of the vectorizer/optimizer for using these routines is:
 *
 *     optshrd_init();
 *     foreach function {
 *         optshrd_finit();
 *         ...
 *         optshrd_fend();
 *     }
 *     optshrd_end();
 *
 */

void
optshrd_init(void)
{

  STG_ALLOC(opt.fgb, 100);   /* flowgraph space */
  opt.fgb.stg_avail = 0;	/* expected starting value */
  gbl.entbih = 1;                /* the first bih */
  STG_ALLOC(opt.rteb, 32); /* retreating edges */

  STG_ALLOC(opt.lpb, 50);   /* loop table */
  LP_PARENT(0) = 0; /* set the parent of region 0 to 0 -- this is
                     * for tests which look at the parent of a
                     * loop without looking at the loop index
                     */

  STG_ALLOC(opt.storeb, 100); /* store area */
  STG_ALLOC(opt.defb, 64); /* definition table */
  STG_ALLOC(opt.useb, 64); /* use lists */

  STG_ALLOC(opt.invb, 100); /* invariant expr area */
  STG_ALLOC_SIDECAR(astb, opt.astb);

  opt.sc = SC_AUTO; /* default storage class for opt-created temps */

  nme_init();

}

void
optshrd_finit(void)
{

}

void
optshrd_fend(void)
{
  flow_end();

  freearea(PSI_AREA);
  freearea(DU_AREA);
  freearea(STL_AREA);

}

void
optshrd_end(void)
{
  STG_DELETE(opt.fgb);
  STG_DELETE(opt.rteb);
  STG_DELETE(opt.lpb);
  STG_DELETE(opt.storeb);
  STG_DELETE(opt.defb);
  STG_DELETE(opt.useb);
  STG_DELETE(opt.invb);
  STG_DELETE_SIDECAR(astb, opt.astb);
  nme_end();

}

/*
 * hlopt_init() & hlopt_end():
 * Initialization & cleanup for 'high level' optimizations (transformations)
 * such as unrolling & invariant if removal.  The optimizations require
 * a flow graph, loop discovery, and flow analysis.  Optional analysis
 * is induction analysis (see HLOPT_... in optimize.h).
 */
void
hlopt_init(int hlopt_bv)
{
  optshrd_init();

  if (hlopt_bv & HLOPT_INDUC)
    induction_init();

  optshrd_finit();

  flowgraph(); /* build the flowgraph for the function */
#if DEBUG
  if (DBGBIT(9, 1))
    dump_flowgraph();
#endif

  findloop(hlopt_bv); /* find the loops */
#if DEBUG
  if (DBGBIT(9, 4)) {
    dump_flowgraph();
    dump_loops();
  }
#endif

  flow(); /* do flow analysis on the loops  */

}

void
hlopt_end(int hlopt_bv, int gbc)
{
  optshrd_fend();

  optshrd_end();

  if (hlopt_bv & HLOPT_INDUC)
    induction_end();

}

static void
optimize_init(void)
{

  /*  initialize for the optimize module  */

  optshrd_init();

  induction_init(); /* induction data areas */

}

static void
optimize_end(void)
{

  /*  free up the space used by the optimize module */

  optshrd_end();

  induction_end();

}

/*  initialize for a function to be optimized  */
static void
function_init(void)
{
  optshrd_finit();
  /* clear temporary used for innermost loop count */

  opt.cntlp.cnt_sym = 0;

  if (DBGBIT(0, 1))
    fprintf(gbl.dbgfil, "***** begin optimizing %s\n",
            getprint((int)gbl.currsub));

}

/*  end for an optimized function  */
static void
function_end(void)
{
  optshrd_fend();

}

/*
 * given a linked list of flow edges;
 * return the linked list with the flow edge to 'r' removed
 */
static PSI_P
remove_flow_list(PSI_P link, int r)
{
  PSI_P head, tail, next;
  head = NULL;
  tail = NULL;
  for (; link; link = next) {
    next = PSI_NEXT(link);
    PSI_NEXT(link) = NULL;
    if (PSI_NODE(link) != r) {
      if (head == NULL) {
        head = link;
      } else {
        PSI_NEXT(tail) = link;
      }
      tail = link;
    }
  }
  return head;
} /* remove_flow_list */

/*
 * remove a loop;
 * remove lpx from the LP_CHILD list of its parent.
 * make the fnodes be simple fall-through fnodes
 * add the fnodes to the fnode list of the parent
 */
static void
remove_loop(int lpx)
{
  int parent, prev, head, tail;
  parent = LP_PARENT(lpx);
  if (LP_CHILD(parent) == lpx) {
    LP_CHILD(parent) = LP_SIBLING(lpx);
    /* does this make 'parent' an innermost loop? */
    if (parent && LP_CHILD(parent) == 0) {
      LP_INNERMOST(parent) = 1;
    }
  } else {
    int lc;
    for (lc = LP_CHILD(parent); lc && LP_SIBLING(lc) != lpx;
         lc = LP_SIBLING(lc))
      ;
    if (lc && LP_SIBLING(lc) == lpx) {
      LP_SIBLING(lc) = LP_SIBLING(lpx);
    }
  }
  head = LP_HEAD(lpx);
  tail = LP_TAIL(lpx);
  FG_PRED(head) = remove_flow_list(FG_PRED(head), tail);
  FG_SUCC(tail) = remove_flow_list(FG_SUCC(tail), head);

  for (prev = FG_LPREV(head); prev; prev = FG_LPREV(prev)) {
    if (FG_LOOP(prev) == parent)
      break;
  }
  if (prev && LP_FG(lpx)) {
    int next, fg;
    next = FG_NEXT(prev);
    FG_NEXT(prev) = LP_FG(lpx);
    for (fg = LP_FG(lpx); fg && FG_NEXT(fg); fg = FG_NEXT(fg))
      ;
    FG_NEXT(fg) = next;
  }
  LP_FG(lpx) = 0;
} /* remove_loop */

/*
 * return '1' if the ast tree has no function calls
 */
static int
no_functions(int ast)
{
  /* ### not written */
  return 1;
} /* no_functions */

void
optimize(int whichpass)
{
  int lpx;
  int i;

  optimize_init();

#if DEBUG
  if (DBGBIT(38, 128)) {
    optimize_end();
    return;
  }
#endif
  if (opt.fgb.stg_avail == 1) {
    optimize_end();
    return;
  }
/* optimize is called for each function */

#if DEBUG
  if (DBGBIT(10, 2)) {
    fprintf(gbl.dbgfil, "STDs before optimizer\n");
    dump_std();
  }
#endif

  function_init();

  flowgraph(); /* build the flowgraph for the function */
#if DEBUG
  if (DBGBIT(9, 1))
    dump_flowgraph();
#endif

  findloop(HLOPT_ALL); /* find the loops */

#if DEBUG
  if (DBGBIT(9, 4)) {
    dump_flowgraph();
    dump_loops();
  }
#endif

  if (XBIT(70, 0x2000)) /* enhanced analysis with HCCSYM variables */
    flg.x[70] |= 0x1000;
  flow(); /* do flow analysis on the loops  */
  flg.x[70] &= ~0x1000;
  if (whichpass == 1) {
    flow_end();
    optimize_end();
    return;
  }

#if DEBUG
  if (DBGBIT(29, 128))
    dump_loops();
#endif

  if (XBIT(0, 0x1000))
    use_before_def(); /* after flow; check for uses before defs */

  delete_stores(); /* find deleteable stores */

  /*
   * do optimizations for each loop.  also, for each loop, mark
   * the blocks which are the head and tail of the loop and mark
   * the head block as innermost if it's the head of an innermost
   * loop (this information is used by the scheduler).
   */
  for (i = 1; i <= opt.nloops; i++) {
    int headfg;

    lpx = LP_LOOP(i);
    headfg = LP_HEAD(lpx);
    gbl.lineno = FG_LINENO(headfg);

    if (LP_PARLOOP(lpx))
      opt.sc = SC_PRIVATE;

    /* apply loop-scoped pragmas/directives available for this loop */
    open_pragma(gbl.lineno);

    if (LP_INNERMOST(lpx)) {
      if (XBIT(70, 0x4000)) {
        int fg, empty, dostd;
        /* remove empty loops */
        empty = 1;
        dostd = 0;
        for (fg = LP_FG(lpx); empty && fg; fg = FG_NEXT(fg)) {
          int std;
          for (std = FG_STDFIRST(fg); empty && std; std = STD_NEXT(std)) {
            switch (A_TYPEG(STD_AST(std))) {
            case A_DO:
              if (dostd == 0) {
                dostd = std;
              } else {
                empty = 0;
              }
              break;
            case A_ENDDO:
            case A_CONTINUE:
              break;
            default:
              empty = 0;
              break;
            }
            if (std == FG_STDLAST(fg))
              break;
          }
        }
        if (empty && dostd) {
          /* if the DO variable is dead, remove all the statements */
          int doast, dovar, donme;
          doast = STD_AST(dostd);
          dovar = A_SPTRG(A_DOVARG(doast));
          donme = add_arrnme(NT_VAR, dovar, 0, (INT)0, 0, FALSE);
          if (!is_live_out(donme, lpx) && no_functions(doast)) {
            /* remove all statements, change all to continue */
            for (fg = LP_FG(lpx); fg; fg = FG_NEXT(fg)) {
              int std;
              for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
                int ast;
                ast = STD_AST(std);
                A_TYPEP(ast, A_CONTINUE);
                if (std == FG_STDLAST(fg))
                  break;
              }
            }
            /* this loop no longer exists.
             * remove the loop from the LP_CHILD list of its parent.
             * if the parent has no other children, mark the parent as INNERMOST
             */
            remove_loop(lpx);
            continue;
          } else {
            /* insert assignment from lower bound, upper bound, stride */
            /* insert proper assignment to DO variable */
            /* ### not done */
          }
        }
      }
      FG_INNERMOST(headfg) = 1;
    }

    if (LP_FG(lpx) == 0) {
      continue;
    }
    FG_HEAD(headfg) = 1;
    FG_MEXITS(headfg) = LP_MEXITS(lpx);
    FG_TAIL(LP_TAIL(lpx)) = 1;
    loop_init(lpx);

    invariant(lpx);

    induction(lpx);

    add_loop_preheader(lpx);
    add_loop_exit(lpx);

    /*  set done bit for all nodes in the loop  */

    close_pragma();

    opt.sc = SC_AUTO;
  }
  if (opt.nloops == 0) {
    ;
  } else {

    /*  do region 0  */

    opt.pre_fg = NEW_NODE(gbl.entbih);
    opt.exit_fg = NEW_NODE(FG_LPREV(opt.exitfg));

    if (OPTDBG(9, 1)) {
      fprintf(gbl.dbgfil, "\n  Loop init for region 0\n");
      fprintf(gbl.dbgfil, "    preheader: %d\n", opt.pre_fg);
      fprintf(gbl.dbgfil, "    exit: %d\n", opt.exit_fg);
    }
  }

  br_to_br();

  merge_blocks(); /* merge the blocks */

  function_end();

#if DEBUG
  if (DBGBIT(10, 4)) {
    fprintf(gbl.dbgfil, "STDs after optimizer\n");
    dump_std();
  }
  if (DBGBIT(10, 64))
    dmpnme();
#endif

  flow_end();
  optimize_end();

}

/*******************************************************************/

#ifdef FLANG_OPTIMIZE_UNUSED
/*
 * attempt to move the label which labels a bih.  The condition for
 * moving the label is if the block contains an unconditional branch.
 * This routine is recursive so that if we have br to br to br, ...,
 * the label labelling last branch in the branch chain is moved first.
 */
static void
move_label(int lab, int bih)
{

}
#endif

static void
br_to_br(void)
{
  if (XBIT(6, 0x8))
    return;
  if (opt.num_nodes < 3)
    return;

}

/*******************************************************************/

static void
merge_blocks(void)
{
  int std;
  int next_std;
  int ast;

  if (XBIT(8, 0x4))
    return;
  for (std = STD_NEXT(0); std; std = next_std) {
    next_std = STD_NEXT(std);
    if (STD_LABEL(std) == 0) {
      ast = STD_AST(std);
      if (A_TYPEG(ast) == A_CONTINUE)
        unlnkilt(std, 0, TRUE);
    }
  }
  if (opt.num_nodes < 3)
    return;

}

/*******************************************************************/

static void
loop_init(int lp)
{
  opt.pre_fg = NEW_NODE(FG_LPREV(LP_HEAD(lp)));
  FG_FT(opt.pre_fg) = 1;

  /* create a block for the loop exit -- code will be added to this
   * block as optimizations are performed.  Note that block is inserted
   * the tail of the loop
   */
  opt.exit_fg = NEW_NODE(LP_TAIL(lp));
  FG_FT(opt.exit_fg) = 1;

  FG_PAR(opt.pre_fg) = LP_PARLOOP(lp);
  FG_PAR(opt.exit_fg) = LP_PARLOOP(lp);
  if (OPTDBG(9, 1)) {
    fprintf(gbl.dbgfil, "\n  Loop init for loop (%d)\n", lp);
    fprintf(gbl.dbgfil, "    preheader: %d, exit: %d\n", opt.pre_fg,
            opt.exit_fg);
  }
}

/*******************************************************************/

/*
 * After a loop has been processed, the flow graph is updated to include
 * the preheader.  This node is physically placed immediatedly before the
 * loop head.
 * The predecessors of head not in the loop are replaced by the preheader
 * flow graph node. The predecessors of this node are the predecessors
 * of head not in the loop.
 * Also, this node replaces head as the successor of all the nodes not
 * in the loop.
 * If there is a path to the head node by branching (the node does not
 * fall thru to the head), the label of head is replaced with a new label
 * and all the branches to the head are modified.  The original label
 * is used to label the preheader.
 * Also, the preheader is added to the region containing the loop.
 */
void
add_loop_preheader(int lp)
{
  int i, v;
  PSI_P p, q, prev;
  int newlabel;

  int head;

  if (FG_STDFIRST(opt.pre_fg) == 0)
    return;
  if (OPTDBG(9, 1))
    fprintf(gbl.dbgfil, "\n  Preheader end for loop (%d)\n", lp);

  /* Every node dominated by lp's dominator will be dominated by the
   * new preheader. */
  head = LP_HEAD(lp); /* loop's head */
  i = FG_DOM(head);   /* dominator of loop's head */
  for (p = FG_SUCC(i); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    v = PSI_NODE(p);
    if (FG_DOM(v) == i)
      FG_DOM(v) = opt.pre_fg;
  }
  FG_DOM(opt.pre_fg) = i;
  FG_DOM(head) = opt.pre_fg;

  i = 0;             /* number of predecessors not in the loop */
  newlabel = 0;      /* non-zero if a pred does fall thru to the head */
  prev = PSI_P_NULL; /* the last predecessor to head that's kept */
  for (p = FG_PRED(head); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    v = PSI_NODE(p);
    if (FG_LOOP(v) != lp) { /* v in pred(head) and not in the loop */
      i++;
      if (FG_LNEXT(v) != opt.pre_fg) {
        /* if pred. doesn't fall thru */
        newlabel = 1;
      }
      /* search v's successor list for the item indicating head */

      for (q = FG_SUCC(v); q != PSI_P_NULL; q = PSI_NEXT(q))
        if (head == PSI_NODE(q))
          break;
      assert(q != PSI_P_NULL, "add_loop_pre:suc(q)nhd", head, 3);

      /* Replace head in succ(v) with the preheader node  */

      PSI_NODE(q) = opt.pre_fg;
      /*
       * Replace v in pred(head) with the preheader node only if this
       * is the first predecessor found which is not in the loop.
       * Otherwise, this predecessor item (p) is deleted.
       */
      if (i == 1) {
        PSI_NODE(p) = opt.pre_fg;
        prev = p;
      } else {
        assert(prev != PSI_P_NULL, "add_loop_pre:prev", head, 3);
        PSI_NEXT(prev) = PSI_NEXT(p);
      }
      /*
       * make v a predecessor of the preheader and make head a
       * successor of the preheader
       */
      (void)add_pred(opt.pre_fg, v);
      q = add_succ(opt.pre_fg, head);
      PSI_FT(q) = 1;
#if DEBUG
      if (OPTDBG(9, 1)) {
        fprintf(gbl.dbgfil, "    old pred of head :\n");
        dump_node(v);
      }
#endif
    } else /* v not in pred(head), remember this predecessor item */
      prev = p;
  }

  if (newlabel) { /* have to create a new label for the loop head */
    int label, newlab;

    label = STD_LABEL(FG_STDFIRST(head)); /* original label */
    newlab = getlab();
    ILIBLKP(newlab, head); /* label the head with the new label */
    STD_LABEL(FG_STDFIRST(head)) = newlab;

    ILIBLKP(label, opt.pre_fg); /* label the preheader with the */
    STD_LABEL(FG_STDFIRST(opt.pre_fg)) = label; /* original label */

    /* find all nodes in the loop which are predecessors of head.
     * the branches in these nodes are changed to access the new
     * label
     */
    for (p = FG_PRED(head); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      v = PSI_NODE(p);
      if (FG_LOOP(v) == lp)
        replace_label((int)FG_TO_BIH(v), label, newlab);
    }
  }
  /*
   * set the loop field of the preheader to an ancestor of lp which
   * contains blocks -- a loop could have been deleted (indicated by a null
   * region)
   */
  for (i = lp; LP_FG(i = LP_PARENT(i)) == 0;)
    ;

  FG_LOOP(opt.pre_fg) = i;

  /* add the preheader to the region for parent of lp  */

  FG_NEXT(opt.pre_fg) = LP_FG(i);
  LP_FG(i) = opt.pre_fg;

  /* link the preheader's stds into the std list */

  {
    int new;
    int old;
    int p;

    /* search for the last std preceding the preheader */
    for (p = FG_LPREV(opt.pre_fg); p; p = FG_LPREV(p))
      if (FG_STDLAST(p))
        break;
    new = FG_STDFIRST(opt.pre_fg);
    old = FG_STDLAST(p);
    STD_NEXT(old) = new;
    STD_PREV(new) = old;

    /* search for the first std following the preheader */
    for (p = FG_LNEXT(opt.pre_fg); p; p = FG_LNEXT(p))
      if (FG_STDFIRST(p))
        break;
    new = FG_STDLAST(opt.pre_fg);
    old = FG_STDFIRST(p);
    STD_NEXT(new) = old;
    STD_PREV(old) = new;
  }

#if DEBUG
  if (OPTDBG(9, 1)) {
    fprintf(gbl.dbgfil, "    preheader:\n");
    dump_node(opt.pre_fg);
    fprintf(gbl.dbgfil, "    head:\n");
    dump_node(head);
  }
  if (OPTDBG(9, 4)) {
    fprintf(gbl.dbgfil, "\n  Region of preheader node %d\n", opt.pre_fg);
    dump_region(i);
  }
#endif
}

/*******************************************************************/

/*
 * After a loop has been processed, the flow graph is updated to include
 * the exit node. A copy of the exit block is added for each exit of the
 * loop. This routine figures out where to add the exit blocks in the
 * bih list.  A block can be added immediately before its exit target
 * if the loop falls through to the target.  If it is not a fall through
 * exit target, the exit block is simply added to the end of the loop
 * and code is added to the end of the block so that it transfers control
 * to the exit target.
 */
void
add_loop_exit(int lp)
{
}

/* Add a loop exit following the tail of loop lp. */
void
add_single_loop_exit(int lp)
{
  int fg, fgSucc, fgNew;
  PSI_P psiSucc;
  int lpSucc;
  int sptrLbl, sptrNewLbl;
  int std;
  int ast;

  assert(lp, "add_loop_exit: no loop", lp, 4);
  assert(!LP_MEXITS(lp), "add_loop_exit: multiple exits", lp, 4);

  fgNew = add_fg(LP_TAIL(lp));
  add_to_parent(fgNew, LP_PARENT(lp));

  for (fg = LP_FG(lp); fg; fg = FG_NEXT(fg)) {
    /* Search for a node with a branch to the exit. */
    for (psiSucc = FG_SUCC(fg); psiSucc; psiSucc = PSI_NEXT(psiSucc)) {
      fgSucc = PSI_NODE(psiSucc);
      lpSucc = FG_LOOP(fgSucc);
      if (!lpSucc)
        break;
      if (lpSucc != lp && LP_PARENT(lpSucc) != lp)
        break;
    }
    if (psiSucc)
      /* ...node with exit branch found. */
      break;
  }
  assert(fg, "add_loop_exit: branch not found", lp, 4);
  FG_FT(fgNew) = PSI_FT(psiSucc);

  /* Add a CONTINUE statement to the new loop trailer. */
  ast = mk_stmt(A_CONTINUE, 0);
  rdilts(fgNew);
  std = add_stmt_after(ast, 0);
  wrilts(fgNew);
  FG_LINENO(fgNew) = STD_LINENO(std) = FG_LINENO(LP_TAIL(lp));

  sptrLbl = FG_LABEL(fgSucc);
  if (!PSI_FT(psiSucc) && sptrLbl) {
    int astlab;
    /* Add a GOTO statement to the new loop trailer. */
    ast = mk_stmt(A_GOTO, 0);
    astlab = mk_label(sptrLbl);
    A_L1P(ast, astlab);
    std = add_stmt_after(ast, std);

    /* Create a new label for fgNew. */
    sptrNewLbl = getlab();
    ILIBLKP(sptrNewLbl, fgNew);
    RFCNTI(sptrNewLbl);
    FG_LABEL(fgNew) = sptrNewLbl;

    /* Revise the branch to go to the new loop trailer. */
    replace_label(FG_TO_BIH(fg), sptrLbl, sptrNewLbl);
  }

  /* Fix up the pred./succ. chains. */
  add_succ(fg, fgNew);     /* succ(fg) = fgNew. */
  add_pred(fgNew, fg);     /* pred(fgNew) = fg. */
  add_succ(fgNew, fgSucc); /* succ(fgNew) = fgSucc. */
  add_pred(fgSucc, fgNew); /* pred(fgSucc) = fgNew. */
  rm_edge(fg, fgSucc);
}

/*******************************************************************/

#ifdef FLANG_OPTIMIZE_UNUSED
/*
 * node s is an exit target of the loop. newtarget_fg represents the
 * new exit target which must be executed before executing s.
 * newtarget_bih is the block header for the new target and has already
 * been inserted into the bih list.
 * s is made a successor of the new target flow graph node.
 * The predecessors of s in the loop are replaced by the new target.
 * These predecessors of s become the predecessors of the new
 * target. Also, the successor lists of these nodes are updated by replacing
 * s with the new target.
 * If the exit in the loop does not fall thru to the new target, then
 * a label is created for the new target and the branch in the exit is
 * modified to access this label.  NOTE that any transfer of control changes
 * from the new target to s have been taken care of by add_loop_exit.
 * Also, the new target is added to the region containing the loop.
 */
static void
process_exit(int lp, int s)
{
}
#endif

/*******************************************************************/

/*
 * Replace all occurrences of the label with a new label.
 */
static void
replace_label(int bihx, int label, int newlab)
{
  int std;
  int ast;

  std = BIH_ILTLAST(bihx);
  ast = STD_AST(std);
  ast_visit(1, 1);
  ast_replace(mk_label(label), mk_label(newlab));
  ast = ast_rewrite(ast);
  ast_unvisit();
  STD_AST(std) = ast;
  A_STDP(ast, std);
}

/*******************************************************************/
/*
 * remove a block of statements;
 * optionally convert the last ELSEIF to IF
 * optionally keep the last ENDIF
 */
static void
remove_block(int std1, int std2, LOGICAL convert_elseif, LOGICAL save_endif)
{
  int stdx, nextstdx, astx;
  for (stdx = std1; stdx; stdx = nextstdx) {
    nextstdx = STD_NEXT(stdx);
    astx = STD_AST(stdx);
    if (stdx == std2 && A_TYPEG(astx) == A_ELSEIF && convert_elseif) {
      A_TYPEP(astx, A_IFTHEN);
      break;
    }
    if (stdx == std2 && A_TYPEG(astx) == A_ENDIF && save_endif) {
      break;
    }
    if (STD_LABEL(stdx)
        || STD_PTA(stdx) || STD_PTASGN(stdx)
            ) {
      /* just convert to continue */
      A_TYPEP(astx, A_CONTINUE);
    } else {
      /* remove altogether */
      delete_stmt(stdx);
    }
    if (stdx == std2)
      break;
  }
} /* remove_block */

/*
 * look for conditional branches with constant conditions.
 * remove the branch, remove unreachable code as well.
 */
void
unconditional_branches(void)
{
  int stdx, nextstdx;
  int astx, nest = 0, stdifx, stdstmtx, sptr, stdelsex, stdendx, astelsex, astendx;
  int condx;
  int *ifnest;
  int ifnestsize;
  ifnestsize = 50;
  NEW(ifnest, int, ifnestsize);
  /* initially, set STD_FG for A_IFTHEN, A_ELSEIF, A_ELSE to the
   * matching following A_ELSEIF, A_ELSE, A_ENDIF, as appropriate */
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    astx = STD_AST(stdx);
    switch (A_TYPEG(astx)) {
    case A_IFTHEN:
      ++nest;
      NEED(nest, ifnest, int, ifnestsize, ifnestsize + 50);
      ifnest[nest] = stdx;
      break;
    case A_ELSEIF:
    case A_ELSE:
      if (nest <= 0) {
        /* bad nesting */
        return;
      }
      stdifx = ifnest[nest];
      STD_FG(stdifx) = stdx;
      ifnest[nest] = stdx;
      break;
    case A_ENDIF:
      if (nest <= 0) {
        /* bad nesting */
        FREE(ifnest);
        return;
      }
      stdifx = ifnest[nest];
      STD_FG(stdifx) = stdx;
      --nest;
      break;
    }
  }
  FREE(ifnest);
  for (stdx = STD_NEXT(0); stdx; stdx = nextstdx) {
    nextstdx = STD_NEXT(stdx);
    astx = STD_AST(stdx);
    switch (A_TYPEG(astx)) {
    case A_IF:
      /* if condition is .FALSE., remove if and its statement */
      /* if condition is .TRUE., replace condition by the statement */
      condx = A_IFEXPRG(astx);
      if (A_ALIASG(condx))
        condx = A_ALIASG(condx);
      if (A_TYPEG(condx) == A_CNST && (sptr = A_SPTRG(condx)) > 0 &&
          DT_ISLOG(DTYPEG(sptr))) {
        if (CONVAL2G(sptr) == 0) {
          /* .false. */
          remove_block(stdx, stdx, FALSE, FALSE);
        } else {
          /* .true. */
          stdstmtx = A_IFSTMTG(astx);
          STD_AST(stdx) = stdstmtx;
        }
      }
      break;
    case A_IFTHEN:
    case A_ELSEIF:
      /* if condition is .FALSE., remove if and all statements up
       * to the ENDIF, ELSEIF, ELSE (change ELSEIF to IF, ELSE/ENDIF to
       * CONTINUE)
       * if the condition is .TRUE., remove this statement, and remove
       * all statements from the ELSEIF/ELSE up to and including the ENDIF */
      condx = A_IFEXPRG(astx);
      if (A_ALIASG(condx))
        condx = A_ALIASG(condx);
      if (A_TYPEG(condx) == A_CNST && (sptr = A_SPTRG(condx)) > 0 &&
          DT_ISLOG(DTYPEG(sptr))) {
        stdelsex = STD_FG(stdx);
        if (CONVAL2G(sptr) == 0) {
          /* .false. */
          /* remove_block changes the ELSEIF to IF, if necessary */
          astelsex = STD_AST(stdelsex);
          if (A_TYPEG(astelsex) == A_ELSEIF) {
            remove_block(stdx, stdelsex, TRUE, FALSE);
          } else {
            stdendx = STD_FG(stdelsex);
            astendx = STD_AST(stdendx);
            if (A_TYPEG(astendx) == A_ENDIF) {
              remove_block(stdx, stdelsex, TRUE, FALSE);
              remove_block(stdendx, stdendx, FALSE, FALSE);
            }
          }
        } else {
          /* .true. */
          astelsex = STD_AST(stdelsex);
          if (A_TYPEG(astelsex) == A_ENDIF) {
            /* simply change if/endif to continue/continue
             * or else/endif */
            if (A_TYPEG(astx) == A_IF) {
              remove_block(stdx, stdx, FALSE, FALSE);
              remove_block(stdelsex, stdelsex, FALSE, FALSE);
            } else {
              A_TYPEG(astx) = A_ELSE;
            }
          } else if (A_TYPEG(astelsex) == A_ELSE ||
                     A_TYPEG(astelsex) == A_ELSEIF) {
            /* simply change if to continue or elseif to else,
             * remove elseif through endif */
            astendx = 0;
            for (stdendx = STD_FG(stdelsex); STD_FG(stdendx);
                 stdendx = STD_FG(stdendx)) {
              astendx = STD_AST(stdendx);
              if (A_TYPEG(astendx) == A_ENDIF)
                break;
            }
            if (stdendx)
              astendx = STD_AST(stdendx);
            if (stdendx && astendx && A_TYPEG(astendx) == A_ENDIF) {
              if (A_TYPEG(astx) == A_IFTHEN) {
                remove_block(stdx, stdx, FALSE, FALSE);
                remove_block(stdelsex, stdendx, FALSE, FALSE);
              } else {
                A_TYPEG(astx) = A_ELSE;
                remove_block(stdelsex, stdendx, FALSE, TRUE);
              }
            }
          }
        }
      }
      break;
    }
  }
  /* clear STD_FG field */
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    STD_FG(stdx) = 0;
  }
} /* unconditional_branches */

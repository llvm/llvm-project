/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief optimizer submodule responsible for building the the flowgraph
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"

/*  static variables  */

static Q_ITEM *assigned_labels; /* labels appearing in ASSIGN statements */

static void set_std_fg(void);
static void partition_blocks(void);
static void dominators(void);
static int eval(int);
static void dfs(int);
#ifdef FG_PDOM
static void rdfs(int);
#endif
static void compress(int);

#define dom_link(v, w) ANCESTOR(w) = v

static PSI_P append_succ(int, int);
static PSI_P add_empty_succ(int);
static void fillin_succ(int, int);

/*  variables local to the dominator routines  */

static struct {
  int *parent;
  int *ancestor;
  int *label;
  int *semi;
  PSI_P *bucket;
} dom;

#define PARENT(i) dom.parent[i]
#define ANCESTOR(i) dom.ancestor[i]
#define LABEL(i) dom.label[i]
#define SEMI(i) dom.semi[i]
#define BUCKET(i) dom.bucket[i]
#define BUCKET_NODE(i) PSI_NODE(i)
#define BUCKET_NEXT(i) PSI_NEXT(i)
#define GET_BUCKET(i) i = (PSI *)getitem(BUCKET_AREA, sizeof(PSI))
#define FREE_BUCKET_AREA freearea(BUCKET_AREA)

void
flowgraph(void)
{

  int stdx,        /* std index */
      astx,        /* ast index */
      astli,       /* ast list item */
      i,           /* temporary */
      fgx,         /* flow graph index */
      nextfg,      /* next flowgraph node */
      lab;         /* label referenced in a branch */
  int cnt;         /* # of arguments */
  int argt;        /* arg table entry */
  LOGICAL is_ujmp; /* last ast of the previous node is an
                    * unconditional branch
                    */
  LOGICAL zerotrip;
  PSI_P q;
  int sym;
  /*
   * Define 'stack' to keep track of any control constructs, possibly
   * nested.
   */
  typedef struct {
    int topfg;    /* fg which is the 'top' of control construct */
    int ast;      /* ast (do, forall) of the 'top' */
    Q_ITEM *endl; /* list of nodes waiting for end of control */
  } CNTL;
  struct {
    int top; /* current 'top' (index) of the control stack */
    CNTL *stg_base;
    int stg_size;
    int stg_avail; /* available top of the stack */
  } cntl;

  Q_ITEM *endi;

#define CNTL_TOPFG(i) cntl.stg_base[i].topfg
#define CNTL_ENDL(i) cntl.stg_base[i].endl
#define CNTL_AST(i) cntl.stg_base[i].ast
#define PUSH_CNTL(f)             \
  {                              \
    cntl.top = cntl.stg_avail++; \
    OPT_NEED(cntl, CNTL, 16);    \
    CNTL_TOPFG(cntl.top) = f;    \
    CNTL_ENDL(cntl.top) = NULL;  \
  }
#define POP_CNTL                   \
  {                                \
    cntl.stg_avail--;              \
    cntl.top = cntl.stg_avail - 1; \
  }

  /* initialize for the flowgraph of the function  */

  opt.fgb.stg_avail = 1;  /* flow graph node of the entry */
  opt.num_nodes = 0;      /* number of nodes in the graph */
  opt.rteb.stg_avail = 0; /* number of retreating edges */

  assigned_labels = NULL; /* labels appearing in ASSIGN statements */
  partition_blocks();

  if (OPTDBG(9, 2))
    fprintf(gbl.dbgfil, "\n---------- flowgraph trace for function \"%s\"\n",
            getprint(STD_LABEL(FG_STDFIRST(gbl.entbih))));

  FG_FT(1) = 1;
  FG_EN(1) = 1;

  /*
   * if processing func or sub with entries. make entry nodes successors
   * of the main entry in the flowgraph.
   */
  if (SYMLKG(sym = gbl.entries) != NOSYM) {
    fgx = gbl.entbih;
    while ((sym = SYMLKG(sym)) != NOSYM) {
      i = BIHNUMG(sym);
      (void)add_succ(fgx, i);
      if (OPTDBG(9, 2)) {
        fprintf(gbl.dbgfil, "           entry pseudo jmp  fg %5d, entry %s\n",
                i, getprint(sym));
      }
    }
  }
  /*
   * loop through the flowgraph nodes for the function. scan blocks to
   * determine the successors of each block
   */
  OPT_ALLOC(cntl, CNTL, 16);
  cntl.stg_avail = 0;
  cntl.top = -1;
  is_ujmp = FALSE;
  fgx = gbl.entbih;    /* entry fg of the function */
  fgx = FG_LNEXT(fgx); /* entry fg is empty */
  q = add_succ(gbl.entbih, fgx);
  PSI_FT(q) = 1;

  for (;;) {
    if (OPTDBG(9, 2)) {
      fprintf(gbl.dbgfil, "--- FG --- %5d,  lineno:%5d", fgx,
              (int)FG_LINENO(fgx));
      if (FG_LABEL(fgx))
        fprintf(gbl.dbgfil, "  \"%s\"\n", getprint((int)FG_LABEL(fgx)));
      else
        fprintf(gbl.dbgfil, "\n");
    }
    if (FG_XT(fgx))
      break;

    nextfg = FG_LNEXT(fgx);
    /*
     * go through the stds of the block and find the node's successors.
     * Branch ASTs are examined to determine the branch targets. The CNTL
     * stack is used to represent the (possibly nested) control constructs
     * implied by many of the branching ASTs.   The top of the stack
     * represents the currently active control construct; entries below
     * the top of the stack represent any structures which are nested.
     *
     * For processing control constructs which may contain 'else'
     * constructs (elseif, else, elsewhere, etc.), a list is created
     * so that the nodes lexically preceding the 'else' constructs
     * can be associated with the the appropriate 'target' node
     * which is only known when the 'end' of the control construct is
     * found.  To accomplish this, an empty successor item is created
     * when the 'else' is processed and is filled in when the control's
     * 'end' is processed.  For the non-looping control constructs, the
     * target node is the node which follows the terminating AST; for the
     * looping control construct (e.g., forall-elseforall-endforall), the
     * target node is just the terminating node.
     *
     * For the looping control constructs, the top of the stack is needed
     * to locate the node which is to be the target of a successor arc
     * from the terminating node.
     *
     * For branching ASTs whose targets are labels, the successor arcs
     * are created whose targets are just those nodes which are 'defined'
     * by the labels.  Note that partition_blocks() associates the labels
     * with their blocks.
     *
     * WARNING:  the variable is_ujmp is a flag used to keep track of
     * whether or not the last std of the previous node is an unconditional
     * branch.  Any new cases for checking branch ASTs must include clearing
     * or setting is_ujmp.
     */
    stdx = FG_STDFIRST(fgx);
    if (stdx == 0)
      goto check_fall_thru;
    astx = STD_AST(stdx);
    switch (A_TYPEG(astx)) {

    /* asts which are leaders and also terminate a block */
    case A_ELSEIF:
      fillin_succ(CNTL_TOPFG(cntl.top), fgx);
      if (!is_ujmp) {
        endi = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
        endi->info = FG_LPREV(fgx);
        endi->next = CNTL_ENDL(cntl.top);
        CNTL_ENDL(cntl.top) = endi;
        (void)add_empty_succ((int)endi->info);
        CNTL_TOPFG(cntl.top) = fgx; /* new 'top' of control */
        (void)add_empty_succ(fgx);
      }
      is_ujmp = FALSE;
      break;
    case A_END:
    case A_ENTRY:
      is_ujmp = FALSE;
      break;
    case A_MP_MASTER:
    case A_MP_SINGLE:
    case A_MP_SECTIONS:
      PUSH_CNTL(fgx);
      (void)add_empty_succ(fgx);
      break;
    case A_MP_SECTION:
    case A_MP_LSECTION:
      fillin_succ(CNTL_TOPFG(cntl.top), fgx);
      CNTL_TOPFG(cntl.top) = fgx; /* new 'top' of control */
      (void)add_empty_succ(fgx);
      break;

    /* asts which are leaders of a block */
    case A_DO:
    case A_DOWHILE:
    case A_MP_PDO:
      PUSH_CNTL(fgx);
      CNTL_AST(cntl.top) = astx;
      goto check_last_std;
    case A_FORALL:
      if (A_IFSTMTG(astx)) {
/* single statement forall */
        (void)add_succ(fgx, fgx);
        is_ujmp = FALSE;
        goto check_fall_thru;
      }
      PUSH_CNTL(fgx);
      CNTL_AST(cntl.top) = astx;
      goto check_last_std;
    case A_ELSEFORALL:
      if (!is_ujmp) {
        endi = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
        endi->info = FG_LPREV(fgx);
        endi->next = NULL;
        CNTL_ENDL(cntl.top) = endi;
        (void)add_empty_succ((int)endi->info);
      }
      goto check_last_std;
    case A_ELSE:
    case A_ELSEWHERE:
      fillin_succ(CNTL_TOPFG(cntl.top), fgx);
      if (!is_ujmp) {
        endi = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
        endi->info = FG_LPREV(fgx);
        endi->next = CNTL_ENDL(cntl.top);
        CNTL_ENDL(cntl.top) = endi;
        (void)add_empty_succ((int)endi->info);
      }
      /* 'end' processing attempts to create an arc from the top
       * of the control construct to the block following the 'end'.
       * Set the 'top' of the control construct to 0 to prevent this
       * from happening.
       */
      CNTL_TOPFG(cntl.top) = 0;
      FLANG_FALLTHROUGH;
    default:
    check_last_std:
      is_ujmp = FALSE;
      stdx = FG_STDLAST(fgx);
      astx = STD_AST(stdx);
      /* asts which terminate a block */
      switch (A_TYPEG(astx)) {
      case A_ENDDO:
      case A_MP_ENDPDO:
        (void)add_succ(fgx, CNTL_TOPFG(cntl.top));
        A_OPT2P(astx, CNTL_AST(cntl.top));
        /*
         * since this is the end of a do loop, need to add a 'zero trip'
         * successor to the do's lexical predecessor which effects a
         * branch around the do loop.  Don't add the successor if it's
         * obvious that the loop is executed at least once.
         */
        zerotrip = TRUE; /* do loop may be zero trip */
        switch (A_TYPEG(CNTL_AST(cntl.top))) {
          int m1, m2, m3;
          DBLINT64 inum1, inum2;
        case A_DO:
        case A_MP_PDO:
          m1 = A_M1G(CNTL_AST(cntl.top));
          m2 = A_M2G(CNTL_AST(cntl.top));
          m3 = A_M3G(CNTL_AST(cntl.top));
          if ((m1 = A_ALIASG(m1)) && (m2 = A_ALIASG(m2)) &&
              (m3 == 0 || (m3 = A_ALIASG(m3)))) {
            /*
             * m1, m2, and possibly m3 locate A_CNST asts.
             */
            int increment;
            increment = 1; /* do increment is non-negative */
            switch (DTY(A_DTYPEG(m1))) {
            case TY_BINT:
            case TY_SINT:
            case TY_INT:
            case TY_BLOG:
            case TY_SLOG:
            case TY_LOG:
              if (m3 && CONVAL2G(A_SPTRG(m3)) < 0)
                increment = 0;
              m1 = CONVAL2G(A_SPTRG(m1));
              m2 = CONVAL2G(A_SPTRG(m2));
              if (increment) {
                if (m1 <= m2)
                  zerotrip = FALSE;
              } else {
                if (m1 >= m2)
                  zerotrip = FALSE;
              }
              break;
            case TY_INT8:
            case TY_LOG8:
              if (m3 && (CONVAL1G(A_SPTRG(m3)) & 0x80000000))
                increment = 0;
              inum1[0] = CONVAL1G(A_SPTRG(m1));
              inum1[1] = CONVAL2G(A_SPTRG(m1));
              inum2[0] = CONVAL1G(A_SPTRG(m2));
              inum2[1] = CONVAL2G(A_SPTRG(m2));
              if (increment) {
                if (cmp64(inum1, inum2) <= 0)
                  zerotrip = FALSE;
              } else {
                if (cmp64(inum1, inum2) >= 0)
                  zerotrip = FALSE;
              }
              break;
            case TY_REAL:
            case TY_DBLE:
            /* don't check non-integer do loops */
            default:
              break;
            }
          }
          break;
        }
        if (zerotrip) {
          q = append_succ(FG_LPREV(CNTL_TOPFG(cntl.top)), nextfg);
          PSI_ZTRP(q) = 1;
        }
        POP_CNTL;
        break;
      case A_ENDFORALL:
        (void)add_succ(fgx, CNTL_TOPFG(cntl.top));
        endi = CNTL_ENDL(cntl.top);
        if (endi)
          (void)add_succ((int)endi->info, fgx);
        A_OPT2P(astx, CNTL_AST(cntl.top));
        POP_CNTL;
        break;
      case A_IF:
        astx = A_IFSTMTG(astx);
        if (A_TYPEG(astx) == A_GOTO)
          goto do_goto;
        break;
      case A_IFTHEN:
        PUSH_CNTL(fgx);
        (void)add_empty_succ(fgx);
        break;
      case A_WHERE:
        if (A_IFSTMTG(astx) == 0) {
          /* not a single statement where */
          PUSH_CNTL(fgx);
          (void)add_empty_succ(fgx);
        }
        break;
      case A_ENDIF:
      case A_ENDWHERE:
        if (CNTL_TOPFG(cntl.top)) {
          /*
           * tpr1356:
           * If the endif/endwhere statement falls thru to the next
           * node, the next node is a successor of the node defined
           * as the top of the control construct (an empty successor
           * arc was created when the top (if/where) was processed).
           *
           * If the endif/endwhere statement does not fall thru, the
           * successor of the top of the control construct cannot yet
           * be filled in.  Presumably, the control construct appears
           * in an if/where nest, and the successor arc can be filled
           * in when the end statement of outer control construct is
           * processed.  Add the node representing the top of the
           * control construct to the 'end' list of the outer
           * control construct.
           */
          if (FG_FT(fgx) || cntl.top <= 0)
            fillin_succ(CNTL_TOPFG(cntl.top), nextfg);
          else {
            i = cntl.top - 1;
            endi = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
            endi->info = CNTL_TOPFG(cntl.top);
            endi->next = CNTL_ENDL(i);
            CNTL_ENDL(i) = endi;
          }
        }
        /*
         * tpr1356 - any deferred successors arcs of this control
         * construct cannot be filled in with the next node if the
         * endif/endwhere statement does not fall thru (again, this
         * implies that the current construct appears in an if/where
         * nest.  The 'end' list is moved to the end list of the outer
         * control construct.
         */
        if (FG_FT(fgx) || cntl.top <= 0)
          for (endi = CNTL_ENDL(cntl.top); endi != NULL; endi = endi->next)
            fillin_succ((int)endi->info, nextfg);
        else {
          for (endi = CNTL_ENDL(cntl.top); endi != NULL; endi = endi->next)
            if (endi->next == NULL)
              break;
          if (endi) {
            i = cntl.top - 1;
            endi->next = CNTL_ENDL(i);
            CNTL_ENDL(i) = CNTL_ENDL(cntl.top);
          }
        }
        POP_CNTL;
        break;
      case A_GOTO:
      do_goto:
        lab = A_SPTRG(A_L1G(astx));
        (void)add_succ(fgx, (int)ILIBLKG(lab));
        break;
      case A_CGOTO:
        astli = A_LISTG(astx);
        while (TRUE) {
          lab = A_SPTRG(ASTLI_AST(astli));
          (void)add_succ(fgx, (int)ILIBLKG(lab));
          astli = ASTLI_NEXT(astli);
          if (astli == 0)
            break;
        }
        break;
      case A_AIF:
        lab = A_SPTRG(A_L1G(astx));
        (void)add_succ(fgx, (int)ILIBLKG(lab));
        lab = A_SPTRG(A_L2G(astx));
        (void)add_succ(fgx, (int)ILIBLKG(lab));
        lab = A_SPTRG(A_L3G(astx));
        (void)add_succ(fgx, (int)ILIBLKG(lab));
        is_ujmp = TRUE;
        break;
      case A_AGOTO:
        /* The successors are the labels in the label list if the
         * the list is present.  Otherwise, the successors are the
         * labels which appeared in the ASSIGN statement; these labels
         * are collected in partition_blocks().
         */
        astli = A_LISTG(astx);
        if (astli)
          while (TRUE) {
            lab = A_SPTRG(ASTLI_AST(astli));
            (void)add_succ(fgx, (int)ILIBLKG(lab));
            astli = ASTLI_NEXT(astli);
            if (astli == 0)
              break;
          }
        else {
          Q_ITEM *q;
          for (q = assigned_labels; q != NULL; q = q->next) {
            lab = q->info;
            (void)add_succ(fgx, (int)ILIBLKG(lab));
          }
        }
        is_ujmp = TRUE;
        break;
      case A_CALL:
        if (!STD_BR(stdx))
          break;
        /* A call marked as branch implies that the call contains
         * alternate returns.
         */
        is_ujmp = FALSE;
        argt = A_ARGSG(astx);
        for (i = 0, cnt = A_ARGCNTG(astx); cnt; cnt--) {
          int opnd;
          opnd = ARGT_ARG(argt, i);
          if (opnd && A_TYPEG(opnd) == A_LABEL) {
            lab = A_SPTRG(opnd);
            (void)add_succ(fgx, (int)ILIBLKG(lab));
          }
          ++i;
        }
        break;
      case A_MP_ENDMASTER:
      case A_MP_ENDSINGLE:
      case A_MP_ENDSECTIONS:
        fillin_succ(CNTL_TOPFG(cntl.top), fgx);
        POP_CNTL;
        break;
      default:
        break;
      }
      break;
    }

  /* check to see if the block falls through to the next one  */

  check_fall_thru:
    if (FG_FT(fgx)) {
      q = add_succ(fgx, nextfg);
      PSI_FT(q) = 1;
      if (OPTDBG(9, 2))
        fprintf(gbl.dbgfil, "           block falls through\n");
    }
    fgx = nextfg;
  }

#if DEBUG
  assert(cntl.top == -1, "flowgraph:cntl.top not empty", cntl.top, 3);
#endif
  OPT_FREE(cntl);
  freearea(Q_AREA);

  /* build the dominator tree and complete the flowgraph */

  dominators();

/*
 * it is determined if any of the flowgraph nodes are unreachable. A node
 * is unreachable if its dfn is -1.  An uncreachable node's bih is
 * deleted from the bih list.  Note that this is only done if the number
 * of nodes which were involved in the depth first search is not the
 * number of nodes created from the bihs. Also, the exit block is not
 * deleted.
 */
    set_std_fg();
}

static void
set_std_fg(void)
{
  int fg;
  int stdx;

  for (fg = 1; fg <= opt.num_nodes; ++fg) {
    if (FG_DFN(fg) == -1)
      continue;
    for (stdx = FG_STDFIRST(fg); stdx; stdx = STD_NEXT(stdx)) {
      STD_FG(stdx) = fg;
      if (stdx == FG_STDLAST(fg)) {
        break;
      }
    }
  }
} /* set_std_fg */

/*
    routines responsible for partitioning the program into basic blocks (flow
    graph nodes):
        partition_blocks() - traverses STDs determining leaders and terminators
        cr_block(std)      - create a basic block. if an empty block exists, the
                             block is used; otherwise, a new block is created.
                             std (may be zero) is added immediately to the
                             block.
        wr_block()         - 'writes' (terminates) a basic block; if a block
                             hasn't been created, nothing is written.  Writing
                             a block implies that the node's last STD entry
                             is filled in.
        chk_block(std)    -  'add' the std to the current block; if a block
                             hasn't been created, one is created.
 */

/* global data for partitioning blocks */
typedef struct {
  int curfg;            /* the current flow graph node created */
  int curstd;           /* the current STD for the node */
  int atomic;           /* in atomic region */
  int par_cnt;          /* parallel region counter */
  unsigned noblock : 1; /* if set, there isn't a current node */
  unsigned master : 1;  /* in master region */
  unsigned cs : 1;      /* in critical section */
  unsigned parsect : 1; /* fg belongs to a parallel section */
  unsigned task : 1;    /* in task*/
} EXP;
static EXP expb;

static void cr_block(int std);
static void wr_block(void);
static void chk_block(int std);

static void
partition_blocks(void)
{
  int std;
  int next_std;
  int ast;
  int atype;
  int label;
  int ent;
  int cnt;
  int argt;
  int i;
  int alab; /* label referenced in an ASSIGN statement */

  BZERO(opt.fgb.stg_base, FG, 1);
  BZERO(&expb, EXP, 1);
  STD_LINENO(0) = FUNCLINEG(gbl.currsub);
  STD_LABEL(0) = gbl.currsub;

  expb.curfg = add_fg(0);
  expb.noblock = 1;
  FG_LINENO(1) = FUNCLINEG(gbl.currsub);
  FG_FT(1) = 1;

  for (std = STD_NEXT(0); std; std = next_std) {
    next_std = STD_NEXT(std); /* 'cause insertions may alter STD_NEXT */
    gbl.lineno = STD_LINENO(std);
    STD_BR(std) = 0;
    label = STD_LABEL(std);
    if (label) {
      wr_block();
      cr_block(0);
      ILIBLKP(label, expb.curfg);
      label = 0;
    }
    ast = STD_AST(std);
    atype = A_TYPEG(ast);
    switch (atype) {
    case A_CONTINUE:
      switch (A_TYPEG(STD_AST(next_std))) {
      case A_DO:
      case A_FORALL:
      case A_MP_PDO:
        if (expb.curstd) {
          wr_block();
          cr_block(std);
        } else {
          chk_block(std);
        }
        break;
      default:
        chk_block(std);
      }
      break;

    /* asts which are leaders of a block */
    /*  loop asts  */
    case A_DO:
    case A_FORALL:
    case A_MP_PDO:
      if ((STD_PREV(std) == 0) ||
          A_TYPEG(STD_AST(STD_PREV(std))) != A_CONTINUE ||
          FG_STDFIRST(expb.curfg) != FG_STDLAST(expb.curfg) || expb.noblock) {
        int s;
        wr_block();
        s = add_stmt_before(mk_stmt(A_CONTINUE, 0), std);
        STD_ACCEL(s) = STD_ACCEL(std);
        STD_KERNEL(s) = STD_KERNEL(std);
        cr_block(s);
      }
      FLANG_FALLTHROUGH;
    case A_DOWHILE:
      wr_block();
      cr_block(std);
      STD_BR(std) = 1;

      /* ast-specific code for the loop asts */
      switch (atype) {
      case A_FORALL:
        if (A_IFSTMTG(ast)) {
          /* single statement forall */
          wr_block();
        }
        break;
      case A_MP_PDO:
        FG_PARLOOP(expb.curfg) = 1;
        break;
      }
      break;
    case A_ELSE:
    case A_ELSEWHERE:
    case A_ELSEFORALL:
      wr_block();
      cr_block(std);
      FG_FT(FG_LPREV(expb.curfg)) = 0;
      STD_BR(std) = 1;
      break;
    case A_ATOMIC:
    case A_ATOMICCAPTURE:
    case A_ATOMICREAD:
    case A_ATOMICWRITE:
      wr_block();
      expb.atomic = std;
      cr_block(std);
      break;
    case A_MASTER:
      wr_block();
      expb.master = 1;
      cr_block(std);
      break;
    case A_CRITICAL:
      wr_block();
      expb.cs = 1;
      cr_block(std);
      break;
    case A_BARRIER:
    case A_MP_BARRIER:
    case A_MP_TASKWAIT:
    case A_MP_TASKYIELD:
    case A_MP_WORKSHARE:
      wr_block();
      cr_block(std);
      break;
    case A_MP_PARALLEL:
      wr_block();
      expb.par_cnt++;
      cr_block(std);
      break;
    case A_MP_BMPSCOPE:
      chk_block(std);
      wr_block();
      break;
    case A_MP_EMPSCOPE:
      chk_block(std);
      wr_block();
      break;
    case A_MP_CRITICAL:
    case A_MP_ATOMIC:
      wr_block();
      expb.cs = 1;
      cr_block(std);
      break;
    case A_MP_TASKREG:
    case A_MP_TASKLOOPREG:
      wr_block();
      cr_block(std);
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      wr_block();
      expb.task++;
      cr_block(std);
      break;

    /* asts which terminate a block */
    case A_ENDDO:
    case A_ENDFORALL:
    case A_MP_ENDPDO:
      chk_block(std);
      wr_block();
      STD_BR(std) = 1;
      break;
    case A_IF:
    case A_IFTHEN:
    case A_ENDIF:
    case A_CGOTO:
    case A_AGOTO:
    case A_WHERE:
    case A_ENDWHERE:
      chk_block(std);
      wr_block();
      STD_BR(std) = 1;
      break;
    case A_AIF:
    case A_GOTO:
      chk_block(std);
      wr_block();
      /*
       * these asts will terminate a block and under normal conditions
       * this block does not fall thru to the next block.  However, the
       * next ast may terminate a control flow structure (if, do, forall,
       * where) in which case it's necessary to set the fall thru flag
       * flag of the block terminated by the goto/aif.
       */
      if (next_std) {
        switch (A_TYPEG(STD_AST(next_std))) {
        case A_ENDIF:
        case A_ENDWHERE:
        case A_ENDDO:
        case A_ENDFORALL:
        case A_MP_ENDPDO:
          break;
        default:
          FG_FT(expb.curfg) = 0;
          break;
        }
      }
      STD_BR(std) = 1;
      break;
    case A_CALL:
      /* call which contains alternate return specifiers terminates
       * a block.
       */
      chk_block(std); /* first, add call to block */
      argt = A_ARGSG(ast);
      for (i = 0, cnt = A_ARGCNTG(ast); cnt; cnt--) {
        int opnd;
        opnd = ARGT_ARG(argt, i);
        if (opnd && A_TYPEG(opnd) == A_LABEL) {
          wr_block();
          STD_BR(std) = 1;
          break;
        }
        ++i;
      }
      break;

    case A_ENDATOMIC:
      chk_block(std);
      wr_block();
      expb.atomic = 0;
      break;
    case A_ENDMASTER:
      chk_block(std);
      wr_block();
      expb.master = 0;
      break;
    case A_ENDCRITICAL:
    case A_MP_ENDATOMIC:
      chk_block(std);
      wr_block();
      expb.cs = 0;
      break;
    case A_MP_ENDPARALLEL:
      chk_block(std);
      wr_block();
      expb.par_cnt--;
      break;
    case A_MP_ENDCRITICAL:
      chk_block(std);
      wr_block();
      expb.cs = 0;
      break;
    case A_MP_ENDWORKSHARE:
      chk_block(std);
      wr_block();
      break;
    case A_MP_ETASKLOOPREG:
      chk_block(std);
      wr_block();
      break;
    case A_MP_ENDTASK:
    case A_MP_ETASKLOOP:
      chk_block(std);
      wr_block();
      expb.task--;
      break;

    /* asts which are leaders and also terminate a block */
    case A_ELSEIF:
      wr_block();
      cr_block(std);
      FG_FT(FG_LPREV(expb.curfg)) = 0;
      wr_block();
      STD_BR(std) = 1;
      break;
    case A_END:
      wr_block();
      cr_block(std);
      wr_block();
      break;
    case A_ENTRY:
      ent = A_SPTRG(ast);
      wr_block();
      cr_block(std);
      BIHNUMP(ent, expb.curfg);
      wr_block();
      break;
    case A_MP_MASTER:
    case A_MP_SINGLE:
    case A_MP_SECTIONS:
      wr_block();
      expb.parsect = 1;
      cr_block(std);
      wr_block();
      break;
    case A_MP_SECTION:
    case A_MP_LSECTION:
      wr_block();
      cr_block(std);
      wr_block();
      break;
    case A_MP_ENDMASTER:
    case A_MP_ENDSINGLE:
    case A_MP_ENDSECTIONS:
      wr_block();
      cr_block(std);
      wr_block();
      expb.parsect = 0;
      break;

    /* other asts which are just added to the block */
    case A_ASNGOTO:
      /* collect all nonFORMAT labels which appear in the ASSIGN
       * statements.
       */
      alab = A_SPTRG(A_SRCG(ast));
      if (FMTPTG(alab) == 0) {
        Q_ITEM *q;
        for (q = assigned_labels; q != NULL; q = q->next)
          if (q->info == alab)
            break;
        if (q == NULL) {
          q = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM));
          q->info = alab;
          q->next = assigned_labels;
          assigned_labels = q;
        }
      }
      chk_block(std);
      break;
    case A_RETURN:
      STD_BR(std) = 1;
      FLANG_FALLTHROUGH;
    default:
      chk_block(std);
      break;
    }
  }
  opt.exitfg = expb.curfg; /* save exit bih */
  FG_XT(opt.exitfg) = 1;
  FG_FT(opt.exitfg) = 0;

}

static void
cr_block(int std)
{
  if (expb.noblock == 0 && expb.curstd == 0)
    /* use empty block */
    ;
  else {
    if (expb.noblock == 0)
      wr_block();
    expb.curfg = add_fg(expb.curfg);
    expb.noblock = 0;
    FG_FT(expb.curfg) = 1; /* assume new block falls thru */
  }
  FG_STDFIRST(expb.curfg) = FG_STDLAST(expb.curfg) = std;
  if (std)
    FG_LINENO(expb.curfg) = STD_LINENO(std);
  FG_MASTER(expb.curfg) = expb.master;
  FG_ATOMIC(expb.curfg) = expb.atomic;
  FG_CS(expb.curfg) = expb.cs;
  FG_PAR(expb.curfg) = expb.par_cnt;
  FG_PARSECT(expb.curfg) = expb.parsect;
  FG_TASK(expb.curfg) = expb.task > 0;
  expb.curstd = std;

}

static void
wr_block(void)
{
  if (expb.noblock)
    /* if no block created, then nothing to write */
    return;
  FG_STDLAST(expb.curfg) = expb.curstd;
  expb.noblock = 1;
  expb.curstd = 0;

}

static void
chk_block(int std)
{
  if (expb.noblock)
    cr_block(0);
  if (expb.curstd == 0) {
    FG_STDFIRST(expb.curfg) = std;
    FG_LINENO(expb.curfg) = STD_LINENO(std);
  }
  expb.curstd = std;

}

/*
    optimizer routines responsible for building the dominator tree
    of a flowgraph.
    The algorithm used is specified in "A Fast Algorithm for Finding
    Dominators in a Flowgraph", Lengauer and Tarjan,  TOPLAS, July
    1979, Vol. 1 No. 1 (pg 121).
*/

static void
dominators(void)
{
  int u, v, w, i;
  PSI_P p;

#define DOM_NEW(x, y)                        \
  {                                          \
    NEW(x, y, opt.num_nodes + 1);            \
    if (x == NULL)                           \
      error(7, 4, gbl.lineno, CNULL, CNULL); \
  }

  DOM_NEW(dom.parent, int);
  DOM_NEW(dom.ancestor, int);
  DOM_NEW(dom.label, int);
  DOM_NEW(dom.semi, int);
  DOM_NEW(dom.bucket, PSI_P);

  /*   step 1  */

  for (v = 1; v <= opt.num_nodes; v++) {
    BUCKET(v) = PSI_P_NULL;
    SEMI(v) = 0;
  }

  opt.dfn = 0;
  dfs(1);

  for (i = opt.dfn; i >= 2; i--) {
    w = VTX_NODE(i);

    /*  step 2  */

    for (p = FG_PRED(w); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      v = PSI_NODE(p);
      u = eval(v);
      if (SEMI(u) < SEMI(w))
        SEMI(w) = SEMI(u);
    }
    /*
     * add w to the bucket list of the semi-dominator of w
     */
    GET_BUCKET(p);
    PSI_NODE(p) = w;
    PSI_NEXT(p) = BUCKET(VTX_NODE(SEMI(w)));
    BUCKET(VTX_NODE(SEMI(w))) = p;

    dom_link(PARENT(w), w);

    /*  step 3  */

    for (p = BUCKET(PARENT(w)); p != PSI_P_NULL;) {
      v = PSI_NODE(p);
      p = BUCKET(PARENT(w)) = PSI_NEXT(p);
      u = eval(v);
      FG_DOM(v) = SEMI(u) < SEMI(v) ? u : PARENT(w);
    }
  }

  /*  step 4  */

  for (i = 2; i <= opt.dfn; i++) {
    w = VTX_NODE(i);
    if (FG_DOM(w) != VTX_NODE(SEMI(w)))
      FG_DOM(w) = FG_DOM(FG_DOM(w));
  }
  FG_DOM(1) = 0;

  FREE(dom.parent);
  FREE(dom.ancestor);
  FREE(dom.label);
  FREE(dom.semi);
  FREE(dom.bucket);
  FREE_BUCKET_AREA;
}

#ifdef FG_PDOM
/* same algorithm in reverse to compute postdominators */
static int rdfn;
void
postdominators(void)
{
  int u, v, w, i, xt;
  PSI_P p;

  /* reuse same dom data structures */
  DOM_NEW(dom.parent, int);
  DOM_NEW(dom.ancestor, int);
  DOM_NEW(dom.label, int);
  DOM_NEW(dom.semi, int);
  DOM_NEW(dom.bucket, PSI_P);

  /*   step 1  */

  for (v = 1; v <= opt.num_nodes; v++) {
    BUCKET(v) = PSI_P_NULL;
    SEMI(v) = 0;
    PARENT(v) = 0;
  }

  rdfn = 0;
  xt = 0;
  /* find exit node(s) */
  for (i = opt.num_nodes; i >= 1; i--) {
    if (FG_XT(i)) {
      if (xt == 0) {
        xt = i;
        rdfs(i);
      } else {
        /* treat multiple exit nodes like they are predecessors of 'xt' */
        PARENT(i) = xt;
        rdfs(i);
      }
    }
  }
  for (i = opt.num_nodes; i >= 1; i--) {
    /* look for nodes that aren't in the rdfs ... infinite loops, etc.
     * add links from 'xt' */
    if (PARENT(i) == 0 && !FG_XT(i) && FG_DFN(i)) {
      PARENT(i) = xt;
      rdfs(i);
    }
  }

  for (i = rdfn; i >= 2; i--) {
    w = RVTX_NODE(i);

    /*  step 2  */

    for (p = FG_SUCC(w); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      v = PSI_NODE(p);
      u = eval(v);
      if (SEMI(u) < SEMI(w))
        SEMI(w) = SEMI(u);
    }
    /*
     * add w to the bucket list of the semi-dominator of w
     */
    GET_BUCKET(p);
    PSI_NODE(p) = w;
    PSI_NEXT(p) = BUCKET(RVTX_NODE(SEMI(w)));
    BUCKET(RVTX_NODE(SEMI(w))) = p;

    dom_link(PARENT(w), w);

    /*  step 3  */

    for (p = BUCKET(PARENT(w)); p != PSI_P_NULL;) {
      v = PSI_NODE(p);
      p = BUCKET(PARENT(w)) = PSI_NEXT(p);
      u = eval(v);
      FG_PDOM(v) = SEMI(u) < SEMI(v) ? u : PARENT(w);
    }
  }

  /*  step 4  */

  for (i = 2; i <= rdfn; i++) {
    w = RVTX_NODE(i);
    if (FG_PDOM(w) != RVTX_NODE(SEMI(w)))
      FG_PDOM(w) = FG_PDOM(FG_PDOM(w));
  }
  FG_PDOM(xt) = 0;

  FREE(dom.parent);
  FREE(dom.ancestor);
  FREE(dom.label);
  FREE(dom.semi);
  FREE(dom.bucket);
  FREE_BUCKET_AREA;
}
#endif

/*
   compute depth first number for flow graph node v
*/
static void
dfs(int v)
{
  PSI_P p;
  int w, rte;

  FG_DFN(v) = SEMI(v) = ++opt.dfn;  /* set dfn of node v */
  VTX_NODE(opt.dfn) = LABEL(v) = v; /* set dfn vector */
  ANCESTOR(v) = 0;
  /*
   * recurse for all of the successors, w, of v
   */
  for (p = FG_SUCC(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (SEMI(w = PSI_NODE(p)) == 0) {
      PARENT(w) = v;
      dfs(w);
    }
    /*
     * add v to the predecessor list of w
     */
    (void)add_pred(w, v);

    if (FG_DFN(v) >= FG_DFN(w)) { /* (v, w) is a retreating edge */
      rte = opt.rteb.stg_avail++;
      OPT_NEED(opt.rteb, EDGE, 32);
      EDGE_PRED(rte) = v;
      EDGE_SUCC(rte) = w;
      EDGE_NEXT(rte) = -1;
    }
  }
}

#ifdef FG_PDOM
/*
 * compute reverse depth first number for flow graph node v
 */
static void
rdfs(int v)
{
  PSI_P p;
  int w;

  if (FG_DFN(v) == -1) {
    FG_RDFN(v) = -1;
    ANCESTOR(v) = 0;
    return;
  }
  FG_RDFN(v) = SEMI(v) = ++rdfn;  /* set dfn of node v */
  RVTX_NODE(rdfn) = LABEL(v) = v; /* set dfn vector */
  ANCESTOR(v) = 0;
  /*
   * recurse for all of the predecessors, w, of v
   */
  for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (SEMI(w = PSI_NODE(p)) == 0) {
      PARENT(w) = v;
      rdfs(w);
    }
  }
} /* rdfs */
#endif

static void
compress(int v)
{
  int ancv;

  if (ANCESTOR(ancv = ANCESTOR(v)) != 0) {
    compress(ancv);
    if (SEMI(LABEL(ancv)) < SEMI(LABEL(v)))
      LABEL(v) = LABEL(ancv);
    ANCESTOR(v) = ANCESTOR(ancv);
  }
}

static int
eval(int v)
{
  if (ANCESTOR(v) == 0)
    return (v);
  compress(v);
  return (LABEL(v));
}

/*
static void dom_link(v, w)
   int v, w;
{
   ANCESTOR(w) = v;
   return;
}
*/

/*---------------------   FLOWGRAPH UTILITIES   ---------------------*/

/** \brief Create an FG and insert it after FG \p after
 */
int
add_fg(int after)
{
  FG *fg;
  int i;
  int a;

  i = opt.fgb.stg_avail++; /* get a flow graph node */
  OPT_NEED(opt.fgb, FG, 100);
  fg = opt.fgb.stg_base + i;

  /* define the flowgraph node  */

  BZERO(fg, FG, 1);
  fg->dfn = -1;
  opt.num_nodes++;
  a = FG_LNEXT(after);
  FG_LNEXT(i) = a;
  FG_LPREV(a) = i;
  FG_LPREV(i) = after;
  FG_LNEXT(after) = i;

  return i;
}

void
delete_fg(int fgx)
{
  int bef, aft;

  bef = FG_LPREV(fgx);
  aft = FG_LNEXT(fgx);
  FG_LNEXT(bef) = aft;
  FG_LPREV(aft) = bef;
}

PSI_P
add_succ(int fgx, int i)
{
  PSI_P p;
  FG *fg;

#if DEBUG
  assert(i != 0, "flowgraph: node is zero", 0, 3);
#endif
  fg = opt.fgb.stg_base + fgx;
  for (p = fg->succ; p != PSI_P_NULL; p = PSI_NEXT(p))
    if (i == PSI_NODE(p))
      return p;
  p = add_empty_succ(fgx);
  PSI_NODE(p) = i;
  return p;
}

static PSI_P
append_succ(int fgx, int i)
{
  PSI_P p, q;
  FG *fg;

#if DEBUG
  assert(i != 0, "flowgraph: node is zero", 0, 3);
#endif
  fg = opt.fgb.stg_base + fgx;
  for (p = fg->succ; p != PSI_P_NULL; p = PSI_NEXT(p))
    if (i == PSI_NODE(p))
      return p;
  p = add_empty_succ(fgx);
  PSI_NODE(p) = i;
  for (q = p; PSI_NEXT(q) != PSI_P_NULL; q = PSI_NEXT(q))
    ;
  if (q != p) {
    fg->succ = PSI_NEXT(p);
    PSI_NEXT(q) = p;
    PSI_NEXT(p) = PSI_P_NULL;
  }
  return p;
}

static PSI_P
add_empty_succ(int fgx)
{
  PSI_P p;
  FG *fg;

  fg = opt.fgb.stg_base + fgx;
  GET_PSI(p);
  PSI_NODE(p) = 0;
  PSI_NEXT(p) = fg->succ;
  PSI_ALL(p) = 0;
  fg->succ = p;
  return p;
}

static void
fillin_succ(int fgx, int succ)
{
  PSI_P p;
  FG *fg;

  fg = opt.fgb.stg_base + fgx;
  for (p = fg->succ; p != PSI_P_NULL; p = PSI_NEXT(p))
    if (PSI_NODE(p) == 0) {
      PSI_NODE(p) = succ;
      return;
    }
#if DEBUG
  interr("fillin_succ: empty succ not fnd", fgx, 3);
#endif
}

PSI_P
add_pred(int fgx, int i)
{
  PSI_P p;
  FG *fg;

#if DEBUG
  assert(i != 0, "flowgraph: node is zero", 0, 3);
#endif
  fg = opt.fgb.stg_base + fgx;
  for (p = fg->pred; p != PSI_P_NULL; p = PSI_NEXT(p))
    if (i == PSI_NODE(p))
      return p;
  GET_PSI(p);
  PSI_NODE(p) = i;
  PSI_NEXT(p) = fg->pred;
  PSI_ALL(p) = 0;
  fg->pred = p;
  return p;
}

/** \brief Remove the edge fg1 -> fg2 from the flowgraph.

    1.  remove fg2 from succ(fg1)
    2.  remove fg1 from pred(fg2)
 */
void
rm_edge(int fg1, int fg2)
{
  PSI_P prev, p;

  if (OPTDBG(9, 1))
    fprintf(gbl.dbgfil, "***** remove edge (%d, %d)\n", fg1, fg2);
  prev = PSI_P_NULL;
  for (p = FG_SUCC(fg1); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (PSI_NODE(p) == fg2)
      break;
    prev = p;
  }
  if (p == PSI_P_NULL)
    return;
  if (prev != PSI_P_NULL)
    PSI_NEXT(prev) = PSI_NEXT(p);
  else
    FG_SUCC(fg1) = PSI_NEXT(p);

  prev = PSI_P_NULL;
  for (p = FG_PRED(fg2); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (PSI_NODE(p) == fg1)
      break;
    prev = p;
  }
#if DEBUG
  assert(p != PSI_P_NULL, "rm_edge:pred not found", fg2, 0);
#endif
  if (prev != PSI_P_NULL)
    PSI_NEXT(prev) = PSI_NEXT(p);
  else {
    FG_PRED(fg2) = PSI_NEXT(p);
    if (FG_PRED(fg2) == PSI_P_NULL && !FG_XT(fg2)) {
      /* since target of the orig. edge no longer has any
       * predecessors, can remove this node from the loop
       * and remove all edges beginning with this node.
       */
      int i, j, lp;

      /* 1.  remove the node from the function */
      j = FG_LNEXT(fg2);
      i = FG_LPREV(j) = FG_LPREV(fg2);
      FG_LNEXT(i) = j;
      if (OPTDBG(9, 1))
        fprintf(gbl.dbgfil, "      deleted fg %d\n", fg2);

      /* 2.  remove the node from its loop */
      lp = FG_LOOP(fg2);
      j = 0;
      for (i = LP_FG(lp); i; i = FG_NEXT(i)) {
        if (i == fg2)
          break;
        j = i;
      }
      if (i) {
        if (j)
          FG_NEXT(j) = FG_NEXT(fg2);
        else
          LP_FG(lp) = FG_NEXT(fg2);
      }

      /* 3.  remove the edge(s) beginning with the node */
      for (p = FG_SUCC(fg2); p != PSI_P_NULL; p = PSI_NEXT(p)) {
        /* do more to update rfcnts of labels if edge represents
         * some sort of branch to succ of fg2
         */
        rm_edge(fg2, (int)PSI_NODE(p));
      }
    }
  }

  if (OPTDBG(9, 1)) {
    dump_node(fg1);
    dump_node(fg2);
  }
}

/** \brief
    \param newfg block to be added
    \param lp descendant of the loop to which block is added
 */
int
add_to_parent(int newfg, int lp)
{
  int parent;

  parent = lp;
  do {
    parent = LP_PARENT(parent);
#if DEBUG
    assert(parent || LP_FG(parent), "add_to_parent: null region 0 of", lp, 0);
#endif
  } while (LP_FG(parent) == 0);
  FG_DOM(newfg) = FG_DOM(LP_HEAD(lp));
  FG_LOOP(newfg) = parent;
  FG_NEXT(newfg) = LP_FG(parent);
  LP_FG(parent) = newfg;

  return newfg;
}

void
_dump_node(int v, FILE *ff)
{
  int i;
  PSI_P p;
  int s;

  if (ff == NULL)
    ff = stderr;

  fprintf(ff, "%5d. ln#:%-5d  lbl:%-5d  lnxt:%-5d  lprv:%-5d", v,
          (int)FG_LINENO(v), (int)FG_LABEL(v), (int)FG_LNEXT(v),
          (int)FG_LPREV(v));
  fprintf(ff, "  first:%-5d  last:%-5d", FG_STDFIRST(v), FG_STDLAST(v));
  fprintf(ff, "\n");
  fprintf(ff, "       dfn:%-5d  dom:%-5d  loop:%-5d  next:%-5d    nat:%-5d\n",
          (int)FG_DFN(v), (int)FG_DOM(v), (int)FG_LOOP(v), (int)FG_NEXT(v),
          (int)FG_NATNXT(v));

  fprintf(ff, "       flags:");
#undef _PFG
#define _PFG(cond, str) \
  if (cond)             \
  fprintf(ff, " %s", str)
  _PFG(FG_FT(v), "FT");
  _PFG(FG_EN(v), "EN");
  _PFG(FG_EX(v), "EX");
  _PFG(FG_XT(v), "XT");
  _PFG(FG_ZTRP(v), "ZT");
  _PFG(FG_HEAD(v), "HEAD");
  _PFG(FG_TAIL(v), "TAIL");
  _PFG(FG_INNERMOST(v), "INNERMOST");
  _PFG(FG_MEXITS(v), "MEXITS");
  _PFG(FG_PAR(v), "PAR");
  _PFG(FG_PTR_STORE(v), "ST*");
  _PFG(FG_JMP_TBL(v), "JMPT");
  _PFG(FG_CS(v), "CS");
  _PFG(FG_MASTER(v), "MASTER");
  _PFG(FG_PARLOOP(v), "PARLOOP");
  _PFG(FG_PARSECT(v), "PARSECT");
  _PFG(FG_TASK(v), "TASK");
  if (FG_ATOMIC(v))
    fprintf(ff, " ATOMIC(%d)", FG_ATOMIC(v));
  fprintf(ff, "\n");

  fprintf(ff, "       succ:");
  i = 0;
  for (p = FG_SUCC(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (i == 9) {
      fprintf(ff, "\n            ");
      i = 0;
    }
    fprintf(ff, " %d", (int)PSI_NODE(p));
    if (PSI_FT(p))
      fprintf(ff, "*");
    i++;
  }
  fprintf(ff, "\n");

  fprintf(ff, "       pred:");
  i = 0;
  for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (i == 9) {
      fprintf(ff, "\n            ");
      i = 0;
    }
    fprintf(ff, " %d", (int)PSI_NODE(p));
    i++;
  }
  fprintf(ff, "\n");
  if (OPTDBG(10, 6) && (s = FG_STDFIRST(v))) {
    if (STD_LABEL(s))
      fprintf(ff, "      %s:\n", SYMNAME(STD_LABEL(s)));
    while (TRUE) {
      fprintf(ff, "        std %d", s);
      if (STD_EX(s))
        fprintf(ff, " callfg");
      if (STD_BR(s))
        fprintf(ff, " br");
      fprintf(ff, "\n");
      dbg_print_ast((int)STD_AST(s), ff);
      if (s == FG_STDLAST(v))
        break;
      s = STD_NEXT(s);
    }
  }

}

void
dump_node(int v)
{
  _dump_node(v, gbl.dbgfil);
}

/*-----------------------------------------------------------------*/
/*  flowgraph read/write utility functions:  names are the same
 *  as the names in the node compilers which provide the identical
 *  functionality.
 */

static struct {
  struct {/* state of STD(0) before a block is read */
    int next;
    int prev;
  } std0;
  int std_before; /* std before the first std of a block */
  int std_after;  /* std after the first std of a block */
} rw_state;

/** \brief 'read' a flowgraph node.

    This defines the 0th STD entry to locate
    the beginning and end STDs of the node.  The STD state (values
    of the STDs in the 0th entry and the beginning and last STDs of
    the flowgraph node) is saved.  After a node has been read, its STD list
    has the following properties:
    1.  the list is terminated by 0.
    2.  STD_NEXT(0) locates the first STD in the list.
    3.  STD_PREV(0) locates the last STD in the list.
 */
void
rdilts(int fgx)
{
  int f, l;
  int p, n;

  f = FG_STDFIRST(fgx);
  l = FG_STDLAST(fgx);
  rw_state.std0.next = STD_NEXT(0);
  rw_state.std0.prev = STD_PREV(0);
  STD_NEXT(0) = f;
  STD_PREV(0) = l;
  STD_PREV(f) = 0;
  STD_NEXT(l) = 0;
  rw_state.std_before = 0;
  p = FG_LPREV(fgx);
  if (p)
    /* search for the last std preceding f */
    for (; p; p = FG_LPREV(p))
      if (FG_STDLAST(p)) {
        rw_state.std_before = FG_STDLAST(p);
        break;
      }
  rw_state.std_after = 0;
  n = FG_LNEXT(fgx);
  if (n)
    /* search for the first std following l */
    for (; n; n = FG_LNEXT(n))
      if (FG_STDFIRST(n)) {
        rw_state.std_after = FG_STDFIRST(n);
        break;
      }
#ifdef STD_TAG
  /* if rw_state.std_before is zero, STD_TAG(0) == 0, so this is ok */
  STD_TAG(0) = STD_TAG(rw_state.std_before);
#endif
}

/** \brief 'write' the flowgraph node.

    This restores the STD list to be a single list of
    all of the STDs in the function.  The STD state is restored.
 */
void
wrilts(int fgx)
{
  int p, n;
  int f, l;

  f = STD_NEXT(0); /* 1st std of fgx */
  l = STD_PREV(0); /* last std of fgx */
#if DEBUG
  assert(STD_PREV(f) == 0, "wrilts: bad 1st std", fgx, 4);
  assert(STD_NEXT(l) == 0, "wrilts: bad last std", fgx, 4);
#endif
  if (f == 0) {
#if DEBUG
    assert(l == 0, "wrilts: bad empty node", fgx, 4);
#endif
    STD_NEXT(0) = rw_state.std0.next;
    STD_PREV(0) = rw_state.std0.prev;
    STD_NEXT(rw_state.std_before) = rw_state.std_after;
    STD_PREV(rw_state.std_after) = rw_state.std_before;
    FG_STDFIRST(fgx) = FG_STDLAST(fgx) = 0;
    return;
  }
  p = rw_state.std_before;
  if (p) {
    /* there is a prior std */
    STD_NEXT(p) = f;
    STD_PREV(f) = p;
    STD_NEXT(0) = rw_state.std0.next;
  } else
    STD_NEXT(0) = f;
  FG_STDFIRST(fgx) = f;

  n = rw_state.std_after;
  if (n) {
    /* there is a next std */
    STD_PREV(n) = l;
    STD_NEXT(l) = n;
    STD_PREV(0) = rw_state.std0.prev;
  } else
    STD_PREV(0) = l;
  FG_STDLAST(fgx) = l;
#ifdef STD_TAG
  STD_TAG(0) = 0;
#endif
}

/**
    \param stdx  std to be deleted
    \param fgx   fg of block from which std is deleted (0 => read)
    \param reuse TRUE if ilt is to be reused
*/
void
unlnkilt(int stdx, int fgx, LOGICAL reuse)
{
  int i, j;
#if DEBUG
  assert(stdx, "unlnkilt: invalid std #", stdx, 4);
#endif
  if (fgx) {
    i = STD_PREV(stdx);
    j = STD_NEXT(stdx);
    STD_PREV(j) = i;
    STD_NEXT(i) = j;
    if (FG_STDFIRST(fgx) == FG_STDLAST(fgx))
      if (STD_LABEL(stdx) && RFCNTG(STD_LABEL(stdx))) {
        i = add_stmt_after(mk_stmt(A_CONTINUE, 0), 0);
        STD_LABEL(i) = STD_LABEL(stdx);
        FG_STDFIRST(fgx) = FG_STDLAST(fgx) = i;
      } else {
        FG_STDFIRST(fgx) = FG_STDLAST(fgx) = 0;
        STD_LABEL(0) = 0;
      }
    else if (stdx == FG_STDFIRST(fgx)) {
      FG_STDFIRST(fgx) = j;
      STD_LABEL(j) = STD_LABEL(stdx);
    } else if (stdx == FG_STDLAST(fgx))
      FG_STDLAST(fgx) = i;
  } else {
    j = STD_NEXT(stdx);
    i = STD_PREV(j) = STD_PREV(stdx);
    STD_NEXT(i) = j;
  }
  STD_DELETE(stdx) = TRUE;
  if (reuse) {
    ; /* TBD: add stdx to free list */
  }

}

void
delilt(int stdx)
{
  unlnkilt(stdx, 0, TRUE);
}

void
dmpilt(int fgx)
{
  dump_node(fgx);
}

void
dump_flowgraph(void)
{

  int v, i;

  fprintf(gbl.dbgfil, "\n\n*****  Flowgraph for Function \"%s\"  *****\n",
          getprint(STD_LABEL(FG_STDFIRST(gbl.entbih))));

  dump_node(0);
  for (v = 1; v <= opt.num_nodes; v++) {
    /*
     * dump_flowgraph is called after the flowgraph has been created and
     * before any optimizations have occurred. Therefore, a node whose
     * dfn is -1 has been deleted. Later, during the optimizations, nodes
     * may be created (preheader, exits) and their dfns will be -1. These
     * nodes are not deleted.
     */
    if (FG_DFN(v) == -1) {
      fprintf(gbl.dbgfil, "%5d.  - deleted\n", v);
    } else
      dump_node(v);
  }

  fprintf(gbl.dbgfil, "\n(dfn,fg):");
  i = 0;
  for (v = 1; v <= opt.dfn; v++) {
    if (i == 6) {
      fprintf(gbl.dbgfil, "\n         ");
      i = 0;
    }
    fprintf(gbl.dbgfil, "  (%d, %d)", v, (int)VTX_NODE(v));
    i++;
  }
  fprintf(gbl.dbgfil, "\n");

  fprintf(gbl.dbgfil, "\nretreating edges:");
  i = 0;
  for (v = 0; v < opt.rteb.stg_avail; v++) {
    if (i == 5) {
      fprintf(gbl.dbgfil, "\n                 ");
      i = 0;
    }
    fprintf(gbl.dbgfil, " (%d, %d)", EDGE_PRED(v), EDGE_SUCC(v));
    i++;
  }
  fprintf(gbl.dbgfil, "\n");
}

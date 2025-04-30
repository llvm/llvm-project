/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Optimizer submodule responsible for performing flow analysis
      Used by the optimizer and vectorizer.

    NOTE: loops' store lists are "computed" during localflow (build_ud).
    To associate a names entry with a store item, the name's stl field is used
    (assumed to be initialized to zero).  At the end of processing a loop,
    the names' stl fields are cleaned up.
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#include "optimize.h"
#include "flow_util.h"
#include "pragma.h"
#include "symutl.h"
#include "fdirect.h"
#include "extern.h"
#include "rtlRtns.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

/* static variables  */

static int add_use(int, int, int, int);
static DEF *add_def(int, int, int, int, int);
static int new_storeitem(int nme);

static void localflow(void);
static void lflow_of_block(void);
static void clean_names(STL *stl);
static void build_ud(int tree);
static void build_do_init(int tree);
static void dump_global(LOGICAL inout);
static void reaching_defs(void);
static void du_ud(void);
static void add_du_ud(int def, int use);
static void chk_ptr_load(int nme);
static void add_store(int nme);

static LOGICAL const_prop(void);
static LOGICAL self_use(int use);
static LOGICAL can_prop_fg(int);
static LOGICAL can_prop_def(int);

static int def_bv_len;

static int cur_lp, cur_fg, cur_std;
static STL *cur_stl;
static int cur_callfg, cur_callinternal;
static int first_use, num_uses;
static int last_def;          /* last def entry created for a fg node */
static LOGICAL local_again;   /* subsequent call of localflow() */
static LOGICAL jumps_deleted; /* any jumps deleted by const prop? */
static LOGICAL do_lv;         /* need to compute lv; also directs flow_end
                               * to free any allocated lv space
                               */
static LOGICAL do_unv;
static int in_doinit = 0; /* 1 if creating a doinit def; ow, 0 */
static struct {
  int first;
  int last;
} g_mark_use = {0, 0};

typedef struct dilttg {
  int ilt;             /* ilt to be deleted */
  int bih;             /* block containing ilt */
  struct dilttg *next; /* next ilt item */
} DILT;
static DILT *dilt_l; /* list of ilts are deleted after localflow */

/*  Live Variable Data Structures  */

typedef struct {
  BV *lin;
  BV *lout;
  BV *luse;
  BV *ldef;
  BV *lunv;
} LV_SETS;

static struct {
  int n;           /* number of elements in each lv sets */
  int bv_len;      /* number of BV units for each bit vector */
  int bv_sz;       /* total number of BV units for all of the sets */
  BV *stg_base;    /* locates space of the flowgraphs' bit vectors */
  LV_SETS *sets;   /* locates space used for locating a nodes sets */
  BV *lp_stg_base; /* locates space for the loops' bit vectors */
  int lp_bv_sz;    /* total number of BV units for all of loop sets */
} lv;

#define LIN(v) lv.sets[v].lin
#define LOUT(v) lv.sets[v].lout
#define LUSE(v) lv.sets[v].luse
#define LDEF(v) lv.sets[v].ldef
#define LUNV(v) lv.sets[v].lunv

static void live_var_init(void);
static void live_var_end(void);
static void live_var(void);
static void lv_print(BV *);
static void uninit_var_init(void);
static void uninit_var_end(void);
static void uninit_var(void);

static LOGICAL bld_ud(int, int *); /**< pre-order visit routine called during
                                      traversal*/
static int bld_use(int, LOGICAL);  /**< visit routine for adding uses */
static DEF *bld_lhs(int, int, LOGICAL, int); /**< visit routine for the lhs of
                                                assignments */
#ifdef FLANG_FLOW_UNUSED
static void def_from_triple(int, int);
#endif

static void _cr_nme(int ast, int *dummy);
static int copy_const(int);
#ifdef FLANG_FLOW_UNUSED
static int lhs_notsubscr(int);
static void cp_cse_store(void);
#endif
static void new_ud(int astx);
static void new_du_ud(void);

static void
flow_init(void)
{

  opt.useb.stg_avail = 1;
  use_hash_alloc();
  opt.defb.stg_avail = FIRST_DEF;
  opt.storeb.stg_avail = 1;
  BZERO(opt.defb.stg_base, DEF, 4);
  DEF_ALL(0) = 0;
  DEF_FG(0) = 0;
  opt.nsyms = 0;
  opt.ndefs = FIRST_DEF - 1;
#if DEBUG
  if (flg.dbg[9]) {
    int ii;
    for (ii = 0; ii < nmeb.stg_avail; ii++)
      if (NME_STL(ii)) {
        interr("flow_init: bad stl", ii, 3);
        fprintf(gbl.dbgfil, "flow_init: bad stl,  nme:%d, rf:%d\n", ii,
                NME_STL(ii));
      }
  }
#endif

}

static void
create_nmes(void)
{
  int std;

  ast_visit(1, 1);

  for (std = STD_NEXT(0); std; std = STD_NEXT(std))
    ast_traverse((int)STD_AST(std), NULL, _cr_nme, NULL);

  ast_unvisit();

#if DEBUG
  if (OPTDBG(10, 16))
    dmpnme();
#endif
}

/* visit is in postorder */
static void
_cr_nme(int ast, int *dummy)
{
  int sym;
  int nme;
  int asd;
  int s;
  int astli;
  int sd, sda;

  switch (A_TYPEG(ast)) {
  case A_ID:
    sym = A_SPTRG(ast);
    for (s = sym; SCG(s) == SC_BASED;) {
      int n, a;
      s = MIDNUMG(s);
      if (s > NOSYM && ST_ISVAR(STYPEG(s))) {
        n = add_arrnme(NT_VAR, s, 0, (INT)0, 0, FALSE);
        a = mk_id(s);
        A_NMEP(a, n);
      } else
        break;
    }
    if (ST_ISVAR(STYPEG(sym))) {
      sd = SDSCG(sym);
      if (sd) {
        /* FS#19312: Needed for source= in sourced allocaton of bld_ud() below.
         * \
         * Shouldn't we always do this for descriptors? \
         */
        _cr_nme(mk_id(sd), dummy);
      }
      nme = add_arrnme(NT_VAR, sym, 0, (INT)0, 0, FALSE);
    } else {
      nme = NME_UNK; /* NME_UNK == 0 */
    }
    A_NMEP(ast, nme);
    break;
  case A_MEM:
    if (A_TYPEG(A_PARENTG(ast)) == A_FUNC) {
      nme = A_NMEG(A_LOPG(A_PARENTG(ast)));
    } else {
      nme = A_NMEG(A_PARENTG(ast));
    }
    sym = A_SPTRG(A_MEMG(ast));
    sd = SDSCG(sym);
    if (sd) {
      sda = check_member(ast, mk_id(sd));
      _cr_nme(sda, dummy);
    }
    nme = add_arrnme(NT_MEM, (int)PSMEMG(sym), nme, (INT)0, 0, FALSE);
    A_NMEP(ast, nme);
    break;
  case A_SUBSCR:
    nme = A_NMEG(A_LOPG(ast));
    asd = A_ASDG(ast);
    nme = add_arrnme(NT_ARR, -1, nme, (INT)0, asd, 0);
    A_NMEP(ast, nme);
    break;
  case A_FORALL:
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      int aa;
      sym = ASTLI_SPTR(astli);
      aa = mk_id(sym);
      _cr_nme(aa, dummy);
    }
    break;
  default:
    break;
  }
}

/*-----------------------------------------------------------------*/

void
flow(void)
{

  flow_init();

  create_nmes();

  dilt_l = NULL;
  local_again = FALSE; /* first call to localflow */
  do_lv = TRUE;
  do_unv = TRUE;

  live_var_init();

  localflow();

  live_var();

  live_var_end();

  if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
    /* live_var_init and localflow must be done first */
    do_lv = FALSE;
    uninit_var_init();
    uninit_var();
    uninit_var_end();
    do_lv = TRUE;
  }

  reaching_defs();
  du_ud();

again:
  jumps_deleted = FALSE;
  if (flg.opt >= 2 && const_prop()) {
    int i;

    for (i = 1; i <= opt.num_nodes; i++) {
      FG_ALL(i) = 0;
      FG_FDEF(i) = 0;
      FG_OUT(i) = FG_IN(i) = 0;
    }
    for (i = 0; i <= opt.nloops; i++) {
      LP_EXT_STORE(i) = 0;
      LP_PTR_LOAD(i) = 0;
      LP_PTR_STORE(i) = 0;
      LP_QJSR(i) = 0;
      LP_STL(i) = 0;
      LP_STL_PAR(i) = 0;
      LP_CS(i) = 0;
      LP_CSECT(i) = 0;
      LP_MASTER(i) = 0;
      LP_PARREGN(i) = 0;
      LP_PARSECT(i) = 0;
      /*
       * These are initialized by findloop:
       *     LP_NOBLA(i)
       *     LP_CALLFG(i)
       *     LP_CALLINTERNAL(i)
       *     LP_ZEROTRIP(i)
       *     LP_JMP_TBL(i)
       *     LP_PARLOOP(i)
       */
    }
    local_again = TRUE;
    if (!jumps_deleted) {
      do_lv = FALSE; /* no need to recompute live variables */
      do_unv = FALSE;
    }
    flow_end();
    freearea(STL_AREA);
    freearea(DU_AREA);
    if (jumps_deleted) {
      freearea(PSI_AREA);

      flowgraph(); /* build the flowgraph for the function */

      findloop(HLOPT_ALL); /* find the loops */
    }
#if DEBUG
    if (OPTDBG(29, 32)) {
      fprintf(gbl.dbgfil, "***** After const prop *****\n");
      dump_flowgraph();
      dump_loops();
    }
    if (DBGBIT(29, 64)) {
      int jj;
      jj = gbl.entbih;
      fprintf(gbl.dbgfil,
              "\n***** BIHs for Function \"%s\" After const prop *****\n",
              getprint((int)FG_LABEL(jj)));
      for (;;) {
        dmpilt(jj);
        if (FG_XT(jj))
          break;
        jj = FG_LNEXT(jj);
      }
    }
#endif

    flow_init();

    if (do_lv)
      live_var_init();

    localflow();

    if (do_lv) {
      live_var();
      live_var_end();
    }
    if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
      /* relies on data structures in live_var_init */
      LOGICAL tdo_lv = do_lv;
      do_lv = FALSE;
      if (do_unv && tdo_lv) {
        uninit_var_init();
        uninit_var();
        uninit_var_end();
      }
      do_lv = tdo_lv;
    }
    reaching_defs();
    du_ud();
    if (jumps_deleted) {
      /* since the flowgraph had to be rebuilt it's possible that
       * additional const prop opportunities are available.
       * This is because that certain defs could have occurred
       * in what was determined to be dead blocks.
       */
      do_lv = TRUE;
      do_unv = TRUE;
      goto again;
    }
  }
  do_lv = TRUE; /* so flow_end() functions properly */
  do_unv = TRUE;
  local_again = FALSE;

  /* delete any ilts saved for deletion during localflow() */

  if (dilt_l != NULL) {
    do {
      unlnkilt((int)dilt_l->ilt, (int)dilt_l->bih, TRUE);
      dilt_l = dilt_l->next;
    } while (dilt_l != NULL);
    freearea(DLT_AREA);
  }

#if DEBUG
  if (flg.dbg[9]) {
    int ii;
    for (ii = 0; ii < nmeb.stg_avail; ii++)
      if (NME_STL(ii)) {
        interr("flow: bad stl", ii, 3);
        fprintf(gbl.dbgfil, "flow: bad stl,  nme:%d, rf:%d\n", ii, NME_STL(ii));
      }
  }
#endif

#if DEBUG
  if (DBGBIT(9, 128))
    dump_global(TRUE);
#endif

}

/** \brief Cleanup up structures created by flow analysis and after a function
    has been optimized.
 */
void
flow_end(void)
{
  int dfx, nme;

  for (dfx = FIRST_DEF; dfx <= opt.ndefs; dfx++) {
    nme = DEF_NM(dfx);
    NME_DEF(nme) = 0;
  }
#if DEBUG
  for (nme = 0; nme < nmeb.stg_avail; nme++)
    if (NME_DEF(nme))
      interr("flow_end: bad rfptr", nme, 1);
#endif

  FREE(opt.def_setb.stg_base); /* allocated in globalflow  */
  if (do_lv) {
    FREE(lv.lp_stg_base);
    FREE(opt.def_setb.nme_base);
  }
  use_hash_free();
}

#if DEBUG
static void
prntstl(STL *p)
{
  static int indent;
  int i;

  for (i = 0; i < indent; i++)
    fprintf(gbl.dbgfil, " ");
  fprintf(gbl.dbgfil, "stl addr %p  store %d\n", (void *)p, p->store);
  {
    int ii;
    for (ii = p->store; ii; ii = STORE_NEXT(ii)) {
      for (i = 0; i < indent; i++)
        fprintf(gbl.dbgfil, " ");
      fprintf(gbl.dbgfil, "--%5d. nme:", ii);
      dumpname((int)STORE_NM(ii));
      fprintf(gbl.dbgfil, ", mark: %d\n", STORE_TYPE(ii));
    }
  }
  indent += 3;
  for (p = p->childlst; p != NULL; p = p->nextsibl)
    prntstl(p);
  indent -= 3;
}
#endif

static void
localflow(void)
{
  int i, parent;
  STL *p;
  int nxtfg;

  /* first allocate an stl for each loop  + region 0  hence i = 0 not 1 */
  for (i = 0; i <= opt.nloops; i++) {
    int tmp;
    if (i == 0)
      tmp = 0;
    else
      tmp = LP_LOOP(i);
    cur_stl = (STL *)getitem(STL_AREA, sizeof(STL));
    LP_STL(tmp) = cur_stl;
    cur_stl->store = 0;
    cur_stl->nextsibl = 0;
    cur_stl->childlst = 0;
  }

#if DEBUG
  {
    int ii;
    for (ii = 1; ii < astb.stg_avail; ii++)
      if (A_VISITG(ii)) {
        interr("local_flow: ast visited", ii, 3);
        A_VISITP(ii, 0);
      }
  }
#endif
  for (i = 1; i <= opt.nloops; i++) {
    cur_lp = LP_LOOP(i);
    cur_stl = LP_STL(cur_lp);
    /* link me into my parent's childlst */
    parent = LP_PARENT(cur_lp);
    p = LP_STL(parent);
    cur_stl->nextsibl = p->childlst;
    p->childlst = cur_stl;
  }

  for (cur_fg = gbl.entbih; cur_fg; cur_fg = nxtfg) {
    nxtfg = FG_LNEXT(cur_fg); /* just in case cur_fg is changed */
    lflow_of_block();
  }

  for (i = 1; i <= opt.nloops; i++) {
    cur_lp = LP_LOOP(i);
    parent = LP_PARENT(cur_lp);
    LP_EXT_STORE(parent) |= LP_EXT_STORE(cur_lp);
    LP_PTR_STORE(parent) |= LP_PTR_STORE(cur_lp);
    LP_PTR_LOAD(parent) |= LP_PTR_LOAD(cur_lp);
    LP_CALLFG(parent) |= LP_CALLFG(cur_lp);
    LP_CALLINTERNAL(parent) |= LP_CALLINTERNAL(cur_lp);
    LP_QJSR(parent) |= LP_QJSR(cur_lp);
    LP_NOBLA(parent) |= LP_NOBLA(cur_lp);
    LP_JMP_TBL(parent) |= LP_JMP_TBL(cur_lp);
    LP_CSECT(parent) |= LP_CSECT(cur_lp);
    LP_PARREGN(parent) |= LP_PARREGN(cur_lp);
    LP_PARSECT(parent) |= LP_PARSECT(cur_lp);
  }

#if DEBUG
  if (OPTDBG(9, 64))
    prntstl(LP_STL(0));
#endif

}

static void
lflow_of_block(void)
{
  if (FG_DFN(cur_fg) == -1)
    return;
  cur_lp = FG_LOOP(cur_fg);
  cur_stl = LP_STL(cur_lp);
  if (cur_lp)
    /* apply loop-scoped pragmas available for this loop */
    open_pragma((int)FG_LINENO(LP_HEAD(cur_lp)));

#if DEBUG
  if (OPTDBG(9, 64))
    fprintf(gbl.dbgfil, "\n---------- local flow trace of node %d, loop %d\n",
            cur_fg, cur_lp);
  if (OPTDBG(9, 64))
    fprintf(gbl.dbgfil, "   flow graph node %d\n", cur_fg);
#endif
  /*
   * Use 0'th def entry as the head of the list of defs, linked
   * together via the lnext field, created for the current flowgraph
   * node.  last_def is set to the 0'th entry; when build_ud is
   * done for this node, DEF_LNEXT(0) (possibly 0 => "no defs") is
   * the list of defs for the node.  NOTE that the defs in the
   * list are in lexical order.
   */
  last_def = 0;
  DEF_LNEXT(0) = 0;
  first_use = opt.useb.stg_avail;
  num_uses = 0;
  use_hash_reset();
  if (FG_CS(cur_fg)) {
    LP_CS(cur_lp) = 1;
    LP_CSECT(cur_lp) = 1;
  }
  if (FG_TASK(cur_fg))
    LP_TASK(cur_lp) = 1;
  if (FG_MASTER(cur_fg))
    LP_MASTER(cur_lp) = 1;
  if (FG_PAR(cur_fg))
    LP_PARREGN(cur_lp) = 1;
  if (FG_PARSECT(cur_fg))
    LP_PARSECT(cur_lp) = 1;
  rdilts(cur_fg);
  for (cur_std = FG_STDFIRST(cur_fg); cur_std; cur_std = STD_NEXT(cur_std)) {
#if DEBUG
    if (OPTDBG(9, 64))
      fprintf(gbl.dbgfil, "      std %d\n", cur_std);
#endif
    build_ud((int)STD_AST(cur_std));
    STD_EX(cur_std) = cur_callfg;
    A_CALLFGP(STD_AST(cur_std), cur_callfg);
    FG_EX(cur_fg) |= STD_EX(cur_std);
    LP_CALLFG(cur_lp) |= FG_EX(cur_fg);
    LP_CALLINTERNAL(cur_lp) |= cur_callinternal;
  }
  wrilts(cur_fg);

  if (cur_lp)
    close_pragma();

  if (!FG_XT(cur_fg)) {
    int nxtfg;

    nxtfg = FG_LNEXT(cur_fg);
    cur_std = FG_STDFIRST(nxtfg);
    if (cur_std) {
      switch (A_TYPEG(STD_AST(cur_std))) {
      case A_DO:
#ifdef A_MP_PDO
      case A_MP_PDO:
#endif
      case A_FORALL:
        build_do_init((int)STD_AST(cur_std));
        break;
      default:
        break;
      }
    }
  }

  FG_FDEF(cur_fg) = DEF_LNEXT(0);
  clean_names(cur_stl);

}

static void
clean_names(STL *stl)
{
  int store;

  for (store = stl->store; store; store = STORE_NEXT(store))
    NME_STL(STORE_NM(store)) = 0;
}

static void
build_ud(int tree)
{
  /* setup to record list of nodes visited;  use nonzero value
   * to denote that an ili has been visited AND the next ili
   * in the list.  astb.i0 represents the end of the list; must start
   * out with this ili as being visited.
   */
  ast_visit(1, 1);
  cur_callfg = 0;
  cur_callinternal = 0;
  ast_traverse(tree, bld_ud, NULL, NULL);
  ast_unvisit();

}

static LOGICAL
bld_ud(int ast, int *dummy)
{
  int atype;
  int sym;
  int i, j;
  int astli;
  int argt;
  int cnt;
  int top; /* ast of matching do/forall of enddo/endforall */
  int opnd, baseopnd;
  int dtype;
  int u;
  DEF *df;
  int idpdsc, iface;
  int intent;
  int parmcnt;
  int templatecall;

  cur_callfg |= A_CALLFGG(ast);
  switch (atype = A_TYPEG(ast)) {
  case A_ID:
  case A_MEM:
  case A_SUBSCR:
  case A_SUBSTR:
    (void)bld_use(ast, FALSE);
    return TRUE;
  case A_ICALL:
    ast_traverse((int)A_LOPG(ast), bld_ud, NULL, NULL);
    argt = A_ARGSG(ast);
    switch (A_OPTYPEG(ast)) {
    case I_ALL:
    case I_ANY:
    case I_COUNT:
    case I_PRODUCT:
    case I_SUM:
    case I_FINDLOC:
    case I_MAXVAL:
    case I_MINVAL:
    case I_CSHIFT:
    case I_DOT_PRODUCT:
    case I_EOSHIFT:
    case I_MAXLOC:
    case I_MINLOC:
    case I_PACK:
    case I_RESHAPE:
    case I_SPREAD:
    case I_TRANSPOSE:
    case I_UNPACK:
    case I_TRANSFER:
    case I_REDUCE_SUM:
    case I_REDUCE_PRODUCT:
    case I_REDUCE_MAXVAL:
    case I_REDUCE_MINVAL:
    case I_REDUCE_ALL:
    case I_REDUCE_IALL:
    case I_REDUCE_ANY:
    case I_REDUCE_IANY:
    case I_REDUCE_PARITY:
    case I_REDUCE_IPARITY:
      /* for these intrinsic calls, the first argument is intent inout.
       * the remaining arguments are uses
       */
      cnt = A_ARGCNTG(ast);
      if (cnt == 0)
        break;
      opnd = ARGT_ARG(argt, 0);
      if (opnd) {
        u = bld_use(opnd, TRUE);
        (void)bld_lhs(opnd, ast, TRUE, 1);
      }

      /* the remaining arguments are uses */
      for (i = 1; i < cnt; i++) {
        opnd = ARGT_ARG(argt, i);
        if (opnd) {
          switch (A_TYPEG(opnd)) {
          case A_ID:
          case A_MEM:
          case A_SUBSCR:
          case A_SUBSTR:
            /* for those ASTs which are lvalues, create use items
             * now and mark as 'arg' uses.
             */
            u = bld_use(opnd, TRUE);
            break;
          default:
            ast_traverse(opnd, bld_ud, NULL, NULL);
            break;
          }
        }
      }
      break;

    case I_MVBITS:
    default:
      idpdsc = 0;
      parmcnt = 0;
      templatecall = 0;
      goto like_call;
    }
    return TRUE;

  case A_CALL:
  case A_FUNC:
    sym = procsym_of_ast(A_LOPG(ast));
    proc_arginfo(sym, NULL, &idpdsc, &iface);
    parmcnt = 0;
    templatecall = 0;
    if (iface) {
      parmcnt = PARAMCTG(iface);
      if (INTERNALG(iface)) {
        cur_callinternal = 1;
      }
      if (HCCSYMG(iface) && NODESCG(iface) &&
          getF90TmplSectRtn(SYMNAME(iface))) {
        templatecall = 1;
      }
    }
  like_call:
    cur_callfg = 1;
    ast_traverse((int)A_LOPG(ast), bld_ud, NULL, NULL);
    argt = A_ARGSG(ast);
    j = 0;
    for (i = 0, cnt = A_ARGCNTG(ast); cnt; cnt--) {
      int arg = 0;
      intent = INTENT_DFLT;
      if (templatecall) {
        if (i == 0) {
          intent = INTENT_OUT;
        } else {
          intent = INTENT_IN;
        }
      } else if (idpdsc) {
        if (i < parmcnt) {
          intent = INTENTG(aux.dpdsc_base[idpdsc + i]);
        } else {
          /* this is a descriptor argument for some fortran argument */
          while (j < parmcnt && !needs_descriptor(aux.dpdsc_base[idpdsc + j]))
            ++j;
          if (j < parmcnt) {
            arg = aux.dpdsc_base[idpdsc + j];
            /* if this is a F90 pointer or HPF dynamic argument,
             * the descriptor may be changed by the call */
            if (!POINTERG(arg)) {
              intent = INTENT_IN;
            }
          }
        }
      }
      opnd = ARGT_ARG(argt, i);
      if (opnd && intent != INTENT_OUT)
        switch (A_TYPEG(opnd)) {
        case A_ID:
        case A_MEM:
        case A_SUBSCR:
        case A_SUBSTR:
          u = bld_use(opnd, TRUE);
          break;
        default:
          ast_traverse(opnd, bld_ud, NULL, NULL);
          break;
        }
      if (opnd && A_ISLVAL(A_TYPEG(opnd)) && intent != INTENT_IN)
        (void)bld_lhs(opnd, ast, TRUE, 1);
      ++i;
    }
    return TRUE;

  case A_ASN:
    (void)bld_lhs((int)A_DESTG(ast), (int)A_SRCG(ast), FALSE, 1);
    return TRUE;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    (void)bld_lhs((int)A_LOPG(ast), (int)A_ROPG(ast), FALSE, 1);
    return TRUE;
  case A_MP_ATOMICREAD:
    (void)bld_use(A_SRCG(ast), FALSE);
    return TRUE;
  case A_ASNGOTO:
    df = bld_lhs((int)A_DESTG(ast), (int)A_SRCG(ast), FALSE, 1);
    df->flags.bits.other = 1;
    return TRUE;
  case A_ENDDO:
#ifdef A_MP_ENDPDO
  case A_MP_ENDPDO:
#endif
    top = A_OPT2G(ast);
    if (A_TYPEG(top) == A_DOWHILE)
      break;
#if DEBUG
#ifdef A_MP_PDO
    assert(A_TYPEG(top) == A_DO || A_TYPEG(top) == A_MP_PDO,
           "bld_ud:mismatched enddo", ast, 3);
#else
    assert(A_TYPEG(top) == A_DO, "bld_ud:mismatched enddo", ast, 3);
#endif
#endif
    ast = A_DOVARG(top);
    dtype = A_DTYPEG(ast);
    if (A_M3G(top))
      cnt = A_M3G(top);
    else if (dtype == DT_REAL4)
      cnt = mk_cnst(stb.flt1);
    else if (dtype == DT_DBLE)
      cnt = mk_cnst(stb.dbl1);
    else
      cnt = astb.i1;
    cnt = mk_binop(OP_ADD, ast, cnt, dtype);
    df = bld_lhs(ast, cnt, FALSE, 1);
    df->flags.bits.doend = 1;
    return TRUE;
  case A_ALLOC:
    opnd = A_SRCG(ast);
    /* If the symbol being allocated has a section descriptor,
     * add a def of the section descriptor here also */
    baseopnd = opnd;
    while (A_TYPEG(baseopnd) == A_SUBSCR) {
      baseopnd = A_LOPG(baseopnd);
    }
    if (A_TYPEG(baseopnd) == A_ID) {
      int sd;
      sym = A_SPTRG(baseopnd);
      sd = SDSCG(sym);
      if (sd) {
        int n, sda;
        sda = mk_id(sd);
        n = A_NMEG(sda);
        df = bld_lhs(sda, ast, FALSE, 0);
        if (df)
          df->flags.bits.other = 1;
      }
    } else if (A_TYPEG(baseopnd) == A_MEM) {
      int sd;
      sym = A_SPTRG(A_MEMG(baseopnd));
      sd = SDSCG(sym);
      if (sd) {
        int n, sda;
        sda = check_member(baseopnd, mk_id(sd));
        n = A_NMEG(sda);
        df = bld_lhs(sda, ast, FALSE, 0);
        if (df)
          df->flags.bits.other = 1;
      }
    }
    df = bld_lhs(opnd, ast, FALSE, 1);
    if (df)
      df->flags.bits.other = 1;
    opnd = A_LOPG(ast);
    if (opnd) {
      df = bld_lhs(opnd, ast, FALSE, 1);
      if (df)
        df->flags.bits.other = 1;
    }
    opnd = A_DESTG(ast);
    if (opnd) {
      df = bld_lhs(opnd, ast, FALSE, 1);
      if (df)
        df->flags.bits.other = 1;
    }
    opnd = A_M3G(ast);
    if (opnd) {
      df = bld_lhs(opnd, ast, FALSE, 1);
      if (df)
        df->flags.bits.other = 1;
    }
    opnd = A_STARTG(ast);
    if (opnd) {
      /* FS#19312: Check for section descriptor in the source=
       * input argument. If one is present, add a def of the
       * section descriptor.
       */
      baseopnd = opnd;
      while (A_TYPEG(baseopnd) == A_SUBSCR) {
        baseopnd = A_LOPG(baseopnd);
      }
      if (A_TYPEG(baseopnd) == A_ID) {
        int sd;
        sym = A_SPTRG(baseopnd);
        sd = SDSCG(sym);
        if (sd) {
          int sda;
          sda = mk_id(sd);
          df = bld_lhs(sda, ast, FALSE, 0);
          if (df)
            df->flags.bits.other = 1;
        }
      } else if (A_TYPEG(baseopnd) == A_MEM) {
        int sd;
        sym = A_SPTRG(A_MEMG(baseopnd));
        sd = SDSCG(sym);
        if (sd) {
          int sda;
          sda = check_member(baseopnd, mk_id(sd));
          df = bld_lhs(sda, ast, FALSE, 0);
          if (df)
            df->flags.bits.other = 1;
        }
      }
    }
    return TRUE;
  case A_FORALL:
    LP_FORALL(cur_lp) = 1;
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      ast_traverse((int)ASTLI_TRIPLE(astli), bld_ud, NULL, NULL);
    if (A_IFEXPRG(ast))
      ast_traverse((int)A_IFEXPRG(ast), bld_ud, NULL, NULL);
    if (A_IFSTMTG(ast)) {
      ast_traverse((int)A_IFSTMTG(ast), bld_ud, NULL, NULL);
      goto endforall_shared;
    }
    return TRUE;
  case A_ENDFORALL:
    top = A_OPT2G(ast);
#if DEBUG
    assert(A_TYPEG(top) == A_FORALL, "bld_ud:mismatched endforall", ast, 3);
#endif
    ast = top;
  endforall_shared:
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      ast = mk_id((int)ASTLI_SPTR(astli));
      cnt = ASTLI_TRIPLE(astli);
      if (A_STRIDEG(cnt))
        cnt = A_STRIDEG(cnt);
      else
        cnt = astb.i1;
      cnt = mk_binop(OP_ADD, ast, cnt, DT_INT);
      df = bld_lhs(ast, cnt, FALSE, 1);
      df->flags.bits.doend = 1;
    }
    return TRUE;
  case A_REDIM:
    opnd = A_SRCG(ast);
    df = bld_lhs(opnd, ast, FALSE, 1);
    df->flags.bits.other = 1;
    return TRUE;
  case A_PRAGMA:
    /* treat appearances in ACC pragmas as uses */
    switch (A_PRAGMATYPEG(ast)) {
    case PR_ACCCOPYIN:
    case PR_ACCCOPYOUT:
    case PR_ACCLOCAL:
    case PR_ACCELLP:
    case PR_ACCVECTOR:
    case PR_ACCPARALLEL:
    case PR_ACCSEQ:
    case PR_ACCHOST:
    case PR_ACCPRIVATE:
    case PR_ACCCACHE:
    case PR_ACCSHORTLOOP:
    case PR_ACCBEGINDIR:
    case PR_ACCIF:
    case PR_ACCUNROLL:
    case PR_ACCKERNEL:
    case PR_ACCCOPY:
    case PR_ACCDATAREG:
    case PR_ACCENDDATAREG:
    case PR_ACCUPDATEHOST:
    case PR_ACCUPDATEDEVICE:
    case PR_ACCUPDATE:
    case PR_ACCINDEPENDENT:
    case PR_ACCWAIT:
    case PR_ACCNOWAIT:
    case PR_ACCIMPDATAREG:
    case PR_ACCENDIMPDATAREG:
    case PR_ACCMIRROR:
    case PR_ACCREFLECT:
    case PR_KERNELBEGIN:
    case PR_KERNEL:
    case PR_ENDKERNEL:
    case PR_KERNELTILE:
    case PR_ACCDEVSYM:
    case PR_ACCIMPDATAREGX:
    case PR_KERNEL_NEST:
    case PR_KERNEL_GRID:
    case PR_KERNEL_BLOCK:
    case PR_ACCDEVICEPTR:
    case PR_ACCPARUNROLL:
    case PR_ACCVECUNROLL:
    case PR_ACCSEQUNROLL:
    case PR_ACCCUDACALL:
    case PR_ACCSCALARREG:
    case PR_ACCENDSCALARREG:
    case PR_ACCPARCONSTRUCT:
    case PR_ACCENDPARCONSTRUCT:
    case PR_ACCKERNELS:
    case PR_ACCENDKERNELS:
    case PR_ACCSERIAL:
    case PR_ACCENDSERIAL:
    case PR_ACCCREATE:
    case PR_ACCPRESENT:
    case PR_ACCPCOPY:
    case PR_ACCPCOPYIN:
    case PR_ACCPCOPYOUT:
    case PR_ACCPCREATE:
    case PR_ACCATTACH:
    case PR_ACCDETACH:
    case PR_ACCASYNC:
    case PR_KERNEL_STREAM:
    case PR_KERNEL_DEVICE:
    case PR_ACCWAITDIR:
    case PR_ACCKLOOP:
    case PR_ACCTKLOOP:
    case PR_ACCPLOOP:
    case PR_ACCTPLOOP:
    case PR_ACCSLOOP:
    case PR_ACCTSLOOP:
    case PR_ACCGANG:
    case PR_ACCGANGDIM:
    case PR_ACCGANGCHUNK:
    case PR_ACCWORKER:
    case PR_ACCFIRSTPRIVATE:
    case PR_ACCNUMGANGS:
    case PR_ACCNUMGANGS2:
    case PR_ACCNUMGANGS3:
    case PR_ACCNUMWORKERS:
    case PR_ACCVLENGTH:
    case PR_ACCWAITARG:
    case PR_ACCREDUCTION:
    case PR_ACCREDUCTOP:
    case PR_ACCCACHEDIR:
    case PR_ACCCACHEREADONLY:
    case PR_ACCCACHEARG:
    case PR_ACCHOSTDATA:
    case PR_ACCENDHOSTDATA:
    case PR_ACCUSEDEVICE:
    case PR_ACCUSEDEVICEIFP:
    case PR_ACCCOLLAPSE:
    case PR_ACCFORCECOLLAPSE:
    case PR_ACCDEVICERES:
      return FALSE;
    }
    return TRUE; /* don't count appearances in pragmas as uses */
  default:
    break;
  }

  return FALSE;
}

static int
bld_use(int lval, LOGICAL isarg)
{
  int ast;
  int sym, nme;
  int i, asd;
  int precise;
  int s;
  int u;

  ast = lval;
  precise = 1;
  u = 0;
again:
  switch (A_TYPEG(ast)) {
  case A_CONV:
    ast = A_LOPG(ast);
    goto again;
  case A_ID:
    sym = A_SPTRG(ast);
    for (s = sym; SCG(s) == SC_BASED;) {
      int n, a;
      s = MIDNUMG(s);
      if (s > NOSYM && ST_ISVAR(STYPEG(s))) {
        a = mk_id(s);
        n = A_NMEG(a);
        u = add_use(n, a, a, TRUE);
        USE_ARG(u) |= isarg;
#if DEBUG
        if (OPTDBG(9, 64))
          fprintf(
              gbl.dbgfil,
              "         ptrast %d, addr %d: use of nme %d, ulist %d \"%s\"\n",
              a, a, n, u, getprint((int)NME_SYM(n)));
#endif
        chk_ptr_load(n);
      } else
        break;
    }
    if (ST_ISVAR(STYPEG(sym))) {
      nme = A_NMEG(ast);
      if (POINTERG(sym))
        chk_ptr_load(nme);
      if (isarg && ast != lval && SEQG(sym))
        /* an aggregate reference or subscripted array as an argument
         * where the object is marked sequential.
         */
        u = add_use(nme, ast, lval, 0);
      else
        u = add_use(nme, ast, lval, precise);
      USE_ARG(u) |= isarg;
#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil,
                "         ast %d, addr %d: use of nme %d, ulist %d \"%s\"\n",
                ast, lval, nme, u, getprint((int)NME_SYM(nme)));
#endif
    }
    break;
  case A_MEM:
    nme = A_NMEG(ast);
    ast = A_PARENTG(ast);
    if (NME_TYPE(nme) != NT_MEM) {
      lval = ast;
      precise = 0;
    } else if (POINTERG(NME_SYM(nme))) {
      chk_ptr_load(nme);
/* ADDRTKN hack - be conservative with the parent symbol;
 * the parent's nme is the one which is actually recorded
 * in the USE and DEF structures.
 */
#ifdef PTRSTOREP
      ptrstore_of(nme);
#else
      loc_of(nme);
#endif
    }
    goto again;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    for (i = 0; i < (int)ASD_NDIM(asd); i++) {
      ast_traverse((int)ASD_SUBS(asd, i), bld_ud, NULL, NULL);
      if (A_ALIASG(ASD_SUBS(asd, i)) == 0)
        precise = 0;
    }
    /* build a use of the section descriptor */
    sym = memsym_of_ast(ast);
    if ((POINTERG(sym) || ALLOCG(sym)) && SDSCG(sym)) {
      int a;
      a = check_member(A_LOPG(ast), mk_id(SDSCG(sym)));
      u = add_use(A_NMEG(a), a, a, TRUE);
      USE_ARG(u) |= isarg;
#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil,
                "         sdsc   %d, addr %d: use of nme %d, ulist %d \"%s\"\n",
                a, a, A_NMEG(a), u, getprint((int)NME_SYM(A_NMEG(a))));
#endif
    }
    ast = A_LOPG(ast);
    goto again;
  case A_SUBSTR:
    if (A_LEFTG(ast)) {
      ast_traverse((int)A_LEFTG(ast), bld_ud, NULL, NULL);
    }
    if (A_RIGHTG(ast)) {
      ast_traverse((int)A_RIGHTG(ast), bld_ud, NULL, NULL);
    }
    precise = 0;
    ast = A_LOPG(ast);
    goto again;
  case A_FUNC:
    break;
  case A_CNST:
    if (!XBIT(7, 0x100000))
      break;
    FLANG_FALLTHROUGH;
  default:
    interr("bld_use, ast nyd", ast, 3);
  }
  return u;
}

static DEF *
bld_lhs(int lhs, int rhs, LOGICAL isarg, int precise)
{
  int ast;
  int sym, nme;
  int i, asd;
  DEF *df = NULL;
  int mark_use; /* to mark beginning of uses found in a rhs */
  int s;

  mark_use = opt.useb.stg_avail;
  ast_traverse(rhs, bld_ud, NULL, NULL);
  ast = lhs;

again:
  switch (A_TYPEG(ast)) {
  case A_ID:
    sym = A_SPTRG(ast);
    if (!ST_ISVAR(STYPEG(sym)))
      return NULL;
    for (s = sym; SCG(s) == SC_BASED;) {
      int n, a;
      s = MIDNUMG(s);
      if (s > NOSYM && ST_ISVAR(STYPEG(s))) {
        a = mk_id(s);
        n = A_NMEG(a);
        i = add_use(n, a, a, TRUE);
#if DEBUG
        if (OPTDBG(9, 64))
          fprintf(
              gbl.dbgfil,
              "         ptrast %d, addr %d: use of nme %d, ulist %d \"%s\"\n",
              a, a, n, i, getprint((int)NME_SYM(n)));
#endif
        chk_ptr_load(n);
      } else
        break;
    }
    nme = A_NMEG(ast);
    if (POINTERG(sym))
      chk_ptr_load(nme);
    add_store(nme);
    df = add_def(nme, ast, lhs, rhs, precise);

    /* if the def is due to an appearance as an actual argument, don't
     * turn off the GEN flag of any defs which precede this point.
     */
    if (!isarg) {
      i = df->next;
      if (precise && ast == lhs && /* def defines the whole variable  */
          DEF_FG(i) == cur_fg)     /* i may be zero, but flags of  */
        DEF_GEN(i) = 0;            /* DEF 0 are zero               */
      else {
        for (; i && DEF_FG(i) == cur_fg; i = DEF_NEXT(i))
          if (precise && DEF_LHS(i) == ast && DEF_ADDR(i) == lhs)
            DEF_GEN(i) = 0;
      }
    } else
      df->flags.bits.arg = 1;

    df->flags.bits.gen = 1;
    if (A_ALIASG(rhs))
      df->flags.bits.cnst = 1;
    if (g_mark_use.first)
      for (i = g_mark_use.first; i <= g_mark_use.last; i++)
        if (USE_NM(i) == nme) {
          df->flags.bits.self = 1;
          break;
        }
    while (mark_use < opt.useb.stg_avail) {
      if (USE_NM(mark_use) == nme) {
        df->flags.bits.self = 1;
        break;
      }
      mark_use++;
    }
#if DEBUG
    if (OPTDBG(9, 64))
      fprintf(gbl.dbgfil, "         lhs %d: store of nme %d \"%s\"\n", lhs, nme,
              getprint((int)NME_SYM(nme)));
#endif
    if ((FG_CS(cur_fg) || FG_PARSECT(cur_fg) || FG_PAR(cur_fg)) &&
        !IS_PRIVATE(NME_SYM(nme))) {
      Q_ITEM *q;
      int ll;

#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil, "         store of nonprivate %s\n",
                getprint((int)NME_SYM(nme)));
#endif
      /*
       * For this loop and all enclosing loops, add this variable
       * (its nme) to the list of non-private variables which were
       * stored while in a critical section, a parallel section, or
       * parallel region.
       */
      for (ll = cur_lp;; ll = LP_PARENT(ll)) {
        for (q = LP_STL_PAR(ll); q != NULL; q = q->next)
          if (q->info == nme)
            goto nme_in_par;

        q = (Q_ITEM *)getitem(STL_AREA, sizeof(Q_ITEM));
        q->info = nme;
        q->next = LP_STL_PAR(ll);
        LP_STL_PAR(ll) = q;
      nme_in_par:
        if (ll == 0)
          break;
      }
    }
    break;
  case A_MEM:
    nme = A_NMEG(ast);
    add_store((int)A_NMEG(ast));
    ast = A_PARENTG(ast);
    if (NME_TYPE(nme) != NT_MEM) {
      lhs = ast;
      precise = 0;
    } else if (POINTERG(NME_SYM(nme))) {
      chk_ptr_load(nme);
      if (!is_ptr_safe(nme)) {
        LP_PTR_STORE(cur_lp) = 1;
        FG_PTR_STORE(cur_fg) = 1;
      }
/* ADDRTKN hack - be conservative with the parent symbol;
 * the parent's nme is the one which is actually recorded
 * in the USE and DEF structures.
 */
#ifdef PTRSTOREP
      ptrstore_of(nme);
#else
      loc_of(nme);
#endif
    }
    goto again;
  case A_SUBSCR:
    add_store((int)A_NMEG(ast));
    asd = A_ASDG(ast);
    for (i = 0; i < (int)ASD_NDIM(asd); i++) {
      ast_traverse((int)ASD_SUBS(asd, i), bld_ud, NULL, NULL);
      if (A_ALIASG(ASD_SUBS(asd, i)) == 0)
        precise = 0;
    }
    /* build a use of the section descriptor */
    sym = memsym_of_ast(ast);
    if ((POINTERG(sym) || ALLOCG(sym)) && SDSCG(sym)) {
      int a, u;
      a = check_member(A_LOPG(ast), mk_id(SDSCG(sym)));
      u = add_use(A_NMEG(a), a, a, TRUE);
      USE_ARG(u) |= isarg;
#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil,
                "         sdsc   %d, addr %d: use of nme %d, ulist %d \"%s\"\n",
                a, a, A_NMEG(a), u, getprint((int)NME_SYM(A_NMEG(a))));
#endif
    }
    ast = A_LOPG(ast);
    goto again;
  case A_SUBSTR:
    if (A_LEFTG(ast)) {
      ast_traverse((int)A_LEFTG(ast), bld_ud, NULL, NULL);
    }
    if (A_RIGHTG(ast)) {
      ast_traverse((int)A_RIGHTG(ast), bld_ud, NULL, NULL);
    }
    precise = 0;
    ast = A_LOPG(ast);
    goto again;
  case A_CNST:
    if (!XBIT(7, 0x100000))
      break;
    FLANG_FALLTHROUGH;
  default:
    interr("bld_lhs, ast nyd", ast, 3);
    df = NULL;
  }
  return df;
}

#ifdef FLANG_FLOW_UNUSED
/* stmt ast containing the triple whose fields are
 * being defined */
static void
def_from_triple(int stmt, int triple)
{
  DEF *df;

  if (triple == 0)
    return;
  if (A_LBDG(triple)) {
    df = bld_lhs((int)A_LBDG(triple), stmt, FALSE, 1);
    if (df)
      df->flags.bits.other = 1;
  }
  if (A_UPBDG(triple)) {
    df = bld_lhs((int)A_UPBDG(triple), stmt, FALSE, 1);
    if (df)
      df->flags.bits.other = 1;
  }
  if (A_STRIDEG(triple)) {
    df = bld_lhs((int)A_STRIDEG(triple), stmt, FALSE, 1);
    if (df)
      df->flags.bits.other = 1;
  }
}
#endif

static void
build_do_init(int tree)
{
  DEF *df;
  int ast;
  int astli;

  in_doinit = 1;
  ast_visit(1, 1);

  switch (A_TYPEG(tree)) {
  case A_DO:
#ifdef A_MP_PDO
  case A_MP_PDO:
#endif
    g_mark_use.first = opt.useb.stg_avail;
    ast_traverse((int)A_M1G(tree), bld_ud, NULL, NULL);
    g_mark_use.last = opt.useb.stg_avail - 1;
    ast_traverse((int)A_M2G(tree), bld_ud, NULL, NULL);
    if (A_M3G(tree))
      ast_traverse((int)A_M3G(tree), bld_ud, NULL, NULL);
    if (A_TYPEG(tree) == A_DO) {
      if (A_M4G(tree))
        ast_traverse((int)A_M4G(tree), bld_ud, NULL, NULL);
    }
#ifdef A_MP_PDO
    if (A_TYPEG(tree) == A_MP_PDO) {
      if (A_LASTVALG(tree))
        ast_traverse((int)A_LASTVALG(tree), bld_ud, NULL, NULL);
    }
#endif
    df = bld_lhs((int)A_DOVARG(tree), (int)A_M1G(tree), FALSE, 1);
    df->flags.bits.doinit = 1;
    g_mark_use.first = 0;
    g_mark_use.last = 0;
    break;
  case A_FORALL:
    for (astli = A_LISTG(tree); astli; astli = ASTLI_NEXT(astli)) {
      ast = ASTLI_TRIPLE(astli);
      ast_traverse(ast, bld_ud, NULL, NULL);
      if (A_LBDG(ast))
        ast = A_LBDG(ast);
      else
        ast = astb.i1;
      df = bld_lhs((int)mk_id((int)ASTLI_SPTR(astli)), ast, FALSE, 1);
      df->flags.bits.doinit = 1;
    }
    break;
  default:
    interr("build_do_init, ast nyd", tree, 3);
    break;
  }
  ast_unvisit();
  in_doinit = 0;

}

static int
add_use(int nme, int ilix, int addr, int precise)
{
  int usex, def;

  usex = use_hash_lookup(false, true, addr, nme, cur_std);
#if (DEBUG && DEBUG_USE_HASH)
  {
    int nuses = num_uses;
    int lusex = 0;
    bool found = false;
    /* Perform the linear use search, and compare to the hash lookup */
    for (lusex = first_use; nuses--; lusex++) {
      if (USE_NM(lusex) == nme && USE_STD(lusex) == cur_std && USE_ADDR(lusex) == addr) {
        found = true;
        break;
      }
    }
    use_hash_check(usex, lusex, found);
  }
#endif /* DEBUG && DEBUG_USE_HASH */
  if (usex) return usex;

  usex = opt.useb.stg_avail++;
  OPT_NEED(opt.useb, USE, 100);

  num_uses++;
  use_hash_insert(usex, false, true, addr, nme, cur_std);
  BZERO(opt.useb.stg_base + usex, USE, 1);
  USE_NM(usex) = nme;
  USE_FG(usex) = cur_fg;
  USE_STD(usex) = cur_std;
  USE_AST(usex) = ilix;
  USE_ADDR(usex) = addr;
  USE_UD(usex) = NULL;
  USE_PRECISE(usex) = precise;
  USE_DOINIT(usex) = in_doinit;
  if ((def = NME_DEF(nme)) && DEF_FG(def) == cur_fg) {
    /* in the same flowgraph node, there exists a def of the same
     * variable prior to the use
     */
    if (ilix == addr) {
      /* use is a whole use or just a scalar variable */
      if (!precise || !DEF_PRECISE(def) || DEF_ADDR(def) != addr)
        USE_EXPOSED(usex) = 1;
      /*
       * add use to du of the definition for nme which is in the block
       * and immediately precedes the use
       */
      add_du_ud(def, usex);
#if DEBUG
      if (OPTDBG(9, 64) && !USE_EXPOSED(usex))
        fprintf(gbl.dbgfil, "-- use %d reached by def %d (use not exposed1)\n",
                usex, def);
#endif
      /* since this an aggregate use, add this as a use of any
       * remaining defs in the same block which precede the use.
       * Stop if the def's addr is the aggregate (scalar falls into this
       * condition) or when we run out of defs for the variable in this
       * block.
       */
      while (TRUE) {
        if (precise && DEF_PRECISE(def) && DEF_ADDR(def) == addr)
          break;
        def = DEF_NEXT(def);
        if (def == 0 || DEF_FG(def) != cur_fg)
          break;
        if (!DEF_GEN(def))
          continue;
        add_du_ud(def, usex);
#if DEBUG
        if (OPTDBG(9, 64))
          fprintf(gbl.dbgfil,
                  "-- use %d reached by def %d (use not exposed2)\n", usex, def);
#endif
      }
      return (usex);
    }
    /* use is a partial use of an array or structure */
    while (TRUE) {
      if (!precise || !DEF_PRECISE(def) || DEF_LHS(def) == ilix ||
          DEF_ADDR(def) == addr || DEF_ADDR(def) == ilix) {
        add_du_ud(def, usex);
#if DEBUG
        if (OPTDBG(9, 64))
          fprintf(gbl.dbgfil,
                  "-- use %d reached by def %d (use not exposed3)\n", usex, def);
#endif
        if (precise && DEF_PRECISE(def) && DEF_ADDR(def) == addr)
          return usex;
      }
      def = DEF_NEXT(def);
      if (def == 0 || DEF_FG(def) != cur_fg)
        break;
    }
  }
  /*
   * this use is exposed to the beginning of the block -- defs which
   * reach this use (those which are in IN(cur_fg)) must be determined
   * later
   */
  USE_EXPOSED(usex) = 1;
  if (do_lv)
    bv_set(LUSE(cur_fg), nme);
  else if (do_unv && flg.opt >= 2 && !XBIT(2, 0x400000)) {
    bv_set(LUSE(cur_fg), nme);
  }

  return (usex);
} /* add_use */

static DEF *
add_def(int nme, int lhs, int addr, int rhs, int precise)
{
  int i;
  DEF *df;

  i = opt.defb.stg_avail++;
  OPT_NEED(opt.defb, DEF, 100);
  DEF_LNEXT(last_def) = i;
  last_def = i;

  df = opt.defb.stg_base + i;
  df->next = NME_DEF(nme);
  df->fg = cur_fg;
  df->std = cur_std;
  df->nm = nme;
  df->lhs = lhs;
  df->addr = addr;
  df->rhs = rhs;
  df->lnext = 0;
  df->flags.all = 0;
  df->du = NULL;
  df->csel = NULL;
  df->flags.bits.precise = precise;
  NME_DEF(nme) = i;

  opt.ndefs++;
  if (do_lv)
    bv_set(LDEF(cur_fg), nme);
  else if (do_unv && flg.opt >= 2 && !XBIT(2, 0x400000)) {
    bv_set(LDEF(cur_fg), nme);
  }

  return (df);
}

static void
chk_ptr_load(int nme)
{
  /*
   * for a loop (including region 0), nme represents a reference occuring
   * in a load.  Determine if this load is via a pointer. A load via a
   * pointer has the potential of conflicting with certain symbols -- this
   * comes into play when considering these symbols for induction analysis
   * and global register assignment (i.e., an alias use of the symbol may
   * occur by this load).
   */
  if (!is_ptr_safe(nme))
    LP_PTR_LOAD(cur_lp) = 1;

}

static void
add_store(int nme)
{
  int sym, store, mark;

  if (cur_lp == 0) {
    /*
     * for region 0, only care about finding stores via ptrs - this is
     * for global register assignment only
     */
    for (; 1; nme = NME_NM(nme)) {
      switch (NME_TYPE(nme)) {
      default:
        break;

      case NT_ARR:
      case NT_MEM:
        continue;

      case NT_VAR:
        sym = NME_SYM(nme);
        if (SCG(sym) == SC_BASED && !is_ptr_safe(nme)) {
          LP_PTR_STORE(cur_lp) = 1;
          FG_PTR_STORE(cur_fg) = 1;
        } else if (POINTERG(sym) && !is_ptr_safe(nme)) {
          LP_PTR_STORE(cur_lp) = 1;
          FG_PTR_STORE(cur_fg) = 1;
        }
        break;

      case NT_UNK:
        LP_PTR_STORE(cur_lp) = 1;
        FG_PTR_STORE(cur_fg) = 1;
        break;

      case NT_IND:
        if (!is_ptr_safe(nme)) {
          LP_PTR_STORE(cur_lp) = 1;
          FG_PTR_STORE(cur_fg) = 1;
        }
        break;
      }
      break;
    }
    return;
  }
  /*
   * for the nme which is being stored, create a store item for it and
   * any "parent" nmes (if an array reference, member reference, etc.).
   * Also, each store item is marked indicating if it represents an
   * aggregate store or it's an array reference with variable subscripts;
   * the mark is propagated to any ancestors.  Initially mark is 0 unless
   * the nme represents an aggregate store (e.g., created by expander to
   * optimize structure assignments).
   */
  if (NME_TYPE(nme) == NT_VAR)
    mark = !ISSCALAR(NME_SYM(nme));
  else if (NME_TYPE(nme) == NT_MEM) {
    if (NME_SYM(nme) <= 1) /* CMPLX real or imag nme */
      mark = 0;
    else
      mark = !ISSCALAR(NME_SYM(nme));
  } else
    mark = 0;
  for (; 1; nme = NME_NM(nme)) {
    if ((store = NME_STL(nme)) == 0) {
      store = new_storeitem(nme);
    } else if (STORE_TYPE(store))
      break;

    STORE_TYPE(store) |= mark;
#if DEBUG
    if (OPTDBG(9, 64))
      fprintf(gbl.dbgfil, "         nme %d marked %d, store %d\n", nme,
              STORE_TYPE(store), store);
#endif

    switch (NME_TYPE(nme)) {
    default:
      break;

    case NT_ARR:
      if (NME_SYM(nme))
        mark = 1;
      continue;

    case NT_MEM:
      if (!XBIT(7, 0x100))
        mark = 0; /* assume ref is within member's boundaries */
      continue;

    case NT_VAR:
      sym = NME_SYM(nme);
      if (SCG(sym) == SC_BASED && !is_ptr_safe(nme)) {
        LP_PTR_STORE(cur_lp) = 1;
        FG_PTR_STORE(cur_fg) = 1;
      } else if (POINTERG(sym) && !is_ptr_safe(nme)) {
        LP_PTR_STORE(cur_lp) = 1;
        FG_PTR_STORE(cur_fg) = 1;
      }
      if (IS_EXTERN(sym) || ADDRTKNG(sym))
        LP_EXT_STORE(cur_lp) = 1;
      break;

    case NT_UNK:
      STORE_TYPE(store) = 1;
      LP_PTR_STORE(cur_lp) = 1;
      FG_PTR_STORE(cur_fg) = 1;
      break;

    case NT_IND:
      if (!is_ptr_safe(nme)) {
        LP_PTR_STORE(cur_lp) = 1;
        FG_PTR_STORE(cur_fg) = 1;
      }
      break;
    }
    break;
  }

}

static int
new_storeitem(int nme)
{
  int store;

  store = opt.storeb.stg_avail++;
  if (store > 32767)
    error(7, 4, 0, CNULL, CNULL);
  OPT_NEED(opt.storeb, STORE, 100);
  STORE_TYPE(store) = 0;
  STORE_NM(store) = nme;
  NME_STL(nme) = store;
  STORE_NEXT(store) = cur_stl->store;
  cur_stl->store = store;

  return store;
}

/** \brief Utility function to add a store item to a loop's store list.

    The store is added after flow analysis has taken place; this is needed
    if we care about considering these stores for later analysis (e.g.,
    the store could affect later invariancy tests).
    The store item is returned to the caller (could be an existing one).
    If one is created, its type is set to zero; the caller is responsible for
    setting it to 1 (variable address).
 */
int
update_stl(int lpx, int nme)
{
  int store;
  STL *stlp;

  stlp = LP_STL(lpx);
  for (store = stlp->store; store; store = STORE_NEXT(store))
    if (STORE_NM(store) == nme)
      return store;
  store = opt.storeb.stg_avail++;
  if (store > 32767)
    error(7, 4, 0, CNULL, CNULL);
  OPT_NEED(opt.storeb, STORE, 100);
  STORE_TYPE(store) = 0;
  STORE_NM(store) = nme;
  STORE_NEXT(store) = stlp->store;
  stlp->store = store;

  return store;
}

/*-----------------------------------------------------------------*/

/*  global flow section  */

/*  live variable analysis */

static void
live_var_init(void)
{
  int i, v;

  /*
   * Allocate the space which will contain pointers to the flowgraph
   * nodes' live variable sets.  This space, organized as a table,
   * parallels the flowgraph table.
   */
  NEW(lv.sets, LV_SETS, opt.num_nodes + 1);
  if (lv.sets == NULL)
    error(7, 4, 0, CNULL, CNULL);
  /*
   * calculate the size (d) in units of BV required for each flow graph's
   * USE, DEF, LIN, and LOUT sets.
   * Then, the space needed for all of the sets in the flow graph is
   * allocated.  This will be 4*num_nodes+1 BV units (don't use dfn even
   * though it's possible that a node will be deleted; algorithm depends
   * on the availability of sets for the function's exit node).
   * The additional BV unit (the first one in the space) is used
   * as a scratch bit vector.
   */
  lv.n = nmeb.stg_avail - 1;
  lv.bv_len = (lv.n + BV_BITS - 1) / BV_BITS;
  lv.bv_sz = (5 * opt.num_nodes + 1) * lv.bv_len;
  NEW(lv.stg_base, BV, lv.bv_sz);
  if (lv.stg_base == NULL)
    error(7, 4, 0, CNULL, CNULL);
  bv_zero(lv.stg_base, lv.bv_sz);
#if DEBUG
  if (OPTDBG(29, 1)) {
    fprintf(gbl.dbgfil, "lv: size of each BV %d,", lv.bv_len);
    fprintf(gbl.dbgfil, "lv: size of bv space %d\n", lv.bv_sz);
  }
#endif

  i = lv.bv_len;
  for (v = 1; v <= opt.num_nodes; v++) {
    LIN(v) = lv.stg_base + i;
    i += lv.bv_len;
    LOUT(v) = lv.stg_base + i;
    i += lv.bv_len;
    LUSE(v) = lv.stg_base + i;
    i += lv.bv_len;
    LDEF(v) = lv.stg_base + i;
    i += lv.bv_len;
    LUNV(v) = lv.stg_base + i;
    i += lv.bv_len;
  }
  /*
   * for the entry node, initialize its LUSE to those variables which
   * are live upon entry.  For the exit node, initialize its LUSE to those
   * variables which are live upon exit.
   */
  v = opt.exitfg;
  for (i = 1; i <= lv.n; i++)
    if (is_optsym(i)) {
      if (is_sym_exit_live(i))
        bv_set(LUSE(v), i);
      if (is_sym_entry_live(i))
        bv_set(LUSE(1), i);
    }

}

static void
live_var_end(void)
{
  /*
   * After the live variable IN/OUT sets have been computed for the flowgraph
   * nodes, we compute the live variable sets for each loop (i.e., those
   * variables which are live upon entry to the loop and those variables
   * which are live out of a loop.  Once this computation is completed,
   * the space used for the flowgraph sets is freed -- the live-in/-out
   * sets of the flowgraph are not preserved.
   */
  int i, j;
  int v;
  PSI_P p;

  /*
   * calculate the size (d) in units of BV required for each loop's
   * LIN, and LOUT sets.
   * Then, the space needed for all of the sets in the loop is
   * allocated.  This will be 2*(opt.nloops+1) BV units.
   * the additional BV units (the first pair in the space) are used
   * as region 0's LIN and LOUT sets.
   */
  lv.lp_bv_sz = (2 * (opt.nloops + 1)) * lv.bv_len;
  NEW(lv.lp_stg_base, BV, lv.lp_bv_sz);
  if (lv.lp_stg_base == NULL)
    error(7, 4, 0, CNULL, CNULL);
  bv_zero(lv.lp_stg_base, lv.lp_bv_sz);
#if DEBUG
  if (OPTDBG(29, 4)) {
    fprintf(gbl.dbgfil, "lvlp: size of each BV %d,", lv.bv_len);
    fprintf(gbl.dbgfil, "lvlp: size of bv space %d\n", lv.lp_bv_sz);
  }
#endif

  i = 0;
  for (j = 0; j <= opt.nloops; j++) {
    LP_LIN(j) = lv.lp_stg_base + i;
    i += lv.bv_len;
    LP_LOUT(j) = lv.lp_stg_base + i;
    i += lv.bv_len;
  }
  /*
   * compute the set of variables which are live into a loop and live
   * out of a loop:
   *    live-in = U LOUT(v), for each predecessor of head not in loop
   *   live-out = U LIN(v),  for each natural exit from the loop
   */
  for (j = 1; j <= opt.nloops; j++) {
    for (p = FG_PRED(LP_HEAD(j)); p; p = PSI_NEXT(p)) {
      v = PSI_NODE(p);
      if (FG_LOOP(v) == j)
        continue;
      bv_union(LP_LIN(j), LOUT(v), lv.bv_len);
    }
    for (p = LP_EXITS(j); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      v = PSI_NODE(p);
      bv_union(LP_LOUT(j), LIN(v), lv.bv_len);
    }
  }
  /*
   * compute the set of variables which are live into a function and live
   * out of a function:
   *    live-in = LIN(entry node)
   *   live-out = LIN(exit node)
   * Note that this is only done for consistency; could just use
   * is_entry_live() and is_exit_live() at the time when it's necessary
   * to check liveness.
   */
  bv_copy(LP_LIN(0), LIN(1), lv.bv_len);
  bv_copy(LP_LOUT(0), LIN(opt.exitfg), lv.bv_len);
#if DEBUG
  if (OPTDBG(29, 4))
    for (i = 0; i <= opt.nloops; i++) {
      fprintf(gbl.dbgfil, "--- lv for loop %d\n", i);
      fprintf(gbl.dbgfil, "LIN :");
      lv_print(LP_LIN(i));
      fprintf(gbl.dbgfil, "LOUT:");
      lv_print(LP_LOUT(i));
    }
#endif
  if (flg.opt >= 2 && !XBIT(2, 0x400000))
    return;

#if DEBUG
  BZERO(lv.sets, LV_SETS, opt.num_nodes + 1);
#endif
  FREE(lv.sets);
  FREE(lv.stg_base);

}

/*
   Compute the live variable sets of the nodes in the flowgraph.
   The purpose of this analysis is to compute live variable sets for
   the loops in the function.

   Algorithm:
      for each node v in the flowgraph in reverse depth first order {
         LIN(v)  = LUSE(v);
         LOUT(v) = 0;
         add v to queue;
      }
      do {
         remove node v from queue;
         new_out = 0;
         new_out U= LIN(s), for each successor s of v;
         if (new_out != LOUT(v)) {
            LOUT(v) = new_out;
            LIN(v)  = (LOUT(v) - LDEF(v)) U LUSE(v);
            for each predecessor s of v {
               if (s is not in queue)
                  add s to queue;
            }
         }
      }  while (first of queue != last of queue);
*/
static void
live_var(void)
{
  Q_ITEM *f_q, *l_q, *q;
  BV *new_out, *bv;
  PSI_P p;
  int i, v;

  l_q = GET_Q_ITEM(f_q); /* a queue is empty when the first and */
  l_q->next = Q_NULL;    /* and last entries locate the same    */
                         /* item                                */

  /*
   * scan the flow graph nodes in reverse depth first order and define
   * lin(fg) to luse(fg) and lout(fg) to the empty set (done by zeroing
   * out the entire area at allocation.
   * Also, the queue of all the nodes is created.
   */
  for (i = opt.dfn; i >= 1; i--) {
    v = VTX_NODE(i);

    bv_copy(LIN(v), LUSE(v), lv.bv_len);
    /* LOUT(v) <- 0; has been init'd to 0 */

    l_q->next = GET_Q_ITEM(q);
    l_q = q;
    l_q->info = v;
    FG_INQ(v) = 1;
#if DEBUG
    if (OPTDBG(29, 1)) {
      fprintf(gbl.dbgfil, "LUSE(%5d):", v);
      lv_print(LUSE(v));
      fprintf(gbl.dbgfil, "LDEF(%5d):", v);
      lv_print(LDEF(v));
    }
#endif
  }
  /*
   * do the flow equations
   */
  new_out = lv.stg_base;
  do {
    f_q = f_q->next; /* get next item from queue */
    v = f_q->info;   /* and remove it from the queue */
    FG_INQ(v) = 0;

    /* do the union of the LIN sets for the successor of v  */

    bv_zero(new_out, lv.bv_len);
    for (p = FG_SUCC(v); p != PSI_P_NULL; p = PSI_NEXT(p))
      bv_union(new_out, LIN(PSI_NODE(p)), lv.bv_len);

    /*
     * this value becomes OUT(v) only if it is different than the current
     * OUT(v)
     */
    if (bv_notequal(new_out, bv = LOUT(v), lv.bv_len)) {

      /* LOUT(v) = new_out  */

      bv_copy(bv, new_out, lv.bv_len);

      /* LIN(v) = LOUT(v) - LDEF(v) U LUSE(v)  */

      bv_sub(new_out, LDEF(v), lv.bv_len);
      bv_union(new_out, LUSE(v), lv.bv_len);
      bv_copy(LIN(v), new_out, lv.bv_len);
#if DEBUG
      if (OPTDBG(29, 1)) {
        fprintf(gbl.dbgfil, "LIN (%5d):", v);
        lv_print(LIN(v));
        fprintf(gbl.dbgfil, "LOUT(%5d):", v);
        lv_print(LOUT(v));
      }
#endif

      for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p))
        if (!FG_INQ(i = PSI_NODE(p))) {
          l_q->next = GET_Q_ITEM(q);
          l_q = q;
          l_q->info = i;
          FG_INQ(i) = 1;
        }
    }
  } while (f_q != l_q);

  freearea(Q_AREA);

#if DEBUG
  if (OPTDBG(29, 2))
    for (i = opt.dfn; i >= 1; i--) {
      v = VTX_NODE(i);
      fprintf(gbl.dbgfil, "--- lv for node %d\n", v);
      fprintf(gbl.dbgfil, "LUSE:");
      lv_print(LUSE(v));
      fprintf(gbl.dbgfil, "LDEF:");
      lv_print(LDEF(v));
      fprintf(gbl.dbgfil, "LIN :");
      lv_print(LIN(v));
      fprintf(gbl.dbgfil, "LOUT:");
      lv_print(LOUT(v));
    }
#endif

}

static void
lv_print(BV *bv)
{
  int i, j, w;
  int maxlen;

  maxlen = lv.n;
  j = 0;
  w = *bv++;
  for (i = 1; i <= maxlen; i++) {
    if (w & 1) {
      if (j == 8) {
        fprintf(gbl.dbgfil, "\n           ");
        j = 0;
      }
      j++;
      fprintf(gbl.dbgfil, " %d>", i);
      (void)print_nme(i);
    }
    w = w >> 1;
    if (i % BV_BITS == 0)
      w = *bv++;
  }
  fprintf(gbl.dbgfil, "\n");

}

/** \brief Utility functions to determine if a variable is live-in to a loop
    and live-out of a loop.

    NOTE: conservative estimates currently in place for a variable which is
    unsafe (e.g., addrtkn, equivalenced, etc.);  for addrtkn, more precise
    information involves checking arguments in bld_ud() and analyzing right-
    hand sides of assignments for acons. Similarly, conservative estimates
    for a static or external variable with respect to calls in the function.
 */
LOGICAL
is_live_in(int nme, int lp)
{
  if (XBIT(6, 0x20)) /* inhibit any effects of the live-variable analysis */
    return TRUE;
  if (nme > lv.n || !is_optsym(nme))
    return TRUE;
  if (!is_sym_live_safe(nme, lp))
    return TRUE;
  return bv_mem(LP_LIN(lp), nme);
}

LOGICAL
is_live_out(int nme, int lp)
{
  if (XBIT(6, 0x20))
    return TRUE;
  if (nme > lv.n || !is_optsym(nme))
    return TRUE;
  if (!is_sym_live_safe(nme, lp))
    return TRUE;
  return bv_mem(LP_LOUT(lp), nme);
}

/* uninitialized variable check */
static void
uninit_var_init(void)
{
  int i, v, j;

  NEW(lv.sets, LV_SETS, opt.num_nodes + 1);
  if (lv.sets == NULL)
    error(7, 4, 0, CNULL, CNULL);
/*
 * Currently use only LDEF and LUSE(not really use it now).
 */
/*
    lv.n = nmeb.stg_avail - 1;
    lv.bv_len = (lv.n + BV_BITS - 1) / BV_BITS;
    lv.bv_sz = (5 * opt.num_nodes + 1) * lv.bv_len;
    NEW(lv.stg_base, BV, lv.bv_sz);
    if (lv.stg_base == NULL)
        error(7,4,0,CNULL,CNULL);
    bv_zero(lv.stg_base, lv.bv_sz);
*/
#if DEBUG
/*
    if (OPTDBG(29,1)) {
        fprintf(gbl.dbgfil, "lv: size of each BV %d,",
                lv.bv_len);
        fprintf(gbl.dbgfil, "lv: size of bv space %d\n", lv.bv_sz);
    }
*/
#endif

  lv.n = nmeb.stg_avail - 1;
  lv.bv_len = (lv.n + BV_BITS - 1) / BV_BITS;
  lv.bv_sz = (opt.num_nodes + 1) * lv.bv_len;
  opt.def_setb.nme_avail = nmeb.stg_avail;
  NEW(opt.def_setb.nme_base, BV, lv.bv_sz);
  bv_zero(opt.def_setb.nme_base, lv.bv_sz);

  i = lv.bv_len;
  j = 0;
  for (v = 1; v <= opt.num_nodes; v++) {
    LIN(v) = lv.stg_base + i;
    i += lv.bv_len;
    LOUT(v) = lv.stg_base + i;
    i += lv.bv_len;
    LUSE(v) = lv.stg_base + i;
    i += lv.bv_len;
    LDEF(v) = lv.stg_base + i;
    i += lv.bv_len;
    LUNV(v) = lv.stg_base + i;
    i += lv.bv_len;
    FG_UNINITED(v) = opt.def_setb.nme_base + j;
    j += lv.bv_len;
  }
  /*
   * for the entry node, initialize its LUSE to those variables which
   * are live upon entry.  For the exit node, initialize its LUSE to those
   * variables which are live upon exit.
   */
  v = opt.exitfg;
  for (i = 1; i <= lv.n; i++)
    if (is_optsym(i)) {
      if (is_sym_exit_live(i))
        bv_set(LUSE(v), i);
      if (is_sym_entry_live(i))
        bv_set(LUSE(1), i);
    }

}

static void
uninit_var(void)
{
  Q_ITEM *f_q, *l_q, *q, *h_q;
  PSI_P succv, tp;
  int i, v;

  l_q = GET_Q_ITEM(f_q);
  l_q->next = Q_NULL;
  v = 1; /* start from node 1 */
  l_q->info = v;
  FG_INQ(v) = 1;
  h_q = l_q; /* keep the head */
  f_q = l_q;

  do {
    v = f_q->info;
    for (succv = FG_SUCC(v); succv != PSI_P_NULL; succv = PSI_NEXT(succv)) {
      if (!FG_INQ(PSI_NODE(succv))) {
        l_q->next = GET_Q_ITEM(q);
        l_q = q;
        l_q->next = Q_NULL;
        l_q->info = PSI_NODE(succv);
        l_q->flag = 0;
        FG_INQ(PSI_NODE(succv)) = 1;
        FG_DONE(l_q->info) = 0;
      }
    }
    f_q = f_q->next;

  } while (f_q);
#if DEBUG
#endif
  f_q = h_q;
  FG_DONE(f_q->info) = 1; /* set flag that this node is done */
  bv_zero(FG_UNINITED(f_q->info),
          lv.bv_len); /* make that nothing is initialized at entry point */
  do {
    tp = NULL;
    v = f_q->info;
    /*FG_INQ(v) = 0;*/
    for (succv = FG_SUCC(v); succv != PSI_P_NULL; succv = PSI_NEXT(succv)) {
      i = PSI_NODE(succv);
      if (i == v) {
        continue;
      }

      /* its successor initialized list is the def of current node union
       * inherited
       * inited of current node
       */
      bv_copy(LUNV(i), LDEF(v), lv.bv_len);
      bv_union(LUNV(i), FG_UNINITED(v), lv.bv_len);

      if (FG_DONE(i)) {
        if (bv_notequal(LUNV(i), FG_UNINITED(i), lv.bv_len)) {

          bv_intersect(FG_UNINITED(i), LUNV(i), lv.bv_len);

          /* add i node to the linked list so that it can propagate down the
           * tree */
          if (!FG_INQ(i)) {
            l_q->next = GET_Q_ITEM(q);
            l_q = q;
            l_q->next = Q_NULL;
            l_q->info = i;
            FG_INQ(i) = 1;
          }
        }
      } else {
        bv_copy(FG_UNINITED(i), LUNV(i), lv.bv_len);
      }
      /* mark that it is already done at least once */

      FG_DONE(i) = 1;
    }
    f_q = f_q->next;
  } while (f_q && f_q != l_q);

  freearea(Q_AREA);

#if DEBUG
  if (OPTDBG(29, 0x40000)) {
    fprintf(gbl.dbgfil, "End dump fgraph of uninit_var\n");
    for (i = opt.dfn; i >= 1; i--) {
      v = VTX_NODE(i);
      fprintf(gbl.dbgfil, "--- lv for node %d\n", v);
      fprintf(gbl.dbgfil, "FG_UNINITED(%5d):", v);
      lv_print(FG_UNINITED(v));
    }
  }
#endif

}

static void
uninit_var_end(void)
{
  int v, i;
#if DEBUG
  BZERO(lv.sets, LV_SETS, opt.num_nodes + 1);
#endif
  FREE(lv.sets);
  FREE(lv.stg_base);

  /* clear visited field */
  for (i = opt.dfn; i >= 1; i--) {
    v = VTX_NODE(i);
    FG_DONE(i) = 0;
    FG_INQ(i) = 0;
  }

}

LOGICAL
is_initialized(int *bv, int nme)
{
  if (nme > lv.n || !is_optsym(nme))
    return TRUE;
  return bv_mem(bv, nme);
}

#if DEBUG
static int
bv_count(int *bv)
{
  int i, w, count;
  count = 0;
  w = *bv++;
  for (i = 1; i <= opt.ndefs; ++i) {
    if (w & 1)
      ++count;
    w = w >> 1;
    if ((i % BV_BITS) == 0)
      w = *bv++;
  }
  return count;
} /* bv_count */

#ifdef FLANG_FLOW_UNUSED
static void
dump_queue(Q_ITEM *head)
{
  int count;
  fprintf(gbl.dbgfil, "queue =");
  count = 0;
  for (; head != NULL; head = head->next) {
    ++count;
    if (count <= 5)
      fprintf(gbl.dbgfil, " %4d", head->info);
  }
  if (count > 6)
    fprintf(gbl.dbgfil, " ...");
  fprintf(gbl.dbgfil, " (%4d entries)\n", count);
} /* dump_queue */
#endif
#endif

/*
 * build reverse-depth-first order
 *  DFO order for a flow graph visits each node,
 *  numbering each node as it is visited before its successors.
 *  RDFO order visits each node, numbering each node
 *  AFTER its successors.  We use the reverse of that order,
 *  or, equivalently, build the list backwards.
 */
static int rdfon = 0;

static void
_rdfo(int v)
{
  PSI_P p;
  FG_RDFO(v) = -1;
  for (p = FG_SUCC(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    int w;
    w = PSI_NODE(p);
    if (FG_RDFO(w) == 0) {
      _rdfo(w);
    }
  }
  if (rdfon <= 0)
    interr("more RDFO nodes than DFN nodes", 0, 4);
  FG_RDFO(v) = rdfon;
  RDFOVTX_NODE(rdfon) = v;
  --rdfon;
} /* _rdfo */

static void
build_rdfo()
{
  int v;
  for (v = 0; v <= opt.num_nodes; ++v) {
    FG_RDFO(v) = 0;
    RDFOVTX_NODE(v) = 0;
  }
  rdfon = opt.dfn;
  _rdfo(1);
  if (rdfon != 0)
    interr("more DFN nodes than RDFO nodes", 0, 4);
} /* build_rdfo */

/*  definition analysis  */

/*
   Algorithm:
      for each node v in the flowgraph in depth first order {
         IN(v) = 0;
         OUT(v) = GEN(v);
      }
      build reverse-depth-first-order
      top = 1
      do{
        changed = 0
        oldtop = top
        top = opt.dfn+1
        do dfo = oldtop to opt.dfn{
         v = RDFOVTX_NODE(dfo)
         IN(v) = 0;
         IN(v) U= OUT(p), for each predecessor p of v;
         new_out = (IN(v) - KILL(v)) U GEN(v);
         if( new_out != OUT(v) ){
            OUT(v) = new_out
            for each successor s of v {
               if( FG_DFN(s) < top && FG_DFN(s) < dfo ){
                top = FG_DFN(s)
                changed = 1
               }
            }
         }
        }
      }while(changed)
*/
static void
reaching_defs(void)
{
  BV *bv, *new_out;
  PSI_P p, s;
  int i, v, def, bih, top, oldtop, changed, dfo, iter;

  /*
   * calculate the size (d) in units of BV required for each flow graph's
   * IN or OUT set.  Then, the space needed for the entire flow graph is
   * allocated.  This will be 2*dfn+1 BV units. (opt.dfn is the number
   * of nodes which are in the flowgraph and not dead -- dead nodes have a
   * dfn of -1); the addtional BV unit (the first one in the space) is used
   * as a scratch bit vector.
   */
  build_rdfo();
  def_bv_len = (opt.ndefs + BV_BITS - 1) / BV_BITS;
  i = (2 * opt.dfn + 1) * def_bv_len;
  NEW(opt.def_setb.stg_base, BV, i);
  bv_zero(opt.def_setb.stg_base, i);
  opt.def_setb.stg_avail = 0;
  new_out = opt.def_setb.stg_base + opt.def_setb.stg_avail;
  opt.def_setb.stg_avail += def_bv_len;
#if DEBUG
  if (OPTDBG(9, 64)) {
    fprintf(gbl.dbgfil, "    size of each def BV %d,", def_bv_len);
    fprintf(gbl.dbgfil, " size of bv space %d\n", i);
  }
#endif
#if DEBUG
  while (i--)
    assert(*(opt.def_setb.stg_base + i) == 0, "gflow: bv not zero", i, 3);
#endif

  /*
   * go through all the flow graph nodes in depth first order and define
   * in(fg) to empty (done by zeroing out the entire area at allocation.
   * out(fg) is set to the definitions in fg which reach the end of the
   * node (gen flag).  Also, the queue of all the nodes is created.
   */
  for (i = 1; i <= opt.dfn; i++) {
    v = RDFOVTX_NODE(i);

    FG_IN(v) = opt.def_setb.stg_base + opt.def_setb.stg_avail;
    opt.def_setb.stg_avail += def_bv_len;

    bv = FG_OUT(v) = opt.def_setb.stg_base + opt.def_setb.stg_avail;
    opt.def_setb.stg_avail += def_bv_len;

    /* OUT(v) = GEN(v) */

    def = FG_FDEF(v);
    while (def) {
      if (DEF_GEN(def))
        bv_set(bv, def);
      def = DEF_LNEXT(def);
    }
    /*
     * if a call occurs in this block, then the GEN set contains
     * "call def"
     */
    bih = v;
    if (FG_EX(bih) && !FG_EN(bih))
      bv_set(bv, CALL_DEF);
    /*
     * if a qjsr occurs in this block, then the GEN set contains
     * "qjsr def"
     */
    if (FG_QJSR(bih))
      bv_set(bv, QJSR_DEF);
    /*
     * if a store via a pointer occurs in this block, then the GEN set
     * contains "ptr store def"
     */
    if (FG_PTR_STORE(v))
      bv_set(bv, PTR_STORE_DEF);
#if DEBUG
    if (OPTDBG(9, 64)) {
      fprintf(gbl.dbgfil, "GEN(%5d):", v);
      bv_print(FG_OUT(v), opt.ndefs);
    }
#endif
  }
  /*
   * do the flow equations
   */
  top = 1;
  iter = 0;
  do {
    ++iter;
    changed = 0;
    oldtop = top;
    top = opt.dfn + 1;
    for (dfo = oldtop; dfo <= opt.dfn; ++dfo) {
      /* do the union of the OUT sets for the predecessors of v  */
      v = RDFOVTX_NODE(dfo);
      bv = FG_IN(v);
      bv_zero(bv, def_bv_len);
      for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p))
        bv_union(bv, FG_OUT(PSI_NODE(p)), def_bv_len);

      /* new_out = IN(v)  */
      bv_copy(new_out, bv, def_bv_len);

/* new_out = IN(v) - KILL(v) U GEN(v)  */
/* go through the definitions in GEN(v)  */
#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil, " defs killed:");
#endif
      def = FG_FDEF(v);
      while (def) {
        if (DEF_GEN(def)) {
          /* since def is in GEN(v), find the other defs defining
           * the same names entry.  These defs are in KILL(v). */
          if (DEF_PRECISE(def) && !DEF_ARG(def))
            for (i = NME_DEF(DEF_NM(def)); i; i = DEF_NEXT(i))
              if (DEF_FG(i) != v && DEF_ADDR(def) == DEF_ADDR(i)) {
                bv_off(new_out, i); /* OUT(v) -= KILL(v) */
#if DEBUG
                if (OPTDBG(9, 64))
                  fprintf(gbl.dbgfil, " %d", i);
#endif
              }
          bv_set(new_out, def); /* OUT(v) U= GEN(v) */
        }
        def = DEF_LNEXT(def);
      }
#if DEBUG
      if (OPTDBG(9, 64))
        fprintf(gbl.dbgfil, "\n");
#endif
      /*
       * if a call occurs in this block, "call def" is added to
       * OUT(v)
       */
      bih = v;
      if (FG_EX(bih) && !FG_EN(bih))
        bv_set(new_out, CALL_DEF);
      /*
       * if a QJSR occurs in this block, "call def" is added to
       * OUT(v)
       */
      if (FG_QJSR(bih))
        bv_set(new_out, QJSR_DEF);
      /*
       * if a store via a pointer occurs in this block, then the GEN set
       * contains "ptr store def"
       */
      if (FG_PTR_STORE(v))
        bv_set(new_out, PTR_STORE_DEF);
#if DEBUG
      if (OPTDBG(9, 64)) {
        fprintf(gbl.dbgfil, "OUT(%5d):", v);
        bv_print(new_out, opt.ndefs);
      }
#endif
      /*
       * this value becomes OUT(v) only if it is different than the
       * current OUT(v)
       */
      if (bv_notequal(new_out, bv = FG_OUT(v), def_bv_len)) {
#if DEBUG
        if (DBGBIT(9, 0x1000000))
          fprintf(gbl.dbgfil, "old(%5d): %5d  new(%5d): %5d\n", v, bv_count(bv),
                  v, bv_count(new_out));
#endif
        /* OUT(v) = new_out  */
        bv_copy(bv, new_out, def_bv_len);
#if DEBUG
        if (OPTDBG(9, 64)) {
          fprintf(gbl.dbgfil, "OUT(%5d):", v);
          bv_print(FG_OUT(v), opt.ndefs);
        }
#endif
        for (s = FG_SUCC(v); s != PSI_P_NULL; s = PSI_NEXT(s)) {
          int ss, o;
          ss = PSI_NODE(s);
          o = FG_RDFO(ss);
          if (o < dfo && o < top) {
            top = o;
            changed = 1;
          }
        }
      }
    }
  } while (changed);

  freearea(Q_AREA);

}

static void
du_ud(void)
{

  /* compute the du chains for the definitions and the ud chains for the uses.
     What's involved is scanning the uses in the use table.  For those uses in
     a flowgraph node which can be reached from the beginning of the
     node (i.e., exposed), the defs of the uses are checked to determine
     if they are members of the flowgraph node's IN set.
  */
  int i, nme, def;
  BV *bv;

  for (i = 1; i < opt.useb.stg_avail; i++)
    if (USE_EXPOSED(i)) {
      nme = USE_NM(i);
      bv = FG_IN(cur_fg = USE_FG(i));
      for (def = NME_DEF(nme); def; def = DEF_NEXT(def))
        if (bv_mem(bv, def)) {
          add_du_ud(def, i);
#if DEBUG
          if (OPTDBG(9, 64))
            fprintf(gbl.dbgfil, "-- use %d reached by def %d\n", i, def);
#endif
        }
    }
}

static void
add_du_ud(int def, int use)
{
  DU *du;
  UD *ud;
  /*
   * add the use to du chain of the definition for nme.
   */
  du = (DU *)getitem(DU_AREA, sizeof(DU));
  du->next = DEF_DU(def);
  DEF_DU(def) = du;
  du->use = use;
  /*
   * add def to ud chain of the use
   */
  ud = (UD *)getitem(DU_AREA, sizeof(UD));
  ud->next = USE_UD(use);
  ud->def = def;
  USE_UD(use) = ud;
}

/*-----------------------------------------------------------------*/

static struct {/* "global" data structure for const propagation */
  int val;     /* ili of constant to be propagated */
  Q_ITEM *f_q;
  Q_ITEM *l_q;
} cp;

static LOGICAL
const_prop(void)
{
  LOGICAL keep_def;
  LOGICAL changes;
  int df;
  int df_ilt;
  int nme;
  DU *du;
  Q_ITEM *q;
  int dvl;

  if (XBIT(6, 0x1))
    return FALSE;
#if DEBUG
  if (OPTDBG(9, 32768)) {
    fprintf(gbl.dbgfil, "const_prop: file %s, function %s\n", gbl.src_file,
            getprint(FG_LABEL(gbl.entbih)));
    dump_global(FALSE);
  }
#endif
  /*
   * Use a queue to record the definitions which are candidates for
   * constant propagation; if a definition's value is propagated, it
   * is possible to create more opportunities for constant propagation.
   * The new candidates are added to the end of the queue.
   * NOTES:
   * 1.  a queue is empty when the first and last entries locate the same
   *     item
   * 2.  the queue is initially filled with the constant definitions
   *     found during localflow.
   * 3.  it's sufficient to use the DEF_CONST flag to indicate that
   *     the definition is in the queue.
   * 4.  for FORTRAN, before the def table is scanned, opportunities
   *     for const prop due to data inits are processed.  The queue
   *     to which new def opportunities are added during copy_const is
   *     released.  Subsequent processing of the def table will still
   *     process the new defs since these defs will have their
   *     DEF_CONST flags set.
   */
  cp.l_q = GET_Q_ITEM(cp.f_q);
  cp.l_q->next = Q_NULL;
  changes = FALSE;
  if (!XBIT(7, 0x100000)) {
    for (dvl = 0; dvl < aux.dvl_avl; dvl++) {
      int use;
      int sym;
      int nocp;
      LOGICAL changes_sym;

      sym = DVL_SPTR(dvl);
      /* allow only if
       *  symbol is a local (SC_LOCAL, or SC_STATIC with SAVE flag)
       *  symbol is never assigned, address it not taken, no storage
       *  overlap, not a host subprogram
       */
      if ((sym &&
           (SCG(sym) != SC_LOCAL && (SCG(sym) != SC_STATIC || !SAVEG(sym)))) ||
          (gbl.internal == 1)
#ifdef THREADG
          || THREADG(sym)
#endif
          || ASSNG(sym) || ADDRTKNG(sym) || SOCPTRG(sym) || ARGG(sym))
        continue;
      cp.val = DVL_CONVAL(dvl);
      nme = addnme(NT_VAR, sym, 0, (INT)0);
      nocp = 0;
      /* keep track of changes for this sym. */
      changes_sym = FALSE;
      for (use = 1; use < opt.useb.stg_avail; use++) {
        if (nme == USE_NM(use)) {
              if (!copy_const(use)) {
            changes = TRUE;
            changes_sym = TRUE;
          } else {
            /* found a use where constant has not been propagated */
            nocp++;
          }
        }
      }
      if (!nocp && changes_sym)
        DATACONSTP(sym, 1);
    }
    aux.dvl_avl = 0; /* ensure doesn't happen again */
    if (changes) {
      freearea(Q_AREA);
      cp.l_q = GET_Q_ITEM(cp.f_q);
      cp.l_q->next = Q_NULL;
    }
  }

  for (df = FIRST_DEF; df <= opt.ndefs; df++) {
    if (!can_prop_fg(DEF_FG(df)))
      continue;
    if (DEF_CONST(df) && can_prop_def(df) && DEF_PRECISE(df)) {
      cp.l_q->next = GET_Q_ITEM(q);
      cp.l_q = q;
      cp.l_q->info = df;
    }
  }

  while (cp.f_q != cp.l_q) {
    cp.f_q = cp.f_q->next;
    df = cp.f_q->info;
    nme = DEF_NM(df);
    df_ilt = DEF_STD(df);
    cp.val = DEF_RHS(df);
#if DEBUG
    if (OPTDBG(9, 32768))
      fprintf(gbl.dbgfil, "--- cand df:%d, val ili %d, %s\n", df, cp.val,
              getprint(NME_SYM(nme)));
#endif
    keep_def = is_sym_imp_live(nme) || XBIT(6, 0x2);
    for (du = DEF_DU(df); du; du = du->next) {
      int use;

      use = du->use;
      if (!single_ud(use) || self_use(use) || !USE_PRECISE(use)) {
        keep_def = TRUE;
        continue;
      }

      if (copy_const(use)) {
        keep_def = TRUE;
      }
      changes = TRUE;
    }
  }

  freearea(Q_AREA);
  return changes;
}

#ifdef FLANG_FLOW_UNUSED
static int
lhs_notsubscr(int use)
{
  int use_std;
  int old_tree;
  use_std = USE_STD(use);
  old_tree = STD_AST(use_std);
  if (A_TYPEG(A_DESTG(old_tree)) == A_SUBSCR ||
      A_TYPEG(A_DESTG(old_tree)) == A_SUBSTR)
    return 0;
  return 1;
}
#endif

static int
copy_const(int use)
{
  int use_std;
  int old_tree, old_ast;
  int new_tree;
  int tmp;
  Q_ITEM *q;
  int fgx;
  int b;   /* bih temporary */

  use_std = USE_STD(use);
  old_tree = STD_AST(use_std);
  b = A_OPT1G(old_tree);
  old_ast = USE_ADDR(use);
  fgx = USE_FG(use);
  switch (A_TYPEG(old_tree)) {
  case A_REALIGN:
  case A_REDISTRIBUTE:
    /* don't propagate into redistribute */
    return 1;
  case A_PRAGMA:
    switch (A_PRAGMATYPEG(old_tree)) {
    case PR_ACCCOPY:
    case PR_ACCCOPYIN:
    case PR_ACCCOPYOUT:
    case PR_ACCLOCAL:
    case PR_ACCPRIVATE:
    case PR_ACCCACHE:
    case PR_ACCUPDATEHOST:
    case PR_ACCUPDATEDEVICE:
    case PR_ACCUPDATE:
    case PR_ACCMIRROR:
    case PR_ACCREFLECT:
    case PR_ACCDEVSYM:
    case PR_ACCDEVICEPTR:
    case PR_ACCCUDACALL:
    case PR_ACCCREATE:
    case PR_ACCPRESENT:
    case PR_ACCPCOPY:
    case PR_ACCPCOPYIN:
    case PR_ACCPCOPYOUT:
    case PR_ACCPCREATE:
    case PR_ACCATTACH:
    case PR_ACCDETACH:
    case PR_ACCFIRSTPRIVATE:
    case PR_ACCREDUCTION:
    case PR_ACCCACHEDIR:
    case PR_ACCCACHEREADONLY:
    case PR_ACCCACHEARG:
    case PR_ACCHOSTDATA:
    case PR_ACCENDHOSTDATA:
    case PR_ACCUSEDEVICE:
    case PR_ACCUSEDEVICEIFP:
    case PR_ACCDEVICERES:
      return 1;
    }
    break;
  }
#ifdef CUDAG
  if ((CUDAG(gbl.currsub) & (CUDA_HOST)) || CUDAG(gbl.currsub) == 0) {
    /* don't forward substitute assignments to device variables
     * in host code */
    int astx = old_ast;
    while (astx) {
      int sptr = 0;
      switch (A_TYPEG(astx)) {
      case A_ID:
        sptr = A_SPTRG(astx);
        astx = 0;
        break;
      case A_MEM:
        sptr = A_SPTRG(A_MEMG(astx));
        astx = A_LOPG(astx);
        break;
      case A_SUBSCR:
      case A_SUBSTR:
        astx = A_LOPG(astx);
        break;
      default:
        astx = 0;
        break;
      }
      if (sptr && DEVICEG(sptr)) {
        /* don't propagate */
        return 1;
      }
    }
  }
#endif
  if (A_TYPEG(old_ast) == A_SUBSTR &&
      A_DTYPEG(cp.val) == A_DTYPEG(A_LOPG(old_ast)))
    old_ast = A_LOPG(old_ast);
#if DEBUG
  if (OPTDBG(9, 32768))
    fprintf(gbl.dbgfil, "    replace use %d in fg %d @ilt %d, ili %d\n", use,
            fgx, use_std, old_ast);
#endif
  /*
   * rewrite the ili tree (we replace old_ast with val);
   */
  ast_visit(1, 1);
  ast_replace(old_ast, cp.val);
  new_tree = ast_rewrite(old_tree);
  ast_unvisit();
#if DEBUG
  if (OPTDBG(9, 32768))
    fprintf(gbl.dbgfil, "    new ilitree %d, old ilitree %d\n", new_tree,
            old_tree);
#endif
  if (new_tree != old_tree) {
    int newval; /* new stored value, if a store */
    int df;     /* def entry, if a store */

    A_OPT1P(new_tree, b);
    STD_AST(use_std) = new_tree;
    A_STDP(new_tree, use_std);

    if (A_TYPEG(new_tree) == A_ASN) {
      newval = A_SRCG(new_tree);
      df = 0;
      tmp = A_NMEG(A_DESTG(new_tree)); /* names entry */
      if (is_optsym(tmp)) {
        /* find def item which corresponds to this store */
        for (tmp = NME_DEF(tmp); tmp; tmp = DEF_NEXT(tmp))
          if (DEF_STD(tmp) == use_std) {
            df = tmp;
            break;
          }
      }
      if (A_TYPEG(newval) == A_CNST) {
        /*
         * created a def of a constant: check the obvious if
         * this store represents a new opportunity for
         * propagation.
         */
        if (df && !DEF_CONST(df) && can_prop_def(df)) {
          cp.l_q->next = GET_Q_ITEM(q);
          cp.l_q = q;
          cp.l_q->info = df;
          DEF_CONST(df) = 1;
          DEF_SELF(df) = 0;
#if DEBUG
          if (OPTDBG(9, 32768))
            fprintf(gbl.dbgfil, "    new df cand %d\n", df);
#endif
        }
      }
    }
  }
  return 0;
}

/* Return TRUE if the statement of use contains a definition of the
 * same variable, or for MASTER or ENDMASTER statements; this to avoid
 * propagating constants to MASTER or ENDMASTER/COPY clauses. */
static LOGICAL
self_use(int use)
{
  int nm = USE_NM(use);
  int std = USE_STD(use);
  int def, ast;

  for (def = NME_DEF(nm); def; def = DEF_NEXT(def))
    if (DEF_STD(def) == std)
      return TRUE;
  ast = STD_AST(std);
  if (ast && (A_TYPEG(ast) == A_MASTER || A_TYPEG(ast) == A_ENDMASTER))
    return TRUE;
  return FALSE;
}

static LOGICAL
can_prop_fg(int fg)
{
  if (FG_CS(fg))
    return FALSE;
  return TRUE;
}

/* Return TRUE if a def is of a constant that can be copied. */
static LOGICAL
can_prop_def(int def)
{
  int dtyp;
  int sptr;

  if (!def)
    return FALSE;
  /* nme could still be an array (our descriptor) */
  sptr = basesym_of(DEF_NM(def));
  if (!sptr)
    return FALSE;
  if (POINTERG(sptr) && MIDNUMG(sptr))
    sptr = MIDNUMG(sptr);
  dtyp = DTYPEG(sptr);
  if (A_TYPEG(DEF_ADDR(def)) == A_ID && !A_SHAPEG(DEF_ADDR(def)) &&
      A_TYPEG(DEF_RHS(def)) == A_CNST &&
      ((!HCCSYMG(sptr) || XBIT(70, 0x1000)) && !PTRVG(sptr)) && !ADDRTKNG(sptr)
#ifdef THREADG
      && !THREADG(sptr)
#endif
      && DTY(dtyp) != TY_CHAR && DTY(dtyp) != TY_NCHAR)
    return TRUE;
  return FALSE;
}

/*-----------------------------------------------------------------*/

/** \brief Mark stores as deletable only if its uses follow
    the definition in the same block.

    In the future, we'll get rid of defs for which there are no uses -- but
    before we do this, we need to determine the side effects of eliminating a
    def (and ilt) from a block.
 */
void
delete_stores(void)
{
  int def, iltz, sym, i;
  DU *du;

  if (XBIT(8, 0x2)) {
#if DEBUG
    if (OPTDBG(9, 8192))
      fprintf(gbl.dbgfil, "delete_st: file %s, function %s INHIBITED\n",
              gbl.src_file, getprint(FG_LABEL(gbl.entbih)));
#endif
    return;
  }
#if DEBUG
  if (OPTDBG(9, 8192))
    fprintf(gbl.dbgfil, "delete_st: file %s, function %s\n", gbl.src_file,
            getprint(FG_LABEL(gbl.entbih)));
#endif
  for (def = FIRST_DEF; def <= opt.ndefs; def++) {
    int nme;
    nme = DEF_NM(def);
    if (NME_TYPE(nme) != NT_VAR)
      continue;
    i = 0;
    if (is_sym_imp_live(nme))
      continue;
    sym = NME_SYM(nme);
    if (SDSCS1G(sym) || DESCARRAYG(sym))
      continue;
    if (FG_CS(DEF_FG(def)) && !IS_PRIVATE(sym)) {
#if DEBUG
      if (OPTDBG(9, 8192)) {
        fprintf(gbl.dbgfil,
                "delete_st: def for %s at line %d in critical section\n",
                getprint(sym), STD_LINENO(DEF_FG(def)));
      }
#endif
      continue;
    }
    du = DEF_DU(def);
    if (du == NULL) {
      iltz = DEF_STD(def);
      if (STD_EX(iltz)) {
/*  don't eliminate yet, we'll need to study this */
#if DEBUG
        if (OPTDBG(9, 8192)) {
          fprintf(gbl.dbgfil, "delete_st: def for %s at line %d, ilt %d not "
                              "deleted (in func)\n",
                  getprint(sym), STD_LINENO(DEF_FG(def)), iltz);
        }
#endif
      } else if (A_TYPEG(STD_AST(iltz)) != A_ASN) {
#if DEBUG
        if (OPTDBG(9, 8192)) {
          fprintf(gbl.dbgfil,
                  "delete_st: def for %s at line %d, ilt %d not assignment\n",
                  getprint(sym), STD_LINENO(DEF_FG(def)), iltz);
        }
#endif
      } else {
        if (flg.smp && PARREFG(sym)) {
          continue;
        }
#if DEBUG
        if (OPTDBG(9, 8192)) {
          fprintf(gbl.dbgfil,
                  "delete_st: def for %s at line %d, ilt %d deleted\n",
                  getprint(sym), STD_LINENO(DEF_FG(def)), iltz);
        }
#endif
        DEF_DELETE(def) = 1;
        unlnkilt(iltz, (int)DEF_FG(def), FALSE);
      }
    }
  }

}

void
use_before_def(void)
{
  int nme, use, sym;

  for (use = 1; use < opt.useb.stg_avail; use++) {
    nme = USE_NM(use);
    sym = basesym_of(USE_NM(use));
    if (!CCSYMG(sym)
        && ((SCG(sym) == SC_LOCAL && !DINITG(sym)) ||
            (SCG(sym) == SC_DUMMY && INTENTG(sym) == INTENT_OUT)) &&
        !USE_ARG(use)
        && !ADDRTKNG(sym)) {
      UD *ud;
      int fg;
      PSI_P p;
      BV *bv;
      int df;
      LOGICAL checked;
      LOGICAL covered;
      int self;
      int df_cnt;
      /* if a def reaching the use occurs in the same block, then
       * 1.  if the use is not exposed, the def precedes the
       *     block -- all is ok.
       * 2.  if  the use is exposed, the def does not precede the
       *     use (it's either the same ilt or occurs after the use.
       * For the def which does not appear in the same block, examine
       * the predecessors of the node containing the use:
       *     if the node's OUT set does not contain a def of the use,
       *     then there exists a path which does not contain a definition.
       */
      ud = USE_UD(use);
      if (ud == NULL) {
        error(279, 2, (int)FG_LINENO(USE_FG(use)), SYMNAME(sym),
              SYMNAME(FG_LABEL(gbl.entbih)));
        goto next_use;
      }
      fg = USE_FG(use);
      self = 0;
      df_cnt = 0;
      checked = FALSE;
      for (; ud != NULL; ud = ud->next) {
        df = ud->def;
        df_cnt++;
        if (DEF_FG(df) == fg) {
          if (USE_EXPOSED(use) && DEF_PRECISE(df) && !DEF_ARG(df) &&
              !DEF_OTHER(df))
            self = df;
        } else if (!checked) {
          for (p = FG_PRED(fg); p != PSI_P_NULL; p = PSI_NEXT(p)) {
            UD *u;
            covered = FALSE;
            bv = FG_OUT(PSI_NODE(p));
            for (u = USE_UD(use); u != NULL; u = u->next) {
              df = u->def;
              if (bv_mem(bv, df)) {
                covered = TRUE;
                break;
              }
            }
            if (!covered) {
              error(279, 2, (int)FG_LINENO(USE_FG(use)), SYMNAME(sym),
                    SYMNAME(FG_LABEL(gbl.entbih)));
              goto next_use;
            }
          }
          checked = TRUE;
        }
      }
      if (df_cnt == 1 && self && FG_LOOP(fg)) {
        /* the only def which reaches the use contains the use;
         * check if any def is in the IN set of the head of the loop
         */
        bv = FG_IN(LP_HEAD(FG_LOOP(fg)));
        covered = FALSE;
        for (df = NME_DEF(nme); df; df = DEF_NEXT(df)) {
          if (df != self && bv_mem(bv, df)) {
            covered = TRUE;
            break;
          }
        }
        if (!covered) {
          error(279, 2, (int)FG_LINENO(USE_FG(use)), SYMNAME(sym),
                SYMNAME(FG_LABEL(gbl.entbih)));
        }
      }
    }
  next_use:;
  }

}

static void
dump_global(LOGICAL inout)
{
  int dfx, nme, i, j;
  DU *du;
  UD *ud;

  fprintf(gbl.dbgfil,
          "\n* * * * * *  Global Info for Function \"%s\"  * * * * * *\n",
          getprint(FG_LABEL(gbl.entbih)));
  fprintf(gbl.dbgfil, "\n* * * * * * * *   Scalar NME   * * * * * * * *\n");
  for (i = 0; i < nmeb.stg_avail; i++) {
    if (NME_DEF(i) == 0)
      continue;
    fprintf(gbl.dbgfil, "%5u   ", i);
    assert(NME_TYPE(i) == NT_VAR, "wrong def nme type", i, 3);
    j = NME_SYM(i);
    fprintf(gbl.dbgfil, "variable            sym:%5u  \"%s\"\n", j,
            getprint((int)j));
    fprintf(gbl.dbgfil, "     defs:");
    j = 0;
    for (dfx = NME_DEF(i); dfx; dfx = DEF_NEXT(dfx)) {
      if (j == 9) {
        fprintf(gbl.dbgfil, "\n          ");
        j = 0;
      }
      fprintf(gbl.dbgfil, " %-5d", dfx);
      j++;
    }
    if (j)
      fprintf(gbl.dbgfil, "\n");
  }

  fprintf(gbl.dbgfil, "\n* * * * * * *  Definitions (%d), Du  * * * * * * *\n",
          opt.ndefs);
  for (dfx = FIRST_DEF; dfx <= opt.ndefs; dfx++) {
    fprintf(gbl.dbgfil, "%5d.  fg:%-5d  ilt:%-5d  next:%-5d  lnext:%-5d", dfx,
            (int)DEF_FG(dfx), (int)DEF_STD(dfx), (int)DEF_NEXT(dfx),
            (int)DEF_LNEXT(dfx));
    nme = DEF_NM(dfx);
    fprintf(gbl.dbgfil, "  nme:%-5d", nme);
    fprintf(gbl.dbgfil, " \"%s\"", getprint((int)NME_SYM(nme)));
    if (DEF_GEN(dfx))
      fprintf(gbl.dbgfil, " -gen");
    if (DEF_CONST(dfx))
      fprintf(gbl.dbgfil, " -cnst");
    if (DEF_SELF(dfx))
      fprintf(gbl.dbgfil, " -self");
    if (DEF_CONFL(dfx))
      fprintf(gbl.dbgfil, " -confl");
    if (DEF_DOINIT(dfx))
      fprintf(gbl.dbgfil, " -doinit");
    if (DEF_DOEND(dfx))
      fprintf(gbl.dbgfil, " -doend");
    if (DEF_ARG(dfx))
      fprintf(gbl.dbgfil, " -arg");
    if (DEF_OTHER(dfx))
      fprintf(gbl.dbgfil, " -other");
    if (!DEF_PRECISE(dfx))
      fprintf(gbl.dbgfil, " -imprecise");
    fprintf(gbl.dbgfil, "\n       lhs:%-5d  addr:%-5d  rhs:%-5d",
            (int)DEF_LHS(dfx), (int)DEF_ADDR(dfx), (int)DEF_RHS(dfx));
    fprintf(gbl.dbgfil, "\n        du:");
    j = 0;
    for (du = DEF_DU(dfx); du != NULL; du = du->next) {
      if (j == 11) {
        j = 0;
        fprintf(gbl.dbgfil, "\n          ");
      }
      j++;
      fprintf(gbl.dbgfil, "  %d", du->use);
    }
    fprintf(gbl.dbgfil, "\n");
    if (DEF_CSEL(dfx)) {
      fprintf(gbl.dbgfil, "        cse:");
      for (du = DEF_CSEL(dfx); du != NULL; du = du->next)
        fprintf(gbl.dbgfil, "  %d", du->use);
      fprintf(gbl.dbgfil, "\n");
    }
  }

  fprintf(gbl.dbgfil, "\n* * * * * *  Uses (%d), Ud  * * * * * *\n",
          opt.useb.stg_avail - 1);
  for (i = 1; i < opt.useb.stg_avail; i++) {
    nme = USE_NM(i);
    fprintf(gbl.dbgfil,
            "%5d.  fg:%-5d  ilt:%-5u  ast:%-5u  addr:%-5u  nme:%-5u", i,
            USE_FG(i), USE_STD(i), USE_AST(i), USE_ADDR(i), nme);
    fprintf(gbl.dbgfil, " \"%s\"", getprint((int)NME_SYM(nme)));
    if (USE_EXPOSED(i))
      fprintf(gbl.dbgfil, " - exposed");
    if (USE_CSE(i))
      fprintf(gbl.dbgfil, " - cse");
    if (!USE_PRECISE(i))
      fprintf(gbl.dbgfil, " - imprecise");
    if (USE_ARG(i))
      fprintf(gbl.dbgfil, " - arg");
    if (USE_DOINIT(i))
      fprintf(gbl.dbgfil, " - doinit");
    fprintf(gbl.dbgfil, "\n       ");
    j = 0;
    if (USE_UD(i) == NULL
        && SCG(basesym_of(USE_NM(i))) == SC_LOCAL &&
        !DINITG(basesym_of(USE_NM(i)))
        && !ADDRTKNG(basesym_of(USE_NM(i)))) {
      fprintf(stderr, "use before def of %s, (%s: %s)\n",
              SYMNAME(basesym_of(USE_NM(i))), gbl.curr_file,
              getprint((int)FG_LABEL(gbl.entbih)));
      fprintf(gbl.dbgfil, "use before def of %s\n",
              SYMNAME(basesym_of(USE_NM(i))));
    }
    for (ud = USE_UD(i); ud != NULL; ud = ud->next) {
      if (j == 11) {
        j = 0;
        fprintf(gbl.dbgfil, "\n       ");
      }
      j++;
      fprintf(gbl.dbgfil, " %-5d", ud->def);
    }
    fprintf(gbl.dbgfil, "\n");
  }

  if (inout) {
    fprintf(gbl.dbgfil, "\n* * * * * *  Defs by Flowgraph  * * * * * *\n");
    for (j = 1; j <= opt.dfn; j++) {
      i = VTX_NODE(j);
      fprintf(gbl.dbgfil, "\n(%5d)  first def: %d\n", i, FG_FDEF(i));
      fprintf(gbl.dbgfil, " IN:");
      bv_print(FG_IN(i), opt.ndefs);
      fprintf(gbl.dbgfil, "OUT:");
      bv_print(FG_OUT(i), opt.ndefs);
    }
  }

  fprintf(gbl.dbgfil, "\n* * * * * *  Info by Loop  * * * * * *\n");
  for (i = 1; i <= opt.nloops; i++) {
    fprintf(gbl.dbgfil, "(%5d)", i);
    if (LP_EXT_STORE(i))
      fprintf(gbl.dbgfil, " ext_store");
    if (LP_PTR_STORE(i))
      fprintf(gbl.dbgfil, " ptr_store");
    if (LP_PTR_LOAD(i))
      fprintf(gbl.dbgfil, " ptr_load");
    if (LP_CALLFG(i))
      fprintf(gbl.dbgfil, " callfg");
    if (LP_CALLINTERNAL(i))
      fprintf(gbl.dbgfil, " callinternal");
    if (LP_QJSR(i))
      fprintf(gbl.dbgfil, " qjsr");
    if (LP_JMP_TBL(i))
      fprintf(gbl.dbgfil, " jmp_tbl");
    fprintf(gbl.dbgfil, "\n");
    {
      STL *p;
      int ii;
      p = LP_STL(i);
      fprintf(gbl.dbgfil, "        stl %08lx  child %08lx  next %08lx\n",
              (long)(p), (long)(p->childlst), (long)(p->nextsibl));
      for (ii = p->store; ii; ii = STORE_NEXT(ii)) {
        fprintf(gbl.dbgfil, "        %5d. nme:", ii);
        dumpname((int)STORE_NM(ii));
        fprintf(gbl.dbgfil, ", mark: %d\n", STORE_TYPE(ii));
      }
    }
  }

}

/** \brief Add use items for optimizable variables which occur in the ili
   expression
    which is a new expression being added to a flowgraph node (where) such
    as the preheader of the loop.

    This is necessary to track new uses of a variable added after the flow
    information has been created.  For example, it's important that the
    induction process adds uses in the loop count value used by a bla
    instruction since this expression was not seen by the flow analysis;
    a later call to induction will not find all of an induction variable's
    uses if it was used in the bla count of a contained loop.

    Along with creating new uses, the du's of the defs reaching the uses
    are updated.

    WARNING:  this will only work when the expression is stored into
    an optimizer created temporary; to be more general (i.e., when adding
    stores of user-variables), new def information will have to be added
    which is impossible since the size of each bit vector cannot be
    increased.
 */
void
add_new_uses(int loop, int where, int newilt, int expr)
{
  first_use = opt.useb.stg_avail; /* mark where the new uses begin */
  num_uses = 0;                   /* number of new uses added */
  use_hash_reset();               /* hash of new uses added */
  cur_fg = where;
  cur_std = newilt;
  cur_lp = loop;
#if DEBUG
  if (OPTDBG(9, 128))
    fprintf(gbl.dbgfil, "add_new_uses for fg %d, ilt %d, expr: %d, loop, %d\n",
            cur_fg, cur_fg, expr, cur_lp);
#endif
  new_ud(expr);
  new_du_ud();

}

/*
 * recursively search for uses of optimizable scalars and create
 * use items.
 */
static void
new_ud(int astx)
{
}

/*
 * compute the du chains for the definitions and the ud chains for the new
 * uses.  The defs reaching the uses are the defs which are members of the
 * union of the OUT sets for all of the predecessors of the loop not in the
 * loop.
 */
static void
new_du_ud(void)
{
  PSI_P p;
  int i, nme, def;
  BV *bv;

  bv = opt.def_setb.stg_base; /* scratch bit vector */
  bv_zero(bv, def_bv_len);    /* bv <-- 0 */
  for (p = FG_PRED(LP_HEAD(cur_lp)); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (FG_LOOP(i = PSI_NODE(p)) == cur_lp)
      continue;
    if (FG_OUT(i) != NULL)
      /*
       *  bv <- bv U OUT(i)
       *
       *  Note:  since this occurs after the compuatation of the flow
       *  equations, we may see flowgraph nodes which were added
       *  after the creation of the flowgraph.  For these nodes, the
       *  IN & OUT fields are NULL.
       *
       */
      bv_union(bv, FG_OUT(i), def_bv_len);
  }
#if DEBUG
  if (OPTDBG(9, 128)) {
    fprintf(gbl.dbgfil, "union of defs\n");
    bv_print(bv, opt.ndefs);
  }
#endif
  for (i = first_use; i < opt.useb.stg_avail; i++) {
#if DEBUG
    assert(USE_EXPOSED(i), "new_ud_du - use not exposed", i, 3);
#endif
    nme = USE_NM(i);
    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      if (bv_mem(bv, def)) {
        add_du_ud(def, i);
/*
 * need to watch out for the case where the def was marked
 * delete (its ilt is marked).  store deletion causes a problem
 * only when an expression has been "copied"; for example,
 * when the initial value of an induction variable is used to
 * compute new initial values of any derived induction
 * variables.  The "copied" expression could contain a use of
 * a def which has been marked.
 *
 * The new use could be in a block which is eventually merged
 * with the block containing the def. block merging takes
 * takes place much later and at this time, not all of the
 * information is available to safely determine if the
 * blocks are mergeable.
 */
#if DEBUG
        if (OPTDBG(9, 128)) {
          fprintf(gbl.dbgfil, "new_ud_du: use %d reached by def %d\n", i, def);
          if (STD_DELETE(DEF_STD(def)))
            fprintf(gbl.dbgfil, "new_ud_du: undelete def %d \n", def);
        }
#endif
        STD_DELETE(DEF_STD(def)) = 0; /* always clear flag */
      }
    }
  }
}


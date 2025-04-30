/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file optimize.h
    \brief arious definitions for the optimizer module
*/

/*  DEBUG-controlled -q stuff  */

#define OPTDBG(x, y) (DEBUG && DBGBIT(x, y))

/* * * * *  symbol table macros for just the optimizer  * * * * */

#define ISSCALAR(s) TY_ISSCALAR(DTY(DTYPEG(s)))
/* FTN's storage classes */
#define IS_LCL(s) (SCG(s) == SC_LOCAL || SCG(s) == SC_PRIVATE)
#define IS_EXTERN(s) (SC_ISCMBLK(SCG(s)) || SCG(s) == SC_EXTERN)
#define IS_STATIC(s) (SCG(s) == SC_STATIC)
#define IS_CMNBLK(s) (SC_ISCMBLK(SCG(s)))
#define IS_DUM(s) (SCG(s) == SC_DUMMY)
#define IS_LCL_OR_DUM(s) (IS_LCL(s) || IS_DUM(s))
#define IS_REGARG(s) (REGARGG(s) && REDUCG(s))

#define IS_PRIVATE(s) (SCG(s) == SC_PRIVATE)

/* * * * *  storage allocation macros  * * * * */

#define OPT_ALLOC(stgb, dt, sz) \
  {                             \
    NEW(stgb.stg_base, dt, sz); \
    stgb.stg_size = sz;         \
  }

#define OPT_NEXT(stb) stb.stg_avail++

#define OPT_NEED(stb, dt, sz)                                               \
  {                                                                         \
    NEED(stb.stg_avail, stb.stg_base, dt, stb.stg_size, stb.stg_size + sz); \
  }

#define OPT_FREE(stb)    \
  {                      \
    FREE(stb.stg_base);  \
    stb.stg_base = NULL; \
    stb.stg_size = 0;    \
    stb.stg_avail = 0;   \
  }

typedef unsigned short US;

/* * * * *  getitem (salloc.c) Areas Used  * * * * */

#define PSI_AREA 0    /* pred/succ area */
#define BUCKET_AREA 1 /* item area for dominators */
#define IUSE_AREA 1   /* induction use area */
#define TOPI_AREA 2   /* item area for topological sort */
#define Q_AREA 2      /* queue area (used local to a routine */
#define DU_AREA 3     /* definition use and use def area */
#define STL_AREA 5    /* store list area */
#define DLT_AREA                                   \
  9 /* list of stds to be deleted in flow(); can't \
     * be shared with flowgraph() and findloops(). \
     */

/* * * * *  Queue list Item  * * * * */

typedef struct Q_TAG {
  int info;
  int flag;
  struct Q_TAG *next;
} Q_ITEM;
#define Q_NULL NULL
#define GET_Q_ITEM(q) q = (Q_ITEM *)getitem(Q_AREA, sizeof(Q_ITEM))

/* * * * *  Predecessor/Successor Item  * * * * */

typedef struct PSI_TAG *PSI_P;
#define PSI_P_NULL NULL

typedef struct PSI_TAG {
  int node;
  union {
    UINT all;
    struct {
      unsigned f1 : 1;
      unsigned f2 : 1;
      unsigned f3 : 1;
      unsigned f4 : 1;
      unsigned f5 : 1;
      unsigned f6 : 1;
      unsigned f7 : 1;
      unsigned f8 : 1;
      unsigned f9 : 1;
      unsigned f10 : 1;
      unsigned f11 : 1;
      unsigned f12 : 1;
      unsigned f13 : 1;
      unsigned f14 : 1;
      unsigned f15 : 1;
      unsigned f16 : 1;
    } bits;
  } flags;
  PSI_P next;
} PSI;

#define PSI_NODE(i) ((i)->node)
#define PSI_ALL(i) ((i)->flags.all)
#define PSI_FT(i) ((i)->flags.bits.f1)
#define PSI_ZTRP(i) ((i)->flags.bits.f2)
#define PSI_EXIT(i) ((i)->flags.bits.f4)
#define PSI_NEXT(i) ((i)->next)
#define GET_PSI(i) i = (PSI *)getitem(PSI_AREA, sizeof(PSI))

/* * * * *  Bit Vector Set Representation  * * * * */

#define BITS_PER_BYTE 8

typedef int BV;
typedef struct {
  BV *stg_base;
  int stg_avail;
  BV *nme_base;
  int nme_avail;
} SET_BASE;

#define BV_BITS (sizeof(BV) * BITS_PER_BYTE)

/* * * * *  Optimizer Augmented AST * * * * */

typedef struct {/* foreach ast index */
  int invar;    /* field for marking an AST invariant/variant */
} OAST;

/* * * * *  Flow Graph Node  * * * * */

typedef struct {
  int lineno;  /* line number of the node */
  int first;   /* first std in node */
  int last;    /* last std in node */
  int lnext;   /* next lexical node */
  int lprev;   /* previous lexical node */
  int dfn;     /* depth first number */
  int dom;     /* immediate dominator */
  int rdfn;    /* reverse graph depth first number */
  int rdfo;    /* reverse depth first number */
  int pdom;    /* immediate post-dominator */
  int vtx;     /* sorted vertex-dfn array */
  int rvtx;    /* sorted vertex-rdfn array */
  int rdfovtx; /* sorted vertex-rdfo array */
  int loop;    /* loop containing the node */
  int next;    /* next in regions list */
  int natnxt;  /* next in natural loop list */
  int fdef;    /* first definition */
  int scr;     /* scratch field -- BIH_ASSN for vectorizer */
  int scr2;    /* scratch field -- BIH_RGSET for vectorizer */
  int atomic;  /* in atomic-update region */
  PSI_P pred;  /* predecssor list */
  PSI_P succ;  /* successor list */
  BV *in;
  BV *out;
  BV *uninited; /* set if variable is not assigned/initialized,
                   currently ignore host program assignment */
  int par;	/* nest level of parallelism */
  int parloop;  /* nest level of parallelism */
  union {
    struct {
      unsigned ft : 1;
      unsigned head : 1;
      unsigned tail : 1;
      unsigned ex : 1;
      unsigned en : 1;
      unsigned xt : 1;
      unsigned qjsr : 1; /* used only by the optimizer */
      unsigned mexits : 1;
      unsigned innermost : 1;
      unsigned ztrp : 1;
      unsigned done : 1;
      unsigned inq : 1;
      unsigned visited : 1;
      unsigned ptr_store : 1; /* store via pointer */
      unsigned jmp_tbl : 1;   /* a jump uses a jump table */
      unsigned cs : 1;     /* in a critical section */
      unsigned master : 1; /* in a master/serial section */
      unsigned parsect : 1;
      unsigned task : 1;     /* in a task */
      unsigned ctlequiv : 1; /* control-equivalent to loop header */
      unsigned mark : 1;
      unsigned sdscunsafe : 1;
      unsigned malf_lp : 1; /* in a malformed loop */
      unsigned spare : 7;
    } bits;
    UINT all;
  } flags;
} FG;

#define FG_LINENO(i) opt.fgb.stg_base[i].lineno
#define FG_STDFIRST(i) opt.fgb.stg_base[i].first
#define FG_STDLAST(i) opt.fgb.stg_base[i].last
#define FG_LABEL(i) STD_LABEL(FG_STDFIRST(i))
#define FG_LNEXT(i) opt.fgb.stg_base[i].lnext
#define FG_LPREV(i) opt.fgb.stg_base[i].lprev
#define FG_DFN(i) opt.fgb.stg_base[i].dfn
#define FG_DOM(i) opt.fgb.stg_base[i].dom
#define FG_RDFN(i) opt.fgb.stg_base[i].rdfn
#define FG_RDFO(i) opt.fgb.stg_base[i].rdfo
#define FG_PDOM(i) opt.fgb.stg_base[i].pdom
#define FG_PRED(i) opt.fgb.stg_base[i].pred
#define FG_SUCC(i) opt.fgb.stg_base[i].succ
#define FG_LOOP(i) opt.fgb.stg_base[i].loop
#define FG_NEXT(i) opt.fgb.stg_base[i].next
#define FG_NATNXT(i) opt.fgb.stg_base[i].natnxt
#define FG_FDEF(i) opt.fgb.stg_base[i].fdef
#define FG_IN(i) opt.fgb.stg_base[i].in
#define FG_OUT(i) opt.fgb.stg_base[i].out
#define FG_UNINITED(i) opt.fgb.stg_base[i].uninited
#define FG_ATOMIC(i) opt.fgb.stg_base[i].atomic
#define FG_FT(i) opt.fgb.stg_base[i].flags.bits.ft
#define FG_HEAD(i) opt.fgb.stg_base[i].flags.bits.head
#define FG_TAIL(i) opt.fgb.stg_base[i].flags.bits.tail
#define FG_EX(i) opt.fgb.stg_base[i].flags.bits.ex
#define FG_EXSDSCUNSAFE(i) opt.fgb.stg_base[i].flags.bits.sdscunsafe
#define FG_EN(i) opt.fgb.stg_base[i].flags.bits.en
#define FG_XT(i) opt.fgb.stg_base[i].flags.bits.xt
#define FG_QJSR(i) opt.fgb.stg_base[i].flags.bits.qjsr
#define FG_MEXITS(i) opt.fgb.stg_base[i].flags.bits.mexits
#define FG_INNERMOST(i) opt.fgb.stg_base[i].flags.bits.innermost
#define FG_ZTRP(i) opt.fgb.stg_base[i].flags.bits.ztrp
#define FG_DONE(i) opt.fgb.stg_base[i].flags.bits.done
#define FG_INQ(i) opt.fgb.stg_base[i].flags.bits.inq
#define FG_VISITED(i) opt.fgb.stg_base[i].flags.bits.visited
#define FG_PTR_STORE(i) opt.fgb.stg_base[i].flags.bits.ptr_store
#define FG_JMP_TBL(i) opt.fgb.stg_base[i].flags.bits.jmp_tbl
#define FG_PAR(i) opt.fgb.stg_base[i].par
#define FG_CS(i) opt.fgb.stg_base[i].flags.bits.cs
#define FG_MASTER(i) opt.fgb.stg_base[i].flags.bits.master
#define FG_PARLOOP(i) opt.fgb.stg_base[i].parloop
#define FG_PARSECT(i) opt.fgb.stg_base[i].flags.bits.parsect
#define FG_TASK(i) opt.fgb.stg_base[i].flags.bits.task
#define FG_CTLEQUIV(i) opt.fgb.stg_base[i].flags.bits.ctlequiv
#define FG_MARK(i) opt.fgb.stg_base[i].flags.bits.mark
#define FG_MALF_LP(i) opt.fgb.stg_base[i].flags.bits.malf_lp
#define FG_ALL(i) opt.fgb.stg_base[i].flags.all
#define VTX_NODE(i) opt.fgb.stg_base[i].vtx
#define RVTX_NODE(i) opt.fgb.stg_base[i].rvtx
#define RDFOVTX_NODE(i) opt.fgb.stg_base[i].rdfovtx

/*** BIH-related flags for which there are HPF equivalents in the flowgraph ***/
/*** Needed for modules which are shared with pghpf                         ***/
#define FG_BIH(i) (i)
#define FG_TO_BIH(i) FG_BIH(i)
#define BIH_TO_FG(i) (i)
#define BIH_LINENO(i) FG_LINENO(i)
#define BIH_LABEL(i) FG_LABEL(i)
#define BIH_ILTFIRST(i) FG_STDFIRST(i)
#define BIH_ILTLAST(i) FG_STDLAST(i)
#define BIH_NEXT(i) FG_LNEXT(i)
#define BIH_PREV(i) FG_LPREV(i)
#define BIH_ASSN(i) opt.fgb.stg_base[i].scr
#define BIH_RGSET(i) opt.fgb.stg_base[i].scr2
#define BIH_FLAGS(i) FG_FLAGS(i)
#define BIH_FT(i) FG_FT(i)
#define BIH_EN(i) FG_EN(i)
#define BIH_EX(i) FG_EX(i)
#define BIH_EXSDSCUNSAFE(i) FG_EXSDSCUNSAFE(i)
#define BIH_XT(i) FG_XT(i)
#define BIH_ZTRP(i) FG_ZTRP(i)
#define BIH_HEAD(i) FG_HEAD(i)
#define BIH_TAIL(i) FG_TAIL(i)
#define BIH_INNERMOST(i) FG_INNERMOST(i)
#define BIH_QJSR(i) FG_QJSR(i)
#define BIH_MEXITS(i) FG_MEXITS(i)
#define BIH_PAR(i) FG_PAR(i)
#define BIH_CS(i) FG_CS(i)
#define BIH_MASTER(i) FG_MASTER(i)
#define BIH_PARLOOP(i) FG_PARLOOP(i)
#define BIH_PARSECT(i) FG_PARSECT(i)
#define BIH_TASK(i) FG_TASK(i)

/*** ILT-related flags for which there are HPF equivalents in the STD ***/
/*** Needed for modules which are shared with pghpf                   ***/

#define NEW_NODE(after) add_fg(after)

/* * * * *  Flowgraph Edge  * * * * */

typedef struct {
  int pred; /* the edge (pred, succ) */
  int succ;
  int next; /* next edge with same succ ("head"), initially -1 */
} EDGE;

#define EDGE_PRED(i) opt.rteb.stg_base[i].pred
#define EDGE_SUCC(i) opt.rteb.stg_base[i].succ
#define EDGE_NEXT(i) opt.rteb.stg_base[i].next
#define NUM_RTE (opt.rteb.stg_avail)

/* * * * *  Stores in a Loop  * * * * */

typedef struct {/* store table entry */
  int nm;       /* names of item being stored */
  INT16 type;   /* type of store:
                 * 0 - addr is "constant"
                 * 1 - addr is "variable"
                 */
  UINT16 next;  /* next store item in loop; 0 terminates list */
} STORE;

#define STORE_NM(i) opt.storeb.stg_base[i].nm
#define STORE_TYPE(i) opt.storeb.stg_base[i].type
#define STORE_NEXT(i) opt.storeb.stg_base[i].next

typedef struct STL_TAG {/* list item for stores in a loop */
  UINT16 store;         /* index into store table of first store */
  INT16 unused;
  struct STL_TAG *nextsibl; /* next STL for any sibling loops
                             * ie. (next node in parent's child list
                             */
  struct STL_TAG *childlst; /* child list STL for any nested loops */
} STL;

/* * * * *  Loop Table  * * * * */

typedef struct {
  int level;   /* level number of loop */
  int parent;  /* parent of loop */
  int head;    /* head (FG node) of the loop */
  int tail;    /* tail (FG node) of the loop */
  int fg;      /* head of the region list */
  int loop;    /* loop for the sorted list */
  int edge;    /* retreating edge index */
  int child;   /* loop immediately enclosed by loop; 0 if
                * innermost
                */
  int sibling; /* next loop at the same level of the loop;
                * the loops immediately enclosed by a loop
                * are represented by the list beginning with
                * child, and linked using the sibling field.
                * a sibling value of 0 terminates the list.
                */
  int parloop; /* loop is to be executed in parallel; nest level of parallelism */
  union {
    struct {
      unsigned innermost : 1;    /* loop is an innermost loop */
      unsigned callfg : 1;       /* loop or any enclosed loop contains
                                  * external calls
                                  */
      unsigned ext_store : 1;    /* loop or any enclosed loop contains a
                                  * store into an external variable or a
                                  * variable with its & taken
                                  */
      unsigned ptr_store : 1;    /* loop or any enclosed loop contains a
                                  * store via a pointer
                                  */
      unsigned zerotrip : 1;     /* loop was converted from a
                                  * while or for to if-do-while
                                  */
      unsigned ptr_load : 1;     /* loop or any enclosed loop contains a
                                  * load via a pointer
                                  */
      unsigned nobla : 1;        /* loop or any enclosed loop contains a
                                  * block with its BIH_NOBLA flag set.
                                  */
      unsigned qjsr : 1;         /* loop or any enclosed loop contains a
                                  * block with BIH_QJSR set
                                  */
      unsigned mexits : 1;       /* loop contains multiple exits */
      unsigned jmp_tbl : 1;      /* loop or any enclosed loop contains a
                                  * jump using a jump tbl
                                  */
      unsigned invarif : 1;      /* invarif loop optz. performed for loop or
                                  * a descendant
                                  */
      unsigned cs : 1;           /* region contains a block with BIH_CS set;
                                  * not propagated to its parent's LP_CS.
                                  */
      unsigned csect : 1;        /* loop or any enclosed loop contains a block
                                  * with BIH_CS set.
                                  */
      unsigned mark : 1;         /* general save area */
      unsigned forall : 1;       /* loop is a forall 'loop' */
      unsigned master : 1;       /* loop contains a master/serial region */
      unsigned parregn : 1;      /* region contains a block with BIH_PAR set:
                                  * 1.  the loop may be a parallel loop
                                  *     (LP_PARLOOP is set),
                                  * 2.  the loop is not executed in parallel
                                  *     but contains a parallel region (the
                                  *     loop's head BIH_PAR is not set),
                                  * 3.  the loop is contained within a parallel
                                  *     region (the LP_PARLOOP is not set).
                                  */
      unsigned parsect : 1;      /* region contains a block with BIH_PARSECT
                                  * set.
                                  */
      unsigned callinternal : 1; /* contains call to internal subprogram */
      unsigned xtndrng : 1;      /* extended range loop */
      unsigned cncall : 1;       /* all calls in loop are call safe */
      unsigned vollab : 1;       /* loop contains block labeled volatile */
      unsigned sdscunsafe : 1;   /* loop contains a call to a routine that
                                  * modifies section descriptors
                                  */
      unsigned task : 1;         /* loop contains a task */
      unsigned tail_aexe : 1;    /* tail is always executed */
      unsigned spare : 6;
    } bits;
    UINT all;
  } flags;
  STL *stl;        /* list of stores in loop and
                    * contained loops
                    */
  BV *lin;         /* set of variables live-in to loop */
  BV *lout;        /* set of variables live-out of loop */
  PSI_P exits;     /* list of natural loop exits - for live var*/
  Q_ITEM *stl_par; /* list of nmes of non-private vars which are
                    * assigned while in a critical section of
                    * a loop or a contained loop, a parallel
                    * section of a loop or a contained loop,
                    * or in a parallel region.
                    */
  int count;       /* number of loops enclosed by the loop,
                    * inclusive.
                    */
  int hstdf;       /* invariant: list of stmt will be hoisted out of loop */
  int dstdf;       /* invariant: deallocate stmts will drop out of loop */
} LP;

#define LP_LEVEL(i) opt.lpb.stg_base[i].level
#define LP_PARENT(i) opt.lpb.stg_base[i].parent
#define LP_HEAD(i) opt.lpb.stg_base[i].head
#define LP_TAIL(i) opt.lpb.stg_base[i].tail
#define LP_FG(i) opt.lpb.stg_base[i].fg
#define LP_LOOP(i) opt.lpb.stg_base[i].loop
#define LP_EDGE(i) opt.lpb.stg_base[i].edge
#define LP_CHILD(i) opt.lpb.stg_base[i].child
#define LP_SIBLING(i) opt.lpb.stg_base[i].sibling
#define LP_RGSET(i) opt.lpb.stg_base[i].rgset
#define LP_INNERMOST(i) opt.lpb.stg_base[i].flags.bits.innermost
#define LP_CALLFG(i) opt.lpb.stg_base[i].flags.bits.callfg
#define LP_CALLSDSCUNSAFE(i) opt.lpb.stg_base[i].flags.bits.sdscunsafe
#define LP_CALLINTERNAL(i) opt.lpb.stg_base[i].flags.bits.callinternal
#define LP_EXT_STORE(i) opt.lpb.stg_base[i].flags.bits.ext_store
#define LP_PTR_STORE(i) opt.lpb.stg_base[i].flags.bits.ptr_store
#define LP_PTR_LOAD(i) opt.lpb.stg_base[i].flags.bits.ptr_load
#define LP_ZEROTRIP(i) opt.lpb.stg_base[i].flags.bits.zerotrip
#define LP_NOBLA(i) opt.lpb.stg_base[i].flags.bits.nobla
#define LP_QJSR(i) opt.lpb.stg_base[i].flags.bits.qjsr
#define LP_MEXITS(i) opt.lpb.stg_base[i].flags.bits.mexits
#define LP_JMP_TBL(i) opt.lpb.stg_base[i].flags.bits.jmp_tbl
#define LP_INVARIF(i) opt.lpb.stg_base[i].flags.bits.invarif
#define LP_PARLOOP(i) opt.lpb.stg_base[i].parloop
#define LP_CS(i) opt.lpb.stg_base[i].flags.bits.cs
#define LP_CSECT(i) opt.lpb.stg_base[i].flags.bits.csect
#define LP_MARK(i) opt.lpb.stg_base[i].flags.bits.mark
#define LP_FORALL(i) opt.lpb.stg_base[i].flags.bits.forall
#define LP_MASTER(i) opt.lpb.stg_base[i].flags.bits.master
#define LP_PARREGN(i) opt.lpb.stg_base[i].flags.bits.parregn
#define LP_PARSECT(i) opt.lpb.stg_base[i].flags.bits.parsect
#define LP_XTNDRNG(i) opt.lpb.stg_base[i].flags.bits.xtndrng
#define LP_CNCALL(i) opt.lpb.stg_base[i].flags.bits.cncall
#define LP_VOLLAB(i) opt.lpb.stg_base[i].flags.bits.vollab
#define LP_TASK(i) opt.lpb.stg_base[i].flags.bits.task
#define LP_TAIL_AEXE(i) opt.lpb.stg_base[i].flags.bits.tail_aexe
#define LP_ALL(i) opt.lpb.stg_base[i].flags.all
#define LP_STL(i) opt.lpb.stg_base[i].stl
#define LP_LIN(i) opt.lpb.stg_base[i].lin
#define LP_LOUT(i) opt.lpb.stg_base[i].lout
#define LP_EXITS(i) opt.lpb.stg_base[i].exits
#define LP_STL_PAR(i) opt.lpb.stg_base[i].stl_par
#define LP_COUNT(i) opt.lpb.stg_base[i].count
#define LP_HSTDF(i) opt.lpb.stg_base[i].hstdf
#define LP_DSTDF(i) opt.lpb.stg_base[i].dstdf

typedef struct UD_TAG {/* def list for a use */
  int def;             /* index into def table */
  struct UD_TAG *next;
} UD;

/* * * * *  Scalar Uses  * * * * */

typedef struct {/* use table entry */
  int nm;
  int fg;
  int std;
  int ast;              /* A_ID ast of the use */
  int addr;             /* address ast of the use */
  union {
    int all;
    struct {
      unsigned exposed : 1; /* use is reached from the beginning
                             * of the block
                             */
      unsigned cse : 1;     /* use is cse of a variable which is
                             * due to the postfix operator
                             */
      unsigned precise : 1; /* address is precise/function-invariant */
      unsigned arg : 1;     /* use is an argument use */
      unsigned doinit : 1;  /* use created when creating a doinit def */
      unsigned mark1 : 1;   /* mark used by fusion, others? */
      unsigned mark2 : 1;   /* mark used by fusion, others? */
      unsigned loop : 1;    /* added at loop entry */
      unsigned aggr : 1;    /* use is of an aggregate */
    } bits;
  } flags;
  UD *ud;               /* defs reaching this use */
} USE;

#define USE_NM(i) opt.useb.stg_base[i].nm
#define USE_FG(i) opt.useb.stg_base[i].fg
#define USE_STD(i) opt.useb.stg_base[i].std
#define USE_AST(i) opt.useb.stg_base[i].ast
#define USE_ADDR(i) opt.useb.stg_base[i].addr
#define USE_EXPOSED(i) opt.useb.stg_base[i].flags.bits.exposed
#define USE_CSE(i) opt.useb.stg_base[i].flags.bits.cse
#define USE_PRECISE(i) opt.useb.stg_base[i].flags.bits.precise
#define USE_ARG(i) opt.useb.stg_base[i].flags.bits.arg
#define USE_DOINIT(i) opt.useb.stg_base[i].flags.bits.doinit
#define USE_MARK1(i) opt.useb.stg_base[i].flags.bits.mark1
#define USE_MARK2(i) opt.useb.stg_base[i].flags.bits.mark2
#define USE_LOOPENTRY(i) opt.useb.stg_base[i].flags.bits.loop
#define USE_AGGR(i) opt.useb.stg_base[i].flags.bits.aggr
#define USE_UD(i) opt.useb.stg_base[i].ud

typedef struct DU_TAG {/* use list for a definition */
  int use;             /* index into use table */
  struct DU_TAG *next;
} DU;

/* * * * *  Scalar Definitions  * * * * */
/*  NOTES:
 *  A called function implicitly defines those variables which can be affected
 *  by a call (e.g., variables, file static variables, &, etc.).  In
 *  order to track these implicit definitions, definition 1 is reserved as
 *  the def implied by a call.  If a block contains a call, definition 1 will
 *  be in its GEN set.  We use just 1 definition instead of adding definitions
 *  for each of the call-affected variables to the block containing the call.
 *  The IN and OUT sets of a block will contain this def if any call can reach
 *  the block.
 *  A store via pointer implicitly defines those variables whose address
 *  has been taken.  Definition 2 is reserved as the def implied by a store
 *  via a pointer.  If a block contains a store via a pointer, def 2 will be
 *  in its GEN set.
 *
 *  Definition 3 is the first user def.
 */
typedef struct {/* def table entry */
  int fg;
  int std;
  int nm;
  int lhs;   /* lefthand side of the def - A_ID ast of the
              * destination */
  int addr;  /* address ast of the destination */
  int rhs;   /* righthand side of the def - source ast */
  int next;  /* next def for this names */
  int lnext; /* next definition in fg; 0 terminates list*/
  union {
    UINT all;
    struct {
      unsigned gen : 1;     /* def reaches end of the block */
      unsigned cnst : 1;    /* def's value is a constant. */
      unsigned delete : 1;  /* def has been deleted */
      unsigned self : 1;    /* sym defined appears in def expr */
      unsigned confl : 1;   /* def's msize conflicts with a use's msize */
      unsigned doinit : 1;  /* initial def for a do/forall */
      unsigned doend : 1;   /* increment def for a enddo/endforall */
      unsigned arg : 1;     /* sym appears as an actual argument*/
      unsigned other : 1;   /* other implicit def: allocate status, etc. */
      unsigned precise : 1; /* address is precise/function-invariant */
      unsigned mark1 : 1;
      unsigned mark2 : 1;
      unsigned loop : 1;    /* added at loop entry */
      unsigned aggr : 1;    /* def is of an aggregate */
    } bits;
  } flags;
  DU *du;   /* uses reached by this def */
  DU *csel; /* cse uses not reached by this def; occurs
             * for postfix expressions:  i = j = k++;
             * k's csel includes the cse uses of k stored
             * into i and j.
             */
} DEF;

#define DEF_FG(i) opt.defb.stg_base[i].fg
#define DEF_STD(i) opt.defb.stg_base[i].std
#define DEF_NM(i) opt.defb.stg_base[i].nm
#define DEF_LHS(i) opt.defb.stg_base[i].lhs
#define DEF_ADDR(i) opt.defb.stg_base[i].addr
#define DEF_RHS(i) opt.defb.stg_base[i].rhs
#define DEF_NEXT(i) opt.defb.stg_base[i].next
#define DEF_LNEXT(i) opt.defb.stg_base[i].lnext
#define DEF_ALL(i) opt.defb.stg_base[i].flags.all
#define DEF_GEN(i) opt.defb.stg_base[i].flags.bits.gen
#define DEF_CONST(i) opt.defb.stg_base[i].flags.bits.cnst
#define DEF_DELETE(i) opt.defb.stg_base[i].flags.bits.delete
#define DEF_SELF(i) opt.defb.stg_base[i].flags.bits.self
#define DEF_CONFL(i) opt.defb.stg_base[i].flags.bits.confl
#define DEF_DOINIT(i) opt.defb.stg_base[i].flags.bits.doinit
#define DEF_DOEND(i) opt.defb.stg_base[i].flags.bits.doend
#define DEF_ARG(i) opt.defb.stg_base[i].flags.bits.arg
#define DEF_OTHER(i) opt.defb.stg_base[i].flags.bits.other
#define DEF_PRECISE(i) opt.defb.stg_base[i].flags.bits.precise
#define DEF_MARK1(i) opt.defb.stg_base[i].flags.bits.mark1
#define DEF_MARK2(i) opt.defb.stg_base[i].flags.bits.mark2
#define DEF_LOOPENTRY(i) opt.defb.stg_base[i].flags.bits.loop
#define DEF_AGGR(i) opt.defb.stg_base[i].flags.bits.aggr
#define DEF_DU(i) opt.defb.stg_base[i].du
#define DEF_CSEL(i) opt.defb.stg_base[i].csel

/* Predefined defs.
 * Define def values for the definition implied by a call, a store via a
 * pointer and the first user definition.
 */
#define CALL_DEF 1
#define PTR_STORE_DEF 2
#define QJSR_DEF 3
#define FIRST_DEF 4

/* * * * *  Invariant ILI Attributes * * * * */

#define NOT_INV -1
#define INV -2
#define T_INV -3 /* probably not used */

#define AST_INVG(i) opt.astb.stg_base[i].invar
#define AST_INVP(i, j) opt.astb.stg_base[i].invar = (j)
#define AST_ISINV(i) (AST_INVG(i) < -1)
#define AST_ISINV_TEMP(i) (AST_INVG(i) < -1)
#define IS_INVARIANT(i) (AST_ISINV(i) || AST_ISINV_TEMP(i))

#define ILI_ISINV(i) AST_ISINV(i)

/* * * * *  optimizer global data  * * * * */

typedef struct {
  int num_nodes; /* number of nodes in a flow graph */
  int dfn;       /* number of nodes which have dfn's */
  STG_DECLARE(fgb, FG); /* flow graph memory area */
  STG_DECLARE(rteb, EDGE);
  int exitfg;  /* fg of a functions's exit block */
  int nloops;  /* number of loops in a function */
  STG_DECLARE(lpb, LP); /* pointer to loop table */
  STG_DECLARE(storeb, STORE);
  STG_DECLARE(defb, DEF);
  STG_DECLARE(useb, USE);
  STG_DECLARE(invb, int);
  int nsyms;
  int ndefs;
  SET_BASE def_setb;
  int pre_fg;     /* preheader flowgraph node of a loop */
  int exit_fg;    /* exit flowgraph node of a loop */
  int rat;        /* rat pointer for the current loop */
  int head_label; /* counter for new loop header labels */
  int exit_label; /* counter for exit block labels */
  struct {        /* countable loop info found in induction
                   * and used by peephole
                   */
    int top;      /* label of top of loop */
    int cnt;      /* ast ptr of loop count */
    int cnt_sym;  /* loop count temporary if reg is needed */
    int skip;     /* ast ptr of loop skip */
    int branch;   /* branch ast to be replaced */
  } cntlp;
  int zstride_ast; /* list of ZSTRIDE check ast which need to be
                    * added before a loop; uses an undefined
                    * operand (# MAX_OPNDS) to link together
                    * (cleared when added ast added to block).
                    */
  int sc;          /* storage class used for optimizer-created
                    * temporaries (SC_LOCAL, SC_PRIVATE).
                    */
  STG_DECLARE(astb, OAST);
} OPT;

extern OPT opt;

/*****  optimize.c *****/
void optshrd_init(void);
void optshrd_finit(void);
void optshrd_fend(void);
void optshrd_end(void);

#define HLOPT_INDUC 0x1
#define HLOPT_ENDTEST 0x2
#define HLOPT_ALL 0x3 /* 'or' of all HLOPT_... bits */
void hlopt_init(int);
void hlopt_end(int, int);

void optimize(int);
void add_loop_preheader(int);
void add_loop_exit(int);
void add_single_loop_exit(int);

/*****  fgraph.c *****/
void flowgraph(void);
int add_fg(int);
void delete_fg(int);
PSI_P add_succ(int, int);
PSI_P add_pred(int, int);
void rm_edge(int, int);
int add_to_parent(int, int);
void dump_node(int);
void rdilts(int);
void wrilts(int);
void unlnkilt(int, int, int);
void delilt(int);
void dmpilt(int);
void dump_flowgraph(void);

/*****  findloop.c *****/
#include "findloop.h"

/*****  flow.c *****/
void flow(void);
void flow_end(void);
int update_stl(int, int);
LOGICAL is_live_in(int, int);
LOGICAL is_live_out(int, int);
void delete_stores(void);
void use_before_def(void);
void add_new_uses(int, int, int, int);

/*****  invar.c *****/
void invariant(int);
void invariant_nomotion(int);
void invariant_mark(int, int);
void invariant_unmark(void);
void invariant_unmarkv(void);
LOGICAL is_invariant(int);
LOGICAL is_sym_invariant_safe(int, int);

/*****  induc.c *****/
void induction_init(void);
void induction_end(void);
void induction(int);
int get_loop_count(int);
void compute_last_values(int, int);
void end_loop_count(void);

/*****  optutil.c *****/
void bv_zero(int *, int);
void bv_copy(int *, int *, int);
void bv_union(int *, int *, int);
void bv_sub(int *, int *, int);
void bv_set(int *, int);
void bv_off(int *, int);
LOGICAL bv_notequal(int *, int *, int);
LOGICAL bv_mem(int *, int);
void bv_print(int *, int);
int get_otemp(void);
LOGICAL is_optsym(int);
LOGICAL is_sym_optsafe(int, int);
LOGICAL is_sym_live_safe(int, int);
LOGICAL is_call_safe(int);
LOGICAL is_ptr_safe(int);
LOGICAL is_sym_ptrsafe(int);
int pred_of_loop(int);
int find_rdef(int, int, LOGICAL);
LOGICAL is_sym_exit_live(int);
LOGICAL is_sym_imp_live(int);
LOGICAL is_sym_entry_live(int);
LOGICAL is_store_via_ptr(int);
LOGICAL can_copy_def(int, int, LOGICAL);
LOGICAL def_ok(int, int, int);
LOGICAL is_avail_expr(int, int, int, int, int);
LOGICAL is_call_in_path(int, int, int, int);
LOGICAL is_ptr_in_path(int, int, int, int);
LOGICAL single_ud(int);
LOGICAL only_one_ud(int);
LOGICAL is_def_imp_live(int);
void rm_def_rloop(int, int);
int copy_to_loop(int, int);
void points_to(void);         /* pointsto.c */
void f90_fini_pointsto(void); /* pointsto.c */
LOGICAL lhs_needtmp(int, int, int);
void postdominators(void);
void findlooptopsort(void);
void reorderloops();
void putstdpta(int);
void putstdassigns(int);
void unconditional_branches(void);
void bv_intersect(BV *a, BV *b, UINT len);
void bv_intersect3(BV *a, BV *b, BV *c, UINT len);
void optimize_alloc(void);                     /* commopt.c */
void points_to_anal(void);                     /* pointsto.c */
void fini_points_to_all(void);                 /* pointsto.c */
bool pta_stride1(int ptrstdx, int ptrsptr); /* pointsto.c */
void pstride_analysis(void);                   /* pstride.c */
void fini_pstride_analysis(void);              /* pstride.c */
void call_analyze(void);                       /* rest.c */
void convert_output(void);                     /* outconv.c */
void sectfloat(void);                          /* outconv.c */
void sectinline(void);                         /* outconv.c */
void linearize_arrays(void);                   /* outconv.c */
void hoist_stmt(int std, int fg, int l);       /* outconv.c */
void redundss(void);                           /* redundss.c */

/* ipa.c */
extern int IPA_Vestigial;
int IPA_isnoconflict(int sptr); /* main.c wrapper */
int IPA_noconflict(int sptr);   /* ipa.c */
void ipa_fini(void);            /* ipa.c */
void ipa_closefile(void);       /* ipa.c */
int IPA_arg_alloc(int funcsptr, int argnum);
int IPA_func_argalloc_safe(int sptr);
int IPA_func_globalalloc_safe(int sptr);
int IPA_safe(int sptr);
void ipa_restore_all(void);
void ipa_restore_back(void);
long IPA_sstride(int sptr); /* ipa.c */
long IPA_pstride(int sptr); /* ipa.c */
void ipa_mfilename(char *name);

/* ipasave.c */
void ipasave_fini(void);
void ipasave_closefile(void);
void ipasave(void);
int ipa_return_in_argument(int func);
int idsym(int cls, int id);
void ipasave_compname(char *, int, char **);
void ipasave_compsw(char *);
void ipasave_mfilename(char *);

/* dump.c */
void dstdp(int stdx);
void dumpdtypes(void);
void dastree(int astx);
void dsa(void);
void dast(int astx);
void past(int astx);
void dstda(void);
void dsym(int sptr);
void dstdpa(void);

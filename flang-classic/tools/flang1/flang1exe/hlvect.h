/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - include file for high-level vectorizer
 */

#ifndef NOVECTORIZE

#include <stdint.h>

/* maximum nesting level we consider */
#define MAX_LOOPS 14

typedef struct {
  unsigned depchk : 1;   /* [no]depchk */
  unsigned assoc : 1;    /* [no]assoc */
  unsigned eqvchk : 1;   /* [no]eqvchk */
  unsigned lstval : 1;   /* [no]lastval */
  unsigned recog : 1;    /* [no]recog */
  unsigned trans : 1;    /* [no]transform */
  unsigned safecall : 1; /* permit calls in loops */
  int shortloop;    /* a.k.a. smallvect */
  int mincnt;       /* flg.x[30] = min lp count for vectorization */
  int ldstsplit;    /* flg.x[40] = load/store threshhold for splitting */
  int opsplit;      /* flg.x[41] = op threshhold for splitting */
} VPRAGMAS;

/*
 * Vectorizer loop structure.
 */
typedef struct {
  int child;   /* first child of this loop (nested loop) */
  int sibling; /* next loop at this nesting level */
  int nest;    /* number of loops enclosing this loop */
  int mrstart; /* start of memory references in this loop */
  int mrcnt;   /* number of memory references in this loop */
  int mrecnt;  /* number of mem refs in this+enclosed loops */
  int istart;  /* start of BIVs for this loop */
  int icnt;    /* number of BIVs in this loop */
  int ubnd;    /* subs ref for loop upper bound  (temporary ) */
  int lbnd;    /* subs ref for loop lower bound  (temporary ) */
  int iubnd;   /* ILT for upper bound */
  int aubnd;   /* AST for upper bound */
  int ilbnd;   /* ILT for lower bound */
  int albnd;   /* AST for upper bound */
  int lpcnt;   /* ILI for loopcount */
  int sclist;  /* expandable scalars */
  int ealist;  /* expandable arrays */
  int prebih;  /* bih of preheader */
  int exitbih; /* bih of exit */
  int ivlist;  /* list of initial value mem refs (store) */
  union {
    uint16_t all;
    struct {
      uint16_t cand : 1;  /* candidate */
      uint16_t ztrip : 1; /* ztrip */
      uint16_t perf : 1;  /* perfectly nested */
      uint16_t del : 1;   /* deleted */
    } bits;
  } flags;
  VPRAGMAS pragmas;
} VLOOP;

#define VL_LOOP(i) hlv.lpbase[i].lp
#define VL_CHILD(i) hlv.lpbase[i].child
#define VL_SIBLING(i) hlv.lpbase[i].sibling
#define VL_NEST(i) hlv.lpbase[i].nest
#define VL_FLAGS(i) hlv.lpbase[i].flags.all
#define VL_MRSTART(i) hlv.lpbase[i].mrstart
#define VL_MRCNT(i) hlv.lpbase[i].mrcnt
#define VL_MRECNT(i) hlv.lpbase[i].mrecnt
#define VL_ISTART(i) hlv.lpbase[i].istart
#define VL_ICNT(i) hlv.lpbase[i].icnt
#define VL_UBND(i) hlv.lpbase[i].ubnd
#define VL_LBND(i) hlv.lpbase[i].lbnd
#define VL_IUBND(i) hlv.lpbase[i].iubnd
#define VL_AUBND(i) hlv.lpbase[i].aubnd
#define VL_ILBND(i) hlv.lpbase[i].ilbnd
#define VL_ALBND(i) hlv.lpbase[i].albnd
#define VL_LPCNT(i) hlv.lpbase[i].lpcnt
#define VL_SCLIST(i) hlv.lpbase[i].sclist
#define VL_EALIST(i) hlv.lpbase[i].ealist
#define VL_PREBIH(i) hlv.lpbase[i].prebih
#define VL_EXITBIH(i) hlv.lpbase[i].exitbih
#define VL_IVLIST(i) hlv.lpbase[i].ivlist
#define VL_PRAGMAS(i) hlv.lpbase[i].pragmas

#define VLP_DEPCHK(i) hlv.lpbase[i].pragmas.depchk
#define VLP_ASSOC(i) hlv.lpbase[i].pragmas.assoc
#define VLP_EQVCHK(i) hlv.lpbase[i].pragmas.eqvchk
#define VLP_LSTVAL(i) hlv.lpbase[i].pragmas.lstval
#define VLP_RECOG(i) hlv.lpbase[i].pragmas.recog
#define VLP_TRANS(i) hlv.lpbase[i].pragmas.trans
#define VLP_SAFECALL(i) hlv.lpbase[i].pragmas.safecall
#define VLP_SHORTLP(i) hlv.lpbase[i].pragmas.shortloop
#define VLP_MINCNT(i) hlv.lpbase[i].pragmas.mincnt
#define VLP_LDSTSPLIT(i) hlv.lpbase[i].pragmas.ldstsplit
#define VLP_OPSPLIT(i) hlv.lpbase[i].pragmas.opsplit

#define VP_DEPCHK(i) i.depchk
#define VP_ASSOC(i) i.assoc
#define VP_EQVCHK(i) i.eqvchk
#define VP_LSTVAL(i) i.lstval
#define VP_RECOG(i) i.recog
#define VP_TRANS(i) i.trans
#define VP_SAFECALL(i) i.safecall
#define VP_SHORTLP(i) i.shortloop
#define VP_MINCNT(i) i.mincnt
#define VP_LDSTSPLIT(i) i.ldstsplit
#define VP_OPSPLIT(i) i.opsplit

/* flags */
#define VL_CAND(i) hlv.lpbase[i].flags.bits.cand
#define VL_PERF(i) hlv.lpbase[i].flags.bits.perf
#define VL_DEL(i) hlv.lpbase[i].flags.bits.del
#define VL_ZTRIP(i) hlv.lpbase[i].flags.bits.ztrip

/*
 * A direction vector is a 32-bit integer.  It consists of up to
 * 7 4-bit fields, together with a 4-bit type field.
 * The 4-bit fields give the directions <, =, >, and reduction;
 * the type field gives the information as to whether the all-equals
 * direction is allowed.
 *
 * For a loop nest, the innermost loop corresponds to the rightmost bit
 * field.  This bit field is accessed with value 0.  And so forth.
 */

typedef BIGUINT64 DIRVEC;

#define DIRV_ENTSIZ 4
#define DIRV_ENTMSK (DIRVEC)15

/* get dirvec entry */
#define DIRV_ENTRYG(dirv, idx) (((dirv) >> ((idx)*DIRV_ENTSIZ)) & DIRV_ENTMSK)
/* put dirvec entry (assume clear) */
#define DIRV_ENTRYP(dirv, idx, dirent) \
  ((dirv) |= ((dirent & DIRV_ENTMSK) << ((idx)*DIRV_ENTSIZ)))
/* clear dirvec entry */
#define DIRV_ENTRYC(dirv, idx) ((dirv) &= ~(DIRV_ENTMSK << ((idx)*DIRV_ENTSIZ)))

#define DIRV_ALLEQ ((DIRVEC)0x8000000000000000L)
#define DIRV_MASK ((DIRVEC)0x0FFFFFFFFFFFFFFFL)

/*
 * Dirvec Bits
 */
#define DIRV_LT (DIRVEC)1L
#define DIRV_EQ (DIRVEC)2L
#define DIRV_GT (DIRVEC)4L
#define DIRV_RD (DIRVEC)8L
#define DIRV_NONE (DIRVEC)0L

#define DIRV_STAR (DIRV_LT | DIRV_EQ | DIRV_GT)

/*
 * Macros to get dirvec portion and information portion
 */
#define DIRV_DIRPART(dirv) ((dirv) & (DIRV_MASK))
#define DIRV_INFOPART(dirv) ((dirv) & ~(DIRV_MASK))

/* getitem area for HLV */
#define HLV_AREA 10

/* tsort successor area */
#define HLV_AREA1 11

/*
 * Data dependence successor
 */
typedef struct DDEDGE {
  int sink;
  union {
    uint16_t all;
    struct {
      uint16_t type : 3;
    } bits;
  } flags;
  DIRVEC dirvec;
  struct DDEDGE *next;
} DDEDGE;

#define DD_SINK(p) (p)->sink
#define DD_DIRVEC(p) (p)->dirvec
#define DD_FLAGS(p) (p)->flags.all
#define DD_NEXT(p) (p)->next

#define DIRV_FFLOW 0
#define DIRV_FANTI 1
#define DIRV_FOUT 2

#define DD_TYPE(i) (i)->flags.bits.type

/* a number bigger than max # of elements in a direction vector */
#define DIRV_BIGPOS (MAX_LOOPS + 1)

/*
 * Subscript information
 *
 * For a given subscript and loop nest, where the loop nest is numbered
 * from 1 to N outer to inner, the subscript has the form
 * base + stride[N-1] * CI1 + ... + stride[0] * CIN, where CIj is the
 * zero-based current iteration for loop j.
 *
 */
typedef struct {
  int bases[MAX_LOOPS + 1];
  int stride[MAX_LOOPS];
  union {
    uint16_t all;
    struct {
      uint16_t stop : 1;
      uint16_t ptr : 1;
      uint16_t ivuse : 1;
    } bits;
  } flags;
} SUBS;

#define SB_STOP(i) hlv.subbase[i].flags.bits.stop
#define SB_PTR(i) hlv.subbase[i].flags.bits.ptr
#define SB_IVUSE(i) hlv.subbase[i].flags.bits.ivuse
#define SB_BASE(i) hlv.subbase[i].bases[0]
#define SB_BASES(i) hlv.subbase[i].bases
#define SB_STRIDE(i) hlv.subbase[i].stride

/*
 * Memory references
 */
typedef struct {
  int fg;         /* flowgraph node containing this ilt */
  int ili;        /* ili for this mem ref */
  int ilt;        /* ilt containing the ili */
  int nme;        /* nme of ILI if load or store */
  int stmt;       /* statement containing this memory reference */
  int subst;      /* index into subscript info */
  int subcnt;     /* number of subscripts */
  int loop;       /* loop containing this memory reference */
  int rg;         /* region containing this memory reference */
  int fsubs;      /* folded subscript entry */
  int cseili;     /* CSE ili for this load, if exists */
  int next, prev; /* linked list pointers */
  int iv;         /* induction var if this is iuse */
  char type;      /* type of mem ref: */
                  /* 'l' = load, */
                  /* 's' = store, */
  char nest;      /* nesting depth of loop containing this mr */
  int rewr;       /* new ili replacing the ili field */
  union {
    uint16_t all;
    struct {
      /* first five are mutually exclusive */
      uint16_t ivuse : 1; /* use of induction variable */
      uint16_t array : 1; /* array reference */
      uint16_t indir : 1; /* indirection */
      uint16_t scalr : 1; /* scalar */
      uint16_t based : 1; /* based */
      /* */
      uint16_t inval : 1; /* invariant assignment or use */
      uint16_t exp : 1;   /* expandable scalar */
      uint16_t init : 1;  /* initial value assignment (store) */
      /* next three only apply to arrays */
      uint16_t induc : 1; /* induction in inner loop */
      uint16_t invar : 1; /* invar array elem in inner loop */
      uint16_t invec : 1; /* invariant vector in two nested loops */
      /* spare bits */
      uint16_t spare : 4;
    } bits;
  } flags;
  DDEDGE *succ; /* successor list */
} MEMREF;

#define MR_ILI(i) hlv.mrbase[i].ili
#define MR_ILT(i) hlv.mrbase[i].ilt
#define MR_FG(i) hlv.mrbase[i].fg
#define MR_NME(i) hlv.mrbase[i].nme
#define MR_STMT(i) hlv.mrbase[i].stmt
#define MR_TYPE(i) hlv.mrbase[i].type
#define MR_NEST(i) hlv.mrbase[i].nest
#define MR_FLAGS(i) hlv.mrbase[i].flags.all
#define MR_SUBST(i) hlv.mrbase[i].subst
#define MR_SUBCNT(i) hlv.mrbase[i].subcnt
#define MR_LOOP(i) hlv.mrbase[i].loop
#define MR_RG(i) hlv.mrbase[i].rg
#define MR_SUCC(i) hlv.mrbase[i].succ
#define MR_FSUBS(i) hlv.mrbase[i].fsubs
#define MR_CSEILI(i) hlv.mrbase[i].cseili
#define MR_NEXT(i) hlv.mrbase[i].next
#define MR_PREV(i) hlv.mrbase[i].prev
#define MR_IV(i) hlv.mrbase[i].iv
#define MR_SCLR(i) hlv.mrbase[i].iv
#define MR_EXPARR(i) hlv.mrbase[i].iv
#define MR_REWR(i) hlv.mrbase[i].rewr

#define MR_IVUSE(i) hlv.mrbase[i].flags.bits.ivuse
#define MR_ARRAY(i) hlv.mrbase[i].flags.bits.array
#define MR_INDIR(i) hlv.mrbase[i].flags.bits.indir
#define MR_SCALR(i) hlv.mrbase[i].flags.bits.scalr
#define MR_BASED(i) hlv.mrbase[i].flags.bits.based

#define MR_EXP(i) hlv.mrbase[i].flags.bits.exp
#define MR_INVAL(i) hlv.mrbase[i].flags.bits.inval
#define MR_INIT(i) hlv.mrbase[i].flags.bits.init

#define MR_INDUC(i) hlv.mrbase[i].flags.bits.induc
#define MR_INVAR(i) hlv.mrbase[i].flags.bits.invar
#define MR_INVEC(i) hlv.mrbase[i].flags.bits.invec

/*
 * Vectorizer induction information
 */
typedef struct {
  int nm;       /* names entry */
  int load;     /* ili of the load */
  int init;     /* initial value -- load or single def */
  int originit; /* original initial value */
  union {
    uint16_t all;
    struct {
      uint16_t ptr : 1;    /* induction variable is a pointer */
      uint16_t delete : 1; /* induction variable can be deleted */
      uint16_t niu : 1;    /* non-induction use */
      uint16_t midf : 1;   /* multiple inductions */
      uint16_t alias : 1;  /* induction alias */
      uint16_t noinv : 1;  /* Initial value is not invariant in loop */
      uint16_t omit : 1;   /* Omit this iv */
      uint16_t visit : 1;  /* Visited */
      uint16_t alast : 1;  /* Last value --> true if count-1 used */
      uint16_t save : 1;   /* must leave def in (perm) */
      uint16_t tsave : 1;  /* must leave def in (temp) */
    } bits;
  } flags;
  short opc;   /* ili opcode of skip expression */
  int totskip; /* total skip */
  int skip;    /* skip ili */
} VIND;

#define VIND_NM(i) hlv.indbase[i].nm
#define VIND_LOAD(i) hlv.indbase[i].load
#define VIND_INIT(i) hlv.indbase[i].init
#define VIND_ORIGINIT(i) hlv.indbase[i].originit
#define VIND_PTR(i) hlv.indbase[i].flags.bits.ptr
#define VIND_DELETE(i) hlv.indbase[i].flags.bits.delete
#define VIND_NIU(i) hlv.indbase[i].flags.bits.niu
#define VIND_MIDF(i) hlv.indbase[i].flags.bits.midf
#define VIND_ALIAS(i) hlv.indbase[i].flags.bits.alias
#define VIND_NOINV(i) hlv.indbase[i].flags.bits.noinv
#define VIND_OMIT(i) hlv.indbase[i].flags.bits.omit
#define VIND_VISIT(i) hlv.indbase[i].flags.bits.visit
#define VIND_ALAST(i) hlv.indbase[i].flags.bits.alast
#define VIND_SAVE(i) hlv.indbase[i].flags.bits.save
#define VIND_OPC(i) hlv.indbase[i].opc
#define VIND_TOTSKIP(i) hlv.indbase[i].totskip
#define VIND_SKIP(i) hlv.indbase[i].skip

/*
 * Statement list for distribution and interchange.
 */
typedef struct {
  int ilt;    /* ilt to which this statement refers */
  int mrlist; /* linked list of memory references */
  int next;   /* next statement */
  int prev;   /* previous statement */
  int flag;   /* useful word */
  union {
    uint16_t all;
    struct {
      uint16_t reduc : 1; /* statement is a reduction */
    } bits;
  } flags;
} STMT;

#define ST_ILT(i) hlv.stbase[i].ilt
#define ST_MRLIST(i) hlv.stbase[i].mrlist
#define ST_NEXT(i) hlv.stbase[i].next
#define ST_PREV(i) hlv.stbase[i].prev
#define ST_FLAG(i) hlv.stbase[i].flag
#define ST_REDUC(i) hlv.stbase[i].flags.bits.reduc

typedef struct VPSI {
  struct VPSI *next;
  int node;
  int flag;
} VPSI, *VPSI_P;

/*
 * Loop interchange and distribute is performed on loop regions.
 */
typedef struct {
  int next;         /* next region at this level */
  int prev;         /* previous region at this level */
  int outer;        /* parent region */
  int inner;        /* child region */
  int orig;         /* original loop for this region */
  int slist;        /* list of statements in this region */
  int nodenum;      /* node number */
  int cycle;        /* cycle number */
  int nest;         /* nesting depth */
  int orignest;     /* original nesting depth */
  int top, ztrip;   /* labels */
  int cnt;          /* loop count temp */
  int llvi;         /* low-level vectorizer info */
  int bih;          /* bih */
  int bihtl;        /* tail bih */
  int bihafter;     /* bih of block after tail */
  int miv;          /* master induction variable */
  int miv_store;    /* ili of initial store into miv */
  int cnt_load;     /* ili of load into loop count */
  int slbnd, subnd; /* lower & upper bound statement */
  int ilbnd, iubnd; /* lower & upper bound subscripts */
  int mreg;         /* marked region # */
  int label;        /* label from orig bih */
  int blockrg;      /* the block loop for a strip loop */
  union {
    uint16_t all;
    struct {
      uint16_t cif : 1;        /* contains if statement */
      uint16_t marked : 1;     /* general marker */
      uint16_t recurrence : 1; /* recurrence region */
      uint16_t strip : 1;      /* must be strip-mined */
      uint16_t self : 1;       /* in a self-cycle */
      uint16_t block : 1;      /* block-loop of stripmine */
      uint16_t parallel : 1;   /* parallelizable */
      uint16_t call : 1;       /* contains call */
    } bits;
  } flags;
  VPSI_P succ; /* list of successors of this region */
} REGION;

#define RG_NEXT(i) hlv.rgbase[i].next
#define RG_PREV(i) hlv.rgbase[i].prev
#define RG_OUTER(i) hlv.rgbase[i].outer
#define RG_INNER(i) hlv.rgbase[i].inner
#define RG_ORIG(i) hlv.rgbase[i].orig
#define RG_SLIST(i) hlv.rgbase[i].slist
#define RG_NODENUM(i) hlv.rgbase[i].nodenum
#define RG_STRIPLEN(i) hlv.rgbase[i].nodenum
#define RG_CYCLE(i) hlv.rgbase[i].cycle
#define RG_NEST(i) hlv.rgbase[i].nest
#define RG_ORIGNEST(i) hlv.rgbase[i].orignest
#define RG_TOP(i) hlv.rgbase[i].top
#define RG_ZTRIP(i) hlv.rgbase[i].ztrip
#define RG_CNT(i) hlv.rgbase[i].cnt
#define RG_SUCC(i) hlv.rgbase[i].succ
#define RG_FLAGS(i) hlv.rgbase[i].flags.all
#define RG_BIH(i) hlv.rgbase[i].bih
#define RG_BIHHD(i) hlv.rgbase[i].bih
#define RG_BIHTL(i) hlv.rgbase[i].bihtl
#define RG_BIHAFTER(i) hlv.rgbase[i].bihafter
#define RG_LLVI(i) hlv.rgbase[i].llvi
#define RG_SLBND(i) hlv.rgbase[i].slbnd
#define RG_SUBND(i) hlv.rgbase[i].subnd
#define RG_LBND(i) hlv.rgbase[i].ilbnd
#define RG_UBND(i) hlv.rgbase[i].iubnd
#define RG_MIV(i) hlv.rgbase[i].miv
#define RG_MREG(i) hlv.rgbase[i].mreg
#define RG_MIVSTORE(i) hlv.rgbase[i].miv_store
#define RG_CNTLOAD(i) hlv.rgbase[i].cnt_load
#define RG_CIF(i) hlv.rgbase[i].flags.bits.cif
#define RG_MARKED(i) hlv.rgbase[i].flags.bits.marked
#define RG_RECURRENCE(i) hlv.rgbase[i].flags.bits.recurrence
#define RG_SELF(i) hlv.rgbase[i].flags.bits.self
#define RG_STRIP(i) hlv.rgbase[i].flags.bits.strip
#define RG_BLOCK(i) hlv.rgbase[i].flags.bits.block
#define RG_PARALLEL(i) hlv.rgbase[i].flags.bits.parallel
#define RG_CALL(i) hlv.rgbase[i].flags.bits.call
#define RG_LABEL(i) hlv.rgbase[i].label
#define RG_BLOCKRG(i) hlv.rgbase[i].blockrg

/*
 * Temp management
 */
typedef struct {
  int iavl;        /* available pointer */
  int iavl_max;    /* high-water for this function */
  int iavl_base;   /* base for this function */
  const char *pfx; /* prefix for this temp */
  int dtype;       /* data type for this temp */
} VTMP;

#define VT_INT 0 /* integer temp */
#define VT_PTR 1 /* pointer temp */
#define VT_SP 2  /* single precision */
#define VT_DP 3  /* double precision */
#define VT_MAX 4

#define VT_PHINIT 0 /* initial value phase */
#define VT_PHIND 1  /* induction phase */
#define VT_PHASES 2 /* number of phases of temps */

extern VTMP hlv_temps[VT_PHASES][VT_MAX];
/* array temps */
extern VTMP hlv_vtemps;

/*
 * Expandable scalars
 */
typedef struct {
  int nme;  /* nme of expandable scalar */
  int next; /* next scalar in list */
  union {
    uint16_t all;
    struct {
      uint16_t flag : 1; /* needs last value */
      uint16_t span : 1; /* spans loops */
    } bits;
  } flags;
  int arrsym;  /* symbol for array if this spans loops */
  int masksym; /* symbol for mask array if conditional assignment */
} SCALAR;

#define SCLR_NME(i) hlv.scbase[i].nme
#define SCLR_NEXT(i) hlv.scbase[i].next
#define SCLR_FLAG(i) hlv.scbase[i].flags.bits.flag
#define SCLR_SPAN(i) hlv.scbase[i].flags.bits.span
#define SCLR_ARRSYM(i) hlv.scbase[i].arrsym
#define SCLR_MASKSYM(i) hlv.scbase[i].masksym

/*
 * Loop masks: ith bit (0 <= i < MAX_LOOPS) reflects information
 *	about the ith outer loop.
 */
typedef UINT16 LOOPMASK;

#define LOOPMASKG(m, i) ((m >> i) & 1)
#define LOOPMASKC(m, i) (m &= ~(1 << i))
#define LOOPMASKS(m, i) (m |= (1 << i))

/*
 * Expandable arrays
 */
typedef struct {
  int nest;          /* length of invloops mask */
  LOOPMASK invloops; /* ith bit is 1 if invariant on ith outer loop
                      * at stddef */
  int tmpsym;        /* symbol for replacement array */
  int next;          /* next expandable array in loop */
  union {
    uint16_t all;
    struct {
      uint16_t lastval : 1;    /* needs last value */
      uint16_t uniquesecn : 1; /* all array refs have a unique section
                                *  descriptor */
    } bits;
  } flags;
} EXPARR;

#define EXPARR_NEST(e) hlv.eabase[e].nest
#define EXPARR_INVLOOPS(e) hlv.eabase[e].invloops
#define EXPARR_TMPSYM(e) hlv.eabase[e].tmpsym
#define EXPARR_NEXT(e) hlv.eabase[e].next
#define EXPARR_LASTVAL(e) hlv.eabase[e].flags.bits.lastval
#define EXPARR_UNIQUESECN(e) hlv.eabase[e].flags.bits.uniquesecn

typedef struct {
  VLOOP *lpbase;
  int lpavail, lpsize;
  int looplist;
  MEMREF *mrbase;
  int mravail, mrsize;
  VIND *indbase;
  int indavail, indsize;
  SUBS *subbase;
  int subavail, subsize;
  int stavail, stsize;
  STMT *stbase;
  int rgavail, rgsize;
  REGION *rgbase;
  int scavail, scsize;
  SCALAR *scbase;
  int eaavail, easize;
  EXPARR *eabase;
  int *fgmap;
  int fgn;
  int natural_loop;
  double (*score_loop)(int *);
} HLVECT;

extern HLVECT hlv;

extern void vect_flg_defaults(void);
extern void vectorize(void);
extern void mark_vinduc(int loop);
extern void unmark_vinduc(int loop);
extern int hlv_getsym(int phase, int type);
extern int hlv_getvsym(int baseDtype);
extern void vdel_bih(int bihx);
extern void open_vpragma(int lp);
extern LOGICAL is_parent_loop(int lpParent, int lpChild);

#if DEBUG
extern void dump_memref_hdr(void);
extern void dump_one_memref(int i);
extern void dump_memrefs(int start, int cnt);
extern void dump_memrefs(int start, int cnt);
extern void dump_vloops(int first, int base);
extern void dump_vinduc(int start, int cnt);
#endif
extern void build_dd(int loop);
extern void dd_edge(int src, int sink, DIRVEC vec);
extern DIRVEC dirv_inverse(DIRVEC vec);
extern DIRVEC dirv_fulldep(int level);
extern DIRVEC dirv_exo(int level, int flag);
extern int dirv_chkzero(DIRVEC dir, int n);
extern void dirv_gen(DIRVEC dir, int *map, int nest, int ig, void (*f)(DIRVEC));
extern LOGICAL dd_array_conflict(int astliTriples, int astArrSrc,
                                 int astArrSink, int bSinkAfterSrc);
extern int dd_symbolic(int il);
#if DEBUG
extern char *dirv_print(DIRVEC dir);
extern void dump_dd(DDEDGE *p);
#endif
extern void hlvtrans(int loop);
extern void regmark(int rg, int rgm, int flag);
extern int remove_accum(int astx, int subast, int optype);
extern int is_perfect(int rg);
extern int get_generic_minmax(int optype);
extern double score_loop_i860(int *a);
extern double score_loop_sparc(int *a);
extern double score_loop_vpu(int *a);
extern double score_loop_hpf(int *a);
extern LOGICAL has_reduction(int rg, int *map, int nest);
#if DEBUG
extern void dump_vregionf(int r, int level, LOGICAL flag);
#endif
extern void vpar(void);
extern void llvect(void);
extern void gen_llinfo(int reg);
extern void reg_parmark(int rg);
extern LOGICAL has_recurrence(int rg, int nest, int *mapx, int *mapy);
extern int parmethod(int rg, int nest, int *mapx, int *mapy);
#if DEBUG
extern void dump1_llv(int k);
extern void dump_llv(void);
extern void dump_one_vloop(int i, int level);
#endif
#endif
#define STRIPSIZE 1024

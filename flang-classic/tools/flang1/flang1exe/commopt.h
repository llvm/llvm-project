/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file commopt.h
    \brief macros, definitions, and prototypes for communications module
*/

#define MAXFUSE 10
#define MAXOMEM 40
typedef struct {
  int lhs;
  int ifast;
  int endifast;
  LITEMF *inner_cyclic;
  int c_lof[7];
  int idx[7];
  int cb_init[7]; /* cyclic_block initialization asts */
  int cb_do[7];
  int cb_block[7];
  int cb_inc[7];
  int cb_enddo[7];

  int c_init[7]; /* cyclic initialization asts */
  int c_inc[7];
  int c_dupl[7];
  int c_idx[7];
  int c_dstt[7];
} CTYPE;

typedef union {
  struct {
    int same;
    int reuse;
    int nomem;
    int omem[MAXOMEM];
    int omemed;
  } getsclr;
  struct {
    int idx;
    int lhs;
    int same;
    int call;
  } bnd;
  struct {
    int alloc;
    int lhs;
    int rhs;
    int sptr;
    int sectl;
    int sectr;
    int uselhs;
    int free;
    int same;
    int out;
    int reuse;
    int lhssec;
    int notlhs;
  } ccopy;
  struct {
    int alloc;
    int lhs;
    int rhs;
    int vsub;
    int nvsub;
    int mask;
    int sectvsub;
    int sectnvsub;
    int sectm;
    int sectv[7];
    int v[7];
    int permute[7];
    int vflag;
    int pflag;
    int vdim;
    int pdim;
    int nvec;
    int nper;
    int type;
    int uselhs;
    int free;
    int same;
    int out;
    int reuse;
    int indexreuse;
    int lhssec;
    int notlhs;
  } cgather;
  struct {
    int rhs;
    int free;
    int same;
    int out;
    int reuse;
    int type;
    int boundary;
  } shift;
  struct {
    int comm;
    int lhs;
    int rhs;
    int sectl;
    int sectr;
    int free;
    int same;
    int out;
    int alloc;
    int ref;
    int reuse;
    int invmvd; /* invariant moved */
    int type;
    int uselhs;
    int usedstd;
  } cstart;
  struct {
    int arr;
    int sptr;
    int alloc;
    int free;
    int flag;
    int same;
    int out;
    int reuse;
  } sect;
  struct {
    int sptr;
    int free;
    int same;
    int out;
    int reuse;
    int used;
    int ptasgn;
  } alloc;
  struct {
    int sptr;
    int ncall;
    LITEMF *call;
    int pos;
  } call;
  struct {
    int sectl;
    int nrt;
    LITEMF *rtl;
    int nmcall;
    LITEMF *mcall;
    int nscall;
    LITEMF *scall;
    int npcall;
    LITEMF *pcall;
    int nmget;
    LITEMF *mget;
    int nsget;
    LITEMF *sget;
    CTYPE *cyclic;
    int fuselp[7][MAXFUSE];
    int fusedstd[7][MAXFUSE];
    int nfuse[7];
    int header;
    int barr1;
    int barr2;
    int fg; /* fg node */
    unsigned ignore : 1;
    unsigned fused : 1;
  } forall;
} FTABLE;

typedef struct {
  int fall;
  int std;
  FTABLE f;
} FT;

typedef struct {
  FT *base;
  int size;
  int avl;
} FTB;

extern FTB ftb;

#define FT_STD(i) ftb.base[i].std
#define FT_FORALL(i) ftb.base[i].fall

#define FT_GETSCLR_SAME(i) ftb.base[i].f.getsclr.same
#define FT_GETSCLR_REUSE(i) ftb.base[i].f.getsclr.reuse
#define FT_GETSCLR_NOMEM(i) ftb.base[i].f.getsclr.nomem
#define FT_GETSCLR_OMEM(i, j) ftb.base[i].f.getsclr.omem[j]
#define FT_GETSCLR_OMEMED(i) ftb.base[i].f.getsclr.omemed

#define FT_BND_LHS(i) ftb.base[i].f.bnd.lhs
#define FT_BND_IDX(i) ftb.base[i].f.bnd.idx
#define FT_BND_SAME(i) ftb.base[i].f.bnd.same
#define FT_BND_CALL(i) ftb.base[i].f.bnd.call

#define FT_ALLOC_SPTR(i) ftb.base[i].f.alloc.sptr
#define FT_ALLOC_FREE(i) ftb.base[i].f.alloc.free
#define FT_ALLOC_SAME(i) ftb.base[i].f.alloc.same
#define FT_ALLOC_OUT(i) ftb.base[i].f.alloc.out
#define FT_ALLOC_REUSE(i) ftb.base[i].f.alloc.reuse
#define FT_ALLOC_USED(i) ftb.base[i].f.alloc.used
#define FT_ALLOC_PTASGN(i) ftb.base[i].f.alloc.ptasgn

#define FT_SECT_ARR(i) ftb.base[i].f.sect.arr
#define FT_SECT_SPTR(i) ftb.base[i].f.sect.sptr
#define FT_SECT_ALLOC(i) ftb.base[i].f.sect.alloc
#define FT_SECT_FREE(i) ftb.base[i].f.sect.free
#define FT_SECT_FLAG(i) ftb.base[i].f.sect.flag
#define FT_SECT_SAME(i) ftb.base[i].f.sect.same
#define FT_SECT_OUT(i) ftb.base[i].f.sect.out
#define FT_SECT_REUSE(i) ftb.base[i].f.sect.reuse

#define FT_CCOPY_ALLOC(i) ftb.base[i].f.ccopy.alloc
#define FT_CCOPY_LHS(i) ftb.base[i].f.ccopy.lhs
#define FT_CCOPY_RHS(i) ftb.base[i].f.ccopy.rhs
#define FT_CCOPY_TSPTR(i) ftb.base[i].f.ccopy.sptr
#define FT_CCOPY_SECTL(i) ftb.base[i].f.ccopy.sectl
#define FT_CCOPY_SECTR(i) ftb.base[i].f.ccopy.sectr
#define FT_CCOPY_USELHS(i) ftb.base[i].f.ccopy.uselhs
#define FT_CCOPY_FREE(i) ftb.base[i].f.ccopy.free
#define FT_CCOPY_SAME(i) ftb.base[i].f.ccopy.same
#define FT_CCOPY_OUT(i) ftb.base[i].f.ccopy.out
#define FT_CCOPY_REUSE(i) ftb.base[i].f.ccopy.reuse
#define FT_CCOPY_LHSSEC(i) ftb.base[i].f.ccopy.lhssec
#define FT_CCOPY_NOTLHS(i) ftb.base[i].f.ccopy.notlhs

#define FT_CGATHER_ALLOC(i) ftb.base[i].f.cgather.alloc
#define FT_CGATHER_LHS(i) ftb.base[i].f.cgather.lhs
#define FT_CGATHER_RHS(i) ftb.base[i].f.cgather.rhs
#define FT_CGATHER_VSUB(i) ftb.base[i].f.cgather.vsub
#define FT_CGATHER_NVSUB(i) ftb.base[i].f.cgather.nvsub
#define FT_CGATHER_MASK(i) ftb.base[i].f.cgather.mask
#define FT_CGATHER_SECTVSUB(i) ftb.base[i].f.cgather.sectvsub
#define FT_CGATHER_SECTNVSUB(i) ftb.base[i].f.cgather.sectnvsub
#define FT_CGATHER_SECTM(i) ftb.base[i].f.cgather.sectm
#define FT_CGATHER_SECTV(i, j) ftb.base[i].f.cgather.sectv[j]
#define FT_CGATHER_V(i, j) ftb.base[i].f.cgather.v[j]
#define FT_CGATHER_PERMUTE(i, j) ftb.base[i].f.cgather.permute[j]
#define FT_CGATHER_VFLAG(i) ftb.base[i].f.cgather.vflag
#define FT_CGATHER_PFLAG(i) ftb.base[i].f.cgather.pflag
#define FT_CGATHER_VDIM(i) ftb.base[i].f.cgather.vdim
#define FT_CGATHER_PDIM(i) ftb.base[i].f.cgather.pdim
#define FT_CGATHER_NVEC(i) ftb.base[i].f.cgather.nvec
#define FT_CGATHER_NPER(i) ftb.base[i].f.cgather.nper
#define FT_CGATHER_TYPE(i) ftb.base[i].f.cgather.type
#define FT_CGATHER_USELHS(i) ftb.base[i].f.cgather.uselhs
#define FT_CGATHER_FREE(i) ftb.base[i].f.cgather.free
#define FT_CGATHER_SAME(i) ftb.base[i].f.cgather.same
#define FT_CGATHER_OUT(i) ftb.base[i].f.cgather.out
#define FT_CGATHER_REUSE(i) ftb.base[i].f.cgather.reuse
#define FT_CGATHER_INDEXREUSE(i) ftb.base[i].f.cgather.indexreuse
#define FT_CGATHER_LHSSEC(i) ftb.base[i].f.cgather.lhssec
#define FT_CGATHER_NOTLHS(i) ftb.base[i].f.cgather.notlhs

#define FT_SHIFT_RHS(i) ftb.base[i].f.shift.rhs
#define FT_SHIFT_FREE(i) ftb.base[i].f.shift.free
#define FT_SHIFT_SAME(i) ftb.base[i].f.shift.same
#define FT_SHIFT_OUT(i) ftb.base[i].f.shift.out
#define FT_SHIFT_REUSE(i) ftb.base[i].f.shift.reuse
#define FT_SHIFT_TYPE(i) ftb.base[i].f.shift.type
#define FT_SHIFT_BOUNDARY(i) ftb.base[i].f.shift.boundary

#define FT_CSTART_COMM(i) ftb.base[i].f.cstart.comm
#define FT_CSTART_LHS(i) ftb.base[i].f.cstart.lhs
#define FT_CSTART_RHS(i) ftb.base[i].f.cstart.rhs
#define FT_CSTART_SECTL(i) ftb.base[i].f.cstart.sectl
#define FT_CSTART_SECTR(i) ftb.base[i].f.cstart.sectr
#define FT_CSTART_FREE(i) ftb.base[i].f.cstart.free
#define FT_CSTART_SAME(i) ftb.base[i].f.cstart.same
#define FT_CSTART_OUT(i) ftb.base[i].f.cstart.out
#define FT_CSTART_ALLOC(i) ftb.base[i].f.cstart.alloc
#define FT_CSTART_REF(i) ftb.base[i].f.cstart.ref
#define FT_CSTART_REUSE(i) ftb.base[i].f.cstart.reuse
#define FT_CSTART_INVMVD(i) ftb.base[i].f.cstart.invmvd
#define FT_CSTART_TYPE(i) ftb.base[i].f.cstart.type
#define FT_CSTART_USELHS(i) ftb.base[i].f.cstart.uselhs
#define FT_CSTART_USEDSTD(i) ftb.base[i].f.cstart.usedstd

#define FT_CALL_SPTR(i) ftb.base[i].f.call.sptr
#define FT_CALL_NCALL(i) ftb.base[i].f.call.ncall
#define FT_CALL_CALL(i) ftb.base[i].f.call.call
#define FT_CALL_POS(i) ftb.base[i].f.call.pos

#define FT_NRT(i) ftb.base[i].f.forall.nrt
#define FT_RTL(i) ftb.base[i].f.forall.rtl
#define FT_NMCALL(i) ftb.base[i].f.forall.nmcall
#define FT_MCALL(i) ftb.base[i].f.forall.mcall
#define FT_NSCALL(i) ftb.base[i].f.forall.nscall
#define FT_SCALL(i) ftb.base[i].f.forall.scall
#define FT_NPCALL(i) ftb.base[i].f.forall.npcall
#define FT_PCALL(i) ftb.base[i].f.forall.pcall
#define FT_NMGET(i) ftb.base[i].f.forall.nmget
#define FT_MGET(i) ftb.base[i].f.forall.mget
#define FT_NSGET(i) ftb.base[i].f.forall.nsget
#define FT_SGET(i) ftb.base[i].f.forall.sget
#define FT_IGNORE(i) ftb.base[i].f.forall.ignore
#define FT_CYCLIC(i) ftb.base[i].f.forall.cyclic
#define FT_SECTL(i) ftb.base[i].f.forall.sectl
#define FT_NFUSE(i, j) ftb.base[i].f.forall.nfuse[j]
#define FT_FUSELP(i, j, k) ftb.base[i].f.forall.fuselp[(j)][(k)]
#define FT_FUSEDSTD(i, j, k) ftb.base[i].f.forall.fusedstd[(j)][(k)]
#define FT_FUSED(i) ftb.base[i].f.forall.fused
#define FT_HEADER(i) ftb.base[i].f.forall.header
#define FT_BARR1(i) ftb.base[i].f.forall.barr1
#define FT_BARR2(i) ftb.base[i].f.forall.barr2
#define FT_FG(i) ftb.base[i].f.forall.fg

typedef struct {
  int fuse;
  int bnd;
  int alloc;
  int sect;
  int copysection;
  int gatherx;
  int scatterx;
  int shift;
  int start;
} OPTSUM;

extern OPTSUM optsum;

#define BOGUSFLAG 0x100
#define NOTSECTFLAG 0 /* remove this flag, 0x200 */
#define NOREINDEX 0x2000000

extern void add_loop_hd(int);
extern LOGICAL same_forall_size(int, int, int);
extern void comm_analyze(void); /* comm.c */
extern void comm_optimize_post(void);
extern void comm_invar(void);     /* comminvar.c */
extern void comm_generator(void); /* commgen.c */
extern void comm_optimize_pre(void);

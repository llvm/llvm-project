/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef FE_COMM_H
#define FE_COMM_H

#include "gbldefs.h"
#include "transfrm.h"
#include <stdio.h>

/*
 * Data structure to record information about subscripts
 */

typedef struct {
  int idx;        /* ASTLI for forall index triplet or 0 */
  int stride;     /* multiplier of idx */
  int base;       /* constant */
  int sub;        /* ast for the subscript */
  int lhs_dim;    /* lhs dimension aligned with the same templ dim */
  int newl;       /* new lower bound for forall */
  int newu;       /* new upper bound for forall */
  int news;       /* new stride for forall */
  int dst_type;   /* distribution type */
  int comm_type;  /* communications type */
  int comm_value; /* generic value for comms */
  int diff;       /* f2(sub2)-f1(sub1) alignment function difference */
  int cnst;       /* to show difference has no index value */
  int lof;        /* cyclic distribution, a(i-lof) */
  int dupl;       /* duplicate, idx appear more than one, a(i,i) */
  int nop;        /* negative overlap*/
  int pop;        /* positive overlap*/
} SUBINFO;

#define COMMT_NOTAG -1    /* no tag attached yet */
#define COMMT_NONE 0      /* no communication */
#define COMMT_MULTI 1     /* multicast */
#define COMMT_SHIFTC 2    /* constant shift */
#define COMMT_SHIFTV 3    /* variable shift */
#define COMMT_TRANSFER 4  /* transfer */
#define COMMT_REPLICATE 5 /* replication */
#define COMMT_CONST 6     /* index */
#define COMMT_UNSTRUCT 7  /* unstructured communication */

#define SUBI_IDX(i) trans.subb.stg_base[i].idx
#define SUBI_STRIDE(i) trans.subb.stg_base[i].stride
#define SUBI_BASE(i) trans.subb.stg_base[i].base
#define SUBI_SUB(i) trans.subb.stg_base[i].sub
#define SUBI_DSTT(i) trans.subb.stg_base[i].dst_type
#define SUBI_LDIM(i) trans.subb.stg_base[i].lhs_dim
#define SUBI_NEWL(i) trans.subb.stg_base[i].newl
#define SUBI_NEWU(i) trans.subb.stg_base[i].newu
#define SUBI_NEWS(i) trans.subb.stg_base[i].news
#define SUBI_DSTT(i) trans.subb.stg_base[i].dst_type
#define SUBI_COMMT(i) trans.subb.stg_base[i].comm_type
#define SUBI_COMMV(i) trans.subb.stg_base[i].comm_value
#define SUBI_CNST(i) trans.subb.stg_base[i].cnst
#define SUBI_DIFF(i) trans.subb.stg_base[i].diff
#define SUBI_LOF(i) trans.subb.stg_base[i].lof
#define SUBI_DUPL(i) trans.subb.stg_base[i].dupl
#define SUBI_NOP(i) trans.subb.stg_base[i].nop
#define SUBI_POP(i) trans.subb.stg_base[i].pop

typedef struct {
  int result;
  int base;
  int operator;
  int function;
  int mask;
  int array;
  int array_simple;
} SCATTER_TYPE;

struct comminfo {
  int std;
  int subinfo;
  int lhs;
  int sub;
  int forall;
  int asn;
  int unstruct;
  int mask_phase;
  int ugly_mask;
  SCATTER_TYPE scat;
  int usedstd;
};

extern struct comminfo comminfo;

#define NO_CLASS 0
#define NO_COMM 1
#define OVERLAP 2
#define COLLECTIVE 3
#define COPY_SECTION 4
#define GATHER 5
#define SCATTER 6
#define IRREGULAR 7

typedef struct {
  int class;  /* communication class */
  int flag;   /* flag for overlap shift */
  int temp;   /* pointer to the temp for this ref */
  int arrsym; /* symbol pointer for the array */
  int arr;    /* array ast */
  int sub;    /* first subscript */
  int ndim;   /* number of dimensions */
  int next;   /* next subscript */
} ARREF;

#define ARREF_CLASS(i) trans.arrb.stg_base[i].class
#define ARREF_FLAG(i) trans.arrb.stg_base[i].flag
#define ARREF_TEMP(i) trans.arrb.stg_base[i].temp
#define ARREF_ARRSYM(i) trans.arrb.stg_base[i].arrsym
#define ARREF_ARR(i) trans.arrb.stg_base[i].arr
#define ARREF_SUB(i) trans.arrb.stg_base[i].sub
#define ARREF_NDIM(i) trans.arrb.stg_base[i].ndim
#define ARREF_NEXT(i) trans.arrb.stg_base[i].next

typedef struct {
  int lhs; /* left hand side array ref */
  int rhs; /* list of rhs array refs */
} TDESC;

#define TD_LHS(i) trans.tdescb.stg_base[i].lhs
#define TD_RHS(i) trans.tdescb.stg_base[i].rhs

typedef struct {
  struct {
    SUBINFO *stg_base;
    int stg_size;
    int stg_avail;
  } subb;
  struct {
    ARREF *stg_base;
    int stg_size;
    int stg_avail;
  } arrb;
  struct {
    TDESC *stg_base;
    int stg_size;
    int stg_avail;
  } tdescb;
  int iardt;           /* array of integer datatype */
  int first;           /* first statement available */
  int dtmp;            /* temp array/pointer to access data structs */
  TLIST *ar_type_list; /* list of array types */
  FILE *ctrfile;       /* constructor file */
  int cmnblksym;       /* common block name */
  int initsym;         /* init subroutine name */
  int rhsbase;         /* first array ref in RHS */
  int lhs;             /* lhs array ref */
  int darray;          /* comm descriptor array (temporary) */
} TRANSFORM;

extern TRANSFORM trans;

#define FORALL_PFX "i_"
#define TMPL_PFX "tmpl_"
#define PROC_PFX "proc_"
#define INIT_PFX "init_"
#define COMMON_PFX "cmn_"

/* storage allocation macros  */

#define TRANS_ALLOC(stgb, dt, sz) \
  {                               \
    NEW(stgb.stg_base, dt, sz);   \
    stgb.stg_size = sz;           \
    stgb.stg_avail = 1;           \
  }

#define TRANS_MORE(stb, dt, nsz)                                            \
  {                                                                         \
    stb.stg_base =                                                          \
        (dt *)sccrelal((char *)stb.stg_base, ((BIGUINT64)((nsz) * sizeof(dt)))); \
    stb.stg_size = nsz;                                                     \
    if (stb.stg_base == NULL)                                               \
      error(7, 4, 0, CNULL, CNULL);                                         \
  }

#define TRANS_NEED(stb, dt, nsz)    \
  if (stb.stg_avail > stb.stg_size) \
    TRANS_MORE(stb, dt, nsz);

#define TRANS_FREE(stb) FREE(stb.stg_base)

LOGICAL is_same_number_of_idx(int dest, int src, int list);
LOGICAL normalize_bounds(int sptr);
int add_lbnd(int dtyp, int dim, int ast, int astmember);
int insert_endmask(int ast, int stdstart);
int insert_mask(int ast, int stdstart);
int record_barrier(LOGICAL, int, int);
int sub_lbnd(int dtyp, int dim, int ast, int astmember);
void check_region(int std);
void comm_analyze(void);
void comm_fini(void);
void init_region(void);
void report_comm(int std, int cause);
int reference_for_temp(int sptr, int a, int forall);
int emit_get_scalar(int a, int std);
void forall_opt1(int ast);
void transform_forall(int std, int ast);
void scalarize(int std, int forall, LOGICAL after_transformer);
void un_fuse(int forall);
void sequentialize(int std, int forall, LOGICAL after_transformer);

#endif /* FE_COMM_H */

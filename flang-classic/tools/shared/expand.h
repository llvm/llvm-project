/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef EXPAND_H_
#define EXPAND_H_

/** \file
 * \brief various definitions for the expand module
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ilmtp.h"
#include <stdint.h>

/*  DEBUG-controlled -q stuff  */

#define EXPDBG(x, y) (DEBUG && DBGBIT(x, y))

/* storage allocation macros  */

#define EXP_ALLOC(stgb, dt, sz) \
  {                             \
    NEW(stgb.stg_base, dt, sz); \
    stgb.stg_size = sz;         \
  }

#define EXP_MORE(stb, dt, nsz)                                              \
  {                                                                         \
    stb.stg_base =                                                          \
        (dt *)sccrelal((char *)stb.stg_base, ((BIGUINT64)((nsz) * sizeof(dt)))); \
    stb.stg_size = nsz;                                                     \
  }

#define EXP_NEED(stb, dt, nsz)      \
  if (stb.stg_avail > stb.stg_size) \
    EXP_MORE(stb, dt, nsz);

#define EXP_FREE(stb) FREE(stb.stg_base)

#ifndef FE90

/*****  expander's view of the ILMs  *****/

typedef struct ILM {
  ILM_T opc;
  ILM_T opnd[1];
} ILM;

#define ILM_OPND(i, opn) ((i)->opnd[opn - 1])

#ifdef __cplusplus
/* clang-format off */
inline ILM_OP ILM_OPC(const ILM *ilm) {
  return static_cast<ILM_OP>(ilm->opc);
}

inline void SetILM_OPC(ILM *ilm, ILM_OP opc) {
  ilm->opc = opc;
}

inline SPTR ILM_SymOPND(const ILM *ilm, int opn) {
  return static_cast<SPTR>(ILM_OPND(ilm, opn));
}

inline DTYPE ILM_DTyOPND(const ILM *ilm, int opn) {
  return static_cast<DTYPE>(ILM_OPND(ilm, opn));
}
/* clang-format on */
#else
#define ILM_OPC(i) ((i)->opc)
#define SetILM_OPC(i,j)  ((i)->opc = (j))
#define ILM_SymOPND ILM_OPND
#define ILM_DTyOPND ILM_OPND
#endif

/*
 * ILM Auxillary Area Declarations - Used to accumulate information about
 * ILMs while being expanded. There is an item for each ILM.  Each item in
 * this area is indexed by the ILM index; since the ILM index is just an
 * offset from the beginning of the ILM area, there will be items in the aux
 * area that are not used.
 */
typedef struct {
  int w1;
  int w2;
  int w3;
  int w4;
  int w5;
  int w6;
  int w7;
  int w8;
} ILM_AUX;

#define ILM_TEMP(i) (expb.temps[i])

#define ILI_OF(i) (expb.ilmb.stg_base[i].w1)
#define NME_OF(i) (expb.ilmb.stg_base[i].w2)
#define SCALE_OF(i) (expb.ilmb.stg_base[i].w4)

#define ILM_RESULT(i) (expb.ilmb.stg_base[i].w1)
#define ILM_NME(i) (expb.ilmb.stg_base[i].w2)
#define ILM_BLOCK(i) (expb.ilmb.stg_base[i].w3)
#define ILM_SCALE(i) (expb.ilmb.stg_base[i].w4)

#define ILM_RRESULT(i) ILM_RESULT(i)
#define ILM_IRESULT(i) (expb.ilmb.stg_base[i].w7)

/* RESTYPE is used to indicate result type */
#define ILM_RESTYPE(i) (expb.ilmb.stg_base[i].w6)
#define ILM_ISCMPLX 1
#define ILM_ISDCMPLX 2
#define ILM_ISCHAR 3
#define ILM_ISI8 4
#define ILM_ISX87CMPLX 5
#define ILM_ISDOUBLEDOUBLECMPLX 6
#define ILM_ISFLOAT128CMPLX 7
#define ILM_ISQCMPLX 8

/* character stuff */
#define ILM_MXLEN(i) (expb.ilmb.stg_base[i].w5)
#define ILM_CLEN(i) (expb.ilmb.stg_base[i].w7)

/* this is used to tell whether an operand was
 * directly expanded for this parent ILM, or some other */
#define ILM_EXPANDED_FOR(i) (expb.ilmb.stg_base[i].w8)

#define DOREG1 (flg.opt == 1 && !XBIT(8, 0x8))
#define ADDRCAND(a, b) if (DOREG1) { exp_rcand((a), (b)); }

/* FTN string stuff */

#define STR_AREA 6

/** \brief string descriptor */
typedef struct _str {
  char aisvar;       /**< string address is variable if TRUE */
  char liscon;       /**< string length is constant */
  char dtype;        /**< TY_CHAR or TY_NCHAR */
  int aval;          /**< address symptr or ili */
  int lval;          /**< string length or ili */
  int cnt;           /**< # items this list */
  int tempnum;       /**< temp # for this var */
  struct _str *next; /**< next strdesc */
} STRDESC;

/* data common to expander module  */

typedef struct {
  int temps[9]; /* ili index temp area during expand */
  struct {
    ILM_AUX *stg_base;
    int stg_size;
  } ilmb;
  union {
    uint16_t wd;
    struct {
      unsigned waitlbl : 1;    /* waiting for a LABEL ILM	 */
      unsigned noblock : 1;    /* no block has been created	 */
      unsigned excstat : 1;    /* excstat was changed		 */
      unsigned dbgline : 1;    /* blocks are to be debugged	 */
      unsigned callfg : 1;     /* function calls an external	 */
      unsigned sdscunsafe : 1; /* call might mod descriptor */
      unsigned noheader : 1;   /* no entry header written (ftn) */
    } bits;
  } flags;
  int nilms;   /* number of (short) words in the ILM block */
  int curlin;  /* line number of the current ILI block	 */
  int curbih;  /* index of BIH of the current ILT block	 */
  int curilt;  /* index of the current (last) ILT		 */
  int saveili; /* ILI (a JMP) not yet added to the block	 */
  SPTR retlbl;  /* ST index to the current return label	 */
  int retcnt;  /* decimal number for the current rtn label */
  int swtcnt;  /* decimal number for the last switch array */
  int arglist; /* ST index of the current argument list	 */
  struct {
    short next;  /* decimal # for the next arglist	 */
    short start; /* start # for arglists in a function	 */
    short max;   /* max "next" # for arglists in a func.	 */
  } arglcnt;
  int uicmp;         /* symbol table index of uicmp function	 */
  int gentmps;       /* general temps */
  bool qjsr_flag; /* qjsr present in the function/subprogram */
  bool intr_flag; /* intrinsic present in the function/subprogram*/
  int isguarded; /* increment when encounter DOBEGNZ */
  INT ilm_words;     /* # of ilm words in the current ili block */
  INT ilm_thresh;    /* if ilm_words > ilm_thresh, break block */
  SC_KIND sc;        /* storage class used for expander-created
                      * temporaries (SC_LOCAL, SC_PRIVATE).
                      */
  int lcpu2;         /* temporary for the current function's
                      * value of mp_lcpu2().
                      */
  int lcpu3;         /* temporary for the current function's
                      * value of mp_lcpu3().
                      */
  int ncpus2;        /* temporary for the  current function's
                      * value of mp_ncpus2().
                      */
  int chartmps;      /* character temps */
  int chardtmps;     /* char descriptor temps */
  STRDESC *str_base; /* string descriptor list */
  int str_size;
  int str_avail;
  int logcjmp;  /* compare & branch ili for logical values:
                 * default is IL_LCJMPZ (odd/even test); -x 125 8
                 * implies IL_ICJMPZ (zero/non-zero test).
                 * initialized by exp_init().
                 */
  SPTR aret_tmp; /* temporary for the alternate return value */
  int clobber_ir; /* gcc-asm clobber list (iregs) info */
  int clobber_pr; /* gcc-asm clobber list (pregs) info */
  SPTR mxcsr_tmp;  /* temporary for the value of the mxcsr */
  int implicitdataregions;
  DTYPE charlen_dtype;
} EXP;

extern EXP expb;

#define CHARLEN_64BIT (XBIT(68,1) || XBIT(68,0x20))

#ifdef EXPANDER_DECLARE_INTERNAL
/* Routines internal to the expander that should not be declared
   as part of the public interface. */

#define expand_throw_point(ilix, dtype, ili_st) \
  (DEBUG_ASSERT(0, "throw points supported only for C++"), (ilix))
#endif /* EXPANDER_DECLARE_INTERNAL */

#endif /* ifndef FE90 */

/**
   \brief ...
 */
int expand(void);

#ifndef FE90
/**
   \brief ...
 */
int exp_mac(ILM_OP opc, ILM *ilmp, int curilm);
#endif

/**
   \brief ...
 */
int getThreadPrivateTp(int sptr);

/**
   \brief ...
 */
int llGetThreadprivateAddr(int sptr);

#ifndef FE90
/**
   \brief ...
 */
int optional_missing_ilm(ILM *ilmpin);
#endif

/**
   \brief ...
 */
int optional_missing(int nme);

/**
   \brief ...
 */
int optional_present(int nme);

/**
   \brief ...
 */
void ds_init(void);

/**
   \brief ...
 */
void eval_ilm(int ilmx);

/**
   \brief ...
 */
void exp_cleanup(void);

/**
   \brief ...
 */
void exp_estmt(int ilix);

/**
   \brief ...
 */
void exp_init(void);

/**
   \brief ...
 */
void exp_label(SPTR lbl);

#ifndef FE90
/**
   \brief ...
 */
void exp_load(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_pure(SPTR extsym, int nargs, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_ref(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_store(ILM_OP opc, ILM *ilmp, int curilm);
#endif

/**
   \brief ...
 */
void ll_set_new_threadprivate(int oldsptr);

/**
   \brief ...
 */
void ref_threadprivate(int cmsym, int *addr, int *nm);

/**
   \brief ...
 */
void ref_threadprivate_var(int cmsym, int *addr, int *nm, int mark);

#ifndef FE90
/**
   \brief ...
 */
void replace_by_one(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void replace_by_zero(ILM_OP opc, ILM *ilmp, int curilm);
#endif

/**
   \brief ...
 */
void set_assn(int nme);

#endif

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILI_H_
#define ILI_H_

/**
   \file
   \brief ILI header file  -  x86-64 version.
 */

#ifndef ILITP_UTIL  /* don't include if building ilitp utility prog*/
#include "global.h"
#include "symtab.h"
#include "iliatt.h" /* defines ILI_OP */
#else
typedef unsigned short ILI_OP;
#endif

#include "atomic_common.h"

#ifndef MAX_OPNDS
#define MAX_OPNDS 5 /* Max number of operands for an ili */
#endif

/***** ILI Declarations *****/

typedef struct {
  ILI_OP opc;
  /* practically all hosts will insert 2 bytes of padding here. */
  int hshlnk;
  int count;
  int visit;
  int alt;
  int opnd[MAX_OPNDS];
  int vlist; /* linked list of ILI visited by a traversal */
  int tndx;
  int tndx2;
  int lili; /* CG: set to the ILI's linear ILI.  This field must be 0
             * on entry to function 'schedule()'. */
} ILI;

typedef struct {
  STG_MEMBERS(ILI);
} ILIB;

extern ILIB ilib;

#define ILI_REPL(i) ilib.stg_base[i].count
#define ILI_OPC(i) ((ILI_OP)ilib.stg_base[i].opc)
#define ILI_OPCP(i, j) ilib.stg_base[i].opc = (j)
#define ILI_HSHLNK(i) ilib.stg_base[i].hshlnk
#define ILI_VISIT(i) ilib.stg_base[i].visit
#define ILI_ALT(i) ilib.stg_base[i].alt
#define ILI_COUNT(i) ilib.stg_base[i].count
#define ILI_RAT(i) ilib.stg_base[i].count
#define ILI_OPND(i, opn) ilib.stg_base[i].opnd[(opn)-1]
#define ILI_VLIST(i) ilib.stg_base[i].vlist
#define ILI_TNDX(i) ilib.stg_base[i].tndx
#define ILI_TNDX2(i) ilib.stg_base[i].tndx2
#define ILI_LILI(i) ilib.stg_base[i].lili

#ifndef ILITP_UTIL
#ifdef __cplusplus
inline SPTR ILI_SymOPND(int i, int opn) {
  return static_cast<SPTR>(ILI_OPND(i, opn));
}

inline DTYPE ILI_DTyOPND(int i, int opn) {
  return static_cast<DTYPE>(ILI_OPND(i, opn));
}
#else
#define ILI_SymOPND ILI_OPND
#define ILI_DTyOPND ILI_OPND
#endif
#endif

/***** ILI Attributes Declarations *****/

typedef struct {
  const char *name;    /* ili name */
  const char *opcod;   /* machine instruction mnemonic (CG only) */
  short oprs;          /* number of operands */
  unsigned short attr; /* AT attributes. e.g. cse, dom,  -
                        * Field size (right to left):
                        *   4 -- IL_TYPE
                        *   1 -- IL_COMM
                        *   5 -- IL_RES
                        *   2 -- IL_DOM/CSEG
                        *   1 -- IL_SSENME
                        *   1 -- IL_VECT
                        * ------------------
                        *  14 -- total
                        */
  /* x86-64 code generator info:
   */
  unsigned notCG : 1;
  unsigned CGonly : 1;
  unsigned notAILI : 1;
  unsigned terminal : 1;
  unsigned move : 1;
  unsigned memdest : 1;
  unsigned ccarith : 1;
  unsigned cclogical : 1;
  unsigned ccmod : 1;
  unsigned shiftop : 1;
  unsigned memarg : 1;
  unsigned ssedp : 1;
  unsigned ssest : 1;
  unsigned conditional_branch : 1;
  unsigned sse_avx : 1;      /* ILI can generate SSE or AVX instructions */
  unsigned avx_only : 1;     /* ILI can only generate AVX instructions */
  unsigned avx_special : 1;  /* AVX version is a special case */
  unsigned avx3_special : 1; /* AVX3 version is a special case */
  unsigned asm_special : 1;
  unsigned asm_nop : 1;
  unsigned accel : 1;

  unsigned short replaceby;
  char size; /* can be 'b', 'w', 'l', 'q' or 'y', or 0 if unspecified */

  char oprflag[MAX_OPNDS]; /* ILIO_ type of each opnd.  See IL_OPRFLAG */
} ILIINFO;

extern ILIINFO ilis[];

typedef enum ILIO_KIND {
  ILIO_NULL = 0,
  ILIO_SYM = 1,
  ILIO_STC = 4,
  ILIO_OFF = 5,
  ILIO_NME = 6,
  ILIO_IR = 7,
  ILIO_HP = 8,
  ILIO_SP = 9,
  ILIO_DP = 10,
  ILIO_QP = 11,
  ILIO_CS = 12,
  ILIO_CD = 13,
  ILIO_AR = 14,
  ILIO_KR = 15,
  ILIO_XMM = 16, /* xmm register number */
  ILIO_X87 = 17,
  ILIO_DOUBLEDOUBLE = 18,
  ILIO_FLOAT128 = 19,
  ILIO_LNK = 20,
  ILIO_IRLNK = 21,
  ILIO_HPLNK = 22,
  ILIO_SPLNK = 23,
  ILIO_DPLNK = 24,
  ILIO_ARLNK = 25,
  ILIO_KRLNK = 26,
  ILIO_QPLNK = 27,
  ILIO_CSLNK = 28,
  ILIO_CDLNK = 29,
  ILIO_CQLNK = 30,
  ILIO_128LNK = 31,
  ILIO_256LNK = 32,
  ILIO_512LNK = 33,
  ILIO_X87LNK = 34,
  ILIO_DOUBLEDOUBLELNK = 35,
  ILIO_FLOAT128LNK = 36
} ILIO_KIND;

#define ILIO_MAX 36
#define ILIO_ISLINK(n) ((n) >= ILIO_IRLNK)

/* Reflexive defines */
#define ILIO_NULL ILIO_NULL
#define ILIO_SYM ILIO_SYM
#define ILIO_STC ILIO_STC
#define ILIO_OFF ILIO_OFF
#define ILIO_NME ILIO_NME
#define ILIO_IR ILIO_IR
#define ILIO_HP ILIO_HP
#define ILIO_SP ILIO_SP
#define ILIO_DP ILIO_DP
/* just for debug to dump ili */
#define ILIO_QP ILIO_QP
#define ILIO_CS ILIO_CS
#define ILIO_CD ILIO_CD
#define ILIO_AR ILIO_AR
#define ILIO_KR ILIO_KR
#define ILIO_XMM ILIO_XMM
#define ILIO_X87 ILIO_X87
#define ILIO_DOUBLEDOUBLE ILIO_DOUBLEDOUBLE
#define ILIO_FLOAT128 ILIO_FLOAT128
#define ILIO_LNK ILIO_LNK
#define ILIO_IRLNK ILIO_IRLNK
#define ILIO_HPLNK ILIO_HPLNK
#define ILIO_SPLNK ILIO_SPLNK
#define ILIO_DPLNK ILIO_DPLNK
#define ILIO_ARLNK ILIO_ARLNK
#define ILIO_KRLNK ILIO_KRLNK
#define ILIO_QPLNK ILIO_QPLNK
#define ILIO_CSLNK ILIO_CSLNK
#define ILIO_CDLNK ILIO_CDLNK
#define ILIO_CQLNK ILIO_CQLNK
#define ILIO_128LNK ILIO_128LNK
#define ILIO_256LNK ILIO_256LNK
#define ILIO_512LNK ILIO_512LNK
#define ILIO_X87LNK ILIO_X87LNK
#define ILIO_DOUBLEDOUBLELNK ILIO_DOUBLEDOUBLELNK
#define ILIO_FLOAT128LNK ILIO_FLOAT128LNK

/* ILIINFO.attr field definitions. */
#define ILIA_NULL 0

#define ILIA_COMM 1 /* comm field */

/* result type field */
typedef enum ILIA_RESULT {
  ILIA_TRM = 0,
  ILIA_LNK = 1,
  ILIA_IR = 2,
  ILIA_HP = 3,
  ILIA_SP = 4,
  ILIA_DP = 5,
  ILIA_AR = 6,
  ILIA_KR = 7,
  ILIA_CC = 8,
  ILIA_FCC = 9,
  ILIA_QP = 10,
  ILIA_CS = 11,
  ILIA_CD = 12,
  ILIA_CQ = 13,
  ILIA_128 = 14,
  ILIA_256 = 15,
  ILIA_512 = 16,
  ILIA_X87 = 17,
  ILIA_DOUBLEDOUBLE = 18,
  ILIA_FLOAT128 = 19
} ILIA_RESULT;

#define ILIA_MAX 19

/* Reflexive defines */
#define ILIA_TRM ILIA_TRM
#define ILIA_LNK ILIA_LNK
#define ILIA_IR ILIA_IR
#define ILIA_HP ILIA_HP
#define ILIA_SP ILIA_SP
#define ILIA_DP ILIA_DP
#define ILIA_AR ILIA_AR
#define ILIA_KR ILIA_KR
#define ILIA_CC ILIA_CC
#define ILIA_FCC ILIA_FCC
#define ILIA_QP ILIA_QP
#define ILIA_CS ILIA_CS
#define ILIA_CD ILIA_CD
#define ILIA_CQ ILIA_CQ
#define ILIA_128 ILIA_128
#define ILIA_256 ILIA_256
#define ILIA_512 ILIA_512
#define ILIA_X87 ILIA_X87
#define ILIA_DOUBLEDOUBLE ILIA_DOUBLEDOUBLE
#define ILIA_FLOAT128 ILIA_FLOAT128

#define ILIA_DOM 1 /* dom/cse field */
#define ILIA_CSE 2

/* Macros use the IL_RES(opc) as a value */
#define ILIA_ISIR(t) ((t) == ILIA_IR)
#define ILIA_ISSP(t) ((t) == ILIA_SP)
#define ILIA_ISDP(t) ((t) == ILIA_DP)
#define ILIA_ISAR(t) ((t) == ILIA_AR)
#define ILIA_ISKR(t) ((t) == ILIA_KR)
#define ILIA_ISCS(t) ((t) == ILIA_CS)
#define ILIA_ISCD(t) ((t) == ILIA_CD)

/* operand type:    ILIO_... e.g. ILIO_DPLNK */

#ifdef __cplusplus
inline ILIO_KIND IL_OPRFLAG(ILI_OP opcode, int opn) {
  return static_cast<ILIO_KIND>(ilis[opcode].oprflag[opn - 1]);
}
#else
#define IL_OPRFLAG(opcode, opn) (ilis[opcode].oprflag[opn - 1])
#endif

#define IL_OPRS(opc) (ilis[opc].oprs)
#define IL_NAME(opc) (ilis[opc].name)
#define IL_MNEMONIC(opc) (ilis[opc].opcod)
#define IL_ISLINK(i, opn) (IL_OPRFLAG(i, opn) >= ILIO_LNK)

#define IL_COMM(i) ((ilis[i].attr >> 4) & 0x1)    /* Yields ILIA_COMM or 0    */
#define IL_RES(i) \
  ((ILIA_RESULT)((ilis[i].attr >> 5) & 0x1f))     /* Yields ILIA_TRM..ILIA_AR */
#define IL_LNK(i) ((ilis[i].attr >> 5) & 0x1f)    /* Yields ILIA_LNK or 0     */
#define IL_DOM(i) ((ilis[i].attr >> 10) & 0x3)    /* Yields ILIA_DOM or 0     */
#define IL_CSEG(i) ((ilis[i].attr >> 10) & 0x3)   /* Yields ILIA_CSE or 0     */
#define IL_IATYPE(i) ((ilis[i].attr >> 10) & 0x3) /* ILIA_DOM, ILIA_CSE or 0*/
#define IL_SSENME(i) ((ilis[i].attr >> 12) & 0x1) /* Yields 1 or 0 */
#define IL_VECT(i) ((ilis[i].attr >> 13) & 0x1)   /* Yields 1 or 0 */
/* Can this operation have a memory fence? */
#define IL_HAS_FENCE(i) (((ilis[i].attr >> 14) & 3) != 0)
/* Is this operation an IL_ATOMICRMWx? */
#define IL_IS_ATOMICRMW(i) (((ilis[i].attr >> 14) & 0x3) == 2)
/* Is this operation an IL_CMPXCHGx? */
#define IL_IS_CMPXCHG(i) (((ilis[i].attr >> 14) & 0x3) == 3)
/* Does this operation perform an atomic update?
   IL_OPND(2) is the address of the operand to be updated. */
#define IL_IS_ATOMIC_UPDATE(i) (((ilis[i].attr >> 14) & 0x3) >= 2)
/* The attribute of the atomic opcodes are non-zero 
 * in the 14th and 15th bit.  */
#define IL_IS_ATOMIC_OPC(i) (((ilis[i].attr >> 14) & 0x3) != 0)

typedef enum ILTY_KIND {
  ILTY_NULL = 0,
  ILTY_ARTH = 1,
  ILTY_BRANCH = 2,
  ILTY_CONS = 3,
  ILTY_DEFINE = 4,
  ILTY_LOAD = 5,
  ILTY_MOVE = 6,
  ILTY_OTHER = 7,
  ILTY_PROC = 8,
  ILTY_STORE = 9,
  ILTY_PLOAD = 10,
  ILTY_PSTORE = 11
} ILTY_KIND;

/* *** operation type:  ILTY_... e.g. ILTY_ARTH  */
#define IL_TYPE(idx) ((ILTY_KIND)(ilis[(idx)].attr & 0xf))

/* Reflexive defines for values inspected by #ifdef. */
#define ILTY_PLOAD ILTY_PLOAD
#define ILTY_PSTORE ILTY_PSTORE

/* Standard offsets for various register set references. */
#define IR_OFFSET 0
#define SP_OFFSET 1
#define DP_OFFSET 2
#define AR_OFFSET 3

/***** Values of conditions in relationals *****/

typedef enum CC_RELATION {
  CC_None,
  CC_EQ = 1,
  CC_NE = 2,
  CC_LT = 3,
  CC_GE = 4,
  CC_LE = 5,
  CC_GT = 6,
  CC_NOTEQ = 7,
  CC_NOTNE = 8,
  CC_NOTLT = 9,
  CC_NOTGE = 10,
  CC_NOTLE = 11,
  CC_NOTGT = 12,
  /* CC values are sometimes negated to denote IEEE floating-point relations.
     The -12 here is a "strut" to ensure that the enum's underlying integral
     type is signed. */
  CC_IEEE_NOTGT = -12
} CC_RELATION;

/* Let subsequent headers know that CC_RELATION is available. */
#define CC_RELATION_IS_DEFINED 1

#define NEW_FMA /* ...to generate FMA3 or FMA4 instructions */

/* The following flags are used in the 'stc' operand of an FMATYPE
 * ILI to describe an FMA instruction.  The FMA operation is:
 *	dest = <sign> (factor1 * factor2)  <addop>  term
 */
#define FMA_MINUS_PROD 1          /* if set <sign> is -, otherwise it's + */
#define FMA_MINUS_TERM 2          /* if set <addop> is -, otherwise it's + */
#define FMA_DEST_IS_FACTOR1 4     /* used for FMA3 */
#define FMA_DEST_IS_TERM 8        /* used for FMA3 & packed reduction FMAs */
#define FMA_MEMOP_IS_FACTOR2 0x10 /* used for [DS]FMA and P[DS]FMA ILIs */
#define FMA_MEMOP_IS_TERM 0x20    /*   "     "     "     "     "     "  */
#define FMA_GEN_FMA3_132 0x40
#define FMA_GEN_FMA3_213 0x80
#define FMA_GEN_FMA3_231 0x100
#define FMA_GEN_FMA4 0x200

/********************************************************************
 * JHM (7 April 2014): The following #define is necessary for
 * compiling comp.shared/llvm/src/llvect.c.  Delete it when possible.
 *******************************************************************/
#define FMA_DEST_IS_SRC1 FMA_DEST_IS_FACTOR1

/* The following flags are used in the 'stc' operand of VEXTRACT and
 * VINSERT ILIs to specify which 'vextract...' or 'vinsert...'
 * instruction to use.
 */
#define SUF_f128 0x10    /* only used in AVX instructions */
#define SUF_f32x4 0x20   /*   "   "   "  AVX3   "    "    */
#define SUF_f32x8 0x40   /*   "   "   "    "    "    "    */
#define SUF_f64x2 0x80   /*   "   "   "    "    "    "    */
#define SUF_f64x4 0x100  /*   "   "   "    "    "    "    */
#define SUF_i128 0x200   /*   "   "   "  AVX2   "    "    */
#define SUF_i32x4 0x400  /*   "   "   "  AVX3   "    "    */
#define SUF_i32x8 0x800  /*   "   "   "    "    "    "    */
#define SUF_i64x2 0x1000 /*   "   "   "    "    "    "    */
#define SUF_i64x4 0x2000 /*   "   "   "    "    "    "    */

#ifdef __cplusplus
inline MSZ MSZ_ILI_OPND(int i, int opn) {
  return static_cast<MSZ>(ILI_OPND(i, opn));
}
#else
#define MSZ_ILI_OPND ILI_OPND
#endif

#define MSZ_TO_BYTES                                                  \
  {                                                                   \
    1 /* SBYTE */, 2 /* SHWORD */, 4 /* SWORD */, 8 /* SLWORD */,     \
      1 /* UBYTE */, 2 /* UHWORD */, 4 /* UWORD */, 8 /* ULWORD */,   \
      0 /* 0x08  */, 2 /* FHALF  */, 4 /* FWORD */, 8 /* FLWORD */,   \
      0 /* 0x0c  */, 0 /* 0x0d   */, 0 /* 0x0e  */, 8 /* I8     */,   \
      0 /* 0x10  */, 0 /* 0x11   */, 0 /* 0x12  */, 8 /* PTR    */,   \
      0 /* 0x14  */, 0 /* 0x15   */, 16 /* F10  */, 16 /* F16   */,   \
      0 /* 0x18  */, 0 /* 0x19   */, 32 /* F32  */, 16 /* F8x2  */,   \
      0 /* 0x1c  */, 0 /* 0x1d   */, 0 /* 0x1e  */, 0 /* 0x1f   */    \
  }

/* Reflexive defines for values that are inspected by preprocessor directives */
#define MSZ_F10 MSZ_F10
#define MSZ_I8 MSZ_I8
#define MSZ_SLWORD MSZ_SLWORD
#define MSZ_ULWORD MSZ_ULWORD
#define MSZ_UWORD MSZ_UWORD

/* Synonyms (beware conflicting case values) */
#define MSZ_WORD MSZ_SWORD
#define MSZ_BYTE MSZ_UBYTE
#define MSZ_F2 MSZ_FHALF
#define MSZ_F4 MSZ_FWORD
#define MSZ_F8 MSZ_FLWORD
#define MSZ_DBLE MSZ_FLWORD
#define MSZ_DFLWORD MSZ_FLWORD
#define MSZ_DSLWORD MSZ_SLWORD

typedef struct {
  unsigned int latency; /* ST | LD | R/R | R/M */
  unsigned int attrs;   /* ST | LD | R/M | R/R */
} SCHINFO;

#define P_FADD 0x01
#define P_FMUL 0x02
#define P_FST 0x04
#define DEC_DIR 0x10
#define DEC_DBL 0x20
#define DEC_VEC 0x40

#define ST_SHIFT 24
#define LD_SHIFT 16
#define RM_SHIFT 8
#define RR_SHIFT 0

#define SCH_ATTR(i) (schinfo[(i)].attrs)
#define SCH_LAT(i) (schinfo[(i)].latency)

/* ---------------------------------------------------------------------- */

#ifndef ILITP_UTIL
extern bool share_proc_ili; /* defd in iliutil.c */
extern bool share_qjsr_ili; /* defd in iliutil.c */

/*  declare external functions iliutil.c, unless building ilitp utility prog */

#define XBIT_NEW_MATH_NAMES XBIT(164, 0x800000)

/* The following macro is for experimenting with the new method for certain
 * complex operations/intrinsics -- when complete, just drop _CMPLX from the 
 * use(s).
 */
#define XBIT_NEW_MATH_NAMES_CMPLX (XBIT_NEW_MATH_NAMES && XBIT(26,1))

#define XBIT_NEW_RELAXEDMATH XBIT(15, 0x400)

#define XBIT_VECTORABI_FOR_SCALAR XBIT(26,2)

/*****  ILT, BIH, NME  declarations  *****/
#include "ilt.h"
#include "bih.h"
#include "nme.h"

/***** Atomic Operation Encodings *****/

/* Extract MSZ from an int that is a MSZ operand or an encoded ATOMIC_INFO.
   This functionality is handy for extracting the MSZ from an instruction
   that might be a plain load/store or atomic/load/store. */ 
#define ILI_MSZ_FROM_STC(x) ((MSZ)(x)&0xFF)

/* Get MSZ of an IL_LD or IL_ATOMICLDx instruction */
#define ILI_MSZ_OF_LD(ilix) (ILI_MSZ_FROM_STC(ILI_OPND((ilix), 3)))

/* Get MSZ of an IL_ST, IL_STHP, IL_STSP, IL_STDP, or IL_ATOMICSTx instruction */
#define ILI_MSZ_OF_ST(ilix) (ILI_MSZ_FROM_STC(ILI_OPND((ilix), 4)))

#include "iliutil.h"

#ifdef __cplusplus
inline MSZ GetILI_MSZ_OF_Load(int ilix) {
  return static_cast<MSZ>(ILI_MSZ_OF_LD(ilix));
}
#undef ILI_MSZ_OF_LD
#define ILI_MSZ_OF_LD GetILI_MSZ_OF_Load
inline MSZ GetILI_MSZ_OF_Store(int ilix) {
  return static_cast<MSZ>(ILI_MSZ_OF_ST(ilix));
}
#undef ILI_MSZ_OF_ST
#define ILI_MSZ_OF_ST GetILI_MSZ_OF_Store
#endif

#endif /* !ILITP_UTIL */

#endif /* ILI_H_ */

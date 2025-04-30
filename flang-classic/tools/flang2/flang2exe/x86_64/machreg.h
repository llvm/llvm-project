/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef MACHREG_H_
#define MACHREG_H_

#include "gbldefs.h"

extern const int scratch_regs[];

/* Define registers for x86-64.
 */
/*------------------------------------------------------------------------
 * Registers must be listed with the callee-saved registers at the
 * end, and must be numbered as follows:
 *
 * non-callee-saved GP regs:  1 --> (IR_FIRST_CALLEE_SAVE - 1)
 * callee-saved GP regs:      IR_FIRST_CALLEE_SAVE --> IR_LAST_CALLEE_SAVE
 *
 * and similarly for the XM registers.  This numbering is assumed in
 * function 'save_callee_save_regs()'.
 *----------------------------------------------------------------------*/

#if !defined(TARGET_WIN_X8664)
/*--------------------------
 * GP registers, Unix-64 ABI
 *------------------------*/
typedef enum {
  NO_REG = -1,
  IR_RAX = 1,
  IR_RCX,    /* = 2 */
  IR_RDX,    /* = 3 */
  IR_RSI,    /* = 4 */
  IR_RDI,    /* = 5 */
  IR_R8,     /* = 6 */
  IR_R9,     /* = 7 */
  IR_R10,    /* = 8 */
  IR_R11,    /* = 9 */
  IR_RBX,    /* = 10; first callee-saved on Unix-64 */
  IR_RBP,    /* = 11 */
  IR_R12,    /* = 12 */
  IR_R13,    /* = 13 */
  IR_R14,    /* = 14 */
  IR_R15,    /* = 15; last callee-saved on Unix-64, i.e. 6 c.s. GP regs */
  IR_RSP     /* = 16 */
} IR_REGS;

#define GP_REG_NAMES    { "%badreg",                          \
                          "%rax",  "%rcx",  "%rdx",  "%rsi",  \
                          "%rdi",  "%r8",   "%r9",   "%r10",  \
                          "%r11",  "%rbx",  "%rbp",  "%r12",  \
                          "%r13",  "%r14",  "%r15",  "%rsp" }

#define WORD_REG_NAMES  { "%badreg",                          \
                          "%eax",  "%ecx",  "%edx",  "%esi",  \
                          "%edi",  "%r8d",  "%r9d",  "%r10d", \
                          "%r11d", "%ebx",  "%ebp",  "%r12d", \
                          "%r13d", "%r14d", "%r15d", "%esp" }

#define HALF_REG_NAMES  { "%badreg",                          \
                          "%ax",   "%cx",   "%dx",   "%si",   \
                          "%di",   "%r8w",  "%r9w",  "%r10w", \
                          "%r11w", "%bx",   "%bp",   "%r12w", \
                          "%r13w", "%r14w", "%r15w", "%sp" }

#define BYTE_REG_NAMES  { "%badreg",                          \
                          "%al",   "%cl",   "%dl",   "%sil",  \
                          "%dil",  "%r8b",  "%r9b",  "%r10b", \
                          "%r11b", "%bl",   "%bpl",  "%r12b", \
                          "%r13b", "%r14b", "%r15b", "%spl" }
#else
/*-----------------------------
 * GP registers, Windows-64 ABI
 *---------------------------*/
typedef enum {
  NO_REG = -1,
  IR_RAX = 1,
  IR_RCX,    /* = 2 */
  IR_RDX,    /* = 3 */
  IR_R8,     /* = 4 */
  IR_R9,     /* = 5 */
  IR_R10,    /* = 6 */
  IR_R11,    /* = 7 */
  IR_RBX,    /* = 8; first callee-saved on Win-64 */
  IR_RBP,    /* = 9 */
  IR_RDI,    /* = 10 */
  IR_RSI,    /* = 11 */
  IR_R12,    /* = 12 */
  IR_R13,    /* = 13 */
  IR_R14,    /* = 14 */
  IR_R15,    /* = 15; last callee-saved on Win-64, i.e. 8 c.s. GP regs */
  IR_RSP     /* = 16 */
} IR_REGS;

#define GP_REG_NAMES    { "%badreg",                          \
                          "%rax",  "%rcx",  "%rdx",  "%r8",   \
                          "%r9",   "%r10",  "%r11",  "%rbx",  \
                          "%rbp",  "%rdi",  "%rsi",  "%r12",  \
                          "%r13",  "%r14",  "%r15",  "%rsp" }

#define WORD_REG_NAMES  { "%badreg",                          \
                          "%eax",  "%ecx",  "%edx",  "%r8d",  \
                          "%r9d",  "%r10d", "%r11d", "%ebx",  \
                          "%ebp",  "%edi",  "%esi",  "%r12d", \
                          "%r13d", "%r14d", "%r15d", "%esp" }

#define HALF_REG_NAMES  { "badreg",                           \
                          "%ax",   "%cx",   "%dx",   "%r8w",  \
                          "%r9w",  "%r10w", "%r11w", "%bx",   \
                          "%bp",   "%di",   "%si",   "%r12w", \
                          "%r13w", "%r14w", "%r15w", "%sp" }

#define BYTE_REG_NAMES  { "%badreg",                          \
                          "%al",   "%cl",   "%dl",   "%r8b",  \
                          "%r9b",  "%r10b", "%r11b", "%bl",   \
                          "%bpl",  "%dil",  "%sil",  "%r12b", \
                          "%r13b", "%r14b", "%r15b", "%spl" }
#endif /* end Windows-64 ABI */

#define IR_FIRST_CALLEE_SAVE  IR_RBX    /* = 8 for Win-64 or 10 for Unix-64 */
#define IR_LAST_CALLEE_SAVE   IR_R15    /* = 15 */

#define N_GP_REGS     16

#define IR_FIRST       1                 /* unused! */
#define IR_LAST       16                 /* only used in invar.c */
#define IR_NUM_NAMES  (N_GP_REGS + 1)    /* only used in dwarf2.c */


/*---------------------------
 * XMM, YMM and ZMM registers
 *-------------------------*/
typedef enum {
  XR_XMM0 = 1,
  XR_XMM1,    /* = 2 */
  XR_XMM2,    /* = 3 */
  XR_XMM3,    /* = 4 */
  XR_XMM4,    /* = 5 */
  XR_XMM5,    /* = 6 */
  XR_XMM6,    /* = 7; first callee-saved on Win-64 */
  XR_XMM7,    /* = 8 */
  XR_XMM8,    /* = 9 */
  XR_XMM9,    /* = 10 */
  XR_XMM10,   /* = 11 */
  XR_XMM11,   /* = 12 */
  XR_XMM12,   /* = 13 */
  XR_XMM13,   /* = 14 */
  XR_XMM14,   /* = 15 */
  XR_XMM15,   /* = 16; last callee-saved on Win-64, i.e. 10 c.s. XMM regs */

  XR_XMM16,   /* = 17; only available in AVX-512 */
  XR_XMM17,   /* = 18;   "      "      "      "  */
  XR_XMM18,   /* = 19;   "      "      "      "  */
  XR_XMM19,   /* = 20;   "      "      "      "  */
  XR_XMM20,   /* = 21;   "      "      "      "  */
  XR_XMM21,   /* = 22;   "      "      "      "  */
  XR_XMM22,   /* = 23;   "      "      "      "  */
  XR_XMM23,   /* = 24;   "      "      "      "  */
  XR_XMM24,   /* = 25;   "      "      "      "  */
  XR_XMM25,   /* = 26;   "      "      "      "  */
  XR_XMM26,   /* = 27;   "      "      "      "  */
  XR_XMM27,   /* = 28;   "      "      "      "  */
  XR_XMM28,   /* = 29;   "      "      "      "  */
  XR_XMM29,   /* = 30;   "      "      "      "  */
  XR_XMM30,   /* = 31;   "      "      "      "  */
  XR_XMM31    /* = 32;   "      "      "      "  */
} XR_REGS;

#define XMM_REG_NAMES  { "%badxmm",                              \
                         "%xmm0",  "%xmm1",  "%xmm2",  "%xmm3",  \
                         "%xmm4",  "%xmm5",  "%xmm6",  "%xmm7",  \
                         "%xmm8",  "%xmm9",  "%xmm10", "%xmm11", \
                         "%xmm12", "%xmm13", "%xmm14", "%xmm15", \
                         "%xmm16", "%xmm17", "%xmm18", "%xmm19", \
                         "%xmm20", "%xmm21", "%xmm22", "%xmm23", \
                         "%xmm24", "%xmm25", "%xmm26", "%xmm27", \
                         "%xmm28", "%xmm29", "%xmm30", "%xmm31" }

#define YMM_REG_NAMES  { "%badymm",                              \
                         "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",  \
                         "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",  \
                         "%ymm8",  "%ymm9",  "%ymm10", "%ymm11", \
                         "%ymm12", "%ymm13", "%ymm14", "%ymm15", \
                         "%ymm16", "%ymm17", "%ymm18", "%ymm19", \
                         "%ymm20", "%ymm21", "%ymm22", "%ymm23", \
                         "%ymm24", "%ymm25", "%ymm26", "%ymm27", \
                         "%ymm28", "%ymm29", "%ymm30", "%ymm31" }

#define ZMM_REG_NAMES  { "%badzmm",                              \
                         "%zmm0",  "%zmm1",  "%zmm2",  "%zmm3",  \
                         "%zmm4",  "%zmm5",  "%zmm6",  "%zmm7",  \
                         "%zmm8",  "%zmm9",  "%zmm10", "%zmm11", \
                         "%zmm12", "%zmm13", "%zmm14", "%zmm15", \
                         "%zmm16", "%zmm17", "%zmm18", "%zmm19", \
                         "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                         "%zmm24", "%zmm25", "%zmm26", "%zmm27", \
                         "%zmm28", "%zmm29", "%zmm30", "%zmm31" }

#if !defined(TARGET_WIN_X8664)
/*-----------------------------------------------------------------
 * Unix-64 ABI: no callee-saved xmm registers.  Note, the last
 * non-callee-saved XM register must be (XR_FIRST_CALLEE_SAVE - 1).
 *---------------------------------------------------------------*/
#define XR_FIRST_CALLEE_SAVE  XR_XMM16    /* i.e. no callee-saved xmm regs */
#define XR_LAST_CALLEE_SAVE   XR_XMM15    /*   "    "    "    "    "    "  */
#else
/*-----------------------------------------------
 * Windows-64 ABI: 10 callee-saved xmm registers.
 *---------------------------------------------*/
#define XR_FIRST_CALLEE_SAVE  XR_XMM6
#define XR_LAST_CALLEE_SAVE   XR_XMM15
#endif

#define MAX_N_XMM_REGS          32    /* 32 for AVX3, else 16 */
#define MAX_N_GP_AND_XMM_REGS   (N_GP_REGS + MAX_N_XMM_REGS)

#define XR_FIRST         1    /* only used in machreg.c */
#define XR_LAST         16    /*    "    "    "    "    */
#define XR_NUM_REGS     16    /* only used in {hammer,llvm}/src/llvect.c */

/*-------------------------
 * AVX-512 opmask registers
 *-----------------------*/
enum {
  OR_K0 = 1,
  OR_K1,
  OR_K2,
  OR_K3,
  OR_K4,
  OR_K5,
  OR_K6,
  OR_K7
};

#define OPMASK_REG_NAMES  { "%badopmask",                  \
                            "%k0",  "%k1",  "%k2",  "%k3", \
                            "%k4",  "%k5",  "%k6",  "%k7" }

/* No callee-saved opmask registers.  Note, the last non-callee-saved
 * opmask register must be (OR_FIRST_CALLEE_SAVE - 1).
 */
#define OR_FIRST_CALLEE_SAVE  (OR_K7 + 1)    /* i.e. no c.s. opmask regs */
#define OR_LAST_CALLEE_SAVE   OR_K7          /*   "    "    "    "    "  */

#define N_OPMASK_REGS     8

#define MAX_N_REGS        (N_GP_REGS + MAX_N_XMM_REGS + N_OPMASK_REGS)

/*------------------------------------------------------------------
 * Assembly code representation of register names.  These arrays are
 * defined and initialised in cgassem.c and read in assem.c,
 * cgassem.c, cggenai.c, exp_rte.c and xprolog.c.
 *----------------------------------------------------------------*/

extern char *gp_reg[N_GP_REGS + 1];         /* GP_REG_NAMES */
extern char *word_reg[N_GP_REGS + 1];       /* WORD_REG_NAMES */
extern char *half_reg[N_GP_REGS + 1];       /* HALF_REG_NAMES */
extern char *byte_reg[N_GP_REGS + 1];       /* BYTE_REG_NAMES */

extern char *xm_reg[MAX_N_XMM_REGS + 1];    /* XMM_REG_NAMES */
extern char *ym_reg[MAX_N_XMM_REGS + 1];    /* YMM_REG_NAMES */
extern char *zm_reg[MAX_N_XMM_REGS + 1];    /* ZMM_REG_NAMES */

extern char *opmask_reg[N_OPMASK_REGS + 1];    /* OPMASK_REG_NAMES */

#define RAX   gp_reg[IR_RAX]
#define RBX   gp_reg[IR_RBX]
#define RCX   gp_reg[IR_RCX]
#define RDX   gp_reg[IR_RDX]
#define RDI   gp_reg[IR_RDI]
#define RSI   gp_reg[IR_RSI]
#define R8    gp_reg[IR_R8 ]
#define R9    gp_reg[IR_R9 ]
#define R10   gp_reg[IR_R10]
#define R11   gp_reg[IR_R11]
#define R12   gp_reg[IR_R12]
#define R13   gp_reg[IR_R13]
#define R14   gp_reg[IR_R14]
#define R15   gp_reg[IR_R15]
#define RBP   gp_reg[IR_RBP]
#define RSP   gp_reg[IR_RSP]

#define EAX   word_reg[IR_RAX]
#define EBX   word_reg[IR_RBX]
#define ECX   word_reg[IR_RCX]
#define EDX   word_reg[IR_RDX]
#define EDI   word_reg[IR_RDI]
#define ESI   word_reg[IR_RSI]
#define R8D   word_reg[IR_R8 ]
#define R9D   word_reg[IR_R9 ]
#define R10D  word_reg[IR_R10]
#define R11D  word_reg[IR_R11]
#define R12D  word_reg[IR_R12]
#define R13D  word_reg[IR_R13]
#define R14D  word_reg[IR_R14]
#define R15D  word_reg[IR_R15]
#define EBP   word_reg[IR_RBP]
#define ESP   word_reg[IR_RSP]

#define XR_SAVE_FOR_ECG  8    /* if llvect is not generating code for entire
                               *   loop, number of registers it should leave
                               *   code generator to generate scalarsse code.
                               */
#define FR_RETVAL XR_XMM0
#define SP_RETVAL XR_XMM0
#define DP_RETVAL XR_XMM0
#define CS_RETVAL XR_XMM0
#define CD_RETVAL XR_XMM0

#define IR_RETVAL IR_RAX
#define AR_RETVAL IR_RAX
#define IR_FRAMEP IR_RBP
#define IR_STACKP IR_RSP
#define MEMARG_OFFSET 8

#if defined(TARGET_WIN_X8664)
#define MR_MAX_IREG_ARGS 4
#define MR_MAX_XREG_ARGS 4
#define MR_MAX_IREG_RES 1
#define MR_MAX_XREG_RES 1
#define MR_MAX_ARGRSRV 32
#else
#define MR_MAX_IREG_ARGS 6
#define MR_MAX_XREG_ARGS 8
#define MR_MAX_IREG_RES 2
#define MR_MAX_XREG_RES 2
#endif

/*  not used to pass args */
#define MR_MAX_FREG_ARGS 0

/* Use macros ARG_IR, ARG_XR, etc.
 */
extern int mr_arg_ir[MR_MAX_IREG_ARGS]; /* defd in machreg.c */
extern int mr_arg_xr[MR_MAX_XREG_ARGS]; /* defd in machreg.c */
extern int mr_res_ir[MR_MAX_IREG_RES];
extern int mr_res_xr[MR_MAX_XREG_RES];

#define ARG_IR(i) (mr_arg_ir[i])
#define ARG_XR(i) (mr_arg_xr[i])
#define RES_IR(i) (mr_res_ir[i])
#define RES_XR(i) (mr_res_xr[i])

#define AR(i) IR_RETVAL /* 32-bit only */
#define IR(i) ARG_IR(i)
#define SP(i) ARG_XR(i)
#define DP(i) ARG_XR(i)
#define ISP(i) (i + 100) /* not used? */
#define IDP(i) (i + 100)

/* Macro for defining alternate-return register for fortran subprograms.
 */
#define IR_ARET IR_RETVAL

/* Macros for unpacking/packing KR registers.
 * NOTE: for hammer, KR regs are just the IR regs - KR_PACK is assumed
 * to be invoked with two IR registers represented by the IR() macros.
 * The IR() macros are assumed to be of the form IR(2*n), IR(2*n+1),
 * where n is the nth KR argument, n >= 0.
 * KR_PACK computes from its first argument which 64-bit register to use.
 */
#define KR_LSH(i) (i)
#define KR_MSH(i) (i)
#define KR_PACK(ms, ls) ARG_IR((&(ms) - &IR(0)) >> 1)

/* Macro for defining the KR register in which the value of a 64-bit integer
 * function is returned.
 */
#define KR_RETVAL IR_RETVAL

/* Define MR_UNIQ, the number of unique register classes for the machine.
 */
#define MR_UNIQ 3

#define GR_THRESHOLD 2

/* Macros for defining the global registers in each of the unique register
 * classes.  For each global set, the lower and upper bounds are specified
 * in the form MR_L<i> .. MR_U<i>, where i = 1 to MR_UNIQ.
 */
/***** i386 general purpose regs - allow 3 global *****/
#define MR_L1 1
#define MR_U1 3
#define MR_MAX1 (MR_U1 - MR_L1 + 1)

/***** i387 floating-point regs - allow 3 global *****/
#define MR_L2 2
#define MR_U2 4
#define MR_MAX2 (MR_U2 - MR_L2 + 1)

/***** i387 xmm floating-point regs - allow 3 global *****/
#define MR_L3 2
#define MR_U3 4
#define MR_MAX3 (MR_U3 - MR_L3 + 1)

/* Total number of globals: used by the optimizer for register history
 * tables.
 */
#define MR_NUMGLB (MR_MAX1 + MR_MAX2 + MR_MAX3)

/* Number of integer registers which are available for global
 * assignment when calls are or are not present.
 */
#define MR_IR_AVAIL(c) 0

/* Define gindex bounds for the set of global irs/ars and scratch
 * irs/ars.  MUST BE CONSISTENT with mr_gindex().
 */
#define MR_GI_IR_LOW 0
#define MR_GI_IR_HIGH (MR_U1 - MR_L1)
#define MR_GI_IS_SCR_IR(i) ((i) > (MR_U1 - MR_L1))

/* Machine Register Information -
 *
 * This information is in two pieces:
 * 1.  a structure exists for each machine's register class which defines
 *     the attributes of registers.
 *     These attributes define a register set with the following properties:
 *         1).  a set is just an increasing set of numbers,
 *         2).  scratch registers are allocated in increasing order (towards
 *              the globals),
 *         3).  globals are allocated in decreasing order (towards the
 *              scratch registers).
 *         4).  the scratch set that can be changed by a procedure
 *              [min .. first_global-1]
 *         5).  the scratch set that can be changed by an intrinisic
 *              [min .. intrinsic]
 *
 * 2.  a structure exists for all of the generic register classes which will
 *     map a register type (macros in registers.h) to the appropriate
 *     machine register class.
 */

/*****  Machine Register Table  *****/

typedef struct {
  char min;       /* minimum register # */
  char max;       /* maximum register # */
  char intrinsic; /* last scratch that can be changed by an intrinsic */
  const char first_global; /* first register # that can be global
                      * Note that the globals are allocated in increasing
                      * order (first_global down to last_global).
                      */
  const char end_global;   /* absolute last register # that can be global. */
                           /* the following two really define the working set
                              of registers that can be assigned. */
  char next_global;        /* next global register # */
  char last_global;        /* last register # that can be global. */
  char nused;              /* number of global registers assigned */
  const char mapbase;      /* offset in register bit vector where
                              this class of MACH_REGS begins. */
  const char Class;        /* class or type of register.  code generator needs
                              to know what kind of registers these represent.
                              'i' (integer), 'f' (float stk), 'x' (float xmm) */
} MACH_REG;

/*****  Register Mapping Table  *****/

typedef struct {
  char max;           /* maximum number of registers */
  char nused;         /* number of registers assigned */
  char joined;        /* non-zero value if registers are formed from
                       * multiple machine registers. 1==>next machine
                       * register is used; other values TBD.
                       */
  int rcand;          /* register candidate list */
  MACH_REG *mach_reg; /* pointer to struct of the actual registers */
  INT const_flag;     /* flag controlling assignment of consts */
} REG;                /*  [rtype]  */

/*****  Register Set Information for a block  *****/

typedef struct {/* three -word bit-vector */
  int xr;
} RGSET;

#define RGSETG(i) rgsetb.stg_base[i]

#define RGSET_XR(i) rgsetb.stg_base[i].xr

#define SET_RGSET_XR(i, reg)     \
  {                              \
    RGSET_XR(i) |= (1 << (reg)); \
  }

#define TST_RGSET_XR(i, reg) ((RGSET_XR(i) >> (reg)) & 1)

typedef struct {
  RGSET *stg_base;
  int stg_avail;
  int stg_size;
} RGSETB;

/*****  External Data  Declarations  *****/

extern REG reg[];
extern RGSETB rgsetb;

/*****  External Function Declarations  *****/

/**
   \brief ...
 */
int mr_getnext(int rtype);

/**
   \brief ...
 */
int mr_getreg(int rtype);

/**
   \brief ...
 */
int mr_get_rgset(void);

/**
   \brief ...
 */
int mr_gindex(int rtype, int regno);

/**
   \brief ...
 */
void mr_end(void);

/**
   \brief ...
 */
void mr_init(void);

/**
   \brief ...
 */
void mr_reset_frglobals(void);

/**
   \brief ...
 */
void mr_reset(int rtype);

/**
   \brief ...
 */
void mr_reset_numglobals(int reduce_by);


#endif // MACHREG_H_

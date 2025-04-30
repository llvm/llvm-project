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

/* Define registers for x86-32.
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

/*-------------
 * GP registers
 *-----------*/

typedef enum {
  NO_REG = -1,
  IR_EAX = 1,
  IR_ECX,
  IR_EDX,
  IR_ESI, /* = 4; first callee-saved on x86-32 */
  IR_EDI,
  IR_EBX,
  IR_EBP, /* = 7; last callee-saved on x86-32, i.e. 4 c.s. GP regs */
  IR_ESP  /* = 8 */
} IR_REGS;

#define IR_FIRST_CALLEE_SAVE IR_ESI
#define IR_LAST_CALLEE_SAVE IR_EBP

#define GP_REG_NAMES                                                           \
  {                                                                            \
    "%badreg0", "%eax", "%ecx", "%edx", "%esi", "%edi", "%ebx", "%ebp", "%esp" \
  }

#define WORD_REG_NAMES                                                         \
  {                                                                            \
    "%badreg1", "%eax", "%ecx", "%edx", "%esi", "%edi", "%ebx", "%ebp", "%esp" \
  }

#define HALF_REG_NAMES                                                 \
  {                                                                    \
    "%badreg2", "%ax", "%cx", "%dx", "%si", "%di", "%bx", "%bp", "%sp" \
  }

#define BYTE_REG_NAMES                                              \
  {                                                                 \
    "%badreg3", "%al", "%cl", "%dl", "%badreg4", "%badreg5", "%bl", \
        "%badreg6", "%badreg7"                                      \
  }

/* Synonyms for GP register symbols.
 */
#define IR_RAX IR_EAX
#define IR_RCX IR_ECX
#define IR_RDX IR_EDX
#define IR_RSI IR_ESI
#define IR_RDI IR_EDI
#define IR_RBX IR_EBX
#define IR_FRAMEP IR_EBP
#define IR_STACKP IR_ESP

#define N_GP_REGS 8
#define IR_FIRST 1
#define IR_LAST 8

/*---------------------------
 * XMM, YMM and ZMM registers
 *-------------------------*/

typedef enum {
  XR_XMM0 = 1,
  XR_XMM1,
  XR_XMM2,
  XR_XMM3,
  XR_XMM4,
  XR_XMM5,
  XR_XMM6,
  XR_XMM7
} XR_REGS;

/* 32-bit ABI - no callee-saved xmm registers.  Note, the last
 * non-callee-saved XM register must be ( XR_FIRST_CALLEE_SAVE - 1 ).
 */
#define XR_FIRST_CALLEE_SAVE 9 /* no callee-saved xmm regs */
#define XR_LAST_CALLEE_SAVE 8

#define XMM_REG_NAMES                                                         \
  {                                                                           \
    "%badxmm", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", \
        "%xmm7"                                                               \
  }

#define YMM_REG_NAMES                                                         \
  {                                                                           \
    "%badymm", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", \
        "%ymm7"                                                               \
  }

#define ZMM_REG_NAMES                                                         \
  {                                                                           \
    "%badzmm", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", \
        "%zmm7"                                                               \
  }

#define MAX_N_XMM_REGS 8
#define XR_FIRST 1
#define XR_LAST 8
#define XR_NUM_REGS 8 /* only used in {hammer,llvm}/src/llvect.c */

#define MAX_N_REGS (N_GP_REGS + MAX_N_XMM_REGS)

/*------------------------------------------------------------------
 * Assembly code representation of register names.  These arrays are
 * defined and initialised in cgassem.c and read in assem.c,
 * cgassem.c, cggenai.c, exp_rte.c and xprolog.c.
 *----------------------------------------------------------------*/

#define IR_NUM_NAMES N_GP_REGS + 1 /* only used in dwarf2.c! */

extern char *gp_reg[N_GP_REGS + 1];      /* GP_REG_NAMES */
extern char *word_reg[N_GP_REGS + 1];    /* WORD_REG_NAMES */
extern char *half_reg[N_GP_REGS + 1];    /* HALF_REG_NAMES */
extern char *byte_reg[N_GP_REGS + 1];    /* BYTE_REG_NAMES */
extern char *xm_reg[MAX_N_XMM_REGS + 1]; /* XMM_REG_NAMES */
extern char *ym_reg[MAX_N_XMM_REGS + 1]; /* YMM_REG_NAMES */
extern char *zm_reg[MAX_N_XMM_REGS + 1]; /* ZMM_REG_NAMES */

#define RAX gp_reg[IR_EAX]
#define RBX gp_reg[IR_EBX]
#define RCX gp_reg[IR_ECX]
#define RDX gp_reg[IR_EDX]
#define RDI gp_reg[IR_EDI]
#define RSI gp_reg[IR_ESI]
#define RBP gp_reg[IR_EBP]
#define RSP gp_reg[IR_ESP]

#define EAX word_reg[IR_EAX]
#define EBX word_reg[IR_EBX]
#define ECX word_reg[IR_ECX]
#define EDX word_reg[IR_EDX]
#define EDI word_reg[IR_EDI]
#define ESI word_reg[IR_ESI]
#define EBP word_reg[IR_EBP]
#define ESP word_reg[IR_ESP]

/* bobt, july 03 ------------------ I did up to here ..... */

#define FR_RETVAL XR_XMM0
#define SP_RETVAL XR_XMM0
#define DP_RETVAL XR_XMM0
#define CS_RETVAL XR_XMM0
#define CD_RETVAL XR_XMM0

#define IR_RETVAL IR_RAX
#define AR_RETVAL IR_RAX
#define MEMARG_OFFSET 8

#define MR_MAX_IREG_ARGS 0
#define MR_MAX_XREG_ARGS 8
/*  not used to pass args */
#define MR_MAX_FREG_ARGS 0

#define MR_MAX_IREG_RES 2
#define MR_MAX_XREG_RES 2

/* Use macros ARG_IR, ARG_XR, etc.
 */
extern int mr_arg_ir[MR_MAX_IREG_ARGS + 1]; /* defd in machreg.c */
extern int mr_arg_xr[MR_MAX_XREG_ARGS + 1]; /* defd in machreg.c */
extern int mr_res_ir[MR_MAX_IREG_RES + 1];
extern int mr_res_xr[MR_MAX_XREG_RES + 1];

#define ARG_IR(i) (scratch_regs[i]) /* initialized in machreg.c */
#define ARG_XR(i) (mr_arg_xr[i])
#define RES_IR(i) (mr_res_ir[i])
#define RES_XR(i) (mr_res_xr[i])

#define AR(i) IR_RETVAL /* used only for pgftn/386 */
#define IR(i) ARG_IR(i)
#define SP(i) ARG_XR(i)
#define DP(i) ARG_XR(i)
#define ISP(i) (i + 100) /* not used? */
#define IDP(i) (i + 100)

/* Macro for defining alternate-return register for fortran subprograms.
 */
#define IR_ARET IR_RETVAL

/* Macros for unpacking/packing KR registers.
 */
#define KR_LSH(i) (((i) >> 8) & 0xff)
#define KR_MSH(i) ((i)&0xff)
#define KR_PACK(ms, ls) (((ls) << 8) | (ms))

/* Macro for defining the KR register in which the value of a 64-bit integer
 * function is returned.
 */
#define KR_RETVAL KR_PACK(IR_EDX, IR_EAX)

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

 int mr_getnext(int rtype);
 int mr_getreg(int rtype);
 int mr_get_rgset();
 int mr_gindex(int rtype, int regno);
 void mr_end();
 void mr_init();
 void mr_reset_frglobals();
 void mr_reset(int rtype);
 void mr_reset_numglobals(int);

#endif

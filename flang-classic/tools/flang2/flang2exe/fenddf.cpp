/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Data definitions for FTN front-end data structures
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "semant.h"
#include "soc.h"
#include "ilm.h"
#include "ilmtp.h"
#include "flgdf.h"
#include "feddesc.h"

GBL gbl;

SEM sem;

SOC soc;

SWEL *switch_base;

AUX aux;
#include "ilmtpdf.h"

/*
 * ilm tables:
 */

/*
Basic types are (starting at 1) :

TY_WORD   TY_DWORD  TY_HOLL   TY_BINT   TY_SINT   TY_INT    TY_INT8   TY_HALF  TY_REAL
TY_DBLE   TY_QUAD   TY_HCMPLX  TY_CMPLX  TY_DCMPLX TY_BLOG   TY_SLOG   TY_LOG    TY_LOG8
TY_128    TY_CHAR   TY_NCHAR

opc opcodes are (starting at 0) :
OP_NEG    OP_ADD    OP_SUB    OP_MUL    OP_DIV    OP_XTOI   OP_XTOX
OP_CMP    OP_AIF    OP_LD     OP_ST     OP_FUNC   OP_CON
*/

/** ILM opcodes for basic types: */
short ilm_opcode[NOPC][2][NTYPE + 1] = {
    {/* NEG */ {0, IM_INEG, 0, 0, IM_INEG, IM_INEG, IM_INEG, IM_KNEG, IM_RNEG, IM_RNEG,
                IM_DNEG, 0, IM_CNEG, IM_CNEG, IM_CDNEG, IM_INEG, IM_INEG, IM_INEG,
                IM_KNEG, 0, 0},
     /* VNEG */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* ADD */ {0, IM_IADD, 0, 0, IM_IADD, IM_IADD, IM_IADD, IM_KADD, IM_RADD, IM_RADD,
                IM_DADD, 0, IM_CADD, IM_CADD, IM_CDADD, IM_IADD, IM_IADD, IM_IADD,
                IM_KADD, 0, 0},
     /* VADD */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* SUB */ {0, IM_ISUB, 0, 0, IM_ISUB, IM_ISUB, IM_ISUB, IM_KSUB, IM_RSUB, IM_RSUB,
                IM_DSUB, 0, IM_CSUB, IM_CSUB, IM_CDSUB, IM_ISUB, IM_ISUB, IM_ISUB,
                IM_KSUB, 0, 0},
     /* VSUB */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* MUL */ {0, IM_IMUL, 0, 0, IM_IMUL, IM_IMUL, IM_IMUL, IM_KMUL, IM_RMUL, IM_RMUL,
                IM_DMUL, 0, IM_CMUL, IM_CMUL, IM_CDMUL, IM_IMUL, IM_IMUL, IM_IMUL,
                IM_KMUL, 0, 0},
     /* VMUL */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* DIV */ {0, IM_IDIV, 0, 0, IM_IDIV, IM_IDIV, IM_IDIV, IM_KDIV, IM_RDIV, IM_RDIV,
                IM_DDIV, 0, IM_CDIV, IM_CDIV, IM_CDDIV, IM_IDIV, IM_IDIV, IM_IDIV,
                IM_KDIV, 0, 0},
     /* VDIV */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* xTOI */ {0, IM_ITOI, 0, 0, IM_ITOI, IM_ITOI, IM_ITOI, IM_KTOI, IM_RTOI, IM_RTOI,
                 IM_DTOI, 0, IM_CTOI, IM_CTOI, IM_CDTOI, IM_ITOI, IM_ITOI, IM_ITOI,
                 IM_KTOI, 0, 0},
     /* VxTOI */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* xTOK */ {0, IM_KTOK, 0, 0, IM_KTOK, IM_KTOK, IM_KTOK, IM_KTOK, IM_RTOK, IM_RTOK,
                 IM_DTOK, 0, IM_CTOK, IM_CTOK, IM_CDTOK, IM_KTOK, IM_KTOK, IM_KTOK,
                 IM_KTOK, 0, 0},
     /* VxTOI */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* xTOx */ {0, IM_ITOI, 0, 0, IM_ITOI, IM_ITOI, IM_ITOI, IM_KTOK, IM_RTOR, IM_RTOR,
                 IM_DTOD, 0, IM_CTOC, IM_CTOC, IM_CDTOCD, IM_ITOI, IM_ITOI, IM_ITOI,
                 IM_KTOK, 0, 0},
     /* VxTOx */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* CMP */ {0, IM_UICMP, IM_UDICMP, 0, IM_ICMP, IM_ICMP, IM_ICMP, IM_KCMP, IM_RCMP,
                IM_RCMP, IM_DCMP, 0, IM_CCMP, IM_CCMP, IM_CDCMP, IM_ICMP, IM_ICMP,
                IM_ICMP, IM_KCMP, 0, IM_SCMP, IM_NSCMP},
     /* VCMP */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* AIF  */ {0, IM_IAIF, 0, 0, IM_IAIF, IM_IAIF, IM_IAIF, IM_KAIF, IM_RAIF, IM_RAIF,
                 IM_DAIF, 0, 0, 0, IM_IAIF, IM_IAIF, IM_IAIF, IM_KAIF, 0, 0},
     /* VAIF non-existent */ {0}},
    {/* LD */ {0, 0, 0, 0, IM_CHLD, IM_SILD, IM_ILD, IM_KLD, IM_RLD, IM_RLD, IM_DLD, 0,
               IM_CLD, IM_CLD, IM_CDLD, IM_CHLD, IM_SLLD, IM_LLD, IM_KLLD, 0, 0},
     /* VLD */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* ST */ {0, IM_IST, IM_IST, 0, IM_CHST, IM_SIST, IM_IST, IM_KST, IM_RST, IM_RST,
               IM_DST, 0, IM_CST, IM_CST, IM_CDST, IM_CHST, IM_SLST, IM_LST, IM_KLST, 0,
               IM_SST, IM_NSST},
     /* VST */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {/* FUNC */ {0, IM_IFUNC, 0, 0, IM_IFUNC, IM_IFUNC, IM_IFUNC, IM_KFUNC,
                 IM_RFUNC, IM_RFUNC, IM_DFUNC, 0, IM_CFUNC, IM_CFUNC, IM_CDFUNC, IM_LFUNC, IM_LFUNC,
                 IM_LFUNC, IM_KFUNC, 0, 0},
     /* VFUNC non-existent */ {0}},
    {/* CON */ {0, IM_UCON, IM_UDCON, 0, IM_ICON, IM_ICON, IM_ICON, IM_KCON,
                IM_RCON, IM_RCON, IM_DCON, 0, IM_CCON, IM_CCON, IM_CDCON, IM_LCON, IM_LCON,
                IM_LCON, IM_KCON, 0, IM_BASE},
     /* VCON non-existent */ {0}}};

/** We only allow casting to and from TY_WORD and TY_DWORD.  Therefore, you
 * index into the following table as follows
 *       cast_types[dtype][{DT_WORD or DT_DWORD} - 1][0 for from or 1 for to]
 */
INT cast_types[NTYPE][2][2] = {
    /* DT_NONE */ {{0, 0}, {0, 0}},
    /* DT_WORD */ {{0, 0}, {IM_UITOUDI, IM_UDITOUI}},
    /* DT_DWORD */ {{IM_UDITOUI, IM_UITOUDI}, {0, 0}},
    /* DT_HOLL */ {{IM_ITOUI, -1}, {IM_ITOUDI, -1}},
    /* DT_BINT */ {{IM_SCTOUI, IM_UITOSC}, {IM_SCTOUDI, IM_UDITOSC}},
    /* DT_SINT */ {{IM_STOUI, IM_UITOS}, {IM_STOUDI, IM_UDITOS}},
    /* DT_INT */ {{IM_ITOUI, IM_UITOI}, {IM_ITOUDI, IM_UDITOI}},
    /* DT_INT8 */ {{IM_K2I, IM_UI2K}, {IM_K2D, IM_D2K}},
    /* DT_HALF */ {{IM_RTOUI, IM_UITOR}, {IM_RTOUDI, IM_UDITOR}},
    /* DT_REAL */ {{IM_RTOUI, IM_UITOR}, {IM_RTOUDI, IM_UDITOR}},
    /* DT_DBLE */ {{IM_DTOUI, IM_UITOD}, {IM_DTOUDI, IM_UDITOD}},
    /* DT_QUAD */ {{-1, -1}, {-1, -1}},
    /* DT_HCMPLX */ {{-1, -1}, {IM_CTOUDI, -1}},
    /* DT_CMPLX */ {{-1, -1}, {IM_CTOUDI, -1}},
    /* DT_DCMPLX */ {{-1, -1}, {IM_CDTOUDI, -1}},
    /* DT_BLOG */ {{IM_SCTOUI, IM_UITOSC}, {IM_SCTOUDI, IM_UDITOSC}},
    /* DT_SLOG */ {{IM_STOUI, IM_UITOS}, {IM_STOUDI, IM_UDITOS}},
    /* DT_LOG */ {{IM_ITOUI, IM_UITOI}, {IM_ITOUDI, IM_UDITOI}},
    /* DT_LOG8 */ {{IM_K2I, IM_UI2K}, {IM_K2D, IM_D2K}},
    /* DT_128 */ {{-1, -1}, {-1, -1}}};

int ty_to_lib[] = {
    -1,      /* TY_NONE */
    __WORD,  /* TY_WORD */
    __DWORD, /* TY_DWORD */
    __HOLL,  /* TY_HOLL */
    __BINT,  /* TY_BINT */
    __SINT,  /* TY_SINT */
    __INT,   /* TY_INT */
    __INT8,  /* TY_INT8 */
    __REAL,  /* TY_HALF */
    __REAL,  /* TY_REAL */
    __DBLE,  /* TY_DBLE */
    __QUAD,  /* TY_QUAD */
    __CPLX,  /* TY_HCMPLX */
    __CPLX,  /* TY_CMPLX */
    __DCPLX, /* TY_DCMPLX */
    __BLOG,  /* TY_BLOG */
    __SLOG,  /* TY_SLOG */
    __LOG,   /* TY_LOG */
    __LOG8,  /* TY_LOG8 */
    -1,      /* TY_128 */
    __CHAR,  /* TY_CHAR */
    __NCHAR, /* TY_NCHAR */
};

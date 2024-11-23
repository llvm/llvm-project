/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef BDSHEMU_X86_
#define BDSHEMU_X86_

#include "bddisasm_types.h"


//
// General purpose registers.
//
typedef struct _SHEMU_X86_GPR_REGS
{
    ND_UINT64       RegRax;
    ND_UINT64       RegRcx;
    ND_UINT64       RegRdx;
    ND_UINT64       RegRbx;
    ND_UINT64       RegRsp;
    ND_UINT64       RegRbp;
    ND_UINT64       RegRsi;
    ND_UINT64       RegRdi;
    ND_UINT64       RegR8;
    ND_UINT64       RegR9;
    ND_UINT64       RegR10;
    ND_UINT64       RegR11;
    ND_UINT64       RegR12;
    ND_UINT64       RegR13;
    ND_UINT64       RegR14;
    ND_UINT64       RegR15;
    ND_UINT64       RegR16;
    ND_UINT64       RegR17;
    ND_UINT64       RegR18;
    ND_UINT64       RegR19;
    ND_UINT64       RegR20;
    ND_UINT64       RegR21;
    ND_UINT64       RegR22;
    ND_UINT64       RegR23;
    ND_UINT64       RegR24;
    ND_UINT64       RegR25;
    ND_UINT64       RegR26;
    ND_UINT64       RegR27;
    ND_UINT64       RegR28;
    ND_UINT64       RegR29;
    ND_UINT64       RegR30;
    ND_UINT64       RegR31;
    ND_UINT64       RegCr2;
    ND_UINT64       RegFlags;
    ND_UINT64       RegDr7;
    ND_UINT64       RegRip;
    ND_UINT64       RegCr0;
    ND_UINT64       RegCr4;
    ND_UINT64       RegCr3;
    ND_UINT64       RegCr8;
    ND_UINT64       RegIdtBase;
    ND_UINT64       RegIdtLimit;
    ND_UINT64       RegGdtBase;
    ND_UINT64       RegGdtLimit;
    ND_UINT64       FpuRip;
} SHEMU_X86_GPR_REGS, *PSHEMU_X86_GPR_REGS;


//
// Segment register (with its hidden part).
//
typedef struct _SHEMU_X86_SEG
{
    ND_UINT64       Base;
    ND_UINT64       Limit;
    ND_UINT64       Selector;
    ND_UINT64       AccessRights;
} SHEMU_X86_SEG, *PSHEMU_X86_SEG;


//
// The segment registers.
//
typedef struct _SHEMU_X86_SEG_REGS
{
    SHEMU_X86_SEG   Es;
    SHEMU_X86_SEG   Cs;
    SHEMU_X86_SEG   Ss;
    SHEMU_X86_SEG   Ds;
    SHEMU_X86_SEG   Fs;
    SHEMU_X86_SEG   Gs;
} SHEMU_X86_SEG_REGS, *PSHEMU_X86_SEG_REGS;

#endif

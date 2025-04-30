/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef EXP_FTN_H_
#define EXP_FTN_H_

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "expand.h"

/**
   \brief ...
 */
int create_array_ref(int nmex, SPTR sptr, DTYPE dtype, int nsubs, int *subs,
                     int ilix, int sdscilix, int inline_flag, int *pnme);

/**
   \brief ...
 */
int exp_get_sdsc_len(int s, int base, int basenm);

/**
   \brief ...
 */
int get_sdsc_element(SPTR sdsc, int indx, int membase, int membase_nme);

/**
   \brief ...
 */
SPTR frte_func(SPTR (*pf)(const char *), const char *root);

/**
   \brief ...
 */
void exp_ac(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_array(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_bran(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_misc(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_restore_mxcsr(void);

#endif // EXP_FTN_H_

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLOPT_H_
#define LLOPT_H_

#include "gbldefs.h"
#include "llutil.h"

/**
   \brief ...
 */
bool block_branches_to(int bih, int target);

/**
   \brief ...
 */
bool funcHasNoDepChk(void);

/**
   \brief ...
 */
void maybe_undo_recip_div(INSTR_LIST *mul);

/**
   \brief ...
 */
void optimize_block(INSTR_LIST *last_block_instr);

/**
   \brief ...
 */
void redundantLdLdElim(void);

/**
   \brief ...
 */
void widenAddressArith(void);

#endif // LLOPT_H_

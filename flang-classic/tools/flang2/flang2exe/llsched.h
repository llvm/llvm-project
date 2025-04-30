/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLSCHED_H_
#define LLSCHED_H_

#include "gbldefs.h"
#include "llutil.h"

/**

/**
   \brief ...
 */
int enhanced_conflict(int nme1, int nme2);

/**
   \brief ...
 */
void check_circular_dep(INSTR_LIST *istart);

/**
   \brief ...
 */
void sched_block_breadth_first(INSTR_LIST *istart, int level);

/**
   \brief ...
 */
void sched_block(INSTR_LIST *istart, INSTR_LIST *iend);

/**
   \brief ...
 */
void sched_instructions(INSTR_LIST *istart);

#endif // LLSCHED_H_

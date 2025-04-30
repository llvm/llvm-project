/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef EXPSMP_H_
#define EXPSMP_H_

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "expand.h"
#include "llmputil.h"

/**
   \brief ...
 */
int add_mp_barrier2(void);

/**
   \brief ...
 */
int add_mp_lcpu(void);

/**
   \brief ...
 */
int add_mp_ncpus3(void);

/**
   \brief ...
 */
int add_mp_ncpus(void);

/**
   \brief ...
 */
int add_mp_penter(int ispar);

/**
   \brief ...
 */
int add_mp_pexit(void);

/// Insert semaphore wait (enter critical section)
int add_mp_p(SPTR semaphore);

/// Insert semaphore signal (end critical section)
int add_mp_v(SPTR semaphore);

/**
   \brief ...
 */
int get_threadprivate_origsize(int sym);

/**
   \brief ...
 */
SPTR lcpu_temp(SC_KIND sc);

/**
   \brief ...
 */
SPTR llTaskAllocSptr(void);

/**
   \brief ...
 */
int _make_mp_get_threadprivate(int data_ili, int size_ili, int cache_ili);

/**
   \brief ...
 */
SPTR ncpus_temp(SC_KIND sc);

/**
   \brief ...
 */
LLTask *llGetTask(int scope);

/**
   \brief ...
 */
void clear_tplnk(void);

/**
   \brief ...
 */
void exp_mp_func_prologue(bool);

/**
   \brief ...
 */
void exp_smp_fini(void);

/**
   \brief ...
 */
void exp_smp(ILM_OP opc, ILM *ilmp, int curilm);

/**
   \brief ...
 */
void exp_smp_init(void);

/**
   \brief ...
 */
void section_create_endblock(SPTR endLabel);

/// \brief ...
LLTask* llGetTask(int scope);

#endif // EXPSMP_H_

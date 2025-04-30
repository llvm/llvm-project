/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef CGRAPH_H_
#define CGRAPH_H_

/** \file
 * \brief Callgraph
 */

#include "platform.h"

/**
   \brief ...
 */
int cgr_barrier(SPTR func);

/**
   \brief ...
 */
int cgr_modifies(SPTR func_sptr, int object_sptr, int flag);

/**
   \brief ...
 */
int get_func_record(SPTR func_sptr);

/**
   \brief ...
 */
void cgr_add_decl(SPTR sptr);

/**
   \brief ...
 */
void cgr_add(SPTR func);

/**
   \brief ...
 */
void cgr_add_mod(SPTR func_sptr, int object);

/**
   \brief ...
 */
void cgr_attribute(char *attr);

/**
   \brief ...
 */
void cgr_call_func(SPTR func, SPTR caller);

/**
   \brief ...
 */
void cgr_call_node(int func_i, int caller_i);

/**
   \brief ...
 */
void cgr_clean(void);

/**
   \brief ...
 */
void cgr_dmp_modifies(void);

/**
   \brief ...
 */
void cgr_dump_decls(int flag);

/**
   \brief ...
 */
void cgr_end(void);

/**
   \brief ...
 */
void cgr_enter_func(SPTR func);

/**
   \brief ...
 */
void cgr_keep(SPTR func);

/**
   \brief ...
 */
void cgr_set_flag(SPTR func, int v);

/**
   \brief ...
 */
void cgr_set_ilmpos(SPTR func, int type, long pos, int start);

/**
   \brief ...
 */
void reset_CGR(void);

#endif // CGRAPH_H_

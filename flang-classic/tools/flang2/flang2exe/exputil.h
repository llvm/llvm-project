/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef EXPUTIL_H_
#define EXPUTIL_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"

/**
   \brief ...
 */
int access_swtab_base_label(int base_label, int sptr, int flag);

/**
   \brief ...
 */
int access_swtab_case_label(int case_label, int *case_val, int sptr, int flag);

/**
   \brief ...
 */
int add_reg_arg_ili(int arglist, int argili, int nmex, DTYPE dtype);

/**
   \brief ...
 */
int check_ilm(int ilmx, int ilix);

/**
   \brief ...
 */
SPTR find_argasym(int sptr);

/**
   \brief ...
 */
SPTR get_byval_local(int argsptr);

/**
   \brief ...
 */
SPTR mk_argasym(int sptr);

/**
   \brief ...
 */
int mk_impsym(SPTR sptr);

/**
   \brief ...
 */
int mk_swlist(INT n, SWEL *swhdr, int doinit);

/**
   \brief ...
 */
int mk_swtab(INT n, SWEL *swhdr, int deflab, int doinit);

/**
   \brief ...
 */
int mk_swtab_ll(INT n, SWEL *swhdr, int deflab, int doinit);

/**
   \brief ...
 */
SPTR mkfunc_cncall(const char *nmptr);

/**
   \brief ...
 */
SPTR mkfunc_sflags(const char *nmptr, const char *flags);

/**
   \brief ...
 */
void chk_block(int newili);

/**
   \brief ...
 */
void chk_block_suppress_throw(int newili);

/**
   \brief ...
 */
void cr_block(void);

/**
   \brief add IL_LD of the rhs, IL_ST to the lhs, for two integer-type symbols
 */
void exp_add_copy(SPTR lhssptr, SPTR rhssptr);

/**
   \brief ...
 */
void expdumpilms(void);

/**
   \brief ...
 */
void flsh_block(void);

/**
   \brief ...
 */
void mkarglist(int cnt, DTYPE dt);

/**
   \brief ...
 */
void put_funccount(void);

/**
   \brief ...
 */
void wr_block(void);

#endif

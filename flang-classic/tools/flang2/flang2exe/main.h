/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief main exports
 */

#ifndef FORTRAN_COMPILER_MAIN_H_
#define FORTRAN_COMPILER_MAIN_H_

#include "gbldefs.h"
#include "error.h"
#include "lz.h"

// FIXME -- move these prototypes
void schedule(void); // cgmain
void acc_add_global(void); // acclin
// FIXME -- end of misplaced prototypes

/* actual exports from module */

/**
   \brief ...
 */
char *user_string(void);

/**
   \brief ...
 */
int bu_auto_inline(void);

/**
   \brief ...
 */
int export_cgraph_sub(lzhandle *fd);

/**
   \brief ...
 */
int main(int argc, char *argv[]);

/**
   \brief FIXME Comments say this belongs in upper.c
 */
void add_llvm_uplevel_symbol(int sptr);

/**
   \brief ...
 */
void finish(void);

/**
   \brief FIXME Comments say this belongs in upper.c
 */
void fixup_llvm_uplevel_symbol(void);

#endif // FORTRAN_COMPILER_MAIN_H_

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LL_ABI_H_
#define LL_ABI_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "llutil.h"
#include "ll_structure.h"

/**
   \brief ...
 */
unsigned ll_abi_classify_va_arg_dtype(LL_Module* module, DTYPE dtype, 
                                      unsigned *num_gp, unsigned *num_fp);

/**
   \brief ...
 */
void ll_abi_classify_arg_dtype(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg,
                               DTYPE dtype);

/**
   \brief ...
 */
void ll_abi_classify_return_dtype(LL_ABI_Info *abi, DTYPE dtype);

/**
   \brief ...
 */
void ll_abi_compute_call_conv(LL_ABI_Info *abi, int func_sptr, int jsra_flags);

#endif

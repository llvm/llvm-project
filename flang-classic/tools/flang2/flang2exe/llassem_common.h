/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLASSEM_COMMON_H_
#define LLASSEM_COMMON_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ll_structure.h"

/**
   \brief ...
 */
ISZ_T put_skip(ISZ_T old, ISZ_T New, bool is_char);

/**
   \brief ...
 */
char *put_next_member(char *ptr);

/**
   \brief ...
 */
int add_member_for_llvm(SPTR sym, int prev, DTYPE dtype, ISZ_T size);

/**
   \brief ...
 */
DTYPE mk_struct_for_llvm_init(const char *name, int size);

/**
   \brief ...
 */
LL_Value *gen_ptr_offset_val(int offset, LL_Type *ret_type, const char *ptr_nm);

/**
   \brief ...
 */
void add_init_routine(char *initroutine);

/**
   \brief ...
 */
void emit_init(DTYPE tdtype, ISZ_T tconval, ISZ_T *addr, ISZ_T *repeat_cnt,
               ISZ_T loc_base, ISZ_T *i8cnt, int *ptrcnt, char **cptr, bool is_char);

/**
   \brief ...
 */
void init_daz(void);

/**
   \brief ...
 */
void init_flushz(void);

/**
   \brief ...
 */
void init_huge_tlb(void);

/**
   \brief ...
 */
void init_ktrap(void);

/**
   \brief ...
 */
void init_Mcuda_compiled(void);

/**
   \brief ...
 */
void put_addr(SPTR sptr, ISZ_T off, DTYPE dtype);

/**
   \brief ...
 */
void put_i32(int val);

/**
   \brief ...
 */
void put_int4(int val);

/**
   \brief ...
 */
void put_short(int val);

/**
   \brief ...
 */
void put_string_n(const char *p, ISZ_T len, int size);

#endif // LLASSEM_COMMON_H_

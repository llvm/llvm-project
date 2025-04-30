/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef CGMAIN_H_
#define CGMAIN_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ili.h"
#include "cgllvm.h"
#include "ll_structure.h"
#include "llutil.h"

#define FFAST_MATH_IS_PRESENT() XBIT(216, 1)
#define QTMP_SIZE 4 /* tmp[] of union qtmp has 4 elements and 128 bits  */

/**
   \brief ...
 */
bool currsub_is_sret(void);

/**
   \brief ...
 */
bool is_cg_llvm_init(void);

/**
   \brief ...
 */
bool ll_check_struct_return(DTYPE dtype);

/**
   \brief ...
 */
bool strict_match(LL_Type *ty1, LL_Type *ty2);

/**
   \brief ...
 */
const char *dtype_struct_name(DTYPE dtype);

/**
   \brief ...
 */
char *gen_llvm_vconstant(const char *ctype, int sptr, DTYPE tdtype, int flags);

/**
   \brief ...
 */
const char *get_label_name(int sptr);

/**
   \brief ...
 */
const char *match_names(MATCH_Kind match_val);

/**
   \brief ...
 */
const char *char_type(DTYPE dtype, SPTR sptr);

/**
   \brief ...
 */
DTYPE msz_dtype(MSZ msz);

/**
   \brief ...
 */
INSTR_LIST *llvm_info_last_instr(void);

/**
   \brief ...
 */
INSTR_LIST *mk_store_instr(OPERAND *val, OPERAND *addr);

/**
   \brief ...
 */
DTYPE cg_get_type(int n, TY_KIND v1, int v2);

/**
   \brief Find the (virtual) function pointer in a JSRA call
   \param ilix  the first argument of the \c IL_JSRA
 */
SPTR find_pointer_to_function(int ilix);

/**
   \brief ...
 */
int match_llvm_types(LL_Type *ty1, LL_Type *ty2);

/**
   \brief ...
 */
int need_ptr(int sptr, int sc, DTYPE sdtype);

/**
   \brief ...
 */
LL_Type *maybe_fixup_x86_abi_return(LL_Type *sig);

/**
   \brief ...
 */
OPERAND *gen_address_operand(int addr_op, int nme, bool lda,
                             LL_Type *llt_expected, MSZ msz);

/**
   \brief ...
 */
OPERAND *gen_call_as_llvm_instr(int sptr, int ilix);

/**
   \brief ...
 */
OPERAND *gen_call_to_builtin(int ilix, char *fname, OPERAND *params,
                             LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                             LL_InstrName i_name, LL_InstrListFlags MathFlag,
                             unsigned flags);

/**
   \brief ...
 */
OPERAND *gen_llvm_expr(int ilix, LL_Type *expected_type);

/**
   \brief ...
 */
OPERAND *mk_alloca_instr(LL_Type *ptrTy);

/**
   \brief ...
 */
TMPS *gen_extract_insert(LL_InstrName i_name, LL_Type *struct_type, TMPS *tmp,
                         LL_Type *tmp_type, TMPS *tmp2, LL_Type *tmp2_type,
                         int index);

/**
   \brief ...
 */
void build_routine_and_parameter_entries(SPTR func_sptr, LL_ABI_Info *abi,
                                         LL_Module *module);

/**
   \brief ...
 */
void cg_fetch_clen_parampos(SPTR *len, int *param, SPTR sptr);

/**
   \brief ...
 */
bool clen_parent_is_param(SPTR length);

/**
   \brief ...
 */
void cg_llvm_end(void);

/**
   \brief ...
 */
void cg_llvm_fnend(void);

/**
   \brief ...
 */
void cg_llvm_init(void);

/**
   \brief ...
 */
void clear_deletable_flags(int ilix);

/**
   \brief ...
 */
void dump_type_for_debug(LL_Type *ll_type);

/**
   \brief ...
 */
void llvm_ctor_add(const char *name);

/**
   \brief ...
 */
void llvm_ctor_add_with_priority(const char *name, int priority);

/**
   \brief ...
 */
void llvm_dtor_add(const char *name);

/**
   \brief ...
 */
void llvm_dtor_add_with_priority(const char *name, int priority);

/**
   \brief ...
 */
void llvmResetSname(int sptr);

/**
   \brief ...
 */
void llvm_write_ctors(void);

/**
   \brief ...
 */
void print_personality(void);

/**
   \brief ...
 */
void print_tmp_name(TMPS *t);

/**
   \brief ...
 */
void process_formal_arguments(LL_ABI_Info *abi);

/**
   \brief ...
 */
void process_global_lifetime_debug(void);

/**
   \brief ...
 */
void process_sptr(SPTR sptr);

/**
   \brief ...
 */
void reset_expr_id(void);

/**
   \brief ...
 */
void schedule(void);

/**
   \brief ...
 */
void set_llvm_sptr_name(OPERAND *operand);

/**
   \brief ...
 */
void update_external_function_declarations(const char *name, char *decl,
                                           unsigned flags);

/**
   \brief ...
 */
void write_external_function_declarations(int first_time);

bool is_vector_x86_mmx(LL_Type *);

/**
   \brief Process common block symbols, adding debug info for it's variables 
   \param sptr  A symbol
 */
void add_debug_cmnblk_variables(LL_DebugInfo *db, SPTR sptr);

/**
   \brief Check if sptr is the mindum of an array and the array has descriptor 
   \param sptr  A symbol
 */
bool ftn_array_need_debug_info(SPTR sptr);

/**
   \brief Insert <tt>@llvm.dbg.declare</tt> call for debug
   \param mdnode  metadata node
   \param sptr    symbol
   \param llTy    preferred type of \p sptr or \c NULL
 */
void insert_llvm_dbg_declare(LL_MDRef mdnode, SPTR sptr, LL_Type *llTy,
                             OPERAND *exprMDOp, OperandFlag_t opflag);

/**
   \brief Insert <tt>@llvm.dbg.value</tt> call for debug
   \param OPERAND operand
   \param sptr    symbol
   \param llTy    preferred type of \p sptr or \c NULL
 */
void insert_llvm_dbg_value(OPERAND *load, LL_MDRef mdnode, SPTR sptr,
                           LL_Type *type);


/**
   \brief Check if sptr is the midnum of a scalar and scalar has POINTER/ALLOCATABLE attribute
   \param sptr  A symbol
 */
bool pointer_scalar_need_debug_info(SPTR sptr);

int get_parnum(SPTR sptr);

int get_entry_parnum(SPTR sptr);
#endif

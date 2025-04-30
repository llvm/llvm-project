/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Include file for ILI to LLVM translation
 */

#ifndef CGLLVM_H__
#define CGLLVM_H__

#include "llutil.h"

void cprintf(char *s, const char *format, INT *val);

#define SNAME(sptr) (sptrinfo.array.stg_base[sptr])
#define LLTYPE(sptr) (sptrinfo.type_array.stg_base[sptr])
#define LLTYPE_kind(sptr) (sptrinfo.type_array.stg_base[sptr]->kind)
#define LLTYPE_size(sptr) (sptrinfo.type_array.stg_base[sptr]->size)

#define AGGREGATE_STYPE(s) \
  ((s) == ST_STRUCT || (s) == ST_UNION || (s) == ST_ARRAY)
#define AGGREGATE_DTYPE(d) \
  ((DTY(d)) == TY_STRUCT || (DTY(d)) == TY_UNION || (DTY(d)) == TY_ARRAY)
#define COMPLEX_DTYPE(d) ((DTY(d)) == TY_CMPLX || (DTY(d)) == TY_DCMPLX)
#define VECTOR_DTYPE(d) ((DTY(d)) == TY_VECT)

#define LLCCF_NEG                                                         \
  {                                                                       \
    LLCCF_NONE, LLCCF_TRUE, LLCCF_UNE, LLCCF_ULE, LLCCF_ULT, LLCCF_UGE,   \
        LLCCF_UGT, LLCCF_UNE, LLCCF_UNO, LLCCF_ONE, LLCCF_OLE, LLCCF_OLT, \
        LLCCF_OGE, LLCCF_OGT, LLCCF_OEQ, LLCCF_ORD, LLCCF_FALSE           \
  }

/*  functions defined in cgmain.c file:  */

void schedule(void);
void process_global_lifetime_debug(void);
OPERAND *gen_llvm_expr(int ilix, LL_Type *expected_type);
void clear_deletable_flags(int ilix);
INSTR_LIST *llvm_info_last_instr(void);
/* Use MSZ_TO_BYTES to detect presence of MSZ */
#ifdef MSZ_TO_BYTES
OPERAND *gen_address_operand(int, int, bool, LL_Type *, MSZ);
DTYPE msz_dtype(MSZ msz);
#endif
void update_external_function_declarations(const char *, char *, unsigned);
void cg_fetch_clen_parampos(SPTR *len, int *param, SPTR sptr);

extern LL_Module *cpu_llvm_module;
#ifdef OMP_OFFLOAD_LLVM
extern LL_Module *gpu_llvm_module;
#endif
typedef enum STMT_Type {
  STMT_NONE = 0,
  STMT_RET = 1,
  STMT_EXPR = 2,
  STMT_LABEL = 3,
  STMT_BR = 4,
  STMT_ST = 5,
  STMT_CALL = 6,
  STMT_SMOVE = 7,
  STMT_SZERO = 8,
  STMT_DECL = 9,
  STMT_LAST = 10
} STMT_Type;

#define BITOP(i) ((i) >= I_SHL && (i) <= I_XOR)
#define BINOP(i) ((i) >= I_ADD && (i) <= I_FREM)
#define CONVERT(i) ((i) >= I_TRUNC && (i) <= I_BITCAST)
#define PICALL(i) ((i) == I_PICALL)

#define CMP_FLT 0
#define CMP_INT 1
#define CMP_USG 2

typedef enum {
  MATCH_NO = -1,
  MATCH_OK = 0,
  MATCH_MEM = 1,
  MATCH_LAST = 2
} MATCH_Kind;

/* TMP flags */
#define CARRAY_TMP 1

/* external declaration flags */
#define EXF_INTRINSIC 1
#define EXF_STRUCT_RETURN 2
#define EXF_VARARG 4
#define EXF_PURE 8

#define IS_OLD_STYLE_CAND(s) (DEFDG(sptr) || CCSYMG(sptr))

typedef struct{
    STG_DECLARE(array, const char*);
    STG_DECLARE(type_array, LL_Type*);
}SPTRINFO_T;

extern SPTRINFO_T sptrinfo;

void cg_llvm_init(void);
void cg_llvm_end(void);
void cg_llvm_fnend(void);
void llvm_ctor_add(const char *);
void llvm_ctor_add_with_priority(const char *name, int priority);
void llvm_dtor_add(const char *);
void llvm_dtor_add_with_priority(const char *name, int priority);
void llvm_write_ctors(void);

extern FILE *par_file1;
extern FILE *par_file2;

void build_routine_and_parameter_entries(SPTR func_sptr, LL_ABI_Info *abi,
                                         LL_Module *module);
bool strict_match(LL_Type *, LL_Type *);
bool is_cg_llvm_init(void);
void process_formal_arguments(LL_ABI_Info *);
void write_external_function_declarations(int);

OPERAND *mk_alloca_instr(LL_Type *ptrTy);
INSTR_LIST *mk_store_instr(OPERAND *val, OPERAND *addr);

#ifdef TARGET_LLVM_X8664
LL_Type *maybe_fixup_x86_abi_return(LL_Type *sig);
#endif

#include "ll_ftn.h"

#endif /* CGLLVM_H__ */

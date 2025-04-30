/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Main source module to translate into LLVM
 */

#include "cgmain.h"
#include "dtypeutl.h"
#include "ll_ftn.h"
#include "exp_rte.h"
#include "error.h"
#include "machreg.h"
#include "dinit.h"
#include "cg.h"
#include "mach.h"
#include "fih.h"
#include "pd.h"
#include "llutil.h"
#include "lldebug.h"
#include "go.h"
#include "sharedefs.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "llassem.h"
#include "ll_write.h"
#include "expand.h"
#include "outliner.h"
#include "mth.h"
#if defined(SOCPTRG)
#include "soc.h"
#endif
#include "llvm/Config/llvm-config.h"
#include "mwd.h"
#include "ccffinfo.h"
#include "main.h"
#include "symfun.h"
#include "ilidir.h"
#include "fdirect.h"

#ifdef OMP_OFFLOAD_LLVM
#include "ompaccel.h"
#define ISNVVMCODEGEN gbl.ompaccel_isdevice
#else
#define ISNVVMCODEGEN false
#endif

#include "upper.h"


typedef enum SincosOptimizationFlags {
  /* used only for sincos() optimization */
  SINCOS_SIN = 1,
  SINCOS_COS = 2,
  SINCOS_EXTRACT = 4,
  SINCOS_MASK = SINCOS_SIN | SINCOS_COS
} SincosOptimizationFlags;

/* clang-format off */

static const char *const llvm_instr_names[I_LAST] = {
  "none", "ret", "br", "switch", "invoke", "unwind", "unreachable",
  "add nsw",
  "fadd",
  "sub nsw",
  "fsub",
  "mul nsw",
  "fmul", "udiv", "sdiv", "fdiv", "urem", "srem", "frem",
  "shl nsw",
  "lshr", "ashr", "and", "or", "xor", "extractelement", "insertelement",
  "shufflevector", "extractvalue", "insertvalue", "malloc", "free", "alloca",
  "load", "store", "getelementptr", "trunc", "zext", "sext", "fptrunc",
  "fpext", "fptoui", "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr",
  "bitcast", "icmp", "fcmp", "vicmp", "vfcmp", "phi", "select", "call",
  "va_arg", "=", "landingpad", "resume", "cleanup", "catch", "fence",
  "atomicrmw", "cmpxchg", "fence", "call", "indirectbr", "filter"
};

static const char *const stmt_names[STMT_LAST] = {
    "STMT_NONE", "STMT_RET",  "STMT_EXPR",  "STMT_LABEL", "STMT_BR",
    "STMT_ST",   "STMT_CALL", "STMT_SMOVE", "STMT_SZERO", "STMT_DECL"
};

/* clang-format on */

const int MEM_EXTRA = 500;

static void insert_entry_label(int);
static void insert_jump_entry_instr(int);
static void store_return_value_for_entry(OPERAND *, int);

static unsigned addressElementSize;

#define ENTOCL_PREFIX "__pgocl_"

#define HOMEFORDEBUG(sptr) (XBIT(183, 8) && SCG(sptr) == SC_DUMMY)

#define ENABLE_CSE_OPT ((flg.opt >= 1) && !XBIT(183, 0x20) && !killCSE)
#define ENABLE_BLK_OPT ((flg.opt >= 2) && XBIT(183, 0x400))
#define ENABLE_ENHANCED_CSE_OPT (flg.opt >= 2 && !XBIT(183, 0x200000))

#ifdef TARGET_LLVM_ARM
/* TO DO: to be revisited, for now we assume we always target NEON unit */
#define NEON_ENABLED 0 /* TEST_FEATURE(FEATURE_NEON) */
#endif

/* debug switches:
   -Mq,11,16 dump ili right before ILI -> LLVM translation
   -Mq,12,16 provides dinit info, ilt trace, and some basic preprocessing info
   -Mq,12,32 provides complete flow debug info through the LLVM routines
*/

#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)

#ifndef TEST_FEATURE
#define TEST_FEATURE(M) 0
#endif

#define HAS_AVX TEST_FEATURE(FEATURE_AVX)
#endif

#define DBGTRACEIN(str) DBGXTRACEIN(DBGBIT(12, 0x20), 1, str)
#define DBGTRACEIN1(str, p1) DBGXTRACEIN1(DBGBIT(12, 0x20), 1, str, p1)
#define DBGTRACEIN2(str, p1, p2) DBGXTRACEIN2(DBGBIT(12, 0x20), 1, str, p1, p2)
#define DBGTRACEIN3(str, p1, p2, p3) \
  DBGXTRACEIN3(DBGBIT(12, 0x20), 1, str, p1, p2, p3)
#define DBGTRACEIN4(str, p1, p2, p3, p4) \
  DBGXTRACEIN4(DBGBIT(12, 0x20), 1, str, p1, p2, p3, p4)
#define DBGTRACEIN7(str, p1, p2, p3, p4, p5, p6, p7) \
  DBGXTRACEIN7(DBGBIT(12, 0x20), 1, str, p1, p2, p3, p4, p5, p6, p7)

#define DBGTRACEOUT(str) DBGXTRACEOUT(DBGBIT(12, 0x20), 1, str)
#define DBGTRACEOUT1(str, p1) DBGXTRACEOUT1(DBGBIT(12, 0x20), 1, str, p1)
#define DBGTRACEOUT2(str, p1, p2) \
  DBGXTRACEOUT2(DBGBIT(12, 0x20), 1, str, p1, p2)
#define DBGTRACEOUT3(str, p1, p2, p3) \
  DBGXTRACEOUT3(DBGBIT(12, 0x20), 1, str, p1, p2, p3)
#define DBGTRACEOUT4(str, p1, p2, p3, p4) \
  DBGXTRACEOUT4(DBGBIT(12, 0x20), 1, str, p1, p2, p3, p4)

#define DBGDUMPLLTYPE(str, llt) DBGXDUMPLLTYPE(DBGBIT(12, 0x20), 1, str, llt)

#define DBGTRACE(str) DBGXTRACE(DBGBIT(12, 0x20), 1, str)
#define DBGTRACE1(str, p1) DBGXTRACE1(DBGBIT(12, 0x20), 1, str, p1)
#define DBGTRACE2(str, p1, p2) DBGXTRACE2(DBGBIT(12, 0x20), 1, str, p1, p2)
#define DBGTRACE3(str, p1, p2, p3) \
  DBGXTRACE3(DBGBIT(12, 0x20), 1, str, p1, p2, p3)
#define DBGTRACE4(str, p1, p2, p3, p4) \
  DBGXTRACE4(DBGBIT(12, 0x20), 1, str, p1, p2, p3, p4)
#define DBGTRACE5(str, p1, p2, p3, p4, p5) \
  DBGXTRACE5(DBGBIT(12, 0x20), 1, str, p1, p2, p3, p4, p5)

#if defined(TARGET_LLVM_X8664)
#define USE_FMA_EXTENSIONS 1
#endif

/* Exported variables */

SPTRINFO_T sptrinfo;

/* This should live in llvm_info, but we need to access this module from other
 * translation units temporarily */
LL_Module *current_module = NULL;
LL_Module *cpu_llvm_module = NULL;
#ifdef OMP_OFFLOAD_LLVM
LL_Module *gpu_llvm_module = NULL;
#endif


/* File static variables */

static struct {
  unsigned _new_ebb : 1;
  unsigned _killCSE : 1;
  unsigned _init_once : 1;
  unsigned _cpp_init_once : 1;
  unsigned _ftn_init_once : 1;
  unsigned _float_jmp : 1;
  unsigned _fcmp_negate : 1;
  unsigned _last_stmt_is_branch : 1;
  unsigned _rw_acc_grp_check : 1;
} CGMain;

#define new_ebb (CGMain._new_ebb)
#define killCSE (CGMain._killCSE)
#define init_once (CGMain._init_once)
#define cpp_init_once (CGMain._cpp_init_once)
#define ftn_init_once (CGMain._ftn_init_once)
#define float_jmp (CGMain._float_jmp)
#define fcmp_negate (CGMain._fcmp_negate)
#define last_stmt_is_branch (CGMain._last_stmt_is_branch)
#define rw_access_group (CGMain._rw_acc_grp_check)

static int funcId;
static int fnegcc[17] = LLCCF_NEG;
static int expr_id;
static int entry_bih = 0;
static int routine_count;
static STMT_Type curr_stmt_type;
static hashmap_t sincos_map;
static hashmap_t sincos_imap;
static LL_MDRef cached_loop_id_md = ll_get_md_null();
static bool cached_loop_id_md_has_vectorize = false;
static LL_MDRef cached_vectorize_enable_metadata = ll_get_md_null();
static LL_MDRef cached_vectorize_disable_metadata = ll_get_md_null();
static LL_MDRef cached_unroll_enable_metadata = ll_get_md_null();
static LL_MDRef cached_unroll_disable_metadata = ll_get_md_null();
static LL_MDRef cached_access_group_metadata = ll_get_md_null();

static bool CG_cpu_compile = false;

static struct ret_tag {
  /** If ILI uses a hidden pointer argument to return a struct, this is it. */
  SPTR sret_sptr;
  bool emit_sret; /**< Should we emit an sret argument in LLVM IR? */
} ret_info;

static struct llvm_tag {
  GBL_LIST *last_global;
  INSTR_LIST *last_instr;
  INSTR_LIST *curr_instr;
  LL_ABI_Info *abi_info;

  /** The LLVM function currently being built. */
  LL_Function *curr_func;

  /** LLVM representation of the current function's return type.
      See comment before analyze_ret_info(). */
  LL_Type *return_ll_type;

  char *buf;

  /** Map sptr -> OPERAND* for those formal function arguments that are saved
      to a local variable in the prolog by process_formal_arguments(). The
      OPERAND* can be used to access the actual LLVM function argument while
      the normal SNAME(sptr) refers to the local variable created by
      process_formal_arguments(). */
  hashmap_t homed_args;

  /** Map name -> func_type for intrinsics that have already been declared by
      get_intrinsic(). */
  hashmap_t declared_intrinsics;

  int last_sym_avail;
  int last_dtype_avail;
  int buf_idx;
  int buf_sz;

  DTYPE curr_ret_dtype;

  unsigned no_debug_info : 1; /* set to emit engineering diagnostics */
} llvm_info;

static GBL_LIST *Globals;
static GBL_LIST *recorded_Globals;
static INSTR_LIST *Instructions;
static CSED_ITEM *csedList;

typedef struct TmpsMap {
  unsigned size;
  TMPS **map;
} TmpsMap;
static TmpsMap tempsMap;

/** \brief list for tracking calls with complex result types */
typedef struct ComplexResultList_t {
  int *list;
  unsigned size;
  unsigned entries;
} ComplexResultList_t;
static ComplexResultList_t complexResultList;
LL_ABI_Info *entry_abi;

/* ---  static prototypes (exported prototypes belong in cgllvm.h) --- */

static void fma_rewrite(INSTR_LIST *isns);
static void undo_recip_div(INSTR_LIST *isns);
static const char *set_local_sname(int sptr, const char *name);
static const char *get_llvm_sname(SPTR sptr);
static int is_special_return_symbol(int sptr);
static bool cgmain_init_call(int);
static OPERAND *gen_call_llvm_intrinsic(const char *, OPERAND *, LL_Type *,
                                        INSTR_LIST *, LL_InstrName);
static OPERAND *gen_call_llvm_fm_intrinsic(const char *, OPERAND *, LL_Type *,
                                           INSTR_LIST *, LL_InstrName);
#ifdef FLANG_GEN_LLVM_ATOMIC_INTRINSICS
static OPERAND *gen_llvm_atomicrmw_instruction(int, int, OPERAND *, DTYPE);
#endif
static void gen_llvm_fence_instruction(int ilix);
static const char *get_atomicrmw_opname(LL_InstrListFlags);
static const char *get_atomic_memory_order_name(int);
static void insert_llvm_memcpy(int, int, OPERAND *, OPERAND *, int, int, int);
static void insert_llvm_memset(int, int, OPERAND *, int, int, int, int);
static void insert_llvm_prefetch(int ilix, OPERAND *dest_op);
static SPTR get_call_sptr(int);
static LL_Type *make_function_type_from_args(LL_Type *return_type,
                                             OPERAND *first_arg_op,
                                             bool is_varargs);
static MATCH_Kind match_types(LL_Type *, LL_Type *);
#ifdef FLANG2_CGMAIN_UNUSED
static int decimal_value_from_oct(int, int, int);
#endif
static char *vect_llvm_intrinsic_name(int);
static void build_unused_global_define_from_params(void);
static void print_function_signature(int func_sptr, const char *fn_name,
                                     LL_ABI_Info *abi, bool print_arg_names);
static void write_global_and_static_defines(void);
static char *gen_constant(SPTR, DTYPE, INT, INT, int);
#ifdef FLANG2_CGMAIN_UNUSED
static char *process_string(char *, int, int);
#endif
static void make_stmt(STMT_Type, int, bool, SPTR next_bih_label, int ilt);
static INSTR_LIST *make_instr(LL_InstrName);
static INSTR_LIST *gen_instr(LL_InstrName, TMPS *, LL_Type *, OPERAND *);
static OPERAND *ad_csed_instr(LL_InstrName, int, LL_Type *, OPERAND *,
                              LL_InstrListFlags, bool);
static void ad_instr(int, INSTR_LIST *);
static OPERAND *gen_call_expr(int ilix, DTYPE ret_dtype, INSTR_LIST *call_instr,
                              int call_sptr);
static INSTR_LIST *gen_switch(int ilix);
static OPERAND *gen_unary_expr(int, LL_InstrName);
static OPERAND *gen_binary_vexpr(int, int, int, int);
static OPERAND *gen_binary_expr(int, int);
static OPERAND *gen_va_arg(int);
#ifdef FLANG2_CGMAIN_UNUSED
/* FIXME: gen_va_arg is used, but gen_va_start/gen_va_end are not. */
static OPERAND *gen_va_start(int);
static OPERAND *gen_va_end(int);
#endif
#ifdef TARGET_POWER
static OPERAND *gen_gep_index(OPERAND *, LL_Type *, int);
#endif
static OPERAND *gen_insert_value(OPERAND *aggr, OPERAND *elem, unsigned index);
static char *gen_vconstant(const char *, int, DTYPE, int);
static LL_Type *make_vtype(DTYPE, int);
static LL_Type *make_type_from_msz(MSZ);
static LL_Type *make_type_from_msz_with_addrspace(MSZ, int);
static LL_Type *make_type_from_opc(ILI_OP);
static bool add_to_cselist(int ilix);
static void clear_csed_list(void);
static void remove_from_csed_list(int);
static void set_csed_operand(OPERAND **, OPERAND *);
static OPERAND **get_csed_operand(int ilix);
static void build_csed_list(int);
static OPERAND *gen_base_addr_operand(int, LL_Type *);
static OPERAND *gen_optext_comp_operand(OPERAND *, ILI_OP, int, int, int, int,
                                        LL_InstrName, int, int);
static OPERAND *gen_sptr(SPTR sptr);
static OPERAND *gen_load(OPERAND *addr, LL_Type *type, LL_InstrListFlags flags);
static void make_store(OPERAND *, OPERAND *, LL_InstrListFlags);
static OPERAND *make_load(int, OPERAND *, LL_Type *, MSZ, unsigned flags);
static OPERAND *convert_operand(OPERAND *convert_op, LL_Type *rslt_type,
                                LL_InstrName convert_instruction);
static OPERAND *convert_float_size(OPERAND *, LL_Type *);
#ifdef FLANG2_CGMAIN_UNUSED
static int follow_sptr_hashlk(SPTR sptr);
static DTYPE follow_ptr_dtype(DTYPE);
#endif
static bool same_op(OPERAND *, OPERAND *);
static void write_instructions(LL_Module *);
static LLIntegerConditionCodes convert_to_llvm_intcc(CC_RELATION cc);
static LLIntegerConditionCodes convert_to_llvm_uintcc(CC_RELATION cc);
static LLFloatingPointConditionCodes convert_to_llvm_fltcc(CC_RELATION cc);
static int convert_to_llvm_cc(CC_RELATION cc, int cc_type);
static OPERAND *get_intrinsic(const char *name, LL_Type *func_type,
                              unsigned flags);
static OPERAND *get_intrinsic_call_ops(const char *name, LL_Type *return_type,
                                       OPERAND *args, unsigned flags);
static OPERAND *sign_extend_int(OPERAND *op, unsigned result_bits);
static OPERAND *zero_extend_int(OPERAND *op, unsigned result_bits);
static bool repeats_in_binary(union xx_u);
static bool zerojump(ILI_OP);
static bool exprjump(ILI_OP);
static OPERAND *gen_resized_vect(OPERAND *, int, int);
static bool is_blockaddr_store(int, int, int);
static SPTR process_blockaddr_sptr(int, int);
#if defined(TARGET_LLVM_X8664)
static bool is_256_or_512_bit_math_intrinsic(int);
#endif
#ifdef FLANG2_CGMAIN_UNUSED
static bool have_masked_intrinsic(int);
#endif
static OPERAND *make_bitcast(OPERAND *, LL_Type *);
static void update_llvm_sym_arrays(void);
static bool need_debug_info(SPTR sptr);
static OPERAND *convert_int_size(int, OPERAND *, LL_Type *);
static OPERAND *convert_int_to_ptr(OPERAND *, LL_Type *);
static OPERAND *gen_call_vminmax_intrinsic(int ilix, OPERAND *op1,
                                           OPERAND *op2);
static OPERAND *gen_extract_value_ll(OPERAND *, LL_Type *, LL_Type *, int);
static OPERAND *gen_extract_value(OPERAND *, DTYPE, DTYPE, int);
static OPERAND *gen_vect_compare_operand(int);

#if defined(TARGET_LLVM_POWER)
static OPERAND *gen_call_vminmax_power_intrinsic(int ilix, OPERAND *op1,
                                                 OPERAND *op2);
#endif
#if defined(TARGET_LLVM_ARM) && NEON_ENABLED
static OPERAND *gen_call_vminmax_neon_intrinsic(int ilix, OPERAND *op1,
                                                OPERAND *op2);
#endif
static INSTR_LIST *remove_instr(INSTR_LIST *instr, bool update_usect_only);

#ifdef __cplusplus
inline static CC_RELATION ILI_ccOPND(int i, int j) {
  CC_RELATION result = static_cast<CC_RELATION>(ILI_OPND(i, j));
  assert((result <= CC_NOTGT) && (result >= -CC_NOTGT), "out of range", result,
         ERR_Fatal);
  return result;
}
#else
#define ILI_ccOPND ILI_OPND
#endif

static void
consTempMap(unsigned size)
{
  if (tempsMap.map) {
    free(tempsMap.map);
  }
  tempsMap.size = size;
  tempsMap.map = (TMPS **)calloc(sizeof(struct TmpsMap), size);
}

static void
gcTempMap(void)
{
  free(tempsMap.map);
  tempsMap.size = 0;
  tempsMap.map = NULL;
}

static TMPS *
getTempMap(unsigned ilix)
{
  return (ilix < tempsMap.size) ? tempsMap.map[ilix] : NULL;
}

static void
setTempMap(unsigned ilix, OPERAND *op)
{
  if (ilix < tempsMap.size) {
    tempsMap.map[ilix] = op->tmps;
  }
}

/* Convert the name of a built-in function to the LLVM intrinsic that
   implements it.  This only works when the built-in function and the LLVM
   intrinsic have the same signature, so no manipulation of the arguments or
   return value is necessary.  (If the list of names gets much longer than two,
   then a table driven approach should be used.  If the list gets really long,
   then a hash table should be considered.) */
static const char *
map_to_llvm_name(const char *function_name)
{
  if (function_name == NULL) {
    return NULL;
  }
  if (strcmp(function_name, "__builtin_return_address") == 0) {
    return "llvm.returnaddress";
  }
  if (strcmp(function_name, "__builtin_frame_address") == 0) {
    return "llvm.frameaddress";
  }
  return function_name;
}

void
set_llvm_sptr_name(OPERAND *operand)
{
  const int sptr = operand->val.sptr;
  operand->string = SNAME(sptr);
}

const char *
get_label_name(int sptr)
{
  const char *nm = SNAME(sptr);
  if (*nm == '@')
    nm++;
  return nm;
}

const char *
get_llvm_sname(SPTR sptr)
{
  const char *p = SNAME(sptr);
  if (p == NULL) {
    process_sptr(sptr);
    p = SNAME(sptr);
  }
  if (p == NULL) {
    p = SYMNAME(sptr);
    if (p == NULL)
      return "";
    p = map_to_llvm_name(p);
    char *buf = (char *)getitem(LLVM_LONGTERM_AREA, strlen(p) + 1);
    SNAME(sptr) = strcpy(buf, p);
    return SNAME(sptr);
  }
  if (*p == '@')
    p++;
  return p;
}

DTYPE
cg_get_type(int n, TY_KIND v1, int v2)
{
  DTYPE ret_dtype = get_type(n, v1, v2);
  update_llvm_sym_arrays();
  return ret_dtype;
}

INSTR_LIST *
llvm_info_last_instr(void)
{
  return llvm_info.last_instr;
}

/**
   \brief Check if the TY_STRUCT fits in registers per the ABI

   This is a backdoor for the expander to access the LLVM bridge.
 */
bool
ll_check_struct_return(DTYPE dtype)
{
  LL_ABI_Info *abi;
  TY_KIND ty = DTY(dtype);

  DEBUG_ASSERT((ty == TY_STRUCT) || (ty == TY_UNION) || DT_ISCMPLX(dtype),
               "must be aggregate type");
  abi = ll_abi_for_func_sptr(cpu_llvm_module, gbl.currsub, DT_NONE);
  ll_abi_classify_return_dtype(abi, dtype);
  return !LL_ABI_HAS_SRET(abi);
}

/*
 * Return value handling.
 *
 * Functions that return a struct or other aggregrate that doesn't fit in
 * registers may require the caller to pass in a return value pointer as a
 * hidden first argument. The callee wil store the returned struct to the
 * pointer.
 *
 * In LLVM IR, this is represented by an sret attribute on the hidden pointer
 * argument:
 *
 *   %struct.S = type { [10 x i32] }
 *
 *   define void @f(ptr noalias sret(%struct.S) %agg.result) ...
 *
 * Some structs can be returned in registers, depending on ABI-specific rules.
 * For example, x86-64 can return a struct {long x, y; } struct in registers
 * %rax and %rdx:
 *
 *   define { i64, i64 } @f() ...
 *
 * When targeting LLVM, ILI for a function returning a struct looks like the
 * caller passed in an sret pointer, no matter how the ABI specifies the struct
 * should be returned. This simplifies the ILI, and we will translate here if
 * the struct can actually be returned in registers for the current ABI.
 */

/*
 * Analyze the return value of the current function and determine how it should
 * be translated to LLVM IR.
 *
 * If the LLVM IR representation uses an sret argument, set:
 *
 *   ret_info.emit_sret = true.
 *   ret_info.sret_sptr = symbol table entry for sret argument.
 *   llvm_info.return_ll_type = void.
 *
 * If the ILI representation uses a hidden struct argument, but the LLVM IR
 * returns in registers, set:
 *
 *   ret_info.emit_sret = false.
 *   ret_info.sret_sptr = symbol table entry for sret argument.
 *   llvm_info.return_ll_type = LLVM function return type.
 *
 * Otherwise when both ILI and LLVM IR return in a register, set:
 *
 *   ret_info.emit_sret = false.
 *   ret_info.sret_sptr = 0.
 *   llvm_info.return_ll_type = LLVM function return type.
 */
static void
analyze_ret_info(SPTR func_sptr)
{
  DTYPE return_dtype;

#if defined(ENTRYG)
  /* Get the symbol table entry for the function's return value. If ILI is
   * using a hidden sret argument, this will be it.
   *
   * Fortran complex return values are handled differently, and don't get an
   * 'sret' attribute.
   */
  ret_info.sret_sptr = aux.entry_base[ENTRYG(func_sptr)].ret_var;
#endif

  if (gbl.arets) {
    return_dtype = DT_INT;
  } else {
    /* get return type from ag_table or ll_abi table */
    return_dtype = get_return_type(func_sptr);
    /* do not set the sret_sptr for 'bind(c)' complex functions in the presence
       of multiple entries */
    if (!has_multiple_entries(gbl.currsub))
      if ((DT_ISCMPLX(return_dtype) && (CFUNCG(func_sptr) || CMPLXFUNC_C)) ||
          LL_ABI_HAS_SRET(llvm_info.abi_info)) {
        ret_info.sret_sptr = FVALG(func_sptr);
      }
  }

  DBGTRACE2("sret_sptr=%d, return_dtype=%d", ret_info.sret_sptr, return_dtype);

  llvm_info.return_ll_type = make_lltype_from_dtype(return_dtype);

  ret_info.emit_sret = LL_ABI_HAS_SRET(llvm_info.abi_info);

  if (ret_info.emit_sret) {
    assert(ret_info.sret_sptr, "ILI should use a ret_var", func_sptr,
           ERR_Fatal);
    llvm_info.return_ll_type = make_void_lltype();
  } else if (llvm_info.return_ll_type != llvm_info.abi_info->arg[0].type) {
    /* Make sure the return type matches the ABI type. */
    llvm_info.return_ll_type =
        make_lltype_from_abi_arg(&llvm_info.abi_info->arg[0]);
  }

  /* Process sret_sptr *after* setting up ret_info. Some decisions in
   * process_auto_sptr() depends on ret_info. */
  if (ret_info.sret_sptr)
    process_sptr(ret_info.sret_sptr);
}

/**
   \brief Generate a return operand when ILI didn't provide a return value.

   LLVM requires a return instruction, even if it is only a "return undef".
   Also handle the case where we have a special return value symbol but want to
   return a value in registers.
 */
INLINE static OPERAND *
gen_return_operand(int ilix)
{
  LL_Type *rtype = llvm_info.return_ll_type;
  DTYPE dtype = DTYPEG(gbl.currsub);
  TY_KIND dty = DTY(dtype);

  if (has_multiple_entries(gbl.currsub) && (rtype->data_type == LL_VOID) &&
      (dty != TY_NONE) && (dty != TY_CHAR) && (dty != TY_NCHAR)
#if !defined(TARGET_LLVM_POWER)
      && (dty != TY_CMPLX) && (dty != TY_DCMPLX)
#ifdef TARGET_SUPPORTS_QUADFP
      && (dty != TY_QCMPLX)
#endif
#endif
  ) {
    LL_Type *rtype = make_lltype_from_dtype(dtype);
    LL_Type *pTy = make_ptr_lltype(rtype);
    const SPTR rv_sptr = FVALG(ILI_OPND(ilix, 1));
    OPERAND *bcast = make_bitcast(gen_sptr(rv_sptr), pTy);
    LL_InstrListFlags flgs = ldst_instr_flags_from_dtype(DTYPEG(rv_sptr));
    return gen_load(bcast, rtype, flgs);
  }
  if (rtype->data_type == LL_VOID) {
    OPERAND *op = make_operand();
    op->ll_type = rtype;
    return op;
  }

  /* ret_sptr is the return value symbol which we want to return in registers.
   *
   * Coerce it to the correct type by bitcasting the pointer and loading
   * the return value type from the stack slot.
   */
  if (ret_info.sret_sptr) {
    /* Bitcast sret_sptr to a pointer to the return type. */
    LL_Type *prtype = make_ptr_lltype(rtype);
    OPERAND *sret_as_prtype =
        make_bitcast(gen_sptr(ret_info.sret_sptr), prtype);
    /* Load sret_sptr as the return type and return that. */
    return gen_load(sret_as_prtype, rtype,
                    ldst_instr_flags_from_dtype(DTYPEG(ret_info.sret_sptr)));
  }
  if (CFUNCG(gbl.currsub) &&
      bindC_function_return_struct_in_registers(gbl.currsub)) {
    /* returning a small struct */
    LL_Type *pTy = make_ptr_lltype(rtype);
    const SPTR rv_sptr = FVALG(ILI_OPND(ilix, 1));
    OPERAND *bcast = make_bitcast(gen_sptr(rv_sptr), pTy);
    LL_InstrListFlags flgs = ldst_instr_flags_from_dtype(DTYPEG(rv_sptr));
    return gen_load(bcast, rtype, flgs);
  }

  (void)ilix; // just to disable any unused warnings

  /* No return value symbol available, so just return undef. */
  return make_undef_op(rtype);
}

INLINE static bool
on_prescan_complex_list(int ilix)
{
  for (unsigned i = 0; i < complexResultList.entries; ++i)
    if (complexResultList.list[i] == ilix)
      return true;
  return false;
}

#ifdef LONG_DOUBLE_FLOAT128
static void
add_prescan_complex_list(int ilix)
{
  if (on_prescan_complex_list(ilix))
    return;
  if (complexResultList.size == complexResultList.entries) {
    int size;
    if (complexResultList.size == 0)
      complexResultList.size = 8;
    else
      complexResultList.size = complexResultList.size * 2;
    size = complexResultList.size * sizeof(int);
    complexResultList.list = (int *)realloc(complexResultList.list, size);
  }
  complexResultList.list[complexResultList.entries++] = ilix;
}
#endif

INLINE static void
clear_prescan_complex_list(void)
{
  if (complexResultList.list) {
    free(complexResultList.list);
    complexResultList.list = NULL;
    complexResultList.size = complexResultList.entries = 0;
  }
}

INLINE static void
fix_nodepchk_flag(int bih)
{
  if (block_branches_to(bih, bih))
    return;
  if (block_branches_to(BIH_NEXT(bih), bih)) {
    BIH_NODEPCHK(BIH_NEXT(bih)) = true;
    BIH_NODEPCHK2(BIH_NEXT(bih)) = true;
    return;
  }
  if (!BIH_NODEPCHK2(bih)) {
    BIH_NODEPCHK(bih) = false;
  }
}

INLINE static void
clear_cached_loop_id_md()
{
  cached_loop_id_md = ll_get_md_null();
  cached_loop_id_md_has_vectorize = false;
}

// Construct a new loop ID node or return previously cached one.
// It looks like: !0 = !{!0}
INLINE static LL_MDRef
cons_loop_id_md()
{
  if (LL_MDREF_IS_NULL(cached_loop_id_md)) {
    cached_loop_id_md = ll_create_flexible_md_node(cpu_llvm_module);
    ll_extend_md_node(cpu_llvm_module, cached_loop_id_md, cached_loop_id_md);
  }
  return cached_loop_id_md;
}

INLINE static void
mark_rw_access_grp(int bih)
{
  rw_access_group = 1;
  if (!BIH_NODEPCHK2(bih))
    clear_cached_loop_id_md();
}

INLINE static void
clear_rw_access_grp(void)
{
  rw_access_group = 0;
  clear_cached_loop_id_md();
}

void
print_personality(void)
{
  print_token(
      " personality ptr @__gxx_personality_v0");
}

/**
   \brief Clear \c SNAME for \p sptr
   \param sptr  the symbol
   Used by auto parallel in C when the optimizer uses the same compiler
   generated variable across loops
 */
void
llvmResetSname(int sptr)
{
  SNAME(sptr) = NULL;
}

bool
currsub_is_sret(void)
{
  return LL_ABI_HAS_SRET(llvm_info.abi_info);
}

INLINE static INSTR_LIST *
find_last_executable(INSTR_LIST *i)
{
  INSTR_LIST *cursor = i;
  for (;;) {
    if (i->i_name != I_NONE)
      return i;
    i = i->prev;
    if ((i == NULL) || (i == cursor))
      return NULL;
  }
}

/* --------------------------------------------------------- */

static int
processOutlinedByConcur(int bih)
{
  int eili, bili, bilt, eilt, gtid;
  int bbih, ebih, bopc, eopc;
  int bconcur = 0;
  SPTR display = SPTR_NULL;
  static int workingBih = 0;

  if (workingBih == 0)
    workingBih = BIH_NEXT(workingBih);

  /* does not support nested auto parallel */
  for (bbih = workingBih; bbih; bbih = BIH_NEXT(bbih)) {

    /* if IL_BCONCUR is always be the first - we can just check the first ilt */
    for (bilt = BIH_ILTFIRST(bbih); bilt; bilt = ILT_NEXT(bilt)) {
      bili = ILT_ILIP(bilt);
      bopc = ILI_OPC(bili);

      if (bopc == IL_BCONCUR) {
        ++bconcur;

        GBL_CURRFUNC = ILI_SymOPND(bili, 1);
        display = llvmAddConcurEntryBlk(bbih);

        /* if IL_ECONCUR is always be the first - we can just check the first
         * ilt */
        for (ebih = bbih; ebih; ebih = BIH_NEXT(ebih)) {
          for (eilt = BIH_ILTFIRST(ebih); eilt; eilt = ILT_NEXT(eilt)) {
            eili = ILT_ILIP(eilt);
            eopc = ILI_OPC(eili);
            if (eopc == IL_ECONCUR) {
              --bconcur;
              llvmAddConcurExitBlk(ebih);
              display = SPTR_NULL;
              workingBih = BIH_NEXT(ebih); /* bih after IL_ECONCUR block */
              BIH_NEXT(ebih) = 0;

              /* Reset SNAME field for gtid which needs to be done for C/C++.
               * gtid can be have SC_LOCAL and ENCLFUNC of the host rotine and
               * the code generator will not process if SNAME already exist.  We
               * want this variable declared in the Mconcur outlined routine.
               */
              gtid = ll_get_gtid();
              if (gtid)
                llvmResetSname(gtid);
              ll_save_gtid_val(0);

#if DEBUG
              if (DBGBIT(10, 4)) {
                dump_blocks(gbl.dbgfil, gbl.entbih,
                            "***** BIHs for Function \"%s\" *****", 0);
              }

#endif
              return ebih;
            }
            if ((eopc == IL_BCONCUR) && (bbih != ebih))
              return 0; /* error happens */
          }
        }
      }
    }
  }
  workingBih = 0; /* no more concur */
  return 0;
}

/*
 * Inspect all variables in the symbol table and change their storage
 * class from SC_LOCAL to SC_STATIC if appropriate.  The CG needs to
 * know the final storage class of variables before it begins code
 * generation.
 */
static void
assign_fortran_storage_classes(void)
{
  int sptr;

  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_PLIST:
    case ST_VAR:
    case ST_ARRAY:
    case ST_STRUCT:
    case ST_UNION:
      if (REFG(sptr))
        break;

      if (SCG(sptr) != SC_LOCAL && SCG(sptr) != SC_NONE)
        break;

      if (DINITG(sptr) || SAVEG(sptr)) {
        SCP(sptr, SC_STATIC);
        if ((flg.smp || (XBIT(34, 0x200) || gbl.usekmpc)) && PARREFG(sptr))
          PARREFP(sptr, 0);
      } else if (STYPEG(sptr) != ST_VAR && !flg.recursive &&
                 (!CCSYMG(sptr) || INLNG(sptr))) {
        SCP(sptr, SC_STATIC);
        if ((flg.smp || (XBIT(34, 0x200) || gbl.usekmpc)) && PARREFG(sptr))
          PARREFP(sptr, 0);
      }
      break;
    default:
      break;
    }
  }
} /* end assign_fortran_storage_classes() */

/*
 * when vector always pragma is specified, "llvm.loop.parallel_accesses" metadata has
 * to be generated along with "llvm.access.group" for each load/store instructions.
 */
INLINE static LL_MDRef
cons_loop_parallel_accesses_metadata(void)
{
  LL_MDRef lvcomp[2];

  lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.parallel_accesses");
  lvcomp[1] = cached_access_group_metadata;
  return ll_get_md_node(cpu_llvm_module, LL_PlainMDNode, lvcomp, 2);
} // cons_loop_parallel_accesses_metadata

/**
   \brief Construct exactly one cached instance of !{!"llvm.loop.vectorize.enable", 0}.
 */
INLINE static LL_MDRef
cons_novectorize_metadata(void)
{
  LL_MDRef lvcomp[2];
  if (LL_MDREF_IS_NULL(cached_vectorize_disable_metadata)) {
    lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.vectorize.enable");
    lvcomp[1] = ll_get_md_i1(0);
    cached_vectorize_disable_metadata = ll_get_md_node(cpu_llvm_module,
        LL_PlainMDNode, lvcomp, 2);
  }
  return cached_vectorize_disable_metadata;
}

/**
   \brief Construct exactly one cached instance of !{!"llvm.loop.unroll.disable"}.
 */
INLINE static LL_MDRef
cons_nounroll_metadata(void)
{
  LL_MDRef lvcomp[1];
  if (LL_MDREF_IS_NULL(cached_unroll_disable_metadata)) {
   lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.unroll.disable");
   cached_unroll_disable_metadata = ll_get_md_node(cpu_llvm_module,
       LL_PlainMDNode, lvcomp, 1);
  }
  return cached_unroll_disable_metadata;
}

/**
   \brief Construct exactly one cached instance of !{!"llvm.loop.vectorize.enable", 1}.
 */
INLINE static LL_MDRef
cons_vectorize_metadata(void)
{
  LL_MDRef lvcomp[2];
  if (LL_MDREF_IS_NULL(cached_vectorize_enable_metadata)) {
    lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.vectorize.enable");
    lvcomp[1] = ll_get_md_i1(1);
    cached_vectorize_enable_metadata = ll_get_md_node(cpu_llvm_module,
        LL_PlainMDNode, lvcomp, 2);
  }
  return cached_vectorize_enable_metadata;
}

/**
   \brief Second pass to clean up all the dead sincos callsites
   \param isns  The list of instructions
 */
INLINE static void
remove_dead_sincos_calls(INSTR_LIST *isns)
{
  INSTR_LIST *p;
  for (p = isns; p; p = p->next) {
    hash_data_t data;
    if (!hashmap_lookup(sincos_map, p, &data))
      continue;
    if ((p->i_name == I_CALL) && (HKEY2INT(data) != SINCOS_EXTRACT) &&
        ((HKEY2INT(data) & SINCOS_MASK) != SINCOS_MASK)) {
      p->operands->next = NULL;
      remove_instr(p, false);
    }
  }

  // finalize
  if (sincos_map)
    hashmap_free(sincos_map);
  sincos_map = NULL;
  if (sincos_imap)
    hashmap_free(sincos_imap);
  sincos_imap = NULL;
}

INLINE static bool
sincos_seen(void)
{
  return sincos_imap != NULL;
}

INLINE static void
sincos_clear_all_args(void)
{
  hashmap_clear(sincos_imap);
}

/**
   \brief First pass to rewrite degenerate sincos to sin (or cos) as needed
   \param isns  The list of instructions
 */
INLINE static void
cleanup_unneeded_sincos_calls(INSTR_LIST *isns)
{
  INSTR_LIST *p;

  DEBUG_ASSERT(sincos_seen(), "function must be marked as containing sincos");
  for (p = isns; p; p = p->next) {
    if (!hashmap_lookup(sincos_map, p, NULL))
      continue;
    if (p->i_name == I_EXTRACTVAL) {
      hash_data_t data;
      const LL_Type *retTy;
      const LL_Type *floatTy;
      char name[36]; /* make_math_name buffer is 32 */
      OPERAND *op;
      TMPS *t;
      INSTR_LIST *call = p->operands->tmps->info.idef;

      if (!hashmap_lookup(sincos_map, call, &data))
        continue;
      if ((HKEY2INT(data) & SINCOS_MASK) == SINCOS_MASK)
        continue;

      // replace this use (scalar)
      retTy = p->ll_type->sub_types[0];
      floatTy = make_lltype_from_dtype(DT_FLOAT);
      if (ILI_OPC(call->ilix) == IL_VSINCOS) {
        const int vecLen = retTy->sub_elements;
        LL_Type *eleTy = retTy->sub_types[0];
        bool hasMask = false;
        int opndCount = ili_get_vect_arg_count(call->ilix);
        DEBUG_ASSERT(retTy->data_type == LL_VECTOR, "vector type expected");
        if (ILI_OPC(ILI_OPND(call->ilix, opndCount - 1)) != IL_NULL) {
          hasMask = true;
        }
        llmk_math_name(name, (HKEY2INT(data) & SINCOS_COS) ? MTH_cos : MTH_sin,
                       vecLen, hasMask,
                       (eleTy == floatTy) ? DT_FLOAT : DT_DBLE);
      } else {
        llmk_math_name(name, (HKEY2INT(data) & SINCOS_COS) ? MTH_cos : MTH_sin,
                       1, false, (retTy == floatTy) ? DT_FLOAT : DT_DBLE);
      }
      t = p->tmps;
      op = call->operands->next;
      op = gen_call_to_builtin(call->ilix, name, op, retTy, p, I_CALL,
                               InstrListFlagsNull, EXF_PURE);
      p->i_name = I_CALL;
      p->tmps = t;
      DEBUG_ASSERT(t->use_count > 0, "must have positive use count");
      DEBUG_ASSERT(t->info.idef == op->tmps->info.idef, "instruction differs");
    }
  }
}

/**
   \brief Is the store ILT really a homing store?
   \param rIli  The value to be stored
   \param nme   The NME argument of the store
 */
INLINE static bool
store_for_homing(int rIli, int nme)
{
  const int fnSym = gbl.currsub;
  const int sym = NME_SYM(nme);
  if ((sym > 0) && (SCG(sym) == SC_DUMMY))
    return true;
  if (CFUNCG(fnSym) && (DTY(DTYPEG(fnSym)) == TY_STRUCT) &&
      bindC_function_return_struct_in_registers(fnSym) &&
      (ILI_OPC(rIli) == IL_LDA)) {
    const int rrIli = ILI_OPND(rIli, 1);
    return (ILI_OPC(rrIli) == IL_ACON) &&
           (SCG(CONVAL1G(ILI_OPND(rrIli, 1))) == SC_DUMMY);
  }
  return false;
}

static void
add_external_function_declaration(const char *key, EXFUNC_LIST *exfunc)
{
  const SPTR sptr = exfunc->sptr;

  if (sptr) {
    LL_ABI_Info *abi =
        ll_abi_for_func_sptr(cpu_llvm_module, sptr, DTYPEG(sptr));
    
    if (strstr(key, "@llvm.x86") != NULL) {
      for (unsigned i = 0; i <= abi->nargs; i++) {
        if (is_vector_x86_mmx(abi->arg[i].type)) {
        /* For x86 intrinsics, transform any vectors with overall 64 bits to 
           X86_mmx. */
          if (abi->arg[i].type->data_type == LL_PTR) {
            abi->arg[i].type = ll_get_pointer_type(ll_create_basic_type(
                                 abi->arg[i].type->module, LL_X86_MMX, 0));
          }
          else {
            abi->arg[i].type = ll_create_basic_type(
                                 abi->arg[i].type->module, LL_X86_MMX, 0);
          }
        } else if (abi->arg[i].type->data_type == LL_PTR) {
        /* All pointer types for x86 intrinsics (that aren't x86_mmx*), becomes 
           i8* pointers.*/
          abi->arg[i].type = ll_get_pointer_type(ll_convert_dtype(
                               abi->arg[i].type->module, DT_CHAR));
        }
      }
    }

    ll_proto_add_sptr(sptr, abi);
    if (exfunc->flags & EXF_INTRINSIC)
      ll_proto_set_intrinsic(ll_proto_key(sptr), exfunc->func_def);
#ifdef WEAKG
    if (WEAKG(sptr))
      ll_proto_set_weak(ll_proto_key(sptr), true);
#endif
  } else {
    DEBUG_ASSERT(key, "key must not be NULL");
    assert(exfunc->func_def && (exfunc->flags & EXF_INTRINSIC),
           "Invalid external function descriptor", 0, ERR_Fatal);
    if (*key == '@')
      ++key; /* do not include leading '@' in the key */

    LL_ABI_Info *abi = NULL;
    if (exfunc->flags & EXF_PURE) {
      /* minimum viable ABI for a builtin */
      abi = ll_abi_alloc(cpu_llvm_module, 0);
      abi->is_pure = true;
    }

    ll_proto_add(key, abi);
    ll_proto_set_intrinsic(key, exfunc->func_def);
  }
}

static void
add_profile_decl(const char *key, char *gname)
{
  EXFUNC_LIST *exfunc =
      (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
  memset(exfunc, 0, sizeof(EXFUNC_LIST));
  exfunc->func_def = gname;
  exfunc->flags |= EXF_INTRINSIC;
  add_external_function_declaration(key, exfunc);
}

/**
   \brief Shared code to emit a call to a profile function
 */
INLINE static void
write_profile_call(const char *profFn)
{
  fprintf(ASMFIL, "\tcall void @%s(ptr %%prof.thisfn, ptr %%prof.callsite)\n",
          profFn);
}

#undef PROF_ENTER
#undef PROF_EXIT
#undef PROF_CALLSITE
#define PROF_ENTER "__cyg_profile_func_enter"
#define PROF_EXIT "__cyg_profile_func_exit"
#define PROF_CALLSITE "llvm.returnaddress"

/**
   \brief Write a call to the profile entry routine
   \param sptr      The symbol of the function we are generating
   \param currFnTy  The type of the function, \p sptr

   \c -finstrument-functions adds an entry and exit profile call to each
   function
 */
INLINE static void
write_profile_enter(SPTR sptr, LL_Type *currFnTy)
{
  static bool protos_defined = false;
  const char *currFn = get_llvm_name(sptr);
  fprintf(ASMFIL,
          "\t%%prof.thisfn = ptr @%s\n"
          "\t%%prof.callsite = call ptr @" PROF_CALLSITE "(i32 0)\n",
          currFn);
  write_profile_call(PROF_ENTER);
  if (!protos_defined) {
    /* add the declarations for output */
    const char retAddr[] = "declare ptr @" PROF_CALLSITE "(i32)";
    const char entFn[] = "declare void @" PROF_ENTER "(ptr, ptr)";
    const char extFn[] = "declare void @" PROF_EXIT "(ptr, ptr)";
    char *gname;

    protos_defined = true;
    gname = (char *)getitem(LLVM_LONGTERM_AREA, sizeof(retAddr));
    strcpy(gname, retAddr);
    add_profile_decl(PROF_CALLSITE, gname);
    gname = (char *)getitem(LLVM_LONGTERM_AREA, sizeof(entFn));
    strcpy(gname, entFn);
    add_profile_decl(PROF_ENTER, gname);
    gname = (char *)getitem(LLVM_LONGTERM_AREA, sizeof(extFn));
    strcpy(gname, extFn);
    add_profile_decl(PROF_EXIT, gname);
  }
}

/**
   \brief Write a call to the profile exit routine

   This should be done before each \c ret in the function.
 */
INLINE static void
write_profile_exit(void)
{
  write_profile_call(PROF_EXIT);
}

#undef PROF_ENTER
#undef PROF_EXIT
#undef PROF_CALLSITE

/**
   \brief Write out the start of an LLVM function definition
 */
INLINE static void
write_routine_definition(SPTR func_sptr, LL_ABI_Info *abi, LL_Module *module,
                         LL_Type *funcTy)
{
  if (has_multiple_entries(func_sptr)) {
    write_master_entry_routine();
    return;
  }
  build_routine_and_parameter_entries(func_sptr, abi, module);
  if (XBIT(129, 0x800)) {
    /* -finstrument-functions */
    write_profile_enter(func_sptr, funcTy);
  }
}

INLINE static void
finish_routine(void)
{
  const int currFn = GBL_CURRFUNC;
  /***** "{" so vi matches *****/
  print_line("}");
  llassem_end_func(cpu_llvm_module->debug_info, currFn);
  if (flg.smp) {
    ll_reset_outlined_func();
  }
}

/**
   \brief Return current loop ID and mark it as vectorizable.

   "NODEPCHK" and "VECTOR ALWAYS" currently both mean to mark a loop ID with
   !"llvm.loop.vectorize.enable", and to mark memory instructions with
   !llvm.access.group referring to the cached access group. This function
   returns the current loop ID metadata via cons_loop_id_md() and marks the
   loop as vectorizable.
 */
static LL_MDRef
cons_vec_always_metadata(void)
{
  LL_MDRef loop_id_md = cons_loop_id_md();
  if (!cached_loop_id_md_has_vectorize) {
    cached_loop_id_md_has_vectorize = true;
    LL_MDRef vectorize = cons_vectorize_metadata();
    LL_MDRef paraccess = cons_loop_parallel_accesses_metadata();
    ll_extend_md_node(cpu_llvm_module, loop_id_md, vectorize);
    ll_extend_md_node(cpu_llvm_module, loop_id_md, paraccess);
  }
  return loop_id_md;
}

/**
   \brief Construct exactly one cached instance of !{!"llvm.loop.unroll.enable"}.
 */
static LL_MDRef
cons_unroll_metadata(void)
{
  LL_MDRef lvcomp[1];
  if (LL_MDREF_IS_NULL(cached_unroll_enable_metadata)) {
   lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.unroll.enable");
   cached_unroll_enable_metadata = ll_get_md_node(cpu_llvm_module,
       LL_PlainMDNode, lvcomp, 1);
  }
  return cached_unroll_enable_metadata;
}

/**
   \brief Construct instance of !{!"llvm.loop.unroll.count", <unroll_factor>}.
 */
static LL_MDRef
cons_unroll_count_metadata(int unroll_factor)
{
  LL_MDRef lvcomp[2];
  LL_MDRef unroll;
  lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.unroll.count");
  lvcomp[1] = ll_get_md_i32(cpu_llvm_module, unroll_factor);
  unroll= ll_get_md_node(cpu_llvm_module, LL_PlainMDNode, lvcomp, 2);
  return unroll;
}

/**
   \brief Construct exactly one instance of !{!"llvm.loop.vectorize.scalable.enable", <arg>}.
 */
INLINE static LL_MDRef
cons_vectorlength_scalable_metadata( int arg )
{
  LL_MDRef lvcomp[2];

  lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.vectorize.scalable.enable");
  lvcomp[1] = ll_get_md_i1( arg );
  return ll_get_md_node( cpu_llvm_module, LL_PlainMDNode, lvcomp, 2 );
}

INLINE static bool
ignore_simd_block(int bih)
{
  return (!XBIT(183, 0x4000000)) && BIH_NOSIMD(bih);
}

/**
   \brief Remove all deletable instructions from the instruction list
 */
INLINE static void
remove_dead_instrs(void)
{
  INSTR_LIST *instr;
  for (instr = llvm_info.last_instr; instr;) {
    if ((instr->i_name == I_STORE) && (instr->flags & DELETABLE))
      instr = remove_instr(instr, false);
    else if ((instr->i_name != I_CALL) && (instr->i_name != I_INVOKE) &&
             (instr->i_name != I_ATOMICRMW) && (instr->tmps != NULL) &&
             (instr->tmps->use_count <= 0))
      instr = remove_instr(instr, false);
    else
      instr = instr->prev;
  }
}

/*
 * Check if the branch instruction is having a loop pragma
 * xbit/xflag pair.
 */
static bool check_for_loop_directive(int branch_line_number, int xbit, int xflag) {
  int iter;
  LPPRG *lpprg;

  // Check if any loop pragmas are specified
  if (direct.lpg.avail > 1) {
    // Loop thru all the loop pragmas
    for (iter = 1; iter < direct.lpg.avail; iter++) {
      lpprg = direct.lpg.stgb + iter;
      // check if xbit/xflag pair is available
      if ((lpprg->dirset.x[xbit] & xflag)
          &&
          (branch_line_number == lpprg->end_line)) {
        return  true;
      } // if

      if (branch_line_number < lpprg->beg_line) {
        // branch instruction is not having any pragma specified.
        break;
      } // if
    } // for
  } // if

  return false;
} // check_for_loop_directive

/**
   \brief process debug info of constants with parameter attribute.
 */
static void
process_params(void)
{
  unsigned smax = stb.stg_avail;
  for (SPTR sptr = get_symbol_start(); sptr < smax; ++sptr) {
    DTYPE dtype = DTYPEG(sptr);
    if (STYPEG(sptr) == ST_PARAM && should_preserve_param(dtype)) {
      if (DTY(dtype) == TY_ARRAY || DTY(dtype) == TY_STRUCT) {
        /* array and derived types have 'var$ac' constant variable
         * lets use that, by renaming that to 'var'.
         */
        SPTR new_sptr = (SPTR)CONVAL1G(sptr);
        NMPTRP(new_sptr, NMPTRG(sptr));
        CCSYMP(new_sptr, 0);
      } else {
        LL_DebugInfo *di = cpu_llvm_module->debug_info;
        int fin = BIH_FINDEX(gbl.entbih);
        LL_Type *type = make_lltype_from_dtype(dtype);
        OPERAND *ld = make_operand();
        ld->ot_type = OT_MDNODE;
        ld->val.sptr = sptr;
        LL_MDRef lcl = lldbg_emit_local_variable(di, sptr, fin, true);

        /* lets generate llvm.dbg.value intrinsic for it.*/
        insert_llvm_dbg_value(ld, lcl, sptr, type);
      }
    }
  }
}

/**
   \brief Construct exactly one cached instance of !{!"llvm.loop.vectorize.width", <width>}.
 */
INLINE static LL_MDRef
cons_vectorize_width_metadata(int width)
{
  LL_MDRef lvcomp[2];
  lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.vectorize.width");
  lvcomp[1] = ll_get_md_i32(cpu_llvm_module, width);
  return ll_get_md_node(cpu_llvm_module, LL_PlainMDNode, lvcomp, 2);
}

/**
   \brief Constructs needed simdlen nodes
 */
static void construct_simdlen_metadata(LL_MDRef *loop_md, int simdlen)
{
  LL_MDRef vectorize = cons_vectorize_metadata();
  ll_extend_md_node(cpu_llvm_module, *loop_md, vectorize);
  if (simdlen > 0) {
    LL_MDRef width = cons_vectorize_width_metadata(simdlen);
    ll_extend_md_node(cpu_llvm_module, *loop_md, width);
  }
}

/**
   \brief Constructs needed vectorlength nodes
 */
static void construct_vectorlength_metadata(LL_MDRef *loop_md, int vectorlength_factor, int scalable)
{
  LL_MDRef vectorlength_enable = cons_vectorize_metadata();
  ll_extend_md_node(cpu_llvm_module, *loop_md, vectorlength_enable);
  LL_MDRef vectorlength_scalable  = cons_vectorlength_scalable_metadata(scalable);
  ll_extend_md_node(cpu_llvm_module, *loop_md, vectorlength_scalable);
  if (vectorlength_factor > 0) {
    LL_MDRef lvcomp[2];
    LL_MDRef width;
    lvcomp[0] = ll_get_md_string(cpu_llvm_module, "llvm.loop.vectorize.width");
    lvcomp[1] = ll_get_md_i32(cpu_llvm_module, vectorlength_factor);
    width = ll_get_md_node(cpu_llvm_module, LL_PlainMDNode, lvcomp, 2);
    ll_extend_md_node(cpu_llvm_module, *loop_md, width);
  }
}

/**
   \brief Perform code translation from ILI to LLVM for one routine
 */
void
schedule(void)
{
  LL_Type *func_type;
  int bihx, ilt, ilix, ilix2, nme;
  ILI_OP opc;
  int rhs_ili, lhs_ili, sptr;
  int bih, bihprev, bihcurr, bihnext;
  int concurBih = 0;
  bool made_return;
  bool merge_next_block;
  bool targetNVVM = false;
  bool processHostConcur = true;
  SPTR func_sptr = GBL_CURRFUNC;
  bool first = true;
  CG_cpu_compile = true;
  int unroll_factor = 0;
  int vectorlength_factor = 0;
  int simdlen = 0;

  funcId++;
  assign_fortran_storage_classes();
  if (XBIT(68, 0x1) && (!XBIT(183, 0x40000000)))
    widenAddressArith();
  if (gbl.outlined && funcHasNoDepChk())
    redundantLdLdElim();

restartConcur:
  FTN_HOST_REG() = 1;
  func_sptr = GBL_CURRFUNC;
  entry_bih = gbl.entbih;

  cg_llvm_init();
#ifdef OMP_OFFLOAD_LLVM
  if (ISNVVMCODEGEN) {
    current_module = gpu_llvm_module;
    use_gpu_output_file();
  } else 
#endif
  {
    current_module = cpu_llvm_module;
  }
  if (!XBIT(53, 0x10000))
    current_module->omnipotentPtr = ll_get_md_null();

  consTempMap(ilib.stg_avail);

  store_llvm_localfptr();

  /* inititalize the definition lists per routine */
  csedList = NULL;
  memset(&ret_info, 0, sizeof(ret_info));
  llvm_info.curr_func = NULL;

#if DEBUG
  if (DBGBIT(11, 1))
    dumpblocks("just before LLVM translation");
  if (DBGBIT(11, 0x810) || DBGBIT(12, 0x30)) {
    fprintf(ll_dfile, "--- ROUTINE %s (sptr# %d) ---\n", SYMNAME(func_sptr),
            func_sptr);
  }
  if (DBGBIT(11, 0x10)) {
    bihx = gbl.entbih;
    for (;;) {
      dmpilt(bihx);
      if (BIH_LAST(bihx))
        break;
      bihx = BIH_NEXT(bihx);
    }
    dmpili();
  }

#endif

  /* Start the LLVM translation here */
  llvm_info.last_instr = NULL;
  llvm_info.curr_instr = NULL;
  Instructions = NULL;
  /* Update symbol table before we process any routine arguments, this must be
   * called before ll_abi_for_func_sptr()
   */
  stb_process_routine_parameters();

  hashmap_clear(llvm_info.homed_args);
  llvm_info.abi_info = ll_abi_for_func_sptr(current_module, func_sptr, DT_NONE);
  func_type = ll_abi_function_type(llvm_info.abi_info);
  process_sptr(func_sptr);

#ifdef OMP_OFFLOAD_LLVM
  if (ISNVVMCODEGEN) {
    /* for now, we generate two ll_function one for host one device. */
    /* it is kernel function in gpu module */
    LL_Function *llfunc = nullptr;
    if (OMPACCFUNCKERNELG(func_sptr)) {
      llfunc = ll_create_device_function_from_type(current_module, func_type,
                                                   &(SNAME(func_sptr)[1]), 1, 0,
                                                   "ptx_kernel", LL_NO_LINKAGE);
    } else if (OMPACCFUNCDEVG(func_sptr)) {
      llfunc = ll_create_device_function_from_type(current_module, func_type,
                                                   &(SNAME(func_sptr)[1]), 0, 0,
                                                   "", LL_INTERNAL_LINKAGE);
    }
    ll_set_device_function_arguments(current_module, llfunc,
                                     llvm_info.abi_info);
  }
#endif
  llvm_info.curr_func =
      ll_create_function_from_type(func_type, SNAME(func_sptr));

  if (LL_ABI_HAS_SRET(llvm_info.abi_info) && CONTAINEDG(func_sptr)) {
    assert(false, "inner function returning derived type not yet implemented",
           func_sptr, ERR_Fatal);
  }
  ad_instr(0, gen_instr(I_NONE, NULL, NULL, make_label_op(SPTR_NULL)));

  ll_proto_add_sptr(func_sptr, llvm_info.abi_info);

  if (flg.debug || XBIT(120, 0x1000)) {
    if (!CCSYMG(func_sptr) || BIH_FINDEX(gbl.entbih)) {
      const DTYPE funcType = get_return_type(func_sptr);
      LL_Value *func_ptr = ll_create_pointer_value_from_type(
              current_module, func_type, SNAME(func_sptr), 0);

#ifdef OMP_OFFLOAD_LLVM
      if(XBIT(232, 0x8))
        targetNVVM = true;
      if (!ISNVVMCODEGEN && !TEXTSTARTUPG(func_sptr))
#endif
      {
        lldbg_emit_subprogram(current_module->debug_info, func_sptr, funcType,
                              BIH_FINDEX(gbl.entbih), targetNVVM);
        lldbg_set_func_ptr(current_module->debug_info, func_ptr);
      }

      if (!ISNVVMCODEGEN) {
        /* FIXME: should this be done for C, C++? */
        lldbg_reset_dtype_array(current_module->debug_info, DT_DEFERCHAR + 1);
      }
    }
  }

  /* set the return type of the function */
  analyze_ret_info(func_sptr);

  /* Build up the additional items/dummys needed for the master sptr if there
     are entries, and call process_formal_arguments on that information. */
  if (has_multiple_entries(gbl.currsub) && get_entries_argnum()) {
    entry_abi = process_ll_abi_func_ftn_mod(current_module, get_master_sptr(), 1);
    process_formal_arguments(entry_abi);
  } else {
    process_formal_arguments(llvm_info.abi_info);
  }
  made_return = false;

  get_local_overlap_size();
  expr_id = 0;
  last_stmt_is_branch = 0;

  bih = BIH_NEXT(0);
  if ((XBIT(34, 0x200) || gbl.usekmpc) && !processHostConcur)
    bih = gbl.entbih;

  cached_access_group_metadata = ll_create_distinct_md_node(cpu_llvm_module, LL_PlainMDNode, NULL, 0);

  /* construct the body of the function */
  for (; bih; bih = BIH_NEXT(bih))
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt))
      build_csed_list(ILT_ILIP(ilt));

  /* Process variables with parameter attribute to generate debug info, if
     debug is on. */
  if (!XBIT(49, 0x10) && flg.debug
#if defined(OMP_OFFLOAD_PGI) || defined(OMP_OFFLOAD_LLVM)
      && !gbl.ompaccel_isdevice
#endif
  )
    process_params();

  merge_next_block = false;
  bih = BIH_NEXT(0);
  if ((XBIT(34, 0x200) || gbl.usekmpc) && !processHostConcur)
    bih = gbl.entbih;
  for (; bih; bih = BIH_NEXT(bih)) {

#if DEBUG
    if (DBGBIT(12, 0x10)) {
      fprintf(ll_dfile, "schedule(): at bih %d\n", bih);
    }
#endif
    DBGTRACE1("Processing bih %d", bih)
    bihcurr = bih;
    if (sincos_seen())
      sincos_clear_all_args();

    /* skip over an entry bih  */
    if (BIH_EN(bih)) {
      if (BIH_ILTFIRST(bih) != BIH_ILTLAST(bih))
        goto do_en_bih;
      else if (has_multiple_entries(gbl.currsub))
        goto do_en_bih;
      bihprev = bih;
      continue;
    }
    /* do we have a label that's the target of a branch? Either a
     * user label (via a goto) or a compiler created label for branching.
     */
    else if ((sptr = BIH_LABEL(bih)) && (DEFDG(sptr) || CCSYMG(sptr))) {
      assert(STYPEG(sptr) == ST_LABEL, "schedule(), not ST_LABEL", sptr,
             ERR_Fatal);
      clear_csed_list();
      make_stmt(STMT_LABEL, sptr, false, SPTR_NULL, 0);
    }

  do_en_bih:

    /* in general, ilts will correspond to statements */
    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt))
      build_csed_list(ILT_ILIP(ilt));

    bihnext = BIH_NEXT(bih);

    if (merge_next_block == false)
      new_ebb = true;

    if (((flg.opt == 1 && BIH_EN(bih)) || (flg.opt >= 2 && !BIH_TAIL(bih))) &&
        bihnext && (!BIH_LABEL(bihnext)) && BIH_PAR(bihnext) == BIH_PAR(bih) &&
        BIH_CS(bihnext) == BIH_CS(bih) && BIH_TASK(bihnext) == BIH_TASK(bih) &&
        !BIH_NOMERGE(bih) && !BIH_NOMERGE(bihnext)) {
      merge_next_block = true;
    } else {
      merge_next_block = false;
    }

    open_pragma(BIH_LINENO(bih));
    BIH_NODEPCHK(bih) = !flg.depchk;
    if (XBIT(19, 0x18))
      BIH_NOSIMD(bih) = true;
    else if (XBIT(19, 0x400))
      BIH_SIMD(bih) = true;
    if ((!XBIT(69, 0x100000) && BIH_NODEPCHK(bih) && !ignore_simd_block(bih)) ||
        XBIT(191, 0x4)) {
      fix_nodepchk_flag(bih);
      mark_rw_access_grp(bih);
    } else {
      clear_rw_access_grp();
    }
    if (flg.x[237] > 0)
      simdlen = flg.x[237];
    if (flg.x[234] > 0) {
      BIH_VECTORLENGTH_ENABLED(bih) = true;
      BIH_VECTORLENGTH_SCALABLE(bih) = (XBIT(234, 0x4) > 0);
      vectorlength_factor = flg.x[235];
    }
    if (flg.x[9] > 0)
      unroll_factor = flg.x[9];
    if (XBIT(11, 0x2) && unroll_factor)
      BIH_UNROLL_COUNT(bih) = true;
    else if (XBIT(11, 0x1))
      BIH_UNROLL(bih) = true;
    else if (XBIT(11, 0x400))
      BIH_NOUNROLL(bih) = true;
    close_pragma();

    for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
      if (BIH_EN(bih) && ilt == BIH_ILTFIRST(bih)) {
        if (!has_multiple_entries(gbl.currsub))
          continue;
        if (first) {
          insert_jump_entry_instr(ilt);
          first = false;
        }
        insert_entry_label(ilt);
        continue;
      }
#if DEBUG
      if (DBGBIT(12, 0x10)) {
        fprintf(ll_dfile, "\tat ilt %d\n", ilt);
      }
#endif

      if (!ISNVVMCODEGEN && (flg.debug || XBIT(120, 0x1000))) {
        lldbg_emit_line(current_module->debug_info, ILT_LINENO(ilt));
      }
      ilix = ILT_ILIP(ilt);
      opc = ILI_OPC(ilix);

      if (processHostConcur && (XBIT(34, 0x200) || gbl.usekmpc)) {
        if (opc == IL_BCONCUR) {
          ++concurBih;
        } else if (opc == IL_ECONCUR) {
          --concurBih;
        }
        if (concurBih)
          continue;
      }

      if (ILT_BR(ilt)) { /* branch */
        SPTR next_bih_label = SPTR_NULL;

        if (!ILT_NEXT(ilt) && bihnext) {
          const SPTR t_next_bih_label = BIH_LABEL(bihnext);
          if (t_next_bih_label &&
              (DEFDG(t_next_bih_label) || CCSYMG(t_next_bih_label)))
            next_bih_label = t_next_bih_label;
        }
        make_stmt(STMT_BR, ilix, false, next_bih_label, ilt);

        int branch_nops = ilis[opc].oprs;
        int branch_target = ILI_OPND(ilix, branch_nops);
        int branch_target_bih = ILIBLKG(branch_target);
        int start_line = BIH_LINENO(branch_target_bih);
        int end_line = ILT_LINENO(ilt);
        bool is_loop_backedge = BIH_HEAD(branch_target_bih) && start_line <= end_line;

        LL_MDRef loop_md = ll_get_md_null();

        bool emitting_debug = !ISNVVMCODEGEN && (flg.debug || XBIT(120, 0x1000));
        if (emitting_debug && is_loop_backedge) {
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();
          lldbg_emit_line(current_module->debug_info, start_line);
          LL_MDRef loop_line_start = lldbg_get_line(current_module->debug_info);
          lldbg_emit_line(current_module->debug_info, end_line);
          LL_MDRef loop_line_end = lldbg_get_line(current_module->debug_info);

          ll_extend_md_node(cpu_llvm_module, loop_md, loop_line_start);
          ll_extend_md_node(cpu_llvm_module, loop_md, loop_line_end);
        }

        if ((!XBIT(69, 0x100000) && BIH_NODEPCHK(bih) && !BIH_NODEPCHK2(bih) &&
             !ignore_simd_block(bih)) ||
            BIH_SIMD(bih)) {
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();

          // cons_vec_always_metadata is different from the others because it
          // arranges that llvm.loop.vectorize is only added to the loop
          // metadata once.
          (void)cons_vec_always_metadata();
        }
        if ((check_for_loop_directive(ILT_LINENO(ilt), 191, 0x4))) {
          LL_MDRef loop_md = cons_vec_always_metadata();
          INSTR_LIST *i = find_last_executable(llvm_info.last_instr);
          if (i) {
            i->flags |= LOOP_BACKEDGE_FLAG;
            i->misc_metadata = loop_md;
          }
        }
        if (simdlen > 0) {
          if (LL_MDREF_IS_NULL(loop_md)) {
            loop_md = cons_loop_id_md();
          }
          construct_simdlen_metadata(&loop_md, simdlen);
        }
        if (BIH_VECTORLENGTH_ENABLED(bih)) {
          if (LL_MDREF_IS_NULL(loop_md)) {
            loop_md = cons_loop_id_md();
          }
          construct_vectorlength_metadata(&loop_md, vectorlength_factor, BIH_VECTORLENGTH_SCALABLE(bih));
        }
        if (BIH_UNROLL(bih)) { // Set on open_pragma() -> if(XBIT(11,0X3))
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();
          ll_extend_md_node(cpu_llvm_module, loop_md, cons_unroll_metadata());
        } else if (BIH_UNROLL_COUNT(bih)) {
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();
          ll_extend_md_node(cpu_llvm_module, loop_md, 
              cons_unroll_count_metadata(unroll_factor));
        } else if (BIH_NOUNROLL(bih)) {
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();
          ll_extend_md_node(cpu_llvm_module, loop_md, cons_nounroll_metadata());
        }
        if (ignore_simd_block(bih)) {
          if (LL_MDREF_IS_NULL(loop_md))
            loop_md = cons_loop_id_md();
          ll_extend_md_node(cpu_llvm_module, loop_md, cons_novectorize_metadata());
        }

        if (!LL_MDREF_IS_NULL(loop_md)) {
          // If any loop metadata is present, mark the last executable
          // instruction as a backedge.
          if (INSTR_LIST *i = find_last_executable(llvm_info.last_instr)) {
            i->flags |= LOOP_BACKEDGE_FLAG;
            i->misc_metadata = loop_md;
          }
        }
      } else if ((ILT_ST(ilt) || ILT_DELETE(ilt)) &&
                 (IL_TYPE(opc) == ILTY_STORE)) {
        /* store */
        rhs_ili = ILI_OPND(ilix, 1);
        lhs_ili = ILI_OPND(ilix, 2);
        nme = ILI_OPND(ilix, 3);
        /* can we ignore homing code? Try it here */
        if (is_rgdfili_opcode(ILI_OPC(rhs_ili)))
          continue;
        if (BIH_EN(bih) && store_for_homing(rhs_ili, nme))
          continue;
        make_stmt(STMT_ST, ilix,
                  ENABLE_CSE_OPT && ILT_DELETE(ilt) &&
                      (IL_TYPE(opc) == ILTY_STORE),
                  SPTR_NULL, ilt);
      } else if (opc == IL_JSR && cgmain_init_call(ILI_OPND(ilix, 1))) {
        make_stmt(STMT_SZERO, ILI_OPND(ilix, 2), false, SPTR_NULL, ilt);
      } else if (opc == IL_SMOVE) {
        make_stmt(STMT_SMOVE, ilix, false, SPTR_NULL, ilt);
      } else if (ILT_EX(ilt)) {
        // ilt contains a call
        if (opc == IL_LABEL)
          continue; /* gen_llvm_expr does not handle IL_LABEL */
        switch (opc) {
        case IL_DFRSP:
        case IL_DFRDP:
#ifdef TARGET_SUPPORTS_QUADFP
        case IL_DFRQP:
#endif
        case IL_DFRCS:
#ifdef LONG_DOUBLE_FLOAT128
        case IL_FLOAT128RESULT:
#endif
          ilix = ILI_OPND(ilix, 1);
          opc = ILI_OPC(ilix);
          break;
        default:
          break;
        }
        if (is_mvili_opcode(opc)) {
          /* call part of the return */
          goto return_with_call;
        } else if (is_freeili_opcode(opc)) {
          remove_from_csed_list(ilix);
          make_stmt(STMT_DECL, ilix, false, SPTR_NULL, ilt);
        } else if ((opc == IL_JSR) || (opc == IL_QJSR) || (opc == IL_JSRA)
#ifdef SJSR
                   || (opc == IL_SJSR) || (opc == IL_SJSRA)
#endif
        ) {
          /* call not in a return */
          make_stmt(STMT_CALL, ilix, false, SPTR_NULL, ilt);
        } else if ((opc != IL_DEALLOC) && (opc != IL_NOP)) {
          make_stmt(STMT_DECL, ilix, false, SPTR_NULL, ilt);
        }
      } else if (opc == IL_FENCE) {
        gen_llvm_fence_instruction(ilix);
      } else if (opc == IL_PREFETCH) {
        LL_Type *optype = make_lltype_from_dtype(DT_CPTR);
        insert_llvm_prefetch(ilix, gen_llvm_expr(ILI_OPND(ilix, 1), optype));
      } else {
      /* may be a return; otherwise mostly ignored */
      /* However, need to keep track of FREE* ili, to match them
       * with CSE uses, since simple dependences need to be checked
       */
      return_with_call:
        if (is_mvili_opcode(opc)) { /* routine return */
          if (ret_info.sret_sptr == 0) {
            ilix2 = ILI_OPND(ilix, 1);
            /* what type of return value */
            switch (IL_TYPE(ILI_OPC(ilix2))) {
            case ILTY_LOAD:
            case ILTY_CONS:
            case ILTY_ARTH:
            case ILTY_DEFINE:
            case ILTY_MOVE:
              make_stmt(STMT_RET, ilix2, false, SPTR_NULL, ilt);
              break;
            case ILTY_OTHER:
              /* handle complex builtin */
              if (XBIT(70, 0x40000000) && (IL_RES(ILI_OPC(ilix2)) == ILIA_DP ||
                                           IL_RES(ILI_OPC(ilix2)) == ILIA_SP)) {
                make_stmt(STMT_RET, ilix2, false, SPTR_NULL, ilt);
                break;
              }
              FLANG_FALLTHROUGH;
            default:
              switch (ILI_OPC(ilix2)) {
              case IL_ISELECT:
              case IL_KSELECT:
              case IL_ASELECT:
              case IL_FSELECT:
              case IL_DSELECT:
              case IL_ATOMICRMWI:
              case IL_ATOMICRMWKR:
              case IL_ATOMICRMWA:
              case IL_ATOMICRMWSP:
              case IL_ATOMICRMWDP:
                make_stmt(STMT_RET, ilix2, false, SPTR_NULL, ilt);
                break;
              default:
                assert(0, "schedule(): incompatible return type",
                       IL_TYPE(ILI_OPC(ilix2)), ERR_Fatal);
              }
            }
            made_return = true;
          }
        } else if (is_freeili_opcode(opc)) {
#if DEBUG
          if (DBGBIT(12, 0x10)) {
            fprintf(ll_dfile, "\tfound free ili: %d(%s)\n", ilix, IL_NAME(opc));
          }
#endif
          remove_from_csed_list(ilix);
          make_stmt(STMT_DECL, ilix, false, SPTR_NULL, ilt);
        } else if (opc == IL_LABEL) {
          continue; /* ignore IL_LABEL */
        } else if (BIH_LAST(bih) && !made_return) {
          /* at end, make a NULL return statement if return not already made */
          make_stmt(STMT_RET, ilix, false, SPTR_NULL, ilt);
        } else if (opc == IL_SMOVE) {
          /* moving/storing a block of memory */
          make_stmt(STMT_SMOVE, ilix, false, SPTR_NULL, ilt);
        }
      }
    }
    bihprev = bih;
  }

  build_unused_global_define_from_params();

/* header already printed; now print global and static defines */
  write_ftn_typedefs();
  write_global_and_static_defines();

#ifdef OMP_OFFLOAD_LLVM
  if (flg.omptarget && ISNVVMCODEGEN)
    use_cpu_output_file();
#endif
  assem_data();
#ifdef OMP_OFFLOAD_LLVM
  if (flg.omptarget && ISNVVMCODEGEN)
    use_gpu_output_file();
  if (flg.omptarget)
    write_libomtparget();
#endif
  /* perform setup for each routine */
  write_routine_definition(func_sptr, llvm_info.abi_info, current_module,
                           func_type);

  /* write out local variable defines */
  ll_write_local_objects(llvm_file(), llvm_info.curr_func);
  /* Emit alloca for local equivalence, c.f. get_local_overlap_var(). */
  write_local_overlap();

  if (ENABLE_BLK_OPT)
    optimize_block(llvm_info.last_instr);

  /*
   * similar code in llvect.c, cgoptim1.c, and llvm's cgmain.c & llvect.c
   * 01/17/17 -- we are no longer attempting to transform the divide into
   *             a multiply by recip; we are simply depending on the user
   *             adding -Mfprelaxed[=div]
   * 02/10/17 -- enabled with -Mnouniform
   *
   */
  if (XBIT_NOUNIFORM && (!XBIT(183, 0x8000)) && XBIT(15, 4) && (!flg.ieee)) {
    undo_recip_div(Instructions);
  }
  if (sincos_seen()) {
    cleanup_unneeded_sincos_calls(Instructions);
    remove_dead_sincos_calls(Instructions);
  }
  /* try FMA rewrite */
  if (XBIT_GENERATE_SCALAR_FMA /* HAS_FMA and x-flag 164 */
      && (get_llvm_version() >= LL_Version_3_7)) {
    fma_rewrite(Instructions);
  }

  if (ENABLE_CSE_OPT) {
    remove_dead_instrs();
    Instructions->prev = NULL;
    if (XBIT(183, 0x40))
      sched_instructions(Instructions);
  }

  /* print out the instructions */
  write_instructions(current_module);

  finish_routine();

#ifdef OMP_OFFLOAD_LLVM
  if (ISNVVMCODEGEN) {
    use_cpu_output_file();
  }
#endif

  clear_prescan_complex_list();
  if (!ISNVVMCODEGEN && (flg.debug || XBIT(120, 0x1000)))
    lldbg_cleanup_missing_bounds(current_module->debug_info,
                                 BIH_FINDEX(gbl.entbih));
  hashmap_clear(llvm_info.homed_args); /* Don't home entry trampoline parms */
  if (processHostConcur)
    print_entry_subroutine(current_module);
  ll_destroy_function(llvm_info.curr_func);
  llvm_info.curr_func = NULL;

  assem_end();
  /* we need to set init_once to zero here because for cuda fortran combine with
   * acc - the constructors can be created without one after the other and
   * cg_llvm_end will not get call between those.  If init_once is not reset,
   * cg_llvm_init will not go through.
   */
  init_once = false;

  if (--routine_count > 0)
  {
    /* free CG_MEDTERM_AREA - done on a per-routine basis */
    freearea(CG_MEDTERM_AREA);
  }
  FTN_HOST_REG() = 1;

  if ((XBIT(34, 0x200) || gbl.usekmpc) &&
      (concurBih = processOutlinedByConcur(concurBih))) {
    processHostConcur = false;
    goto restartConcur;
  }
  ll_reset_gtid();
  if (flg.smp || (XBIT(34, 0x200) || gbl.usekmpc))
    ll_reset_gtid();

  if ((gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) &&
      ((flg.inliner && !XBIT(14, 0x10000)) || flg.autoinline)) {
      GBL_CURRFUNC = SPTR_NULL;
  }

  if (LL_DebugInfo *di = current_module->debug_info) {
    if (lldbg_get_di_routine_idx(di) > 0)
      lldbg_emit_lv_list(di);
  }

  gcTempMap();
  CG_cpu_compile = false;
} /* schedule */

INLINE static bool
call_sym_is(SPTR sptr, const char *sym_name)
{
  return sptr && (strncmp(SYMNAME(sptr), sym_name, strlen(sym_name)) == 0);
}

static OPERAND *
gen_llvm_instr(int ilix, ILI_OP opc, LL_Type *return_type,
               LL_Type *param_lltype, LL_InstrName itype)
{
  OPERAND *operand;
  OPERAND *param_op;
  INSTR_LIST *Curr_Instr;
  int arg_ili = ILI_OPND(ilix, 2);

  operand = make_tmp_op(return_type, make_tmps());
  Curr_Instr = gen_instr(itype, operand->tmps, operand->ll_type, NULL);
  assert(ILI_OPC(arg_ili) == opc,
         "gen_llvm_instr(): unexpected opc for parameter ", ILI_OPC(arg_ili),
         ERR_Fatal);
  param_op = gen_llvm_expr(ILI_OPND(arg_ili, 1), param_lltype);
  Curr_Instr->operands = param_op;
  arg_ili = ILI_OPND(arg_ili, 2);
  while ((arg_ili > 0) && (ILI_OPC(arg_ili) != IL_NULL)) {
    assert(ILI_OPC(arg_ili) == opc,
           "gen_llvm_instr(): unexpected opc for parameter ", ILI_OPC(arg_ili),
           ERR_Fatal);
    param_op->next = gen_llvm_expr(ILI_OPND(arg_ili, 1), param_lltype);
    param_op = param_op->next;
    arg_ili = ILI_OPND(arg_ili, 2);
  }
  ad_instr(ilix, Curr_Instr);

  return operand;
}

#ifdef FLANG_GEN_LLVM_ATOMIC_INTRINSICS
static OPERAND *
gen_llvm_atomic_intrinsic_for_builtin(int pdnum, int sptr, int ilix,
                                      INSTR_LIST *Call_Instr)
{
  OPERAND *operand;
  int call_sptr = sptr;
  LL_Type *op_type;
  char routine_name[MAXIDLEN];
  DTYPE base_dtype;
  int first_arg_ili;
  bool incdec = false;
  int arg_ili = ILI_OPND(ilix, 2);
  DTYPE call_dtype = DTYPEG(call_sptr);
  DTYPE return_dtype = DTyReturnType(call_dtype);

  switch (pdnum) {
  default:
    assert(0, "gen_llvm_atomic_intrinsic_for_builtin(): invalid pdnum value ",
           pdnum, ERR_Fatal);
  }
  base_dtype = return_dtype;
  first_arg_ili = ILI_OPND(arg_ili, 1);
  switch (DTY(base_dtype)) {
  case TY_BINT:
    strcat(routine_name, "i8.p0i8");
    break;
  case TY_USINT:
    strcat(routine_name, "i16.p0i16");
    break;
  case TY_SINT:
    strcat(routine_name, "i16.p0i16");
    break;
  case TY_UINT:
  case TY_INT:
    strcat(routine_name, "i32.p0i32");
    break;
  case TY_INT8:
  case TY_UINT8:
    strcat(routine_name, "i64.p0i64");
    break;
  case TY_REAL:
    return NULL;
  default:
    assert(0,
           "gen_llvm_atomic_intrinsic_for_builtin(): invalid base type for "
           "call to sptr",
           sptr, ERR_Fatal);
  }
  op_type = make_lltype_from_dtype(cg_get_type(2, TY_PTR, return_dtype));
  operand = gen_llvm_expr(first_arg_ili, op_type);
  op_type = make_lltype_from_dtype(return_dtype);
  if (incdec) {
    operand->next = gen_llvm_expr(ad_icon(1), op_type);
  } else {
    int next_arg_ili = ILI_OPND(arg_ili, 2);
    operand->next = gen_llvm_expr(ILI_OPND(next_arg_ili, 1), op_type);
    next_arg_ili = ILI_OPND(next_arg_ili, 2);
    if (ILI_OPC(next_arg_ili) != IL_NULL) {
      int next = ILI_OPND(next_arg_ili, 1);
      operand->next->next = gen_llvm_expr(next, op_type);
    }
  }

  return gen_llvm_atomicrmw_instruction(ilix, pdnum, operand, return_dtype);
}
#endif

static OPERAND *
gen_call_vminmax_intrinsic(int ilix, OPERAND *op1, OPERAND *op2)
{
  DTYPE vect_dtype;
  int vect_size;
  int type_size;
  char sign = 'u';
  char type = 'i';
  const char *mstr = "maxnum";
  static char buf[MAXIDLEN];

  if (ILI_OPC(ilix) == IL_VMIN) {
    mstr = "minnum";
  } else {
    assert(ILI_OPC(ilix) == IL_VMAX, "gen_call_vminmax_intrinsic(): bad opc",
           ILI_OPC(ilix), ERR_Fatal);
  }
  vect_dtype = ILI_DTyOPND(ilix, 3);
  vect_size = DTyVecLength(vect_dtype);
  switch (DTY(DTySeqTyElement(vect_dtype))) {
  case TY_FLOAT:
  case TY_DBLE:
    type = 'f';
    FLANG_FALLTHROUGH;
  case TY_INT:
    sign = 's';
    FLANG_FALLTHROUGH;
  case TY_UINT:
    if (vect_size != 2 && vect_size != 4 && vect_size != 8 && vect_size != 16)
      return NULL;
    break;
  case TY_SINT:
    sign = 's';
    FLANG_FALLTHROUGH;
  case TY_USINT:
    if (vect_size != 4 && vect_size != 8 && vect_size != 16)
      return NULL;
    break;
  case TY_BINT:
    sign = 's';
    return NULL;
    break;
  default:
    return NULL;
  }
  op1->next = op2;
  type_size = zsize_of(DTySeqTyElement(vect_dtype)) * BITS_IN_BYTE;
  sprintf(buf, "@llvm.%s.v%d%c%d", mstr, vect_size, type, type_size);
  return gen_call_to_builtin(ilix, buf, op1, make_lltype_from_dtype(vect_dtype),
                             NULL, I_PICALL, InstrListFlagsNull, 0);
}

#if defined(TARGET_LLVM_POWER)
static OPERAND *
gen_call_vminmax_power_intrinsic(int ilix, OPERAND *op1, OPERAND *op2)
{
  DTYPE vect_dtype;
  int vect_size; /* number of elements per vector */
  int type_size;
  char *type = "sp";
  const char *mstr = "max";
  static char buf[MAXIDLEN];

  if (ILI_OPC(ilix) == IL_VMIN)
    mstr = "min";
  vect_dtype = ILI_DTyOPND(ilix, 3);
  vect_size = DTyVecLength(vect_dtype);
  if (vect_size != 2 && vect_size != 4)
    return NULL;

  if (vect_size == 2)
    type = "dp";

  switch (DTY(DTySeqTyElement(vect_dtype))) {
  case TY_FLOAT:
  case TY_DBLE:
    break;
  default:
    return NULL;
  }
  op1->next = op2;
  type_size = zsize_of(DTySeqTyElement(vect_dtype)) * BITS_IN_BYTE;
  sprintf(buf, "@llvm.ppc.vsx.xv%s%s", mstr, type);
  return gen_call_to_builtin(ilix, buf, op1, make_lltype_from_dtype(vect_dtype),
                             NULL, I_PICALL, InstrListFlagsNull, 0);
}
#endif

#if defined(TARGET_LLVM_ARM) && NEON_ENABLED
static OPERAND *
gen_call_vminmax_neon_intrinsic(int ilix, OPERAND *op1, OPERAND *op2)
{
  DTYPE vect_dtype;
  int vect_size;
  int type_size;
  char sign = 'u';
  char type = 'i';
  char *mstr = "vmax";
  static char buf[MAXIDLEN];

  if (!NEON_ENABLED)
    return NULL;
  if (ILI_OPC(ilix) == IL_VMIN)
    mstr = "vmin";
  vect_dtype = (DTYPE)ILI_OPND(ilix, 3);
  vect_size = DTyVecLength(vect_dtype);
  switch (DTY(DTySeqTyElement(vect_dtype))) {
  case TY_FLOAT:
    type = 'f';
  case TY_INT:
    sign = 's';
  case TY_UINT:
    if (vect_size != 2 && vect_size != 4)
      return NULL;
    break;
  case TY_SINT:
    sign = 's';
  case TY_USINT:
    if (vect_size != 4 && vect_size != 8)
      return NULL;
    break;
  case TY_BINT:
    sign = 's';
  default:
    return NULL;
  }
  op1->next = op2;
  type_size = zsize_of(DTySeqTyElement(vect_dtype)) * BITS_IN_BYTE;
  sprintf(buf, "@llvm.arm.neon.%s%c.v%d%c%d", mstr, sign, vect_size, type,
          type_size);
  return gen_call_to_builtin(ilix, buf, op1, make_lltype_from_dtype(vect_dtype),
                             NULL, I_PICALL, InstrListFlagsNull, 0);
}
#endif

/* If the function being called is __builtin_alloca(n), generate an alloca
   instruction.  Otherwise, do nothing. */
static OPERAND *
gen_alloca_call_if_necessary(SPTR sptr, int ilix)
{
  if (call_sym_is(sptr, "__builtin_alloca")) {
    if (size_of(DT_CPTR) == 8)
      return gen_llvm_instr(ilix, IL_ARGKR, make_lltype_from_dtype(DT_CPTR),
                            make_lltype_from_dtype(DT_INT8), I_ALLOCA);
    return gen_llvm_instr(ilix, IL_ARGIR, make_lltype_from_dtype(DT_CPTR),
                          make_lltype_from_dtype(DT_INT), I_ALLOCA);
  }
  return NULL;
}

static OPERAND *
gen_unreachable_if_necessary(SPTR sptr, int ilix)
{
  if (call_sym_is(sptr, "__builtin_unreachable")) {
    ad_instr(ilix, gen_instr(I_UNREACH, NULL, NULL, NULL));
    return make_undef_op(make_void_lltype());
  }
  return NULL;
}

OPERAND *
gen_call_as_llvm_instr(SPTR sptr, int ilix)
{
  OPERAND *special_call;
  special_call = gen_alloca_call_if_necessary(sptr, ilix);
  if (special_call == NULL) {
    special_call = gen_unreachable_if_necessary(sptr, ilix);
  }
  return special_call;
}

static bool
cgmain_init_call(int sptr)
{
  return sptr && (strncmp(SYMNAME(sptr), "__c_bzero", 9) == 0);
}

DTYPE
msz_dtype(MSZ msz)
{
  switch (msz) {
  case MSZ_SBYTE:
    return DT_BINT;
  case MSZ_SHWORD:
    return DT_SINT;
  case MSZ_SWORD:
    return DT_INT;
  case MSZ_SLWORD:
    return DT_INT;
  case MSZ_BYTE:
    return DT_BINT;
  case MSZ_UHWORD:
    return DT_USINT;
  case MSZ_UWORD:
    return DT_UINT;
  case MSZ_ULWORD:
    return DT_INT;
  case MSZ_FWORD:
    return DT_FLOAT;
  case MSZ_DFLWORD:
    return DT_DBLE;
  case MSZ_I8:
    return DT_INT8;
  case MSZ_PTR:
    return DT_CPTR;
  case MSZ_F16:
#if defined(LONG_DOUBLE_FLOAT128)
    return DT_FLOAT128;
#elif defined(TARGET_LLVM_X8664)
    return DT_128;
#else
    return DT_QUAD;
#endif
  case MSZ_F32:
    return DT_256;
  default:
    assert(0, "msz_dtype, bad value", msz, ERR_Fatal);
  }
  return DT_NONE;
}

/* Begin define calling conventions */
#define CALLCONV                         \
  PRESENT(cc_default, ""),               \
    PRESENT(arm_aapcscc, "arm_aapcscc"), \
      PRESENT(arm_aapcs_vfpcc, "arm_aapcs_vfpcc")

#define PRESENT(x, y) x
enum calling_conventions { CALLCONV };
#undef PRESENT

#define PRESENT(x, y) y
const char *cc_as_str[] = {CALLCONV};
#undef PRESENT

#undef CALLCONV
/* End define calling conventions */

/**
   \brief Create and append a !dbg info metadata from \p module
   \param module   The module from which to get \c debug_info
 */
static void
emit_dbg_from_module(LL_Module *module)
{
  const LL_MDRef linemd = lldbg_cons_line(module->debug_info);
  if (!LL_MDREF_IS_NULL(linemd)) {
    print_dbg_line(linemd);
  }
}

static LL_Type *
fixup_x86_abi_return(LL_Type *sig)
{
  LL_Type *rv;
  const unsigned numArgs = sig->sub_elements;
  const bool isVarArgs = (sig->flags & LL_TYPE_IS_VARARGS_FUNC) != 0;
  LL_Type **args = (LL_Type **)malloc(numArgs * sizeof(LL_Type *));
  memcpy(args, sig->sub_types, numArgs * sizeof(LL_Type *));
  args[0] = make_lltype_from_dtype(DT_INT);
  rv = ll_create_function_type(sig->module, args, numArgs - 1, isVarArgs);
  free(args);
  return rv;
}

#if defined(TARGET_LLVM_X8664)
LL_Type *
maybe_fixup_x86_abi_return(LL_Type *sig)
{
  if (!XBIT(183, 0x400000) && (sig->data_type == LL_PTR)) {
    LL_Type *pt = sig->sub_types[0];
    if (pt->data_type == LL_FUNCTION) {
      LL_Type *rt = pt->sub_types[0];
      if (rt->data_type == LL_I16)
        return ll_get_pointer_type(fixup_x86_abi_return(pt));
    }
  }
  return sig;
}
#endif

/**
 * \brief write \c I_CALL instruction
 * \param curr_instr  pointer to current instruction instance
 * \param emit_func_signature_for_call
 * \return 1 if debug op was written, 0 otherwise
 */
static int
write_I_CALL(INSTR_LIST *curr_instr, bool emit_func_signature_for_call)
{
  /* Function invocation description as a list of OPERAND values */
  int i_name = curr_instr->i_name;
  /* get the return type of the call */
  LL_Type *return_type = curr_instr->ll_type;
  /* Get invocation description */
  OPERAND *call_op = curr_instr->operands;
  /* Debug has not been printed yet */
  bool dbg_line_op_written = false;
  bool routine_label_written = false;
  /* Start with default calling conventions */
  bool callRequiresTrunc = false;
  bool simple_callee = true;
  LL_Type *callee_type = call_op->ll_type;
  int sptr;
  char callRequiresTruncName[32];

  /* operand pattern:
   *   result (optional - only if return type of call not null)
   *   if var_args need to provide call signature
   *   call sptr (if null return type, this is the first operand)
   *   zero or more operands for the call arguments
   */
  print_token("\t");
#if defined(TARGET_LLVM_X8664)
  /* by default on X86-64, a function returning INTEGER*2 is promoted to return INTEGER*4
     and the return value truncated.*/
  if (return_type->data_type == LL_I16) {
      callRequiresTrunc = !XBIT(183, 0x400000);
  }
#endif
  assert(return_type, "write_I_CALL: missing return type for call instruction",
         0, ERR_Fatal);
  assert(call_op, "write_I_CALL: missing operand for call instruction", 0,
         ERR_Fatal);

  /* The callee is either a function pointer (before LLVM 3.7) or a
   * function (3.7).
   *
   * We don't have to print the entire callee type unless it is a varargs
   * function or a function returning a function pointer.  In the common case,
   * print the function return type instead of the whole function type. LLVM
   * will infer the rest from the arguments.
   *
   * FIXME: We still generate function calls with bad callee types that
   * are not function pointers:
   * - gen_call_to_builtin()
   * - gen_va_start()
   */

  /* This should really be an assertion, see above: */
  if (ll_type_is_pointer_to_function(callee_type)) {
    callee_type = callee_type->sub_types[0];

    /* Varargs callee => print whole function pointer type. */
    if (callee_type->flags & LL_TYPE_IS_VARARGS_FUNC)
      simple_callee = false;
    /* Call returns pointer to function => print whole type. */
    if (ll_type_is_pointer_to_function(return_type))
      simple_callee = false;
  }

  if (return_type->data_type != LL_VOID) {
    if (callRequiresTrunc) {
      snprintf(callRequiresTruncName, 32, "%%call.%d", expr_id);
      print_token(callRequiresTruncName);
    } else {
      print_tmp_name(curr_instr->tmps);
    }
    print_token(" = ");
  }
  print_token(llvm_instr_names[i_name]);
  print_space(1);

  if ((!flg.ieee || XBIT(216, 1)) && (curr_instr->flags & FAST_MATH_FLAG))
    print_token("fast ");
  if (!XBIT(216, 1)) {
    if (curr_instr->flags & NSZ_MATH_FLAG)
      print_token("nsz ");
    if (curr_instr->flags & REASSOC_MATH_FLAG)
      print_token("reassoc ");
  }

  /* Print calling conventions */
  if (curr_instr->flags & CALLCONV_MASK) {
    enum LL_CallConv cc = (enum LL_CallConv)(
        (curr_instr->flags & CALLCONV_MASK) >> CALLCONV_SHIFT);
    print_token(ll_get_calling_conv_str(cc));
    print_space(1);
  }

  sptr = call_op->val.sptr;
  /* write out call signature if var_args */
  if (curr_instr->flags & FAST_CALL) {
    print_token("fastcc ");
  }

  if (simple_callee) {
    LL_Type *retTy = return_type;
    /* In simple case it is sufficient to write just the return type */
    if (callRequiresTrunc)
      retTy = make_lltype_from_dtype(DT_INT);
    write_type(retTy);
  } else {
    LL_Type *sig =
        emit_func_signature_for_call ? callee_type : call_op->ll_type;
    if (callRequiresTrunc)
      sig = fixup_x86_abi_return(sig);
    /* Write out either function type or pointer type for the callee */
    write_type(sig);
  }
  print_space(1);

    if (!routine_label_written)
      write_operand(call_op, " (", FLG_OMIT_OP_TYPE);
    write_operands(call_op->next, 0);
    /* if no arguments, write out the parens */
    print_token(")");
  if (callRequiresTrunc) {
    print_dbg_line(curr_instr->dbg_line_op);
    print_token("\n\t");
    print_tmp_name(curr_instr->tmps);
    print_token(" = trunc i32 ");
    print_token(callRequiresTruncName);
    print_token(" to i16");
  }
  {
    const bool wrDbg = true;
    if (wrDbg && cpu_llvm_module->debug_info &&
        ll_feature_subprogram_not_in_cu(&cpu_llvm_module->ir) &&
        LL_MDREF_IS_NULL(curr_instr->dbg_line_op)) {
      /* we must emit !dbg metadata in this case */
      emit_dbg_from_module(cpu_llvm_module);
      return true;
    }
  }
  return dbg_line_op_written;
} /* write_I_CALL */

/**
   \brief Create the root and omnipotent pointer nodes of the TBAA tree

   These metadata nodes are unique per LLVM module and should be cached there.
 */
static LL_MDRef
get_omnipotent_pointer(LL_Module *module)
{
  LL_MDRef omni = module->omnipotentPtr;
  if (LL_MDREF_IS_NULL(omni)) {
    LL_MDRef s0;
    LL_MDRef r0;
    LL_MDRef a[3];
    char baseBuff[32];
    const char *baseName = "Flang FAA";
    const char *const omniName = "unlimited ptr";
    const char *const unObjName = "unref ptr";
    snprintf(baseBuff, 32, "%s %x", baseName, funcId);
    s0 = ll_get_md_string(module, baseBuff);
    r0 = ll_get_md_node(module, LL_PlainMDNode, &s0, 1);
    a[0] = ll_get_md_string(module, unObjName);
    a[1] = r0;
    a[2] = ll_get_md_i64(module, 0);
    module->unrefPtr = ll_get_md_node(module, LL_PlainMDNode, a, 3);
    a[0] = ll_get_md_string(module, omniName);
    a[1] = r0;
    a[2] = ll_get_md_i64(module, 0);
    omni = ll_get_md_node(module, LL_PlainMDNode, a, 3);
    module->omnipotentPtr = omni;
  }
  return omni;
}

static bool
assumeWillAlias(int nme)
{
  do {
    int sym = NME_SYM(nme);
    if (sym > 0) {
#if defined(VARIANTG)
      const int variant = VARIANTG(sym);
      if (variant > 0)
        sym = variant;
#endif
      if (NOCONFLICTG(sym) || CCSYMG(sym)) {
        ; /* do nothing */
#if defined(PTRSAFEG)
      } else if (PTRSAFEG(sym)) {
        ; /* do nothing */
#endif
      } else if (DTY(DTYPEG(sym)) == TY_PTR) {
        return true;
#if defined(POINTERG)
      } else if (POINTERG(sym)) {
        return true;
#endif
      }
    }
    switch (NME_TYPE(nme)) {
    default:
      return false;
    case NT_MEM:
    case NT_IND:
    case NT_ARR:
    case NT_SAFE:
      nme = NME_NM(nme);
      break;
    }
  } while (nme != 0);
  return false;
}

/**
   \brief Fortran location set to "TBAA" translation

   In Fortran, there isn't any TBAA. But we use the LLVM mechanism to hint to
   the backend what may alias.
 */
static LL_MDRef
locset_to_tbaa_info(LL_Module *module, LL_MDRef omniPtr, int ilix)
{
  const int NAME_SZ = 32;
  char name[NAME_SZ];
  LL_MDRef a[3];
  int bsym, rv;
  const ILI_OP opc = ILI_OPC(ilix);
  const ILTY_KIND ty = IL_TYPE(opc);
  const int nme = ILI_OPND(ilix, (ty == ILTY_LOAD) ? 2 : 3);
  const int base = basenme_of(nme);

  if (!base)
    return omniPtr;

  bsym = NME_SYM(base);
  switch (STYPEG(bsym)) {
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
    /* do nothing */
    break;
  default:
    return module->unrefPtr;
  }

  if (!strncmp(SYMNAME(bsym), "reshap$r", 8))
    return LL_MDREF_ctor(0, 0);
  if ((NME_SYM(nme) != bsym) && assumeWillAlias(nme))
    return omniPtr;

#if defined(REVMIDLNKG)
  if (REVMIDLNKG(bsym)) {
    const int ptr = REVMIDLNKG(bsym);
    if (!NOCONFLICTG(ptr) && !PTRSAFEG(ptr) && !TARGETG(ptr))
      return LL_MDREF_ctor(0, 0);
    bsym = ptr;
  }
#endif

  if (NOCONFLICTG(bsym) || CCSYMG(bsym)) {
    ; /* do nothing */
#if defined(PTRSAFEG)
  } else if (PTRSAFEG(bsym)) {
    ; /* do nothing */
#endif
  } else if (DTY(DTYPEG(bsym)) == TY_PTR) {
    return omniPtr;
#if defined(POINTERG)
  } else if (POINTERG(bsym)) {
    return omniPtr;
#endif
  }

#if defined(SOCPTRG)
  if (SOCPTRG(bsym)) {
    int ysoc = SOCPTRG(bsym);
    while (SOC_NEXT(ysoc))
      ysoc = SOC_NEXT(ysoc);
    rv = snprintf(name, NAME_SZ, "s%x.%x", funcId, ysoc);
    DEBUG_ASSERT(rv < NAME_SZ, "buffer overrun");
    a[0] = ll_get_md_string(module, name);
    a[1] = omniPtr;
    a[2] = ll_get_md_i64(module, 0);
    return ll_get_md_node(module, LL_PlainMDNode, a, 3);
  }
#endif
  /* variable can't alias type-wise. It's Fortran! */
  rv = snprintf(name, NAME_SZ, "t%x.%x", funcId, base);
  DEBUG_ASSERT(rv < NAME_SZ, "buffer overrun");
  a[0] = ll_get_md_string(module, name);
  a[1] = omniPtr;
  a[2] = ll_get_md_i64(module, 0);
  return ll_get_md_node(module, LL_PlainMDNode, a, 3);
}

/**
   \brief Write TBAA metadata for the address \p opnd
   \param module  The module
   \param opnd  a pointer to a typed location
   \param isVol   Is this a volatile access?

   To do this correctly for C, we have use the effective type.
 */
static LL_MDRef
get_tbaa_metadata(LL_Module *module, int ilix, OPERAND *opnd, bool isVol)
{
  LL_MDRef a[3];
  LL_MDRef myPtr, omniPtr;
  LL_Type *ty;

  ty = opnd->ll_type;
  assert(ty->data_type == LL_PTR, "must be a ptr", ty->data_type, ERR_Fatal);
  omniPtr = get_omnipotent_pointer(module);

  /* volatile memory access aliases all */
  if (isVol)
    return omniPtr;

  ty = ty->sub_types[0];
  assert(ty->data_type != LL_NOTYPE, "must be a type", 0, ERR_Fatal);

  myPtr = locset_to_tbaa_info(module, omniPtr, ilix);

  if (!myPtr)
    return myPtr;

  a[0] = a[1] = myPtr;
  a[2] = ll_get_md_i64(module, 0);
  return ll_get_md_node(module, LL_PlainMDNode, a, 3);
}

/**
   \brief Is TBAA disabled?
 */
INLINE static bool
tbaa_disabled(void)
{
#ifdef OMP_OFFLOAD_LLVM
  /* Always disable tbaa for device code. */
  if (ISNVVMCODEGEN)
    return true;
#endif
  return (flg.opt < 2) || XBIT(183, 0x20000);
}

/**
   \brief Write out the TBAA metadata, if needed
 */
static void
write_tbaa_metadata(LL_Module *mod, int ilix, OPERAND *opnd, int flags)
{
  if (!tbaa_disabled()) {
    const bool isVol = (flags & VOLATILE_FLAG) != 0;
    LL_MDRef md = get_tbaa_metadata(mod, ilix, opnd, isVol);
    if (!LL_MDREF_IS_NULL(md)) {
      print_token(", !tbaa ");
      write_mdref(gbl.asmfil, mod, md, 1);
    }
  }
}

/**
   \brief Test for improperly constructed instruction streams
   \param insn   The instruction under the cursor
   \return true  iff we don't need to emit a dummy label
 */
INLINE static int
dont_force_a_dummy_label(INSTR_LIST *insn)
{
  const int i_name = insn->i_name;
  if (i_name == I_NONE) {
    /* insn is a label, no need for a dummy */
    return true;
  }
  if ((i_name == I_BR) && insn->next && (insn->next->i_name == I_NONE)) {
    /* odd case: two terminators appearing back-to-back followed by a
       label. write_instructions() will skip over this 'insn' and
       emit the next one. Don't emit two labels back-to-back. */
    return true;
  }
  return false;
}

/**
   \brief For the given instruction, write [singlethread] <memory ordering>
   to the LLVM IR output file.
 */
static void
write_memory_order(INSTR_LIST *instrs)
{
  if (instrs->flags & ATOMIC_SINGLETHREAD_FLAG) {
    print_token(" singlethread");
  }
  print_space(1);
  print_token(get_atomic_memory_order_name(instrs->flags));
}

/**
   \brief For the given instruction, write [singlethread] <memory ordering>,
   <alignment> to the LLVM IR output file.
 */
static void
write_memory_order_and_alignment(INSTR_LIST *instrs)
{
  int align;
  DEBUG_ASSERT(
      instrs->i_name == I_LOAD || instrs->i_name == I_STORE,
      "write_memory_order_and_alignment: not a load or store instruction");

  /* Print memory order if instruction is atomic. */
  if (instrs->flags & ATOMIC_MEM_ORD_FLAGS) {
    write_memory_order(instrs);
  } else {
    DEBUG_ASSERT(
        (instrs->flags & ATOMIC_SINGLETHREAD_FLAG) == 0,
        "write_memory_order_and_alignment: inappropriate singlethread");
  }

  /* Extract the alignment in bytes from the flags field. It's stored as
   * log2(bytes). */
  align = LDST_BYTEALIGN(instrs->flags);
  if (align) {
    char align_token[4];
    print_token(", align ");
    sprintf(align_token, "%d", align);
    print_token(align_token);
  }
}

INLINE static void
write_llaccgroup_metadata(LL_Module *module, INSTR_LIST *insn,
                          LL_InstrListFlags flag_check, unsigned int val)
{
  if (insn->flags & flag_check) {
    char buf[64];
    int n;
    DEBUG_ASSERT(insn->misc_metadata, "missing metadata");
    n = snprintf(buf, 64, ", !llvm.access.group !%u",
                 LL_MDREF_value(val));
    DEBUG_ASSERT(n < 64, "buffer overrun");
    print_token(buf);
  }
}

/* write out the struct member types */
static void
write_verbose_type(LL_Type *ll_type)
{
  print_token(ll_type->str);
}

/* whether debug location should be suppressed */
static bool
should_suppress_debug_loc(INSTR_LIST *instrs)
{
  if (!instrs)
    return false;

  // return true if not a call instruction
  switch (instrs->i_name) {
  case I_INVOKE:
    return false;
  case I_CALL:
    // f90 runtime functions fort_init and f90_* dont need debug location
    if (instrs->prev && (instrs->operands->ot_type == OT_TMP) &&
        (instrs->operands->tmps == instrs->prev->tmps) &&
        (instrs->prev->operands->ot_type == OT_VAR)) {
      // We dont need to expose those internals in prolog to user
      // %1 = bitcast void (...)* @fort_init to void (i8*, ...)*
      // call void (i8*, ...) %1(i8* %0)
      // %8 = bitcast void (...)* @f90_template1_i8 to void (i8*, i8*, i8*, i8*,
      //      i8*, i8*, ...)*
      // call void (i8*, i8*, i8*, i8*, i8*, i8*, ...) %8(i8*
      //      %2, i8* %3, i8* %4, i8* %5, i8* %6, i8* %7)

      if (const char *name_str = instrs->prev->operands->string) {
        return (!strncmp(name_str, "@fort_init", strlen("@fort_init")) ||
                !strncmp(name_str, "@f90_", strlen("@f90_")));
      }
    }
    return false;
  default:
    return true;
  }
}

/**
   \brief Write the instruction list to the LLVM IR output file
 */
static void
write_instructions(LL_Module *module)
{
  INSTR_LIST *instrs;
  OPERAND *p, *p1;
  LL_InstrName i_name;
  SPTR sptr;
  bool forceLabel = true;
  bool dbg_line_op_written;
  bool ret_scalar;
  int entry;

  DBGTRACEIN("")

  for (instrs = Instructions; instrs; instrs = instrs->next) {
    llvm_info.curr_instr = instrs;
    i_name = instrs->i_name;
    dbg_line_op_written = false;

    asrt(i_name >= 0 && i_name < I_LAST);
    DBGTRACE3("#instruction(%d) %s for ilix %d\n", i_name,
              llvm_instr_names[i_name], instrs->ilix);

    if (dont_force_a_dummy_label(instrs))
      forceLabel = false;
    if (forceLabel) {
      char buff[32];
      static unsigned counter = 0;
      snprintf(buff, 32, "L.dead%u:\n", counter++);
      print_token(buff);
      forceLabel = false;
    }
    if (instrs->flags & CANCEL_CALL_DBG_VALUE) {
      DBGTRACE("#instruction llvm.dbg.value canceled")
      continue;
    } else if (BINOP(i_name) || BITOP(i_name)) {
      print_token("\t");
      print_tmp_name(instrs->tmps);
      print_token(" = ");
      print_token(llvm_instr_names[i_name]);
      if ((!flg.ieee) || XBIT(216, 1))
        switch (i_name) {
        case I_FADD:
          if (!XBIT(216, 2))
            print_token(" fast");
          break;
        case I_FDIV:
          if (!XBIT(216, 4))
            print_token(" fast");
          break;
        case I_FSUB:
        case I_FMUL:
        case I_FREM:
          print_token(" fast");
          break;
        default:
          break;
        }
      if (!XBIT(216, 1))
        switch (i_name) {
        case I_FADD:
        case I_FDIV:
        case I_FSUB:
        case I_FMUL:
        case I_FREM:
          if (XBIT(216, 0x8))
            print_token(" nsz");
          if (XBIT(216, 0x10))
            print_token(" reassoc");
          break;
        default:
          break;
        }
      p = instrs->operands;
      assert(p->ll_type, "write_instruction(): missing binary type", 0,
             ERR_Fatal);
      asrt(match_types(instrs->ll_type, p->ll_type) == MATCH_OK);
      print_space(1);
      /* write_type(p->ll_type); */
      write_type(instrs->ll_type);
      print_space(1);
      write_operand(p, ", ", FLG_OMIT_OP_TYPE);
      p = p->next;
      assert(p->ll_type, "write_instruction(): missing binary type", 0,
             ERR_Fatal);
      asrt(match_types(instrs->ll_type, p->ll_type) == MATCH_OK);
      write_operand(p, "", FLG_OMIT_OP_TYPE);
    } else if (CONVERT(i_name)) {
      p = instrs->operands;
      assert(p->next == NULL, "write_instructions(),bad next ptr", 0,
             ERR_Fatal);
      print_token("\t");
      print_tmp_name(instrs->tmps);
      print_token(" = ");
      print_token(llvm_instr_names[i_name]);
      print_space(1);
#if defined(PGFTN) && defined(TARGET_LLVM_X8664)
      write_operand(p, " to ", FLG_FIXUP_RETURN_TYPE);
      write_type(maybe_fixup_x86_abi_return(instrs->ll_type));
#else
      write_operand(p, " to ", 0);
      write_type(instrs->ll_type);
#endif
    } else {
      switch (i_name) {
      case I_NONE: /* should be a label */
        forceLabel = false;
        sptr = instrs->operands->val.sptr;
        if (instrs->prev == NULL && sptr == 0) {
          /* entry label we just ignore it*/
          break;
        }
        assert(sptr, "write_instructions(): missing symbol", 0, ERR_Fatal);
        if (sptr != instrs->operands->val.sptr)
          printf("sptr mixup sptr= %d, val = %d\n", sptr,
                 instrs->operands->val.sptr);
        /* Every label must be immediately preceded by a branch or other
           terminal instruction. */
        if (!INSTR_PREV(instrs) || !INSTR_IS_TERMINAL(INSTR_PREV(instrs))) {
          print_token("\t");
          print_token(llvm_instr_names[I_BR]);
          print_token(" label %L");
          print_token(get_llvm_name(sptr));
          print_nl();
        }

        write_operand(instrs->operands, "", 0);
        /* if label is last instruction in the module we need
         * a return instruction as llvm requires a termination
         * instruction at the end of the block.
         */
        if (!instrs->next) {
          print_nl();
          print_token("\t");
          print_token(llvm_instr_names[I_RET]);
          print_space(1);
          if (has_multiple_entries(gbl.currsub)) {
            if (gbl.arets)
              llvm_info.return_ll_type = make_lltype_from_dtype(DT_INT);
            else
              llvm_info.return_ll_type = make_lltype_from_dtype(DT_NONE);
          }
          write_type(llvm_info.abi_info->extend_abi_return
                         ? make_lltype_from_dtype(DT_INT)
                         : llvm_info.return_ll_type);
          if (llvm_info.return_ll_type->data_type != LL_VOID) {
            switch (llvm_info.return_ll_type->data_type) {
            case LL_PTR:
              print_token(" null");
              break;
            case LL_I1:
            case LL_I8:
            case LL_I16:
            case LL_I24:
            case LL_I32:
            case LL_I40:
            case LL_I48:
            case LL_I56:
            case LL_I64:
            case LL_I128:
            case LL_I256:
              print_token(" 0");
              break;
            case LL_DOUBLE:
            case LL_FLOAT:
              print_token(" 0.0");
              break;
            case LL_X86_FP80:
              print_token(" 0xK00000000000000000000");
              break;
            case LL_FP128:
              print_token(" 0xL00000000000000000000000000000000");
              break;
            case LL_PPC_FP128:
              print_token(" 0xM00000000000000000000000000000000");
              break;
            default:
              print_token(" zeroinitializer");
            }
          }
        }
        break;
      case I_EXTRACTVAL:
      case I_INSERTVAL: {
        /* extractvalue lhs, rhs, int
         * lhs = extractvalue rhs_type rhs, int
         * lhs = insertvalue rhs_type rhs, int
         */
        OPERAND *cc = instrs->operands;
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_verbose_type(instrs->ll_type);
        print_space(1);
        write_operand(cc, ", ", FLG_OMIT_OP_TYPE);
        cc = cc->next;
        if (i_name == I_INSERTVAL) {
          write_operand(cc, ", ", 0);
          cc = cc->next;
        }
        write_operand(cc, "", FLG_OMIT_OP_TYPE);
      } break;
      case I_RESUME: {
        /* resume { i8*, i32 } %33 */
        OPERAND *cc;
        forceLabel = true; // is needed here
        cc = instrs->operands;
        print_token("\t");
        print_token(llvm_instr_names[I_RESUME]);
        print_space(1);
        write_verbose_type(cc->ll_type);
        print_space(1);
        write_operand(cc, " ", FLG_OMIT_OP_TYPE);
      } break;
      case I_CLEANUP:
        print_token("\t");
        print_token(llvm_instr_names[I_CLEANUP]);
        break;
      case I_LANDINGPAD:
        /* landingpad: typeinfo_var, catch_clause_sptr,
         * caught_object_sptr
         */
        /* LABEL */
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[I_LANDINGPAD]);
        print_space(1);
        write_verbose_type(instrs->ll_type);
        if (ll_feature_eh_personality_on_landingpad(&module->ir))
          print_personality();
        dbg_line_op_written = true;
        break;
      case I_CATCH: {
        OPERAND *cc;
        cc = instrs->operands;

        print_token("\tcatch ptr ");
        write_operand(cc, " ", FLG_OMIT_OP_TYPE);
      } break;
      case I_FILTER: {
        /* "filter <array-type> [ <array-of-typeinfo-vars> ]"
           Each operand is a typeinfo variable for a type in the exception
           specification. */
        if (instrs->operands == NULL) {
          /* A no-throw exception spec, "throw()" */
          /* LLVM documentation says that "filter [0xi8*] undef" is fine, but
             the LLVM compiler rejects it.  So we have to do it differently. */
          print_token("\t\tfilter [0 x ptr] zeroinitializer");
        } else {
          OPERAND *esti;       /* One typeinfo var for the exception spec. */
          int count = 0;       /* Number of types in the exception spec. */
          char buffer[19 + 9]; /* Format string + small integer */
          for (esti = instrs->operands; esti != NULL; esti = esti->next) {
            ++count;
          }
          snprintf(buffer, sizeof buffer, "\tfilter [%d x ptr] [", count);
          print_token(buffer);
          for (esti = instrs->operands; esti != NULL; esti = esti->next) {
            print_token("ptr ");
            write_operand(esti, NULL, FLG_OMIT_OP_TYPE);
            if (esti->next != NULL) {
              print_token(", ");
            }
          }
          print_token("]");
        }
        break;
      }
      case I_INVOKE:
      /* forceLabel = true; is not needed here, already handled */
      case I_PICALL:
      case I_CALL:
        dbg_line_op_written = write_I_CALL(
            instrs, ll_feature_emit_func_signature_for_call(&module->ir));
        break;
      case I_SW:
        forceLabel = true;
        p = instrs->operands;
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_operand(p, ", ", 0);
        write_operand(p->next, "[\n\t\t", 0);
        p1 = p->next->next;
        while (p1) {
          write_operand(p1, ", ", 0);
          p1 = p1->next;
          if (p1) {
            write_operand(p1, "\n\t\t", 0);
            p1 = p1->next;
          }
        }
        print_token("]");
        break;
      case I_RET:
        forceLabel = true;
        p = instrs->operands;
        if (XBIT(129, 0x800)) {
          /* -finstrument-functions */
          write_profile_exit();
        }
        ret_scalar = false;
        for (entry = gbl.currsub; entry > NOSYM; entry = SYMLKG(entry)) {
          int fval = FVALG(entry);
          if(fval && SCG(fval) != SC_DUMMY && SCG(fval) != SC_BASED) {
            ret_scalar = true;
            break;
          }
        }
        /* This is a way to return value for multiple entries with return type
         * pass as argument to the master/common routine */
        if (has_multiple_entries(gbl.currsub) && ret_scalar) {
          /* (1) bitcast result(second argument) from i8* to type of p->ll_type
           * (2) store result into (1)
           * (3) return void.
           */
          store_return_value_for_entry(p, i_name);
          break;
        }
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_type(llvm_info.abi_info->extend_abi_return
                       ? make_lltype_from_dtype(DT_INT)
                       : llvm_info.return_ll_type);
        /*  If a function return type is VOID, we don't have to
         *  append any operands after LLVM instruction "ret void" */
        if (llvm_info.return_ll_type->data_type != LL_VOID &&
            (p->ot_type != OT_NONE) && (p->ll_type->data_type != LL_VOID)) {
          print_space(1);
          write_operand(p, "", FLG_OMIT_OP_TYPE);
          assert(p->next == NULL, "write_instructions(), bad next ptr", 0,
                 ERR_Fatal);
        }
        break;
      case I_LOAD:
        p = instrs->operands;
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        if (instrs->flags & ATOMIC_MEM_ORD_FLAGS) {
          print_token("atomic ");
        }
        if (instrs->flags & VOLATILE_FLAG) {
          print_token("volatile ");
        }

        /* Print out the loaded type. */
        if (ll_feature_explicit_gep_load_type(&module->ir)) {
          LL_Type *t = p->ll_type;
          assert(t && t->data_type == LL_PTR, "load operand must be a pointer",
                 0, ERR_Fatal);
          t = t->sub_types[0];
          print_token(t->str);
          print_token(", ");
        }

        /* Print out the pointer operand. */
        write_operand(p, "", 0);

        write_memory_order_and_alignment(instrs);

        assert(p->next == NULL, "write_instructions(), bad next ptr", 0,
               ERR_Fatal);
        write_llaccgroup_metadata(module, instrs, LDST_HAS_ACCESSGRP_METADATA,
                                  cached_access_group_metadata);
        write_tbaa_metadata(module, instrs->ilix, instrs->operands,
                            instrs->flags);
        break;
      case I_STORE:
        p = instrs->operands;
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        if (instrs->flags & ATOMIC_MEM_ORD_FLAGS) {
          print_token("atomic ");
        }
        if (instrs->flags & VOLATILE_FLAG) {
          print_token("volatile ");
        }
        write_operand(p, ", ", 0);
        p = p->next;
        write_operand(p, "", 0);

        write_memory_order_and_alignment(instrs);
        write_llaccgroup_metadata(module, instrs, LDST_HAS_ACCESSGRP_METADATA,
                                  cached_access_group_metadata);
        write_tbaa_metadata(module, instrs->ilix, instrs->operands->next,
                            instrs->flags & VOLATILE_FLAG);
        break;
      case I_BR:
        if (!INSTR_PREV(instrs) || !INSTR_IS_TERMINAL(INSTR_PREV(instrs))) {
          forceLabel = true;
          print_token("\t");
          print_token(llvm_instr_names[i_name]);
          print_space(1);
          write_operands(instrs->operands, 0);
          if (instrs->flags & LOOP_BACKEDGE_FLAG) {
            char buf[32];
            LL_MDRef loop_md = instrs->misc_metadata;
            snprintf(buf, 32, ", !llvm.loop !%u", LL_MDREF_value(loop_md));
            print_token(buf);
          }
        } else {
          /* The branch is dead code.  Don't write it out.  And don't write out
             the debug information either. */
          dbg_line_op_written = true;
        }
        break;
      case I_INDBR:
        forceLabel = true;
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_operands(instrs->operands, 0);
        break;
      case I_GEP:
        p = instrs->operands;
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);

        /* Print out the indexed type. */
        if (ll_feature_explicit_gep_load_type(&module->ir)) {
          LL_Type *t = p->ll_type;
          assert(t && t->data_type == LL_PTR, "gep operand must be a pointer",
                 0, ERR_Fatal);
          t = t->sub_types[0];
          print_token(t->str);
          print_token(", ");
        }

        write_operands(p, FLG_AS_UNSIGNED);
        break;
      case I_VA_ARG:
        p = instrs->operands;
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_operand(p, "", 0);
        write_type(instrs->ll_type);
        break;
      case I_DECL:
        break;
      case I_FCMP:
      case I_ICMP:
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        if ((i_name == I_FCMP) && ((!flg.ieee) || XBIT(216, 1)) &&
            (instrs->operands->ll_type->data_type != LL_FP128))
          print_token(" fast");
        print_space(1);
        p = instrs->operands;
        write_operand(p, " ", 0);
        /* use the type of the comparison operators */
        write_type(instrs->operands->next->ll_type);
        print_space(1);
        p = p->next;
        write_operand(p, ", ", FLG_OMIT_OP_TYPE);
        p = p->next;
        write_operand(p, "", FLG_OMIT_OP_TYPE);
        break;
      case I_ALLOCA:
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_type(instrs->ll_type->sub_types[0]);
        p = instrs->operands;
        if (p) {
          print_token(", ");
          write_operand(p, "", 0);
        }
        break;
      case I_SELECT:
      case I_EXTELE:
      case I_INSELE:
      case I_SHUFFVEC:
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        p = instrs->operands;
        print_space(1);
        write_operands(p, 0);
        break;
      case I_BARRIER:
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        print_token("acq_rel");
        break;
      case I_ATOMICRMW:
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        print_token(get_atomicrmw_opname(instrs->flags));
        print_space(1);
        write_operands(instrs->operands, 0);
        write_memory_order(instrs);
        break;
      case I_CMPXCHG:
        print_token("\t");
        print_tmp_name(instrs->tmps);
        print_token(" = ");
        print_token(llvm_instr_names[i_name]);
        print_space(1);
        write_operands(instrs->operands, 0);
        print_space(1);
        print_token(get_atomic_memory_order_name(instrs->flags));
        print_space(1);
        print_token(get_atomic_memory_order_name(
            FROM_CMPXCHG_MEMORDER_FAIL(instrs->flags)));
        break;
      case I_FENCE:
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        write_memory_order(instrs);
        break;
      case I_UNREACH:
        forceLabel = true;
        print_token("\t");
        print_token(llvm_instr_names[i_name]);
        break;
      default:
        DBGTRACE1("### write_instructions(): unknown instr name: %s",
                  llvm_instr_names[i_name])
        assert(0, "write_instructions(): unknown instr name", instrs->i_name,
               ERR_Fatal);
      }
    }
    /*
     *  Do not dump debug location here if
     *  - it is NULL
     *  - it is already written (dbg_line_op_written) or
     *  - it is a known internal (f90 runtime) call in prolog (fort_init &
     * f90_*)
     */
    if (!(LL_MDREF_IS_NULL(instrs->dbg_line_op) || dbg_line_op_written ||
          ((instrs->dbg_line_op ==
            lldbg_get_subprogram_line(module->debug_info)) &&
           should_suppress_debug_loc(instrs)))) {
      print_dbg_line(instrs->dbg_line_op);
    }
#if DEBUG
    if (instrs->traceComment) {
      print_token("\t\t;) ");
      print_token(instrs->traceComment);
    }
    if (XBIT(183, 0x800)) {
      char buf[200];

      if (instrs->tmps)
        sprintf(buf, "\t\t\t; ilix %d, usect %d", instrs->ilix,
                instrs->tmps->use_count);
      else
        sprintf(buf, "\t\t\t; ilix %d", instrs->ilix);

      if (instrs->flags & DELETABLE)
        strcat(buf, " deletable");
      if (instrs->flags & STARTEBB)
        strcat(buf, " startebb");
      if (instrs->flags & ROOTDG)
        strcat(buf, " rootdg");
      print_token(buf);
      sprintf(buf, "\ti%d", instrs->rank);
      print_token(buf);
    }
#endif
    print_nl();
  }

  DBGTRACEOUT("")
} /* write_instructions */

OPERAND *
mk_alloca_instr(LL_Type *ptrTy)
{
  OPERAND *op = make_tmp_op(ptrTy, make_tmps());
  INSTR_LIST *insn = gen_instr(I_ALLOCA, op->tmps, ptrTy, NULL);
  ad_instr(0, insn);
  return op;
}

INSTR_LIST *
mk_store_instr(OPERAND *val, OPERAND *addr)
{
  INSTR_LIST *insn;
  val->next = addr;
  insn = gen_instr(I_STORE, NULL, NULL, val);
  if (rw_access_group) {
    insn->flags |= LDST_HAS_ACCESSGRP_METADATA;
    insn->misc_metadata = cons_vec_always_metadata();
  }
  ad_instr(0, insn);
  return insn;
}

#if DEBUG
void
dump_type_for_debug(LL_Type *ll_type)
{
  if (ll_type == NULL) {
    fprintf(ll_dfile, "(UNKNOWN)\n");
  }
  fprintf(ll_dfile, "%s\n", ll_type->str);
}
#endif

/* create the instructions for:
 * 	%new_tmp =  extractvalue { struct mems } %tmp, index
 *  or
 * 	%new_tmp =  insertvalue { struct mems } %tmp, tmp2_type tmp2, index

 */
TMPS *
gen_extract_insert(LL_InstrName i_name, LL_Type *struct_type, TMPS *tmp,
                   LL_Type *tmp_type, TMPS *tmp2, LL_Type *tmp2_type, int index)
{
  OPERAND *cc;
  int num[2];
  TMPS *new_tmp;
  INSTR_LIST *Curr_Instr;

  /* %new_tmp =  extractvalue { struct mems } %tmp, index */
  if (tmp)
    cc = make_tmp_op(tmp_type, tmp);
  else
    cc = make_undef_op(tmp_type);
  new_tmp = make_tmps();
  Curr_Instr = gen_instr(i_name, new_tmp, struct_type, cc);

  /* insertval requires one more temp */
  if (tmp2) {
    cc->next = make_tmp_op(tmp2_type, tmp2);
    cc = cc->next;
  }
  cc->next = make_operand();
  cc = cc->next;

  cc->ot_type = OT_CONSTSPTR;
  cc->ll_type = make_lltype_from_dtype(DT_INT);
  switch (index) {
  case 0:
    cc->val.sptr = stb.i0;
    break;
  case 1:
    cc->val.sptr = stb.i1;
    break;
  default:
    num[0] = 0;
    num[1] = index;
    cc->val.sptr = getcon(num, DT_INT);
    break;
  }

  ad_instr(0, Curr_Instr);
  return new_tmp;
}

/**
   \brief Generate an insertvalue instruction
   \param aggr    an aggregate (destination object)
   \param elem    an element (item to be inserted)
   \param index   index to insert element
 */
static OPERAND *
gen_insert_value(OPERAND *aggr, OPERAND *elem, unsigned index)
{
  aggr->next = elem;
  elem->next = make_constval32_op(index);
  return ad_csed_instr(I_INSERTVAL, 0, aggr->ll_type, aggr, InstrListFlagsNull,
                       false);
}

/**
   \brief Construct an \c INSTR_LIST object

   Initializes fields i_name (and dbg_line_op if appropriate).  Zeros the other
   fields.
 */
static INSTR_LIST *
make_instr(LL_InstrName instr_name)
{
  INSTR_LIST *iptr;

  iptr = (INSTR_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(INSTR_LIST));
  memset(iptr, 0, sizeof(INSTR_LIST));
  iptr->i_name = instr_name;
  if (flg.debug || XBIT(120, 0x1000)) {
    switch (instr_name) {
    default:
      iptr->dbg_line_op = lldbg_get_line(cpu_llvm_module->debug_info);
      break;
    case I_NONE:
    case I_DECL:
    case I_CLEANUP:
    case I_CATCH:
      break;
    }
  }
  return iptr;
} /* make_instr */

/**
   \brief Like make_instr, but also sets tmps, ll_type, and operands.
 */
static INSTR_LIST *
gen_instr(LL_InstrName instr_name, TMPS *tmps, LL_Type *ll_type,
          OPERAND *operands)
{
  INSTR_LIST *iptr;

  iptr = make_instr(instr_name);
  iptr->tmps = tmps;
  if (tmps != NULL)
    tmps->info.idef = iptr;
  iptr->ll_type = ll_type;
  iptr->operands = operands;
  return iptr;
}

INLINE static OPERAND *
convert_sint_to_float(OPERAND *convert_op, LL_Type *rslt_type)
{
  return convert_operand(convert_op, rslt_type, I_SITOFP);
}

INLINE static OPERAND *
convert_float_to_sint(OPERAND *convert_op, LL_Type *rslt_type)
{
  return convert_operand(convert_op, rslt_type, I_FPTOSI);
}

INLINE static OPERAND *
convert_uint_to_float(OPERAND *convert_op, LL_Type *rslt_type)
{
  return convert_operand(convert_op, rslt_type, I_UITOFP);
}

INLINE static OPERAND *
convert_float_to_uint(OPERAND *convert_op, LL_Type *rslt_type)
{
  return convert_operand(convert_op, rslt_type, I_FPTOUI);
}

INLINE static OPERAND *
convert_ptr_to_int(OPERAND *convert_op, LL_Type *rslt_type)
{
  return convert_operand(convert_op, rslt_type, I_PTRTOINT);
}

static OPERAND *
convert_mismatched_types(OPERAND *operand, LL_Type *expected_type, int ilix)
{
  if (ll_type_int_bits(operand->ll_type) && ll_type_int_bits(expected_type)) {
    return convert_int_size(ilix, operand, expected_type);
  } else if (expected_type->data_type == LL_PTR &&
             operand->ll_type->data_type == LL_PTR) {
    DBGDUMPLLTYPE("#adding bitcast to match expected type:", expected_type);
    return make_bitcast(operand, expected_type);
  } else if (ll_type_is_fp(expected_type) && ll_type_is_fp(operand->ll_type)) {
    return convert_float_size(operand, expected_type);
  } else if (ll_type_is_fp(operand->ll_type) &&
             ll_type_int_bits(expected_type)) {
    return convert_float_to_sint(operand, expected_type);
  } else if (ll_type_int_bits(operand->ll_type) &&
             (expected_type->data_type == LL_PTR)) {
    return convert_int_to_ptr(operand, expected_type);
  } else if ((operand->ll_type->data_type == LL_PTR) &&
             ll_type_int_bits(expected_type)) {
    return convert_ptr_to_int(operand, expected_type);
  } else if (ll_type_int_bits(operand->ll_type) &&
             (expected_type->data_type == LL_X86_FP80)) {
    assert(ll_type_bytes(operand->ll_type) <= ll_type_bytes(expected_type),
           "bitcast of int larger than long long to FP80",
           ll_type_bytes(operand->ll_type), ERR_Fatal);
    return convert_sint_to_float(operand, expected_type);
  } else if (ll_type_int_bits(operand->ll_type) &&
             ll_type_is_fp(expected_type)) {
    assert(ll_type_bytes(operand->ll_type) == ll_type_bytes(expected_type),
           "bitcast with differing sizes",
           ll_type_bytes(operand->ll_type) - ll_type_bytes(expected_type),
           ERR_Fatal);
    return make_bitcast(operand, expected_type);
  }
  assert(false, "no type conversion available", 0, ERR_Fatal);
  return operand;
}

static OPERAND *
ad_csed_instr(LL_InstrName instr_name, int ilix, LL_Type *ll_type,
              OPERAND *operands, LL_InstrListFlags flags, bool do_cse)
{
  OPERAND *operand, *new_op;
  INSTR_LIST *instr;
  if (do_cse && ENABLE_CSE_OPT && !new_ebb) {
    instr = llvm_info.last_instr;
    while (instr) {
      if (instr->i_name == instr_name) {
        operand = instr->operands;
        new_op = operands;
        while (operand && new_op) {
          if (!same_op(operand, new_op))
            break;
          operand = operand->next;
          new_op = new_op->next;
        }
        if (operand == NULL && new_op == NULL) {
          new_op = make_tmp_op(instr->ll_type, instr->tmps);
          if (instr->ll_type != ll_type) {
            new_op = convert_mismatched_types(new_op, ll_type, ilix);
          }
          return new_op;
        }
      }
      switch (instr->i_name) {
      case I_SW:
      case I_INVOKE:
      case I_CALL:
        instr = NULL;
        break;
      case I_BR:
      case I_INDBR:
      case I_NONE:
        if (!ENABLE_ENHANCED_CSE_OPT) {
          instr = NULL;
          break;
        }
        FLANG_FALLTHROUGH;
      default:
        instr = (instr->flags & STARTEBB) ? NULL : instr->prev;
        break;
      }
    }
  }
  operand = make_tmp_op(ll_type, make_tmps());
  instr = gen_instr(instr_name, operand->tmps, ll_type, operands);
  if ((instr_name == I_LOAD) && rw_access_group) {
    flags |= LDST_HAS_ACCESSGRP_METADATA;
    instr->misc_metadata = cons_vec_always_metadata();
  }
  instr->flags = flags;
  ad_instr(ilix, instr);
  return operand;
}

static void
ad_instr(int ilix, INSTR_LIST *instr)
{
  OPERAND *operand;

  if (instr == NULL)
    return;

  instr->ilix = ilix;
  DEBUG_ASSERT(instr != llvm_info.last_instr, "looped link");

  for (operand = instr->operands; operand; operand = operand->next) {
    if (operand->ot_type == OT_TMP) {
      assert(operand->tmps, "ad_instr(): missing last instruction", 0,
             ERR_Fatal);
      operand->tmps->use_count++;
    }
  }
  if (Instructions) {
    assert(llvm_info.last_instr, "ad_instr(): missing last instruction", 0,
           ERR_Fatal);
    llvm_info.last_instr->next = instr;
    instr->prev = llvm_info.last_instr;
  } else {
    assert(!llvm_info.last_instr, "ad_instr(): last instruction not NULL", 0,
           ERR_Fatal);
    Instructions = instr;
  }
  llvm_info.last_instr = instr;
  if (new_ebb) {
    instr->flags |= STARTEBB;
    new_ebb = false;
  }
}

static bool
cancel_store(int ilix, int op_ili, int addr_ili)
{
  if(!ENABLE_CSE_OPT)
    return false;
  ILI_OP op_opc = ILI_OPC(op_ili);
  bool csed = false;

  if (is_cseili_opcode(op_opc)) {
    op_ili = ILI_OPND(op_ili, 1);
    op_opc = ILI_OPC(op_ili);
    csed = true;
  }
  if (IL_TYPE(op_opc) == ILTY_LOAD) {
    bool ret_val = (ILI_OPND(op_ili, 1) == addr_ili);
    if (ret_val && csed) {
      DBGTRACE1("#store of CSE'd operand removed for ilix(%d)", ilix);
    }
    return ret_val;
  }
  return false;
}

static LL_InstrListFlags
ll_instr_flags_from_memory_order(MEMORY_ORDER mo)
{
  switch (mo) {
  default:
    assert(false,
           "ll_instr_flags_from_memory_order:"
           " unimplemented memory order",
           mo, ERR_Fatal);
    return (LL_InstrListFlags)0;
  case MO_RELAXED:
    return ATOMIC_MONOTONIC_FLAG;
  case MO_CONSUME:
  /* LLVM does not support "consume", so round up to acquire. */
  case MO_ACQUIRE:
    return ATOMIC_ACQUIRE_FLAG;
  case MO_RELEASE:
    return ATOMIC_RELEASE_FLAG;
  case MO_ACQ_REL:
    return ATOMIC_ACQ_REL_FLAG;
  case MO_SEQ_CST:
    return ATOMIC_SEQ_CST_FLAG;
  }
}

/**
  \brief From an ILI atomic instruction with a fence,
         get instruction flags for [singlethread] <memory order>.
 */
static LL_InstrListFlags
ll_instr_flags_for_memory_order_and_scope(int ilix)
{
  LL_InstrListFlags flags =
      ll_instr_flags_from_memory_order(memory_order(ilix));
  ATOMIC_INFO info = atomic_info(ilix);
  if (info.scope == SS_SINGLETHREAD)
    flags |= ATOMIC_SINGLETHREAD_FLAG;
  return flags;
}

/**
   \brief Invalidate cached sincos intrinsics on write to input expression
 */
static bool
sincos_input_uses(int ilix, int nme)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;
  const ILTY_KIND ilty = IL_TYPE(opc);
  if (ilty == ILTY_LOAD)
    return (ILI_OPND(ilix, 2) == nme);
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i)) {
      bool isUse = sincos_input_uses(ILI_OPND(ilix, i), nme);
      if (isUse)
        return true;
    }
  }
  return false;
}

/**
   \brief Remove all loads that correspond to a given NME
   \param key      an ILI value
   \param data     is NULL for a load
   \param context  the NME we want to remove
 */
static void
sincos_clear_arg_helper(hash_key_t key, hash_data_t data, void *context)
{
  const int lhs_ili = ((int *)context)[0];
  const int seek_nme = ((int *)context)[1];
  const int ilix = HKEY2INT(key);
  const int ilix_nme = ILI_OPND(ilix, 2);
  if ((ilix == lhs_ili) || ((data == NULL) && (seek_nme == ilix_nme)) ||
      sincos_input_uses(ilix, seek_nme))
    hashmap_erase(sincos_imap, key, NULL);
}

INLINE static void
sincos_clear_specific_arg(int lhs_ili, int nme)
{
  int ctxt[] = {lhs_ili, nme};
  hashmap_iterate(sincos_imap, sincos_clear_arg_helper, (void *)ctxt);
}

INLINE static OPERAND *
gen_sret_expr(int ilix, LL_Type *expected_type)
{
  OPERAND *value = gen_llvm_expr(ilix, expected_type);
  ret_info.sret_sptr = NME_SYM(ILI_OPND(ilix, 2));
  process_sptr(ret_info.sret_sptr);
  value = make_operand();
  value->ll_type = make_lltype_from_dtype(DT_NONE);
  return value;
}

static void
make_stmt(STMT_Type stmt_type, int ilix, bool deletable, SPTR next_bih_label,
          int ilt)
{
  int lhs_ili, rhs_ili, nme;
  SPTR sptr, sptr_lab;
  int ts;
  SPTR sym, pd_sym;
  DTYPE dtype;
  int to_ili, from_ili, length_ili, opnd, bytes, from_nme, cc;
  ILI_OP opc;
  TMPS *tmps;
  OPERAND *ret_op, *store_op, *op1;
  OPERAND *dst_op, *src_op, *first_label, *second_label;
  bool has_entries;
  MSZ msz;
  LL_Type *llt_expected;
  int alignment;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN2(" type: %s ilix: %d", stmt_names[stmt_type], ilix)

  curr_stmt_type = stmt_type;
  if (last_stmt_is_branch && stmt_type != STMT_LABEL) {
    sptr_lab = getlab();
    update_llvm_sym_arrays();
    make_stmt(STMT_LABEL, sptr_lab, false, SPTR_NULL, ilt);
  }
  last_stmt_is_branch = 0;
  switch (stmt_type) {
  case STMT_RET: {
    LL_Type *retTy = llvm_info.abi_info->extend_abi_return
                         ? make_lltype_from_dtype(DT_INT)
                         : llvm_info.return_ll_type;
    last_stmt_is_branch = 1;
    has_entries = has_multiple_entries(gbl.currsub);
    switch (ILI_OPC(ilix)) {
    case IL_AADD:
    case IL_ASUB:
    case IL_ACON:
    case IL_IAMV:
    case IL_KAMV:
    case IL_LDA:
      if (has_entries && !gbl.arets) {
        ret_op = gen_base_addr_operand(ilix, NULL);
      } else if (llvm_info.abi_info->is_iso_c) {
        if (currsub_is_sret()) {
          ret_op = gen_sret_expr(ilix, llvm_info.abi_info->arg[0].type);
        } else {
          ret_op = gen_base_addr_operand(ilix, make_ptr_lltype(retTy));
          ret_op = gen_load(ret_op, retTy, InstrListFlagsNull);
        }
      } else {
        if ((ILI_OPC(ilix) != IL_LDA) && (retTy->data_type != LL_PTR))
          retTy = make_ptr_lltype(retTy);
        ret_op = gen_base_addr_operand(ilix, retTy);
      }
      break;
    default:
      /* IL_EXIT */
      if (has_entries && !gbl.arets) {
        ret_op = gen_llvm_expr(ilix, NULL);
      } else {
        ret_op = gen_llvm_expr(ilix, retTy);
      }
    }
    if (ret_op) {
      Curr_Instr = gen_instr(I_RET, NULL, NULL, ret_op);
      ad_instr(ilix, Curr_Instr);
    }
  } break;
  case STMT_DECL:
    Curr_Instr = gen_instr(I_DECL, NULL, NULL, gen_llvm_expr(ilix, NULL));
    ad_instr(ilix, Curr_Instr);
    break;
  case STMT_LABEL: {
    sptr = (SPTR)ilix; // FIXME: is this a bug?
    process_sptr(sptr);
    Curr_Instr = gen_instr(I_NONE, NULL, NULL, make_label_op(sptr));
    ad_instr(ilix, Curr_Instr);

    break;
  }
  case STMT_CALL:
    if (getTempMap(ilix))
      return;
    sym = pd_sym = get_call_sptr(ilix);

    if (sym != pd_sym && STYPEG(pd_sym) == ST_PD) {
      switch (PDNUMG(pd_sym)) {
      default:
        break;
      }
    }
    if (gen_alloca_call_if_necessary(sym, ilix) != NULL ||
        gen_unreachable_if_necessary(sym, ilix) != NULL) {
      /* A builtin function that gets special handling. */
      goto end_make_stmt;
    }
    gen_call_expr(ilix, DT_NONE, NULL, sym);
    break;

    /* Add instruction if it hasn't been added already by gen_call_expr(). */
    if (!Instructions || !Curr_Instr->prev)
      ad_instr(ilix, Curr_Instr);
    break;

  case STMT_BR:
    opc = ILI_OPC(ilix);
    if (opc == IL_JMP) { /* unconditional jump */
      last_stmt_is_branch = 1;
      sptr = ILI_SymOPND(ilix, 1);
      /* also in gen_new_landingpad_jump */
      process_sptr(sptr);
      Curr_Instr = gen_instr(I_BR, NULL, NULL, make_target_op(sptr));
      ad_instr(ilix, Curr_Instr);
    } else if (exprjump(opc) || zerojump(opc)) { /* cond or zero jump */
      if (exprjump(opc)) { /* get sptr pointing to jump label */
        sptr = ILI_SymOPND(ilix, 4);
        cc = ILI_OPND(ilix, 3);
      } else {
        sptr = ILI_SymOPND(ilix, 3);
        cc = ILI_OPND(ilix, 2);
      }
      process_sptr(sptr);
      Curr_Instr = make_instr(I_BR);
      tmps = make_tmps();
      Curr_Instr->operands = make_tmp_op(make_int_lltype(1), tmps);

      /* make the condition code */
      switch (opc) {
      case IL_FCJMP:
      case IL_FCJMPZ:
      case IL_DCJMP:
      case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
      case IL_QCJMP:
      case IL_QCJMPZ:
#endif
        ad_instr(ilix, gen_instr(I_FCMP, tmps, Curr_Instr->operands->ll_type,
                                 gen_llvm_expr(ilix, NULL)));
        break;
      default:
        ad_instr(ilix, gen_instr(I_ICMP, tmps, Curr_Instr->operands->ll_type,
                                 gen_llvm_expr(ilix, NULL)));
      }
      first_label = make_target_op(sptr);
      /* need to make a label for the false condition -- llvm conditional
       * branch requires this step.
       */
      if (next_bih_label) {
        sptr_lab = next_bih_label;
      } else {
        sptr_lab = getlab();
        update_llvm_sym_arrays();
      }
      second_label = make_target_op(sptr_lab);
      first_label->next = second_label;
      Curr_Instr->operands->next = first_label;
      ad_instr(ilix, Curr_Instr);
      /* now add the label instruction */
      if (!next_bih_label)
        make_stmt(STMT_LABEL, sptr_lab, false, SPTR_NULL, ilt);
      DBGTRACE1("#goto statement: jump to label sptr %d", sptr);
    } else if (opc == IL_JMPM || opc == IL_JMPMK) {
      /* unconditional jump */
      Curr_Instr = gen_switch(ilix);
      last_stmt_is_branch = 1;
    } else if (opc == IL_JMPA) {
      int arg1 = ILI_OPND(ilix, 1);
      last_stmt_is_branch = 1;
      op1 = gen_llvm_expr(arg1, make_lltype_from_dtype(DT_CPTR));
      Curr_Instr = gen_instr(I_INDBR, NULL, NULL, op1);
      ad_instr(ilix, Curr_Instr);
    } else {
      /* unknown jump type */
      assert(0, "ilt branch: unexpected branch code", opc, ERR_Fatal);
    }
    break;
  case STMT_SMOVE:
    from_ili = ILI_OPND(ilix, 1);
    to_ili = ILI_OPND(ilix, 2);
    length_ili = ILI_OPND(ilix, 3);
    opnd = ILI_OPND(length_ili, 1);
    assert(DTYPEG(opnd) == DT_CPTR, "make_stmt(): expected DT_CPTR",
           DTYPEG(opnd), ERR_Fatal);
    bytes = CONVAL2G(opnd);
/* IL_SMOVE 3rd opnd has a 4-byte or 8-byte unit, the rest of the
   data are copied using other STORE ili.
   we use it as bytes.
*/
    bytes = bytes * 8;
    assert(bytes, "make_stmt(): expected smove byte size", 0, ERR_Fatal);
    from_nme = ILI_OPND(ilix, 4);
    ts = BITS_IN_BYTE * size_of(DT_CPTR);
    src_op = gen_llvm_expr(from_ili, make_lltype_from_dtype(DT_CPTR));
    dst_op = gen_llvm_expr(to_ili, make_lltype_from_dtype(DT_CPTR));
    dtype = dt_nme(from_nme);
#ifdef DT_ANY
    if (dtype == DT_ANY)
      alignment = align_of(DT_CPTR);
    else
#endif
        if (dtype)
      alignment = align_of(dtype);
    else
      alignment = 1;

    insert_llvm_memcpy(ilix, ts, dst_op, src_op, bytes, alignment, 0);
    break;
  case STMT_SZERO:
    assert(ILI_OPC(ilix) == IL_ARGIR || ILI_OPC(ilix) == IL_DAIR,
           "make_stmt(): expected ARGIR/DAIR for ilix ", ilix, ERR_Fatal);
    length_ili = ILI_OPND(ilix, 1);
    opnd = ILI_OPND(length_ili, 1);
    if (ILI_OPC(ilix) == IL_DAIR)
      to_ili = ILI_OPND(ilix, 3);
    else
      to_ili = ILI_OPND(ilix, 2);
    assert(ILI_OPC(to_ili) == IL_ARGAR || ILI_OPC(to_ili) == IL_DAAR,
           "make_stmt(): expected ARGAR/DAAR for ili ", to_ili, ERR_Fatal);
    to_ili = ILI_OPND(to_ili, 1);
    bytes = CONVAL2G(opnd);
    if (bytes) {
      ts = BITS_IN_BYTE * size_of(DT_CPTR);
      dst_op = gen_llvm_expr(to_ili, make_lltype_from_dtype(DT_CPTR));
      insert_llvm_memset(ilix, ts, dst_op, bytes, 0, 1, 0);
    }
    break;
  case STMT_ST:
    /* STORE statement */
    llvm_info.curr_ret_dtype = DT_NONE;
    nme = ILI_OPND(ilix, 3);
    lhs_ili = ILI_OPND(ilix, 2);
    rhs_ili = ILI_OPND(ilix, 1);
    if (sincos_seen())
      sincos_clear_specific_arg(lhs_ili, nme);
    if (!cancel_store(ilix, rhs_ili, lhs_ili)) {
      DTYPE vect_dtype = DT_NONE;
      LL_InstrListFlags store_flags = InstrListFlagsNull;
      LL_Type *int_llt = NULL;
      LL_Type *v4_llt = NULL;
      msz = ILI_MSZ_OF_ST(ilix);
      vect_dtype = ili_get_vect_dtype(ilix);
#if defined(TARGET_LLVM_ARM)
      if (vect_dtype) {
        store_flags = ldst_instr_flags_from_dtype(vect_dtype);
        if ((DTyVecLength(vect_dtype) == 3) && (ILI_OPC(ilix) == IL_VST)) {
          v4_llt = make_lltype_sz4v3_from_dtype(vect_dtype);
        } else {
          switch (zsize_of(vect_dtype)) {
          case 2:
            int_llt = make_lltype_from_dtype(DT_SINT);
            break;
          case 4:
            if (DTyVecLength(vect_dtype) != 3)
              int_llt = make_lltype_from_dtype(DT_INT);
            break;
          default:
            break;
          }
        }
      }
#endif
      if (ILI_OPC(ilix) == IL_STA) {
        LL_Type *ptrTy = make_lltype_from_dtype(DT_CPTR);
        op1 = gen_base_addr_operand(rhs_ili, ptrTy);
        store_flags = ldst_instr_flags_from_dtype(DT_CPTR);
      } else {
        if (vect_dtype) {
          if (v4_llt) {
            op1 = gen_llvm_expr(rhs_ili, v4_llt);
          } else {
            LL_Type *ty = make_lltype_from_dtype(vect_dtype);
            if (ILI_OPC(rhs_ili) == IL_VCMP)
              ty = 0;
            op1 = gen_llvm_expr(rhs_ili, ty);
            if (ILI_OPC(rhs_ili) == IL_VCMP)
              int_llt = op1->ll_type;
          }
          if (int_llt)
            op1 = make_bitcast(op1, int_llt);
          /* Clear alignment bits ==> alignment = 1 byte. */
          if (ILI_OPC(ilix) == IL_VSTU)
            store_flags &= (LL_InstrListFlags)~LDST_LOGALIGN_MASK;
        } else if (is_blockaddr_store(ilix, rhs_ili, lhs_ili)) {
          return;
        } else if (ILI_OPC(ilix) == IL_STSCMPLX) {
          LL_Type *ty = make_lltype_from_dtype(DT_CMPLX);
          op1 = gen_llvm_expr(rhs_ili, ty);
          store_flags = ldst_instr_flags_from_dtype(DT_CMPLX);
        } else if (ILI_OPC(ilix) == IL_STDCMPLX) {
          LL_Type *ty = make_lltype_from_dtype(DT_DCMPLX);
          op1 = gen_llvm_expr(rhs_ili, ty);
          store_flags = ldst_instr_flags_from_dtype(DT_DCMPLX);
#ifdef TARGET_SUPPORTS_QUADFP
        } else if (ILI_OPC(ilix) == IL_STQCMPLX) {
          LL_Type *ty = make_lltype_from_dtype(DT_QCMPLX);
          op1 = gen_llvm_expr(rhs_ili, ty);
          store_flags = ldst_instr_flags_from_dtype(DT_QCMPLX);
#endif
#ifdef LONG_DOUBLE_FLOAT128
        } else if (ILI_OPC(ilix) == IL_FLOAT128ST) {
          LL_Type *ty = make_lltype_from_dtype(DT_FLOAT128);
          op1 = gen_llvm_expr(rhs_ili, ty);
          store_flags = ldst_instr_flags_from_dtype(DT_FLOAT128);
#endif
          /* we let any other LONG_DOUBLE_X87 fall into the default case as
           * conversion is done (if needed) implicitly via convert_float_size()
           */
        } else {
          LL_Type *ty = make_type_from_msz(msz);
          op1 = gen_llvm_expr(rhs_ili, ty);
          store_flags = ldst_instr_flags_from_dtype(msz_dtype(msz));
        }
      }
      llt_expected = NULL;
      if ((ILI_OPC(ilix) == IL_STA) || (op1->ll_type->data_type == LL_STRUCT)) {
        llt_expected = make_ptr_lltype(op1->ll_type);
      }
      if (vect_dtype) {
        if (v4_llt)
          llt_expected = make_ptr_lltype(v4_llt);
        else if (int_llt)
          llt_expected = make_ptr_lltype(int_llt);
        else
          llt_expected = make_ptr_lltype(make_lltype_from_dtype(vect_dtype));
        store_op =
            gen_address_operand(lhs_ili, nme, false, llt_expected, (MSZ)-1);
      } else {
        store_op = gen_address_operand(lhs_ili, nme, false, llt_expected, msz);
      }
      if ((store_op->ll_type->data_type == LL_PTR) &&
          ll_type_int_bits(store_op->ll_type->sub_types[0]) &&
          ll_type_int_bits(op1->ll_type) &&
          (ll_type_bytes(store_op->ll_type->sub_types[0]) !=
           ll_type_bytes(op1->ll_type))) {
        /* Need to add a conversion here */
        op1 = convert_int_size(ilix, op1, store_op->ll_type->sub_types[0]);
      }

      if (nme == NME_VOL)
        store_flags |= VOLATILE_FLAG;
      if (IL_HAS_FENCE(ILI_OPC(ilix)))
        store_flags |= ll_instr_flags_for_memory_order_and_scope(ilix);
      DBGTRACE2("#store_op %p, op1 %p\n", store_op, op1);
      if (deletable)
        store_flags |= DELETABLE;
      Curr_Instr = mk_store_instr(op1, store_op);
      Curr_Instr->ilix = ilix;
      Curr_Instr->flags |= store_flags;
    }
    break;
  default:
    assert(0, "make_stmt(): unknown statment type", stmt_type, ERR_Fatal);
    break;
  }
end_make_stmt:;

  DBGTRACEOUT("")
} /* make_stmt */

// FIXME: gen_va_arg is used, but gen_va_start is never used.
#ifdef FLANG2_CGMAIN_UNUSED
static OPERAND *
gen_va_start(int ilix)
{
  OPERAND *call_op, *arg_op;
  char *va_start_name, *gname;
  int arg;
  static bool va_start_defined = false;
  EXFUNC_LIST *exfunc;
  LL_Type *expected_type;

  DBGTRACEIN1(" called with ilix %d\n", ilix)

  call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  va_start_name = (char *)getitem(LLVM_LONGTERM_AREA, 17);
  sprintf(va_start_name, "@llvm.va_start");
  call_op->string = va_start_name;
  arg = ILI_OPND(ilix, 2);
  assert(arg && is_argili_opcode(ILI_OPC(arg)), "gen_va_start(): bad argument",
         arg, ERR_Fatal);
  expected_type = make_lltype_from_dtype(DT_CPTR);
  arg_op = gen_llvm_expr(ILI_OPND(arg, 1), expected_type);
  call_op->next = arg_op;
  /* add prototype if needed */
  if (!va_start_defined) {
    va_start_defined = true;
    gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(va_start_name) + 35);
    sprintf(gname, "declare void %s(ptr)", va_start_name);
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration(va_start_name, exfunc);
  }

  DBGTRACEOUT1(" returns operand %p", call_op)

  return call_op;
} /* gen_va_start */
#endif

/**
   \brief Create a variable of type \p dtype
   \param ilix
   \param dtype
   \return an sptr to the newly created instance.

  */
static SPTR
make_arg_tmp(int ilix, DTYPE dtype)
{
  SPTR tmp;
  SYMTYPE stype;
  char tmp_name[32];
  NEWSYM(tmp);
  snprintf(tmp_name, sizeof(tmp_name), "argtmp.%d", ilix);
  NMPTRP(tmp, putsname(tmp_name, strlen(tmp_name)));

  switch (DTY(dtype)) {
  case TY_ARRAY:
    stype = ST_ARRAY;
    break;
  case TY_STRUCT:
    stype = ST_STRUCT;
    break;
  case TY_UNION:
    stype = ST_UNION;
    break;
  default:
    stype = ST_VAR;
  }
  STYPEP(tmp, stype);
  SCP(tmp, SC_AUTO);
  DTYPEP(tmp, dtype);
  PDALNP(tmp, align_bytes2power(align_of(dtype)));
  return tmp;
}

/**
 * \brief Expand an IL_VA_ARG instruction
 *
 * <tt>VA_ARG arlnk dtype</tt>
 * The first argument is a pointer to the va_list, the second is the dtype of
 * the argument to be extracted. Produce a pointer where the argument can be
 * loaded.
 *
 * There are two versions of this function (one for x86-64 and one for
 * non-x86-64).
 */
static OPERAND *
gen_va_arg(int ilix)
{

  /*
   * va_arg for other targets: va_list is an i8**, arguments are contiguous in
   * memory.
   *
   * %ap_cast = bitcast i8** %ap to uintptr_t*
   * %addr = load uintptr_t* %ap_cast
   * if (arg_align > reg_size)
   *   %addr = round-up-to-align(arg_align)
   *
   * %next = getelementptr argtype* %addr, 1
   * store argtype* %next, %ap_cast
   * return argtype %ptr
   */
  OPERAND *addr_op, *result_op, *next_op;
  const int ap_ili = ILI_OPND(ilix, 1);
  const DTYPE arg_dtype = ILI_DTyOPND(ilix, 2);
  const unsigned reg_size = size_of(DT_CPTR);
  unsigned arg_align = alignment(arg_dtype) + 1;
  unsigned arg_size = size_of(arg_dtype);
  LL_Type *uintptr_type = make_int_lltype(8 * reg_size);
  OPERAND *ap_cast = gen_llvm_expr(ap_ili, make_ptr_lltype(uintptr_type));
  const LL_InstrListFlags flags = ldst_instr_flags_from_dtype(DT_CPTR);

  addr_op = gen_load(ap_cast, uintptr_type, flags);

  switch (DTY(arg_dtype)) {
  default:
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case TY_FLOAT128:
  case TY_CMPLX128:
#endif
    /* These types are (needlessly) aligned to 16 bytes when laying out
     * structures, but treated as pairs or quadruplets of doubles in the
     * context of argument passing.
     */
    arg_align = 8;
    break;
  }

  if (arg_align > reg_size) {
    /* This argument has alignment greater than the pointer register size.
     * We need to dynamically align the address. */
    /* addr_op += arg_align-1 */
    addr_op->next = make_constval_op(uintptr_type, arg_align - 1, 0);
    addr_op =
        ad_csed_instr(I_ADD, 0, uintptr_type, addr_op, NOUNSIGNEDWRAP, false);
    /* addr_op &= -arg_align */
    addr_op->next = make_constval_op(uintptr_type, -arg_align, -1);
    addr_op = ad_csed_instr(I_AND, 0, uintptr_type, addr_op, InstrListFlagsNull,
                            false);
  }
  result_op = convert_int_to_ptr(
      addr_op, make_ptr_lltype(make_lltype_from_dtype(arg_dtype)));

#ifdef TARGET_POWER
  /* POWER ABI: va_args are passed in the parameter save region of the stack.
   * The caller is responsible for setting up the stack space for this (LLVM
   * will do this for us).
   *
   * The special case here is for 'float complex' where each complex has
   * two components, treated as the same type and alignment as the first
   * component (the real component of the complex value).
   *
   * The reason for this special case is because we need to treat the
   * components of the complex as coming from two separate float arguments.
   * These are stored into a temp complex {float, float} and a pointer to that
   * temp is returned.
   */
  if (arg_dtype == DT_CMPLX) {
    LL_Type *llt_float = make_lltype_from_dtype(DT_FLOAT);
    LL_Type *llt_float_ptr = make_ptr_lltype(llt_float);
    LL_Type *llt_cptr = make_lltype_from_dtype(DT_CPTR);
    const LL_InstrListFlags flt_flags =
        (LL_InstrListFlags)ldst_instr_flags_from_dtype(DT_FLOAT);
    OPERAND *tmp_op, *cmplx_op, *val_op;

    /* Pointer to temp real */
    SPTR tmp = make_arg_tmp(ilix, arg_dtype);
    cmplx_op = tmp_op = make_var_op(tmp); /* points to {float,float} */
    tmp_op = make_bitcast(tmp_op, llt_cptr);
    tmp_op = gen_gep_index(tmp_op, llt_cptr, 0);
    tmp_op = make_bitcast(tmp_op, llt_float_ptr);

    /* Pointer to actual real */
    result_op = make_bitcast(result_op, llt_cptr);
    result_op = gen_gep_index(result_op, llt_cptr, 0);
    result_op = make_bitcast(result_op, llt_float_ptr);
    val_op = gen_load(result_op, llt_float, flt_flags);
    make_store(val_op, tmp_op, flt_flags);

    /* Now for imaginary (must skip 2 * DT_FLOAT bytes) */
    tmp_op = make_bitcast(tmp_op, llt_cptr);
    tmp_op = gen_gep_index(tmp_op, llt_cptr, size_of(DT_FLOAT));
    tmp_op = make_bitcast(tmp_op, llt_float_ptr);
    result_op = make_bitcast(result_op, llt_cptr);
    result_op = gen_gep_index(result_op, llt_cptr, size_of(DT_FLOAT) * 2);
    result_op = make_bitcast(result_op, llt_float_ptr);
    val_op = gen_load(result_op, llt_float, flt_flags);
    make_store(val_op, tmp_op, flt_flags);

    result_op = gen_copy_op(cmplx_op);
    arg_size *= 2; /* Skip two floats instead of one float */
  }
#endif /* TARGET_POWER */

  /* Compute the address of the next argument.
   * Round up to a multiple of reg_size.
   */
  arg_size = (arg_size + reg_size - 1) & -reg_size;
  addr_op = gen_copy_op(addr_op);
  addr_op->next = make_constval_op(uintptr_type, arg_size, 0);
  next_op =
      ad_csed_instr(I_ADD, 0, uintptr_type, addr_op, NOUNSIGNEDWRAP, false);
  make_store(next_op, gen_copy_op(ap_cast), flags);

  return result_op;
} /* gen_va_arg */

// FIXME: gen_va_arg is used, but gen_va_end is never used.
#ifdef FLANG2_CGMAIN_UNUSED
static OPERAND *
gen_va_end(int ilix)
{
  OPERAND *call_op, *arg_op;
  char *va_end_name, *gname;
  int arg;
  static bool va_end_defined = false;
  EXFUNC_LIST *exfunc;
  LL_Type *expected_type;

  DBGTRACEIN1(" called with ilix %d\n", ilix)

  call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  va_end_name = (char *)getitem(LLVM_LONGTERM_AREA, 17);
  sprintf(va_end_name, "@llvm.va_end");
  call_op->string = va_end_name;
  arg = ILI_OPND(ilix, 2);
  assert(arg && is_argili_opcode(ILI_OPC(arg)), "gen_va_end(): bad argument",
         arg, ERR_Fatal);
  expected_type = make_lltype_from_dtype(DT_CPTR);
  arg_op = gen_llvm_expr(ILI_OPND(arg, 1), expected_type);
  call_op->next = arg_op;
  /* add prototype if needed */
  if (!va_end_defined) {
    va_end_defined = true;
    gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(va_end_name) + 35);
    sprintf(gname, "declare void %s(ptr)", va_end_name);
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration(va_end_name, exfunc);
  }

  DBGTRACEOUT1(" returns operand %p", call_op)

  return call_op;
} /* gen_va_end */
#endif

OPERAND *
gen_call_to_builtin(int ilix, char *fname, OPERAND *params,
                    LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                    LL_InstrName i_name, LL_InstrListFlags MathFlag,
                    unsigned flags)
{
  OPERAND *operand = NULL;
  char *intrinsic_name;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN1(" for ilix %d\n", ilix)

  intrinsic_name = (char *)getitem(LLVM_LONGTERM_AREA, strlen(fname) + 1);
  strcpy(intrinsic_name, fname);
  operand = make_tmp_op(return_ll_type, make_tmps());
  if (!Call_Instr)
    Curr_Instr = make_instr(i_name);
  else
    Curr_Instr = Call_Instr;
  Curr_Instr->flags |= CALL_INTRINSIC_FLAG;
  Curr_Instr->flags |= MathFlag;
  Curr_Instr->tmps = operand->tmps; /* result operand */
  Curr_Instr->tmps->info.idef = Curr_Instr;
  Curr_Instr->ll_type = return_ll_type;
  Curr_Instr->operands =
      get_intrinsic_call_ops(intrinsic_name, return_ll_type, params, flags);
  if (!Call_Instr)
    ad_instr(ilix, Curr_Instr);

  DBGTRACEOUT("")

  return operand;
} /* gen_call_to_builtin */

static const char *
get_atomicrmw_opname(LL_InstrListFlags instr_flags)
{
  switch (instr_flags & ATOMIC_RMW_OP_FLAGS) {
  case ATOMIC_SUB_FLAG:
    return "sub";
  case ATOMIC_ADD_FLAG:
    return "add";
  case ATOMIC_XCHG_FLAG:
    return "xchg";
  case ATOMIC_UMIN_FLAG:
    return "umin";
  case ATOMIC_MIN_FLAG:
    return "min";
  case ATOMIC_UMAX_FLAG:
    return "umax";
  case ATOMIC_MAX_FLAG:
    return "max";
  case ATOMIC_AND_FLAG:
    return "and";
  case ATOMIC_OR_FLAG:
    return "or";
  case ATOMIC_XOR_FLAG:
    return "xor";
  default:
    interr("Unexpected atomic rmw flag: ", instr_flags & ATOMIC_RMW_OP_FLAGS,
           ERR_Severe);
    return "";
  }
}

static const char *
get_atomic_memory_order_name(int instr_flags)
{
  switch (instr_flags & ATOMIC_MEM_ORD_FLAGS) {
  case ATOMIC_MONOTONIC_FLAG:
    return "monotonic";
  case ATOMIC_ACQUIRE_FLAG:
    return "acquire";
  case ATOMIC_RELEASE_FLAG:
    return "release";
  case ATOMIC_ACQ_REL_FLAG:
    return "acq_rel";
  case ATOMIC_SEQ_CST_FLAG:
    return "seq_cst";
  default:
    interr("Unexpected atomic mem ord flag: ",
           instr_flags & ATOMIC_MEM_ORD_FLAGS, ERR_Severe);
    return "";
  }
}

#ifdef FLANG_GEN_LLVM_ATOMIC_INTRINSICS
static OPERAND *
gen_llvm_atomicrmw_instruction(int ilix, int pdnum, OPERAND *params,
                               DTYPE return_dtype)
{
  return NULL; // TODO?
}
#endif

static OPERAND *
gen_call_llvm_intrinsic_impl(const char *fname, OPERAND *params,
                             LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                             LL_InstrName i_name, LL_InstrListFlags MathFlag)
{
  static char buf[MAXIDLEN];

  sprintf(buf, "@llvm.%s", fname);
  return gen_call_to_builtin(0, buf, params, return_ll_type, Call_Instr, i_name,
                             MathFlag, 0);
}

static OPERAND *
gen_call_llvm_intrinsic(const char *fname, OPERAND *params,
                        LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                        LL_InstrName i_name)
{
  return gen_call_llvm_intrinsic_impl(fname, params, return_ll_type,
                                      Call_Instr, i_name, InstrListFlagsNull);
}

static OPERAND *
gen_call_llvm_fm_intrinsic(const char *fname, OPERAND *params,
                           LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                           LL_InstrName i_name)
{
  return gen_call_llvm_intrinsic_impl(fname, params, return_ll_type,
                                      Call_Instr, i_name, FAST_MATH_FLAG);
}

static OPERAND *
gen_call_llvm_non_fm_math_intrinsic(const char *fname, OPERAND *params,
                           LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                           LL_InstrName i_name)
{
  LL_InstrListFlags MathFlag = InstrListFlagsNull;
  if (XBIT(216, 0x8))
    MathFlag |= NSZ_MATH_FLAG;
  if (XBIT(216, 0x10))
    MathFlag |= REASSOC_MATH_FLAG;
  return gen_call_llvm_intrinsic_impl(fname, params, return_ll_type,
                                      Call_Instr, i_name, MathFlag);
}

static OPERAND *
gen_call_pgocl_intrinsic(const char *fname, OPERAND *params,
                         LL_Type *return_ll_type, INSTR_LIST *Call_Instr,
                         LL_InstrName i_name)
{
  static char buf[MAXIDLEN];

  sprintf(buf, "@%s%s", ENTOCL_PREFIX, fname);
  return gen_call_to_builtin(0, buf, params, return_ll_type, Call_Instr,
                             i_name, InstrListFlagsNull, 0);
}

static void
insert_llvm_prefetch(int ilix, OPERAND *dest_op)
{
  OPERAND *call_op;

  DBGTRACEIN("")

  const char *intrinsic_name = "@llvm.prefetch";
  char *fname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(intrinsic_name) + 1);
  strcpy(fname, intrinsic_name);
  INSTR_LIST *Curr_Instr = make_instr(I_CALL);
  Curr_Instr->flags |= CALL_INTRINSIC_FLAG;
  Curr_Instr->operands = call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  Curr_Instr->ll_type = call_op->ll_type;
  call_op->string = fname;
  call_op->next = dest_op;

  /* setup rest of the parameters for llvm.prefetch */
  LL_Type *int32_type = make_int_lltype(32);
  /* prefetch type: 0 = read, 1 = write */
  dest_op->next = make_constval_op(int32_type, 0, 0);
  /* temporal locality specifier: 3 = extremely local, keep in cache */
  dest_op->next->next = make_constval_op(int32_type, 3, 0);
  /* cache type: 0 = instruction, 1 = data */
  dest_op->next->next->next = make_constval_op(int32_type, 1, 0);
  ad_instr(ilix, Curr_Instr);

  /* add global define of @llvm.prefetch to external function list, if needed */
  static bool prefetch_defined = false;
  if (!prefetch_defined) {
    prefetch_defined = true;
    const char *intrinsic_decl = "declare void @llvm.prefetch(ptr nocapture, i32, i32, i32)";
    char *gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(intrinsic_decl) + 1);
    strcpy(gname, intrinsic_decl);
    EXFUNC_LIST *exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration(fname, exfunc);
  }

  DBGTRACEOUT("")
} /* insert_llvm_prefetch */

static void
insert_llvm_memset(int ilix, int size, OPERAND *dest_op, int len, int value,
                   int align, int is_volatile)
{
  EXFUNC_LIST *exfunc;
  OPERAND *call_op;
  static bool memset_defined = false;
  char *memset_name, *gname;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN("")

  memset_name = (char *)getitem(LLVM_LONGTERM_AREA, 22);
  sprintf(memset_name, "@llvm.memset.p0i8.i%d", size);
  Curr_Instr = make_instr(I_CALL);
  Curr_Instr->flags |= CALL_INTRINSIC_FLAG;
  Curr_Instr->operands = call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  Curr_Instr->ll_type = call_op->ll_type;
  call_op->string = memset_name;
  call_op->next = dest_op;

  dest_op->next = make_constval_op(make_int_lltype(8), value, 0);
  /* length in bytes of memset */
  dest_op->next->next = make_constval_op(make_int_lltype(size), len, 0);
  /* alignment */
  dest_op->next->next->next = make_constval32_op(align);
  dest_op->next->next->next->next =
      make_constval_op(make_int_lltype(1), is_volatile, 0);
  ad_instr(ilix, Curr_Instr);
  /* add global define of llvm.memset to external function list, if needed */
  if (!memset_defined) {
    memset_defined = true;
    gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(memset_name) + 45);
    sprintf(gname, "declare void %s(ptr, i8, i%d, i32, i1)", memset_name, size);
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration(memset_name, exfunc);
  }
  DBGTRACEOUT("")
} /* insert_llvm_memset */

static void
insert_llvm_memcpy(int ilix, int size, OPERAND *dest_op, OPERAND *src_op,
                   int len, int align, int is_volatile)
{
  EXFUNC_LIST *exfunc;
  OPERAND *call_op;
  static bool memcpy_defined = false;
  char *memcpy_name, *gname;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN("")

  memcpy_name = (char *)getitem(LLVM_LONGTERM_AREA, 27);
  sprintf(memcpy_name, "@llvm.memcpy.p0i8.p0i8.i%d", size);
  Curr_Instr = make_instr(I_CALL);
  Curr_Instr->flags |= CALL_INTRINSIC_FLAG;
  Curr_Instr->operands = call_op = make_operand();
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  Curr_Instr->ll_type = call_op->ll_type;
  call_op->string = memcpy_name;
  call_op->next = dest_op;
  dest_op->next = src_op;
  /* length in bytes of memcpy */
  src_op->next = make_constval_op(make_int_lltype(size), len, 0);
  src_op->next->next = make_constval32_op(align); /* alignment */
  src_op->next->next->next =
      make_constval_op(make_int_lltype(1), is_volatile, 0);
  ad_instr(ilix, Curr_Instr);
  /* add global define of llvm.memcpy to external function list, if needed */
  if (!memcpy_defined) {
    memcpy_defined = true;
    gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(memcpy_name) + 49);
    sprintf(gname, "declare void %s(ptr, ptr, i%d, i32, i1)", memcpy_name,
            size);
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration(memcpy_name, exfunc);
  }

  DBGTRACEOUT("")
} /* insert_llvm_memcpy */

/**
   \brief Insert <tt>@llvm.dbg.declare</tt> call for debug
   \param mdnode  metadata node
   \param sptr    symbol
   \param llTy    preferred type of \p sptr or \c NULL
 */
void
insert_llvm_dbg_declare(LL_MDRef mdnode, SPTR sptr, LL_Type *llTy,
                        OPERAND *exprMDOp, OperandFlag_t opflag)
{
  EXFUNC_LIST *exfunc;
  OPERAND *call_op;
  static bool dbg_declare_defined = false;
  const char *gname;
  INSTR_LIST *Curr_Instr;

  Curr_Instr = make_instr(I_CALL);
  Curr_Instr->flags |= CALL_INTRINSIC_FLAG;
  Curr_Instr->operands = call_op = make_operand();
  Curr_Instr->dbg_line_op =
      lldbg_get_var_line(cpu_llvm_module->debug_info, sptr);
  call_op->ot_type = OT_CALL;
  call_op->ll_type = make_void_lltype();
  Curr_Instr->ll_type = call_op->ll_type;
  call_op->string = "@llvm.dbg.declare";

  call_op->next = make_metadata_wrapper_op(sptr, llTy);
  call_op->next->flags |= opflag;
  call_op->next->next = make_mdref_op(mdnode);
  if (ll_feature_dbg_declare_needs_expression_md(&cpu_llvm_module->ir)) {
    if (exprMDOp) {
      call_op->next->next->next = exprMDOp;
    } else {
      LL_DebugInfo *di = cpu_llvm_module->debug_info;
      LL_MDRef md;
      if (ll_feature_debug_info_ver90(&cpu_llvm_module->ir)) {
        md = lldbg_emit_empty_expression_mdnode(di);
      } else {
        /* Handle the Fortran allocatable array cases. Emit expression
         * mdnode with single argument of DW_OP_deref to workaround known
         * gdb bug not able to debug array bounds.
         */
        if (ftn_array_need_debug_info(sptr)) {
          const unsigned deref = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
          md = lldbg_emit_expression_mdnode(di, 1, deref);
        } else
          md = lldbg_emit_empty_expression_mdnode(di);
      }
      call_op->next->next->next = make_mdref_op(md);
    }
  }

  ad_instr(0, Curr_Instr);
  /* add global define of llvm.dbg.declare to external function list, if needed
   */
  if (!dbg_declare_defined) {
    dbg_declare_defined = true;
    if (ll_feature_dbg_declare_needs_expression_md(&cpu_llvm_module->ir)) {
      gname = "declare void @llvm.dbg.declare(metadata, metadata, metadata)";
    } else {
      gname = "declare void @llvm.dbg.declare(metadata, metadata)";
    }
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration("llvm.dbg.declare", exfunc);
  }
}

const char *
match_names(MATCH_Kind match_val)
{
  char *tt;

  switch (match_val) {
  case MATCH_NO:
    return "MATCH_NO";
  case MATCH_OK:
    return "MATCH_OK";
  case MATCH_MEM:
    return "MATCH_MEM";
  default:
    asrt(match_val > 1);
    tt = (char *)getitem(LLVM_LONGTERM_AREA, 10);
    sprintf(tt, "MATCH_%d", match_val);
    return tt;
  }
} /* match_names */

static OPERAND *
gen_const_expr(int ilix, LL_Type *expected_type)
{
  OPERAND *operand;
  SPTR sptr = ILI_SymOPND(ilix, 1);

  /* Generate null pointer operands when requested.
   * Accept both IL_ICON and IL_KCON nulls. */
  if (expected_type && (expected_type->data_type == LL_PTR) &&
      CONVAL2G(sptr) == 0 && CONVAL1G(sptr) == 0)
    return make_null_op(expected_type);

  operand = make_operand();
  operand->ot_type = OT_CONSTSPTR;

  switch (ILI_OPC(ilix)) {
  case IL_KCON:
    operand->ll_type = make_lltype_from_dtype(DT_INT8);
    operand->val.sptr = sptr;
    break;
  case IL_ICON:
    if (expected_type && ll_type_int_bits(expected_type) &&
        (ll_type_int_bits(expected_type) < 32)) {
      operand->ot_type = OT_CONSTVAL;
      operand->val.conval[0] = CONVAL2G(sptr);
      operand->val.conval[1] = 0;
      assert(ll_type_int_bits(expected_type), "expected int",
             expected_type->data_type, ERR_Fatal);
      operand->ll_type = expected_type;
    } else {
       if (expected_type && expected_type->data_type == LL_PTR)
         operand->ll_type = make_lltype_from_dtype(DT_INT8);
       else
         operand->ll_type = make_lltype_from_dtype(DT_INT);
      operand->val.sptr = sptr;
    }
    break;
  case IL_FCON:
    operand->ll_type = make_lltype_from_dtype(DT_FLOAT);
    operand->val.sptr = sptr;
    break;
  case IL_DCON:
    operand->ll_type = make_lltype_from_dtype(DT_DBLE);
    operand->val.sptr = sptr;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCON:
    operand->ll_type = make_lltype_from_dtype(DT_QUAD);
    operand->val.sptr = sptr;
    break;
#endif
  case IL_VCON:
    operand->ll_type = make_lltype_from_sptr(sptr);
    operand->val.sptr = sptr;
    break;
  case IL_SCMPLXCON:
  case IL_DCMPLXCON:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXCON:
#endif
    operand->ll_type = make_lltype_from_dtype(DTYPEG(sptr));
    operand->val.sptr = sptr;
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CON:
    operand->ll_type = make_lltype_from_dtype(DT_FLOAT128);
    operand->val.sptr = sptr;
    break;
#endif
  default:
    interr("Unknown gen_const_expr opcode", ILI_OPC(ilix), ERR_Fatal);
  }
  return operand;
} /* gen_const_expr */

static OPERAND *
gen_unary_expr(int ilix, LL_InstrName itype)
{
  int op_ili;
  ILI_OP opc = ILI_OPC(ilix);
  OPERAND *operand;
  LL_Type *opc_type, *instr_type;

  DBGTRACEIN2(" ilix: %d(%s) \n", ilix, IL_NAME(opc))

  instr_type = opc_type = make_type_from_opc(opc);
  assert(opc_type != NULL, "gen_unary_expr(): no type information", 0,
         ERR_Fatal);

  op_ili = ILI_OPND(ilix, 1);

  switch (opc) {
  case IL_DFIXUK:
  case IL_DFIXU:
  case IL_DFIX:
  case IL_DFIXK:
  case IL_SNGL:
#ifdef TARGET_SUPPORTS_QUADFP
  /* convert double to quad precision */
  case IL_DQUAD:
#endif
    opc_type = make_lltype_from_dtype(DT_DBLE);
    break;
  case IL_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  /* convert float to quad precision */
  case IL_RQUAD:
#endif
  case IL_UFIX:
  case IL_FIX:
  case IL_FIXK:
  case IL_FIXUK:
    opc_type = make_lltype_from_dtype(DT_FLOAT);
    break;
  case IL_FLOAT:
  case IL_FLOATU:
  case IL_DFLOATU:
  case IL_DFLOAT:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFLOAT:
#endif
  case IL_ALLOC:
    opc_type = make_lltype_from_dtype(DT_INT);
    break;
  case IL_FLOATK:
  case IL_FLOATUK:
  case IL_DFLOATUK:
  case IL_DFLOATK:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFLOATK:
#endif
    opc_type = make_lltype_from_dtype(DT_INT8);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  /* convert quad to integer */
  case IL_QFIX:
  /* convert quad to 64 bit integer */
  case IL_QFIXK:
  /* convert quad to single precision */
  case IL_SNGQ:
  /* convert quad to double precision */
  case IL_DBLEQ:
    opc_type = make_lltype_from_dtype(DT_QUAD);
    break;
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FROM:
    opc_type = make_lltype_from_dtype(DT_DBLE);
    break;
  case IL_FLOAT128TO:
    opc_type = make_lltype_from_dtype(DT_FLOAT128);
    break;
#endif
  default:
    break;
  }

  DBGTRACE2("#generating unary operand, op_ili: %d(%s)", op_ili,
            IL_NAME(ILI_OPC(op_ili)));

  /* now make the new unary expression */
  operand =
      ad_csed_instr(itype, ilix, instr_type, gen_llvm_expr(op_ili, opc_type),
                    InstrListFlagsNull, true);

  DBGTRACEOUT1(" return operand %p\n", operand)

  return operand;
} /* gen_unary_expr */

static OPERAND *
gen_abs_expr(int ilix)
{
  int lhs_ili;
  ILI_OP opc = ILI_OPC(ilix);
  OPERAND *operand, *cmp_op, *op1, *op2, *zero_op, *comp_operands;
  LL_Type *opc_type, *bool_type;
  LL_InstrName cc_itype;
  int cc_val;
  INT tmp[2];
  union {
    double d;
    INT tmp[2];
  } dtmp;
  float f;
  double d;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN2(" ilix: %d(%s) \n", ilix, IL_NAME(opc))

  lhs_ili = ILI_OPND(ilix, 1);
  opc_type = make_type_from_opc(opc);
  assert(opc_type, "gen_abs_expr(): no type information", 0, ERR_Fatal);
  operand = make_tmp_op(opc_type, make_tmps());
  op1 = gen_llvm_expr(lhs_ili, operand->ll_type);
  /* now make the new binary expression */
  Curr_Instr = gen_instr(I_SELECT, operand->tmps, operand->ll_type, NULL);
  bool_type = make_int_lltype(1);
  switch (ILI_OPC(ilix)) {
  default:
    interr("Unknown abs opcode", ILI_OPC(ilix), ERR_Fatal);
    return NULL;
  case IL_IABS:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_intcc(CC_LT);
    op2 = gen_llvm_expr(ad1ili(IL_INEG, lhs_ili), operand->ll_type);
    zero_op = gen_llvm_expr(ad_icon(0), operand->ll_type);
    break;
  case IL_KABS:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_intcc(CC_LT);
    op2 = gen_llvm_expr(ad1ili(IL_KNEG, lhs_ili), operand->ll_type);
    zero_op = gen_llvm_expr(ad_kconi(0), operand->ll_type);
    break;
  case IL_FABS:
    cc_itype = I_FCMP;
    cc_val = convert_to_llvm_fltcc(CC_LT);
    op2 = gen_llvm_expr(ad1ili(IL_FNEG, lhs_ili), operand->ll_type);
    tmp[0] = 0;
    f = 0.0;
    mftof(f, tmp[1]);
    zero_op =
        gen_llvm_expr(ad1ili(IL_FCON, getcon(tmp, DT_FLOAT)), operand->ll_type);
    break;
  case IL_DABS:
    cc_itype = I_FCMP;
    cc_val = convert_to_llvm_fltcc(CC_LT);
    op2 = gen_llvm_expr(ad1ili(IL_DNEG, lhs_ili), operand->ll_type);
    d = 0.0;
    xmdtod(d, dtmp.tmp);
    zero_op = gen_llvm_expr(ad1ili(IL_DCON, getcon(dtmp.tmp, DT_DBLE)),
                            operand->ll_type);
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ABS:
    cc_itype = IL_FLOAT128CMP;
    cc_val = convert_to_llvm_fltcc(CC_LT);
    op2 = gen_llvm_expr(ad1ili(IL_FLOAT128CHS, lhs_ili), operand->ll_type);
    zero_op =
        gen_llvm_expr(ad1ili(IL_FLOAT128CON, stb.float128_0), operand->ll_type);
    break;
#endif
  }
  cmp_op = make_tmp_op(bool_type, make_tmps());

  Curr_Instr->operands = cmp_op;
  Curr_Instr->operands->next = op2;
  Curr_Instr->operands->next->next = op1;

  comp_operands = make_operand();
  comp_operands->ot_type = OT_CC;
  comp_operands->val.cc = cc_val;
  comp_operands->ll_type = bool_type;
  comp_operands->next = gen_copy_op(op1);
  comp_operands->next->next = gen_copy_op(zero_op);

  ad_instr(ilix,
           gen_instr(cc_itype, cmp_op->tmps, cmp_op->ll_type, comp_operands));

  ad_instr(ilix, Curr_Instr);

  DBGTRACEOUT1(" returns operand %p", operand)

  return operand;
}

static OPERAND *
gen_minmax_expr(int ilix, OPERAND *op1, OPERAND *op2)
{
  ILI_OP opc = ILI_OPC(ilix);
  OPERAND *operand, *cmp_op;
  LL_Type *llt, *bool_type;
  LL_InstrName cc_itype;
  int cc_val;
  CC_RELATION cc_ctype;
  DTYPE vect_dtype;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN2(" ilix: %d(%s)", ilix, IL_NAME(opc))

  operand = make_tmp_op(NULL, make_tmps());
  vect_dtype = ili_get_vect_dtype(ilix);
  if (vect_dtype) {
    llt = make_lltype_from_dtype(vect_dtype);
    operand->ll_type = llt->sub_types[0];
  } else
  {
    llt = make_type_from_opc(opc);
    operand->ll_type = llt;
  }

  /* now make the new binary expression */
  bool_type = make_int_lltype(1);
  switch (opc) {
  case IL_UIMIN:
  case IL_UKMIN:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_uintcc(CC_LT);
    break;
  case IL_IMIN:
  case IL_KMIN:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_intcc(CC_LT);
    break;
  case IL_FMIN:
  case IL_DMIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMIN:
#endif
    cc_itype = I_FCMP;
    cc_val = convert_to_llvm_fltcc(CC_NOTGE);
    break;
  case IL_UIMAX:
  case IL_UKMAX:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_uintcc(CC_GT);
    break;
  case IL_IMAX:
  case IL_KMAX:
    cc_itype = I_ICMP;
    cc_val = convert_to_llvm_intcc(CC_GT);
    break;
  case IL_FMAX:
  case IL_DMAX:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMAX:
#endif
    cc_itype = I_FCMP;
    cc_val = convert_to_llvm_fltcc(CC_NOTLE);
    break;
  case IL_VMIN:
    cc_ctype = CC_NOTGE;
    goto common_minmax;
  /* Fall through */
  case IL_VMAX:
    cc_ctype = CC_NOTLE;
  common_minmax:
    switch (DTY(DTySeqTyElement(vect_dtype))) {
    case TY_FLOAT:
    case TY_DBLE:
      cc_itype = I_FCMP;
      cc_val = convert_to_llvm_fltcc(cc_ctype);
      break;
    default:
      cc_itype = I_ICMP;
      if (DT_ISUNSIGNED(DTySeqTyElement(vect_dtype)))
        cc_val = convert_to_llvm_uintcc(cc_ctype);
      else
        cc_val = convert_to_llvm_intcc(cc_ctype);
      break;
    }
    break;
  default:
    interr("Unknown minmax opcode", ILI_OPC(ilix), ERR_Fatal);
    return NULL;
  }
  cmp_op = make_tmp_op(bool_type, make_tmps());

  Curr_Instr = gen_instr(cc_itype, cmp_op->tmps, bool_type, NULL);
  Curr_Instr->operands = make_operand();
  Curr_Instr->operands->ot_type = OT_CC;
  Curr_Instr->operands->val.cc = cc_val;
  Curr_Instr->operands->ll_type = llt; /* opc type computed at top of routine */
  Curr_Instr->operands->next = op1;
  Curr_Instr->operands->next->next = op2;
  ad_instr(ilix, Curr_Instr);

  cmp_op->next = gen_copy_op(op1);
  cmp_op->next->next = gen_copy_op(op2);
  Curr_Instr = gen_instr(I_SELECT, operand->tmps, operand->ll_type, cmp_op);
  ad_instr(ilix, Curr_Instr);

  DBGTRACEOUT1(" returns operand %p", operand)

  return operand;
}

static OPERAND *
gen_select_expr(int ilix)
{
  int cmp_ili, lhs_ili, rhs_ili;
  ILI_OP opc = ILI_OPC(ilix);
  OPERAND *operand;
  LL_Type *opc_type, *bool_type;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN2(" ilix: %d(%s) \n", ilix, IL_NAME(opc))

  cmp_ili = ILI_OPND(ilix, 1);
  lhs_ili = ILI_OPND(ilix, 3);
  rhs_ili = ILI_OPND(ilix, 2);
  opc_type = make_type_from_opc(opc);
  assert(opc_type, "gen_select_expr(): no type information", 0, ERR_Fatal);
  operand = make_tmp_op(opc_type, make_tmps());

  /* now make the new binary expression */
  Curr_Instr = gen_instr(I_SELECT, operand->tmps, operand->ll_type, NULL);

  DBGTRACE2("#generating comparison operand, cmp_ili: %d(%s)", cmp_ili,
            IL_NAME(ILI_OPC(cmp_ili)))

  bool_type = make_int_lltype(1);
  if (IEEE_CMP)
    float_jmp = true;
  Curr_Instr->operands = gen_llvm_expr(cmp_ili, bool_type);
  float_jmp = false;

  DBGTRACE2("#generating second operand, lhs_ili: %d(%s)", lhs_ili,
            IL_NAME(ILI_OPC(lhs_ili)))

  Curr_Instr->operands->next = gen_llvm_expr(lhs_ili, operand->ll_type);

  DBGTRACE2("#generating third operand, rhs_ili: %d(%s)", rhs_ili,
            IL_NAME(ILI_OPC(rhs_ili)))

  Curr_Instr->operands->next->next = gen_llvm_expr(rhs_ili, operand->ll_type);
  ad_instr(ilix, Curr_Instr);

  DBGTRACEOUT1(" returns operand %p", operand)

  return operand;
}

static int
get_vconi(DTYPE dtype, INT value)
{
  INT v[TY_VECT_MAXLEN];
  int i;

  for (i = 0; i < DTyVecLength(dtype); i++) {
    v[i] = value;
  }
  return get_vcon(v, dtype);
}

static SPTR
get_vcon0_n(DTYPE dtype, int start, int N)
{
  INT v[TY_VECT_MAXLEN];
  int i;

  for (i = 0; i < N; i++) {
    v[i] = start + i;
  }
  return get_vcon(v, dtype);
}

static OPERAND *
gen_imask(SPTR sptr)
{
  OPERAND *operand;
  DTYPE vdtype = DTYPEG(sptr);

  operand = make_operand();
  operand->ot_type = OT_CONSTSPTR;
  operand->ll_type = make_vtype(DT_INT, DTyVecLength(vdtype));
  operand->val.sptr = sptr;

  return operand;
}

/*
 * Here we generate LLVM instruction to insert scalar operand <sop> at index
 * <idx>
 * into vector operand <vop>
 * So in LLVM it will tranlate into:
 * %0 = insertelement <<sz> x <ty>> <vop>, <sop>, i32 <idx>
 */
static OPERAND *
gen_insert_vector(OPERAND *vop, OPERAND *sop, int idx)
{
  OPERAND *operand;
  INSTR_LIST *Curr_Instr;

  operand = make_tmp_op(vop->ll_type, make_tmps());

  Curr_Instr = gen_instr(I_INSELE, operand->tmps, operand->ll_type, vop);
  vop->next = sop;
  vop->next->next = make_constval32_op(idx);
  ad_instr(0, Curr_Instr);

  return operand;
}

/*
 * Here we generate LLVM instruction to extract a scalar at a index <idx>
 * from vector operand <vop>
 * So in LLVM it will tranlate into:
 * %0 = extractelement <<sz> x <ty>> <vop>, i32 <idx>
 */
#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
static OPERAND *
gen_extract_vector(OPERAND *vop, int idx)
{
  OPERAND *operand;
  INSTR_LIST *Curr_Instr;

  assert(vop->ll_type->data_type == LL_VECTOR,
         "gen_extract_vector(): vector type expected for operand\n",
         vop->ll_type->data_type, ERR_Fatal);
  operand = make_tmp_op(vop->ll_type->sub_types[0], make_tmps());

  Curr_Instr = gen_instr(I_EXTELE, operand->tmps, operand->ll_type, vop);
  vop->next = make_constval32_op(idx);
  ad_instr(0, Curr_Instr);

  return operand;
}
#endif

/**
   \brief Create a new vector

   The new vactor will have the same type as \p vop with size \p new_size and
   filed with values from \p vop.  This is useful for converting 3 element
   vectors to 4 element vectors and vice-versa.

   Let's assume \p vop is a vector of 4 floats and \p new_size is 3.  This will
   be expanded into following LLVM instruction:

   <pre>
   %0 = shufflevector <4 x float> %vop, <4 x float> undef,
             <3 x i32> <i32 0, i32 1, i32 2>
   </pre>

   This will build a vector of 3 floats with 3 first elements from \p vop.
 */
static OPERAND *
gen_resized_vect(OPERAND *vop, int new_size, int start)
{
  OPERAND *operand;
  LL_Type *llt;
  INSTR_LIST *Curr_Instr;
  INT v[TY_VECT_MAXLEN];
  int i;

  assert(vop->ll_type->data_type == LL_VECTOR, "expecting vector type",
         vop->ll_type->data_type, ERR_Fatal);
  llt = ll_get_vector_type(vop->ll_type->sub_types[0], new_size);
  operand = make_tmp_op(llt, make_tmps());

  Curr_Instr = gen_instr(I_SHUFFVEC, operand->tmps, operand->ll_type, vop);

  vop->next = make_operand();
  vop->next->ot_type = OT_UNDEF;
  vop->next->ll_type = vop->ll_type;

  if ((ll_type_bytes(vop->ll_type) * BITS_IN_BYTE) > new_size) {
    vop->next->next = gen_imask(
        get_vcon0_n(get_vector_dtype(DT_INT, new_size), start, new_size));
  } else {
    for (i = 0; i < ll_type_bytes(vop->ll_type) * BITS_IN_BYTE; i++)
      v[i] = i + start;
    for (; i < new_size; i++)
      v[i] = ll_type_bytes(vop->ll_type) * BITS_IN_BYTE + start;
    vop->next->next =
        gen_imask(get_vcon(v, get_vector_dtype(DT_INT, new_size)));
  }

  ad_instr(0, Curr_Instr);

  return operand;
}

static OPERAND *
gen_scalar_to_vector_helper(int ilix, int from_ili, LL_Type *ll_vecttype)
{
  OPERAND *operand, *undefop, *arg;
  INSTR_LIST *Curr_Instr;

  operand = make_tmp_op(ll_vecttype, make_tmps());

  Curr_Instr = gen_instr(I_SHUFFVEC, operand->tmps, operand->ll_type, NULL);

  undefop = make_undef_op(ll_vecttype);
  arg = gen_llvm_expr(from_ili, ll_vecttype->sub_types[0]);
  Curr_Instr->operands = gen_insert_vector(undefop, arg, 0);

  Curr_Instr->operands->next = make_undef_op(ll_vecttype);

  Curr_Instr->operands->next->next = gen_imask((SPTR)get_vcon0( // ???
      get_vector_dtype(DT_INT, ll_vecttype->sub_elements)));
  ad_instr(ilix, Curr_Instr);

  return operand;
}

/*
 * Create a vector from a scalar value represented by 'ilix'
 * ilix -> <ilix, ilix, ilix, ilix>
 * Let's assume ilix needs to be promoted to a vector of 4 floats
 * This will be expanded into following LLVM instructions:
 * %0 = insertelement <4 x float> undef, <ilix>, i32 0
 * %1 = shufflevector <4 x float> %0, <4 x float> undef, <4 x i32> <i32 0, i32
 * 0, i32, 0 i32 0>
 */
INLINE static OPERAND *
gen_scalar_to_vector(int ilix, LL_Type *ll_vecttype)
{
  const int from_ili = ILI_OPND(ilix, 1);
  return gen_scalar_to_vector_helper(ilix, from_ili, ll_vecttype);
}

#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
static OPERAND *
gen_scalar_to_vector_no_shuffle(int ilix, LL_Type *ll_vecttype)
{
  const int from_ili = ILI_OPND(ilix, 1);
  OPERAND *undefop = make_undef_op(ll_vecttype);
  OPERAND *arg = gen_llvm_expr(from_ili, ll_vecttype->sub_types[0]);
  OPERAND *operand = gen_insert_vector(undefop, arg, 0);
  return operand;
}
#endif

INLINE static OPERAND *
gen_temp_to_vector(int from_ili, LL_Type *ll_vecttype)
{
  const int ilix = 0;
  return gen_scalar_to_vector_helper(ilix, from_ili, ll_vecttype);
}

static OPERAND *
gen_gep_op(int ilix, OPERAND *base_op, LL_Type *llt, OPERAND *index_op)
{
  base_op->next = index_op;
  return ad_csed_instr(I_GEP, ilix, llt, base_op, InstrListFlagsNull, true);
}

#ifdef TARGET_POWER
INLINE static OPERAND *
gen_gep_index(OPERAND *base_op, LL_Type *llt, int index)
{
  return gen_gep_op(0, base_op, llt, make_constval32_op(index));
}
#endif

void
insert_llvm_dbg_value(OPERAND *load, LL_MDRef mdnode, SPTR sptr, LL_Type *type)
{
  static bool defined = false;
  OPERAND *callOp;
  OPERAND *oper;
  LLVMModuleRef mod = cpu_llvm_module;
  LL_DebugInfo *di = mod->debug_info;
  INSTR_LIST *callInsn = make_instr(I_CALL);

  if (!defined) {
    EXFUNC_LIST *exfunc;
    const char *gname =
        "declare void @llvm.dbg.value(metadata, i64, metadata, metadata)";
    exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
    memset(exfunc, 0, sizeof(EXFUNC_LIST));
    exfunc->func_def = gname;
    exfunc->flags |= EXF_INTRINSIC;
    add_external_function_declaration("llvm.dbg.value", exfunc);
    defined = true;
  }

  callInsn->flags |= CALL_INTRINSIC_FLAG;
  callInsn->operands = callOp = make_operand();
  callInsn->dbg_line_op = lldbg_get_var_line(di, sptr);
  callOp->ot_type = OT_CALL;
  callOp->ll_type = make_void_lltype();
  callInsn->ll_type = callOp->ll_type;
  callOp->string = "@llvm.dbg.value";

  callOp->next = oper = make_operand();
  oper->ot_type = OT_MDNODE;
  oper->tmps = load->tmps;
  oper->val = load->val;
  oper->ll_type = type;
  oper->flags |= OPF_WRAPPED_MD;
  oper = make_constval_op(ll_create_int_type(mod, 64), 0, 0);
  callOp->next->next = oper;
  oper->next = make_mdref_op(mdnode);
  oper->next->next = make_mdref_op(lldbg_emit_empty_expression_mdnode(di));

  ad_instr(0, callInsn);
}

/**
   \brief Construct llvm.dbg.value calls on the load sites
   \param ld    The load
   \param addr  The address being loaded
   \param type  The type of the object being loaded
 */
static void
consLoadDebug(OPERAND *ld, OPERAND *addr, LL_Type *type)
{
  SPTR sptr = addr->val.sptr;
  if (sptr && need_debug_info(sptr)) {
    LL_DebugInfo *di = cpu_llvm_module->debug_info;
    int fin = BIH_FINDEX(gbl.entbih);
    LL_MDRef lcl = lldbg_emit_local_variable(di, sptr, fin, true);
    insert_llvm_dbg_value(ld, lcl, sptr, type);
  }
}

/**
   \brief Insert an LLVM load instruction and return the loaded value.

   The address operand must be a pointer type that points to the load type.

   The flags provide the alignment and volatility, see
   ldst_instr_flags_from_dtype().
 */
static OPERAND *
gen_load(OPERAND *addr, LL_Type *type, LL_InstrListFlags flags)
{
  OPERAND *ld = ad_csed_instr(I_LOAD, 0, type, addr, flags, false);
  consLoadDebug(ld, addr, type);
  return ld;
}

INLINE static void
make_store(OPERAND *sop, OPERAND *address_op, LL_InstrListFlags flags)
{
  INSTR_LIST *Curr_Instr = mk_store_instr(sop, address_op);
  Curr_Instr->flags |= flags;
}

static OPERAND *
gen_convert_vector(int ilix)
{
  LL_Type *ll_src, *ll_dst;
  OPERAND *operand;
  DTYPE dtype_dst = ILI_DTyOPND(ilix, 2);
  DTYPE dtype_src = ILI_DTyOPND(ilix, 3);

  ll_dst = make_lltype_from_dtype(dtype_dst);
  ll_src = make_lltype_from_dtype(dtype_src);
  assert(ll_dst->data_type == LL_VECTOR,
         "gen_convert_vector(): vector type expected for dst",
         ll_dst->data_type, ERR_Fatal);
  assert(ll_src->data_type == LL_VECTOR,
         "gen_convert_vector(): vector type expected for src",
         ll_src->data_type, ERR_Fatal);
  operand = gen_llvm_expr(ILI_OPND(ilix, 1), ll_src);
  switch (ll_dst->sub_types[0]->data_type) {
  case LL_I1:
  case LL_I8:
  case LL_I16:
  case LL_I24:
  case LL_I32:
  case LL_I40:
  case LL_I48:
  case LL_I56:
  case LL_I64:
  case LL_I128:
  case LL_I256:
    switch (ll_src->sub_types[0]->data_type) {
    case LL_I1:
    case LL_I8:
    case LL_I16:
    case LL_I24:
    case LL_I32:
    case LL_I40:
    case LL_I48:
    case LL_I56:
    case LL_I64:
    case LL_I128:
    case LL_I256:
      if (DT_ISUNSIGNED(dtype_dst))
        operand->flags |= OPF_ZEXT;
      return convert_int_size(ilix, operand, ll_dst);
    case LL_FLOAT:
    case LL_DOUBLE:
      if (DT_ISUNSIGNED(dtype_dst))
        return convert_float_to_uint(operand, ll_dst);
      return convert_float_to_sint(operand, ll_dst);
    default:
      break;
    }
    break;
  case LL_FLOAT:
    switch (ll_src->sub_types[0]->data_type) {
    case LL_I1:
    case LL_I8:
    case LL_I16:
    case LL_I24:
    case LL_I32:
    case LL_I40:
    case LL_I48:
    case LL_I56:
    case LL_I64:
    case LL_I128:
    case LL_I256:
      if (DT_ISUNSIGNED(dtype_src))
        return convert_uint_to_float(operand, ll_dst);
      return convert_sint_to_float(operand, ll_dst);
    case LL_DOUBLE:
      return convert_float_size(operand, ll_dst);
    default:
      break;
    }
    break;
  case LL_DOUBLE:
    switch (ll_src->sub_types[0]->data_type) {
    case LL_I1:
    case LL_I8:
    case LL_I16:
    case LL_I24:
    case LL_I32:
    case LL_I40:
    case LL_I48:
    case LL_I56:
    case LL_I64:
    case LL_I128:
    case LL_I256:
      if (DT_ISUNSIGNED(dtype_src))
        return convert_uint_to_float(operand, ll_dst);
      return convert_sint_to_float(operand, ll_dst);
    case LL_FLOAT:
      return convert_float_size(operand, ll_dst);
    default:
      break;
    }
    break;
  default:
    assert(0, "gen_convert_vector(): unhandled vector type for dst",
           ll_dst->sub_types[0]->data_type, ERR_Fatal);
  }
  assert(0, "gen_convert_vector(): unhandled vector type for src",
         ll_src->sub_types[0]->data_type, ERR_Fatal);
  return NULL;
}

static OPERAND *
gen_bitcast_vector(int ilix)
{
  LL_Type *ll_src, *ll_dst;
  OPERAND *operand;

  ll_src = make_lltype_from_dtype((DTYPE)ILI_OPND(ilix, 3));
  ll_dst = make_lltype_from_dtype((DTYPE)ILI_OPND(ilix, 2));
  assert(ll_src->data_type == LL_VECTOR,
         "gen_bitcast_vector(): source type is not a vector", ll_src->data_type,
         ERR_Fatal);
  assert(ll_dst->data_type == LL_VECTOR,
         "gen_bitcast_vector(): destination type is not a vector",
         ll_dst->data_type, ERR_Fatal);
  operand = gen_llvm_expr(ILI_OPND(ilix, 1), ll_src);
  return convert_operand(operand, ll_dst, I_BITCAST);
}

static OPERAND *
gen_binary_vexpr(int ilix, int itype_int, int itype_uint, int itype_float)
{
  DTYPE vect_dtype = ili_get_vect_dtype(ilix);
  assert(vect_dtype,
         "gen_binary_vexpr(): called with non vector type for ilix ", ilix,
         ERR_Fatal);
  switch (DTY(DTySeqTyElement(vect_dtype))) {
  case TY_REAL:
  case TY_DBLE:
    return gen_binary_expr(ilix, itype_float);
  case TY_INT:
  case TY_SINT:
  case TY_BINT:
  case TY_INT8:
    return gen_binary_expr(ilix, itype_int);
  case TY_UINT:
  case TY_USINT:
  case TY_LOG:
  case TY_UINT8:
    return gen_binary_expr(ilix, itype_uint);
  default:
    assert(0, "gen_binary_vexpr(): vector type not yet handled for ilix ", ilix,
           ERR_Fatal);
  }
  return NULL;
}

/**
   \brief Is \p ilix a candidate for translation to FMA?
   \param ilix   The index of the ILI
   \return ILI of the multiply operand or 0 if not a candidate

   This inspects the operation at \p ilix to determine if it matches any of the
   suitable forms for a fused multiply add instruction.
   <pre>
   ([-] (([-] A) * ([-] B))) + [-] C
   </pre>
   where <tt>[-]</tt> is an optional negation/subraction.
 */
static int
fused_multiply_add_candidate(int ilix)
{
  int l, r, lx, rx;

  switch (ILI_OPC(ilix)) {
  default:
    break;
#if defined(USE_FMA_EXTENSIONS)
  case IL_FSUB:
#endif
  case IL_FADD:
    lx = ILI_OPND(ilix, 1);
    l = ILI_OPC(lx);
    if (l == IL_FMUL)
      return lx;
    rx = ILI_OPND(ilix, 2);
    r = ILI_OPC(rx);
    if (r == IL_FMUL)
      return rx;
#if defined(USE_FMA_EXTENSIONS)
    if ((l == IL_FNEG) && (ILI_OPC(ILI_OPND(lx, 1)) == IL_FMUL))
      return lx;
    if ((r == IL_FNEG) && (ILI_OPC(ILI_OPND(rx, 1)) == IL_FMUL))
      return rx;
#endif
    break;
#if defined(USE_FMA_EXTENSIONS)
  case IL_DSUB:
#endif
  case IL_DADD:
    lx = ILI_OPND(ilix, 1);
    l = ILI_OPC(lx);
    if (l == IL_DMUL)
      return lx;
    rx = ILI_OPND(ilix, 2);
    r = ILI_OPC(rx);
    if (r == IL_DMUL)
      return rx;
#if defined(USE_FMA_EXTENSIONS)
    if ((l == IL_DNEG) && (ILI_OPC(ILI_OPND(lx, 1)) == IL_DMUL))
      return lx;
    if ((r == IL_DNEG) && (ILI_OPC(ILI_OPND(rx, 1)) == IL_DMUL))
      return rx;
#endif
    break;
  }
  return 0;
}

#if defined(TARGET_LLVM_X8664)
/**
   \brief Get the x86 intrinsic name of the MAC instruction
   \param swap  [output] true if caller must swap arguments
   \param fneg  [output] true if multiply result is negated
   \param ilix  The root of the MAC (must be an ADD or SUB)
   \param l     The (original) lhs ili
   \param r     The (original) rhs ili
   \return the name of the intrinsic (sans "llvm." prefix)
 */
static const char *
get_mac_name(int *swap, int *fneg, int ilix, int matches, int l, int r)
{
  int opc;

  *swap = (matches == r);
  opc = ILI_OPC((*swap) ? r : l);
  *fneg = (opc == IL_FNEG) || (opc == IL_DNEG);
  switch (ILI_OPC(ilix)) {
  default:
    break;
  case IL_FADD:
    return (*fneg) ? "x86.fma.vfnmadd.ss" : "x86.fma.vfmadd.ss";
  case IL_DADD:
    return (*fneg) ? "x86.fma.vfnmadd.sd" : "x86.fma.vfmadd.sd";
  case IL_FSUB:
    if (*swap) {
      return (*fneg) ? "x86.fma.vfmadd.ss" : "x86.fma.vfnmadd.ss";
    }
    return (*fneg) ? "x86.fma.vfnmsub.ss" : "x86.fma.vfmsub.ss";
  case IL_DSUB:
    if (*swap) {
      return (*fneg) ? "x86.fma.vfmadd.sd" : "x86.fma.vfnmadd.sd";
    }
    return (*fneg) ? "x86.fma.vfnmsub.sd" : "x86.fma.vfmsub.sd";
  }
  assert(0, "does not match MAC", opc, ERR_Fatal);
  return "";
}
#endif

#ifndef USE_FMA_EXTENSIONS
/**
   \brief Put the candidate in proper canonical form

   The canonical form is lhs = (a * b), rhs = c, lhs + rhs.
 */
static void
fused_multiply_add_canonical_form(INSTR_LIST *addInsn, int matches, ILI_OP opc,
                                  OPERAND **l, OPERAND **r, int *lhs_ili,
                                  int *rhs_ili)
{
  ILI_OP lopc;
  ILI_OP negOpc;

  (*l)->next = (*r)->next = NULL;

  if (opc == IL_FSUB) {
    /* negate the rhs. t1 - t2 => t1 + (-t2) */
    negOpc = IL_FNEG;
  } else if (opc == IL_DSUB) {
    negOpc = IL_DNEG;
  } else {
    /* it's already an ADD */
    negOpc = IL_NONE;
  }

  if (matches == *rhs_ili) {
    /* multiply on right, exchange lhs and rhs. Don't rewrite anything yet. */
    int tmp = *rhs_ili;
    OPERAND *t = *r;
    *rhs_ili = *lhs_ili;
    *r = *l;
    *lhs_ili = tmp;
    *l = t;
  } else if (negOpc) {
    /* handle subtract form when multiply was already on the left */
    const int ropc = ILI_OPC(*rhs_ili);
    if ((ropc == IL_FNEG) || (ropc == IL_DNEG)) {
      /* double negative */
      (*r)->tmps->use_count--;
      *r = (*r)->tmps->info.idef->operands->next;
    } else {
      OPERAND *neg = gen_llvm_expr(ad1ili(negOpc, *rhs_ili), (*r)->ll_type);
      if (neg->tmps)
        neg->tmps->info.idef->operands->next = *r;
      *r = neg;
    }
    negOpc = IL_NONE;
  }

  /* negOpc implies a swap was made. Fixup any negations now. */
  lopc = ILI_OPC(*lhs_ili);
  if (negOpc && ((lopc == IL_FNEG) || (lopc == IL_DNEG))) {
    /* double negation. -(-(a * b)) => (a * b) */
    (*l)->tmps->use_count--;
    *l = (*l)->tmps->info.idef->operands->next;
  } else if (negOpc || (lopc == IL_FNEG) || (lopc == IL_DNEG)) {
    /* swap mult and negate.  -(a * b) => (-a) * b */
    OPERAND *n, *newMul;
    /* l has form: (a * b) or (0 - (a * b)) */
    OPERAND *mul = negOpc ? *l : (*l)->tmps->info.idef->operands->next;
    OPERAND *mul_l = mul->tmps->info.idef->operands; /* a term */
    OPERAND *mul_r = mul_l->next;                    /* b term */
    int mulili = negOpc ? *lhs_ili : ILI_OPND(*lhs_ili, 1);
    int muliliop = ILI_OPC(mulili);
    int mulili_l = ILI_OPND(mulili, 1);
    LL_Type *fTy = mul_l->ll_type;
    /* create n, where n ::= -a */
    if (muliliop == IL_FMUL) {
      n = gen_llvm_expr(ad1ili(IL_FNEG, mulili_l), fTy);
    } else {
      assert(ILI_OPC(mulili) == IL_DMUL, "unexpected expr", mulili, ERR_Fatal);
      n = gen_llvm_expr(ad1ili(IL_DNEG, mulili_l), fTy);
    }
    /* rebuild the multiply */
    if (n->tmps)
      n->tmps->info.idef->operands->next = mul_l;
    n->next = mul_r;
    newMul = make_tmp_op(mul->ll_type, make_tmps());
    ad_instr(mulili, gen_instr(I_FMUL, newMul->tmps, mul->ll_type, n));
    *l = newMul; /* l ::= (n * b) = (-a * b) */
  }
}
#endif // !USE_FMA_EXTENSIONS

/**
   \brief Does this multiply op have more than one use?
   \param multop A multiply operation (unchecked)
 */
INLINE static bool
fma_mult_has_mult_uses(OPERAND *multop)
{
  return (!multop->tmps) || (multop->tmps->use_count > 1);
}

static void
overwrite_fma_add(INSTR_LIST *oldAdd, INSTR_LIST *newFma, INSTR_LIST *last)
{
  INSTR_LIST *prev = oldAdd->prev;
  INSTR_LIST *start = last->next;
  OPERAND *op;
  last->next = NULL;
  start->prev = NULL;
  if (start != newFma) {
    prev->next = start;
    start->prev = prev;
    newFma->prev->next = oldAdd;
    oldAdd->prev = newFma->prev;
  }
  /* updated in place since uses have pointers to oldAdd */
  oldAdd->rank = newFma->rank;
  oldAdd->i_name = newFma->i_name;
  oldAdd->ilix = newFma->ilix;
  oldAdd->flags = newFma->flags;
  oldAdd->ll_type = newFma->ll_type;
  oldAdd->operands = newFma->operands;
  /* force the DCE pass to keep the FMA arguments alive */
  for (op = oldAdd->operands; op; op = op->next) {
    if (op->tmps)
      op->tmps->use_count += 100;
  }
}

#if defined(TARGET_LLVM_X8664)
static OPERAND *
x86_promote_to_vector(LL_Type *eTy, LL_Type *vTy, OPERAND *op)
{
  OPERAND *op1, *undefop, *next;

  if (op == NULL)
    return NULL;

  undefop = make_undef_op(vTy);
  next = op->next;
  op1 = gen_insert_vector(undefop, gen_copy_op(op), 0);
  op1->next = x86_promote_to_vector(eTy, vTy, next);
  return op1;
}
#endif

/**
   \brief Replace add instruction with an FMA when all conditions are met
 */
static void
maybe_generate_fma(int ilix, INSTR_LIST *insn)
{
  int lhs_ili = ILI_OPND(ilix, 1);
  int rhs_ili = ILI_OPND(ilix, 2);
  int matches;
  ILI_OP opc;
  int isSinglePrec;
  const char *intrinsicName;
  OPERAND *l_l, *l_r, *l, *r, *binops, *fmaop;
  LL_Type *llTy;
  INSTR_LIST *fma, *last;
#if defined(USE_FMA_EXTENSIONS)
  int swap, fneg;
  OPERAND *mul;
#endif
#if defined(TARGET_LLVM_X8664)
  LL_Type *vTy;
#endif

  last = llvm_info.last_instr;
  binops = insn->operands;
  if (lhs_ili == rhs_ili)
    return;

  matches = fused_multiply_add_candidate(ilix);
  if (!matches)
    return;

  l = (matches == lhs_ili) ? binops : binops->next;
  if (fma_mult_has_mult_uses(l))
    return;

  opc = ILI_OPC(matches);
  if ((opc == IL_FNEG) || (opc == IL_DNEG))
    if (fma_mult_has_mult_uses(l->tmps->info.idef->operands->next))
      return;
  l = binops;
  r = binops->next;
  opc = ILI_OPC(ilix);
  /* put in canonical form: left:mult, right:_ */
  killCSE = true;
  clear_csed_list();
  isSinglePrec = (opc == IL_FADD) || (opc == IL_FSUB);
  llTy = make_lltype_from_dtype(isSinglePrec ? DT_FLOAT : DT_DBLE);
#if defined(USE_FMA_EXTENSIONS)
  /* use intrinsics for specific target instructions */
  intrinsicName = get_mac_name(&swap, &fneg, ilix, matches, lhs_ili, rhs_ili);
  if (swap) {
    OPERAND *t = r;
    r = l;
    l = t;
  }
  mul = fneg ? l->tmps->info.idef->operands->next : l;
  l_l = mul->tmps->info.idef->operands;
  l_r = l_l->next;
  l_r->next = r;
  r->next = NULL;
  l->tmps->use_count--;
#if defined(TARGET_LLVM_X8664)
  vTy = ll_get_vector_type(llTy, (isSinglePrec ? 4 : 2));
  l_l = x86_promote_to_vector(llTy, vTy, l_l);
  llTy = vTy;
#endif
#else /* not Power/LLVM or X86-64/LLVM */
  /* Instead of forcing the compiler's hand to emit an FMA instruction
   * for @llvm.fma.*, it is better to use @llvm.fmuladd.* to suggest
   * using an FMA if available and profitable. */
  fused_multiply_add_canonical_form(insn, matches, opc, &l, &r, &lhs_ili,
                                    &rhs_ili);
  /* llvm.fmuladd ::= madd(l.l * l.r + r), assemble args in the LLVM order */
  l_l = l->tmps->info.idef->operands;
  l_r = l_l->next;
  l_r->next = r;
  l->tmps->use_count--;
  intrinsicName = isSinglePrec ? "fmuladd.f32" : "fmuladd.f64";
#endif
  int FastMath = !flg.ieee || XBIT(216, 1);
  fmaop = (FastMath ? gen_call_llvm_fm_intrinsic :
                      gen_call_llvm_non_fm_math_intrinsic)
            (intrinsicName, l_l, llTy, NULL, I_PICALL);
#if defined(TARGET_LLVM_X8664)
  fmaop->tmps->use_count++;
  fmaop = gen_extract_vector(fmaop, 0);
#endif
  fmaop->tmps->use_count++;
  fma = fmaop->tmps->info.idef;
  overwrite_fma_add(insn, fma, last);
  if (DBGBIT(12, 0x40000))
    printf("fma %s inserted at ili %d\n", intrinsicName, ilix);
  ccff_info(MSGOPT, "OPT051", gbl.findex, gbl.lineno,
            "FMA (fused multiply-add) instruction(s) generated", NULL);
  llvm_info.last_instr = last;
  killCSE = false;
}

/**
   \brief Find and rewrite any multiply-add opportunities
   \param isns  The list of instructions
 */
static void
fma_rewrite(INSTR_LIST *isns)
{
  INSTR_LIST *p;

  for (p = isns; p; p = p->next) {
    int ilx = p->ilix;
    if (ilx && p->tmps && p->operands && (p->tmps->use_count > 0))
      maybe_generate_fma(ilx, p);
  }
}

/**
   \brief Undoes the multiply-by-reciprocal transformation
   \param isns  The list of instructions
 */
static void
undo_recip_div(INSTR_LIST *isns)
{
  INSTR_LIST *p;

  for (p = isns; p; p = p->next)
    if (p->ilix && p->tmps && p->operands && (p->tmps->use_count > 0) &&
        (p->i_name == I_FMUL))
      maybe_undo_recip_div(p);
}

static OPERAND *
gen_binary_expr(int ilix, int itype)
{
  int lhs_ili, rhs_ili;
  int vect_type;
  DTYPE vect_dtype = DT_NONE;
  LL_InstrListFlags flags = InstrListFlagsNull;
  ILI_OP opc = ILI_OPC(ilix);
  OPERAND *operand, *binops;
  LL_Type *instr_type;
  INT val[2];

  DBGTRACEIN2(" ilix: %d(%s)", ilix, IL_NAME(opc))

  lhs_ili = ILI_OPND(ilix, 1);
  rhs_ili = ILI_OPND(ilix, 2);

  switch (opc) {
  case IL_VMUL:
  case IL_IMUL:
  case IL_KMUL:
  case IL_VSUB:
  case IL_ISUB:
  case IL_KSUB:
  case IL_VADD:
  case IL_IADD:
  case IL_KADD:
  case IL_VLSHIFTV:
  case IL_VLSHIFTS:
  case IL_LSHIFT:
  case IL_KLSHIFT:
  case IL_VNEG:
  case IL_INEG:
  case IL_KNEG:
    flags |= NOSIGNEDWRAP;
    break;
  default:
    break;
  }

  /* handle conditional vectorization  where we want the inverse mask -
   * (1) in the case opc == IL_VNOT and the lhs_ili is a VCMP
   * (2) in the case opc == IL_VNOT and the lhs_ili is a VPERMUTE pointing
   * to a VCMP. Here we have a half-size predicate compared to computation.
   */
  if (opc == IL_VNOT && (ILI_OPC(lhs_ili) == IL_VCMP ||
                         (ILI_OPC(lhs_ili) == IL_VPERMUTE &&
                          ILI_OPC(ILI_OPND(lhs_ili, 1)) == IL_VCMP))) {
    int num_elem, constant;
    LL_Type *bit_type, *mask_type;
    OPERAND *bit_mask_of_ones;
    DTYPE vdt, ones_dtype;
    SPTR vcon1_sptr = SPTR_NULL;
    vect_dtype = ili_get_vect_dtype(lhs_ili);
    num_elem = DTyVecLength(vect_dtype);
    switch (DTySeqTyElement(vect_dtype)) {
    case DT_INT:
    case DT_FLOAT:
      if (ILI_OPC(lhs_ili) == IL_VPERMUTE)
        bit_type = make_int_lltype(64); /* half-size predicate */
      else
        bit_type = make_int_lltype(32);
      ones_dtype = DT_INT;
      vdt = get_vector_dtype(ones_dtype, num_elem);
      vcon1_sptr = get_vcon_scalar(0xffffffff, vdt);
      break;
    case DT_INT8:
    case DT_DBLE:
      bit_type = make_int_lltype(64);
      val[0] = 0xffffffff;
      val[1] = 0xffffffff;
      ones_dtype = DT_INT8;
      vdt = get_vector_dtype(ones_dtype, num_elem);
      constant = getcon(val, ones_dtype);
      vcon1_sptr = get_vcon_scalar(constant, vdt);
      break;
    default:
      assert(0, "Unexpected dtype for VNOT", DTySeqTyElement(vect_dtype),
             ERR_Fatal);
    }
    bit_mask_of_ones = gen_llvm_expr(ad1ili(IL_VCON, vcon1_sptr), 0);
    mask_type = ll_get_vector_type(bit_type, num_elem);
    binops = gen_llvm_expr(lhs_ili, mask_type);
    instr_type = binops->ll_type;
    binops->next = convert_int_size(ilix, bit_mask_of_ones, instr_type);
    goto make_binary_expression;
  }

  /* account for the *NEG ili - LLVM treats all of these as subtractions
   * from zero.
   */
  if (!rhs_ili || !IL_ISLINK(opc, 2)) {
    rhs_ili = lhs_ili;
    switch (opc) {
    case IL_NOT:
    case IL_UNOT:
      lhs_ili = ad_icon(-1);
      break;
    case IL_KNOT:
    case IL_UKNOT:
      lhs_ili = ad_kconi(-1);
      break;
    case IL_VNOT:
      vect_dtype = ili_get_vect_dtype(ilix);
      switch (DTY(DTySeqTyElement(vect_dtype))) {
      case TY_UINT8:
      case TY_INT8: {
        INT num[2];
        ISZ_2_INT64(-1, num);
        lhs_ili = ad1ili(
            IL_VCON, get_vconi((DTYPE)ILI_OPND(ilix, 2), getcon(num, DT_INT8)));
      } break;
      case TY_REAL:
      case TY_DBLE:
        assert(0, "gen_binary_expr(): VNOT of float/double not handled yet", 0,
               ERR_Fatal);
        break;
      default:
        lhs_ili = ad1ili(IL_VCON, get_vconi((DTYPE)ILI_OPND(ilix, 2), -1));
      }
      break;
    case IL_INEG:
    case IL_UINEG:
      lhs_ili = ad_icon(0);
      break;
    case IL_KNEG:
    case IL_UKNEG:
      lhs_ili = ad_kconi(0);
      break;
    case IL_FNEG:
      lhs_ili = ad1ili(IL_FCON, stb.fltm0);
      break;
    case IL_DNEG:
      lhs_ili = ad1ili(IL_DCON, stb.dblm0);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case IL_QNEG:
      lhs_ili = ad1ili(IL_QCON, stb.quad0);
      break;
#endif
#ifdef LONG_DOUBLE_FLOAT128
    case IL_FLOAT128CHS:
      lhs_ili = ad1ili(IL_FLOAT128CON, stb.float128_0);
      break;
#endif
    case IL_VNEG:
      vect_dtype = ILI_DTyOPND(ilix, 2);
      lhs_ili = ad1ili(IL_VCON, get_vconm0(vect_dtype));
      vect_type = DTY(DTySeqTyElement(vect_dtype));
      break;
    default:
      DBGTRACE1("#opcode %s not handled as *NEG ili", IL_NAME(opc))
      assert(0, "gen_binary_expr(): confusion with opcode", opc, ERR_Fatal);
    }
  }
  vect_dtype = ili_get_vect_dtype(ilix);
  if (vect_dtype) {
    instr_type = make_lltype_from_dtype(vect_dtype);
  } else
      if ((instr_type = make_type_from_opc(opc)) == NULL) {
    assert(0, "gen_binary_expr(): no type information", 0, ERR_Fatal);
  }

  DBGTRACE2("#generating first binary operand, lhs_ili: %d(%s)", lhs_ili,
            IL_NAME(ILI_OPC(lhs_ili)))

  binops = gen_llvm_expr(lhs_ili, instr_type);

  DBGTRACE2("#generating second binary operand, rhs_ili: %d(%s)", rhs_ili,
            IL_NAME(ILI_OPC(rhs_ili)))

  switch (opc) {
  case IL_KLSHIFT:
  case IL_KURSHIFT:
  case IL_KARSHIFT:
    binops->next = gen_llvm_expr(rhs_ili, make_lltype_from_dtype(DT_UINT8));
    break;
  case IL_LSHIFT:
  case IL_ULSHIFT:
  case IL_URSHIFT:
  case IL_RSHIFT:
  case IL_ARSHIFT:
    binops->next = gen_llvm_expr(rhs_ili, make_lltype_from_dtype(DT_UINT));
    break;
  case IL_VLSHIFTS:
  case IL_VRSHIFTS:
  case IL_VURSHIFTS:
    binops->next =
        gen_temp_to_vector(rhs_ili, make_vtype(DTySeqTyElement(vect_dtype),
                                               DTyVecLength(vect_dtype)));
    break;
  default:
    binops->next = gen_llvm_expr(rhs_ili, instr_type);
  }

make_binary_expression:
  /* now make the new binary expression */
  operand =
      ad_csed_instr((LL_InstrName)itype, ilix, instr_type, binops, flags, true);

  DBGTRACEOUT1(" returns operand %p", operand)

  return operand;
} /* gen_binary_expr */

/* Compute the high result bits of a multiplication.
 *
 * ilix should be an IL_KMULH or IL_UKMULH instruction.
 */
static OPERAND *
gen_mulh_expr(int ilix)
{
  LL_InstrName ext_instr, shr_instr;
  LL_InstrListFlags mul_flags;
  int bits = 64;
  OPERAND *lhs, *rhs, *result;
  LL_Type *op_llt, *big_llt;

  switch (ILI_OPC(ilix)) {
  case IL_KMULH:
    ext_instr = I_SEXT;
    mul_flags = NOSIGNEDWRAP;
    shr_instr = I_ASHR;
    break;
  case IL_UKMULH:
    ext_instr = I_ZEXT;
    mul_flags = NOUNSIGNEDWRAP;
    shr_instr = I_LSHR;
    break;
  default:
    interr("Unknown mulh opcode", ILI_OPC(ilix), ERR_Fatal);
    return NULL;
  }

  /* Extend both sides to i128. */
  op_llt = make_int_lltype(bits);
  big_llt = make_int_lltype(2 * bits);
  lhs = gen_llvm_expr(ILI_OPND(ilix, 1), op_llt);
  rhs = gen_llvm_expr(ILI_OPND(ilix, 2), op_llt);
  lhs = ad_csed_instr(ext_instr, ilix, big_llt, lhs, InstrListFlagsNull, true);
  rhs = ad_csed_instr(ext_instr, ilix, big_llt, rhs, InstrListFlagsNull, true);

  /* Do the multiplication in 128 bits. */
  lhs->next = rhs;
  result = ad_csed_instr(I_MUL, ilix, big_llt, lhs, mul_flags, true);

  /* Shift down to get the high bits. */
  result->next = make_constval_op(big_llt, bits, 0);
  result =
      ad_csed_instr(shr_instr, ilix, big_llt, result, InstrListFlagsNull, true);

  /* Finally truncate down to 64 bits */
  return ad_csed_instr(I_TRUNC, ilix, op_llt, result, InstrListFlagsNull, true);
}

/**
   return new operand of type OT_TMP as result of converting cast_op.  If
   cast_op has a next, make sure the next pointer is dealt with properly BEFORE
   the call to make_bitcast().
 */
static OPERAND *
make_bitcast(OPERAND *cast_op, LL_Type *rslt_type)
{
  OPERAND *operand;
  TMPS *new_tmps;
  INSTR_LIST *Curr_Instr, *instr;

  if (cast_op->ll_type->data_type == LL_PTR &&
      rslt_type->data_type == LL_PTR) {
    operand = gen_copy_op(cast_op);
    operand->ll_type = rslt_type;
    return operand;
  }

  if (strict_match(cast_op->ll_type, rslt_type))
    return gen_copy_op(cast_op);

  assert(ll_type_bytes(cast_op->ll_type) == ll_type_bytes(rslt_type),
         "sizes do not match", 0, ERR_Fatal);

  if (cast_op->ot_type == OT_TMP) {
    instr = cast_op->tmps->info.idef;
    if (instr && (instr->i_name == I_BITCAST) &&
        strict_match(instr->operands->ll_type, rslt_type)) {
      return gen_copy_op(instr->operands);
    }
  }

  DBGTRACEIN1(" cast op: %p", cast_op)
  DBGDUMPLLTYPE("result type ", rslt_type)
  DBGDUMPLLTYPE("cast_op type ", cast_op->ll_type)

  if (ENABLE_CSE_OPT) {
    instr = llvm_info.last_instr;
    while (instr) {
      switch (instr->i_name) {
      case I_BR:
      case I_INDBR:
      case I_NONE:
        instr = NULL;
        break;
      case I_BITCAST:
        if (same_op(cast_op, instr->operands) &&
            strict_match(rslt_type, instr->ll_type)) {
          operand = make_tmp_op(rslt_type, instr->tmps);
          DBGTRACEOUT1(" returns CSE'd operand %p\n", operand)

          return operand;
        }
        FLANG_FALLTHROUGH;
      default:
        if (instr->flags & STARTEBB)
          instr = NULL;
        else
          instr = instr->prev;
      }
    }
  }
  Curr_Instr = gen_instr(I_BITCAST, new_tmps = make_tmps(), rslt_type, cast_op);
  cast_op->next = NULL;
  ad_instr(0, Curr_Instr);
  /* now build the operand */
  operand = make_tmp_op(rslt_type, new_tmps);

  DBGTRACEOUT1(" returns operand %p\n", operand)

  return operand;
} /* make_bitcast */

/* return new operand of type OT_TMP as result of converting convert_op,
 * which is floating pt but needs coercion to the larger size within rslt_type.
 * If the passed operand convert_op has a next pointer, make sure it
 * is handled BEFORE this call!
 */
static OPERAND *
convert_float_size(OPERAND *convert_op, LL_Type *rslt_type)
{
  LL_Type *ty1, *ty2;
  int kind1, kind2;
  int conversion_instr;
  OPERAND *op_tmp;
  TMPS *new_tmps;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN1(" convert op %p", convert_op)
  DBGDUMPLLTYPE("result type ", rslt_type)

  ty1 = convert_op->ll_type;
  ty2 = rslt_type;
  kind1 = (ty1->data_type == LL_VECTOR) ? ty1->sub_types[0]->data_type
                                        : ty1->data_type;
  kind2 = (ty2->data_type == LL_VECTOR) ? ty2->sub_types[0]->data_type
                                        : ty2->data_type;
  if (kind1 > kind2)
    conversion_instr = I_FPTRUNC;
  else
    conversion_instr = I_FPEXT;
  new_tmps = make_tmps();
  op_tmp = make_tmp_op(ty2, new_tmps);
  Curr_Instr = gen_instr((LL_InstrName)conversion_instr, new_tmps,
                         op_tmp->ll_type, convert_op);
  convert_op->next = NULL;
  ad_instr(0, Curr_Instr);

  DBGTRACEOUT1(" returns operand %p", op_tmp)
  return op_tmp;
} /* convert_float_size */

/** return new operand of type OT_TMP as result of converting convert_op, which
    is an int but needs coercion to the int size within rslt_type.  If the
    passed operand convert_op has a next pointer, make sure it is handled BEFORE
    this call! */
static OPERAND *
convert_int_size(int ilix, OPERAND *convert_op, LL_Type *rslt_type)
{
  LL_Type *ty1, *ty2, *ll_type;
  int size1, size2, flags1, conversion_instr;
  enum LL_BaseDataType kind1, kind2;
  OPERAND *op_tmp;
  TMPS *new_tmps;

  DBGTRACEIN1(" convert op %p", convert_op)
  DBGDUMPLLTYPE("result type ", rslt_type)

  ty1 = convert_op->ll_type;
  ty2 = rslt_type;
  if (ty1->data_type == LL_VECTOR) {
    kind1 = ty1->sub_types[0]->data_type;
    size1 = ll_type_int_bits(ty1->sub_types[0]);
    if (!size1)
      size1 = ll_type_bytes(ty1->sub_types[0]) * BITS_IN_BYTE;
  } else {
    kind1 = ty1->data_type;
    size1 = ll_type_int_bits(ty1);
    if (!size1)
      size1 = ll_type_bytes(ty1) * BITS_IN_BYTE;
  }
  flags1 = convert_op->flags;
  if (ty2->data_type == LL_VECTOR) {
    kind2 = ty2->sub_types[0]->data_type;
    size2 = ll_type_int_bits(ty2->sub_types[0]);
    if (!size2)
      size2 = ll_type_bytes(ty2->sub_types[0]) * BITS_IN_BYTE;
  } else {
    kind2 = ty2->data_type;
    size2 = ll_type_int_bits(ty2);
    if (!size2)
      size2 = ll_type_bytes(ty2) * BITS_IN_BYTE;
  }
  if (ty1->data_type != LL_VECTOR) {
    assert(ll_type_int_bits(ty1),
           "convert_int_size(): expected int type for src", kind1, ERR_Fatal);
  }
  if (ty2->data_type != LL_VECTOR) {
    assert(ll_type_int_bits(ty2),
           "convert_int_size(): expected int type for dst", kind2, ERR_Fatal);
  }
  /* need conversion, either extension or truncation */
  if (size1 < size2) {
    /* extension */
    conversion_instr = (flags1 & OPF_ZEXT) ? I_ZEXT : I_SEXT;
  } else if (size1 > size2) {
    /* size1 > size2, truncation */
    conversion_instr = I_TRUNC;
  } else {
    DBGTRACE("#conversion of same size, should be a conversion signed/unsigned")
    DBGTRACEOUT1(" returns operand %p", convert_op)
    return convert_op;
  }

  DBGTRACE2("#coercing ints to size %d with instruction %s", size2,
            llvm_instr_names[conversion_instr])

  new_tmps = make_tmps();
  ll_type = ty2;
  op_tmp = ad_csed_instr((LL_InstrName)conversion_instr, ilix, ll_type,
                         convert_op, InstrListFlagsNull, true);

  DBGTRACEOUT1(" returns operand %p", op_tmp)
  return op_tmp;
} /* convert_int_size */

static OPERAND *
convert_operand(OPERAND *convert_op, LL_Type *rslt_type,
                LL_InstrName convert_instruction)
{
  LL_Type *ty, *ll_type;
  int size;
  OPERAND *op_tmp;
  TMPS *new_tmps;
  INSTR_LIST *Curr_Instr;

  DBGTRACEIN1(" convert op %p", convert_op)
  DBGDUMPLLTYPE("result type ", rslt_type)

  ty = convert_op->ll_type;
  size = ll_type_bytes(ty) * BITS_IN_BYTE;
  new_tmps = make_tmps();
  ll_type = rslt_type;
  op_tmp = make_tmp_op(ll_type, new_tmps);
  Curr_Instr = gen_instr(convert_instruction, new_tmps, ll_type, convert_op);
  ad_instr(0, Curr_Instr);
  DBGTRACEOUT1(" returns operand %p", op_tmp)
  return op_tmp;
}

static OPERAND *
convert_int_to_ptr(OPERAND *convert_op, LL_Type *rslt_type)
{
  const LL_Type *llt = convert_op->ll_type;
  assert(llt,"convert_int_to_ptr(): missing incoming type",0,ERR_Fatal);
  assert(ll_type_int_bits(llt) == BITS_IN_BYTE * size_of(DT_CPTR),
         "Unsafe type for inttoptr", ll_type_int_bits(llt), ERR_Fatal);
  return convert_operand(convert_op, rslt_type, I_INTTOPTR);
}

static OPERAND *
sign_extend_int(OPERAND *op, unsigned result_bits)
{
  const LL_Type *llt = op->ll_type;
  assert(ll_type_int_bits(llt) && (ll_type_int_bits(llt) < result_bits),
         "sign_extend_int: bad type", ll_type_int_bits(llt), ERR_Fatal);
  return convert_operand(op, make_int_lltype(result_bits), I_SEXT);
}

static OPERAND *
zero_extend_int(OPERAND *op, unsigned result_bits)
{
  const LL_Type *llt = op->ll_type;
  assert(ll_type_int_bits(llt) && (ll_type_int_bits(llt) < result_bits),
         "zero_extend_int: bad type", ll_type_int_bits(llt), ERR_Fatal);
  return convert_operand(op, make_int_lltype(result_bits), I_ZEXT);
}

static INSTR_LIST *
remove_instr(INSTR_LIST *instr, bool update_usect_only)
{
  INSTR_LIST *prev, *next;
  OPERAND *operand;

  prev = instr->prev;
  next = instr->next;
  if (!update_usect_only) {
    if (next)
      next->prev = prev;
    else
      llvm_info.last_instr = prev;
    if (prev)
      prev->next = next;
    else
      Instructions = next;
  }
  for (operand = instr->operands; operand; operand = operand->next) {
    if (operand->ot_type == OT_TMP) {
      assert(operand->tmps, "remove_instr(): missing temp operand", 0,
             ERR_Fatal);
      operand->tmps->use_count--;
    }
  }

  return prev;
}

static bool
same_op(OPERAND *op1, OPERAND *op2)
{
  if (op1->ot_type != op2->ot_type)
    return false;
  switch (op1->ot_type) {
  case OT_TMP:
    return (op1->tmps == op2->tmps);
  case OT_VAR:
    return (op1->val.sptr == op2->val.sptr);
  case OT_CONSTVAL:
    return (op1->val.conval[0] == op2->val.conval[0]) &&
           (op1->val.conval[1] == op2->val.conval[1]);
  default:
    return false;
  }
}

/** Return true if a load can be moved upwards (backwards in time)
    over fencing specified by the given instruction. */
static bool
can_move_load_up_over_fence(INSTR_LIST *instr)
{
  switch (instr->flags & ATOMIC_MEM_ORD_FLAGS) {
  case ATOMIC_ACQUIRE_FLAG:
  case ATOMIC_ACQ_REL_FLAG:
  case ATOMIC_SEQ_CST_FLAG:
    return false;
  default:
    return true;
  }
}

/**
   \brief Clear DELETABLE flag from previous instructions
   \param ilix  The ILI of a LOAD instruction
 */
void
clear_deletable_flags(int ilix)
{
  INSTR_LIST *instr;
  int ld_nme;

  DEBUG_ASSERT(IL_TYPE(ILI_OPC(ilix)) == ILTY_LOAD, "must be load");
  ld_nme = ILI_OPND(ilix, 2);
  for (instr = llvm_info.last_instr; instr; instr = instr->prev) {
    if (instr->i_name == I_STORE) {
      if (instr->ilix == 0) {
        instr->flags &= ~DELETABLE;
        continue;
      }
      if (ld_nme == ILI_OPND(instr->ilix, 3)) {
        instr->flags &= ~DELETABLE;
        break;
      }
    }
  }
}

INLINE static OPERAND *
find_load_cse(int ilix, OPERAND *load_op, LL_Type *llt)
{
  INSTR_LIST *instr, *del_store_instr, *last_instr;
  int del_store_flags;
  int ld_nme;
  int c;

  if (new_ebb || (!ilix) || (IL_TYPE(ILI_OPC(ilix)) != ILTY_LOAD))
    return NULL;

  ld_nme = ILI_OPND(ilix, 2);
  if (ld_nme == NME_VOL) /* don't optimize a VOLATILE load */
    return NULL;

  /* If there is a deletable store to 'ld_nme', 'del_store_li', set
   * its 'deletable' flag to false.  We do this because 'ld_ili'
   * loads from that address, so we mustn't delete the preceding
   * store to it.  However, if the following LILI scan reaches
   * 'del_store_li', *and* we return the expression that is stored
   * by 'del_store_li', then we restore its 'deletable' flag, since
   * in that case the store *can* be deleted.
   * We track deletable store in EBB but perform CSE load opt only in
   * BB to avoid LLVM opt to fail, so we have to mark stores in EBB as
   * undeletable
   */
  del_store_instr = NULL;
  last_instr = NULL;

  for (instr = llvm_info.last_instr; instr; instr = instr->prev) {
    if ((instr->i_name == I_STORE) && instr->ilix &&
        (ld_nme == ILI_OPND(instr->ilix, 3))) {
      del_store_instr = instr;
      del_store_flags = del_store_instr->flags;
      del_store_instr->flags &= ~DELETABLE;
      break;
    }
    if (instr->flags & STARTEBB) {
      last_instr = (instr->i_name != I_NONE) ? instr : instr->prev;
      break;
    }
  }

  for (instr = llvm_info.last_instr; instr != last_instr; instr = instr->prev) {
    if (instr->ilix == ilix)
      return same_op(instr->operands, load_op)
                 ? make_tmp_op(instr->ll_type, instr->tmps)
                 : NULL;

    switch (instr->i_name) {
    case I_LOAD:
    case I_CMPXCHG:
    case I_ATOMICRMW:
    case I_FENCE:
      DEBUG_ASSERT(instr->ilix != 0 || instr->i_name == I_LOAD,
                   "missing ilix for I_CMPXCHG, I_ATOMICRMW, or I_FENCE");
      if (!can_move_load_up_over_fence(instr))
        return NULL;
      if (instr->i_name == I_LOAD)
        break;
      goto check_conflict;
    case I_STORE:
      if (instr->ilix == 0)
        return NULL;
      if (IL_TYPE(ILI_OPC(instr->ilix)) != ILTY_STORE)
        return NULL;
      /* must use ili_opnd() call to skip by CSExx, otherwise
       * may not get latest store to the load location.
       */
      if (ILI_OPND(ilix, 1) == ili_opnd(instr->ilix, 2)) {
        /* Maybe revisited to add conversion op */
        if (match_types(instr->operands->ll_type, llt) != MATCH_OK)
          return NULL;
        if (!same_op(instr->operands->next, load_op))
          return NULL;
        if (instr == del_store_instr)
          instr->flags = (LL_InstrListFlags)del_store_flags;
        return gen_copy_op(instr->operands);
      }
    check_conflict:
      c = enhanced_conflict(ld_nme, ILI_OPND(instr->ilix, 3));
      if (c == SAME || (flg.depchk && c != NOCONFLICT))
        return NULL;
      break;
    case I_INVOKE:
    case I_CALL:
      if (!(instr->flags & FAST_CALL))
        return NULL;
      break;
    case I_NONE:
    case I_BR:
    case I_INDBR:
      if (!ENABLE_ENHANCED_CSE_OPT)
        return NULL;
      break;
    default:
      break;
    }
  }

  return NULL;
}

/**
   \brief return new operand of type OT_TMP as result of loading \p load_op

   If \p load_op has a next, make sure the next pointer is dealt with properly
   \e BEFORE the call to make_load().

   \p flags is the instruction flags to set. Should usually be
   ldst_instr_flags_from_dtype() for natural alignment.
 */
static OPERAND *
make_load(int ilix, OPERAND *load_op, LL_Type *rslt_type, MSZ msz,
          unsigned flags)
{
  OPERAND *operand, *cse_op;
  TMPS *new_tmps;
  LL_Type *load_type;
  INSTR_LIST *Curr_Instr;

  assert(((int)msz) != -1, "make_load():adding a load because of a matchmem ?",
         0, ERR_Fatal);

  cse_op = NULL;
  if (ENABLE_CSE_OPT && !is_omp_atomic_ld(ilix)) {
    operand = find_load_cse(ilix, load_op, rslt_type);
    if (operand != NULL) {
      const int bits = ll_type_int_bits(operand->ll_type);
      if ((bits > 0) && (bits < 32)) {
        LL_Type *ll_tmp;

        switch (msz) {
        case MSZ_SBYTE:
        case MSZ_SHWORD:
          ll_tmp = operand->ll_type;
          operand->flags |= OPF_SEXT;
          operand->ll_type = ll_tmp;
          break;
        case MSZ_BYTE:
        case MSZ_UHWORD:
          ll_tmp = make_lltype_from_dtype(DT_UINT);
          operand->flags |= OPF_ZEXT;
          operand = convert_int_size(0, operand, ll_tmp);
          FLANG_FALLTHROUGH;
        default:
          break;
        }
      }
      cse_op = operand;
      return cse_op;
    }
  }
  load_type = load_op->ll_type;

  DBGTRACEIN2(" ilix %d, load op: %p", ilix, load_op)
  DBGDUMPLLTYPE("result type ", rslt_type)

  assert(load_type->data_type == LL_PTR, "make_load(): op not ptr type",
         load_type->data_type, ERR_Fatal);
  assert(match_types(load_type->sub_types[0], rslt_type) == MATCH_OK,
         "make_load(): types don't match", 0, ERR_Fatal);
  new_tmps = make_tmps();
  Curr_Instr = gen_instr(I_LOAD, new_tmps, rslt_type, load_op);
  if (rw_access_group) {
    flags |= LDST_HAS_ACCESSGRP_METADATA;
    Curr_Instr->misc_metadata = cons_vec_always_metadata();
  }
  Curr_Instr->flags = (LL_InstrListFlags)flags;
  load_op->next = NULL;
  ad_instr(ilix, Curr_Instr);
  /* make the new operand to be the temp */
  operand = make_tmp_op(rslt_type, new_tmps);
  consLoadDebug(operand, load_op, rslt_type);
  /* Need to make sure the char type is unsigned */
  if (ll_type_int_bits(operand->ll_type) &&
      (ll_type_int_bits(operand->ll_type) < 16)) {
    switch (msz) {
    case MSZ_UBYTE:
    case MSZ_UHWORD:
      operand->flags |= OPF_ZEXT;
      break;
    default:
      break;
    }
  }
  if (ll_type_int_bits(operand->ll_type) &&
      (ll_type_int_bits(operand->ll_type) < 32)) {
    switch (msz) {
    case MSZ_BYTE:
    case MSZ_UHWORD: {
      LL_Type *ll_tmp = make_lltype_from_dtype(DT_UINT);
      operand->flags |= OPF_ZEXT;
      operand = convert_int_size(0, operand, ll_tmp);
    } break;
    default:
      break;
    }
  }

  DBGTRACEOUT1(" returns operand %p", operand);
  return cse_op ? cse_op : operand;
}

SPTR
find_pointer_to_function(int ilix)
{
  int addr, addr_acon_ptr;
  SPTR sptr = SPTR_NULL;

  addr = ILI_OPND(ilix, 1);
  while (ILI_OPC(addr) == IL_LDA) {
    if (ILI_OPC(ILI_OPND(addr, 1)) == IL_ACON) {
      addr_acon_ptr = ILI_OPND(addr, 1);
      sptr = ILI_SymOPND(addr_acon_ptr, 1);
      if (CONVAL1G(sptr)) {
        sptr = SymConval1(sptr);
      }
    } else if (ILI_OPC(ILI_OPND(addr, 1)) == IL_AADD) {
      if (ILI_OPC(ILI_OPND(ILI_OPND(addr, 1), 1)) == IL_ACON) {
        addr_acon_ptr = ILI_OPND(ILI_OPND(addr, 1), 1);
        sptr = SymConval1(ILI_SymOPND(addr_acon_ptr, 1));
      }
      addr = ILI_OPND(addr, 1);
    }
    addr = ILI_OPND(addr, 1);
  }

  return sptr;
}

static SPTR
get_call_sptr(int ilix)
{
  SPTR sptr = SPTR_NULL;
  int addr;
  SPTR addr_acon_ptr;
  ILI_OP opc = ILI_OPC(ilix);

  DBGTRACEIN2(" called with ilix %d (opc=%s)", ilix, IL_NAME(opc))

  switch (opc) {
  case IL_JSR:
  case IL_GJSR:
  case IL_QJSR:
    sptr = ILI_SymOPND(ilix, 1);
    break;
  case IL_GJSRA:
  case IL_JSRA:
    addr = ILI_OPND(ilix, 1);
    if (ILI_OPC(addr) == IL_LDA) {
      sptr = find_pointer_to_function(ilix);
    } else if (ILI_OPC(addr) == IL_ACON) {
      addr_acon_ptr = ILI_SymOPND(addr, 1);
      if (!CONVAL1G(addr_acon_ptr))
        sptr = addr_acon_ptr;
      else
        sptr = SymConval1(addr_acon_ptr);
    } else if (ILI_OPC(addr) == IL_DFRAR) {
      const int addr_acon_ptr = ILI_OPND(addr, 1);
      if (ILI_OPC(addr_acon_ptr) == IL_JSR)
        /* this sptr is the called function, but the DFRAR is
         * returning a function pointer from that sptr, and that
         * returned indirect function sptr is unknown.
         */
        /* sptr = ILI_OPND(addr_acon_ptr,1); */
        sptr = SPTR_NULL;
      else if (ILI_OPC(addr_acon_ptr) == IL_JSRA)
        return get_call_sptr(addr_acon_ptr);
      else
        assert(false, "get_call_sptr(): indirect call via DFRAR not JSR/JSRA",
               ILI_OPC(addr_acon_ptr), ERR_Fatal);
    } else {
      assert(false, "get_call_sptr(): indirect call not via LDA/ACON",
             ILI_OPC(addr), ERR_Fatal);
    }
    break;
  default:
    DBGTRACE2("###get_call_sptr unknown opc %d (%s)", opc, IL_NAME(opc))
    assert(false, "get_call_sptr(): unknown opc", opc, ERR_Fatal);
    break;
  }

  DBGTRACEOUT1(" returns %d", sptr)

  return sptr;
} /* get_call_sptr */

static void
update_return_type_for_ccfunc(int ilix, ILI_OP opc)
{
  int sptr = ILI_OPND(ilix, 1);
  DTYPE dtype = DTYPEG(sptr);
  DTYPE new_dtype;
  switch (opc) {
  case IL_DFRAR:
    new_dtype = cg_get_type(3, DTY(dtype), DT_CPTR);
    break;
#ifdef IL_DFRSPX87
  case IL_DFRSPX87:
#endif
  case IL_DFRSP:
    new_dtype = cg_get_type(3, DTY(dtype), DT_FLOAT);
    break;
#ifdef IL_DFRDPX87
  case IL_DFRDPX87:
#endif
  case IL_DFRDP:
    new_dtype = cg_get_type(3, DTY(dtype), DT_DBLE);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DFRQP:
    new_dtype = cg_get_type(3, DTY(dtype), DT_QUAD);
    break;
#endif
  case IL_DFRIR:
    new_dtype = cg_get_type(3, DTY(dtype), DT_INT);
    break;
  case IL_DFRKR:
    new_dtype = cg_get_type(3, DTY(dtype), DT_INT8);
    break;
  case IL_DFRCS:
    new_dtype = cg_get_type(3, DTY(dtype), DT_CMPLX);
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
    new_dtype = cg_get_type(3, DTY(dtype), DT_FLOAT128);
    break;
#endif
  default:
    assert(false,
           "update_return_type_for_ccfunc(): return type not handled for opc ",
           opc, ERR_Fatal);
    return;
  }
  DTySetParamList(new_dtype, DTyParamList(dtype));
  DTYPEP(sptr, new_dtype);
}

/**
   \brief Create a function type from a return type and an argument list
 */
static LL_Type *
make_function_type_from_args(LL_Type *return_type, OPERAND *first_arg_op,
                             bool is_varargs)
{
  unsigned nargs = 0;
  LL_Type **types;
  OPERAND *op;
  unsigned i = 0;
  LL_Type *func_type;

  /* Count the arguments. */
  for (op = first_arg_op; op; op = op->next)
    nargs++;

  /* [0] = return type, [1..] = args. */
  types = (LL_Type **)calloc(1 + nargs, sizeof(LL_Type *));
  types[i++] = return_type;
  for (op = first_arg_op; op; op = op->next)
    types[i++] = op->ll_type;

  func_type =
      ll_create_function_type(cpu_llvm_module, types, nargs, is_varargs);
  free(types);
  return func_type;
}

/*
 * Generate a single operand for a function call.
 *
 * abi_arg is the index into abi->args. 0 is the return value, 1 is the first
 * argument, ...
 * arg_ili is the ILI instruction setting up the argument, IL_ARG*, IL_GARG, or
 * IL_DA*
 */
static OPERAND *
gen_arg_operand(LL_ABI_Info *abi, unsigned abi_arg, int arg_ili)
{
  int val_res;
  const ILI_OP arg_opc = ILI_OPC(arg_ili);
  const int value_ili = ILI_OPND(arg_ili, 1);
  LL_ABI_ArgInfo arg_info, *arg;
  LL_Type *arg_type;
  DTYPE dtype = DT_NONE;
  /* Is the ILI value argument a pointer to the value? */
  bool indirect_ili_value = false;
  bool need_load = false;
  unsigned flags = 0;
  OPERAND *operand;
  bool missing = false;

  /* Determine the dtype of the argument, or at least an approximation. Also
   * compute whether indirect_ili_value should be set. */
  switch (arg_opc) {
  case IL_GARGRET:
    assert(abi_arg == 0, "GARGRET out of place", arg_ili, ERR_Fatal);
    /* GARGRET value next-lnk dtype */
    dtype = ILI_DTyOPND(arg_ili, 3);
    /* The GARGRET value is a pointer to where the return value should be
     * stored. */
    indirect_ili_value = true;
    break;

  case IL_GARG:
    /* GARG value next-lnk dtype */
    dtype = ILI_DTyOPND(arg_ili, 3);

    /* The ili argument may be a pointer to the value to be passed. This
     * happens when passing structs by value, for example.  Assume
     * that pointers are never passed indirectly.  This also considers
     * LDSCMPLX and LDDCMPLX (complex value loads).
     */
    val_res = IL_RES(ILI_OPC(value_ili));
    if ((DTY(dtype) != TY_PTR) && ILIA_ISAR(val_res))
      indirect_ili_value = true;
    break;

  default:
    /* Without a GARG, we'll assume that any pointers (IL_ARGAR) are passed
     * by value. The indirect_ili_value stays false, and we don't support
     * passing structs by value. */
    dtype = get_dtype_from_arg_opc(arg_opc);
  }

  /* Make sure arg points to relevant lowering information, generate it if
   * required. */
  if (abi_arg <= abi->nargs) {
    /* This is one of the known arguments. */
    arg = &abi->arg[abi_arg];
  } else {
    missing = true;
    /* This is a trailing argument to a varargs function, or we don't have a
     * prototype. */
    memset(&arg_info, 0, sizeof(arg_info));
    arg = &arg_info;
    assert(dtype, "Can't infer argument dtype from ILI", arg_ili, ERR_Fatal);
    if (abi->is_fortran && !abi->is_iso_c && indirect_ili_value) {
      arg->kind = LL_ARG_INDIRECT;
      ll_abi_classify_arg_dtype(abi, arg, DT_ADDR);
      ll_abi_complete_arg_info(abi, arg, DT_ADDR);
    } else
    {
      ll_abi_classify_arg_dtype(abi, arg, dtype);
      ll_abi_complete_arg_info(abi, arg, dtype);
    }
  }

/* For fortan we want to follow the ILI as close as possible.
 * The exception is if GARG ILI field '4' is set (the arg is byval).
 * Return early in the case of fortran.
 * TODO: Allow code to pass straight through this routine and not return
 * early. Just set 'need_load' properly.
 */
  arg_type = make_lltype_from_abi_arg(arg);
  if (arg->kind != LL_ARG_BYVAL && indirect_ili_value &&
      (ILI_OPND(arg_ili, 4) || arg->kind == LL_ARG_COERCE)) {
    operand = gen_llvm_expr(value_ili, make_ptr_lltype(arg_type));
    return gen_load(operand, arg_type, ldst_instr_flags_from_dtype(dtype));
  }
    operand = gen_llvm_expr(value_ili, arg_type);
  if (arg->kind == LL_ARG_BYVAL && !missing)
    operand->flags |= OPF_SRARG_TYPE;
  if (arg->kind == LL_ARG_INDIRECT && !missing && (abi_arg == 0))
    operand->flags |= OPF_SRET_TYPE;
  return operand;

  arg_type = make_lltype_from_abi_arg(arg);

  switch (arg->kind) {
  case LL_ARG_DIRECT:
    need_load = indirect_ili_value;
    break;

  case LL_ARG_ZEROEXT:
    need_load = indirect_ili_value;
    /* flags |= OP_ZEROEXT_FLAG */
    break;

  case LL_ARG_SIGNEXT:
    need_load = indirect_ili_value;
    /* flags |= OP_SIGNEXT_FLAG */
    break;

  case LL_ARG_COERCE:
    /* It is possible to coerce with a bitcast, but we only implement
     * coercion via memory for now.
     *
     * Complex values are treated as coercion types, due to different abi
     * representations.
     *
     * Note that bitcast coercion works differently on little-endian and
     * big-endian architectures. The coercion cast always works as if the
     * value was stored with the old type and loaded with the new type.
     */
    assert(indirect_ili_value, "Can only coerce indirect args", arg_ili,
           ERR_Fatal);
    need_load = true;
    break;

  case LL_ARG_INDIRECT:
    assert(indirect_ili_value, "Indirect arg required", arg_ili, ERR_Fatal);
    /* Tag an 'sret' attribute on an indirect return value. */
    if (abi_arg == 0)
      flags |= OPF_SRET_TYPE;
    break;

  case LL_ARG_INDIRECT_BUFFERED:
    assert(indirect_ili_value, "Indirect arg required", arg_ili, ERR_Fatal);
    break;

  case LL_ARG_BYVAL:
    assert(indirect_ili_value, "Indirect arg required for byval", arg_ili,
           ERR_Fatal);
    flags |= OPF_SRARG_TYPE;
    break;

  default:
    interr("Unknown ABI argument kind", arg->kind, ERR_Fatal);
  }

  if (need_load) {
    LL_Type *ptr_type = make_ptr_lltype(arg_type);
    OPERAND *ptr = gen_llvm_expr(value_ili, ptr_type);
    operand = gen_load(ptr, arg_type, ldst_instr_flags_from_dtype(dtype));
  } else {
      operand = gen_llvm_expr(value_ili, arg_type);
  }

  if (arg->kind == LL_ARG_INDIRECT_BUFFERED) {
    /* Make a copy of the argument in the caller's scope and pass a pointer to it */
    SPTR tmp = make_arg_tmp(value_ili, dtype);
    OPERAND *new_op = make_var_op(tmp);
    OPERAND *dest_op = make_bitcast(new_op, make_lltype_from_dtype(DT_CPTR));
    OPERAND *src_op = make_bitcast(operand, make_lltype_from_dtype(DT_CPTR));
    int ts = BITS_IN_BYTE * size_of(DT_CPTR);
    insert_llvm_memcpy(0, ts, dest_op, src_op, size_of(dtype), align_of(dtype), 0);
    operand = make_var_op(tmp);
  }
  /* Set sret, byval, sign/zeroext flags. */
  operand->flags |= flags;
  return operand;
}

/* Get the next argument ILI from an IL_ARG* or IL_DA* ILI. The list will be
 * terminated by ILI_NULL. */
static int
get_next_arg(int arg_ili)
{
  switch (ILI_OPC(arg_ili)) {
  case IL_ARGAR:
  case IL_ARGDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_ARGQP:
#endif
  case IL_ARGIR:
  case IL_ARGKR:
  case IL_ARGSP:
  case IL_GARG:
  case IL_GARGRET:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ARG:
#endif
    return ILI_OPND(arg_ili, 2);

  case IL_DAAR:
  case IL_DADP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DAQP:
#endif
  case IL_DAIR:
  case IL_DAKR:
  case IL_DASP:
    return ILI_OPND(arg_ili, 3);

  default:
    interr("Unknown IL_ARG opcode", arg_ili, ERR_Fatal);
    return IL_NULL;
  }
}

/*
 * Generate linked list of operands for a call.
 *
 * The returned operand list represent the function arguments. The operand
 * representing the callee is not included.
 */
static OPERAND *
gen_arg_operand_list(LL_ABI_Info *abi, int arg_ili)
{
  unsigned abi_arg, max_abi_arg = ~0u;
  OPERAND *first_arg_op = NULL, *arg_op = NULL;

  if (LL_ABI_HAS_SRET(abi)) {
    /* ABI requires a hidden argument to return a struct. We require ILI to
     * contain a GARGRET instruction in this case. */
    /* GARGRET value next-lnk dtype */
    first_arg_op = arg_op = gen_arg_operand(abi, 0, arg_ili);
    arg_ili = get_next_arg(arg_ili);
  } else if (ILI_OPC(arg_ili) == IL_GARGRET) {
    /* It is up to gen_call_expr() to save the function return value in
     * this case. We'll just ignore the GARGRET. */
    arg_ili = get_next_arg(arg_ili);
  }

  /* If we know the exact prototype of the callee and it isn't a varargs
   * function, don't create more arguments than the function accepts.
   * Old-style C allows functions to be called with extra arguments, but LLVM
   * does not. */
  if (!abi->missing_prototype && !abi->is_varargs)
    max_abi_arg = abi->nargs;

  /* Generate operands for all the provided call arguments. */
  for (abi_arg = 1; abi_arg <= max_abi_arg && ILI_OPC(arg_ili) != IL_NULL;
       arg_ili = get_next_arg(arg_ili), abi_arg++) {
    OPERAND *op = gen_arg_operand(abi, abi_arg, arg_ili);
    if (arg_op == NULL)
      first_arg_op = op;
    else
      arg_op->next = op;
    arg_op = op;
  }

  return first_arg_op;
}

/**
   \brief Generate LLVM instructions for a call.
   \param ilix      is the IL_JSR or IL_JSRA instruction representing the call
   \param ret_dtype is the dtype for the return value, or 0 if return unused
   \param call_instr is either a newly allocated I_CALL or NULL; if NULL, a new
   instruction will be allocated
   \param call_sptr is the sptr of the called function or function pointer, if
   known
   \returns an OPERAND representing the value returned by the call
*/
static OPERAND *
gen_call_expr(int ilix, DTYPE ret_dtype, INSTR_LIST *call_instr, int call_sptr)
{
  int first_arg_ili;
  LL_ABI_Info *abi;
  LL_Type *return_type;
  OPERAND *first_arg_op;
  OPERAND *callee_op = NULL;
  LL_Type *func_type = NULL;
  OPERAND *result_op = NULL;
  bool intrinsic_modified = false;
  int throw_label = ili_throw_label(ilix);

  if (call_instr == NULL)
    call_instr = make_instr((throw_label > 0) ? I_INVOKE : I_CALL);

  /* Prefer IL_GJSR / IL_GJSRA when available. */
  if (ILI_ALT(ilix))
    ilix = ILI_ALT(ilix);

  /* GJSR sym args, or GJSRA addr args flags. */
  first_arg_ili = ILI_OPND(ilix, 2);

  /* Get an ABI descriptor which at least knows about the function return type.
     We may have more arguments than the descriptor knows about if this is a
     varargs call, or if the prototype is missing. */
  abi = ll_abi_from_call_site(cpu_llvm_module, ilix, ret_dtype);

  first_arg_op = gen_arg_operand_list(abi, first_arg_ili);

  /* Set the calling convention, read by write_I_CALL. */
  call_instr->flags |= (LL_InstrListFlags)(abi->call_conv << CALLCONV_SHIFT);

  if (abi->fast_math)
    call_instr->flags |= FAST_MATH_FLAG;

  /* Functions without a prototype are represented in LLVM IR as f(...) varargs
     functions.  Do what clang does and bitcast to a function pointer which is
     varargs, but with all the actual argument types filled in. */
  if (abi->missing_prototype) {
#if defined(TARGET_LLVM_X8664)
    /* Fortran argument lists of dtype currently not precsise. So when
     * we make 256/512-bit math intrinsic calls, which are not really covered
     * by the ABI, LLVM can get confused with stack alignment. This
     * check is just a temporary workaround (which is to not generate
     * the bitcast into a varargs, but just use the known argument.)
     */
    const DTYPE ddsize = DTYPEG(call_sptr);
    int dsize = (ddsize == DT_NONE) ? 0 : zsize_of(ddsize);
    if ((dsize == 32 || dsize == 64) &&
        is_256_or_512_bit_math_intrinsic(call_sptr) && !XBIT(183, 0x4000))
      func_type = make_function_type_from_args(ll_abi_return_type(abi),
                                               first_arg_op, 0);
    else
#endif
      func_type = make_function_type_from_args(
          ll_abi_return_type(abi), first_arg_op, abi->call_as_varargs);
  }

  /* Now figure out the callee itself. */
  switch (ILI_OPC(ilix)) {
  case IL_JSR:
  case IL_GJSR:
  case IL_QJSR: {
    /* Direct call: JSR sym arg-lnk */
    SPTR callee_sptr = ILI_SymOPND(ilix, 1);
    callee_op = make_var_op(callee_sptr);
    /* Create an alternative function type from arguments for a function that
       is defined in the current source file and does not have an alternative
       function type defined yet.  Don't perform this conversion for varargs
       functions, because for those arguments the invocation doesn't match the
       definition. */
    if (!func_type && !abi->is_varargs) {
      func_type = make_function_type_from_args(
          ll_abi_return_type(abi), first_arg_op, abi->call_as_varargs);
      if(intrinsic_modified) {
        /* For llvm intrinsics whose arg op has been modified, correct
           callee_op to match and to prevent function bitcast.*/
        callee_op->ll_type = make_ptr_lltype(func_type);
      }
    }
    /* Cast function points that are missing a prototype. */
    if (func_type && func_type != callee_op->ll_type->sub_types[0]) {
      callee_op = make_bitcast(callee_op, make_ptr_lltype(func_type));
    }
    break;
  }
  case IL_JSRA:
  case IL_GJSRA: {
    /* Indirect call: JSRA addr arg-lnk flags */
    int addr_ili = ILI_OPND(ilix, 1);
    if (!func_type) {
      func_type = abi->is_varargs ? ll_abi_function_type(abi) :
        make_function_type_from_args(
            ll_abi_return_type(abi), first_arg_op, abi->call_as_varargs);
    }
    /* Now that we know the desired type we can create the callee address
       expression. */
    callee_op = gen_llvm_expr(addr_ili, make_ptr_lltype(func_type));
    call_instr->flags |= CALL_FUNC_PTR_FLAG;
    break;
  }
  default:
    interr("Unhandled call instruction", ilix, ERR_Fatal);
    break;
  }

  callee_op->next = first_arg_op;
  call_instr->operands = callee_op;

  if (throw_label == -1) {
    /* The function might throw, but the exception should just propagate out to
       the calling function.  Nothing to do. */
  } else if (throw_label == 0) {
    if (callee_op->string &&
        (!strcmp("@__cxa_call_unexpected", callee_op->string))) {
      /* Ignore __cxa_call_unexpected as nounwind, due to bugs in PowerPC
         backend. */
    } else {
      /* The called function should never throw. */
      call_instr->flags |= NOUNWIND_CALL_FLAG;
    }
  } else {
    /* The function might throw, and if it does, control should jump to the
       given label. The normal return label and the exception label are added
       to the end of the operand list. */
    OPERAND *label_op;
    OPERAND *op = call_instr->operands;
    while (op->next) {
      op = op->next;
    }
    label_op = make_label_op(getlab());
    op->next = label_op;
    op = label_op;
    label_op = make_operand();
    label_op->ot_type = OT_LABEL;
    label_op->val.cc = throw_label;
    op->next = label_op;
  }

  return_type = ll_abi_return_type(abi);
  if (return_type->data_type == LL_VOID) {
    call_instr->ll_type = make_void_lltype();
    /* This function may return a struct via a hidden argument.  See if ILI is
       expecting the function to return the hidden argument pointer, like the
       x86-64 ABI requires.  LLVM handles this in the code generator, so we
       have produced a void return type for the LLVM IR.  Just return the
       hidden argument directly for the ret_dtype to consume. */
    if (LL_ABI_HAS_SRET(abi) &&
        (DTY(ret_dtype) == TY_STRUCT || DTY(ret_dtype) == TY_UNION)) {
      result_op = gen_copy_op(first_arg_op);
      result_op->flags = 0;
    }
  } else {
    /* When we stop wrapping alt_type, this can simply be return_type. */
    call_instr->ll_type = make_lltype_from_abi_arg(&abi->arg[0]);
    call_instr->tmps = make_tmps();

    /* If ret_dtype is not set, no return value is expected. */
    if (ret_dtype) {
      result_op = make_tmp_op(call_instr->ll_type, call_instr->tmps);
    }
  }

  ad_instr(ilix, call_instr);

  /* Check if the expander created a GARGRET call but the ABI returns in
     registers.  In that case, coerce the returned value by storing it to the
     GARGRET value pointer. */
  if (ILI_OPC(first_arg_ili) == IL_GARGRET && !LL_ABI_HAS_SRET(abi)) {
    int addr_ili = ILI_OPND(first_arg_ili, 1);
    DTYPE return_dtype = ILI_DTyOPND(first_arg_ili, 3);
    OPERAND *addr;
    assert(ILIA_ISAR(IL_RES(ILI_OPC(addr_ili))),
           "GARGRET must be indirect value", ilix, ERR_Fatal);
    addr = gen_llvm_expr(addr_ili, make_ptr_lltype(call_instr->ll_type));
    make_store(make_tmp_op(call_instr->ll_type, call_instr->tmps), addr,
               ldst_instr_flags_from_dtype(return_dtype));

    /* Does ILI expect the function call to return the hidden argument, like an
       x86-64 sret function call? */
    if (DTY(ret_dtype) == TY_STRUCT || DTY(ret_dtype) == TY_UNION) {
      result_op = gen_copy_op(addr);
    }
  }

  return result_op;
} /* gen_call_expr */

#if defined(TARGET_LLVM_X8664)
static bool
is_256_or_512_bit_math_intrinsic(int sptr)
{
  int new_num, new_type;
  const char *sptrName;
  bool is_g_name = false; /* the first cut at generic names */
  bool is_newest_name = false;

  if (sptr == 0 || !CCSYMG(sptr))
    return false;
  sptrName = SYMNAME(sptr);
  if (sptrName == NULL)
    return false;

  /* test for generic name that matches "__gv<s|d|c|z>_<math-func>_<2|4|8>" */
  if (!strncmp(sptrName, "__gv", 4)) {
    is_g_name = true;
    new_type = sptrName[4];
  }

  /* test for newest name that matches
   * "__<frp><s|d|c|z>_<math-func>_<2|4|8|16>
   */
  else if (!strncmp(sptrName, "__f", 4) || !strncmp(sptrName, "__p", 4) ||
           !strncmp(sptrName, "__r", 4)) {
    new_type = sptrName[3];
    switch (new_type) {
    case 's':
    case 'd':
    case 'c':
    case 'z':
      is_newest_name = true;
      break;
    default:
      break;
    }
  }

  /* names match: generic name or  "__<f|r>v<s|d>_<math-func>_<vex|fma4>_256" */
  if (is_newest_name || is_g_name || !strncmp(sptrName, "__fv", 4) ||
      !strncmp(sptrName, "__rv", 4))
    sptrName += 4;
  else
    return false;

  if (is_newest_name)
    sptrName++;
  else if ((*sptrName) &&
           ((*sptrName == 's') || (*sptrName == 'd') || (*sptrName == 'c') ||
            (*sptrName == 'z')) &&
           (sptrName[1] == '_'))
    sptrName += 2;
  else
    return false;

  if (!(*sptrName))
    return false;
  if ((!strncmp(sptrName, "sin", 3)) || (!strncmp(sptrName, "cos", 3)) ||
      (!strncmp(sptrName, "tan", 3)) || (!strncmp(sptrName, "pow", 3)) ||
      (!strncmp(sptrName, "log", 3)) || (!strncmp(sptrName, "exp", 3)) ||
      (!strncmp(sptrName, "mod", 3)) || (!strncmp(sptrName, "div", 3)))
    sptrName += 3;
  else if ((!strncmp(sptrName, "sinh", 4)) || (!strncmp(sptrName, "cosh", 4)))
    sptrName += 4;
  else if (!strncmp(sptrName, "log10", 5))
    sptrName += 5;
  else if (!strncmp(sptrName, "sincos", 6))
    sptrName += 6;
  else
    return false;

  if (is_newest_name) {
    sptrName++;
    new_num = atoi(sptrName);
    switch (new_type) {
    case 's':
      if (new_num == 8 || new_num == 16)
        return true;
      break;
    case 'd':
      if (new_num == 4 || new_num == 8)
        return true;
      break;
    case 'c':
      if (new_num == 4 || new_num == 8)
        return true;
      break;
    case 'z':
      if (new_num == 2 || new_num == 4)
        return true;
      break;
    default:
      return false;
    }
    return false;
  } else if (is_g_name) {
    new_num = atoi(sptrName);
    if (isdigit(sptrName[0]))
      switch (new_type) {
      case 's':
        if (new_num == 8)
          return true;
        break;
      case 'd':
        if (new_num == 4)
          return true;
        break;
      default:
        return false;
      }
    return false;
  } else if (*sptrName == '_') {
    sptrName++;
  } else {
    return false;
  }

  if (!(*sptrName))
    return false;
  if (!strncmp(sptrName, "vex_", 4))
    sptrName += 4;
  else if (!strncmp(sptrName, "fma4_", 5))
    sptrName += 5;
  else
    return false;

  return (!strcmp(sptrName, "256")); /* strcmp: check for trailing garbage */
}
#endif

#ifdef FLANG2_CGMAIN_UNUSED
static bool
have_masked_intrinsic(int ilix)
{
  ILI_OP vopc;
  DTYPE vdtype;
  int mask;

  vopc = ILI_OPC(ilix);
  vdtype = ili_get_vect_dtype(ilix);
  if (!vdtype)
    return false;

  switch (vopc) {
  case IL_VDIV:
  case IL_VMOD:
  case IL_VSQRT:
  case IL_VSIN:
  case IL_VCOS:
  case IL_VTAN:
  case IL_VSINCOS:
  case IL_VASIN:
  case IL_VACOS:
  case IL_VATAN:
  case IL_VATAN2:
  case IL_VSINH:
  case IL_VCOSH:
  case IL_VTANH:
  case IL_VEXP:
  case IL_VLOG:
  case IL_VLOG10:
  case IL_VPOW:
  case IL_VPOWI:
  case IL_VPOWK:
  case IL_VPOWIS:
  case IL_VPOWKS:
  case IL_VFPOWK:
  case IL_VFPOWKS:
  case IL_VDPOWI:
  case IL_VDPOWIS:
  case IL_VRSQRT:
  case IL_VRCP:
  /* case IL_VFLOOR:*/
  /* case IL_VCEIL: */
  /* case IL_VAINT: */
    mask = ILI_OPND(ilix, IL_OPRS(vopc) - 1); /* get potential mask */
    if (ILI_OPC(mask) != IL_NULL) {
      /* have mask */
      return true;
    }
    break;
  default:
    break;
  }
  return false;
}
#endif

/* LLVM extractvalue instruction:
 * Given an aggregate and index return the value at that index.
 *
 * <result> = extractvalue <aggregate type> <val>, <idx>{, <idx>}*
 */
static OPERAND *
gen_extract_value_ll(OPERAND *aggr, LL_Type *aggr_ty, LL_Type *elt_ty, int idx)
{
  OPERAND *res = make_tmp_op(elt_ty, make_tmps());
  INSTR_LIST *Curr_Instr = gen_instr(I_EXTRACTVAL, res->tmps, aggr_ty, aggr);
  aggr->next = make_constval32_op(idx);
  ad_instr(0, Curr_Instr);
  return res;
}

/* Like gen_extract_value_ll, but takes DTYPE instead of LL_TYPE argumentss. */
static OPERAND *
gen_extract_value(OPERAND *aggr, DTYPE aggr_dtype, DTYPE elt_dtype, int idx)
{
  LL_Type *aggr_ty = make_lltype_from_dtype(aggr_dtype);
  LL_Type *elt_ty = make_lltype_from_dtype(elt_dtype);
  return gen_extract_value_ll(aggr, aggr_ty, elt_ty, idx);
}

static OPERAND *
gen_eval_cmplx_value(int ilix, DTYPE dtype)
{
  OPERAND *c1;
  LL_Type *cmplx_type = make_lltype_from_dtype(dtype);

  c1 = gen_llvm_expr(ilix, cmplx_type);

  /* should move this to a %temp? */

  return c1;
}

static OPERAND *
gen_copy_operand(OPERAND *opnd)
{
  OPERAND *curr;
  OPERAND *head;

  /* copy operand opnd -> c1 */
  head = gen_copy_op(opnd);
  curr = head;
  while (opnd->next) {
    curr->next = gen_copy_op(opnd->next);
    opnd = opnd->next;
    curr = curr->next;
  }

  return head;
}

/* Math operations for complex values.
 * 'itype' should be the I_FADD, I_FSUB, I_xxxx etc.
 * 'dtype' should either be DT_CMPLX or DT_DCMPLX.
 */
static OPERAND *
gen_cmplx_math(int ilix, DTYPE dtype, LL_InstrName itype)
{
  OPERAND *r1, *r2, *i1, *i2, *rmath, *imath, *res, *c1, *c2, *cse1, *cse2;
  LL_Type *cmplx_type, *cmpnt_type;
  const DTYPE cmpnt = (dtype == DT_CMPLX)  ? DT_FLOAT
#ifdef TARGET_SUPPORTS_QUADFP
                    : (dtype == DT_QCMPLX) ? DT_QUAD
#endif
                                           : DT_DBLE;

  assert(DT_ISCMPLX(dtype), "gen_cmplx_math: Expected DT_CMPLX, DT_DCMPLX or DT_QCMPLX",
         dtype, ERR_Fatal);

  cmplx_type = make_lltype_from_dtype(dtype);
  cmpnt_type = make_lltype_from_dtype(cmpnt);

  /* Obtain the components (real and imaginary) for both operands */

  c1 = gen_eval_cmplx_value(ILI_OPND(ilix, 1), dtype);
  c2 = gen_eval_cmplx_value(ILI_OPND(ilix, 2), dtype);
  cse1 = gen_copy_operand(c1);
  cse2 = gen_copy_operand(c2);

  r1 = gen_extract_value(c1, dtype, cmpnt, 0);
  i1 = gen_extract_value(cse1, dtype, cmpnt, 1);

  r2 = gen_extract_value(c2, dtype, cmpnt, 0);
  i2 = gen_extract_value(cse2, dtype, cmpnt, 1);

  r1->next = r2;
  i1->next = i2;

  rmath = ad_csed_instr(itype, 0, cmpnt_type, r1, InstrListFlagsNull, true);
  imath = ad_csed_instr(itype, 0, cmpnt_type, i1, InstrListFlagsNull, true);

  /* Build a temp complex in registers and store the mathed values in that */
  res = make_undef_op(cmplx_type);
  res = gen_insert_value(res, rmath, 0);
  return gen_insert_value(res, imath, 1);
}

/* Complex multiply:
 * (a + bi) * (c + di) ==  (a*c) + (a*di) + (bi*c) + (bi*di)
 */
static OPERAND *
gen_cmplx_mul(int ilix, DTYPE dtype)
{
  const DTYPE elt_dt = (dtype == DT_CMPLX)  ? DT_FLOAT
#ifdef TARGET_SUPPORTS_QUADFP
                     : (dtype == DT_QCMPLX) ? DT_QUAD
#endif
                                            : DT_DBLE;
  LL_Type *cmpnt_type = make_lltype_from_dtype(elt_dt);
  OPERAND *a, *bi, *c, *di, *cse1, *cse2;
  OPERAND *r1, *r2, *r3, *r4, *imag, *real, *res, *c1, *c2;

  c1 = gen_eval_cmplx_value(ILI_OPND(ilix, 1), dtype);
  c2 = gen_eval_cmplx_value(ILI_OPND(ilix, 2), dtype);
  cse1 = gen_copy_operand(c1);
  cse2 = gen_copy_operand(c2);

  a = gen_extract_value(c1, dtype, elt_dt, 0);
  bi = gen_extract_value(cse1, dtype, elt_dt, 1);
  c = gen_extract_value(c2, dtype, elt_dt, 0);
  di = gen_extract_value(cse2, dtype, elt_dt, 1);

  /* r1 = (a * c) */
  a->next = c;
  r1 = ad_csed_instr(I_FMUL, 0, cmpnt_type, a, InstrListFlagsNull, true);

  /* r2 = (a * di) */
  cse1 = gen_copy_operand(c1);
  a = gen_extract_value(cse1, dtype, elt_dt, 0);
  a->next = di;
  r2 = ad_csed_instr(I_FMUL, 0, cmpnt_type, a, InstrListFlagsNull, true);

  /* r3 = (bi * c) */
  bi->next = c;
  r3 = ad_csed_instr(I_FMUL, 0, cmpnt_type, bi, InstrListFlagsNull, true);

  /* r4 = (bi * di) */
  cse1 = gen_copy_operand(c1);
  bi = gen_extract_value(cse1, dtype, elt_dt, 1);
  bi->next = di;
  r4 = ad_csed_instr(I_FMUL, 0, cmpnt_type, bi, InstrListFlagsNull, true);

  /* Real: r1 - r4 */
  r1->next = r4;
  real = ad_csed_instr(I_FSUB, 0, cmpnt_type, r1, InstrListFlagsNull, true);

  /* Imag: r2 + r3 */
  r2->next = r3;
  imag = ad_csed_instr(I_FADD, 0, cmpnt_type, r2, InstrListFlagsNull, true);

  res = make_undef_op(make_lltype_from_dtype(dtype));
  res = gen_insert_value(res, real, 0);
  return gen_insert_value(res, imag, 1);
}

static OPERAND *
gen_llvm_atomicrmw_expr(int ilix)
{
  OPERAND *result;
  ATOMIC_INFO info = atomic_info(ilix);
  LL_Type *instr_type = make_type_from_msz((MSZ)info.msz);
  /* LLVM instruction atomicrmw has operands in opposite order of ILI
   * instruction. */
  OPERAND *op1 = gen_llvm_expr(ILI_OPND(ilix, 2), make_ptr_lltype(instr_type));
  OPERAND *op2 = gen_llvm_expr(ILI_OPND(ilix, 1), instr_type);
  LL_InstrListFlags flags;
  op1->next = op2;
  flags = ll_instr_flags_for_memory_order_and_scope(ilix);
  if (ILI_OPND(ilix, 3) == NME_VOL)
    flags |= VOLATILE_FLAG;
  flags |= ll_instr_flags_from_aop((ATOMIC_RMW_OP)info.op);
  /* Caller will deal with doing zero-extend/sign-extend if necessary. */
  result = ad_csed_instr(I_ATOMICRMW, ilix, instr_type, op1, flags, false);
  return result;
}

static void
gen_llvm_fence_instruction(int ilix)
{
  LL_InstrListFlags flags = ll_instr_flags_for_memory_order_and_scope(ilix);
  INSTR_LIST *fence = gen_instr(I_FENCE, NULL, NULL, NULL);
  fence->flags |= flags;
  ad_instr(0, fence);
}

INLINE static OPERAND *
gen_llvm_cmpxchg(int ilix)
{
  LL_Type *aggr_type;
  LL_InstrListFlags flags;
  OPERAND *op1, *op2, *op3;
  LL_Type *elements[2];
  CMPXCHG_MEMORY_ORDER order;

  /* Construct aggregate type for result of cmpxchg. */
  MSZ msz = atomic_info(ilix).msz;
  LL_Module *module = cpu_llvm_module;
  elements[0] = make_type_from_msz(msz);
  elements[1] = ll_create_basic_type(module, LL_I1, 0);
  aggr_type = ll_create_anon_struct_type(module, elements, 2,
                                         /*is_packed=*/false, LL_AddrSp_Default);

  /* address of location */
  op1 = gen_llvm_expr(cmpxchg_loc(ilix), make_ptr_lltype(elements[0]));
  /* comparand */
  op2 = gen_llvm_expr(ILI_OPND(ilix, 5), elements[0]);
  /* new value */
  op3 = gen_llvm_expr(ILI_OPND(ilix, 1), elements[0]);
  op1->next = op2;
  op2->next = op3;

  /* Construct flags for memory order, volatile, and weak. */
  order = cmpxchg_memory_order(ilix);
  flags = ll_instr_flags_for_memory_order_and_scope(ilix);
  flags |=
      TO_CMPXCHG_MEMORDER_FAIL(ll_instr_flags_from_memory_order(order.failure));
  if (ILI_OPND(ilix, 3) == NME_VOL)
    flags |= VOLATILE_FLAG;
  if (cmpxchg_is_weak(ilix))
    flags |= CMPXCHG_WEAK_FLAG;
  return ad_csed_instr(I_CMPXCHG, ilix, aggr_type, op1, flags, false);
}

static OPERAND *
gen_llvm_cmpxchg_component(int ilix, int idx)
{
  OPERAND *result, *ll_cmpxchg;
  int ilix_cmpxchg = ILI_OPND(ilix, 1);

  DEBUG_ASSERT((unsigned)idx < 2u, "gen_llvm_cmpxchg_component: bad index");
  /* Generate the cmpxchg */
  ll_cmpxchg = gen_llvm_expr(ilix_cmpxchg, NULL);
  ll_cmpxchg->next = make_constval32_op(idx);
  result = gen_extract_value_ll(ll_cmpxchg, ll_cmpxchg->ll_type,
                                ll_cmpxchg->ll_type->sub_types[idx], idx);
  return result;
}

static void
add_sincos_map(INSTR_LIST *insn, unsigned flag)
{
  hash_data_t data = NULL;
  if (!sincos_map)
    sincos_map = hashmap_alloc(hash_functions_direct);
  if (hashmap_lookup(sincos_map, insn, &data) && (HKEY2INT(data) & flag))
    return;
  data = INT2HKEY(HKEY2INT(data) | flag);
  hashmap_replace(sincos_map, insn, &data);
}

/**
   \brief Generate the \c extractvalue for the \c sin or \c cos part
 */
INLINE static OPERAND *
gen_llvm_select_sin_or_cos(OPERAND *op, LL_Type *argTy, LL_Type *retTy,
                           int component)
{
  OPERAND *rv;
  add_sincos_map(op->tmps->info.idef, (component ? SINCOS_COS : SINCOS_SIN));
  rv = gen_extract_value_ll(op, retTy, argTy, component);
  add_sincos_map(rv->tmps->info.idef, SINCOS_EXTRACT);
  return rv;
}

INLINE static OPERAND *
gen_llvm_select_vsincos(OPERAND *op, LL_Type *argTy, LL_Type *retTy,
                        int component)
{
  OPERAND *rv;
  add_sincos_map(op->tmps->info.idef, (component ? SINCOS_COS : SINCOS_SIN));
  rv = gen_extract_value_ll(op, retTy, argTy, component);
  add_sincos_map(rv->tmps->info.idef, SINCOS_EXTRACT);
  return rv;
}

static void
add_sincos_imap_loads(int ilix)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;
  const ILTY_KIND ilty = IL_TYPE(opc);
  if (ilty == ILTY_LOAD) {
    hash_data_t data = NULL;
    hashmap_replace(sincos_imap, INT2HKEY(ilix), &data);
  }
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i))
      add_sincos_imap_loads(ILI_OPND(ilix, i));
  }
}

INLINE static void
add_sincos_imap(int ilix, hash_data_t data)
{
  hashmap_replace(sincos_imap, INT2HKEY(ilix), &data);
}

/**
   \brief Generate a CALL to \c sincos, the scalar version
 */
INLINE static OPERAND *
gen_llvm_sincos_call(int ilix)
{
  OPERAND *rv;
  const ILI_OP opc = ILI_OPC(ilix);
  const bool isSingle = (opc == IL_FSINCOS);
  LL_Type *llTy = make_lltype_from_dtype(isSingle ? DT_FLOAT : DT_DBLE);
  OPERAND *arg = gen_llvm_expr(ILI_OPND(ilix, 1), llTy);
  LL_Type *retTy = make_lltype_from_dtype(isSingle ? DT_CMPLX : DT_DCMPLX);
  char sincosName[36]; /* make_math_name buffer is 32 */
  llmk_math_name(sincosName, MTH_sincos, 1, false,
                 isSingle ? DT_FLOAT : DT_DBLE);
  if (!sincos_imap)
    sincos_imap = hashmap_alloc(hash_functions_direct);
  add_sincos_imap_loads(ILI_OPND(ilix, 1));
  rv = gen_call_to_builtin(ilix, sincosName, arg, retTy, NULL, I_CALL,
                           InstrListFlagsNull, EXF_PURE);
  add_sincos_imap(ilix, rv);
  return rv;
}

INLINE static LL_Type *
gen_vsincos_return_type(LL_Type *vecTy)
{
  LL_Type *elements[2] = {vecTy, vecTy};
  return ll_create_anon_struct_type(cpu_llvm_module, elements, 2, false, LL_AddrSp_Default);
}

INLINE static OPERAND *
gen_llvm_vsincos_call(int ilix)
{
  const DTYPE dtype = ili_get_vect_dtype(ilix);
  LL_Type *floatTy = make_lltype_from_dtype(DT_FLOAT);
  LL_Type *vecTy = make_lltype_from_dtype(dtype);
  DTYPE mask_dtype;
  LL_Type *maskTy;
  DTYPE dtypeName = (vecTy->sub_types[0] == floatTy) ? DT_FLOAT : DT_DBLE;
  LL_Type *retTy = gen_vsincos_return_type(vecTy);
  OPERAND *opnd = gen_llvm_expr(ILI_OPND(ilix, 1), vecTy);
  char sincosName[36]; /* make_math_name buffer is 32 */
  int vecLen = vecTy->sub_elements;
  int opndCount = ili_get_vect_arg_count(ilix);
  int mask_arg_ili = ILI_OPND(ilix, opndCount - 1);
  bool hasMask = false;

  /* Mask operand is always the one before the last operand */
  if (ILI_OPC(mask_arg_ili) != IL_NULL) {
     /* mask is always a vector of integers; same number and size as 
      * the regular argument.
      */
     mask_dtype = get_vector_dtype(dtypeName==DT_FLOAT?DT_INT:DT_INT8,vecLen);
     maskTy = make_lltype_from_dtype(mask_dtype);
      opnd->next = gen_llvm_expr(mask_arg_ili, maskTy);
    hasMask = true;
  }
  llmk_math_name(sincosName, MTH_sincos, vecLen, hasMask, dtypeName);
  if (!sincos_imap)
    sincos_imap = hashmap_alloc(hash_functions_direct);
  add_sincos_imap_loads(ILI_OPND(ilix, 1));
  opnd = gen_call_to_builtin(ilix, sincosName, opnd, retTy, NULL, I_CALL,
                             InstrListFlagsNull, EXF_PURE);
  add_sincos_imap(ilix, opnd);
  return opnd;
}

static bool
sincos_argument_valid(int ilix)
{
  int i;
  const ILI_OP opc = ILI_OPC(ilix);
  const int noprs = ilis[opc].oprs;
  const ILTY_KIND ilty = IL_TYPE(opc);
  if (ilty == ILTY_LOAD) {
    if (!hashmap_lookup(sincos_imap, INT2HKEY(ilix), NULL))
      return false;
  }
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i) && (!sincos_argument_valid(ILI_OPND(ilix, i))))
      return false;
  }
  return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
INLINE static OPERAND *
get_last_sincos(int ilix)
{
  hash_data_t data = NULL;
  if (sincos_imap && hashmap_lookup(sincos_imap, INT2HKEY(ilix), &data) &&
      sincos_argument_valid(ILI_OPND(ilix, 1)))
    return (OPERAND *)data;
  return NULL;
}
#pragma GCC diagnostic pop

INLINE static OPERAND *
gen_llvm_sincos_builtin(int ilix)
{
  OPERAND *sincos = get_last_sincos(ilix);
  return sincos ? sincos : gen_llvm_sincos_call(ilix);
}

INLINE static OPERAND *
gen_llvm_vsincos_builtin(int ilix)
{
  OPERAND *vsincos = get_last_sincos(ilix);
  return vsincos ? vsincos : gen_llvm_vsincos_call(ilix);
}

INLINE static bool
is_complex_result(int ilix)
{
  if (on_prescan_complex_list(ILI_OPND(ilix, 1))) {
    ILI_OP opc = ILI_OPC(ilix);
    switch (opc) {
#ifdef LONG_DOUBLE_FLOAT128
    case IL_FLOAT128RESULT:
      return true;
#endif
    default:
      break;
    }
  }
  return false;
}

INLINE static LL_Type *
complex_result_type(ILI_OP opc)
{
  switch (opc) {
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
    return make_lltype_from_dtype(DT_CMPLX128);
#endif
  default:
    return NULL;
  }
}

INLINE static OPERAND *
gen_comp_operand(OPERAND *operand, ILI_OP opc, int lhs_ili, int rhs_ili,
                 int cc_ili, int cc_type, LL_InstrName itype)
{
  return gen_optext_comp_operand(operand, opc, lhs_ili, rhs_ili, cc_ili,
                                 cc_type, itype, 1, 0);
}

OPERAND *
gen_llvm_expr(int ilix, LL_Type *expected_type)
{
  int nme_ili, ld_ili;
  SPTR sptr;
  MSZ msz;
  int lhs_ili, rhs_ili;
  CC_RELATION ili_cc;
  int zero_ili = 0;
  int first_ili, second_ili;
  int ct;
  DTYPE dt, cmpnt, dtype;
  ILI_OP opc = IL_NONE;
  bool cse_opc = false;
  DTYPE call_dtype = DT_NONE;
  SPTR call_sptr = SPTR_NULL;
  MATCH_Kind ret_match;
  LL_Type *comp_exp_type = (LL_Type *)0;
  LL_Type *intrinsic_type;
  OPERAND *operand, *args, *call_op;
  OPERAND *cc_op1, *cc_op2, *c1, *cse1;
  INT tmp[2];
  const char *intrinsic_name;
  float f;
  union {
    double d;
    INT tmp[2];
  } dtmp;
  double d;
#ifdef TARGET_SUPPORTS_QUADFP
  union {
    long double q;
    INT tmp[QTMP_SIZE];
  } qtmp;
  long double q;
#endif

  switch (ILI_OPC(ilix)) {
  case IL_JSR:
  case IL_JSRA:
    /*  ILI_ALT may be IL_GJSR/IL_GJSRA */
    break;
  case IL_VFLOOR:
  case IL_VCEIL:
  case IL_VAINT:
  case IL_FFLOOR:
  case IL_DFLOOR:
  case IL_FCEIL:
  case IL_DCEIL:
  case IL_AINT:
  case IL_DINT:
    /* floor/ceil/aint use llvm intrinsics, not calls via alt-ili */
    break;
  case IL_VSIN:
  case IL_VCOS:
    if (ILI_OPC(ILI_OPND(ilix, 1)) == IL_VSINCOS)
      break;
    FLANG_FALLTHROUGH;
  default:
    if (ILI_ALT(ilix)) {
      ilix = ILI_ALT(ilix);
    }
    break;
  }
  opc = ILI_OPC(ilix);

  DBGTRACEIN2(" ilix: %d(%s)", ilix, IL_NAME(opc));
  DBGDUMPLLTYPE("#expected type: ", expected_type);

  assert(ilix, "gen_llvm_expr(): no incoming ili", 0, ERR_Fatal);
  operand = make_operand();

  switch (opc) {
  case IL_JSRA:
  case IL_GJSRA:
    call_dtype = ILI_DTyOPND(ilix, 4);
    if (call_dtype) {
      /* iface symbol table value */
      call_dtype = (DTYPE)DTY(DTYPEG(call_dtype)); // FIXME: bug?
    } else {
      call_dtype = llvm_info.curr_ret_dtype;
    }
    goto call_processing;
  case IL_QJSR:
  case IL_JSR:
  case IL_GJSR:
    sptr = ILI_SymOPND(ilix, 1);
    call_op = gen_call_as_llvm_instr(sptr, ilix);
    if (call_op) {
      operand = call_op;
      break;
    }

/* check for return dtype */
    call_dtype = (DTYPE)DTY(DTYPEG(sptr)); // FIXME: is this a bug?

    DBGTRACE1("#CALL to %s", SYMNAME(sptr))

    call_sptr = sptr;

  call_processing:
    call_op = gen_call_expr(ilix, call_dtype, NULL, call_sptr);
    if (call_op) {
      operand = call_op;
      if (DT_ISUNSIGNED(call_dtype))
        operand->flags |= OPF_ZEXT;
      else if (DT_ISINT(call_dtype))
        operand->flags |= OPF_SEXT;
    } else {
      operand->ll_type = make_void_lltype();
    }
    break;
  case IL_EXIT:
    operand = gen_return_operand(ilix);
    break;
  case IL_VA_ARG:
    operand = gen_va_arg(ilix);
    break;
  case IL_ACON:
    operand = gen_base_addr_operand(ilix, expected_type);
    break;
  case IL_LDA:
    nme_ili = ILI_OPND(ilix, 2);
    ld_ili = ILI_OPND(ilix, 1);
    if (ILI_OPC(ld_ili) != IL_ACON && expected_type &&
        (expected_type->data_type == LL_PTR)) {
      LL_Type *pt_expected_type = make_ptr_lltype(expected_type);
      operand = gen_base_addr_operand(ld_ili, pt_expected_type);
    } else {
      operand = gen_address_operand(ld_ili, nme_ili, true, NULL,
                                    MSZ_ILI_OPND(ilix, 3));
    }
    sptr = basesym_of(nme_ili);
    if ((operand->ll_type->data_type == LL_PTR) ||
        (operand->ll_type->data_type == LL_ARRAY)) {
      DTYPE dtype = DTYPEG(sptr);

      /* If no type found assume generic pointer */
      if (dtype == DT_NONE)
        dtype = DT_CPTR;

      if (operand->ll_type->sub_types[0]->data_type == LL_PTR ||
          ILI_OPC(ld_ili) != IL_ACON) {
        operand =
            make_load(ilix, operand, operand->ll_type->sub_types[0], (MSZ)-2,
                      ldst_instr_flags_from_dtype_nme(dtype, nme_ili));
      } else {
        if ((SCG(sptr) != SC_DUMMY))
        {
          LL_Type *llt = make_ptr_lltype(expected_type);
          operand = make_bitcast(operand, llt);
          /* ??? what is the magic constant -2? */
          operand =
              make_load(ilix, operand, operand->ll_type->sub_types[0], (MSZ)-2,
                        ldst_instr_flags_from_dtype_nme(dtype, nme_ili));
        }
      }
    }
    break;
  case IL_VLD: {
    LL_Type *llt, *vect_lltype, *int_llt = NULL;
    DTYPE vect_dtype = ILI_DTyOPND(ilix, 3);

    nme_ili = ILI_OPND(ilix, 2);
    ld_ili = ILI_OPND(ilix, 1);
    vect_lltype = make_lltype_from_dtype(vect_dtype);
    llt = make_ptr_lltype(vect_lltype);
    if (expected_type && (expected_type->data_type == LL_VECTOR) &&
        (expected_type->sub_elements == 4 ||
         expected_type->sub_elements == 3) &&
        (llt->sub_types[0]->data_type == LL_VECTOR) &&
        (llt->sub_types[0]->sub_elements == 3)) {
      LL_Type *veleTy = llt->sub_types[0]->sub_types[0];
      LL_Type *vTy = ll_get_vector_type(veleTy, 4);
      llt = make_ptr_lltype(vTy);
    }
#ifdef TARGET_LLVM_ARM
    switch (zsize_of(vect_dtype)) {
    case 2:
      int_llt = make_ptr_lltype(make_lltype_from_dtype(DT_SINT));
      break;
    case 3:
    case 4:
      int_llt = make_ptr_lltype(make_lltype_from_dtype(DT_INT));
      break;
    default:
      break;
    }
#endif
    operand = gen_address_operand(ld_ili, nme_ili, false,
                                  (int_llt ? int_llt : llt), (MSZ)-1);
    if ((operand->ll_type->data_type == LL_PTR) ||
        (operand->ll_type->data_type == LL_ARRAY)) {
      operand = make_load(ilix, operand, operand->ll_type->sub_types[0],
                          (MSZ)-2, ldst_instr_flags_from_dtype(vect_dtype));
      if (int_llt != NULL) {
        if (expected_type == NULL ||
            !strict_match(operand->ll_type, int_llt->sub_types[0]))
          operand = make_bitcast(operand, llt->sub_types[0]);
      }
    } else if (int_llt) {
      operand = make_bitcast(operand, llt);
    }
    if (expected_type && (expected_type->data_type == LL_VECTOR) &&
        !strict_match(operand->ll_type, expected_type)) {
      if (expected_type->sub_elements == 3) {
        if (int_llt && (zsize_of(vect_dtype) == 4))
          operand = make_bitcast(operand, vect_lltype);
      } else {
        operand = make_bitcast(operand, expected_type);
      }
    }
  } break;
  case IL_VLDU: {
    LL_Type *llt, *vect_lltype, *int_llt = NULL;
    DTYPE vect_dtype = ILI_DTyOPND(ilix, 3);

    nme_ili = ILI_OPND(ilix, 2);
    ld_ili = ILI_OPND(ilix, 1);
    vect_lltype = make_lltype_from_dtype(vect_dtype);
    llt = make_ptr_lltype(vect_lltype);
    if (expected_type && (expected_type->data_type == LL_VECTOR) &&
        (expected_type->sub_elements == 4) &&
        (llt->sub_types[0]->data_type == LL_VECTOR) &&
        (llt->sub_types[0]->sub_elements == 3)) {
      LL_Type *veleTy = llt->sub_types[0]->sub_types[0];
      LL_Type *vTy = ll_get_vector_type(veleTy, 4);
      llt = make_ptr_lltype(vTy);
    }
#ifdef TARGET_LLVM_ARM
    if (vect_lltype->sub_elements != 3) {
      if (vect_lltype->sub_elements != 3) {
        switch (zsize_of(vect_dtype)) {
        case 2:
          int_llt = make_ptr_lltype(make_lltype_from_dtype(DT_SINT));
          break;
        case 4:
          if (expected_type && (expected_type->data_type == LL_VECTOR) &&
              (expected_type->sub_elements != 3))
            int_llt = make_ptr_lltype(make_lltype_from_dtype(DT_INT));
          break;
        default:
          break;
        }
      } else if (expected_type && ll_type_int_bits(expected_type)) {
        int_llt = make_ptr_lltype(expected_type);
      }
    }
#endif
    operand = gen_address_operand(ld_ili, nme_ili, false,
                                  (int_llt ? int_llt : llt), (MSZ)-1);
    if (ll_type_is_mem_seq(operand->ll_type)) {
      operand = make_load(ilix, operand, operand->ll_type->sub_types[0],
                          (MSZ)-2, /* unaligned */ 0);
      if (int_llt != NULL) {
        if (expected_type == NULL ||
            !strict_match(operand->ll_type, int_llt->sub_types[0]))
          operand = make_bitcast(operand, llt->sub_types[0]);
      } else if (vect_lltype->sub_elements == 3 && expected_type &&
                 ll_type_int_bits(expected_type) &&
                 !strict_match(operand->ll_type, expected_type)) {
        operand = gen_resized_vect(operand, ll_type_bytes(expected_type), 0);
        operand = make_bitcast(operand, expected_type);
      }
    } else if (int_llt) {
      operand = make_bitcast(operand, llt);
    }
    if (expected_type && expected_type->data_type == LL_VECTOR &&
        !strict_match(operand->ll_type, expected_type)) {
      if (expected_type->sub_elements == 3) {
        if (int_llt && (zsize_of(vect_dtype) == 4))
          operand = make_bitcast(operand, vect_lltype);
      } else {
        operand = make_bitcast(operand, expected_type);
      }
    }
  } break;
  case IL_LDSCMPLX:
  case IL_LDDCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_LDQCMPLX:
#endif
  {
    unsigned flags;
    ld_ili = ILI_OPND(ilix, 1);
    nme_ili = ILI_OPND(ilix, 2);
    msz = (MSZ)ILI_OPND(ilix, 3);
    flags = opc == IL_LDSCMPLX ? DT_CMPLX
#ifdef TARGET_SUPPORTS_QUADFP
          : opc == IL_LDQCMPLX ? DT_QCMPLX
#endif
                               : DT_DCMPLX;
    operand = gen_address_operand(ld_ili, nme_ili, false,
                                  make_ptr_lltype(expected_type), (MSZ)-1);
    assert(operand->ll_type->data_type == LL_PTR,
           "Invalid operand for cmplx load", ilix, ERR_Fatal);
    operand =
        make_load(ilix, operand, operand->ll_type->sub_types[0], msz, flags);
  } break;
  case IL_LD:
  case IL_LDSP:
  case IL_LDDP:
  case IL_LDKR:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128LD:
#endif
#ifdef TARGET_LLVM_X8664
  case IL_LDQ:
  case IL_LD256:
#endif
#ifdef TARGET_SUPPORTS_QUADFP
  /* to support quad precision load */
  case IL_LDQP:
#endif
    ld_ili = ILI_OPND(ilix, 1);
    nme_ili = ILI_OPND(ilix, 2);
    msz = MSZ_ILI_OPND(ilix, 3);
    operand = gen_address_operand(ld_ili, nme_ili, false, NULL, msz);
    if ((operand->ll_type->data_type == LL_PTR) ||
        (operand->ll_type->data_type == LL_ARRAY)) {
      LL_InstrListFlags flags =
          ldst_instr_flags_from_dtype_nme(msz_dtype(msz), nme_ili);
      operand =
          make_load(ilix, operand, operand->ll_type->sub_types[0], msz, flags);
    }
    break;
  case IL_ATOMICLDSP:
  case IL_ATOMICLDDP: {
    DTYPE fromdtype, todtype;
    MSZ newmsz;
    LL_InstrListFlags flags;
    ld_ili = ILI_OPND(ilix, 1);
    nme_ili = ILI_OPND(ilix, 2);
    msz = ILI_MSZ_OF_LD(ilix);
    if (opc == IL_ATOMICLDSP) {
      fromdtype = DT_FLOAT;
      todtype = DT_INT;
      newmsz = MSZ_WORD;
    } else {
      fromdtype = DT_DBLE;
      todtype = DT_INT8;
      newmsz = MSZ_I8;
    }
    flags = (LL_InstrListFlags)(
        ll_instr_flags_for_memory_order_and_scope(ilix) |
        ldst_instr_flags_from_dtype_nme(msz_dtype(msz), nme_ili));
    operand = gen_address_operand(ld_ili, nme_ili, false, NULL, newmsz);
    operand =
        make_load(ilix, operand, operand->ll_type->sub_types[0], newmsz, flags);
    operand = make_bitcast(operand, make_lltype_from_dtype(fromdtype));
  } break;

  case IL_ATOMICLDA:
  case IL_ATOMICLDI:
  case IL_ATOMICLDKR: {
    LL_InstrListFlags flags;
    ld_ili = ILI_OPND(ilix, 1);
    nme_ili = ILI_OPND(ilix, 2);
    msz = (MSZ)ILI_MSZ_OF_LD(ilix);
    flags = (LL_InstrListFlags)(
        ll_instr_flags_for_memory_order_and_scope(ilix) |
        ldst_instr_flags_from_dtype_nme(msz_dtype(msz), nme_ili));
    operand = gen_address_operand(ld_ili, nme_ili, false, NULL, msz);
    operand =
        make_load(ilix, operand, operand->ll_type->sub_types[0], msz, flags);
  } break;
  case IL_VCON:
  case IL_KCON:
  case IL_ICON:
  case IL_FCON:
  case IL_DCON:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCON:
#endif
  case IL_SCMPLXCON:
  case IL_DCMPLXCON:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXCON:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CON:
#endif
    operand = gen_const_expr(ilix, expected_type);
    break;
  case IL_FIX:
  case IL_DFIX:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFIX:
#endif
    operand = gen_unary_expr(ilix, I_FPTOSI);
    break;
  case IL_FIXK:
  case IL_DFIXK:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFIXK:
#endif
    operand = gen_unary_expr(ilix, I_FPTOSI);
    break;
  case IL_FIXUK:
  case IL_DFIXUK:
  case IL_DFIXU:
  case IL_UFIX:
    operand = gen_unary_expr(ilix, I_FPTOUI);
    break;
  case IL_FLOATU:
  case IL_DFLOATU:
  case IL_FLOATUK:
  case IL_DFLOATUK:
    operand = gen_unary_expr(ilix, I_UITOFP);
    break;
  case IL_FLOAT:
  case IL_DFLOAT:
  case IL_DFLOATK:
  case IL_FLOATK:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFLOAT:
  case IL_QFLOATK:
#endif
    operand = gen_unary_expr(ilix, I_SITOFP);
    break;
  case IL_SNGL:
#ifdef TARGET_SUPPORTS_QUADFP
  /* convert the quad precision to single precision */
  case IL_SNGQ:
  /* convert the quad precision to double precision */
  case IL_DBLEQ:
#endif
    operand = gen_unary_expr(ilix, I_FPTRUNC);
    break;
  case IL_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_RQUAD:
  case IL_DQUAD:
#endif
    operand = gen_unary_expr(ilix, I_FPEXT);
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FROM:
    operand = gen_unary_expr(ilix, I_FPEXT);
    break;
  case IL_FLOAT128TO:
    operand = gen_unary_expr(ilix, I_FPTRUNC);
    break;
#endif
  case IL_ALLOC:
    operand = gen_unary_expr(ilix, I_ALLOCA);
    break;
  case IL_DEALLOC:
    break;
  case IL_VADD:
    operand = gen_binary_vexpr(ilix, I_ADD, I_ADD, I_FADD);
    break;
  case IL_IADD:
  case IL_KADD:
  case IL_UKADD:
  case IL_UIADD:
    operand = gen_binary_expr(ilix, I_ADD);
    break;
  case IL_FADD:
  case IL_DADD:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QADD:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ADD:
#endif
    operand = gen_binary_expr(ilix, I_FADD);
    break;
  case IL_SCMPLXADD:
    operand = gen_cmplx_math(ilix, DT_CMPLX, I_FADD);
    break;
  case IL_DCMPLXADD:
    operand = gen_cmplx_math(ilix, DT_DCMPLX, I_FADD);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXADD:
    operand = gen_cmplx_math(ilix, DT_QCMPLX, I_FADD);
    break;
#endif
  case IL_VSUB:
    operand = gen_binary_vexpr(ilix, I_SUB, I_SUB, I_FSUB);
    break;
  case IL_ISUB:
  case IL_KSUB:
  case IL_UKSUB:
  case IL_UISUB:
    operand = gen_binary_expr(ilix, I_SUB);
    break;
  case IL_FSUB:
  case IL_DSUB:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QSUB:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128SUB:
#endif
    operand = gen_binary_expr(ilix, I_FSUB);
    break;
  case IL_SCMPLXSUB:
    operand = gen_cmplx_math(ilix, DT_CMPLX, I_FSUB);
    break;
  case IL_DCMPLXSUB:
    operand = gen_cmplx_math(ilix, DT_DCMPLX, I_FSUB);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXSUB:
    operand = gen_cmplx_math(ilix, DT_QCMPLX, I_FSUB);
    break;
#endif
  case IL_VMUL:
    operand = gen_binary_vexpr(ilix, I_MUL, I_MUL, I_FMUL);
    break;
  case IL_IMUL:
  case IL_KMUL:
  case IL_UKMUL:
  case IL_UIMUL:
    operand = gen_binary_expr(ilix, I_MUL);
    break;
  case IL_KMULH:
  case IL_UKMULH:
    operand = gen_mulh_expr(ilix);
    break;
  case IL_FMUL:
  case IL_DMUL:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMUL:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128MUL:
#endif
    operand = gen_binary_expr(ilix, I_FMUL);
    break;
  case IL_SCMPLXMUL:
    operand = gen_cmplx_mul(ilix, DT_CMPLX);
    break;
  case IL_DCMPLXMUL:
    operand = gen_cmplx_mul(ilix, DT_DCMPLX);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXMUL:
    operand = gen_cmplx_mul(ilix, DT_QCMPLX);
    break;
#endif
  case IL_VDIV:
    operand = gen_binary_vexpr(ilix, I_SDIV, I_UDIV, I_FDIV);
    break;
  case IL_KDIV:
  case IL_IDIV:
    operand = gen_binary_expr(ilix, I_SDIV);
    break;
  case IL_UKDIV:
  case IL_UIDIV:
    operand = gen_binary_expr(ilix, I_UDIV);
    break;
  case IL_FDIV:
  case IL_DDIV:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QDIV:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128DIV:
#endif
    operand = gen_binary_expr(ilix, I_FDIV);
    break;
  case IL_VLSHIFTV:
  case IL_VLSHIFTS:
  case IL_LSHIFT:
  case IL_ULSHIFT:
  case IL_KLSHIFT:
    operand = gen_binary_expr(ilix, I_SHL);
    break;
  case IL_VRSHIFTV:
  case IL_VRSHIFTS:
    operand = gen_binary_vexpr(ilix, I_ASHR, I_LSHR, I_ASHR);
    break;
  case IL_VURSHIFTS:
    operand = gen_binary_vexpr(ilix, I_LSHR, I_LSHR, I_LSHR);
    break;
  case IL_URSHIFT:
  case IL_KURSHIFT:
    operand = gen_binary_expr(ilix, I_LSHR);
    break;
  case IL_RSHIFT:
  case IL_ARSHIFT:
  case IL_KARSHIFT:
    operand = gen_binary_expr(ilix, I_ASHR);
    break;
  case IL_VAND:
    /* need to check dtype - if floating type need special code to
     * cast to int, compare, then cast back to float. Similar to
     * what is done with the IL_FAND case, except with vectors.
     * Else just fall through.
     * NB: currently this method only works for float values, not
     * doubles (and when using -Mfprelaxed that is all our compiler
     * currently operates on anyway.)
     */
    dtype = (DTYPE)ILI_OPND(ilix, 3); /* get the vector dtype */
    assert(TY_ISVECT(DTY(dtype)), "gen_llvm_expr(): expected vect type",
           DTY(dtype), ERR_Fatal);
    /* check the base type for float/real */
    if (DTY(DTySeqTyElement(dtype)) == TY_FLOAT) {
      OPERAND *op1, *op2, *op3, *op4, *op5, *op6;
      INSTR_LIST *instr1, *instr2, *instr3;
      int vsize = DTyVecLength(dtype);
      LL_Type *viTy = make_vtype(DT_INT, vsize);
      LL_Type *vfTy = make_vtype(DT_FLOAT, vsize);
      op1 = gen_llvm_expr(ILI_OPND(ilix, 1), NULL);
      op2 = make_tmp_op(viTy, make_tmps());
      instr1 = gen_instr(I_BITCAST, op2->tmps, viTy, op1);
      ad_instr(ilix, instr1);
      op3 = gen_llvm_expr(ILI_OPND(ilix, 2), NULL);
      op4 = make_tmp_op(viTy, make_tmps());
      instr2 = gen_instr(I_BITCAST, op4->tmps, viTy, op3);
      ad_instr(ilix, instr2);
      op6 = make_tmp_op(vfTy, make_tmps());
      op2->next = op4;
      op5 = ad_csed_instr(I_AND, 0, viTy, op2, InstrListFlagsNull, false);
      instr3 = gen_instr(I_BITCAST, op6->tmps, vfTy, op5);
      ad_instr(ilix, instr3);
      operand = op6;
      break;
    }
    FLANG_FALLTHROUGH;
  case IL_KAND:
  case IL_AND:
    operand = gen_binary_expr(ilix, I_AND);
    break;
  case IL_VOR:
  case IL_KOR:
  case IL_OR:
    operand = gen_binary_expr(ilix, I_OR);
    break;
  case IL_VXOR:
  case IL_KXOR:
  case IL_XOR:
    operand = gen_binary_expr(ilix, I_XOR);
    break;
  case IL_VMOD:
    operand = gen_binary_vexpr(ilix, I_SREM, I_UREM, I_FREM);
    break;
  case IL_SCMPLXXOR:
    operand = gen_cmplx_math(ilix, DT_CMPLX, I_XOR);
    break;
  case IL_DCMPLXXOR:
    operand = gen_cmplx_math(ilix, DT_DCMPLX, I_XOR);
    break;
  case IL_KMOD:
  case IL_MOD:
    operand = gen_binary_expr(ilix, I_SREM);
    break;
  case IL_KUMOD:
  case IL_UIMOD:
    operand = gen_binary_expr(ilix, I_UREM);
    break;
  case IL_ASUB:
  case IL_AADD: {
    LL_Type *t =
        expected_type ? expected_type : make_lltype_from_dtype(DT_CPTR);
    operand = gen_base_addr_operand(ilix, t);
  } break;
  /* jumps on zero with cc */
  case IL_FCJMPZ:
    tmp[0] = 0.0;
    f = 0.0;
    mftof(f, tmp[1]);
    zero_ili = ad1ili(IL_FCON, getcon(tmp, DT_FLOAT));
    comp_exp_type = make_lltype_from_dtype(DT_FLOAT);
    FLANG_FALLTHROUGH;
  case IL_DCJMPZ:
    if (!zero_ili) {
      d = 0.0;
      xmdtod(d, dtmp.tmp);
      zero_ili = ad1ili(IL_DCON, getcon(dtmp.tmp, DT_DBLE));
      comp_exp_type = make_lltype_from_dtype(DT_DBLE);
    }
    operand->ot_type = OT_CC;
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    ili_cc = ILI_ccOPND(ilix, 2);
    if (IEEE_CMP)
      float_jmp = true;
    operand->val.cc = convert_to_llvm_fltcc(ili_cc);
    float_jmp = false;
    operand->ll_type = make_type_from_opc(opc);
    goto process_cc;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
    if (!zero_ili) {
      q = 0.0;
      xmqtoq(q, qtmp.tmp);
      zero_ili = ad1ili(IL_QCON, getcon(qtmp.tmp, DT_QUAD));
      comp_exp_type = make_lltype_from_dtype(DT_QUAD);
    }
    operand->ot_type = OT_CC;
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    ili_cc = ILI_ccOPND(ilix, 2);
    if (IEEE_CMP)
      float_jmp = true;
    operand->val.cc = convert_to_llvm_fltcc(ili_cc);
    float_jmp = false;
    operand->ll_type = make_type_from_opc(opc);
    goto process_cc;
    break;
#endif
  case IL_UKCJMPZ:
    zero_ili = ad_kconi(0);
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_uintcc(ILI_ccOPND(ilix, 2));
    operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    comp_exp_type = make_lltype_from_dtype(DT_INT8);
    goto process_cc;
    break;
  case IL_UICJMPZ:
    zero_ili = ad_icon(0);
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_uintcc(ILI_ccOPND(ilix, 2));
    operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    comp_exp_type = make_lltype_from_dtype(DT_INT);
    goto process_cc;
    break;

  case IL_KCJMPZ:
    zero_ili = ad_kconi(0);
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_intcc(ILI_ccOPND(ilix, 2));
    operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    comp_exp_type = make_lltype_from_dtype(DT_INT8);
    goto process_cc;
    break;
  case IL_ICJMPZ:
    zero_ili = ad_icon(0);

    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_intcc(ILI_ccOPND(ilix, 2));
    operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    comp_exp_type = make_lltype_from_dtype(DT_INT);
    goto process_cc;
    break;
  case IL_ACJMPZ:
    zero_ili = ad_icon(0);

    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_uintcc(ILI_ccOPND(ilix, 2));
    comp_exp_type = operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = zero_ili;
    goto process_cc;
    break;
  /* jumps with cc and expression */
  case IL_FCJMP:
  case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
#endif
    operand->ot_type = OT_CC;
    first_ili = ILI_OPND(ilix, 1);
    second_ili = ILI_OPND(ilix, 2);
    ili_cc = ILI_ccOPND(ilix, 3);
    if (IEEE_CMP)
      float_jmp = true;
    operand->val.cc = convert_to_llvm_fltcc(ili_cc);
    float_jmp = false;
    comp_exp_type = operand->ll_type = make_type_from_opc(opc);
    goto process_cc;
    break;
  case IL_UKCJMP:
  case IL_UICJMP:
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_uintcc(ILI_ccOPND(ilix, 3));
    comp_exp_type = operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = ILI_OPND(ilix, 2);
    goto process_cc;
    break;
  case IL_KCJMP:
  case IL_ICJMP:
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_intcc(ILI_ccOPND(ilix, 3));
    comp_exp_type = operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = ILI_OPND(ilix, 2);
    goto process_cc;
    break;
  case IL_ACJMP:
    operand->ot_type = OT_CC;
    operand->val.cc = convert_to_llvm_uintcc(ILI_ccOPND(ilix, 3));
    comp_exp_type = operand->ll_type = make_type_from_opc(opc);
    first_ili = ILI_OPND(ilix, 1);
    second_ili = ILI_OPND(ilix, 2);
  process_cc:
    operand->next = cc_op1 = gen_llvm_expr(first_ili, operand->ll_type);
    operand->next->next = cc_op2 = gen_llvm_expr(second_ili, comp_exp_type);
    break;
  case IL_FCMP:
  case IL_DCMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMP:
#endif
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CMP:
#endif
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    ili_cc = ILI_ccOPND(ilix, 3);
    if (IEEE_CMP)
      float_jmp = true;
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili, ili_cc, CMP_FLT,
                               I_FCMP);
    break;
  case IL_CMPNEQSS: {
    OPERAND *op1;
    INSTR_LIST *instr1;
    unsigned bits = BITS_IN_BYTE * size_of(DT_FLOAT);
    LL_Type *iTy = make_int_lltype(bits);
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    ili_cc = CC_NE;
    if (IEEE_CMP)
      float_jmp = true;
    operand = gen_optext_comp_operand(operand, opc, lhs_ili, rhs_ili, ili_cc,
                                      CMP_FLT, I_FCMP, 0, 0);
    /* sext i1 to i32 */
    op1 = make_tmp_op(iTy, make_tmps());
    instr1 = gen_instr(I_SEXT, op1->tmps, iTy, operand);
    ad_instr(ilix, instr1);
    operand = op1;
  } break;
  case IL_VCMPNEQ: {
    OPERAND *op1;
    INSTR_LIST *instr1;
    int vsize;
    LL_Type *viTy;
    dtype = (DTYPE)ILI_OPND(ilix, 3); /* get the vector dtype */
    assert(TY_ISVECT(DTY(dtype)), "gen_llvm_expr(): expected vect type",
           DTY(dtype), ERR_Fatal);
    vsize = DTyVecLength(dtype);
    viTy = make_vtype(DT_INT, vsize);
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    ili_cc = CC_NE;
    if (IEEE_CMP)
      float_jmp = true;
    operand = gen_optext_comp_operand(operand, opc, lhs_ili, rhs_ili, ili_cc,
                                      CMP_FLT, I_FCMP, 0, ilix);
    /* sext i1 to i32 */
    op1 = make_tmp_op(viTy, make_tmps());
    instr1 = gen_instr(I_SEXT, op1->tmps, viTy, operand);
    ad_instr(ilix, instr1);
    operand = op1;
  } break;
  case IL_KCMPZ:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ad_kconi(0);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 2), CMP_INT, I_ICMP);
    break;
  case IL_UKCMPZ:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ad_kconi(0);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 2), CMP_INT | CMP_USG, I_ICMP);
    break;
  case IL_ICMPZ:
    /* what are we testing for here? We may have an ICMPZ pointing to
     * an FCMP, which is negating the sense of the FCMP. To account for
     * NaNs (hence the IEEE_CMP test) we need to correctly negate
     * the floating comparison operator, taking into account both
     * ordered and unordered cases. That is why we set fcmp_negate
     * for use in convert_to_llvm_cc().
     */
    if (IEEE_CMP && ILI_OPC(ILI_OPND(ilix, 1)) == IL_FCMP) {
      int fcmp_ili = ILI_OPND(ilix, 1);

      lhs_ili = ILI_OPND(fcmp_ili, 1);
      rhs_ili = ILI_OPND(fcmp_ili, 2);
      fcmp_negate = true;
      operand = gen_comp_operand(operand, IL_FCMP, lhs_ili, rhs_ili,
                                 ILI_OPND(fcmp_ili, 3), CMP_FLT, I_FCMP);
      fcmp_negate = false;
      break;
    }
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ad_icon(0);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 2), CMP_INT, I_ICMP);
    break;
  case IL_ACMPZ:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ad_icon(0);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 2), CMP_INT | CMP_USG, I_ICMP);
    break;
  case IL_UICMPZ:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ad_icon(0);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 2), CMP_INT | CMP_USG, I_ICMP);
    break;
  case IL_UKCMP:
  case IL_UICMP:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 3), CMP_INT | CMP_USG, I_ICMP);
    break;
  case IL_ACMP:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 3), CMP_INT | CMP_USG, I_ICMP);
    break;
  case IL_KCMP:
  case IL_ICMP:
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    operand = gen_comp_operand(operand, opc, lhs_ili, rhs_ili,
                               ILI_OPND(ilix, 3), CMP_INT, I_ICMP);
    break;
  case IL_AIMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_CPTR));
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_INT);
    break;
  case IL_AKMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_CPTR));
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_INT8);
    break;
  case IL_KIMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_INT));
    break;
  case IL_IKMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_INT));
    operand = sign_extend_int(operand, 64);
    break;
  case IL_UIKMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_UINT));
    operand = zero_extend_int(operand, 64);
    break;
  case IL_IAMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_INT));
    /* This ILI is sometimes generated on 64-bit targets. Make sure it is
     * sign-extended, the LLVM inttoptr instruction zero-extends. */
    if (size_of(DT_CPTR) == 8)
      operand = sign_extend_int(operand, 64);
    break;

  case IL_KAMV:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_int_lltype(64));
#if TARGET_PTRSIZE < 8
    /* Explicitly truncate to a 32-bit int - convert_int_to_ptr() won't work
     * because it can't truncate. */
    operand =
        convert_int_size(ilix, operand, make_int_lltype(BITS_IN_BYTE * TARGET_PTRSIZE));
#endif
    break;

    operand = gen_llvm_expr(ILI_OPND(ilix, 1), expected_type);
    break;
#ifdef IL_DFRSPX87
  case IL_FREESPX87:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRSPX87:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_FLOAT);
    goto _process_define_ili;
  case IL_FREEDPX87:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRDPX87:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_DBLE);
    goto _process_define_ili;
#endif
  case IL_FREEKR:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRKR:
    if (expected_type == NULL) {
      expected_type = make_lltype_from_dtype(DT_INT8);
    }
    goto _process_define_ili;
  case IL_FREEIR:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRIR:
    if (expected_type == NULL) {
      expected_type = make_lltype_from_dtype(DT_INT);
    }
    goto _process_define_ili;
  case IL_FREESP:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRSP:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_FLOAT);
    goto _process_define_ili;
  case IL_FREEDP:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRDP:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_DBLE);
    goto _process_define_ili;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DFRQP:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_QUAD);
    goto _process_define_ili;
#endif
  case IL_DFR128:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_128);
    goto _process_define_ili;
  case IL_DFR256:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_256);
    goto _process_define_ili;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128FREE:
    cse_opc = 1;
    FLANG_FALLTHROUGH;
  case IL_FLOAT128RESULT:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_FLOAT128);
    goto _process_define_ili;
#endif
  case IL_FREEAR:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRAR:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_CPTR);
    goto _process_define_ili;
  case IL_FREECS:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRCS:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_CMPLX);
    goto _process_define_ili;
  case IL_FREECD:
    cse_opc = true;
    FLANG_FALLTHROUGH;
  case IL_DFRCD:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(DT_DCMPLX);

  _process_define_ili:
    /* llvm_info.curr_ret_ili = ilix; */
    llvm_info.curr_ret_dtype = cse_opc ? DT_NONE : dtype_from_return_type(opc);
    switch (ILI_OPC(ILI_OPND(ilix, 1))) {
    default:
      break;
#ifdef PGPLUS
    case IL_JSRA:
#endif
    case IL_QJSR:
    case IL_JSR:
    case IL_GJSR:
      /*
       * For compiler-created functions, its DTYPE record is
       * believable if its dtype value is not a 'predeclared,
       * e.g., DT_IFUNC.
       */
      if (CCSYMG(ILI_OPND(ILI_OPND(ilix, 1), 1)) &&
          DTYPEG(ILI_OPND(ILI_OPND(ilix, 1), 1)) < DT_MAX) {
        update_return_type_for_ccfunc(ILI_OPND(ilix, 1), opc);
      }
    }

    if (is_complex_result(ilix)) {
      comp_exp_type = expected_type;
      expected_type = complex_result_type(ILI_OPC(ilix));
    }

    /* Identical calls in the same block must be csed for correctness,
     * identical calls that are supposed to be repeated are given different
     * ILI numbers.
     *
     * Don't cse QJSR/GJSR calls. They are hashed as other instructions, so
     * cse'ing could inadvertently move loads across stores. See
     * pgc_correctll/gf40.c on an architecture that calls __mth_i_floatk
     * with QJSR.
     */
    if (ILI_OPC(ILI_OPND(ilix, 1)) == IL_QJSR ||
        ILI_OPC(ILI_OPND(ilix, 1)) == IL_GJSR) {
      operand = gen_llvm_expr(ILI_OPND(ilix, 1), expected_type);
    } else {
      OPERAND **csed_operand = get_csed_operand(ILI_OPND(ilix, 1));

      if (csed_operand == NULL) {
        operand = gen_llvm_expr(ILI_OPND(ilix, 1), expected_type);
        add_to_cselist(ILI_OPND(ilix, 1));
        csed_operand = get_csed_operand(ILI_OPND(ilix, 1));
        set_csed_operand(csed_operand, operand);
      } else if (!ILI_COUNT(ILI_OPND(ilix, 1))) {
        operand = gen_llvm_expr(ILI_OPND(ilix, 1), expected_type);
      } else if (*csed_operand == NULL) {
        operand = gen_llvm_expr(ILI_OPND(ilix, 1), expected_type);
        set_csed_operand(csed_operand, operand);
      } else {
        operand = gen_copy_op(*csed_operand);
      }
      assert(operand, "null operand in cse list for ilix ", ILI_OPND(ilix, 1),
             ERR_Fatal);
    }
    if (is_complex_result(ilix)) {
      operand = gen_extract_value_ll(operand, expected_type, comp_exp_type,
                                     (ILI_OPND(ilix, 2) == 'r') ? 0 : 1);
      expected_type = comp_exp_type;
    }
    break;
  case IL_FREE:
    if (expected_type == NULL)
      expected_type = make_lltype_from_dtype(ILI_DTyOPND(ilix, 2));
    goto _process_define_ili;

  case IL_VCVTV:
    operand = gen_convert_vector(ilix);
    break;

  case IL_VCVTR:
    operand = gen_bitcast_vector(ilix);
    break;

  case IL_VCVTS:
    operand = gen_scalar_to_vector(
        ilix, make_lltype_from_dtype(ILI_DTyOPND(ilix, 2)));
    break;

  case IL_VNOT:
  case IL_NOT:
  case IL_UNOT:
  case IL_KNOT:
  case IL_UKNOT:
    operand = gen_binary_expr(ilix, I_XOR);
    if (opc == IL_VNOT && (ILI_OPC(ILI_OPND(ilix, 1)) == IL_VCMP ||
                           ILI_OPC(ILI_OPND(ilix, 1)) == IL_VPERMUTE))
      expected_type = operand->ll_type;
    break;
  case IL_VNEG:
    operand = gen_binary_vexpr(ilix, I_SUB, I_SUB, I_FSUB);
    break;
  case IL_INEG:
  case IL_UINEG:
  case IL_KNEG:
  case IL_UKNEG:
    operand = gen_binary_expr(ilix, I_SUB);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QNEG:
#endif
  case IL_DNEG:
  case IL_FNEG:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CHS:
#endif
    operand = gen_binary_expr(ilix, I_FSUB);
    break;
  case IL_SCMPLXNEG:
  case IL_DCMPLXNEG:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXNEG:
#endif
  {
    OPERAND *res, *op_rneg, *op_ineg, *c1, *cse1;
    LL_Type *cmpnt_ty;
    const DTYPE dt = opc == IL_SCMPLXNEG ? DT_CMPLX
#ifdef TARGET_SUPPORTS_QUADFP
                   : opc == IL_QCMPLXNEG ? DT_QCMPLX
#endif
                                         : DT_DCMPLX;
    const DTYPE et = opc == IL_SCMPLXNEG ? DT_FLOAT
#ifdef TARGET_SUPPORTS_QUADFP
                   : opc == IL_QCMPLXNEG ? DT_QUAD
#endif
                                         : DT_DBLE;

    cmpnt_ty = make_lltype_from_dtype(dt == DT_CMPLX ? DT_FLOAT
#ifdef TARGET_SUPPORTS_QUADFP
                                   : dt == DT_QCMPLX ? DT_QUAD
#endif
                                                     : DT_DBLE);

    c1 = gen_eval_cmplx_value(ILI_OPND(ilix, 1), dt);
    cse1 = gen_copy_operand(c1);

    /* real = 0 - real */
    op_rneg = (et == DT_QUAD) ? make_constval_opL(cmpnt_ty, 0, 0, 0, 0) :
	    make_constval_op(cmpnt_ty, 0, 0);
    op_rneg->next = gen_extract_value(c1, dt, et, 0);
    op_rneg =
        ad_csed_instr(I_FSUB, 0, cmpnt_ty, op_rneg, InstrListFlagsNull, true);

    /* imag = 0 - imag */
    op_ineg = (et == DT_QUAD) ? make_constval_opL(cmpnt_ty, 0, 0, 0, 0) :
	    make_constval_op(cmpnt_ty, 0, 0);
    op_ineg->next = gen_extract_value(cse1, dt, et, 1);
    op_ineg =
        ad_csed_instr(I_FSUB, 0, cmpnt_ty, op_ineg, InstrListFlagsNull, true);

    /* {real, imag} */
    res = make_undef_op(make_lltype_from_dtype(dt));
    res = gen_insert_value(res, op_rneg, 0);
    operand = gen_insert_value(res, op_ineg, 1);
  } break;
  case IL_CSE:
  case IL_CSEKR:
  case IL_CSEIR:
  case IL_CSESP:
  case IL_CSEDP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_CSEQP:
#endif
  case IL_CSEAR:
  case IL_CSECS:
  case IL_CSECD:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CSE:
#endif
  {
    int csed_ilix;
    OPERAND **csed_operand;

    csed_ilix = ILI_OPND(ilix, 1);
    if (ILI_ALT(csed_ilix))
      csed_ilix = ILI_ALT(csed_ilix);
    csed_operand = get_csed_operand(csed_ilix);

    assert(csed_operand, "missing cse operand list for ilix ", csed_ilix,
           ERR_Fatal);
    if (!ILI_COUNT(csed_ilix)) {
      operand = gen_llvm_expr(csed_ilix, expected_type);
    } else {
      operand = gen_copy_op(*csed_operand);
    }
    assert(operand, "null operand in cse list for ilix ", csed_ilix, ERR_Fatal);
  } break;
  case IL_IR2SP:
    operand = make_bitcast(gen_llvm_expr(ILI_OPND(ilix, 1), 0),
                           make_lltype_from_dtype(DT_REAL));
    break;
  case IL_KR2DP:
    operand = make_bitcast(gen_llvm_expr(ILI_OPND(ilix, 1), 0),
                           make_lltype_from_dtype(DT_DBLE));
    break;
  /* these next ILI are currently generated by idiom recognition within
   * induc, and as arguments to our __c_mset* routines we want them treated
   * as integer bits without conversion.
   */
  case IL_SP2IR:
    operand = make_bitcast(gen_llvm_expr(ILI_OPND(ilix, 1), 0),
                           make_lltype_from_dtype(DT_INT));
    break;
  case IL_DP2KR:
    operand = make_bitcast(gen_llvm_expr(ILI_OPND(ilix, 1), 0),
                           make_lltype_from_dtype(DT_INT8));
    break;
  case IL_CS2KR:
    comp_exp_type = make_lltype_from_dtype(DT_CMPLX);
    cc_op1 = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_CMPLX));

    /* ILI_OPND(ilix, 1) can be expression */
    if (ILTY_CONS == IL_TYPE(ILI_OPC(ILI_OPND(ilix, 1)))) {
      cc_op2 = make_var_op(ILI_SymOPND(ILI_OPND(ilix, 1), 1));
    } else {
      assert(0, "gen_llvm_expr(): unsupport operand for CS2KR ", opc,
             ERR_Fatal);
      /* it is not worth it to do it */
      break;
    }

    operand =
        make_bitcast(cc_op2, make_ptr_lltype(make_lltype_from_dtype(DT_INT8)));
    operand = make_load(ilix, operand, operand->ll_type->sub_types[0], (MSZ)-2,
                        ldst_instr_flags_from_dtype(DT_INT8));
    break;
  case IL_SCMPLX2REAL:
    dt = DT_CMPLX;
    cmpnt = DT_NONE;
    goto component;
  case IL_DCMPLX2REAL:
    dt = DT_DCMPLX;
    cmpnt = DT_NONE;
    goto component;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLX2REAL:
    dt = DT_QCMPLX;
    cmpnt = DT_NONE;
    goto component;
#endif
  case IL_SCMPLX2IMAG:
    dt = DT_CMPLX;
    cmpnt = (DTYPE)1;
    goto component;
  case IL_DCMPLX2IMAG:
    dt = DT_DCMPLX;
    cmpnt = (DTYPE)1;
    goto component;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLX2IMAG:
    dt = DT_QCMPLX;
    cmpnt = (DTYPE)1;
    goto component;
#endif
  component:
    c1 = gen_eval_cmplx_value(ILI_OPND(ilix, 1), dt);
    operand = gen_extract_value(c1, dt,
                                dt == DT_CMPLX  ? DT_FLOAT
#ifdef TARGET_SUPPORTS_QUADFP
                              : dt == DT_QCMPLX ? DT_QUAD
#endif
                                                : DT_DBLE, cmpnt);
    break;
  case IL_SPSP2SCMPLX:
  case IL_DPDP2DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QPQP2QCMPLX:
#endif
  {
    LL_Type *dt, *et;
    if (opc == IL_SPSP2SCMPLX) {
      dt = make_lltype_from_dtype(DT_CMPLX);
      et = make_lltype_from_dtype(DT_FLOAT);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (opc == IL_QPQP2QCMPLX) {
      dt = make_lltype_from_dtype(DT_QCMPLX);
      et = make_lltype_from_dtype(DT_QUAD);
#endif
    } else {
      dt = make_lltype_from_dtype(DT_DCMPLX);
      et = make_lltype_from_dtype(DT_DBLE);
    }
    cc_op1 = gen_llvm_expr(ILI_OPND(ilix, 1), et);
    cc_op2 = gen_llvm_expr(ILI_OPND(ilix, 2), et);
    operand = make_undef_op(dt);
    operand = gen_insert_value(operand, cc_op1, 0);
    operand = gen_insert_value(operand, cc_op2, 1);
  } break;
  case IL_SPSP2SCMPLXI0:
    dt = DT_CMPLX;
    cmpnt = DT_FLOAT;
    goto component_zero;
  case IL_DPDP2DCMPLXI0:
    dt = DT_DCMPLX;
    cmpnt = DT_DBLE;
    goto component_zero;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QPQP2QCMPLXI0:
    dt = DT_QCMPLX;
    cmpnt = DT_QUAD;
    goto component_zero;
#endif
  component_zero: /* Set imaginary value to 0 */
    cc_op1 = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(cmpnt));
    cc_op2 = (cmpnt == DT_QUAD) ? make_constval_opL(make_lltype_from_dtype(cmpnt), 0, 0, 0, 0) :
	    make_constval_op(make_lltype_from_dtype(cmpnt), 0, 0);
    operand = make_undef_op(make_lltype_from_dtype(dt));
    operand = gen_insert_value(operand, cc_op1, 0);
    operand = gen_insert_value(operand, cc_op2, 1);
    break;
  case IL_SCMPLXCONJG:
    dt = DT_CMPLX;
    cmpnt = DT_FLOAT;
    goto cmplx_conj;
  case IL_DCMPLXCONJG:
    dt = DT_DCMPLX;
    cmpnt = DT_DBLE;
    goto cmplx_conj;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCMPLXCONJG:
    dt = DT_QCMPLX;
    cmpnt = DT_QUAD;
    goto cmplx_conj;
#endif
  cmplx_conj:
    /* result = {real , 0 - imag} */
    c1 = gen_eval_cmplx_value(ILI_OPND(ilix, 1), dt);
    cse1 = gen_copy_operand(c1);
    cc_op1 = gen_extract_value(c1, dt, cmpnt, 1);
    cc_op2 = make_constval_op(make_lltype_from_dtype(cmpnt), 0, 0);
    cc_op2->next = cc_op1;
    cc_op2 = ad_csed_instr(I_FSUB, 0, make_lltype_from_dtype(cmpnt), cc_op2,
                           InstrListFlagsNull, true);
    cc_op1 = gen_extract_value(cse1, dt, cmpnt, 0);
    operand = make_undef_op(make_lltype_from_dtype(dt));
    operand = gen_insert_value(operand, cc_op1, 0);
    operand = gen_insert_value(operand, cc_op2, 1);
    break;
  case IL_FABS:
    operand = gen_call_llvm_intrinsic(
        "fabs.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_VABS:
  case IL_VFLOOR:
  case IL_VCEIL:
  case IL_VAINT:
  case IL_VSQRT:
    intrinsic_name = vect_llvm_intrinsic_name(ilix);
    intrinsic_type = make_lltype_from_dtype(ili_get_vect_dtype(ilix));
    operand = gen_call_llvm_intrinsic(
        intrinsic_name, gen_llvm_expr(ILI_OPND(ilix, 1), intrinsic_type),
        make_lltype_from_dtype(ili_get_vect_dtype(ilix)), NULL, I_PICALL);
    break;
  case IL_VRSQRT: {
    int vsize;
#if defined(TARGET_LLVM_POWER) || defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    const int arg = ILI_OPND(ilix, 1);
#endif
    dtype = ili_get_vect_dtype(ilix); /* get the vector dtype */
    intrinsic_type = make_lltype_from_dtype(dtype);
    assert(TY_ISVECT(DTY(dtype)), "gen_llvm_expr(): expected vect type",
           DTY(dtype), ERR_Fatal);
    vsize = DTyVecLength(dtype);
#if defined(TARGET_LLVM_POWER)
    operand = gen_llvm_expr(arg, intrinsic_type);
    intrinsic_name = "ppc.vsx.xvrsqrtesp";
#elif defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    if (vsize == 4) {
      operand = gen_llvm_expr(arg, intrinsic_type);
      intrinsic_name = "x86.sse.rsqrt.ps";
    } else if (vsize == 8) {
      operand = gen_llvm_expr(arg, intrinsic_type);
      intrinsic_name = "x86.avx.rsqrt.ps.256";
    } else if (vsize == 16) {
      LL_Type *i16Ty = ll_create_int_type(cpu_llvm_module, 16);
      OPERAND *op3 = gen_llvm_expr(ad_icon(~0), i16Ty);
      OPERAND *op2 = gen_llvm_expr(arg, intrinsic_type);
      operand = gen_copy_op(op2);
      operand->next = op2;
      op2->next = op3;
      intrinsic_name = "x86.avx512.rsqrt14.ps.512";
      // Xeon Phi also supports 28 bit precision
    } else {
      assert(false, "gen_llvm_expr(): unexpected vector size", vsize,
             ERR_Fatal);
    }
#else
    assert(false, "gen_llvm_expr(): unsupported target", vsize, ERR_Fatal);
#endif
    operand = gen_call_llvm_intrinsic(intrinsic_name, operand, intrinsic_type,
                                      NULL, I_PICALL);
  } break;
  case IL_VRCP: {
    int vsize;
#if defined(TARGET_LLVM_POWER) || defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    const int arg = ILI_OPND(ilix, 1);
#endif
    dtype = ili_get_vect_dtype(ilix); /* get the vector dtype */
    intrinsic_type = make_lltype_from_dtype(dtype);
    assert(TY_ISVECT(DTY(dtype)), "gen_llvm_expr(): expected vect type",
           DTY(dtype), ERR_Fatal);
    vsize = DTyVecLength(dtype);
#if defined(TARGET_LLVM_POWER)
    operand = gen_llvm_expr(arg, intrinsic_type);
    intrinsic_name = "ppc.vsx.xvresp";
#elif defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    if (vsize == 4) {
      operand = gen_llvm_expr(arg, intrinsic_type);
      intrinsic_name = "x86.sse.rcp.ps";
    } else if (vsize == 8) {
      operand = gen_llvm_expr(arg, intrinsic_type);
      intrinsic_name = "x86.avx.rcp.ps.256";
    } else if (vsize == 16) {
      LL_Type *i16Ty = ll_create_int_type(cpu_llvm_module, 16);
      OPERAND *op3 = gen_llvm_expr(ad_icon(~0), i16Ty);
      OPERAND *op2 = gen_llvm_expr(arg, intrinsic_type);
      operand = gen_copy_op(op2);
      operand->next = op2;
      op2->next = op3;
      intrinsic_name = "x86.avx512.rcp14.ps.512";
      // Xeon Phi also supports 28 bit precision
    } else {
      assert(false, "gen_llvm_expr(): unexpected vector size", vsize,
             ERR_Fatal);
    }
#else
    assert(false, "gen_llvm_expr(): unsupported target", vsize, ERR_Fatal);
#endif
    operand = gen_call_llvm_intrinsic(intrinsic_name, operand, intrinsic_type,
                                      NULL, I_PICALL);
  } break;
  case IL_VFMA1:
  case IL_VFMA2:
  case IL_VFMA3:
  case IL_VFMA4:
    intrinsic_name = vect_llvm_intrinsic_name(ilix);
    intrinsic_type = make_lltype_from_dtype(ili_get_vect_dtype(ilix));
    args = gen_llvm_expr(ILI_OPND(ilix, 1), intrinsic_type);
    args->next = gen_llvm_expr(ILI_OPND(ilix, 2), intrinsic_type);
    args->next->next = gen_llvm_expr(ILI_OPND(ilix, 3), intrinsic_type);
    operand = gen_call_llvm_intrinsic(intrinsic_name, args, intrinsic_type,
                                      NULL, I_PICALL);
    break;
  case IL_VSINCOS:
    operand = gen_llvm_vsincos_builtin(ilix);
    break;
  case IL_VCOS:
  case IL_VSIN: {
    LL_Type *vecTy = make_lltype_from_dtype(ili_get_vect_dtype(ilix));
    if (ILI_OPC(ILI_OPND(ilix, 1)) == IL_VSINCOS) {
      // overloaded use: this is an extract value operation
      LL_Type *retTy = gen_vsincos_return_type(vecTy);
      OPERAND *op = gen_copy_op(gen_llvm_expr(ILI_OPND(ilix, 1), retTy));
      operand = gen_llvm_select_vsincos(op, vecTy, retTy, (opc == IL_VCOS));
    } else {
      // standard call to vector sin (or vector cos)
      OPERAND *opnd = gen_llvm_expr(ILI_OPND(ilix, 1), vecTy);
      char *name = vect_llvm_intrinsic_name(ilix);
      operand = gen_call_llvm_intrinsic(name, opnd, vecTy, NULL, I_PICALL);
    }
  } break;
  case IL_DABS:
    operand = gen_call_llvm_intrinsic(
        "fabs.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_IABS:
  case IL_KABS:
    operand = gen_abs_expr(ilix);
    break;
  case IL_FFLOOR:
    operand = gen_call_llvm_intrinsic(
        "floor.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DFLOOR:
    operand = gen_call_llvm_intrinsic(
        "floor.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FCEIL:
    operand = gen_call_llvm_intrinsic(
        "ceil.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DCEIL:
    operand = gen_call_llvm_intrinsic(
        "ceil.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_AINT:
    operand = gen_call_llvm_intrinsic(
        "trunc.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DINT:
    operand = gen_call_llvm_intrinsic(
        "trunc.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_IMIN:
  case IL_UIMIN:
  case IL_KMIN:
  case IL_UKMIN:
  case IL_FMIN:
  case IL_DMIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMIN:
#endif
  case IL_IMAX:
  case IL_UIMAX:
  case IL_KMAX:
  case IL_UKMAX:
  case IL_FMAX:
  case IL_DMAX:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QMAX:
#endif
  {
    LL_Type *llTy;
    lhs_ili = ILI_OPND(ilix, 2);
    rhs_ili = ILI_OPND(ilix, 1);
    llTy = make_type_from_opc(opc);
    operand = gen_minmax_expr(ilix, gen_llvm_expr(lhs_ili, llTy),
                              gen_llvm_expr(rhs_ili, llTy));
  } break;
  case IL_VMIN:
  case IL_VMAX: {
    DTYPE vect_dtype = ili_get_vect_dtype(ilix);
    OPERAND *op1, *op2;
    LL_Type *llTy;
    lhs_ili = ILI_OPND(ilix, 2);
    rhs_ili = ILI_OPND(ilix, 1);
    llTy = make_lltype_from_dtype(vect_dtype);
    op1 = gen_llvm_expr(lhs_ili, llTy);
    op2 = gen_llvm_expr(rhs_ili, llTy);
#if defined(TARGET_LLVM_POWER)
    if ((operand = gen_call_vminmax_power_intrinsic(ilix, op1, op2)) == NULL) {
      operand = gen_minmax_expr(ilix, op1, op2);
    }
#elif defined(TARGET_LLVM_ARM) && NEON_ENABLED
    if ((operand = gen_call_vminmax_neon_intrinsic(ilix, op1, op2)) == NULL) {
      operand = gen_minmax_expr(ilix, op1, op2);
    }
#else
    if ((operand = gen_call_vminmax_intrinsic(ilix, op1, op2)) == NULL) {
      operand = gen_minmax_expr(ilix, op1, op2);
    }
#endif
  } break;
  case IL_ISELECT:
  case IL_KSELECT:
  case IL_ASELECT:
  case IL_FSELECT:
  case IL_DSELECT:
  case IL_CSSELECT:
  case IL_CDSELECT:
    operand = gen_select_expr(ilix);
    break;
  case IL_FSQRT:
    operand = gen_call_llvm_intrinsic(
        "sqrt.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DSQRT:
    operand = gen_call_llvm_intrinsic(
        "sqrt.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QSQRT:
    operand = gen_call_llvm_intrinsic(
        "sqrt.f128",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_QUAD)),
        make_lltype_from_dtype(DT_QUAD), NULL, I_PICALL);
    break;
#endif
  case IL_FLOG:
    operand = gen_call_llvm_intrinsic(
        "log.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DLOG:
    operand = gen_call_llvm_intrinsic(
        "log.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FLOG10:
    operand = gen_call_llvm_intrinsic(
        "log10.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DLOG10:
    operand = gen_call_llvm_intrinsic(
        "log10.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FSIN:
    operand = gen_call_llvm_intrinsic(
        "sin.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DSIN:
    operand = gen_call_llvm_intrinsic(
        "sin.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FTAN:
    operand = gen_call_pgocl_intrinsic(
        "tan_f",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_CALL);
    break;
  case IL_DTAN:
    operand = gen_call_pgocl_intrinsic(
        "tan_d",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_CALL);
    break;
  case IL_FPOWF:
    operand =
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT));
    operand->next =
        gen_llvm_expr(ILI_OPND(ilix, 2), make_lltype_from_dtype(DT_FLOAT));
    operand = gen_call_pgocl_intrinsic(
        "pow_f", operand, make_lltype_from_dtype(DT_FLOAT), NULL, I_CALL);
    break;
  case IL_DPOWD:
    operand = gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE));
    operand->next =
        gen_llvm_expr(ILI_OPND(ilix, 2), make_lltype_from_dtype(DT_DBLE));
    operand = gen_call_pgocl_intrinsic(
        "pow_d", operand, make_lltype_from_dtype(DT_DBLE), NULL, I_CALL);
    break;
  case IL_DPOWI:
    // TODO: won't work because our builtins expect args in registers (xxm0 in
    // this case) and the call generated here (with llc) puts the args on the
    // stack
    assert(ILI_ALT(ilix), "gen_llvm_expr: missing ILI_ALT field for DPOWI ili",
           ilix, ERR_Fatal);
    operand = gen_llvm_expr(ilix, make_lltype_from_dtype(DT_DBLE));
    break;
  case IL_FCOS:
    operand = gen_call_llvm_intrinsic(
        "cos.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DCOS:
    operand = gen_call_llvm_intrinsic(
        "cos.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FEXP:
    operand = gen_call_llvm_intrinsic(
        "exp.f32",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
    break;
  case IL_DEXP:
    operand = gen_call_llvm_intrinsic(
        "exp.f64",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_DBLE)),
        make_lltype_from_dtype(DT_DBLE), NULL, I_PICALL);
    break;
  case IL_FAND: {
    /* bitwise logical AND op. operand has floating-point type
       %copnd1 = bitcast float %opnd1 to iX
       %copnd2 = bitcast float %opnd1 to iX
       %crslt = and iX %copnd1, %copnd2
       %result = bitcast iX %crslt to float
    */
    OPERAND *op3, *op4, *op5, *op6;
    INSTR_LIST *instr2, *instr3;
    unsigned bits = BITS_IN_BYTE * size_of(DT_FLOAT);
    LL_Type *iTy = make_int_lltype(bits);
    LL_Type *fltTy = make_lltype_from_dtype(DT_FLOAT);
    OPERAND *op1 = gen_llvm_expr(ILI_OPND(ilix, 1), NULL);
    OPERAND *op2 = make_tmp_op(iTy, make_tmps());
    INSTR_LIST *instr1 = gen_instr(I_BITCAST, op2->tmps, iTy, op1);
    ad_instr(ilix, instr1);
    op3 = gen_llvm_expr(ILI_OPND(ilix, 2), NULL);
    op4 = make_tmp_op(iTy, make_tmps());
    instr2 = gen_instr(I_BITCAST, op4->tmps, iTy, op3);
    ad_instr(ilix, instr2);
    op6 = make_tmp_op(fltTy, make_tmps());
    op2->next = op4;
    op5 = ad_csed_instr(I_AND, 0, iTy, op2, InstrListFlagsNull, false);
    instr3 = gen_instr(I_BITCAST, op6->tmps, fltTy, op5);
    ad_instr(ilix, instr3);
    operand = op6;
  } break;
  case IL_RSQRTSS:
#if defined(TARGET_LLVM_POWER)
    operand = gen_call_llvm_intrinsic(
        "ppc.vsx.xsrsqrtesp",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
#endif
#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    {
      /* intrinsic has type <4 x float> -> <4 x float>, so need to build
         and extract from vectors */
      const char *nm = "x86.sse.rsqrt.ss";
      LL_Type *vTy = make_vtype(DT_FLOAT, 4);
      OPERAND *op1 = gen_scalar_to_vector_no_shuffle(ilix, vTy);
      OPERAND *op2 = gen_call_llvm_intrinsic(nm, op1, vTy, NULL, I_PICALL);
      operand = gen_extract_vector(op2, 0);
    }
#endif
    break;
  case IL_RCPSS:
#if defined(TARGET_LLVM_POWER)
    operand = gen_call_llvm_intrinsic(
        "ppc.vsx.xsresp",
        gen_llvm_expr(ILI_OPND(ilix, 1), make_lltype_from_dtype(DT_FLOAT)),
        make_lltype_from_dtype(DT_FLOAT), NULL, I_PICALL);
#endif
#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_X8664)
    {
      /* intrinsic has type <4 x float> -> <4 x float>, so need to build
         and extract from vectors */
      const char *nm = "x86.sse.rcp.ss";
      LL_Type *vTy = make_vtype(DT_FLOAT, 4);
      OPERAND *op1 = gen_scalar_to_vector(ilix, vTy);
      OPERAND *op2 = gen_call_llvm_intrinsic(nm, op1, vTy, NULL, I_PICALL);
      operand = gen_extract_vector(op2, 0);
    }
#endif
    break;
  case IL_VPERMUTE: {
    OPERAND *op1;
    OPERAND *mask_op;
    LL_Type *vect_lltype, *op_lltype;
    DTYPE vect_dtype = ili_get_vect_dtype(ilix);
    int mask_ili;
    int edtype;
    unsigned long long undef_mask = 0;

    /* LLVM shufflevector instruction has a mask whose selector takes
     * the concatenation of two vectors and numbers the elements as
     * 0,1,2,3,... from left to right.
     */

    vect_lltype = make_lltype_from_dtype(vect_dtype);
    lhs_ili = ILI_OPND(ilix, 1);
    rhs_ili = ILI_OPND(ilix, 2);
    mask_ili = ILI_OPND(ilix, 3);
    op_lltype = make_lltype_from_dtype(ili_get_vect_dtype(lhs_ili));

    op1 = gen_llvm_expr(lhs_ili, op_lltype);
    if (ILI_OPC(rhs_ili) == IL_NULL) /* a don't care, generate an undef */
      op1->next = make_undef_op(op1->ll_type);
    else
      op1->next = gen_llvm_expr(rhs_ili, op_lltype);
    mask_op = gen_llvm_expr(mask_ili, 0);
    edtype = CONVAL1G(mask_op->val.sptr);
    // VPERMUTE mask values of -1 are considered llvm undef.
    int i;
    for(i = mask_op->ll_type->sub_elements - 1; i >= 0; i--) {
      undef_mask <<= 1;
      if(VCON_CONVAL(edtype + i) == -1) {
        undef_mask |= 1;
        mask_op->flags |= OPF_CONTAINS_UNDEF;
      } else if(VCON_CONVAL(edtype + i) < -1) {
        interr("VPERMUTE shuffle mask cannot contain values less than -1.", 
               ilix, ERR_Severe);
      }
    }
    if(mask_op->flags & OPF_CONTAINS_UNDEF) {
      mask_op->val.sptr_undef.sptr = mask_op->val.sptr;
      mask_op->val.sptr_undef.undef_mask = undef_mask;
    }
    op1->next->next = mask_op;
    if (vect_lltype->sub_elements != op1->next->next->ll_type->sub_elements) {
      assert(0,
             "VPERMUTE: result and mask must have the same number of elements.",
             vect_lltype->sub_elements, ERR_Severe);
    }
    operand = ad_csed_instr(I_SHUFFVEC, ilix, vect_lltype, op1,
                            InstrListFlagsNull, true);
  } break;
  case IL_VBLEND: {
    int num_elem;
    OPERAND *op1;
    LL_Type *vect_lltype, *int_type, *select_type, *op1_subtype,
        *compare_ll_type;
    DTYPE vect_dtype = ili_get_vect_dtype(ilix);
    int mask_ili = ILI_OPND(ilix, 1);
    lhs_ili = ILI_OPND(ilix, 2);
    rhs_ili = ILI_OPND(ilix, 3);

    select_type = 0;
    vect_lltype = make_lltype_from_dtype(vect_dtype);
    if (ILI_OPC(mask_ili) == IL_VCMP) {
      op1 = gen_vect_compare_operand(mask_ili);
    } else if (ILI_OPC(mask_ili) == IL_VPERMUTE) {
      /* half size predicate */
      op1 = gen_llvm_expr(mask_ili, 0);
      num_elem = DTyVecLength(vect_dtype);
      int_type = make_int_lltype(1);
      select_type = ll_get_vector_type(int_type, num_elem);
      /* The result of the VPERMUTE will be a shuffle of bit mask values,
         so need to set the type correctly. */
      op1->ll_type = select_type;
    } else {
      num_elem = DTyVecLength(vect_dtype);
      int_type = make_int_lltype(1);
      select_type = ll_get_vector_type(int_type, num_elem);
      op1 = gen_llvm_expr(mask_ili, 0);
      asrt(op1->ll_type->data_type == LL_VECTOR);
      op1_subtype = op1->ll_type->sub_types[0];
      /* because DTYPEs do not support i1 bit masks we need to narrow */
      if (ll_type_int_bits(op1_subtype) > 0) /* int type */
        op1 = convert_int_size(mask_ili, op1, select_type);
      else if (ll_type_is_fp(op1_subtype) > 0) /* fp type */
      {
        INT val[2];
        enum LL_BaseDataType bdt = expected_type->sub_types[0]->data_type;
        OPERAND *opm;
        SPTR vcon1_sptr = SPTR_NULL, constant;
        DTYPE vdt;
        switch (bdt) {
        case LL_FLOAT:
          vdt = get_vector_dtype(DT_FLOAT, num_elem);
          vcon1_sptr = get_vcon_scalar(0xffffffff, vdt);
          break;
        case LL_DOUBLE:
          vdt = get_vector_dtype(DT_DBLE, num_elem);
          val[0] = 0xffffffff;
          val[1] = 0xffffffff;
          constant = getcon(val, DT_DBLE);
          vcon1_sptr = get_vcon_scalar(constant, vdt);
          break;
        default:
          assert(0, "Unexpected basic type for VBLEND mask", bdt, ERR_Fatal);
        }
        opm = make_operand();
        opm->ot_type = OT_CC;
        opm->val.cc = LLCCF_OEQ;
        opm->tmps = make_tmps();
        /* type of the compare is the operands: convert from vect_dtype  */
        compare_ll_type = make_lltype_from_dtype(vect_dtype);
        opm->ll_type = compare_ll_type;
        opm->next = op1;
        opm->next->next =
            gen_llvm_expr(ad1ili(IL_VCON, vcon1_sptr), compare_ll_type);
        op1 = ad_csed_instr(I_FCMP, mask_ili, select_type, opm,
                            InstrListFlagsNull, true);
      } else
        assert(false, "gen_llvm_expr(): bad VCMP type", op1_subtype->data_type,
               ERR_Fatal);
    }
    op1->next = gen_llvm_expr(lhs_ili, vect_lltype);
    op1->next->next = gen_llvm_expr(rhs_ili, vect_lltype);
    operand = ad_csed_instr(I_SELECT, ilix, vect_lltype, op1,
                            InstrListFlagsNull, true);
  } break;
  case IL_VCMP:
    /* VCMP is either to select the value from conditional branch or is
     * part of an argument to a masked intrinsic call.
     */
    operand = gen_vect_compare_operand(ilix);
      expected_type = operand->ll_type; /* turn into bit-vector */
    break;
  case IL_ATOMICRMWI:
  case IL_ATOMICRMWA:
  case IL_ATOMICRMWKR:
    operand = gen_llvm_atomicrmw_expr(ilix);
    break;
  case IL_CMPXCHG_OLDA:
  case IL_CMPXCHG_OLDI:
  case IL_CMPXCHG_OLDKR:
    operand = gen_llvm_cmpxchg_component(ilix, 0);
    break;
  case IL_CMPXCHG_SUCCESS:
    operand = gen_llvm_cmpxchg_component(ilix, 1);
    /* Any widening should do zero-extend, not sign-extend. */
    operand->flags |= OPF_ZEXT;
    break;
  case IL_CMPXCHGA:
  case IL_CMPXCHGI:
  case IL_CMPXCHGKR:
    operand = gen_llvm_cmpxchg(ilix);
    break;
  case IL_FNSIN:
  case IL_DNSIN:
  case IL_FNCOS:
  case IL_DNCOS: {
    DTYPE ety = ((opc == IL_FNSIN) || (opc == IL_FNCOS)) ? DT_FLOAT : DT_DBLE;
    LL_Type *argTy = make_lltype_from_dtype(ety);
    DTYPE dty = ((opc == IL_FNSIN) || (opc == IL_FNCOS)) ? DT_CMPLX : DT_DCMPLX;
    LL_Type *retTy = make_lltype_from_dtype(dty);
    const int isCos = (opc == IL_FNCOS) || (opc == IL_DNCOS);
    operand = gen_copy_op(gen_llvm_expr(ILI_OPND(ilix, 1), retTy));
    operand = gen_llvm_select_sin_or_cos(operand, argTy, retTy, isCos);
  } break;
  case IL_FSINCOS:
  case IL_DSINCOS:
    operand = gen_llvm_sincos_builtin(ilix);
    break;
  default:
    DBGTRACE3("### gen_llvm_expr; ilix %d, unknown opcode: %d(%s)\n", ilix, opc,
              IL_NAME(opc))
    assert(false, "gen_llvm_expr(): unknown opcode", opc, ERR_Fatal);
    break;
  } /* End of switch(opc) */

  assert(operand, "gen_llvm_expr(): missing operand", ilix, ERR_Fatal);
  if (!operand->ll_type) {
    DBGTRACE2("# missing type for operand %p (ilix %d)", operand, ilix)
    assert(false, "gen_llvm_expr(): missing type", ilix, ERR_Fatal);
  }
  {
    OPERAND **csed_operand = get_csed_operand(ilix);
    if (csed_operand != NULL)
      set_csed_operand(csed_operand, operand);
    if (sincos_seen() && (IL_HAS_FENCE(opc) || (IL_TYPE(opc) == ILTY_PROC)))
      sincos_clear_all_args();
  }
  ILI_COUNT(ilix)++;
  if (expected_type) {
    LL_Type *tty1, *tty2;
    ret_match = match_types(expected_type, operand->ll_type);
    switch (ret_match) {
    case MATCH_MEM:
      if ((operand->ll_type->data_type == LL_PTR) &&
          ll_type_int_bits(expected_type)) {
        operand = convert_ptr_to_int(operand, expected_type);
      } else {
        operand = make_bitcast(operand, expected_type);
      }
      break;
    case MATCH_OK:
      if ((operand->ll_type->data_type == LL_VECTOR) &&
          (expected_type->data_type == LL_VECTOR) &&
          (operand->ll_type->sub_types[0] == expected_type->sub_types[0]) &&
          (ll_type_bytes(operand->ll_type) != ll_type_bytes(expected_type))) {
        operand = gen_resized_vect(operand, expected_type->sub_elements, 0);
        break;
      }
      tty1 = expected_type;
      tty2 = operand->ll_type;
      ct = 0;
      while (tty1->data_type == tty2->data_type) {
        if ((tty1->data_type == LL_PTR) || (tty1->data_type == LL_ARRAY)) {
          tty1 = tty1->sub_types[0];
          tty2 = tty2->sub_types[0];
          ct++;
        } else {
          break;
        }
      }
      if (tty1 != tty2) {
        operand = make_bitcast(operand, expected_type);
      }
      break;
    case MATCH_NO:
      /* binop1 points to int of different size than instr_type */
      operand = convert_mismatched_types(operand, expected_type, ilix);
      break;
    default:
      assert(0, "gen_llvm_expr(): bad match type for operand", ret_match,
             ERR_Fatal);
    }
  }

  DBGDUMPLLTYPE("#returned type: ", operand->ll_type);
  DBGTRACEOUT2(" returns operand %p, count %d", operand, ILI_COUNT(ilix));
  setTempMap(ilix, operand);
  return operand;
} /* gen_llvm_expr */

static LLIntegerConditionCodes
convert_to_llvm_intcc(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
  case CC_NOTNE: return LLCC_EQ;
  case CC_NE:
  case CC_NOTEQ: return LLCC_NE;
  case CC_LT:
  case CC_NOTGE: return LLCC_SLT;
  case CC_GE:
  case CC_NOTLT: return LLCC_SGE;
  case CC_LE:
  case CC_NOTGT: return LLCC_SLE;
  case CC_GT:
  case CC_NOTLE: return LLCC_SGT;
  default:
    assert(false, "unknown condition code", cc, ERR_Fatal);
  }
  return LLCC_NONE;
}

static LLIntegerConditionCodes
convert_to_llvm_uintcc(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
  case CC_NOTNE: return LLCC_EQ;
  case CC_NE:
  case CC_NOTEQ: return LLCC_NE;
  case CC_LT:
  case CC_NOTGE: return LLCC_ULT;
  case CC_GE:
  case CC_NOTLT: return LLCC_UGE;
  case CC_LE:
  case CC_NOTGT: return LLCC_ULE;
  case CC_GT:
  case CC_NOTLE: return LLCC_UGT;
  default:
    assert(false, "unknown condition code", cc, ERR_Fatal);
  }
  return LLCC_NONE;
}

static LLFloatingPointConditionCodes
convert_to_llvm_fltcc(CC_RELATION cc)
{
  switch (cc) {
  case CC_EQ:
  case CC_NOTNE: return LLCCF_OEQ; // see section 5.11 of IEEE 754
  case CC_NE:
  case CC_NOTEQ: return LLCCF_UNE;
  case CC_LT:    return LLCCF_OLT;
  case CC_NOTGE: return LLCCF_ULT;
  case CC_GE:    return LLCCF_OGE;
  case CC_NOTLT: return LLCCF_UGE;
  case CC_LE:    return LLCCF_OLE;
  case CC_NOTGT: return LLCCF_ULE;
  case CC_GT:    return LLCCF_OGT;
  case CC_NOTLE: return LLCCF_UGT;
  default:
    assert(false, "unknown condition code", cc, ERR_Fatal);
  }
  return LLCCF_NONE;
}

static OPERAND *
gen_vect_compare_operand(int mask_ili)
{
  int num_elem;
  CC_RELATION incoming_cc_code;
  int lhs_ili, rhs_ili, cmp_type = 0;
  LL_Type *int_type, *instr_type, *compare_ll_type;
  LL_InstrName cmp_inst_name = I_NONE;
  DTYPE vect_dtype, elem_dtype;
  OPERAND *operand, *op1;
  ILI_OP mask_opc = ILI_OPC(mask_ili);

  assert(mask_opc == IL_VCMP,
         "gen_vect_compare_operand(): expected vector compare", mask_opc,
         ERR_Fatal);

  incoming_cc_code = ILI_ccOPND(mask_ili, 1);
  lhs_ili = ILI_OPND(mask_ili, 2);
  rhs_ili = ILI_OPND(mask_ili, 3);
  vect_dtype = ili_get_vect_dtype(mask_ili);
  elem_dtype = DTySeqTyElement(vect_dtype);
  num_elem = DTyVecLength(vect_dtype);

  int_type = make_int_lltype(1);
  instr_type = ll_get_vector_type(int_type, num_elem);
  compare_ll_type = make_lltype_from_dtype(vect_dtype);
  if (DT_ISINT(elem_dtype)) {
    cmp_inst_name = I_ICMP;
    cmp_type = CMP_INT;
    if (DT_ISUNSIGNED(elem_dtype)) {
      cmp_type = cmp_type | CMP_USG;
    }
  } else if (DT_ISREAL(elem_dtype)) {
    cmp_inst_name = I_FCMP;
    cmp_type = CMP_FLT;
  } else {
    assert(false, "gen_vect_compare_operand(): unsupported dtype", elem_dtype,
           ERR_Fatal);
  }
  op1 = make_operand();
  op1->ot_type = OT_CC;
  op1->val.cc = convert_to_llvm_cc(incoming_cc_code, cmp_type);
  op1->tmps = make_tmps();
  /* type of the compare is the operands: use compare_ll_type */
  op1->ll_type = compare_ll_type;
  op1->next = gen_llvm_expr(lhs_ili, compare_ll_type);
  op1->next->next = gen_llvm_expr(rhs_ili, compare_ll_type);
  /* type of the instruction is a bit-vector: use instr_type */
  operand = ad_csed_instr(cmp_inst_name, mask_ili, instr_type, op1,
                          InstrListFlagsNull, true);
  return operand;
} /* gen_vect_compare_operand */

static char *
vect_llvm_intrinsic_name(int ilix)
{
  int type, n, fsize = 0;
  DTYPE dtype;
  ILI_OP opc = ILI_OPC(ilix);
  const char *basename = NULL;
  char *retc;
  assert(IL_VECT(opc), "vect_llvm_intrinsic_name(): not vect ili", ilix,
         ERR_Fatal);
  dtype = ili_get_vect_dtype(ilix);

  assert(DTY(dtype) == TY_VECT, "vect_llvm_intrinsic_name(): not vect dtype",
         DTY(dtype), ERR_Fatal);
  type = DTySeqTyElement(dtype);
  retc = getitem(LLVM_LONGTERM_AREA, 20);
  n = DTyVecLength(dtype);
  switch (opc) {
  case IL_VSQRT:
    basename = "sqrt";
    break;
  case IL_VABS:
    basename = "fabs";
    break;
  case IL_VFLOOR:
    basename = "floor";
    break;
  case IL_VCEIL:
    basename = "ceil";
    break;
  case IL_VAINT:
    basename = "trunc";
    break;
  case IL_VFMA1:
    basename = "fma";
    break;
  case IL_VSIN: /* VSIN here for testing purposes */
    basename = "sin";
    break;
  default:
    assert(0, "vect_llvm_intrinsic_name(): unhandled opc", opc, ERR_Fatal);
  }
  switch (type) {
  case DT_FLOAT:
    fsize = 32;
    break;
  case DT_DBLE:
    fsize = 64;
    break;
  default:
    assert(0, "vect_llvm_intrinsic_name(): unhandled type", type, ERR_Fatal);
  }

  sprintf(retc, "%s.v%df%d", basename, n, fsize);

  return retc;
} /* vect_llvm_intrinsic_name */

/**
   \brief Generate comparison operand. Optionally extending the result.
   \param optext  if this is false, do not extend the result to 32 bits.
 */
static OPERAND *
gen_optext_comp_operand(OPERAND *operand, ILI_OP opc, int lhs_ili, int rhs_ili,
                        int cc_ili, int cc_type, LL_InstrName itype, int optext,
                        int ilix)
{
  LL_Type *expected_type, *op_type;
  INSTR_LIST *Curr_Instr;
  DTYPE dtype;
  int vsize;

  operand->ot_type = OT_TMP;
  operand->tmps = make_tmps();

  operand->ll_type = make_int_lltype(1);
  if (opc == IL_VCMPNEQ) {
    assert(ilix, "gen_optext_comp_operand(): missing ilix", 0, ERR_Fatal);
    dtype = ILI_DTyOPND(ilix, 3);
    vsize = DTyVecLength(dtype);
    op_type = operand->ll_type;
    operand->ll_type = make_vector_lltype(vsize, op_type);
  }

  /* now make the new binary expression */
  Curr_Instr =
      gen_instr(itype, operand->tmps, operand->ll_type, make_operand());
  Curr_Instr->operands->ot_type = OT_CC;
  Curr_Instr->operands->val.cc =
    convert_to_llvm_cc((CC_RELATION)cc_ili, cc_type);
  if (opc == IL_VCMPNEQ)
    Curr_Instr->operands->ll_type = expected_type =
        make_lltype_from_dtype(dtype);
  else
    Curr_Instr->operands->ll_type = expected_type = make_type_from_opc(opc);
  Curr_Instr->operands->next = gen_llvm_expr(lhs_ili, expected_type);
  if (opc == IL_ACMPZ || opc == IL_ACMP) {
    LL_Type *ty0 = Curr_Instr->operands->next->ll_type;
    OPERAND *opTo = Curr_Instr->operands->next;
    opTo->next = gen_base_addr_operand(rhs_ili, ty0);
  } else {
    Curr_Instr->operands->next->next = gen_llvm_expr(rhs_ili, expected_type);
  }
  if (opc == IL_ACMPZ)
    Curr_Instr->operands->next->next->flags |= OPF_NULL_TYPE;

  ad_instr(0, Curr_Instr);
  if (!optext)
    return operand;
  if (XBIT(125, 0x8))
    return zero_extend_int(operand, 32);
  else
    /* Result type is bool which is signed, -1 for true, 0 for false. */
    return sign_extend_int(operand, 32);
}

/*
 * Given an ilix that is either an IL_JMPM or an IL_JMPMK, generate and insert
 * the corresponding switch instruction.
 *
 * Return the switch instruction.
 */
static INSTR_LIST *
gen_switch(int ilix)
{
  int is_64bit = false;
  LL_Type *switch_type;
  INSTR_LIST *instr;
  OPERAND *last_op;
  int switch_sptr;
  int sw_elt;

  switch (ILI_OPC(ilix)) {
  case IL_JMPM:
    is_64bit = false;
    break;
  case IL_JMPMK:
    is_64bit = true;
    break;
  default:
    interr("gen_switch(): Unexpected jump ili", ilix, ERR_Fatal);
  }

  instr = make_instr(I_SW);
  switch_type = make_int_lltype(is_64bit ? 64 : 32);

  /*
   * JMPM  irlnk1 irlnk2 sym1 sym2
   * JMPMK krlnk1 irlnk2 sym1 sym2
   *
   * irlnk1 / krlnk1 is the value being switched on.
   * irlnk2 is the table size.
   * sym1 is the label for the memory table.
   * sym2 is the default label.
   *
   * Produce: switch <expr>, <default> [value, label]+
   */
  instr->operands = last_op = gen_llvm_expr(ILI_OPND(ilix, 1), switch_type);

  switch_sptr = ILI_OPND(ilix, 3);

  /* Add the default case. */
  last_op->next = make_target_op(DEFLABG(switch_sptr));
  last_op = last_op->next;

  /* Get all the switch elements out of the switch_base table. */
  for (sw_elt = SWELG(switch_sptr); sw_elt; sw_elt = switch_base[sw_elt].next) {
    OPERAND *label = make_target_op(switch_base[sw_elt].clabel);
    OPERAND *value;
    if (is_64bit)
      value = make_constsptr_op((SPTR)switch_base[sw_elt].val); // ???
    else
      value = make_constval32_op(switch_base[sw_elt].val);
    /* Remaining switch operands are (value, target) pairs. */
    last_op->next = value;
    value->next = label;
    last_op = label;
  }

  ad_instr(ilix, instr);
  return instr;
}

/**
   \brief Add \p ilix to the CSE list
   \param ilix  The ILI index to be added
   \return true iff \p ilix already appears in the CSE list
 */
static bool
add_to_cselist(int ilix)
{
  CSED_ITEM *csed;

  if (ILI_ALT(ilix))
    ilix = ILI_ALT(ilix);

  DBGTRACE1("#adding to cse list ilix %d", ilix)

  for (csed = csedList; csed; csed = csed->next) {
    if (ilix == csed->ilix) {
      DBGTRACE2("#ilix %d already in cse list, count %d", ilix, ILI_COUNT(ilix))
      return true;
    }
  }
  csed = (CSED_ITEM *)getitem(LLVM_LONGTERM_AREA, sizeof(CSED_ITEM));
  memset(csed, 0, sizeof(CSED_ITEM));
  csed->ilix = ilix;
  csed->next = csedList;
  csedList = csed;
  build_csed_list(ilix);
  return false;
}

static void
set_csed_operand(OPERAND **csed_operand, OPERAND *operand)
{
  if (operand->tmps) {
    OPERAND *new_op;

    DBGTRACE("#set_csed_operand using tmps created for operand")
    if (*csed_operand && (*csed_operand)->tmps &&
        (*csed_operand)->tmps != operand->tmps) {
      DBGTRACE("#tmps are different")
    }
    new_op = make_tmp_op(operand->ll_type, operand->tmps);
    *csed_operand = new_op;
  } else {
    DBGTRACE("#set_csed_operand replace csed operand")
    *csed_operand = operand;
  }
}

static void
clear_csed_list(void)
{
  CSED_ITEM *csed;

  for (csed = csedList; csed; csed = csed->next) {
    ILI_COUNT(csed->ilix) = 0;
    csed->operand = NULL;
  }
}

static void
remove_from_csed_list(int ili)
{
  int i, noprs;
  ILI_OP opc;
  CSED_ITEM *csed;

  opc = ILI_OPC(ili);
  for (csed = csedList; csed; csed = csed->next) {
    if (is_cseili_opcode(ILI_OPC(ili)))
      return;
    if (ili == csed->ilix) {
      DBGTRACE1("#remove_from_csed_list ilix(%d)", ili)
      ILI_COUNT(ili) = 0;
      csed->operand = NULL;
    }
  }

  noprs = ilis[opc].oprs;
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i))
      remove_from_csed_list(ILI_OPND(ili, i));
  }
}

static OPERAND **
get_csed_operand(int ilix)
{
  CSED_ITEM *csed;

  if (ILI_ALT(ilix))
    ilix = ILI_ALT(ilix);
  for (csed = csedList; csed; csed = csed->next) {
    if (ilix == csed->ilix) {
      OPERAND *p = csed->operand;

      if (p != NULL) {
        DBGTRACE3(
            "#get_csed_operand for ilix %d, operand found %p, with type (%s)",
            ilix, p, OTNAMEG(p))
        DBGDUMPLLTYPE("cse'd operand type ", p->ll_type)
      } else {
        DBGTRACE1("#get_csed_operand for ilix %d, operand found is null", ilix);
      }
      return &csed->operand;
    }
  }

  DBGTRACE1("#get_csed_operand for ilix %d not found", ilix)

  return NULL;
}

static void
build_csed_list(int ilix)
{
  int i, noprs;
  ILI_OP opc = ILI_OPC(ilix);

  if (is_cseili_opcode(opc)) {
    int csed_ilix = ILI_OPND(ilix, 1);
    if (ILI_ALT(csed_ilix))
      csed_ilix = ILI_ALT(csed_ilix);
    if (add_to_cselist(csed_ilix))
      return;
  }
  switch (opc) {
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
    if (ILI_OPND(ilix, 2) == 'i')
      add_prescan_complex_list(ILI_OPND(ilix, 1));
    break;
#endif
  default:
    break;
  }

  noprs = ilis[opc].oprs;
  for (i = 1; i <= noprs; ++i) {
    if (IL_ISLINK(opc, i))
      build_csed_list(ILI_OPND(ilix, i));
  }
}

static int
convert_to_llvm_cc(CC_RELATION cc, int cc_type)
{
  int ret_code;
  if (cc_type & CMP_INT) {
    if (cc_type & CMP_USG)
      ret_code = convert_to_llvm_uintcc(cc);
    else 
      ret_code = convert_to_llvm_intcc(cc);
  } else {
    ret_code = convert_to_llvm_fltcc(cc);
  }
  if (IEEE_CMP && fcmp_negate)
    ret_code = fnegcc[ret_code];
  return ret_code;
}

INLINE static bool
check_global_define(GBL_LIST *cgl)
{
  GBL_LIST *gl, *gitem;

  for (gl = recorded_Globals; gl; gl = gl->next) {
    if (gl->sptr > 0 && gl->sptr == cgl->sptr) {
      DBGTRACE1("#sptr %d already in Global list; exiting", gl->sptr)
      return true;
    }
  }
  gitem = (GBL_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(GBL_LIST));
  memset(gitem, 0, sizeof(GBL_LIST));
  gitem->sptr = cgl->sptr;
  gitem->global_def = cgl->global_def;
  gitem->next = recorded_Globals;
  recorded_Globals = gitem;
  return false;
}

static void
add_global_define(GBL_LIST *gitem)
{
  DBGTRACEIN2(": '%s', (sptr %d)", gitem->global_def, gitem->sptr);

  /* make sure the global def for this sptr has not already been added;
   * can occur with -Mipa=inline that multiple versions exist.
   */
  if (!check_global_define(gitem)) {
    if (Globals) {
      llvm_info.last_global->next = gitem;
    } else {
      Globals = gitem;
    }
    llvm_info.last_global = gitem;
    if (flg.debug) {
      if (gitem->sptr && ST_ISVAR(STYPEG(gitem->sptr)) &&
          !CCSYMG(gitem->sptr)) {
        LL_Type *type = make_lltype_from_sptr(gitem->sptr);
        LL_Value *value = ll_create_value_from_type(cpu_llvm_module, type,
                                                    SNAME(gitem->sptr));
        lldbg_emit_global_variable(cpu_llvm_module->debug_info, gitem->sptr, 0,
                                   1, value);
      }
    }
  }

  DBGTRACEOUT("");
} /* add_global_define */

void
update_external_function_declarations(const char *name, char *decl,
                                      unsigned flags)
{
  EXFUNC_LIST *efl;
  char *gname;

  gname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(decl) + 1);
  strcpy(gname, decl);
  efl = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
  memset(efl, 0, sizeof(EXFUNC_LIST));
  efl->func_def = gname;
  efl->flags |= flags;
  add_external_function_declaration(name, efl);
}

/**
   \brief Get an intrinsic function with \p name and \p func_type.
   \param name       The name of the function
   \param func_type  The signature of the function

   Create an external function declaration if necessary, or verify that the
   requested function type matches previous uses.

   Use this to generate calls to LLVM intrinsics or runtime library functions.

   The function name must include the leading '@'. The \p name string will not
   be copied.
 */
static OPERAND *
get_intrinsic(const char *name, LL_Type *func_type, unsigned flags)
{
  hash_data_t old_type = NULL;
  OPERAND *op;

  if (hashmap_lookup(llvm_info.declared_intrinsics, name, &old_type)) {
    assert(old_type == func_type,
           "Intrinsic already declared with different signature", 0, ERR_Fatal);
  } else {
    /* First time we see this intrinsic. */
    char *decl = (char *)getitem(LLVM_LONGTERM_AREA,
                                 strlen(name) + strlen(func_type->str) + 50);
    if (!strncmp(name, "asm ", 4)) {
      /* do nothing - CALL asm() */
    } else {
      sprintf(decl, "declare %s %s(", func_type->sub_types[0]->str, name);
      for (BIGUINT64 i = 1; i < func_type->sub_elements; i++) {
        if (i > 1)
          strcat(decl, ", ");
        strcat(decl, func_type->sub_types[i]->str);
      }
      strcat(decl, ")");
      update_external_function_declarations(name, decl, EXF_INTRINSIC | flags);
      hashmap_insert(llvm_info.declared_intrinsics, name, func_type);
    }
  }

  op = make_operand();
  op->ot_type = OT_CALL;
  op->ll_type = make_ptr_lltype(func_type);
  op->string = name;
  return op;
}

/**
   \brief Prepend the callee to a list of operands for an intrinsic call

   When preparing a call to <code>float @llvm.foo(i32 %a, i32 %b)</code>, pass
   the \c %a, \c %b operands to
   <code>get_intrinsic_call_ops("@llvm.foo", float, a_b_ops);</code>
 */
static OPERAND *
get_intrinsic_call_ops(const char *name, LL_Type *return_type, OPERAND *args,
                       unsigned flags)
{
  LL_Type *func_type = make_function_type_from_args(return_type, args, false);
  OPERAND *op = get_intrinsic(name, func_type, flags);
  op->next = args;
  return op;
}

#ifdef FLANG2_CGMAIN_UNUSED
#define OCTVAL(v) ((v >= 48) && (v <= 55))

static int
decimal_value_from_oct(int c, int b, int a)
{
  int val, vc, vb, va;

  vc = c - 48;
  vb = b - 48;
  va = a - 48;
  val = vc * 64 + vb * 8 + va;
  return val;
} /* decimal value */

/**
   \brief Format a string for LLVM output

   LLVM uses hex ASCII characters in strings in place of escape sequences. So
   process the string here making all needed replacements.
 */
static char *
process_string(char *name, int pad, int string_length)
{
  int i, value, remain, count = 0;
  int len = strlen(name);
  char *new_name = (char *)getitem(LLVM_LONGTERM_AREA, 3 * (len + pad) + 2);

  DBGTRACEIN4(" arg name: %s, pad: %d, len: %d, string_length %d", name, pad,
              len, string_length);

  for (i = 0; i <= len; i++) {
    if (name[i] == 92 && i < len) /* backslash that might be an escape */
    {
      switch (name[i + 1]) {
      case 39: /* \' in string => ' */
        new_name[count++] = name[i + 1];
        i++;
        break;
      case 48: /* look for octal values */
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
        if (i <= len - 2 && OCTVAL(name[i + 2]) && OCTVAL(name[i + 3])) {
          value = decimal_value_from_oct(name[i + 1], name[i + 2], name[i + 3]);
          remain = value % 16;
          value = value / 16;
          new_name[count++] = name[i]; /* copy the \ character */
          if (value < 10)
            new_name[count++] = 48 + value;
          else
            new_name[count++] = 55 + value;
          if (remain < 10)
            new_name[count++] = 48 + remain;
          else
            new_name[count++] = 55 + remain;
          i += 3;
        } else
          new_name[count++] = name[i]; /* copy the \ character */
        break;
      case 97: /* bell character (bel) - \a in string => \07 */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 55;
        i++;
        break;
      case 98:                       /* backspace (bs) - \b in string => \08 */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 56;
        i++;
        break;
      case 116: /* horizontal tab (ht) - \t in string => \09 */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 57;
        i++;
        break;
      case 110:                      /* newline (nl) - \n in string => \0a */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 97;
        i++;
        break;
      case 102:                      /* form feed (np) - \f in string => \0c */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 99;
        i++;
        break;
      case 114: /* carriage return (cr) - \r in string => \0d */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 100;
        i++;
        break;
      case 34:                       /* quote character - \" in string => \22 */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 50;
        new_name[count++] = 50;
        i++;
        break;
      case 92: /* backslash  character - \\ in string => \5C */
        new_name[count++] = name[i]; /* copy the \ character */
        new_name[count++] = 53;
        new_name[count++] = 67;
        i++;
        break;
      default:                       /* don't do anything */
        new_name[count++] = name[i]; /* copy the \ character */
        break;
      }
    } else {
      switch (name[i]) {
      case 10:
        new_name[count++] = 92; /* copy the \ character */
        new_name[count++] = 48;
        new_name[count++] = 97;
        break;
      case 34:
        if (i && i != (len - 1)) {
          new_name[count++] = 92; /* copy the \ character */
          new_name[count++] = '2';
          new_name[count++] = '2';
          break;
        }
      default:
        new_name[count++] = name[i];
      }
    }
  }
  len = strlen(new_name);
  /* add any needed padding */
  for (i = 0; i < pad; i++) {
    new_name[len + (i * 3 - 1)] = 92; /* \ */
    new_name[len + (i * 3)] = 48;     /* 0 */
    new_name[len + (i * 3 + 1)] = 48; /* 0 */
  }

  if (pad) /* if needed, fix up the end of the padding */
  {
    new_name[len + (3 * pad - 1)] = 34; /* " */
    new_name[len + (3 * pad)] = 0;      /* '\0' */
  }

  len = strlen(new_name);
  /* need to have the string end with \00" unless tight
   * character array initialization.
   */
  if (!string_length || len - 2 != string_length) {
    new_name[len - 1] = 92; /* \ */
    new_name[len] = 48;     /* 0 */
    new_name[len + 1] = 48; /* 0 */
    new_name[len + 2] = 34; /* " */
    new_name[len + 3] = 0;  /* '\0' */
  }

  DBGTRACEOUT1(" returns '%s'", new_name)

  return new_name;
} /* process_string */
#endif

/**
    \brief Get string name for a struct type
    \param dtype  dtype index
    \return string containing dtype name
 */
const char *
dtype_struct_name(DTYPE dtype)
{
  const char *dtype_str = process_dtype_struct(dtype);
  return dtype_str;
}

/* Set the LLVM name of a global sptr to '@' + name.
 *
 * This is appropriate for external identifiers and internal identifiers with a
 * module-unique name.
 */
static const char *
set_global_sname(int sptr, const char *name)
{
  name = map_to_llvm_name(name);
  char *buf = (char *)getitem(LLVM_LONGTERM_AREA, strlen(name) + 2);
  sprintf(buf, "@%s", name);
  SNAME(sptr) = buf;
  return SNAME(sptr);
}

/* Set the LLVM name of a global sptr to '@' + name + '.' + sptr.
 *
 * This is appropriate for internal globals that don't have a unique name
 * because they belong to some scope. The sptr suffix makes the name unique.
 */
static const char *
set_numbered_global_sname(int sptr, const char *name)
{
  char *buf = (char *)getitem(LLVM_LONGTERM_AREA, strlen(name) + 12);
  sprintf(buf, "@%s.%d", name, sptr);
  SNAME(sptr) = buf;
  return SNAME(sptr);
}

/* Set the LLVM name of a local sptr to '%' + name.
 *
 * This is appropriate for function-local identifiers.
 */
static const char *
set_local_sname(int sptr, const char *name)
{
  char *buf = (char *)getitem(LLVM_LONGTERM_AREA, strlen(name) + 2);
  sprintf(buf, "%%%s", name);
  SNAME(sptr) = buf;
  return SNAME(sptr);
}

/* Create an LLVM initializer for a global define and record it as
 * gitem->global_def.
 *
 * This will either use the dinit_string() or an appropriate zero-initializer.
 *
 * The flag_str modifies the global variable linkage, visibility, and other
 * flags.
 *
 * The type_str is the name of of global type as returned from char_type().
 *
 */
static void
create_global_initializer(GBL_LIST *gitem, const char *flag_str,
                          const char *type_str)
{
  int dty, stype;
  int sptr = gitem->sptr;
  const char *initializer;
  char *gname;

  assert(sptr, "gitem must be initialized", 0, ERR_Fatal);
  assert(gitem->global_def == NULL, "gitem already has an initializer", sptr,
         ERR_Fatal);
  assert(SNAME(sptr), "sptr must have an LLVM name", sptr, ERR_Fatal);

  /* Create an initializer string. */
  if (DINITG(sptr))
    return;

  dty = DTY(DTYPEG(sptr));
  stype = STYPEG(sptr);

  if (
      (stype == ST_VAR && dty == TY_PTR))
    initializer = "null";
  else if (AGGREGATE_STYPE(stype) || COMPLEX_DTYPE(DTYPEG(sptr)) ||
           VECTOR_DTYPE(DTYPEG(sptr)))
    initializer = "zeroinitializer";
  else if (stype == ST_VAR && TY_ISREAL(dty))
    initializer = "0.0";
  else
    initializer = "0";
  gname = (char *)getitem(LLVM_LONGTERM_AREA,
                          strlen(SNAME(sptr)) + strlen(flag_str) +
                              strlen(type_str) + strlen(initializer) + 8);
  sprintf(gname, "%s = %s %s %s", SNAME(sptr), flag_str, type_str, initializer);
  gitem->global_def = gname;
}

/**
   \brief Check if sptr is the midnum of a scalar and scalar has POINTER/ALLOCATABLE attribute
   \param sptr  A symbol
 */
bool
pointer_scalar_need_debug_info(SPTR sptr)
{
  if ((sptr > NOSYM) && REVMIDLNKG(sptr)) {
    SPTR scalar_sptr = (SPTR)REVMIDLNKG(sptr);
    if ((POINTERG(scalar_sptr) || ALLOCATTRG(scalar_sptr)) &&
        ((STYPEG(scalar_sptr) == ST_VAR) || (STYPEG(scalar_sptr) == ST_STRUCT)))
      return true;
  }
  return false;
}

/**
   \brief Check if sptr is the midnum of an array and the array has descriptor 
   \param sptr  A symbol
 */
bool
ftn_array_need_debug_info(SPTR sptr)
{
  if ((sptr > NOSYM) && REVMIDLNKG(sptr)) {
    SPTR array_sptr = (SPTR)REVMIDLNKG(sptr);
    if (!CCSYMG(array_sptr) && SDSCG(array_sptr))
      return true;
  }
  return false;
}

/**
   \brief Separate symbols that should NOT have debug information
   \param sptr  a symbol
   \return false iff \p sptr ought NOT to have debug info
 */
INLINE static bool
needDebugInfoFilt(SPTR sptr)
{
  if (!sptr)
    return true;
  /* Fortran case needs to be revisited when we start to support debug, for now
   * just the obvious case */
  return (!CCSYMG(sptr) || DCLDG(sptr) ||
          is_procedure_ptr((SPTR)REVMIDLNKG(sptr)) ||
          ftn_array_need_debug_info(sptr));
}
#ifdef OMP_OFFLOAD_LLVM
INLINE static bool
is_ompaccel(SPTR sptr)
{
  return OMPACCDEVSYMG(sptr);
}
#endif
INLINE static bool
generating_debug_info(void)
{
  return flg.debug && cpu_llvm_module->debug_info;
}

/**
   \brief Determine if debug information is needed for a particular symbol
   \param sptr  The symbol

   Checks debug flags and symbol properties.
 */
INLINE static bool
need_debug_info(SPTR sptr)
{
#ifdef OMP_OFFLOAD_LLVM
  if (is_ompaccel(sptr) && ISNVVMCODEGEN)
    return false;
#endif
  return generating_debug_info() && needDebugInfoFilt(sptr);
}

/**
   \brief Fixup debug info types

   The declared type and the type of the symbol may have diverged. We want to
   use the declared type for debug info so the user sees the expected
   representation.
 */
INLINE static LL_Type *
mergeDebugTypesForGlobal(const char **glob, LL_Type *symTy, LL_Type *declTy)
{
  if (symTy != declTy) {
    const size_t strlenSum =
        16 + strlen(symTy->str) + strlen(declTy->str) + strlen(*glob);
    char *buff = (char *)getitem(LLVM_LONGTERM_AREA, strlenSum);
    snprintf(buff, strlenSum, "bitcast (%s %s to %s)", symTy->str, *glob,
             declTy->str);
    *glob = buff;
  }
  return declTy;
}

static void
addDebugForGlobalVar(SPTR sptr, ISZ_T off)
{
  if (need_debug_info(sptr)) {
    LL_Module *mod = cpu_llvm_module;
    /* TODO: defeat unwanted side-effects. make_lltype_from_sptr() will update
       the LLTYPE() type (sptr_type_array) along some paths. This may be
       undesirable at this point, because the array gets updated with an
       unexpected/incorrect type. Work around this buggy behavior by caching and
       restoring the type value.  Figure out why this works the way it does. */
    LL_Type *cache = LLTYPE(sptr);
    LL_Type *sty = make_lltype_from_sptr(sptr);
    LL_Type *dty = ll_get_pointer_type(make_lltype_from_dtype(DTYPEG(sptr)));
    const char *glob = SNAME(sptr);
    LL_Type *vty = mergeDebugTypesForGlobal(&glob, sty, dty);
    LL_Value *val = ll_create_value_from_type(mod, vty, glob);
    lldbg_emit_global_variable(mod->debug_info, sptr, off, 1, val);
    LLTYPE(sptr) = cache;
  }
}

static void
process_cmnblk_data(SPTR sptr, ISZ_T off)
{
  SPTR cmnblk = MIDNUMG(sptr);
  SPTR scope = SCOPEG(cmnblk);

  if (flg.debug && !CCSYMG(cmnblk) && (scope > 0)) {
    const char *name = new_debug_name(SYMNAME(scope), SYMNAME(cmnblk), NULL);
    if (!ll_get_module_debug(cpu_llvm_module->common_debug_map, name))
      lldbg_emit_common_block_mdnode(cpu_llvm_module->debug_info, cmnblk);
  }
}

/**
   \brief process \c SC_STATIC \p sptr representing a file-local variable
   \param sptr  A symbol
   \param off   offset into a structure (should be >= 0)
 */
static void
process_static_sptr(SPTR sptr, ISZ_T off)
{
  const int stype = STYPEG(sptr);

  DEBUG_ASSERT(SCG(sptr) == SC_STATIC, "Expected static variable sptr");
  DEBUG_ASSERT(!SNAME(sptr), "Already processed sptr");

  set_global_sname(sptr, get_llvm_name(sptr));
  sym_is_refd(sptr);

  if ((stype == ST_ENTRY) || (stype == ST_PROC))
    return;
  if ((stype == ST_CONST) || (stype == ST_PARAM))
    return;

  addDebugForGlobalVar(sptr, off);
}

static bool
is_blockaddr_store(int ilix, int rhs, int lhs)
{
  if (ILI_OPC(rhs) == IL_AIMV || ILI_OPC(rhs) == IL_AKMV)
    rhs = ILI_OPND(rhs, 1);

  if (ILI_OPC(rhs) == IL_ACEXT) {
    SPTR gl_sptr;
    int ili, newnme;
    int nme = ILI_OPND(ilix, 3);
    SPTR sptr = basesym_of(nme);
    SPTR label = SymConval1(ILI_SymOPND(rhs, 1));
    process_sptr(label);
    gl_sptr = process_blockaddr_sptr(sptr, label);

    /* MSZ could be 64 if it is 64-bit */
    ili = ad_acon(gl_sptr, 0);
    STYPEP(gl_sptr, ST_VAR);
    newnme = addnme(NT_VAR, gl_sptr, 0, 0);
    ili = ad3ili(IL_LD, ili, newnme, MSZ_WORD);

    ili = ad4ili(IL_ST, ili, lhs, nme, MSZ_WORD);
    make_stmt(STMT_ST, ili, false, SPTR_NULL, 0);

    return true;
  }
  return false;
}

/**
   \brief Process block address symbol

   We want to generate something similar to:
   \verbatim
     @MAIN_iab = internal global  i8*  blockaddress(@MAIN_, %L.LB1_351)

   MAIN_:
      %0 = load i8** @iab2
      store i8* %0, i8** %iab
      ; next instruction is use when branching
      ; indirectbr i8** %iab, [label %the_label]
   \endverbatim
  */
static SPTR
process_blockaddr_sptr(int sptr, int label)
{
  SPTR gl_sptr;
  char *curfnm = getsname(gbl.currsub);
  char *sptrnm = SYMNAME(sptr);
  int size = strlen(curfnm) + strlen(sptrnm);

  DEBUG_ASSERT(size <= MXIDLN, "strcat exceeds available space");
  gl_sptr = getsymbol(strcat(curfnm, sptrnm));
  DTYPEP(gl_sptr, DT_CPTR);
  STYPEP(gl_sptr, ST_VAR);
  SCP(gl_sptr, SC_EXTERN);
  ADDRESSP(gl_sptr, 0);
  CCSYMP(gl_sptr, 1);

  if (SNAME(gl_sptr) == NULL) {
    LL_Type *ttype;
    char *sname, *gname;
    const char *retc;
    const char *labelName;
    GBL_LIST *gitem = (GBL_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(GBL_LIST));
    memset(gitem, 0, sizeof(GBL_LIST));
    gitem->sptr = gl_sptr;

    sname = (char *)getitem(LLVM_LONGTERM_AREA, strlen(SYMNAME(gl_sptr)));
    sprintf(sname, "@%s", SYMNAME(gl_sptr));
    SNAME(gl_sptr) = sname;
    ttype = make_lltype_sz4v3_from_sptr(gl_sptr);
    LLTYPE(gl_sptr) = ttype;

    size = size + 80;

    retc = char_type(DTYPEG(gl_sptr), gl_sptr);
    // FIXME: should use snprintf or check. How do we know +80 is big enough?
    gname = (char *)getitem(LLVM_LONGTERM_AREA, size);
    labelName = get_label_name(label);
    sprintf(gname, "@%s = internal global %s blockaddress(@%s, %%L%s)",
            SYMNAME(gl_sptr), retc, getsname(gbl.currsub), labelName);
    gitem->global_def = gname;
    add_global_define(gitem);
  }

  return gl_sptr;
}

/**
   \brief Process \p sptr and initialize \c SNAME(sptr)
   \param sptr  an external function
 */
static void
process_extern_function_sptr(SPTR sptr)
{
  DTYPE dtype = DTYPEG(sptr);
  DTYPE return_dtype;
  EXFUNC_LIST *exfunc;
  char *gname;
  const char *name;
  const char *extend_prefix;
  LL_Type *ll_ttype;

  assert(SCG(sptr) == SC_EXTERN, "Expected extern sptr", sptr, ERR_Fatal);
  assert(SNAME(sptr) == NULL, "Already processed sptr", sptr, ERR_Fatal);
  assert(STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY,
         "Can only process extern procedures", sptr, ERR_Fatal);

  name = set_global_sname(sptr, get_llvm_name(sptr));

  sym_is_refd(sptr);
  if (CFUNCG(sptr) && STYPEG(sptr) == ST_PROC) {
    DTYPE ttype = DDTG(dtype);
    if (DTY(ttype) == TY_CHAR) {
      ll_ttype = make_ptr_lltype(make_lltype_from_dtype(DT_BINT));
      LLTYPE(sptr) = ll_ttype;
    }
  }
  return_dtype = dtype;

  if (DEFDG(sptr))
    return; /* defined in the file, so no need to declare separately */
  if (INMODULEG(sptr) && INMODULEG(sptr) == INMODULEG(gbl.currsub) &&
      FUNCLINEG(sptr))
    return; /* module subroutine call its module subroutine*/

#if defined ALIASG && defined WEAKG
  /* Don't emit an external reference if the name needs to be defined
   * as a weak alias in write_aliases().
   */
  if (ALIASG(sptr) > NOSYM && WEAKG(sptr))
    return;
#endif

  /* In the case of a function, we want the return type, not the type of
   * the sptr, which we know is "function"
   */
  exfunc = (EXFUNC_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(EXFUNC_LIST));
  memset(exfunc, 0, sizeof(EXFUNC_LIST));
  exfunc->sptr = sptr;
  if (cgmain_init_call(sptr)) {
    gname = (char *)getitem(LLVM_LONGTERM_AREA, 34);
    sprintf(gname, "declare void @__c_bzero(i32, ptr)");
    exfunc->flags |= EXF_INTRINSIC;
  } else {
    const DTYPE dTy =
        get_return_dtype(return_dtype, &(exfunc->flags), EXF_STRUCT_RETURN);
    const char *retc = char_type(dTy, SPTR_NULL);
    const int size = strlen(retc) + strlen(name) + 50;
    gname = getitem(LLVM_LONGTERM_AREA, size);
#ifdef VARARGG
    if (VARARGG(sptr))
      exfunc->flags |= EXF_VARARG;
#endif
    /* do we return a char? If so, must add
     * attribute "zeroext" or "signext"
     */
    switch (DTY(dTy)) {
    default:
      extend_prefix = "";
      break;
    case TY_SINT:
#if defined(TARGET_LLVM_X8664)
      /* Workaround: LLVM on x86 does not sign extend i16 types */
      retc = char_type(DT_INT, SPTR_NULL);
#endif
      FLANG_FALLTHROUGH;
    case TY_BINT:
      extend_prefix = "signext";
      break;
    case TY_USINT:
#if defined(TARGET_LLVM_X8664)
      /* Workaround: LLVM on x86 does not sign extend i16 types */
      retc = char_type(DT_INT, SPTR_NULL);
#endif
      extend_prefix = "zeroext";
      break;
    }
    sprintf(gname, "%s %s %s", extend_prefix, retc, name);
  }
  exfunc->func_def = gname;
  exfunc->use_dtype = DTYPEG(sptr);
  add_external_function_declaration(name, exfunc);
}

INLINE static bool
externVarHasDefinition(SPTR sptr)
{
  return DEFDG(sptr);
}

INLINE static bool
externVarMustInitialize(SPTR sptr)
{
  return true;
}

/**
   \brief Process extern variable \p sptr and initialize <tt>SNAME(sptr)</tt>
   \param sptr  a symbol
   \param off
 */
static void
process_extern_variable_sptr(SPTR sptr, ISZ_T off)
{
  const char *name;
  const char *retc;
  const char *flag_str;
  GBL_LIST *gitem;

  DEBUG_ASSERT(SCG(sptr) == SC_EXTERN, "Expected extern sptr");
  DEBUG_ASSERT(!SNAME(sptr), "Already processed sptr");

  name = set_global_sname(sptr, get_llvm_name(sptr));
  retc = char_type(DTYPEG(sptr), sptr);

  gitem = (GBL_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(GBL_LIST));
  memset(gitem, 0, sizeof(GBL_LIST));
  gitem->sptr = sptr;
  gitem->alignment = align_of_var(sptr);

  /* Add debug information for global variable */
  addDebugForGlobalVar(sptr, off);

  if (!externVarHasDefinition(sptr))
    return;

  flag_str = IS_TLS(sptr) ? "thread_local global" : "global";

  /* Defined as a global, not initialized. */
  if (!DINITG(sptr))
    flag_str = IS_TLS(sptr) ? "common thread_local global" : "common global";

  if (externVarMustInitialize(sptr))
    create_global_initializer(gitem, flag_str, retc);
}

/**
   \brief add debug information for variable \p sptr
   \param sptr  the symbol to be added
   \param type  the type to be used for \p sptr
 */
INLINE static void
addDebugForLocalVar(SPTR sptr, LL_Type *type)
{
  if (need_debug_info(sptr) || pointer_scalar_need_debug_info(sptr)) {
    /* Dummy sptrs are treated as local (see above) */
    if ((ll_feature_debug_info_ver90(&cpu_llvm_module->ir) &&
        ftn_array_need_debug_info(sptr)) &&
        (DTYPEG(REVMIDLNKG(sptr)) != DT_DEFERCHAR)) {
      SPTR array_sptr = (SPTR)REVMIDLNKG(sptr);
      LL_MDRef array_md =
          lldbg_emit_local_variable(cpu_llvm_module->debug_info, array_sptr,
                                    BIH_FINDEX(gbl.entbih), true);
      LL_Type *sd_type = LLTYPE(SDSCG(array_sptr));
      if (sd_type && sd_type->data_type == LL_PTR)
        sd_type = sd_type->sub_types[0];
      insert_llvm_dbg_declare(array_md, SDSCG(array_sptr),
                              sd_type, NULL, OPF_NONE);
    } else {
      LL_MDRef param_md = lldbg_emit_local_variable(
          cpu_llvm_module->debug_info, sptr, BIH_FINDEX(gbl.entbih), true);
      insert_llvm_dbg_declare(param_md, sptr, type, NULL, OPF_NONE);
    }
  }
}

/**
   \brief process an \c SC_LOCAL \p sptr and initialize \c SNAME(sptr)
   \param sptr  a symbol
 */
static void
process_local_sptr(SPTR sptr)
{
  LL_Type *type = NULL;
  assert(SCG(sptr) == SC_LOCAL, "Expected local sptr", sptr, ERR_Fatal);
  assert(SNAME(sptr) == NULL, "Already processed sptr", sptr, ERR_Fatal);

  sym_is_refd(sptr);

  if (REFG(sptr) && DINITG(sptr)) {
    const char *name = get_llvm_name(sptr);
    if (SCOPEG(sptr) == 0) {
      name = set_global_sname(sptr, name);
    } else {
      name = set_numbered_global_sname(sptr, name);
    }
    DBGTRACE2("#variable #%d(%s) is data initialized", sptr, SYMNAME(sptr))
  } else if (DINITG(sptr) || SAVEG(sptr)) {
    const char *name = get_llvm_name(sptr);
    GBL_LIST *gitem = (GBL_LIST *)getitem(LLVM_LONGTERM_AREA, sizeof(GBL_LIST));
    memset(gitem, 0, sizeof(GBL_LIST));
    gitem->sptr = sptr;

    if (SCOPEG(sptr) == 0) {
      name = set_global_sname(sptr, name);
    } else {
      name = set_numbered_global_sname(sptr, name);
    }

    DBGTRACE2("#variable #%d(%s) is data initialized", sptr, SYMNAME(sptr));
    create_global_initializer(gitem, "internal global",
                              char_type(DTYPEG(sptr), sptr));
    add_global_define(gitem);
  } else if (SOCPTRG(sptr)) {
    SNAME(sptr) = get_local_overlap_var();
  } else {
    /* This is an actual local variable. Create an alloca. */
    LL_Object *local;
    type = LLTYPE(sptr);

    /* make_lltype_from_sptr() should have added a pointer to the type of
     * this local variable. Remove it */
    CHECK(type->data_type == LL_PTR);
    type = type->sub_types[0];

    /* Now create the alloca for this variable.
     * FIXME: Apparently, the AG table is keeping track of local symbols by
     * name, but we have no guarantee that locval names are unique. This
     * will end in tears. */
    local =
        ll_create_local_object(llvm_info.curr_func, type, align_of_var(sptr),
                               "%s", get_llvm_name(sptr));
    SNAME(sptr) = local->address.data;
  }

  addDebugForLocalVar(sptr, type);
}

static void
gen_name_private_sptr(SPTR sptr)
{
  /* This is an actual local variable. Create an alloca. */
  LL_Type *type = LLTYPE(sptr);
  LL_Object *local;

  /* make_lltype_from_sptr() should have added a pointer to the type of
   * this local variable. Remove it */
  CHECK(type->data_type == LL_PTR);
  type = type->sub_types[0];

  /* Now create the alloca for this variable.
   * FIXME: Apparently, the AG table is keeping track of local symbols by
   * name, but we have no guarantee that locval names are unique. This
   * will end in tears.
   */
  local = ll_create_local_object(llvm_info.curr_func, type, align_of_var(sptr),
                                 "%s", get_llvm_name(sptr));
  SNAME(sptr) = local->address.data;
  addDebugForLocalVar(sptr, type);
}
/* May need to be revisited */
static void
process_private_sptr(SPTR sptr)
{
  if (!gbl.outlined && !TASKG(sptr) && !ISTASKDUPG(GBL_CURRFUNC))
    return;

  assert(SCG(sptr) == SC_PRIVATE, "Expected local sptr", sptr, ERR_Fatal);
  assert(SNAME(sptr) == NULL, "Already processed sptr", sptr, ERR_Fatal);

  /* TODO: Check enclfuncg's scope and if its is not the same as the
   * scope level for -g, then return early, this is not a private sptr
   */
  sym_is_refd(sptr);

  gen_name_private_sptr(sptr);
}

/*
 * if compiling CPU code, return nonzero if this procedure is attributes(global)
 * if compiling GPU code, return nonzero if this proceudre is attributes(global)
 *   OR attributes(device)
 * in particular, when compiling for CPU, return zero for attributes(device,host),
 *   because we're generating the host code
 */
INLINE static int
compilingGlobalOrDevice()
{
  int cudag = CUDA_GLOBAL;
  if (!CG_cpu_compile)
    cudag |= CUDA_DEVICE;
  return CUDAG(gbl.currsub) & cudag;
}

/**
   \brief Does this arg's pointer type really need to be dereferenced?
   \param sptr   The argument
   \return true iff this argument should NOT use the pointer's base type
 */
INLINE static bool
processAutoSptr_skip(SPTR sptr)
{
  if (compilingGlobalOrDevice() && DEVICEG(sptr) && !PASSBYVALG(sptr)) {
    return (SCG(sptr) == SC_DUMMY) ||
           ((SCG(sptr) == SC_BASED) && (SCG(MIDNUMG(sptr)) == SC_DUMMY));
  }
  return false;
}

INLINE static LL_Type *
fixup_argument_type(SPTR sptr, LL_Type *type)
{
  if (processAutoSptr_skip(sptr))
    return type;
  /* type = pointer base type */
  return type->sub_types[0];
}

/**
   \brief Process an \c SC_AUTO or \c SC_REGISTER \p sptr
   \param sptr  A symbol
   Also initialize <tt>SNAME(sptr)</tt>.
 */
static void
process_auto_sptr(SPTR sptr)
{
  LL_Type *type = LLTYPE(sptr);
  LL_Object *local;

  /* Accept SC_DUMMY sptrs if they are arguments that have been given local
   * variable storage. */
  if (SCG(sptr) == SC_DUMMY) {
    assert(hashmap_lookup(llvm_info.homed_args, INT2HKEY(sptr), NULL),
           "Expected coerced dummy sptr", sptr, ERR_Fatal);
  } else {
  }
  assert(SNAME(sptr) == NULL, "Already processed sptr", sptr, ERR_Fatal);

  /* The hidden return argument is created as an SC_AUTO sptr containing the
   * pointer, but it does not need a local entry if we're actually going to
   * emit an LLVM IR sret argument which is just a constant pointer.
   */
  if (ret_info.emit_sret && is_special_return_symbol(sptr)) {
    SNAME(sptr) = ll_create_local_name(llvm_info.curr_func, "sretaddr");
    return;
  }

  /* make_lltype_from_sptr() should have added a pointer to the type of this
   * local variable. Remove it */
  CHECK(type->data_type == LL_PTR);
  type = fixup_argument_type(sptr, type);

  /* Now create the alloca for this variable. Since the alloca produces the
   * address of the local, name it "%foo.addr". */
  local = ll_create_local_object(llvm_info.curr_func, type, align_of_var(sptr),
                                 "%s.addr", SYMNAME(sptr));
  SNAME(sptr) = local->address.data;

  addDebugForLocalVar(sptr, type);
}

static void
process_label_sptr_c(SPTR sptr)
{
  const char *name = get_llvm_name(sptr);
  char *buf = (char *)getitem(LLVM_LONGTERM_AREA, strlen(name) + 1);
  strcpy(buf, name);
  SNAME(sptr) = buf;
}

/**
   \brief Process an <tt>SC_NONE</tt> \p sptr
   \param sptr represents a label
   Also initialize <tt>SNAME(sptr)</tt>.
 */
static void
process_label_sptr(SPTR sptr)
{
  assert(SCG(sptr) == SC_NONE, "Expected label sptr", sptr, ERR_Fatal);
  assert(SNAME(sptr) == NULL, "Already processed sptr", sptr, ERR_Fatal);

  switch (STYPEG(sptr)) {
  case ST_CONST:
    /* TODO: Move this sooner, into the bridge */
    sym_is_refd(sptr);
    return;
  case ST_MEMBER:
    return;
  default:
    break;
  }
  process_label_sptr_c(sptr);
}

static void
process_sptr_offset(SPTR sptr, ISZ_T off)
{
  SC_KIND sc;
  int midnum;
  LL_Type *ttype;

  sym_is_refd(sptr);
  update_llvm_sym_arrays();
  sc = SCG(sptr);

  if (SNAME(sptr))
    return;

  DBGTRACEIN7(" sptr %d = '%s' (%s) SNAME(%d)=%p, sc %d, ADDRTKNG(%d)", sptr,
              getprint(sptr), stb.scnames[sc], sptr, SNAME(sptr), sc,
              ADDRTKNG(sptr));

  ttype = make_lltype_sz4v3_from_sptr(sptr);
  LLTYPE(sptr) = ttype;

  switch (sc) {
  case SC_CMBLK:
    process_cmnblk_data(sptr, off);
    set_global_sname(sptr, get_llvm_name(sptr));
    break;
  case SC_STATIC:
    process_static_sptr(sptr, off);
    break;

  case SC_EXTERN:
    if (
        STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY
    ) {
      process_extern_function_sptr(sptr);
    } else {
      process_extern_variable_sptr(sptr, off);
    }
    break;

  case SC_DUMMY:
    midnum = MIDNUMG(sptr);
    if (DTYPEG(sptr) == DT_ADDR && midnum &&
        hashmap_lookup(llvm_info.homed_args, INT2HKEY(midnum), NULL)) {
      LLTYPE(sptr) = LLTYPE(midnum);
      SNAME(sptr) = SNAME(midnum);
      return;
    }
    if (hashmap_lookup(llvm_info.homed_args, INT2HKEY(sptr), NULL)) {
      process_auto_sptr(sptr);
    } else {
      set_local_sname(sptr, get_llvm_name(sptr));
    }
    if ((flg.smp || (XBIT(34, 0x200) || gbl.usekmpc)) &&
        (gbl.outlined || ISTASKDUPG(GBL_CURRFUNC))) {
      if (sptr == ll_get_shared_arg(gbl.currsub)) {
        LLTYPE(sptr) = make_ptr_lltype(make_lltype_from_dtype(DT_INT8));
      }
    }
    DBGTRACE1("#dummy argument: %s", SNAME(sptr));
    break;

  case SC_LOCAL:
    process_local_sptr(sptr);
    break;

  case SC_BASED:
    if (compilingGlobalOrDevice() && DEVICEG(sptr)) {
      if (hashmap_lookup(llvm_info.homed_args, INT2HKEY(MIDNUMG(sptr)), NULL)) {
        process_auto_sptr(sptr);
        LLTYPE(MIDNUMG(sptr)) = LLTYPE(sptr);
        SNAME(MIDNUMG(sptr)) = SNAME(sptr);
      } else {
        set_local_sname(sptr, get_llvm_name(sptr));
      }
    } else
      set_local_sname(sptr, get_llvm_name(sptr));
    DBGTRACE1("#dummy argument: %s", SNAME(sptr));
    break;

  case SC_NONE: /* should be a label */
    process_label_sptr(sptr);
    break;

#ifdef SC_PRIVATE
  case SC_PRIVATE: /* OpenMP */
    process_private_sptr(sptr);
    if(!SNAME(sptr)) {
      gen_name_private_sptr(sptr);
    }
    break;
#endif
  }

  DBGTRACEOUT("")
}

/**
   \brief Computes byte offset into aggregate structure of \p sptr
   \param sptr  the symbol
   \param idx   additional addend to offset
   \return an offset into a memory object or 0

   NB: sym_is_refd() must be called prior to this function in order to return
   the correct result.
 */
static ISZ_T
variable_offset_in_aggregate(SPTR sptr, ISZ_T idx)
{
  if (ADDRESSG(sptr) && (SCG(sptr) != SC_DUMMY) && (SCG(sptr) != SC_LOCAL)) {
    /* expect:
          int2                           int
          sptr:301  dtype:6  nmptr:2848  sc:4=CMBLK  stype:6=variable
          symlk:1=NOSYM
          address:8  enclfunc:295=mymod
          midnum:302=_mymod$0

       This can be found in a common block.  Don't add address on stack for
       local/dummy arguments */
    idx += ADDRESSG(sptr);
  } else if ((SCG(sptr) == SC_LOCAL) && SOCPTRG(sptr)) {
    idx += get_socptr_offset(sptr);
  }
  return idx;
}

void
process_sptr(SPTR sptr)
{
  process_sptr_offset(sptr, variable_offset_in_aggregate(sptr, 0));
}

#ifdef FLANG2_CGMAIN_UNUSED
/* ipa sometimes makes additional symbol entries for external variables (I have
 * noticed this mainly on globally-defined anonymous structures). However, since
 * LLVM requires all references to be declared within the file that they are
 * referenced, this may result in multiple declararions of the same symbol. Does
 * not work with the opt and llc tools of LLVM.  Thus we try to track down, if
 * possible, the original symbol of storage class extern. Follow the links as
 * far as possible.
 */
static int
follow_sptr_hashlk(SPTR sptr)
{
  char *hash_name, *name = get_llvm_name(sptr);
  int ret_val = 0;
  int hashlk = HASHLKG(sptr);
  while (hashlk > 0) {
    hash_name = get_llvm_name((SPTR)hashlk);
    if (SCG(hashlk) == SC_EXTERN && !strcmp(name, hash_name))
      ret_val = hashlk;
    ret_val = hashlk;
    hashlk = HASHLKG(hashlk);
  }
  return ret_val;
}

static DTYPE
follow_ptr_dtype(DTYPE dtype)
{
  DTYPE dty = dtype;
  while (DTY(dty) == TY_PTR)
    dty = DTySeqTyElement(dty);
  return dty;
}
#endif

bool
strict_match(LL_Type *ty1, LL_Type *ty2)
{
  return (ty1 == ty2);
}

/**
   \brief Does \c ty2 "match" \c ty1 or can \c ty2 be converted to \c ty1?
   \param ty1  the result type
   \param ty2  a type

   If \c ty1 is a ptr and \c ty2 is not, we have an error.  In general, if the
   nesting level of \c ty1 is greater than that of \c ty2, then we have an
   error.

   NB: The original algorithm did \e NOT enforce the latter condition above. The
   old algorithm would peel off all outer levels of array types blindly and
   until a non-array element type is found. This implied that this function
   would return \c MATCH_OK when the input types are \c i32 and <code>[A x [B x
   [C x [D x i32]]]]</code>.
 */
static MATCH_Kind
match_types(LL_Type *ty1, LL_Type *ty2)
{
  MATCH_Kind ret_type;
  int ct1, ct2;
  LL_Type *llt1, *llt2;

  assert(ty1 && ty2, "match_types(): missing argument", 0, ERR_Fatal);

  DBGTRACEIN2("match_types: ty1=%s, ty2=%s\n", ty1->str, ty2->str);
  if (ty1 == ty2)
    return MATCH_OK;

  if (ty1->data_type == LL_ARRAY) {
    LL_Type *ele1 = ll_type_array_elety(ty1);
    LL_Type *ele2 = ll_type_array_elety(ty2);
    return ele2 ? match_types(ele1, ele2) : MATCH_NO;
  }

  if ((ty1->data_type == LL_PTR) || (ty2->data_type == LL_PTR)) {
    /* at least one pointer type */
    if (ty2->data_type != LL_PTR) {
      /* reject as only ty1 is a ptr */
      ret_type = MATCH_NO;
    } else {
      /* get the depth of each pointer type */
      ct1 = 0;
      llt1 = ty1;
      while (llt1->data_type == LL_PTR) {
        asrt(llt1);
        ct1++;
        llt1 = llt1->sub_types[0];
      }
      ct2 = 0;
      llt2 = ty2;
      while (llt2->data_type == LL_PTR) {
        asrt(llt2);
        ct2++;
        llt2 = llt2->sub_types[0];
      }
      if (ct1 > ct2) {
        ret_type = MATCH_NO;
      } else if (match_types(llt1, llt2) == MATCH_OK) {
        if (ct1 == ct2)
          ret_type = MATCH_OK; // ptrs have same level of indirection only
        else if (ct1 + 1 == ct2)
          ret_type = MATCH_MEM;
        else
          ret_type = MATCH_NO;
      } else if ((llt1->data_type == LL_VOID) || (llt2->data_type == LL_VOID)) {
        // one or the other is ptr-to-void; implies, void* == T*****
        ret_type = MATCH_OK;
      } else {
        ret_type = MATCH_NO;
      }
    }
  } else if (ty1->data_type == ty2->data_type) {
    if (ty1->data_type == LL_STRUCT) {
      ret_type = ((ty1 == ty2) ? MATCH_OK : MATCH_NO);
    } else if (ty1->data_type == LL_FUNCTION) {
      /* ??? used to check if both FUNC types were "old-style" or not.
         "Old-style" meant that (DTY(dtype) == TY_FUNC). Why would it matter?
         This doesn't otherwise check the signature. */
      ret_type = MATCH_OK;
    } else {
      ret_type = MATCH_OK;
    }
  } else {
    ret_type = MATCH_NO;
  }

  if (ll_type_int_bits(ty1)) {
    DBGTRACEOUT4(" returns %d(%s) ty1 = %s%d", ret_type, match_names(ret_type),
                 ty1->str, (int)(ll_type_bytes(ty1) * BITS_IN_BYTE))
  } else {
    DBGTRACEOUT3(" returns %d(%s) ty1 = %s", ret_type, match_names(ret_type),
                 ty1->str)
  }
  return ret_type;
} /* match_types */

int
match_llvm_types(LL_Type *ty1, LL_Type *ty2)
{
  return match_types(ty1, ty2);
}

static LL_Type *
make_type_from_opc(ILI_OP opc)
{
  LL_Type *llt;

  DBGTRACEIN1(" (%s)", IL_NAME(opc))
  /* these opcodes will come from conversion operations and expression
   * evaluation without a store, such as:
   *           if( j << 2 )
   *           if (j - (float)3.0)
   * the other possibility is jump ILI with expressions, or cast due
   * to array manipulations. These are mostly
   * of integer type, as the the evaluation of a condition is inherently
   * integral. However, notice first two cases, which are of type LLT_PTR.
   */
  switch (opc) {
  case IL_ACMP:
  case IL_ACMPZ:
  case IL_ACJMP:
  case IL_ACJMPZ:
  case IL_ASELECT:
    llt = make_lltype_from_dtype(DT_CPTR);
    break;
  case IL_ICJMP:
  case IL_UICJMP:
  case IL_FIX:
  case IL_AND:
  case IL_OR:
  case IL_XOR:
  case IL_NOT:
  case IL_MOD:
  case IL_MODZ:
  case IL_LSHIFT:
  case IL_RSHIFT:
  case IL_ARSHIFT:
  case IL_ICON:
  case IL_ICJMPZ:
  case IL_UICJMPZ:
  case IL_UICMPZ:
  case IL_IADD:
  case IL_ISUB:
  case IL_IMUL:
  case IL_IDIVZ:
  case IL_IDIV:
  case IL_UIADD:
  case IL_UISUB:
  case IL_UIMUL:
  case IL_UIDIV:
  case IL_INEG:
  case IL_UINEG:
  case IL_DFIX:
  case IL_DFIXU:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFIX:
#endif
  case IL_ICMP:
  case IL_ICMPZ:
  case IL_ISELECT:
  case IL_IMIN:
  case IL_IMAX:
  case IL_UIMIN:
  case IL_UIMAX:
  case IL_IABS:
  case IL_CMPXCHG_OLDI:
    llt = make_lltype_from_dtype(DT_INT);
    break;
  case IL_KAND:
  case IL_KLSHIFT:
  case IL_KCJMP:
  case IL_KCON:
  case IL_KADD:
  case IL_KSUB:
  case IL_KMUL:
  case IL_KDIV:
  case IL_KDIVZ:
  case IL_KNOT:
  case IL_KCJMPZ:
  case IL_KOR:
  case IL_FIXK:
  case IL_KXOR:
  case IL_KMOD:
  case IL_KARSHIFT:
  case IL_KNEG:
  case IL_KCMP:
  case IL_KCMPZ:
  case IL_UKMIN:
  case IL_UKMAX:
  case IL_KMIN:
  case IL_KMAX:
  case IL_KSELECT:
  case IL_KABS:
  case IL_CMPXCHG_OLDKR:
    llt = make_lltype_from_dtype(DT_INT8);
    break;
  case IL_KUMOD:
  case IL_KUMODZ:
  case IL_UKDIV:
  case IL_UKDIVZ:
  case IL_FIXUK:
  case IL_KURSHIFT:
  case IL_UKCMP:
  case IL_DFIXK:
  case IL_DFIXUK:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QFIXK:
#endif
  case IL_UKCJMP:
  case IL_UKADD:
  case IL_UKSUB:
  case IL_UKMUL:
  case IL_UKCJMPZ:
  case IL_UKCMPZ:
  case IL_UKNEG:
  case IL_UKNOT:
    llt = make_lltype_from_dtype(DT_UINT8);
    break;
  case IL_UNOT:
  case IL_UFIX:
  case IL_UIMOD:
  case IL_UIMODZ:
  case IL_UIDIVZ:
  case IL_ULSHIFT:
  case IL_URSHIFT:
  case IL_UICMP:
    llt = make_lltype_from_dtype(DT_UINT);
    break;
  case IL_FLOAT:
  case IL_FLOATU:
  case IL_FLOATK:
  case IL_FLOATUK:
  case IL_FMOD:
  case IL_SNGL:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_SNGQ:
#endif
  case IL_FSUB:
  case IL_FMUL:
  case IL_FDIV:
  case IL_FADD:
  case IL_FCON:
  case IL_FNEG:
  case IL_FCJMP:
  case IL_FCJMPZ:
  case IL_FCMP:
  case IL_CMPNEQSS:
  case IL_FMIN:
  case IL_FMAX:
  case IL_FABS:
  case IL_FSELECT:
    llt = make_lltype_from_dtype(DT_FLOAT);
    break;
  case IL_DCJMP:
  case IL_DCJMPZ:
  case IL_DFLOAT:
  case IL_DFLOATU:
  case IL_DFLOATK:
  case IL_DFLOATUK:
  case IL_DMOD:
  case IL_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DBLEQ:
#endif
  case IL_DADD:
  case IL_DSUB:
  case IL_DNEG:
  case IL_DMAX:
  case IL_DMIN:
  case IL_DMUL:
  case IL_DDIV:
  case IL_DCON:
  case IL_DCMP:
  case IL_DSELECT:
  case IL_DABS:
    llt = make_lltype_from_dtype(DT_DBLE);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_RQUAD:
  case IL_DQUAD:
  case IL_QFLOATK:
  case IL_QFLOAT:
  case IL_QCJMP:
  case IL_QCJMPZ:
  case IL_QADD:
  case IL_QSUB:
  case IL_QMUL:
  case IL_QDIV:
  case IL_QMAX:
  case IL_QMIN:
  case IL_QNEG:
  case IL_QCMP:
    llt = make_lltype_from_dtype(DT_QUAD);
    break;
#endif
  case IL_CSSELECT:
  case IL_SCMPLXADD:
    llt = make_lltype_from_dtype(DT_CMPLX);
    break;
  case IL_CDSELECT:
  case IL_DCMPLXADD:
    llt = make_lltype_from_dtype(DT_DCMPLX);
    break;
  case IL_ALLOC:
    llt = make_lltype_from_dtype(DT_CPTR);
    break;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128CON:
  case IL_FLOAT128ABS:
  case IL_FLOAT128CHS:
  case IL_FLOAT128RNDINT:
  case IL_FLOAT128FROM:
  case IL_FLOAT128ADD:
  case IL_FLOAT128SUB:
  case IL_FLOAT128MUL:
  case IL_FLOAT128DIV:
  case IL_FLOAT128CMP:
    llt = make_lltype_from_dtype(DT_FLOAT128);
    break;
  case IL_FLOAT128TO:
    llt = make_lltype_from_dtype(DT_DBLE);
    break;
#endif
  default:
    DBGTRACE2("###make_type_from_opc(): unknown opc %d(%s)", opc, IL_NAME(opc))
    assert(0, "make_type_from_opc: unknown opc", opc, ERR_Fatal);
    llt = NULL;
  }

  DBGTRACEOUT1(" returns %p", llt)
  return llt;
} /* make_type_from_opc */

static LL_Type *
make_type_from_msz_with_addrspace(MSZ msz, int addrspace)
{
  return make_lltype_from_dtype_with_addrspace(msz_dtype(msz), addrspace);
} /* make_type_from_msz_with_addrspace */

static LL_Type *
make_type_from_msz(MSZ msz)
{
  return make_type_from_msz_with_addrspace(msz, LL_AddrSp_Default);
} /* make_type_from_msz */

static LL_Type *
make_vtype(DTYPE dtype, int sz)
{
  DTYPE vect_dtype;
  vect_dtype = get_vector_dtype(dtype, sz);
  return make_lltype_from_dtype(vect_dtype);
} /* make_vtype */

static int
is_special_return_symbol(int sptr)
{
  return ret_info.sret_sptr == sptr;
}

int
need_ptr(int sptr, int sc, DTYPE sdtype)
{
  if (is_special_return_symbol(sptr))
    return DTY(sdtype) != TY_PTR;

  switch (sc) {
  case SC_EXTERN:
    return true;

  case SC_STATIC:
    return !DINITG(sptr) || !SAVEG(sptr);

#ifdef SC_PRIVATE
  case SC_PRIVATE:
    return true;
#endif
  case SC_LOCAL:
  case SC_CMBLK:
    return true;

  case SC_DUMMY:
    /* process_formal_arguments() homes all dummies. */
    return true;
  }

  if (sptr)
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
      return true;
    case ST_MEMBER:
      if (DTY(sdtype) == TY_ARRAY)
        return true;
      break;
    default:
      break;
    }

  return false;
}

static OPERAND *
gen_sptr(SPTR sptr)
{
  SC_KIND sc;
  OPERAND *sptr_operand;

  DBGTRACEIN2(" sptr %d (%s)", sptr, SYMNAME(sptr))

  sptr_operand = make_operand();
  sc = SCG(sptr);
  process_sptr(sptr);
  sptr_operand->ll_type = LLTYPE(sptr);
  switch (sc) {
  case SC_CMBLK:
  case SC_DUMMY:
#ifdef SC_PRIVATE
  case SC_PRIVATE:
#endif
  case SC_STATIC:
  case SC_EXTERN:
  case SC_AUTO:
    DBGTRACE2("#using this name for %s; %s", SYMNAME(sptr), SNAME(sptr))

    sptr_operand->ot_type = OT_VAR;
    sptr_operand->val.sptr = sptr;
    sptr_operand->string = SNAME(sptr);
    break;
  case SC_NONE:
    /* For some constants, we need its address to pass to our runtime.
       They also need to be initialized.
     */
    if (STYPEG(sptr) == ST_CONST) {
      sptr_operand->ot_type = OT_VAR;
      sptr_operand->val.sptr = sptr;
      sptr_operand->string = SNAME(sptr);
      break;
    }
    /* TBD */
    FLANG_FALLTHROUGH;
  case SC_BASED:
    assert(0, "gen_sptr(): unexpected storage type", sc, ERR_Fatal);
  }
#ifdef OMP_OFFLOAD_LLVM
#endif
  DBGTRACEOUT1(" returns operand %p", sptr_operand)
  return sptr_operand;
} /* gen_sptr */

/**
   \brief Generate an address expression as an operand
   \param addr_op	ILI of address expression
   \param nme		NME value
   \param lda		is this an IL_LDA?
   \param llt_expected  expected LL_Type
   \param msz		memory access size
 */
OPERAND *
gen_address_operand(int addr_op, int nme, bool lda, LL_Type *llt_expected,
                    MSZ msz)
{
  OPERAND *operand;
  OPERAND **csed_operand;
  LL_Type *llt = llt_expected;
  SPTR sptr = basesym_of(nme);
  unsigned savedAddressSize = addressElementSize;
  int addrspace = LL_AddrSp_Default;
  DBGTRACEIN2(" for ilix: %d(%s)", addr_op, IL_NAME(ILI_OPC(addr_op)))
  DBGDUMPLLTYPE("expected type ", llt_expected)

#ifdef OMP_OFFLOAD_LLVM
  addrspace = OMPACCSHMEMG(sptr) ? LL_AddrSp_NVVM_Shared : LL_AddrSp_NVVM_Generic;
#endif

  if (!llt && !lda && (((int)msz) >= 0)) {
    llt = make_ptr_lltype(make_type_from_msz_with_addrspace(msz, addrspace));
  }
  sptr = basesym_of(nme);

  if (llt) {
    /* do nothing */
  } else {
    if (sptr <= 0) {
      int sub_add = addr_op, sub_opc;
      llt = make_lltype_from_dtype(DT_CPTR);
      while (sub_add) {
        sub_opc = ILI_OPC(sub_add);
        switch (sub_opc) {
        case IL_AADD:
        case IL_ASUB:
          sub_add = ILI_OPND(sub_add, 1);
          continue;
        case IL_DFRAR:
        case IL_ACON:
          llt = make_ptr_lltype(llt);
          sub_add = 0;
          break;
        case IL_LDA:
          llt = make_ptr_lltype(llt);
          sub_add = ILI_OPND(sub_add, 1);
          break;
        default:
          sub_add = 0;
        }
      }
    } else {
      llt = make_lltype_from_sptr(sptr);
      DBGTRACE4("#lda of sptr '%s' for type %d %d %d", getprint(sptr),
                STYPEG(sptr), SCG(sptr), llt->data_type);

      if ((SCG(sptr) == SC_DUMMY) || (SCG(sptr) == SC_AUTO)) {
        const int midnum = MIDNUMG(sptr);
        if (midnum && STYPEG(midnum) == ST_PROC) {
          assert(LLTYPE(midnum), "process_sptr() never called for sptr", sptr,
                 ERR_Fatal);
          llt = LLTYPE(midnum);
        } else if ((flg.smp || XBIT(34, 0x200) || gbl.usekmpc) &&
                   (gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) &&
                   (sptr == ll_get_shared_arg(gbl.currsub))) {
          llt = LLTYPE(sptr);
        } else
#ifdef TARGET_LLVM_ARM
            if ((llt->data_type == LL_STRUCT) && (NME_SYM(nme) != sptr)) {
          llt = make_ptr_lltype(make_lltype_from_dtype(DT_CPTR));
        } else
#endif
            if ((llt->data_type == LL_PTR) &&
                (llt->sub_types[0]->data_type != LL_PTR) &&
                NME_SYM(nme) != sptr) {
          llt = make_ptr_lltype(make_lltype_from_dtype(DT_CPTR));
        }
        if ((STYPEG(sptr) != ST_VAR) && (ASSNG(sptr) || ADDRTKNG(sptr))) {
          if ((llt->data_type == LL_PTR) &&
              (llt->sub_types[0]->data_type != LL_PTR)) {
            llt = make_ptr_lltype(make_lltype_from_dtype(DT_CPTR));
          }
        }
      } else if ((STYPEG(sptr) != ST_VAR) &&
                 ((llt->data_type != LL_PTR) ||
                  (llt->sub_types[0]->data_type != LL_PTR))) {
        llt = make_ptr_lltype(make_lltype_from_dtype(DT_CPTR));
      }
    }
  }
  addressElementSize = (llt->data_type == LL_PTR)
                           ? ll_type_bytes_unchecked(llt->sub_types[0])
                           : 0;
  operand = gen_base_addr_operand(addr_op, llt);

  DBGTRACEOUT("")

  csed_operand = get_csed_operand(addr_op);
  if (csed_operand != NULL) {
    set_csed_operand(csed_operand, operand);
  }

  ILI_COUNT(addr_op)++;
  addressElementSize = savedAddressSize;
  return operand;
}

/**
   \brief Generate an OPERAND representing the value of the ACON ilix.
   \param ilix           An IL_ACON ilix
   \param expected_type  The expected type of the result
   \return an OPERAND
 */
static OPERAND *
gen_acon_expr(int ilix, LL_Type *expected_type)
{
  SPTR sptr;
  DTYPE dtype;
  ISZ_T idx;
  LL_Type *ty1;
  OPERAND *base_op, *index_op;
  OPERAND *operand = NULL;
  const SPTR opnd = ILI_SymOPND(ilix, 1);
  const int ptrbits = BITS_IN_BYTE * size_of(DT_CPTR);
  INT val[2];
  ISZ_T num;

  assert(ILI_OPC(ilix) == IL_ACON, "gen_acon_expr: acon expected", ilix,
         ERR_Fatal);
  assert(STYPEG(opnd) == ST_CONST, "gen_acon_expr: ST_CONST argument expected",
         ilix, ERR_Fatal);

  /* Handle integer constants, converting to a pointer-sized integer */
  dtype = DTYPEG(opnd);
  if (DT_ISINT(dtype)) {
    INT hi = CONVAL1G(opnd);
    INT lo = CONVAL2G(opnd);

    /* Sign-extend DT_INT to 64 bits */
    if (dtype == DT_INT && ptrbits == 64)
      hi = lo < 0 ? -1 : 0;
    return make_constval_op(make_int_lltype(ptrbits), lo, hi);
  }

  /* With integers handled above, there should only be DT_CPTR constants left.
   * Apparently we sometimes generate DT_IPTR constants too (for wide string
   * constants) */
  assert(DTY(dtype) == TY_PTR,
         "gen_acon_expr: Expected pointer or integer constant", ilix,
         ERR_Fatal);

  /* Handle pointer constants with no base symbol table pointer.
   * This also becomes a pointer-sized integer */
  sptr = SymConval1(opnd);
  if (!sptr) {
    num = ACONOFFG(opnd);
    ISZ_2_INT64(num, val);
    return make_constval_op(make_int_lltype(ptrbits), val[1], val[0]);
  }
  sym_is_refd(sptr);
  idx = (STYPEG(sptr) == ST_STRUCT || STYPEG(sptr) == ST_ARRAY
         || ACONOFFG(opnd) < 0) ? 0 : ACONOFFG(opnd);
  process_sptr_offset(sptr, variable_offset_in_aggregate(sptr, idx));
  idx = ACONOFFG(opnd); /* byte offset */

  ty1 = make_lltype_from_dtype(DT_ADDR);
  idx = variable_offset_in_aggregate(sptr, idx);
  if (idx) {
    base_op = gen_sptr(sptr);
    index_op = NULL;
    base_op = make_bitcast(base_op, ty1);
    ISZ_2_INT64(idx, val); /* make a index operand */
    index_op = make_constval_op(make_int_lltype(ptrbits), val[1], val[0]);
    operand = gen_gep_op(ilix, base_op, ty1, index_op);
  } else {
    operand = gen_sptr(sptr);
    /* SC_DUMMY - address constant .cxxxx */
    if (compilingGlobalOrDevice() && SCG(sptr) == SC_DUMMY &&
        DTYPEG(sptr) == DT_ADDR) {
      /* scalar argument */
      int midnum = MIDNUMG(sptr);
      if (midnum && DEVICEG(midnum) && !PASSBYVALG(midnum))
        operand->ll_type = make_ptr_lltype(operand->ll_type);
      else if (DTY(DTYPEG(midnum)) == TY_PTR ||
               DTY(DTYPEG(midnum)) == TY_ARRAY) /* pointer */
        operand->ll_type = make_ptr_lltype(operand->ll_type);
    }
  }

  if (operand->ll_type && VOLG(sptr))
    operand->flags |= OPF_VOLATILE;
  return operand;
}

/**
   \brief Pattern match the ILI tree and fold when there is a match
   \param addr  The ILI to pattern match
   \param size  The expected type size
 */
static OPERAND *
attempt_gep_folding(int addr, BIGINT64 size)
{
  int kmul, kcon;
  BIGINT64 val;
  OPERAND *op;

  if (ILI_OPC(addr) != IL_KAMV)
    return NULL;
  kmul = ILI_OPND(addr, 1);
  if (ILI_OPC(kmul) != IL_KMUL)
    return NULL;
  kcon = ILI_OPND(kmul, 2);
  if (ILI_OPC(kcon) != IL_KCON)
    return NULL;
  val = ((BIGINT64)CONVAL1G(ILI_OPND(kcon, 1))) << 32;
  val |= CONVAL2G(ILI_OPND(kcon, 1)) & (0xFFFFFFFF);
  if (val != size)
    return NULL;
  /* at this point we are going to drop the explicit multiply */
  op = gen_llvm_expr(ILI_OPND(kmul, 1), make_int_lltype(64));
  return op;
}

/**
   \brief Attempt to convert explicit pointer scaling into GEP
   \param aadd	An IL_AADD
   \param idxOp	The index expression to be checked

   Do \e not assume \p idxOp is the same as <tt>ILI_OPND(aadd, 2)</tt>.
 */
static OPERAND *
maybe_do_gep_folding(int aadd, int idxOp, LL_Type *ty)
{
  int baseOp;
  OPERAND *rv;
  LL_Type *i8ptr;
  unsigned savedAddressElementSize;

  if (addressElementSize == 0)
    return NULL;

  baseOp = ILI_OPND(aadd, 1);
  i8ptr = make_lltype_from_dtype(DT_CPTR);
  if (ty == i8ptr) {
    if (addressElementSize != TARGET_PTRSIZE)
      return NULL;
    ty = ll_get_pointer_type(ty);
  }

  savedAddressElementSize = addressElementSize;
  addressElementSize = 0;

  /* 1. check if idxOp is a scaled expression */
  rv = attempt_gep_folding(idxOp, savedAddressElementSize);
  if (rv) {
    OPERAND *base = gen_base_addr_operand(baseOp, ty);
    rv = gen_gep_op(aadd, base, ty, rv);
    return rv;
  }

  /* 2. check if baseOp is a scaled expression */
  rv = attempt_gep_folding(baseOp, savedAddressElementSize);
  if (rv) {
    OPERAND *index = gen_base_addr_operand(idxOp, ty);
    rv = gen_gep_op(aadd, index, ty, rv);
    return rv;
  }

  addressElementSize = savedAddressElementSize;
  return NULL;
}

static OPERAND *
gen_base_addr_operand(int ilix, LL_Type *expected_type)
{
  OPERAND *operand = NULL, *base_op, *index_op;
  OPERAND **csed_operand;
  LL_Type *ty1, *ty2;
  int opnd = 0;

  DBGTRACEIN2(" for ilix: %d(%s), expected_type ", ilix, IL_NAME(ILI_OPC(ilix)))
  DBGDUMPLLTYPE("expected type ", expected_type)

  switch (ILI_OPC(ilix)) {
  case IL_ASUB:
    if (!ll_type_int_bits(expected_type)) {
      switch (ILI_OPC(ILI_OPND(ilix, 2))) {
      case IL_IAMV:
        opnd = ad1ili(IL_AIMV, ILI_OPND(ilix, 2));
        opnd = ad2ili(IL_ISUB, ad_icon(0), opnd);
        break;
      case IL_KAMV:
        opnd = ad1ili(IL_AKMV, ILI_OPND(ilix, 2));
        opnd = ad2ili(IL_KSUB, ad_kconi(0), opnd);
        break;
      default:
        if (size_of(DT_CPTR) == 8) {
          opnd = ad1ili(IL_AKMV, ILI_OPND(ilix, 2));
          opnd = ad2ili(IL_KSUB, ad_kconi(0), opnd);
        } else {
          opnd = ad1ili(IL_AIMV, ILI_OPND(ilix, 2));
          opnd = ad2ili(IL_ISUB, ad_icon(0), opnd);
        }
      }
    } else {
      if (size_of(DT_CPTR) == 8) {
        opnd = ad1ili(IL_AKMV, ILI_OPND(ilix, 2));
        opnd = ad2ili(IL_KSUB, ad_kconi(0), opnd);
        opnd = ad1ili(IL_KAMV, opnd);
      } else {
        opnd = ad1ili(IL_AIMV, ILI_OPND(ilix, 2));
        opnd = ad2ili(IL_ISUB, ad_icon(0), opnd);
        opnd = ad1ili(IL_IAMV, opnd);
      }
    }
    FLANG_FALLTHROUGH;
  case IL_AADD:
    opnd = opnd ? opnd : ILI_OPND(ilix, 2);
    operand = (XBIT(183, 0x40000))
                  ? NULL
                  : maybe_do_gep_folding(ilix, opnd, expected_type);
    if (!operand) {
      ty1 = make_lltype_from_dtype(DT_CPTR);
      base_op = gen_base_addr_operand(ILI_OPND(ilix, 1), ty1);
      ty2 = make_int_lltype(BITS_IN_BYTE * size_of(DT_CPTR));
      index_op = gen_base_addr_operand(opnd, ty2);
      operand = gen_gep_op(ilix, base_op, ty1, index_op);
    }
    break;
  case IL_ACON:
    operand = gen_acon_expr(ilix, expected_type);
    break;
  default:
    /* third arg must be 0 since we're not generating a GEP in this case */
    operand = gen_llvm_expr(ilix, expected_type);
  }
  if (expected_type)
    ty1 = expected_type;
  else
    goto _exit_gen_base_addr_operand;
  ty2 = operand->ll_type;

  DBGDUMPLLTYPE("#operand type ", ty2);
  DBGDUMPLLTYPE("#expected type ", ty1);

  if (ll_type_int_bits(ty1) && ll_type_int_bits(ty2)) {
    if (ll_type_int_bits(ty1) != ll_type_int_bits(ty2)) {
      operand = convert_int_size(ilix, operand, ty1);
    }
    goto _exit_gen_base_addr_operand;
  } else if ((ty1->data_type == LL_PTR) && (ty2->data_type == LL_PTR)) {
    /* both pointers, but pointing to different types */
    LL_Type *tty1 = NULL, *tty2 = NULL;

    DBGTRACE("#both are pointers, checking if they are pointing to same type")

    if (ty2->sub_types[0]->data_type == LL_ARRAY) {
      tty1 = ty1;
      tty2 = ty2;
    }
    if (tty1 || tty2) {

      while (tty1->data_type == tty2->data_type) {
        if ((tty1->data_type == LL_PTR) || (tty1->data_type == LL_ARRAY)) {
          tty1 = tty1->sub_types[0];
          tty2 = tty2->sub_types[0];
        } else {
          break;
        }
      }
      if (ll_type_int_bits(tty1) && ll_type_int_bits(tty2) &&
          (ll_type_int_bits(tty1) != ll_type_int_bits(tty2))) {
        const int flags = operand->flags & (OPF_SEXT | OPF_ZEXT | OPF_VOLATILE);
        operand = make_bitcast(operand, ty1);
        operand->flags |= flags;
      } else if (tty1->data_type != LL_NOTYPE) {
        operand = make_bitcast(operand, ty1);
      }
    } else if (!strict_match(ty1->sub_types[0], ty2->sub_types[0])) {
      DBGTRACE("#no strict match between pointers")

      operand = make_bitcast(operand, ty1);
    } else {
      LL_Type *ety1 = ty1->sub_types[0];
      LL_Type *ety2 = ty2->sub_types[0];
      DBGTRACE("#strict match between pointers,"
               " checking signed/unsigned conflicts");
      while (ety1->data_type == ety2->data_type) {
        if ((ety1->data_type == LL_PTR) || (ety1->data_type == LL_ARRAY)) {
          ety1 = ety1->sub_types[0];
          ety2 = ety2->sub_types[0];
        } else {
          break;
        }
      }
      if (ll_type_int_bits(ety1) && ll_type_int_bits(ety2) &&
          (ll_type_int_bits(ety1) != ll_type_int_bits(ety2))) {
        const int flags = operand->flags & (OPF_SEXT | OPF_ZEXT | OPF_VOLATILE);
        operand = make_bitcast(operand, ty1);
        operand->flags |= flags;
      }
    }
  } else if ((ty1->data_type == LL_PTR) && ll_type_int_bits(ty2)) {
    if ((operand->ot_type == OT_CONSTVAL) && (!operand->val.conval[0]) &&
        (!operand->val.conval[1])) {
      // rewrite: cast(iN 0) to T*  ==>  (T* null)
      operand = make_constval_op(ty1, 0, 0);
      operand->flags |= OPF_NULL_TYPE;
    } else if ((operand->ot_type != OT_VAR) ||
               (!ll_type_int_bits(ty1->sub_types[0]))) {
      operand = convert_int_to_ptr(operand, ty1);
    }
  } else if (ty1->data_type == LL_PTR && ty2->data_type == LL_STRUCT) {
    operand->ll_type = make_ptr_lltype(ty2);
    operand = make_bitcast(operand, ty1);
  } else if (ty1->data_type == LL_PTR && ty2->data_type == LL_VECTOR &&
             !strict_match(ty1->sub_types[0], ty2)) {
    operand->ll_type = make_ptr_lltype(ty2);
    operand = make_bitcast(operand, ty1);
  } else if (ll_type_int_bits(ty1) && (ty2->data_type == LL_PTR)) {
    operand = convert_ptr_to_int(operand, ty1);
  } else if (ty1->data_type == LL_PTR && ty2->data_type == LL_ARRAY) {
    operand = make_bitcast(operand, ty1);
  } else if (ty1->data_type != ty2->data_type) {
    if (ty1->data_type == LL_PTR && operand->ot_type == OT_VAR) {
      ty1 = ty1->sub_types[0];
      while (ty1->data_type == ty2->data_type) {
        if (ty1->data_type == LL_PTR || ty1->data_type == LL_ARRAY) {
          ty1 = ty1->sub_types[0];
          ty2 = ty2->sub_types[0];
        } else {
          break;
        }
      }
      if (ty1->data_type == ty2->data_type || ty1->data_type == LL_VOID)
        goto _exit_gen_base_addr_operand;
      if (ll_type_int_bits(ty1) && (ty2->data_type == LL_FLOAT) &&
          ll_type_bytes(ty1) == 4) {
        operand = make_bitcast(operand, ty1);
        goto _exit_gen_base_addr_operand;
      }
    }
    assert(0, "gen_base_addr_operand(): unexpected conversion", 0, ERR_Fatal);
  }
_exit_gen_base_addr_operand:
  csed_operand = get_csed_operand(ilix);
  if (csed_operand != NULL)
    set_csed_operand(csed_operand, operand);
  ILI_COUNT(ilix)++;

  DBGTRACEOUT4(" returns operand %p, tmps %p, count %d for ilix %d", operand,
               operand->tmps, ILI_COUNT(ilix), ilix)
  setTempMap(ilix, operand);
  return operand;
}

void
print_tmp_name(TMPS *t)
{
  char tmp[10];
  int idx = 0;

  if (!t) {
    idx = ++expr_id;
    sprintf(tmp, "%%%d", idx - 1);
    print_token(tmp);
    return;
  }

  if (!t->id)
    t->id = ++expr_id;
  sprintf(tmp, "%%%d", t->id - 1);
  print_token(tmp);
}

static bool
repeats_in_binary(union xx_u xx)
{
  bool ret_val;
  double dd = (double)xx.ff;

  if (!llvm_info.no_debug_info) {
    DBGTRACEIN1(" input value: %g \n", dd)
  }

  ret_val = true;
  if (!llvm_info.no_debug_info) {
    DBGTRACEOUT1(" returns %s", ret_val ? "True" : "False")
  }
  return ret_val;
} /* repeats_in_binary */

static char *
gen_vconstant(const char *ctype, int sptr, DTYPE tdtype, int flags)
{
  DTYPE vdtype;
  int vsize;
  int i;
  int edtype;
  static char tmp_vcon_buf[2000];
  char *constant;

  vdtype = DTySeqTyElement(tdtype);
  vsize = DTyVecLength(tdtype);
  edtype = CONVAL1G(sptr);

  if (flags & FLG_OMIT_OP_TYPE) {
    tmp_vcon_buf[0] = '<';
    tmp_vcon_buf[1] = '\0';
  } else
    sprintf(tmp_vcon_buf, "%s <", ctype);

  for (i = 0; i < vsize; i++) {
    if (i)
      strcat(tmp_vcon_buf, ", ");
    switch (DTY(vdtype)) {
    case TY_REAL:
      strcat(tmp_vcon_buf,
             gen_constant(SPTR_NULL, vdtype, VCON_CONVAL(edtype + i), 0,
                          flags & ~FLG_OMIT_OP_TYPE));
      break;
    case TY_INT8:
    case TY_DBLE:
      strcat(tmp_vcon_buf, gen_constant((SPTR)VCON_CONVAL(edtype + i), DT_NONE,
                                        0, 0, flags & ~FLG_OMIT_OP_TYPE));
      break;
    default:
      strcat(tmp_vcon_buf,
             gen_constant(SPTR_NULL, vdtype, VCON_CONVAL(edtype + i), 0,
                          flags & ~FLG_OMIT_OP_TYPE));
    }
  }
  strcat(tmp_vcon_buf, ">");
  constant = (char *)getitem(LLVM_LONGTERM_AREA,
                             strlen(tmp_vcon_buf) + 1); /* room for \0 */
  strcpy(constant, tmp_vcon_buf);
  return constant;
}

char *
gen_llvm_vconstant(const char *ctype, int sptr, DTYPE tdtype, int flags)
{
  return gen_vconstant(ctype, sptr, tdtype, flags);
}

static char *
gen_constant(SPTR sptr, DTYPE tdtype, INT conval0, INT conval1, int flags)
{
  DTYPE dtype;
  INT num[2] = {0, 0};
  union xx_u xx;
  union {
    double d;
    INT tmp[2];
  } dtmp, dtmp2;
  char *constant = NULL, *constant1, *constant2;
  const char *ctype = "";
  int size = 0;

  static char d[MAXIDLEN];
  static char *b = NULL;

  if (b == NULL) {
    NEW(b, char, 100);
  }

  assert((sptr || tdtype), "gen_constant(): missing arguments", 0, ERR_Fatal);
  if (sptr)
    dtype = DTYPEG(sptr);
  else
    dtype = tdtype;

  if (!llvm_info.no_debug_info) {
    DBGTRACEIN3(" sptr %d, dtype:%d(%s)", sptr, dtype, stb.tynames[DTY(dtype)])
  }

  if (!(flags & FLG_OMIT_OP_TYPE)) {
    ctype = llvm_fc_type(dtype);
    size += strlen(ctype) + 1; /* include room for space after the type */
  }
/* Use an enum's underlying type. */

  if (dtype && DTY(dtype) == TY_VECT)
    return gen_vconstant(ctype, sptr, dtype, flags);

  switch (dtype) {
  case DT_INT:
  case DT_SINT:
  case DT_BINT:
  case DT_USINT:
  case DT_UINT:
  case DT_LOG:
  case DT_SLOG:
  case DT_BLOG:
#if !LONG_IS_64
#endif

    if (sptr)
      sprintf(b, "%ld", (long)CONVAL2G(sptr));
    else
      sprintf(b, "%ld", (long)conval0); /* from dinit info */
    size += strlen(b);

    if (!llvm_info.no_debug_info) {
      DBGTRACE2("#generating integer value: %s %s\n", char_type(dtype, sptr), b)
    }

    constant = (char *)getitem(LLVM_LONGTERM_AREA, size + 1); /* room for \0 */
    if (flags & FLG_OMIT_OP_TYPE)
      sprintf(constant, "%s", b);
    else
      sprintf(constant, "%s %s", ctype, b);
    break;
#if LONG_IS_64
#endif
  case DT_INT8:
  case DT_UINT8:
  case DT_LOG8:
    if (sptr) {
      num[1] = CONVAL2G(sptr);
      num[0] = CONVAL1G(sptr);
    } else {
      num[1] = conval0;
      num[0] = conval1;
    }
    ui64toax(num, b, 22, 0, 10);
    size += strlen(b);

    if (!llvm_info.no_debug_info) {
      DBGTRACE2("#generating integer value: %s %s\n", char_type(dtype, sptr), b)
    }

    constant = (char *)getitem(LLVM_LONGTERM_AREA, size + 1); /* room for \0 */
    if (flags & FLG_OMIT_OP_TYPE)
      sprintf(constant, "%s", b);
    else
      sprintf(constant, "%s %s", ctype, b);
    break;

  case DT_DBLE:
  case DT_QUAD:

    if (sptr) {
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
    } else {
      num[0] = conval0;
      num[1] = conval1;
    }

    cprintf(d, "%.17le", num);
    /* Check for  `+/-Infinity` and 'NaN' based on the IEEE bit patterns */
    if ((num[0] & 0x7ff00000) == 0x7ff00000) /* exponent == 2047 */
      sprintf(d, "0x%08x00000000", num[0]);
    /* also check for -0 */
    else if (num[0] == (INT)0x80000000 && num[1] == (INT)0x00000000)
      sprintf(d, "-0.00000000e+00");
    /* remember to make room for /0 */
    constant =
        (char *)getitem(LLVM_LONGTERM_AREA, strlen(d) + strlen(ctype) + 1);
    if (flags & FLG_OMIT_OP_TYPE)
      sprintf(constant, "%s", d);
    else
      sprintf(constant, "%s %s", ctype, d);
    if (!llvm_info.no_debug_info) {
      DBGTRACE1("#set double exponent value to %s", d)
    }
    break;
  case DT_REAL:
    /* our internal representation of floats is in 8 digit hex form;
     * internal LLVM representation of floats in hex form is 16 digits;
     * thus we must make the conversion. Also need to decide when to
     * represent final float form in exponential or hexadecimal form.
     */
    if (sptr)
      xx.ww = CONVAL2G(sptr);
    else
      xx.ww = conval0;
    xdble(xx.ww, dtmp2.tmp);
    xdtomd(dtmp2.tmp, &dtmp.d);
    snprintf(d, 200, "%.8e", dtmp.d);
    size += 19;
    constant = (char *)getitem(
        LLVM_LONGTERM_AREA,
        size + 2); /* room for \0  and potentially a '-' sign for neg_zero */
    constant1 = (char *)getitem(LLVM_LONGTERM_AREA, 9);
    constant2 = (char *)getitem(LLVM_LONGTERM_AREA, 9);
    if (repeats_in_binary(xx)) {
      /* put in hexadecimal form unless neg 0 */
      if (dtmp.tmp[0] == -1) /* pick up the quiet nan */
        sprintf(constant1, "7FF80000");
      else if (!dtmp.tmp[1])
        sprintf(constant1, "00000000");
      else
        sprintf(constant1, "%X", dtmp.tmp[1]);
      if (!dtmp.tmp[0] || dtmp.tmp[0] == -1)
        sprintf(constant2, "00000000");
      else
        sprintf(constant2, "%X", dtmp.tmp[0]);
      if (flags & FLG_OMIT_OP_TYPE)
        sprintf(constant, "0x%s%s", constant1, constant2);
      else
        sprintf(constant, "%s 0x%s%s", ctype, constant1, constant2);

      /* check for negative zero */
      if (dtmp.tmp[1] == (INT)0x80000000 && !dtmp.tmp[0]) {
        if (flags & FLG_OMIT_OP_TYPE)
          sprintf(constant, "-0.000000e+00");
        else
          sprintf(constant, "%s -0.000000e+00", ctype);
      }
    } else {
      /*  put in exponential form */
      if (flags & FLG_OMIT_OP_TYPE)
        sprintf(constant, "%s", d);
      else
        sprintf(constant, "%s %s", ctype, d);
    }

    if (!llvm_info.no_debug_info) {
      DBGTRACE1("#set float exp value to %s", d)
      DBGTRACE2("#set float hex value to 0x%X%x", dtmp.tmp[1], dtmp.tmp[0])
    }

    break;

#ifdef LONG_DOUBLE_FLOAT128
  case DT_FLOAT128:
    size += 36;
    constant = getitem(LLVM_LONGTERM_AREA, size);
    if (flags & FLG_OMIT_OP_TYPE)
      snprintf(constant, size, "0xL%08x%08x%08x%08x", CONVAL1G(sptr),
               CONVAL2G(sptr), CONVAL3G(sptr), CONVAL4G(sptr));
    else
      snprintf(constant, size, "%s 0xL%08x%08x%08x%08x", ctype, CONVAL1G(sptr),
               CONVAL2G(sptr), CONVAL3G(sptr), CONVAL4G(sptr));
    break;
#endif

  default:
    if (!llvm_info.no_debug_info) {
      DBGTRACE3("### gen_constant; sptr %d, unknown dtype: %d(%s)", sptr, dtype,
                stb.tynames[DTY(dtype)])
    }
    assert(0, "gen_constant(): unexpected constant dtype", dtype, ERR_Fatal);
  }

  if (!llvm_info.no_debug_info) {
    DBGTRACEOUT1(" returns %s", constant)
  }
  return constant;
} /* gen_constant */

#ifdef FLANG2_CGMAIN_UNUSED
static char *
add_tmp_buf_list_item(TEMP_BUF_LIST **tempbuflist_ptr, int sz)
{
  int i;
  TEMP_BUF_LIST *last;

  for (last = *tempbuflist_ptr; last && last->next; last = last->next)
    ;

  if (*tempbuflist_ptr) {
    last->next =
        (TEMP_BUF_LIST *)getitem(CG_MEDTERM_AREA, sizeof(TEMP_BUF_LIST));
    last = last->next;
  } else {
    *tempbuflist_ptr = last =
        (TEMP_BUF_LIST *)getitem(CG_MEDTERM_AREA, sizeof(TEMP_BUF_LIST));
  }

  last->next = NULL;
  last->buf.buffer = (char *)getitem(CG_MEDTERM_AREA, sz);
  *last->buf.buffer = '\0';
  return last->buf.buffer;
}
#endif

#ifdef OMP_OFFLOAD_LLVM
INLINE static bool
isNVVM(char *fn_name)
{
  if (!flg.omptarget)
    return false;
  return (strncmp(fn_name, "__kmpc", 6) == 0) ||
         (strncmp(fn_name, "llvm.nvvm", 9) == 0) ||
         (strncmp(fn_name, "omp_", 4) == 0) ||
         (strncmp(fn_name, "llvm.fma", 8) == 0);
}
#endif

static void
write_extern_fndecl(struct LL_FnProto_ *proto)
{
  /* Only print decls if we have not seen a body (must be external) */
  if (!proto->has_defined_body) {
#ifdef OMP_OFFLOAD_LLVM
    bool isnvvm = false;
    do {
#endif
      if (proto->intrinsic_decl_str) {
        print_token(proto->intrinsic_decl_str);
        if (proto->abi && proto->abi->is_pure)
          print_token(" nounwind readnone willreturn");
        print_nl();
      } else {
        print_token("declare");
        if (proto->is_weak)
          print_token(" extern_weak");
        print_function_signature(0, proto->fn_name, proto->abi, false);
        if (proto->abi->is_nomerge)
          print_token(" nomerge");
        if (proto->abi->is_pure)
          print_token(" nounwind readnone willreturn");
        if ((!flg.ieee || XBIT(216, 1)) && proto->abi->fast_math)
          print_token(" \"no-infs-fp-math\"=\"true\" "
                      "\"no-nans-fp-math\"=\"true\" "
                      "\"unsafe-fp-math\"=\"true\" \"use-soft-float\"=\"false\""
                      " \"no-signed-zeros-fp-math\"=\"true\"");
        print_nl();
      }
#ifdef OMP_OFFLOAD_LLVM
      if (isnvvm) {
        isnvvm = false;
        use_cpu_output_file();
      } else if (isNVVM(proto->fn_name)) {
        isnvvm = true;
        use_gpu_output_file();
      }
    } while (isnvvm);
#endif
  }
}

void
write_external_function_declarations(int first_time)
{
  DBGTRACEIN("");

  if (first_time)
    print_nl();
  ll_proto_iterate(write_extern_fndecl);
  DBGTRACEOUT("");
} /* write_external_function_declarations */

INLINE static void
write_target_features(void)
{
  if (flg.target_features) {
    print_token(" \"target-features\"=\"");
    print_token(flg.target_features);
    print_token("\"");
  }
}

INLINE static void
write_vscale_range(void)
{
  if (flg.vscale_range_min) {
    char vsrange[64U];
    snprintf(vsrange, sizeof vsrange, " vscale_range(%d,%d)",
                                      flg.vscale_range_min,
                                      flg.vscale_range_max);
    print_token(vsrange);
  }
}

/**
   \brief Emit function attributes in debugging mode output

   The <code>"no-frame-pointer-elim-non-leaf"</code> flag is included to
   generate better coverage of the function in the \c .eh_frame section. This is
   done primarily to help the profiler unwind the stack.
 */
INLINE static void
write_function_attributes(void)
{
  if (need_debug_info(SPTR_NULL)) {
    print_token("attributes #0 = {");

    if (!XBIT(183, 0x10))
      print_token(" noinline");
    print_token(" \"no-frame-pointer-elim-non-leaf\"");
    if (XBIT(216, 0x1000))
      print_token(" \"fp-contract\"=\"fast\"");
    write_target_features();
    write_vscale_range();
    print_token(" }\n");
  }
}

static void
write_global_and_static_defines(void)
{
  GBL_LIST *gl;
  for (gl = Globals; gl; gl = gl->next) {
    if ((STYPEG(gl->sptr) == ST_CONST) ||
        ((SCG(gl->sptr) == SC_LOCAL) && DINITG(gl->sptr) && !REFG(gl->sptr)) ||
        ((SCG(gl->sptr) == SC_EXTERN) && (STYPEG(gl->sptr) == ST_VAR) &&
         (DTYPEG(gl->sptr) == DT_ADDR))) {
      print_line(gl->global_def);
    }
  }
  Globals = NULL;
}

static void
build_unused_global_define_from_params(void)
{
  return;
}

/**
   \brief Helper function: In Fortran, test if \c MIDNUM is not \c SC_DUMMY
   \param sptr   a symbol
 */
INLINE static bool
formalsMidnumNotDummy(SPTR sptr)
{
  return SCG(MIDNUMG(sptr)) != SC_DUMMY;
}

/**
   \brief Helper function: Get \c DTYPE of \p s
   \param s  a symbol
   Fortran requires special handling for ST_PROC.
 */
INLINE static DTYPE
formalsGetDtype(SPTR s)
{
  return ((STYPEG(s) == ST_PROC) && (!DTYPEG(s))) ? DT_ADDR : DTYPEG(s);
}

INLINE static OPERAND *
cons_expression_metadata_operand(LL_Type *llTy)
{
  // FIXME: we don't need to always do this, do we? do a type check here
  if (llTy->data_type == LL_PTR) {
    LL_DebugInfo *di = cpu_llvm_module->debug_info;
    unsigned v = lldbg_encode_expression_arg(LL_DW_OP_deref, 0);
    LL_MDRef exprMD = lldbg_emit_expression_mdnode(di, 1, v);
    return make_mdref_op(exprMD);
  }
  return NULL;
}

INLINE static bool
formalsNeedDebugInfo(SPTR sptr)
{
#ifdef OMP_OFFLOAD_LLVM
  if(is_ompaccel(sptr)) return false;
#endif
  return generating_debug_info();
}

/**
   \brief Helper function: add debug information for formal argument
   \param sptr  a symbol
   \param i     parameter position
 */
INLINE static void
formalsAddDebug(SPTR sptr, unsigned i, LL_Type *llType, bool mayHide)
{
  if (formalsNeedDebugInfo(sptr)) {
    bool is_ptr_alc_arr = false;
    SPTR new_sptr = (SPTR)REVMIDLNKG(sptr);
    if (ll_feature_debug_info_ver90(&cpu_llvm_module->ir) &&
        CCSYMG(sptr) /* Otherwise it can be a cray pointer */ &&
        (new_sptr && (STYPEG(new_sptr) == ST_ARRAY) &&
         (POINTERG(new_sptr) || ALLOCATTRG(new_sptr))) &&
        SDSCG(new_sptr)) {
      is_ptr_alc_arr = true;
      sptr = new_sptr;
    }
    LL_DebugInfo *db = current_module->debug_info;
    if (ll_feature_debug_info_ver90(&cpu_llvm_module->ir) &&
        STYPEG(sptr) == ST_ARRAY && CCSYMG(sptr) &&
        !LL_MDREF_IS_NULL(get_param_mdnode(db, sptr)))
      return;
    LL_MDRef param_md = lldbg_emit_param_variable(
        db, sptr, BIH_FINDEX(gbl.entbih), i, CCSYMG(sptr));
    if (!LL_MDREF_IS_NULL(param_md)) {
      LL_Type *llTy = fixup_argument_type(sptr, llType);
      OPERAND *exprMDOp = (STYPEG(sptr) == ST_ARRAY)
                              ? NULL
                              : cons_expression_metadata_operand(llTy);
      OperandFlag_t flag = (mayHide && CCSYMG(sptr)) ? OPF_HIDDEN : OPF_NONE;
      // For pointer, allocatable, assumed shape and assumed rank arrays, pass
      // descriptor in place of base address.
      if (ll_feature_debug_info_ver90(&cpu_llvm_module->ir) &&
          (is_ptr_alc_arr || ASSUMRANKG(sptr) || ASSUMSHPG(sptr)) &&
          SDSCG(sptr))
        sptr = SDSCG(sptr);
      insert_llvm_dbg_declare(param_md, sptr, llTy, exprMDOp, flag);
    }
  }
}

/**
   \brief Process the formal arguments to the current function

   Generate the required prolog code to home all arguments that need it.

   \c llvm_info.abi_info must be initialized before calling this function.

   Populates \c llvm_info.homed_args, which must be allocated and empty before
   the call.
 */
void
process_formal_arguments(LL_ABI_Info *abi)
{
  /* Entries already have been processed */
  unsigned i;

  for (i = 1; i <= abi->nargs; i++) {
    OPERAND *arg_op;
    OPERAND *store_addr;
    LL_Type *llTy;
    LL_InstrListFlags flags;
    SPTR key;
    LL_ABI_ArgInfo *arg = &abi->arg[i];
    const char *suffix = ".arg";
    bool ftn_byval = false;

    assert(arg->sptr, "Unnamed function argument", i, ERR_Fatal);
    if (!ll_feature_debug_info_ver90(&cpu_llvm_module->ir)) {
      assert(SNAME(arg->sptr) == NULL, "Argument sptr already processed",
             arg->sptr, ERR_Fatal);
    }
    if ((SCG(arg->sptr) != SC_DUMMY) && formalsMidnumNotDummy(arg->sptr)) {
      process_sptr(arg->sptr);
      continue;
    }

    switch (arg->kind) {
    case LL_ARG_BYVAL:
      if (abi->is_fortran && !abi->is_iso_c) {
        ftn_byval = true;
        break;
      }
      FLANG_FALLTHROUGH;
    case LL_ARG_INDIRECT:
    case LL_ARG_INDIRECT_BUFFERED:
      /* For device pointer, we need to home it because we will need to pass it
       * as &&arg(pointer to pointer), make_var_op will call process_sptr later.
       */
      if (compilingGlobalOrDevice() && DEVICEG(arg->sptr))
        break;
      /* These arguments already appear as pointers. Should we make a copy of
       * an indirect arg? The caller doesn't expect us to modify the memory.
       */
      process_sptr(arg->sptr);
      key = ((SCG(arg->sptr) == SC_BASED) && MIDNUMG(arg->sptr))
                ? MIDNUMG(arg->sptr)
                : arg->sptr;
      llTy = llis_dummied_arg(key) ? make_generic_dummy_lltype() : LLTYPE(key);
      formalsAddDebug(key, i, llTy, false);
      continue;

    case LL_ARG_COERCE:
      /* This argument is passed by value as arg->type which is not the real
       * type of the argument. Generate code to save the LLVM argument into a
       * local variable of the right type. */
      suffix = ".coerce";
      break;

    default:
      /* Other by-value kinds. */
      break;
    }

    /* This op represents the real LLVM argument, not the local variable. */
    arg_op = make_operand();
    arg_op->ot_type = OT_VAR;
    arg_op->ll_type = make_lltype_from_abi_arg(arg);
    arg_op->val.sptr = arg->sptr;

    key = arg->sptr;
    /* if it is a pointer, should use midnum as hash key because most of
     * the time, the ILI is referencing to is MIDNUMG(x$p).
     * If there will ever be a reference to this SC_BASED directly,
     * we should always use its MIDNUMG for hashing.
     */
    if (SCG(arg->sptr) == SC_BASED && MIDNUMG(arg->sptr))
      key = MIDNUMG(arg->sptr);
    hashmap_insert(llvm_info.homed_args, INT2HKEY(key), arg_op);

    /* Process the argument sptr *after* updating homed_args.
     * process_sptr() will look at this map to treat the argument as an
     * auto instead of a dummy. */
    store_addr = make_var_op(arg->sptr);

    /* make sure it is pointer to pointer */
    if (compilingGlobalOrDevice() && DEVICEG(arg->sptr) &&
        !(ftn_byval || PASSBYVALG(arg->sptr)))
      store_addr->ll_type = ll_get_pointer_type(store_addr->ll_type);

    /* Make a name for the real LLVM IR argument. This will also be used by
     * build_routine_and_parameter_entries(). */
    arg_op->string = ll_create_local_name(
        llvm_info.curr_func, "%s%s", SYMNAME(arg->sptr), suffix);

    /* Emit code in the entry block that saves the argument into the local
     * variable. */
    if (store_addr->ll_type->sub_types[0] != arg->type) {
      LL_Type *var_type = store_addr->ll_type->sub_types[0];
      if (ll_type_bytes(arg->type) > ll_type_bytes(var_type)) {
        /* This can happen in C (but not C++) with a new-style declaration and
           an old-style definition:
             int f(int);
             int f(c) char c; { ... }
           Cast the argument value to the local variable type. */
        if (ll_type_int_bits(arg->type) && ll_type_int_bits(var_type)) {
          arg_op = convert_operand(arg_op, var_type, I_TRUNC);
        } else if (ll_type_is_fp(arg->type) && ll_type_is_fp(var_type)) {
          arg_op = convert_operand(arg_op, var_type, I_FPTRUNC);
        } else {
#ifdef TARGET_LLVM_ARM64
           /* 
            On ARM64 the ABI requires for instance that a 3 4-byte struct (12 bytes)
            be coerced into 2 8-byte registers. This is achieved by declaring the type of the 
            formal argument to be a [2 x i64] and assigning its value to a local variable of 
            type {i32, i32, i32}. This has been handled, so far, by performing a store of the 
            formal to the address of the local which only really works when both are the same size 
            otherwise the store spills over the next element on the stack. That's bad.
            
            To fix this:
              - the local storage of {i32, i32, i32}, that has already been created at this point, 
                is replaced with the bigger one of size [2 x i64]
              - the store_addr operand is repurposed to point to the new storage
              - in ll_write_local_objects the kind LLObj_LocalBuffered is detected so 
                the appropriate bitcast is performed 
            */
          LL_Object * object, *local, *prev_object = NULL;
          LL_Function * function = llvm_info.curr_func;
          local = function->last_local;

          // The last local variable introduced must have been for this argument otherwise error
          if (local && strcmp(local->address.data, SNAME(arg->sptr)) == 0) {
            // locals are singly chained so iterate from the start to find the previous one
            for (object = function->first_local; object->next; object = object->next) {
              prev_object = object;
            }

            LL_Object *bigger_local
             = ll_create_local_object(llvm_info.curr_func, arg->type, 8,
                               "%s.buffer", get_llvm_name(arg->sptr));

            // Help detect discrepancy in ll_write_local_objects                 
            bigger_local->sptr = arg->sptr;
            bigger_local->kind = LLObj_LocalBuffered;

            // Repurpose store_addr to the bigger storage
            store_addr->ll_type = ll_get_pointer_type(arg->type);
            store_addr->string = bigger_local->address.data;

            // Replace last local in the list with the bigger one
            if (prev_object == NULL) {
              function->first_local = function->last_local = bigger_local;
            } else {
              prev_object->next = bigger_local;
              function->last_local = bigger_local;
            }
          } else {
            assert(false,
                  "process_formal_arguments: Function argument with missing "
                  "local storage",
                  0, ERR_Fatal);
          }
          
#else
          assert(false,
                 "process_formal_arguments: Function argument with mismatched "
                 "size that is neither integer nor floating-point",
                 0, ERR_Fatal);
#endif
        }
      } else {
        /* Use a pointer bitcast on the address of the local variable to coerce
           the argument to the local variable type. */
        store_addr =
            make_bitcast(store_addr, ll_get_pointer_type(arg_op->ll_type));
      }
    }

    flags = ldst_instr_flags_from_dtype(formalsGetDtype(arg->sptr));
    if (ftn_byval) {
      arg_op = make_load(0, arg_op, arg_op->ll_type->sub_types[0],
                         mem_size(TY_INT), 0);
      store_addr = make_var_op(arg->sptr);
    }
    make_store(arg_op, store_addr, flags);

    if (CCSYMG(arg->sptr)) {
      llTy = arg_op->ll_type;
      if (llTy->data_type != LL_PTR)
        llTy = ll_get_pointer_type(llTy);
    } else {
      llTy = LLTYPE(arg->sptr);
    }
    assert(llTy->data_type == LL_PTR, "expected a pointer type",
           llTy->data_type, ERR_Fatal);
    /* Emit an @llvm.dbg.declare right after the store. */
    /* if arg->sptr is the compiler created symbol which represents the length
     * of assumed length string type, then make the first metadata argument type
     * of this symbol as address instead of value in the llvm.dbg.declare
     * intrinsic.
     */
    if ((arg->kind == LL_ARG_DIRECT) && CCSYMG(arg->sptr) &&
            PASSBYVALG(arg->sptr) && clen_parent_is_param(arg->sptr))
      formalsAddDebug(arg->sptr, i, llTy, false);
    else
      formalsAddDebug(arg->sptr, i, llTy, true);
  }
}

/**
   \brief Write out attributes for a function argument or return value.
   \param arg  an argument's info record
 */
static void
print_arg_attributes(LL_ABI_ArgInfo *arg)
{
  switch (arg->kind) {
  case LL_ARG_DIRECT:
  case LL_ARG_COERCE:
  case LL_ARG_INDIRECT:
  case LL_ARG_INDIRECT_BUFFERED:
    break;
  case LL_ARG_ZEROEXT:
    print_token(" zeroext");
    break;
  case LL_ARG_SIGNEXT:
    print_token(" signext");
    break;
  case LL_ARG_BYVAL:
    print_token(" byval(");
    print_token(arg->type->sub_types[0]->str);
    print_token(")");
    break;
  default:
    interr("Unknown argument kind", arg->kind, ERR_Fatal);
  }
  if (arg->inreg)
    print_token(" inreg");
}

/**
 * \brief Print the signature of func_sptr, omitting the leading define/declare,
 * ending after the function attributes.
 *
 * When print_arg_names is set, also print the names of arguments. (From
 * abi->arg[n].sptr).
 *
 * fn_name is passed separately from the sptr, since Fortran also calls this
 * routine.  In the Fortran case, the sptr will not always be valid, but the
 * LL_FnProto contains a valid fn_name string.
 */
static void
print_function_signature(int func_sptr, const char *fn_name, LL_ABI_Info *abi,
                         bool print_arg_names)
{
  unsigned i;
  bool need_comma = false;

  /* Fortran treats functions with unknown prototypes as varargs,
   * we cannot decorate them with fastcc.
   */
  if (abi->call_conv &&
      !(abi->call_conv == LL_CallConv_Fast && abi->call_as_varargs)) {
    print_space(1);
    print_token(
        ll_get_calling_conv_str((enum LL_CallConv)abi->call_conv)); // ???
  }
#ifdef WEAKG
  if (func_sptr > NOSYM && WEAKG(func_sptr)) {
    print_token(" weak");
  }
#endif

  /* Print function return type with attributes. */
  if (LL_ABI_HAS_SRET(abi)) {
    print_token(" void");
  } else {
    print_arg_attributes(&abi->arg[0]);
    print_space(1);
    print_token(abi->extend_abi_return ? make_lltype_from_dtype(DT_INT)->str
                                       : abi->arg[0].type->str);
  }

  print_token(" @");
  print_token(map_to_llvm_name(fn_name));
  print_token("(");

  /* Hidden sret argument for struct returns. */
  if (LL_ABI_HAS_SRET(abi)) {
    print_token(abi->arg[0].type->str);
    print_token(" sret(");
    print_token(abi->arg[0].type->sub_types[0]->str);
    print_token(")");
    if (print_arg_names) {
      print_space(1);
      print_token(SNAME(ret_info.sret_sptr));
    }
    need_comma = true;
  }

  /* Iterate over function arguments. */
  for (i = 1; i <= abi->nargs; i++) {
    LL_ABI_ArgInfo *arg = &abi->arg[i];

    if (need_comma)
      print_token(", ");

    print_token(arg->type->str);
    print_arg_attributes(arg);

    if (print_arg_names && arg->sptr) {
      int key;
      const OPERAND *coerce_op = NULL;
      print_space(1);
      key = arg->sptr;
      if (SCG(arg->sptr) == SC_BASED && MIDNUMG(arg->sptr))
        key = MIDNUMG(arg->sptr);

      if (hashmap_lookup(llvm_info.homed_args, INT2HKEY(key),
                         (hash_data_t *)&coerce_op)) {
        print_token(coerce_op->string);
      } else {
        assert(SNAME(arg->sptr),
               "print_function_signature: "
               "No SNAME for sptr",
               arg->sptr, ERR_Fatal);
        print_token(SNAME(arg->sptr));
      }
    }
    need_comma = true;
  }

  /* Finally, append ... for varargs functions. */
  if (ll_abi_use_llvm_varargs(abi)) {
    if (need_comma)
      print_token(", ");
    print_token("...");
  }

  print_token(")");

  /* Function attributes.  With debugging turned on, the debug attributes
     contain "noinline" (and others), so there is no need to repeat it here. */
  if (need_debug_info(SPTR_NULL)) {
    /* 'attributes #0 = { ... }' to be emitted later */
    print_token(" #0");
  } else {
    if (!XBIT(183, 0x10)) {
      /* Nobody sets -x 183 0x10, besides Flang. We're disabling LLVM inlining for
       * proprietary compilers. */
      print_token(" noinline");
    }
    if (XBIT(216, 0x1000)) {
      print_token(" \"fp-contract\"=\"fast\"");
    }
    write_target_features();
    write_vscale_range();
  }
  if (XBIT(14, 0x8)) {
    /* Apply noinline attribute if the pragma "noinline" is given */
    print_token(" noinline");
  }
  if (XBIT(191, 0x2)) {
    /* Apply alwaysinline attribute if the pragma "forceinline" is given */
    print_token(" alwaysinline");
  }

  if (func_sptr > NOSYM) {
/* print_function_signature() can be called with func_sptr=0 */
  }

#ifdef ELFSCNG
  if (ELFSCNG(func_sptr)) {
    print_token(" section \"");
    print_token(SYMNAME(ELFSCNG(func_sptr)));
    print_token("\"");
  }
#endif
#ifdef TEXTSTARTUPG
  if (TEXTSTARTUPG(func_sptr)) {
    print_token(" section \".text.startup \"");
  }
#endif

}

#ifdef OMP_OFFLOAD_LLVM
INLINE void static add_property_struct(char *func_name, int nreductions,
                                       int reductionsize)
{
  print_token("@");
  print_token(func_name);
  print_token("__exec_mode = weak constant i8 0\n");
}
#endif

/**
   \brief write out the header of the function definition

   Writes text from \c define to the label of the entry block.
 */
void
build_routine_and_parameter_entries(SPTR func_sptr, LL_ABI_Info *abi,
                                    LL_Module *module)
{
  const char *linkage = NULL;
#ifdef OMP_OFFLOAD_LLVM
  int reductionsize = 0;
  if (OMPACCFUNCKERNELG(func_sptr)) {
    OMPACCEL_TINFO *tinfo = ompaccel_tinfo_get(func_sptr);
    if (tinfo->n_reduction_symbols == 0) {
      add_property_struct(SYMNAME(func_sptr), 0, 0);
    } else {
      for (int i = 0; i < tinfo->n_reduction_symbols; ++i) {
        reductionsize +=
            (size_of(DTYPEG(tinfo->reduction_symbols[i].shared_sym)) *
             BITS_IN_BYTE);
      }
      add_property_struct(SYMNAME(func_sptr), tinfo->n_reduction_symbols,
                          reductionsize);
    }
  }
#endif
  /* Start printing the defining line to the output file. */
  print_token("define");

/* Function linkage. */
      if (SCG(func_sptr) == SC_STATIC)
    linkage = " internal";
#ifdef OMP_OFFLOAD_LLVM
  if (OMPACCFUNCKERNELG(func_sptr)) {
    linkage = " ptx_kernel";
  }
#endif
  if (linkage)
    print_token(linkage);
  if (SCG(func_sptr) != SC_STATIC)
    llvm_set_unique_sym(func_sptr);
#ifdef WEAKG
  if (WEAKG(func_sptr))
    ll_proto_set_weak(ll_proto_key(func_sptr), true);
#endif

  print_function_signature(func_sptr, get_llvm_name(func_sptr), abi, true);

  /* As of LLVM 3.8 the DISubprogram metadata nodes no longer bear
   * 'function' members that address the code for the subprogram.
   * Now, the references are reverse, the function definition carries
   * a !dbg metadata reference to the subprogram.
   */
  if (module && module->debug_info &&
      ll_feature_debug_info_ver38(&module->ir)) {
    LL_MDRef subprogram = lldbg_subprogram(module->debug_info);
    if (!LL_MDREF_IS_NULL(subprogram)) {
      print_dbg_line_no_comma(subprogram);
    }
  }

  print_line(" {\nL.entry:"); /* } so vi matches */

#ifdef CONSTRUCTORG
  if (CONSTRUCTORG(func_sptr)) {
    llvm_ctor_add_with_priority(get_llvm_sname(func_sptr),
                                PRIORITYG(func_sptr));
  }
#endif
#ifdef DESTRUCTORG
  if (DESTRUCTORG(func_sptr)) {
    llvm_dtor_add_with_priority(get_llvm_sname(func_sptr),
                                PRIORITYG(func_sptr));
  }
#endif

  ll_proto_set_defined_body(ll_proto_key(func_sptr), true);
}

static bool
exprjump(ILI_OP opc)
{
  switch (opc) {
  case IL_UKCJMP:
  case IL_KCJMP:
  case IL_ICJMP:
  case IL_FCJMP:
  case IL_DCJMP:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMP:
#endif
  case IL_ACJMP:
  case IL_UICJMP:
    return true;
  default:
    return false;
  }
}

static bool
zerojump(ILI_OP opc)
{
  switch (opc) {
  case IL_KCJMPZ:
  case IL_UKCJMPZ:
  case IL_ICJMPZ:
  case IL_FCJMPZ:
  case IL_DCJMPZ:
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_QCJMPZ:
#endif
  case IL_ACJMPZ:
  case IL_UICJMPZ:
    return true;
  default:
    return false;
  }
}

/**
   \brief Get the string representation of the type of \p sptr or \p dtype
   \param sptr  symbol to use for type consing or 0
   \param dtype used when \p sptr not provided
 */
const char *
char_type(DTYPE dtype, SPTR sptr)
{
  LL_Type *ty;

  if (sptr && (DTYPEG(sptr) == dtype)) {
    ty = make_lltype_from_sptr(sptr);
    if (need_ptr(sptr, SCG(sptr), dtype))
      ty = ty->sub_types[0];
  } else {
    ty = make_lltype_from_dtype(dtype);
  }
  return ty->str;
}

/**
   \brief Update the shadow symbol arrays

   When adding new symbols or starting a new routine, make sure the shadow
   symbol arrays and dtype debug array are updated.
 */
static void
update_llvm_sym_arrays(void)
{
  if ((flg.debug || XBIT(120, 0x1000)) && cpu_llvm_module) {
    lldbg_update_arrays(cpu_llvm_module->debug_info, llvm_info.last_dtype_avail,
                        stb.dt.stg_avail + MEM_EXTRA);
  }
}

void
cg_llvm_init(void)
{
  int i;
  const char *triple = "";
  enum LL_IRVersion ir_version;

  if (init_once) {
    update_llvm_sym_arrays();
    return;
  }
  ll_proto_init();
  routine_count = 0;

  CHECK(TARGET_PTRSIZE == size_of(DT_CPTR));

  if (flg.llvm_target_triple)
    triple = flg.llvm_target_triple;
  else
    triple = LLVM_DEFAULT_TARGET_TRIPLE;

  ir_version = get_llvm_version();

  if (!cpu_llvm_module)
    cpu_llvm_module = ll_create_module(gbl.file_name, triple, ir_version);
#ifdef OMP_OFFLOAD_LLVM
  if (flg.omptarget && !gpu_llvm_module) {
    gpu_llvm_module =
        ll_create_module(gbl.file_name, ompaccel_get_targetriple(), ir_version);
  }
#endif
  llvm_info.declared_intrinsics = hashmap_alloc(hash_functions_strings);

  llvm_info.homed_args = hashmap_alloc(hash_functions_direct);

#if DEBUG
  ll_dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
#endif

  llvm_info.last_dtype_avail = stb.dt.stg_avail + 2000;
  /* set up sptr array - some extra for symbols that may need to be added */
  /* last_sym_avail is used for all the arrays below */
  llvm_info.last_sym_avail = stb.stg_avail + MEM_EXTRA;

  if (sptrinfo.array.stg_base) {
    STG_CLEAR_ALL(sptrinfo.array);
    STG_CLEAR_ALL(sptrinfo.type_array);
  } else {
    STG_ALLOC_SIDECAR(stb, sptrinfo.array);
    /* set up the type array shadowing the symbol table */
    STG_ALLOC_SIDECAR(stb, sptrinfo.type_array);
  }

  Globals = NULL;
  recorded_Globals = NULL;

  /* get a count of the number of routines in this file */
  for (i = gbl.entries; i > NOSYM; i = SYMLKG(i)) {
    routine_count++;
  }

  entry_bih = gbl.entbih;
#if DEBUG
  if (DBGBIT(12, 0x10)) {
    indent(0);
    if (routine_count)
      fprintf(ll_dfile, "# %d routines in file %s\n", routine_count,
              entry_bih ? FIH_FILENAME(BIH_FINDEX(entry_bih))
                        : "FILENAME(gbl.entbih) NOT SET");
    else
      fprintf(ll_dfile, "# no routine in file\n");
  }
#endif

  if (flg.debug || XBIT(120, 0x1000)) {
    lldbg_init(cpu_llvm_module);
#ifdef OMP_OFFLOAD_LLVM
    if (flg.omptarget && XBIT(232, 0x8))
      lldbg_init(gpu_llvm_module);
#endif
  }

  init_once = true;
  assem_init();
  if (!ftn_init_once && FTN_HAS_INIT() == 0)
    init_output_file();
#ifdef OMP_OFFLOAD_LLVM
  init_gpu_output_file();
#endif
  ftn_init_once = true;

  write_ftn_typedefs();
} /* cg_llvm_init */

/**
   \brief Process the end of the file (Fortran)

   Dumps the metadata for the Module.
 */
void
cg_llvm_end(void)
{
  write_function_attributes();
  ll_write_metadata(llvm_file(), cpu_llvm_module);
#ifdef OMP_OFFLOAD_LLVM
  if (flg.omptarget) {
    ll_write_metadata(llvm_file(), gpu_llvm_module);
    ll_build_metadata_device(gbl.ompaccfile, gpu_llvm_module);
    ll_write_metadata(gbl.ompaccfile, gpu_llvm_module);
  }
#endif
}

/**
   \brief Process the end of the SUBROUTINE (Fortran)

   In Fortran, we carry over data from the LONGTERM_AREA to the next subroutine
   to be processed.
 */
void
cg_llvm_fnend(void)
{
  if (!init_once) {
    cg_llvm_init();
  }
  write_global_and_static_defines();
  write_ftn_typedefs();
  Globals = NULL;

  /* Note that this function is called for every routine.  */
  assem_end();
  init_once = false;
  llutil_struct_def_reset();
  ll_reset_module_types(cpu_llvm_module);

  recorded_Globals = NULL;
  SYMLKP(gbl.entries, NOSYM);

  freearea(CG_MEDTERM_AREA);
}

bool
is_cg_llvm_init(void)
{
  return init_once;
}

/**
   \brief Insert the jump entry instruction

   Insert compare and jump instruction to correct "entry" based on the first
   argument of the routine.
 */
static void
insert_jump_entry_instr(int ilt)
{
  SPTR sptr, lab, sym;
  SPTR *dpdscp;
  INT val = 0;

  if (!has_multiple_entries(gbl.currsub))
    return;

  dpdscp = (SPTR *)(aux.dpdsc_base + DPDSCG(master_sptr));
  sym = *dpdscp;
  assert(hashmap_lookup(llvm_info.homed_args, INT2HKEY(sym), NULL),
         "Expected homed master-entry-choice sptr", sym, ERR_Fatal);

  for (sptr = (SPTR)gbl.entries; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    /* The first arg (choice) is homed via process_formal_arguments() */
    INSTR_LIST *Curr_Instr;
    OPERAND *choice_op = make_var_op(sym);
    OPERAND *load_op = gen_load(choice_op, make_lltype_from_dtype(DT_INT),
                                ldst_instr_flags_from_dtype(DT_INT));

    OPERAND *operand = make_tmp_op(make_int_lltype(1), make_tmps());
    operand->tmps->use_count++;
    Curr_Instr =
        gen_instr(I_ICMP, operand->tmps, operand->ll_type, make_operand());
    Curr_Instr->operands->ot_type = OT_CC;
    Curr_Instr->operands->val.cc = convert_to_llvm_intcc(CC_EQ);
    Curr_Instr->operands->ll_type = make_type_from_opc(IL_ICMP);
    Curr_Instr->operands->next = load_op;
    Curr_Instr->operands->next->next =
        gen_llvm_expr(ad_icon(val), make_lltype_from_dtype(DT_INT));
    ad_instr(0, Curr_Instr);
    val++;

    lab = getlab();
    Curr_Instr = make_instr(I_BR);
    Curr_Instr->operands = make_tmp_op(make_int_lltype(1), operand->tmps);

    Curr_Instr->operands->next = make_target_op(sptr);
    Curr_Instr->operands->next->next = make_target_op(lab);
    ad_instr(0, Curr_Instr);

    /* label lab: */
    Curr_Instr = gen_instr(I_NONE, NULL, NULL, make_label_op(lab));
    ad_instr(0, Curr_Instr);
  }
}

static void
insert_entry_label(int ilt)
{
  const int ilix = ILT_ILIP(ilt);
  SPTR sptr = ILI_SymOPND(ilix, 1);
  INSTR_LIST *Curr_Instr = gen_instr(I_NONE, NULL, NULL, make_label_op(sptr));
  ad_instr(0, Curr_Instr);
  llvm_info.return_ll_type = make_lltype_from_dtype(
      gbl.arets ? DT_INT : get_return_type(sptr)); // FIXME: possible bug
}

void
reset_expr_id(void)
{
  expr_id = 0;
}

static void
store_return_value_for_entry(OPERAND *p, int i_name)
{
  const LL_Type *retTy;
  TMPS *new_tmps;
  // extract the scalar value if datatype is a pointer type
  if (p->ll_type->data_type == LL_PTR) {
    retTy = p->ll_type->sub_types[0];
    assert((retTy->data_type  != LL_PTR), 
	    "scalar type is expected : got %d ", LL_PTR, ERR_Fatal);
    new_tmps = make_tmps();
    print_token("\t");
    print_tmp_name(new_tmps);
    print_space(1);
    print_token("=");
    print_space(1);
    print_token("load ");
    write_type(retTy);
    print_token(", ");
    write_type(p->ll_type);
    write_operand(p, "", FLG_OMIT_OP_TYPE);
    print_token(", align 4\n");
  }

  print_token("\tstore ");
  if (p->ll_type->data_type == LL_PTR) {
    write_type(retTy);
    print_space(1);
    print_tmp_name(new_tmps);
  } else {
   write_type(p->ll_type);
   print_space(1);
   write_operand(p, "", FLG_OMIT_OP_TYPE);
  }
  print_token(", ");
  write_type(make_ptr_lltype(p->ll_type));
  print_token(" %");
  print_token(get_entret_arg_name());
  print_token(", align 4\n");

  print_token("\t");
  print_token(llvm_instr_names[i_name]);
  print_token(" void");
}

/*
 * Global initialization and finalization routines
 */

#define LLVM_DEFAULT_PRIORITY 65535

typedef struct init_node {
  const char *name;
  int priority;
  struct init_node *next;
} init_node;

typedef struct init_list_t {
  struct init_node *head;
  struct init_node *tail;
  int size;
} init_list_t;

static init_list_t llvm_ctor_list;
static init_list_t llvm_dtor_list;

static void
llvm_add_to_init_list(const char *name, int priority, init_list_t *list)
{
  init_node *node = (init_node *)malloc(sizeof(init_node));
  node->name = llutil_strdup(name);
  if (priority < 0 || priority > LLVM_DEFAULT_PRIORITY) {
    priority = LLVM_DEFAULT_PRIORITY;
  }
  node->priority = priority;
  node->next = NULL;

  if (list->head == NULL) {
    list->head = node;
    list->tail = node;
  } else {
    list->tail->next = node;
    list->tail = node;
  }
  ++(list->size);
}

void
llvm_ctor_add(const char *name)
{
  llvm_add_to_init_list(name, LLVM_DEFAULT_PRIORITY, &llvm_ctor_list);
}

void
llvm_ctor_add_with_priority(const char *name, int priority)
{
  llvm_add_to_init_list(name, priority, &llvm_ctor_list);
}

void
llvm_dtor_add(const char *name)
{
  llvm_add_to_init_list(name, LLVM_DEFAULT_PRIORITY, &llvm_dtor_list);
}

void
llvm_dtor_add_with_priority(const char *name, int priority)
{
  llvm_add_to_init_list(name, priority, &llvm_dtor_list);
}

static void
llvm_write_ctor_dtor_list(init_list_t *list, const char *global_name)
{
  struct init_node *node;
  char int_str_buffer[20];

  if (list->size == 0)
    return;

  print_token("@");
  print_token(global_name);
  print_token(" = appending global [");
  sprintf(int_str_buffer, "%d", list->size);
  print_token(int_str_buffer);

  if (ll_feature_three_argument_ctor_and_dtor(&current_module->ir)) {
    print_token(" x { i32, ptr, ptr }][");
    for (node = list->head; node != NULL; node = node->next) {
      print_token("{ i32, ptr, ptr } { i32 ");
      sprintf(int_str_buffer, "%d", node->priority);
      print_token(int_str_buffer);
      print_token(", ptr @");
      print_token(node->name);
      print_token(", ptr null }");
      if (node->next != NULL) {
        print_token(", ");
      }
    }
  } else {
    print_token(" x { i32, ptr }][");
    for (node = list->head; node != NULL; node = node->next) {
      print_token("{ i32, ptr } { i32 ");
      sprintf(int_str_buffer, "%d", node->priority);
      print_token(int_str_buffer);
      print_token(", ptr @");
      print_token(node->name);
      print_token(" }");
      if (node->next != NULL) {
        print_token(", ");
      }
    }
  }

  print_token("]");
  print_nl();
}

void
llvm_write_ctors()
{
  llvm_write_ctor_dtor_list(&llvm_ctor_list, "llvm.global_ctors");
  llvm_write_ctor_dtor_list(&llvm_dtor_list, "llvm.global_dtors");
}

void
cg_fetch_clen_parampos(SPTR *len, int *param, SPTR sptr)
{
  if (llvm_info.abi_info) {
    *len = CLENG(sptr);
    for (unsigned i = 1; i <= llvm_info.abi_info->nargs; ++i)
      if (llvm_info.abi_info->arg[i].sptr == *len) {
        *param = i;
        return;
      }
  }
  *param = -1; /* param not found */
}

/**
   \brief Helper function: test if the param length exists as compiler created
          symbol which represents length of any assumed length string argument
          in the arg list
   \param length is the compiler created symbol which represents length of
          assumed length string argument
 */
bool
clen_parent_is_param(SPTR length)
{
  int i;
  SPTR parent;
  for (i = 1; i <= llvm_info.abi_info->nargs; ++i) {
    parent = llvm_info.abi_info->arg[i].sptr;
    if ((DTY(DTYPEG(parent)) == TY_CHAR) && (DTYPEG(parent) == DT_ASSCHAR) &&
        (CLENG(parent) == length)) return true;
  }
  return false;
}

void
add_debug_cmnblk_variables(LL_DebugInfo *db, SPTR sptr)
{
  static hashset_t sptr_added;
  SPTR scope, var;
  const char *debug_name;
  bool has_alias = false;

  if (!sptr_added)
    sptr_added = hashset_alloc(hash_functions_strings);
  scope = SCOPEG(sptr);
  for (var = CMEMFG(sptr); var > NOSYM; var = SYMLKG(var)) {
    if ((!SNAME(var)) || strcmp(SNAME(var), SYMNAME(var))) {
      if (gbl.rutype != RU_BDATA && NEEDMODG(scope) &&
          lookup_modvar_alias(var)) {
        has_alias = true;
        break;
      }
    }
  }
  if (gbl.rutype != RU_BDATA && NEEDMODG(scope) &&
      !RESTRICTEDG(sptr)) {
    /* This is a MODULE to be imported to a subroutine
     * later in lldbg_emit_subprogram(). */
    lldbg_add_pending_import_entity(db, scope, IMPORTED_MODULE);
  }
  for (var = CMEMFG(sptr); var > NOSYM; var = SYMLKG(var)) {
    if ((!SNAME(var)) || strcmp(SNAME(var), SYMNAME(var))) {
      if (CCSYMG(sptr)) {
        debug_name = new_debug_name(SYMNAME(scope), SYMNAME(var), NULL);
      } else {
        debug_name = new_debug_name(SYMNAME(scope), SYMNAME(sptr),
                                    SYMNAME(var));
      }
      if (gbl.rutype != RU_BDATA && NEEDMODG(scope) &&
          (lookup_modvar_alias(var) ||
           (has_alias && !RESTRICTEDG(sptr)))) {
        /* This is a DECLARATION to be imported to a subroutine
         * later in lldbg_emit_subprogram().
         * Case 1: Aliased members of restricted / non-restricted modules.
         * Case 2: All members of non-restricted module if it has atlease one
         *         aliased member. */
        if (!ll_feature_debug_info_ver11(&cpu_llvm_module->ir) ||
            (RESTRICTEDG(sptr) && lookup_modvar_alias(var))) {
          lldbg_add_pending_import_entity(db, var, IMPORTED_DECLARATION);
        } else if (!RESTRICTEDG(sptr) && lookup_modvar_alias(var))
          lldbg_add_pending_import_entity_to_child(db, var,
                                                   IMPORTED_DECLARATION);
      }
      if (hashset_lookup(sptr_added, debug_name))
        continue;
      hashset_insert(sptr_added, debug_name);
      const char *save_ptr = SNAME(var);
      SNAME(var) = SYMNAME(var);
      addDebugForGlobalVar(var, variable_offset_in_aggregate(var, 0));
      SNAME(var) = save_ptr;
    }
  }
}

/**
   \brief Process symbols with global lifetime and cons their metadata
 */
void
process_global_lifetime_debug(void)
{
  bool host_version = true;
  if (cpu_llvm_module->global_debug_map)
    hashmap_clear(cpu_llvm_module->global_debug_map);
  if (cpu_llvm_module->debug_info && gbl.cmblks) {
    LL_DebugInfo *db = cpu_llvm_module->debug_info;
    SPTR sptr = gbl.cmblks;
    update_llvm_sym_arrays();
    lldbg_reset_module(db);
    if (gbl.currsub>NOSYM) {
       if (CUDAG(gbl.currsub) && 
	   !(CUDAG(gbl.currsub) & CUDA_HOST)) {
         host_version = false;
       }
    }
    if (!gbl.cuda_constructor &&
        host_version) {
      for (; sptr > NOSYM; sptr = SYMLKG(sptr)) {
        const SPTR scope = SCOPEG(sptr);
        if (scope > 0) {
          if (CCSYMG(sptr)) {
            lldbg_emit_module_mdnode(db, scope);
            add_debug_cmnblk_variables(db, sptr);
          } else {
            if (FROMMODG(sptr) || (gbl.rutype == RU_BDATA && scope == gbl.currsub)) {
              lldbg_emit_common_block_mdnode(db, sptr);
              add_debug_cmnblk_variables(db, sptr);
            }
          }
        }
      }
    }
  }
}


bool 
is_vector_x86_mmx(LL_Type *type) {
  /* Check if type is a vector with 64 bits overall. Works on pointer types. */
  LL_Type *t = type;
  if (type->data_type == LL_PTR) {
    t = *type->sub_types;
  }
  if (t->data_type == LL_VECTOR &&
      (strcmp(t->str, "<1 x i64>") == 0 ||
       strcmp(t->str, "<2 x i32>") == 0 ||
       strcmp(t->str, "<4 x i16>") == 0 ||
       strcmp(t->str, "<8 x i8>") == 0 ||
       strcmp(t->str, "<1 x double>") == 0 ||
       strcmp(t->str, "<2 x float>") == 0)) {
    return true;
  }
  return false;
}

int
get_parnum(SPTR sptr)
{
  for (unsigned parnum = 1; parnum <= llvm_info.abi_info->nargs; parnum++) {
    if (llvm_info.abi_info->arg[parnum].sptr == sptr) {
      return parnum;
    }
  }

  return 0;
}

int
get_entry_parnum(SPTR sptr)
{
  for (unsigned parnum = 1; parnum <= entry_abi->nargs; parnum++) {
    if (entry_abi->arg[parnum].sptr == sptr) {
      return parnum;
    }
  }

  return 0;
}

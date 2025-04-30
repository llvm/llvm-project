/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LLUTIL_H_
#define LLUTIL_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ll_structure.h"
#include "ili.h"

/** \brief need a getitem() area that can persist across routine compilation */
#define LLVM_LONGTERM_AREA 25
#define BITS_IN_BYTE 8

/** \brief OPERAND flag values */
typedef enum OperandFlag_t {
  OPF_NONE,
  OPF_WRAPPED_MD = (1 << 0),
  OPF_NULL_TYPE = (1 << 1),
  OPF_SRET_TYPE = (1 << 2),
  OPF_SRARG_TYPE = (1 << 3),
  OPF_SEXT = (1 << 4),
  OPF_ZEXT = (1 << 5),
  OPF_VOLATILE = (1 << 6),
  OPF_HIDDEN = (1 << 7),
  OPF_CONTAINS_UNDEF = (1 << 8),
} OperandFlag_t;

typedef enum OperandOutputFlag_t {
  FLG_OMIT_OP_TYPE = (1 << 0),
  FLG_AS_UNSIGNED = (1 << 1),
  FLG_FIXUP_RETURN_TYPE = (1 << 2),
  FLG_INNER_TYPED_COMPLEX = (1 << 3)
} OperandOutputFlag_t;

#define LLTNAMEG(llt) llvm_type_names[(llt)->kind]

/** \brief OPERAND types */
typedef enum OperandType_t {
  OT_NONE, /**< must be first (0) member */
  OT_CONSTSPTR,
  OT_VAR,
  OT_TMP,
  OT_LABEL,
  OT_CC,
  OT_TARGET,
  OT_CALL,
  OT_CONSTVAL,
  OT_UNDEF,
  OT_MDNODE,
  OT_MEMBER,
  OT_DEF,
  OT_CONSTSTRING,
  OT_LAST /**< must be last member */
} OperandType_t;

#if DEBUG
const char *get_ot_name(unsigned ot);
#define OTNAMEG(p) get_ot_name((p)->ot_type)
#endif

/** \brief LLVM integer condition codes */
typedef enum LLIntegerConditionCodes {
  LLCC_NONE,
  LLCC_EQ,  /**< equal */
  LLCC_NE,  /**< not equal */
  LLCC_UGT, /**< usigned greater than */
  LLCC_UGE, /**< unsigned greater or equal */
  LLCC_ULT, /**< unsigned less than */
  LLCC_ULE, /**< unsigned less or equal */
  LLCC_SGT, /**< signed greater than */
  LLCC_SGE, /**< signed greater or equal */
  LLCC_SLT, /**< signed less than */
  LLCC_SLE, /**< signed less or equal */
  LLCC_LAST
} LLIntegerConditionCodes;

/** \brief LLVM fp condition codes */
typedef enum LLFloatingPointConditionCodes {
  LLCCF_NONE,
  LLCCF_FALSE, /**< always false */
  LLCCF_OEQ,   /**< ordered and equal */
  LLCCF_OGT,   /**< ordered and greater than */
  LLCCF_OGE,   /**< ordered and greater than or equal */
  LLCCF_OLT,   /**< ordered and less than */
  LLCCF_OLE,   /**< ordered and less than or equal */
  LLCCF_ONE,   /**< ordered and not equal */
  LLCCF_ORD,   /**< ordered (no nans) */
  LLCCF_UEQ,   /**< unordered or equal */
  LLCCF_UGT,   /**< unordered or greater than */
  LLCCF_UGE,   /**< unordered or greater than or equal */
  LLCCF_ULT,   /**< unordered or less than */
  LLCCF_ULE,   /**< unordered or less than or equal */
  LLCCF_UNE,   /**< unordered or not equal */
  LLCCF_UNO,   /**< unordered (either nans) */
  LLCCF_TRUE,  /**< always true */
  LLCCF_LAST
} LLFloatingPointConditionCodes;

/** \brief Dimension information */
typedef struct LLDimInfo {
  int size; /**< size of this dimension */
  int sptr; /**< corresponding sptr */
  int dim;  /**< dimension number */
  /** associated dtype - char_type(dtype) gives sub-array type as string */
  int dtype;
} DIM_INFO;

/** \brief Data initialization record */
typedef struct DINIT_REC {
  int dtype; /**< first dtype encountered, so if array then type of elements */
  ISZ_T conval;
  int offset;
  int len;
  char *payload;
  struct DINIT_REC *next;
} DINIT_REC;

/** \brief Symbol pointer drec */
typedef struct {
  int sptr;          /**< symbol sptr */
  int offset;        /**< offset value */
  int dtype;         /**< dtype of record entries */
  int pad;           /**< padding before sptr ? */
  DINIT_REC *record; /**< link to dinit records */
} SPTR_DREC;

/** \brief Node type in a list of globals */
typedef struct GBL_TAG {
  SPTR sptr;          /**< sptr of the variable */
  unsigned alignment; /**< in bytes */
  char *global_def;   /**< global definition */
  struct GBL_TAG *next;
} GBL_LIST;

/** \brief Extern function list node */
typedef struct EXFUNC_LIST {
  SPTR sptr;                /**< sptr of the variable */
  SPTR ha_sptr;             /**< sptr of hidden structure argument, if present */
  const char *func_def;     /**< external function definition */
  unsigned flags;           /**< details about the external function */
  DTYPE use_dtype;          /**< the dtype to use when generating prototypes */
  struct EXFUNC_LIST *next;
} EXFUNC_LIST;

struct INSTR_TAG;

typedef struct TMP_TAG {
  int id;		/**< assigned temporary id - current expr_id */
  union {
    struct INSTR_TAG *idef; /**< pointer to instr that define this tmp */
    char *string;	/**< string definition for named metadata (LLVM debug
                           format 2.9) */
  } info;
  int use_count;	/**< use_count for tmp value */
} TMPS;

typedef struct OPERAND {
  OperandType_t ot_type; /**< operand type */
  TMPS *tmps;            /**< for OT_TMP types, the corresponding temporary */
  LL_Type *ll_type;      /**< operand type */
  union {
    int cc;        /**< condition code value */
    SPTR sptr;      /**< sptr value */
    INT conval[4]; /**< constant value */
    struct {
      SPTR sptr;
      unsigned long long undef_mask;
    } sptr_undef;
  } val;
  const char *string;   /**< hold routine name for llvm intrinsic calls */
  unsigned flags;       /**< dependent on operand */
  struct OPERAND *next; /**< link to next in list */
} OPERAND;

/**
   \brief LLVM instructions
 */
typedef enum LL_InstrName {
  I_NONE,	/**< must be 0 */
  I_RET,
  I_BR,
  I_SW,
  I_INVOKE,
  I_UNWIND,
  I_UNREACH,
  I_ADD,
  I_FADD,
  I_SUB,
  I_FSUB,
  I_MUL,
  I_FMUL,
  I_UDIV,
  I_SDIV,
  I_FDIV,
  I_UREM,
  I_SREM,
  I_FREM,
  I_SHL,
  I_LSHR,
  I_ASHR,
  I_AND,
  I_OR,
  I_XOR,
  I_EXTELE,
  I_INSELE,
  I_SHUFFVEC,
  I_EXTRACTVAL,
  I_INSERTVAL,
  I_MALLOC,
  I_FREE,
  I_ALLOCA,
  I_LOAD,
  I_STORE,
  I_GEP,
  I_TRUNC,
  I_ZEXT,
  I_SEXT,
  I_FPTRUNC,
  I_FPEXT,
  I_FPTOUI,
  I_FPTOSI,
  I_UITOFP,
  I_SITOFP,
  I_PTRTOINT,
  I_INTTOPTR,
  I_BITCAST,
  I_ICMP,
  I_FCMP,
  I_VICMP,
  I_VFCMP,
  I_PHI,
  I_SELECT,
  I_CALL,
  I_VA_ARG,
  I_DECL,
  I_LANDINGPAD,
  I_RESUME,
  I_CLEANUP,
  I_CATCH,
  I_BARRIER,
  I_ATOMICRMW,
  I_CMPXCHG,
  I_FENCE,
  I_PICALL,
  I_INDBR,
  I_FILTER,
  I_LAST	/**< must be last in enum */
} LL_InstrName;

/* clang-format off */

/**
   \brief INSTR_LIST flag values
 */
typedef enum LL_InstrListFlags {
  InstrListFlagsNull,
  VAR_ARGS_FLAG       = (1 << 0),
  CALL_FUNC_PTR_FLAG  = (1 << 1),
  CALL_INTRINSIC_FLAG = (1 << 2),
  HIDDEN_ARG_FLAG     = (1 << 3),
  LOOP_BACKEDGE_FLAG  = (1 << 4), /**< I_BR only */
  FAST_MATH_FLAG      = (1 << 4), /**< I_CALL only */
  VOLATILE_FLAG       = (1 << 4), /**< I_LOAD, I_STORE, I_ATOMICRMW,
                                       I_CMPXCHG only */
  NSZ_MATH_FLAG       = (1 << 5),
  REASSOC_MATH_FLAG   = (1 << 6),

  FAST_CALL               = (1 << 7),

  ARM_AAPCS               = (1 << 8),
  ARM_AAPCS_VFP           = (1 << 9),
  CANCEL_CALL_DBG_VALUE   = (1 << 10),
  NOSIGNEDWRAP            = (1 << 11),
  NOUNSIGNEDWRAP          = (1 << 12),
  FUNC_RETURN_IS_FUNC_PTR = (1 << 13),
  LDST_HAS_ACCESSGRP_METADATA = (1 << 14), /**< I_LOAD, I_STORE only, for llvm.loop.parallel_accesses */

  /* Information for atomic operations.
     This information overlaps 12 of the calling convention bits.  In earlier
     versions of the code, these were one-per-bit flags, hence the suffix
     "FLAG".  The "flags" are all non-zero values so that testing for the
     presence of any of them can be fast.

     For load/store instructions, presence of a memory ordering flag will cause
     the "atomic" modifier to be printed too. */

  /* 4 bits for the rmw operation */
          ATOMIC_RMW_OP_FLAGS = (0xF << 13),
  ATOMIC_XCHG_FLAG = (1 << 13),
  ATOMIC_ADD_FLAG  = (2 << 13),
  ATOMIC_SUB_FLAG  = (3 << 13),
  ATOMIC_AND_FLAG  = (4 << 13),
  ATOMIC_NAND_FLAG = (5 << 13),
  ATOMIC_OR_FLAG   = (6 << 13),
  ATOMIC_XOR_FLAG  = (7 << 13),
  ATOMIC_MAX_FLAG  = (8 << 13),
  ATOMIC_MIN_FLAG  = (9 << 13),
  ATOMIC_UMAX_FLAG = (10 << 13),
  ATOMIC_UMIN_FLAG = (11 << 13),

  /* 1 bit for singlethread aspect of memory order */
          ATOMIC_SINGLETHREAD_FLAG = (1 << 17),

  /* 3 bits for the memory order (if cmpxchg, then success case) */
          ATOMIC_MEM_ORD_FLAGS = (0x7 << 18),
  ATOMIC_MONOTONIC_FLAG = (1 << 18),
  ATOMIC_ACQUIRE_FLAG = (3 << 18),
  ATOMIC_RELEASE_FLAG = (4 << 18),
  ATOMIC_ACQ_REL_FLAG = (5 << 18),
  ATOMIC_SEQ_CST_FLAG = (6 << 18),

  /* 3 bits for the memory order for failed cmpxchg.  Use macros
     TO_CMPXCHG_MEMORDER_FAIL and FROM_CMPXCHG_MEMORDER_FAIL to access them. */
          ATOMIC_MEM_ORD_FAIL_FLAGS = (0x7 << 21),

  /* 1 bit for marking "weak" cmpxchg. */
          CMPXCHG_WEAK_FLAG = (1 << 24),

  DELETABLE          = (1 << 25),
  STARTEBB           = (1 << 26),
  ROOTDG             = (1 << 27),
  INST_ADDED         = (1 << 28),
  INST_VISITED       = (1 << 29),
  NOUNWIND_CALL_FLAG = (1 << 30),
  INST_LIST_FLAGS_MAX = NOUNWIND_CALL_FLAG + (NOUNWIND_CALL_FLAG - 1)
} LL_InstrListFlags;

#ifdef __cplusplus
// determine a type that is lage enough to cast to and from
#include <climits>
#if INSTR_LIST_FLAGS_MAX <= UINT_MAX
#define ILFint unsigned
#elif INSTR_LIST_FLAGS_MAX <= ULLONG_MAX
#define ILFint unsigned long long
#else
#error "LL_InstrListFlags is larger than 'unsigned long long'"
#endif

inline LL_InstrListFlags operator|=(LL_InstrListFlags& set,
                                    LL_InstrListFlags f) {
  set = static_cast<LL_InstrListFlags>(static_cast<ILFint>(set) |
                                       static_cast<ILFint>(f));
  return set;
}

inline LL_InstrListFlags operator&=(LL_InstrListFlags& set,
                                    LL_InstrListFlags f) {
  set = static_cast<LL_InstrListFlags>(static_cast<ILFint>(set) &
                                       static_cast<ILFint>(f));
  return set;
}

inline LL_InstrListFlags operator~(const LL_InstrListFlags& set) {
  return static_cast<LL_InstrListFlags>(~static_cast<ILFint>(set));
}

#undef ILFint
#endif

#define TO_CMPXCHG_MEMORDER_FAIL(flags) ((LL_InstrListFlags)((flags)<<3))
#define FROM_CMPXCHG_MEMORDER_FAIL(flags) ((LL_InstrListFlags) \
                                           ((flags)>>3 & ATOMIC_MEM_ORD_FLAGS))

#define LDST_LOGALIGN_SHIFT 5
/* log2(alignment) is encoded in three bits. See
   ll_logalign_flags_from_dtype(). */
#define LDST_LOGALIGN_MASK (7 << LDST_LOGALIGN_SHIFT)

/* convert access alignment from log2 encoding to number of bytes */
#define LDST_BYTEALIGN(flags) \
  (1U << (((flags) & LDST_LOGALIGN_MASK) >> LDST_LOGALIGN_SHIFT))

#define CALLCONV_SHIFT 14
/* Calling convention encoded in 8 bits on call instructions in the
   LL_InstrListFlags. See enum LL_CallConv in ll_structure.h */
#define CALLCONV_MASK (0xff << CALLCONV_SHIFT)

/* clang-format on */

typedef struct INSTR_TAG {
  int rank;             /**< instruction rank for scheduling */
  LL_InstrName i_name;  /**< see LL_InstrName */
  int ilix;      /**< original ilix for instruction, required for LOAD/STORE */
  LL_InstrListFlags flags; /**< dependent on instruction */
  TMPS *tmps;           /**< used to hold intermediate results */
  LL_Type *ll_type;     /**< type of intermediate results */
  OPERAND *operands;    /**< list of instruction operands */
  LL_MDRef dbg_line_op; /**< line info for debug */
  LL_MDRef misc_metadata;
#if DEBUG
  const char *traceComment;
#endif
  struct INSTR_TAG *prev;
  struct INSTR_TAG *next;
} INSTR_LIST;

/* old-style FUNCTION tag */
#define OLD_STYLE_FUNC -99

#define INSTR_PREV(i) ((i)->prev)
#define INSTR_NEXT(i) ((i)->next)
#define INSTR_IS_TERMINAL(i) \
    ((i)->i_name == I_BR || (i)->i_name == I_SW || (i)->i_name == I_RET || \
     (i)->i_name == I_RESUME || (i)->i_name == I_UNREACH)

typedef struct EXPR_STK_TAG {
  int tmp_id;
  INSTR_LIST *instr;
  int sptr;
  char *tmp_name;
  TMPS *tmps;
  int i_type; /* instruction type: I_LOAD, I_STORE, ... */
  struct EXPR_STK_TAG *next;
} EXPR_STK;

typedef struct STRUCT_UNION_TAG {
  int saved; /* save the current_element for restoring when stack pop */
  int dtype;
  bool first;
  int has_bitfield;
  struct STRUCT_UNION_TAG *next;
} SU_STK;

typedef struct CSED_TAG {
  int ilix; /* cse'd ili */
  OPERAND *operand;
  struct CSED_TAG *next;
} CSED_ITEM;

/* union definition used for float types */
union xx_u {
  float ff;
  int ww;
};

typedef enum LLDEF_Flags {
  LLDEF_NONE = 0,
  LLDEF_IS_TYPE        = (1 << 0),
  LLDEF_IS_INITIALIZED = (1 << 1),
  LLDEF_IS_STATIC      = (1 << 2),
  LLDEF_IS_EMPTY       = (1 << 3),
  LLDEF_IS_EXTERNAL    = (1 << 4),
  LLDEF_IS_STRUCT      = (1 << 5),
  LLDEF_IS_ARRAY       = (1 << 6),
  LLDEF_IS_ACCSTRING   = (1 << 7),
  LLDEF_IS_CONST       = (1 << 8),
  LLDEF_IS_UNPACKED_STRUCT = (1 << 9)
} LLDEF_Flags;

LL_Type *get_struct_def_type(char *def_name, LL_Module *module);
void write_struct_defs(void);

/* Routines defined in cgmain.c for now, it will require too much work to move
 * them to llutil.c */
void append_llvm_used(OPERAND *op);
void print_dbg_line_no_comma(LL_MDRef md);
void print_dbg_line(LL_MDRef md);

#if DEBUG
void indent(int);
extern FILE *ll_dfile;


#define DBGXTRACEIN(xsw, in, str)                    \
  if (xsw) {                                         \
    if (in) {                                        \
      indent(1);                                     \
    }                                                \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__); \
  }
#define DBGXTRACEIN1(xsw, in, str, p1)                   \
  if (xsw) {                                             \
    if (in) {                                            \
      indent(1);                                         \
    }                                                    \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__, p1); \
  }
#define DBGXTRACEIN2(xsw, in, str, p1, p2)                   \
  if (xsw) {                                                 \
    if (in) {                                                \
      indent(1);                                             \
    }                                                        \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__, p1, p2); \
  }
#define DBGXTRACEIN3(xsw, in, str, p1, p2, p3)                   \
  if (xsw) {                                                     \
    if (in) {                                                    \
      indent(1);                                                 \
    }                                                            \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__, p1, p2, p3); \
  }
#define DBGXTRACEIN4(xsw, in, str, p1, p2, p3, p4)                   \
  if (xsw) {                                                         \
    if (in) {                                                        \
      indent(1);                                                     \
    }                                                                \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__, p1, p2, p3, p4); \
  }
#define DBGXTRACEIN7(xsw, in, str, p1, p2, p3, p4, p5, p6, p7)              \
  if (xsw) {                                                                \
    if (in) {                                                               \
      indent(1);                                                            \
    }                                                                       \
    fprintf(ll_dfile, "<%s" str "\n", __FUNCTION__, p1, p2, p3, p4, p5, p6, \
            p7);                                                            \
  }
#define DBGXTRACEOUT(xsw, in, str)                   \
  if (xsw) {                                         \
    if (in) {                                        \
      indent(-1);                                    \
    }                                                \
    fprintf(ll_dfile, ">%s" str "\n", __FUNCTION__); \
  }
#define DBGXTRACEOUT1(xsw, in, str, p1)                  \
  if (xsw) {                                             \
    if (in) {                                            \
      indent(-1);                                        \
    }                                                    \
    fprintf(ll_dfile, ">%s" str "\n", __FUNCTION__, p1); \
  }
#define DBGXTRACEOUT2(xsw, in, str, p1, p2)                  \
  if (xsw) {                                                 \
    if (in) {                                                \
      indent(-1);                                            \
    }                                                        \
    fprintf(ll_dfile, ">%s" str "\n", __FUNCTION__, p1, p2); \
  }
#define DBGXTRACEOUT3(xsw, in, str, p1, p2, p3)                  \
  if (xsw) {                                                     \
    if (in) {                                                    \
      indent(-1);                                                \
    }                                                            \
    fprintf(ll_dfile, ">%s" str "\n", __FUNCTION__, p1, p2, p3); \
  }
#define DBGXTRACEOUT4(xsw, in, str, p1, p2, p3, p4)                  \
  if (xsw) {                                                         \
    if (in) {                                                        \
      indent(-1);                                                    \
    }                                                                \
    fprintf(ll_dfile, ">%s" str "\n", __FUNCTION__, p1, p2, p3, p4); \
  }
#define DBGXDUMPLLTYPE(xsw, in, str, llt) \
  if (xsw) {                              \
    if (in) {                             \
      indent(0);                          \
    }                                     \
    fprintf(ll_dfile, str);               \
    if (llt)                              \
      dump_type_for_debug(llt);           \
    fprintf(ll_dfile, "\n");              \
  }
#define DBGXTRACE(xsw, in, str)  \
  if (xsw) {                     \
    if (in) {                    \
      indent(0);                 \
    }                            \
    fprintf(ll_dfile, str "\n"); \
  }
#define DBGXTRACE1(xsw, in, str, p1) \
  if (xsw) {                         \
    if (in) {                        \
      indent(0);                     \
    }                                \
    fprintf(ll_dfile, str "\n", p1); \
  }
#define DBGXTRACE2(xsw, in, str, p1, p2) \
  if (xsw) {                             \
    if (in) {                            \
      indent(0);                         \
    }                                    \
    fprintf(ll_dfile, str "\n", p1, p2); \
  }
#define DBGXTRACE3(xsw, in, str, p1, p2, p3) \
  if (xsw) {                                 \
    if (in) {                                \
      indent(0);                             \
    }                                        \
    fprintf(ll_dfile, str "\n", p1, p2, p3); \
  }
#define DBGXTRACE4(xsw, in, str, p1, p2, p3, p4) \
  if (xsw) {                                     \
    if (in) {                                    \
      indent(0);                                 \
    }                                            \
    fprintf(ll_dfile, str "\n", p1, p2, p3, p4); \
  }
#define DBGXTRACE5(xsw, in, str, p1, p2, p3, p4, p5) \
  if (xsw) {                                         \
    if (in) {                                        \
      indent(0);                                     \
    }                                                \
    fprintf(ll_dfile, str "\n", p1, p2, p3, p4, p5); \
  }
#else
#define DBGXTRACEIN(xsw, in, str) ;
#define DBGXTRACEIN1(xsw, in, str, p1) ;
#define DBGXTRACEIN2(xsw, in, str, p1, p2) ;
#define DBGXTRACEIN3(xsw, in, str, p1, p2, p3) ;
#define DBGXTRACEIN4(xsw, in, str, p1, p2, p3, p4) ;
#define DBGXTRACEIN7(xsw, in, str, p1, p2, p3, p4, p5, p6, p7) ;
#define DBGXTRACEOUT(xsw, in, str) ;
#define DBGXTRACEOUT1(xsw, in, str, p1) ;
#define DBGXTRACEOUT2(xsw, in, str, p1, p2) ;
#define DBGXTRACEOUT3(xsw, in, str, p1, p2, p3) ;
#define DBGXTRACEOUT4(xsw, in, str, p1, p2, p3, p4) ;
#define DBGXDUMPLLTYPE(xsw, in, str, llt) ;
#define DBGXTRACE(xsw, in, str) ;
#define DBGXTRACE1(xsw, in, str, p1) ;
#define DBGXTRACE2(xsw, in, str, p1, p2) ;
#define DBGXTRACE3(xsw, in, str, p1, p2, p3) ;
#define DBGXTRACE4(xsw, in, str, p1, p2, p3, p4) ;
#define DBGXTRACE5(xsw, in, str, p1, p2, p3, p4, p5) ;

#endif

#if defined(TARGET_LLVM_X8632) || defined(TARGET_LLVM_POWER)
#define PASS_STRUCT_BY_REF
#define IS_SMALLT_STRUCT(dtype) (XBIT(121, 0x400000) && DTY(dtype) == TY_CMPLX)
#elif defined(TARGET_LLVM_X8664)
#define PASS_STRUCT_BY_REF
#define IS_SMALLT_STRUCT(dtype) (0)
#elif defined(TARGET_LLVM_ARM)
#define PASS_STRUCT_BY_VALUE
#define IS_SMALLT_STRUCT(dtype) (0)
#else
#define PASS_STRUCT_BY_REF
#define IS_SMALLT_STRUCT(dtype) (XBIT(121, 0x400000) && DTY(dtype) == TY_CMPLX)
#endif

#define ZSIZEOF(dtype) size_of(dtype)

extern SPTR_DREC *sptr_dinit_array;

/* Routines in llutil.h using the ll_structure.h types. */
struct LL_Module;

/* ABI lowering for LLVM.
 *
 * LLVM divides the lowering of function arguments and return values between
 * the front end and the back end. The front end simplifies certain language
 * constructs when it generates LLVM IR prototypes for functions. The back end
 * selects argument registers based on the LLVM IR function prototype and
 * calling convention.
 *
 * - The LLVM code generator will flatten any aggregate function arguments, so
 *   C structs that need to be passed in registers must be coerced to
 *   register-sized arguments by the front end.
 *
 * - A hidden struct return argument must be represented explicitly in LLVM IR
 *   as a pointer function argument with the 'sret' attribute.
 *
 * - Large struct arguments that are passed by reference should be converted
 *   to pointer arguments by the front end.
 *
 * - Large struct arguments that must be copied to the stack may be converted
 *   to 'byval' pointer arguments.
 *
 * - Some calling conventions require small integer arguments to be sign or
 *   zero-extended. Since LLVM IR doesn't have signed/unsigned integer types,
 *   the extension must be requested by the front end by setting the sext/zext
 *   argument attributes.
 *
 * - The 'inreg' attribute can be used to request that an argument is passed
 *   in a register (for x86-32). Other target architectures may use this bit
 *   for other purposes.
 */

/**
   These are the possible ways of lowering a single function argument (or return
   value) to LLVM IR.
 */
enum LL_ABI_ArgKind {
  /* Unknown argument type. This type can't be lowered to LLVM IR. */
          LL_ARG_UNKNOWN,

  /* Argument or return value is passed directly without any translation.
   * The LLVM IR type that is normally used to represent to argument is also
   * used in the function prototype. */
          LL_ARG_DIRECT,

  /* Unsigned integers only: Argument or return value is passed directly,
   * with a zext attribute indicating that it should be zero-extended to the
   * full register. */
          LL_ARG_ZEROEXT,

  /* Signed integers only: Argument or return value is passed directly, with
   * a sext attribute indicating that it should be sign-extended to the full
   * register. */
          LL_ARG_SIGNEXT,

  /* Argument or return value is coerced to one or more register-sized
   * arguments of varying types. The coercion is equivalent to storing the
   * original argument to memory followed by loading the LLVM IR arguments
   * from the same memory.
   *
   * The coercion_type pointer in LL_ABI_ArgInfo describes the sequence of
   * arguments to produce. */
          LL_ARG_COERCE,

  /* Argument should be passed indirectly as a pointer.  The argument is
   * saved to stack memory and a pointer to that memory is passed instead.
   *
   * If this is used for a return value, it means that the caller passes an
   * 'sret' argument pointing to temporary stack space. */
          LL_ARG_INDIRECT,

  /* Argument should be copied in the scope of the caller and then
   * passed indirectly as a pointer.  The argument is
   * copied to stack memory and a pointer to that memory is passed instead.
   *
   * This is used when arguments are passed by value but the ABI only expects 
   * a pointer */
          LL_ARG_INDIRECT_BUFFERED,

  /* Argument is passed by value on the stack.
   *
   * This is represented in LLVM IR as a pointer argument with the 'byval'
   * attribute set. This looks like an indirect argument, but the LLVM code
   * generator will pass the argument by value by copying it to the stack.
   *
   * This option does not apply to return values. */
          LL_ARG_BYVAL
};

/**
   \brief Information about how a single argument should be lowered to LLVM IR.
 */
typedef struct LL_ABI_ArgInfo_ {
  enum LL_ABI_ArgKind kind;

  /* Set the 'inreg' attribute. */
  unsigned inreg;

  /* Fortran pass by value. */
  bool ftn_pass_by_val;

  /* LLVM type of this argument. When kind is LL_ARG_COERCE, points to the
   * coercion type. */
  const struct LL_Type_ *type;

  /* Symbol table entry representing this function argument, if available. */
  SPTR sptr;
} LL_ABI_ArgInfo;

/**
   \brief Information about LLVM lowering the return value and all arguments of
  a function.
 */
typedef struct LL_ABI_Info_ {
  /** The LL_Module containing any referenced types. */
  struct LL_Module *module;

  /** Number of formal function arguments. */
  unsigned nargs;

  /** This is a varargs function. */
  unsigned is_varargs : 1;

  /** This is an old-style declaration without a prototype: int f(). */
  unsigned missing_prototype : 1;

  /** This function should be called as a varargs function, even thouh we
      don't know if it is a varargs function. Typically only used with
      missing_prototype. */
  unsigned call_as_varargs : 1;

  /** This represents a fortran call to a ISO c function. */
  unsigned is_iso_c : 1;

  /** This represents a call within a Fortran program. */
  unsigned is_fortran : 1;

  /** Callee is a pure function */
  unsigned is_pure : 1;

  /** Workaround for X86 ABI, does not sign-extend shorts */
  unsigned extend_abi_return : 1;

  /** This is a fast_math function */
  unsigned fast_math : 1;

  /** This is a nomerge function */
  unsigned is_nomerge : 1;

  /** Calling convention to use. See enum LL_CallConv. */
  unsigned call_conv;

  /** Number of integer registers used for arguments so for, including any
      hidden struct return pointers. */
  unsigned used_iregs;

  /** Number of floating point / vector registers used. */
  unsigned used_fregs;

  /** Lowering information for the return value [0] and arguments [1..nargs].
      This array contains nargs+1 elements.
      A void function will have an LL_ARG_UNKNOWN entry in arg[0]. */
  LL_ABI_ArgInfo arg[1];
} LL_ABI_Info;

/* Allocate an empty ABI instance, call ll_abi_free when done. */
LL_ABI_Info *ll_abi_alloc(struct LL_Module *, unsigned nargs);

/* Free an allocated ABI instance, returns NULL to avoid dangling pointers. */
LL_ABI_Info *ll_abi_free(LL_ABI_Info *);

#ifdef DT_INT /* Use DT_INT to detect whether DTYPE is defined. */
/* Get ABI lowering info for a function dtype (TY_FUNC / TY_PFUNC).
 *
 * Note that this does not provide lowering information for the ellipsis part
 * of a varargs function.
 */
LL_ABI_Info *ll_abi_for_func_dtype(struct LL_Module *, DTYPE dtype);

#endif

/* Get the LLVM return type of a function represented by abi. */
const struct LL_Type_ *ll_abi_return_type(LL_ABI_Info *abi);

/* Get the LLVM function type corresponding to an LL_ABI_Info instance. */
const struct LL_Type_ *ll_abi_function_type(LL_ABI_Info *abi);

/* Does this function use an sret argument to return a struct? */
#define LL_ABI_HAS_SRET(abi) ((abi)->arg[0].kind == LL_ARG_INDIRECT)

/* Create an LL_Type corresponding to an ABI argument.
 */
LL_Type *make_lltype_from_abi_arg(LL_ABI_ArgInfo *arg);

/* Target-specific low-level interface for ABI lowering.
 *
 * These functions will be called in this order:
 *
 * 1. abi is allocated and nargs, is_varargs, and missing_prototype is
 * initialized.
 * 2. ll_abi_compute_call_conv().
 * 3. ll_abi_classify_return_dtype().
 * 4. for i in 1..nargs: ll_abi_classify_arg_dtype(arg[i]).
 * 5. for extra arguments passed: ll_abi_classify_arg_dtype().
 */

/* Determine the calling convention to use for calling func_sptr.
 *
 * This target-dependent function will be called after the nargs, is_varargs,
 * and missing_prototype fields in abi have been set, but before calling any of
 * the ll_abi_classify_*() functions. The function should determine which
 * calling convention to use and make any other necessary changes to abi such
 * as setting call_as_varargs.
 *
 * - func_sptr is the symbol table entry for the called function, or 0 if
 * unknown.
 * - jsra_flags is the flags operand on the JSRA instruction for an indirect
 *   call, or 0 if unknown.
 *
 * This function may be called with (0, 0) arguments when translating function
 * pointer types, for example.
 */
void ll_abi_compute_call_conv(LL_ABI_Info *abi, int func_sptr, int jsra_flags);

#ifdef DT_INT /* Use DT_INT to detect whether DTYPE is defined. */
/* Classify a function return type for the current target ABI.
 *
 * On entry, these fields should be initialized to 0:
 *
 *  abi->used_iregs
 *  abi->used_fregs
 *  abi->arg[0]
 *
 * The used_iregs and used_fregs members are updated to reflect the number of
 * registers used on *function entry*. Usually, that means that used_iregs
 * will be set when abi->arg[0].kind == LL_ARG_INDIRECT, and 0 otherwise.
 *
 * The return value classification will be stored in abi->arg[0].
 */
void ll_abi_classify_return_dtype(LL_ABI_Info *abi, DTYPE return_dtype);

/* Classify a function argument dtype for the current target ABI.
 *
 * The counters abi->used_iregs and abi->used_fregs should reflect the number
 * of registers used by arguments before the current one, including hidden
 * arguments used to return large structs. They will be incremented to reflect
 * the registers used by this argument.
 *
 * The arg pointer does not necessarily have to point to an argument in the
 * abi->arg[] array. It should point to a zero-initialized struct.
 */
void ll_abi_classify_arg_dtype(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg,
                               DTYPE arg_dtype);

/* The ll_abi_classify_* functions above don't always fill out the entire
 * LL_ABI_ArgInfo struct. Call this function to ensure that arg->type is always
 * present, even when it is trivial. */
void ll_abi_complete_arg_info(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg,
                              DTYPE dtype);

/* Classify a dtype for va_arg().
 *
 * This is needed for x86-64 and aarch64 which use a separate register save
 * area in varargs functions.
 *
 * Return the number of general-purpose and and floating point registers needed
 * to pass dtype in num_gp, num_fp. Return (0,0) for an argument that must be
 * passed in memory.
 *
 * The returned value is the 'map' argument for __builtin_va_genargs. It is one
 * of GP_XM(0), XM_GP(1), or XM_XM(2). See rte/pgc/hammer/src/va_arg.c.
 */
unsigned ll_abi_classify_va_arg_dtype(LL_Module* module, DTYPE dtype, 
                                      unsigned *num_gp, unsigned *num_fp);

/* Function for visiting struct members.
 *
 * Recursion / iteration keeps going as long as struct_visitor returns 0.
 *
 * The member_sptr is the ST_MEMBER entry, or 0 when visiting copmponents that
 * are not struct members.
 */
typedef int (*dtype_visitor)(void *context, DTYPE dtype, unsigned address,
                             int member_sptr);

/* Given a dtype possibly containing nested structs and unions, recursively
 * visit all non-struct members in address order.
 *
 * Stop as soon as a visit call returns non-zero, return the value returned
 * from the last visit call.
 */
int visit_flattened_dtype(dtype_visitor visitor, void *context, DTYPE dtype,
                          unsigned address, unsigned member_sptr);

FILE *llvm_file(void);
LL_Type *ll_convert_dtype(LL_Module *module, DTYPE dtype);
LL_Type *ll_convert_dtype_with_addrspace(LL_Module *module, DTYPE dtype, int addrspace);
#endif /* ifdef DT_INT */

/* Compute the appropriate coercion type for passing dtype in GPRs. */
LL_Type * ll_coercion_type(LL_Module *module, DTYPE dtype, ISZ_T size, ISZ_T reg_size);

/* llopt.c */
void optimize_block(INSTR_LIST *last_block_instr);
void maybe_undo_recip_div(INSTR_LIST *isns);
void widenAddressArith(void);
bool funcHasNoDepChk(void);
void redundantLdLdElim(void);
bool block_branches_to(int bih, int target);

/* llsched.c */
int enhanced_conflict(int nme1, int nme2);
void sched_instructions(INSTR_LIST *);


/**
   \brief set the module as gpu/cpu module
 */
void llvm_set_acc_module(void);
void llvm_set_cpu_module(void);
LL_Module* get_current_module(void);
LL_Module* llvm_get_current_module(void);

typedef struct FTN_LLVM_ST {
  union {
    UINT all;
    struct {
      unsigned host_reg : 1;
      unsigned has_init : 1;
      unsigned gpu_init : 1;
    } bits;
  } flags;
} FTN_LLVM_ST;

extern FTN_LLVM_ST ftn_llvm_st;

#define FTN_HOST_REG() ftn_llvm_st.flags.bits.host_reg
#define FTN_HAS_INIT() ftn_llvm_st.flags.bits.has_init
#define FTN_GPU_INIT() ftn_llvm_st.flags.bits.gpu_init

void reset_equiv_var(void);
void reset_master_sptr(void);
void process_cmblk_sptr(int sptr);
void write_ftn_typedefs(void);
void write_local_overlap(void);
void print_entry_subroutine(LL_Module *module);
LL_Type *make_generic_dummy_lltype(void);
bool llis_dummied_arg(SPTR sptr);
bool currsub_is_sret(void);
void write_gblvar_defs(FILE *target_file, LL_Module *module);

/**
   \brief ...
 */
bool is_function(int sptr);

/**
   \brief Should this function be represented as a varags LLVM function type?
 */
bool ll_abi_use_llvm_varargs(LL_ABI_Info *abi);

/**
   \brief ...
 */
bool llis_dummied_arg(SPTR sptr);

/**
   \brief ...
 */
bool small_aggr_return(DTYPE dtype);

/**
   \brief ...
 */
const char *llvm_fc_type(DTYPE dtype);

/**
   \brief ...
 */
const char *process_dtype_struct(DTYPE dtype);

/**
   \brief ...
 */
const char *process_ftn_dtype_struct(DTYPE dtype, char *tname, bool printed);

/**
   \brief ...
 */
const char *get_ot_name(unsigned ot);

/**
   \brief ...
 */
const char *llutil_strdup(const char *str);

/**
   \brief ...
 */
DTYPE dtype_from_return_type(ILI_OP ret_opc);

/**
   \brief ...
 */
DTYPE get_dtype_from_arg_opc(ILI_OP opc);

/**
   \brief ...
 */
DTYPE get_dtype_from_tytype(TY_KIND ty);

/**
   \brief ...
 */
DTYPE get_dtype_for_vect_type_nme(int nme);

/**
   \brief ...
 */
DTYPE get_param_equiv_dtype(DTYPE dtype);

/**
   \brief ...
 */
FILE *llvm_file(void);

/**
   \brief ...
 */
DTYPE generic_dummy_dtype(void);

/**
   \brief ...
 */
int get_dim_size(ADSC *ad, int dim);

/**
   \brief ...
 */
DTYPE get_int_dtype_from_size(int size);

/**
   \brief ...
 */
DTYPE get_return_dtype(DTYPE dtype, unsigned *flags, unsigned new_flag);

/**
   \brief ...
 */
int is_struct_kind(DTYPE dtype, bool check_return,
                   bool return_vector_as_struct);

/**
   \brief ...
 */
int visit_flattened_dtype(dtype_visitor visitor, void *context, DTYPE dtype,
                          unsigned address, unsigned member_sptr);

/**
   \brief ...
 */
LL_ABI_Info *ll_abi_alloc(LL_Module *module, unsigned nargs);

/**
   \brief Get ABI lowering info for a function given its symbol table entry.

   Since IPA can change the dtype of functions between calls to schedule(), use
   the passed dtype instead of DTYPEG(func_sptr).
 */
LL_ABI_Info *ll_abi_for_func_sptr(LL_Module *module, SPTR func_sptr,
                                  DTYPE dtype);

/**
   \brief ...
 */
LL_ABI_Info *ll_abi_free(LL_ABI_Info *abi);

/**
   \brief Get ABI lowering info for a specific call site.

   - ilix is the IL_JSR or IL_JSRA instruction representing the call.
   - ret_ili is the IL_DFRx instruction extracting the return value for the
     call, or 0 if the return value is unused.

   This may not provide lowering information for all function arguments if the
   callee is varargs or if a prototype is not available.
 */
LL_ABI_Info *ll_abi_from_call_site(LL_Module *module, int ilix,
                                   DTYPE ret_dtype);

/**
   \brief ...
 */
LL_ABI_Info *process_ll_abi_func_ftn(SPTR func_sptr, bool use_sptrs);

/**
   \brief ...
 */
LL_ABI_Info *process_ll_abi_func_ftn_mod(LL_Module *mod, SPTR func_sptr,
                                         bool update);


/**
   \brief ...
 */
LL_InstrListFlags ldst_instr_flags_from_dtype(DTYPE dtype);

/**
   \brief ...
 */
LL_InstrListFlags ldst_instr_flags_from_dtype_nme(DTYPE dtype, int nme);

/**
   \brief ...
 */
LL_Type *get_ftn_cbind_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_cmblk_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_dummy_lltype(int sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_extern_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_func_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_hollerith_type(int sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_static_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_typedesc_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_struct_def_type(char *def_name, LL_Module *module);

/**
   \brief ...
 */
LL_Type *ll_abi_function_type(LL_ABI_Info *abi);

/**
   \brief ...
 */
LL_Type *ll_abi_return_type(LL_ABI_Info *abi);

/**
   \brief ...
 */
LL_Type *ll_convert_array_dtype(LL_Module *module, DTYPE dtype, int addrspace);

/**
   \brief ...
 */
LL_Type *ll_convert_dtype(LL_Module *module, DTYPE dtype);

/**
   \brief ...
 */
LL_Type *make_array_lltype(int size, LL_Type *pts_to);

/**
   \brief ...
 */
LL_Type *make_generic_dummy_lltype(void);

/**
   \brief ...
 */
LL_Type *make_int_lltype(unsigned bits);

/**
   \brief ...
 */
LL_Type *make_lltype_from_abi_arg(LL_ABI_ArgInfo *arg);

/**
   \brief ...
 */
LL_Type *make_lltype_from_arg(int arg);

/**
   \brief ...
 */
LL_Type *make_lltype_from_arg_noproto(int arg);

/**
   \brief ...
 */
LL_Type *make_lltype_from_dtype(DTYPE dtype);

/**
   \brief ...
 */
LL_Type *
make_lltype_from_dtype_with_addrspace(DTYPE dtype, int addrspace);

/**
   \brief ...
 */
LL_Type *make_lltype_from_iface(SPTR sptr);

/**
   \brief ...
 */
LL_Type *make_lltype_from_sptr(SPTR sptr);

/**
   \brief ...
 */
LL_Type *make_lltype_sz4v3_from_dtype(DTYPE dtype);

/**
   \brief ...
 */
LL_Type *make_lltype_sz4v3_from_sptr(SPTR sptr);

/**
   \brief ...
 */
LL_Type *make_ptr_lltype(LL_Type *pts_to);

/**
   \brief ...
 */
LL_Type *make_vector_lltype(int size, LL_Type *pts_to);

/**
   \brief ...
 */
LL_Type *make_void_lltype(void);

/**
   \brief ...
 */
void layout_struct_body(LL_Module *module, LL_Type *struct_type,
			int member_sptr, ISZ_T size_bytes);

/**
   \brief ...
 */
LL_Type *process_acc_decl_common(LL_Module *module, int first, BIGINT size,
                                 int is_constant, char *d_name,
                                 int externcommon);

/**
   \brief it defines a global variable, used by lili2llvm_decl_globalvar.
 */
LL_Type *
process_acc_decl_gblvar(LL_Module *module, LL_Type* lltype, SPTR sptr, char* name);

/**
   \brief ...
 */
LL_Type *process_acc_decl_statics(LL_Module *module, int first, char *d_name,
                                  int init, int addrspace);

/**
   \brief ...
 */
LL_Type *process_acc_string(LL_Module *module, char *name, int sptr,
                            bool initialize);

/**
   \brief ...
 */
OPERAND *gen_copy_list_op(OPERAND *operands);

/**
   \brief ...
 */
OPERAND *gen_copy_op(OPERAND *op);

/**
   \brief ...
 */
OPERAND *make_constsptr_op(SPTR sptr);

/**
   \brief ...
 */
OPERAND *make_constval32_op(int idx);

/**
   \brief ...
 */
OPERAND *make_constval_opL(LL_Type *ll_type, INT conval0, INT conval1, INT conval2, INT conval3);

/**
   \brief ...
 */
OPERAND *make_constval_op(LL_Type *ll_type, INT conval0, INT conval1);

/**
   \brief ...
 */
OPERAND *make_def_op(char *str);

/**
   \brief ...
 */
OPERAND *make_label_op(SPTR sptr);

/**
   \brief ...
 */
OPERAND *make_mdref_op(LL_MDRef mdref);

/**
   \brief Create metadata operand that wraps an LLVM value

   For example, <tt>metadata !{i32 %x}</tt>.
   Used by \c llvm.dbg.declare intrinsic.
 */
OPERAND *make_metadata_wrapper_op(SPTR sptr, LL_Type *llTy);

/**
   \brief ...
 */
OPERAND *make_null_op(LL_Type *llt);

/**
   \brief ...
 */
OPERAND *make_operand(void);

/**
   \brief ...
 */
OPERAND *make_target_op(SPTR sptr);

/**
   \brief ...
 */
OPERAND *make_tmp_op(LL_Type *llt, TMPS *tmps);

/**
   \brief ...
 */
OPERAND *make_undef_op(LL_Type *llt);

/**
   \brief ...
 */
OPERAND *make_var_op(SPTR sptr);

/**
   \brief ...
 */
OPERAND *process_symlinked_sptr(int sptr, int total_init_sz, int is_union, int max_field_sz);

/**
   \brief ...
 */
TMPS *make_tmps(void);

/**
   \brief ...
 */
void indent(int change);

/**
   \brief ...
 */
void init_metadata_index(TMPS *t);

/**
   \brief ...
 */
void init_output_file(void);

/**
   \brief ...
 */
void ll_abi_complete_arg_info(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype);

/**
   \brief Add a function prototype declaration to the LLVM module
   \param sptr  The symbol of the function
   \param nargs Number of arguments to the function
   \param args  An array of DTYPEs for the arguments
 */
void ll_add_func_proto(int sptr, unsigned flags, int nargs, DTYPE *args);

/**
   \brief ...
 */
void ll_override_type_string(LL_Type *llt, const char *str);

/**
   \brief ...
 */
void llutil_def_reset(void);

/**
   \brief ...
 */
void llutil_dfile_init(void);

/**
   \brief ...
 */
void llutil_gblvar_def_reset(void);

/**
   \brief ...
 */
void llutil_struct_def_reset(void);

/**
   \brief ...
 */
void print_line(const char *ln);

/**
   \brief ...
 */
void print_line_tobuf(char *ln, char *buf);

/**
   \brief ...
 */
void print_llsize(LL_Type *llt);

/**
   \brief ...
 */
void print_llsize_tobuf(LL_Type *llt, char *buf);

/**
   \brief ...
 */
void print_metadata_name(TMPS *t);

/**
   \brief ...
 */
void print_nl_tobuf(char *buf);

/**
   \brief ...
 */
void print_nl(void);

/**
   \brief ...
 */
void print_space(int num);

/**
   \brief ...
 */
void print_space_tobuf(int num, char *buf);

/**
   \brief ...
 */
void print_token(const char *tk);

/**
   \brief ...
 */
void print_token_tobuf(char *tk, char *buf);

/**
   \brief ...
 */
void process_acc_decl_const_param(LL_Module *module, char *name, int sptr);

/**
   \brief ...
 */
void process_acc_put_dinit(int cmem, int x, bool loop);

/**
   \brief ...
 */
void set_metadata_string(TMPS *t, char *string);

/**
   \brief ...
 */
void write_constant_value(int sptr, LL_Type *type, INT conval0, INT conval1,
                          bool uns);

/**
   \brief ...
 */
void write_ftn_typedefs(void);

/**
   \brief ...
 */
void write_gblvar_defs(FILE *target_file, LL_Module *module);

/**
   \brief ...
 */
void write_operand(OPERAND *p, const char *punc_string, int flags);

/**
   \brief ...
 */
void write_operands(OPERAND *operand, int flags);

/**
   \brief ...
 */
void write_struct_defs(void);

/**
   \brief ...
 */
void write_type(LL_Type *ll_type);

/// \brief Compute the LL_InstrListFlag set from an ATOMIC_RMW_OP
/// \param aop   an ATOMIC_RMW_OP value
LL_InstrListFlags ll_instr_flags_from_aop(ATOMIC_RMW_OP aop);

bool llis_integral_kind(DTYPE dtype);
bool llis_pointer_kind(DTYPE dtype);
bool llis_array_kind(DTYPE dtype);
bool llis_vector_kind(DTYPE dtype);
bool llis_struct_kind(DTYPE dtype);
bool llis_function_kind(DTYPE dtype);

/**
   \brief return whether param debug info should be preserved
 */
bool should_preserve_param(const DTYPE dtype);

#ifdef OMP_OFFLOAD_LLVM
/**
   \brief Create a file to write the device code if it has not already been created,
 */
void init_gpu_output_file(void);

/**
   \brief Assign the cpu file to LLVMFIL
 */
void use_cpu_output_file(void);

/**
   \brief Assign the gpu file to LLVMFIL
 */
void use_gpu_output_file(void);
#endif
#endif

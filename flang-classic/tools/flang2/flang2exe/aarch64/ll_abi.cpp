/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* ll_abi.c - Lowering arm function calls to LLVM IR.
 *
 * This file implements the AAPCS_VFP procedure call standard for the ARMv7
 * architecture.
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "llutil.h"
#include "ll_structure.h"
#include "dtypeutl.h"

#define DT_VOIDNONE DT_NONE

#define DT_BASETYPE(dt) (dt)

void
ll_abi_compute_call_conv(LL_ABI_Info *abi, int func_sptr, int jsra_flags)
{
  abi->call_conv = LL_CallConv_C; /* Default */
}

/* AAPCS has the concept of a homogeneous aggregrate. It is an aggregate type
 * where all the fundamental types are the same after flattening all structs
 * and arrays. */
struct arm_homogeneous_aggr {
  LL_Module *module;
  LL_Type *base_type;
  unsigned base_bytes;
};

typedef struct ARM_ABI_ArgInfo_ {
  enum LL_ABI_ArgKind kind;
  LL_Type *type;
  bool is_return_val;
} ARM_ABI_ArgInfo;

inline static void
update_arg_info(LL_ABI_ArgInfo *arg, ARM_ABI_ArgInfo *arm_arg)
{
  arg->kind = arm_arg->kind;
  if (arm_arg->type) {
    arg->type = arm_arg->type;
  }
}

/* Return 1 if dtype is inconsistent with the homogeneous aggregate
 * information pointed to by context. */
static int
update_homogeneous(void *context, DTYPE dtype, unsigned address,
                   int member_sptr)
{
  struct arm_homogeneous_aggr *ha = (struct arm_homogeneous_aggr *)context;
  unsigned size;
  LL_Type *llt;

  dtype = DT_BASETYPE(dtype);

  if (DTY(dtype) == TY_ARRAY)
    dtype = (DTYPE)DTY(dtype + 1); // ???

  switch (dtype) {
  default:
    break;
  case DT_CMPLX:
    dtype = DT_FLOAT;
    break;
  case DT_DCMPLX:
    dtype = DT_DBLE;
    break;
  }

  size = size_of(dtype);
  llt = ll_convert_dtype(ha->module, dtype);

  if (!ha->base_type) {
    if (address != 0)
      return 1;
    ha->base_type = llt;
    ha->base_bytes = size;
    return 0;
  }

  /* Check if dtype is consistent with the existing base type. */
  if (size != ha->base_bytes)
    return 1;

  if (!size || address % size != 0)
    return 1;

  /* Vector types just need matching sizes. Elements don't need to match. */
  if (ha->base_type->data_type == LL_VECTOR && llt->data_type == LL_VECTOR)
    return 0;

  /* Other base types must be identical. */
  return ha->base_type != llt;
}

/* Check if dtype is a VFP register candidate. Return the coercion type or NULL.
 */
static LL_Type *
check_vfp(LL_Module *module, DTYPE dtype, struct arm_homogeneous_aggr *aggr)
{
  ISZ_T size = size_of(dtype);

  /* Check if dtype is a homogeneous aggregate. */
  if (visit_flattened_dtype(update_homogeneous, aggr, dtype, 0, 0))
    return NULL;
  if (!aggr->base_type)
    return NULL;

  /* A non-aggregated scalar will simply be copied to base_type. */
  switch (aggr->base_type->data_type) {
  case LL_FLOAT:
  case LL_DOUBLE:
    break;
  case LL_VECTOR:
    /* Only 64-bit or 128-bit vectors supported. */
    if (aggr->base_bytes != 8 && aggr->base_bytes != 16)
      return NULL;
    break;
  default:
    return NULL;
  }

  /* We have a scalar or a homogeneous aggregate of the right type. The ABI
   * supports one to four elements of the base type. */
  if (size > 4 * aggr->base_bytes)
    return NULL;

  /* Single-element aggregate? */
  if (size == aggr->base_bytes)
    return aggr->base_type;

  /* Multiple elements coerced to an array type. */
  return ll_get_array_type(aggr->base_type, size / aggr->base_bytes, 0);
}

/* Classify an integer type for return or arg. */
static enum LL_ABI_ArgKind
classify_int(DTYPE dtype)
{
  /* Integer types smaller than a register must be sign/zero extended. */
  if (size_of(dtype) < 4)
    return DT_ISUNSIGNED(dtype) ? LL_ARG_ZEROEXT : LL_ARG_SIGNEXT;

  return LL_ARG_DIRECT;
}

static inline int
ll_abi_num_regs(int num_bytes)
{
  return (num_bytes + 7) / 8;
}

/* Classify common to args and return values. */
static bool
classify_common(LL_Module *module, LL_ABI_Info *abi, ARM_ABI_ArgInfo *arg,
                DTYPE dtype)
{
  if (DT_ISINT(dtype)) {
    arg->kind = classify_int(dtype);
    return true;
  }

  /* Basic types can be returned in registers directly. Complex types also
   * get handled correctly. */
  if (dtype == DT_VOIDNONE || DT_ISSCALAR(dtype)) {
    arg->kind = LL_ARG_DIRECT;
    return true;
  }

  struct arm_homogeneous_aggr aggr = {module, NULL, 0};
  LL_Type *haggr_lltype = check_vfp(module, dtype, &aggr);
  if (haggr_lltype) {
    arg->kind = LL_ARG_COERCE;
    arg->type = haggr_lltype;
    return true;
  }

  // AAPCS64: Arm 64 bit Architecture Procedure Call Standard
  if (DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_UNION) {
    ISZ_T size = size_of(dtype);
    if (size > 16) {
      // AAPCS64: Stage B3
      // If the argument is a composite type that is larger than 16 bytes, then
      // the argument is copied to memory by the caller and the argument is
      // replaced by a pointer to the copy
      if (arg->is_return_val) {
        arg->kind = LL_ARG_INDIRECT;
      } else {
        arg->kind = LL_ARG_INDIRECT_BUFFERED;
      }
    } else {
      arg->kind = LL_ARG_COERCE;
      if (arg->is_return_val) {
        // Whether directly or indirectly, returned values are always passed
        // through registers. However, unlike input arguments whose size has to
        // be rounded up to the nearest 8 bytes (see LLObj_LocalBuffered), the
        // type of the returned value has to match the size of the actual LHS.
        // This is because the returned value is immediately stored into the
        // local variable for the LHS and that store has to match the size of
        // the local otherwise it will spill over the next local in the stack.
        // We use a coercion type to signify that :
        //    - returned value is to be passed through registers
        //    - return value size matches that of LHS
        // Example:
        //    {i8,i8,i8,i8,i32,i8} -> {i64, i8}
        arg->type = ll_coercion_type(abi->module, dtype, size, 8);
      } else {
        // AAPCS64: Stage B4
        // If the argument is a composite type then the size of the argument is
        // rounded up to the nearest multiple of 8 bytes
        //
        // AAPCS64: Stage  C14
        // If the size of the argument is less than 8 bytes then the size of the
        // argument is set to 8 bytes
        arg->type = ll_create_basic_type(abi->module, LL_I64, 0);
        if (size > 8) {
          arg->type = ll_get_array_type(arg->type, ll_abi_num_regs(size), 0);
        }
      }
    }
    return true;
  }
  return false;
}

void
ll_abi_classify_return_dtype(LL_ABI_Info *abi, DTYPE dtype)
{
  enum LL_BaseDataType bdt = LL_NOTYPE;
  ARM_ABI_ArgInfo tmp_arg_info = {LL_ARG_UNKNOWN, NULL, true};

  dtype = DT_BASETYPE(dtype);

  if (classify_common(abi->module, abi, &tmp_arg_info, dtype)) {
    update_arg_info(&(abi->arg[0]), &tmp_arg_info);
    return;
  }
  /* Small structs can be returned in r0.
   * FIXME: can also be returned in register pair of floating-point registers.
   */
  switch (size_of(dtype)) {
  case 1:
    bdt = LL_I8;
    break;
  case 2:
    bdt = LL_I16;
    break;
  case 3:
  case 4:
    bdt = LL_I32;
    break;
  case 8:
    bdt = LL_I64;
    break;
  }
  if (bdt != LL_NOTYPE) {
    abi->arg[0].kind = LL_ARG_COERCE;
    abi->arg[0].type = ll_create_basic_type(abi->module, bdt, 0);
    return;
  }

  /* Large types must be returned in memory via an sret pointer argument. */
  abi->arg[0].kind = LL_ARG_INDIRECT;
}

void
ll_abi_classify_arg_dtype(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype)
{
  ISZ_T size;
  ARM_ABI_ArgInfo tmp_arg_info = {LL_ARG_UNKNOWN, NULL, false};

  dtype = DT_BASETYPE(dtype);

  if (classify_common(abi->module, abi, &tmp_arg_info, dtype)) {
    update_arg_info(arg, &tmp_arg_info);
    return;
  }

  /* All other arguments are coerced into an array of 32-bit registers. */
  size = size_of(dtype);
  arg->kind = LL_ARG_COERCE;
  if (alignment(dtype) > 4 && size % 8 == 0) {
    /* The coercion type needs to have the same alignment as the original type.
     */
    arg->type = ll_create_basic_type(abi->module, LL_I64, 0);
    if (size > 8)
      arg->type = ll_get_array_type(arg->type, size / 8, 0);
  } else {
    arg->type = ll_create_basic_type(abi->module, LL_I32, 0);
    if (size > 4)
      arg->type = ll_get_array_type(arg->type, (size + 3) / 4, 0);
  }
}

unsigned
ll_abi_classify_va_arg_dtype(LL_Module *module, DTYPE dtype, unsigned *num_gp,
                             unsigned *num_fp)
{
  ISZ_T size;
  struct arm_homogeneous_aggr aggr = {module, NULL, 0};
  LL_Type *haggr_lltype;

  size = size_of(dtype);
  *num_gp = 0;
  *num_fp = 0;

  haggr_lltype = check_vfp(module, dtype, &aggr);
  if (haggr_lltype) {
    /*
      Only one member per register. a struct of 4 32-bit floats is scattered
      over 4 128-bit registers. Recomputing the number of members as the size
      of the whole type / the size of a member.
      __builtin_va_fpargs gathers these register back into a temporary that
      matches the original layout */
    *num_fp = size / aggr.base_bytes;
    return aggr.base_bytes;
  }

  if (DT_ISINT(dtype) || DTY(dtype) == TY_PTR) {
    *num_gp = ll_abi_num_regs(size);
    return 0;
  }

  if (DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_UNION) {
    if (size > 16) {
    } else {
      *num_gp = ll_abi_num_regs(size);
    }
    return 0;
  }

  return 0;
}

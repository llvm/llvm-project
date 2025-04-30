/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* ll_abi.c - Lowering power function calls to LLVM IR.
 *
 * This file implements the 64-Bit ELF V2 ABI Specification for the Power
 * architecture.
 *
 * Based on version 1.0, 21 July 2014.
 */

#include "gbldefs.h"
#include "error.h"
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
  /* Always use the default CC. */
}

/* ELF v2 ABI has the concept of a homogeneous aggregrate. It is an aggregate
 * type where all the fundamental types are the same after flattening all
 * structs and arrays.
 */
struct pwr_homogeneous_aggr {
  LL_Module *module;
  LL_Type *base_type;
  ISZ_T base_bytes;
};

/* Return 1 if dtype is inconsistent with the homogeneous aggregate
 * information pointed to by context. */
static int
update_homogeneous(void *context, DTYPE dtype, unsigned address,
                   int member_sptr)
{
  struct pwr_homogeneous_aggr *ha = (struct pwr_homogeneous_aggr *)context;
  ISZ_T size;
  LL_Type *llt;

  dtype = DT_BASETYPE(dtype);

  if (DTY(dtype) == TY_ARRAY)
    dtype = (DTYPE)DTY(dtype + 1); // ???

  switch (dtype) {
  case DT_CMPLX:
    dtype = DT_FLOAT;
    break;
  case DT_DCMPLX:
    dtype = DT_DBLE;
    break;
  default:
    break;
  }

  size = zsize_of(dtype);
  llt = ll_convert_dtype(ha->module, dtype);

  /* Check if this type is allowed as the base type of a homogeneous
   * aggregate. */
  switch (llt->data_type) {
  case LL_FLOAT:
  case LL_DOUBLE:
    /* OK. */
    break;
  case LL_VECTOR:
    /* Allow naturally sized vectors only. */
    if (size != 16)
      return 1;
    break;
  default:
    /* Nope. */
    return 1;
  }

  /* Record base type on the first call. */
  if (!ha->base_type) {
    if (address != 0)
      return 1;
    ha->base_type = llt;
    ha->base_bytes = size;
    return 0;
  }

  /* Check if dtype is consistent with the existing base type. */
  if (llt != ha->base_type)
    return 1;

  /* Reject unaligned entries. */
  if (!size || address % size != 0)
    return 1;

  return 0;
}

/* Check if dtype is a vector register candidate. */
static LL_Type *
check_vector_registers(LL_Module *module, DTYPE dtype)
{
  struct pwr_homogeneous_aggr aggr = {module, NULL, 0};
  ISZ_T size = zsize_of(dtype);

  /* Check if dtype is a homogeneous aggregate, or a single value which can
   * be passed in vector registers. */
  if (visit_flattened_dtype(update_homogeneous, &aggr, dtype, 0, 0))
    return NULL;
  if (!aggr.base_type)
    return NULL;

  /* A non-aggregated scalar will simply be copied to base_type. */
  switch (aggr.base_type->data_type) {
  case LL_FLOAT:
  case LL_DOUBLE:
    break;
  case LL_VECTOR:
    if (aggr.base_bytes != 8 && aggr.base_bytes != 16)
      return NULL;
    break;
  default:
    return NULL;
  }

  /* We have a scalar or a homogeneous aggregate of the right type. The ABI
   * supports one to eight elements of the base type. */
  if (size > 8 * aggr.base_bytes)
    return NULL;

  if (size == aggr.base_bytes)
    return aggr.base_type;

  /* If we supported IBM extended precision or _Decimal128 data types, we
   * should impose a 4 element limit, but all other types allow up to eight
   * elements to be passed in vector registers.
   */

  return ll_get_array_type(aggr.base_type, size / aggr.base_bytes, 0);
}

/* Classify an integer type for return or arg. */
static enum LL_ABI_ArgKind
classify_int(DTYPE dtype)
{
  /* Integer types smaller than a register must be sign/zero extended. */
  if (zsize_of(dtype) < 8)
    return DT_ISUNSIGNED(dtype) ? LL_ARG_ZEROEXT : LL_ARG_SIGNEXT;

  return LL_ARG_DIRECT;
}

/* Classify common to args and return values. */
static bool
classify_common(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype)
{
  LL_Type *haggr;
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

  /* Check for vector register arguments, including homogeneous aggregrates. */
  if ((haggr = check_vector_registers(abi->module, dtype))) {
    arg->kind = LL_ARG_DIRECT;
    arg->type = haggr;
    return true;
  }

  return false;
}

void
ll_abi_classify_return_dtype(LL_ABI_Info *abi, DTYPE dtype)
{
  enum LL_BaseDataType bdt = LL_NOTYPE;
  ISZ_T size;

  dtype = DT_BASETYPE(dtype);

  if (classify_common(abi, &abi->arg[0], dtype))
    return;

  /* Small structs can be returned in up to two GPRs. */
  size = zsize_of(dtype);
  if (size <= 16) {
    abi->arg[0].kind = LL_ARG_COERCE;
    abi->arg[0].type = ll_coercion_type(abi->module, dtype, size, 8);
    return;
  }

  /* Large types must be returned in memory via an sret pointer argument. */
  abi->arg[0].kind = LL_ARG_INDIRECT;
}

void
ll_abi_classify_arg_dtype(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype)
{
  dtype = DT_BASETYPE(dtype);

  if (classify_common(abi, arg, dtype))
    return;

  /* Fortran, by default, is pass by reference.
   * dtype information is not enough to determine if this argument is direct
   * or not.  All classification occurs in process_ll_abi_func_ftn_mod and
   * only when there is a known prototype.
   */
  if (arg->ftn_pass_by_val &&
      (DT_ISCMPLX(dtype) || (DTY(dtype) == TY_STRUCT))) {
    arg->kind = LL_ARG_COERCE;
    arg->type = ll_coercion_type(abi->module, dtype, zsize_of(dtype), 8);
    return;
  }

  /* ???: 64 == 8 x eightwords, which is the most we can pass in GPRs. */
  if ((DTY(dtype) == TY_STRUCT) && (zsize_of(dtype) > 64)) {
    LL_Type *llt;
    arg->kind = LL_ARG_BYVAL;
    llt = ll_get_struct_type(abi->module, dtype, 0);
    if (!llt) {
      llt = ll_convert_dtype(abi->module, dtype);
      assert(llt, "expected LL_Type*", dtype, ERR_Fatal);
    }
    arg->type = ll_get_pointer_type(llt);
    return;
  }

  /* All other arguments are coerced. LLVM will figure out which parts go in
   * registers. */
  arg->kind = LL_ARG_COERCE;
  arg->type = ll_coercion_type(abi->module, dtype, zsize_of(dtype), 8);
}

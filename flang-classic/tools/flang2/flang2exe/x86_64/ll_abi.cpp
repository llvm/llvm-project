/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* ll_abi.c - Lowering x86-64 function calls to LLVM IR. */

#include "ll_abi.h"
#include "dtypeutl.h"
#include "error.h"
#include "symfun.h"

#define DT_VOIDNONE DT_NONE

void
ll_abi_compute_call_conv(LL_ABI_Info *abi, int func_sptr, int jsra_flags)
{
  /* Functions without a prototype must be called as varargs functions.
   * Otherwise %AL isn't set up correrctly if the function turns out to be a
   * varargs function. */
  if (abi->missing_prototype)
    abi->call_as_varargs = true;
}

/* Parameter classes from the AMD64 ABI Draft 0.99.6 section on parameter
 * passing. */
enum amd64_class {
  AMD64_NO_CLASS = 0,
  AMD64_INTEGER,
  AMD64_SSE,
  AMD64_SSEUP,
  AMD64_X87,
  AMD64_X87UP,
  AMD64_COMPLEX_X87,
  AMD64_MEMORY
};

/* Merge two argument classes for the case when there are multiple objects in
 * one eight-byte. */
static enum amd64_class
amd64_merge(enum amd64_class c1, enum amd64_class c2)
{
  /* (a) If both classes are equal, this is the resulting class. */
  if (c1 == c2)
    return c1;

  /* (b) If one of the classes is NO_CLASS, the resulting class is the other
   * class. */
  if (c1 == AMD64_NO_CLASS)
    return c2;
  if (c2 == AMD64_NO_CLASS)
    return c1;

  /* (c) If one of the classes is MEMORY, the result is the MEMORY class. */
  if (c1 == AMD64_MEMORY || c2 == AMD64_MEMORY)
    return AMD64_MEMORY;

  /* (d) If one of the classes is INTEGER, the result is the INTEGER. */
  if (c1 == AMD64_INTEGER || c2 == AMD64_INTEGER)
    return AMD64_INTEGER;

  /* (e) If one of the classes is X87, X87UP, COMPLEX_X87 class, MEMORY is used
   * as class. */
  if (c1 == AMD64_X87 || c2 == AMD64_X87 || c1 == AMD64_X87UP ||
      c2 == AMD64_X87UP || c1 == AMD64_COMPLEX_X87 || c2 == AMD64_COMPLEX_X87)
    return AMD64_MEMORY;

  /* (f) Otherwise class SSE is used. */
  return AMD64_SSE;
}

/* The maximum size of a single argument is 4 eightbytes, but that is only when
 * passing a single __m256 in a %ymm register. In all other cases, the argument
 * must be at most two eightbytes.
 *
 * We'll special-case the __mm256 case and only classify the first two
 * eightbytes.
 *
 * LLVM: we want to simply pass vectors as arguments in LLVM.
 */

/* Update classification in eightbyte[] when adding a dtype at offset, or
 * return 1 to indicate that everything needs to be passed in memory.
 */
static int
amd64_update_class(void *context, DTYPE dtype, unsigned address,
                   int member_sptr)
{
  enum amd64_class *ebc = (enum amd64_class *)context;

  /* Special case for __mm256 as mentioned above. */
  if (DTY(dtype) == TY_256 && address == 0) {
    ebc[0] = amd64_merge(ebc[0], AMD64_SSE);
    ebc[1] = amd64_merge(ebc[1], AMD64_SSEUP);
    return 0;
  }

  /* Since this is LLVM, then we want to pass vectors by value and not as
     temps on the stack. */
  if (DT_ISVECT(dtype)) {
    ebc[0] = AMD64_SSE; /* force amd64_classify() */
    ebc[1] = AMD64_SSEUP;
    return 0;
  }


  /* Check if this overflows our two ebcs. */
  unsigned eight_num = address / 8;
  unsigned size = zsize_of(dtype);
  if (!size || address + size > 16)
    return 1;

  /* Unaligned fields cause the whole argument to be passed in memory. */
  unsigned offset = address % 8;
  if (offset & alignment(dtype))
    return 1;

  /* Handle member arrays by recursively calling amd64_update_class() with
   * one or two representative array elements. */
  if (DTY(dtype) == TY_ARRAY) {
    /* TY_ARRAY dtype dim */
    DTYPE ddtype = DTySeqTyElement(dtype);
    int retval = amd64_update_class(context, ddtype, address, 0);
    /* Does this array extend into the next 8-byte segment? */
    if (retval == 0 && address < 8 && address + size > 8)
      retval = amd64_update_class(context, ddtype, address + 8, 0);
    return retval;
  }

  if (size <= 8) {
    bool is_ptr = DTY(dtype) == TY_PTR;
    enum amd64_class cls = AMD64_MEMORY;
    if (DT_ISINT(dtype) || is_ptr)
      cls = AMD64_INTEGER;
    else if (DT_ISREAL(dtype) || DT_ISCMPLX(dtype))
      cls = AMD64_SSE;
    ebc[eight_num] = amd64_merge(ebc[eight_num], cls);
    return ebc[eight_num] == AMD64_MEMORY;
  }

  /* This type is larger than 8 bytes. It must be aligned. */
  if (address != 0)
    return 1;

  enum amd64_class cls[2] = {AMD64_MEMORY, AMD64_MEMORY};
  switch (DTY(dtype)) {
  default:
    break;

  case TY_VECT:
  case TY_128:
  case TY_FLOAT128:
    cls[0] = AMD64_SSE;
    cls[1] = AMD64_SSEUP;
    break;

  case TY_INT128:
    cls[0] = AMD64_INTEGER;
    cls[1] = AMD64_INTEGER;
    break;

  case TY_DCMPLX:
    cls[0] = AMD64_SSE;
    cls[1] = AMD64_SSE;
    break;
  }
  ebc[0] = amd64_merge(ebc[0], cls[0]);
  ebc[1] = amd64_merge(ebc[1], cls[1]);

  return ebc[0] == AMD64_MEMORY;
}

/* Classify dtype for passing as an argument or return value.
 *
 * Return 1 if argument must be passed in memory, otherwise return 0 and update
 * the eightbyte classes.
 */
static int
amd64_classify(enum amd64_class ebc[2], DTYPE dtype)
{
  unsigned i;
  for (i = 0; i != 2; i++)
    ebc[i] = AMD64_NO_CLASS;
  if (visit_flattened_dtype(amd64_update_class, (void *)ebc, dtype, 0, 0))
    return 1;

  /* Post-merge cleanup.
     (a) If one of the classes is MEMORY, the whole argument is passed in
     memory. */
  if (ebc[0] == AMD64_MEMORY || ebc[1] == AMD64_MEMORY)
    return 1;
  /* (b) If X87UP is not preceded by X87, the whole argument is passed in
     memory. */
  if (ebc[1] == AMD64_X87UP && ebc[0] != AMD64_X87)
    return 1;
  /* (c) If the size of the aggregate exceeds two eightbytes and the first
     eightbyte isn’t SSE or any other eightbyte isn’t SSEUP, the whole
     argument is passed in memory. */
  if (zsize_of(dtype) > 16 && (ebc[0] != AMD64_SSE || ebc[1] != AMD64_SSEUP))
    return 1;

  /* This is a register argument, assuming there are free registers available.
   */
  return 0;
}

/* Compute coercion types for passing or returning dtype in registers, using
 * ebc from amd64_classify(). */
static void
amd64_coerce(LL_Module *module, LL_ABI_ArgInfo *arg,
             const enum amd64_class ebc[2], DTYPE dtype)
{
  unsigned i;
  LL_Type *types[2] = {NULL, NULL};
  ISZ_T size = zsize_of(dtype);

  arg->kind = LL_ARG_COERCE;

  /* This is a single vector register. */
  if (ebc[0] == AMD64_SSE && ebc[1] == AMD64_SSEUP) {
    /* Possible coercion types: <2 x double>, <4 x float>, <4 x i32>
       Also 256-bit vectors: <4 x double>, ...
       TODO: Create a coercion type that better matches the original. */
    LL_Type *ltype = ll_create_basic_type(module, LL_FLOAT, 0);
    unsigned lanes = (size > 16) ? 8 : 4;
    arg->type = ll_get_vector_type(ltype, lanes);
    return;
  }

  if (ebc[0] == AMD64_X87 && ebc[1] == AMD64_X87UP) {
    /* A single 16-byte fp80 value. */
    arg->type = ll_create_basic_type(module, LL_X86_FP80, 0);
    return;
  }

  /* This is one or two registers. */
  for (i = 0; i != 2; ++i) {
    switch (ebc[i]) {
    case AMD64_NO_CLASS:
      break;
    case AMD64_INTEGER:
      if (size < 8 * i + 8)
        types[i] = ll_create_int_type(module, 8 * (size - 8 * i));
      else
        types[i] = ll_create_basic_type(module, LL_I64, 0);
      break;
    case AMD64_SSE:
      /* Possibilities: float, double, <2 x float> */
      if (dtype == DT_CMPLX) {
        LL_Type *ftype = ll_create_basic_type(module, LL_FLOAT, 0);
        LL_Type *pair[2] = {ftype, ftype};
        types[i] = ll_create_anon_struct_type(module, pair, 2,  true, 0);
      }
      else if (size == 8 * i + 4)
        types[i] = ll_create_basic_type(module, LL_FLOAT, 0);
      else
        types[i] = ll_create_basic_type(module, LL_DOUBLE, 0);
      break;
    default:
      assert(0, "Unexpected eightbyte class", ebc[i], ERR_Fatal);
      break;
    }
  }

  /* An undefined struct shows up with size = 0 and ebc = { NO_CLASS,
   * NO_CLASS }.  We can't compute the correct handling of that struct here,
   * so just use an i64.  This wil be OK for function pointers, not for
   * actually making a call. */
  if (size == 0)
    types[0] = ll_create_int_type(module, 64);
  else
    assert(types[0], "No class for first register", 0, ERR_Fatal);

  /* Build an anonymous struct if we have two registers. */
  if (types[1]) {
    arg->type = ll_create_anon_struct_type(module, types, 2, false, LL_AddrSp_Default);
  } else {
    arg->type = types[0];
  }
}

/* Classify an integer type for return or arg. */
static enum LL_ABI_ArgKind
classify_int(DTYPE dtype)
{
  /* Integer types smaller than a register must be sign/zero extended. */
  if (size_of(dtype) < 8)
    return DT_ISUNSIGNED(dtype) ? LL_ARG_ZEROEXT : LL_ARG_SIGNEXT;

  return LL_ARG_DIRECT;
}

void
ll_abi_classify_return_dtype(LL_ABI_Info *abi, DTYPE dtype)
{
  enum amd64_class ebc[2];

  if (dtype == DT_VOIDNONE) {
    abi->arg[0].kind = LL_ARG_DIRECT;
    return;
  }


  if (amd64_classify(ebc, dtype)) {
    /* Must be returned in memory via an sret pointer argument. */
    abi->used_iregs = 1;
    abi->arg[0].kind = LL_ARG_INDIRECT;
    return;
  }

  /* Integer types can be returned in registers, possibly sign/zero extended. */
  if (DT_ISINT(dtype)) {
    abi->arg[0].kind = classify_int(dtype);
    return;
  }

  /* Basic types can be returned in registers directly. */
  if ((DT_ISSCALAR(dtype) && !DT_ISCMPLX(dtype)) || DT_ISVECT(dtype)) {
    abi->arg[0].kind = LL_ARG_DIRECT;
    return;
  }

  /* Other types need to be coerced into one or two arguments corresponding
     to the registers. */
  amd64_coerce(abi->module, &abi->arg[0], ebc, dtype);
}

void
ll_abi_classify_arg_dtype(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype)
{
  enum amd64_class ebc[2];
  bool inregs;

  inregs = amd64_classify(ebc, dtype) == 0;

  /* If inregs is true, this argument can be passed in registers, but only if
     enough are available. */
  unsigned need_iregs = (ebc[0] == AMD64_INTEGER) + (ebc[1] == AMD64_INTEGER);
  unsigned need_fregs = (ebc[0] == AMD64_SSE) + (ebc[1] == AMD64_SSE);

  /* %rdi, %rsi, %rdx, %rcx, %r8 and %r9. */
  if (abi->used_iregs + need_iregs > 6)
    inregs = 0;
  /* %xmm0 - %xmm7 */
  if (abi->used_fregs + need_fregs > 8)
    inregs = 0;

  if (inregs) {
    /* Registers are available, update the counters. */
    abi->used_iregs += need_iregs;
    abi->used_fregs += need_fregs;
  }

  /* Integer types can be passed in registers, possibly sign/zero extended. */
  if (DT_ISINT(dtype)) {
    arg->kind = classify_int(dtype);
    return;
  }

  /* For simple types, LLVM figures out whether they go on the stack or in
     registers. */
  if ((DT_ISSCALAR(dtype) && !DT_ISCMPLX(dtype)) || DT_ISVECT(dtype)) {
    arg->kind = LL_ARG_DIRECT;
    return;
  }

  /* This disables coercion for Fortran... for now, including ISO-C.
   * We probably want to lift the ISO-C restriction in the future.
   */
  if (inregs)
    amd64_coerce(abi->module, arg, ebc, dtype);
  else
    arg->kind = LL_ARG_BYVAL;
}

/* Map values for __builtin_va_genarg(). See rte/pgc/hammer/src/va_arg.c. */
#define GP_XM 0
#define XM_GP 1
#define XM_XM 2

unsigned
ll_abi_classify_va_arg_dtype( LL_Module* module, DTYPE dtype, 
                              unsigned *num_gp, unsigned *num_fp)
{
  enum amd64_class ebc[2];

  if (amd64_classify(ebc, dtype) == 0) {
    *num_gp = (ebc[0] == AMD64_INTEGER) + (ebc[1] == AMD64_INTEGER);
    *num_fp = (ebc[0] == AMD64_SSE) + (ebc[1] == AMD64_SSE);

    if (ebc[0] == AMD64_INTEGER && ebc[1] == AMD64_SSE)
      return GP_XM;
    if (ebc[0] == AMD64_SSE && ebc[1] == AMD64_INTEGER)
      return XM_GP;
  } else {
    *num_fp = 0;
    *num_gp = 0;
  }
  /* Return value is only used when we end up calling __builtin_va_genarg(). */
  return XM_XM;
}

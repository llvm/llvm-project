/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file llutil.c
   Contains misc. utility routines for LLVM Code Generator
 */

#include "llutil.h"
#include "dinitutl.h"
#include "dinit.h"
#include "ll_write.h"
#include "lldebug.h"
#include "dtypeutl.h"
#include "llassem.h"
#include "llassem_common.h"
#include "cgllvm.h"
#include "cgmain.h"
#include "mach.h"
#include "symfun.h"

typedef struct LLDEF {
  DTYPE dtype;
  LL_Type *ll_type;
  int sptr;
  int rank;
  unsigned flags;	/**< bitmask value. See LLDEF_Flags */
  const char *name;
  int printed;
  int addrspace;
  OPERAND *values;
  struct LLDEF *next;
} LLDEF;

#if DEBUG
static const char *ot_names[OT_LAST] = {
    "OT_NONE",   "OT_CONSTSPTR", "OT_VAR",  "OT_TMP",        "OT_LABEL",
    "OT_CC",     "OT_TARGET",    "OT_CALL", "OT_CONSTVAL",   "OT_UNDEF",
    "OT_MDNODE", "OT_MEMBER",    "OT_DEF",  "OT_CONSTSTRING"};

const char *
get_ot_name(unsigned ot)
{
  return (ot < OT_LAST) ? ot_names[ot] : "";
}
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

#define DT_VOID_NONE DT_NONE

#define DT_SBYTE DT_BINT

static const char *llvm_cc_names[LLCC_LAST] = {
    "none", "eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"};

static const char *llvm_ccfp_names[LLCCF_LAST] = {
    "none", "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
    "ueq",  "ugt",   "uge", "ult", "ule", "une", "uno", "true"};

/* struct definition only used in CPU llvm backend
 * accel takes a different approach */
static LLDEF *struct_def_list = NULL;
static LLDEF *llarray_def_list = NULL;
/* global variable declaration for GPU llvm backend
 * CPU takes another approach, please check assemble_end in llassem_c.c. */
static LLDEF *gblvar_def_list = NULL;

FTN_LLVM_ST ftn_llvm_st;
FILE *LLVMFIL = NULL;

static LL_ABI_Info *ll_abi_for_missing_prototype(LL_Module *module,
                                                 DTYPE return_dtype,
                                                 int func_sptr, int jsra_flags);
static bool LLTYPE_equiv(LL_Type *ty1, LL_Type *ty2);

static int is_gpu_module = false;

void 
llvm_set_acc_module(void)
{
  is_gpu_module = true;
}

void 
llvm_set_cpu_module(void)
{
  is_gpu_module = false;
}

LL_Module*
llvm_get_current_module(void)
{
  /* only TARGET_LLVM is defined; it is impossible to have both
   * TARGET_ACCEL_LLVM and TARGET_LLVM undefined (Accel LLVM hasn't
   * been enabled yet on ARM platform) */
  return cpu_llvm_module;
}

void
llutil_struct_def_reset(void)
{
  /* TODO: Please don't leak this */
  struct_def_list = NULL;
}

void
llutil_gblvar_def_reset(void)
{
  /* TODO: Please don't leak this either */
  gblvar_def_list = NULL;
}

void
llutil_def_reset(void)
{
  llutil_struct_def_reset();
  llutil_gblvar_def_reset();
}

void
llutil_dfile_init(void)
{
#if DEBUG
  ll_dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
#endif
}

static char *
llutil_alloc(INT size)
{
  char *p = (char *)getitem(LLVM_LONGTERM_AREA, size);
  memset(p, 0, size);
  return p;
}

const char *
llutil_strdup(const char *str)
{
  char *p = llutil_alloc(strlen(str) + 1);
  return strcpy(p, str);
}

/**
   \brief allocate a new \c TMPS structure
 */
TMPS *
make_tmps(void)
{
  return (TMPS *)llutil_alloc(sizeof(TMPS));
}

void
ll_add_func_proto(int sptr, unsigned flags, int nargs, DTYPE *args)
{
  int i;
  LL_Type *fty;
  const DTYPE dtype = DTYPEG(sptr);
  LL_Type **fsig = (LL_Type **)malloc(sizeof(LL_Type *) * (nargs + 1));
  LL_ABI_Info *abi = ll_abi_alloc(llvm_get_current_module(), nargs);

  ll_proto_init();
  abi->arg[0].type = fsig[0] = make_lltype_from_dtype(dtype);
  abi->arg[0].kind = LL_ARG_DIRECT;
  for (i = 0; i < nargs; ++i) {
    abi->arg[1 + i].type = fsig[1 + i] =
      make_lltype_from_dtype(args[i]);
    abi->arg[1 + i].kind = LL_ARG_DIRECT;
  }
  fty = ll_create_function_type(llvm_get_current_module(), fsig, nargs, false);
  abi->is_fortran = true;
  abi->is_iso_c = CFUNCG(sptr);
  abi->is_pure = PUREG(sptr);
  abi->fast_math = (flags & FAST_MATH_FLAG) != 0;
  ll_proto_add(SYMNAME(sptr), abi);
  free(fsig);
}

/**
   \brief Compute load/store instruction flag bits corresponding to dtype.
   \param dtype  The DTYPE

   The flags encode alignment in the \c LDST_LOGALIGN_MASK bits and volatile
   types have the \c VOLATILE_FLAG bit set.

   The returned flags are pre-shifted so they can be or'ed onto the instruction
   flags.
 */
LL_InstrListFlags
ldst_instr_flags_from_dtype(DTYPE dtype)
{
  unsigned align = alignment(dtype);
  unsigned logalign = 0;
  unsigned flags = 0;

  /* Align is on the form 2^n-1. Compute n. */
  while (align) {
    logalign++;
    align >>= 1;
  }
  flags |= logalign << LDST_LOGALIGN_SHIFT;

#ifdef MOD_VOLATILE
  /* We should not be relying on MOD_VOLATILE to detect volatile loads
     and stores in ILI.  See routine ldst_instr_flags_from_dtype_and_nme
     for right way to do it.  When we're sure we have it right, the
     code here should be deleted, and the description of the routine updateb. */
  if (DTY(dtype) == TY_MOD && (DTY(dtype + 2) & MOD_VOLATILE))
    flags |= VOLATILE_FLAG;
#endif

  return (LL_InstrListFlags)flags;
}

/**
   \brief Compute load/store instruction flag bits corresponding to dtype/nme.
   \param dtype  The DTYPE
   \param nme    The NME

   The flags encode alignment in the \c LDST_LOGALIGN_MASK bits and the nme
   NME_VOL
   has the \c VOLATILE_FLAG bit set.

   The returned flags are pre-shifted so they can be or'ed onto the instruction
   flags.
 */
LL_InstrListFlags
ldst_instr_flags_from_dtype_nme(DTYPE dtype, int nme)
{
  unsigned flags = ldst_instr_flags_from_dtype(dtype);
  if (nme == NME_VOL)
    flags |= VOLATILE_FLAG;
  return (LL_InstrListFlags)flags;
}

/*
 * Convert a basic non-integer dtype to the corresponding LL_Type in module.
 */
static LL_Type *
ll_convert_basic_dtype_with_addrspace(LL_Module *module, DTYPE dtype, int addrspace)
{
  enum LL_BaseDataType basetype = LL_NOTYPE;
  LL_Type *type;

  switch (DTY(dtype)) {
  case TY_NONE:
    basetype = LL_VOID;
    break;
  case TY_FLOAT:
  case TY_CMPLX:
    basetype = LL_FLOAT;
    break;
  case TY_DBLE:
  case TY_DCMPLX:
    basetype = LL_DOUBLE;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
  case TY_QCMPLX:
    /* TY_QUAD maps to an IEEE128 quad precision. */
#endif
  case TY_FLOAT128:
  case TY_CMPLX128:
    /* TY_FLOAT128 represents a long double (or __float128) on
     * systems where it maps to an IEEE128 quad precision. */
    basetype = LL_FP128;
    break;

  default:
    interr("ll_convert_basic_dtype: unknown data type", dtype, ERR_Fatal);
  }

  type = ll_create_basic_type(module, basetype, addrspace);

  if (DT_ISCMPLX(dtype)) {
    LL_Type *pair[2] = {type, type};
    type = ll_create_anon_struct_type(module, pair, 2, /*FIXME*/ true, addrspace);
  }

  return type;
}

#if defined(TARGET_LLVM_X8664)
/**
 * \brief Convert a SIMD dtype to the corresponding LLVM type.
 *
 * Examples of SIMD dtypes are DT_128, DT_128F, DT_256, DT_512.
 */
static LL_Type *
ll_convert_simd_dtype(LL_Module *module, DTYPE dtype)
{
  enum LL_BaseDataType base;
  unsigned num_elements;
  LL_Type *base_type;
  switch (dtype) {
  case DT_128:
  case DT_128I:
  case DT_256:
  case DT_256I:
  case DT_512:
  case DT_512I:
    base = LL_I32;
    break;
  case DT_128F:
  case DT_256F:
    base = LL_FLOAT;
    break;
  case DT_128D:
  case DT_256D:
    base = LL_DOUBLE;
    break;
  default:
    interr("ll_convert_simd_dtype: unhandled dtype", dtype, ERR_Fatal);
    return NULL;
  }
  base_type = ll_create_basic_type(module, base, 0);
  num_elements = size_of(dtype) / ll_type_bytes(base_type);
  return ll_get_vector_type(base_type, num_elements);
}
#endif

/**
   This routine is for use with fortran interfaces, specified by sptr
 */
static LL_Type *
ll_convert_iface_sptr(LL_Module *module, SPTR iface_sptr)
{
  int i, n_args;
  SPTR gblsym;
  LL_Type **args, *res;
  LL_Type *llt;
  DTLIST *dtl;

  if (INMODULEG(iface_sptr))
    gblsym = find_ag(get_llvm_name(iface_sptr));
  else {
    if (!(gblsym = find_ag(get_llvm_ifacenm(iface_sptr))))
      gblsym = local_funcptr_sptr_to_gblsym(iface_sptr);
  }
  assert(gblsym, "ll_convert_iface_sptr: No gblsym found", iface_sptr, ERR_Fatal);

  n_args = get_ag_argdtlist_length(gblsym);
  args = (LL_Type**)calloc(1, (1 + n_args) * sizeof(LL_Type *));

  /* Return type */
  llt = get_ag_lltype(gblsym);
  args[0] = ll_convert_dtype(module, DTYPEG(iface_sptr));

  for (i = 1, dtl = get_argdtlist(gblsym); dtl;
       dtl = get_next_argdtlist(dtl), ++i) {
    llt = (LL_Type *)get_lltype_from_argdtlist(dtl);
    args[i] = llt;
  }

  res = ll_create_function_type(module, args, n_args, false);
  free(args);
  return res;
}

/**
 * \brief Layout the body of a struct type by scanning the member symbol table
 * entries starting at member_sptr, and call ll_set_struct_body(struct_type).
 *
 * This code can layout both struct/union dtypes and common blocks.
 *
 * The provided struct_type should be created with
 * ll_create_named_struct_type().
 *
 * Padding will be added to make the size of the new struct size_bytes, unless
 * size_bytes is -1 which is ignored.
 */
void
layout_struct_body(LL_Module *module, LL_Type *struct_type, int member_sptr,
                   ISZ_T size_bytes)
{
  int sptr;
  int packed = 0;
  int padded = 0;
  unsigned nmemb = 0;
  LL_Type **memb_type;
  unsigned *memb_off;
  char *memb_pad, *cp;
  ISZ_T bytes = 0;

  /* Count the number of struct members so we can size the allocations. */
  for (sptr = member_sptr; sptr > NOSYM; sptr = SYMLKG(sptr))
    nmemb++;

  /* Worst case struct we have to build has padding before every member + tail
   * padding. */
  memb_type = (LL_Type**)malloc(sizeof(LL_Type *) * (2 * nmemb + 1));
  memb_off = (unsigned*)malloc(sizeof(unsigned) * (2 * nmemb + 2));
  memb_pad = (char*)calloc((2 * nmemb) + 1, 1);
  nmemb = 0;

  /* Revisit struct members while keeping track if the built struct size so
   * far in 'bytes'. Only add a typed member if:
   *
   * - Member is aligned according to its datatype. This way we can avoid LLVM
   * packed structs.
   * - Member doesn't overlap the struct built so far. This would happen for
   *   union members and bitfields.
   * - Member doesn't extend beyond the end of the struct.
   *
   * If we choose to not add a member, it will be part of the padding added
   * after it.
   */
  for (sptr = member_sptr; sptr > NOSYM; sptr = SYMLKG(sptr)) {
    unsigned cur_size = 0;
    LL_Type *cur_type = NULL;

#ifdef PACKG
    packed = packed || PACKG(sptr);
#endif

    if (ADDRESSG(sptr) < bytes)
      continue;

    if (size_bytes != -1 && ADDRESSG(sptr) >= size_bytes)
      continue;

    if ((!packed) && (alignment(DTYPEG(sptr)) & ADDRESSG(sptr)))
      continue;

#ifdef POINTERG
    if (POINTERG(sptr)) {
      cur_type = ll_convert_dtype(module, DDTG(DTYPEG(sptr)));
      cur_type = ll_get_pointer_type(cur_type);
      cur_size = ll_type_bytes(cur_type);
    }
#endif /* POINTERG */

    /* Otherwise use the normal dtype. */
    if (!cur_type) {
      cur_type = ll_convert_dtype(module, DTYPEG(sptr));
      if (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
          DDTG(DTYPEG(sptr)) == DT_DEFERCHAR)
        cur_size = ZSIZEOF(DT_ADDR);
      else if (DTY(DTYPEG(sptr)) == TY_ARRAY && extent_of(DTYPEG(sptr)) == 0)
        cur_size = 0;
      else
        cur_size = ZSIZEOF(DTYPEG(sptr));
    }

    /* Skip empty struct array members. */
    if (!cur_size)
      continue;

    if (size_bytes != -1 && ADDRESSG(sptr) + cur_size > size_bytes)
      continue;

    /* Add padding before. Use an [n x i8] array if needed. */
    if (ADDRESSG(sptr) > bytes) {
      unsigned pad_size = ADDRESSG(sptr) - bytes;
      LL_Type *pad_type = ll_create_basic_type(module, LL_I8, 0);
      if (pad_size > 1)
        pad_type = ll_get_array_type(pad_type, pad_size, 0);

      memb_off[nmemb] = bytes;
      memb_pad[nmemb] = padded = 1;
      memb_type[nmemb++] = pad_type;
      bytes += pad_size;
    }

    /* Add current member. */
    memb_off[nmemb] = bytes;
    memb_type[nmemb++] = cur_type;
    bytes += cur_size;
  }

  /* Finally add tail padding. */
  if (size_bytes > bytes) {
    unsigned pad_size = size_bytes - bytes;
    LL_Type *pad_type = ll_create_basic_type(module, LL_I8, 0);
    if (pad_size > 1)
      pad_type = ll_get_array_type(pad_type, pad_size, 0);
    memb_off[nmemb] = bytes;
    memb_pad[nmemb] = padded = 1;
    memb_type[nmemb++] = pad_type;
    bytes += pad_size;
  }

  assert(size_bytes == -1 || bytes == size_bytes, "Inconsistent size", bytes,
         ERR_Fatal);
  memb_off[nmemb] = size_bytes;
  cp = padded ? memb_pad : NULL;
  ll_set_struct_body(struct_type, memb_type, memb_off, cp, nmemb, packed);
  free(memb_pad);
  free(memb_type);
  free(memb_off);
}

/*
 * Convert a TY_STRUCT or TY_UNION dtype to an LLVM LL_STRUCT type.
 *
 * LLVM can't represent full C structs and unions; it has no bitfield concept
 * or union support. We build an LLVM struct type that has matching members
 * where possible, and i8 padding otherwise.
 */
static LL_Type *
ll_convert_struct_dtype(LL_Module *module, DTYPE dtype)
{
  /* TY_STRUCT sptr size tag align ict */
  const SPTR member_sptr = DTyAlgTyMember(dtype);
  const unsigned size_bytes = DTyAlgTySize(dtype);
  const SPTR tag_sptr = DTyAlgTyTag(dtype);
  const char *prefix = DTY(dtype) == TY_UNION ? "union" : "struct";
  LL_Type *old_type;
  LL_Type *new_type;

  /* Was this struct converted previously? Named structs are indexed by dtype.
   */
  old_type = ll_get_struct_type(module, dtype, false);
  if (old_type)
    return old_type;

  /* No, this has not been converted yet, so we need to create a new named
   * struct.
   *
   * Create an empty struct right away and fill in the body later. This is
   * necessary because we recursively call ll_convert_dtype() while
   * converting the struct body. Once the empty struct is created, the
   * recursion will be terminated by ll_get_struct_type() above.
   *
   * The name picked for the type is not important,
   * ll_create_named_struct_type() will ensure a unique type name.
   */
  if (tag_sptr)
    new_type = ll_create_named_struct_type(module, dtype, true, "%s.%s", prefix,
                                           SYMNAME(tag_sptr));
  else
    new_type = ll_create_named_struct_type(module, dtype, true, "a%s.dt%d",
                                           prefix, dtype);

/* Make sure that the old-style struct definition exists. For now this is
 * how struct definitions are printed. The mutual recursion between these
 * functions is terminated by the ll_get_struct_type() call above returning
 * non-NULL.
 *
 * This is only required for the CPU code generator. The GPU code
 * generators use ll_write_user_structs(), so don't depend on
 * process_dtype_struct().
 */
  if (module == cpu_llvm_module)
    process_dtype_struct(dtype);

  layout_struct_body(module, new_type, member_sptr, size_bytes);

  return new_type;
}

/**
 * \brief Convert a Fortran-style \c TY_ARRAY dtype to an LLVM array.
 *
 * This routine obtains the length information via the array descriptor.
 */
LL_Type *
ll_convert_array_dtype(LL_Module *module, DTYPE dtype, int addrspace)
{
  int len = 0;
  LL_Type *type = NULL;

  if (DTY(dtype) == TY_ARRAY) {
    DTYPE ddtype = DTySeqTyElement(dtype);
    ADSC *ad = AD_DPTR(dtype);
    int numdim = AD_NUMDIM(ad);
    int numelm = AD_NUMELM(ad);

    type = ll_convert_dtype(module, ddtype);

    if (numdim >= 1 && numdim <= 7) {
      /* Create nested LLVM arrays. */
      int i;
      for (i = 0; i < numdim; i++)
        type = ll_get_array_type(type, get_dim_size(ad, i), addrspace);
      return type;
    }

    if (numelm) {
      assert((STYPEG(numelm) == ST_CONST) || (STYPEG(numelm) == ST_VAR),
             "Array length is neither a constant nor variable", numelm, ERR_unused);
      len = (STYPEG(numelm) == ST_CONST) ? get_bnd_cval(numelm) : 0;
    } else {
      len = 0;
    }
  } else if (DTY(dtype) == TY_CHAR) {
    len = DTyCharLength(dtype);
    if (len == 0)
      len = 1;
    type = ll_convert_dtype(module, DT_BINT);
  } else if (DTY(dtype) == TY_NCHAR) {
    len = DTyCharLength(dtype);
    if (len == 0)
      len = 1;
    type = ll_convert_dtype(module, DT_SINT);
  } else
    interr("ll_convert_array_dtype: unhandled dtype", dtype, ERR_Fatal);

  /* The array dimension is a symbol table reference.
   * Use [0 x t] for variable-sized array types.
   */
  return ll_get_array_type(type, len, 0);
}



static LL_Type *
convert_dtype(LL_Module *module, DTYPE dtype, int addrspace)
{
  LL_Type *subtype;
  DTYPE dt;

  switch (DTY(dtype)) {
  default:
    break;

  case TY_NONE:
  case TY_ANY:
  case TY_NUMERIC:
    return ll_create_basic_type(module, LL_VOID, addrspace);

  case TY_PTR:
    dt = DTySeqTyElement(dtype);
    if (DTY(dt) == TY_PROC)
      subtype = ll_create_basic_type(module, LL_I8, addrspace);
    else
      subtype = ll_convert_dtype_with_addrspace(module, DTySeqTyElement(dtype), addrspace);
    /* LLVM doesn't have void pointers. Use i8* instead. */
    if (subtype->data_type == LL_VOID)
      subtype = ll_create_basic_type(module, LL_I8, addrspace);
    return ll_get_pointer_type(subtype);

  case TY_CHAR:
  case TY_NCHAR:
  case TY_ARRAY:
    return ll_convert_array_dtype(module, dtype, addrspace);

  case TY_STRUCT:
  case TY_UNION:
    return ll_convert_struct_dtype(module, dtype);

  case TY_VECT:
    subtype = ll_convert_dtype_with_addrspace(module, DTySeqTyElement(dtype), addrspace);
    return ll_get_vector_type(subtype, DTyVecLength(dtype));

#if defined(TARGET_LLVM_X8664)
  case TY_128:
  case TY_256:
  case TY_512:
    return ll_convert_simd_dtype(module, dtype);
#endif
  }
  if (DT_ISINT(dtype))
    return ll_create_int_type_with_addrspace(module, 8 * size_of(dtype), addrspace);

  if (DT_ISBASIC(dtype))
    return ll_convert_basic_dtype_with_addrspace(module, dtype, addrspace);

  interr("ll_convert_dtype: unhandled dtype", dtype, ERR_Fatal);
  return NULL;
}

/**
 * \brief Convert any kind of dtype to an LLVM type.
 */
LL_Type *
ll_convert_dtype(LL_Module *module, DTYPE dtype)
{
  return convert_dtype(module, dtype, 0);
}

/**
 * \brief Convert any kind of dtype to an LLVM type with address space.
 */
LL_Type *
ll_convert_dtype_with_addrspace(LL_Module *module, DTYPE dtype, int addrspace)
{
  return convert_dtype(module, dtype, addrspace);
}

bool
llis_integral_kind(DTYPE dtype)
{
  switch (DTY(dtype)) {
#if defined(PGC) || defined(PG0CL)
  case TY_LONG:
  case TY_ULONG:
  case TY_SCHAR:
  case TY_UCHAR:
  case TY_ENUM:
  case TY_BOOL:
#endif
  case TY_WORD:
  case TY_DWORD:
  case TY_HOLL:
  case TY_BINT:
  case TY_UBINT:
  case TY_INT128:
  case TY_UINT128:
  case TY_LOG:
  case TY_SLOG:
  case TY_BLOG:
  case TY_LOG8:
  case TY_INT:
  case TY_UINT:
  case TY_SINT:
  case TY_USINT:
  case TY_INT8:
  case TY_UINT8:
    return 1;
  default:
    break;
  }
  return 0;
}

bool
llis_pointer_kind(DTYPE dtype)
{
  return (DTY(dtype) == TY_PTR);
}

bool
llis_array_kind(DTYPE dtype)
{
  switch (DTY(dtype)) {
  case TY_CHAR:
  case TY_NCHAR:
  case TY_ARRAY:
    return true;
  default:
    break;
  }
  return false;
}

bool
llis_dummied_arg(SPTR sptr)
{
  return sptr && (SCG(sptr) == SC_DUMMY) &&
         (llis_pointer_kind(DTYPEG(sptr)) || llis_array_kind(DTYPEG(sptr)));
}

bool
llis_vector_kind(DTYPE dtype)
{
  return (DTY(dtype) == TY_VECT);
}

bool
llis_struct_kind(DTYPE dtype)
{
  switch (DTY(dtype)) {
  case TY_CMPLX128:
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
  case TY_STRUCT:
  case TY_UNION:
    return true;
  default:
    break;
  }
  return false;
}

bool
llis_function_kind(DTYPE dtype)
{
  switch (DTY(dtype)) {
  case TY_PROC:
    return true;
  default:
    break;
  }
  return false;
}

int
is_struct_kind(DTYPE dtype, bool check_return,
               bool return_vector_as_struct)
{
  switch (DTY(dtype)) {
  case TY_STRUCT:
  case TY_UNION:
    return true;
  case TY_VECT:
    return return_vector_as_struct;
  case TY_CMPLX:
    return check_return;
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
  case TY_CMPLX128:
    return true;
  default:
    return false;
  }
}

LL_Type *
make_ptr_lltype(LL_Type *pts_to)
{
  return ll_get_pointer_type(pts_to);
}

LL_Type *
make_int_lltype(unsigned bits)
{
  return ll_create_int_type(llvm_get_current_module(), bits);
}

LL_Type *
make_void_lltype(void)
{
  return ll_create_basic_type(llvm_get_current_module(), LL_VOID, LL_AddrSp_Default);
}

LL_Type *
make_vector_lltype(int size, LL_Type *pts_to)
{
  return ll_get_vector_type(pts_to, size);
}

LL_Type *
make_array_lltype(int size, LL_Type *pts_to)
{
  return ll_get_array_type(pts_to, size, LL_AddrSp_Default);
}

int
get_dim_size(ADSC *ad, int dim)
{
  int dim_size = 0;
  const int lower_bnd = AD_LWBD(ad, dim);
  const int upper_bnd = AD_UPBD(ad, dim);

  if (STYPEG(upper_bnd) == ST_CONST && STYPEG(lower_bnd) == ST_CONST)
    dim_size = (int)(ad_val_of(upper_bnd) - ad_val_of(lower_bnd) + 1);
  return dim_size;
}

static LL_Type *
lltype_from_dtype(DTYPE dtype, int addrspace)
{
  DTYPE sdtype;

  sdtype = dtype;
  return ll_convert_dtype_with_addrspace(llvm_get_current_module(), sdtype, addrspace);
}

LL_Type *
make_lltype_from_dtype(DTYPE dtype)
{
  return lltype_from_dtype(dtype, 0);
}

LL_Type *
make_lltype_from_dtype_with_addrspace(DTYPE dtype, int addrspace)
{
  return lltype_from_dtype(dtype, addrspace);
}

DTYPE
generic_dummy_dtype(void)
{
  return TARGET_PTRSIZE == 8 ? DT_UINT8 : DT_UINT;
}

/* This was originally just i8*, but to avoid only loading 1 byte,
 * we now represent dummys as i32* or i64* in fortran.
 */
LL_Type *
make_generic_dummy_lltype(void)
{
  return make_ptr_lltype(make_lltype_from_dtype(generic_dummy_dtype()));
}

  /* Until we have prototype available, we are making assumption here:
   *
   * 1) This function is called for module subroutine calling its own module
   * subroutine
   * 2) Sectional arguments may not be handled correctly.
   * 3) Assumed-size/adjustable/defered char arguments if passing as arguments
   *    to another contained subroutine in the same module - will need to be
   *    the same type?
   */

LL_Type *
make_lltype_from_arg(int arg)
{
  assert(0, "", 0, ERR_Fatal);
  return 0;
} /* make_lltype_from_dtype */

/* create expected type from actual arguments - all arguments are char*(or i8*)
 * else if pass by value - pass actual type.
 */

LL_Type *
make_lltype_from_arg_noproto(int arg)
{
  DTYPE dtype;
  LL_Type *llt, *llt2;
  int argili;

  argili = ILI_OPND(arg, 1);
  dtype = ILI_DTyOPND(arg, 3);
  if (IL_RES(ILI_OPC(argili)) == ILIA_AR) { /* by reference */
    if (DTY(dtype) != TY_ARRAY && DTY(dtype) != TY_PTR && DTY(dtype) != TY_ANY)
      llt2 = make_lltype_from_dtype(dtype);
    else
      llt2 = make_lltype_from_dtype(DT_BINT);
    llt = make_ptr_lltype(llt2);

  } else {
    llt = make_lltype_from_dtype(dtype);
  }

  DBGTRACEOUT2(" return type %p: %s\n", llt, llt->str);

  return llt;
} /* make_lltype_from_dtype */

/**
   \brief Get a function argument dtype from an IL_ARG* instruction opcode.
 */
DTYPE
get_dtype_from_arg_opc(ILI_OP opc)
{
  switch (opc) {
  case IL_ARGIR:
  case IL_DAIR:
    return DT_INT;
  case IL_ARGSP:
  case IL_DASP:
    return DT_FLOAT;
  case IL_ARGDP:
  case IL_DADP:
    return DT_DBLE;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_ARGQP:
  case IL_DAQP:
    return DT_QUAD;
#endif
  case IL_ARGAR:
  case IL_DAAR:
    return DT_CPTR;
  case IL_ARGKR:
  case IL_DAKR:
    return DT_INT8;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128ARG:
    return DT_FLOAT128;
#endif
  default:
    return DT_NONE;
  }
} /* get_dtype_from_arg_opc */

/**
   \brief Convert a <tt>TY_</tt><i>*</i> to a <tt>DT_</tt><i>*</i> value

   If the TY type isn't a basic type, returns <tt>DT_NONE</tt>.
 */
DTYPE
get_dtype_from_tytype(TY_KIND ty)
{
  assert((ty >= TY_NONE) && (ty < TY_MAX), "DTY not in range", ty, ERR_Fatal);
  switch (ty) {
  case TY_WORD:
    return DT_WORD;
  case TY_DWORD:
    return DT_DWORD;
  case TY_HOLL:
    return DT_HOLL;
  case TY_BINT:
    return DT_BINT;
  case TY_INT:
    return DT_INT;
  case TY_UINT:
    return DT_UINT;
  case TY_SINT:
    return DT_SINT;
  case TY_USINT:
    return DT_USINT;
#ifdef PGF
  case TY_CHAR:
    return DT_CHAR;
#endif
  case TY_NCHAR:
    return DT_NCHAR;
#ifdef PGF
  case TY_REAL:
    return DT_REAL;
#endif
  case TY_DBLE:
    return DT_DBLE;
  case TY_QUAD:
    return DT_QUAD;
  case TY_CMPLX:
    return DT_CMPLX;
  case TY_DCMPLX:
    return DT_DCMPLX;
  case TY_INT8:
    return DT_INT8;
  case TY_UINT8:
    return DT_UINT8;
  case TY_128:
    return DT_128;
  case TY_256:
    return DT_256;
  case TY_512:
    return DT_512;
  case TY_INT128:
    return DT_INT128;
  case TY_UINT128:
    return DT_UINT128;
  case TY_FLOAT128:
    return DT_FLOAT128;
  case TY_CMPLX128:
    return DT_CMPLX128;
  case TY_PTR:
    return DT_CPTR;
  case TY_BLOG:
    return DT_BLOG;
  case TY_SLOG:
    return DT_SLOG;
  case TY_LOG:
    return DT_LOG;
  case TY_LOG8:
    return DT_LOG8;
  default:
    return DT_NONE;
  }
}

/**
   \brief Convert a <tt>nme</tt><i>*</i> to a <tt>DT_</tt><i>*</i> value

   If the DT_ value can't be determined, returns <tt>DT_NONE</tt>.
 */
DTYPE
get_dtype_for_vect_type_nme(int nme) {
  DTYPE dtype = DT_NONE;
  if (nme) {
    SPTR sym = basesym_of(nme);
    if (sym != SPTR_NULL) {
      dtype = DTYPEG(sym);
      assert(DTY(dtype) == TY_VECT, "Not a vect type", dtype, ERR_Fatal);
      switch(size_of(dtype)) {
      case 1:
        dtype = DT_CHAR;
        break;
      case 2:
        dtype = DT_SINT;
        break;
      case 4:
        switch (DTySeqTyElement(dtype)) {
        case DT_FLOAT:
          dtype = DT_FLOAT;
          break;
        default:
          dtype = DT_INT;
        }
        break;
      case 8:
        switch (DTySeqTyElement(dtype)) {
        case DT_FLOAT:
        case DT_DBLE:
          dtype = DT_DBLE;
          break;
        default:
          dtype = DT_INT8;
        }
        break;
      case 16:
        switch (DTySeqTyElement(dtype)) {
        case DT_FLOAT:
          dtype = DT_128F;
          break;
        case DT_DBLE:
          dtype = DT_128D;
          break;
        default:
          dtype = DT_128;
        }
        break;
      case 32:
        switch (DTySeqTyElement(dtype)) {
        case DT_FLOAT:
          dtype = DT_256F;
          break;
        case DT_DBLE:
          dtype = DT_256D;
          break;
        default:
          dtype = DT_256;
        }
        break;
      }
    }
  }

  return dtype;
}

/**
   \brief Get the function return type coprresponding to an IL_DFR* opcode.
 */
DTYPE
dtype_from_return_type(ILI_OP ret_opc)
{
  switch (ret_opc) {
  case IL_DFRAR:
    return DT_CPTR;
#ifdef IL_DFRSPX87
  case IL_DFRSPX87:
#endif
  case IL_DFRSP:
    return DT_FLOAT;
  case IL_DFR128:
    return DT_128;
  case IL_DFR256:
    return DT_256;
#ifdef IL_DFRDPX87
  case IL_DFRDPX87:
#endif
  case IL_DFRDP:
    return DT_DBLE;
#ifdef TARGET_SUPPORTS_QUADFP
  case IL_DFRQP:
    return DT_QUAD;
#endif
  case IL_DFRIR:
    return DT_INT;
  case IL_DFRKR:
    return DT_INT8;
  case IL_DFRCS:
    return DT_CMPLX;
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
    return DT_FLOAT128;
#endif
  default:
    interr("dtype_from_return_type(), bad return opc", ret_opc, ERR_Fatal);
  }
  return DT_NONE;
}

LL_Type *
make_lltype_from_iface(SPTR sptr)
{
  return ll_convert_iface_sptr(llvm_get_current_module(), sptr);
}

/* Convenience macro (aids readability for is_function predicate) */
#define IS_FTN_PROC_PTR(sptr) \
  ((DTY(DTYPEG(sptr)) == TY_PTR) && \
   (DTY(DTySeqTyElement(DTYPEG(sptr))) == TY_PROC))

bool
is_function(int sptr)
{
  const int stype = STYPEG(sptr);
  return (stype == ST_ENTRY || stype == ST_PROC || IS_FTN_PROC_PTR(sptr));
}

static void
add_def(LLDEF *new_def, LLDEF **def_list)
{
  new_def->next = *def_list;
  *def_list = new_def;
  if ((new_def->ll_type == NULL) && (new_def->dtype > 0))
    new_def->ll_type = make_lltype_from_dtype(new_def->dtype);
}

/**
   \brief Make an \c LL_Type from symbol \p sptr
   \param sptr  a symbol
 */
LL_Type *
make_lltype_from_sptr(SPTR sptr)
{
  DTYPE sdtype, atype;
  int anum;
  SPTR iface;
  int stype = 0, sc = 0;
  LL_Type *llt;
  int addrspace = LL_AddrSp_Default;
  ADSC *ad;
  INT d;

  if (sptr) {
    sdtype = DTYPEG(sptr);
    stype = STYPEG(sptr);
    sc = SCG(sptr);
  }
#if defined(HOLLG)
  if ((CUDAG(gbl.currsub) & (CUDA_GLOBAL | CUDA_DEVICE)) &&
      (SCG(sptr) == SC_DUMMY)) {
    /* do nothing */
  } else if (HOLLG(sptr) && STYPEG(sptr) == ST_CONST) {
    return make_ptr_lltype(get_ftn_hollerith_type(sptr));
  } else
#endif
      if (SCG(sptr) == SC_CMBLK) {
    return make_ptr_lltype(get_ftn_cmblk_lltype(sptr));
  } else if (SCG(sptr) == SC_DUMMY) {
    return get_ftn_dummy_lltype(sptr);
  } else if (DESCARRAYG(sptr) && CLASSG(sptr)) {
    return make_ptr_lltype(get_ftn_typedesc_lltype(sptr));
  } else if (SCG(sptr) == SC_STATIC) {
    return make_ptr_lltype(get_ftn_static_lltype(sptr));
  } else if (CFUNCG(sptr) && SCG(sptr) == SC_EXTERN) {
    return make_ptr_lltype(get_ftn_cbind_lltype(sptr));
  } else if (SCG(sptr) == SC_LOCAL && SOCPTRG(sptr)) {
    return make_ptr_lltype(get_local_overlap_vartype());
  }

  assert(sptr, "make_lltype_from_sptr(), no incoming arguments", 0, ERR_Fatal);
  DBGTRACEIN7(" sptr %d (%s), stype = %d (%s), dtype = %d (%s,%d)\n", sptr,
              SYMNAME(sptr), stype, stb.stypes[stype], sdtype,
              stb.tynames[DTY(sdtype)], (int)DTY(sdtype))

  /* Labels */
  if (stype == ST_LABEL) {
    return ll_create_basic_type(llvm_get_current_module(), LL_LABEL, 0);
  }

  /* Functions */
  if (is_function(sptr)) {
    LL_ABI_Info *abi;
    if (IS_FTN_PROC_PTR(sptr)) {
      if ((iface = get_iface_sptr(sptr)))
        return make_ptr_lltype(make_ptr_lltype(make_lltype_from_iface(iface)));
      return make_ptr_lltype(make_lltype_from_dtype(DT_CPTR));
    }
    abi = ll_abi_for_func_sptr(llvm_get_current_module(), sptr, DT_NONE);
    llt = ll_abi_function_type(abi);
    return make_ptr_lltype(llt);
  }

  /* Volatiles */
  if (sptr && VOLG(sptr)) {
    // FIXME -- do nothing? -- should flag for metadata
    DBGTRACE1("#setting type for '%s' to VOLATILE", SYMNAME(sptr));
  }
#ifdef OMP_OFFLOAD_LLVM
  addrspace = OMPACCSHMEMG(sptr) ? LL_AddrSp_NVVM_Shared : LL_AddrSp_NVVM_Generic;
#endif
  /* Initialize llt information, and set initial type */
  llt = ll_convert_dtype_with_addrspace(llvm_get_current_module(), sdtype, addrspace);

      if (llis_integral_kind(sdtype)) {
    /* do nothing */
  } else if (llis_pointer_kind(sdtype)) {
    /* make it i8* - use i32* or i64*  */
    if (sc == SC_DUMMY)
      return make_generic_dummy_lltype();
    if (DTY(sdtype) == TY_PTR && sdtype != DT_ADDR)
      llt = ll_get_pointer_type(make_lltype_from_dtype(DTySeqTyElement(sdtype)));
    else if (sdtype == DT_ADDR)
      llt = ll_get_pointer_type(make_lltype_from_dtype(DT_BINT));
    else
      llt = ll_get_pointer_type(make_lltype_from_dtype(sdtype));
    if (llt->sub_types[0]->data_type == LL_VOID) {
      llt = ll_get_pointer_type(ll_create_int_type(llvm_get_current_module(), 8));
    }
  } else if (llis_array_kind(sdtype)) {
    /* all dummy argument are i32* or i64* */
    if (SCG(sptr) == SC_DUMMY)
      return make_generic_dummy_lltype();
    /* Make all arrays to be <type>* */
    if (DTY(sdtype) == TY_CHAR)
      atype = DT_BINT;
    else if (DTY(sdtype) == TY_NCHAR)
      atype = DT_SINT;
    else
      atype = DDTG(sdtype);
    llt = ll_get_pointer_type(make_lltype_from_dtype_with_addrspace(atype, addrspace));
    if (DTY(sdtype) != TY_CHAR && DTY(sdtype) != TY_NCHAR) {
      ad = AD_DPTR(sdtype);
      d = AD_NUMELM(ad);
      if (d == 0 || STYPEG(d) != ST_CONST) {
        if (XBIT(68, 0x1))
          d = AD_NUMELM(ad) = stb.k1;
        else
          d = AD_NUMELM(ad) = stb.i1;
      }
      anum = ad_val_of(d);
    } else {
      anum = DTySeqTyElement(sdtype);
    }
    if (anum > 0) {
      llt = ll_get_array_type(make_lltype_from_dtype(atype), anum,
                              addrspace);
    }
  } else if (llis_vector_kind(sdtype)) {
    LL_Type *oldLlt = llt;
    DBGTRACE1("#setting dtype %d for vector type", sdtype)

#ifdef TARGET_LLVM_ARM
    if (sc == SC_DUMMY) {
      switch (ZSIZEOF(sdtype)) {
      case 1:
        llt = ll_create_int_type(llvm_get_current_module(), 8);
        break;
      case 2:
        llt = ll_create_int_type(llvm_get_current_module(), 16);
        break;
      case 3:
        // FIXME: why is this promoted to 32 bits?
        // llt = ll_create_int_type(module, 24);
        // break;
      case 4:
        llt = ll_create_int_type(llvm_get_current_module(), 32);
        break;
      default:
        assert(0, "", __LINE__, ERR_Fatal);
      }
    }
#endif // TARGET_LLVM_ARM
    if (oldLlt == llt) {
      // LL_Type *t = make_lltype_from_dtype(DTY(sdtype + 1));
      // llt = ll_get_pointer_type(t);
    }
  } else if (llis_struct_kind(sdtype)) {
    process_dtype_struct(sdtype);
  } else if (llis_function_kind(sdtype)) {
    LL_ABI_Info *abi = ll_abi_for_func_sptr(llvm_get_current_module(), sptr, DT_NONE);
    llt = ll_abi_function_type(abi);
    DBGTRACE1("#setting dtype %d for function type", sdtype)
  }

  /* in LLVM, all variables, except dummies, have memory address
   * by default (either on the stack in the case of locals, or
   * global addresses with global variables), and thus a pointer
   * needs to be prepended to the type.
   */
  if (need_ptr(sptr, sc, sdtype)) {
    llt = ll_get_pointer_type(llt);
  }

  DBGDUMPLLTYPE("returned type is ", llt)
  DBGTRACEOUT1(" return type address %p", llt)

  if ((llt->data_type == LL_ARRAY) || (llt->data_type == LL_PTR)) {
    LLDEF *def = (LLDEF *)llutil_alloc(sizeof(LLDEF));
    def->dtype = sdtype;
    def->sptr = sptr;
    def->ll_type = llt;
    def->addrspace = addrspace;
    add_def(def, &llarray_def_list);
  }
  return llt;
} /* make_lltype_from_sptr */

/* Create an OT_CONSTSPTR operand for the constant sptr. */
OPERAND *
make_constsptr_op(SPTR sptr)
{
  OPERAND *op;

  assert(STYPEG(sptr) == ST_CONST, "Constant sptr required", sptr, ERR_Fatal);
  op = make_operand();
  op->ot_type = OT_CONSTSPTR;
  op->ll_type = make_lltype_from_dtype(DTYPEG(sptr));
  op->val.sptr = sptr;

  return op;
}

static char *
ll_get_string_buf(int string_len, char *base, int skip_quotes)
{
  char *name;
  char *from, *to;
  int len;
  name = (char *)llutil_alloc(string_len * sizeof(char));
  to = name;
  from = base;
  len = string_len;
  while (len--) {
    *to++ = *from++;
  }
  return name;
}

/* Create an OT_CONSTSTRING operand for the constant sptr. */
static OPERAND *
make_conststring_op(int sptr)
{
  OPERAND *op = NULL;
  assert(STYPEG(sptr) == ST_CONST, "Constant sptr required", sptr, ERR_Fatal);
  op = make_operand();
  op->ot_type = OT_CONSTSTRING;
  op->ll_type = make_lltype_from_dtype(DTYPEG(sptr));

  if (sptr && DTY(DTYPEG(sptr)) == TY_CHAR) {
    const int length = ll_type_bytes(op->ll_type);
    op->string = ll_get_string_buf(length, stb.n_base + CONVAL1G(sptr), 1);
  }
  return op;
}

OPERAND *
make_constval_op(LL_Type *ll_type, INT conval0, INT conval1)
{
  OPERAND *op;

  op = make_operand();
  op->ot_type = OT_CONSTVAL;
  op->ll_type = ll_type;
  op->val.conval[0] = conval0;
  op->val.conval[1] = conval1;

  return op;
}

OPERAND *
make_constval_opL(LL_Type *ll_type, INT conval0, INT conval1, INT conval2,
                  INT conval3)
{
  OPERAND *op;

  op = make_operand();
  op->ot_type = OT_CONSTVAL;
  op->ll_type = ll_type;
  op->val.conval[0] = conval0;
  op->val.conval[1] = conval1;
  op->val.conval[2] = conval2;
  op->val.conval[3] = conval3;

  return op;
}

OPERAND *
make_constval32_op(int idx)
{
  return make_constval_op(make_lltype_from_dtype(DT_INT), idx, 0);
}

static LL_Type *
set_vect3_to_size4(LL_Type *ll_type)
{
  switch (ll_type->data_type) {
  case LL_ARRAY:
    ll_type = ll_get_array_type(set_vect3_to_size4(ll_type->sub_types[0]),
                                ll_type->sub_elements, ll_type->addrspace);
    break;
  case LL_VECTOR:
    if (ll_type->sub_elements == 3)
      ll_type = ll_get_vector_type(ll_type->sub_types[0], 4);
    break;
  case LL_PTR:
    ll_type = ll_get_pointer_type(set_vect3_to_size4(ll_type->sub_types[0]));
    break;
  default:
    break;
  }
  return ll_type;
}

LL_Type *
make_lltype_sz4v3_from_sptr(SPTR sptr)
{
  LL_Type *llt = make_lltype_from_sptr(sptr);
  return set_vect3_to_size4(llt);
}

LL_Type *
make_lltype_sz4v3_from_dtype(DTYPE dtype)
{
  LL_Type *llt = make_lltype_from_dtype(dtype);
  return set_vect3_to_size4(llt);
}

OPERAND *
make_var_op(SPTR sptr)
{
  OPERAND *op;

  process_sptr(sptr);
  op = make_operand();
  op->ot_type = OT_VAR;
  op->ll_type = make_lltype_from_sptr(sptr);
  op->val.sptr = sptr;
  set_llvm_sptr_name(op);

  return op;
}

INLINE static OPERAND *
make_arg_op(SPTR sptr)
{
  OPERAND *op;
  unsigned size;
  char *base_name;
  char *buffer;

  process_sptr(sptr);
  op = make_operand();
  op->ot_type = OT_VAR;
  op->ll_type = make_lltype_from_sptr(sptr);
  op->val.sptr = sptr;
  base_name = get_llvm_name(sptr);
  size = strlen(base_name) + 6;
  buffer = (char *)llutil_alloc(size);
  snprintf(buffer, size, "%%%s.arg", base_name);
  op->string = buffer;
  return op;
}

OPERAND *
make_def_op(char *str)
{
  OPERAND *op;

  op = make_operand();
  op->ot_type = OT_DEF;
  op->string = str;

  return op;
}

static OPERAND *
make_member_op_with_lltype(int address, LL_Type *llTy)
{
  OPERAND *op = make_operand();
  op->ot_type = OT_MEMBER;
  op->ll_type = llTy;
  op->next = NULL;
  return op;
}

INLINE static OPERAND *
make_member_op(int address, DTYPE dtype)
{
  return make_member_op_with_lltype(address, make_lltype_from_dtype(dtype));
}

OPERAND *
make_tmp_op(LL_Type *llt, TMPS *tmps)
{
  OPERAND *op;

  op = make_operand();
  op->ot_type = OT_TMP;
  op->ll_type = llt;
  op->tmps = tmps;
  return op;
}

OPERAND *
make_undef_op(LL_Type *llt)
{
  OPERAND *op;

  op = make_operand();
  op->ot_type = OT_UNDEF;
  op->ll_type = llt;
  return op;
}

OPERAND *
make_null_op(LL_Type *llt)
{
  OPERAND *op;

  assert(llt->data_type == LL_PTR, "make_null_op: Need pointer type", 0, ERR_Fatal);
  op = make_operand();
  op->ot_type = OT_CONSTVAL;
  op->ll_type = llt;
  op->flags |= OPF_NULL_TYPE;

  return op;
}

/* Create a metadata operand that references a numbered metadata node. */
OPERAND *
make_mdref_op(LL_MDRef mdref)
{
  OPERAND *op;

  assert(LL_MDREF_kind(mdref) == MDRef_Node,
         "Can only reference metadata nodes", 0, ERR_Fatal);
  op = make_operand();
  op->ot_type = OT_MDNODE;
  op->tmps = make_tmps();
  op->tmps->id = LL_MDREF_value(mdref) + 1;

  return op;
}

OPERAND *
make_metadata_wrapper_op(SPTR sptr, LL_Type *llTy)
{
  OPERAND *op;

  if (sptr)
    process_sptr(sptr);
  op = make_operand();
  op->ot_type = OT_MDNODE;
  op->val.sptr = sptr;
  op->ll_type = llTy;
  return op;
}

OPERAND *
make_target_op(SPTR sptr)
{
  OPERAND *op;

  if (sptr)
    process_sptr(sptr);
  op = make_operand();
  op->ot_type = OT_TARGET;
  op->val.sptr = sptr;
  if (sptr)
    op->string = get_label_name(sptr);
  return op;
}

OPERAND *
make_label_op(SPTR sptr)
{
  OPERAND *op;

  if (sptr)
    process_sptr(sptr);
  op = make_operand();
  op->ot_type = OT_LABEL;
  op->val.sptr = sptr;
  if (sptr)
    op->string = get_label_name(sptr);
  return op;
}

OPERAND *
make_operand(void)
{
  OPERAND *op = (OPERAND *)llutil_alloc(sizeof(OPERAND));
  return op;
}

static void
set_llasm_output_file(FILE *fd)
{
  LLVMFIL = fd;
}

void
init_output_file(void)
{
  if (FTN_HAS_INIT())
    return;
  FTN_HAS_INIT() = 1;
  set_llasm_output_file(gbl.asmfil);
  ll_write_module_header(gbl.asmfil, llvm_get_current_module());
}

void
init_gpu_output_file(void)
{
  if (FTN_GPU_INIT())
    return;
  FTN_GPU_INIT() = 1;
#ifdef OMP_OFFLOAD_LLVM
  if(flg.omptarget)
    ll_write_module_header(gbl.ompaccfile, gpu_llvm_module);
#endif
}

#ifdef OMP_OFFLOAD_LLVM
void
use_gpu_output_file(void)
{
  set_llasm_output_file(gbl.ompaccfile);
}
void
use_cpu_output_file(void)
{
  set_llasm_output_file(gbl.asmfil);
}
#endif
/**
   \brief Write size of \c LL_Type to llvm file
 */
void
print_llsize(LL_Type *llt)
{
  assert(llt, "print_llsize(): missing llt", 0, ERR_Fatal);
  fprintf(LLVMFIL, "%" BIGIPFSZ "d", ll_type_bytes(llt) * 8);
}

void
print_llsize_tobuf(LL_Type *llt, char *buf)
{
  assert(llt, "print_llsize(): missing llt", 0, ERR_Fatal);
  sprintf(buf, "%" BIGIPFSZ "d", ll_type_bytes(llt) * 8);
}

/**
   \brief Write \p num spaces to llvm file
   \p num  The number of spaces to write
 */
void
print_space(int num)
{
  int i;

  for (i = 0; i < num; i++)
    fputc(' ', LLVMFIL);
}

void
print_space_tobuf(int num, char *buf)
{
  int i;

  for (i = 0; i < num; i++)
    sprintf(buf, " ");
}

/**
   \brief Write any line which does not need a tab
 */
void
print_line(const char *ln)
{
  if (ln != NULL)
    fprintf(LLVMFIL, "%s\n", ln);
  else
    fprintf(LLVMFIL, "\n");
}

/**
   \brief Print any line which does not need a tab
 */
void
print_line_tobuf(char *ln, char *buf)
{
  if (ln != NULL)
    sprintf(buf, "%s\n", ln);
  else
    sprintf(buf, "\n");
}

FILE *
llvm_file(void)
{
  return LLVMFIL;
}

/**
   \brief Write a token at the current location with no nl
 */
void
print_token(const char *tk)
{
  assert(tk, "print_token(): missing token", 0, ERR_Fatal);
  fprintf(LLVMFIL, "%s", tk);
}

/**
   \brief print a token at the current location with no nl
 */
void
print_token_tobuf(char *tk, char *buf)
{
  assert(tk, "print_token(): missing token", 0, ERR_Fatal);
  sprintf(buf, "%s", tk);
}

/**
   \brief Write a new line in the output llvm file
 */
void
print_nl(void)
{
  fprintf(LLVMFIL, "\n");
}

void
print_nl_tobuf(char *buf)
{
  sprintf(buf, "\n");
}

/**
   \brief Emit line info debug information.

   Output the string " !dbg !<i>n</i>", where <i>n</i> is a metadata ref.
 */
void
print_dbg_line_no_comma(LL_MDRef md)
{
  char buf[32];
  snprintf(buf, 32, " !dbg !%u", LL_MDREF_value(md));
  print_token(buf);
}

void
print_dbg_line(LL_MDRef md)
{
  print_token(",");
  print_dbg_line_no_comma(md);
}

/**
   \brief Compare two types to make sure something isn't already sideways

   This is for use in sanity assertions.
   FIXME: i32 and i64 types are conflated in many f90_correct tests.
 */
static bool
LLTYPE_equiv(LL_Type *ty1, LL_Type *ty2)
{
  return true;
  // FIXME - return (ty1 == ty2) || (ty1->data_type == ty2->data_type);
  return false;
}

static void
write_vconstant_value(int sptr, LL_Type *type, unsigned long long undef_bitmask)
{
  LL_Type *vtype = type->sub_types[0];
  int vsize = type->sub_elements;
  int i;
  int edtype;

  edtype = CONVAL1G(sptr);

  fputc('<', LLVMFIL);

  for (i = 0; i < vsize; i++) {
    if (i)
      fputs(", ", LLVMFIL);
    write_type(vtype);
    fputc(' ', LLVMFIL);

    if (undef_bitmask & 1) {
      print_token("undef");
      undef_bitmask >>= 1;
      continue;
    }
    undef_bitmask >>= 1;

    switch (vtype->data_type) {
    case LL_DOUBLE:
      write_constant_value(VCON_CONVAL(edtype + i), 0, 0, 0, false);
      break;
    case LL_I40:
    case LL_I48:
    case LL_I56:
    case LL_I64:
    case LL_I128:
    case LL_I256: {
      write_constant_value(VCON_CONVAL(edtype + i), 0, 0, 0, false);
      break;
    }
    /* Fall through. */
    default:
      write_constant_value(0, vtype, VCON_CONVAL(edtype + i), 0, false);
    }
  }
  fputc('>', LLVMFIL);
}

/**
   \brief Write a constant value to the output llvm file
 */
void
write_constant_value(int sptr, LL_Type *type, INT conval0, INT conval1,
                     bool uns)
{
  const char *ctype;
  INT num[2] = {0, 0};
  union xx_u xx;
  union {
    double d;
    INT tmp[2];
  } dtmp, dtmp2;
  char constant1[9], constant2[9];

  static char d[256];
  static char b[100];

  assert((sptr || type), "write_constant_value(): missing arguments", sptr, ERR_Fatal);
  if (sptr && !type)
    type = make_lltype_from_dtype(DTYPEG(sptr));

  switch (type->data_type) {
  case LL_VECTOR:
    write_vconstant_value(sptr, type, 0);
    return;

  case LL_ARRAY:

    if (sptr && DTY(DTYPEG(sptr)) == TY_CHAR) {
      int len = type->sub_elements;
      char *p;
      fprintf(LLVMFIL, "c\"");

      p = stb.n_base + CONVAL1G(sptr);
      ;
      while (len--)
        fprintf(LLVMFIL, "%c", *p++);
      fprintf(LLVMFIL, "\"");
      return;
    }

    if (conval0 == 0 && conval1 == 0) {
      fprintf(LLVMFIL, "zeroinitializer");
    } else {
      unsigned elems = type->sub_elements;

      if (sptr && DTY(DTYPEG(sptr)) == TY_NCHAR) {
        ctype = llvm_fc_type(DTYPEG(sptr));
        fprintf(LLVMFIL, "[");
      } else
        fprintf(LLVMFIL, "{");
      while (elems > 0) {
        if (sptr && DTY(DTYPEG(sptr)) == TY_NCHAR) {
          fprintf(LLVMFIL, "%s ", ctype);
        }
        write_constant_value(0, type->sub_types[0], conval0, conval1, uns);
        elems--;
        if (elems > 0)
          fprintf(LLVMFIL, ", ");
      }
      if (sptr && DTY(DTYPEG(sptr)) == TY_NCHAR) {
        fprintf(LLVMFIL, "]");
      } else
        fprintf(LLVMFIL, "}");
    }
    return;

  case LL_STRUCT:
    /* Complex data types are represented as LLVM structs. */
    if (sptr && DT_ISCMPLX(DTYPEG(sptr))) {
      if (DTY(DTYPEG(sptr)) == TY_CMPLX) {
        LL_Type *float_type = make_lltype_from_dtype(DT_FLOAT);
        ctype = llvm_fc_type(DT_FLOAT);
        fprintf(LLVMFIL, "<{ %s ", ctype);
        write_constant_value(0, float_type, CONVAL1G(sptr), 0, uns);
        fprintf(LLVMFIL, ", %s ", ctype);
        write_constant_value(0, float_type, CONVAL2G(sptr), 0, uns);
        fprintf(LLVMFIL, "}>");
      } else {
        ctype = llvm_fc_type(DTYPEG(CONVAL1G(sptr)));
        fprintf(LLVMFIL, "<{ %s ", ctype);
        write_constant_value(CONVAL1G(sptr), 0, 0, 0, uns);
        fprintf(LLVMFIL, ", %s ", ctype);
        write_constant_value(CONVAL2G(sptr), 0, 0, 0, uns);
        fprintf(LLVMFIL, "}>");
      }
    } else {
      assert(conval0 == 0 && conval1 == 0,
             "write_constant_value(): non zero struct", 0, ERR_Fatal);
      fprintf(LLVMFIL, "zeroinitializer");
    }
    return;

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
    if (sptr) {
      num[1] = CONVAL2G(sptr);
      num[0] = CONVAL1G(sptr);
    } else {
      num[1] = conval0;
      num[0] = conval1;
    }
    if (ll_type_bytes(type) <= 4) {
      fprintf(LLVMFIL, uns ? "%lu" : "%ld", (long)num[1]);
    } else {
      ui64toax(num, b, 22, uns, 10);
      fprintf(LLVMFIL, "%s", b);
    }
    return;

  case LL_DOUBLE:
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
      sprintf(d, "0x%08x%08x", num[0], num[1]);
    /* also check for -0 */
    else if (num[0] == (INT)0x80000000 && num[1] == (INT)0x00000000)
      sprintf(d, "-0.00000000e+00");
    /* remember to make room for /0 */
    fprintf(LLVMFIL, "%s", d);
    return;

  case LL_FLOAT:
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

    /* check for negative zero */
    if (dtmp.tmp[1] == (INT)0x80000000 && !dtmp.tmp[0])
      fprintf(LLVMFIL, "-0.000000e+00");
    else
      fprintf(LLVMFIL, "0x%s%s", constant1, constant2);

    break;

  case LL_X86_FP80:
    assert(sptr, "write_constant_value(): x87 constant without sptr", 0, ERR_Fatal);
    fprintf(LLVMFIL, "0xK%08x%08x%04x", CONVAL1G(sptr), CONVAL2G(sptr),
            (unsigned short)(CONVAL3G(sptr) >> 16));
    return;
#ifdef TARGET_LLVM_ARM
  case LL_FP128:
    assert(sptr, "write_constant_value(): fp128 constant without sptr", 0, ERR_Fatal);
    fprintf(LLVMFIL, "0xL%08x%08x%08x%08x", CONVAL3G(sptr), CONVAL4G(sptr),
            CONVAL1G(sptr), CONVAL2G(sptr));
    return;
#else
  case LL_FP128:
    assert(sptr, "write_constant_value(): fp128 constant without sptr", 0, ERR_Fatal);
    fprintf(LLVMFIL, "0xL%08x%08x%08x%08x", CONVAL1G(sptr), CONVAL2G(sptr),
            CONVAL3G(sptr), CONVAL4G(sptr));
    return;
#endif
  case LL_PPC_FP128:
    assert(sptr, "write_constant_value(): double-double constant without sptr",
           0, ERR_Fatal);
    fprintf(LLVMFIL, "0xM%08x%08x%08x%08x", CONVAL1G(CONVAL1G(sptr)),
            CONVAL2G(CONVAL1G(sptr)), CONVAL1G(CONVAL2G(sptr)),
            CONVAL2G(CONVAL2G(sptr)));
    return;

  case LL_PTR:
    if (sptr) {
      num[1] = CONVAL2G(sptr);
      num[0] = CONVAL1G(sptr);
    } else {
      num[1] = conval0;
      num[0] = conval1;
    }
    if (num[0] == 0 && num[1] == 0) {
      fprintf(LLVMFIL, "null");
    } else {
      ui64toax(num, b, 22, uns, 10);
      fprintf(LLVMFIL, "%s", b);
    }
    return;
  default:
    assert(false, "write_constant_value(): unexpected constant ll_type",
           type->data_type, ERR_Fatal);
  }
} /* write_constant_value */

/**
   \brief Write LL_Type to llvm file
 */
void
write_type(LL_Type *ll_type)
{
  print_token(ll_type->str);
}

INLINE static bool
metadata_args_need_struct(void)
{
  return ll_feature_metadata_args_struct(&llvm_get_current_module()->ir);
}

/**
 * This function returns true for the types supported
 * in function make_param_op
 */
bool
should_preserve_param(const DTYPE dtype)
{
  switch (DTY(dtype)) {
  // handled cases
  case TY_ARRAY:
    {
      ADSC *ad = AD_DPTR(dtype);
      SPTR size_sptr = AD_NUMELM(ad);
      ISZ_T size = ad_val_of(size_sptr);
      /* Do not preserve zero-sized array, which would be optimized out later */
      if (size == 0)
        return false;
      else
        return true;
    }
  case TY_STRUCT:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_REAL:
  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_CHAR:
    return true;
  // unsupported cases
  case TY_WORD:
  case TY_DWORD:
  case TY_HOLL:
  case TY_NCHAR:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    return false;
  default:
    assert(0, "should_preserve_param(dtype): unexpected DTYPE", 0, ERR_Fatal);
    return false;
  }
}

OPERAND *
make_param_op(SPTR sptr)
{
  OPERAND *oper = nullptr;
  DTYPE dtype = DTYPEG(sptr);

  switch (DTY(dtype)) {
  // Below are the supported types, please note that two types TY_ARRAY,
  // TY_STRUCT present in should_preserve_param but absent here that is
  // because these two type are handled differently in function process_params.
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_REAL:
    oper = make_constval_op(make_lltype_from_dtype(dtype), CONVAL1G(sptr),
                            CONVAL2G(sptr));
    break;
  case TY_INT8:
  case TY_LOG8:
    oper = make_constval_op(make_lltype_from_dtype(dtype),
                            CONVAL2G(CONVAL1G(sptr)), CONVAL1G(CONVAL1G(sptr)));
    break;
  case TY_DBLE:
    oper = make_constval_op(make_lltype_from_dtype(dtype),
                            CONVAL1G(CONVAL1G(sptr)), CONVAL2G(CONVAL1G(sptr)));
    break;
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
    oper = make_constsptr_op((SPTR)CONVAL1G(sptr));
    break;
  case TY_CHAR:
    oper = make_conststring_op((SPTR)CONVAL1G(sptr));
    break;
  // TODO: to add support for other types
  default:
    break;
  }

  return oper;
}

/**
   \brief Write a single operand
 */
void
write_operand(OPERAND *p, const char *punc_string, int flags)
{
  OPERAND *new_op;
  LL_Type *pllt;
  const bool uns = (flags & FLG_AS_UNSIGNED) != 0;
  int sptr = p->val.sptr;
  if (p->flags & OPF_CONTAINS_UNDEF) {
    sptr = p->val.sptr_undef.sptr;
  }

  DBGTRACEIN2(" operand %p (%s)", p, OTNAMEG(p))
  DBGDUMPLLTYPE(" with type ", p->ll_type)

  switch (p->ot_type) {
  case OT_MEMBER:
  case OT_NONE:
    write_type(p->ll_type);
    break;
  case OT_CONSTVAL:
    if (p->flags & OPF_NULL_TYPE) {
      if (!(flags & FLG_OMIT_OP_TYPE))
        write_type(p->ll_type);
      print_token(" null");
    } else {
      assert(p->ll_type, "write_operand(): no type when expected", 0, ERR_Fatal);
      if (!(flags & FLG_OMIT_OP_TYPE)) {
        write_type(p->ll_type);
        print_space(1);
      }

      /* write_constant_value() can't handle LL_FP128 when sptr is 0.
       * Build a sptr with getcon() before calling it. */
      if (p->ll_type->data_type == LL_FP128) {
        SPTR sptr = getcon(p->val.conval, DT_QUAD);
        write_constant_value(sptr, p->ll_type, p->val.conval[0],
                             p->val.conval[1], uns);
      } else {
        write_constant_value(0, p->ll_type, p->val.conval[0],
                             p->val.conval[1], uns);
      }
    }
    break;
  case OT_UNDEF:
    if (!(flags & FLG_OMIT_OP_TYPE)) {
      write_type(p->ll_type);
      print_space(1);
    }
    print_token("undef");
    break;
  case OT_CONSTSTRING:
    assert(p->string, "write_operand(): no string when expected", 0, ERR_Fatal);
    if (p->flags & OPF_NULL_TYPE)
      print_token("null");
    else {
      if (!(flags & FLG_OMIT_OP_TYPE)) {
        write_type(p->ll_type);
        print_space(1);
      }
      if (p->ll_type->sub_types[0]->data_type == LL_I16) {
          print_token(p->string);
      } else {
        char buffer[6];
        print_token("[");
        for (BIGUINT64 i = 0; i < p->ll_type->sub_elements; i++) {
          if (i)
            print_token(", ");
          print_token("i8 ");
          char c = p->string[i];
          sprintf(buffer, "%d", c);
          print_token(buffer);
        }
        print_token(" ] ");
      }
    }
    break;
  case OT_CONSTSPTR:
    assert(sptr, "write_operand(): no sptr when expected", 0, ERR_Fatal);
    if (p->flags & OPF_NULL_TYPE)
      print_token("null");
    else {
      LL_Type *sptrType = make_lltype_from_dtype(DTYPEG(sptr));
      assert(LLTYPE_equiv(sptrType, p->ll_type),
             "write_operand(): operand has incorrect type", sptr, ERR_Fatal);
      if (!(flags & FLG_OMIT_OP_TYPE)) {
        write_type(p->ll_type);
        print_space(1);
      }
      if (p->flags & OPF_CONTAINS_UNDEF) {
        write_vconstant_value(sptr, sptrType, p->val.sptr_undef.undef_mask);
      } else {
        write_constant_value(sptr, sptrType, 0, 0, uns);
      }
    }
    break;
  case OT_TARGET:
    assert(sptr, "write_operand(): no sptr when expected", 0, ERR_Fatal);
    print_token("label %L");
    print_token(p->string);
    break;
  case OT_VAR:
    assert(sptr, "write_operand(): no sptr when expected", 0, ERR_Fatal);
    pllt = p->ll_type;
    if (pllt->data_type == LL_FUNCTION)
      pllt = make_ptr_lltype(pllt);
#if defined(TARGET_LLVM_X8664)
    if ((flags & FLG_FIXUP_RETURN_TYPE) && (pllt->data_type == LL_PTR))
      pllt = maybe_fixup_x86_abi_return(pllt);
#endif
    if (!(flags & FLG_OMIT_OP_TYPE))
      write_type(pllt);
    if (p->flags & OPF_SRET_TYPE) {
      print_token(" sret(");
      print_token(p->ll_type->sub_types[0]->str);
      print_token(")");
    }
    if (p->flags & OPF_SRARG_TYPE) {
      print_token(" byval(");
      print_token(p->ll_type->sub_types[0]->str);
      print_token(")");
    }
    print_space(1);
    print_token(p->string);
    break;
  case OT_DEF:
  case OT_CALL: /* currently just used for llvm intrinsics */
    print_token(p->string);
    break;
  case OT_LABEL:
    print_token("L");
    print_token(p->string);
    print_token(":");
    break;
  case OT_TMP:
    if (!(flags & FLG_OMIT_OP_TYPE)) {
      assert(p->ll_type, "write_operand(): missing type information", 0, ERR_Fatal);
      write_type(p->ll_type);
    }
    if (p->flags & OPF_SRET_TYPE) {
      print_token(" sret(");
      print_token(p->ll_type->sub_types[0]->str);
      print_token(")");
    }
    if (p->flags & OPF_SRARG_TYPE) {
      print_token(" byval(");
      print_token(p->ll_type->sub_types[0]->str);
      print_token(")");
    }
    print_space(1);
    if (p->tmps)
      print_tmp_name(p->tmps);
    else
      assert(0, "write_operand(): missing temporary value", 0, ERR_Fatal);
    break;
  case OT_CC:
    assert(p->val.cc, "write_operand(): expecting condition code", 0, ERR_Fatal);
    assert(p->ll_type, "write_operand(): missing type", 0, ERR_Fatal);
    if (ll_type_int_bits(p->ll_type) || p->ll_type->data_type == LL_PTR)
      print_token(llvm_cc_names[p->val.cc]);
    else if (ll_type_is_fp(p->ll_type))
      print_token(llvm_ccfp_names[p->val.cc]);
    else if (p->ll_type->data_type == LL_VECTOR) {
      LL_Type *ty;
      assert(p->ll_type->data_type == LL_VECTOR, "expected vector",
             p->ll_type->data_type, ERR_Fatal);
      ty = p->ll_type->sub_types[0];
      if (ll_type_is_fp(ty)) {
        print_token(llvm_ccfp_names[p->val.cc]);
      } else if (ll_type_int_bits(ty)) {
        print_token(llvm_cc_names[p->val.cc]);
      } else {
        assert(0, "unexpected type", ty->data_type, ERR_Fatal);
      }
    } else {
#if DEBUG
      assert(0, "write_operand(): bad LL type", p->ll_type->data_type, ERR_Fatal);
#endif
    }
    break;
  case OT_MDNODE:
    if (p->tmps) {
      if (p->flags & OPF_WRAPPED_MD) {
        print_token("metadata ");
        print_token(p->ll_type->str);
        print_space(1);
        if (p->tmps->id)
          print_tmp_name(p->tmps);
        else
          print_token("undef");
      } else {
        if (!(flags & FLG_OMIT_OP_TYPE))
          print_token("metadata ");
        print_metadata_name(p->tmps);
      }
    } else if (p->val.sptr) {
      if (!(flags & FLG_OMIT_OP_TYPE))
        print_token("metadata ");
      if (metadata_args_need_struct())
        print_token("!{");
      if (p->flags & OPF_HIDDEN) {
        new_op = make_arg_op(p->val.sptr);
        if (p->ll_type)
          new_op->ll_type = p->ll_type;
      } else if (STYPEG(p->val.sptr) == ST_PARAM) {
        new_op = make_param_op(p->val.sptr);
      } else {
        new_op = make_var_op(p->val.sptr);
        if (p->ll_type)
          new_op->ll_type = ll_get_pointer_type(p->ll_type);
      }

      new_op->flags = p->flags;
      write_operand(new_op, "", 0);
      if (metadata_args_need_struct())
        print_token("}");
    } else {
      print_token("null");
    }
    break;
  default:
    DBGTRACE1("### write_operand(): unknown operand type: %s",
              ot_names[p->ot_type])
    assert(0, "write_operand(): unknown operand type", p->ot_type, ERR_Fatal);
  }
  /* check for commas and closing paren */
  if (punc_string != NULL)
    print_token(punc_string);
  DBGTRACEOUT("")
}

/**
   \brief Write operand list
   \param operand  The head of the list
   \param flags

   Write out the operands in order. Not always possible, depends on instruction
   format. Assumes the separator is a comma.
 */
void
write_operands(OPERAND *operand, int flags)
{
  OPERAND *p;

  DBGTRACEIN1(" starting at operand %p", operand)

  /* write out the operands to the instructions */
  for (p = operand; p; p = p->next)
    write_operand(p, (p->next) ? ", " : "", flags);

  DBGTRACEOUT("")
}

static int metadata_id = 0;

/**
   \brief Set name for named metadata
 */
void
set_metadata_string(TMPS *t, char *string)
{
  DBGTRACEIN2(" TMPS* %p, string %s", t, string)

  t->id = -1;
  t->info.string = string;

  DBGTRACEOUT("")
}

/**
   \brief Init metadata index, for anonymous metadata
 */
void
init_metadata_index(TMPS *t)
{
  DBGTRACEIN1(" TMPS* %p", t)

  if (!t->id)
    t->id = ++metadata_id;

  DBGTRACEOUT1(" %d", t->id)
}

/**
   \brief Print metadata name
 */
void
print_metadata_name(TMPS *t)
{
  char tmp[50];

  DBGTRACEIN1(" TMPS* %p", t)

  if (!t->id)
    t->id = ++metadata_id;
  if (t->id < 0) {
    print_token(t->info.string);
  } else {
    sprintf(tmp, "!%d", t->id - 1);
    print_token(tmp);
  }
  DBGTRACEOUT("")
} /* print_metadata_name */

#if DEBUG
static int indentlev = 0;
FILE *ll_dfile;

void
indent(int change)
{
  int i;

  if (change < 0)
    indentlev += change;
  for (i = 1; i <= indentlev; i++)
    fprintf(ll_dfile, "  ");
  if (change > 0)
    indentlev += change;
}
#endif

bool
small_aggr_return(DTYPE dtype)
{
#if   defined(TARGET_LLVM_X8664)
  /* TO DO : to be revisited when needed */
  return false;
#else
  return false;
#endif
  return false;
}

DTYPE
get_return_dtype(DTYPE dtype, unsigned *flags, unsigned new_flag)
{
#ifdef TARGET_LLVM_ARM
  if (!small_aggr_return(dtype)) {
    if (is_struct_kind(dtype, !XBIT(121, 0x400000), true)) {
      if (flags)
        *flags |= new_flag;
      return DT_VOID_NONE;
    }
  } else {
    switch (ZSIZEOF(dtype)) {
    case 1:
      return DT_SBYTE;
    case 2:
      return DT_SINT;
    case 3:
    case 4:
      return DT_INT;
    default:
      assert(0, "get_return_dtype(): bad return dtype size for small struct",
             ZSIZEOF(dtype), ERR_Fatal);
    }
  }
#else  /* !TARGET_LLVM_ARM */
  if (is_struct_kind(dtype, !XBIT(121, 0x400000), true)) {
    if (flags)
      *flags |= new_flag;
    return DT_VOID_NONE;
  }
#endif /* TARGET_LLVM_ARM */
  if (DT_ISCMPLX(dtype))
    return DT_NONE;
  if (XBIT(121, 0x400000) && DTY(dtype) == TY_CMPLX)
    return DT_INT8;
  return dtype;
}

DTYPE
get_param_equiv_dtype(DTYPE dtype)
{
#ifdef TARGET_LLVM_ARM
  if (DTY(dtype) == TY_VECT) {
    switch (ZSIZEOF(dtype)) {
    case 1:
      return DT_BINT;
    case 2:
      return DT_SINT;
    case 3:
    case 4:
      return DT_INT;
    }
  }
#endif
  return dtype;
}

/**
   \brief return string for a first class type
 */
const char *
llvm_fc_type(DTYPE dtype)
{
  const char *retc = "";
  ISZ_T sz;

  switch (DTY(dtype)) {
  case TY_NONE:
    retc = "void"; /* TODO need to check where it is be used */
    break;
  case TY_INT:
  case TY_UINT:
  case TY_LOG:
  case TY_DWORD:
    sz = size_of(dtype);
    if (sz == 4)
      retc = "i32";
    else if (sz == 8)
      retc = "i64";
    else
      assert(0, "llvm_fc_type(): incompatible size", sz, ERR_Fatal);
    break;

  case TY_CHAR:
    retc = "i8";
    break;
  case TY_NCHAR:
    retc = "i16";
    break;
  case TY_BINT:
  case TY_BLOG:
    retc = "i8";
    break;
  case TY_SINT:
  case TY_SLOG:
  case TY_WORD:
    retc = "i16";
    /* retc = "i16 signext"; */
    break;
  case TY_USINT:
    retc = "i16";
    /* retc = "i16 zeroext"; */
    break;
  case TY_FLOAT:
    retc = "float";
    break;
  case TY_DBLE:
    retc = "double";
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_FLOAT128:
  case TY_128:
    retc = "fp128";
    break;
  case TY_CMPLX128:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    retc = "{fp128, fp128}";
    break;
  case TY_INT8:
  case TY_UINT8:
  case TY_LOG8:
    retc = "i64";
    break;
  case TY_LOG128:
  case TY_INT128:
    retc = "i128";
    break;
  case TY_DCMPLX:
    retc = "{double, double}";
    break;
  case TY_CMPLX:
    retc = "{float, float}";
    break;
  case -TY_UNION:
    retc = "union";
    break;
  case -TY_STRUCT:
    retc = "struct";
    break;
  default:
    DBGTRACE2("###llvm_fc_type(): unhandled data type: %ld (%s), might not be "
              "first class ?",
              DTY(dtype), (stb.tynames[DTY(dtype)]))
    assert(0, "llvm_fc_type: unhandled data type", DTY(dtype), ERR_Fatal);
    break;
  }
  return retc;
} /* llvm_fc_type */

OPERAND *
gen_copy_op(OPERAND *op)
{
  OPERAND *copy_operand;

  copy_operand = make_operand();
  memmove(copy_operand, op, sizeof(OPERAND));
  copy_operand->next = NULL;
  return copy_operand;
}

OPERAND *
gen_copy_list_op(OPERAND *operands)
{
  OPERAND *list_op = NULL, *prev_op = NULL;

  if (operands) {
    list_op = gen_copy_op(operands);
    prev_op = list_op;
    operands = operands->next;
  }
  while (operands) {
    prev_op->next = gen_copy_op(operands);
    prev_op = prev_op->next;
    operands = operands->next;
  }
  return list_op;
}

static LLDEF *
make_def(DTYPE dtype, int sptr, int rank, const char *name, int flags)
{
  LLDEF *new_def;

  new_def = (LLDEF *)llutil_alloc(sizeof(LLDEF));
  new_def->dtype = dtype;
  new_def->ll_type = NULL;
  new_def->sptr = sptr;
  new_def->rank = rank;
  new_def->flags = flags;
  new_def->printed = 0;
  new_def->name = name;
  new_def->addrspace = 0;
  new_def->values = NULL;
  new_def->next = NULL;
  return new_def;
}

static LLDEF *
get_def(DTYPE dtype, int sptr, int rank, LLDEF *def_list)
{
  LLDEF *p_def;

  p_def = def_list;
  while (p_def != NULL) {
    if (p_def->dtype == dtype && p_def->sptr == sptr && p_def->rank == rank)
      break;
    p_def = p_def->next;
  }
  return p_def;
}

#ifdef TARGET_LLVM_ARM
void
write_alt_struct_def(LLDEF *def)
{
  char buf[80];
  DTYPE dtype = def->dtype;
  int struct_sz, field_sz;

  print_token(def->name);
  print_token(".alt = type ");
  if (ZSIZEOF(def->dtype) == 0) {
    print_token("opaque");
    print_nl();
    return;
  }
  print_token("< { ");
  struct_sz = ZSIZEOF(dtype);
  if (DTyAlgTyAlign(dtype) > 3)
    field_sz = 8;
  else
    field_sz = 4;
  while (field_sz && struct_sz) {
    int field_count = struct_sz / field_sz;
    struct_sz = struct_sz & (field_sz - 1);
    if (field_count > 0) {
      sprintf(buf, "[%d x i%d]", field_count, field_sz * 8);
      print_token(buf);
    }
    field_sz >>= 1;
    if (field_count && struct_sz)
      print_token(", ");
  }
  print_token(" } >");
  print_nl();
}
#endif

/*
 * Write out an initializer of the given type, consuming as many operands from
 * the def_op chain as required.
 *
 * Return the first unused def_op operand.
 */
static OPERAND *
write_def_values(OPERAND *def_op, LL_Type *type)
{
  if (def_op == NULL) {
    print_token(type->str);
    print_token(" undef");
    return NULL;
  }

  switch (type->data_type) {
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
  case LL_HALF:
  case LL_FLOAT:
  case LL_DOUBLE:
  case LL_FP128:
  case LL_X86_FP80:
  case LL_PPC_FP128:
  case LL_PTR:
    print_token(type->str);
    print_token(" ");
    write_operand(def_op, "", FLG_OMIT_OP_TYPE);
    return def_op->next;

  case LL_ARRAY:
    print_token(type->str);
    if (def_op->ot_type == OT_CONSTSTRING && type->data_type == LL_ARRAY &&
        (type->sub_types[0]->data_type == LL_I8 ||
         type->sub_types[0]->data_type == LL_I16)) {
      print_token(" ");
      write_operand(def_op, "", FLG_OMIT_OP_TYPE);
      def_op = def_op->next;
      return def_op;
    } else if (def_op->ot_type == OT_CONSTVAL &&
               type->data_type == LL_ARRAY &&
               def_op->ll_type->data_type == LL_ARRAY) {
      /* We are initializing an array with a constant value that is also array type.
         This means that every array element needs to get same value. */
      if (def_op->val.conval[0] == 0 && def_op->val.conval[1] == 0 &&
          def_op->val.conval[2] == 0 && def_op->val.conval[3] == 0) {
        /* If value is zero, use zeroinitializer to improve readability */
        print_token(" zeroinitializer ");
        def_op = def_op->next;
      } else {
        OPERAND *new_def_op = def_op;
        print_token(" [ ");
        for (BIGUINT64 i = 0; i < type->sub_elements; i++) {
          if (i)
            print_token(", ");
          /* The idea here is that we reuse the same def_op for each array member.
             The new_def_op is supposed to be the next value and thus we only
             make use of that once we are done processing each array member. */
          new_def_op = write_def_values(def_op, type->sub_types[0]);
        }
        print_token(" ] ");
        def_op = new_def_op;
      }
    } else {
      print_token(" [ ");
      for (BIGUINT64 i = 0; i < type->sub_elements; i++) {
        if (i)
          print_token(", ");
        def_op = write_def_values(def_op, type->sub_types[0]);
      }
      print_token(" ] ");
    }

    return def_op;

  case LL_VECTOR:
    print_token(type->str);
    print_token(" < ");
    for (BIGUINT64 i = 0; i < type->sub_elements; i++) {
      if (i)
        print_token(", ");
      assert(def_op, "write_def_values(): missing def for type", 0, ERR_Fatal);
      def_op = write_def_values(def_op, type->sub_types[0]);
    }
    print_token(" > ");
    return def_op;

  case LL_STRUCT:
    print_token(type->str);
    if (type->flags & LL_TYPE_IS_PACKED_STRUCT)
      print_token(" <{ ");
    else
      print_token(" { ");
    for (BIGUINT64 i = 0; i < type->sub_elements; i++) {
      if (i)
        print_token(", ");
      def_op = write_def_values(def_op, type->sub_types[i]);
    }
    if (type->flags & LL_TYPE_IS_PACKED_STRUCT)
      print_token(" }>");
    else
      print_token(" }");
    return def_op;

  default:
    interr("write_def_values(): unknown datatype", type->data_type, ERR_Fatal);
  }
  return NULL;
}

static void
write_alt_field_types(LL_Type *llty)
{
  if (llty->sub_elements > 0) {
    int i;
    int I = llty->sub_elements - 1;

    for (i = 0; i < I; ++i) {
      print_token(llty->sub_types[i]->str);
      print_token(", ");
    }
    print_token(llty->sub_types[I]->str);
  }
}

static void
write_def(LLDEF *def, int check_type_in_struct_def_type)
{
  LLDEF *lltypedef = NULL;

  print_token(def->name);
  print_token(" = ");
  if (check_type_in_struct_def_type && def->dtype) {
    lltypedef = get_def(def->dtype, 0, 0, struct_def_list);
  }
  if (def->flags & LLDEF_IS_TYPE) {
    print_token("type ");
    if (def->flags & LLDEF_IS_EMPTY) {
      print_token("<{ }>");
      print_nl();
      return;
    }
    if(def->flags & LLDEF_IS_UNPACKED_STRUCT)
      print_token("{ ");
    else
      print_token("<{ ");
    write_alt_field_types(def->ll_type);
    if(def->flags & LLDEF_IS_UNPACKED_STRUCT)
      print_token("} ");
    else
      print_token("}> ");
  } else {
    char buf[50];
    if (def->flags & LLDEF_IS_EXTERNAL)
      sprintf(buf, "external addrspace(%d) global ", def->addrspace);
    else if ((def->flags & LLDEF_IS_INITIALIZED) && (def->values != NULL) &&
             (def->flags & LLDEF_IS_ACCSTRING))
      sprintf(buf, "private addrspace(%d) constant ", def->addrspace);
    else if (def->flags & LLDEF_IS_STATIC)
      sprintf(buf, "internal addrspace(%d) global ", def->addrspace);
    else if ((def->flags & LLDEF_IS_INITIALIZED) && (def->values != NULL))
      sprintf(buf, "addrspace(%d) global ", def->addrspace);
    else if (def->flags & LLDEF_IS_CONST)
      sprintf(buf, "addrspace(%d) global ", def->addrspace);
    else
      sprintf(buf, "common addrspace(%d) global ", def->addrspace);

    print_token(buf);

    if ((def->flags & (LLDEF_IS_INITIALIZED | LLDEF_IS_EXTERNAL)) ==
        LLDEF_IS_INITIALIZED) {
      if (def->values != NULL) {
        write_def_values(def->values, def->ll_type);
      } else {
        write_type(def->ll_type);
        print_token(" zeroinitializer");
      }
    } else {
      if (lltypedef)
        print_token(lltypedef->name);
      else if (def->ll_type)
        write_type(def->ll_type);
      else
        write_type(make_lltype_from_dtype(def->dtype));
      if (def->flags & LLDEF_IS_STATIC)
        print_token(" zeroinitializer");
    }
    print_token(", align 16");
  }

  print_nl();
#ifdef TARGET_LLVM_ARM
  if (def->flags & LLDEF_IS_TYPE)
    write_alt_struct_def(def);
#endif
}

static void
write_defs(LLDEF *def_list, int check_type_in_struct_def_type)
{
  LLDEF *cur_def;

  cur_def = def_list;
  print_nl();
  while (cur_def) {
    if (!cur_def->printed) {
      write_def(cur_def, check_type_in_struct_def_type);
      cur_def->printed = 1;
    }
    cur_def = cur_def->next;
  }
  print_nl();
}

/* Check whethere there are any definitons to write
 * @param def_list -- definition list
 * @return true if there is any entry with printed==0, false if all are printed
 * or the list is empty
 */
static bool
defs_to_write(LLDEF *def_list)
{
  LLDEF *cur_def;
  if (!def_list)
    return false;

  cur_def = def_list;
  while (cur_def) {
    if (!cur_def->printed) {
      return true;
    }
    cur_def = cur_def->next;
  }
  return false;
}

/* Write structure definitions to the output LLVM file */
void
write_struct_defs(void)
{
  write_defs(struct_def_list, 0);
  /* Keep on processing list of structure defs until it stops changing
   */
  while (defs_to_write(struct_def_list)) {
    write_defs(struct_def_list, 0);
  }
}

void
write_ftn_typedefs(void)
{
  LLDEF *cur_def;
  int gblsym;

  cur_def = struct_def_list;
  while (cur_def) {
    if (!cur_def->printed && cur_def->name && cur_def->dtype) {
      gblsym = get_typedef_ag(cur_def->name,
                              process_dtype_struct(cur_def->dtype));
      if (gblsym == 0) {
        write_def(cur_def, 0);
      }
      cur_def->printed = 1;
    }
    cur_def = cur_def->next;
  }
}

DTYPE
get_int_dtype_from_size(int size)
{
  switch (size) {
  case 1:
    return DT_BINT;
    break;
  case 2:
    return DT_SINT;
  case 4:
    return DT_INT;
  case 8:
    return DT_INT8;
  }
  return DT_NONE;
}

#if DEBUG
int
struct_typedef_name(DTYPE dtype)
{
  int sptr;

  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_TYPEDEF && DTYPEG(sptr) == dtype)
      return sptr;
  }
  return 0;
} /* struct_typedef_name */
#endif

#ifdef FLANG2_LLUTIL_UNUSED
static char *
def_name(DTYPE dtype, int tag)
{
  char *tag_name;
  char *d_name;
  char buf[200];
  char idbuf[MAXIDLEN];
  static int count = 0;
  int tag_len = 0;

  if (tag) {
    tag_name = getprint(tag);
  } else {
    tag = struct_typedef_name(dtype);
    if (tag) {
      tag_name = getprint(tag);
    } else {
      sprintf(buf, "_anon%d", count++);
      tag_name = buf;
    }
  }
  if (tag) {
    sprintf(idbuf, "%d_%d", dtype, tag);
    tag_len = strlen(idbuf) + 1;
  }
  tag_len += strlen(tag_name) + 10;
  d_name = (char *)llutil_alloc(tag_len * sizeof(char));
  if (tag)
    sprintf(d_name, "%%struct.%s.%s", tag_name, idbuf);
  else
    sprintf(d_name, "%%struct.%s", tag_name);
  return d_name;
}
#endif

OPERAND *
process_symlinked_sptr(int sptr, int total_init_sz, int is_union,
                       int max_field_sz)
{
  OPERAND *cur_op;
  OPERAND head;
  int pad, field_sz, sptr_sz, max_sz, update_union;
  int cur_addr, prev_addr;
  OPERAND *union_from = NULL, *union_to = NULL;

  if (sptr > NOSYM)
    prev_addr = ADDRESSG(sptr);
  field_sz = 0;
  max_sz = 0;
  update_union = 0;
  pad = 0;
  head.next = 0;
  cur_op = &head;
  while (sptr > NOSYM) {
    if (POINTERG(sptr)) {
      sptr = SYMLKG(sptr);
      continue;
    }
    cur_addr = ADDRESSG(sptr);
    if (cur_addr > prev_addr) {
      while (prev_addr < cur_addr) {
        cur_op->next = make_member_op(prev_addr, get_int_dtype_from_size(1));
        cur_op = cur_op->next;
        prev_addr++;
        pad++;
      }
    }
    {
      if (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
          DDTG(DTYPEG(sptr)) == DT_DEFERCHAR)
        sptr_sz = ZSIZEOF(DT_ADDR);
      else
        sptr_sz = ZSIZEOF(DTYPEG(sptr));
      pad += sptr_sz;
      cur_op->next = make_member_op(prev_addr, DTYPEG(sptr));
      if (sptr_sz > max_sz) {
        max_sz = sptr_sz;
        union_from = union_to = cur_op->next;
      }
      cur_op = cur_op->next;
      if (DDTG(DTYPEG(sptr)) == DT_ASSCHAR ||
          DDTG(DTYPEG(sptr)) == DT_DEFERCHAR)
        prev_addr = cur_addr + ZSIZEOF(DT_ADDR);
      else
        prev_addr = cur_addr + ZSIZEOF(DTYPEG(sptr));
      sptr = SYMLKG(sptr);
    }
  }
  if (is_union && max_sz) {
    cur_op = union_to;
    union_to->next = NULL;
    head.next = union_from;
    pad = total_init_sz - max_sz;
  } else {
    pad = total_init_sz - pad;
  }
  if (pad > 8) {
    LL_Type *i8 = ll_create_int_type(llvm_get_current_module(), 8);
    LL_Type *arr = ll_get_array_type(i8, pad, 0);
    cur_op->next = make_member_op_with_lltype(prev_addr, arr);
  } else {
    while (pad > 0) {
      cur_op->next = make_member_op(prev_addr, get_int_dtype_from_size(1));
      cur_op = cur_op->next;
      prev_addr++;
      pad--;
    }
  }
  return head.next;
}

const char *
process_dtype_struct(DTYPE dtype)
{
  const char *d_name;
  SPTR tag;
  TY_KIND dty;
  LLDEF *def;
#ifdef OMP_OFFLOAD_LLVM
  //bool is_omptarget_type = (bool)OMPACCSTRUCTG(DTY((DTYPE)(dtype + 3)));
  bool is_omptarget_type = DTyArgNext(dtype);
#endif
  dty = DTY(dtype);
  def = get_def(dtype, 0, 0, struct_def_list);
  if (dty != TY_UNION && dty != TY_STRUCT && def == NULL)
    return NULL;
  tag = DTyAlgTyTag(dtype);

  DBGTRACEIN1(" called with dtype %d\n", dtype)

  /* if already computed, just return */
  if (def != NULL) {
    DBGTRACEOUT1(" returns %s", def->name)
    return def->name;
  }
  /* Use consistent struct type names. */
  d_name = ll_convert_struct_dtype(llvm_get_current_module(), dtype)->str;
  if (ZSIZEOF(dtype) == 0 && DTyAlgTyMember(dtype) == 0)
    def = make_def(dtype, 0, 0, d_name,
                   LLDEF_IS_TYPE | LLDEF_IS_EMPTY | LLDEF_IS_STRUCT);
#ifdef OMP_OFFLOAD_LLVM
  else if(is_omptarget_type)
    def = make_def(dtype, 0, 0, d_name, LLDEF_IS_TYPE | LLDEF_IS_UNPACKED_STRUCT);
#endif
  else
    def = make_def(dtype, 0, 0, d_name, LLDEF_IS_TYPE | LLDEF_IS_STRUCT);
  add_def(def, &struct_def_list);
  /* if empty (extended) type - don't call process_symlinked_sptr -> oop508 */
  if (is_empty_typedef(dtype))
    def->values = 0;
  def->values = process_symlinked_sptr(
      DTyAlgTyMember(dtype), ZSIZEOF(dtype), (dty == TY_UNION),
                             (DTyAlgTyAlign(dtype) + 1) * 8);
  DBGTRACEOUT1(" returns %s", def->name);

  return def->name;
}

/**
   \brief Make a fake struct for static/common block

   This differs from process_dtype_struct and that it overrides the unique name
   generated by ll_convert_struct_dtype().

   The printed flag tells write_ftn_typedefs that this type has already been
   printed out to the .ll output file.  If true, write_ftn_typedefs() will not
   print the type out (assuming that it has already been 'printed').
 */
const char *
process_ftn_dtype_struct(DTYPE dtype, char *tname, bool printed)
{
  int tag;
  TY_KIND dty;
  char *d_name;
  LLDEF *def;

  dty = DTY(dtype);
  def = get_def(dtype, 0, 0, struct_def_list);
  if (dty != TY_UNION && dty != TY_STRUCT && def == NULL)
    return NULL;
  tag = DTyAlgTyTag(dtype);

  DBGTRACEIN1(" called with dtype %d\n", dtype)

  d_name = (char *)llutil_alloc(strlen(tname) + 2);
  sprintf(d_name, "%%%s", tname);

  /* if already computed, just return */
  if (def != NULL) {
    DBGTRACEOUT1(" returns %s", def->name)
    return def->name;
  }

  if (ZSIZEOF(dtype) == 0)
    def = make_def(dtype, 0, 0, d_name,
                   LLDEF_IS_TYPE | LLDEF_IS_EMPTY | LLDEF_IS_STRUCT);
  else
    def = make_def(dtype, 0, 0, d_name, LLDEF_IS_TYPE | LLDEF_IS_STRUCT);
  add_def(def, &struct_def_list);
  def->values = process_symlinked_sptr(
      DTyAlgTyMember(dtype), ZSIZEOF(dtype), (dty == TY_UNION),
                             (DTyAlgTyAlign(dtype) + 1) * 8);
  def->printed = printed;
  ll_override_type_string(def->ll_type, d_name);
  DBGTRACEOUT1(" returns %s", def->name)
  return def->name;
}

#ifdef FLANG2_LLUTIL_UNUSED
static OPERAND *
add_init_zero_const_op(int sptr, OPERAND *cur_op, ISZ_T *offset,
                       ISZ_T *lastoffset)
{
  DTYPE dtype;
  ISZ_T address;

  dtype = DTYPEG(sptr);
  address = ADDRESSG(sptr);
  cur_op->next = make_constval_op(make_lltype_from_dtype(dtype), 0, 0);
  if (lastoffset)
    *lastoffset = address + ZSIZEOF(dtype);
  *offset = address;
  return cur_op->next;
}

static OPERAND *
add_init_const_op(DTYPE dtype, OPERAND *cur_op, ISZ_T conval, ISZ_T *repeat_cnt,
                  ISZ_T *offset)
{
  ISZ_T address;
  const SPTR convalSptr = (SPTR)conval;

  address = *offset;
  switch (dtype) {
  case 0:
    /* alignment record? */
    interr("cf_data_init: unexpected alignment", 0, ERR_Fatal);
    break;
  case DINIT_ZEROES:
    /* output zeroes */
    interr("cf_data_init: unexpected zeroes", 0, ERR_Fatal);
    break;
  case DINIT_LABEL:
    /* initialize to address */
    cur_op->next = make_var_op(convalSptr);
    cur_op = cur_op->next;
    address += size_of(DT_CPTR);
    break;
#ifdef DINIT_OFFSET
  case DINIT_OFFSET:
    interr("cf_data_init: unexpected offset", 0, ERR_Fatal);
    break;
#endif
#ifdef DINIT_REPEAT
  case DINIT_REPEAT:
    *repeat_cnt = conval;
    break;
#endif
#ifdef DINIT_STRING
  case DINIT_STRING:
    interr("cf_data_init: unexpected string", 0, ERR_Fatal);
    break;
#endif
  default:
    if (!DTyValidRange(dtype))
      interr("cf_data_init: unknown datatype", dtype, ERR_Fatal);
    do {
      switch (DTY(dtype)) {
      case TY_INT8:
      case TY_LOG8:
        cur_op->next = make_constval_op(make_lltype_from_dtype(dtype),
                                        CONVAL2G(conval), CONVAL1G(conval));
        cur_op = cur_op->next;
        address += 8;
        break;
      case TY_INT:
      case TY_UINT:
      case TY_LOG:
      case TY_SINT:
      case TY_SLOG:
      case TY_BINT:
      case TY_BLOG:
      case TY_FLOAT:
        cur_op->next =
            make_constval_op(make_lltype_from_dtype(dtype), conval, 0);
        cur_op = cur_op->next;
        address += size_of(dtype);
        break;
      case TY_128:
        break;
      case TY_DBLE:
        cur_op->next = make_constval_op(make_lltype_from_dtype(dtype),
                                        CONVAL1G(conval), CONVAL2G(conval));
        cur_op = cur_op->next;
        address += 8;
        break;
      case TY_CMPLX:
        cur_op->next = make_constval_op(make_lltype_from_dtype(DT_FLOAT),
                                        CONVAL1G(conval), 0);
        cur_op->next->next = make_constval_op(make_lltype_from_dtype(DT_FLOAT),
                                              CONVAL2G(conval), 0);
        cur_op = cur_op->next->next;
        address += 8;
        break;
#ifdef LONG_DOUBLE_FLOAT128
      case TY_FLOAT128:
        cur_op->next->next = make_constval_opL(
            make_lltype_from_dtype(DT_FLOAT128), CONVAL1G(conval),
            CONVAL2G(conval), CONVAL3G(conval), CONVAL4G(conval));
        cur_op = cur_op->next->next;
        address += 16;
        break;
#endif
      case TY_DCMPLX:
        cur_op->next = make_constval_op(make_lltype_from_dtype(DT_DBLE),
                                        CONVAL2G(CONVAL1G(conval)),
                                        CONVAL1G(CONVAL1G(conval)));
        cur_op->next->next = make_constval_op(make_lltype_from_dtype(DT_DBLE),
                                              CONVAL2G(CONVAL2G(conval)),
                                              CONVAL1G(CONVAL2G(conval)));
        cur_op = cur_op->next->next;
        address += 16;
        break;
      case TY_CHAR:
        address += DTyCharLength(DTYPEG(conval));
        if (STYPEG(conval) == ST_CONST)
          cur_op->next = make_conststring_op(conval);
        else
          cur_op->next = make_constsptr_op(convalSptr);
        cur_op = cur_op->next;
        break;
      case TY_NCHAR:
        address += DTyCharLength(DTYPEG(conval));
        if (STYPEG(conval) == ST_CONST)
          cur_op->next = make_conststring_op(conval);
        else
          cur_op->next = make_constsptr_op(convalSptr);
        cur_op = cur_op->next;
        break;
      case TY_PTR:
        /* almost always a null pointer */
        if (DT_ISINT(DTYPEG(conval))) {
          cur_op->next = make_constval_op(make_lltype_from_dtype(dtype),
                                        CONVAL2G(conval), CONVAL1G(conval));
          cur_op = cur_op->next;
          address += size_of(dtype);
        } else {
          interr("process_acc_put_dinit: unexpected datatype", dtype, ERR_Fatal);
        }
        break;
      default:
        interr("process_acc_put_dinit: unexpected datatype", dtype, ERR_Fatal);
        break;
      }
    } while (--*repeat_cnt);
    *repeat_cnt = 1;
    break;
  }
  *offset = address;
  return cur_op;
}

static OPERAND *
add_init_pad(OPERAND *cur_op, ISZ_T sz)
{
  while (sz > 0) {
    cur_op->next = make_constval_op(
        make_lltype_from_dtype(get_int_dtype_from_size(1)), 0, 0);
    cur_op = cur_op->next;
    sz--;
  }
  return cur_op;
}

static OPERAND *
add_init_subzero_consts(DTYPE dtype, OPERAND *cur_op, ISZ_T *offset,
                        ISZ_T lastoffset)
{
  ISZ_T sz;
  DTYPE ddtype;
  int mem;
  DTYPE memdtype;
  ISZ_T address;
  LL_Type* llddtype;

  address = *offset;
  switch (DTY(dtype)) {
  case TY_ARRAY:
    sz = ZSIZEOF(dtype);
    if (lastoffset - address >= sz) {
      cur_op->next = make_constval_op(make_lltype_from_dtype(dtype), 0, 0);
      *offset = address + sz;
      return cur_op->next;
    }
    /* only part of the array */
    ddtype = DTySeqTyElement(dtype);
    sz = size_of(ddtype);
    if (lastoffset - address < sz) {
      /* Less than size of one element, we are partially initializing an element
       * of array of struct */
      return add_init_subzero_consts(ddtype, cur_op, offset, lastoffset);
    }
    while (address < lastoffset) {
      cur_op->next = make_constval_op(make_lltype_from_dtype(ddtype), 0, 0);
      cur_op = cur_op->next;
      address += sz;
    }
    *offset = address;
    return cur_op;
  case TY_CHAR:
    sz = DTyCharLength(dtype);
    llddtype = make_lltype_from_dtype(DT_BINT);
    while (address < lastoffset) {
      cur_op->next = make_constval_op(llddtype, 0, 0);
      cur_op = cur_op->next;
      address += 1;
    }
    *offset = address;
    return cur_op;
  case TY_STRUCT:
    mem = DTyAlgTyMember(dtype);
    while (ADDRESSG(mem) < address && mem > NOSYM)
      mem = SYMLKG(mem);
    if (mem > NOSYM) {
      if (address > ADDRESSG(mem)) {
        memdtype = DTYPEG(mem);
        sz = size_of(memdtype);
        address = 0;
        cur_op = add_init_subzero_consts(DTYPEG(mem), cur_op, &address,
                                         lastoffset - ADDRESSG(mem));
        if (address == lastoffset) {
          *offset = address;
          return cur_op;
        }
        if (address >= ADDRESSG(mem) + sz)
          mem = SYMLKG(mem);
      } else if (address < ADDRESSG(mem)) {
        if (lastoffset <= ADDRESSG(mem)) {
          cur_op = add_init_pad(cur_op, lastoffset - address);
          *offset = lastoffset;
          return cur_op;
        } else {
          cur_op = add_init_pad(cur_op, ADDRESSG(mem) - address);
          address = ADDRESSG(mem);
        }
      }
    }
    if (mem > NOSYM) {
      memdtype = DTYPEG(mem);
      sz = size_of(memdtype);
      while (mem > NOSYM && ADDRESSG(mem) + sz <= lastoffset) {
        cur_op = add_init_subzero_consts(DTYPEG(mem), cur_op, &address,
                                         lastoffset - ADDRESSG(mem));
        mem = SYMLKG(mem);
        memdtype = DTYPEG(mem);
        sz = size_of(memdtype);
      }
    }
    if (address < lastoffset) {
      if (mem == NOSYM || ADDRESSG(mem) == lastoffset) {
        cur_op = add_init_pad(cur_op, lastoffset - address);
        address = lastoffset;
      } else {
        address = 0;
        cur_op = add_init_subzero_consts(DTYPEG(mem), cur_op, &address,
                                         lastoffset - ADDRESSG(mem));
      }
    }
    *offset = address;
    return cur_op;
  default:
    sz = size_of(dtype);
    cur_op->next = make_constval_op(make_lltype_from_dtype(dtype), 0, 0);
    cur_op = cur_op->next;
    *offset = address + sz;
  }
  return cur_op;
}
#endif

/* Allocate an LL_ABI_Info object with room for nargs arguments. */
LL_ABI_Info *
ll_abi_alloc(LL_Module *module, unsigned nargs)
{
  LL_ABI_Info *abi = (LL_ABI_Info*)calloc(
      1, sizeof(LL_ABI_Info) + nargs * sizeof(LL_ABI_ArgInfo));
  abi->module = module;
  abi->nargs = nargs;
  return abi;
}

/* Reclaim: Returns NULL, just to discourage dangling pointers */
LL_ABI_Info *
ll_abi_free(LL_ABI_Info *abi)
{
#if DEBUG
  assert(abi, "No abi to free", 0, ERR_Fatal);
  memset(abi, 0, sizeof(LL_ABI_Info) + (abi->nargs * sizeof(LL_ABI_ArgInfo)));
#endif
  free(abi);
  return NULL;
}

LL_Type *
ll_abi_return_type(LL_ABI_Info *abi)
{
  if (LL_ABI_HAS_SRET(abi))
    return ll_create_basic_type(abi->module, LL_VOID, 0);
  else
    return abi->arg[0].type;
}

bool
ll_abi_use_llvm_varargs(LL_ABI_Info *abi)
{
  if (abi->is_varargs)
    return true;

  if (abi->missing_prototype)
    return abi->call_as_varargs;

  return false;
}

LL_Type *
ll_abi_function_type(LL_ABI_Info *abi)
{
  unsigned i;
  LL_Type **types, **argtypes;
  LL_Type *func_type;

  /* Return type + optional sret + arguments. */
  types = (LL_Type **)calloc(abi->nargs + 2, sizeof(LL_Type *));
  argtypes = types;

  /* Prepend a void return and make the return type in arg[0] an argument. */
  if (LL_ABI_HAS_SRET(abi))
    *argtypes++ = ll_create_basic_type(abi->module, LL_VOID, 0);

  for (i = 0; i <= abi->nargs; i++)
    argtypes[i] = abi->arg[i].type;

  func_type = ll_create_function_type(
      abi->module, types, LL_ABI_HAS_SRET(abi) ? abi->nargs + 1 : abi->nargs,
      ll_abi_use_llvm_varargs(abi));

  free(types);

  return func_type;
}

void
ll_abi_complete_arg_info(LL_ABI_Info *abi, LL_ABI_ArgInfo *arg, DTYPE dtype)
{
  LL_Type *type;
  enum LL_ABI_ArgKind kind = arg->kind;

  if (arg->type)
    return;

  assert(kind != LL_ARG_COERCE, "Missing coercion type", 0, ERR_Fatal);

  type = ll_convert_dtype(abi->module, dtype);
  if (kind == LL_ARG_INDIRECT || kind == LL_ARG_INDIRECT_BUFFERED
   || kind == LL_ARG_BYVAL) {
    assert(type->data_type != LL_VOID,
           "ll_abi_complete_arg_info: void function argument", dtype,
           ERR_Fatal);
    type = ll_get_pointer_type(type);
  }

  arg->type = type;
}

/**
   \brief Process the return type and arguments for func_sptr
   \param mod
   \param func_sptr
   \param update    flag for special handling

   If the update flag is \c true, then the ABI is reconstructed from the AG
   table, taking into account any changes added to the AG table.  Update also
   will set the sptrs which means that this routine should only be called with
   \c true when the sptrs are valid: (i.e., if this routine exists in the
   current module).

   TODO: Rename this function since process_sptr is not called in here.
 */
LL_ABI_Info *
process_ll_abi_func_ftn_mod(LL_Module *mod, SPTR func_sptr, bool update)
{
  int i, ty;
  DTYPE ret_dtype;
  DTLIST *param;
  LL_ABI_Info *abi;
  LL_Type *llt;
  int gblsym = 0;
  int iface = 0;
  unsigned nargs = 0;
  const int stype = STYPEG(func_sptr);

  /* Find the number of arguments, if not found, check if this is an iface */
  if (stype == ST_ENTRY && (gblsym = find_ag(get_llvm_name(func_sptr)))) {
    nargs = get_ag_argdtlist_length(gblsym);
  } else if ((gblsym = find_ag(get_llvm_ifacenm(func_sptr)))) {
    iface = get_llvm_funcptr_ag(func_sptr, get_llvm_ifacenm(func_sptr));
    nargs = get_ag_argdtlist_length(iface);
  } else if ((gblsym = find_ag(get_llvm_name(func_sptr)))) {
    nargs = get_ag_argdtlist_length(gblsym);
  }

  /* If we have already added this, and don't want to update, then return */
  abi = ll_proto_get_abi(ll_proto_key(func_sptr));
  if (!update && gblsym && abi) {
    return abi;
  } else if (!update && abi && stype == ST_PROC && !INMODULEG(func_sptr)) {
    return abi; /* We already have an abi */
  } else if (update && abi) {
    abi = ll_abi_free(abi);
  }

  abi = ll_abi_alloc(mod, nargs);
  abi->is_fortran = true;

  /* If fortran is calling an iso-c function */
  abi->is_iso_c = CFUNCG(func_sptr);

  ll_abi_compute_call_conv(abi, func_sptr, 0);

  /* Update the gblsym abi pointer */
  if (update)
    ll_proto_set_abi(ll_proto_key(func_sptr), abi);

  /* External and never discovered arguments, then we will declare this as a
   * varargs function.  When a call to this function is made, the callsite
   * args from the JSR/GJSR will be used and we will cast away the varargs.
   */
  /*
   * IS_INTERFACE check allows abstract interfaces which have INMODULE
   * bit set to pass through this check.
   */
  if (!nargs && (!INMODULEG(func_sptr) || IS_INTERFACEG(func_sptr)) &&
      (IS_FTN_PROC_PTR(func_sptr) || stype == ST_PROC)) {
    assert(IS_FTN_PROC_PTR(func_sptr) || SCG(func_sptr) == SC_EXTERN ||
               SCG(func_sptr) == SC_NONE || SCG(func_sptr) == SC_DUMMY ||
               STYPEG(func_sptr) == ST_PROC || STYPEG(func_sptr) == ST_ENTRY,
           "process_ll_abi_func_ftn: "
           "Unknown function prototype",
           func_sptr, ERR_Fatal);
    abi->missing_prototype = true;
#if defined(TARGET_ARM)
    abi->call_as_varargs = false;
#else
    abi->call_as_varargs = true;
#endif
  }

  /* Obtain, classify, and create an arg for the return value */
  ret_dtype = get_return_type(func_sptr);
  ty = DTY(ret_dtype);
  if (ty == TY_CHAR || ty == TY_NCHAR ||
      (TY_ISCMPLX(ty) && !CFUNCG(func_sptr) && !CMPLXFUNC_C))
    ret_dtype = DT_NONE;

#if defined(TARGET_LLVM_X8664)
  /* Workaround the X86 ABI */
  switch (ty) {
  case TY_SINT:
  case TY_USINT:
  case TY_SLOG:
    abi->extend_abi_return = !XBIT(183, 0x400000);
    break;
  default:
    break;
  }
#endif
  ll_abi_classify_return_dtype(abi, ret_dtype);
  ll_abi_complete_arg_info(abi, &abi->arg[0], ret_dtype);

  /* Override with a more correct type, to avoid using the
   * fortran-default float if that was specified in ret_dtype.
   * ll_process_routine_parameters() decides to override
   * (See ll_process_routine_parameters() where it calls
   *  set_ag_return_lltype()).
   */
  if (gblsym && (llt = get_ag_return_lltype(gblsym)))
    abi->arg[0].type = llt;

  /* Determine how each arg should be handled */
  if (!abi->missing_prototype) {
    for (i = 1, param = get_argdtlist(gblsym); param;
         ++i, param = get_next_argdtlist(param)) {
      LL_Type *llt = get_lltype_from_argdtlist(param);
      const bool byval = get_byval_from_argdtlist(param);
      abi->arg[i].type = llt; /* HACK FIXME */
      abi->arg[i].kind = byval ? LL_ARG_DIRECT : LL_ARG_INDIRECT;
      abi->arg[i].ftn_pass_by_val = byval;

      /* Only for process_formal_arguments(), and for the current
       * function being compiled (this function).
       *
       * sptr is only valid if it was created in the same translation
       * object that this abi instance is being created in.
       */
      if (update || gbl.currsub == func_sptr ||
          get_master_sptr() == func_sptr || gbl.entries == func_sptr) {
        const SPTR sptr = get_sptr_from_argdtlist(param);
        DTYPE dtype = DTYPEG(sptr);
        abi->arg[i].sptr = sptr;
        if (!dtype || is_iso_cptr(dtype))
          dtype = DT_ADDR;
        else if (byval)
          ll_abi_classify_arg_dtype(abi, &abi->arg[i], dtype);
        if (abi->arg[i].kind == LL_ARG_SIGNEXT) /* Get rid of this */
          abi->arg[i].kind = LL_ARG_DIRECT;
      }
    }
  }

  return abi;
}

/**
   \brief Wrapper to process_ll_abi_func_ftn_mod() passing the default module
 */
LL_ABI_Info *
process_ll_abi_func_ftn(SPTR func_sptr, bool use_sptrs)
{
  return process_ll_abi_func_ftn_mod(llvm_get_current_module(), func_sptr, use_sptrs);
}

/* Generate LL_ABI_Info for a function without a prototype. The return type
 * must be known. */
static LL_ABI_Info *
ll_abi_for_missing_prototype(LL_Module *module, DTYPE return_dtype,
                             int func_sptr, int jsra_flags)
{
  LL_ABI_Info *abi = ll_abi_alloc(module, 0);
  abi->is_varargs = false;
  abi->missing_prototype = true;

  ll_abi_compute_call_conv(abi, func_sptr, jsra_flags);

  ll_abi_classify_return_dtype(abi, return_dtype);
  assert(abi->arg[0].kind, "ll_abi_for_missing_prototype: Unknown return type",
         return_dtype, ERR_Fatal);
  assert(abi->arg[0].kind != LL_ARG_BYVAL, "Return value can't be byval",
         return_dtype, ERR_Fatal);
  ll_abi_complete_arg_info(abi, &abi->arg[0], return_dtype);

  abi->is_fortran = true;

  return abi;
}

LL_ABI_Info *
ll_abi_for_func_sptr(LL_Module *module, SPTR func_sptr, DTYPE dtype)
{
  return process_ll_abi_func_ftn_mod(module, func_sptr, false);
}

LL_ABI_Info *
ll_abi_from_call_site(LL_Module *module, int ilix, DTYPE ret_dtype)
{
  DTYPE return_dtype = DT_NONE;
  int jsra_flags = 0;

  switch (ILI_OPC(ilix)) {
  case IL_GJSR:
  case IL_JSR:
  case IL_QJSR:
    /* Direct call: JSR sym arg-lnk */
    return ll_abi_for_func_sptr(module, ILI_SymOPND(ilix, 1), DT_NONE);

  case IL_GJSRA: {
    /* Indirect call: Look for a GARGRET return type indicator.
     * GARGRET value next-lnk dtype
     * GJSRA addr arg-lnk attr-flags
     */
    const SPTR iface = ILI_SymOPND(ilix, 4);
    const int gargret = ILI_OPND(ilix, 2);
    jsra_flags = ILI_OPND(ilix, 3);
    if (iface == 0)
      return ll_abi_for_missing_prototype(module, ret_dtype, 0, 0);
    if (find_ag(get_llvm_ifacenm(iface)))
      return ll_abi_for_func_sptr(module, iface, DT_NONE);
    get_llvm_funcptr_ag(iface, get_llvm_name(iface));
    if (ILI_OPC(gargret) == IL_GARGRET)
      return_dtype = ILI_DTyOPND(gargret, 3);
  } break;

  case IL_JSRA:
    /* Indirect call: JSRA addr arg-lnk attr-flags */
    jsra_flags = ILI_OPND(ilix, 3);
    break;
  default:
    interr("ll_abi_from_call_site: Unknown call ILI", ilix, ERR_Fatal);
  }

  /* No prototype found, just analyze the return value. */
  if (!return_dtype && ret_dtype)
    return_dtype = ret_dtype;
    /* return_dtype = dtype_from_return_type(ILI_OPC(ret_ili)); */

  if (!return_dtype)
    return_dtype = DT_NONE;

  return ll_abi_for_missing_prototype(module, return_dtype, 0, jsra_flags);
}

/* Create an LL_Type wrapper for an argument type. */
LL_Type *
make_lltype_from_abi_arg(LL_ABI_ArgInfo *arg)
{
  return arg->type;
}

int
visit_flattened_dtype(dtype_visitor visitor, void *context, DTYPE dtype,
                      unsigned address, unsigned member_sptr)
{
  int retval = 0;
  SPTR sptr;

  if (DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_UNION) {
    /* TY_STRUCT sptr tag size align. */
    for (sptr = DTyAlgTyMember(dtype); sptr > NOSYM && retval == 0;
         sptr = SYMLKG(sptr)) {
      assert(STYPEG(sptr) == ST_MEMBER, "Non-member in struct", sptr,
             ERR_Fatal);
      if (DTYPEG(sptr) == dtype) {
        return -1; /* next pointer */
      }
      retval = visit_flattened_dtype(visitor, context, DTYPEG(sptr),
                                     address + ADDRESSG(sptr), sptr);
    }
    return retval;
  }

  return visitor(context, dtype, address, member_sptr);
}

/* HACK, FIXME: This is only to support Fortran.
 * Structs in fortran are stroed in the AG table and searched for in the AG
 * table by our own fortran nameing scheme: struct<struct name>.  This does
 * not mix well with the newer, more unique naming scheme used by our llvm
 * backend... mainly that generates unique struct names via unique_name().
 * Eventually we will want to use the latter functionality everywhere.
 * This casts-away constness.
 */
void
ll_override_type_string(LL_Type *llt, const char *str)
{
  char *clone = llutil_alloc(strlen(str) + 1);
  strcpy(clone, str);

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  /* Cast away constness *eww gross*, gcc hates me */
  // FIXME -- this is wrong headed
  ((struct LL_Type_ *)llt)->str = clone;
#pragma GCC diagnostic pop
}

/**
   \brief Scan the list of struct types and find the corresponding LLDEF
   \arg dtype  The dtype to search for
   \return null iff the struct type is not found

   This is an <i>O(n)</i> operation, where <i>n</i> is the number of struct
   types.
 */
static LLDEF *
LLABI_find_su_type_def(DTYPE dtype)
{
  LLDEF *p;
  for (p = struct_def_list; p; p = p->next) {
    if (p->dtype == dtype)
      return p;
  }
  return NULL;
}

/**
   \brief Scan the list of array types and find the corresponding LLDEF
   \arg dtype  The dtype to search for
   \return null iff the array type is not found

   This is an <i>O(n)</i> operation, where <i>n</i> is the number of array
   types.
 */
static LLDEF *
LLABI_find_array_type_def(DTYPE dtype)
{
  LLDEF *p;
  for (p = llarray_def_list; p; p = p->next) {
    if (p->dtype == dtype)
      return p;
  }
  return NULL;
}

LL_Type *
llfind_su_type_def(DTYPE dtype)
{
  LLDEF *def = LLABI_find_su_type_def(dtype);
  return (def && def->ll_type) ? def->ll_type : NULL;
}

LL_Type *
llfind_array_type_def(DTYPE dtype)
{
  LLDEF *def = LLABI_find_array_type_def(dtype);
  return (def && def->ll_type) ? def->ll_type : NULL;
}


LL_Type *
get_ftn_static_lltype(SPTR sptr)
{
  /* 3 kinds of static
     1) constant
     2) dinited static
     3) uninited static
     we process 2) and 3) the same way.
   */
  LL_Type *llt = NULL;
  char *name;
  char tname[MXIDLN];
  int gblsym;
  DTYPE dtype;

  assert(SCG(sptr) == SC_STATIC, "Expected SC_STATIC storage class", sptr, ERR_Fatal);

  dtype = DTYPEG(sptr);
  // In case of a FTN proc ptr generate lltype as its done for any ptr
  if (is_function(sptr) && !IS_FTN_PROC_PTR(sptr))
    return get_ftn_func_lltype(sptr);
  if (STYPEG(sptr) == ST_CONST)
    return make_lltype_from_dtype(dtype);
  if (DESCARRAYG(sptr) && CLASSG(sptr))
    return make_ptr_lltype(get_ftn_typedesc_lltype(sptr));

  name = get_llvm_name(sptr);
  sprintf(tname, "struct%s", name);

  /* get_typedef_ag will return 0 if lltype does not exist and will create a new
     ag entry with tname as a side effect. dinit processing should fill struct
     layout later. */
  gblsym = get_typedef_ag(tname, NULL);
  if (!gblsym)
    gblsym = get_typedef_ag(tname, NULL); /* now get an ag entry */

  if (AG_LLTYPE(gblsym))
    return get_ag_lltype(gblsym);

  if (ACCINITDATAG(sptr) && (CFUNCG(sptr) || CUDAG(gbl.currsub))) {
    if (DDTG(dtype) != TY_CHAR) {
      dtype = mk_struct_for_llvm_init(getsname(sptr), 0);
      llt = make_lltype_from_dtype(dtype);
      gblsym = get_typedef_ag(getsname(sptr), 0);
      /* the next line is NOT a typo, it is needed for correctness */
      gblsym = get_typedef_ag(getsname(sptr), 0);
      set_ag_lltype(gblsym, llt);
      DTYPEP(sptr, dtype);
      AG_STYPE(gblsym) = STYPEG(sptr);
      return llt;
    }
    return make_lltype_from_dtype(dtype);
  }
  llt = make_lltype_from_dtype(dtype);
  set_ag_lltype(gblsym, llt);
  return llt;
}

LL_Type *
get_ftn_cmblk_lltype(SPTR sptr)
{
  char *name;
  char tname[MXIDLN];
  int midnum;
  LL_Type *llt;
  int gblsym;

  assert(SCG(sptr) == SC_CMBLK, "Expected SC_CMBLK storage class", sptr, ERR_Fatal);

  /* For all SC_CMBLK. We should delay filling out the common block layout until
   * the end of the file or until processing dinit.  If it is dinit'd, then
   * don't change its layout as dinit will fill its layout and cannot be
   * changed.  Otherwise use SIZE field to define the layout - which will be in
   * the form of [i8 x SIZE].  SIZE includes the alignment of common block
   * member, i.e, common /myc/ myint, mychar, myint2 integer myint character
   * mychar integer myint2
   *
   * SIZE of myc will be 12
   */
  name = get_llvm_name(sptr);
  sprintf(tname, "struct%s", name);
  gblsym = find_ag(tname);
  if (!gblsym) {
    get_typedef_ag(tname, NULL);
    gblsym = find_ag(tname);
    llt = make_lltype_from_dtype(DTYPEG(sptr));
    set_ag_lltype(gblsym, llt);
    return llt;
  }
  llt = get_ag_lltype(gblsym);

  midnum = MIDNUMG(sptr);

  if (midnum) {
    LLTYPE(midnum) = llt;
    if (SNAME(midnum) == NULL)
      SNAME(midnum) = SNAME(sptr);
    LLTYPE(midnum) = llt;
  }
  return llt;
}

LL_Type *
get_ftn_typedesc_lltype(SPTR sptr)
{
  LL_Type *llt = NULL;
  char *name;
  char tname[MXIDLN];
  int gblsym;
  DTYPE dtype;

  assert(DESCARRAYG(sptr) && CLASSG(sptr), "Expected DESCARRAY && CLASS symbol",
         sptr, ERR_Fatal);

  name = getsname(sptr);
  gblsym = find_ag(name);
  if (!gblsym) /* create an entry for tihs symbol which will set ag_global */
    gblsym = get_ag(sptr);
  if (SCG(sptr) == SC_STATIC)
    AG_DEFD(gblsym) = 1;

  sprintf(tname, "struct%s", name); /* search for its type */
  gblsym = find_ag(tname);
  if (!gblsym) {
    dtype = get_ftn_typedesc_dtype(sptr);
    llt = make_lltype_from_dtype(dtype);
    gblsym = get_typedef_ag(tname, NULL);
    if (!gblsym)
      gblsym = get_typedef_ag(tname, NULL);
    set_ag_lltype(gblsym, llt);
  }
  llt = get_ag_lltype(gblsym);
  return llt;
}

LL_Type *
get_ftn_extern_lltype(SPTR sptr)
{
  assert(SCG(sptr) == SC_EXTERN, "Expected SC_EXTERN storage class", sptr, ERR_Fatal);

  if (is_function(sptr))
    return get_ftn_func_lltype(sptr);
  if (CFUNCG(sptr))
    return get_ftn_cbind_lltype(sptr);
  if (CLASSG(sptr) && DESCARRAYG(sptr))
    return get_ftn_typedesc_lltype(sptr);
  return make_lltype_from_dtype(DTYPEG(sptr));
}

LL_Type *
get_ftn_cbind_lltype(SPTR sptr)
{
  DTYPE dtype = DTYPEG(sptr);
  DTYPE sdtype;
  ISZ_T anum = 0;
  int tag, numdim, gblsym, d;
  LL_Type *llt = NULL;
  const char *typed;
  char *name;
  char tname[MXIDLN];
  ADSC *ad;

  assert(CFUNCG(sptr), "Expected CBIND type", sptr, ERR_Fatal);

  /* currently BIND(C) type is only allowed on module. If that were to change,
   * we will need to handle here
   */

  if (is_function(sptr))
    return get_ftn_func_lltype(sptr);

  if (SCG(sptr) == SC_STATIC) /* internal procedure bind(c) */
    return get_ftn_static_lltype(sptr);

  if (SCG(sptr) == SC_EXTERN) {
    sdtype = dtype;
    if (DTY(dtype) == TY_ARRAY)
      sdtype = DTySeqTyElement(dtype);
    if (DTY(sdtype) == TY_STRUCT) {
      tag = DTyAlgTyTag(sdtype);
      name = SYMNAME(tag);
      sprintf(tname, "struct%s", name);
      gblsym = find_ag(tname);
      if (!gblsym) {
        llt = make_lltype_from_dtype(sdtype);
        gblsym = get_typedef_ag(tname, NULL);
        typed = process_dtype_struct(sdtype);
        gblsym = get_typedef_ag(tname, typed);
        set_ag_lltype(gblsym, llt);
      }
      llt = get_ag_lltype(gblsym);

      /* We chose to flatten Fortran array into single dimension array because
       * how the dinit processing was done and how we access to its address in
       * the ili, which is linearized.  Not really sure how it dwarf generation
       * should be done - wait until then ...
       */
      if (DTY(dtype) == TY_ARRAY) {
        ad = AD_DPTR(dtype);
        numdim = AD_NUMDIM(ad);
        d = AD_NUMELM(ad);
        if (numdim >= 1 && numdim <= 7) {
          if (d == 0 || STYPEG(d) != ST_CONST) {
            if (XBIT(68, 0x1))
              d = AD_NUMELM(ad) = stb.k1;
            else
              d = AD_NUMELM(ad) = stb.i1;
          }
          anum = ad_val_of(d);
        }
        llt = make_array_lltype(anum, llt);
      }
      return llt;
    }
  }
  return make_lltype_from_dtype(DTYPEG(sptr));
}

LL_Type *
get_ftn_func_lltype(SPTR sptr)
{
  if (is_function(sptr)) {
    LL_ABI_Info *abi;
    if (IS_FTN_PROC_PTR(sptr)) {
      const SPTR iface = get_iface_sptr(sptr);
      if (iface)
        return make_lltype_from_iface(iface);
      return make_lltype_from_dtype(DT_CPTR);
    }
    abi = ll_abi_for_func_sptr(llvm_get_current_module(), sptr, DT_NONE);
    return ll_abi_function_type(abi);
  }
  assert(0, "Expected function type", sptr, ERR_Fatal);
  return NULL;
}

LL_Type *
get_ftn_dummy_lltype(int sptr)
{
  if (!PASSBYVALG(sptr)) {
    const int func_sptr = gbl.currsub;
    const int midnum = MIDNUMG(sptr);
    LL_Type *llt = make_generic_dummy_lltype();
#ifdef OMP_OFFLOAD_LLVM
    const bool is_nvvm = gbl.ompaccel_isdevice && PASSBYVALG(midnum);
#else
    const bool is_nvvm = false;
#endif
    if (is_nvvm || gbl.outlined || ISTASKDUPG(GBL_CURRFUNC)) {
      const DTYPE dtype = DTYPEG(midnum ? midnum : sptr);
      llt = make_ptr_lltype(make_lltype_from_dtype(dtype));
    }
    if (CFUNCG(func_sptr) && currsub_is_sret()) {
      const int fval = FVALG(func_sptr);
      const DTYPE dtype = DTYPEG(func_sptr);
      if ((sptr == fval) || (midnum == fval))
        llt = make_ptr_lltype(make_lltype_from_dtype(dtype));
      if (midnum == fval)
        LLTYPE(midnum) = llt;
    } else if (DTYPEG(sptr) == DT_ADDR && midnum) {
      LLTYPE(midnum) = llt;
    }
    LLTYPE(sptr) = llt;
    return llt;
  }
  return make_ptr_lltype(make_lltype_from_dtype(DTYPEG(sptr)));
}

LL_Type *
get_ftn_hollerith_type(int sptr)
{
  /* we need to cheat for hollerith type if we need to print out the space after
   * the dtype For example, for 'a', we may need to put 3 empty space after the
   * 'a' to keep it the memory after 'a' clean.  This is needed when we pass 'a'
   * to function and it expects integer.
   */
  LL_Type *llt = NULL;
  DTYPE dtype = DTYPEG(sptr);

  if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
    if (HOLLG(sptr) && STYPEG(sptr) == ST_CONST) {
      int len = get_hollerith_size(sptr);
      len = len + DTyCharLength(dtype);
      /* need to create a char of this size */
      dtype = get_type(2, DTY(dtype), len);
      llt = make_lltype_from_dtype(dtype);
      LLTYPE(sptr) = llt;
      return llt;
    }
  }
  return make_lltype_from_dtype(dtype);
}

LL_InstrListFlags
ll_instr_flags_from_aop(ATOMIC_RMW_OP aop)
{
  switch (aop) {
  default:
    assert(false, "gen_llvm_atomicrmw_expr: unimplemented op", aop, ERR_Fatal);
    return InstrListFlagsNull;
  case AOP_XCHG:
    return ATOMIC_XCHG_FLAG;
  case AOP_ADD:
    return ATOMIC_ADD_FLAG;
  case AOP_SUB:
    return ATOMIC_SUB_FLAG;
  case AOP_AND:
    return ATOMIC_AND_FLAG;
  case AOP_OR:
    return ATOMIC_OR_FLAG;
  case AOP_XOR:
    return ATOMIC_XOR_FLAG;
  case AOP_MIN:
    return ATOMIC_MIN_FLAG;
  case AOP_MAX:
    return ATOMIC_MAX_FLAG;
  }
}

/* Compute the appropriate coercion type for passing dtype in GPRs. */
LL_Type *
ll_coercion_type(LL_Module *module, DTYPE dtype, ISZ_T size, ISZ_T reg_size)
{
  LL_Type *parts[2] = {NULL, NULL};

  /* An empty or unknown struct will have size 0. Treat it as a single byte
   * which won't be correct, but at least we can create function pointers
   * with plausible signature. */
  if (size == 0)
    size = 1;

  /* Depending on size, create one of these coercion types:

     i<n> for size <= 8,
     [n x i64] for size % 8 == 0, or
     { [n x i64], i<m> } otherwise.
   */

  if (size >= reg_size) {
    parts[0] = ll_create_int_type(module, reg_size * BITS_IN_BYTE);
    if (size > reg_size)
      parts[0] = ll_get_array_type(parts[0], size / reg_size, 0);
  }

  /* parts[1] is the odd padding. */
  if (size % reg_size)
    parts[1] = ll_create_int_type(module, reg_size * (size % reg_size));

  /* Put the parts together in a struct if necessary. */
  if (parts[0] && parts[1])
    return ll_create_anon_struct_type(module, parts, 2, false, LL_AddrSp_Default);

  return parts[0] ? parts[0] : parts[1];
}


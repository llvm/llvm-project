/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief ompaccel.c - OpenMP GPU Offload for NVVM Targets. It uses
 * libomptarget
 */
#ifdef OMP_OFFLOAD_LLVM

#include "kmpcutil.h"
#include "error.h"
#include "semant.h"
#include "ilmtp.h"
#include "ilm.h"
#include "ili.h"
#include "expand.h"
#include "exputil.h"
#include "outliner.h"
#include "machreg.h"
#include "mp.h"
#include "ll_structure.h"
#include "llmputil.h"
#include "ccffinfo.h"
#include "llutil.h"
#include "ompaccel.h"
#include "tgtutil.h"
#include "dinit.h"
#include "assem.h"
#include "dinitutl.h"
#include "cgllvm.h"
#include "cgmain.h"

#include "regutil.h"
#include "dtypeutl.h"
#include "llassem.h"
#include "ll_ftn.h"
#include "symfun.h"
#include "../../flang1/flang1exe/global.h"

#define NOT_IMPLEMENTED(_pragma) \
  error((error_code_t)1200, ERR_Fatal, 0, _pragma, NULL)
#define NOT_IMPLEMENTED_CANTCOMBINED(_pragma, _pragma2) \
  error((error_code_t)1201, ERR_Fatal, 0, _pragma, _pragma2)
#define NOT_IMPLEMENTED_NEEDCOMBINED(_pragma, _pragma2) \
  error((error_code_t)1202, ERR_Fatal, 0, _pragma, _pragma2)

/* Initial Max target region */
#define INC_EXP 2
int tinfo_size = 50;
int tinfo_size_reductions = 10;

int num_tinfos = 0;
OMPACCEL_TINFO **tinfos;
OMPACCEL_TINFO *current_tinfo = nullptr;
OMP_TARGET_MODE NextTargetMode = mode_none_target;

const char *nvvm_target_triple;
void
ompaccel_set_targetriple(const char *tp)
{
  nvvm_target_triple = tp;
}
const char *
ompaccel_get_targetriple()
{
  return nvvm_target_triple;
}
static int reductionFunctionCounter = 0;

static void
_long_unsigned(int lilix, int *dt, bool *punsigned, DTYPE dtype)
{
  ILI_OP opc;
  opc = ILI_OPC(lilix);
  int dty = DTY(dtype);
  if (dtype == DT_INT || dtype == DT_UINT) {
    if (size_of(dtype) > 4)
      *dt = 2;
    else
      *dt = 1;
  } else if (dtype == DT_FLOAT || dtype == DT_UINT8) {
    *dt = 3;
  } else if (dty == TY_INT8) {
    *dt = 2;
  } else if (dty == TY_DBLE) {
    *dt = 4;
  }

  // todo ompaccel I don't know how to handle others

  switch (DTY(dtype)) {
  case TY_UINT:
  case TY_USINT:
  case TY_UINT8:
  case TY_UINT128:
    if (opc != IL_ICON)
      *punsigned = TRUE;
    break;
  default:
    break;
  }

} /* _long_unsigned */

static bool
_pointer_type(DTYPE dtype)
{
  if (dtype && DTY(dtype) == TY_PTR)
    return TRUE;
  return FALSE;
} /* _pointer_type */

int
mk_ompaccel_ldsptr(SPTR sptr)
{
  ISZ_T sz;
  DTYPE dtype;
  // it's function
  if (DTYPEG(sptr) == DT_NONE && STYPEG(sptr) == ST_ENTRY) {
    int nme = addnme(NT_VAR, sptr, 0, 0);
    int ili = mk_address(sptr);
    return ad3ili(IL_LDA, ili, nme, MSZ_PTR);
  } else {
    sz = size_of(DTYPEG(sptr));
    dtype = DTYPEG(sptr);

    if (STYPEG(sptr) == ST_CONST) {
      if (sz == 8)
        return ad_kcon(CONVAL1G(sptr), CONVAL2G(sptr));
      else
        return ad_icon(CONVAL2G(sptr));
    } else {
      int nme = addnme(NT_VAR, sptr, 0, 0);
      int ili = mk_address(sptr);
      if (ILI_OPC(ili) == IL_LDA)
        nme = ILI_OPND(ili, 2);
      if (_pointer_type(dtype) || DTY(dtype) == TY_ARRAY) {
        return ad3ili(IL_LDA, ili, nme, MSZ_PTR);
      } else {
        if (sz == 8)
          return ad3ili(IL_LDKR, ili, nme, MSZ_I8);
        else
          return ad3ili(IL_LD, ili, nme, mem_size(DTY(DTYPEG(sptr))));
      }
    }
  }
}

int
mk_ompaccel_load(int ili, DTYPE dtype, int nme)
{
  if (_pointer_type(dtype))
    return ad3ili(IL_LDA, ili, nme, MSZ_PTR);
  else {
    switch (dtype) {
    case DT_INT:
      if (size_of(dtype) > 4)
        return ad3ili(IL_LDKR, ili, nme, MSZ_WORD);
      else
        return ad3ili(IL_LD, ili, nme, MSZ_WORD);
    case DT_REAL:
      if (size_of(dtype) > 4)
        return ad3ili(IL_LDKR, ili, nme, MSZ_F8);
      else
        return ad3ili(IL_LDSP, ili, nme, MSZ_F8);
      break;
    case DT_DBLE:
      return ad3ili(IL_LDDP, ili, nme, MSZ_DBLE);
      break;
    case DT_CMPLX:
      return ad3ili(IL_LDDCMPLX, ili, nme, MSZ_F16);
      break;
    case DT_NONE:
      return ad3ili(IL_LD, ili, nme, MSZ_WORD);
      break;
    default:
      return 0;
      break;
    }
  }
}

static int
mk_ompaccel_ld(int ili, int nme)
{
  return mk_ompaccel_load(ili, DT_NONE, nme);
}

int
mk_ompaccel_store(int ili_value, DTYPE dtype, int nme, int ili_address)
{
  if (_pointer_type(dtype))
    return ad4ili(IL_STA, ili_value, ili_address, nme, MSZ_PTR);
  else {
    switch (dtype) {
    case DT_LOG:
      return ad4ili(IL_ST, ili_value, ili_address, nme, MSZ_WORD);
      break;
    case DT_INT:
      return ad4ili(IL_ST, ili_value, ili_address, nme, MSZ_WORD);
      break;
    case DT_REAL:
      return ad4ili(IL_STSP, ili_value, ili_address, nme, MSZ_F4);
      break;
    case DT_DBLE:
      return ad4ili(IL_STDP, ili_value, ili_address, nme, MSZ_DBLE);
      break;
    case DT_INT8:
      return ad4ili(IL_STKR, ili_value, ili_address, nme, MSZ_I8);
      break;
    case DT_NONE:
      return ad4ili(IL_ST, ili_value, ili_address, nme, MSZ_WORD);
      break;
    default:
      return 0;
      break;
    }
  }
}

static int
mk_ompaccel_stsptr(int ili_value, SPTR sptr)
{
  ISZ_T sz = size_of(DTYPEG(sptr));
  DTYPE dtype = DTYPEG(sptr);
  int ili;
  int nme = addnme(NT_VAR, sptr, 0, 0);
  if (STYPEG(sptr) == ST_CONST) {
    if (sz == 8)
      ili = ad_kcon(CONVAL1G(sptr), CONVAL2G(sptr));
    else
      ili = ad_icon(CONVAL2G(sptr));
  } else {
    ili = mk_address(sptr);
    if (ILI_OPC(ili) == IL_LDA)
      nme = ILI_OPND(ili, 2);
  }
  return mk_ompaccel_store(ili_value, dtype, nme, ili);
}

static int
mk_ompaccel_and(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2)
{
  ILI_OP opc;
  int dt = 0;
  bool uu = FALSE;
  if (!ili1)
    return ili2;
  if (!ili2)
    return ili1;
  if (_pointer_type(dtype1) || _pointer_type(dtype2)) {
    return ad3ili(IL_AADD, ili1, ili2, 0);
  } else {
    _long_unsigned(ili1, &dt, &uu, dtype1);
    _long_unsigned(ili2, &dt, &uu, dtype2);
    /* signed */
    if (!uu) {
      opc = IL_AND;
    } else {
      opc = IL_KAND;
    }
  }
  return ad2ili(opc, ili1, ili2);
}

static int
mk_ompaccel_iand(int ili1, int ili2)
{
  return mk_ompaccel_and(ili1, DT_INT, ili2, DT_INT);
}

static int
mk_ompaccel_shift(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2)
{
  ILI_OP opc = IL_NONE;
  int dt = 0;
  bool uu = FALSE;
  if (!ili1)
    return ili2;
  if (!ili2)
    return ili1;

  _long_unsigned(ili1, &dt, &uu, dtype1);
  _long_unsigned(ili2, &dt, &uu, dtype2);
  /* signed */
  if (!uu) {
    if (dt == 1)
      opc = IL_RSHIFT;
    else if (dt == 2)
      opc = IL_KARSHIFT;
  } else {
    if (dt == 1)
      opc = IL_URSHIFT;
    else if (dt == 2)
      opc = IL_KURSHIFT;
  }
  assert(opc != IL_NONE, "Correct IL is not found.", 0, ERR_Fatal);
  return ad2ili(opc, ili1, ili2);
}

int
mk_ompaccel_compare(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2, int CC)
{
  ILI_OP opc = IL_NONE;
  int dt = 0;
  bool uu = FALSE;
  if (!ili1)
    return ili2;
  if (!ili2)
    return ili1;

  _long_unsigned(ili1, &dt, &uu, dtype1);
  _long_unsigned(ili2, &dt, &uu, dtype2);
  /* signed */
  if (!uu) {
    if (dt == 1)
      opc = IL_ICMP;
    else if (dt == 2)
      opc = IL_KCMP;
  } else {
    if (dt == 1)
      opc = IL_UICMP;
    else if (dt == 2)
      opc = IL_UKCMP;
  }
  assert(opc != IL_NONE, "Correct IL is not found.", 0, ERR_Fatal);
  return ad3ili(opc, ili1, ili2, CC);
}

int
mk_ompaccel_add(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2)
{
  ILI_OP opc = IL_NONE;
  int dt = 0;
  bool uu = FALSE;
  if (!ili1)
    return ili2;
  if (!ili2)
    return ili1;
  if (_pointer_type(dtype1) || _pointer_type(dtype2)) {
    return ad3ili(IL_AADD, ili1, ili2, 0);
  } else {
    _long_unsigned(ili1, &dt, &uu, dtype1);
    _long_unsigned(ili2, &dt, &uu, dtype2);
    /* signed */
    if (!uu) {
      if (dt == 1)
        opc = IL_IADD;
      else if (dt == 2)
        opc = IL_KADD;
      else if (dt == 3)
        opc = IL_FADD;
      else if (dt == 4)
        opc = IL_DADD;
      else if (dt == 5)
        opc = IL_SCMPLXADD;
      else if (dt == 6)
        opc = IL_DCMPLXADD;
    } else {
      if (dt == 1)
        opc = IL_UIADD;
      else if (dt == 2)
        opc = IL_UKADD;
    }
  }
  assert(opc != IL_NONE, "Correct IL is not found.", 0, ERR_Fatal);
  return ad2ili(opc, ili1, ili2);
} /* mk_ompaccel_add */

int
mk_ompaccel_mul(int ili1, DTYPE dtype1, int ili2, DTYPE dtype2)
{
  ILI_OP opc = IL_NONE;
  int dt = 0;
  bool uu = FALSE;
  if (!ili1)
    return ili2;
  if (!ili2)
    return ili1;
  if (_pointer_type(dtype1) || _pointer_type(dtype2)) {
    // todo ompaccel not sure what to do here.
    return ad3ili(IL_KMUL, ili1, ili2, 0);
  } else {
    _long_unsigned(ili1, &dt, &uu, dtype1);
    _long_unsigned(ili2, &dt, &uu, dtype2);
    /* signed */
    if (!uu) {
      if (dt == 1)
        opc = IL_IMUL;
      else if (dt == 2)
        opc = IL_KMUL;
      else if (dt == 3)
        opc = IL_FMUL;
      else if (dt == 4)
        opc = IL_DMUL;
      else if (dt == 5)
        opc = IL_SCMPLXMUL;
      else if (dt == 6)
        opc = IL_DCMPLXMUL;
    } else {
      if (dt == 1)
        opc = IL_UIMUL;
      else if (dt == 2)
        opc = IL_UKMUL;
    }
  }
  assert(opc != IL_NONE, "Correct IL is not found.", 0, ERR_Fatal);
  return ad2ili(opc, ili1, ili2);
} /* mk_ompaccel_mul */

static SPTR
mk_ompaccel_getnewccsym(int letter, int n, DTYPE dtype, SC_KIND SCkind,
                        SYMTYPE symtype)
{
  SPTR sptr = getnewccsym(letter, n, symtype);
  DTYPEP(sptr, dtype);
  SCP(sptr, SCkind);
  OMPACCDEVSYMP(sptr, 1);
  return sptr;
}

SPTR
mk_ompaccel_addsymbol(const char *name, DTYPE dtype, SC_KIND SCkind,
                      SYMTYPE symtype)
{
  SPTR sptr = getsymbol(name);
  DTYPEP(sptr, dtype);
  STYPEP(sptr, symtype);
  SCP(sptr, SCkind);
  OMPACCDEVSYMP(sptr, 1);
  return sptr;
}

static void
mk_ompaccel_function_end(SPTR func_sptr)
{
  int bihx, endlab;
  bihx = expb.curbih;
  bihx = addbih(bihx);
  rdilts(bihx);
  addilt(0, ad1ili(IL_EXIT, func_sptr));
  wrilts(bihx);
  BIH_XT(bihx) = 1;
  BIH_LAST(bihx) = 1;
  endlab = getlab();
  STYPEP(endlab, ST_LABEL);
  RFCNTP(endlab, 1);
  CCSYMP(endlab, 1);
  ILIBLKP(endlab, bihx);
  BIH_LABEL(bihx) = SPTR(endlab);
}

static SPTR
mk_ompaccel_function(const char *name, int n_params, const SPTR *param_sptrs,
                     bool isDeviceFunc)
{
  /* Create a function symbol along with parameters */
  int dpdscp, bihx;
  SPTR func_sptr, sym;
  func_sptr = getsymbol(name);
  TASKFNP(func_sptr, FALSE);
  ISTASKDUPP(func_sptr, FALSE);
  FUNCLINEP(func_sptr, gbl.lineno);
  STYPEP(func_sptr, ST_ENTRY);
  CFUNCP(func_sptr, 1);
  DEFDP(func_sptr, 1);
  SCP(func_sptr, SC_EXTERN);
  ADDRTKNP(func_sptr, 1);
  DCLDP(func_sptr, 1);
  DTYPEP(func_sptr, DT_NONE);

  if (isDeviceFunc)
    OMPACCFUNCDEVP(func_sptr, 1);
  PARAMCTP(func_sptr, n_params);
  dpdscp = aux.dpdsc_avl;
  DPDSCP(func_sptr, dpdscp);
  aux.dpdsc_avl += n_params;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + n_params + 100);

  for (int i = 0; i < n_params; ++i) {
    sym = param_sptrs[i];
    aux.dpdsc_base[dpdscp++] = sym;
  }

  /* Initialize with an Entry Block */
  GBL_CURRFUNC = func_sptr;
  gbl.entries = GBL_CURRFUNC;

  ds_init();

  gbl.lineno = 0;
  gbl.findex = 0;
  bihx = addbih(0);
  gbl.entbih = bihx;
  BIH_LABEL(bihx) = GBL_CURRFUNC;
  rdilts(bihx);
  addilt(0, ad1ili(IL_ENTRY, GBL_CURRFUNC));
  wrilts(bihx);
  BIH_FT(bihx) = 1;
  BIH_EN(bihx) = 1;
  BIHNUMP(GBL_CURRFUNC, bihx);
  BIH_LABEL(bihx) = GBL_CURRFUNC;

  expb.curbih = bihx;

  return func_sptr;
}

static int
mk_reduction_op(int redop, int lili, DTYPE dtype1, int rili, DTYPE dtype2)
{
  switch (redop) {
  case 1:
  case 2:
    return mk_ompaccel_add(lili, dtype1, rili, dtype2);
  case 3:
    return mk_ompaccel_mul(lili, dtype1, rili, dtype2);
  default:
    static_assert(true, "Rest of reduction operators are not implemented yet.");
    break;
  }
  return 0;
}

DTYPE
mk_ompaccel_array_dtype(DTYPE atype, int size)
{
  DTYPE dtype;
  {
    ADSC *adsc;
    INT con[2] = {0, size};

    dtype = get_array_dtype(1, atype);
    adsc = AD_DPTR(dtype);
    AD_LWBD(adsc, 0) = stb.i1;
    AD_UPBD(adsc, 0) = getcon(con, DT_INT);
    AD_NUMELM(adsc) = AD_UPBD(adsc, 0);
  }

  return dtype;
} /* make_array_dtype */

static void
open_OMP_OFFLOAD_LLVM_file()
{
  FILE *F;
  F = fopen(gbl.ompaccfilename, "w");
  if (F == nullptr) {
#if DEBUG
    fprintf(stderr, "Trying to open temp file %s\n", gbl.ompaccfilename);
#endif
  }
  gbl.ompaccfile = F;
}

INLINE static SPTR
create_nvvm_sym(const char *name, DTYPE dtype)
{
  SPTR sptr = getsymbol(name);
  DEFDP(sptr, 1);
  DTYPEP(sptr, dtype);
  CFUNCP(sptr, 1);
  STYPEP(sptr, ST_ENTRY);
  SCP(sptr, SC_STATIC);
  ADDRTKNP(sptr, 1);
  PARAMCTP(sptr, 0);
  return sptr;
}

INLINE static SPTR
create_sregs(const char *name)
{
  return create_nvvm_sym(name, DT_INT);
}

void
ompaccel_init()
{
  /* Create file to write device code */
  open_OMP_OFFLOAD_LLVM_file();
  /* Create target pool */
  tinfos = (OMPACCEL_TINFO **)sccrelal(
      (char *)tinfos, ((BIGUINT64)((tinfo_size) * sizeof(OMPACCEL_TINFO *))));
}

void
ompaccel_initsyms()
{
  /* Create thread id sreg symbols */
  init_nvvm_syms = create_sregs(NVVM_SREG[threadIdX]);
  create_sregs(NVVM_SREG[threadIdY]);
  create_sregs(NVVM_SREG[threadIdZ]);
  /* Create block id sreg symbols */
  create_sregs(NVVM_SREG[blockIdX]);
  create_sregs(NVVM_SREG[blockIdY]);
  create_sregs(NVVM_SREG[blockIdZ]);
  /* Create block id sreg symbols */
  create_sregs(NVVM_SREG[blockDimX]);
  create_sregs(NVVM_SREG[blockDimY]);
  create_sregs(NVVM_SREG[blockDimZ]);
  /* Create block id sreg symbols */
  create_sregs(NVVM_SREG[gridDimX]);
  create_sregs(NVVM_SREG[gridDimY]);
  create_sregs(NVVM_SREG[gridDimZ]);
  // todo create others nvvm things too
  create_sregs(NVVM_SREG[warpSize]);

  /* Create llvm intrinsics symbols */
  init_nvvm_intrinsics = create_nvvm_sym(NVVM_INTRINSICS[barrier0], DT_NONE);
  create_nvvm_sym(NVVM_INTRINSICS[barrier], DT_NONE);
}

int
ompaccel_nvvm_get(nvvm_sregs sreg)
{
  SPTR sptr = SPTR(init_nvvm_syms + sreg);
  ll_make_ftn_outlined_params(sptr, 0, nullptr);
  ll_process_routine_parameters(sptr);
  return ll_ad_outlined_func2(IL_DFRIR, IL_JSR, sptr, 0, nullptr);
}

int
ompaccel_nvvm_mk_barrier(nvvm_barriers btype)
{
  SPTR sptr;
  if (btype == CTA_BARRIER) {
    sptr = (SPTR)(init_nvvm_intrinsics + barrier0);
    ll_make_ftn_outlined_params(sptr, 0, 0);
    ll_process_routine_parameters(sptr);
    return ll_ad_outlined_func2(IL_NONE, IL_JSR, sptr, 0, nullptr);
  }
  static_assert(true, "Other nvvm intrinsics are not implemented yet.");
}

int
ompaccel_nvvm_get_gbl_tid()
{
  int ilix, iliy, iliz;
  ilix = ad2ili(IL_ISUB, ompaccel_nvvm_get(blockDimX), ad_icon(32));
  ilix = ad2ili(IL_IMUL, ompaccel_nvvm_get(blockIdX), ilix);

  iliy = ad2ili(IL_ISUB, ompaccel_nvvm_get(warpSize), ad_icon(1));
  iliy = ad2ili(IL_XOR, iliy, ad_icon(-1));
  iliz = ad2ili(IL_ISUB, ompaccel_nvvm_get(blockDimX), ad_icon(1));
  iliz = ad2ili(IL_AND, iliy, iliz);
  iliz = ad2ili(IL_ISUB, iliz, ad_icon(1));
  iliz = ad2ili(IL_AND, iliz, ompaccel_nvvm_get(threadIdX));

  iliy = ad2ili(IL_IADD, iliz, ilix);
  return iliy;
}

void
ompaccel_tinfo_current_set_mode(OMP_TARGET_MODE type)
{
  current_tinfo->mode = type;
}

void
ompaccel_tinfo_set_mode_next_target(OMP_TARGET_MODE type)
{
  NextTargetMode = type;
}

OMP_TARGET_MODE
ompaccel_tinfo_current_target_mode()
{
  return current_tinfo->mode;
}

OMPACCEL_TINFO *
ompaccel_tinfo_create(SPTR func_sptr, int max_nargs)
{
  OMPACCEL_TINFO *info;
  if (DBGBIT(61, 0x10) && gbl.dbgfil != nullptr)
    fprintf(gbl.dbgfil, "#target add request for sptr:%d [%s]\n", func_sptr,
            SYMNAME(func_sptr));

  NEW(info, OMPACCEL_TINFO, 1);
  info->func_sptr = func_sptr;
  info->n_symbols = 0;
  if (max_nargs != 0) {
    NEW(info->symbols, OMPACCEL_SYM, max_nargs);
    NEW(info->quiet_symbols, OMPACCEL_SYM, max_nargs);
  } else {
    info->symbols = nullptr;
    info->quiet_symbols = nullptr;
  }
  info->sz_symbols = info->sz_quiet_symbols = max_nargs;
  info->mode = NextTargetMode;
  NextTargetMode = mode_none_target;
  info->nowait = false;
  info->n_quiet_symbols = 0;
  NEW(info->reduction_symbols, OMPACCEL_RED_SYM, tinfo_size_reductions);
  info->n_reduction_symbols = 0;

  /* add ot to array */
  NEED(num_tinfos + 1, tinfos, OMPACCEL_TINFO *, tinfo_size,
       tinfo_size * INC_EXP);
  tinfos[num_tinfos++] = info;

  /* linking */
  if (current_tinfo != nullptr)
    info->parent_tinfo = current_tinfo;
  else
    info->parent_tinfo = nullptr;
  current_tinfo = info;
  return info;
}

bool
ompaccel_tinfo_has(int func_sptr)
{
  for (int i = 0; i < num_tinfos; ++i) {
    if (tinfos[i]->func_sptr == func_sptr) {
      return true;
    }
  }
  return false;
}

OMPACCEL_TINFO *
ompaccel_tinfo_get(int func_sptr)
{
  int i;
  for (i = 0; i < num_tinfos; ++i) {
    if (tinfos[i]->func_sptr == func_sptr) {
      return tinfos[i];
    }
  }
  return nullptr;
}

SPTR
ompaccel_create_device_symbol(SPTR sptr, int count)
{
  SPTR sym, sptr_alloc;
  char name[252];
  DTYPE dtype = DTYPEG(sptr);
  bool byval;
  if (DTYPEG(sptr) == DT_ADDR || DTY(DTYPEG(sptr)) == TY_ARRAY)
    byval = false;
  else
    byval = true;
  if (byval) {
    sprintf(name, "Arg_%s_%d", SYMNAME(sptr), count);
  } else {
    if (strlen(SYMNAME(sptr)) == 0)
      sprintf(name, "Arg_%s%d", SYMNAME(sptr), count);
    else
      sprintf(name, "Arg_%s", SYMNAME(sptr));
  }
  sym = getsymbol(name);

  SCP(sym, SC_DUMMY);

  if (dtype == DT_CPTR) {
    dtype = DT_INT8;
  }
  // assume it's base of allocatable descriptor
  if (strncmp(SYMNAME(sptr), ".Z", 2) == 0) {
    for (int j = 0; j < current_tinfo->n_quiet_symbols; ++j)
      if (MIDNUMG(current_tinfo->quiet_symbols[j].host_sym) == sptr)
        sptr_alloc = current_tinfo->quiet_symbols[j].host_sym;
    byval = false;
    DTYPEP(sym, DTYPE(DTYPEG(sptr_alloc) + 1));
    sptr_alloc = ((SPTR)0);

  } else {
    DTYPEP(sym, dtype);
  }
  STYPEP(sym, ST_VAR);
  PASSBYVALP(sym, byval);

  OMPACCDEVSYMP(sym, TRUE);
  return sym;
}

INLINE static SPTR
add_symbol_to_function(SPTR func, SPTR sym)
{
  int paramct;
  paramct = PARAMCTG(func);
  paramct += 1;
  aux.dpdsc_base[paramct] = sym;
  PARAMCTP(func, paramct);
  aux.dpdsc_avl += 1;
}

INLINE static SPTR
get_devsptr(OMPACCEL_TINFO *tinfo, SPTR host_symbol)
{
  int i;
  if (tinfo == nullptr)
    return host_symbol;

  for (i = 0; i < tinfo->n_symbols; ++i) {
    if (tinfo->symbols[i].host_sym == host_symbol) {
      if (tinfo->symbols[i].device_sym == NOSYM) {
        /* It is second case that we catch the symbols in target region from the
         * ILM. In case there is a symbol that has no device symbol created, we
         * should create device symbol for it also we should add it function
         * parameter. */
        tinfo->symbols[i].device_sym =
            ompaccel_create_device_symbol(tinfo->symbols[i].host_sym, 1);
        add_symbol_to_function(tinfo->func_sptr, tinfo->symbols[i].device_sym);
      }
      return tinfo->symbols[i].device_sym;
    }
  }
  return host_symbol;
}

INLINE static SPTR
get_devsptr2(OMPACCEL_TINFO *tinfo, SPTR host_symbol)
{
  int i;
  for (i = 0; i < tinfo->n_symbols; ++i) {
    if (tinfo->symbols[i].device_sym == host_symbol) {
      return tinfo->symbols[i].host_sym;
    }
  }
  return host_symbol;
}

OMPACCEL_TINFO *
ompaccel_tinfo_current_get_targetdata()
{
  OMPACCEL_TINFO *tinfo = current_tinfo;
  while (tinfo != nullptr) {
    if (tinfo->mode == mode_target_data_region)
      return tinfo;
    if (tinfo->parent_tinfo == nullptr)
      break;
    tinfo = tinfo->parent_tinfo;
  }
  ompaccel_msg_interr("XXX", "Beginning of 'target data' is not found. ");
  return nullptr;
}

OMPACCEL_TINFO *
ompaccel_tinfo_current_get()
{
  return current_tinfo;
}

DTYPE
ompaccel_tinfo_current_get_dev_dtype(DTYPE org_dtype)
{
  int i;
  DTYPE dev_dtype = org_dtype;
  if (current_tinfo != nullptr) {
    for (i = 0; i < current_tinfo->n_quiet_symbols; ++i) {
      if (DTYPEG(current_tinfo->quiet_symbols[i].host_sym) == org_dtype) {
        dev_dtype = DTYPEG(current_tinfo->quiet_symbols[i].device_sym);
        break;
      }
    }

    for (i = 0; i < current_tinfo->n_symbols; ++i) {
      if (DTYPEG(current_tinfo->symbols[i].host_sym) == org_dtype) {
        dev_dtype = DTYPEG(current_tinfo->symbols[i].device_sym);
        break;
      }
    }
  }
  if (DBGBIT(61, 2) && gbl.dbgfil != nullptr) {
    if (org_dtype != dev_dtype) {
      fprintf(gbl.dbgfil, "[ompaccel] REPLACED org_dtype:%d --> dev_dtype:%d",
              org_dtype, dev_dtype);
    }
  }
  return dev_dtype;
}

SPTR
ompaccel_tinfo_parent_get_devsptr(SPTR host_symbol)
{
  int i;
  if (current_tinfo->parent_tinfo == nullptr)
    return host_symbol;
  for (i = 0; i < current_tinfo->parent_tinfo->n_quiet_symbols; ++i) {
    if (current_tinfo->parent_tinfo->quiet_symbols[i].host_sym == host_symbol) {
      return current_tinfo->parent_tinfo->quiet_symbols[i].device_sym;
    }
  }
  return host_symbol;
}

bool
ompaccel_tinfo_current_is_registered(SPTR host_symbol)
{
  int i;
  if (current_tinfo == nullptr || !host_symbol)
    return false;

  for (i = 0; i < current_tinfo->n_symbols; ++i) {
    if (current_tinfo->symbols[i].host_sym == host_symbol) {
      return true;
    }
  }
  return false;
}

SPTR
ompaccel_tinfo_current_get_devsptr(SPTR host_symbol)
{
  SPTR device_symbol;
  if (current_tinfo == nullptr || !host_symbol)
    return host_symbol;

  device_symbol = get_devsptr(current_tinfo, host_symbol);

  if (device_symbol == host_symbol && current_tinfo->parent_tinfo != nullptr)
    device_symbol = get_devsptr2(current_tinfo->parent_tinfo, host_symbol);

  if ((DBGBIT(61, 2)) && gbl.dbgfil != nullptr &&
      device_symbol != host_symbol) {
    fprintf(gbl.dbgfil,
            "[ompaccel] REPLACED host_symbol:%d[%s] --> device_symbol:%d[%s]",
            host_symbol, SYMNAME(host_symbol), device_symbol,
            SYMNAME(device_symbol));
    fprintf(gbl.dbgfil, "\n");
  }

  return device_symbol;
}

static bool
tinfo_update_maptype(OMPACCEL_SYM *tsyms, int nargs, SPTR host_symbol,
                     int map_type)
{
  int i;
  for (i = 0; i < nargs; ++i) {
    if (tsyms[i].host_sym == host_symbol) {
      tsyms[i].map_type = map_type;
      if (STYPEG(tsyms[i].host_sym) != ST_ARRAY) {
        /* if scalar variables are used in map clause, pass them by reference */
        if (map_type & OMP_TGT_MAPTYPE_FROM || map_type & OMP_TGT_MAPTYPE_TO)
          PASSBYREFP(tsyms[i].device_sym, 1);
        PASSBYVALP(tsyms[i].device_sym, 0);
      }
      return true;
    }
  }
  return false;
}

void
ompaccel_tinfo_current_add_reductionitem(SPTR private_sym, SPTR shared_sym,
                                         int redop)
{
  if (current_tinfo == nullptr)
    ompaccel_msg_interr("XXX", "Current target info is not found.\n");

  current_tinfo->reduction_symbols[current_tinfo->n_reduction_symbols]
      .private_sym = private_sym;
  current_tinfo->reduction_symbols[current_tinfo->n_reduction_symbols]
      .shared_sym = shared_sym;
  current_tinfo->reduction_symbols[current_tinfo->n_reduction_symbols].redop =
      redop;
  current_tinfo->n_reduction_symbols++;
  // it is initially created pass by value. It should be address, it should be
  // copied back to the host.
  PASSBYVALP(private_sym, 0);

  /* Mark reduction variable as tofrom */
  if (ompaccel_tinfo_current_target_mode() ==
      mode_target_teams_distribute_parallel_for ||
      ompaccel_tinfo_current_target_mode() ==
          mode_target_teams_distribute_parallel_for_simd)
    ompaccel_tinfo_current_addupdate_mapitem((SPTR)HASHLKG(private_sym),
                                             OMP_TGT_MAPTYPE_TARGET_PARAM |
                                                 OMP_TGT_MAPTYPE_TO |
                                                 OMP_TGT_MAPTYPE_FROM);
  else
    ompaccel_tinfo_current_addupdate_mapitem((SPTR)HASHLKG(shared_sym),
                                             OMP_TGT_MAPTYPE_TARGET_PARAM |
                                                 OMP_TGT_MAPTYPE_TO |
                                                 OMP_TGT_MAPTYPE_FROM);
}

void
ompaccel_tinfo_current_addupdate_mapitem(SPTR host_symbol, int map_type)
{
  SPTR midsptr;
  if (current_tinfo == nullptr)
    ompaccel_msg_interr("XXX", "Current target info is not found\n");

  // check whether it is allocatable or not
  if (SCG(host_symbol) == SC_BASED) {
    /* if it is in data mode, we should keep midnum at active symbols*/
    if (current_tinfo->mode == mode_target_data_enter_region ||
        current_tinfo->mode == mode_target_data_exit_region ||
        current_tinfo->mode == mode_target_data_region) {
      midsptr = (SPTR)MIDNUMG(host_symbol);
      if (!tinfo_update_maptype(current_tinfo->symbols,
                                current_tinfo->n_symbols, midsptr, map_type))
        ompaccel_tinfo_current_add_sym(midsptr, NOSYM, map_type);
    }
    /* Main argument will be kept at passive */
    if (!tinfo_update_maptype(current_tinfo->quiet_symbols,
                              current_tinfo->n_quiet_symbols, host_symbol,
                              map_type))
      ompaccel_tinfo_current_add_sym(host_symbol, NOSYM, map_type);
  } else {
    if (!tinfo_update_maptype(current_tinfo->symbols, current_tinfo->n_symbols,
                              host_symbol, map_type))
      ompaccel_tinfo_current_add_sym(host_symbol, NOSYM, map_type);
  }
}

void
ompaccel_tinfo_current_add_sym(SPTR host_symbol, SPTR device_symbol,
                               int map_type)
{
  if ((MIDNUMG(host_symbol) && SCG(host_symbol) == SC_BASED)) {
    NEED((current_tinfo->n_quiet_symbols + 1), current_tinfo->quiet_symbols,
         OMPACCEL_SYM, current_tinfo->sz_quiet_symbols,
         current_tinfo->sz_quiet_symbols * INC_EXP);
    current_tinfo->quiet_symbols[current_tinfo->n_quiet_symbols].host_sym =
        host_symbol;
    current_tinfo->quiet_symbols[current_tinfo->n_quiet_symbols].device_sym =
        device_symbol;
    current_tinfo->quiet_symbols[current_tinfo->n_quiet_symbols].map_type =
        map_type;
    current_tinfo->n_quiet_symbols++;
  } else {
    NEED((current_tinfo->n_symbols + 1), current_tinfo->symbols, OMPACCEL_SYM,
         current_tinfo->sz_symbols, current_tinfo->sz_symbols * INC_EXP);
    current_tinfo->symbols[current_tinfo->n_symbols].host_sym = host_symbol;
    current_tinfo->symbols[current_tinfo->n_symbols].device_sym = device_symbol;
    current_tinfo->symbols[current_tinfo->n_symbols].map_type = map_type;
    current_tinfo->n_symbols++;
  }
}

INLINE static void
dumptargetsym(OMPACCEL_SYM targetsym)
{
  const char *dev_sptr_name, *org_sptr_name;
  if (gbl.dbgfil == nullptr)
    return;

  dev_sptr_name =
      targetsym.device_sym == NOSYM ? "NOSYM" : SYMNAME(targetsym.device_sym);
  org_sptr_name =
      targetsym.host_sym == NOSYM ? "NOSYM" : SYMNAME(targetsym.host_sym);

  fprintf(gbl.dbgfil,
          "\t(org:%d[%s], dev:%d[%s], map-type: ", targetsym.host_sym,
          org_sptr_name, targetsym.device_sym, dev_sptr_name);

  if (targetsym.map_type & OMP_TGT_MAPTYPE_ALWAYS)
    fprintf(gbl.dbgfil, "always ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_TO)
    fprintf(gbl.dbgfil, "to ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_FROM)
    fprintf(gbl.dbgfil, "from ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_DELETE)
    fprintf(gbl.dbgfil, "delete ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
    fprintf(gbl.dbgfil, "ptr_obj ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_TARGET_PARAM)
    fprintf(gbl.dbgfil, "target ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_RETURN_PARAM)
    fprintf(gbl.dbgfil, "return ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_PRIVATE)
    fprintf(gbl.dbgfil, "private ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_LITERAL)
    fprintf(gbl.dbgfil, "literal ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_IMPLICIT)
    fprintf(gbl.dbgfil, "implicit ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_MEMBER_OF)
    fprintf(gbl.dbgfil, "member ");
  if (targetsym.map_type & OMP_TGT_MAPTYPE_NONE)
    fprintf(gbl.dbgfil, "none ");
  fprintf(gbl.dbgfil, " )\n");
}

INLINE static void
dumptargetreduction(OMPACCEL_RED_SYM targetred)
{
  if (gbl.dbgfil == nullptr)
    return;
  switch (targetred.redop) {
  case 1:
    fprintf(gbl.dbgfil, "+:  ");
    break;
  case 2:
    fprintf(gbl.dbgfil, "-:  ");
    break;
  case 3:
    fprintf(gbl.dbgfil, "*:  ");
    break;
  case 346:
    fprintf(gbl.dbgfil, "max:");
    break;
  case 347:
    fprintf(gbl.dbgfil, "min:");
    break;
  case 327:
    fprintf(gbl.dbgfil, "iand:");
    break;
  case 328:
    fprintf(gbl.dbgfil, "ior:");
    break;
  case 329:
    fprintf(gbl.dbgfil, "ieor:");
    break;
  case 14:
    fprintf(gbl.dbgfil, "???:");
    break;
  default:
    break;
  }

  fprintf(gbl.dbgfil, "\t(shared:%d[%s], private:%d[%s] \n",
          targetred.shared_sym, SYMNAME(targetred.shared_sym),
          targetred.private_sym, SYMNAME(targetred.private_sym));
}

void
dumpomptarget(OMPACCEL_TINFO *tinfo)
{
  if (tinfo == nullptr)
    return;
  if (gbl.dbgfil == nullptr)
    return;

  switch (tinfo->mode) {
  default:
    break;
  case mode_none_target:
    fprintf(gbl.dbgfil, " <mode none>");
    break;
  case mode_target:
    fprintf(gbl.dbgfil, " <target>");
    break;
  case mode_target_teams:
    fprintf(gbl.dbgfil, " <target teams>");
    break;
  case mode_target_parallel_for_simd:
    fprintf(gbl.dbgfil, " <target parallel for simd>");
    break;
  case mode_target_parallel_for:
    fprintf(gbl.dbgfil, " <target parallel for>");
    break;
  case mode_target_teams_distribute_parallel_for:
    fprintf(gbl.dbgfil, " <target teams distribute parallel for>");
    break;
  case mode_target_teams_distribute_parallel_for_simd:
    fprintf(gbl.dbgfil, " <target teams distribute parallel for simd>");
    break;
  case mode_target_teams_distribute:
    fprintf(gbl.dbgfil, " <target teams distribute >");
    break;
  case mode_target_data_region:
    fprintf(gbl.dbgfil, " <target data>");
    break;
  case mode_target_data_enter_region:
    fprintf(gbl.dbgfil, " <target data enter>");
    break;
  case mode_target_data_exit_region:
    fprintf(gbl.dbgfil, " <target data exit>");
    break;
  }
  fprintf(gbl.dbgfil, " \n");
  //}

  if ((tinfo->mode != mode_target_data_region) &&
      (tinfo->mode != mode_target_data_enter_region) &&
      (tinfo->mode != mode_target_data_exit_region)) {
    if (OMPACCFUNCKERNELG(tinfo->func_sptr))
      fprintf(gbl.dbgfil, " (__global__) ");
    else if (OMPACCFUNCDEVG(tinfo->func_sptr))
      fprintf(gbl.dbgfil, " (__device__) ");
    else
      fprintf(gbl.dbgfil, " ??? ");
    fprintf(gbl.dbgfil, "%s\t sptr: %d \n", SYMNAME(tinfo->func_sptr),
            tinfo->func_sptr);
  }

  fprintf(gbl.dbgfil, " ** Active Symbols ** \n");
  for (int j = 0; j < tinfo->n_symbols; ++j) {
    dumptargetsym(tinfo->symbols[j]);
  }
  fprintf(gbl.dbgfil, " ** Passive Symbols ** \n");
  for (int j = 0; j < tinfo->n_quiet_symbols; ++j) {
    dumptargetsym(tinfo->quiet_symbols[j]);
  }
  fprintf(gbl.dbgfil, " ** Reductions ** \n");
  for (int j = 0; j < tinfo->n_reduction_symbols; ++j) {
    dumptargetreduction(tinfo->reduction_symbols[j]);
  }
  fprintf(gbl.dbgfil, "\n");
}

void
dumpomptargets()
{
  int i;
  if (gbl.dbgfil == NULL)
    return;
  fprintf(gbl.dbgfil, "------------OpenMP Target Regions ---------------\n");
  for (i = 0; i < num_tinfos; ++i) {
    dumpomptarget(tinfos[i]);
  }
}

void
dumpomptargetsymbols()
{
  int i, l, u;
  l = stb.firstusym;
  u = stb.stg_avail - 1;
  if (u >= stb.stg_avail)
    u = stb.stg_avail - 1;
  for (i = l; i <= u; ++i) {
    if (OMPACCDEVSYMG(i))
      fprintf(gbl.dbgfil, "(sym) sptr:%d [%s]\n", i, SYMNAME(i));
    if (OMPACCFUNCDEVG(i))
      fprintf(gbl.dbgfil, "(func) sptr:%d [%s]\n", i, SYMNAME(i));
    if (OMPACCFUNCKERNELG(i))
      fprintf(gbl.dbgfil, "(kernel) sptr:%d [%s]\n", i, SYMNAME(i));
  }
}

void
dumptargetsymbols(OMPACCEL_SYM *targetsyms, int n)
{
  for (int i = 0; i < n; ++i) {
    dumptargetsym(targetsyms[i]);
  }
}

void
ompaccel_msg_interr(const char *id, const char *message)
{
  interr(message, MSGOMPACCEL, ERR_Fatal);
}

void
ompaccel_msg_info(const char *id, const char *message)
{
  ccff_info(MSGOMPACCEL, id, gbl.findex, gbl.lineno, message, NULL);
}

bool
ompaccel_is_tgt_registered()
{
  return isOmpaccelRegistered;
}

void
ompaccel_register_tgt()
{
  isOmpaccelRegistered = true;
}

void
ompaccel_emit_tgt_register()
{
  int ilix;
  SPTR sptrFn;
  const char *name = "ompaccel.register";
  sptrFn = mk_ompaccel_function(name, 0, NULL, false);
  CONSTRUCTORP(sptrFn, 1);
  TEXTSTARTUPP(sptrFn, 1);
  PRIORITYP(sptrFn, 65535 /* LLVM_DEFAULT_PRIORITY */);
  cr_block();
  ilix = ll_make_tgt_register_lib();
  iltb.callfg = 1;
  chk_block(ilix);
  wr_block();
  mk_ompaccel_function_end(sptrFn);
}

SPTR
ompaccel_nvvm_emit_reduce(OMPACCEL_RED_SYM *ReductionItems, int NumReductions)
{
  int ili, bili, rili;
  SPTR sptrFn, sptrRhs, sptrReduceData, func_params[2];
  DTYPE dtypeReductionItem, dtypeReduceData;
  int nmeReduceData, nmeRhs;
  char name[30];

  /* Generate function parameters */
  dtypeReduceData = get_type(2, TY_PTR, DT_ANY);
  sptrReduceData = func_params[0] =
      mk_ompaccel_addsymbol(".reducedata", dtypeReduceData, SC_DUMMY, ST_ARRAY);
  sptrRhs = func_params[1] =
      mk_ompaccel_addsymbol(".rhs", dtypeReduceData, SC_DUMMY, ST_VAR);

  /* Generate function symbol */
  sprintf(name, "%s%d", "ompaccel_reduction", reductionFunctionCounter++);
  sptrFn = mk_ompaccel_function(name, 2, func_params, true);
  cr_block();

  for (int i = 0; i < NumReductions; ++i) {
    dtypeReductionItem = DTYPEG(ReductionItems[i].shared_sym);

    bili = mk_ompaccel_ldsptr(sptrReduceData);
    rili = mk_ompaccel_ldsptr(sptrRhs);

    if (i != 0) {
      bili = mk_ompaccel_add(bili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
      rili = mk_ompaccel_add(rili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
    }

    nmeReduceData =
        add_arrnme(NT_IND, SPTR_NULL, addnme(NT_VAR, sptrReduceData, 0, 0), 0,
                   ad_icon(i), FALSE);
    nmeRhs =
        add_arrnme(NT_IND, SPTR_NULL,
                   addnme(NT_IND, SPTR_NULL, addnme(NT_VAR, sptrRhs, 0, 0), 0),
                   i, ad_icon(i), FALSE);

    bili =
        mk_ompaccel_load(bili, DT_ADDR, addnme(NT_VAR, sptrReduceData, 0, 0));
    rili = mk_ompaccel_load(rili, DT_ADDR, addnme(NT_VAR, sptrRhs, 0, 0));

    rili = mk_ompaccel_load(rili, dtypeReductionItem, nmeRhs);
    ili = mk_ompaccel_load(bili, dtypeReductionItem, nmeReduceData);

    ili = mk_reduction_op(ReductionItems[i].redop, ili, dtypeReductionItem,
                          rili, dtypeReductionItem);

    ili = mk_ompaccel_store(ili, dtypeReductionItem, nmeReduceData, bili);
    chk_block(ili);
  }
  wr_block();
  mk_ompaccel_function_end(sptrFn);
  return sptrFn;
}

SPTR
ompaccel_nvvm_emit_shuffle_reduce(OMPACCEL_RED_SYM *ReductionItems,
                                  int NumReductions, SPTR sptrFnReduce)
{
  int ili, rili, bili;
  SPTR sptrFn, sptrRhs, sptrReduceData, sptrShuffleReturn, sptrLaneOffset,
      func_params[4];
  DTYPE dtypeReductionItem, dtypeReduceData, dtypeRHS;
  int nmeReduceData, nmeRhs, params[2];
  char name[30];
  DTYPE params_dtypes[2] = {DT_ADDR, DT_ADDR};

  /* Generate function parameters */
  dtypeReduceData = get_type(2, TY_PTR, DT_ANY);
  sptrReduceData = func_params[0] = mk_ompaccel_addsymbol(
      ".reducedata2", dtypeReduceData, SC_DUMMY, ST_ARRAY);
  func_params[1] = mk_ompaccel_addsymbol(".laneid", DT_SINT, SC_DUMMY, ST_VAR);
  sptrLaneOffset = func_params[2] =
      mk_ompaccel_addsymbol(".laneoffset", DT_SINT, SC_DUMMY, ST_VAR);
  func_params[3] =
      mk_ompaccel_addsymbol(".shortcircuit", DT_SINT, SC_DUMMY, ST_VAR);
  PASSBYVALP(func_params[1], 1);
  PASSBYVALP(func_params[2], 1);
  PASSBYVALP(func_params[3], 1);

  /* Generate function symbol */
  sprintf(name, "%s%d", "ompaccel_shufflereduce", reductionFunctionCounter++);
  sptrFn = mk_ompaccel_function(name, 4, func_params, true);
  cr_block();

  dtypeRHS = mk_ompaccel_array_dtype(dtypeReduceData, NumReductions);
  sptrRhs = mk_ompaccel_addsymbol(".rhs", dtypeRHS, SC_LOCAL, ST_ARRAY);

  for (int i = 0; i < NumReductions; ++i) {

    dtypeReductionItem = DTYPEG(ReductionItems[i].shared_sym);
    sptrShuffleReturn =
        mk_ompaccel_getnewccsym('r', i, dtypeReductionItem, SC_LOCAL, ST_VAR);

    bili = mk_ompaccel_ldsptr(sptrReduceData);
    rili = mk_address(sptrRhs);

    nmeReduceData =
        add_arrnme(NT_IND, SPTR_NULL, addnme(NT_VAR, sptrReduceData, 0, 0), i,
                   ad_icon(i), FALSE);

    if (i != 0) {
      rili = mk_ompaccel_add(rili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
      bili = mk_ompaccel_add(bili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
    }

    ili = mk_ompaccel_load(bili, DT_ADDR, nmeReduceData);
    ili = mk_ompaccel_load(ili, dtypeReductionItem, nmeReduceData);

    if (dtypeReductionItem == DT_DBLE)
      ili =
          ll_make_kmpc_shuffle(ili, mk_ompaccel_ldsptr(func_params[2]),
                               ad_icon(size_of(dtypeReductionItem) * 8), true);
    else
      ili =
          ll_make_kmpc_shuffle(ili, mk_ompaccel_ldsptr(func_params[2]),
                               ad_icon(size_of(dtypeReductionItem) * 8), false);

    ili = mk_ompaccel_store(ili, dtypeReductionItem,
                            addnme(NT_VAR, sptrShuffleReturn, 0, 0),
                            ad_acon(sptrShuffleReturn, 0));
    chk_block(ili);

    nmeRhs = add_arrnme(NT_ARR, NME_NULL, addnme(NT_VAR, sptrRhs, 0, 0), i,
                        ad_icon(i), FALSE);
    ili =
        mk_ompaccel_store(ad_acon(sptrShuffleReturn, 0), DT_ADDR, nmeRhs, rili);

    chk_block(ili);
  }

  params[0] = mk_address(sptrRhs);
  params[1] = mk_address(sptrReduceData);

  /* Call reduce function */
  ili = mk_function_call(DT_NONE, 2, params_dtypes, params, sptrFnReduce);

  /* Write to block */
  iltb.callfg = 1;
  chk_block(ili);

  wr_block();
  mk_ompaccel_function_end(sptrFn);

  return sptrFn;
}

/**
   \brief This function emits code that gathers reduce_data from the first lane
   of every active warp to lanes in the first warp.
 */
SPTR
ompaccel_nvvm_emit_inter_warp_copy(OMPACCEL_RED_SYM *ReductionItems,
                                   int NumReductions)
{
  int ili, rili;
  SPTR sptrFn, sptrReduceData, sptrWarpNum, sptrShmem, sptrWarpId,
      sptrMasterWarp, sptrRedItem, sptrRedItemAddress, func_params[2];
  SPTR lFirstLane, lBarrier, lFirstWarp, lFinalBarrier;
  int nmeShmem;
  DTYPE dtypeReductionItem;
  char name[30];

  sprintf(name, "%s%d", "ompaccel_InterWarpCopy", reductionFunctionCounter++);
  sptrReduceData = func_params[0] = mk_ompaccel_addsymbol(
      ".reduceData", mk_ompaccel_array_dtype(DT_INT8, NumReductions), SC_DUMMY,
      ST_ARRAY);
  sptrWarpNum = func_params[1] =
      mk_ompaccel_addsymbol(".warpNum", DT_INT, SC_DUMMY, ST_VAR);
  PASSBYVALP(sptrWarpNum, 1);
  /* Generate function symbol, Create a block */
  sptrFn = mk_ompaccel_function(name, 2, func_params, true);
  cr_block();

  sptrShmem = mk_ompaccel_addsymbol(
      "ompaccelshmem", mk_ompaccel_array_dtype(DT_INT8, NVVM_WARPSIZE),
      SC_EXTERN, ST_ARRAY);
  OMPACCSHMEMP(sptrShmem, true);
  SYMLKP(sptrShmem, gbl.externs);
  gbl.externs = sptrShmem;

  /* MasterWarp */
  sptrMasterWarp =
      mk_ompaccel_addsymbol(".masterwarp", DT_INT, SC_LOCAL, ST_VAR);
  ili = ompaccel_nvvm_get(threadIdX);
  ili = mk_ompaccel_iand(ili, ad_icon(31));
  ili = mk_ompaccel_stsptr(ili, sptrMasterWarp);

  chk_block(ili);

  /* MasterWarp */
  sptrWarpId = mk_ompaccel_addsymbol(".warpid", DT_INT, SC_LOCAL, ST_VAR);
  ili = ompaccel_nvvm_get(threadIdX);
  ili = mk_ompaccel_shift(ili, DT_UINT, ad_icon(5), DT_UINT);
  ili = mk_ompaccel_stsptr(ili, sptrWarpId);

  chk_block(ili);
  iltb.callfg = 1;
  wr_block();

  sptrRedItem =
      mk_ompaccel_addsymbol(".reductionitem", DT_ADDR, SC_LOCAL, ST_VAR);
  sptrRedItemAddress =
      mk_ompaccel_addsymbol(".reductionitemaddr", DT_ADDR, SC_LOCAL, ST_VAR);

  for (int i = 0; i < NumReductions; ++i) {
    cr_block();
    dtypeReductionItem = DTYPEG(ReductionItems[i].shared_sym);
    rili = mk_ompaccel_ldsptr(sptrReduceData);
    if (i != 0) {
      rili = mk_ompaccel_add(rili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
    }
    rili =
        mk_ompaccel_load(rili, DT_ADDR, addnme(NT_VAR, sptrReduceData, 0, 0));

    ili =
        mk_ompaccel_store(rili, DT_ADDR, addnme(NT_VAR, sptrRedItem, 0, (INT)0),
                          mk_address(sptrRedItem));
    chk_block(ili);
    ili = mk_ompaccel_store(rili, DT_ADDR,
                            addnme(NT_VAR, sptrRedItemAddress, 0, (INT)0),
                            mk_address(sptrRedItemAddress));
    chk_block(ili);

    ili = mk_ompaccel_ldsptr(sptrMasterWarp);
    lFirstLane = getlab();
    lBarrier = getlab();
    ili = ad3ili(IL_ICJMPZ, ili, CC_NE, lBarrier);
    RFCNTI(lFirstLane);
    chk_block(ili);
    iltb.callfg = 1;
    wr_block();

    cr_block();
    exp_label(lFirstLane);

    rili = ad1ili(IL_IKMV, mk_ompaccel_ldsptr(sptrWarpId));
    ili = mk_ompaccel_mul(rili, DT_INT8, ad_kconi(8), DT_INT8);
    ili = ad1ili(IL_KAMV, ili);
    ili = mk_ompaccel_add(mk_address(sptrShmem), DT_ADDR, ili, DT_ADDR);
    nmeShmem = add_arrnme(NT_ARR, NME_NULL, addnme(NT_VAR, sptrShmem, 0, 0), 0,
                          rili, FALSE);

    rili = mk_ompaccel_ldsptr(sptrRedItem);
    // todo ompaccel more
    if (dtypeReductionItem == DT_DBLE) {
      rili = mk_ompaccel_load(rili, DT_DBLE, addnme(NT_VAR, sptrRedItem, 0, 0));
    } else if (dtypeReductionItem == DT_INT) {
      rili = mk_ompaccel_ld(rili, addnme(NT_IND, SPTR_NULL,
                                         addnme(NT_VAR, sptrRedItem, 0, 0), 0));
      rili = ad1ili(IL_FLOAT, rili);
    } else if (dtypeReductionItem == DT_FLOAT) {
      rili =
          mk_ompaccel_load(rili, DT_FLOAT, addnme(NT_VAR, sptrRedItem, 0, 0));
      rili = ad1ili(IL_DBLE, rili);
    }
    ili = mk_ompaccel_store(rili, DT_DBLE, nmeShmem, ili);
    chk_block(ili);
    iltb.callfg = 1;
    wr_block();

    /* Sync */
    cr_block();
    RFCNTI(lBarrier);
    exp_label(lBarrier);
    ili = ompaccel_nvvm_mk_barrier(CTA_BARRIER);
    iltb.callfg = 1;
    chk_block(ili);
    wr_block();

    cr_block();
    ili = mk_ompaccel_ldsptr(sptrWarpNum);
    rili = ompaccel_nvvm_get(threadIdX);
    ili = mk_ompaccel_compare(rili, DT_INT, ili, DT_INT, CC_GE);
    lFirstWarp = getlab();
    lFinalBarrier = getlab();
    ili = ad3ili(IL_ICJMPZ, ili, CC_NE, lFinalBarrier);
    chk_block(ili);
    iltb.callfg = 1;
    wr_block();

    cr_block();
    RFCNTI(lFirstWarp);
    exp_label(lFirstWarp);
    rili = ad1ili(IL_IKMV, ompaccel_nvvm_get(threadIdX));
    ili = mk_ompaccel_mul(rili, DT_INT8, ad_kconi(8), DT_INT8);
    ili = ad1ili(IL_KAMV, ili);
    ili = mk_ompaccel_add(mk_address(sptrShmem), DT_ADDR, ili, DT_ADDR);
    nmeShmem = add_arrnme(NT_ARR, NME_NULL, addnme(NT_VAR, sptrShmem, 0, 0), 0,
                          rili, FALSE);

    ili = mk_ompaccel_load(ili, DT_DBLE, nmeShmem);
    rili = mk_ompaccel_ldsptr(sptrRedItemAddress);

    if (dtypeReductionItem == DT_DBLE) {
      ili = mk_ompaccel_store(
          ili, DT_DBLE,
          addnme(NT_IND, NME_NULL, addnme(NT_VAR, sptrRedItemAddress, 0, 0), 0),
          rili);
    } else if (dtypeReductionItem == DT_INT) {
      ili = ad1ili(IL_DFIX, ili);
      ili = mk_ompaccel_store(
          ili, DT_NONE,
          addnme(NT_IND, NME_NULL, addnme(NT_VAR, sptrRedItemAddress, 0, 0), 0),
          rili);
    } else if (dtypeReductionItem == DT_FLOAT) {
      ili = ad1ili(IL_SNGL, ili);
      ili = mk_ompaccel_store(
          ili, DT_FLOAT,
          addnme(NT_IND, NME_NULL, addnme(NT_VAR, sptrRedItemAddress, 0, 0), 0),
          rili);
    }
    chk_block(ili);
    iltb.callfg = 1;
    wr_block();

    /* Sync */
    cr_block();
    RFCNTI(lFinalBarrier);
    exp_label(lFinalBarrier);
    ili = ompaccel_nvvm_mk_barrier(CTA_BARRIER);
    iltb.callfg = 1;
    chk_block(ili);
    wr_block();
  }
  /* Finalize the function */
  mk_ompaccel_function_end(sptrFn);
  return sptrFn;
}

/* Expander - OpenMP Accelerator Model */
void
exp_ompaccel_bpar(ILM *ilmp, int curilm, SPTR uplevel_sptr, SPTR scopeSptr,
                  int(incrOutlinedCnt()))
{
  int ili, outlinedCnt;
  SPTR sptr;
  if (flg.opt != 0) {
    wr_block();
    cr_block();
  }
  if (!XBIT(232, 0x1)) {
    ll_rewrite_ilms(-1, curilm, 0);
    return;
  }
  outlinedCnt = incrOutlinedCnt();
  BIH_FT(expb.curbih) = TRUE;
  BIH_QJSR(expb.curbih) = TRUE;
  BIH_NOMERGE(expb.curbih) = TRUE;
  if (gbl.outlined)
    expb.sc = SC_PRIVATE;
  if (outlinedCnt == 1) {
    sptr = ll_make_outlined_ompaccel_func(uplevel_sptr, scopeSptr, FALSE);

    if (!PARENCLFUNCG(scopeSptr))
      PARENCLFUNCP(scopeSptr, sptr);
    ll_write_ilm_header(sptr, curilm);

    ili = ompaccel_nvvm_get(threadIdX);
    ili = ll_make_kmpc_spmd_kernel_init(ili);
    iltb.callfg = 1;
    chk_block(ili);

    ili = ll_make_outlined_ompaccel_call(gbl.ompoutlinedfunc, sptr);
    iltb.callfg = 1;
    chk_block(ili);
    gbl.ompoutlinedfunc = sptr;

  } else if (outlinedCnt > 1) {
    ll_rewrite_ilms(-1, curilm, 0);
  }
}

void
exp_ompaccel_epar(ILM *ilmp, int curilm, int outlinedCnt,
                  int(decrOutlinedCnt()))
{
  if (XBIT(232, 0x1)) {
    if (flg.opt != 0) {
      wr_block();
      cr_block();
    }

    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    outlinedCnt = decrOutlinedCnt();
  }
  if (outlinedCnt >= 1)
    ll_rewrite_ilms(-1, curilm, 0);

  if (gbl.outlined)
    expb.sc = SC_AUTO;
  ccff_info(MSGOPENMP, "OMP002", gbl.findex, gbl.lineno,
            "Parallel region terminated", NULL);
}

void
exp_ompaccel_eteams(ILM *ilmp, int curilm, int outlinedCnt,
                    int(decrOutlinedCnt()))
{
  if (XBIT(232, 0x1)) {
    if (outlinedCnt == 1) {
      ilm_outlined_pad_ilm(curilm);
    }
    outlinedCnt = decrOutlinedCnt();
  }
  if (outlinedCnt >= 1)
    ll_rewrite_ilms(-1, curilm, 0);

  if (gbl.outlined)
    expb.sc = SC_AUTO;
  ccff_info(MSGOPENMP, "OMP023", gbl.findex, gbl.lineno,
            "Teams region terminated", NULL);
}

void
exp_ompaccel_mploopfini(ILM *ilmp, int curilm, int outlinedCnt)
{
  int ili;
  if (outlinedCnt >= 1)
    return;
  const int sched = mp_sched_to_kmpc_sched(ILM_OPND(ilmp, 2));
  if (sched == KMP_ORD_STATIC || sched == KMP_ORD_DYNAMIC_CHUNKED) {
    ili = ll_make_kmpc_dispatch_fini((DTYPE)ILM_OPND(ilmp, 1));
    iltb.callfg = 1;
    chk_block(ili);
  } else if (sched == KMP_SCH_STATIC || sched == KMP_SCH_STATIC_CHUNKED) {
    ili = ll_make_kmpc_for_static_fini();
    iltb.callfg = 1;
    chk_block(ili);
  }
}

void
exp_ompaccel_mploop(ILM *ilmp, int curilm)
{
  SPTR nlower, nupper, nstride;
  int sched, ili;
  loop_args_t loop_args;
#if LLVM_YKT
  /* frontend generates two MPLOOP ILM, one for distribute, other for parallel
   * If it is combined construct like ttdpf, I don't need to do something
   * special for distribute I need to pass different scheduling type to device
   * runtime.
   */
  if (mp_sched_to_kmpc_sched(ILM_OPND(ilmp, 7)) == KMP_DISTRIBUTE_STATIC) {
    if ((ompaccel_tinfo_current_target_mode() ==
             mode_target_teams_distribute_parallel_for ||
         ompaccel_tinfo_current_target_mode() ==
             mode_target_teams_distribute_parallel_for_simd))
      return;
  }
#endif
  nlower = ILM_SymOPND(ilmp, 1);
  nupper = ILM_SymOPND(ilmp, 2);
  nstride = ILM_SymOPND(ilmp, 3);
  if (!XBIT(183, 0x100000)) {
    nlower = (SPTR)getccsym_copy(nlower);   // ???
    nupper = (SPTR)getccsym_copy(nupper);   // ???
    nstride = (SPTR)getccsym_copy(nstride); // ???
    SCP(nlower, SC_PRIVATE);
    ENCLFUNCP(nlower, GBL_CURRFUNC);
    ENCLFUNCP(nupper, GBL_CURRFUNC);
    ENCLFUNCP(nstride, GBL_CURRFUNC);
    exp_add_copy(nlower, ILM_SymOPND(ilmp, 1));
    exp_add_copy(nupper, ILM_SymOPND(ilmp, 2));
    exp_add_copy(nstride, ILM_SymOPND(ilmp, 3));
  }
  loop_args.lower = nlower;
  loop_args.upper = nupper;
  loop_args.stride = nstride;
  loop_args.chunk = ILM_SymOPND(ilmp, 4);
  loop_args.last = ILM_SymOPND(ilmp, 5);
  loop_args.dtype = (DTYPE)ILM_OPND(ilmp, 6); // ???
  loop_args.sched = (kmpc_sched_e)ILM_OPND(ilmp, 7);
  sched = mp_sched_to_kmpc_sched(loop_args.sched);
  switch (sched) {
  case KMP_SCH_STATIC:
  case KMP_SCH_STATIC_CHUNKED:
    if ((ILM_OPND(ilmp, 7) & 0xff00) == MP_SCH_CHUNK_1) {
      ccff_info(MSGOPENMP, "OMP014", gbl.findex, gbl.lineno,
                "Parallel loop activated with %schedule schedule",
                "schedule=%s", "static cyclic", NULL);
    }
    FLANG_FALLTHROUGH;
  case KMP_DISTRIBUTE_STATIC_CHUNKED:
  case KMP_DISTRIBUTE_STATIC:
    ili = ll_make_kmpc_for_static_init_simple_spmd(&loop_args, sched);
    break;
  default:
    ili = ll_make_kmpc_dispatch_init(&loop_args);
  }

  iltb.callfg = 1;
  chk_block(ili);
  BIH_NOMERGE(expb.curbih) = TRUE;
  if (!XBIT(183, 0x100000)) {
    exp_add_copy(ILM_SymOPND(ilmp, 1), nlower);
    exp_add_copy(ILM_SymOPND(ilmp, 2), nupper);
    exp_add_copy(ILM_SymOPND(ilmp, 3), nstride);
  }

  /* constant propagation stop when it sees function call. We may have some
   * stride that needs to propagate for computation of tripcount. */
  if (flg.opt != 0) {
    wr_block();
    cr_block();
  }
}

void
exp_ompaccel_btarget(ILM *ilmp, int curilm, SPTR uplevel_sptr, SPTR scopeSptr,
                     int(incrOutlinedCnt()), SPTR *targetfunc_sptr,
                     int *isTargetDevice)
{
  SPTR sptr;
  /* lexically nested begin parallel */
  int outlinedCnt = incrOutlinedCnt();
  if (outlinedCnt > 1) {
    ll_rewrite_ilms(-1, curilm, 0);
    return;
  }
  ompaccel_symreplacer(false);
  if (flg.opt != 0) {
    wr_block();
    cr_block();
  }

  BIH_FT(expb.curbih) = TRUE;
  BIH_QJSR(expb.curbih) = TRUE;
  BIH_NOMERGE(expb.curbih) = TRUE;
  if (outlinedCnt == 1) {
    /* inomptarget used to figure out whether other directives, statements are
     * in target region or not */
    gbl.ompaccel_intarget = true;
    /* Outline function, create sptr as ptx kernel, duplicate all the sptrs*/
    sptr = ll_make_outlined_ompaccel_func(uplevel_sptr, scopeSptr, TRUE);
    /* set global outlined function with the latest */
    gbl.ompoutlinedfunc = sptr;

    if (!PARENCLFUNCG(scopeSptr))
      PARENCLFUNCP(scopeSptr, sptr);
    ll_write_ilm_header(sptr, curilm);
  }
  ccff_info(MSGOPENMP, "OMP020", gbl.findex, gbl.lineno,
            "Target region activated for offload", NULL);
  *targetfunc_sptr = sptr;
  *isTargetDevice = ILI_OF(ILM_OPND(ilmp, 1));
  return;
}

static void
exp_ompaccel_ereduction(ILM *ilmp, int curilm)
{
  int ili;
  cr_block();
  ili = ll_make_kmpc_nvptx_end_reduce_nowait();

  iltb.callfg = 1;
  chk_block(ili);
  wr_block();
}

void
exp_ompaccel_etarget(ILM *ilmp, int curilm, SPTR targetfunc_sptr,
                     int outlinedCnt, SPTR uplevel_sptr, int(decrOutlinedCnt()))
{
  int ili;
  if (outlinedCnt == 1) {
    ilm_outlined_pad_ilm(curilm);
  }
  outlinedCnt = decrOutlinedCnt();
  if (outlinedCnt >= 1) {
    ll_rewrite_ilms(-1, curilm, 0);
    return;
  }
  if (gbl.outlined)
    expb.sc = SC_AUTO;

  if (ompaccel_tinfo_current_target_mode() == mode_target) {
    ili = ll_make_tgt_target(gbl.ompoutlinedfunc, OMPACCEL_DEFAULT_DEVICEID,
                             uplevel_sptr);
  } else if (ompaccel_tinfo_current_target_mode() == mode_target_parallel_for ||
      ompaccel_tinfo_current_target_mode() ==
          mode_target_parallel_for_simd) {
    // Create kernel with single team.
    ili = ll_make_tgt_target_teams(
        gbl.ompoutlinedfunc, OMPACCEL_DEFAULT_DEVICEID, uplevel_sptr, 1, 0);
  } else {
    ili = ll_make_tgt_target_teams(
        gbl.ompoutlinedfunc, OMPACCEL_DEFAULT_DEVICEID, uplevel_sptr, 0, 0);
  }

  iltb.callfg = 1;
  chk_block(ili);

  gbl.ompaccel_intarget = false;

  ccff_info(MSGOPENMP, "OMP021", gbl.findex, gbl.lineno,
            "Target region terminated", NULL);
}

void
exp_ompaccel_reduction(ILM *ilmp, int curilm)
{
  int ili, bili, nmeReduceData, sizeRed = 0;
  SPTR lAssignReduction, sptrReduceData, sptrReductionItem;
  DTYPE dtypeReduceData, dtypeReductionItem;
  dtypeReduceData = mk_ompaccel_array_dtype(
      get_type(2, TY_PTR, DT_ANY),
      ompaccel_tinfo_current_get()->n_reduction_symbols);
  sptrReduceData =
      mk_ompaccel_addsymbol(".reduceData", dtypeReduceData, SC_LOCAL, ST_ARRAY);

  cr_block();
  for (int i = 0; i < ompaccel_tinfo_current_get()->n_reduction_symbols; ++i) {
    sptrReductionItem =
        ompaccel_tinfo_current_get()->reduction_symbols[i].shared_sym;
    dtypeReductionItem = DTYPEG(sptrReductionItem);

    ili = mk_address(sptrReductionItem);
    nmeReduceData =
        add_arrnme(NT_ARR, NME_NULL, addnme(NT_VAR, sptrReduceData, 0, 0), i,
                   ad_icon(i), FALSE);

    bili = mk_address(sptrReduceData);
    if (i != 0)
      bili = mk_ompaccel_add(bili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);

    ili = mk_ompaccel_store(ili, DT_ADDR, nmeReduceData, bili);
    chk_block(ili);
  }
  wr_block();

  cr_block();
  ili = ll_make_kmpc_nvptx_parallel_reduce_nowait_simple_spmd(
      ad_icon(ompaccel_tinfo_current_get()->n_reduction_symbols),
      ad_icon(sizeRed), mk_address(sptrReduceData),
      ompaccel_tinfo_current_get()->reduction_funcs.shuffleFn,
      ompaccel_tinfo_current_get()->reduction_funcs.interWarpCopy);
  iltb.callfg = 1;
  chk_block(ili);
  wr_block();

  exp_ompaccel_ereduction(ilmp, curilm);

  lAssignReduction = getlab();
  RFCNTI(lAssignReduction);

  ili = ompaccel_nvvm_get(threadIdX);
  ili = mk_ompaccel_compare(ili, DT_INT, ad_icon(0), DT_INT, CC_NE);
  ili = ad3ili(IL_ICJMPZ, ili, CC_NE, lAssignReduction);
  chk_block(ili);

  // Load reduced items to the origina laddress
  for (int i = 0; i < ompaccel_tinfo_current_get()->n_reduction_symbols; ++i) {
    bili = mk_address(sptrReduceData);
    sptrReductionItem =
        ompaccel_tinfo_current_get()->reduction_symbols[i].private_sym;
    dtypeReductionItem = DTYPEG(sptrReductionItem);

    if (i != 0) {
      bili = mk_ompaccel_add(bili, DT_ADDR, ad_aconi(i * size_of(DT_ADDR)),
                             DT_ADDR);
    }

    bili = mk_ompaccel_load(bili, DT_ADDR, nmeReduceData);
    bili = mk_ompaccel_load(bili, dtypeReductionItem, nmeReduceData);

    ili = mk_ompaccel_ldsptr(sptrReductionItem);

    switch (ompaccel_tinfo_current_get()->reduction_symbols[i].redop) {
    case 1:
    case 2:
      ili = mk_ompaccel_add(ili, dtypeReductionItem, bili, dtypeReductionItem);
      ili = mk_ompaccel_store(ili, dtypeReductionItem,
                              addnme(NT_VAR, sptrReductionItem, 0, 0),
                              mk_address(sptrReductionItem));
      break;
    }

    chk_block(ili);
  }
  wr_block();
  cr_block();
  exp_label(lAssignReduction);
}

void
exp_ompaccel_bteams(ILM *ilmp, int curilm, int outlinedCnt, SPTR uplevel_sptr,
                    SPTR scopeSptr, int(incrOutlinedCnt()))
{
  int ili;
  SPTR sptr;
  if (flg.opt != 0) {
    wr_block();
    cr_block();
  }

  if (flg.omptarget) {
    ll_rewrite_ilms(-1, curilm, 0);
    return;
  }

  if (XBIT(232, 0x1)) {
    outlinedCnt = incrOutlinedCnt();
  }
  BIH_FT(expb.curbih) = TRUE;
  BIH_QJSR(expb.curbih) = TRUE;
  BIH_NOMERGE(expb.curbih) = TRUE;
  if (gbl.outlined)
    expb.sc = SC_PRIVATE;
  if (outlinedCnt == 1) {
    if (flg.omptarget)
      sptr = ll_make_outlined_ompaccel_func(uplevel_sptr, scopeSptr, FALSE);
    else
      sptr = ll_make_outlined_func(uplevel_sptr, scopeSptr);
    if (!PARENCLFUNCG(scopeSptr))
      PARENCLFUNCP(scopeSptr, sptr);
    ll_write_ilm_header(sptr, curilm);
    if (flg.omptarget) {
      ili = ompaccel_nvvm_get(threadIdX);
      ili = ll_make_kmpc_spmd_kernel_init(ili);
      iltb.callfg = 1;
      chk_block(ili);
      ili = ll_make_outlined_ompaccel_call(gbl.ompoutlinedfunc, sptr);
      iltb.callfg = 1;
      chk_block(ili);
      gbl.ompoutlinedfunc = sptr;
      return;
    }
    ccff_info(MSGOPENMP, "OMP022", gbl.findex, gbl.lineno,
              "Teams region activated", NULL);

  } else if (outlinedCnt > 1) {
    ll_rewrite_ilms(-1, curilm, 0);
  }
}
void
exp_ompaccel_map(ILM *ilmp, int curilm, int outlinedCnt)
{
  int label = 0, argilm;
  SPTR sptr = SPTR_NULL;
  if (outlinedCnt >= 2)
    return;
  argilm = ILM_OPND(ilmp, 1);
  ILM *mapop = (ILM *)(ilmb.ilm_base + argilm);
  if (ILM_OPC(mapop) == IM_BASE) {
    sptr = ILM_SymOPND(mapop, 1); // make 2
    label = ILM_OPND(ilmp, 2);    /* map type */
  } else if (ILM_OPC(mapop) == IM_PLD) {
    sptr = ILM_SymOPND(mapop, 2); // make 2
    label = ILM_OPND(ilmp, 2);    /* map type */
  }
  ompaccel_tinfo_current_addupdate_mapitem(sptr, label);
}

void
exp_ompaccel_emap(ILM *ilmp, int curilm)
{
  int ili;
  OMPACCEL_TINFO *targetinfo;
  if (ompaccel_tinfo_has(gbl.currsub))
    return;
  ompaccel_symreplacer(true);
  targetinfo = ompaccel_tinfo_current_get();
  if (targetinfo != NULL) {
    if (ompaccel_tinfo_current_target_mode() == mode_target_data_enter_region ||
        ompaccel_tinfo_current_target_mode() == mode_target_data_region) {
      wr_block();
      cr_block();
      ili =
          ll_make_tgt_target_data_begin(OMPACCEL_DEFAULT_DEVICEID, targetinfo);
      iltb.callfg = 1;
      chk_block(ili);
    } else if (ompaccel_tinfo_current_target_mode() ==
        mode_target_data_exit_region) {
      wr_block();
      cr_block();
      ili = ll_make_tgt_target_data_end(OMPACCEL_DEFAULT_DEVICEID, targetinfo);
      iltb.callfg = 1;
      chk_block(ili);
    }
  }
}

void
exp_ompaccel_looptripcount(ILM *ilmp, int curilm)
{
  /* push loop trip count is disabled because of performance issue */
  if (XBIT(232, 0x20)) {
    SPTR sptr;
    int ili;
    wr_block();
    cr_block();
    sptr = ILM_SymOPND(ilmp, 1);
    ili = ll_make_kmpc_push_target_tripcount(OMPACCEL_DEFAULT_DEVICEID, sptr);
    iltb.callfg = 1;
    chk_block(ili);
  }
}

void
exp_ompaccel_reductionitem(ILM *ilmp, int curilm)
{
  ompaccel_tinfo_current_add_reductionitem(
          ILM_SymOPND(ilmp, 1), ILM_SymOPND(ilmp, 2), ILM_SymOPND(ilmp, 3));
}

void
exp_ompaccel_targetdata(ILM *ilmp, int curilm, ILM_OP opc)
{
  int dotarget;
  SPTR beg_label, end_label;
  ompaccel_symreplacer(false);
  ompaccel_tinfo_create(OMPACCEL_DATA_FUNCTION, OMPACCEL_DATA_MAX_SYM);
  if (opc == IM_TARGETEXITDATA)
    ompaccel_tinfo_current_set_mode(mode_target_data_exit_region);
  else if (opc == IM_TARGETENTERDATA)
    ompaccel_tinfo_current_set_mode(mode_target_data_enter_region);
  else if (opc == IM_BTARGETDATA)
    ompaccel_tinfo_current_set_mode(mode_target_data_region);
  dotarget = ILI_OF(ILM_OPND(ilmp, 1));
  beg_label = getlab();
  end_label = getlab();

  dotarget = ad3ili(IL_ICJMPZ, dotarget, CC_EQ, end_label);
  RFCNTI(end_label);
  chk_block(dotarget);

  wr_block();
  cr_block();
  exp_label(beg_label);

  /* .... TODO: call to runtime target data here  */

  exp_label(end_label);
}
void
exp_ompaccel_etargetdata(ILM *ilmp, int curilm)
{
  OMPACCEL_TINFO *targetinfo;
  int ili;
  if (gbl.outlined)
    return;
  ompaccel_symreplacer(true);
  targetinfo = ompaccel_tinfo_current_get_targetdata();
  wr_block();
  cr_block();
  ili = ll_make_tgt_target_data_end(OMPACCEL_DEFAULT_DEVICEID, targetinfo);
  iltb.callfg = 1;
  chk_block(ili);
}

void
init_test()
{
  init_tgtutil();
}

#endif
/* Expander - OpenMP Accelerator Model */

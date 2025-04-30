/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Routines used by lower.c for lowering symbols.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "ast.h"
#include "semant.h"
#include "dinit.h"
#include "soc.h"
#include "pragma.h"
#include "rte.h"
#include "fih.h"
#include "dpm_out.h"
#include "rtlRtns.h"
#include "sharedefs.h"

#include "llmputil.h"

#define INSIDE_LOWER
#include "lower.h"
#include "dbg_out.h"
void scan_for_dwarf_module();
extern int print_file(int fihx);
static int valid_kind_parm_expr(int ast);
static int is_descr_expression(int ast);
static int lower_getnull(void);

/* table of data types to be exported */
static char *datatype_used;
static char *datatype_output;
static int last_datatype_used;
/* flag whether to mark linearized arrays yet */
static LOGICAL lower_linearized_dtypes = FALSE;

#define STB_LOWER() ((gbl.outfil == lowersym.lowerfile) && gbl.stbfil)
#define IS_STB_FILE() (gbl.stbfil == lowersym.lowerfile)
static void _stb_fixup_ifacearg(int);

/* keep a stack of information */
static int stack_top, stack_size;
static int *stack;

/* keep track of fih that has been written to file */
static int curr_findex;

/** \brief List of ILMs for function/subroutine arguments */
int *lower_argument;
int lower_argument_size;

/* header of linked list of pointer or allocatable variables whose
 * pointer/offset/descriptors need to be initialized */
static int lower_pointer_list_head;

/* head of linked list of pointer/offset/section descriptors in the order they
 * need to be given addresses */
static int lower_refd_list_head;

/* size of private area needed for private descriptors & their pointer &
 * offset variables.
 */
static ISZ_T private_addr;

struct lower_syms lowersym;

static int first_avail_scalarptr_temp, first_used_scalarptr_temp, first_temp;
static int first_avail_scalar_temp, first_used_scalar_temp;
static void lower_put_datatype(int, int);
static bool has_opt_args(SPTR sptr);

static void lower_fileinfo_llvm();
static LOGICAL llvm_iface_flag = FALSE;
static void stb_lower_sym_header();
static void check_debug_alias(SPTR sptr);

/** \brief
 * ASSCHAR = -1 assumed size character
 * ADJCHAR = -2 backend maps to DT_ASSCHAR
 * DEFERCHAR = -3 deferred-length character */
enum LEN {ASSCHAR = -1, ADJCHAR = -2, DEFERCHAR = -3};

/** \brief Returns true if the procedure (sptr) has optional arguments.
 */
static bool
has_opt_args(SPTR sptr)
{
 int i, psptr, nargs, dpdsc;

  if (STYPEG(sptr) != ST_ENTRY && STYPEG(sptr) != ST_PROC) {
    return false;
  }
  nargs = PARAMCTG(sptr);
  dpdsc = DPDSCG(sptr);
  for (i = 0; i < nargs; ++i) {
    psptr = *(aux.dpdsc_base + dpdsc + i);
    if (OPTARGG(psptr)) {
       return true;
    }
  }
  return false;
}
/** \brief Set 'EXTRA' bit for arrays, descriptors, array members
    that have IPA no conflict information, or that are compiler temps,
    or that can't conflict because they aren't targets and aren't pointers
 */
void
lower_set_symbols(void)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    /* allocatable arrays and members that are not POINTER
     * arrays can be 'noconflict'; arrays without TARGET can be
     * 'noconflict'; temp arrays are 'noconflict' */
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
      if (!IGNOREG(sptr)) {
        if ((!ADDRTKNG(sptr) || (ALLOCG(sptr) && !POINTERG(sptr))) &&
            SCG(sptr) != SC_BASED && IPA_isnoconflict(sptr)) {
          VISIT2P(sptr, 1);
          if (STYPEG(sptr) == ST_ARRAY && NEWARGG(sptr)) {
            VISIT2P(NEWARGG(sptr), 1);
          }
        }
      }
      FLANG_FALLTHROUGH;
    case ST_MEMBER:
    case ST_DESCRIPTOR:
      if (!IGNOREG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY) {
        if ((!TARGETG(sptr) && !POINTERG(sptr) &&
             (ALLOCG(sptr) || !ADDRTKNG(sptr))) ||
            CCSYMG(sptr) || HCCSYMG(sptr)) {
          VISIT2P(sptr, 1);
        }
      }
      FLANG_FALLTHROUGH;
    case ST_VAR:
      if (SCG(sptr) == SC_BASED) {
        /* look at section descriptor, pointer */
        int d, p;
        p = MIDNUMG(sptr);
        if (p && HCCSYMG(p))
          VISIT2P(p, 1);
        d = SDSCG(sptr);
        if (d && HCCSYMG(d))
          VISIT2P(d, 1);
      }
      break;
    default:;
    }
  }
} /* lower_set_symbols */

/** \brief Set datatype of 'cray pointers' to derived types.
 */
void
lower_set_craypointer(void)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
    case ST_VAR:
    case ST_MEMBER:
      if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
        int ptr;
        ptr = MIDNUMG(sptr);
        if (DTYPEG(ptr) == DT_PTR) {
          int dtype, ndtype;
          dtype = DTYPEG(sptr);
          if (DTY(dtype) == TY_ARRAY)
            dtype = DTY(dtype + 1);
          if (DTY(dtype) == TY_PTR)
            ndtype = dtype;
          else {
            ndtype = get_type(2, TY_PTR, dtype);
          }
          DTYPEP(ptr, ndtype);
          if (VISITG(ptr) || ndtype >= last_datatype_used) {
            lower_use_datatype(ndtype, 1);
          }
        }
      }
      break;
    default:;
    }
  }
} /* lower_set_craypointer */

/** \brief Reset data types of derived type pointers to DT_PTR.
 */
void
lower_unset_symbols(void)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
    case ST_VAR:
    case ST_MEMBER:
      if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
        int ptr;
        ptr = MIDNUMG(sptr);
        DTYPEP(ptr, DT_PTR);
      }
      break;
    default:;
    }
  }
} /* lower_unset_symbols */

#ifdef FLANG_LOWERSYM_UNUSED
static void save_vol_descriptors(int);
#endif

/* call this first so the symbol count and datatype count won't change later */
static void
lower_make_all_descriptors(void)
{
  int sptr;
  int stp = 0;
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
    case ST_DESCRIPTOR:
    case ST_VAR:
    case ST_IDENT:
    case ST_STRUCT:
      if (IGNOREG(sptr))
        break;
      /* see if setting LNRZD fixes REDIM statement processing */
      if (ALLOCG(sptr) && !NODESCG(sptr)) {
        LNRZDP(sptr, 1);
      }
      if (ENCLFUNCG(sptr) != 0 && !CONSTRUCTSYMG(sptr)) {
        /* module symbols */
        if (!POINTERG(sptr) && SDSCG(sptr) != 0 &&
            STYPEG(SDSCG(sptr)) != ST_PARAM) {
          if (!ASSUMSHPG(sptr) ||
              (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))) {
            /* set SDSCS1 for sdsc */
            SDSCS1P(SDSCG(sptr), 1);
          }
        }
        break;
      }
      /* names that weren't resolved might be variables used by internal
       * subroutines */
      if (SCG(sptr) == SC_NONE)
        SCP(sptr, SC_LOCAL);
      if (SAVEG(sptr) && SCG(sptr) == SC_LOCAL)
        SCP(sptr, SC_STATIC);
      if (STYPEG(sptr) == ST_IDENT)
        STYPEP(sptr, ST_VAR);
      if (POINTERG(sptr) || ALLOCG(sptr) || ALLOCATTRG(sptr)) {
        if (SDSCG(sptr) == 0 || STYPEG(SDSCG(sptr)) == ST_PARAM) {
          if (MIDNUMG(sptr) == 0) {
            stp = sym_get_ptr(sptr);
            MIDNUMP(sptr, stp);
            if (SCG(sptr) == SC_PRIVATE)
              SCP(stp, SC_PRIVATE);
          }
          PTRSAFEP(MIDNUMG(sptr), 1);
        } else {
          if (PTROFFG(sptr) == 0) {
            if (MIDNUMG(sptr) == 0) {
              stp = sym_get_ptr(sptr);
              MIDNUMP(sptr, stp);
              if (SCG(sptr) == SC_PRIVATE)
                SCP(stp, SC_PRIVATE);
            }
            if (SCG(sptr) == SC_DUMMY) {
              if (!stp)
                stp = sym_get_ptr(sptr);
              SCP(stp, SC_DUMMY);
              MIDNUMP(sptr, stp);
            }
          }
          if (!POINTERG(sptr)) {
            /* set SDSCS1 for sdsc */
            SDSCS1P(SDSCG(sptr), 1);
          }
          if (MIDNUMG(sptr))
            PTRSAFEP(MIDNUMG(sptr), 1);
        }
        SCP(sptr, SC_BASED);
        if (SAVEG(sptr) && SCG(sptr) == SC_STATIC) {
          int ptr, sdsc, off;
          ptr = MIDNUMG(sptr);
          SAVEP(MIDNUMG(sptr), 1);
          if (ptr && SCG(ptr) == SC_LOCAL)
            SCP(ptr, SC_STATIC);
          sdsc = SDSCG(sptr);
          if (sdsc && STYPEG(sdsc) != ST_PARAM) {
            SAVEP(sdsc, 1);
            if (SCG(sdsc) == SC_LOCAL)
              SCP(sdsc, SC_STATIC);
          }
          off = PTROFFG(sptr);
          if (off && STYPEG(off) != ST_PARAM) {
            SAVEP(off, 1);
            if (SCG(off) == SC_LOCAL)
              SCP(off, SC_STATIC);
          }
          SAVEP(sptr, 0);
        }
      } else if (AUTOBJG(sptr) || (ADJARRG(sptr) && SCG(sptr) == SC_LOCAL)) {
        if (MIDNUMG(sptr) == 0) {
          SCP(sptr, SC_BASED);
          stp = sym_get_ptr(sptr);
          MIDNUMP(sptr, stp);
        }
        else if (flg.smp && MIDNUMG(sptr)) {
          SCP(sptr, SC_BASED);
        }
        PTRSAFEP(MIDNUMG(sptr), 1);
        if (SAVEG(sptr) && SCG(sptr) == SC_STATIC) {
          int ptr, sdsc, off;
          ptr = MIDNUMG(sptr);
          SAVEP(MIDNUMG(sptr), 1);
          if (ptr && SCG(ptr) == SC_LOCAL)
            SCP(ptr, SC_STATIC);
          sdsc = SDSCG(sptr);
          if (sdsc && STYPEG(sdsc) != ST_PARAM) {
            SAVEP(sdsc, 1);
            if (SCG(sdsc) == SC_LOCAL)
              SCP(sdsc, SC_STATIC);
          }
          off = PTROFFG(sptr);
          if (off && STYPEG(off) != ST_PARAM) {
            SAVEP(off, 1);
            if (SCG(off) == SC_LOCAL)
              SCP(off, SC_STATIC);
          }
          SAVEP(sptr, 0);
        }
      }
      break;
    case ST_MEMBER:
      if (!POINTERG(sptr)) {
        if (SDSCG(sptr) != 0 && STYPEG(SDSCG(sptr)) != ST_PARAM) {
          /* set SDSCS1 for sdsc */
          SDSCS1P(SDSCG(sptr), 1);
        }
      }
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        if (DISTG(sptr) || ALIGNG(sptr) || ADJARRG(sptr) || RUNTIMEG(sptr)) {
          /* implement by handling like a pointer */
          POINTERP(sptr, 1);
        }
      }
      if (POINTERG(sptr)) {
        if (SDSCG(sptr) == 0 || STYPEG(SDSCG(sptr)) == ST_PARAM) {
          if (MIDNUMG(sptr) == 0) {
            if (!is_procedure_ptr(sptr)) {
              stp = sym_get_ptr(sptr);
              MIDNUMP(sptr, stp);
            } else {
              MIDNUMP(sptr, sptr);
            }
          }
        } else {
          if (PTROFFG(sptr) == 0) {
            if (MIDNUMG(sptr) == 0) {
              stp = sym_get_ptr(sptr);
              MIDNUMP(sptr, stp);
            }
            stp = sym_get_offset(sptr);
            PTROFFP(sptr, stp);
          }
        }
      }
      break;
    default:;
    }
  }
} /* lower_make_all_descriptors */

#ifdef FLANG_LOWERSYM_UNUSED
static void
save_vol_descriptors(int sptr)
{
  int ptr, sdsc, off;
  if (SAVEG(sptr) && SCG(sptr) == SC_STATIC) {
    ptr = MIDNUMG(sptr);
    SAVEP(MIDNUMG(sptr), 1);
    if (ptr && SCG(ptr) == SC_LOCAL)
      SCP(ptr, SC_STATIC);
    sdsc = SDSCG(sptr);
    if (sdsc && STYPEG(sdsc) != ST_PARAM) {
      SAVEP(sdsc, 1);
      if (SCG(sdsc) == SC_LOCAL)
        SCP(sdsc, SC_STATIC);
    }
    off = PTROFFG(sptr);
    if (off && STYPEG(off) != ST_PARAM) {
      SAVEP(off, 1);
      if (SCG(off) == SC_LOCAL)
        SCP(off, SC_STATIC);
    }
    SAVEP(sptr, 0);
  }
  if (VOLG(sptr)) {
    ptr = MIDNUMG(sptr);
    VOLP(MIDNUMG(sptr), 1);
    sdsc = SDSCG(sptr);
    if (sdsc && STYPEG(sdsc) != ST_PARAM) {
      VOLP(sdsc, 1);
    }
    off = PTROFFG(sptr);
    if (off && STYPEG(off) != ST_PARAM) {
      VOLP(off, 1);
    }
    VOLP(sptr, 0);
  }
}
#endif

static int
remove_list(int list, int sym)
{
  int l, prev = 0;
  for (l = list; l > NOSYM; l = SYMLKG(l)) {
    if (l == sym) {
      if (prev) {
        SYMLKP(prev, SYMLKG(sym));
      } else {
        list = SYMLKG(sym);
      }
      SYMLKP(sym, NOSYM);
      return list;
    }
    prev = l;
  }
  /* not found */
  return list;
} /* remove_list */

static void
push_lower_refd_list(int sym)
{
  if (LOWER_REFD_LIST(sym)) {
    int l, prev;
    prev = 0;
    for (l = lower_refd_list_head; l > NOSYM; l = LOWER_REFD_LIST(l)) {
      if (l == sym) {
        if (prev) {
          LOWER_REFD_LIST(prev) = LOWER_REFD_LIST(sym);
        } else {
          lower_refd_list_head = LOWER_REFD_LIST(sym);
        }
        break;
      }
      prev = l;
    }
  }
  LOWER_REFD_LIST(sym) = lower_refd_list_head;
  lower_refd_list_head = sym;
} /* push_lower_refd_list */

/* fill in LWAST, UPAST, MLPYR, ZBASE, NUMELM fields */
static void
fill_fixed_array_dtype(int dtype)
{
  int i, ndim, m;
  ISZ_T mlpyr, zbase;
  ndim = ADD_NUMDIM(dtype);
  mlpyr = 1;
  zbase = 0;

  m = ADD_MLPYR(dtype, 0);
  if (m == 0) {
    mlpyr = 1;
  } else {
    if (A_ALIASG(m))
      m = A_ALIASG(m);
    if (A_TYPEG(m) != A_CNST) {
      lerror("nonconstant multiplier for dimension 1 for datatype %d", dtype);
      mlpyr = 1;
    } else {
      int mlpyrsym;
      mlpyrsym = A_SPTRG(m);
      lower_visit_symbol(mlpyrsym);
      if (STYPEG(mlpyrsym) == ST_CONST) {
        mlpyr = ad_val_of(mlpyrsym);
      } else {
        lerror("nonconstant multiplier for dimension 1 for datatype %d", dtype);
        mlpyr = 1;
      }
    }
  }

  for (i = 0; i < ndim; ++i) {
    int lw, up;
    ISZ_T lwval, upval;

    lw = ADD_LWAST(dtype, i);
    if (lw != 0 && A_ALIASG(lw))
      lw = A_ALIASG(lw);
    if (lw == 0) {
      lwval = 1;
      ADD_LWAST(dtype, i) = mk_cnst(lower_getiszcon(lwval));
    } else if (A_TYPEG(lw) == A_CNST) {
      lwval = ad_val_of(A_SPTRG(lw));
    } else {
      lerror("nonconstant array lower bound for dimension %d for datatype %d",
             i, dtype);
      lwval = 1;
      ADD_LWAST(dtype, i) = mk_cnst(lower_getiszcon(lwval));
    }

    if (mlpyr > 0) {
      ADD_MLPYR(dtype, i) = mk_cnst(lower_getiszcon(mlpyr));
      zbase = zbase + mlpyr * lwval;
    }

    up = ADD_UPAST(dtype, i);

    if (up != 0 && A_ALIASG(up))
      up = A_ALIASG(up);
    if (up == 0) {
      if (i != ndim - 1) {
        lerror("no upper bound for dimension %d of datatype %d", i, dtype);
      }
      mlpyr = -1;
    } else if (A_TYPEG(up) != A_CNST && !valid_kind_parm_expr(up)) {
      if (i != ndim - 1) {
        lerror("nonconstant upper bound for dimension %d of datatype %d", i,
               dtype);
      }
      mlpyr = -1;
    } else {
      upval = ad_val_of(A_SPTRG(up));

      /* update multiplier for next dimension;
       * mlpyr = mlpyr * (upper - lower + 1) */
      if (mlpyr > 0) {
        mlpyr *= (upval - lwval + 1);
      }
    }
  }
  ADD_ZBASE(dtype) = mk_cnst(lower_getiszcon(zbase));

  if (mlpyr > 0) {
    ADD_NUMELM(dtype) = mk_cnst(lower_getiszcon(mlpyr));
  } else {
    ADD_NUMELM(dtype) = astb.bnd.zero;
  }
} /* fill_fixed_array_dtype */

/* fill in LWAST, UPAST, MLPYR, ZBASE, NUMELM fields */
static void
fill_pointer_array_dtype(int dtype, int sptr)
{
  int i, ndim, zbaseast, numelmast, desc;

  desc = SDSCG(sptr);
  if (desc == 0) {
    lerror("no descriptor for %s, datatype %d", SYMNAME(sptr), dtype);
    return;
  }
  ndim = ADD_NUMDIM(dtype);
  for (i = 0; i < ndim; ++i) {
    int lwast, upast, extntast, mast;
    lwast = ADD_LWAST(dtype, i);
    if (!lwast || A_TYPEG(lwast) != A_CNST) {
      ADD_LWAST(dtype, i) = get_global_lower(desc, i);
    }

    upast = ADD_UPAST(dtype, i);
    if (!upast || A_TYPEG(upast) != A_CNST) {
      int a;
      a = get_extent(desc, i);
      a = mk_binop(OP_SUB, a, astb.i1, A_DTYPEG(a)),
      ADD_UPAST(dtype, i) =
          mk_binop(OP_ADD, get_global_lower(desc, i), a, A_DTYPEG(a));
    }

    extntast = ADD_EXTNTAST(dtype, i);
    if (!extntast || A_TYPEG(extntast) != A_CNST) {
      ADD_EXTNTAST(dtype, i) = get_extent(desc, i);
    }

    mast = ADD_MLPYR(dtype, i);
    if (!mast || A_TYPEG(mast) != A_CNST) {
      ADD_MLPYR(dtype, i) = get_local_multiplier(desc, i);
    }
  }
  zbaseast = ADD_ZBASE(dtype);
  if (!zbaseast || A_TYPEG(zbaseast) != A_CNST) {
    ADD_ZBASE(dtype) = get_xbase(desc);
  }
  numelmast = ADD_NUMELM(dtype);
  if (!numelmast || A_TYPEG(numelmast) != A_CNST) {
    ADD_NUMELM(dtype) = get_desc_gsize(desc);
  }
} /* fill_pointer_array_dtype */

static int
adjarr_class(int sptr)
{
  int midnum;
  if (!XBIT(52, 4)) {
    if (POINTERG(sptr) || MDALLOCG(sptr)) {
      return SC_NONE;
    }
  }
  midnum = MIDNUMG(sptr);
  if (!midnum) {
    if (!THREADG(sptr)) {
      if (SAVEG(sptr) || SCG(sptr) == SC_STATIC) {
        return SC_STATIC;
      }
    }
  } else {
    if (!THREADG(sptr)) {
      if (SAVEG(midnum) || SCG(midnum) == SC_STATIC ||
          SCG(midnum) == SC_CMBLK) {
        return SC_STATIC;
      }
    }
    if (SCG(midnum) == SC_PRIVATE)
      return SC_PRIVATE;
  }
  return SC_LOCAL;
} /* adjarr_class */

static int
get_atmp(int tempsc, int dt, int saveg)
{
  int s;
  s = getccsym('A', ++lowersym.acount, ST_VAR);
  SCP(s, tempsc);
  DTYPEP(s, dt);
  SAVEP(s, saveg);
  return s;
}

/* fill in LWAST, UPAST, MLPYR, ZBASE, NUMELM fields
 * if assumed-shape, lower bounds are the actual values used */
static void
fill_adjustable_array_dtype(int dtype, int assumedshape, int stride1,
                            int tempsc, int alltemp, int keeptemp, int saveg,
                            int sptr)
{
  int i, ndim, zbase, zbasesym, nonconstant;
  int mlpyr, mlpyrsym;
  ISZ_T mlpyrval;
  int dt_bnd;
  int enclfunc, taskp;

  enclfunc = 0;
  taskp = 0;

  if (XBIT(68, 0x1))
    dt_bnd = DT_INT8;
  else
    dt_bnd = DT_INT4;

  ndim = ADD_NUMDIM(dtype);
  nonconstant = 0;

  mlpyr = ADD_MLPYR(dtype, 0);
  if (mlpyr == 0 || stride1) {
    mlpyrval = 1;
    mlpyrsym = 0;
  } else {
    if (A_ALIASG(mlpyr))
      mlpyr = A_ALIASG(mlpyr);
    if (A_TYPEG(mlpyr) == A_ID || A_TYPEG(mlpyr) == A_CNST) {
      mlpyrsym = A_SPTRG(mlpyr);
      if (!alltemp && STYPEG(mlpyrsym) == ST_CONST) {
        mlpyrval = ad_val_of(mlpyrsym);
        mlpyrsym = 0;
      } else if (!keeptemp || STYPEG(mlpyrsym) != ST_VAR) {
        mlpyrsym = get_atmp(tempsc, dt_bnd, saveg);
        mlpyrval = 0;
        if (enclfunc) {
          ENCLFUNCP(mlpyrsym, enclfunc);
          TASKP(mlpyrsym, 1);
        }
      }
    } else {
      mlpyrsym = get_atmp(tempsc, dt_bnd, saveg);
      if (enclfunc) {
        ENCLFUNCP(mlpyrsym, enclfunc);
        TASKP(mlpyrsym, 1);
      }
      mlpyrval = 0;
    }
  }
  /* update multiplier */
  if (mlpyrsym == 0) {
    /* so far, multiplier is constant */
    ADD_MLPYR(dtype, 0) = mk_cnst(lower_getiszcon(mlpyrval));
  } else {
    ADD_MLPYR(dtype, 0) = mk_id(mlpyrsym);
    lower_visit_symbol(mlpyrsym);
  }
  for (i = 0; i < ndim; ++i) {
    int lw, lwsym, up, upsym, extnt;
    ISZ_T lwval, upval;
    lw = ADD_LWAST(dtype, i);
    if (lw != 0 && A_ALIASG(lw))
      lw = A_ALIASG(lw);
    if (lw == 0 && assumedshape && !XBIT(54, 2) &&
        !(XBIT(58, 0x400000) && TARGETG(sptr))) {
      ADD_LWAST(dtype, i) = astb.bnd.one;
      lwsym = 0;
      lwval = 1;
    } else if (lw && A_TYPEG(lw) == A_CNST && !alltemp) {
      lwval = ad_val_of(A_SPTRG(lw));
      lwsym = 0;
    } else if (keeptemp && lw && A_TYPEG(lw) == A_ID) {
      lwval = 0;
      lwsym = A_SPTRG(lw);
    } else {
      lwsym = get_atmp(tempsc, dt_bnd, saveg);
      if (enclfunc) {
        ENCLFUNCP(lwsym, enclfunc);
        TASKP(lwsym, 1);
      }
      ADD_LWAST(dtype, i) = mk_id(lwsym);
      lwval = 0;
      lower_visit_symbol(lwsym);
    }

    up = ADD_UPAST(dtype, i);
    if (up != 0 && A_ALIASG(up))
      up = A_ALIASG(up);
    if (up && A_TYPEG(up) == A_CNST && !alltemp) {
      upval = ad_val_of(A_SPTRG(up));
      upsym = 0;
    } else if (keeptemp && up && A_TYPEG(up) == A_ID) {
      upval = 0;
      upsym = A_SPTRG(up);
    } else {
      upsym = get_atmp(tempsc, dt_bnd, saveg);
      if (enclfunc) {
        ENCLFUNCP(upsym, enclfunc);
        TASKP(upsym, 1);
      }
      ADD_UPAST(dtype, i) = mk_id(upsym);
      upval = 0;
      lower_visit_symbol(upsym);
    }

    extnt = ADD_EXTNTAST(dtype, i);
    if (extnt != 0 && A_ALIASG(extnt))
      extnt = A_ALIASG(extnt);
    if (extnt && A_TYPEG(extnt) == A_CNST && !alltemp) {
      extnt = CONVAL2G(A_SPTRG(extnt));
    } else if (keeptemp && extnt && A_TYPEG(extnt) == A_ID) {
      extnt = A_SPTRG(extnt);
    } else if (ALLOCATTRG(sptr) && THREADG(sptr) && extnt) {
      /*
       * do not create a scalar temp for the extent of an allocatable
       * threadprivate; use the desriptor as-is.
       * Perhaps, another routine should be called instead of
       * fill_adjustable_array_dtype(), e.g., for POINTERs, we call
       * fill_pointer_array_dtype()
       */
      ;
    } else {
      extnt = get_atmp(tempsc, dt_bnd, saveg);
      if (enclfunc) {
        ENCLFUNCP(extnt, enclfunc);
        TASKP(extnt, 1);
      }
      ADD_EXTNTAST(dtype, i) = mk_id(extnt);
      lower_visit_symbol(extnt);
    }

    if (mlpyrsym == 0 && lwsym == 0 && upsym == 0) {
      mlpyrval *= (upval - lwval + 1);
      ADD_MLPYR(dtype, i + 1) = mk_cnst(lower_getiszcon(mlpyrval));
    } else {
      mlpyr = ADD_MLPYR(dtype, i + 1);
      if (keeptemp && mlpyr && A_TYPEG(mlpyr) == A_ID) {
        mlpyrval = 0;
        mlpyrsym = A_SPTRG(mlpyr);
      } else {
        mlpyrsym = get_atmp(tempsc, lowersym.bnd.dtype, saveg);
        if (enclfunc) {
          ENCLFUNCP(mlpyrsym, enclfunc);
          TASKP(mlpyrsym, 1);
        }
        ADD_MLPYR(dtype, i + 1) = mk_id(mlpyrsym);
      }
      lower_visit_symbol(mlpyrsym);
    }
  }

  zbase = ADD_ZBASE(dtype);
  if (keeptemp && (A_TYPEG(zbase) == A_ID || A_TYPEG(zbase) == A_CNST)) {
    zbasesym = A_SPTRG(zbase);
  } else {
    zbasesym = get_atmp(tempsc, dt_bnd, saveg);
    if (enclfunc) {
      ENCLFUNCP(zbasesym, enclfunc);
      TASKP(zbasesym, 1);
    }
    ADD_ZBASE(dtype) = mk_id(zbasesym);
  }
  lower_visit_symbol(zbasesym);
} /* fill_adjustable_array_dtype */

static void
lower_prepare_symbols()
{
  int sptr, link, fval;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    int dtype, stype;
    stype = STYPEG(sptr);
    dtype = DTYPEG(sptr);
    if (GSCOPEG(sptr)) {
      fixup_reqgs_ident(sptr);
    }
    switch (stype) {
    case ST_ARRAY:
      if ((gbl.internal <= 1 && !gbl.empty_contains) || INTERNALG(sptr)) {
        int saveg;
        saveg = 0;
        if (SAVEG(sptr) && !THREADG(sptr))
          saveg = 1;
        if (POINTERG(sptr) || MDALLOCG(sptr) || ALIGNG(sptr) || DISTG(sptr)) {
          if (!XBIT(52, 4)) {
            if (SDSCG(sptr) && STYPEG(SDSCG(sptr)) != ST_PARAM) {
              /* use section descriptor elements in the array datatype */
              fill_pointer_array_dtype(dtype, sptr);
            }
          } else {
            /* insert .A variables in the descriptor */
            fill_adjustable_array_dtype(dtype, ASSUMSHPG(sptr), 0,
                                        adjarr_class(sptr), POINTERG(sptr), 0,
                                        saveg, sptr);
          }
        } else if (XBIT(57, 0x10000) && ASSUMSHPG(sptr)) {
          /* don't need to insert .A variables in the descriptor */
        } else if (ASSUMSHPG(sptr) ||
                   (ALLOCG(sptr) && SCG(sptr) == SC_BASED && MIDNUMG(sptr) &&
                    SCG(MIDNUMG(sptr)) == SC_CMBLK)) {
          if (!XBIT(52, 4)) {
            int subdtype;
            subdtype = DTY(dtype + 1);
            subdtype = DTY(subdtype);
            if (SDSCG(sptr) && !NODESCG(sptr) && subdtype != TY_CHAR &&
                subdtype != TY_NCHAR && STYPEG(SDSCG(sptr)) != ST_PARAM) {
              /* use section descriptor elements in the array datatype */
              fill_pointer_array_dtype(dtype, sptr);
            } else {
              fill_adjustable_array_dtype(dtype, ASSUMSHPG(sptr), 1,
                                          adjarr_class(sptr), ALLOCG(sptr), 1,
                                          saveg, sptr);
            }
          } else if (!XBIT(52, 8)) {
            fill_adjustable_array_dtype(dtype, ASSUMSHPG(sptr), 1,
                                        adjarr_class(sptr), ALLOCG(sptr), 1,
                                        saveg, sptr);
          } else {
            /* insert .A variables in the datatype */
            fill_adjustable_array_dtype(dtype, ASSUMSHPG(sptr), 1,
                                        adjarr_class(sptr), ALLOCG(sptr), 0,
                                        saveg, sptr);
          }
        } else if (gbl.internal && ALLOCATTRG(sptr) && !INTERNALG(sptr) &&
                   MIDNUMG(sptr) &&
                   (SCG(MIDNUMG(sptr)) == SC_LOCAL ||
                    SCG(MIDNUMG(sptr)) == SC_DUMMY)) {
          /*
           * nothing to do --- Host local allocatables will be
           * descriptor-based in the presence of internal procedures
           */
          ;
        } else if (ALLOCG(sptr) || AUTOBJG(sptr) ||
                   (ADJARRG(sptr) && SCG(sptr) == SC_LOCAL)) {
          if (flg.smp && MIDNUMG(sptr) && TASKG(MIDNUMG(sptr)))
            ;
          else
              if (!XBIT(52, 8)) {
            /* insert .A variables in the datatype */
            fill_adjustable_array_dtype(dtype, 0, 1, adjarr_class(sptr),
                                        ALLOCG(sptr), 1, saveg, sptr);
          } else {
            /* insert .A variables in the datatype */
            fill_adjustable_array_dtype(dtype, 0, 1, adjarr_class(sptr),
                                        ALLOCG(sptr), 0, saveg, sptr);
          }
        } else if (!ADJARRG(sptr)) {
          /* fixed-size datatype */
          fill_fixed_array_dtype(dtype);
        }
      }
      FLANG_FALLTHROUGH;
    case ST_VAR:
    case ST_IDENT:
    case ST_STRUCT:
      if (MDALLOCG(sptr))
        break;
      if (SCG(sptr) == SC_CMBLK)
        break;
      if (SCG(sptr) == SC_DUMMY)
        break;
      if (SCG(sptr) == SC_STATIC)
        break;
      if (CCSYMG(sptr) && !RESULTG(sptr))
        break;
      if (ENCLFUNCG(sptr) != 0 && !CONSTRUCTSYMG(sptr))
        break;
      if (POINTERG(sptr) || ALLOCG(sptr)) {
        /* this gets confused if the same ptr/off/desc are used
         * for more than one symbol (as for function return arrays).
         * We don't want to put them on the gbl.locals list more than
         * once, and do want to make them static if any of the symbols
         * using them are static */
        int ptr, off, desc;
        ptr = MIDNUMG(sptr);
        if (ptr == 0)
          break;
        off = PTROFFG(sptr);
        desc = SDSCG(sptr);
        if (desc != 0) {
          if (STYPEG(desc) == ST_PARAM || STYPEG(desc) == ST_MEMBER)
            break;
          IGNOREP(ptr, 0);
          if (off)
            IGNOREP(off, 0);
          IGNOREP(desc, 0);

          /* give new addresses */
          if (REFG(ptr)) {
            if (SCG(ptr) == SC_STATIC) {
              gbl.statics = remove_list(gbl.statics, ptr);
            } else {
              gbl.locals = remove_list(gbl.locals, ptr);
            }
            REFP(ptr, 0);
          }
          if (off && REFG(off)) {
            if (SCG(off) == SC_STATIC) {
              gbl.statics = remove_list(gbl.statics, off);
            } else {
              gbl.locals = remove_list(gbl.locals, off);
            }
            REFP(off, 0);
          }
          if (REFG(desc)) {
            if (SCG(desc) == SC_STATIC) {
              gbl.statics = remove_list(gbl.statics, desc);
            } else {
              gbl.locals = remove_list(gbl.locals, desc);
            }
            REFP(desc, 0);
          }

          /* astout.c would put the pointer/offset/descriptor
           * triplet in a common block to make sure they are
           * allocated continguously.  Here, we simply give them
           * consecutively addresses */
          if (SAVEG(sptr)) {
            SAVEP(ptr, 1);
            SCP(ptr, SC_STATIC);
            if (off) {
              SAVEP(off, 1);
              SCP(off, SC_STATIC);
            }
            /* FS#18004: If descriptor is for a polymorphic entity
             * and the descriptor is a dummy argument, then do not
             * turn it into a save variable/static. Otherwise,
             * we may lose type information at runtime.
             */
            if (!CLASSG(sptr) || SCG(desc) != SC_DUMMY) {
              SAVEP(desc, 1);
              SCP(desc, SC_STATIC);
            }
          } else if (SCG(ptr) != SC_DUMMY &&
                     (SCG(ptr) == SC_STATIC || SCG(desc) == SC_STATIC ||
                      (off && SCG(off) == SC_STATIC))) {
            SCP(ptr, SC_STATIC);
            if (off)
              SCP(off, SC_STATIC);
            SCP(desc, SC_STATIC);
          }
          if (ptr >= stb.firstusym && off > stb.firstusym &&
              desc > stb.firstusym) {
            if (SCG(desc) != SC_DUMMY) {
              if (SCG(ptr) == SC_LOCAL) {
                push_lower_refd_list(ptr);
                push_lower_refd_list(off);
                push_lower_refd_list(desc);
              } else {
                push_lower_refd_list(desc);
                push_lower_refd_list(off);
                push_lower_refd_list(ptr);
              }
            }
          }
        }
        if (XBIT(47, 0x8000000)) {
          if (desc)
            ADDRTKNP(desc, 1);
          if (off)
            ADDRTKNP(off, 1);
          ADDRTKNP(ptr, 1);
        }
        if (!SAVEG(ptr) && SCG(ptr) != SC_CMBLK && SCG(ptr) != SC_STATIC &&
            SCG(ptr) != SC_DUMMY &&
            !(ALLOCATTRG(sptr) && SCG(SDSCG(sptr)) == SC_DUMMY)) {
          /* Also, we must be sure the pointer, offset,
           * and first descriptor word are initially zero;
           * keep a list of the symbols */
          if (ptr >= stb.firstusym) {
            LOWER_POINTER_LIST(sptr) = lower_pointer_list_head;
            lower_pointer_list_head = sptr;
          }
        }
      }
      break;
    case ST_DESCRIPTOR:
      fill_fixed_array_dtype(dtype);
      break;
    case ST_MEMBER:
      if (DTY(dtype) == TY_ARRAY && IFACEG(sptr) &&
          STYPEG(IFACEG(sptr)) == ST_PROC && ABSTRACTG(IFACEG(sptr))) {
        dtype = get_array_dtype(rank_of_sym(sptr), DTY(dtype + 1));
        DTYPEP(sptr, dtype);
        lower_use_datatype(dtype, 1);
      }

      if (IGNOREG(sptr))
        break;

      if (DTY(dtype) == TY_ARRAY) {
        if ((POINTERG(sptr) || ALLOCG(sptr)) && SDSCG(sptr) &&
            STYPEG(SDSCG(sptr)) != ST_PARAM) {
          fill_pointer_array_dtype(dtype, sptr);
        } else if (ADD_ADJARR(dtype) || ADD_DEFER(dtype)) {
          break;
        } else {
          /* fixed-size datatype */
          fill_fixed_array_dtype(dtype);
        }
      }
      break;
    case ST_CONST:
      break;
    case ST_ALIAS:
      /* if this is an alias for a function and the function
       * return value's name is not the same as the function name
       * then create an alias for the return value that has the
       * same name as the function.
       */
      link = SYMLKG(sptr);
      if (STYPEG(link) == ST_ENTRY) {
        fval = FVALG(link);
        if (fval && NMPTRG(fval) != NMPTRG(sptr)) {
          int retval_sptr = insert_sym(sptr);
          STYPEP(retval_sptr, ST_ALIAS);
          DTYPEP(retval_sptr, DTYPEG(fval));
          SCOPEP(retval_sptr, SCOPEG(fval));
          IGNOREP(retval_sptr, 0);
          SYMLKP(retval_sptr, fval);
        }
      }
      break;
    case ST_LABEL:
      if (!VOLG(sptr))
        RFCNTP(sptr, 0);
      break;
    case ST_PROC:
    case ST_ENTRY:
      fval = FVALG(sptr);
      if (fval) {
        CCSYMP(fval, 1);
      }
      FLANG_FALLTHROUGH;
    default:
      break;
    }
  }
  first_temp = stb.stg_avail;
  first_avail_scalarptr_temp = first_used_scalarptr_temp = NOSYM;
  first_avail_scalar_temp = first_used_scalar_temp = NOSYM;
} /* lower_prepare_symbols */

static void
lower_finish_symbols(void)
{
  int sptr;
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (IGNOREG(sptr))
      continue;
    switch (STYPEG(sptr)) {
    case ST_PARAM:
      if (CCSYMG(sptr))
        break;
      if (ENCLFUNCG(sptr) == 0 ||
          (ENCLFUNCG(sptr) == gbl.currsub && flg.debug)) {
        lower_visit_symbol(sptr);
      }
      break;
    case ST_TYPEDEF:
      /* if this is a typedef for the current routine, export it */
      if (ENCLFUNCG(sptr) == 0 || ENCLFUNCG(sptr) == gbl.currsub) {
        lower_visit_symbol(sptr);
      }
      /* if this is a type descriptor for mod object file, export it */
      else if (SDSCG(sptr) && CLASSG(SDSCG(sptr)) && !PARENTG(sptr)) {
        lower_visit_symbol(sptr);
      }
      break;
    case ST_ARRAY:
    case ST_VAR:
    case ST_IDENT:
    case ST_STRUCT:
      /* if debug, or if contains routines, put out all locals */
      if (HCCSYMG(sptr))
        break;

      if (ENCLFUNCG(sptr) != 0 && !flg.debug)
        break;
      if (!flg.debug && !XBIT(57, 0x20) && gbl.internal != 1)
        break;
      if (LOWER_SYMBOL_REPLACE(sptr))
        break;

      lower_visit_symbol(sptr);
      break;
    case ST_MODULE:
      lower_visit_symbol(sptr);
      break;
    case ST_PROC:
      /* if -x 124 0x1000, and this appeared in an EXTERNAL statement,
       * export it */
      if (XBIT(124, 0x1000)) {
        if (TYPDG(sptr)) {
          lower_visit_symbol(sptr);
        }
      }
      break;
    case ST_BLOCK:
      lower_visit_symbol(sptr);
      break;
    default:
      break;
    }
  }
} /* lower_finish_symbols */

void
lower_pointer_init(void)
{
  int sptr;
  for (sptr = lower_pointer_list_head; sptr > 0;
       sptr = LOWER_POINTER_LIST(sptr)) {
    int ptr, off, desc;
    int lilm, rilm;
      if (STYPEG(sptr) != ST_MEMBER &&
          (XBIT(47, 0x2000000) || !HCCSYMG(sptr))) {
        ptr = MIDNUMG(sptr);
        if (SCG(ptr) != SC_PRIVATE) {
          lilm = plower("oS", "BASE", ptr);
          if (XBIT(49, 0x100)) {
            /* 64-bit pointers */
          } else {
          }
          if (!PTR_TARGETG(sptr)) {
            rilm = lower_null();
          } else {
            rilm = plower("oS", "BASE", PTR_TARGETG(sptr));
          }
          if (!XBIT(49, 0x20000000)) {
            plower("oii", "PST", lilm, rilm);
          } else if (XBIT(49, 0x100)) {
            plower("oii", "KST", lilm, rilm);
          } else {
            plower("oii", "IST", lilm, rilm);
          }
          off = PTROFFG(sptr);
              if (off && STYPEG(off) != ST_PARAM && !ENCLFUNCG(off) &&
                  XBIT(47, 0x2000000)) {
            lilm = plower("oS", "BASE", off);
            if (XBIT(49, 0x100)) {
              /* 64-bit pointers */
              rilm = plower("oS", "KCON", lowersym.intzero);
            } else {
              rilm = plower("oS", "ICON", lowersym.intzero);
            }
            if (XBIT(49, 0x100)) {
              plower("oii", "KST", lilm, rilm);
            } else {
              plower("oii", "IST", lilm, rilm);
            }
          }
        }
      }
      desc = SDSCG(sptr);
      if (desc && STYPEG(desc) != ST_PARAM && !ENCLFUNCG(desc) &&
          SCG(desc) != SC_DUMMY && SCG(desc) != SC_PRIVATE &&
          (XBIT(47, 0x2000000) || !HCCSYMG(sptr))) {
        lilm = plower("oS", "BASE", desc);
        rilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
        lilm = plower("onidi", "ELEMENT", 1, lilm, DTYPEG(desc), rilm);
        rilm = plower("oS", "ICON", lowersym.intzero);
        if (XBIT(68, 1)) {
          plower("oii", "KST", lilm, rilm);
        } else {
          plower("oii", "IST", lilm, rilm);
        }
      }
  }
} /* lower_pointer_init */

/* When prepend_func_result_as_first_arg(semfin.c) has been called for an
 * entry, the FVAL symbol and its descriptor symbol if exist are referred in
 * the entry's dummy arguments.
 * When we are going to identify all result variables of same dtype from
 * different entry points with a single symbol, here we traverse all the dummy
 * arguments, and replace the FVAL symbol and its descriptor symbol with this
 * single symbol and corresponding descriptor symbol.
 */
static void
replace_fval_in_params(SPTR entry, SPTR entrysame)
{
  SPTR fval, fvalsame, newdsc, newdscsame, newarg, newargsame;
  int params, narg, i;

  narg = PARAMCTG(entry);
  if (narg == 0)
    return;
  params = DPDSCG(entry);
  fval = FVALG(entry);
  fvalsame = FVALG(entrysame);
  newdsc = NEWDSCG(fval);
  newarg = NEWARGG(fval);
  /* If the return variable is an adjustable length character without POINTER
   * attribute, its NEWARG keeps SPTR_NULL(see init_newargs in bblock.c), and
   * itself will be put into dummy parameters(see newargs_for_entry in
   * dpm_out.c), so we match and replace the return variable directly.
   */
  if (newarg == SPTR_NULL)
    newarg = fval;
  newdscsame = NEWDSCG(fvalsame);
  newargsame = NEWARGG(fvalsame);
  if (newargsame == SPTR_NULL)
    newargsame = fvalsame;
  for (i = 0; i < narg; i++) {
    int arg = aux.dpdsc_base[params + i];
    if (arg != SPTR_NULL && arg == newarg) {
      aux.dpdsc_base[params + i] = newargsame;
      continue;
    }
    if (arg != SPTR_NULL && arg == newdsc) {
      aux.dpdsc_base[params + i] = newdscsame;
      continue;
    }
  }
}

/* replace the symbol used in the ast of type A_ID taking advantage of the hash
 * in the AST table
 */
static void
replace_sptr_in_ast(SPTR sptr)
{
  SPTR newsptr;
  int ast;

  if (sptr <= NOSYM) {
    return;
  }
  newsptr = LOWER_SYMBOL_REPLACE(sptr);
  if (newsptr <= NOSYM) {
    return;
  }
  ast = mk_id(sptr);
  A_SPTRP(ast, newsptr);
}

static inline void
add_replace_map(SPTR sptr, SPTR newsptr)
{
  if (sptr <= NOSYM || newsptr <= NOSYM) {
    return;
  }
  LOWER_SYMBOL_REPLACE(sptr) = newsptr;
}

/* replace the fval symbol and associated symbols when the fval symbol is
 * pointer or array
 */
static void
replace_fval_in_ast(SPTR fval, SPTR fvalsame)
{
  SPTR var, var_same;

  replace_sptr_in_ast(fval);

  var = MIDNUMG(fval);
  var_same = MIDNUMG(fvalsame);
  add_replace_map(var, var_same);
  replace_sptr_in_ast(var);

  var = PTROFFG(fval);
  var_same = PTROFFG(fvalsame);
  add_replace_map(var, var_same);
  replace_sptr_in_ast(var);

  var = DESCRG(fval);
  var_same = DESCRG(fvalsame);
  add_replace_map(var, var_same);
  replace_sptr_in_ast(var);

  var = SDSCG(fval);
  var_same = SDSCG(fvalsame);
  add_replace_map(var, var_same);
  replace_sptr_in_ast(var);

  var = CVLENG(fval);
  var_same = CVLENG(fvalsame);
  add_replace_map(var, var_same);
  replace_sptr_in_ast(var);
}

extern int pghpf_type_sptr;
extern int pghpf_local_mode_sptr;

void
lower_init_sym(void)
{
  int sym, dtype;
  LOGICAL from_func;

  lowersym.sc = SC_LOCAL;
  lowersym.parallel_depth = 0;
  lowersym.task_depth = 0;
  lower_linearized_dtypes = FALSE;
  lower_make_all_descriptors();
  /* reassign member addresses to account for distributed derived
   * type members, late additions of section descriptors, pointers, etc. */
  for (dtype = 0; dtype < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
    if (DTY(dtype) == TY_DERIVED) {
      chkstruct(dtype);
    }
  }
  /* allocate the table of datatypes */
  last_datatype_used = stb.dt.stg_avail;
  NEW(datatype_used, char, last_datatype_used);
  BZERO(datatype_used, char, last_datatype_used);
  NEW(datatype_output, char, last_datatype_used);
  BZERO(datatype_output, char, last_datatype_used);
  if (gbl.internal < 2) {
    lowersym.acount = 0;
    lowersym.Ccount = 0;
  }
  lowersym.ptr0 = lowersym.ptr0c = 0;
  lowersym.license = lowersym.localmode = 0;
  lowersym.intzero = lower_getintcon(0);
  lowersym.intone = lower_getintcon(1);
  lowersym.realzero = stb.flt0;
  lowersym.dblezero = stb.dbl0;
  lowersym.quadzero = stb.quad0;
  lowersym.ptrnull = lower_getnull();
  if (XBIT(68, 0x1)) {
    lowersym.bnd.zero = stb.k0;
    lowersym.bnd.one = stb.k1;
    lowersym.bnd.max = lower_getiszcon(0x7fffffffffffffff);
    lowersym.bnd.dtype = DT_INT8;
    lowersym.bnd.load = "KLD";
    lowersym.bnd.store = "KST";
    lowersym.bnd.con = "KCON";
    lowersym.bnd.add = "KADD";
    lowersym.bnd.sub = "KSUB";
    lowersym.bnd.mul = "KMUL";
    lowersym.bnd.div = "KDIV";
  } else {
    lowersym.bnd.zero = stb.i0;
    lowersym.bnd.one = stb.i1;
    lowersym.bnd.max = lower_getintcon(0x7fffffff);
    lowersym.bnd.dtype = DT_INT;
    lowersym.bnd.load = "ILD";
    lowersym.bnd.store = "IST";
    lowersym.bnd.con = "ICON";
    lowersym.bnd.add = "IADD";
    lowersym.bnd.sub = "ISUB";
    lowersym.bnd.mul = "IMUL";
    lowersym.bnd.div = "IDIV";
  }
  lowersym.loc = lowersym.exit = lowersym.alloc = lowersym.alloc_chk =
      lowersym.ptr_alloc = lowersym.dealloc = lowersym.dealloc_mbr =
          lowersym.lmalloc = lowersym.lfree = lowersym.calloc =
              lowersym.ptr_calloc = lowersym.auto_alloc = lowersym.auto_calloc =
                  lowersym.auto_dealloc = 0;
  if (XBIT(70, 2)) {
    /* add subchk subroutine */
    if (XBIT(68, 0x1))
      lowersym.sym_subchk =
          lower_makefunc(mkRteRtnNm(RTE_subchk64), DT_INT, TRUE);
    else
      lowersym.sym_subchk =
          lower_makefunc(mkRteRtnNm(RTE_subchk), DT_INT, TRUE);
    lowersym.intmax = lower_getintcon(0x7fffffff);
  }
  if (XBIT(70, 4)) {
    /* add ptrchk subroutine */
    lowersym.sym_ptrchk = lower_makefunc(mkRteRtnNm(RTE_ptrchk), DT_INT, TRUE);
  }

  lowersym.oldsymavl = stb.stg_avail;
  lowersym.sched_dtype = 0;
  lowersym.scheds_dtype = 0;

  STG_ALLOC_SIDECAR(stb, lsymlists);
  lower_pointer_list_head = -1;
  lower_refd_list_head = NOSYM;
  lower_prepare_symbols();

  private_addr = 0;
  for (sym = lower_refd_list_head; sym > NOSYM; sym = LOWER_REFD_LIST(sym)) {
    if (SCG(sym) != SC_PRIVATE)
      sym_is_refd(sym);
    else {
      /* Assume the descriptor, pointer, and offset variables have the
       * same alignment requirements; therefore, don't bother with
       * explicitly aligning their offsets as sym_is_refd() does.
       * NOTE:  Assigning offsets for these variables is performed
       *        here instead of in sym_is_refd() since  sym_is_refd()
       *        ignores private variables (doesn't set their REF
       *        bits).  The backend will adjust the offsets per
       *        the target's first private address.
       */
      ADDRESSP(sym, private_addr);
      private_addr += size_of(DTYPEG(sym));
      REFP(sym, 1);
    }
  }

  /* any variables in locals or statics list need to be exported */
  for (sym = gbl.locals; sym > NOSYM; sym = SYMLKG(sym)) {
    lower_visit_symbol(sym);
  }
  for (sym = gbl.statics; sym > NOSYM; sym = SYMLKG(sym)) {
    lower_visit_symbol(sym);
  }

  /* If this symbol is used in a contained subprogram but not in the
   * contained subprogram's host, then the symbol in the host will not
   * automatically be lowered.
   */
  if (pghpf_type_sptr)
    lower_visit_symbol(pghpf_type_sptr);
  if (pghpf_local_mode_sptr)
    lower_visit_symbol(pghpf_local_mode_sptr);

  /* prepare stack for use */
  stack_top = 0;
  stack_size = 100;
  NEW(stack, int, stack_size);

  from_func = gbl.rutype == RU_SUBR && gbl.entries > NOSYM && FVALG(gbl.entries);
  /* look for ENTRY points; make all ENTRY points with the same
   * return type use the same FVAL symbol */
  if (from_func || gbl.rutype == RU_FUNC) {
    int ent, esame;
    for (ent = gbl.entries; ent > NOSYM; ent = SYMLKG(ent)) {
      for (esame = gbl.entries; esame != ent; esame = SYMLKG(esame)) {
        int fval, fvalsame;
        fval = FVALG(ent);
        fvalsame = FVALG(esame);
        if (fval && fvalsame && fval != fvalsame &&
            same_dtype(DTYPEG(fval), DTYPEG(fvalsame))) {
          /* esame is the earlier entry point, make ent use the
           * FVAL of esame */
          LOWER_SYMBOL_REPLACE(fval) = fvalsame;
          replace_fval_in_params(ent, esame);
          FVALP(ent, fvalsame);
          replace_fval_in_ast(fval, fvalsame);
          break; /* leave inner loop */
        }
      }
    }
  }

  /* if an internal routine, change the entry points of the containing
   * routine to ST_PROC */
  if (gbl.internal > 1) {
    for (sym = lowersym.first_outer_sym; sym < lowersym.last_outer_sym; ++sym) {
      if (STYPEG(sym) == ST_ENTRY) {
        STYPEP(sym, ST_PROC);
      }
    }
  }
  lower_argument_size = 100;
  NEW(lower_argument, int, lower_argument_size);
  BZERO(lower_argument, int, lower_argument_size);
} /* lower_init_sym */

void
lower_finish_sym(void)
{
  FREE(lower_argument);
  lower_argument = NULL;
  lower_argument_size = 0;
  FREE(stack);
  stack = NULL;
  STG_DELETE_SIDECAR(stb, lsymlists);
  FREE(datatype_output);
  datatype_output = NULL;
  FREE(datatype_used);
  datatype_used = NULL;
} /* lower_finish_sym */

typedef struct initem {
  char *name, *cname, *filename;
  struct initem *next;
  long offset, objoffset;
  int level, which, staticbase, size;
} INITEM;

static INITEM *inlist = NULL, *inlistend = NULL;
#define PERM_AREA 8
#define STASH(p) strcpy(getitem(PERM_AREA, strlen(p) + 1), p);

void
lower_add_func_call(int level, long objoffset, long offset, char *name,
                    char *cname, char *filename, char which, int staticbase,
                    int size)
{
  INITEM *p;
  p = (INITEM *)getitem(PERM_AREA, sizeof(INITEM));
  p->level = level;
  p->offset = offset;
  p->objoffset = objoffset;
  p->name = STASH(name);
  p->cname = STASH(cname);
  p->filename = STASH(filename);
  p->which = which;
  p->staticbase = staticbase;
  p->size = size;
  p->next = NULL;
  if (inlistend) {
    inlistend->next = p;
  } else {
    inlist = p;
  }
  inlistend = p;
} /* lower_add_func_call */

static int saveblockname = 0;

void
create_static_base(int blockname)
{
  saveblockname = blockname;
} /* create_static_base */

static void
putvline(const char *n, ISZ_T v)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, "%s:%" ISZ_PF "d\n", n, v);
  } else
#endif
    fprintf(lowersym.lowerfile, "%c:%" ISZ_PF "d\n", n[0], v);
} /* putvline */

static void
putbit(const char *bitname, int bit)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, " %s%c", bitname, bit ? '+' : '-');
  } else
#endif
    fprintf(lowersym.lowerfile, " %c%c", bitname[0], bit ? '+' : '-');
} /* putbit */

static void
putsym(const char *valname, int sym)
{
  if (valname) {
#if DEBUG
    if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
      fprintf(lowersym.lowerfile, " %s:", valname);
    } else
#endif
      fprintf(lowersym.lowerfile, " %c:", valname[0]);
  } else {
    fprintf(lowersym.lowerfile, " ");
  }
#if DEBUG
  if (DBGBIT(47, 8) && sym > NOSYM) {
    fprintf(lowersym.lowerfile, "%s", getprint(sym));
  } else
#endif
    fprintf(lowersym.lowerfile, "%d", sym);
} /* putsym */

static void
putval(const char *valname, ISZ_T val)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, " %s:%" ISZ_PF "d", valname, val);
  } else
#endif
    fprintf(lowersym.lowerfile, " %c:%" ISZ_PF "d", valname[0], val);
} /* putval */

static void
putival(const char *valname, int val)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, "%s:%d", valname, val);
  } else
#endif
    fprintf(lowersym.lowerfile, "%c:%d", valname[0], val);
} /* putival */

static void
putlval(const char *valname, long val)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, " %s:%ld", valname, val);
  } else
#endif
    fprintf(lowersym.lowerfile, " %c:%ld", valname[0], val);
} /* putlval */

static void
putpair(int first, int second)
{
#if DEBUG
  if (DBGBIT(47, 8)) {
    fprintf(lowersym.lowerfile, " %s", getprint(first));
    fprintf(lowersym.lowerfile, ":%s", getprint(second));
  } else
#endif
    fprintf(lowersym.lowerfile, " %d:%d", first, second);
} /* putpair */

static void
puthex(int hex)
{
  fprintf(lowersym.lowerfile, " %x", hex);
} /* puthex */

static void
putstring(const char *s)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, " %s", s);
  } else
#endif
    fprintf(lowersym.lowerfile, " %c", s[0]);
} /* putstring */

static void
putwhich(const char *s, const char *ss)
{
#if DEBUG
  if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
    fprintf(lowersym.lowerfile, " %s", s);
  } else
#endif
    fprintf(lowersym.lowerfile, " %s", ss);
} /* putwhich */

/** \brief Print file table information
 */
void
lower_fileinfo(void)
{
  int fihx;
  const char *dirname, *filename, *funcname, *fullname;

  fihx = curr_findex;

  for (; fihx < fihb.stg_avail; ++fihx) {
    dirname = FIH_DIRNAME(fihx);
    if (dirname == NULL)
      dirname = "";
    filename = FIH_FILENAME(fihx);
    if (filename == NULL)
      filename = "";
    funcname = FIH_FUNCNAME(fihx);
    if (funcname == NULL)
      funcname = "";
    fullname = FIH_FULLNAME(fihx);
    if (fullname == NULL)
      fullname = "";

    fprintf(lowersym.lowerfile,
            "fihx:%d tag:%d parent:%d flags:%d "
            "lineno:%d srcline:%d level:%d next:%d %" GBL_SIZE_T_FORMAT
            ":%s %" GBL_SIZE_T_FORMAT ":%s %" GBL_SIZE_T_FORMAT
            ":%s %" GBL_SIZE_T_FORMAT ":%s\n",
            fihx, FIH_FUNCTAG(fihx), FIH_PARENT(fihx), FIH_FLAGS(fihx),
            FIH_LINENO(fihx), FIH_SRCLINE(fihx), FIH_LEVEL(fihx),
            FIH_NEXT(fihx), strlen(dirname), dirname, strlen(filename),
            filename, strlen(funcname), funcname, strlen(fullname), fullname);
  }

  lower_fileinfo_llvm();
  curr_findex = fihx;

} /* lower_fileinfo */

/* Note: If you make any change to this function, please also update
          stb_lower_sym_header ()
*/
void
lower_sym_header(void)
{
  ISZ_T bss_addr;
  INITEM *p;
  static int first_time = 1;

  /* last chance to fix up symbols and datatypes */
  lower_finish_symbols();

  if (first_time) {
    first_time = 0;
    /* put out any saved inlining information */
    for (p = inlist; p; p = p->next) {
      putival("inline", p->level);
      putlval("offset", p->offset);
      putval("which", p->which);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s", strlen(p->name),
              p->name);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s",
              strlen(p->cname), p->cname);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s",
              strlen(p->filename), p->filename);
      putlval("objoffset", p->objoffset);
      putval("base", p->staticbase);
      putval("size", p->size);
      fprintf(lowersym.lowerfile, "\n");
    }
    fprintf(lowersym.lowerfile, "ENDINLINE\n");
  }

  /* put out header lines */
  fprintf(lowersym.lowerfile, "TOILM version %d/%d\n", VersionMajor,
          VersionMinor);
  if (gbl.internal == 1 && gbl.empty_contains)
    putvline("Internal", 0);
  else 
    putvline("Internal", gbl.internal);
  if (gbl.internal > 1) {
    putvline("Outer", lowersym.outersub);
    putvline("First", stb.firstusym);
  }
  putvline("Symbols", stb.stg_avail - 1);
  putvline("Datatypes", stb.dt.stg_avail - 1);
  bss_addr = get_bss_addr();
  putvline("BSS", bss_addr);
  putvline("GBL", gbl.saddr);
  putvline("LOC", gbl.locaddr);
  putvline("STATICS", gbl.statics);
  putvline("LOCALS", gbl.locals);
  putvline("PRIVATES", private_addr);
  if (saveblockname) {
    putvline("GNAME", saveblockname);
  }

  stb_lower_sym_header();
} /* lower_sym_header */

static void
set_common_size(int common)
{
  int elsym, lastelsym;
  ISZ_T offset = 0;
  ISZ_T size = 0;
  int aln_n = 1;
  lastelsym = 0;

  /* for equivalence symbols, save the difference between
   * their starting address and the starting address of
   * their first non-EQV 'soc' overlap symbol */
  for (elsym = CMEMFG(common); elsym > NOSYM; elsym = SYMLKG(elsym)) {
    if (EQVG(elsym) && SOCPTRG(elsym)) {
      int socptr;
      for (socptr = SOCPTRG(elsym); socptr; socptr = SOC_NEXT(socptr)) {
        int socsptr = SOC_SPTR(socptr);
        if (!EQVG(socsptr)) {
          /* compute difference with nonEQV symbol */
          ISZ_T diff = ADDRESSG(elsym) - ADDRESSG(socsptr);
          ADDRESSP(elsym, diff);
          break;
        }
      }
    }
  }
  for (elsym = CMEMFG(common); elsym > NOSYM; elsym = SYMLKG(elsym)) {
    int dtype;
    lastelsym = elsym;
    dtype = DTYPEG(elsym);
    if (STYPEG(elsym) == ST_IDENT || STYPEG(elsym) == ST_UNKNOWN) {
      switch (DTY(dtype)) {
      case TY_STRUCT:
        STYPEP(elsym, ST_STRUCT);
        break;
      case TY_UNION:
        STYPEP(elsym, ST_UNION);
        break;
      case TY_DERIVED:
        STYPEP(elsym, ST_VAR);
        break;
      case TY_ARRAY:
        STYPEP(elsym, ST_ARRAY);
        break;
      default:
        STYPEP(elsym, ST_VAR);
        break;
      }
    }
    REFP(elsym, 1);
    if (!EQVG(elsym)) {
      int addr;
      ISZ_T msz;
      addr = alignment_of_var(elsym);
      offset = ALIGN(offset, addr);
      ADDRESSP(elsym, offset);
      msz = size_of_var(elsym);
      msz = pad_cmn_mem(elsym, msz, &aln_n);
      offset += msz;
      if (offset > size) {
        size = offset;
      }
    }
    /* note: common may not be volatile but a member may */
    if (VOLG(common))
      VOLP(elsym, 1);
  }
  for (elsym = CMEMFG(common); elsym > NOSYM; elsym = SYMLKG(elsym)) {
    if (EQVG(elsym)) {
      ISZ_T end_of_eqv;
      int socptr;
      /* look at the first non-EQV overlap symbol, add difference of their
       * old addresses to the new address of the overlap symbol,
       * to be the new address of this symbol */
      for (socptr = SOCPTRG(elsym); socptr; socptr = SOC_NEXT(socptr)) {
        int socsptr = SOC_SPTR(socptr);
        if (!EQVG(socsptr)) {
          /* compute difference with nonEQV symbol */
          ISZ_T diff = ADDRESSG(elsym) + ADDRESSG(socsptr);
          ADDRESSP(elsym, diff);
          break;
        }
      }
      end_of_eqv = ADDRESSG(elsym) + size_of_var(elsym);
      if (end_of_eqv > size)
        size = end_of_eqv;
    }
  }
  if (size == 0) {
    /* zero-size common block, ugh, add a nonzero-size element */
    NEWSYM(elsym);
    DTYPEP(elsym, DT_INT);
    SCP(elsym, SC_CMBLK);
    STYPEP(elsym, ST_VAR);
    CCSYMP(elsym, 1);
    SCOPEP(elsym, stb.curr_scope);
    SYMLKP(elsym, NOSYM);
    if (INTERNALG(common))
      INTERNALP(elsym, 1);
    if (lastelsym) {
      SYMLKP(lastelsym, elsym);
    } else {
      CMEMFP(common, elsym);
    }
    CMEMLP(common, elsym);
    size = size_of(DT_INT);
  }
  SIZEP(common, size);
} /* set_common_size */

/** \brief Mark all commons to be exported, and fill in sizes for
    compiler commons that are unfinished.
 */
void
lower_common_sizes(void)
{
  int sptr, s, inmod;
  for (sptr = gbl.cmblks; sptr != NOSYM; sptr = SYMLKG(sptr)) {
    /* set 'visit' bit for all commons and all members */
    VISITP(sptr, 1);
    DTYPEP(sptr, 0);
    inmod = SCOPEG(sptr);
    if (inmod && STYPEG(inmod) == ST_ALIAS)
      inmod = SCOPEG(inmod);
    if (inmod && STYPEG(inmod) == ST_MODULE)
      lower_visit_symbol(inmod);
    set_common_size(sptr);
    if (IGNOREG(sptr))
      continue;
    for (s = CMEMFG(sptr); s != NOSYM; s = SYMLKG(s)) {
      lower_visit_symbol(s);
    }
    /* propagate altnames of common blocks */
    if (ALTNAMEG(sptr))
      lower_visit_symbol(ALTNAMEG(sptr));
  }
} /* lower_common_sizes */

static void
check_additional_common(int newcom)
{
  int hash, link;
  int s, lasts;

  /* if no members, already done */
  if (CMEMFG(newcom) == 0)
    return;

  /* get hash address of this name */
  HASH_ID(hash, SYMNAME(newcom), strlen(SYMNAME(newcom)));

  /* look through all symbols on that hash list, look for another
   * common block of the same name with VISIT bit set */
  for (link = stb.hashtb[hash]; link; link = HASHLKG(link)) {
    if (link != newcom && NMPTRG(link) == NMPTRG(newcom) &&
        STYPEG(link) == ST_CMBLK && VISITG(link))
      break;
  }

  if (link == 0) {
    /* there is no such common block; we must instead just treat
     * this common block as the only one of its name */
    VISITP(newcom, 1);
    lower_use_datatype(DTYPEG(newcom), 1);
    set_common_size(newcom);
    for (s = CMEMFG(newcom); s != NOSYM; s = SYMLKG(s)) {
      lower_visit_symbol(s);
    }
    return;
  }

  /* here, link is a common with the same name.
   * fill in the address fields if necessary, then
   * set the 'equivalence' bit for the members and add them
   * to the original common block as equivalences.
   * Theoretically, this should work whether the new names and
   * types are the same as the original or not. */

  set_common_size(newcom);

  lasts = 0;
  for (s = CMEMFG(newcom); s != NOSYM; lasts = s, s = SYMLKG(s)) {
    lower_visit_symbol(s);
    EQVP(s, 1);
    CMBLKP(s, link);
  }
  /* last common member should point to new common list */
  SYMLKP(CMEMLG(link), CMEMFG(newcom));
  CMEMLP(link, lasts);

  /* unset visit flag for the 'equivalenced' common */
  VISITP(newcom, 0);
  /* remove all member pointers */
  CMEMFP(newcom, 0);
  CMEMLP(newcom, 0);
} /* check_additional_common */

/* determine whether to make this function return value variable
 * a local or a dummy */
static int
makefvallocal(int rutype, int fval)
{
  int dtype;
  /* if this was turned into a subroutine, make the fval a dummy */
  if (rutype != RU_FUNC)
    return 0;
  /* if the fval is a POINTER variable, make local */
  if (POINTERG(fval))
    return 1;
  dtype = DTYPEG(fval);
  /* if the datatype is a structure, derived type, make a dummy */
  if ((DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_DERIVED))
    return 0;
  /* if the datatype is character, make a dummy */
  if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR)
    return 0;
  /* if the datatype is complex, make a dummy */
  if (DTY(dtype) == TY_CMPLX || DTY(dtype) == TY_DCMPLX ||
      DTY(dtype) == TY_QCMPLX) {
    return 0;
  }
  /* else, make local */
  return 1;
} /* makefvallocal */

void
lower_visit_symbol(int sptr)
{
  int socptr, dtype, params, i, fval, inmod, stype;
  if (LOWER_SYMBOL_REPLACE(sptr)) {
    lower_visit_symbol(LOWER_SYMBOL_REPLACE(sptr));
    lerror("visit symbol %s(%d) which was replaced by %s(%d)", SYMNAME(sptr),
           sptr, SYMNAME(LOWER_SYMBOL_REPLACE(sptr)),
           LOWER_SYMBOL_REPLACE(sptr));
    return;
  }
  if (VISITG(sptr))
    return;

  if ((STYPEG(sptr) == ST_ALIAS || STYPEG(sptr) == ST_PROC ||
      STYPEG(sptr) == ST_ENTRY) &&
      SEPARATEMPG(sptr) &&
      STYPEG(SCOPEG(sptr)) == ST_MODULE)
    INMODULEP(sptr, 1);

  VISITP(sptr, 1);
  dtype = DTYPEG(sptr);
  stype = STYPEG(sptr);
  if (stype == ST_PROC || stype == ST_ENTRY) {
    if (DTY(dtype) == TY_ARRAY) {
      dtype = DTY(dtype + 1);
    }
  }
  if (lower_linearized_dtypes || DTY(dtype) != TY_ARRAY || !XBIT(52, 4) ||
      !LNRZDG(sptr)) {
    /* linearized array data types are 'used' later */
    lower_use_datatype(dtype, 1);
  }
  switch (stype) {
  case ST_IDENT:
  case ST_UNKNOWN:
    if (dtype) {
      switch (DTY(dtype)) {
      case TY_STRUCT:
        STYPEP(sptr, ST_STRUCT);
        break;
      case TY_UNION:
        STYPEP(sptr, ST_UNION);
        break;
      case TY_DERIVED:
        STYPEP(sptr, ST_VAR);
        break;
      case TY_ARRAY:
        STYPEP(sptr, ST_ARRAY);
        break;
      default:
        STYPEP(sptr, ST_VAR);
        break;
      }
    }
    if (SCG(sptr) == SC_NONE) {
      SCP(sptr, SC_LOCAL);
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  switch (STYPEG(sptr)) {
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_VAR:
  case ST_STRUCT:
  case ST_UNION:
    if (SCG(sptr) == SC_CMBLK) {
      /* mark the whole common block as visited */
      int common;
      common = CMBLKG(sptr);
      if (VISITG(common) == 0)
        lower_visit_symbol(common);
    }
    /* does it overlap with anything (equivalence overlaps?) */
    for (socptr = SOCPTRG(sptr); socptr; socptr = SOC_NEXT(socptr)) {
      int overlap;
      overlap = SOC_SPTR(socptr);
      lower_visit_symbol(overlap);
    }
    if (MIDNUMG(sptr))
      lower_visit_symbol(MIDNUMG(sptr));
    if (PTROFFG(sptr))
      lower_visit_symbol(PTROFFG(sptr));
    if (SDSCG(sptr))
      lower_visit_symbol(SDSCG(sptr));
    if (CVLENG(sptr))
      lower_visit_symbol(CVLENG(sptr));
    if (ALTNAMEG(sptr))
      lower_visit_symbol(ALTNAMEG(sptr));
    break;
  case ST_IDENT:
    /* not classified as ID or anything else as yet */
    if (SCG(sptr) == SC_CMBLK) {
      /* mark the whole common block as visited */
      int common;
      common = CMBLKG(sptr);
      if (VISITG(common) == 0)
        lower_visit_symbol(common);
    }
    if (MIDNUMG(sptr))
      lower_visit_symbol(MIDNUMG(sptr));
    break;
  case ST_ENTRY:
    fval = FVALG(sptr);
    if (fval) {
      lower_visit_symbol(FVALG(sptr));
      /* semant marks class of function return value temp as DUMMY so it
       * won't be deleted by the optimizer; pgftn wants it to be LOCAL;
       * if this is a real subroutine, it was converted from a function,
       * so leave it as dummy */
      if (SCG(fval) == SC_BASED) {
        /* ADDRESS field was used to hold symtab pointers
         * for optimizer */
        ADDRESSP(fval, 0);
      } else {
        if (makefvallocal(gbl.rutype, fval)) {
          SCP(fval, SC_LOCAL);
          if (is_iso_cptr(DTYPEG(fval))) {
            DTYPEP(fval, DT_CPTR);
          }
        } else {
          SCP(fval, SC_DUMMY);
        }
      }
    }
    params = DPDSCG(sptr);
    for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
      int param = aux.dpdsc_base[params + i];
      if (param) {
        lower_visit_symbol(param);
      }
    }
    inmod = SCOPEG(sptr);
    if (inmod && STYPEG(inmod) == ST_ALIAS) {
      inmod = SCOPEG(inmod);
    }
    if (inmod && STYPEG(inmod) == ST_MODULE) {
      lower_visit_symbol(inmod);
    }
    if (ALTNAMEG(sptr))
      lower_visit_symbol(ALTNAMEG(sptr));
    break;
  case ST_PROC:
    inmod = SCOPEG(sptr);
    if (inmod && STYPEG(inmod) == ST_ALIAS)
      inmod = SCOPEG(inmod);
    if (inmod && STYPEG(inmod) == ST_MODULE)
      lower_visit_symbol(inmod);
    if (ALTNAMEG(sptr))
      lower_visit_symbol(ALTNAMEG(sptr));
    if (SCG(sptr) == SC_NONE ||
        (SCG(sptr) == SC_EXTERN && VISITG(sptr) &&
         (inmod || INMODULEG(sptr) ||
          (TYPDG(sptr) && DCLDG(sptr)) /* interface */))) {
      fval = FVALG(sptr);
      if (fval) {
        lower_visit_symbol(FVALG(sptr));
        if (SCG(fval) == SC_BASED) {
          ADDRESSP(fval, 0);
        } else {
          SCP(fval, SC_DUMMY);
        }
      }
      params = DPDSCG(sptr);
      for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
        int param = aux.dpdsc_base[params + i];
        if (param) {
          lower_visit_symbol(param);
        }
      }
    }
    break;
  case ST_CONST:
    switch (DTY(DTYPEG(sptr))) {
    case TY_PTR:
      if (CONVAL1G(sptr))
        lower_visit_symbol(CONVAL1G(sptr));
      break;
    case TY_DCMPLX:
    case TY_QCMPLX:
      lower_visit_symbol(CONVAL1G(sptr));
      lower_visit_symbol(CONVAL2G(sptr));
      break;
    case TY_HOLL:
      /* symbol table ptr of char constant */
      lower_use_datatype(DTYPEG(CONVAL1G(sptr)), 1);
      break;
    }
    break;
  case ST_CMBLK:
    /* since all common blocks are visited by lower_common_sizes,
     * this should only be reached when there is another common block
     * of the same name, such as for inlined routines, interface blocks,
     * or the like.  In any case, add this common also, making sure
     * it has a size and so on */
    check_additional_common(sptr);
    if (ALTNAMEG(sptr))
      lower_visit_symbol(ALTNAMEG(sptr));
    break;
  case ST_PARAM:
    if (!TY_ISWORD(DTY(DTYPEG(sptr)))) {
      lower_visit_symbol(CONVAL1G(sptr));
    }
    break;
  case ST_BLOCK:
    if (STARTLABG(sptr))
      lower_visit_symbol(STARTLABG(sptr));
    if (ENDLABG(sptr))
      lower_visit_symbol(ENDLABG(sptr));
    if (PARUPLEVELG(sptr))
      lower_visit_symbol(PARUPLEVELG(sptr));
    break;
  default:
    break;
  }

  if (SCG(sptr) == SC_DUMMY) {
    int origdummy;
    origdummy = NEWARGG(sptr);
    if (origdummy) {
      lower_visit_symbol(origdummy);
    }
  }
} /* lower_visit_symbol */

/*
 * return FALSE if this symbol is from a module that was implicitly 'used'
 */
static LOGICAL
notimplicit(int sptr)
{
  int s;
  s = SCOPEG(sptr);
  if (!s)
    return TRUE;
  if (STYPEG(s) != ST_MODULE)
    return TRUE;
  if (strcmp(SYMNAME(s), "cudadevice") == 0)
    return FALSE;
  if (strcmp(SYMNAME(s), "cudafor") == 0)
    return FALSE;
  if (strcmp(SYMNAME(s), "cudafor_la") == 0)
    return FALSE;
  return TRUE;
} /* notimplicit */

void
lower_check_generics(void)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_USERGENERIC) {
      int desc;
      if (XBIT(57, 0x20) && notimplicit(sptr)) {
        VISITP(sptr, 1);
        lower_use_datatype(DTYPEG(sptr), 1);
        for (desc = GNDSCG(sptr); desc; desc = SYMI_NEXT(desc)) {
          int s = SYMI_SPTR(desc);
          if (STYPEG(s) != ST_MODPROC) {
            lower_visit_symbol(s);
          }
        }
      } else {
        VISITP(sptr, 0);
        /* look for any actuals that were used */
        for (desc = GNDSCG(sptr); desc; desc = SYMI_NEXT(desc)) {
          int s = SYMI_SPTR(desc);
          if (s && CLASSG(sptr)) {
            VISITP(s, 1);
          }
          if (VISITG(s)) {
            VISITP(sptr, 1);
            lower_use_datatype(DTYPEG(sptr), 1);
            break;
          }
        }
      }
    }
  }
} /* lower_check_generics */

/** \brief For contained subprograms, mark all the regular symbols
    of the host subprogram
 */
void
lower_outer_symbols(void)
{
  int sptr;
  for (sptr = lowersym.first_outer_sym; sptr < lowersym.last_outer_sym;
       ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
    case ST_DESCRIPTOR:
    case ST_VAR:
    case ST_UNION:
    case ST_STRUCT:
    case ST_PLIST:
      if (!IGNOREG(sptr) &&
          (LOWER_SYMBOL_REPLACE(sptr) == 0))
        lower_visit_symbol(sptr);
      break;
    default:
      break;
    }
  }
} /* lower_outer_symbols */

void
lower_use_datatype(int dtype, int usage)
{
  int ndim, i, sptr, zbase, numelm;
  if (dtype <= 0)
    return;
  if (dtype < last_datatype_used) {
    if (datatype_used[dtype]) {
      datatype_used[dtype] |= usage;
      return;
    }
    datatype_used[dtype] = usage;
  }

  switch (DTY(dtype)) {
  case TY_PTR:
    if (dtype != DT_ADDR) {
      /* pointer datatype internal to lower */
      lower_use_datatype(DTY(dtype + 1), 1);
    } else {
      datatype_used[dtype] = 0;
      if (XBIT(49, 0x100)) { /* 64-bit pointers */
        lower_use_datatype(DT_INT8, 1);
      } else {
        lower_use_datatype(DT_INT, 1);
      }
    }
    break;
  case TY_ARRAY:
    lower_use_datatype(DTY(dtype + 1), 1);
    ndim = ADD_NUMDIM(dtype);
    for (i = 0; i < ndim; ++i) {
      int lb, ub, mpy;
      lb = ADD_LWAST(dtype, i);
      ub = ADD_UPAST(dtype, i);
      if (lb == 0) {
        lb = lowersym.intone;
      } else if (A_TYPEG(lb) == A_ID || A_TYPEG(lb) == A_CNST) {
        lb = A_SPTRG(lb);
      } else {
        lb = lowersym.intone;
      }
      lower_visit_symbol(lb);
      if (ub != 0) {
        if (A_TYPEG(ub) == A_ID || A_TYPEG(ub) == A_CNST) {
          ub = A_SPTRG(ub);
          lower_visit_symbol(ub);
        } else {
          ub = 0;
        }
      }
      if (ADD_DEFER(dtype)) {
        lb = ADD_LWBD(dtype, i);
        ub = ADD_UPBD(dtype, i);
        if (lb != 0) {
          if (A_TYPEG(lb) == A_ID || A_TYPEG(lb) == A_CNST) {
            lb = A_SPTRG(lb);
            lower_visit_symbol(lb);
          }
        }
        if (ub != 0) {
          if (A_TYPEG(ub) == A_ID || A_TYPEG(ub) == A_CNST) {
            ub = A_SPTRG(ub);
            lower_visit_symbol(ub);
          }
        }
      }
      mpy = ADD_MLPYR(dtype, i);
      if (mpy != 0) {
        if (A_TYPEG(mpy) == A_ID || A_TYPEG(mpy) == A_CNST) {
          mpy = A_SPTRG(mpy);
          lower_visit_symbol(mpy);
        }
      }
    }
    zbase = ADD_ZBASE(dtype);
    if (zbase == 0) {
      zbase = 0;
    } else {
      if (A_TYPEG(zbase) == A_ID || A_TYPEG(zbase) == A_CNST) {
        zbase = A_SPTRG(zbase);
        lower_visit_symbol(zbase);
      }
    }
    numelm = ADD_NUMELM(dtype);
    if (numelm != 0) {
      if (A_TYPEG(numelm) == A_ID || A_TYPEG(numelm) == A_CNST) {
        numelm = A_SPTRG(numelm);
        lower_visit_symbol(numelm);
      }
    }
    break;
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    /* mark all members */
    for (sptr = DTY(dtype + 1); sptr > NOSYM; sptr = SYMLKG(sptr)) {
      lower_visit_symbol(sptr);
    }
    /* mark tag (structure name) */
    if (DTY(dtype + 3))
      lower_visit_symbol(DTY(dtype + 3));
    break;
  case TY_PROC: {
    int restype = DTY(dtype + 1);
    if (is_array_dtype(restype)) {
      /* array result types must be lowered later to avoid
       * lowering errors, but don't neglect the element type
       */
      restype = array_element_dtype(restype);
    }
    if (restype > 0)
      lower_use_datatype(restype, 1);
  }
    if (gbl.stbfil && DTY(dtype + 2)) {
      int iface = DTY(dtype + 2);
      int fval = DTY(dtype + 5);
      int params = DPDSCG(iface);
      if (STYPEG(iface) == ST_ALIAS) {
        iface = SYMLKG(iface);
        fval = FVALG(iface);
        params = DPDSCG(iface);
      }
      if (STYPEG(iface) == ST_MODPROC) {
        if (SCOPEG(iface) == gbl.currsub || ENCLFUNCG(iface) == gbl.currsub)
          break;
        if (ENCLFUNCG(iface) == ENCLFUNCG(gbl.currsub))
          break;
      }
      llvm_iface_flag = TRUE;
      lower_visit_symbol(iface);
      for (i = 0; i < (int)(PARAMCTG(iface)); ++i) {
        int param = aux.dpdsc_base[params + i];
        if (param) {
          lower_visit_symbol(param);
        }
      }
      if (fval)
        lower_visit_symbol(fval);
      llvm_iface_flag = FALSE;
    }

    break;
  }
} /* lower_use_datatype */

#ifdef FLANG_LOWERSYM_UNUSED
/* Return TRUE if this dtype was not already marked used */
static int
lower_unused_datatype(int dtype)
{
  if (dtype <= 0)
    return 1;
  if (dtype >= last_datatype_used)
    return 0;
  if (datatype_used[dtype])
    return 0;
  return 1;
} /* lower_unused_datatype */
#endif

static int
eval_con_expr(int ast, int *val, int *dtyp)
{
  int val1;
  int val2;
  int tmp_ast1;
  int success = 0;

  if (!ast)
    return 0;

  if (A_ALIASG(ast)) {
    *dtyp = A_DTYPEG(ast);
    ast = A_ALIASG(ast);
  }

  switch (A_TYPEG(ast)) {
  case A_CNST:
    *dtyp = A_DTYPEG(ast);
    *val = CONVAL2G(A_SPTRG(ast));
    success = 1;
    break;
  case A_UNOP:
    if (eval_con_expr(A_LOPG(ast), &val1, dtyp)) {
      if (A_OPTYPEG(ast) == OP_SUB)
        *val = negate_const(val1, A_DTYPEG(ast));
      if (A_OPTYPEG(ast) == OP_LNOT)
        *val = ~val1;
      *dtyp = A_DTYPEG(ast);
      success = 1;
    }
    break;
  case A_BINOP:
    if (eval_con_expr(A_LOPG(ast), &val1, dtyp) &&
        eval_con_expr(A_ROPG(ast), &val2, dtyp)) {
      *val = const_fold(A_OPTYPEG(ast), val1, val2, A_DTYPEG(ast));
      *dtyp = A_DTYPEG(ast);
      success = 1;
    }
    break;
  case A_SUBSCR:
  case A_MEM:
    tmp_ast1 = complex_alias(ast);
    if (eval_con_expr(tmp_ast1, &val1, dtyp)) {
      *val = val1;
      success = 1;
    }
    break;
  }

  return success;
}

static void
lower_put_datatype(int dtype, int usage)
{
  int ndim, i, zbase, numelm;
  int dty, iface;
  /* if this was a 'stashed' old datatype */
  if (DTY(dtype) < 0)
    return;
  if (dtype < last_datatype_used) {
    if (datatype_output[dtype] > 1)
      return;
    else if (datatype_output[dtype] == 1) {
      if (!IS_STB_FILE())
        return;
    }
    datatype_output[dtype]++;
  }
  /* first character disambiguates:
   * a - any
   * A - array
   * c - character
   * C - complex
   * D - derived type
   * H - Hollerith
   * I - Integer
   * L - Logical
   * n - ncharacter
   * N - none
   * P - pointer
   * R - real
   * S - struct
   * U - union
   * W - word
   * Z - numeric
   */

  if (DTY(dtype) == TY_ARRAY) {
    /* FS#19796: Make sure we lower the element type of array.
     * Otherwise, we might miss lowering dtypes for element indices
     * such as DT_INT8 if the array has a DT_INT8 array size or if
     * the user compiles with -i8.
     */
    ndim = ADD_NUMDIM(dtype);
    for (i = 0; i < ndim; ++i) {
      int lb, ub, extnt;
      lb = ADD_LWAST(dtype, i);
      ub = ADD_UPAST(dtype, i);
      extnt = ADD_EXTNTAST(dtype, i);

      if (A_TYPEG(lb) == A_INTR) {
        switch (A_OPTYPEG(lb)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          lb = A_ARGSG(lb);
          lb = ARGT_ARG(lb, 0);
          dty = A_DTYPEG(ub);
          lower_put_datatype(dty, datatype_used[dty]);
        }
      }
      if (A_TYPEG(ub) == A_INTR) {
        switch (A_OPTYPEG(ub)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          ub = A_ARGSG(ub);
          ub = ARGT_ARG(ub, 0);
          dty = A_DTYPEG(ub);
          lower_put_datatype(dty, datatype_used[dty]);
        }
      }
      if (A_TYPEG(extnt) == A_INTR) {
        switch (A_OPTYPEG(extnt)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          extnt = A_ARGSG(extnt);
          extnt = ARGT_ARG(extnt, 0);
          dty = A_DTYPEG(extnt);
          lower_put_datatype(dty, datatype_used[dty]);
        }
      }
    }
  }

  putival("datatype", dtype);

  switch (DTY(dtype)) {
  case TY_NONE:
    putwhich("none", "n");
    break;
  case TY_WORD:
    putwhich("Word4", "W4");
    break;
  case TY_DWORD:
    putwhich("Word8", "W8");
    break;
  case TY_HOLL:
    putwhich("Hollerith", "H");
    break;

  case TY_BINT:
    putwhich("Integer1", "I1");
    break;
  case TY_SINT:
    putwhich("Integer2", "I2");
    break;
  case TY_INT:
    putwhich("Integer4", "I4");
    break;
  case TY_INT8:
    putwhich("Integer8", "I8");
    break;

  case TY_HALF:
    putwhich("Real2", "R2");
    break;
  case TY_REAL:
    putwhich("Real4", "R4");
    break;
  case TY_DBLE:
    putwhich("Real8", "R8");
    break;
  case TY_QUAD:
    putwhich("Real16", "R16");
    break;

  case TY_HCMPLX:
    putwhich("Complex4", "C4");
    break;
  case TY_CMPLX:
    putwhich("Complex8", "C8");
    break;
  case TY_DCMPLX:
    putwhich("Complex16", "C16");
    break;
  case TY_QCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    putwhich("Complex32", "C32");
#else
    putwhich("Complex16", "C16");
#endif
    break;

  case TY_BLOG:
    putwhich("Logical1", "L1");
    break;
  case TY_SLOG:
    putwhich("Logical2", "L2");
    break;
  case TY_LOG:
    putwhich("Logical4", "L4");
    break;
  case TY_LOG8:
    putwhich("Logical8", "L8");
    break;

  case TY_CHAR:
    putwhich("character", "c");
    if (dtype == DT_ASSCHAR) {
      putval("len", ASSCHAR);
    } else if (dtype == DT_DEFERCHAR) {
      putval("len", DEFERCHAR);
    } else {
      int clen = DTY(dtype + 1);
      if (A_ALIASG(clen)) {
        clen = A_ALIASG(clen);
        clen = A_SPTRG(clen);
        clen = CONVAL2G(clen);
        putval("len", clen);
      } else {
        if (sem.gcvlen && is_deferlenchar_dtype(dtype)) {
          putval("len", DEFERCHAR);
        } else {
          putval("len", ADJCHAR);
        }
      }
    }
    break;
  case TY_NCHAR:
    putwhich("kcharacter", "k");
    if (dtype == DT_ASSNCHAR) {
      putval("len", ASSCHAR);
    } else if (dtype == DT_DEFERNCHAR) {
      putval("len", DEFERCHAR);
    } else {
      int clen = DTY(dtype + 1);
      if (A_ALIASG(clen)) {
        clen = A_ALIASG(clen);
        clen = A_SPTRG(clen);
        clen = CONVAL2G(clen);
        putval("len", clen);
      } else {
        putval("len", ASSCHAR);
      }
    }
    break;

  case TY_PTR:
    putwhich("Pointer", "P");
    putval("ptrto", DTY(dtype + 1));
    break;

  case TY_STRUCT:
    putwhich("Struct", "S");
    goto SUD;
  case TY_UNION:
    putwhich("Union", "U");
    goto SUD;
  case TY_DERIVED:
    putwhich("Derived", "D");
  SUD:
    /* first member (symbol), size (bytes), alignment (0/1/3/7) */
    putsym("member", DTY(dtype + 1));
    putval("size", DTY(dtype + 2));
    putsym("tag", DTY(dtype + 3));
    putval("align", DTY(dtype + 4));
    break;

  case TY_NUMERIC:
    putwhich("Numeric", "N");
    break;
  case TY_ANY:
    putwhich("any", "a");
    break;

  case TY_PROC: {
    int restype = DTY(dtype + 1);
    if (is_array_dtype(restype))
      restype = array_element_dtype(restype);
  }
    putwhich("proc", "p");
    putval("result", DTY(dtype + 1));
    iface = DTY(dtype + 2);
    if (iface) {
      /* Based revision 68096 - need to lower its symlink instead if it is
       * ST_ALIAS */
      if (STYPEG(iface) == ST_ALIAS) {
        putsym("iface", SYMLKG(iface));
      } else
        putsym("iface", iface);

    } else
      putsym("iface", iface);
    putval("paramct", DTY(dtype + 3));
    putval("dpdsc", DTY(dtype + 4));
    putval("fval", DTY(dtype + 5));
    if (gbl.stbfil && DTY(dtype + 2)) {
      int fval = DTY(dtype + 5);
      int params = DPDSCG(iface);
      if (STYPEG(iface) == ST_ALIAS) {
        iface = SYMLKG(iface);
        fval = FVALG(iface);
        params = DPDSCG(iface);
      }
      if (STYPEG(iface) == ST_MODPROC) {
        if (SCOPEG(iface) == gbl.currsub || ENCLFUNCG(iface) == gbl.currsub)
          break;
        if (ENCLFUNCG(iface) == ENCLFUNCG(gbl.currsub))
          break;
      }
      llvm_iface_flag = TRUE;
      lower_visit_symbol(iface);
      for (i = 0; i < (int)(PARAMCTG(iface)); ++i) {
        int param = aux.dpdsc_base[params + i];
        if (param) {
          lower_visit_symbol(param);
        }
      }
      if (fval)
        lower_visit_symbol(fval);
      llvm_iface_flag = FALSE;
    }
    break;

  case TY_ARRAY:
    ndim = ADD_NUMDIM(dtype);
    putwhich("Array", "A");
    putval("type", DTY(dtype + 1));
    putval("dims", ndim);
    for (i = 0; i < ndim; ++i) {
      int lb, ub, extnt, mpy;
      lb = ADD_LWAST(dtype, i);
      ub = ADD_UPAST(dtype, i);
      extnt = ADD_EXTNTAST(dtype, i);
    lb_again:
      if (lb == 0) {
        lb = lowersym.intone;
        lower_visit_symbol(lb);
      } else if (A_TYPEG(lb) == A_INTR) {
        switch (A_OPTYPEG(lb)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          lb = A_ARGSG(lb);
          lb = ARGT_ARG(lb, 0);
          goto lb_again;
        case I_SIZE: {
          int arr, con, dty, val;
          ADSC *ad;
          lb = A_ARGSG(lb);
          arr = ARGT_ARG(lb, 0);
          con = ARGT_ARG(lb, 1);
          if (!eval_con_expr(con, &val, &dty)) {
            goto lb_error;
          }
          dty = DTYPEG(memsym_of_ast(arr));
          ad = AD_DPTR(dty);
          lb = AD_UPAST(ad, val);
          goto lb_again;
        }
        }
        goto lb_error;
      } else if (A_TYPEG(lb) == A_ID || A_TYPEG(lb) == A_CNST) {
        lb = A_SPTRG(lb);
        lower_visit_symbol(lb);
      } else {
        if (!XBIT(52, 4)) {
          if (A_TYPEG(lb) == A_SUBSCR) {
            int l = A_LOPG(lb);
            if (A_TYPEG(l) == A_MEM)
              l = A_MEMG(l);
            if (A_TYPEG(l) == A_ID && DESCARRAYG(A_SPTRG(l))) {
              lb = 0;
            }
          }
        }
        if (lb) {
        lb_error:
          if (usage == 1)
            lerror("array lower bound is not a symbol for datatype %d", dtype);
          lb = lowersym.intone;
          lower_visit_symbol(lb);
        }
      }
    ub_again:
      if (ub == 0) {
      } else if (A_TYPEG(ub) == A_INTR) {
        switch (A_OPTYPEG(ub)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          ub = A_ARGSG(ub);
          ub = ARGT_ARG(ub, 0);
          goto ub_again;
        case I_SIZE: {
          int arr, con, dty, val;
          ADSC *ad;
          ub = A_ARGSG(ub);
          arr = ARGT_ARG(ub, 0);
          con = ARGT_ARG(ub, 1);
          if (!eval_con_expr(con, &val, &dty)) {
            if (A_TYPEG(A_LOPG(con)) == A_ID) {
              ub = 0;
              goto ub_again;
            }
            goto ub_error;
          }
          dty = DTYPEG(memsym_of_ast(arr));
          ad = AD_DPTR(dty);
          ub = AD_UPAST(ad, val);
          goto ub_again;
        }
        }
        goto ub_error;

      } else if (A_TYPEG(ub) == A_ID || A_TYPEG(ub) == A_CNST) {
        ub = A_SPTRG(ub);
        lower_visit_symbol(ub);
      } else {
        if (!XBIT(52, 4)) {
          if (A_TYPEG(ub) == A_SUBSCR) {
            int u = A_LOPG(ub);
            if (A_TYPEG(u) == A_MEM)
              u = A_MEMG(u);
            if (A_TYPEG(u) == A_ID && DESCARRAYG(A_SPTRG(u))) {
              ub = 0;
            }
          } else if (A_TYPEG(ub) == A_BINOP && A_OPTYPEG(ub) == OP_ADD) {
            /* handle special case of lower+(extent-1) */
            int l, r, rl, rr;
            l = A_LOPG(ub);
            r = A_ROPG(ub);
            if (A_TYPEG(l) == A_BINOP && A_OPTYPEG(l) == OP_SUB) {
              rl = l;
              l = r;
              r = rl;
            }
            if (A_TYPEG(r) == A_BINOP && A_OPTYPEG(r) == OP_SUB) {
              rl = A_LOPG(r);
              rr = A_ROPG(r);
              if (A_TYPEG(l) == A_SUBSCR && A_TYPEG(rl) == A_SUBSCR &&
                  A_TYPEG(rr) == A_CNST) {
                l = A_LOPG(l);
                rl = A_LOPG(rl);
                if (A_TYPEG(l) == A_ID && DESCARRAYG(A_SPTRG(l)) &&
                    A_TYPEG(rl) == A_ID && DESCARRAYG(A_SPTRG(rl))) {
                  ub = 0;
                }
              }
            }
          }
        }
        if (ub && !valid_kind_parm_expr(ub)) {
        ub_error:
          if (usage == 1) {
            lerror("array upper bound is not a symbol for datatype %d", dtype);
          }
          ub = 0;
        }
      }
      putpair(lb, ub);
    extnt_again:
      if (extnt == 0) {
        extnt = lowersym.intone;
        lower_visit_symbol(extnt);
      } else if (A_TYPEG(extnt) == A_INTR) {
        switch (A_OPTYPEG(extnt)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          extnt = A_ARGSG(extnt);
          extnt = ARGT_ARG(extnt, 0);
          goto extnt_again;
        case I_SIZE: {
          int arr, con, dty, val;
          extnt = A_ARGSG(extnt);
          arr = ARGT_ARG(extnt, 0);
          con = ARGT_ARG(extnt, 1);
          if (!eval_con_expr(con, &val, &dty)) {
            if (A_TYPEG(A_LOPG(con)) == A_ID) {
              extnt = 0;
              goto extnt_again;
            }
            goto extnt_error;
          }
          dty = DTYPEG(memsym_of_ast(arr));
          extnt = ADD_EXTNTAST(dty, val);
          goto extnt_again;
        }
        }
        goto extnt_error;
      } else if (A_TYPEG(extnt) == A_ID || A_TYPEG(extnt) == A_CNST) {
        extnt = A_SPTRG(extnt);
        lower_visit_symbol(extnt);
      } else {
        if (!XBIT(52, 4)) {
          if (A_TYPEG(extnt) == A_SUBSCR) {
            int l = A_LOPG(extnt);
            if (A_TYPEG(l) == A_MEM)
              l = A_MEMG(l);
            if (A_TYPEG(l) == A_ID && DESCARRAYG(A_SPTRG(l))) {
              extnt = 0;
            }
          }
        }
        if (extnt && !valid_kind_parm_expr(extnt)) {
        extnt_error:
          if (usage == 1)
            lerror("array extnt is not a symbol for datatype %d", dtype);
          extnt = lowersym.intone;
          lower_visit_symbol(extnt);
        }
      }
      mpy = ADD_MLPYR(dtype, i);
      if (mpy == 0) {
      } else if (A_TYPEG(mpy) == A_ID || A_TYPEG(mpy) == A_CNST) {
        mpy = A_SPTRG(mpy);
        lower_visit_symbol(mpy);
      } else {
        mpy = 0;
      }
      putsym("mpy", mpy);
    }
    zbase = ADD_ZBASE(dtype);
    if (zbase == 0) {
      zbase = 0;
      /*lerror( "array zero-base is unknown for datatype %d", dtype );*/
      /* it will be left as zero for assumed-shape arguments
       * of module subprograms */
    } else if (A_TYPEG(zbase) == A_ID || A_TYPEG(zbase) == A_CNST) {
      zbase = A_SPTRG(zbase);
      lower_visit_symbol(zbase);
    } else {
      if (!XBIT(52, 4)) {
        if (A_TYPEG(zbase) == A_SUBSCR) {
          int z = A_LOPG(zbase);
          if (A_TYPEG(z) == A_MEM)
            z = A_MEMG(z);
          if (A_TYPEG(z) == A_ID && DESCARRAYG(A_SPTRG(z))) {
            zbase = 0;
          }
        }
      }
      if (zbase) {
        zbase = 0;
        /*We need to avoid the case that logic array has been used for
         * intrinsics*/
        if (usage == 1 && ndim)
          lerror("array zero-base is not a symbol for datatype %d", dtype);
      }
    }
    if (zbase == 0)
      zbase = stb.i1;
    putsym("zbase", zbase);
    numelm = ADD_NUMELM(dtype);
    if (numelm == 0) {
    } else if (A_TYPEG(numelm) == A_ID || A_TYPEG(numelm) == A_CNST) {
      numelm = A_SPTRG(numelm);
      lower_visit_symbol(numelm);
    } else {
      if (!XBIT(52, 4)) {
        if (is_descr_expression(numelm)) {
          numelm = 0;
        } else if (A_TYPEG(numelm) == A_SUBSCR) {
          int n = A_LOPG(numelm);
          if (A_TYPEG(n) == A_ID && DESCARRAYG(A_SPTRG(n))) {
            numelm = 0;
          }
        }
      }
      if (numelm && !valid_kind_parm_expr(numelm)) {
        numelm = 0;
        if (usage == 1)
          lerror("array numelm is not a symbol for datatype %d", dtype);
      }
    }
    putsym("numelm", numelm);
    break;

  default:
    fprintf(lowersym.lowerfile, "?????");
    lerror("unknown data type %d (value %d)", dtype, DTY(dtype));
    break;
  }
  fprintf(lowersym.lowerfile, "\n");
} /* lower_put_datatype */

/* put dtype to ilm file and optionally to stb file */
static void
lower_put_datatype_stb(int dtype)
{
  int usage = dtype >= last_datatype_used ? 1 : datatype_used[dtype];
  lower_put_datatype(dtype, usage);
  if (STB_LOWER()) {
    FILE *tmpfile = lowersym.lowerfile;
    lowersym.lowerfile = gbl.stbfil;
    lower_put_datatype(dtype, usage);
    lowersym.lowerfile = tmpfile;
  }
}

/** \brief Lower all of the data types */
void
lower_data_types(void)
{
  int dtype;

  for (dtype = 0; dtype < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
    if (dtype >= last_datatype_used || datatype_used[dtype]) {
      lower_put_datatype_stb(dtype);
    }
  }
} /* lower_data_types */

void
lower_push(int value)
{
  ++stack_top;
  NEED(stack_top + 1, stack, int, stack_size, stack_size + 100);
  stack[stack_top] = value;
} /* lower_push */

int
lower_pop(void)
{
  if (stack_top <= 0) {
    error(0, 4, 0, "stack underflow while lowering", "");
  }
  --stack_top;
  return stack[stack_top + 1];
} /* lower_pop */

void
lower_check_stack(int check)
{
  if (stack_top <= 0) {
    interr("stack underflow while lowering", stack_top, 4);
  }
  if (stack[stack_top] != check) {
    interr("stack error while lowering", check, 4);
  }
  --stack_top;
} /* lower_check_stack */

int
lower_getintcon(int val)
{
  INT v[4];
  int sptr;
  v[0] = v[2] = v[3] = 0;
  v[1] = val;
  sptr = getcon(v, DT_INT4);
  VISITP(sptr, 1);
  lower_use_datatype(DT_INT4, 1);
  return sptr;
} /* lower_getintcon */

static int
lower_getnull(void)
{
  INT v[4];
  int sptr;
  v[0] = v[1] = v[2] = v[3] = 0;
  sptr = getcon(v, DT_ADDR);
  return sptr;
} /* lower_getnull */

int
lower_getiszcon(ISZ_T val)
{
  if (XBIT(68, 0x1)) {
    INT num[2], sptr;

    ISZ_2_INT64(val, num);
    sptr = getcon(num, DT_INT8);
    VISITP(sptr, 1);
    lower_use_datatype(DT_INT8, 1);
    return sptr;
  } else
    return lower_getintcon(val);
} /* lower_getiszcon */

int
lower_getlogcon(int val)
{
  INT v[4];
  int sptr;
  v[0] = v[2] = v[3] = 0;
  v[1] = val;
  sptr = getcon(v, DT_LOG4);
  VISITP(sptr, 1);
  lower_use_datatype(DT_LOG4, 1);
  return sptr;
} /* lower_getlogcon */

int
lower_getrealcon(int val)
{
  INT v[4];
  int sptr;
  v[0] = v[2] = v[3] = 0;
  v[1] = val;
  sptr = getcon(v, DT_REAL4);
  VISITP(sptr, 1);
  lower_use_datatype(DT_REAL4, 1);
  return sptr;
} /* lower_getrealcon */

void
lower_namelist_plists(void)
{
  int sptr;
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_NML) {
      /* change the data type of the namelist PLIST from DT_INT
       * to an array of proper size */
      int plist = ADDRESSG(sptr);
      int dtype = get_array_dtype(1, DT_PTR);
      int member;
      lower_use_datatype(DT_INT, 1);
      lower_use_datatype(DT_PTR, 1);
      ADD_ZBASE(dtype) = astb.bnd.one;
      ADD_MLPYR(dtype, 0) = astb.bnd.one;
      ADD_LWBD(dtype, 0) = ADD_LWAST(dtype, 0) = astb.bnd.one;
      ADD_NUMELM(dtype) = ADD_UPBD(dtype, 0) = ADD_UPAST(dtype, 0) =
          ADD_EXTNTAST(dtype, 0) = mk_cnst(lower_getiszcon(PLLENG(plist)));
      DTYPEP(plist, dtype);
      STYPEP(plist, ST_ARRAY);
      PLLENP(plist, 0);

      /* export the namelist variable also */
      lower_visit_symbol(sptr);
      /* export all symbols in the namelist */
      for (member = CMEMFG(sptr); member; member = NML_NEXT(member)) {
        int sptr = NML_SPTR(member);
        if (LOWER_SYMBOL_REPLACE(sptr)) {
          sptr = LOWER_SYMBOL_REPLACE(sptr);
        }
        lower_visit_symbol(sptr);
      }
    }
  }
} /* lower_namelist_plists */

/** \brief Convert the datatype for linearized arrays to assumed-size array */
void
lower_linearized(void)
{
  int sptr;
  if (!XBIT(52, 4))
    return;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    if (DTY(DTYPEG(sptr)) == TY_ARRAY && LNRZDG(sptr)) {
      /* type should be basetype(1:1):: array */
      int olddtype, dtype, savedtype;
      olddtype = DTYPEG(sptr);
      /* stash the old datatype; it can be retrieved
       * from the DTY('newdtype'-1) */
      savedtype = get_type(1, -olddtype, 0);
      dtype = get_array_dtype(1, DTY(olddtype + 1));
      ADD_ZBASE(dtype) = astb.bnd.one;
      ADD_MLPYR(dtype, 0) = astb.bnd.one;
      ADD_LWBD(dtype, 0) = ADD_LWAST(dtype, 0) = astb.bnd.one;
      ADD_NUMELM(dtype) = ADD_UPBD(dtype, 0) = ADD_UPAST(dtype, 0) =
          ADD_EXTNTAST(dtype, 0) = astb.bnd.one;
      lower_visit_symbol(lowersym.intone);
      DTYPEP(sptr, dtype);
    }
  }
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    int dtype;
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY && LNRZDG(sptr)) {
      lower_use_datatype(DTY(dtype + 1), 1);
    }
    if (STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY) {
      if (FVALG(sptr)) {
        DTYPEP(sptr, DTYPEG(FVALG(sptr)));
      }
    }
  }
  lower_linearized_dtypes = TRUE;
} /* lower_linearized */

/*
 * find a NMPTR that shares NMPTR for different symbols with the same name
 * note that putsname always inserts a new name into the name table
 */
static int
find_nmptr(const char *symname, int len)
{
  int hash, hptr;
  HASH_ID(hash, symname, len);
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if (strcmp(SYMNAME(hptr), symname) == 0) {
      return NMPTRG(hptr);
    }
  }
  return putsname(symname, len);
} /* find_nmptr */

static int
lower_newsymbol(const char *name, int stype, int dtype, int sclass)
{
  int sptr, hashid;
  int namelen = strlen(name);
  HASH_ID(hashid, name, namelen);
  ADDSYM(sptr, hashid);
  NMPTRP(sptr, find_nmptr(name, namelen));
  SYMLKP(sptr, NOSYM);
  STYPEP(sptr, stype);
  DTYPEP(sptr, dtype);
  SCP(sptr, sclass);
  SCOPEP(sptr, stb.curr_scope);
  switch (stype) {
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
    CCSYMP(sptr, 1);
    break;
  default:
    break;
  }
  VISITP(sptr, 1);
  lower_use_datatype(dtype, 1);
  return sptr;
} /* lower_newsymbol */

int
lower_newfunc(const char *name, int stype, int dtype, int sclass)
{
  int namelen, sptr;
  namelen = strlen(name);
  sptr = lookupsym(name, namelen);
  if (sptr <= NOSYM)
    sptr = lower_newsymbol(name, stype, dtype, sclass);
  return sptr;
} /* lower_newfunc */

int
lower_makefunc(const char *name, int dtype, LOGICAL isDscSafe)
{
  int symfunc;
  symfunc = lower_newfunc(name, ST_PROC, dtype, SC_EXTERN);
  HCCSYMP(symfunc, 1);
  if (isDscSafe)
    SDSCSAFEP(symfunc, 1);
  return symfunc;
} /* lower_makefunc */

void
lower_clear_visit_fields(void)
{
  int sptr;
  for (sptr = 0; sptr < stb.stg_avail; ++sptr) {
    VISITP(sptr, 0);
    VISIT2P(sptr, 0);
  }
} /* lower_clear_visit_fields */

static int lower_cmptrvar(const char *, int, int, int *);
#ifdef FLANG_LOWERSYM_UNUSED
static int get_cmptrvar(char *, int, int, int *);
#endif

/** \brief Add common blocks to hold various zeros

    <pre>
    common/pghpf_0/ pghpf_01, pghpf_02, pghpf_03, pghpf_04
    integer pghpf_01, pghpf_02, pghpf_03, pghpf_04
    common/pghpf_0c/ pghpf_0c
    character*1 pghpf_0c
    common /pghpf_lineno/ pghpf_lineno
    common /pghpf_np/ hpf_np$
    common /pghpf_me/ hpf_me$
    </pre>
 */
void
lower_add_pghpf_commons(void)
{
  int symcommon, sym1, sym2, sym3, sym4, dtype;
  int bsym1, bsym2, bsym3, bsym4;
  int cmsz; /* common member size */

  if (!XBIT(57, 0x8000)) {
    lowersym.ptr0 = lowersym.ptrnull;
  } else {
    symcommon = lower_newsymbol("pghpf_0", ST_CMBLK, 0, SC_NONE);
    SYMLKP(symcommon, gbl.cmblks);
    gbl.cmblks = symcommon;
    HCCSYMP(symcommon, 1);
    sym1 = lower_cmptrvar("pghpf_01", ST_VAR, DT_INT4, &bsym1);
    sym2 = lower_cmptrvar("pghpf_02", ST_VAR, DT_INT4, &bsym2);
    sym3 = lower_cmptrvar("pghpf_03", ST_VAR, DT_INT4, &bsym3);
    sym4 = lower_cmptrvar("pghpf_04", ST_VAR, DT_INT4, &bsym4);
#if defined(TARGET_WIN)
    if (!XBIT(70, 0x80000000)) {
      DLLP(symcommon, DLL_IMPORT);
    }
#endif
    if (!XBIT(70, 0x80000000)) {
      cmsz = 4;
      lowersym.ptr0 = sym3;
    } else {
      /* win dll target */
      cmsz = size_of(DT_PTR);
      lowersym.ptr0 = bsym3;
    }
    CMEMFP(symcommon, sym1);
    SYMLKP(sym1, sym2);
    SYMLKP(sym2, sym3);
    SYMLKP(sym3, sym4);
    SYMLKP(sym4, NOSYM);
    CMEMLP(symcommon, sym4);
    CMBLKP(sym1, symcommon);
    CMBLKP(sym2, symcommon);
    CMBLKP(sym3, symcommon);
    CMBLKP(sym4, symcommon);
    ADDRESSP(sym1, 0 * cmsz);
    ADDRESSP(sym2, 1 * cmsz);
    ADDRESSP(sym3, 2 * cmsz);
    ADDRESSP(sym4, 3 * cmsz);
    SIZEP(symcommon, 4 * cmsz);
  }

  if (!XBIT(57, 0x8000)) {
    lowersym.ptr0c = lowersym.ptr0;
  } else {
    dtype = get_type(2, TY_CHAR, astb.i1);
    lower_use_datatype(dtype, 1);
    symcommon = lower_newsymbol("pghpf_0c", ST_CMBLK, 0, SC_NONE);
    SYMLKP(symcommon, gbl.cmblks);
    gbl.cmblks = symcommon;
    HCCSYMP(symcommon, 1);
    sym1 = lower_cmptrvar("pghpf_0c", ST_VAR, dtype, &bsym1);
#if defined(TARGET_WIN)
    if (!XBIT(70, 0x80000000)) {
      DLLP(symcommon, DLL_IMPORT);
    }
#endif
    if (!XBIT(70, 0x80000000)) {
      lowersym.ptr0c = sym1;
      SIZEP(symcommon, 1);
    } else {
      lowersym.ptr0c = bsym1;
      SIZEP(symcommon, size_of(DT_PTR));
    }
    CMEMFP(symcommon, sym1);
    SYMLKP(sym1, NOSYM);
    CMEMLP(symcommon, sym1);
    CMBLKP(sym1, symcommon);
  }

  if (XBIT(70, 6)) {
    int l;
    l = strlen(gbl.src_file);
    lowersym.sym_chkfile = getstring(gbl.src_file, l + 1);
  }
} /* lower_add_pghpf_commons */

static int
lower_cmptrvar(const char *name, int stype, int dtype, int *bsym)
{
  char bname[16];
  int len;
  int sym;

  if (!XBIT(70, 0x80000000)) {
    sym = lower_newsymbol(name, stype, dtype, SC_CMBLK);
    return sym;
  }

  len = strlen(name);
#if DEBUG
  assert(len < (sizeof(bname) - 1), "lower_cmptrvar name overflow", 0, 0);
#endif
  /* win dll target: the variable is actually a pointer-based object,
   * so what's added to the common is the object's pointer variable.
   * The name of the pointer variable is formed by appending 'p' to
   * the original name.
   */
  strcpy(bname, name);
  bname[len] = 'p';
  bname[len + 1] = 0;

  sym = lower_newsymbol(bname, ST_VAR, DT_PTR, SC_CMBLK);
  *bsym = lower_newsymbol(name, stype, dtype, SC_BASED);
  MIDNUMP(*bsym, sym);
  return sym;
}

#ifdef FLANG_LOWERSYM_UNUSED
static int
get_cmptrvar(char *name, int stype, int dtype, int *bsym)
{
  int sym;

  if (!XBIT(70, 0x80000000)) {
    sym = getsymbol(name);
    STYPEP(sym, stype);
    DTYPEP(sym, dtype);
    SCP(sym, SC_CMBLK);
    VISITP(sym, 1);
    return sym;
  }

  /* win dll target: the variable is actually a pointer-based object,
   * so what's added to the common is the object's pointer variable.
   * The name of the pointer variable is formed by appending 'p' to
   * the original name.
   */
  sym = getsymf("%sp", name);
  STYPEP(sym, ST_VAR);
  DTYPEP(sym, DT_PTR);
  SCP(sym, SC_CMBLK);
  VISITP(sym, 1);

  *bsym = getsymbol(name);
  STYPEP(*bsym, stype);
  DTYPEP(*bsym, dtype);
  SCP(*bsym, SC_BASED);
  VISITP(*bsym, 1);
  MIDNUMP(*bsym, sym);

  return sym;
}
#endif

#if TY_MAX != 36
#error "Need to edit lowersym.c to add new TY_... data types"
#endif

static const char *
putstype(int stype, int sptr)
{
/* TRY TO KEEP THESE UNIQUE IN THE FIRST CHARACTER! */
#if ST_MAX != 35
#error \
    "Need to edit lowersym.c to add new or remove old ST_... symbol types or need to run the symtab utility"
#endif
  if (stype == ST_MODULE) {
    if (sptr == gbl.currsub) {
      stype = ST_ENTRY;
    } else {
      stype = ST_PROC;
    }
  }
  switch (stype) {
  case ST_ARRAY:
    return "Array";
  case ST_BLOCK:
    return "Block";
  case ST_CMBLK:
    return "Common";
  case ST_CONST:
    return "constant";
  case ST_DESCRIPTOR:
    return "Array";
  case ST_ENTRY:
    return "Entry";
  case ST_GENERIC:
    return "Generic";
  case ST_INTRIN:
    return "Intrinsic";
  case ST_PD:
    return "Known";
  case ST_LABEL:
    return "Label";
  case ST_PLIST:
    return "list";
  case ST_MEMBER:
    return "Member";
  case ST_MODULE:
    return "module";
  case ST_NML:
    return "Namelist";
  case ST_PARAM:
    return "parameter";
  case ST_PROC:
  case ST_MODPROC:
    return "Procedure";
  case ST_STRUCT:
    return "Struct";
  case ST_STAG:
    return "Tag";
  case ST_TYPEDEF:
    return "typedef";
  case ST_UNION:
    return "Union";
  case ST_USERGENERIC:
    return "Generic";
  case ST_VAR:
    return "Variable";

  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_STFUNC:
  case ST_ISOC:
  case ST_ISOFTNENV:
  case ST_ARRDSC:
  case ST_ALIAS:
  case ST_OPERATOR:
  case ST_CONSTRUCT:
  case ST_CRAY:
  default:
    lerror("unexpected symbol type %s(%d)",
           stype >= 0 && stype <= ST_MAX ? stb.stypes[stype] : "", stype);
#if DEBUG
    symdentry(gbl.dbgfil, sptr);
    if (STYPEG(sptr) == ST_ALIAS)
      symdentry(gbl.dbgfil, SYMLKG(sptr));
#endif
    return "?";
  }
} /* putstype */

static const char *
putsclass(int sclass, int sptr)
{
#if SC_MAX != 7
#error "Need to edit lowersym.c to add new SC_... symbol classes"
#endif
  switch (sclass) {
  case SC_BASED:
    return "Based";
  case SC_CMBLK:
    return "Common";
  case SC_DUMMY:
    return "Dummy";
  case SC_EXTERN:
    return "Extern";
  case SC_LOCAL:
    return "Local";
  case SC_NONE:
    return "none";
  case SC_PRIVATE:
    return "Private";
  case SC_STATIC:
    return "Static";
  default:
    lerror("unexpected symbol class %s(%d)",
           sclass >= 0 && sclass <= SC_MAX ? stb.scnames[sclass] : "", sclass);
#if DEBUG
    symdentry(gbl.dbgfil, sptr);
#endif
    return "?";
  }
} /* putsclass */

static void
lower_symbol(int sptr)
{
  int i, params, count, namelen, strip, newline, dtype, altreturn, desc;
  int fvalfirst, fvallast, sc, inmod, pdaln, frommod, cudamodule = 0;
  int conval, stype;
  int dll;
  int cudaemu, routx = 0;
  const char *name;
  char tempname[15];
  int retdesc;

  strip = 0;
  newline = 0;
  name = SYMNAME(sptr);
  namelen = ((name == NULL) ? 0 : strlen(name));
#if DEBUG
  if (DBGBIT(47, 8)) {
    fprintf(lowersym.lowerfile, "symbol:%s ", getprint(sptr));
  } else
#endif
    putival("symbol", sptr);
  stype = STYPEG(sptr);
  sc = SCG(sptr);

  if ((STYPEG(sptr) == ST_ALIAS || STYPEG(sptr) == ST_PROC ||
      STYPEG(sptr) == ST_ENTRY) &&
      SEPARATEMPG(sptr) &&
      STYPEG(SCOPEG(sptr)) == ST_MODULE)
    INMODULEP(sptr, 1);

  dtype = DTYPEG(sptr);
  if (stype == ST_CONST && DTY(dtype) == TY_HOLL)
    dtype = DTYPEG(CONVAL1G(sptr));
  if (stype == ST_PROC || stype == ST_ENTRY) {
    if (DTY(dtype) == TY_ARRAY) {
      dtype = DTY(dtype + 1);
      if (DTY(dtype) == TY_CHAR)
        dtype = DT_NONE;
    }
  }
  if (stype == ST_PROC || stype == ST_MODPROC) {
    if (sc == SC_NONE)
      sc = SC_EXTERN;
  }
  if (dtype == DT_ADDR) {
    if (XBIT(49, 0x100)) { /* 64-bit pointers */
      dtype = DT_INT8;
    } else {
      dtype = DT_INT;
    }
  }

  putstring(putstype(stype, sptr));
  putstring(putsclass(sc, sptr));
#if DEBUG
  if (DBGBIT(47, 8)) {
    fprintf(lowersym.lowerfile, " dtype:%d ", (int)DTY(dtype));
  } else
#endif
    putval("dtype", dtype);
  putval("palign", PALIGNG(sptr));
  /* type specific information */
  switch (stype) {
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
  case ST_VAR:
    putbit("addrtaken", ADDRTKNG(sptr));
    putbit("argument", ARGG(sptr));
    putbit("assigned", ASSNG(sptr));
    putbit("decl", DCLDG(sptr));
#if defined(TARGET_WIN)
    putval("dll", DLLG(sptr));
    putbit("mscall", MSCALLG(sptr));
    putbit("cref", CREFG(sptr));
#else
    putval("dll", 0);
    putbit("mscall", 0);
    putbit("cref", 0);
#endif
    putbit("ccsym", CCSYMG(sptr));
    putbit("hccsym", HCCSYMG(sptr));
    if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE &&
        SCOPEG(sptr) != stb.curr_scope) {
      putbit("init", 0);
    } else {
      putbit("init", DINITG(sptr));
    }
    if (!XBIT(7, 0x100000)) {
      putbit("datacnst", DATACONSTG(sptr));
    } else {
      putbit("datacnst", 0);
    }
    putbit("namelist", NMLG(sptr));
    putbit("optional", OPTARGG(sptr));
    putbit("pointer",
           POINTERG(sptr) || MDALLOCG(sptr) ||
               (ALLOCG(sptr) && (SCG(sptr) == SC_BASED) && !NODESCG(sptr)));
    putbit("private", PRIVATEG(sptr));
    pdaln = 0;
#ifdef PDALNG
    if (!PDALN_IS_DEFAULT(sptr)) {
      pdaln = PDALNG(sptr);
      if (pdaln == 0)
        pdaln = PDALN_EXPLICIT_0;
    }
#endif
#ifdef QALNG
    if (QALNG(sptr) && (pdaln < 3 || pdaln == PDALN_EXPLICIT_0))
      pdaln = 3;
#endif
    putval("pdaln", pdaln);
#ifdef TQALNG
    if (stype == ST_VAR) {
      putbit("tqaln", TQALNG(sptr));
    } else
#endif
      putbit("tqaln", 0);
    putbit("ref", REFG(sptr));
    putbit("save", SAVEG(sptr));
    putbit("seq", SEQG(sptr));
    putbit("target", TARGETG(sptr));
    putbit("param", PARAMG(sptr));
    if (gbl.internal <= 1 || INTERNALG(sptr)) {
      /* for outer procedures, no symbols are uplevel */
      putbit("uplevel", 0);
      putbit("internref", 0);
    } else if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_MODULE) {
      /* module symbols are not uplevel */
      putbit("uplevel", 0);
      putbit("internref", 0);
    } else {
      putbit("uplevel", 1);
      if (INTERNREFG(sptr))
        putbit("internref", 1);
      else
        putbit("internref", 0);
    }
    putbit("ptrsafe", PTRSAFEG(sptr));
    putbit("thread", THREADG(sptr));
    putval("etls", ETLSG(sptr));
    putbit("tls", TLSG(sptr));

#ifdef TASKG
    putbit("task", TASKG(sptr));
#else
    putbit("task", 0);
#endif
    putbit("volatile", VOLG(sptr));
    if (sc == SC_DUMMY || sc == SC_BASED ||
        (CLASSG(sptr) && stype == ST_DESCRIPTOR)) {
      putval("address", 0);
    } else {
      putval("address", ADDRESSG(sptr));
    }
    if (ADJLENG(sptr)) {
      putsym("clen", CVLENG(sptr));
    } else {
      putval("clen", 0);
    }
    putsym("common", CMBLKG(sptr));
#if DEBUG
    if (DBGBIT(47, 8)) { /* don't put out 'link' with this switch */
    } else
#endif
      putsym("link", SYMLKG(sptr));
    putsym("midnum", MIDNUMG(sptr));
    if (flg.debug)
      check_debug_alias(sptr);
    if (sc == SC_DUMMY) {
      int a;
      a = NEWARGG(sptr);
      putval("origdummy", a);
    }
    if (stype == ST_ARRAY || stype == ST_DESCRIPTOR) {
      putbit("adjustable", ADJARRG(sptr));
      putbit("afterentry", AFTENTG(sptr));
      putbit("assumedrank", ASSUMRANKG(sptr));
      putbit("assumedshape", ASSUMSHPG(sptr));
      putbit("assumedsize", ASUMSZG(sptr));
      putbit("autoarray",
             AUTOBJG(sptr) || (ADJARRG(sptr) && SCG(sptr) == SC_LOCAL));
      putbit("noconflict", VISIT2G(sptr));
      putbit("s1", SDSCS1G(sptr));
      putbit("isdesc", stype == ST_DESCRIPTOR ? 1 : 0);
#ifdef SDSCCONTIGG
      putbit("contig", stype == ST_DESCRIPTOR ? SDSCCONTIGG(sptr) : 0);
#else
      putbit("contig", 0);
#endif
      if (LNRZDG(sptr) && XBIT(52, 4)) {
        /* get original datatype */
        int origdtype;
        origdtype = -DTY(dtype - 1);
        putval("origdim", ADD_NUMDIM(origdtype));
      } else {
        putval("origdim", ADD_NUMDIM(dtype));
      }
      putsym("descriptor", SDSCG(sptr));
    }
    putbit("parref", PARREFG(sptr));
    putsym("enclfunc", ENCLFUNCG(sptr));
    putbit("passbyval", PASSBYVALG(sptr));
    putbit("passbyref", PASSBYREFG(sptr));
    putbit("Cfunc", CFUNCG(sptr));
    putsym("altname", ALTNAMEG(sptr));
    putbit("contigattr", CONTIGATTRG(sptr));
    putbit("device", 0);
    putbit("pinned", 0);
    putbit("shared", 0);
    putbit("constant", 0);
    putbit("texture", 0);
    putbit("managed", 0);
    putbit("intentin", (SCG(sptr) == SC_DUMMY && INTENTG(sptr) == INTENT_IN));
#if defined(CLASSG)
    putbit("class", CLASSG(sptr));
    putval("parent", PARENTG(sptr));
    if (stype == ST_VAR) { /* TBD - need this for poly variable? */
      if (DTYPEG(sptr) == DT_DEFERCHAR || DTYPEG(sptr) == DT_ASSCHAR) {
        putsym("descriptor", SDSCG(sptr));
      } else if (sc == SC_DUMMY && CLASSG(sptr)) {
        putsym("descriptor", PARENTG(sptr));
      } else if ((sc == SC_DUMMY ||
                  (sc == SC_BASED && SCG(MIDNUMG(sptr)) == SC_DUMMY)) &&
                 NEWDSCG(sptr) && SDSCG(sptr)) {
        putsym("descriptor", NEWDSCG(sptr));
      } else if ((sc == SC_DUMMY || SCG(SDSCG(sptr)) == SC_DUMMY) &&
                 needs_descriptor(sptr)) {
        putsym("descriptor", SDSCG(sptr));
      } else {
        putsym("descriptor", (CLASSG(sptr)) ? SDSCG(sptr) : 0);
      }
    }
#else
    putbit("class", 0);
    putval("parent", 0);
    if (stype == ST_VAR) {
      if (DTYPEG(sptr) == DT_DEFERCHAR || DTYPEG(sptr) == DT_DEFERCHAR)
        putsym("descriptor", SDSCG(sptr));
      else
        putsym("descriptor", 0);
    }
#endif
    if (DTY(dtype) == TY_DERIVED && PARENTG(DTY(dtype + 1)) &&
        DINITG(DTY(dtype + 1)) && sc == SC_STATIC) {
      /* Set reref bit for type extensions with initializations
       * in the parent component since we need to compute
       * assn_static_off() in back end's sym_is_refd() function.
       */
      putbit("reref", 1);
    } else {
      putbit("reref", 0);
    }
    putbit("reflected", 0);
    putbit("mirrored", 0);
    putbit("create", 0);
    putbit("copyin", 0);
    putbit("resident", 0);
    putbit("link", 0);
    putbit("devicecopy", 0);
    putbit("devicesd", 0);
    putval("devcopy", 0);
    putbit("allocattr", ALLOCATTRG(sptr));
    putbit("f90pointer", 0); /* F90POINTER will denote the POINTER attribute */
                             /* but first need to remove FE legacy use */
    putbit("procdescr", IS_PROC_DESCRG(sptr));
    strip = 1;
    break;

  case ST_CMBLK:
    putsym("altname", ALTNAMEG(sptr));
    putbit("ccsym", CCSYMG(sptr) || HCCSYMG(sptr));
    putbit("Cfunc", CFUNCG(sptr));
#if defined(TARGET_WIN)
    putval("dll", DLLG(sptr));
#else
    putval("dll", 0);
#endif
    if (SCOPEG(sptr) == stb.curr_scope) {
      putbit("init", DINITG(sptr));
    } else {
      putbit("init", 0);
    }
    putsym("member", CMEMFG(sptr));
    putbit("mscall", MSCALLG(sptr));
    pdaln = 0;
#ifdef PDALNG
    if (!PDALN_IS_DEFAULT(sptr)) {
      pdaln = PDALNG(sptr);
      if (pdaln == 0)
        pdaln = PDALN_EXPLICIT_0;
    }
#endif
#ifdef QALNG
    if (QALNG(sptr) && (pdaln < 3 || pdaln == PDALN_EXPLICIT_0))
      pdaln = 3;
#endif
    putval("pdaln", pdaln);
    putbit("save", SAVEG(sptr));
    putval("size", SIZEG(sptr));
    putbit("stdcall", STDCALLG(sptr));
    putbit("thread", THREADG(sptr));
    putval("etls", ETLSG(sptr));
    putbit("tls", TLSG(sptr));
    putbit("volatile", VOLG(sptr));
    frommod = FROMMODG(sptr);
    if (MODCMNG(sptr) && frommod) {
      /*  Just a module with specifications only */
      if (SCOPEG(sptr) == gbl.currsub)
        frommod = 0;
    }
    putbit("frommod", frommod);
    putbit("modcmn", MODCMNG(sptr));
    putsym("scope", SCOPEG(sptr));
    putbit("restricted", RESTRICTEDG(SCOPEG(sptr)));
    putbit("device", 0);
    putbit("constant", 0);
    putbit("create", 0);
    putbit("copyin", 0);
    putbit("resident", 0);
    putbit("link", 0);
    if (BLANKCG(sptr)) {
      namelen = 6;
      name = "_BLNK_";
    }
    strip = 1;
    break;

  case ST_CONST:
    /* hollerith? and value */
    putbit("hollerith", HOLLG(sptr));
    switch (DTY(dtype)) {
    case TY_DWORD:
    case TY_INT8:
    case TY_LOG8:
    case TY_DBLE:
    case TY_CMPLX:
      puthex(CONVAL1G(sptr));
      puthex(CONVAL2G(sptr));
      break;
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_REAL:
    case TY_WORD:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      puthex(CONVAL2G(sptr));
      break;
    case TY_DCMPLX:
    case TY_QCMPLX:
      putsym("sym", CONVAL1G(sptr));
      putsym("sym", CONVAL2G(sptr));
      break;
    case TY_QUAD:
      puthex(CONVAL1G(sptr));
      puthex(CONVAL2G(sptr));
      puthex(CONVAL3G(sptr));
      puthex(CONVAL4G(sptr));
      break;
    case TY_PTR:
      putsym("sym", CONVAL1G(sptr));
      putval("offset", CONVAL2G(sptr));
      break;
    case TY_CHAR:
    case TY_NCHAR:
      /* put out the char string instead of the name */
      /* is this really a hollerith? */
      if (DTY(DTYPEG(sptr)) == TY_HOLL || DTY(dtype) == TY_NCHAR) {
        conval = CONVAL1G(sptr);
        name = stb.n_base + CONVAL1G(conval);
        namelen = string_length(DTYPEG(conval));
      } else {
        namelen = string_length(dtype);
        name = stb.n_base + CONVAL1G(sptr);
      }
      newline = 1;
      break;
    default:
      lerror("unexpected constant symbol data type (%d)", dtype);
#if DEBUG
      symdentry(gbl.dbgfil, sptr);
#endif
      break;
    }
    break;

  case ST_MODULE:
    if (sptr == gbl.currsub) {
      /* put out like an ENTRY */
      putbit("currsub", 1);
      putbit("adjustable", 0);
      putbit("afterentry", 0);
      putsym("altname", 0);
#if defined(TARGET_WIN_X86)
      putbit("Cfunc", 1);
#else
      putbit("Cfunc", 0);
#endif
      putbit("decl", 0);
#if defined(TARGET_WIN)
      putval("dll", DLLG(sptr));
#else
      putval("dll", 0);
#endif
      putval("cmode", 0);
      putval("end", ENDLINEG(sptr));
      putsym("inmodule", 0);
      putval("line", FUNCLINEG(sptr));
#if defined(TARGET_WIN_X86)
      putbit("mscall", 1);
#else
      putbit("mscall", 0);
#endif
      putbit("pure", 0);
      putbit("recursive", 0);
      putbit("elemental", 0);
      putval("returnval", 0);
      putbit("passbyval", 0);
      putbit("passbyref", 0);
      putbit("stdcall", 0);
      putbit("decorate", 0);
      putbit("cref", 0);
      putbit("nomixedstrlen", 0);
      putval("cudaemu", 0);
      putval("rout", 0);
      putval("paramcount", 0);
      putval("altreturn", 0);
      putval("vtoff", 0);
      putval("invobj", 0);
      putbit("invobjinc", 0);
      putbit("class", 0);
      putbit("denorm", 0);
      putbit("aret", 0);
      putbit("vararg", 0);
      putbit("has_opts", 0);
      strip = 1;
    } else {
      /* put out like a PROC */
      putsym("altname", 0);
      putbit("ccsym", 0);
      putbit("decl", 0);
      putval("dll", 0);
      i = 0;
#if defined(TARGET_WIN)
      if (ENCLFUNCG(gbl.currsub) == sptr && DLLG(sptr) != DLL_EXPORT &&
          DLLG(gbl.currsub) == DLL_EXPORT) {
        /*
         * dllexport of a normal ST_PROC is illegal; however, it
         * could represent a MODULE whose dllexport only occurs within
         * a contained procedure.
         */
        i = 1;
      }
#endif
      putbit("dllexportmod", i);
      putval("cmode", 0);
      putbit("func", 0);
      putsym("inmodule", 0);
#if defined(TARGET_WIN_X86)
      putbit("mscall", 1);
#else
      putbit("mscall", 0);
#endif
      putbit("needmod", NEEDMODG(sptr));
      putbit("pure", 0);
      putbit("ref", 0);
      putbit("passbyval", 0);
      putbit("passbyref", 0);
      putbit("cstructret", CSTRUCTRETG(sptr));
      putbit("sdscsafe", 0);
      putbit("stdcall", 0);
      putbit("decorate", 0);
      putbit("cref", 0);
      putbit("nomixedstrlen", 0);
      putbit("typed", TYPDG(sptr));
      putbit("recursive", 0);
      putval("returnval", 0);
#if defined(TARGET_WIN_X86)
      putbit("Cfunc", 1);
#else
      putbit("Cfunc", 0);
#endif
      putbit("uplevel", 0);
      putbit("internref", 0);
      putval("rout", 0);
      putval("paramcount", 0);
      putval("vtoff", 0);
      putval("invobj", 0);
      putbit("invobjinc", 0);
      putbit("class", 0);
      putbit("mlib", 0);
      putbit("clib", 0);
      putbit("inmodproc", 0);
      putbit("cudamodule", 0);
      putbit("fwdref", 0);
      putbit("aret", 0);
      putbit("vararg", VARARGG(sptr));
      putbit("has_opts", 0);
      putbit("parref", PARREFG(sptr));
      /*
       * emit this bit only if emitting ST_MODULE as ST_PROC
       * this conversion happens in putstype()
       */
      if (sptr != gbl.currsub) {
        putbit("is_interface", IS_INTERFACEG(sptr));
        putval("assocptr", ASSOC_PTRG(sptr));
        putval("ptrtarget",PTR_TARGETG(sptr));
        putbit("prociface", IS_PROC_PTR_IFACEG(sptr));
      }

      strip = 1;
    }
    break;
  case ST_ENTRY:
    inmod = SCOPEG(sptr);
    if (inmod && STYPEG(inmod) == ST_ALIAS) {
      inmod = SCOPEG(inmod);
    }
    if (!INMODULEG(sptr) || (inmod && STYPEG(inmod) != ST_MODULE)) {
      inmod = 0;
    }
    putbit("currsub", sptr == gbl.currsub);
    putbit("adjustable", ADJARRG(sptr));
    putbit("afterentry", AFTENTG(sptr));
    putsym("altname", ALTNAMEG(sptr));
    putbit("Cfunc", CFUNCG(sptr));
    putbit("decl", DCLDG(sptr));
#if defined(TARGET_WIN)
    putval("dll", DLLG(sptr));
#else
    putval("dll", 0);
#endif
#if defined(CUDAG)
    putval("cmode", CUDAG(sptr));
#else
    putval("cmode", 0);
#endif
    putval("end", ENDLINEG(sptr));
    putsym("inmodule", inmod);
    putval("line", FUNCLINEG(sptr));
    putbit("mscall", MSCALLG(sptr));
    putbit("pure", PUREG(sptr));
    putbit("recursive", RECURG(sptr));
    putbit("elemental", ELEMENTALG(sptr));
    putsym("returnval", FVALG(sptr));
    putbit("passbyval", PASSBYVALG(sptr));
    putbit("passbyref", PASSBYREFG(sptr));
    putbit("stdcall", STDCALLG(sptr));
    putbit("decorate", DECORATEG(sptr));
#ifdef CREFP
    putbit("cref", CREFG(sptr));
    putbit("nomixedstrlen", NOMIXEDSTRLENG(sptr));
#else
    putbit("cref", 0);
    putbit("nomixedstrlen", 0);
#endif
    cudaemu = 0;
    putval("cudaemu", cudaemu);
    fvalfirst = fvallast = 0;
    retdesc = CLASS_NONE;
    if (CFUNCG(sptr)) {
      retdesc = check_return(DTYPEG(FVALG(sptr)));
      if (retdesc != CLASS_MEM && retdesc != CLASS_PTR) {
        SCP(FVALG(sptr), SC_LOCAL); /* change retval from dummy to local */
      }
    } else if (CMPLXFUNC_C && FVALG(sptr) && DT_ISCMPLX(DTYPEG(FVALG(sptr)))) {
      SCP(FVALG(sptr), SC_LOCAL); /* change retval from dummy to local */
    }
    if (!POINTERG(sptr) && (retdesc == CLASS_NONE || retdesc == CLASS_MEM ||
                            retdesc == CLASS_PTR)) {
      switch (DTY(dtype)) {
      case TY_CMPLX:
      case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
#endif
        if (!CMPLXFUNC_C && FVALG(sptr))
          fvallast = 1;
        break;
      case TY_CHAR:
      case TY_NCHAR:
        if (FVALG(sptr) && !ADJLENG(FVALG(sptr)))
          fvallast = 1;
        break;
      case TY_DERIVED:
      case TY_STRUCT:
        if (FVALG(sptr))
          fvalfirst = 1;
        break;
      default:
        break;
      }
    }
    count = 0;
    altreturn = 0;
    params = DPDSCG(sptr);
    for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
      if (aux.dpdsc_base[params + i]) {
        ++count;
      } else {
        ++altreturn;
      }
    }
#if defined(ACCROUTG)
    putval("rout", ACCROUTG(sptr));
    routx = ACCROUTG(sptr);
#else
    putval("rout", 0);
#endif
    putval("paramcount", count + fvalfirst + fvallast);
    putval("altreturn", altreturn);
    putval("vtoff", VTOFFG(sptr));
    putval("invobj", INVOBJG(sptr));
    putbit("invobjinc", INVOBJINCG(sptr));
    putbit("class", CLASSG(sptr));
    putbit("denorm", gbl.denorm);
    putbit("aret", ARETG(sptr));
    putbit("vararg", 0);
    putbit("has_opts", has_opt_args(sptr) ? 1 : 0);
    if (fvalfirst) {
      putsym(NULL, FVALG(sptr));
    }
    for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
      if (aux.dpdsc_base[params + i]) {
        putsym(NULL, aux.dpdsc_base[params + i]);
      }
    }
    if (fvallast) {
      putsym(NULL, FVALG(sptr));
    }
    strip = 1;
    break;

  case ST_LABEL:
    putbit("ccsym", CCSYMG(sptr));
    putbit("assigned", ASSNG(sptr));
    putbit("format", FMTPTG(sptr));
    putbit("volatile", VOLG(sptr));
    putval("refs", RFCNTG(sptr));
    putval("agoto", AGOTOG(sptr));
    strip = 1;
    break;

  case ST_MEMBER:
    putbit("ccsym", CCSYMG(sptr));
    putbit("s1", SDSCS1G(sptr));
    putbit("isdesc", DESCARRAYG(sptr));
#ifdef SDSCCONTIGG
    putbit("contig", DESCARRAYG(sptr) ? SDSCCONTIGG(sptr) : 0);
#else
    putbit("contig", 0);
#endif
    putbit("contigattr", CONTIGATTRG(sptr));
    putbit("pointer", POINTERG(sptr) || ALLOCG(sptr));
    putval("address", ADDRESSG(sptr));
    if (DTY(dtype) == TY_ARRAY) {
      putsym("descriptor", SDSCG(sptr));
    } else if (DTYPEG(sptr) == DT_DEFERCHAR || DTYPEG(sptr) == DT_DEFERNCHAR) {
      putsym("descriptor", SDSCG(sptr));
    }
#ifdef CLASSG
    else if (SDSCG(sptr) && (CLASSG(sptr) || FINALIZEDG(sptr))) {
      int sdsc_mem = SYMLKG(sptr);
      if (sdsc_mem == MIDNUMG(sptr)) {
        sdsc_mem = SYMLKG(sdsc_mem);
        if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem))
          sdsc_mem = SYMLKG(sdsc_mem);
      }
      putsym("descriptor", sdsc_mem);
    }
#endif
    else {
      putsym("descriptor", 0);
    }
    putbit("noconflict", VISIT2G(sptr));
    putsym("link", SYMLKG(sptr));
    if ((STYPEG(BINDG(sptr)) == ST_OPERATOR ||
         STYPEG(BINDG(sptr)) == ST_USERGENERIC)) {
      /* FS#17251: TBD - if bind is an ST_OPERATOR/ST_USERGENERIC, then
       * fill in with a type bound procedure or 0 if generic is
       * currently empty.
       */
      int mem;
      mem = get_specific_member(TBPLNKG(VTABLEG(sptr)), VTABLEG(sptr));
      putval("tbplnk", BINDG(mem));
      putval("vtable", VTABLEG(mem));
      putval("iface", 0);
    } else {
      char *vt = SYMNAME(VTABLEG(sptr));
      putval("tbplnk", BINDG(sptr));
      if (!IFACEG(sptr) && strlen(vt) > 4 &&
          strcmp(vt + (strlen(vt) - 4), "$tbp") == 0) {
        putval("vtable", 0);
        putval("iface", 0);
      } else {
        putval("vtable", (IFACEG(sptr)) ? 0 : VTABLEG(sptr));
        putval("iface", IFACEG(sptr));
      }
    }
    putbit("class", CLASSG(sptr));
#if defined(TARGET_WIN)
    if (VTABLEG(sptr)) {
      putbit("mscall", MSCALLG(VTABLEG(sptr)));
      putbit("cref", CREFG(VTABLEG(sptr)));
    } else {
      putbit("mscall", MSCALLG(sptr));
      putbit("cref", CREFG(sptr));
    }
#else
    putbit("mscall", 0);
    putbit("cref", 0);
#endif
    putbit("allocattr", ALLOCATTRG(sptr));
    putbit("f90pointer", 0); /* need to remove FE legacy use of F90POINTER */
#ifdef FINALG
    putval("final", (!ELEMENTALG(VTABLEG(sptr))) ? FINALG(sptr) : MAXDIMS + 2);
#else
    putval("final", 0);
#endif
#ifdef FINALIZEDG
    putbit("finalized", FINALIZEDG(sptr));
#else
    putbit("finalized", 0);
#endif
#ifdef KINDG
    putbit("kindparm", KINDG(sptr) != 0);
#else
    putbit("kindparm", 0);
#endif
#ifdef LENPARMG
    putbit("lenparm", LENPARMG(sptr));
#else
    putbit("lenparm", 0);
#endif
#ifdef TPALLOCG
    putbit("tpalloc", TPALLOCG(sptr));
#else
    putbit("tpalloc", 0);
#endif
    putval("assocptr", ASSOC_PTRG(sptr));
    putval("ptrtarget", PTR_TARGETG(sptr));
    putbit("prociface", IS_PROC_PTR_IFACEG(sptr));
    strip = 1;
    break;

  case ST_MODPROC:
    /* fake a procedure */
    putsym("altname", 0);
    putbit("ccsym", 0);
    putbit("decl", 0);
    putval("dll", 0);
    putbit("dllexportmod", 0);
    putval("cmode", 0);
    putbit("func", 0);
    putsym("inmodule", 0);
    putbit("mscall", 0);
    putbit("needmod", 0);
    putbit("pure", 0);
    putbit("ref", 0);
    putbit("passbyval", 0);
    putbit("passbyref", 0);
    putbit("cstructret", 0);
    putbit("sdscsafe", 0);
    putbit("stdcall", 0);
    putbit("decorate", 0);
    putbit("cref", 0);
    putbit("nomixedstrlen", 0);
    putbit("typed", 0);
    putbit("recursive", 0);
    putval("returnval", 0);
    putbit("Cfunc", 0);
    putbit("uplevel", 0);
    putbit("internref", 0);
    putval("rout", 0);
    putval("paramcount", 0);
    putval("vtoff", 0);
    putval("invobj", 0);
    putbit("invobjinc", 0);
    putbit("class", 0);
    putbit("mlib", 0);
    putbit("clib", 0);
    putbit("inmodproc", 0);
    putbit("cudamodule", 0);
    putbit("fwdref", 0);
    putbit("aret", 0);
    putbit("vararg", 0);
    putbit("has_opts", 0);
    putbit("parref", 0);
    putbit("is_interface", 0);
    putval("assocptr", 0);
    putval("ptrtarget", 0);
    putbit("prociface", 0);
    strip = 1;
    break;

  case ST_NML:
    putval("line", NML_LINENO(CMEMFG(sptr)));
    putbit("ref", REFG(sptr));
    putval("plist", ADDRESSG(sptr));
    count = 0;
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i)) {
      ++count;
    }
    putval("count", count);
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i)) {
      putsym(NULL, NML_SPTR(i));
    }
    strip = 1;
    break;

  case ST_PARAM:
    putbit("decl", DCLDG(sptr));
    putbit("private", PRIVATEG(sptr));
    putbit("ref", REFG(sptr));
    if (TY_ISWORD(DTY(dtype))) {
      putval("val", CONVAL1G(sptr));
    } else {
      putsym("sym", CONVAL1G(sptr));
    }
    strip = 1;
    break;

  case ST_PLIST:
    putbit("ccsym", CCSYMG(sptr));
    putbit("init", DINITG(sptr));
    /*if( SCOPEG(sptr) == stb.curr_scope ){
        putbit( "init", DINITG(sptr) );
    }else{
        putbit( "init", 0 );
    }*/
    putbit("ref", 0); /* ref bit needs to be zero, so an address
                       * can be assigned */
    if (gbl.internal <= 1 || INTERNALG(sptr)) {
      /* for outer procedures, all symbols are not uplevel */
      putbit("uplevel", 0);
      putbit("internref", 0);
    } else {
      putbit("uplevel", 1);
      if (INTERNREFG(sptr))
        putbit("internref", 1);
      else
        putbit("internref", 0);
    }
    putbit("parref", PARREFG(sptr));
    putval("count", PLLENG(sptr));
    putval("etls", ETLSG(sptr));
    putbit("tls", TLSG(sptr));
    strip = 1;
    break;

  case ST_PROC:
    inmod = SCOPEG(sptr);
    if (inmod && STYPEG(inmod) == ST_ALIAS) {
      inmod = SCOPEG(inmod);
    }
    if (inmod && STYPEG(inmod) == ST_MODULE) {
      if (strcmp(SYMNAME(inmod), "cudadevice") == 0)
        cudamodule = 1;
    }
    if (!INMODULEG(sptr) || (inmod && STYPEG(inmod) != ST_MODULE)) {
      /* not actually in the module */
      inmod = 0;
    }
    putsym("altname", ALTNAMEG(sptr));
    putbit("ccsym", CCSYMG(sptr) || HCCSYMG(sptr));
    putbit("decl", DCLDG(sptr));
    dll = 0;
#if defined(TARGET_WIN)
    if (SCG(sptr) != SC_DUMMY)
      dll = DLLG(sptr);
#endif
    putval("dll", dll);
    putbit("dllexportmod", 0);
#if defined(CUDAG)
    putval("cmode", CUDAG(sptr));
#else
    putval("cmode", 0);
#endif
    putbit("func", FUNCG(sptr));
    putsym("inmodule", inmod);
    putbit("mscall", MSCALLG(sptr));
    putbit("needmod", 0);
    putbit("pure", PUREG(sptr));
    putbit("ref", REFG(sptr));
    putbit("passbyval", PASSBYVALG(sptr));
    putbit("passbyref", PASSBYREFG(sptr));
    putbit("cstructret", CSTRUCTRETG(sptr));
    putbit("sdscsafe", SDSCSAFEG(sptr));
    putbit("stdcall", STDCALLG(sptr));
    putbit("decorate", DECORATEG(sptr));
#ifdef CREFP
    putbit("cref", CREFG(sptr));
    putbit("nomixedstrlen", NOMIXEDSTRLENG(sptr));
#else
    putbit("cref", 0);
    putbit("nomixedstrlen", 0);
#endif
    putbit("typed", TYPDG(sptr));
    putbit("recursive", RECURG(sptr));
    putsym("returnval", FVALG(sptr));
    putbit("Cfunc", CFUNCG(sptr));
    if (SCG(sptr) != SC_DUMMY || gbl.internal <= 1 || INTERNALG(sptr)) {
      /* nondummy procedures are not uplevel; dummy
       * outer procedures, all symbols are not uplevel.
       */
      putbit("uplevel", 0);
      putbit("internref", 0);
    } else {
      /* dummy procedure, defined in host */
      putbit("uplevel", 1);
      if (INTERNREFG(sptr))
        putbit("internref", 1);
      else
        putbit("internref", 0);
    }

    if (gbl.stbfil && DTY(DTYPEG(sptr) + 2)) {
      /* Need to do this earlier so that it lowers the descriptor */
      fvalfirst = fvallast = 0;
      retdesc = CLASS_NONE;
      if (CFUNCG(sptr)) {
        retdesc = check_return(DTYPEG(FVALG(sptr)));
        if (retdesc != CLASS_MEM && retdesc != CLASS_PTR) {
          /* retval is sc_local */
        }
      }
      if (!POINTERG(sptr) && (retdesc == CLASS_NONE || retdesc == CLASS_MEM ||
                              retdesc == CLASS_PTR)) {
        switch (DTY(dtype)) {
        case TY_CMPLX:
        case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
        case TY_QCMPLX:
#endif
          if (FVALG(sptr))
            fvallast = 1;
          break;
        case TY_CHAR:
        case TY_NCHAR:
          if (FVALG(sptr) && !ADJLENG(FVALG(sptr)))
            fvallast = 1;
          break;
        case TY_DERIVED:
        case TY_STRUCT:
          if (FVALG(sptr))
            fvalfirst = 1;
          break;
        default:
          break;
        }
      }
      count = 0;
      altreturn = 0;
      params = DPDSCG(sptr);
      for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
        if (aux.dpdsc_base[params + i]) {
          ++count;
        } else {
          ++altreturn;
        }
      }
    }

#if defined(ACCROUTG)
    putval("rout", ACCROUTG(sptr));
    routx = ACCROUTG(sptr);
#else
    putval("rout", 0);
#endif
    if (gbl.stbfil && DTY(DTYPEG(sptr) + 2))
      putval("paramcount", count + fvalfirst + fvallast);
    else
      putval("paramcount", 0);
    putval("vtoff", VTOFFG(sptr));
    putval("invobj", INVOBJG(sptr));
    putbit("invobjinc", INVOBJINCG(sptr));
    putbit("class", CLASSG(sptr));
#ifdef LIBMG
    putbit("mlib", LIBMG(sptr));
    putbit("clib", LIBCG(sptr));
#else
    putbit("mlib", 0);
    putbit("clib", 0);
#endif
    putbit("inmodproc", SYMIG(sptr));
    putbit("cudamodule", cudamodule);
    putbit("fwdref", (inmod && IGNOREG(sptr)));
    putbit("aret", ARETG(sptr));
    putbit("vararg", 0);
    putbit("has_opts", has_opt_args(sptr) ? 1 : 0);
    putbit("parref", PARREFG(sptr));
    putbit("is_interface", IS_INTERFACEG(sptr));
    if (SCG(sptr) == SC_DUMMY)
      putval("descriptor", IS_PROC_DUMMYG(sptr) ? SDSCG(sptr) : 0);
    putsym("assocptr", ASSOC_PTRG(sptr));
    putsym("ptrtarget", PTR_TARGETG(sptr));
    putbit("prociface", IS_PROC_PTR_IFACEG(sptr)); 
    if (gbl.stbfil && DTY(DTYPEG(sptr) + 2)) {
      if (fvalfirst) {
        putsym(NULL, FVALG(sptr));
      }
      for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
        if (aux.dpdsc_base[params + i]) {
          putsym(NULL, aux.dpdsc_base[params + i]);
        }
      }
      if (fvallast) {
        putsym(NULL, FVALG(sptr));
      }
    }
    strip = 1;
    break;

  case ST_TYPEDEF:
    putbit("frommod", FROMMODG(sptr));
#if !defined(PARENTG)
    putval("parent", 0);
    putval("descriptor", 0);
    putbit("class", 0);
    if (all_default_init(DTYPEG(sptr))) {
      putbit("alldefaultinit", 1);
    } else {
      putbit("alldefaultinit", 0);
    }
    putbit("unlpoly", 0);
    putbit("isocbind", 0);
#else
    putval("parent", PARENTG(sptr));
    putval("descriptor", SDSCG(sptr));
    putbit("class", CLASSG(sptr));
    if (all_default_init(DTYPEG(sptr))) {
      putbit("alldefaultinit", 1);
    } else {
      putbit("alldefaultinit", 0);
    }
    putbit("unlpoly", UNLPOLYG(sptr));
    putbit("isoctype", ISOCTYPEG(sptr));
    putval("typedef_init", TYPDEF_INITG(sptr));
#endif
    strip = 1;
    break;

  case ST_GENERIC:
    putval("gsame", -1);
    putval("count", -1);
    strip = 1;
    break;
  case ST_USERGENERIC:
    putval("gsame", GSAMEG(sptr));
    count = 0;
    for (desc = GNDSCG(sptr); desc; desc = SYMI_NEXT(desc)) {
      int s = SYMI_SPTR(desc);
      if (VISITG(s))
        ++count;
    }
    putval("count", count);
    for (desc = GNDSCG(sptr); desc; desc = SYMI_NEXT(desc)) {
      int s = SYMI_SPTR(desc);
      if (VISITG(s)) {
        putsym(NULL, s);
      }
    }
    strip = 1;
    break;

  case ST_INTRIN:
  case ST_PD:
  case ST_STAG:
    break;

  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_STFUNC:
  case ST_ISOC:
  case ST_ISOFTNENV:
  case ST_ARRDSC:
  case ST_ALIAS:
  case ST_OPERATOR:
  case ST_CONSTRUCT:
  case ST_CRAY:
    break;

  case ST_BLOCK:
    putsym("enclfunc", ENCLFUNCG(sptr));
    putval("startline", STARTLINEG(sptr));
    putval("end", ENDLINEG(sptr));
    putsym("startlab", STARTLABG(sptr));
    putsym("endlab", ENDLABG(sptr));
#ifdef PARUPLEVELG
    putval("paruplevel", PARUPLEVELG(sptr));
#endif
    if (PARSYMSG(sptr)) {
      LLUplevel *up = llmp_get_uplevel(sptr);
      int count = 0;
      putval("parent", up->parent);
      /* recount parsymsct, don't count ST_ARRDSC */
      for (i = 0; i < up->vals_count; ++i) {
        if (up->vals[i] && STYPEG(up->vals[i]) == ST_ARRDSC)
          count++;
      }
      putval("parsymsct", (up->vals_count - count));
      for (i = 0; i < up->vals_count; ++i) {
        if (up->vals[i] && STYPEG(up->vals[i]) == ST_ARRDSC)
          continue;
        putsym(NULL, up->vals[i]);
      }
    } else {
      LLUplevel *up = llmp_has_uplevel(sptr);
      if (up) {
        putval("parent", up->parent);
      } else {
        putval("parent", 0);
      }
      putval("parsymsct", 0);
    }

    strip = 1;
    break;
  }
  if (name == NULL && sptr >= first_temp) {
    sprintf(tempname, "T$%d", sptr);
    namelen = strlen(tempname);
    name = tempname;
  }
  if (namelen > 0 && strip) {
    while (name[namelen - 1] == ' ')
      --namelen;
  }
  fprintf(lowersym.lowerfile, " %d:", namelen);
  if (namelen > 0) {
    if (newline) {
      putc('\n', lowersym.lowerfile);
      putc('=', lowersym.lowerfile);
      while (namelen) {
        int namec;
        namec = *name;
        namec = namec & 0xff;
        fprintf(lowersym.lowerfile, "%2.2x", namec);
        /* yes, this could be all on one line, but a good compiler
         * should generate good code nevertheless.
         * [end of patronizing religious proselytizing] */
        ++name;
        --namelen;
      }
    } else {
      /* printf doesn't work, since the 'name' can have embedded '\0's */
      while (namelen) {
        putc(*name, lowersym.lowerfile);
        /* yes, this could be all on one line, but a good compiler
         * should generate good code nevertheless.
         * [end of patronizing religious proselytizing] */
        ++name;
        --namelen;
      }
    }
  }
  fprintf(lowersym.lowerfile, "\n");
} /* lower_symbol */

/* lower symbol to ilm file and optionally to stb file */
static void
lower_symbol_stb(int sptr)
{
  lower_symbol(sptr);
  if (STB_LOWER()) {
    FILE *tmpfile = lowersym.lowerfile;
    lowersym.lowerfile = gbl.stbfil;
    lower_symbol(sptr);
    lowersym.lowerfile = tmpfile;
  }
}

/* If the  _V_ passbyvalue variable has  been marked
   VISITP, then propagate that info to the corresping
   SC_LOCAL variable
 */
static void
propagate_byval_visit(int sptr)
{
  char *name;
  int origptr;

  if (!PASSBYVALG(sptr) || !VISITG(sptr))
    return;
  origptr = MIDNUMG(sptr);
  if (origptr) {
    VISITP(origptr, 1);
    return;
  }
  name = SYMNAME(sptr);

  if (SCG(sptr) == SC_DUMMY && SCOPEG(sptr) != gbl.currsub)
    return;
}

void
lower_symbols(void)
{
  SPTR sptr;
  FILE *tfile;
  bool is_interface;
  SPTR scope;

  if (OUTPUT_DWARF)
    scan_for_dwarf_module();

  for (sptr = 1; sptr < stb.stg_avail; ++sptr) {
    if (SCG(sptr) == SC_DUMMY)
      propagate_byval_visit(sptr);

    if (FVALG(gbl.currsub) == sptr) {
      if (CFUNCG(gbl.currsub) || (CMPLXFUNC_C && DT_ISCMPLX(DTYPEG(sptr)))) {
        SCP(sptr, SC_LOCAL);
      }
    }
    if (VISITG(sptr) && STYPEG(sptr) == ST_ALIAS) {
      /* do not lower ST_ALIAS */
      int sptr2 = SYMLKG(sptr);
      VISITP(sptr, 0);
      if (sptr2 > NOSYM) {
        VISITP(sptr2, 1);
        if (sptr2 < sptr) {
          lower_symbol_stb(sptr2);
          VISIT2P(sptr2, 0);
        }
      }
    }
    if (VISITG(sptr) && STYPEG(sptr) == ST_TYPEDEF && BASETYPEG(sptr)) {
      lower_put_datatype_stb(BASETYPEG(sptr));
    }
    if (VISITG(sptr) && STYPEG(sptr) == ST_PROC) {
      SPTR sym = ASSOC_PTRG(sptr);
      if (sym > NOSYM && !VISITG(sym)) {
        lower_symbol(sym);
        VISITP(sym, 1);
      }
      sym = PTR_TARGETG(sptr);
      if (sym > NOSYM && !VISITG(sym)) {
        lower_symbol(sym);
        VISITP(sym, 1);
      }
    }
    if (VISITG(sptr) && is_procedure_ptr(sptr)) {
      /* make sure we lower type and subtype of procedure ptr */
      int dtype = DTYPEG(sptr);
      lower_put_datatype_stb(dtype);
      lower_put_datatype_stb(DTY(dtype + 1));
    }
    scope = SCOPEG(sptr);
    is_interface = ((STYPEG(scope) == ST_PROC || STYPEG(scope) == ST_ENTRY) &&
    IS_INTERFACEG(scope));

    if (!is_interface && STYPEG(sptr) == ST_TYPEDEF) {
      SPTR tag = DTY(DTYPEG(sptr) + 3);
      if (!VISITG(tag)) {
        SPTR sdsc = SDSCG(tag);
        lower_put_datatype_stb(DTYPEG(tag));
        lower_symbol_stb(tag);
        VISITP(tag, 1);
        if (sdsc && !VISITG(sdsc)) {
          VISITP(sdsc, 1);
          lower_put_datatype_stb(DTYPEG(sdsc));
        }
      }
    } else if (!VISITG(sptr) && CLASSG(sptr) && DESCARRAYG(sptr) &&
               STYPEG(sptr) == ST_DESCRIPTOR) {
      if (PARENTG(sptr) && !is_interface) {
        /* Only perform this if PARENT is set. Also do not create type
         * descriptors for derived types defined inside interfaces. When
         * derived types are defined inside interfaces, type descriptors are
         * not needed because there is no executable code inside an interface.
         * Furthermore, if we generate them, we might get multiple definitions
         * of the same type descriptor.
         */
        lower_put_datatype_stb(DTYPEG(sptr));
        VISITP(sptr, 1);
        lower_symbol_stb(sptr);
        VISIT2P(sptr, 0);
        lower_put_datatype_stb(PARENTG(sptr));
      }
    } else if (!VISITG(sptr) && STYPEG(sptr) == ST_MEMBER && FINALG(sptr)) {
      int vt = VTABLEG(sptr);
      lower_put_datatype_stb(ENCLDTYPEG(sptr));
      VISITP(sptr, 1);
      lower_symbol_stb(sptr);
      VISIT2P(sptr, 0);
      if (INMODULEG(vt) && !VISITG(vt)) {
        VISITP(vt, 1);
        if (vt < sptr) {
          lower_symbol_stb(vt);
        }
        VISIT2P(vt, 0);
      }
    } else if (/*!VISITG(sptr) &&*/ CLASSG(sptr) && DESCARRAYG(sptr) &&
               STYPEG(sptr) == ST_DESCRIPTOR &&
               (!UNLPOLYG(sptr) || STYPEG(SCOPEG(sptr)) != ST_MODULE)) {
      /* this occurs when we have a parent type descriptor
       * that's not directly used but we still need to
       * generate it for its children types.
       */
      VISITP(sptr, 1);
      lower_put_datatype_stb(DTYPEG(sptr));
      lower_put_datatype_stb(PARENTG(sptr));
    } else if (!CLASSG(sptr) && !VISITG(sptr) && STYPEG(sptr) == ST_MEMBER) {
      /* FS#18558 - need to lower members if derived type
       * contains type bound procedures. Otherwise, we may
       * not be able to generate "virtual function tables".
       */
      int dtype = ENCLDTYPEG(sptr);
      if (has_tbp_or_final(dtype)) {
        int mem;
        lower_put_datatype_stb(dtype);
        for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
          int dt_mem = DTYPEG(mem);
          lower_put_datatype_stb(dt_mem);
          if (DTY(dt_mem) == TY_ARRAY) {
            /* FS#19034 - must also lower array subtype */
            lower_put_datatype_stb(DTY(dt_mem + 1));
          }
          if (0 && mem != sptr) {
            lower_symbol_stb(mem);
          }
          VISITP(mem, 1);
        }
      }

    } else if (CLASSG(sptr) && CCSYMG(sptr) && STYPEG(sptr) == ST_MEMBER) {
      int bind = BINDG(sptr);
      int vt = VTABLEG(sptr);
      if (STYPEG(vt) == ST_PROC || STYPEG(vt) == ST_ENTRY ||
          STYPEG(vt) == ST_OPERATOR || STYPEG(vt) == ST_USERGENERIC ||
          STYPEG(vt) == ST_MODPROC) {
        if (vt && !VISITG(vt) && !IFACEG(sptr)) {
          STYPEP(vt, ST_PROC);
          CCSYMP(vt, 0);
          VISITP(vt, 1);
          lower_put_datatype_stb(DTYPEG(vt));
          if (bind && vt < sptr) {
            lower_symbol_stb(vt);
          }
          VISIT2P(vt, 0);
        }
        if (bind && !VISITG(bind) && STYPEG(bind) == ST_PROC) {
          VISITP(bind, 1);
          lower_put_datatype_stb(DTYPEG(bind));
          lower_symbol_stb(bind);
          VISIT2P(bind, 0);
        }
      }
    } else if (!VISITG(sptr) && STYPEG(sptr) == ST_TYPEDEF && SDSCG(sptr) &&
               CLASSG(SDSCG(sptr)) && !PARENTG(sptr)) {
      /* Force generation of type descriptors in the mod object file */
      VISITP(sptr, 1);
      lower_put_datatype_stb(DTYPEG(sptr));
    } else if (flg.debug && !VISITG(sptr) && STYPEG(sptr) == ST_PARAM) {
      /* lower parameter in module for debugging purpose */
      int sym = 0;
      if (!ENCLFUNCG(sptr) || !SCOPEG(sptr))
        continue;
      if (ENCLFUNCG(sptr) && !NEEDMODG(ENCLFUNCG(sptr))) {
        continue;
      } else if (SCOPEG(sptr) && !NEEDMODG(SCOPEG(sptr))) {
        continue;
      }
      if (DTY(DTYPEG(sptr)) == TY_ARRAY || DTY(DTYPEG(sptr)) == TY_DERIVED)
        sym = CONVAL1G(sptr);
      else if (CONVAL2G(sptr))
        sym = sym_of_ast(CONVAL2G(sptr));
      if (sym && VISITG(sym)) {
        VISITP(sptr, 1);
        lower_put_datatype_stb(DTYPEG(sptr));
      }
    }
    else if (VISITG(sptr)) {
      int scope = SCOPEG(sptr);
      if (scope && STYPEG(scope) == ST_PROC && FVALG(scope) == sptr) {
        lower_put_datatype_stb(DTYPEG(sptr));
      }
    }

    if (VISITG(sptr)) {
      lower_symbol_stb(sptr);
    }
    VISIT2P(sptr, 0);

    /* Unfreeze intrinsics for re/use in internal routines.
     *
     * This isn't quite right.  It favors declarations in an internal routine
     * at the possible expense of cases where a host routine declaration
     * should be accessible in an internal routine.  It might be useful to
     * have multiple freeze bits, such as one for a host routine and one
     * for the current internal routine.  That would allow more accurate
     * diagnosis of errors in internal routines.
     *
     * Unfortunately, multiple bits would require analysis of existing cases
     * where the bit is set and referenced, and there is a combinatorial
     * explosion of cases mixing various declarations and uses.  For the LEN
     * intrinsic, for example, some possible declaration cases are:
     *
     *  - INTEGER :: LEN ! (ambiguous) LEN may be a var or an intrinsic
     *  - INTEGER, INTRINISC :: LEN ! LEN is an intrinsic
     *  - <no declaration> -- (first) use determines what LEN is
     *
     * Some reference possibilities are:
     *
     *  - LEN() is an (intrinsic) function call
     *  - LEN is a (scalar) var reference
     *
     * These declarations and references can be present in any combination
     * in a host routine, in an internal routine, or both.  Many of these
     * combinations are valid, but not all.  Compilation currently mishandles
     * some of these variants.  The choice to clear the "freeze" bit here is
     * a compromise attempt intended to favor correct compilation of valid
     * programs above diagnosis of error cases.
     */
    if (IS_INTRINSIC(STYPEG(sptr)))
      EXPSTP(sptr, 0);
  }
  if (gbl.internal > 1) {
    for (sptr = gbl.outerentries; sptr > NOSYM; sptr = SYMLKG(sptr)) {
      if (sptr != gbl.outersub) {
        putival("Entry", sptr);
        fprintf(lowersym.lowerfile, "\n");
        if (STB_LOWER())
          fprintf(gbl.stbfil, "\n");
      }
    }
  }
  if (XBIT(53, 2)) {
    lower_pstride_info(lowersym.lowerfile);
    if (STB_LOWER())
      lower_pstride_info(gbl.stbfil);
  }
  for (sptr = 1; sptr < stb.stg_avail; ++sptr) {
    int socptr;
    if (!VISITG(sptr))
      continue;
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
      socptr = SOCPTRG(sptr);
      if (socptr) {
        int s, n;
        n = 0;
        for (s = socptr; s; s = SOC_NEXT(s)) {
          ++n;
        }
#if DEBUG
        if (DBGBIT(47, 8) && sptr > NOSYM) {
          fprintf(lowersym.lowerfile, "overlap:%s", getprint(sptr));
        } else
#endif
          putival("overlap", sptr);
        putval("count", n);
        if (STB_LOWER()) {
          tfile = lowersym.lowerfile;
          lowersym.lowerfile = gbl.stbfil;
          if (DBGBIT(47, 8) && sptr > NOSYM)
            fprintf(gbl.stbfil, "overlap:%s", getprint(sptr));
          else
            putival("overlap", sptr);
          putval("count", n);
          lowersym.lowerfile = tfile;
        }
        for (s = socptr; s; s = SOC_NEXT(s)) {
          int overlap;
          overlap = SOC_SPTR(s);
#if DEBUG
          if (DBGBIT(47, 8) && overlap > NOSYM) {
            fprintf(lowersym.lowerfile, " %s", getprint(overlap));
            if (STB_LOWER())
              fprintf(gbl.stbfil, " %s", getprint(overlap));
          } else
#endif
          {
            fprintf(lowersym.lowerfile, " %d", overlap);
            if (STB_LOWER())
              fprintf(gbl.stbfil, " %d", overlap);
          }
        }
        fprintf(lowersym.lowerfile, "\n");
        if (STB_LOWER())
          fprintf(gbl.stbfil, "\n");
      }
      break;
    default:
      break;
    }
  }
  /* restore TY_PTR stuff to its original type */
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    int dtype;
    switch (STYPEG(sptr)) {
    case ST_MEMBER:
      dtype = DTYPEG(sptr);
      if (DTY(dtype) == TY_PTR && dtype != DT_ADDR &&
          DTY(DTY(dtype + 1)) != TY_PROC) {
        DTYPEP(sptr, DTY(dtype + 1));
      }
      break;
    default:;
    }
    if (DTY(DTYPEG(sptr)) == TY_ARRAY && LNRZDG(sptr) && XBIT(52, 4)) {
      /* restore the old linearized datatype from the stashed type */
      dtype = DTYPEG(sptr);
      dtype = -DTY(dtype - 1);
      DTYPEP(sptr, dtype);
    }
  }
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    /* restore data types of procedures/entries */
    if (STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_ENTRY) {
      if (FVALG(sptr)) {
        DTYPEP(sptr, DTYPEG(FVALG(sptr)));
      }
    }
  }
} /* lower_symbols */

/** \brief Reset temps for next statement */
void
lower_reset_temps(void)
{
  int sptr, nextsptr;
  for (sptr = first_used_scalarptr_temp; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = SYMLKG(sptr);
    SYMLKP(sptr, first_avail_scalarptr_temp);
    first_avail_scalarptr_temp = sptr;
  }
  first_used_scalarptr_temp = 0;
  for (sptr = first_used_scalar_temp; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = SYMLKG(sptr);
    SYMLKP(sptr, first_avail_scalar_temp);
    first_avail_scalar_temp = sptr;
  }
  first_used_scalar_temp = 0;
} /* lower_reset_temps */

/** \brief Return a symbol which is a temp scalar of DTYPE 'dtype' */
int
lower_scalar_temp(int dtype)
{
  int sptr, lastsptr, nextsptr;
  for (lastsptr = 0, sptr = first_avail_scalar_temp; sptr > NOSYM;
       lastsptr = sptr, sptr = nextsptr) {
    nextsptr = SYMLKG(sptr);

    if (DTYPEG(sptr) == dtype && SCG(sptr) == lowersym.sc) {
      /* remove from this list, add to 'used' list, return it */
      if (lastsptr) {
        SYMLKP(lastsptr, nextsptr);
      } else {
        first_avail_scalar_temp = nextsptr;
      }
      SYMLKP(sptr, first_used_scalar_temp);
      first_used_scalar_temp = sptr;
      return sptr;
    }
  }
  /* make a 'dtype' variable to be the temp */
  sptr = getccsym_sc('C', ++lowersym.Ccount, ST_VAR, lowersym.sc);
  DTYPEP(sptr, dtype);
  SYMLKP(sptr, first_used_scalar_temp);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  first_used_scalar_temp = sptr;
  return sptr;
} /* lower_scalar_temp */

/** \brief For an ST_MEMBER of an anonymous structure/union,
    fill member_parent[sptr] with the symbol name of
    its parent structure
 */
void
lower_fill_member_parent(void)
{
  int sptr;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    int tag, s;
    int dtype = DTYPEG(sptr);
    switch (DTY(dtype)) {
    case TY_DERIVED:
    case TY_STRUCT:
    case TY_UNION:
      tag = DTY(dtype + 3);
      if (tag == 0) {
        /* look through the linked list of members;
         * make each member point back to this tag */
        for (s = DTY(dtype + 1); s > NOSYM; s = SYMLKG(s)) {
          if (LOWER_MEMBER_PARENT(s)) {
            lerror("symbol %s (%d) appears in two anonymous structs",
                   SYMNAME(s), s);
          }
          LOWER_MEMBER_PARENT(s) = sptr;
        }
      }
      break;
    default:
      break;
    }
  }
} /* lower_fill_member_parent */

void
lower_mark_entries(void)
{
  int ent;
  /* always mark the current routine or block data, ... */
  lower_visit_symbol(gbl.currsub);

  /* mark any entry points, also */
  for (ent = gbl.entries; ent > NOSYM; ent = SYMLKG(ent)) {
    int params, i;
    lower_visit_symbol(ent);
    /* mark any parameters, unless a module name */
    if (STYPEG(ent) != ST_MODULE) {
      params = DPDSCG(ent);
      for (i = 0; i < (int)(PARAMCTG(ent)); ++i) {
        int parm = aux.dpdsc_base[params + i];
        if (parm) {
          lower_visit_symbol(parm);
        }
      }
    }
  }
  if (gbl.internal > 1) {
    if (lowersym.outersub) {
      lower_visit_symbol(lowersym.outersub);
    }
    for (ent = gbl.outerentries; ent > NOSYM; ent = SYMLKG(ent)) {
      int params, i;
      lower_visit_symbol(ent);
      /* mark any parameters, unless a module name */
      params = DPDSCG(ent);
      for (i = 0; i < (int)(PARAMCTG(ent)); ++i) {
        int parm = aux.dpdsc_base[params + i];
        if (parm) {
          lower_visit_symbol(parm);
        }
      }
    }
  }
} /* lower_mark_entries */

int
lower_lab(void)
{
  int lab;
  lab = getlab();
  RFCNTP(lab, 0);
  return lab;
} /* lower_lab */

int
lowersym_pghpf_cmem(int *whichmem)
{
  int ptr;
  int base;

  if (*whichmem == 0)
    lower_add_pghpf_commons();

  if (!XBIT(57, 0x8000)) {
    if (whichmem == &lowersym.ptr0)
      return plower("oS", "ACON", *whichmem);
    if (whichmem == &lowersym.ptr0c)
      return plower("oS", "ACON", *whichmem);
  }

  if (!XBIT(70, 0x80000000))
    return plower("oS", "BASE", *whichmem);

  ptr = MIDNUMG(*whichmem);
  base = plower("oS", "BASE", ptr);
  return plower("oiS", "PLD", base, *whichmem);
}

/* Checks to see if array bound ast is an expression that uses a type parameter.
 * This function is mirrored in semutil2.c.
 * TO DO: Move this function to dtypeutl.c, make it extern, and remove the
 * instance in semutil2.c.
 */
static int
valid_kind_parm_expr(int ast)
{
  int sptr, rslt, i;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return valid_kind_parm_expr(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 1;
  case A_MEM:
    sptr = memsym_of_ast(ast);
    if (KINDG(sptr))
      return 1;
    return 0;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (KINDG(sptr))
      return 1;
    return 0;
  case A_CONV:
  case A_UNOP:
    return valid_kind_parm_expr(A_LOPG(ast));
  case A_BINOP:
    rslt = valid_kind_parm_expr(A_LOPG(ast));
    if (!rslt)
      return 0;
    rslt = valid_kind_parm_expr(A_ROPG(ast));
    if (!rslt)
      return 0;
    return 1;
  }
  return 0;
}

static int
is_descr_expression(int ast)
{

  int sptr, rslt, i;

  if (!ast)
    return 0;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return is_descr_expression(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 0;
  case A_MEM:
    sptr = memsym_of_ast(ast);
    if (DESCARRAYG(sptr))
      return 1;
    return 0;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (DESCARRAYG(sptr))
      return 1;
    return 0;
  case A_SUBSCR:
  case A_CONV:
  case A_UNOP:
    return is_descr_expression(A_LOPG(ast));
  case A_BINOP:
    rslt = is_descr_expression(A_LOPG(ast));
    if (rslt)
      return 1;
    rslt = is_descr_expression(A_ROPG(ast));
    if (!rslt)
      return 0;
    return 1;
  }
  return 0;
}

static void
lower_fileinfo_llvm()
{
  int fihx;
  const char *dirname, *filename, *funcname, *fullname;

  if (!STB_LOWER())
    return;
  fihx = curr_findex;

  for (; fihx < fihb.stg_avail; ++fihx) {
    dirname = FIH_DIRNAME(fihx);
    if (dirname == NULL)
      dirname = "";
    filename = FIH_FILENAME(fihx);
    if (filename == NULL)
      filename = "";
    funcname = FIH_FUNCNAME(fihx);
    if (funcname == NULL)
      funcname = "";
    fullname = FIH_FULLNAME(fihx);
    if (fullname == NULL)
      fullname = "";

    fprintf(gbl.stbfil,
            "fihx:%d tag:%d parent:%d flags:%d lineno:%d "
            "srcline:%d level:%d next:%d %" GBL_SIZE_T_FORMAT
            ":%s %" GBL_SIZE_T_FORMAT ":%s %" GBL_SIZE_T_FORMAT
            ":%s %" GBL_SIZE_T_FORMAT ":%s\n",
            fihx, FIH_FUNCTAG(fihx), FIH_PARENT(fihx), FIH_FLAGS(fihx),
            FIH_LINENO(fihx), FIH_SRCLINE(fihx), FIH_LEVEL(fihx),
            FIH_NEXT(fihx), strlen(dirname), dirname, strlen(filename),
            filename, strlen(funcname), funcname, strlen(fullname), fullname);
  }
  curr_findex = fihx;

} /* lower_fileinfo_llvm */

static void
stb_lower_sym_header()
{
  ISZ_T bss_addr;
  INITEM *p;
  static int first_time = 1;
  FILE *tmpfile = lowersym.lowerfile;

  if (!STB_LOWER()) {
    if (first_time)
      first_time = 0;
    return;
  }

  lowersym.lowerfile = gbl.stbfil;

  /* Following code is copied from lower_sym_header */
  if (first_time) {
    /* put out any saved inlining information */
    first_time = 0;
    for (p = inlist; p; p = p->next) {
      putival("inline", p->level);
      putlval("offset", p->offset);
      putval("which", p->which);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s", strlen(p->name),
              p->name);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s",
              strlen(p->cname), p->cname);
      fprintf(lowersym.lowerfile, " %" GBL_SIZE_T_FORMAT ":%s",
              strlen(p->filename), p->filename);
      putlval("objoffset", p->objoffset);
      putval("base", p->staticbase);
      putval("size", p->size);
      fprintf(lowersym.lowerfile, "\n");
    }
    fprintf(lowersym.lowerfile, "ENDINLINE\n");
  }

  /* put out header lines */
  fprintf(lowersym.lowerfile, "TOILM version %d/%d\n", VersionMajor,
          VersionMinor);
  if (gbl.internal == 1 && gbl.empty_contains)
    putvline("Internal", 0);
  else 
    putvline("Internal", gbl.internal);
  if (gbl.internal > 1) {
    putvline("Outer", lowersym.outersub);
    putvline("First", stb.firstusym);
  }
  putvline("Symbols", stb.stg_avail - 1);
  putvline("Datatypes", stb.dt.stg_avail - 1);
  bss_addr = get_bss_addr();
  putvline("BSS", bss_addr);
  putvline("GBL", gbl.saddr);
  putvline("LOC", gbl.locaddr);
  putvline("STATICS", gbl.statics);
  putvline("LOCALS", gbl.locals);
  putvline("PRIVATES", private_addr);
  if (saveblockname) {
    putvline("GNAME", saveblockname);
  }
  lowersym.lowerfile = tmpfile;

} /* lower_sym_header */

typedef struct old_dscp {
  int sptr;
  int dpdsc;
  int paramct;
  int fval;
} OLD_DPDSC;

static OLD_DPDSC *save_dpdsc = NULL;
static int save_dpdsc_cnt = 0;

static void
llvm_check_retval_inargs(int sptr)
{
  int fval = FVALG(sptr);
  if (fval) {
    int dtype;
    int ent_dtype = DTYPEG(sptr);
    llvm_fix_args(sptr);
    dtype = DTYPEG(fval);
    fix_class_args(sptr);
    if (DTYPEG(sptr) != DT_NONE && makefvallocal(RU_FUNC, fval)) {
      SCP(fval, SC_LOCAL);
      if (is_iso_cptr(DTYPEG(fval))) {
        DTYPEP(fval, DT_CPTR);
      }
    }
    switch (DTY(dtype)) {
    case TY_ARRAY:
      if (aux.dpdsc_base[DPDSCG(sptr)] != fval) {
        DPDSCP(sptr, DPDSCG(sptr) - 1);
        *(aux.dpdsc_base + DPDSCG(sptr)) = fval;
        PARAMCTP(sptr, PARAMCTG(sptr) + 1);
        DTYPEP(sptr, DT_NONE);
        SCP(fval, SC_DUMMY);
      }
      break;
    case TY_CHAR:
    case TY_NCHAR:
      if (dtype != ent_dtype)
        return;
      if (!POINTERG(sptr) && ADJLENG(fval) && DPDSCG(sptr)) {

        if (aux.dpdsc_base[DPDSCG(sptr)] != fval) {
          DPDSCP(sptr, DPDSCG(sptr) - 1);
          *(aux.dpdsc_base + DPDSCG(sptr)) = fval;
          PARAMCTP(sptr, PARAMCTG(sptr) + 1);
          DTYPEP(sptr, DT_NONE);
          SCP(fval, SC_DUMMY);
        }
      }
      FLANG_FALLTHROUGH;
    case TY_DCMPLX:
      if (DTY(ent_dtype) != TY_DCMPLX) {
        return;
      }
      goto pointer_check;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      if (DTY(ent_dtype) != TY_QCMPLX) {
        return;
      }
      goto pointer_check;
#endif

    default:
      if (DTY(ent_dtype) == TY_DCMPLX ||
#ifdef TARGET_SUPPORTS_QUADFP
          DTY(ent_dtype) == TY_QCMPLX ||
#endif
          DTY(ent_dtype) == TY_CHAR || DTY(ent_dtype) == TY_NCHAR)
        return;

    pointer_check:
      if (aux.dpdsc_base[DPDSCG(sptr)] != fval &&
          (POINTERG(sptr) || ALLOCATTRG(fval) || (DTY(ent_dtype) == TY_DCMPLX)
#ifdef TARGET_SUPPORTS_QUADFP
          || (DTY(ent_dtype) == TY_QCMPLX)
#endif
      )) {
        if (DPDSCG(sptr) && DTYPEG(sptr) != DT_NONE) {

          DPDSCP(sptr, DPDSCG(sptr) - 1);
          *(aux.dpdsc_base + DPDSCG(sptr)) = fval;
          PARAMCTP(sptr, PARAMCTG(sptr) + 1);
          DTYPEP(sptr, DT_NONE);
          SCP(fval, SC_DUMMY);
        }
      }
      break;
    }
  }
}

static void
_stb_fixup_ifacearg(int sptr)
{
  int params, i, newdsc, fval;

  i = save_dpdsc_cnt;
  if (save_dpdsc_cnt == 0) {
    save_dpdsc_cnt = 1;
    NEW(save_dpdsc, OLD_DPDSC, save_dpdsc_cnt);
  } else {
    NEED(save_dpdsc_cnt + 1, save_dpdsc, OLD_DPDSC, save_dpdsc_cnt,
         save_dpdsc_cnt + 1);
  }

  fval = FVALG(sptr);
  save_dpdsc[i].sptr = sptr;
  save_dpdsc[i].dpdsc = DPDSCG(sptr);
  save_dpdsc[i].paramct = PARAMCTG(sptr);
  save_dpdsc[i].fval = fval;

  fix_class_args(sptr);
  if (INTERFACEG(sptr))
    return;

  llvm_check_retval_inargs(sptr);

  newdsc = newargs_for_llvmiface(sptr);
  llvm_iface_flag = TRUE;
  interface_for_llvmiface(sptr, newdsc);
  undouble_callee_args_llvmf90(sptr);
  params = DPDSCG(sptr);
  if (fval && NEWARGG(fval)) {
    FVALP(sptr, NEWARGG(fval));
    lower_visit_symbol(FVALG(sptr));
  }
  for (i = 0; i < (int)(PARAMCTG(sptr)); ++i) {
    int param = aux.dpdsc_base[params + i];
    if (param) {
      lower_visit_symbol(param);
    }
  }
  llvm_iface_flag = FALSE;
}

/* TODO: Note that for contained subroutine, we need remove to the added
 * argument
 * before entering the contained routine.  Then at lower, we need to put it back
 * again.
 */

void
stb_fixup_llvmiface()
{
  int sptr;
  /* go through iface symbols */
  for (sptr = 1; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_PROC) {
      if (SCG(sptr) == SC_NONE ||
          (SCG(sptr) == SC_EXTERN &&
           ((VISITG(sptr) && INMODULEG(sptr)) ||
            (DPDSCG(sptr) && VISITG(sptr)) ||
            (gbl.currsub && gbl.currsub == SCOPEG(sptr) &&
             NEEDMODG(gbl.currsub))))

      ) {
        _stb_fixup_ifacearg(sptr);
      }
    }
  }
}

void
uncouple_callee_args()
{
  int i, sptr;
  /* do it backward just in case there is a case where we overwrite the existing
   * one */
  for (i = (save_dpdsc_cnt - 1); i >= 0; i--) {
    sptr = save_dpdsc[i].sptr;
    DPDSCP(sptr, save_dpdsc[i].dpdsc);
    PARAMCTP(sptr, save_dpdsc[i].paramct);
    FVALP(sptr, save_dpdsc[i].fval);
    INTERFACEP(sptr, 0);
  }
  FREE(save_dpdsc);
  save_dpdsc = NULL;
  save_dpdsc_cnt = 0;
}

/**
   \brief Inspect a common block variable symbol to see if it has a alias 
   name, if YES, write to ilm file with attribute "has_alias" be 1 and
   followed by the length and name of the alias; if NO, put 0 to "has_alias".
 */
static void
check_debug_alias(SPTR sptr)
{
  if (gbl.rutype != RU_BDATA && STYPEG(sptr) == ST_VAR && SCG(sptr) == SC_CMBLK) {
    /* Create debug info for restricted import of module variables
     * and renaming of module variables */
    if (HASHLKG(sptr)) {
      if (STYPEG(HASHLKG(sptr)) == ST_ALIAS &&
          !strcmp(SYMNAME(sptr), SYMNAME(HASHLKG(sptr)))) {
        putbit("has_alias", 1);
        fprintf(lowersym.lowerfile, " %d:%s",
                strlen(SYMNAME(sptr)), SYMNAME(HASHLKG(sptr)));
      } else {
        SPTR candidate = sptr;
        while (candidate) {
          if (dbgref_symbol.altname[candidate] && 
              SYMLKG(dbgref_symbol.altname[candidate]->sptr) == sptr)
            break;
          candidate = HASHLKG(candidate);
        }
        if (candidate) {
          putbit("has_alias", 1);
          fprintf(lowersym.lowerfile, " %d:%s",
                  strlen(SYMNAME(dbgref_symbol.altname[candidate]->sptr)),
                  SYMNAME(dbgref_symbol.altname[candidate]->sptr));
        } else {
          putbit("has_alias", 0);
        }
      }
    } else {
      putbit("has_alias", 0);
    }
  }
}


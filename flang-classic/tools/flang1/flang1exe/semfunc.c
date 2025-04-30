/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran front-end utility routines used by Semantic Analyzer to
           process functions, subroutines, predeclareds, etc.
 */

#include "gbldefs.h"
#include "global.h"
#include "gramtk.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "ilmtp.h"
#include "semstk.h"
#include "pd.h"
#include "machar.h"
#include "ast.h"
#include "rte.h"
#include "rtlRtns.h"
#include "version.h"
#include "atomic_common.h"

#define MAX_ARGS_NUMBER 3
#define ARGS_NUMBER 3
#define MIN_ARGS_NUMBER 0

static struct {
  int nent;  /* number of arguments specified by user */
  int nargt; /* number actually needed for AST creation */
} carg;
static void add_typroc(int);
static void count_actuals(ITEM *);
static void count_formals(int);
static void count_formal_args(int, int);
static void check_dim_error(int, int);
static int mk_array_type(int, int);
static int gen_pointer_result(int, int, int, LOGICAL, int);
static int gen_allocatable_result(int, int, int, LOGICAL, int);
static int gen_array_result(int, int, int, LOGICAL, int);
static int gen_char_result(int, int, int);
static void precompute_arg_intrin(int, int);
static void precompute_args(int, int);
static void replace_arguments(int, int);
static void rewrite_triples(int, int, int);
static void rewrite_subscr(int, int, int);
static void replace_formal_triples(int, int, int);
static void replace_formal_bounds(int, int, int, int);
static int getMergeSym(int, int);
static void ref_pd_subr(SST *, ITEM *);
static void ref_intrin_subr(SST *, ITEM *);
static int set_kind_result(SST *, int, int);
static int set_shape_result(int, int);
static int _adjustl(int);
static int _adjustr(int);
static int _index(int, int, int);
static int _len_trim(int);
static int _repeat(int, int);
static int _scan(int, int, int);
static int _trim(int);
static int _verify(int, int, int);
static int find_byval_ref(int, int, int);
static int cmp_mod_scope(SPTR);

static void gen_init_intrin_call(SST *, int, int, int, int);
#ifdef I_C_ASSOCIATED
static int _c_associated(SST *, int);
#endif

static int get_type_descr_dummy(int sptr, int arg);
static int get_tbp(int sptr);
static void fix_proc_pointer_call(SST *, ITEM **);
static int find_by_name_stype_arg(char *, int, int, int, int, int);

static int _e74_sym;
static int _e74_cnt;
static int _e74_l;
static int _e74_u;
static int _e74_pos;
static char *_e74_kwd;
static void e74_cnt(int, int, int, int);
static void e74_arg(int, int, char *);
static int byvalue_ref_arg(SST *, int *, int, int);
static int gen_finalized_result(int fval, int func_sptr);

#define E74_CNT(s, c, l, u) (_e74_sym = s, _e74_cnt = c, _e74_l = l, _e74_u = u)
#define E74_ARG(s, p, k) (_e74_sym = s, _e74_pos = p, _e74_kwd = k)

#define ERR170(s) error(170, 2, gbl.lineno, s, CNULL)
#define HL_UF(s) \
  error(0, 3, gbl.lineno, "HPF Library procedure not implemented", SYMNAME(s))

#define GET_CVAL_ARG(i) get_sst_cval(ARG_STK(i))
#define GET_DBLE(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y)
#define GET_QUAD(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y);  \
  x[2] = CONVAL3G(y);  \
  x[3] = CONVAL4G(y);

static int byval_func_ptr = 0;
static int byval_dscptr = 0;
static int byval_paramct = 0;

#define PASS_BYVAL 1
#define PASS_BYREF 2
#define PASS_BYREF_NO_LEN 3
#define PASS_BYDEFAULT 0

/** \brief Return the "static type descriptor" for object sptr. The static
           type descriptor holds the "declared type" of an object.
 */
int
get_static_type_descriptor(int sptr)
{
  int sptrsdsc, dtype;

  dtype = DTYPEG(sptr);

  switch (DTY(dtype)) {
  case TY_DERIVED:
    break;
  case TY_ARRAY:
    dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED) {
      sptr = DTY(dtype + 3);
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    return 0; /* TBD - probably need other cases for unlimited
               * polymorphic entities.
               */
  }

  sptrsdsc = SDSCG(sptr);
  if (sptrsdsc <= NOSYM) {
    set_descriptor_class(1);
    get_static_descriptor(sptr);
    set_descriptor_class(0);
    sptrsdsc = SDSCG(sptr);
  }
  DESCUSEDP(sptr, TRUE);
  NODESCP(sptr, FALSE);
  PARENTP(sptrsdsc, DTYPEG(sptr));
  if (DTY(DTYPEG(sptr)) == TY_DERIVED) {
    /* make sure all parent types get a descriptor as well */
    DTYPE dt = DTYPEG(sptr);
    SPTR tag = get_struct_tag_sptr(dt);
    SPTR member = get_struct_members(dt);
    int init_ict = get_struct_initialization_tree(dt);

    if (init_ict > 0) {
      SPTR init_template = get_dtype_init_template(dt);
      if (init_template > NOSYM)
        sym_is_refd(init_template);
    }

    while (member > NOSYM && PARENTG(member)) {
      DTYPE dt = DTYPEG(member);
      if ((tag = get_struct_tag_sptr(dt)) <= NOSYM)
        break;
      if (!SDSCG(member)) {
        set_descriptor_class(TRUE); /* this means "needs a type pointer" */
        get_static_descriptor(member);
        set_descriptor_class(FALSE); /* reset static flag that was set above */
        DESCUSEDP(member, TRUE);
        NODESCP(member, FALSE);
        PARENTP(SDSCG(member), dt);
      }
      member = get_struct_members(DTYPEG(tag));
    }
  }
  return sptrsdsc;
}

static int
get_type_descr_dummy(int sptr, int arg)
{

  int count, i, count_class;
  int dscptr, count_descr;
  LOGICAL found = FALSE;

  fix_class_args(sptr);
  count = PARAMCTG(sptr);
  dscptr = DPDSCG(sptr);
  count_class = count_descr = 0;
  for (i = 0; i < count; ++i) {
    int arg2 = aux.dpdsc_base[dscptr + i];
    if (!found) {
      if (strcmp(SYMNAME(arg), SYMNAME(arg2)) != 0) {
        if (CLASSG(arg2) && !needs_descriptor(arg2))
          ++count_class;
      } else {
        found = TRUE;
      }
    } else if (CCSYMG(arg2) && CLASSG(arg2)) {
      if (count_class == count_descr) {
        return arg2;
      }
      ++count_descr;
    }
  }

  return 0;
}

/** \brief Return the type descriptor associated with \a arg (and \a func_sptr
   when
           \a arg is a dummy argument of routine \a func_sptr).
 */
int
get_type_descr_arg(int func_sptr, int arg)
{
  int sptr;

  if (needs_descriptor(arg)) {
    if (SDSCG(arg) <= NOSYM)
      get_static_descriptor(arg);
    return SDSCG(arg);
  }

  if (CLASSG(arg) && SCG(arg) == SC_DUMMY) {
    sptr = get_type_descr_dummy(func_sptr, arg);
    if (!sptr && gbl.internal > 1) {
      sptr = get_type_descr_dummy(gbl.outersub, arg);
    }
#if DEBUG
    assert(sptr, "get_type_descr_arg: NULL dummy descriptor ", arg, 4);
#endif
    return sptr;
  }
  if (!CLASSG(arg)) {
    DTYPE dtype = DTYPEG(arg);
    if (DTY(dtype) == TY_DERIVED) {
      /* not polymorphic, so just return declared type descriptor */
      arg = DTY(dtype + 3);
    }
  }
  sptr = get_static_type_descriptor(arg);

#if DEBUG
  assert(sptr, "get_type_descr_arg: NULL descriptor ", arg, 4);
#endif

  return sptr;
}

/** \brief Same as get_type_descr_arg(), but do not perform error check.
 */
int
get_type_descr_arg2(int func_sptr, int arg)
{
  int sptr;
  if (needs_descriptor(arg)) {
    int desc;
    if (SDSCG(arg))
      desc = SDSCG(arg);
    else {
      int orig_sc = get_descriptor_sc();
      set_descriptor_sc(SC_STATIC);
      get_static_descriptor(arg);
      set_descriptor_sc(orig_sc);
      desc = SDSCG(arg);
    }
    return desc;
  }

  if (CLASSG(arg) && SCG(arg) == SC_DUMMY) {
    sptr = get_type_descr_dummy(func_sptr, arg);
    return sptr;
  }

  sptr = get_static_type_descriptor(arg);

  return sptr;
}

/* check if this is a character parameter, passed by reference,
   no length needed in the function parameter list
  */
static int
pass_char_no_len(int func_sptr, int param_sptr)
{
  return (find_byval_ref(func_sptr, param_sptr, 0) == PASS_BYREF_NO_LEN);
}

/** \brief Return true if \a sptr is an SC_LOCAL and a pass by value parameter
   of
           \a func_sptr.
 */
int
sc_local_passbyvalue(int sptr, int func_sptr)
{
  int dscptr;
  int i;
  int param_sptr;
  char *param_name;

  if (SCG(sptr) != SC_LOCAL)
    return 0;

  /* find the _V_var on the function list */
  dscptr = DPDSCG(func_sptr);
  for (i = PARAMCTG(func_sptr); i > 0; dscptr++, i--) {
    param_sptr = aux.dpdsc_base[dscptr];
    param_name = SYMNAME(param_sptr);
    if ((strncmp(param_name, "_V_", 3) == 0) &&
        (strcmp(param_name + 3, SYMNAME(sptr)) == 0))
      return 1;
  }
  return 0;
}

/* param_sptr is a character string.  return  PASS_BYVAL,
   PASS_BYREF, PASS_BYREF_NO_LEN
 */
static int
set_char_ref_val(int func, int param)
{
  if (func == 0)
    return (PASS_BYREF);
  if (PASSBYVALG(param))
    return PASS_BYVAL;
  if (STDCALLG(func) || CFUNCG(func)) {
    if (PASSBYREFG(param))
      return PASS_BYREF_NO_LEN;

    if (PASSBYREFG(func))
      return PASS_BYREF;

    /* plain func= c/stdcall is pass by value */
    return PASS_BYVAL;
  }

  return PASS_BYREF;
}

/* find_byval_ref: check STCALLG , CFUNCG, PASSBYREFG, PASSBYVALG  and
   decide if this parameter is pass by value , pass by reference,
   or a character parameter pass by ref without length
 */
static int
find_byval_ref(int func_sptr, int param_sptr, int any_type)
{
  int iface;
  /* special care must be taken to mark string types
     pass by reference when we do not pass a length
   */
  /* CDEC$ VALUE or REFERENCE set explicitly for this parameter */

  proc_arginfo(func_sptr, NULL, NULL, &iface);
  if (param_sptr <= 0) {
    if (iface == 0)
      return (PASS_BYDEFAULT);
    if (PASSBYVALG(iface)) {
      return (PASS_BYVAL);
    }
    if (PASSBYREFG(iface)) {
      return (PASS_BYREF);
    }
/* sub defaults implied by STDARG or CFUNC */
#ifdef CREFP
    if (!CREFG(iface) && (STDCALLG(iface) || CFUNCG(iface))) {
      return (PASS_BYVAL);
    }
#else
    if (STDCALLG(iface) || CFUNCG(iface)) {
      return (PASS_BYVAL);
    }
#endif
    return PASS_BYDEFAULT;
  }

  if ((DTY(DTYPEG(param_sptr)) == TY_CHAR) ||
      (DTY(DTYPEG(param_sptr)) == TY_NCHAR)) {
    return (set_char_ref_val(iface, param_sptr));
  }

  if (is_iso_cptr(DTYPEG(param_sptr)) && PASSBYVALG(param_sptr)) {
    return (PASS_BYVAL);
  }

  if (!any_type && ((DTY(DTYPEG(param_sptr)) == TY_ARRAY) ||
                    (DTY(DTYPEG(param_sptr)) == TY_UNION))) {
    return (PASS_BYREF);
  }

  if (PASSBYVALG(param_sptr)) {
    return (PASS_BYVAL);
  }
  if (PASSBYREFG(param_sptr)) {
    return (PASS_BYREF);
  }

  /* subroutine default setting of parameters :
     sub defaults were directly set CDEC$ ATTRIBUTE VALUE or REFERENCE
   */
  if (iface == 0)
    return (PASS_BYDEFAULT);
  if (PASSBYVALG(iface)) {
    return (PASS_BYVAL);
  }
  if (PASSBYREFG(iface)) {
    return (PASS_BYREF);
  }
  /* sub defaults implied by STDARG or CFUNC */
  if (STDCALLG(iface) || CFUNCG(iface)) {
    return (PASS_BYVAL);
  }

  return (PASS_BYDEFAULT);
}

static void
init_byval()
{
  byval_func_ptr = 0;
  byval_dscptr = 0;
  byval_paramct = 0;
}

/* return the next dummy parameter to check for
   by value
 */
static int
inc_dummy_param(int func_sptr)
{
  int param_sptr;

  if (byval_func_ptr == 0) {
    byval_func_ptr = func_sptr;
    byval_dscptr = DPDSCG(func_sptr);
    byval_paramct = PARAMCTG(func_sptr);
  }

  if (byval_paramct == 0)
    return 0;
  param_sptr = *(aux.dpdsc_base + byval_dscptr);
  byval_dscptr++;
  return (param_sptr);
}

/** \brief Return true if param is pass by value.
 */
int
get_byval(int func_sptr, int param_sptr)
{
  return find_byval_ref(func_sptr, param_sptr, 0) == PASS_BYVAL;
}

/* rewrite references to types c_ptr, c_loc_ptr as
   c-_ptr->member
 */
static int
rewrite_cptr_references(int ast)
{
  int mast;
  int new_ast = 0;
  int psptr;
  int iso_dtype;

  switch (A_TYPEG(ast)) {
  case A_ID:
    mast = ast;
    break;
  case A_MEM:
    mast = A_MEMG(ast);
    break;
  case A_SUBSCR:
    mast = A_LOPG(ast);
    break;
  default:
    /* no need to process further  all cases of possible
       nested C_PTR must be in cases above  */
    return 0;
  }

  /* check for type C_PTR, C_FUNC_PTR, and process */
  iso_dtype = is_iso_cptr(A_DTYPEG(mast));
  if (iso_dtype) {
    psptr = DTY(iso_dtype + 1);
    new_ast = mk_member(ast, mk_id(psptr), DTYPEG(psptr));
  }
  return new_ast;
}

/*---------------------------------------------------------------------*/
/*
 * This stack entry represents a subprogram argument to be passed by value.
 *
 */
/* from %VAL() and %REF() processing */
static int
byvalue_ref_arg(SST *e1, int *dtype, int op, int func_sptr)
{
  int dum;
  int saved_dtype;
  int new_ast = 0;

  if (op == OP_VAL || op == OP_BYVAL) {
    int argdt;
    if (SST_ISNONDECC(e1))
      cngtyp(e1, DT_INT);

    saved_dtype = A_DTYPEG(SST_ASTG(e1));

    if ((A_TYPEG(SST_ASTG(e1)) == A_FUNC) && (is_iso_cptr(saved_dtype)) && !CFUNCG(func_sptr)) {
      /* functions returning c_ptr structs become funcs
         returning ints, so that we simply copy the
         (integer)pointer
       */
      A_DTYPEP(SST_ASTG(e1), DT_PTR);
    } else {
      new_ast = rewrite_cptr_references(SST_ASTG(e1));
      if (new_ast) {
        SST_ASTP(e1, new_ast);
        SST_IDP(e1, S_EXPR);
        SST_DTYPEP(e1, A_DTYPEG(new_ast));
      }
    }

    /* checking the AST dtype, resetting the semantic stack dtype */
    if (A_DTYPEG(SST_ASTG(e1)) != saved_dtype) {
      SST_DTYPEP(e1, A_DTYPEG(SST_ASTG(e1)));
    }

    mkexpr(e1);
    SST_IDP(e1, S_VAL);
    argdt = SST_DTYPEG(e1);
    *dtype = argdt;
    if (ELEMENTALG(func_sptr))
      argdt = DDTG(argdt);

    if (!is_iso_cptr(argdt) && !DT_ISBASIC(argdt) && DTY(argdt) != TY_STRUCT &&
        DTY(argdt) != TY_DERIVED) {
      /* also allow passing chars with no loc */
      cngtyp(e1, DT_INT);
      errsev(52);
    }
    SST_ASTP(e1, mk_unop(op, SST_ASTG(e1), *dtype));
    return mkarg(e1, dtype);
  }
#if DEBUG
  assert(op == OP_REF, "byvalue_ref_arg bad op", op, 3);
#endif
  /* OP_REF(character) , no length passed */
  mkarg(e1, &dum);
  SST_IDP(e1, S_REF);

  SST_ASTP(e1, mk_unop(op, SST_ASTG(e1), DT_INT));
  return 1;
}

/** \brief Return TRUE if sptr is a derived type with an allocatable member */
LOGICAL
allocatable_member(int sptr)
{
  DTYPE dtype = DTYPEG(sptr);
  if (DTYG(dtype) == TY_DERIVED) {
    int sptrmem;
    for (sptrmem = DTY(DDTG(dtype) + 1); sptrmem > NOSYM;
         sptrmem = SYMLKG(sptrmem)) {
      if (ALLOCATTRG(sptrmem)) {
        return TRUE;
      }
      if (USELENG(sptrmem) && ALLOCG(sptrmem) && TPALLOCG(sptrmem)) {
        return TRUE; /* uses length type parameter */
      }
      if (is_tbp_or_final(sptrmem)) {
        continue; /* skip tbp */
      }
      if (dtype != DTYPEG(sptrmem) && !POINTERG(sptrmem) &&
          allocatable_member(sptrmem)) {
        return TRUE;
      }
    }
  }
  return FALSE;
}

/*---------------------------------------------------------------------*/
LOGICAL
in_kernel_region()
{
  int df;
  for (df = 1; df <= sem.doif_depth; df++) {
    switch (DI_ID(df)) {
    default:
      break;
    case DI_CUFKERNEL:
    case DI_ACCDO:
    case DI_ACCLOOP:
    case DI_ACCREGDO:
    case DI_ACCREGLOOP:
    case DI_ACCKERNELSDO:
    case DI_ACCKERNELSLOOP:
    case DI_ACCPARALLELDO:
    case DI_ACCPARALLELLOOP:
    case DI_ACCSERIALLOOP:
      return TRUE;
    }
  }
  return FALSE;
} /* in_kernel_region */
/*---------------------------------------------------------------------*/

static int
get_sym_from_sst_if_available(SST *sst_actual)
{
  int sptr = 0;
  int unused;

  if (SST_IDG(sst_actual) == S_LVALUE)
    sptr = SST_LSYMG(sst_actual);
  else if (SST_IDG(sst_actual) == S_DERIVED || SST_IDG(sst_actual) == S_IDENT)
    sptr = SST_SYMG(sst_actual);
  else if (SST_IDG(sst_actual) == S_SCONST) {
    (void)mkarg(sst_actual, &unused);
    sptr = SST_SYMG(sst_actual);
  }
  return sptr;
}

static LOGICAL
is_ptr_arg(SST *sst_actual)
{
  SPTR sptr = get_sym_from_sst_if_available(sst_actual);

  if (sptr <= NOSYM) {
    int ast = SST_ASTG(sst_actual);
    if (A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_NULL) {
      return TRUE;
    }
    if (A_TYPEG(ast) == A_ID) {
      sptr = A_SPTRG(ast);
      if (sptr > NOSYM && SCG(sptr) == SC_BASED && !ALLOCATTRG(sptr) &&
          MIDNUMG(sptr) > NOSYM && PTRVG(MIDNUMG(sptr)))
        return TRUE;
    }
    if (SST_IDG(sst_actual) == S_EXPR && A_TYPEG(ast) == A_FUNC) {
      sptr = memsym_of_ast(A_LOPG(ast));
      sptr = FVALG(sptr);
    }
  }

  return sptr > NOSYM && POINTERG(sptr);
}

/* Non-pointer passed to a pointer dummy: geneerate a pointer temp, associate
 * the temp with the actual arg, and pass the temp.
 */
static int
gen_and_assoc_tmp_ptr(SST *sst_actual, int std)
{
  int sptrtmp;
  int ast_actual;
  int asttmp;
  int ast;
  int dtype;
  int dtype1;

  ast_actual = SST_ASTG(sst_actual);

  if (SST_IDG(sst_actual) == S_EXPR) {
    dtype1 = A_DTYPEG(ast_actual);
    ast = sem_tempify(sst_actual);
    (void)add_stmt(ast);
    ast = A_DESTG(ast);
  } else if (ast_actual) {
    dtype1 = A_DTYPEG(ast_actual);
    ast = ast_actual;
  } else {
    int sptractual = get_sym_from_sst_if_available(sst_actual);
    assert(sptractual, "gen_and_assoc_tmp_ptr: no symbol or AST for actual arg",
           0, 4);
    dtype1 = DTYPEG(sptractual);
    ast = mk_id(sptractual);
  }

  dtype = dtype1;
  if (DTY(dtype) == TY_ARRAY) {
    dtype = dup_array_dtype(dtype);
    DTY(dtype + 1) = DTY(dtype1 + 1);
  }

  sptrtmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, SC_LOCAL);
  asttmp = mk_id(sptrtmp);
  POINTERP(sptrtmp, 1);
  CCSYMP(sptrtmp, 1);
  ARGP(sptrtmp, 1);
  get_static_descriptor(sptrtmp);
  get_all_descriptors(sptrtmp);
  ADDRTKNP(sym_of_ast(ast), 1);
  (void)add_stmt(add_ptr_assign(asttmp, ast, std));
  return asttmp;
}

static LOGICAL
need_tmp_retval(int func_sptr, int param_dummy)
{
  int fval;
  int func_dtype;

  fval = func_sptr;
  if (FVALG(func_sptr))
    fval = FVALG(func_sptr);

  func_dtype = DTYPEG(func_sptr);

  if (POINTERG(fval)) {
    return TRUE;
  }
  if (POINTERG(fval)) {
    return TRUE;
  }
  if (ALLOCATTRG(fval) || allocatable_member(fval)) {
    return TRUE;
  }
  if (DTY(func_dtype) == TY_ARRAY) {
    return TRUE;
  }
  if (ADJLENG(fval)) {
    if (!ELEMENTALG(func_sptr)) {
      return TRUE;
    } else if (!ARG_STK(0) || !A_SHAPEG(SST_ASTG(ARG_STK(0)))) {
      return TRUE;
    }
  }

  return FALSE;
}

/** \brief If applicable, generate finalization code for function result.
 *
 * \param fval is the result symbol.
 * \param func_sptr is the function symbol table pointer
 *
 * \returns the result symbol; either fval or a new result symbol.
 */
static int
gen_finalized_result(int fval, int func_sptr)
{
  if (!ALLOCATTRG(fval) && !POINTERG(fval) && has_finalized_component(fval)) {
    /* Need to finalize the function result after it's assigned to LHS.
     * If the result is allocatable, then finalization is handled during
     * automatic deallocation (i.e., the runtime call to dealloc_poly03,
     * dealloc_poly_mbr03). If the result is pointer, then we do not finalize
     * the object (the language spec indicates that it is processor dependent
     * whether such objects are finalized).
     */
    int std = add_stmt(mk_stmt(A_CONTINUE, 0));

    if (STYPEG(fval) == ST_UNKNOWN || STYPEG(fval) == ST_IDENT) {
      fval = getsymbol(SYMNAME(fval));
      if (STYPEG(fval) == ST_PROC) {
        /* function result variable name same as its function */
        fval = insert_sym(fval);
      } else {
        /* function result variable name overloads another object */
        fval = get_next_sym(SYMNAME(fval), NULL);
      }
      fval = declsym(fval, ST_VAR, TRUE);
      SCP(fval, SC_LOCAL);
      DTYPEP(fval, DTYPEG(func_sptr));
      DCLDP(fval, 1);
      init_derived_type(fval, 0, std);
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
    }
    gen_finalization_for_sym(fval, std, 0);
  }
  return fval;
}

/** \brief Write ILMs to call a function.
    \param stktop function to call
    \param list   arguments to pass to function
    \param flag   set if called from a generic resolution routine
 */
int
func_call2(SST *stktop, ITEM *list, int flag)
{
  int func_sptr, sptr1, fval_sptr = 0;
  ITEM *itemp;
  int count, i, ii;
  int dum;
  int dtype;
  int ast;
  int argt;
  SST *sp;
  int param_dummy;
  int return_value, isarray, save_func_arrinfo;
  char *kwd_str; /* where make_kwd_str saves the string */
  int argt_count;
  int shaper;
  int new_ast;
  int psptr, msptr;
  int callee;
  int invobj;

  return_value = 0;
  save_func_arrinfo = 0;
  SST_CVLENP(stktop, 0);
  ast = astb.i0; /* initialize just in case error occurs */
  kwd_str = NULL;
  func_sptr = SST_SYMG(stktop);
  if (func_sptr < 0) {
    func_sptr = -func_sptr;
    SST_SYMP(stktop, func_sptr);
  }
  switch (A_TYPEG(SST_ASTG(stktop))) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    callee = memsym_of_ast(SST_ASTG(stktop));
    if (STYPEG(callee) == ST_PROC && CLASSG(callee) && IS_TBP(callee)) {
      /* special case for user defined generic type bound operators */
      i = 0;
      func_sptr = get_implementation(TBPLNKG(callee), callee, 0, &i);
      if (STYPEG(BINDG(i)) == ST_OPERATOR ||
          STYPEG(BINDG(i)) == ST_USERGENERIC) {
        i = get_specific_member(TBPLNKG(callee), callee);
        func_sptr = VTABLEG(i);
      }
      callee = i;
      SST_ASTP(stktop, replace_memsym_of_ast(SST_ASTG(stktop), i));
      dtype = TBPLNKG(BINDG(i));
      goto process_tbp;
    }
    break;
  default:
    callee = 0;
  }
  if (callee && CLASSG(callee) && CCSYMG(callee) &&
      STYPEG(callee) == ST_MEMBER) {
    func_sptr = pass_sym_of_ast(SST_ASTG(stktop));
    dtype = DTYPEG(func_sptr);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    if (STYPEG(BINDG(callee)) == ST_USERGENERIC) {
      int mem;
      func_sptr = generic_tbp_func(BINDG(callee), stktop, list);
      if (func_sptr) {
        if (get_implementation(dtype, func_sptr, 0, &mem) == 0) {
          char *name_cpy, *name;
          name_cpy = getitem(0, strlen(SYMNAME(func_sptr)) + 1);
          strcpy(name_cpy, SYMNAME(func_sptr));
          name = strchr(name_cpy, '$');
          if (name)
            *name = '\0';
          error(155, 3, gbl.lineno,
                "Could not resolve generic type bound "
                "procedure",
                name_cpy);
          sptr1 = 0;
        } else {
          SST_ASTP(stktop, replace_memsym_of_ast(SST_ASTG(stktop), mem));
          callee = mem;
        }
      }
    }
    func_sptr = get_implementation(dtype, BINDG(callee), !flag, NULL);
  process_tbp:
    invobj = get_tbp_argno(BINDG(callee), dtype);
    set_pass_objects(invobj - 1, pass_sym_of_ast(SST_ASTG(stktop)));
    callee = SST_ASTG(stktop);
  } else
    callee = 0;
  FUNCP(func_sptr, 1); /* mark sptr as a function */
  TYPDP(func_sptr, 1); /* put in 'external' statement */
  dtype = DTYPEG(func_sptr);
  shaper = 0;
  isarray = DTY(dtype) == TY_ARRAY;

  if (DPDSCG(func_sptr))
    kwd_str = make_kwd_str(func_sptr);

  /* store function st in ERRSYM for error messages; used to be set only
   * for CHAR
   */
  SST_ERRSYMP(stktop, func_sptr);

  if (list == NULL)
    list = ITEM_END;
  if (STYPEG(func_sptr) == ST_PROC && SLNKG(func_sptr) == 0 && 
      !sem.proc_initializer) { 
    SLNKP(func_sptr, aux.list[ST_PROC]);
    aux.list[ST_PROC] = func_sptr;
  }
  count_actuals(list);
  count = carg.nent;
  argt_count = carg.nargt;

  if (!FUNCLINEG(func_sptr) && POINTERG(func_sptr)) {
    error(465, 3, gbl.lineno, CNULL, CNULL);
  }
  init_byval();

  if (kwd_str) {
    int dscptr; /* ptr to dummy parameter descriptor list */
    int fval;

    if (check_arguments(func_sptr, count, list, kwd_str))
      goto exit_;
    for (i = 0; i < carg.nent; i++) {
      sp = ARG_STK(i);
      if (sp) {
        /* add to ARGT list, handling derived type arguments as
         * special case.
         */
        sptr1 = get_sym_from_sst_if_available(sp);
        {
          param_dummy = inc_dummy_param(func_sptr);

          if (is_iso_cloc(SST_ASTG(sp))) {
            if (find_byval_ref(func_sptr, param_dummy, 1) == PASS_BYVAL) {
              /* pass by val iso_c pointer to arg:
                 C_LOC(arg)   C_FUN_LOC(arg)
                 is plain old pass by reference
                 without type checking: get rid of the
                C_LOC:
               */
              new_ast = ARGT_ARG(A_ARGSG(SST_ASTG(sp)), 0);
              if (A_TYPEG(new_ast) == A_ID && (!TARGETG(A_SPTRG(new_ast))) &&
                  (!POINTERG(A_SPTRG(new_ast))))
                errwarn(468);

              SST_ASTP(sp, new_ast);
              SST_IDP(sp, S_EXPR);
            } else if (A_TYPEG(ARGT_ARG(A_ARGSG(SST_ASTG(sp)), 0)) != A_ID) {
              // Inlining has problems with an expression in this context.
              // Downstream code can always handle simple variables.
              (void)tempify(sp);
            }
            /* else
             * iso_c_loc by reference pointer to pointer */
          } else if (get_byval(func_sptr, param_dummy)) {
            /*  function arguments not processed by lowerilm */
            if (PASSBYVALG(param_dummy)) {
              if (OPTARGG(param_dummy)) {
                int assn = sem_tempify(sp);
                (void)add_stmt(assn);
                SST_ASTP(sp, A_DESTG(assn));
                byvalue_ref_arg(sp, &dum, OP_REF, func_sptr);
              } else if (!need_tmp_retval(func_sptr, param_dummy))
                byvalue_ref_arg(sp, &dum, OP_BYVAL, func_sptr);
              else
                byvalue_ref_arg(sp, &dum, OP_VAL, func_sptr);
            } else {
              byvalue_ref_arg(sp, &dum, OP_VAL, func_sptr);
            }
          } else if (pass_char_no_len(func_sptr, param_dummy)) {
            byvalue_ref_arg(sp, &dum, OP_REF, func_sptr);
          } else if (INTENTG(param_dummy) == INTENT_IN &&
                     POINTERG(param_dummy) && !is_ptr_arg(sp)) {
            /* F2008: pass non-pointer actual arg for an
             *        INTENT(IN), POINTER formal arg */
            ARG_AST(i) = SST_ASTG(sp) = gen_and_assoc_tmp_ptr(sp, sem.last_std);
          } else {
          }
        }
      }
    }

    count_formals(func_sptr);
    argt_count = carg.nargt;
    dscptr = DPDSCG(func_sptr);
    fval = func_sptr;
    if (FVALG(func_sptr))
      fval = FVALG(func_sptr);
    /* for ST_ENTRY, the data type info is set in the return value symbol */
    if (POINTERG(fval)) {
      /*
       * since the result of the function is a pointer, a pointer
       * temporary must be created.
       * Note that for an 'adjustable' return value, its size
       * may be dependent on the actual arguments.
       *
       * Would like to call set_descriptor_sc() at the beginning
       * of func2_call() and restore at the end; however, there
       * are still semsym things that might need to be done to user
       * variables.  So, only call set_descriptor_sc() when we know
       * we are creating temps.
       */
      set_descriptor_sc(sem.sc);
      if (isarray) {
        return_value = ref_entry(func_sptr);
      } else {
        return_value = get_next_sym(SYMNAME(func_sptr), "v");
        STYPEP(return_value, ST_VAR);
        SCP(return_value, SC_BASED);
        DTYPEP(return_value, dtype);
        DCLDP(return_value, 1);
        POINTERP(return_value, 1);
        if (DTYG(dtype) == TY_DERIVED && XBIT(58, 0x40000)) {
          F90POINTERP(return_value, 1);
        } else {
          get_static_descriptor(return_value);
          get_all_descriptors(return_value);
        }
      }
#ifdef CLASSG
      if (HCCSYMG(return_value) && !CLASSG(return_value))
        CLASSP(return_value, CLASSG(FVALG(func_sptr)));
#endif
      {
        /* Be warned: "return_value" is a symbol table index coming into
         * this block of code, but it's an AST index coming out!
         */
        return_value = gen_pointer_result(return_value, dscptr, carg.nent,
                                          FALSE, func_sptr);
        argt_count++;
        argt = mk_argt(argt_count); /* mk_argt stuffs away count */
        ARGT_ARG(argt, 0) = return_value;
        ii = 1;
        save_func_arrinfo = 1;
      }
      set_descriptor_sc(SC_LOCAL);
    } else if (ALLOCATTRG(fval)) {
      /*
       * result of the function is an allocatable, should be similiar
       * to a pointer
       */
      if (isarray) {
        fval_sptr = ref_entry(func_sptr);
      } else {
        fval_sptr = get_next_sym(SYMNAME(func_sptr), "v");
        STYPEP(fval_sptr, ST_VAR);
        SCP(fval_sptr, SC_BASED);
        DTYPEP(fval_sptr, dtype);
        DCLDP(fval_sptr, 1);
        set_descriptor_sc(sem.sc);
        get_static_descriptor(fval_sptr);
        get_all_descriptors(fval_sptr);
        set_descriptor_sc(SC_LOCAL);
      }

      return_value = gen_allocatable_result(
          fval_sptr, dscptr, carg.nent, (DTYG(dtype) == TY_DERIVED), func_sptr);
#ifdef RVALLOCP
      if (XBIT(54, 0x1) && !isarray && DTY(dtype) != TY_DERIVED) {
        int sym;
        sym = sym_of_ast(return_value);
        if (MIDNUMG(sym)) {
          RVALLOCP(MIDNUMG(sym), 1);
        }
      }
#endif

#ifdef CLASSG
      if (HCCSYMG(fval_sptr) && !CLASSG(fval_sptr)) {
        CLASSP(fval_sptr, CLASSG(FVALG(func_sptr)));
        CLASSP(sym_of_ast(return_value), CLASSG(FVALG(func_sptr)));
      }
#endif
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
      add_p_dealloc_item(memsym_of_ast(return_value));
    } else if (allocatable_member(fval)) {
      if (ELEMENTALG(func_sptr)) {
        int i;
        for (i = 0; i < argt_count; ++i) {
          shaper = A_SHAPEG(ARG_AST(i));
          if (shaper) {
            int dt = dtype_with_shape(dtype, shaper);
            fval_sptr = get_arr_temp(dt, FALSE, FALSE, FALSE);
            DTYPEP(fval_sptr, dt);
            STYPEP(fval_sptr, ST_ARRAY);
            break;
          }
        }
      }
      if (!shaper) {
        if (ADJARRG(fval)) {
          return_value = ref_entry(func_sptr);
          return_value = gen_array_result(return_value, dscptr, carg.nent,
                                          FALSE, func_sptr);
          fval_sptr = A_SPTRG(return_value);
        } else {
          fval_sptr = get_next_sym(SYMNAME(func_sptr), "d");
          if (isarray) {
            STYPEP(fval_sptr, ST_ARRAY);
          } else {
            STYPEP(fval_sptr, ST_VAR);
          }
          DTYPEP(fval_sptr, dtype);
        }
      }

      SCP(fval_sptr, sem.sc);
      if (ASSUMSHPG(fval) || ASUMSZG(fval)) {
        set_descriptor_sc(sem.sc);
        get_static_descriptor(fval_sptr);
        get_all_descriptors(fval_sptr);
        set_descriptor_sc(SC_LOCAL);
      }
      init_derived_type(fval_sptr, 0, STD_PREV(0));
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      return_value = mk_id(fval_sptr);
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
      add_p_dealloc_item(fval_sptr);
    } else if (isarray) {
      /*
       * since the result of the function is an array, a temporary
       * must be allocated at run-time even if its bounds are contant.
       * Note that for an 'adjustable' return value, its size
       * may be dependent on the actual arguments.
       */
      return_value = ref_entry(func_sptr);
      if (!ADJLENG(fval))
        return_value =
            gen_array_result(return_value, dscptr, carg.nent, FALSE, func_sptr);
      else
        return_value = gen_char_result(return_value, dscptr, carg.nent);
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
      /*
       * have an array-valued function; save up information
       * which would allow substituting the result temp with
       * the LHS of an assignment.
       */
      save_func_arrinfo = 1;
    } else if (ADJLENG(fval)) {
      if (ELEMENTALG(func_sptr)) {
        sp = ARG_STK(0);
        if (sp && (shaper = A_SHAPEG(SST_ASTG(sp)))) {
          argt_count++;
          argt = mk_argt(argt_count);
          ARGT_ARG(argt, 0) = gen_char_result(fval, dscptr, carg.nent);
          ii = 1;
          return_value = 0;
        } else {
          return_value = gen_char_result(fval, dscptr, carg.nent);
        }
      } else {
        return_value = gen_char_result(fval, dscptr, carg.nent);
      }
      if (return_value) {
        argt_count++;
        argt = mk_argt(argt_count); /* mk_argt stuffs away count */
        ARGT_ARG(argt, 0) = return_value;
        ii = 1;
      }
    } else {
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ii = 0;
    }

    fval = gen_finalized_result(fval, func_sptr);

    /* return value handled, copy in the function args */
    for (i = 0; i < carg.nent; i++, ii++) {
      if (ARG_STK(i)) {
        ARGT_ARG(argt, ii) = SST_ASTG(ARG_STK(i));
      } else {
        /* OPTIONAL arg not present */
        ARGT_ARG(argt, ii) = astb.ptr0;
      }
    }

    if (return_value) {
      /* return_value is symbol if result is of derived type;
       * otherwise, it's an ast.
       */
      dtype = DTYPEG(A_SPTRG(return_value));
      if (callee) {
        int mem = memsym_of_ast(callee);
        if (STYPEG(mem) == ST_MEMBER && !strstr(SYMNAME(func_sptr), "$tbp")) {
          VTABLEP(mem, func_sptr);
        }
        /*dtype = DTYPEG(mem);*/
      }
      ast = mk_func_node(A_CALL, (callee) ? callee : mk_id(func_sptr),
                         argt_count, argt);
      sem.arrfn.call_std = add_stmt(ast);
      sem.arrfn.sptr = func_sptr;
      if (save_func_arrinfo) {
        sem.arrfn.return_value = return_value;
        if (ALLOCG(A_SPTRG(return_value)))
          sem.arrfn.alloc_std = sem.alloc_std;
      }
      ast = return_value;
    } else {
      if (callee) {
        int mem = memsym_of_ast(callee);
        if (STYPEG(mem) == ST_MEMBER && !strstr(SYMNAME(func_sptr), "$tbp")) {
          VTABLEP(mem, func_sptr);
        }
        /*dtype = DTYPEG(mem);*/
      }
      ast = mk_func_node(A_FUNC, (callee) ? callee : mk_id(func_sptr),
                         argt_count, argt);
    }
    if (ELEMENTALG(func_sptr)) {
      int argc;
      for (argc = 0; argc < argt_count; ++argc) {
        /* Use first shaped argument */
        shaper = A_SHAPEG(ARGT_ARG(argt, argc));
        if (shaper)
          break;
      }
      if (shaper == 0) {
        shaper = mkshape(dtype);
      } else {
        dtype = dtype_with_shape(dtype, shaper);
        A_SHAPEP(ast, shaper);
      }
    } else {
      shaper = mkshape(dtype);
    }
    A_DTYPEP(ast, dtype);
    if (DFLTG(func_sptr)) {
      int newdt = dtype;
      switch (DTY(dtype)) {
      case TY_INT:
        newdt = stb.user.dt_int;
        break;
      case TY_LOG:
        newdt = stb.user.dt_log;
        break;
      case TY_REAL:
        newdt = stb.user.dt_real;
        break;
      case TY_CMPLX:
        newdt = stb.user.dt_cmplx;
        break;
      }
      if (newdt != dtype) {
        ast = mk_convert(ast, newdt);
        dtype = newdt;
      }
    }
    goto exit_;
  }
  ii = 0;
  /* An assumed-length character function is handled as if it is a subroutine.
   * Implicit temporary variables are created for the return value and its
   * length, and passed to the function as output arguments.
   */
  if ((DTY(dtype) == TY_CHAR) && ADJLENG(func_sptr)) {
    int functmp;
    int functmp_ast, alloc_ast, len_ast;
    int alloc_stmt, len_stmt;

    functmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_DEFERCHAR, SC_BASED);
    functmp_ast = mk_id(functmp);
    get_static_descriptor(functmp);
    get_all_descriptors(functmp);
    ALLOCDESCP(functmp, 1);
    ALLOCP(functmp, 1);
    len_ast = DTY(dtype + 1);
    len_stmt = mk_assn_stmt(get_len_of_deferchar_ast(functmp_ast), len_ast, A_DTYPEG(len_ast));
    /* Allocate the temporary */
    alloc_ast = mk_stmt(A_ALLOC, 0);
    A_TKNP(alloc_ast, TK_ALLOCATE);
    A_SRCP(alloc_ast, functmp_ast);
    alloc_stmt = add_stmt(alloc_ast);
    sem.alloc_std = alloc_stmt;
    add_stmt_before(len_stmt, alloc_stmt);
    add_auto_dealloc(functmp);
    /* add the temporary and length of character to ARGT list */
    return_value = functmp_ast;
    argt_count = argt_count + 2;
    argt = mk_argt(argt_count);
    ARGT_ARG(argt, 0) = return_value;
    ARGT_ARG(argt, 1) = mk_convert(len_ast, stb.user.dt_int);
    ii = ii + 2;
  } else {
    argt = mk_argt(argt_count); /* mk_argt stuffs away count */
  }

  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_KEYWORD) {
      /* form is <ident> = <expression> */
      error(79, 3, gbl.lineno, scn.id.name + SST_CVALG(itemp->t.stkp), CNULL);
      itemp->t.sptr = 1;
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    if (SST_IDG(sp) == S_TRIPLE) {
      /* form is e1:e2:e3 */
      error(76, 3, gbl.lineno, SYMNAME(func_sptr), CNULL);
      itemp->t.sptr = 1;
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    if (SST_IDG(sp) == S_LABEL) {
      error(155, 3, gbl.lineno, "Illegal use of alternate return specifier",
            CNULL);
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    /* check arguments and add to ARGT list, handling derived type
       arguments as special case */
    sptr1 = 0;
    if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
      sptr1 = SST_SYMG(sp);
    else if (SST_IDG(sp) == S_LVALUE)
      sptr1 = SST_LSYMG(sp);
    else if (SST_IDG(sp) == S_SCONST) {
      (void)mkarg(sp, &dum);
      sptr1 = SST_SYMG(sp);
    }
    {
      /* form is <ident> or <expression> */
      param_dummy = inc_dummy_param(func_sptr);
      /*  function arguments not processed bylowerilm */

      if ((A_TYPEG(SST_ASTG(sp)) == A_ID) &&
          is_iso_cptr(DTYPEG(A_SPTRG(SST_ASTG(sp))))) {
        if (find_byval_ref(func_sptr, param_dummy, 1) == PASS_BYVAL) {
          /* iso cptr passed by value needs to transform into
             pass by value cptr->member : (pass the pointer
             sitting in cptr->member by value) */

          psptr = A_SPTRG(SST_ASTG(sp));
          msptr = DTY(DTYPEG(psptr) + 1);
          new_ast = mk_member(SST_ASTG(sp), mk_id(msptr), DTYPEG(msptr));
          SST_ASTP(sp, new_ast);
          SST_IDP(sp, S_EXPR);
          SST_DTYPEP(sp, DTYPEG(msptr));

          byvalue_ref_arg(sp, &dum, OP_VAL, func_sptr);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        } else {
          /* plain pass by ref */
          itemp->t.sptr = chkarg(sp, &dum);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        }
      } else if (is_iso_cloc(SST_ASTG(sp))) {

        if (find_byval_ref(func_sptr, param_dummy, 1) == PASS_BYVAL) {
          /* pass by val iso_c pointer to arg:
             C_LOC(arg)   C_FUN_LOC(arg)
             is plain old pass by reference
             without type checking: get rid of the c_LOC
           */
          new_ast = ARGT_ARG(A_ARGSG(SST_ASTG(sp)), 0);
          if (A_TYPEG(new_ast) == A_ID && (!TARGETG(A_SPTRG(new_ast))) &&
              (!POINTERG(A_SPTRG(new_ast))))
            errwarn(468);

          SST_ASTP(sp, new_ast);
          SST_IDP(sp, S_EXPR);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);

        } else {
          /* iso_c_loc by reference: pointer to pointer */
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        }
      } else if (get_byval(func_sptr, param_dummy)) {
        if (PASSBYVALG(param_dummy)) {
          itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_BYVAL, func_sptr);
        } else {
          itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_VAL, func_sptr);
        }
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      } else if (pass_char_no_len(func_sptr, param_dummy)) {
        itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_REF, func_sptr);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      } else {
        itemp->t.sptr = chkarg(sp, &dum);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      }
      ii++;
    }
  }
  if (callee) {
    int mem = memsym_of_ast(callee);
    if (STYPEG(mem) == ST_MEMBER && !strstr(SYMNAME(func_sptr), "$tbp")) {
      VTABLEP(mem, func_sptr);
    }
    dtype = DTYPEG(mem);
  }
  /* Use A_CALL instead of A_FUNC for an assumed-length character function. */
  if ((DTY(dtype) == TY_CHAR) && ADJLENG(func_sptr)) {
    ast = mk_func_node(A_CALL, (callee) ? callee : mk_id(func_sptr),
                       argt_count, argt);
    sem.arrfn.call_std = add_stmt(ast);
    sem.arrfn.sptr = func_sptr;
    ast = return_value;
  } else {
    ast = mk_func_node(A_FUNC, (callee) ? callee : mk_id(func_sptr),
                       argt_count, argt);
  }
  A_DTYPEP(ast, dtype);
  A_SHAPEP(ast, mkshape(dtype));
  if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR)
    error(89, 3, gbl.lineno, SYMNAME(func_sptr), CNULL);

exit_:
  SST_IDP(stktop, S_EXPR);
  SST_ASTP(stktop, ast);
  if (shaper)
    SST_SHAPEP(stktop, shaper);
  else
    SST_SHAPEP(stktop, A_SHAPEG(ast));
  SST_DTYPEP(stktop, dtype);

  if (kwd_str)
    FREE(kwd_str);

  return 1;
}

/** \brief Resolve forward references in function func_call().
 *
 * Used by func_call() to resolve any forward refs we may
 * encounter since resolve_fwd_refs() in semutil.c gets called after we
 * finish processing this function. We also want to check to see if this
 * reference resolves to a generic procedure.
 */
static void
resolve_fwd_ref(int ref)
{
  int mod, decl, hashlk;
  int found;

  if (STYPEG(ref) == ST_PROC && FWDREFG(ref)) {
    found = 0;
    /* Find the module that contains the reference. */
    for (mod = SCOPEG(ref); mod; mod = SCOPEG(mod))
      if (STYPEG(mod) == ST_MODULE)
        break;
    if (mod == 0)
      return; /* Not in a module. */

    /* Look for the matching declaration. */
    for (decl = first_hash(ref); decl; decl = HASHLKG(decl)) {
      if (NMPTRG(decl) != NMPTRG(ref))
        continue;
      if (STYPEG(decl) == ST_PROC && ENCLFUNCG(decl) == mod) {
        hashlk = HASHLKG(ref);
        *(stb.stg_base + ref) = *(stb.stg_base + decl);
        HASHLKP(ref, hashlk);
        found = 1;
        break;
      }
    }
    if (found)
      return;
    /* Look for the matching generic declaration. */
    for (decl = first_hash(ref); decl; decl = HASHLKG(decl)) {
      if (NMPTRG(decl) != NMPTRG(ref))
        continue;
      if (STYPEG(decl) == ST_USERGENERIC && ENCLFUNCG(decl) == mod) {
        hashlk = HASHLKG(ref);
        *(stb.stg_base + ref) = *(stb.stg_base + decl);
        HASHLKP(ref, hashlk);
        found = 1;
        break;
      }
    }
  }
}

int
func_call(SST *stktop, ITEM *list)
{
  int func_sptr;
  /* Note: If we have a generic tbp (or operator), pass a 0
   * flag only if the generic is private. We do this to turn off
   * the private error check on the resolved tbp.
   */
  int ast, sptr, sptr1 = NOSYM;
  ast = SST_ASTG(stktop);
  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    sptr1 = memsym_of_ast(ast);
    sptr = BINDG(sptr1);
    break;
  }

  if (A_TYPEG(ast) != A_MEM && sptr1 > NOSYM && IS_TBP(sptr1)) {
    /* Check for generic function that might be sharing the same
     * name as a type bound procedure
     */
    generic_func(SST_SYMG(stktop), stktop, list);
    sptr = SST_SYMG(stktop);
  }

  if ((STYPEG(sptr) == ST_USERGENERIC || STYPEG(sptr) == ST_OPERATOR) &&
      IS_TBP(sptr)) {
    return func_call2(stktop, list, sptr1 <= NOSYM || !PRIVATEG(sptr1));
  }
  /* Check to see if func_sptr is a forward reference that
   * resolves to an ST_PROC or a ST_USERGENERIC
   */
  func_sptr = SST_SYMG(stktop);
  if (func_sptr < 0) {
    func_sptr = -func_sptr;
  }
  resolve_fwd_ref(func_sptr);
  if (STYPEG(func_sptr) == ST_USERGENERIC)
    return generic_func(func_sptr, stktop, list);

  return func_call2(stktop, list, 0);
}

int
ptrfunc_call(SST *stktop, ITEM *list)
{
  int func_sptr, sptr1, fval_sptr;
  int callee;
  ITEM *itemp;
  int count, i, ii;
  int dum;
  int dtproc, iface, paramct, dpdsc, fval;
  int dtype;
  int ast;
  int argt;
  SST *sp;
  int param_dummy;
  int return_value, isarray, save_func_arrinfo;
  char *kwd_str; /* where make_kwd_str saves the string */
  int argt_count;
  int shaper;
  int new_ast;
  int psptr, msptr;
  int pass_pos;

  fix_proc_pointer_call(stktop, &list);
  return_value = 0;
  save_func_arrinfo = 0;
  SST_CVLENP(stktop, 0);
  ast = astb.i0; /* initialize just in case error occurs */
  kwd_str = NULL;
  dtype = A_DTYPEG(astb.i0);
  shaper = 0;
  pass_pos = -1;
  if (SST_IDG(stktop) != S_LVALUE) {
    func_sptr = SST_SYMG(stktop);
    callee = mk_id(func_sptr);
  } else {
    func_sptr = SST_LSYMG(stktop);
    if (!is_procedure_ptr(func_sptr)) {
      /* error must have occurred */
      goto exit_;
    }
    callee = SST_ASTG(stktop);
  }
  dtype = DTYPEG(func_sptr);
#if DEBUG
  assert(DTY(dtype) == TY_PTR, "ptrfunc_call, expected TY_PTR dtype", func_sptr,
         4);
#endif
  dtproc = DTY(dtype + 1);
#if DEBUG
  assert(DTY(dtproc) == TY_PROC, "ptrfunc_call, expected TY_PROC dtype",
         func_sptr, 4);
#endif
  dtype = DTY(dtproc + 1);
  iface = DTY(dtproc + 2);
  paramct = DTY(dtproc + 3);
  dpdsc = DTY(dtproc + 4);
  fval = DTY(dtproc + 5);
  if (iface) {
    FUNCP(iface, 1); /* mark sptr as a function */
  }
  if (iface != func_sptr && !paramct) {
    proc_arginfo(iface, &paramct, &dpdsc, NULL);
    DTY(dtproc + 3) = paramct;
    DTY(dtproc + 4) = dpdsc;
  }
  add_typroc(dtproc);
  shaper = 0;
  if (iface)
    isarray = is_array_dtype(DTYPEG(iface));
  else
    isarray = is_array_dtype(dtype);
  if (dpdsc)
    kwd_str = make_keyword_str(paramct, dpdsc);
  /* store function st in ERRSYM for error messages; used to be set only
   * for CHAR
   */
  SST_ERRSYMP(stktop, func_sptr);

  if (list == NULL)
    list = ITEM_END;
  count_actuals(list);
  count = carg.nent;
  argt_count = carg.nargt;

  init_byval();

  if (kwd_str) {
    if (chk_arguments(func_sptr, count, list, kwd_str, paramct, dpdsc, callee,
                      &pass_pos))
      goto exit_;
    count_formal_args(paramct, dpdsc);
    argt_count = carg.nargt;
    if (!fval)
      fval = iface;
    /* for ST_ENTRY, the data type info is set in the return value symbol */
    if (POINTERG(fval)) {
      /*
       * since the result of the function is a pointer, a pointer
       * temporary must be created.
       * Note that for an 'adjustable' return value, its size
       * may be dependent on the actual arguments.
       */
      set_descriptor_sc(sem.sc);
      if (isarray) {
        return_value = fval;
      } else {
        return_value = get_next_sym(SYMNAME(iface), "v");
        STYPEP(return_value, ST_VAR);
        SCP(return_value, SC_BASED);
        DTYPEP(return_value, dtype);
        DCLDP(return_value, 1);
        POINTERP(return_value, 1);
        if (DTYG(dtype) == TY_DERIVED && XBIT(58, 0x40000)) {
          F90POINTERP(return_value, 1);
        } else {
          get_static_descriptor(return_value);
          get_all_descriptors(return_value);
        }
      }
#ifdef CLASSG
      if (HCCSYMG(return_value) && !CLASSG(return_value))
        CLASSP(return_value, CLASSG(FVALG(func_sptr)));
#endif
      {
        return_value =
            gen_pointer_result(return_value, dpdsc, carg.nent, FALSE, iface);
        argt_count++;
        argt = mk_argt(argt_count); /* mk_argt stuffs away count */
        ARGT_ARG(argt, 0) = return_value;
        ii = 1;
        save_func_arrinfo = 1;
      }
      set_descriptor_sc(SC_LOCAL);
    } else if (ALLOCATTRG(fval)) {
      /*
       * result of the function is an allocatable, should be similiar
       * to a pointer
       */
      if (isarray) {
        fval_sptr = fval;
      } else {
        fval_sptr = get_next_sym(SYMNAME(iface), "v");
        STYPEP(fval_sptr, ST_VAR);
        SCP(fval_sptr, SC_BASED);
        DTYPEP(fval_sptr, dtype);
        DCLDP(fval_sptr, 1);
        set_descriptor_sc(sem.sc);
        get_static_descriptor(fval_sptr);
        get_all_descriptors(fval_sptr);
        set_descriptor_sc(SC_LOCAL);
      }
      return_value = gen_allocatable_result(fval_sptr, dpdsc, carg.nent,
                                            (DTYG(dtype) == TY_DERIVED), iface);
#ifdef CLASSG
      if (HCCSYMG(fval_sptr) && !CLASSG(fval_sptr))
        CLASSP(fval_sptr, CLASSG(FVALG(func_sptr)));
#endif
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;

      add_p_dealloc_item(memsym_of_ast(return_value));
    } else if (allocatable_member(fval)) {
      if (ELEMENTALG(iface)) {
        int i;
        for (i = 0; i < argt_count; ++i) {
          shaper = A_SHAPEG(ARG_AST(i));
          if (shaper) {
            int dt = dtype_with_shape(dtype, shaper);
            fval_sptr = get_arr_temp(dt, FALSE, FALSE, FALSE);
            DTYPEP(fval_sptr, dt);
            STYPEP(fval_sptr, ST_ARRAY);
            break;
          }
        }
      }
      if (!shaper) {
        if (ADJARRG(fval)) {
          return_value = ref_entry(iface);
          return_value =
              gen_array_result(return_value, dpdsc, carg.nent, FALSE, iface);
          fval_sptr = A_SPTRG(return_value);
        } else {
          fval_sptr = get_next_sym(SYMNAME(func_sptr), "d");
          if (isarray) {
            STYPEP(fval_sptr, ST_ARRAY);
          } else {
            STYPEP(fval_sptr, ST_VAR);
          }
          DTYPEP(fval_sptr, dtype);
        }
      }

      SCP(fval_sptr, sem.sc);
      if (ASSUMSHPG(fval) || ASUMSZG(fval)) {
        set_descriptor_sc(sem.sc);
        get_static_descriptor(fval_sptr);
        get_all_descriptors(fval_sptr);
        set_descriptor_sc(SC_LOCAL);
      }
      init_derived_type(fval_sptr, 0, STD_PREV(0));
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      return_value = mk_id(fval_sptr);
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
      add_p_dealloc_item(fval_sptr);
    } else if (isarray) {
      /*
       * since the result of the function is an array, a temporary
       * must be allocated at run-time even if its bounds are contant.
       * Note that for an 'adjustable' return value, its size
       * may be dependent on the actual arguments.
       */
      if (iface)
        return_value = ref_entry(iface);
      else
        return_value = fval;
      if (!ADJLENG(fval))
        return_value =
            gen_array_result(return_value, dpdsc, carg.nent, FALSE, iface);
      else
        return_value = gen_char_result(return_value, dpdsc, carg.nent);
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
      /*
       * have an array-valued function; save up information
       * which would allow substituting the result temp with
       * the LHS of an assignment.
       */
      save_func_arrinfo = 1;
    } else if (ADJLENG(fval)) {
      return_value = gen_char_result(fval, dpdsc, carg.nent);
      argt_count++;
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ARGT_ARG(argt, 0) = return_value;
      ii = 1;
    } else {
      argt = mk_argt(argt_count); /* mk_argt stuffs away count */
      ii = 0;
    }

    fval = gen_finalized_result(fval, func_sptr);

    for (i = 0; i < carg.nent; i++) {
      sp = ARG_STK(i);
      if (sp) {
        /* add to ARGT list, handling derived type arguments as
         * special case.
         */
        sptr1 = 0;
        if (SST_IDG(sp) == S_LVALUE)
          sptr1 = SST_LSYMG(sp);
        else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
          sptr1 = SST_SYMG(sp);
        else if (SST_IDG(sp) == S_SCONST) {
          (void)mkarg(sp, &dum);
          sptr1 = SST_SYMG(sp);
        }
        {
          param_dummy = inc_dummy_param(iface);

          if (is_iso_cloc(SST_ASTG(sp))) {
            if (find_byval_ref(func_sptr, param_dummy, 1) == PASS_BYVAL) {
              /* pass by val iso_c pointer to arg:
                 C_LOC(arg)   C_FUN_LOC(arg)
                 is plain old pass by reference
                 without type checking: get rid of the
                C_LOC:
               */
              new_ast = ARGT_ARG(A_ARGSG(SST_ASTG(sp)), 0);
              if (A_TYPEG(new_ast) == A_ID && (!TARGETG(A_SPTRG(new_ast))) &&
                  (!POINTERG(A_SPTRG(new_ast))))
                errwarn(468);

              SST_ASTP(sp, new_ast);
              SST_IDP(sp, S_EXPR);
              ARGT_ARG(argt, ii) = SST_ASTG(sp);
            } else {
              /* iso_c_loc by reference pointer to pointer */
              ARGT_ARG(argt, ii) = ARG_AST(i);
            }

          } else if (get_byval(func_sptr, param_dummy)) {
            /*  function arguments not processed by lowerilm */
            if (PASSBYVALG(param_dummy)) {
              if (OPTARGG(param_dummy)) {
                int assn = sem_tempify(sp);
                (void)add_stmt(assn);
                SST_ASTP(sp, A_DESTG(assn));
                byvalue_ref_arg(sp, &dum, OP_REF, func_sptr);
              } else if (!need_tmp_retval(iface, param_dummy)) {
                byvalue_ref_arg(sp, &dum, OP_BYVAL, iface);
              } else {
                byvalue_ref_arg(sp, &dum, OP_VAL, iface);
              }
            } else {
              byvalue_ref_arg(sp, &dum, OP_VAL, iface);
            }
            ARGT_ARG(argt, ii) = SST_ASTG(sp);
          } else if (pass_char_no_len(func_sptr, param_dummy)) {
            byvalue_ref_arg(sp, &dum, OP_REF, func_sptr);
            ARGT_ARG(argt, ii) = SST_ASTG(sp);
          } else {
            ARGT_ARG(argt, ii) = ARG_AST(i);
          }
          ii++;
        }
      } else if (i == pass_pos) {
        ARGT_ARG(argt, ii) = A_PARENTG(callee);
        ii++;
      } else {
        int npad;
        for (npad = ARG_AST(i); npad > 0; npad--) {
          ARGT_ARG(argt, ii) = astb.ptr0;
          ii++;
        }
      }
    }
    if (return_value) {
      /* return_value is symbol if result is of derived type;
       * otherwise, it's an ast.
       */
      dtype = DTYPEG(A_SPTRG(return_value));
      ast = mk_func_node(A_CALL, callee, argt_count, argt);
      sem.arrfn.call_std = add_stmt(ast);
      sem.arrfn.sptr = iface;
      if (save_func_arrinfo) {
        sem.arrfn.return_value = return_value;
        if (ALLOCG(A_SPTRG(return_value)))
          sem.arrfn.alloc_std = sem.alloc_std;
      }
      ast = return_value;
    } else {
      ast = mk_func_node(A_FUNC, callee, argt_count, argt);
    }
    if (ELEMENTALG(iface)) {
      int argc;
      for (argc = 0; argc < argt_count; ++argc) {
        /* Use first shaped argument */
        shaper = A_SHAPEG(ARGT_ARG(argt, argc));
        if (shaper)
          break;
      }
      if (shaper == 0) {
        shaper = mkshape(dtype);
      } else {
        dtype = dtype_with_shape(dtype, shaper);
        A_SHAPEP(ast, shaper);
      }
    } else {
      shaper = mkshape(dtype);
    }
    A_DTYPEP(ast, dtype);
    goto exit_;
  }
  ii = 0;
  /* before processing arguments, add derived type return values if needed */
  argt = mk_argt(argt_count); /* mk_argt stuffs away count */

  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_KEYWORD) {
      /* form is <ident> = <expression> */
      error(79, 3, gbl.lineno, scn.id.name + SST_CVALG(itemp->t.stkp), CNULL);
      itemp->t.sptr = 1;
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    if (SST_IDG(sp) == S_TRIPLE) {
      /* form is e1:e2:e3 */
      error(76, 3, gbl.lineno, SYMNAME(func_sptr), CNULL);
      itemp->t.sptr = 1;
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    if (SST_IDG(sp) == S_LABEL) {
      error(155, 3, gbl.lineno, "Illegal use of alternate return specifier",
            CNULL);
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    /* check arguments and add to ARGT list, handling derived type
       arguments as special case */
    sptr1 = 0;
    if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
      sptr1 = SST_SYMG(sp);
    else if (SST_IDG(sp) == S_LVALUE)
      sptr1 = SST_LSYMG(sp);
    else if (SST_IDG(sp) == S_SCONST) {
      (void)mkarg(sp, &dum);
      sptr1 = SST_SYMG(sp);
    }
    {
      /* form is <ident> or <expression> */
      param_dummy = inc_dummy_param(iface);
      /*  function arguments not processed bylowerilm */

      if ((A_TYPEG(SST_ASTG(sp)) == A_ID) &&
          is_iso_cptr(DTYPEG(A_SPTRG(SST_ASTG(sp))))) {
        if (find_byval_ref(iface, param_dummy, 1) == PASS_BYVAL) {
          /* iso cptr passed by value needs to transform into
             pass by value cptr->member : (pass the pointer
             sitting in cptr->member by value) */

          psptr = A_SPTRG(SST_ASTG(sp));
          msptr = DTY(DTYPEG(psptr) + 1);
          new_ast = mk_member(SST_ASTG(sp), mk_id(msptr), DTYPEG(msptr));
          SST_ASTP(sp, new_ast);
          SST_IDP(sp, S_EXPR);
          SST_DTYPEP(sp, DTYPEG(msptr));

          byvalue_ref_arg(sp, &dum, OP_VAL, iface);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        } else {
          /* plain pass by ref */
          itemp->t.sptr = chkarg(sp, &dum);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        }
      } else if (is_iso_cloc(SST_ASTG(sp))) {

        if (find_byval_ref(iface, param_dummy, 1) == PASS_BYVAL) {
          /* pass by val iso_c pointer to arg:
             C_LOC(arg)   C_FUN_LOC(arg)
             is plain old pass by reference
             without type checking: get rid of the c_LOC
           */
          new_ast = ARGT_ARG(A_ARGSG(SST_ASTG(sp)), 0);
          if (A_TYPEG(new_ast) == A_ID && (!TARGETG(A_SPTRG(new_ast))) &&
              (!POINTERG(A_SPTRG(new_ast))))
            errwarn(468);

          SST_ASTP(sp, new_ast);
          SST_IDP(sp, S_EXPR);
          ARGT_ARG(argt, ii) = SST_ASTG(sp);

        } else {
          /* iso_c_loc by reference: pointer to pointer */
          ARGT_ARG(argt, ii) = SST_ASTG(sp);
        }
      } else if (get_byval(iface, param_dummy)) {

        itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_VAL, iface);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      } else if (pass_char_no_len(iface, param_dummy)) {
        itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_REF, iface);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);

      } else {
        itemp->t.sptr = chkarg(sp, &dum);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      }
      ii++;
    }
  }

  ast = mk_func_node(A_FUNC, callee, argt_count, argt);
  A_DTYPEP(ast, dtype);
  A_SHAPEP(ast, mkshape(dtype));
  if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR)
    error(89, 3, gbl.lineno, SYMNAME(func_sptr), CNULL);

exit_:
  SST_IDP(stktop, S_EXPR);
  SST_ASTP(stktop, ast);
  if (shaper)
    SST_SHAPEP(stktop, shaper);
  else
    SST_SHAPEP(stktop, A_SHAPEG(ast));
  SST_DTYPEP(stktop, dtype);
  if (kwd_str)
    FREE(kwd_str);

  return 1;
}

/*
 * add the proc data type to a list so that semfin can
 * adjust the PARAMCT and DPDSC values for functions
 * returning certain types.
 */
static void
add_typroc(int dt)
{
  int i;

  for (i = 0; i < sem.typroc_avail; i++) {
    if (sem.typroc_base[i] == dt)
      return;
  }
  sem.typroc_avail++;
  NEED(sem.typroc_avail, sem.typroc_base, int, sem.typroc_size,
       sem.typroc_avail + 50);
  sem.typroc_base[sem.typroc_avail - 1] = dt;
}

static void
count_actuals(ITEM *list)
{
  ITEM *itemp;
  SST *sp;
  int dum;

  carg.nargt = carg.nent = 0;
  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_KEYWORD)
      sp = SST_E3G(sp);
    /* adjust argument count, if derived type arguments are used as
       individual entities */
    if (SST_IDG(sp) == S_SCONST) {
      mkarg(sp, &dum); /* mkarg will assign to tmp- S_IDENT */
      carg.nargt++;
    } else
      carg.nargt++;
    carg.nent++;
  }
}

static void
count_formals(int sptr)
{
  count_formal_args(PARAMCTG(sptr), DPDSCG(sptr));
}

static void
count_formal_args(int paramct, int dpdsc)
{
  int *dscptr;
  int arg;
  int i;

  carg.nargt = carg.nent = paramct;
  dscptr = aux.dpdsc_base + dpdsc;
  for (i = paramct; i > 0; i--) {
    arg = *dscptr++;
    if (CLASSG(arg) && CCSYMG(arg) /*&& OPTARGG(arg)*/) {
      carg.nargt--;
      carg.nent--;
    }
    if (DESCARRAYG(arg) && NODESCG(arg) && DTY(DTYPEG(arg)) == TY_ARRAY &&
        NODESCG(arg)) {
      carg.nargt--;
      carg.nent--;
    }
  }
}

static int
fix_character_length(int dtype, int func_sptr)
{
  int dscptr, paramct, clen;
  if (DTY(dtype) != TY_CHAR
      && DTY(dtype) != TY_NCHAR
  )
    return dtype;

  /* we have a character datatype, replace any formal arguments in
   * the character length by their values, rewrite the length */
  dscptr = DPDSCG(func_sptr);
  paramct = PARAMCTG(func_sptr);
  ast_visit(1, 1);
  replace_arguments(dscptr, paramct);
  clen = ast_rewrite(DTY(dtype + 1));
  ast_unvisit();
  if (clen == DTY(dtype + 1))
    return dtype;
  /* character length has changed, create new character datatype */
  dtype = get_type(2, DTY(dtype), clen);
  return dtype;
} /* fix_character_length */

static int
gen_pointer_result(int array_value, int dscptr, int nactuals,
                   LOGICAL is_derived, int func_sptr)
{
  int o_dt;
  int dt;
  int arr_tmp;

  o_dt = DTYPEG(array_value);
  if (DTY(o_dt) == TY_ARRAY) {
    int l;
    dt = dup_array_dtype(o_dt);
    l = fix_character_length(DTY(dt + 1), func_sptr);
    DTY(dt + 1) = l;
  } else {
    dt = fix_character_length(o_dt, func_sptr);
  }
  /*
   * Create a new pointer temporary with a new dtype record
   */
  if (is_derived) {
    arr_tmp = array_value;
    DTYPEP(arr_tmp, dt);
  } else {
    int ddt; 
    arr_tmp = get_next_sym(SYMNAME(array_value), NULL);
    dup_sym(arr_tmp, stb.stg_base + array_value);
    DTYPEP(arr_tmp, dt);
    DESCRP(arr_tmp, 0);
    /*
     * set_descriptor_sc(sem.sc); already called in the caller
     */
    get_static_descriptor(arr_tmp);
    get_all_descriptors(arr_tmp);
    /* need to have different MIDNUM than arr_value */
    /* otherwise multiple declaration */
    MIDNUMP(arr_tmp, 0);
    NODESCP(arr_tmp, 0);
    ddt = DDTG(dt);
    if ((DTY(dt) == TY_CHAR && dt != DT_DEFERCHAR) ||
        (DTY(dt) == TY_NCHAR && dt != DT_DEFERNCHAR)) {
      add_auto_len(arr_tmp, 0);
      if (CVLENG(arr_tmp))
        EARLYSPECP(CVLENG(arr_tmp), 1);
    }
  }
  if (gbl.internal > 1) {
    INTERNALP(arr_tmp, 1);
  } else {
    INTERNALP(arr_tmp, 0);
  }
  if (DTY(o_dt) == TY_ARRAY) {
    STYPEP(arr_tmp, ST_ARRAY);
    ALLOCP(arr_tmp, 1);
  } else
    STYPEP(arr_tmp, ST_VAR);
  SCOPEP(arr_tmp, stb.curr_scope);
  IGNOREP(arr_tmp, 0);
  SLNKP(arr_tmp, 0);
  SOCPTRP(arr_tmp, 0);
  SCP(arr_tmp, SC_BASED);
  ref_based_object(arr_tmp);

  return mk_id(arr_tmp);
}

static int
gen_allocatable_result(int array_value, int dscptr, int nactuals,
                       LOGICAL is_derived, int func_sptr)
{
  int o_dt;
  int dt;
  int arr_tmp;
  int pvar;
  int astrslt;

  o_dt = DTYPEG(array_value);
  if (DTY(o_dt) == TY_ARRAY) {
    int l;
    dt = dup_array_dtype(o_dt);
    l = fix_character_length(DTY(dt + 1), func_sptr);
    DTY(dt + 1) = l;
  } else {
    dt = fix_character_length(o_dt, func_sptr);
  }
  /*
   * Create a new allocatable temporary with a new dtype record
   */
  arr_tmp = get_next_sym(SYMNAME(array_value), NULL);
  dup_sym(arr_tmp, stb.stg_base + array_value);
  DTYPEP(arr_tmp, dt);
  DESCRP(arr_tmp, 0);
  /*
   * Would like to call set_descriptor_sc() at the beginning
   * of func2_call() and restore at the end; however, there
   * are still semsym things that might need to be done to user
   * variables.  So, only call set_descriptor_sc() when we know
   * we are creating temps.
   */
  set_descriptor_sc(sem.sc);
  get_static_descriptor(arr_tmp);
  get_all_descriptors(arr_tmp);
  /* need to have different MIDNUM than arr_value */
  /* otherwise multiple declaration */
  pvar = sym_get_ptr(arr_tmp);
  MIDNUMP(arr_tmp, pvar);
  NODESCP(arr_tmp, 0);
  ALLOCATTRP(arr_tmp, 1);
  set_descriptor_sc(SC_LOCAL);
  if (DTY(o_dt) == TY_ARRAY) {
    STYPEP(arr_tmp, ST_ARRAY);
    ALLOCP(arr_tmp, 1);
  } else
    STYPEP(arr_tmp, ST_VAR);
  if (gbl.internal > 1) {
    INTERNALP(arr_tmp, 1);
  } else {
    INTERNALP(arr_tmp, 0);
  }
  SCOPEP(arr_tmp, stb.curr_scope);
  IGNOREP(arr_tmp, 0);
  SLNKP(arr_tmp, 0);
  SOCPTRP(arr_tmp, 0);
  SCP(arr_tmp, SC_BASED);
  astrslt = ref_based_object_sc(arr_tmp, sem.sc);
  ALLOCATTRP(arr_tmp, 1);
  astrslt = mk_id(arr_tmp);

  return astrslt;
}

/*
 * check whether an array descriptor has fixed bounds
 * and whether the bounds are 'small enough'
 */
static int
small_enough(ADSC *ad, int numdim)
{
  int i;
  ISZ_T size;
  size = 1;
  for (i = 0; i < numdim; ++i) {
    int l, u;
    ISZ_T lv, uv;
    l = AD_LWBD(ad, i);
    if (l && !A_ALIASG(l))
      return 0;
    lv = 1; /* default */
    if (l) {
      l = A_ALIASG(l);
      assert(A_TYPEG(l) == A_CNST,
             "small_enough: expecting constant lower bound", l, 4);
      lv = get_isz_cval(A_SPTRG(l));
    }
    u = AD_UPBD(ad, i);
    if (!u || !A_ALIASG(u))
      return 0; /* not fixed size, or assumed-size */
    u = A_ALIASG(u);
    assert(A_TYPEG(u) == A_CNST, "small_enough: expecting constant upper bound",
           l, 4);
    uv = get_isz_cval(A_SPTRG(u));
    size *= (uv - lv + 1);
    if (size > 1000)
      return 0;
  }
  return 1;
} /* small_enough */

static int
gen_array_result(int array_value, int dscptr, int nactuals, LOGICAL is_derived,
                 int callee)
{
  int numdim;
  int o_dt;
  int dt;
  int arr_tmp;
  int ii;
  ADSC *ad;

  o_dt = DTYPEG(array_value);
  ad = AD_DPTR(o_dt);
  numdim = AD_NUMDIM(ad);
  /*
   * 0.  Check whether the return array size is fixed size and
   *     small enough to simply use a local array
   */
  if (small_enough(ad, numdim)) {
    /* use same name, etc. */
    arr_tmp = get_next_sym(SYMNAME(array_value), NULL);
    dup_sym(arr_tmp, stb.stg_base + array_value);
    NODESCP(arr_tmp, 0);
    DESCRP(arr_tmp, 0);
    ARGP(arr_tmp, 1);
    STYPEP(arr_tmp, ST_ARRAY);
    SCOPEP(arr_tmp, stb.curr_scope);
    IGNOREP(arr_tmp, 0);
    DTYPEP(arr_tmp, o_dt);
    SLNKP(arr_tmp, 0);
    if (gbl.internal > 1) {
      INTERNALP(arr_tmp, 1);
    } else {
      INTERNALP(arr_tmp, 0);
    }
    SCP(arr_tmp, sem.sc);
    return mk_id(arr_tmp);
  }
  /*
   * 1.  Create an allocatable temporary with a deferred-shape dtype
   *     using the sem.arrdim data structure.
   */
  sem.arrdim.ndefer = sem.arrdim.ndim = numdim;
  for (ii = 0; ii < numdim; ii++)
    sem.bounds[ii].lowtype = S_NULL;
  dt = mk_arrdsc();
  DTY(dt + 1) = DTY(o_dt + 1);

  if (is_derived)
    arr_tmp = array_value;
  else {
    arr_tmp = get_next_sym(SYMNAME(array_value), NULL);
    dup_sym(arr_tmp, stb.stg_base + array_value);
    NODESCP(arr_tmp, 0);
    DESCRP(arr_tmp, 0);
    PARAMCTP(arr_tmp, 0);
  }

  ARGP(arr_tmp, 1);
  STYPEP(arr_tmp, ST_ARRAY);
  SCOPEP(arr_tmp, stb.curr_scope);
  IGNOREP(arr_tmp, 0);
  DTYPEP(arr_tmp, dt);
  SLNKP(arr_tmp, 0);
  if (gbl.internal > 1) {
    INTERNALP(arr_tmp, 1);
  } else {
    INTERNALP(arr_tmp, 0);
  }
  SCP(arr_tmp, SC_BASED);
  ALLOCP(arr_tmp, 1);
  HCCSYMP(arr_tmp, 1);
  ref_based_object_sc(arr_tmp, sem.sc);

  /*
   * 2.  Generate the assignments to the bounds temporaries
   *     of the array temp and allocate it.
   * 2a. The values of the temporaries may depend on the actual arguments
   *     (e.g., a specification expression may refer to a formal); therefore,
   *     the 'formals' are replaced with the actuals.
   * 2b. If the current context is an internal procedure whose host is a
   *     module subroutine and the function called is also internal. The
   *     values of the bounds temps may depend on the dummy arguments of
   *     the host.  At this point, there are two symbol table entries for
   *     the host:
   *     1) ST_ENTRY and this is the parent scope of the current internal
   *        routine
   *     2) ST_PROC since the host is within a module -- recall that when a
   *        module is compiled, two syms are created for the module routine:
   *        an ST_PROC representing the routine's interface and an ST_ENTRY
   *        for when the body of the routine is actually compiled.
   *     These sym entries are distinct and each will have their own sym
   *     entries for their dummy arguments.  If there are bounds declarations
   *     in any array formal or result which refer to a host dummy, the
   *     corresponding sym entry for the dummy is the ST_PROC.  When the
   *     callee is invoked, the host dummy is in scope of the ST_ENTRY.
   *     Consequently, the bounds values refer to the incorrect instance of
   *     the host dummy.  The ASTs of the ST_PROC's host dummies referenced
   *     in the bounds computations must be replaced with the ASTs of the
   *     corresponding ST_ENTRY's host host dummies.
   */
  ad = AD_DPTR(o_dt);
  if (AD_ADJARR(ad)) {
    precompute_arg_intrin(dscptr, nactuals);
    precompute_args(dscptr, nactuals);
  }
  ast_visit(1, 1);
  if (gbl.currmod != 0 && gbl.internal > 1 && callee && INTERNALG(callee)) {
    /*
     * In an internal procedure whose host is a module routine and the
     * called function is also internal.
     */
    int host = SCOPEG(gbl.currsub); /* module routine (probably an ST_ALIAS) */
    /*
     * if sem.modhost_proc is non-zero, the host's ST_PROC & ST_ENTRY were
     * already discovered
     */
    if (sem.modhost_proc == 0) {
      /* starting with the first entry in the hash list, find the ST_PROC*/
      sem.modhost_proc = get_symtype(ST_PROC, first_hash(host));
      if (sem.modhost_proc != 0) {
        /*
         * if ST_PROC found, now find the ST_ENTRY - it will follow the ST_PROC
         * so do not have start over at first_hash(host).
         */
        sem.modhost_entry = get_symtype(ST_ENTRY, HASHLKG(sem.modhost_proc));
        if (sem.modhost_entry == 0)
          sem.modhost_proc = 0;
      }
    }
    if (sem.modhost_entry != 0) {
      /*
       * scan the ST_PROC's and ST_ENTRY's arguments and replace the
       * ASTs of the ST_PROC's args with the ASTs of the ST_ENTRY's args.
       */
      int i;
      for (i = PARAMCTG(sem.modhost_proc); i > 0; i--) {
        int old = aux.dpdsc_base[DPDSCG(sem.modhost_proc) + i - 1];
        int new = aux.dpdsc_base[DPDSCG(sem.modhost_entry) + i - 1];
        ast_replace(mk_id(old), mk_id(new));
      }
    }
  }
  replace_arguments(dscptr, nactuals);
  /*
   * 3.  Rewrite the bounds expressions of the original
   *     declaration of the function.  These values become
   *     the bounds expressions of the array temp and are
   *     stored in the sem.bounds data structure.
   *     Reset the sem.arrdim fields of (1) since
   *     precompute_arg_intrin() could cause them to be set
   *     for another context
   */
  sem.arrdim.ndefer = sem.arrdim.ndim = numdim;
  for (ii = 0; ii < numdim; ii++) {
    sem.bounds[ii].lowtype = S_NULL;
    if (AD_LWBD(ad, ii)) {
      replace_formal_triples(AD_LWBD(ad, ii), dscptr, nactuals);
      replace_formal_bounds(AD_LWBD(ad, ii), dscptr, nactuals, 0);
      sem.bounds[ii].lwast = ast_rewrite((int)AD_LWBD(ad, ii));
    } else {
      sem.bounds[ii].lwast = astb.bnd.one;
    }
    replace_formal_triples(AD_UPBD(ad, ii), dscptr, nactuals);
    replace_formal_bounds(AD_UPBD(ad, ii), dscptr, nactuals, 0);
    sem.bounds[ii].upast = ast_rewrite((int)AD_UPBD(ad, ii));
  }
  ast_unvisit();
  /*
   * 4.  assign values to the bounds temporaries and
   *     allocate the array; the utility routine also
   *     saves enough information so that the array
   *     temporary can be deallocated.
   */
  gen_allocate_array(arr_tmp);
  return mk_id(arr_tmp);
}

static int
gen_char_result(int fval, int dscptr, int nactuals)
{
  int dt, edt;
  int ctemp;
  int len;

  dt = DTYPEG(fval);
  edt = dt;
  if (DTY(dt) == TY_ARRAY)
    edt = DTY(dt + 1);
  ast_visit(1, 1);
  replace_arguments(dscptr, nactuals);
  replace_formal_bounds(DTY(edt + 1), dscptr, nactuals, 0);
  len = ast_rewrite(DTY(edt + 1));
  ast_unvisit();
  if (A_TYPEG(len) == A_INTR && A_OPTYPEG(len) == I_LEN) {
    int aaa;
    aaa = ARGT_ARG(A_ARGSG(len), 0);
    if (A_TYPEG(aaa) == A_INTR && A_OPTYPEG(aaa) == I_TRIM) {
      len = ast_intr(I_LEN_TRIM, astb.bnd.dtype, 1, ARGT_ARG(A_ARGSG(aaa), 0));
    }
  }
  if (len != DTY(edt + 1)) {
    edt = get_type(2, TY_CHAR, len);
    if (DTY(dt) != TY_ARRAY)
      dt = edt;
    else {
      dt = dup_array_dtype(dt);
      DTY(dt + 1) = edt;
    }
  }
  ctemp = get_ch_temp(dt);
  return mk_id(ctemp);
}

static void
precompute_arg_intrin(int dscptr, int nactuals)
{
  int ii;
  int dtype;

  for (ii = 0; ii < nactuals; ii++) {
    int arg, tmp, assn;
    if (!ARG_STK(ii))
      continue;
    arg = ARG_AST(ii);
    if (A_TYPEG(arg) == A_INTR) {
      dtype = A_DTYPEG(arg);
      if (DTY(dtype) == TY_ARRAY) {
        int shape;
        shape = A_SHAPEG(arg);
        if (shape) {
          if (SHD_NDIM(shape) != ADD_NUMDIM(dtype)) {
            tmp = get_shape_arr_temp(arg);
          } else {
            ADSC *ad;
            ad = AD_DPTR(dtype);
            if (AD_DEFER(ad) || AD_ADJARR(ad) || AD_NOBOUNDS(ad)) {
              tmp = get_shape_arr_temp(arg);
            } else
              tmp = get_arr_temp(dtype, FALSE, TRUE, FALSE);
          }
        } else
          tmp = get_arr_temp(dtype, FALSE, TRUE, FALSE);
      } else {
        dtype = get_temp_dtype(dtype, arg);
        tmp = get_temp(dtype);
      }
      assn = mk_assn_stmt(mk_id(tmp), arg, dtype);
      (void)add_stmt(assn);
      ARG_AST(ii) = A_DESTG(assn);
      SST_ASTP(ARG_STK(ii), ARG_AST(ii));
    }
  }
}

static void
precompute_args(int dscptr, int nactuals)
{
  int ii;

  for (ii = 0; ii < nactuals; ii++) {
    int arg, dtype, assn;
    if (!ARG_STK(ii))
      continue;
    arg = ARG_AST(ii);
    if (!A_CALLFGG(arg))
      continue;
    dtype = A_DTYPEG(arg);
    if (!DT_ISSCALAR(dtype) && DTY(dtype) != TY_DERIVED)
      continue;
    assn = sem_tempify(ARG_STK(ii));
    (void)add_stmt(assn);
    ARG_AST(ii) = A_DESTG(assn);
    SST_ASTP(ARG_STK(ii), ARG_AST(ii));
  }
}

static void
rewrite_triples(int ast_subscr, int dscptr, int nactuals)
{
  int i, j;
  int sptr_actual;
  int ast_actual = A_LOPG(ast_subscr);

  if (A_TYPEG(ast_actual) == A_ID) {
    sptr_actual = A_SPTRG(ast_actual);
  } else if (A_TYPEG(ast_actual) == A_MEM) {
    sptr_actual = A_SPTRG(A_MEMG(ast_actual));
  } else {
    return;
  }

  for (i = 0; i < nactuals; i++) {
    if (ARG_STK(i)) {
      int sptr_arg;
      int arg = ARG_AST(i);
      if (A_TYPEG(arg) == A_ID) {
        sptr_arg = A_SPTRG(arg);
      } else if (A_TYPEG(arg) == A_MEM) {
        sptr_arg = A_SPTRG(A_MEMG(arg));
      } else {
        continue;
      }
      if (sptr_arg == sptr_actual) {
        int asd = A_ASDG(ast_subscr);
        int ndim = ASD_NDIM(asd);
        int dt_formal = DTYPEG(aux.dpdsc_base[dscptr + i]);
        ADSC *ad_formal = AD_DPTR(dt_formal);
        for (j = 0; j < ndim; j++) {
          int sub = ASD_SUBS(asd, j);
          if (A_TYPEG(sub) == A_TRIPLE &&
              AD_LWBD(ad_formal, j) == A_LBDG(sub) &&
              AD_UPBD(ad_formal, j) == A_UPBDG(sub)) {
            /* the triple is from the dummy arg, replace it */
            ADSC *ad_actual = AD_DPTR(DTYPEG(sptr_actual));
            int triple = mk_triple(AD_LWBD(ad_actual, j), AD_UPBD(ad_actual, j),
                                   AD_EXTNTAST(ad_actual, j));
            ast_replace(sub, triple);
          }
        }
      }
    }
  }
}

/*
 * A formal array can be subscripted in a specification expression;
 * when this occurs, need to check if the corresponding actual argument is
 * an array section.   The original processing can create something like:
 *    act(1:10)(1)
 * where the formal appears as formal(1) is some expression and the actual
 * argument is act(1:10).  Eventually, the illegal subscripting could  lead
 * to an ICE.
 */
static void
rewrite_subscr(int ast_subscr, int dscptr, int nactuals)
{
  int ast;
  int sptr;
  int arr, rpl;
  int flg;
  int i;
  int actarr;
  int asd, numdim;
  int subs[7]; /* maximum number of dimensions */
  int triple;
  int subscr;

  arr = A_LOPG(ast_subscr);
  if (A_TYPEG(arr) != A_ID)
    return;
  /*
   * Make sure what's being subscripted is a formal array which is being
   * replaced by some interesting array expression ...
   * is
   */
  rpl = A_REPLG(arr);
  if (!rpl)
    /* not being replaced */
    return;
  sptr = A_SPTRG(arr);
  if (STYPEG(sptr) != ST_ARRAY && SCG(sptr) != SC_DUMMY)
    return;
  flg = 0;
  for (i = 0; i < nactuals; i++) {
    if (sptr == aux.dpdsc_base[dscptr + i]) {
      /* is a formal argument of the called routine */
      flg = 1;
      break;
    }
  }
  if (!flg)
    /* not a formal array argument */
    return;

  if (A_TYPEG(rpl) != A_SUBSCR)
    /* the replacing expression is not being subscripted */
    return;

  /*
   *+++++++++++++++++  WARNING  +++++++++++++++++
   * only allow a single subscript of the formal for now. This covers
   * the bug in f15222, but eventually, this will need to be generalized.
   */
  asd = A_ASDG(ast_subscr);
  if (ASD_NDIM(asd) != 1)
    return;
  subscr = ASD_SUBS(asd, 0);

  actarr = A_LOPG(rpl);
  if (A_TYPEG(actarr) != A_ID)
    /* the actual arg being subscripted is not a simple array */
    return;

  asd = A_ASDG(rpl);
  numdim = ASD_NDIM(asd);
  flg = 0;
  for (i = 0; i < numdim; i++) {
    subs[i] = ASD_SUBS(asd, i);
    if (A_TYPEG(subs[i]) == A_TRIPLE) {
      flg = 1;
      triple = i;
    }
  }
  if (!flg) {
    /*
     * strictly speaking, this is an error that should have already
     * been caught since the formal is subscripted, and the actual
     * argument which is subscripted is not array-valued!
     */
    return;
  }
  subs[triple] = subscr;
  /*
   * create a new subscripted reference where the subscript expression
   * of the formal is folded into the subscript expression of the
   * actual argument.  The new subscripted references replaces the
   * current subscripted reference of the formal.
   */
  ast = mk_subscr(actarr, subs, numdim, A_DTYPEG(ast_subscr));
  ast_replace(ast_subscr, ast);
}

static void
replace_formal_triples(int ast, int dscptr, int nactuals)
{
  int cnt;
  int argt;
  int i;

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    replace_formal_triples(A_LOPG(ast), dscptr, nactuals);
    replace_formal_triples(A_ROPG(ast), dscptr, nactuals);
    break;
  case A_UNOP:
  case A_PAREN:
  case A_CONV:
    replace_formal_triples(A_LOPG(ast), dscptr, nactuals);
    break;
  case A_INTR:
    cnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    for (i = 0; i < cnt; i++) {
      /* watch for optional args */
      if (ARGT_ARG(argt, i) != 0) {
        replace_formal_triples(ARGT_ARG(argt, i), dscptr, nactuals);
      }
    }
    break;
  case A_SUBSCR:
    rewrite_triples(ast, dscptr, nactuals);
    rewrite_subscr(ast, dscptr, nactuals);
    break;
  default:
    ast_visit(ast, 1);
  }
}

/** \brief Replace the intrinsic call of LBOUND/UBOUND on assumed-shape formal.
 * Model after replace_formal_triples which replaces triples in bounds.
 *
 * The upper bound of the assumed-shape formal should be determined by the
 * lower bound of the formal and the extent of the actual.
 *
 * \param ast      the ast to traverse
 * \param dscptr   dummy parameter descriptor
 * \param nactuals count of actuals (or formals)
 * \param nextstd  insert the generated stmts before this stmt
 */
static void
replace_formal_bounds(int ast, int dscptr, int nactuals, int nextstd) {
  int cnt, argt, i, asd;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    replace_formal_bounds(A_LOPG(ast), dscptr, nactuals, nextstd);
    replace_formal_bounds(A_ROPG(ast), dscptr, nactuals, nextstd);
    break;
  case A_UNOP:
  case A_PAREN:
  case A_CONV:
    replace_formal_bounds(A_LOPG(ast), dscptr, nactuals, nextstd);
    break;
  case A_INTR:
    cnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    for (i = 0; i < cnt; i++) {
      /* watch for optional args */
      if (ARGT_ARG(argt, i) != 0)
        replace_formal_bounds(ARGT_ARG(argt, i), dscptr, nactuals, nextstd);
    }
    /* Do the replace */
    if ((A_OPTYPEG(ast) == I_LBOUND || A_OPTYPEG(ast) == I_UBOUND) &&
        A_TYPEG(ARGT_ARG(argt, 0)) == A_ID) {
      SPTR array = A_SPTRG(ARGT_ARG(argt, 0));
      if (ASSUMSHPG(array)) {
        for (i = 0; i < nactuals; i++) {
          if (array == aux.dpdsc_base[dscptr + i]) {
            int astnew = rewrite_lbound_ubound(ast, A_REPLG(ARGT_ARG(argt, 0)),
                                               nextstd);
            ast_replace(ast, astnew);
            break;
          }
        }
      }
    }
    break;
  case A_SUBSCR:
    replace_formal_bounds(A_LOPG(ast), dscptr, nactuals, nextstd);
    asd = A_ASDG(ast);
    for (i = 0; i < ASD_NDIM(asd); ++i)
      replace_formal_bounds(ASD_SUBS(asd, i), dscptr, nactuals, nextstd);
    break;
  case A_MEM:
    replace_formal_bounds(A_PARENTG(ast), dscptr, nactuals, nextstd);
    break;
  default:
    ast_visit(ast, 1);
  }
}

/*
 * Substitute the formal arguments with the actual arguments.
 * Also, the appearance of formal arguments in descriptors need to
 * be replaced.
 */
static void
replace_arguments(int dscptr, int nactuals)
{
  int ii;

  for (ii = 0; ii < nactuals; ii++) {
    if (ARG_STK(ii)) {
      int formal, formalid, arg, argid, astmem;
      formalid = aux.dpdsc_base[dscptr + ii];
      formal = mk_id(formalid);
      arg = ARG_AST(ii);
      ast_replace(formal, arg); /*formal <- actual*/
      argid = 0;
      if (A_TYPEG(arg) == A_ID) {
        argid = A_SPTRG(arg);
        astmem = 0;
      } else if (A_TYPEG(arg) == A_MEM) {
        argid = A_SPTRG(A_MEMG(arg));
        astmem = arg;
      }
      if (argid && formalid) {
        /* see if we should also replace any SDSC references
         * in the bounds, such as might come from translated
         * LBOUND(a,1) refs */
        if (SDSCG(formalid)) {
          formal = mk_id(SDSCG(formalid));
          if (!SDSCG(argid)) {
            get_static_descriptor(argid);
            get_all_descriptors(argid);
          }
          arg = check_member(astmem, mk_id(SDSCG(argid)));
          ast_replace(formal, arg);
        }
      }
    }
  }
}

static int
get_tbp(int sptr)
{
  /* Get a type bound procedure. Assume that sptr points to a user
   * defined type bound procedure. We then mangle it with a $tbp suffix.
   * This returns the sptr of the mangled type bound procedure (binding
   * name).
   */

  int len;
  char *name;

  if (STYPEG(sptr) != ST_PROC) {
    /* If we get here with a symbol that isn't a procedure, don't create
     * a new ...$tbp symbol that'll never be used.
     */
    return sptr;
  }

  name = SYMNAME(sptr);
  len = strlen(name);
  if (len > 4 && strcmp("$tbp", name + (len - 4)) == 0) {
    return sptr;
  }
  return getsymf("%s$tbp", name);
}

int
get_tbp_argno(int sptr, int dty)
{
  if (dty <= 0)
    dty = TBPLNKG(sptr);
  if (dty > 0 && VTOFFG(sptr) != 0) {
    int mem, imp = get_implementation(dty, sptr, 0, &mem), first = imp;
    while (imp > NOSYM) {
      int paramct, dpdsc, bind;
      assert(mem > NOSYM, "get_tbp_argno: bad mem sptr", sptr, 3);
      /* set bind to VTABLEG(mem) if bind is a generic type bound procedure */
      bind = STYPEG(sptr) == ST_PROC ? BINDG(mem) : VTABLEG(mem);
      if (PASSG(mem) <= NOSYM && !NOPASSG(mem) && INVOBJG(bind) > 0)
        return INVOBJG(bind);
      proc_arginfo(imp, &paramct, &dpdsc, 0);
      if (dpdsc > 0) {
        /* found what must be the implementation */
        int invobj = find_dummy_position(imp, PASSG(mem));
        if (invobj == 0) {
          if (PASSG(mem) > NOSYM) {
            char *name = SYMNAME(sptr), *name2 = name;
            int len = strlen(name);
            if (len > 4 && strcmp("$tbp", name + (len - 4)) == 0) {
              name2 = getitem(0, len + 1);
              strncpy(name2, name, len - 4);
            }
            error(155, 3, gbl.lineno,
                  "PASS arguments for type bound procedure "
                  "must have same name and position as overridden type bound "
                  "procedure",
                  name2);
          } else if (!NOPASSG(mem)) {
            invobj = 1; /* when no PASS or NOPASS, pass in the first position */
          }
        }
        if (invobj > 0 && STYPEG(sptr) == ST_PROC)
          INVOBJP(sptr, invobj);
        return invobj;
      }
      /* Try next hash link before giving up */
      get_next_hash_link(imp, 0 /* magic code to clear name's VISIT flags */);
      imp = get_next_hash_link(imp, 1 /* magic code, STYPE must match */);
      if (imp > NOSYM && test_scope(imp) != 0)
        imp = 0;
    }

    if (first <= NOSYM)
      first = sptr;
    error(155, 3, gbl.lineno,
          "Type bound procedure must be a module procedure "
          "or an external procedure with an explicit interface - ",
          SYMNAME(first));
  }
  return 0;
}

int
get_generic_member(int dtype, int sptr)
{

  /* This function is used to find the generic type bound procedure member
   * for a given dtype by matching the sptr with a member's VTABLE entry.
   * This function is also used in finding the type bound procedure
   * member with a given implementation (see chk_arguments() in
   * semfunc2.c).
   */

  int tag, mem;

  if (!dtype || DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (CLASSG(mem) && VTABLEG(mem) && BINDG(mem) &&
        strcmp(SYMNAME(sptr), SYMNAME(VTABLEG(mem))) == 0) {
      return mem;
    }
  }

  tag = DTY(dtype + 3);
  if (PARENTG(tag)) {
    mem = get_generic_member(DTYPEG(PARENTG(tag)), sptr);
  }

  return (mem > NOSYM) ? mem : 0;
}

int
get_generic_member2(int dtype, int sptr, int argcnt, int *argno)
{

  /* Similar to get_generic_member() above, except it assumes sptr is the
   * generic type bound procedure symbol (i.e., has a $tbpg suffix).
   */
  int tag, mem, candidate, exact_match;

  if (!dtype || DTY(dtype) != TY_DERIVED)
    return 0;
  if (argno)
    *argno = 0;
  candidate = exact_match = 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (CLASSG(mem) && VTABLEG(mem) && BINDG(mem) &&
        strcmp(SYMNAME(sptr), SYMNAME(BINDG(mem))) == 0) {
      if (argcnt) {
        int mem2, func;
        mem2 = 0;
        func = get_implementation(dtype, VTABLEG(mem), 0, &mem2);
        if (mem2) {
          int i, paramct, dpdsc, reqargs, optargs, arg2, pass_arg;
          proc_arginfo(func, &paramct, &dpdsc, NULL);
          for (pass_arg = reqargs = optargs = i = 0; i < paramct; ++i) {
            arg2 = aux.dpdsc_base[dpdsc + i];
            if (OPTARGG(arg2)) {
              ++optargs;
            } else {
              ++reqargs;
            }
            if (PASSG(mem2) &&
                strcmp(SYMNAME(PASSG(mem2)), SYMNAME(arg2)) == 0) {
              pass_arg = arg2;
              if (argno)
                *argno = i + 1;
            } else if (i == 0 && !PASSG(mem2) && !NOPASSG(mem2)) {
              pass_arg = arg2;
              if (argno)
                *argno = i + 1;
            }
          }
          reqargs = (reqargs > 0) ? reqargs - (pass_arg > NOSYM) : 0;
          if (!optargs && argcnt == reqargs) {
            if (eq_dtype2(DTYPEG(pass_arg), dtype, 0))
              return mem;
            else if (eq_dtype2(DTYPEG(pass_arg), dtype, 1) && !exact_match)
              candidate = mem;
            else if (!pass_arg)
              candidate = mem;
          } else if (optargs && argcnt <= (optargs + reqargs)) {
            if (eq_dtype2(DTYPEG(pass_arg), dtype, 0)) {
              exact_match = 1;
              candidate = mem;
            } else if (eq_dtype2(DTYPEG(pass_arg), dtype, 1) && !exact_match)
              candidate = mem;
            else if (!pass_arg)
              candidate = mem;
          }
        }
      }
    }
  }
  tag = DTY(dtype + 3);
  if (candidate > NOSYM) {
    return candidate;
  }

  if (PARENTG(tag)) {
    mem = get_generic_member2(DTYPEG(PARENTG(tag)), sptr, argcnt, argno);
  }

  return (mem > NOSYM) ? mem : 0;
}

int
generic_tbp_has_pass_and_nopass(int dtype, int sptr)
{

  /* Checks for the special case where a generic type bound procedure has
   * two identical specific type bound procedures except one has nopass
   * and the other has pass set. Assumes that sptr is a generic tbp.
   */

  int found_nopass, found_pass;
  int tag, mem;

  if (STYPEG(sptr) != ST_USERGENERIC && STYPEG(sptr) != ST_OPERATOR)
    return 0;
  if (!dtype || DTY(dtype) != TY_DERIVED)
    return 0;
  found_nopass = found_pass = 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (CLASSG(mem) && VTABLEG(mem) && BINDG(mem) &&
        strcmp(SYMNAME(sptr), SYMNAME(BINDG(mem))) == 0) {
      if (NOPASSG(mem))
        found_nopass = 1;
      else
        found_pass = 1;
    }
  }

  tag = DTY(dtype + 3);
  if (PARENTG(tag)) {
    return generic_tbp_has_pass_and_nopass(DTYPEG(PARENTG(tag)), sptr);
  }

  return found_nopass && found_pass;
}

int
get_generic_tbp_pass_or_nopass(int dtype, int sptr, int flag)
{

  /* Get the generic tbp sptr from dtype. If flag is set, then
   * this routine will return the NOPASS version (if available),
   * else the PASS version (if available). It returns 0 if generic
   * tbp is not available or none available from the flag criteria.
   */
  int found_nopass, found_pass;
  int tag, mem;

  if (STYPEG(sptr) != ST_USERGENERIC && STYPEG(sptr) != ST_OPERATOR)
    return 0;
  if (!dtype || DTY(dtype) != TY_DERIVED)
    return 0;
  found_nopass = found_pass = 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (CLASSG(mem) && VTABLEG(mem) && BINDG(mem) &&
        strcmp(SYMNAME(sptr), SYMNAME(BINDG(mem))) == 0) {
      if (NOPASSG(mem))
        found_nopass = mem;
      else
        found_pass = mem;
    }
  }

  tag = DTY(dtype + 3);
  if (PARENTG(tag)) {
    return generic_tbp_has_pass_and_nopass(DTYPEG(PARENTG(tag)), sptr);
  }

  return (flag) ? found_nopass : found_pass;
}

int
get_specific_member(int dtype, int sptr)
{

  /* Similar to get_generic_member() except it returns the member of
   * the specific type bound procedure. This is needed when a user
   * operator has the same name (except for the leading and trailing
   * dot `.') as a specific type bound procedure.
   */

  int tag, mem, mem2;

  if (!dtype || DTY(dtype) != TY_DERIVED)
    return 0;
  mem2 = 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (CLASSG(mem) && VTABLEG(mem) && BINDG(mem) &&
        STYPEG(BINDG(mem)) != ST_OPERATOR &&
        STYPEG(BINDG(mem)) != ST_USERGENERIC &&
        strcmp(SYMNAME(sptr), SYMNAME(BINDG(mem))) == 0) {
      return mem;
    }
  }

  tag = DTY(dtype + 3);
  if (PARENTG(tag)) {
    mem = get_specific_member(DTYPEG(PARENTG(tag)), sptr);
  }

  return (mem > NOSYM) ? mem : 0;
}

static int
find_by_name_stype_arg(char *symname, int stype, int scope, int dtype, int inv,
                       int exact)
{
  int hash, hptr, len;
  int dpdsc, dtype2, arg;
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if (STYPEG(hptr) == stype && strcmp(SYMNAME(hptr), symname) == 0) {
      if (scope == 0 || scope == SCOPEG(hptr)) {
        if (!inv)
          return hptr;
        dpdsc = DPDSCG(hptr);
        arg = aux.dpdsc_base[dpdsc + (inv - 1)];
        dtype2 = DTYPEG(arg);
        if (eq_dtype2(dtype2, dtype, !exact && CLASSG(arg)) ||
            eq_dtype2(dtype, dtype2, !exact && CLASSG(arg)))
          return hptr;
      }
    }
  }
  return 0;
}

/** \brief For type bound procedures, find the implementation for the
 * type bound procedure binding name in dtype.
 *
 * If flag is set, then we check to see if we're accessing a PRIVATE
 * type bound procedure. If so, we issue an error message.
 *
 * \param dtype is the derived type record that we are searching.
 * \param orig_sptr is the symbol table pointer of the binding name of the
 *        type bound procedure to look up.
 * \param flag is set to check for accessing a PRIVATE type bound procedure.
 * \param memout if set, the function will store the type bound procedure
 *        symbol table pointer in this pointer argument.
 *
 * \return a symbol table pointer to the type bound procedure implementation;
 *         otherwise 0 (if not found).
 */
int
get_implementation(int dtype, int orig_sptr, int flag, int *memout)
{
  int sptr = orig_sptr;
  int mem, tag;
  int imp = 0, bind;
  int rslt = 0;
  int invobj = 0;
  const char *tbp_name, *suffix;
  int tbp_name_len;
  int my_mem;
  int inherited_imp = 0;
  int scope;
  SPTR tag_scope;
  static bool force_resolve_once = false;

  if (!memout)
    memout = &my_mem;
  *memout = 0;

  if (dtype > 0 && DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  if (dtype <= 0 || DTY(dtype) != TY_DERIVED)
    return 0;

  inherited_imp = 0;
  sptr = get_tbp(orig_sptr);
  tbp_name = SYMNAME(sptr);
  tbp_name_len = strlen(tbp_name);
  if ((suffix = strstr(tbp_name, "$tbp")))
    tbp_name_len = suffix - tbp_name;
  tag = DTY(dtype + 3);

  for(tag_scope = SCOPEG(tag); STYPEG(tag_scope) == ST_ALIAS;) {
    tag_scope = SYMLKG(tag_scope);
  }
  if (sem.which_pass > 0 && STYPEG(tag_scope) != ST_MODULE &&
      !force_resolve_once) {
    /* We have a derived type that's defined inside a procedure. We
     * need to force a resolution on the type bound procedures since they
     * do not normally get resolved until we see an ENDMODULE statement
     * (which would not necessarily apply in this case).
     *
     * Because queue_tbp() might also call get_implementation(), we need to
     * use the "force_resolve_once" variable to make sure queue_tbp() is 
     * only called once with TBP_FORCE_RESOLVED.
     */
    force_resolve_once = true;
    queue_tbp(0, 0, 0, 0, TBP_FORCE_RESOLVE);
    force_resolve_once = false;
  }

  if (PARENTG(tag)) {
    imp = get_implementation(DTYPEG(PARENTG(tag)), sptr, 0, memout);
    if (imp) {
      bind = BINDG(*memout);
      invobj = INVOBJG(bind);
      inherited_imp = imp;
    }
  }
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    bind = BINDG(mem);
    if (bind > NOSYM && CCSYMG(mem) && CLASSG(mem) && VTABLEG(mem)) {
      const char *bind_name = SYMNAME(bind);
      int bind_name_len = strlen(bind_name);
      if ((suffix = strstr(bind_name, "$tbp")))
        bind_name_len = suffix - bind_name;
      if (bind_name_len == tbp_name_len &&
          memcmp(tbp_name, bind_name, bind_name_len) == 0) {
        imp = IFACEG(mem) ? IFACEG(mem) : VTABLEG(mem);
        invobj = INVOBJG(bind);
        *memout = mem;
        break;
      }
    }
  }

  if (!imp)
    return 0;
  
  /*for submod, it needs to make comparison again with gbl.currsub, as
    submod's scope is 0 which doesn't equal to the proc defined in 
    parent mod with scope to it's parent mod
  */
  if (flag && PRIVATEG(*memout) && SCOPEG(*memout) != gbl.currmod &&
      SCOPEG(*memout) != SCOPEG(gbl.currsub)) {
    error(155, 3, gbl.lineno, "cannot access PRIVATE type bound procedure",
          SYMNAME(orig_sptr));
  }

  if (!invobj && !NOPASSG(*memout)) {
    invobj = 1;
    bind = BINDG(*memout);
    if (STYPEG(bind) == ST_PROC)
      INVOBJP(bind, invobj);
  }
  scope = DTY(dtype) == TY_DERIVED ? SCOPEG(DTY(dtype + 3)) : 0;

  if (scope != SCOPEG(SCOPEG(imp)) && imp != inherited_imp) {
/* If imp is declared in same scoping unit as dtype, don't
 * perform the additional checks below.
 */
    /* Perform the additional checks below if the dtype's
     * implementation is not inherited from a parent type and its
     * defined in another scope.
     */
    rslt =
        find_by_name_stype_arg(SYMNAME(imp), ST_PROC, scope, dtype, invobj, 1);
    if (!rslt) {
      rslt = find_by_name_stype_arg(SYMNAME(imp), ST_PROC, scope, dtype, invobj,
                                    0);
    }

    if (!rslt) {
      rslt = find_by_name_stype_arg(SYMNAME(imp), ST_PROC, 0, dtype, invobj, 1);
    }

    if (!rslt) {
      rslt = find_by_name_stype_arg(SYMNAME(imp), ST_PROC, 0, dtype, invobj, 0);
    }

    if (!rslt) {
      rslt = find_by_name_stype_arg(SYMNAME(imp), ST_PROC, 0, 0, invobj, 0);
    }

    if (!rslt) {
      rslt = find_by_name_stype_arg(SYMNAME(imp), ST_PROC, 0, 0, 0, 0);
    }
  }

  if (!rslt) {
    rslt = imp;
  }

  if (rslt != VTABLEG(mem)) {
    VTABLEP(mem, rslt);
    if (DTYPEG(rslt))
      DTYPEP(mem, DTYPEG(rslt));
  }

  return rslt;
}

/*---------------------------------------------------------------------*/

/** \brief Write ILMs to call a subroutine.
    \param stktop function to call
    \param list   arguments to pass to function
    \param flag   set if called from a generic resolution routine
 */
void
subr_call2(SST *stktop, ITEM *list, int flag)
{
  int sptr, sptr1, stype;
  ITEM *itemp;
  int count, alt_ret;
  int dum, i, ii, check_generic;
  int ast;
  int argt;
  SST *sp;
  int param_dummy;
  char *kwd_str; /* where make_kwd_str saves the string */
  int tbp_mem;

  tbp_mem = 0;
  ast = 0; /* initialize just in case error occurs */
  kwd_str = NULL;
  sptr = SST_SYMG(stktop);
  if (sptr > 0) {
    check_generic = 1;
  } else {
    sptr = -sptr;
    SST_SYMP(stktop, sptr);
    check_generic = 0;
  }
try_next_sptr:
  stype = STYPEG(sptr);
  if (stype == ST_ALIAS) {
    sptr = SYMLKG(sptr);
    stype = STYPEG(sptr);
  }
  get_next_hash_link(sptr, 0);
try_next_hash_link:

  init_byval();
  if (stype != ST_PROC) {
    if (stype == ST_PD) {
      ref_pd_subr(stktop, list);
      return;
    }
    if (stype == ST_USERGENERIC && check_generic) {
      if (CLASSG(sptr)) {
        sptr = generic_tbp_call(sptr, stktop, list, 0);
        goto do_call;
      }
      generic_call(sptr, stktop, list, 0);
      return;
    }
    if (stype == ST_INTRIN) {
      /* class subroutine intrinsic? */
      switch (INTASTG(sptr)) {
      case I_C_F_POINTER:
      case I_C_F_PROCPOINTER:
        ref_intrin_subr(stktop, list);
        return;
      default:
        break;
      }
    }
    if (IS_INTRINSIC(stype)) {
      /* check if intrinsic is frozen */
      if ((sptr = newsym(sptr)) == 0) {
        ast = 0;
        goto exit_;
      }
    } else if (stype == ST_IDENT) {
      if (SCG(sptr) != SC_LOCAL) {
        if (SCG(sptr) == SC_DUMMY) {
          /*
           *  this is a dummy procedure call, but may be a user
           *  error.
           */
          error(125, 1, gbl.lineno, SYMNAME(sptr), CNULL);
        } else if (SCG(sptr) != SC_NONE) {
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- attempt to CALL a non-SUBROUTINE");
          ast = 0;
          goto exit_;
        } else
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- attempt to CALL a FUNCTION");
      }
    } else if (stype == ST_ENTRY) {
      int sptr2;
      if (GSAMEG(sptr) && check_generic) {
        if (CLASSG(sptr)) {
          sptr = generic_tbp_call(sptr, stktop, list, 0);
          goto do_call;
        }
        generic_call(GSAMEG(sptr), stktop, list, 0);
        return;
      }
      if (flg.recursive || RECURG(sptr)) {
        if (gbl.rutype != RU_SUBR) {
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- attempt to CALL a non-SUBROUTINE");
          ast = 0;
          goto exit_;
        }
        if (DPDSCG(sptr))
          kwd_str = make_kwd_str(sptr);
        goto do_call;
      }
      if (sptr != gbl.currsub) {
        sptr2 = findByNameStypeScope(SYMNAME(sptr), ST_PROC, 0);
        if (sptr2) {
          sptr = sptr2;
          goto try_next_sptr;
        }
      }
      error(88, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      ast = 0;
      goto exit_;
    } else if (stype != ST_UNKNOWN) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- attempt to CALL a non-SUBROUTINE");
      ast = 0;
      goto exit_;
    } else {
      SCP(sptr, SC_NONE); /* <var ref> could have SET storage class */
    }
    /*
     * it's okay to make the symbol a procedure
     */
    STYPEP(sptr, ST_PROC);
    DTYPEP(sptr, 0);
    if (SCG(sptr) == SC_NONE)
      SCP(sptr, SC_EXTERN);
    if (SLNKG(sptr) == 0) {
      SLNKP(sptr, aux.list[ST_PROC]);
      aux.list[ST_PROC] = sptr;
    }
  } else { /* stype == ST_PROC */
    if (GSAMEG(sptr) && check_generic) {
      if (CLASSG(sptr)) {
        sptr = generic_tbp_call(sptr, stktop, list, 0);
        goto do_call;
      }
      generic_call(GSAMEG(sptr), stktop, list, 0);
      return;
    }
    if (DTYPEG(sptr) != 0 && (DCLDG(sptr) || FUNCG(sptr)))
      /* sptr is a function */
      error(84, 3, gbl.lineno, SYMNAME(sptr), "- attempt to CALL a FUNCTION");
    else
      /* first occurrence could have been
       * in an EXTERNAL statement in which case its dtype
       * was set due to the implicit handling.
       */
      DTYPEP(sptr, 0);
    if (DPDSCG(sptr))
      kwd_str = make_kwd_str(sptr);
    if (STYPEG(sptr) == ST_PROC && SLNKG(sptr) == 0) {
      SLNKP(sptr, aux.list[ST_PROC]);
      aux.list[ST_PROC] = sptr;
    }
  }

do_call:
  if (flg.xref)
    xrefput(sptr, 'r');

  alt_ret = 0;
  count_actuals(list);
  count = carg.nent;

  if (CLASSG(sptr)) {
    int sptr2;
    ast = SST_ASTG(stktop);
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr1 = memsym_of_ast(ast);
      sptr2 = pass_sym_of_ast(ast);
      if (STYPEG(BINDG(sptr1)) != ST_USERGENERIC) {
        sptr = BINDG(sptr1);
      } else {
        /* Replace the generic type bound procedure with the specific
         * type bound procedure.
         */
        int mem, dtype;
        dtype = DTYPEG(sptr2);
        if (DTY(dtype) == TY_ARRAY)
          dtype = DTY(dtype + 1);

        if (get_implementation(dtype, sptr, 0, &mem) == 0) {
          dtype = TBPLNKG(sptr);
        }

        if (get_implementation(dtype, sptr, 0, &mem) == 0) {
          char *name_cpy, *name;
          name_cpy = getitem(0, strlen(SYMNAME(sptr1)) + 1);
          strcpy(name_cpy, SYMNAME(sptr1));
          name = strchr(name_cpy, '$');
          if (name)
            *name = '\0';
          error(155, 3, gbl.lineno,
                "Could not resolve generic type bound "
                "procedure",
                name_cpy);
          sptr1 = 0;
          break;
        }
        ast = replace_memsym_of_ast(ast, mem);
        SST_ASTP(stktop, ast);
        sptr = BINDG(mem);
        sptr1 = mem;
      }
      break;
    default:
      if (check_generic && CLASSG(sptr) && list != ITEM_END &&
          SST_DTYPEG(list->t.stkp) &&
          !tk_match_arg(TBPLNKG(sptr), SST_DTYPEG(list->t.stkp), FALSE)) {
        /* FS20530: this handles the case where there is a TBP bind name and a
         * user
         * generic with the same name and sptr points to the TBP when what is
         * needed
         * is one of the generic implementations.
         */
        sptr1 = SST_SYMG(stktop);
        generic_call(sptr, stktop, list, 0);
        if (sptr1 != SST_SYMG(stktop)) {
          return;
        }
      }
      SST_SYMP(stktop, sptr1);
      sptr1 = 0;
    }

    if (sptr1 && (INVOBJG(sptr) || NOPASSG(sptr1))) {
      int imp, dty2;
      int dty, basedt, basedt2;
      int invobj, invobj2;
      int i;
      ITEM *itemp;

      dty = TBPLNKG(sptr);
      if (dty) {
        if (DTY(dty) == TY_ARRAY)
          basedt = DTY(dty + 1);
        else
          basedt = dty;
        imp = get_implementation(DTYPEG(sptr2), sptr, 0, NULL);
        if (imp) {
          invobj = get_tbp_argno(sptr, DTYPEG(sptr2));
        } else {
          invobj = get_tbp_argno(sptr, basedt);
        }
        if (invobj) {
          for (sp = 0, i = 1, itemp = list; i <= invobj && itemp != ITEM_END;
               ++i) {
            sp = itemp->t.stkp;
            itemp = itemp->next;
          }
          sptr1 = 0;
          if (SST_IDG(sp) == S_LVALUE || SST_IDG(sp) == S_EXPR)
            sptr1 = SST_LSYMG(sp);
          else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
            sptr1 = SST_SYMG(sp);
          else if (SST_IDG(sp) == S_SCONST) {
            (void)mkarg(sp, &dum);
            sptr1 = SST_SYMG(sp);
          }
          dty2 = DTYPEG(sptr1);
          if (DTY(dty2) == TY_ARRAY)
            basedt2 = DTY(dty2 + 1);
          else
            basedt2 = dty2;
          if (0 && !eq_dtype2(basedt, basedt2, 1)) { /* TBD */
            error(155, 3, gbl.lineno,
                  "Incompatible PASS argument in type "
                  "bound procedure call",
                  CNULL);
          } else {
            imp = get_implementation(basedt2, sptr, !flag, NULL);
            if (!imp) {
              error(155, 3, gbl.lineno,
                    "Incompatible PASS argument in type "
                    "bound procedure call",
                    CNULL);
            }
            invobj2 = get_tbp_argno(sptr, basedt2);
            if (invobj != invobj2) {
              error(155, 4, gbl.lineno,
                    "Type bound procedure "
                    "PASS arguments must have the same "
                    "name and position as PASS arguments in the overloaded "
                    "type bound procedure",
                    SYMNAME(imp));
            }

            set_pass_objects(invobj - 1, sptr1);

            CLASSP(imp, 1);
            sptr = imp;

            tbp_mem = ast;

            if (kwd_str)
              FREE(kwd_str);
            if (DPDSCG(sptr)) {
              kwd_str = make_kwd_str(sptr);
            }
          }
        } else if (NOPASSG(sptr1)) {
          sptr = sym_of_ast(ast);
          imp = get_implementation(basedt, BINDG(sptr1), !flag, NULL);
          sptr = imp;
          tbp_mem = ast;
          if (kwd_str)
            FREE(kwd_str);
          if (DPDSCG(sptr))
            kwd_str = make_kwd_str(sptr);
        }
      }
    }
  }

  if (!tbp_mem && sptr > NOSYM && !IS_PROC_DUMMYG(sptr) && TBPLNKG(sptr)) {
    int sym;
    do {
      sym = get_next_hash_link(sptr, 1);
    } while (sym && test_scope(SCOPEG(sym)) < 0);
    if (sym) {
      sptr = sym;
      if (kwd_str) {
        FREE(kwd_str);
        kwd_str = NULL;
      }
      goto try_next_hash_link;
    }
    if (!kwd_str) {
      for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
        sp = itemp->t.stkp;
        if (SST_IDG(sp) == S_KEYWORD) {
          kwd_str = make_kwd_str(sptr);
          break;
        }
      }
    }
  }

  /*
   * loop through the argument list to evaluate all of the arguments and
   * saving their values (ILM pointers);
   */
  if (kwd_str) {
    if (check_arguments(sptr, count, list, kwd_str))
      goto exit_;
    count_formals(sptr);
    count = carg.nent;
    argt = mk_argt(carg.nargt); /* mk_argt stuffs away count */
    ii = 0;
    for (i = 0; i < count; i++) {
      sp = ARG_STK(i);
      if (sp) {
        /* add to ARGT list, handling derived type arguments as
         * special case.
         */
        sptr1 = get_sym_from_sst_if_available(sp);
        {
          param_dummy = inc_dummy_param(sptr);

          if (!is_iso_cloc(SST_ASTG(sp)) && (A_TYPEG(SST_ASTG(sp)) != A_FUNC) &&
              is_iso_cptr(A_DTYPEG(SST_ASTG(sp)))) {
            /* rewrite iso cptr references,
               do not rewrite functions returning iso_cptr,
               do not rewrite iso c_loc
             */

            ARGT_ARG(argt, ii) = rewrite_cptr_references(SST_ASTG(sp));
          } else if (get_byval(sptr, param_dummy)
                    && PASSBYVALG(param_dummy)
                    && OPTARGG(param_dummy)) {
            int assn = sem_tempify(sp);
            (void)add_stmt(assn);
            SST_ASTP(sp, A_DESTG(assn));
            byvalue_ref_arg(sp, &dum, OP_REF, sptr);
            ARGT_ARG(argt, ii) = SST_ASTG(sp);
          } else if (pass_char_no_len(sptr, param_dummy)) {
            byvalue_ref_arg(sp, &dum, OP_REF, sptr);
            ARGT_ARG(argt, ii) = SST_ASTG(sp);
          } else if (INTENTG(param_dummy) == INTENT_IN &&
                     POINTERG(param_dummy) && !is_ptr_arg(sp)) {
            /* F2008: pass non-pointer actual arg for an
             *        INTENT(IN), POINTER formal arg */
            ARGT_ARG(argt, ii) = gen_and_assoc_tmp_ptr(sp, sem.last_std);
          } else {
            /* byval arguments done in lowerilm.c for  subroutines */
            ARGT_ARG(argt, ii) = ARG_AST(i);
          }
          ii++;
          if (sptr1 && STYPEG(sptr1) == ST_PROC && DPDSCG(sptr1) &&
              SLNKG(sptr1) == 0) {
            SLNKP(sptr1, aux.list[ST_PROC]);
            aux.list[ST_PROC] = sptr1;
          }
        }
      } else {
        int npad;
        for (npad = ARG_AST(i); npad > 0; npad--) {
          ARGT_ARG(argt, ii) = astb.ptr0;
          ii++;
        }
      }
    }
    if (tbp_mem) {
      int mem = memsym_of_ast(tbp_mem);
      if (STYPEG(mem) == ST_MEMBER && !strstr(SYMNAME(sptr), "$tbp")) {
        VTABLEP(mem, sptr);
      }
    }
    ast = mk_func_node(A_CALL, (tbp_mem) ? tbp_mem : mk_id(sptr), carg.nargt,
                       argt);
    goto exit_;
  }
  argt = mk_argt(carg.nargt); /* mk_argt stuffs away count */
  if (tbp_mem) {
    int mem = memsym_of_ast(tbp_mem);
    if (STYPEG(mem) == ST_MEMBER && !strstr(SYMNAME(sptr), "$tbp")) {
      VTABLEP(mem, sptr);
    }
  }
  ast =
      mk_func_node(A_CALL, (tbp_mem) ? tbp_mem : mk_id(sptr), carg.nargt, argt);
  ii = count = 0;

  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_KEYWORD) {
      /* form is <ident> = <expression> */
      error(79, 3, gbl.lineno, scn.id.name + SST_CVALG(itemp->t.stkp), CNULL);
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    /* check arguments and add to ARGT list, handling derived type
     * arguments as special case
     */
    sptr1 = 0;
    if (SST_IDG(sp) == S_LVALUE)
      sptr1 = SST_LSYMG(sp);
    else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
      sptr1 = SST_SYMG(sp);
    else if (SST_IDG(sp) == S_SCONST) {
      (void)mkarg(sp, &dum);
      sptr1 = SST_SYMG(sp);
    }
    {

      /* get_byvalue parameter processing is handled in lowerilm.c for
         subroutine calls.
       */
      param_dummy = inc_dummy_param(sptr);

      if (pass_char_no_len(sptr, param_dummy)) {
        itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_REF, sptr);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);

      } else {
        itemp->t.sptr = chkarg(sp, &dum);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      }
      ii++;

      if (sptr1 && STYPEG(sptr1) == ST_PROC && DPDSCG(sptr1) &&
          SLNKG(sptr1) == 0) {
        SLNKP(sptr1, aux.list[ST_PROC]);
        aux.list[ST_PROC] = sptr1;
      }
    }
    /*
     * a negative value returned by mkarg is a negated alternate
     * return label
     */
    if (itemp->t.sptr <= 0)
      alt_ret++;
  }

exit_:
  SST_ASTP(stktop, ast);

  if (kwd_str)
    FREE(kwd_str);
}

void
subr_call(SST *stktop, ITEM *list)
{
  subr_call2(stktop, list, 0);
}

static void
fix_proc_pointer_call(SST *stktop, ITEM **list)
{
  /* Fix up pointer procedure call. If it's missing the pass object in the
   * arg list, add it. Also resolve the procedure pointer's iface if it has
   * not yet been resolved.
   */

  int func, pass_sym;
  int paramct, dpdsc, iface, ast, i;
  int dtype, dtproc;
  SST *e1;
  ITEM *itemp, *itemp2;
  ast = SST_ASTG(stktop);
  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    func = memsym_of_ast(ast);
    pass_sym = pass_sym_of_ast(ast);
    proc_arginfo(func, &paramct, &dpdsc, &iface);
    break;
  default:
    return;
  }
  if (STYPEG(iface) != ST_PROC) {
    iface = findByNameStypeScope(SYMNAME(iface), ST_PROC, 0);
    if (iface) {
      proc_arginfo(iface, &paramct, &dpdsc, NULL);
      if (is_procedure_ptr(func)) {
        dtype = DTYPEG(func);
        dtproc = DTY(dtype + 1);
        DTY(dtproc + 3) = paramct;
        DTY(dtproc + 4) = dpdsc;
        DTY(dtproc + 2) = iface;
        DTY(dtproc + 1) = DTYPEG(iface);
      }
    } else
      return;
  }

  if (NOPASSG(func) || paramct <= 0)
    return;

  for (i = 0, itemp = *list; itemp != ITEM_END; itemp = itemp->next) {
    ++i;
  }

  if (*list != ITEM_END && (paramct - 1) <= i)
    return;

  if (!PASSG(func)) {
    /* check first arg */
    if (*list == ITEM_END) {
    insert_first_arg:
      e1 = (SST *)getitem(0, sizeof(SST));
      SST_IDP(e1, S_EXPR);
      SST_SYMP(e1, pass_sym);
      SST_ASTP(e1, check_member(ast, mk_id(pass_sym)));

      itemp = (ITEM *)getitem(0, sizeof(ITEM));
      itemp->t.stkp = e1;
      itemp->next = ITEM_END;
      *list = itemp;
    }
  } else {
    int pass_pos = find_dummy_position(iface, PASSG(func));
    if (pass_pos == 1 && *list == ITEM_END)
      goto insert_first_arg;
    if (pass_pos <= 1)
      return;
    for (i = 0, itemp = *list; itemp != ITEM_END; itemp = itemp->next) {
      e1 = itemp->t.stkp;
      if (i == pass_pos - 2) {
        e1 = (SST *)getitem(0, sizeof(SST));
        SST_IDP(e1, S_EXPR);
        SST_SYMP(e1, pass_sym);
        SST_ASTP(e1, check_member(ast, mk_id(pass_sym)));
        itemp2 = (ITEM *)getitem(0, sizeof(ITEM));
        itemp2->t.stkp = e1;
        itemp2->next = itemp->next;
        itemp->next = itemp2;
        break;
      }
      ++i;
    }
  }
}

void
ptrsubr_call(SST *stktop, ITEM *list)
{
  int sptr, sptr1;
  int callee;
  ITEM *itemp;
  int count, alt_ret;
  int dum, i, ii;
  int dtproc, iface, paramct, dpdsc;
  int dtype;
  int ast;
  int argt;
  SST *sp;
  int param_dummy;
  char *kwd_str; /* where make_kwd_str saves the string */
  int pass_pos;

  fix_proc_pointer_call(stktop, &list);
  ast = 0; /* initialize just in case error occurs */
  kwd_str = NULL;
  pass_pos = -1;
  if (SST_IDG(stktop) != S_LVALUE) {
    sptr = SST_SYMG(stktop);
    callee = mk_id(sptr);
  } else {
    sptr = SST_LSYMG(stktop);
    if (!is_procedure_ptr(sptr))
      /* error must have occurred */
      goto exit_;
    callee = SST_ASTG(stktop);
  }
  if (FUNCG(sptr))
    /* sptr is a function */
    error(84, 3, gbl.lineno, SYMNAME(sptr), "- attempt to CALL a FUNCTION");
  dtype = DTYPEG(sptr);
#if DEBUG
  assert(DTY(dtype) == TY_PTR, "ptrsubr_call, expected TY_PTR dtype", sptr, 4);
#endif
  dtproc = DTY(dtype + 1);
#if DEBUG
  assert(DTY(dtproc) == TY_PROC, "ptrsubr_call, expected TY_PROC dtype", sptr,
         4);
#endif
  dtype = DTY(dtproc + 1);
  iface = DTY(dtproc + 2);
  paramct = DTY(dtproc + 3);
  dpdsc = DTY(dtproc + 4);
  if (iface != sptr && !paramct) {
    proc_arginfo(iface, &paramct, &dpdsc, NULL);
    DTY(dtproc + 3) = paramct;
    DTY(dtproc + 4) = dpdsc;
  }
  init_byval();
  if (dpdsc)
    kwd_str = make_keyword_str(paramct, dpdsc);

  if (flg.xref)
    xrefput(sptr, 'r');

  alt_ret = 0;
  count_actuals(list);
  count = carg.nent;

  /*
   * loop through the argument list to evaluate all of the arguments and
   * saving their values (ILM pointers);
   */
  if (kwd_str) {
    if (chk_arguments(sptr, count, list, kwd_str, paramct, dpdsc, callee,
                      &pass_pos))
      goto exit_;
    count_formal_args(paramct, dpdsc);
    count = carg.nent;
    argt = mk_argt(carg.nargt); /* mk_argt stuffs away count */
    ii = 0;
    for (i = 0; i < count; i++) {
      sp = ARG_STK(i);
      if (sp) {
        /* add to ARGT list, handling derived type arguments as
         * special case.
         */
        sptr1 = 0;
        if (SST_IDG(sp) == S_LVALUE)
          sptr1 = SST_LSYMG(sp);
        else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
          sptr1 = SST_SYMG(sp);
        else if (SST_IDG(sp) == S_SCONST) {
          (void)mkarg(sp, &dum);
          sptr1 = SST_SYMG(sp);
        }
        {
          param_dummy = inc_dummy_param(sptr);
          if (!is_iso_cloc(SST_ASTG(sp)) && (A_TYPEG(SST_ASTG(sp)) != A_FUNC) &&
              is_iso_cptr(A_DTYPEG(SST_ASTG(sp)))) {
            /* rewrite iso cptr references,
               do not rewrite functions returning iso_cptr,
               do not rewrite iso c_loc
             */

            ARGT_ARG(argt, ii) = rewrite_cptr_references(SST_ASTG(sp));
            ii++;
          } else if (pass_char_no_len(sptr, param_dummy)) {
            byvalue_ref_arg(sp, &dum, OP_REF, sptr);
            ARGT_ARG(argt, ii) = SST_ASTG(sp);
            ii++;
          } else {
            /* byval arguments done in lowerilm.c for  subroutines */
            ARGT_ARG(argt, ii) = ARG_AST(i);
            ii++;
          }
          if (sptr1 && STYPEG(sptr1) == ST_PROC && DPDSCG(sptr1) &&
              SLNKG(sptr1) == 0) {
            SLNKP(sptr1, aux.list[ST_PROC]);
            aux.list[ST_PROC] = sptr1;
          }
        }
      } else if (i == pass_pos) {
        ARGT_ARG(argt, ii) = A_PARENTG(callee);
        ii++;
      } else {
        int npad;
        for (npad = ARG_AST(i); npad > 0; npad--) {
          ARGT_ARG(argt, ii) = astb.ptr0;
          ii++;
        }
      }
    }
    ast = mk_func_node(A_CALL, callee, carg.nargt, argt);
    goto exit_;
  }
  argt = mk_argt(carg.nargt); /* mk_argt stuffs away count */
  ast = mk_func_node(A_CALL, callee, carg.nargt, argt);
  ii = count = 0;

  for (itemp = list; itemp != ITEM_END; itemp = itemp->next) {
    sp = itemp->t.stkp;
    if (SST_IDG(sp) == S_KEYWORD) {
      /* form is <ident> = <expression> */
      error(79, 3, gbl.lineno, scn.id.name + SST_CVALG(itemp->t.stkp), CNULL);
      ARGT_ARG(argt, ii) = astb.i0;
      ii++;
      continue;
    }
    /* check arguments and add to ARGT list, handling derived type
     * arguments as special case
     */
    sptr1 = 0;
    if (SST_IDG(sp) == S_LVALUE)
      sptr1 = SST_LSYMG(sp);
    else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
      sptr1 = SST_SYMG(sp);
    else if (SST_IDG(sp) == S_SCONST) {
      (void)mkarg(sp, &dum);
      sptr1 = SST_SYMG(sp);
    }
    {
      /* get_byvalue parameter processing is handled in lowerilm.c for
         subroutine calls.
       */
      param_dummy = inc_dummy_param(sptr);
      if (pass_char_no_len(sptr, param_dummy)) {
        itemp->t.sptr = byvalue_ref_arg(sp, &dum, OP_REF, sptr);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);

      } else {
        itemp->t.sptr = chkarg(sp, &dum);
        ARGT_ARG(argt, ii) = SST_ASTG(sp);
      }
      ii++;
    }
    /*
     * a negative value returned by mkarg is a negated alternate
     * return label
     */
    if (itemp->t.sptr <= 0)
      alt_ret++;
  }

exit_:
  SST_ASTP(stktop, ast);

  if (kwd_str)
    FREE(kwd_str);
}

/*---------------------------------------------------------------------*/

/* the purpose of these ASTs is to transfer information to the
 * ACL constructors in semutil2.c.  They should be ignored by
 * by anything not involved in data initialization.
 */
static void
gen_init_intrin_call(SST *stkp, int pdsym, int argt_count, int dtype,
                     int elemental)
{
  int argt = mk_argt(argt_count); /* space for arguments */
  int func_ast;
  int ast;
  int i;
  int dtyper = dtype;
  SST *arg1;
  int arg1dtype;
  int dum;
  SST *s;

  for (i = 0; i < argt_count; i++) {
    s = (ARG_STK(i));
    if (!s) {
      ARGT_ARG(argt, i) = astb.i0;
    } else if (SST_IDG(s) == S_IDENT || SST_IDG(s) == S_ACONST) {
      SST_ASTP(s, 0);
      (void)mkarg(s, &dum);
      XFR_ARGAST(i);
      ARGT_ARG(argt, i) = ARG_AST(i);
    } else if (ARG_AST(i)) {
      ARGT_ARG(argt, i) = ARG_AST(i);
    }
  }
  func_ast = mk_id(pdsym);

  ast = mk_func_node(A_INTR, func_ast, argt_count, argt);
  A_DTYPEP(ast, dtype);

  if (elemental) {
    arg1 = ARG_STK(0);
    arg1dtype = SST_DTYPEG(arg1);
    if (DTY(arg1dtype) == TY_ARRAY) {
      dtyper = mk_array_type(arg1dtype, dtype);
      A_DTYPEP(ast, dtyper);
      A_SHAPEP(ast, SST_SHAPEG(arg1));
    }
  }
  SST_DTYPEP(stkp, dtyper);

  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stkp, S_EXPR);
  SST_ASTP(stkp, ast);
  A_OPTYPEP(ast, INTASTG(pdsym));
}

/*
 * Generate a symbol for newer specifics of older generic intrinsics, i.e.,
 * those not
 * defined in syminidf.h
 */
static int
gen_newer_intrin(int sptrgenr, int dtype)
{
  char *intrin_nmptr = SYMNAME(sptrgenr);
  char nmptr[STANDARD_MAXIDLEN + 3] = ".";
  int sptr = 0;

  if (strcmp(intrin_nmptr, "acos") == 0 || strcmp(intrin_nmptr, "asin") == 0 ||
      strcmp(intrin_nmptr, "atan") == 0 || strcmp(intrin_nmptr, "cosh") == 0 ||
      strcmp(intrin_nmptr, "sinh") == 0 || strcmp(intrin_nmptr, "tanh") == 0 ||
      strcmp(intrin_nmptr, "tan") == 0) {
    if (DT_ISCMPLX(dtype)) {
      switch (DTY(dtype)) {
      case TY_DCMPLX:
        strcat(nmptr, "cd");
        break;
      case TY_CMPLX:
        strcat(nmptr, "c");
        break;
      default:
        interr(
            "gen_newer_intrin: unknown type for inverse trigonmetric intrinsic",
            DTY(dtype), 2);
        return 0;
      }
      strcat(nmptr, intrin_nmptr);

      sptr = getsymbol(nmptr);
      STYPEP(sptr, ST_INTRIN);
      DTYPEP(sptr, 0);
      SYMLKP(sptr, sptrgenr);
      PNMPTRP(sptr, PNMPTRG(GREALG(sptrgenr)));
      PARAMCTP(sptr, 1);
      ILMP(sptr, ILMG(GREALG(sptrgenr)));
      ARRAYFP(sptr, ARRAYFG(GREALG(sptrgenr)));
      ARGTYPP(sptr, dtype);
      INTTYPP(sptr, dtype);
      INTASTP(sptr, NEW_INTRIN);

      switch (DTY(dtype)) {
      case TY_DCMPLX:
        GDCMPLXP(sptrgenr, sptr);
        break;
      case TY_CMPLX:
        GCMPLXP(sptrgenr, sptr);
        break;
      }
    }
    return sptr;
  }

  return 0;
}

static int
cmp_mod_scope(SPTR sptr)
{
  SPTR scope1, scope2;

  scope1 = stb.curr_scope;
  if (IS_PROC(STYPEG(scope1))) {
    scope1 = SCOPEG(scope1);
  }
  scope2 = SCOPEG(sptr);
  return scope1 == scope2;
}

/** \brief Handle Generic and Intrinsic function calls.
 */
int
ref_intrin(SST *stktop, ITEM *list)
{
  int sptr, fsptr, sptre, dtype, dtype1, argtyp, paramct;
  int f_dt, ddt;
  int opc, count, const_cnt;
  ITEM *ip1;
  SST *sp;
  LOGICAL frozen;
  int ast;
  int argt;
  int i;
  int intast;
  int shaper;
  int func_ast;
  int argdtype;
  int dtyper;
  int func_type;
  int dum;
  int dt_cast_word;
  int hpf_sym;
  int tmp, tmp_ast;
  FtnRtlEnum rtlRtn;
  int intrin; /* one of the I_* constants */

  dtyper = 0;
  dtype1 = 0;
  sptr = 0; /* for min and max character */
  SST_CVLENP(stktop, 0);
  sptre = SST_SYMG(stktop);
  if (STYPEG(sptre) == ST_INTRIN) {
    SPTR sptr2 = findByNameStypeScope(SYMNAME(sptre), ST_ALIAS, 0);
    if (sptr2 > NOSYM && SYMLKG(sptr2) == sptre && PRIVATEG(sptr2) &&
        (!IN_MODULE || cmp_mod_scope(sptr2))) {
      error(1015, 3, gbl.lineno, SYMNAME(sptr2), NULL);
    }
  }

  if (sptre >= stb.firstusym)
    return generic_func(sptre, stktop, list);

  frozen = EXPSTG(sptre);
  if (list == ITEM_END)
    goto intrinsic_error;
  /*
   * Count number of arguments without type changing arguments in case
   * we need to recover by assuming reference is to an external function.
   */
  count = 0;
  for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
    count++;
    switch (SST_IDG(ip1->t.stkp)) {
    case S_TRIPLE:
      goto intrinsic_error;
    default:
      break;
    }
  }
  /* position the arguments per the keyword argument string. note
   * that the number of arguments processed by get_kwd_args is
   *     max(actual arg count, number of 'non-variable' arguments).
   */
  i = KWDCNTG(sptre);
  if (count > i)
    i = count;
  if (get_kwd_args(list, i, KWDARGSTR(sptre)))
    goto intrinsic_error;

  intrin = INTASTG(sptre);
  dt_cast_word = 0;
  if (STYPEG(sptre) == ST_GENERIC) {
    /*
     * f2003 says that a boz literal can appear as an argument to
     * the real, dble, cmplx, and dcmplx intrinsics and its value
     * is used as the respective internal respresentation
     */
    switch (intrin) {
    case I_DBLE:
    case I_DCMPLX:
      dt_cast_word = DT_DBLE;
      break;
    case I_IAND:
      sem.mpaccatomic.rmw_op = AOP_AND;
      break;
    case I_IOR:
      sem.mpaccatomic.rmw_op = AOP_OR;
      break;
    case I_IEOR:
      sem.mpaccatomic.rmw_op = AOP_XOR;
      break;
    case I_MIN:
      sem.mpaccatomic.rmw_op = AOP_MIN;
      break;
    case I_MAX:
      sem.mpaccatomic.rmw_op = AOP_MAX;
      break;
    }
  }
  sp = ARG_STK(0); /* Save 1st arg's semantic stack pointer */
  dtype1 = 0;
  for (i = 0; i < count; i++) {
    sp = ARG_STK(i);
    argdtype = SST_DTYPEG(sp);
    if (argdtype == DT_WORD || argdtype == DT_DWORD) {
      if (dt_cast_word) {
        cngtyp(sp, dt_cast_word);
        argdtype = SST_DTYPEG(sp);
      } else if (argdtype == DT_WORD) {
      }
    }
    if (!dtype1) {
      f_dt = dtype1 = argdtype; /* Save 1st arg's data type */
      if (DTY(argdtype) == TY_ARRAY)
        break;
    } else {
      /* check rest of args to see if they might be array. */
      /* assert.  haven't seen an array argument yet. */
      if (DTY(argdtype) == TY_ARRAY) {
        f_dt = dtype1 = argdtype; /* Save data type */
        break;
      }
    }
  }

  if (STYPEG(sptre) == ST_GENERIC) {
    if (SST_ISNONDECC(sp)) {
      cngtyp(sp, DT_INT);
    }
    dtype = DDTG(dtype1);
    /* apply the KIND argument if applicable */
    /* determine specific intrinsic name from data type of first argument */
    switch (DTY(dtype)) {
    case TY_BLOG:
    case TY_BINT:
      sptr = GINTG(sptre);
      if (ARGTYPG(sptr) == INTTYPG(sptr))
        dtyper = dtype;
      break;
    case TY_SLOG:
    case TY_SINT:
      if ((sptr = GSINTG(sptre)))
        break;
      FLANG_FALLTHROUGH;
    case TY_WORD:
    case TY_LOG:
    case TY_INT:
      sptr = GINTG(sptre);
      break;
    case TY_DWORD:
    case TY_LOG8:
    case TY_INT8:
      sptr = GINT8G(sptre);
      break;
    case TY_REAL:
      sptr = GREALG(sptre);
      break;
    case TY_DBLE:
      sptr = GDBLEG(sptre);
      break;
    case TY_QUAD:
      sptr = GQUADG(sptre);
      break;
    case TY_CMPLX:
      sptr = GCMPLXG(sptre);
      break;
    case TY_DCMPLX:
      sptr = GDCMPLXG(sptre);
      break;
    case TY_QCMPLX:
      sptr = GQCMPLXG(sptre);
      break;
    case TY_CHAR:
    case TY_NCHAR:
      if ((intrin == I_MAX || intrin == I_MIN) && sem.dinit_data) {
        paramct = 12;
        argtyp = dtype1;
        /* Should really check type of next argument is char also */
        rtlRtn = intrin == I_MAX ? RTE_max : RTE_min;
        sptr = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtyper);
        gen_init_intrin_call(stktop, sptr, count, DDTG(dtype1), TRUE);
        A_OPTYPEP(SST_ASTG(stktop), intrin);
        return 1;
      }
      FLANG_FALLTHROUGH;
    default:
      sptr = 0;
      break;
    }

    if (sptr == 0) {
      sptr = gen_newer_intrin(SST_SYMG(stktop), dtype);
    }

    if (sptr <= 0)
      goto intrinsic_error;
    assert(STYPEG(sptr) == ST_INTRIN, "ref_intrin: bad intrinsic sptr", sptr,
           3);
    /*
     * determine if resolved specific has the same name as the generic;
     * If it is, must 'freeze' the specific.
     */
    if (strcmp(SYMNAME(sptr), SYMNAME(sptre)) == 0)
      EXPSTP(sptr, 1);
  } else {
    /*  SPECIFICs  */
    static int float_intr_warn = 0;
    if (XBIT(124, 0x10)) {
      /* -i8 */
      /* the intrinsic ast opcodes of the following integer*8
       * intrinsics, must appear as special cases in
       * semfunc2.c:intrinsic_as_arg() so that the correct
       * function name is selected given the integer name.
       */
      switch (intrin) {
      case I_IABS:
        sptre = intast_sym[I_KIABS];
        break;
      case I_IDIM:
        sptre = intast_sym[I_KIDIM];
        break;
      case I_IDNINT:
        sptre = intast_sym[I_KIDNNT];
        break;
      case I_ISIGN:
        sptre = intast_sym[I_KISIGN];
        break;
      case I_MAX0:
        sptre = intast_sym[I_KMAX0];
        break;
      case I_MIN0:
        sptre = intast_sym[I_KMIN0];
        break;
      case I_MAX1:
        sptre = intast_sym[I_KMAX1];
        break;
      case I_MIN1:
        sptre = intast_sym[I_KMIN1];
        break;
      }
    }
    if (XBIT(124, 0x8)) {
      /* -r8 */
      /* the intrinsic ast opcodes of the following double real/complex
       * intrinsics, must appear as special cases in
       * semfunc2.c:intrinsic_as_arg() so that the correct
       * function name is selected given the real/complex name.
       */
      switch (intrin) {
      case I_ALOG:
        sptre = intast_sym[I_DLOG];
        break;
      case I_ALOG10:
        sptre = intast_sym[I_DLOG10];
        break;
      case I_AMAX1:
        sptre = intast_sym[I_DMAX1];
        break;
      case I_AMIN1:
        sptre = intast_sym[I_DMIN1];
        break;
      case I_AMOD:
        sptre = intast_sym[I_DMOD];
        break;
      case I_CABS:
        sptre = intast_sym[I_CDABS];
        break;
      case I_CSQRT:
        sptre = intast_sym[I_CDSQRT];
        break;
      case I_CLOG:
        sptre = intast_sym[I_CDLOG];
        break;
      case I_CEXP:
        sptre = intast_sym[I_CDEXP];
        break;
      case I_CSIN:
        sptre = intast_sym[I_CDSIN];
        break;
      case I_CCOS:
        sptre = intast_sym[I_CDCOS];
        break;
      case I_FLOATI:
        if (XBIT(124, 0x80000)) {
          sptre = intast_sym[I_DFLOTI];
          if (!float_intr_warn) {
            float_intr_warn = 1;
            error(155, 2, gbl.lineno,
                  "The type of FLOAT is now double precision with -r8", CNULL);
          }
        }
        break;
      case I_FLOATJ:
        if (XBIT(124, 0x80000)) {
          sptre = intast_sym[I_DFLOTJ];
          if (!float_intr_warn) {
            float_intr_warn = 1;
            error(155, 2, gbl.lineno,
                  "The type of FLOAT is now double precision with -r8", CNULL);
          }
        }
        break;
      case I_FLOAT:
        if (XBIT(124, 0x80000)) {
          sptre = intast_sym[I_DFLOAT];
          if (!float_intr_warn) {
            float_intr_warn = 1;
            error(155, 2, gbl.lineno,
                  "The type of FLOAT is now double precision with -r8", CNULL);
          }
        }
        break;
      }
    }
    sptr = sptre;
  }

  intast = INTASTG(sptr);

  /*
   * Assertion: sptr now points to the specific intrinsic entry ST_INTRIN
   * that was either specified with a generic name or a specific name.
   * sptre EITHER points to the generic name symbol entry or the specific
   * name symbol entry (if generic and specific have same names).
   */
  dtype = INTTYPG(sptr);

  /*
   * Determine intrinsic's ILM and number and type of arguments.
   */
  if (DTY(SST_DTYPEG(sp)) == TY_ARRAY) {
    opc = ARRAYFG(sptr); /* Get ilm for Vectors */
    /* Check if vectors disallowed and not a type conversion intrinsic.
     * Vectors okay for type conversion intrinsics.
     */
    if (ILMG(sptr) == IM_LOC)
      opc = IM_LOC;
    else if (opc == 0 && ILMG(sptr) != 0)
      goto intrinsic_error;
    /* opc == 0 */
  } else
    opc = ILMG(sptr);
  argtyp = ARGTYPG(sptr);
  paramct = PARAMCTG(sptr);

  if (paramct != 12 && paramct != 11 && count > paramct) {
    goto intrinsic_error;
  }

  if (paramct == 11) { /* CMPLX/DCMPLX intrinsic */
    if (ARG_STK(1))
      /* Two arguments in reference, cause conversion of each part to
       * real/dble
       */

      dtype = dtype == DT_CMPLX  ? stb.user.dt_real
#ifdef TARGET_SUPPORTS_QUADFP
            : dtype == DT_QCMPLX ? DT_QUAD
#endif
                                 : DT_DBLE;

    else /* treat like typical type conversion intrinsic */
      paramct = 1;
  } else {
    switch (intast) {
    case I_FLOAT:
    case I_DFLOAT:
      ddt = DDTG(f_dt);
      if (ddt == DT_INT8)
        argtyp = DT_INT8;
      break;
    }
  }

  if (sem.dinit_data) {
    switch (ILMG(sptr)) {
    case IM_ICHAR:
      gen_init_intrin_call(stktop, sptr, count, stb.user.dt_int, TRUE);
      return 1;
    case IM_IISHFT:
    case IM_JISHFT:
    case IM_KISHFT:
      gen_init_intrin_call(stktop, sptr, count, stb.user.dt_int, TRUE);
      return 1;
    case IM_IMAX:
    case IM_I8MAX:
    case IM_RMAX:
    case IM_DMAX:
#ifdef IM_QMAX
    case IM_QMAX:
#endif
    case IM_IMIN:
    case IM_I8MIN:
    case IM_RMIN:
    case IM_DMIN:
#ifdef IM_QMIN
    case IM_QMIN:
#endif
      gen_init_intrin_call(stktop, sptr, count, DDTG(dtype1), TRUE);
      return 1;
    case 0:
      switch (intrin) {
      case I_DBLE:
      case I_DFLOAT:
      case I_FLOAT:
      case I_REAL:
        gen_init_intrin_call(stktop, sptre, count, DDTG(dtype1), TRUE);
        return 1;
      }
    }
  }

  /*
   * Count number of constant arguments.
   */
  const_cnt = 0;
  for (i = 0; i < count; i++)
    if (ARG_STK(i) && is_sst_const(ARG_STK(i)))
      const_cnt++;

  /*  If all arguments are constants, attempt to constant fold  */

  if (const_cnt == count) {

    INT conval, con1, con2, res[4], num1[4], num2[4];
    char ch;

    switch (opc) {
    case IM_LOC:
#ifdef I_C_ASSOCIATED
    case IM_C_ASSOC:
#endif
      goto no_const_fold;
    }

    argt = mk_argt(count); /* space for arguments */
    for (i = 0; i < count; i++) {
      sp = ARG_STK(i);
      if (opc == 0) {
        /* type conversion: for the two argument CMPLX/DCMPLX, each
         * part is converted to the real type implied by the intrinsic;
         * otherwise, the operands are converted to the result type
         * of the intrinsic.
         */
        if (XBIT(124, 0x8)) {
          /* -r8 */
          if (intast == I_SNGL) {
            dtype = DT_REAL8;
          }
        }
        cngtyp(sp, dtype);
      } else if (DTY(argtyp) == TY_CHAR && DTY(SST_DTYPEG(sp)) == TY_CHAR) {
        if (opc == IM_ICHAR && i == 0)
          dtyper = stb.user.dt_int;
      } else if ((DTY(argtyp) == TY_NCHAR || DTY(argtyp) == TY_CHAR) &&
                 DTY(SST_DTYPEG(sp)) == TY_NCHAR) {
        /*
         * if the argument is character and the expected argument is
         * character, we don't call cngtyp since we represent argtyp
         * as a character of length 1
         */
        if (opc == IM_ICHAR && i == 0)
          dtyper = stb.user.dt_int;
      } else if (i == 2 && opc == IM_NINDEX)
        cngtyp(sp, DT_LOG);
      else if (opc == IM_ICHAR) {
        if (i == 0) {
          chktyp(sp, argtyp, TRUE);
          dtyper = stb.user.dt_int;
        } else {
          dtyper = set_kind_result(sp, DT_INT, TY_INT);
          if (!dtyper) {
            goto intrinsic_error;
          }
        }
      } else
        cngtyp(sp, argtyp);
      ARGT_ARG(argt, i) = SST_ASTG(sp);
    }

    con1 = GET_CVAL_ARG(0);
    if (paramct < 12) {
      if (paramct == 11) {
        /* CMPLX/DCMPLX with 2 args: cause both to make complex # */
        num1[0] = con1;
        num1[1] = GET_CVAL_ARG(1);

        if (DTY(dtype) == TY_REAL)
          conval = getcon(num1, DT_CMPLX);
        else
          conval = getcon(num1, DT_DCMPLX);

        goto const_return;
      }
      if (opc == 0) { /* type conversion intrinsic */
        conval = GET_CVAL_ARG(0);
        if (XBIT(124, 0x8)) {
          /* -r8 */
          if (intast == I_SNGL) {
            dtype = DT_REAL8;
            goto const_return_2;
          }
        }
        goto const_return;
      }
      switch (opc) {
      case IM_IABS:
        conval = con1 >= 0 ? con1 : -con1;
        goto const_return;
      case IM_ABS:
        xfabsv(con1, &res[0]);
        conval = res[0];
        goto const_return;
      case IM_DABS:
        GET_DBLE(num1, con1);
        xdabsv(num1, res);
        goto const_getcon;
#ifdef IM_QABS
      case IM_QABS:
        GET_QUAD(num1, con1);
        xqabsv(num1, res);
        goto const_getcon;
#endif
      case IM_NINT:
        num1[0] = CONVAL2G(stb.flt0);
        if (xfcmp(con1, num1[0]) >= 0) {
          INT fv2_23 = 0x4b000000;
          if (xfcmp(con1, fv2_23) >= 0)
            xfadd(con1, CONVAL2G(stb.flt0), &res[0]);
          else
            xfadd(con1, CONVAL2G(stb.flthalf), &res[0]);
        } else {
          INT fvm2_23 = 0xcb000000;
          if (xfcmp(con1, fvm2_23) <= 0)
            xfsub(con1, CONVAL2G(stb.flt0), &res[0]);
          else
            xfsub(con1, CONVAL2G(stb.flthalf), &res[0]);
        }
        conval = cngcon(res[0], DT_REAL4, stb.user.dt_int);
        goto const_return;
      case IM_IDNINT:
        if (const_fold(OP_CMP, con1, stb.dbl0, DT_REAL8) >= 0) {
          INT dv2_52[2] = {0x43300000, 0x00000000};
          INT d2_52;
          d2_52 = getcon(dv2_52, DT_DBLE);
          if (const_fold(OP_CMP, con1, d2_52, DT_REAL8) >= 0)
            res[0] = const_fold(OP_ADD, con1, stb.dbl0, DT_REAL8);
          else
            res[0] = const_fold(OP_ADD, con1, stb.dblhalf, DT_REAL8);
        } else {
          INT dvm2_52[2] = {0xc3300000, 0x00000000};
          INT dm2_52;
          dm2_52 = getcon(dvm2_52, DT_DBLE);
          if (const_fold(OP_CMP, con1, dm2_52, DT_REAL8) <= 0)
            res[0] = const_fold(OP_SUB, con1, stb.dblhalf, DT_REAL8);
          else
            res[0] = const_fold(OP_SUB, con1, stb.dbl0, DT_REAL8);
        }
        conval = cngcon(res[0], DT_REAL8, stb.user.dt_int);
        goto const_return;
      case IM_IMAG:
      case IM_DIMAG:
#ifdef IM_QIMAG
      case IM_QIMAG:
#endif
        conval = CONVAL2G(con1);
        goto const_return;
      case IM_CONJG:
        res[0] = CONVAL1G(con1);
        con2 = CONVAL2G(con1);
        xfsub(CONVAL2G(stb.flt0), con2, &res[1]);
        goto const_getcon;
      case IM_DCONJG:
        res[0] = CONVAL1G(con1);
        con2 = CONVAL2G(con1);
        res[1] = const_fold(OP_SUB, (INT)stb.dbl0, con2, DT_REAL8);
        goto const_getcon;
#ifdef IM_QCONJG
      case IM_QCONJG:
        res[0] = CONVAL1G(con1);
        con2 = CONVAL2G(con1);
        res[1] = const_fold(OP_SUB, (INT)stb.quad0, con2, DT_QUAD);
        goto const_getcon;
#endif
#ifdef IM_DPROD
      case IM_DPROD:
        con2 = GET_CVAL_ARG(1);
        xdble(con1, num1);
        xdble(con2, num2);
        xdmul(num1, num2, res);
        goto const_getcon;
#endif
      case IM_AND8:
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num1, con1);
        GET_DBLE(num2, con2);
        and64(num1, num2, res);
        goto const_getcon;
      case IM_AND:
        con2 = GET_CVAL_ARG(1);
        conval = con1 & con2;
        goto const_return;
      case IM_OR8:
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num1, con1);
        GET_DBLE(num2, con2);
        or64(num1, num2, res);
        goto const_getcon;
      case IM_OR:
        con2 = GET_CVAL_ARG(1);
        conval = con1 | con2;
        goto const_return;
      case IM_XOR8:
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num1, con1);
        GET_DBLE(num2, con2);
        xor64(num1, num2, res);
        goto const_getcon;
      case IM_XOR:
        con2 = GET_CVAL_ARG(1);
        conval = con1 ^ con2;
        goto const_return;
      case IM_NOT8:
        GET_DBLE(num1, con1);
        not64(num1, res);
        goto const_getcon;
      case IM_NOT:
        conval = ~con1;
        goto const_return;
      case IM_I8MOD:
        /* i % j = i - (i / j)*j */
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num1, con1);
        GET_DBLE(num2, con2);
        div64(num1, num2, res);
        mul64(num2, res, res);
        sub64(num1, res, res);
        goto const_getcon;
      case IM_MOD:
        con2 = GET_CVAL_ARG(1);
        conval = con1 % con2;
        goto const_return;
      case IM_IDIM:
        con2 = GET_CVAL_ARG(1);
        conval = con1 > con2 ? con1 - con2 : 0;
        goto const_return;
      case IM_I8DIM:
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num1, con1);
        GET_DBLE(num2, con2);
        if (cmp64(num1, num2) > 0)
          sub64(num1, num2, res);
        else
          res[0] = res[1] = 0;
        goto const_getcon;
      case IM_DIM:
        con2 = GET_CVAL_ARG(1);
        if (xfcmp(con1, con2) > 0) {
          xfsub(con1, con2, &res[0]);
          conval = res[0];
        } else
          conval = CONVAL2G(stb.flt0);
        goto const_return;
      case IM_DDIM:
        con2 = GET_CVAL_ARG(1);
        if (const_fold(OP_CMP, con1, con2, DT_REAL8) > 0)
          conval = const_fold(OP_SUB, con1, con2, DT_REAL8);
        else
          conval = stb.dbl0;
        goto const_return;
#ifdef IM_QDIM
      case IM_QDIM:
        con2 = GET_CVAL_ARG(1);
        if (const_fold(OP_CMP, con1, con2, DT_QUAD) > 0)
          conval = const_fold(OP_SUB, con1, con2, DT_QUAD);
        else
          conval = stb.quad0;
        goto const_return;
#endif
      case IM_IISHFT:
        con2 = GET_CVAL_ARG(1);
        /*
         * because this ilm is used for the ISHFT intrinsic, count
         * is defined for values -16 to 16.
         */
        if (con2 >= 0) {
          if (con2 >= 16)
            conval = 0;
          else {
            conval = ULSHIFT(con1, con2);
            conval = ULSHIFT(conval, 16);
            conval = ARSHIFT(conval, 16);
          }
        } else {
          if (con2 <= -16)
            conval = 0;
          else {
            con1 &= 0xffff;
            conval = URSHIFT(con1, -con2);
          }
        }
        goto const_return;
      case IM_JISHFT:
        con2 = GET_CVAL_ARG(1);
        /*
         * because this ilm is used for the ISHFT intrinsic, count
         * is defined for values -32 to 32; some hw (i.e., n10) shifts
         * by cnt mod 32.
         */
        if (con2 >= 0) {
          if (con2 >= 32)
            conval = 0;
          else
            conval = ULSHIFT(con1, con2);
        } else {
          if (con2 <= -32)
            conval = 0;
          else
            conval = URSHIFT(con1, -con2);
        }
        goto const_return;
      case IM_KISHFT:
        con2 = GET_CVAL_ARG(1);
        /* con1 and con2 are symbol pointers */
        /* get the value for con2 */
        con2 = CONVAL2G(con2);
        res[0] = CONVAL1G(con1);
        res[1] = CONVAL2G(con1);
        if (con2 >= 0) {
          if (con2 >= 64) {
            res[0] = 0;
            res[1] = 0;
          } else if (con2 >= 32) {
            /* shift con1 by 32 bits or more */
            res[0] = ULSHIFT(res[1], con2 - 32);
            res[1] = 0;
          } else {
            /* shift by less than 32 bits; shift high-order
             * bits of low-order word into high-order word */
            res[0] = ULSHIFT(res[0], con2) | URSHIFT(res[1], 32 - con2);
            res[1] = ULSHIFT(res[1], con2);
          }
        } else {
          con2 = -con2;
          if (con2 >= 64) {
            res[0] = 0;
            res[1] = 0;
          } else if (con2 >= 32) {
            /* shift con1 by 32 bits or more */
            res[1] = URSHIFT(res[0], con2 - 32);
            res[0] = 0;
          } else {
            /* shift by less than 32 bits; shift low-order
             * bits of high-order word into low-order word */
            res[1] = URSHIFT(res[1], con2) | ULSHIFT(res[0], 32 - con2);
            res[0] = URSHIFT(res[0], con2);
          }
        }
        conval = getcon(res, DT_INT8);
        goto const_return;
      case IM_ICHAR:
        if (DTY(SST_DTYPEG(ARG_STK(0))) == TY_NCHAR) { /* kanji */
          int dum, clen;
          assert(DTY(DTYPEG(con1)) == TY_CHAR || DTY(DTYPEG(con1)) == TY_NCHAR,
                 "ref_intrin:KK", con1, 3);
          con2 = CONVAL1G(con1);
          clen = string_length(DTYPEG(con2));
          conval = kanji_char((unsigned char *)stb.n_base + CONVAL1G(con2),
                              clen, &dum);
        } else
          conval = stb.n_base[CONVAL1G(con1)] & 0xff;

        if (!dtyper)
          dtyper = stb.user.dt_int;
        dtype = dtyper;
        if (DTY(dtyper) == TY_INT8) {
          /* The user default integer is integer*8, but INTTYP(ICHAR)
           * may still be DT_INT4 because of -i8.  Force the type to
           * DT_INT8 -- a better way to do this may be to store
           * DT_INT8 in the INTTYP field in sym_init() if -i8
           * (-x 124 0x10) was present.
           */
          res[0] = 0;
          res[1] = conval;
          conval = getcon(res, DT_INT8);
          dtype = DT_INT8;
        }
        goto const_return_2;
      case IM_CHAR:
        ch = con1;
        conval = getstring(&ch, 1);
        goto const_return;

      case IM_GE:
      case IM_GT:
      case IM_LE:
      case IM_LT:
        dtype = SST_DTYPEG(ARG_STK(0));
        /* two arguments must both be either TY_CHAR or TY_NCHAR: */
        if (DTY(dtype) != DTY(SST_DTYPEG(ARG_STK(1))))
          goto intrinsic_error;
        con2 = GET_CVAL_ARG(1);
        conval = const_fold(OP_CMP, con1, con2, dtype);

        switch (opc) {
        case IM_GE:
          conval = conval >= 0 ? SCFTN_TRUE : SCFTN_FALSE;
          break;
        case IM_GT:
          conval = conval > 0 ? SCFTN_TRUE : SCFTN_FALSE;
          break;
        case IM_LE:
          conval = conval <= 0 ? SCFTN_TRUE : SCFTN_FALSE;
          break;
        case IM_LT:
          conval = conval < 0 ? SCFTN_TRUE : SCFTN_FALSE;
        }

        /* Convert constant result logical type if -i8 turned on */

        if (DTY(stb.user.dt_log) == TY_LOG8) {
          dtype = DT_LOG8;
          conval = cngcon(conval, DT_LOG4, dtype);
          goto const_return_2;
        }
        goto const_return;
      case IM_IIBSET:
      case IM_JIBSET:
        /* how many bits to use from the first argument */
        i = size_of(dtype);
        i = i * 8;
        con2 = GET_CVAL_ARG(1);
        /* take only lower bits of con2, that is, modulo i */
        con2 = con2 % i;
        /* set bit 'con2' in 'con1' */
        conval = con1 | (1 << con2);
        goto const_return;
      case IM_KIBSET:
        /* how many bits to use from the first argument */
        i = size_of(dtype);
        i = i * 8;
        GET_DBLE(num1, con1);
        con2 = GET_CVAL_ARG(1);
        GET_DBLE(num2, con2);
        con2 = num2[1];
        /* take only lower bits of con2, that is, modulo i */
        con2 = con2 % i;
        res[2] = res[3] = 0;
        res[0] = num1[0];
        res[1] = num1[1];
        if (con2 >= 32) {
          res[0] |= 1 << (con2 - 32);
        } else {
          res[1] |= 1 << con2;
        }
        goto const_getcon;

      default:
        switch (intast) {
        case I_IISIGN:
        case I_JISIGN:
        case I_ISIGN:
          conval = con1;
          if (conval < 0 && conval != 0x80000000)
            conval = -conval;
          con2 = GET_CVAL_ARG(1);
          if (con2 < 0 && conval != 0x80000000)
            conval = -conval;
          goto const_return;
        case I_KISIGN:
          GET_DBLE(res, con1);
          GET_DBLE(num1, stb.k0);
          if (cmp64(res, num1) < 0)
            neg64(res, res);
          con2 = GET_CVAL_ARG(1);
          GET_DBLE(num2, con2);
          if (cmp64(num2, num1) < 0)
            neg64(res, res);
          goto const_getcon;
        case I_SIGN:
          xfabsv(con1, &conval);
          con2 = GET_CVAL_ARG(1);
          num1[0] = CONVAL2G(stb.flt0);
          if (con2 == CONVAL2G(stb.fltm0) || xfcmp(con2, num1[0]) < 0) {
            /* IEEE -0.0 , or < 0.0 */
            xfneg(conval, &conval);
          }
          goto const_return;
        case I_DSIGN:
          GET_DBLE(res, con1);
          xdabsv(res, res);
          con2 = GET_CVAL_ARG(1);
          GET_DBLE(num2, con2);
          GET_DBLE(num1, stb.dbl0);
          if (con2 == stb.dblm0 || xdcmp(num2, num1) < 0) {
            /* IEEE -0.0 , or < 0.0 */
            xdneg(res, res);
          }
          goto const_getcon;
        default:
          break;
        }
        break;
      }
    } else { /* max or min intrinsic */
      switch (opc) {
      case IM_IMAX:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (con1 > conval)
            conval = con1;
        }
        break;
      case IM_I8MAX:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_INT8) > 0)
            conval = con1;
        }
        break;
      case IM_RMAX:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (xfcmp(con1, conval) > 0)
            conval = con1;
        }
        break;
      case IM_DMAX:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_REAL8) > 0)
            conval = con1;
        }
        break;
#ifdef IM_QMAX
      case IM_QMAX:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_QUAD) > 0)
            conval = con1;
        }
      break;
#endif
      case IM_IMIN:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (con1 < conval)
            conval = con1;
        }
        break;
      case IM_I8MIN:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_INT8) < 0)
            conval = con1;
        }
        break;
      case IM_RMIN:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (xfcmp(con1, conval) < 0)
            conval = con1;
        }
        break;
      case IM_DMIN:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_REAL8) < 0)
            conval = con1;
        }
        break;
#ifdef IM_QMIN
      case IM_QMIN:
        conval = con1;
        for (i = 1; i < count; i++) {
          con1 = GET_CVAL_ARG(i);
          if (const_fold(OP_CMP, con1, conval, DT_QUAD) < 0)
            conval = con1;
        }
        break;
#endif
      default:
        goto no_const_fold;
      }
      if (argtyp != dtype)
        conval = cngcon(conval, argtyp, dtype);
      goto const_return;
    }
    goto no_const_fold;

  const_getcon:
    conval = getcon(res, dtype);
  const_return:
    if (ARGTYPG(sptr) == INTTYPG(sptr) && dtyper) {
      dtype = dtyper;
    } else {
      dtype = INTTYPG(sptr);
    }
  const_return_2:
    SST_IDP(stktop, S_CONST);
    SST_DTYPEP(stktop, dtype);
    SST_CVALP(stktop, conval);
    EXPSTP(sptre, 1); /* freeze generic or specific name */
    SST_SHAPEP(stktop, 0);

    ast = mk_cval1(conval, dtype);
    SST_ASTP(stktop, ast);

    return conval;
  }

no_const_fold:
  /*
   * Validate arguments specified.
   */
  shaper = 0;
  if (opc == 0 && paramct == 11)
    /* CMPLX/DCMPLX intrinsic */
    for (i = 0; i < count; XFR_ARGAST(i), i++) {
      sp = ARG_STK(i);
      chktyp(sp, DT_NUMERIC, FALSE);
      if (!shaper)
        shaper = SST_SHAPEG(sp);
    }
  else
    for (i = 0; i < count; XFR_ARGAST(i), i++) {
      sp = ARG_STK(i);
      if (opc == IM_LOC) {
        if (sc_local_passbyvalue(SST_SYMG(sp), GBL_CURRFUNC)) {
          error(155, 3, gbl.lineno,
                "unsupported LOC of VALUE parameter:", SYMNAME(SST_SYMG(sp)));
        } else if (mklvalue(sp, 3) == 0)
          goto intrinsic_error;
      }
      else if (DTYG(SST_DTYPEG(sp)) == TY_NCHAR) {
        switch (opc) {
        case IM_ICHAR:
          dtyper = stb.user.dt_int;
          FLANG_FALLTHROUGH;
        case IM_NCHAR:
        case IM_NINDEX:
        case IM_NLEN:
        case IM_GE:
        case IM_GT:
        case IM_LE:
        case IM_LT:
          break;
        default:
          chktyp(sp, argtyp, TRUE);
          continue;
        }
        mkexpr(sp);
      }
      else {
        switch (opc) {
        case IM_GE:
        case IM_GT:
        case IM_LE:
        case IM_LT:
          if (DTYG(SST_DTYPEG(sp)) != TY_CHAR)
            goto intrinsic_error;
          mkexpr(sp);
          break;
        case IM_ICHAR:
          if (i == 0) {
            chktyp(sp, argtyp, TRUE);
            dtyper = stb.user.dt_int;
          } else {
            dtyper = set_kind_result(sp, DT_INT, TY_INT);
            if (!dtyper) {
              goto intrinsic_error;
            }
          }
          break;
#ifdef I_C_ASSOCIATED
        case IM_C_ASSOC:
          if (SST_IDG(sp) == S_EXPR)
            (void)tempify(sp);
          mkarg(sp, &dum);
          break;
#endif
        default:
          if (i == 2 && opc == IM_NINDEX)
            cngtyp(sp, DT_LOG);
          else
            chktyp(sp, argtyp, TRUE);
          break;
        }
      }

      if (!shaper)
        shaper = SST_SHAPEG(sp);
    }

  if (paramct < 12) {
    if (paramct == 11) {
      /* complex intrinsic with 2 args: cause both to make complex # */
      /* just mark as a type conversion, vectors ok - ILMG & ARRAYF
       * fields of type conversions intrinsics are 0.
       */
      opc = 0;
    }
  } else { /* max or min intrinsic */
    if (dtype != argtyp) {
      SST_IDP(stktop, S_EXPR);
      SST_DTYPEP(stktop, argtyp);
      cngtyp(stktop, dtype);
    }
  }

  /* SUCCESSFUL GENERIC/INTRINSIC PROCESSING */
  /* The data type of the result comes from the specific intrinsic used.
   * The shape of the result comes from the shape of the 1st argument.
   */
  if (opc == IM_LOC) {
    shaper = 0;
    dtyper = DT_PTR;
    switch (intast) {
    case I_C_LOC:
      ddt = get_iso_ptrtype("c_ptr");
      if (ddt)
        dtyper = ddt;
      break;
    case I_C_FUNLOC:
      ddt = get_iso_ptrtype("c_funptr");
      if (ddt)
        dtyper = ddt;
      break;
    }
  } else {
    if (!dtyper) {
      switch (intast) {
      case I_BITEST:
      case I_BJTEST:
      case I_BKTEST:
      case I_BTEST:
        dtyper = stb.user.dt_log;
        break;
      default:
        dtyper = INTTYPG(sptr);
        break;
      }
    }
    if (DTY(dtype1) == TY_ARRAY && (ARRAYFG(sptr) || !opc)) {
      /* Assertion:  First argument is an array AND intrinsic can
       *             handle vectors (this includes the type conversion
       *             intrinsics).  Create an array data type.
       */
      dtype = dup_array_dtype(dtype1);
      DTY(dtype + 1) = dtyper;
      dtyper = dtype;
    } else {
      if (shaper)
        interr("ref_intrin: result has shape, but dtype is not array", dtyper,
               2);
    }
  }

  SST_DTYPEP(stktop, dtyper);
  SST_IDP(stktop, S_EXPR);

  /* It is time to freeze the symbol's use as an intrinsic reference.
   * Use sptre which points to the generic or specific name that was found
   * in the source code.  Freezing generic names does not automatically
   * freeze specific names unless the names are the same.
   */

  func_type = A_INTR;
  switch (intast) {
  case I_ICHAR:
    if (count == 2) {
      count = 1;
    }
    func_ast = mk_id(sptre);
    break;
  case I_MODULO:
    switch ((int)INTTYPG(sptr)) {
    case DT_SINT:
      rtlRtn = RTE_imodulov;
      break;
    case DT_INT4:
      rtlRtn = RTE_modulov;
      break;
    case DT_INT8:
      rtlRtn = RTE_i8modulov;
      break;
    case DT_REAL4:
      rtlRtn = RTE_amodulov;
      break;
    case DT_REAL8:
      rtlRtn = RTE_dmodulov;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case DT_QUAD:
      rtlRtn = RTE_qmodulov;
      break;
#endif
    }
    fsptr = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), (int)INTTYPG(sptr));
    EXTSYMP(sptr, fsptr);
    ELEMENTALP(sptr, 1);
    func_ast = mk_id(fsptr);
    break;
#ifdef I_C_ASSOCIATED
  case I_C_ASSOCIATED:
    if (_c_associated(stktop, count)) {
      count = 2;
      goto use_intr_sym;
    }
    goto intrinsic_error;
#endif
  case I_SNGL:
    if (XBIT(124, 0x8)) {
      /* -r8 */
      ast = ARG_AST(0);
      SST_ASTP(stktop, ast);
      SST_DTYPEP(stktop, DT_REAL8);
      SST_SHAPEP(stktop, shaper);
      EXPSTP(sptre, 1);
      return 1;
    }
    goto use_intr_sym;
  case I_IISHFTC:
  case I_JISHFTC:
  case I_ISHFTC:
  case I_KISHFTC:
    if (count == 2) { /* need to provide a size argument */
      ARG_AST(2) = mk_cval((INT)bits_in((int)DDTG(f_dt)), DT_INT);
      count++;
    }
    FLANG_FALLTHROUGH;
  default: /* name is just the name of the specific or generic */
  use_intr_sym:
    func_ast = mk_id(sptre);
    break;
  }

  argt = mk_argt(count); /* space for arguments */
  for (i = 0; i < count; i++)
    ARGT_ARG(argt, i) = ARG_AST(i);

  ast = mk_func_node(func_type, func_ast, count, argt);
  A_DTYPEP(ast, dtyper);
  A_OPTYPEP(ast, intast);
  A_SHAPEP(ast, shaper);
  SST_ASTP(stktop, ast);
  SST_SHAPEP(stktop, shaper);
  EXPSTP(sptre, 1);

  return 1;

/*
 * Error recovery: Generate ILM's, and fix semantic stack
 */
intrinsic_error:

  /* Need to add a check for min and max first */
  if (STYPEG(sptre) == ST_GENERIC && (intrin == I_MAX || intrin == I_MIN)) {
    if (count > 1 && ((DTY(dtype1) == TY_CHAR || DTY(dtype1) == TY_NCHAR) ||
                      (DTYG(dtype1) == TY_CHAR || DTYG(dtype1) == TY_NCHAR))) {

      /* Need to check if all arguments are the same type.
       * Not sure if we can check shape here, I think so(later).
       */
      argt = mk_argt(count + 2);
      for (i = 0; i < count; i++) {
        sp = ARG_STK(i);
        argdtype = SST_DTYPEG(sp);
        if (DTY(argdtype) != DTY(dtype1)) {
          goto intrinsic_error2;
        }
        if (ARG_AST(i)) {
          ARGT_ARG(argt, i + 2) = ARG_AST(i);
        } else if (SST_IDG(sp) == S_IDENT || SST_IDG(sp) == S_ACONST) {
          SST_ASTP(sp, 0);
          (void)mkarg(sp, &dum);
          XFR_ARGAST(i);
          ARGT_ARG(argt, i + 2) = ARG_AST(i);
          if (rank_of_ast((int)ARG_AST(0)) != rank_of_ast((int)ARG_AST(i))) {
            goto intrinsic_error2;
          }
        }
      }
      rtlRtn = intrin == I_MAX ? RTE_max : RTE_min;
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtyper);
      func_ast = mk_id(hpf_sym);
      /* Add 2 arguments
       * 1) the number of argument in the list, excluding itself and the result
       * 2) the result
       */
      sp = ARG_STK(0);
      chktyp(sp, dtype1, TRUE);
      shaper = SST_SHAPEG(sp);

      /* check only the first argument */
      if (DTY(dtype1) == TY_ARRAY) {
        if (shaper) {
          if (SHD_NDIM(shaper) != ADD_NUMDIM(dtype1)) {
            tmp = get_shape_arr_temp(ARG_AST(0));
          } else {
            ADSC *ad;
            ad = AD_DPTR(dtype1);
            if (AD_DEFER(ad) || AD_ADJARR(ad) || AD_NOBOUNDS(ad)) {
              tmp = get_shape_arr_temp(ARG_AST(0));
            } else
              tmp = get_arr_temp(dtype1, FALSE, TRUE, FALSE);
          }
        } else
          tmp = get_arr_temp(dtype1, FALSE, TRUE, FALSE);

      } else {
        dtype1 = get_temp_dtype(dtype1, ARG_AST(0));
        tmp = get_temp(dtype1);
      }
      tmp_ast = mk_id(tmp);

      func_type = A_CALL;
      /* First number of argument list, and a result */
      ARGT_ARG(argt, 0) = mk_cval(count, DT_INT);
      ARGT_ARG(argt, 1) = tmp_ast;

      ast = mk_func_node(func_type, func_ast, count + 2, argt);

      add_stmt(ast);
      dtyper = dtype1;
      A_DTYPEP(ast, dtyper);
      A_DTYPEP(func_ast, dtyper);
      A_SHAPEP(ast, shaper);

      SST_ASTP(stktop, tmp_ast);
      SST_SHAPEP(stktop, shaper);
      SST_DTYPEP(stktop, dtyper);
      SST_IDP(stktop, S_EXPR);

      EXPSTP(hpf_sym, 1);
      ELEMENTALP(hpf_sym, 1);
      return 1;
    }
  }

intrinsic_error2:
  /* Wrong number or type of arguments to intrinsic */
  if (frozen) {
    /* Replace expression term with constant 0.  Save sptr to intrinsic
     * in stack so that during lvalue processing the error message
     * generated can get the symbol's name.
     */
    error(74, 3, gbl.lineno, SYMNAME(sptre), CNULL);
    fix_term(stktop, stb.i0);
    SST_ERRSYMP(stktop, sptre);
  } else {
    /* Intrinsic name without argument list is assumed to be a variable
     * Intrinsic name with wrong argument list is assumed to be external
     */
    if (list == NULL) {
      sptr = newsym(sptre);
      STYPEP(sptre, ST_VAR);
    } else {
      sptr = newsym(sptre);
      STYPEP(sptre, ST_IDENT);
    }

    mkident(stktop);
    SST_SYMP(stktop, sptr);
    mkvarref(stktop, list);
  }

  SST_IDP(stktop, S_EXPR);
  return 1;
}

#ifdef I_C_ASSOCIATED
static int
_c_associated(SST *stkp, int count)
{
  int lop, rop;

  lop = ARG_AST(0);
  if (!is_iso_cptr(A_DTYPEG(lop)))
    return 0;
  lop = rewrite_cptr_references(lop);
  ARG_AST(0) = lop;
  if (count == 2) {
    rop = ARG_AST(1);
    if (!is_iso_cptr(A_DTYPEG(rop)))
      return 0;
    rop = rewrite_cptr_references(rop);
    ARG_AST(1) = rop;
  }
  return 1;
}
#endif

static void
e74_cnt(int sym, int cnt, int l, int u)
{
  char buf[64];

  buf[0] = '-';
  buf[1] = ' ';
  if (l == u)
    sprintf(buf + 2, "%d argument(s) present, %d argument(s) expected", cnt, l);
  else
    sprintf(buf + 2, "%d argument(s) present, %d-%d argument(s) expected", cnt,
            l, u);
  error(74, 3, gbl.lineno, SYMNAME(sym), buf);
}

static void
e74_arg(int sym, int pos, char *kwd)
{
  char buf[128];
  int i;
  int kwd_len;
  char *np;
  char *p, *q;

  if (sem.which_pass == 0)
    return;
  strcpy(buf, "- keyword argument ");
  if (kwd != NULL)
    strcat(buf, kwd);
  else {
    kwd = KWDARGSTR(sym);
    for (i = 0; TRUE; i++) {
      if (*kwd == '*' || *kwd == ' ')
        kwd++;
      if (*kwd == '#' || *kwd == '\0') {
        sprintf(buf + strlen(buf), "position %d", pos + 1);
        goto report_;
      }
      kwd_len = 0;
      for (np = kwd; TRUE; np++) {
        if (*np == ' ' || *np == '\0')
          break;
        kwd_len++;
      }
      if (i == pos)
        break;
      kwd = np;
    }
    p = kwd;
    q = buf + strlen(buf);
    while (kwd_len > 0) {
      *q++ = *p++;
      --kwd_len;
    }
    *q = 0;
  }
report_:
  error(74, 3, gbl.lineno, SYMNAME(sym), buf);
}

static int
gen_call_class_obj_size(int sptr)
{
  int ast;
  int argt;
  int arg;
  int func_ast;
  int hpf_sym;

  argt = mk_argt(1);
  if (SCG(sptr) == SC_DUMMY) {
    arg = get_type_descr_arg(gbl.currsub, sptr);
  } else {
    arg = SDSCG(sptr) ? SDSCG(sptr) : get_static_type_descriptor(sptr);
  }

  ARGT_ARG(argt, 0) = mk_id(arg);
  DESCUSEDP(sptr, 1);

  hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_class_obj_size), DT_INT8);
  func_ast = mk_id(hpf_sym);
  ast = mk_func_node(A_FUNC, func_ast, 1, argt);
  A_DTYPEP(ast, DT_INT8);
  return ast;
}

/* this flag disables an error message in mkexpr1 (semutil.c)
 * about assumed-size arrays */
int dont_issue_assumedsize_error = 0;

/** \brief Handle calls to Predeclared functions.
    \param stktop function to call
    \param list   arguments to pass to function
 */
int
ref_pd(SST *stktop, ITEM *list)
{
  INT con1, con2;
  INT num1[4];
  INT res[4];
  INT conval = 0;
  char ch;
  int dtype1, dtype2, dtyper, dtyper2;
  int count;
  INT val[4];
  ISZ_T iszval;
  int dum;
  ITEM *ip1;
  int ast, arg1, rank;
  int argt;
  int argt_count, argt_extra;
  int i;
  ADSC *ad;
  SST *stkp, *stkp1, *stkp2;
  SST *dim;
  SST *mask;
  int shape1, shape2, shaper;
  int tmp;
  int hpf_sym; /* hpf-specific sptr, if special name required for
                * the predeclared for hpf
                */
  int func_type;
  int arrtmp_ast;
  const char *name;
  int func_ast;
  ACL *shape_acl;
  int sptr = 0, fsptr, baseptr;
  LOGICAL is_constant;
  int asumsz;
  int pvar;
  int nelems, eltype;
  const char *sname = NULL;
  char verstr[140]; /*140, get_version_str returns max 128 char + pf90 prefix */
  FtnRtlEnum rtlRtn = 0;
  SPTR pdsym = SST_SYMG(stktop);
  int pdtype = PDNUMG(pdsym);

/* any integer type, or hollerith, or, if -x 51 0x20 not set, real/double */
#define TYPELESS(dt)                     \
  (DT_ISINT(dt) || DTY(dt) == TY_HOLL || \
   (!XBIT(51, 0x20) && (DTY(dt) == TY_REAL || DTY(dt) == TY_DBLE)))

  dont_issue_assumedsize_error = 0;
  SST_CVLENP(stktop, 0);
  hpf_sym = 0;
  func_type = A_INTR;
  /* Count the number of arguments to function */
  count = 0;
  for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
    count++;
    if (SST_IDG(ip1->t.stkp) == S_TRIPLE) {
      /* form is e1:e2:e3 */
      error(76, 3, gbl.lineno, SYMNAME(pdsym), CNULL);
      goto bad_args;
    }
  }

  argt_count = count;
  argt_extra = 0;
  shaper = 0;
  switch (pdtype) {
  case PD_and:
  case PD_eqv:
  case PD_neqv:
  case PD_or:
    /* Validate the number of arguments and their data types */
    if (count != 2 || get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    dtype2 = SST_DTYPEG(ARG_STK(1));
    if (!TYPELESS(dtype1) || !TYPELESS(dtype2))
      goto bad_args;

    /* Choose size of operation and thus the result from the argument
     * having the largest size.  Then cast both arguments to this size.
     */
    dtype1 = (size_of(dtype1) > 4) ? DT_DWORD : DT_WORD;
    dtype2 = (size_of(dtype2) > 4) ? DT_DWORD : DT_WORD;
    dtyper = (dtype1 > dtype2) ? dtype1 : dtype2;
    (void)casttyp(ARG_STK(0), dtyper);
    (void)casttyp(ARG_STK(1), dtyper);
    XFR_ARGAST(0);
    XFR_ARGAST(1);
    break;

  case PD_compl:
    /* Validate the number of arguments and their data types */
    if (count != 1 || get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype1 = SST_DTYPEG(ARG_STK(0));

    if (!TYPELESS(dtype1))
      goto bad_args;

    /* Choose size of operation and thus result from the argument */
    if (size_of(dtype1) > 4) {
      (void)casttyp(ARG_STK(0), DT_DWORD);
      dtyper = DT_DWORD;
    } else {
      (void)casttyp(ARG_STK(0), DT_WORD);
      dtyper = DT_WORD;
    }
    XFR_ARGAST(0);
    break;

  case PD_zext:
  case PD_jzext:
    if (count != 1 || get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (!DT_ISINT(dtype1) && !DT_ISLOG(dtype1))
      goto bad_args;
    (void)mkexpr(ARG_STK(0));
    XFR_ARGAST(0);
    dtyper = DT_INT;
    break;
  case PD_izext:
    if (count != 1 || get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (!DT_ISINT(dtype1) && !DT_ISLOG(dtype1))
      goto bad_args;
    if (size_of(dtype1) > size_of(DT_SINT))
      goto bad_args;
    (void)mkexpr(ARG_STK(0));
    XFR_ARGAST(0);
    dtyper = DT_SINT;
    break;

  case PD_matmul:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    stkp1 = ARG_STK(0); /* matrix_a */
    dtyper = SST_DTYPEG(stkp1);
    shape1 = SST_SHAPEG(stkp1);
    if (shape1 == 0) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    ast = SST_ASTG(stkp1);
    sptr = SST_SYMG(stkp1);

    stkp = ARG_STK(1); /* matrix_b */
    dtype2 = SST_DTYPEG(stkp);
    shape2 = SST_SHAPEG(stkp);
    if (shape2 == 0) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    /* Recognize and rewrite the idiom MATMUL(TRANSPOSE(...), ...).  At
     * present, we only handle the matrix by vector case for real and
     * complex.  This is an attempt to improve the performance of spec
     * benchmark galgel.
     */
    if (SST_IDG(stkp1) == S_EXPR && A_TYPEG(ast) == A_INTR)
      if (STYPEG(sptr) == ST_PD && PDNUMG(sptr) == PD_transpose)
        if (SHD_NDIM(shape1) == 2 && SHD_NDIM(shape2) == 1)
          if (DT_ISREAL_ARR(dtyper) || DT_ISCMPLX_ARR(dtyper))
            if (DTYG(dtyper) == DTYG(dtype2)) {

              pdsym = getsymbol("matmul_transpose");
              ARG_AST(0) = ARGT_ARG(A_ARGSG(ast), 0);
              /*SST_ASTP(stkp, A_LOPG(ast));*/
            }

    if (DT_ISLOG(DTY(dtyper + 1))) {
      if (!DT_ISLOG(DTY(dtype2 + 1))) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else if (DT_ISNUMERIC(DTY(dtyper + 1))) {
      if (!DT_ISNUMERIC(DTY(dtype2 + 1))) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }

    switch (SHD_NDIM(shape1)) {
    case 1:
      if (SHD_NDIM(shape2) != 2) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      /* (n) * (n, k) = (k) */
      /* TBD: cmp_bnd_shape(shape1, 1, shape2, 1) */
      add_shape_rank(1);
      add_shape_spec((int)SHD_LWB(shape2, 1), (int)SHD_UPB(shape2, 1),
                     (int)SHD_STRIDE(shape2, 1));
      break;
    case 2:
      switch (SHD_NDIM(shape2)) {
      case 1: /* (n, m) * (m) = (n) */
        /* TBD: cmp_bnd_shape(shape1, 2, shape2, 1) */
        add_shape_rank(1);
        add_shape_spec((int)SHD_LWB(shape1, 0), (int)SHD_UPB(shape1, 0),
                       (int)SHD_STRIDE(shape1, 0));
        break;
      case 2: /* (n, m) * (m, k) = (n, k) */
        /* TBD: cmp_bnd_shape(shape1, 2, shape2, 1) */
        add_shape_rank(2);
        add_shape_spec((int)SHD_LWB(shape1, 0), (int)SHD_UPB(shape1, 0),
                       (int)SHD_STRIDE(shape1, 0));
        add_shape_spec((int)SHD_LWB(shape2, 1), (int)SHD_UPB(shape2, 1),
                       (int)SHD_STRIDE(shape2, 1));
        break;
      default:
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      break;
    default:
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    shaper = mk_shape();

    /* check data types with respect to the rules of the equivalent binary
     * operations.
     */
    if (DTY(dtyper + 1) < DTY(dtype2 + 1)) {
      cngtyp(ARG_STK(0), dtype2);
      dtyper = dtype2;
      XFR_ARGAST(0);
    } else {
      cngtyp(ARG_STK(1), dtyper);
      XFR_ARGAST(1);
    }
    break;
  case PD_isnan:
    if (count != 1 || get_kwd_args(list, count, KWDARGSTR(pdsym))) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (DTYG(dtype1) != TY_REAL && DTYG(dtype1) != TY_DBLE)
      goto call_e74_arg;
    (void)mkexpr(ARG_STK(0));
    XFR_ARGAST(0);
    dtyper = DT_LOG;
    if (DTY(dtype1) == TY_ARRAY) {
      shape1 = A_SHAPEG(ARG_AST(0));
      count = SHD_NDIM(shape1);
      dtyper = get_array_dtype(count, DT_LOG);
    }
    break;
  case PD_dotproduct:
    if (!XBIT(49, 0x40)) /* if xbit set, CM fortran intrinsics allowed */
      goto bad_args;
    FLANG_FALLTHROUGH;
  case PD_dot_product:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 2;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtype1) != TY_ARRAY || rank_of_ast(ARG_AST(0)) != 1) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtype2 = SST_DTYPEG(ARG_STK(1));
    if (DTY(dtype2) != TY_ARRAY || rank_of_ast(ARG_AST(1)) != 1) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    dtyper = DTY(dtype1 + 1);
    if (DT_ISLOG(dtyper)) {
      if (!DT_ISLOG(DTY(dtype2 + 1))) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else if (DT_ISNUMERIC(DTY(dtyper))) {
      if (!DT_ISNUMERIC(DTY(dtype2 + 1))) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    /* check data types with respect to the rules of the equivalent binary
     * operations.
     */
    if (dtyper < DTY(dtype2 + 1)) {
      cngtyp(ARG_STK(0), dtype2);
      dtyper = DTY(dtype2 + 1);
      XFR_ARGAST(0);
    } else {
      cngtyp(ARG_STK(1), dtype1);
      XFR_ARGAST(1);
    }
    if (pdtype == PD_dotproduct) {
      INTASTP(pdsym, I_DOT_PRODUCT);
      if (flg.standard)
        ERR170("dotproduct should be dot_product");
    }
    break;
  case PD_all:
  case PD_any:
    if (count == 0 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 2;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (!DT_ISLOG_ARR(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = DTY(dtype1 + 1);
    if ((stkp = ARG_STK(1))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      shaper = reduc_shape((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp),
                           (int)STD_PREV(0));
      if (shaper)
        dtyper = dtype1;
    }
    break;
  case PD_count:
    if (count == 0 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 2;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (!DT_ISLOG_ARR(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = stb.user.dt_int;

    if ((stkp = ARG_STK(1))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      shaper = reduc_shape((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp),
                           (int)STD_PREV(0));
      if (shaper)
        dtyper = aux.dt_iarray;
    }
    break;
  case PD_findloc:
    if (count < 2 || count > 6) {
      E74_CNT(pdsym, count, 1, 6);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 6, KWDARGSTR(pdsym)))
      goto exit_;

    argt_count = 5;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISNUMERIC_ARR(dtype1) &&
        !(DTY(dtype1) == TY_ARRAY &&
          (DTYG(dtype1) == TY_CHAR || DTYG(dtype1) == TY_NCHAR))) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    stkp = ARG_STK(1); /* value */
    dtype2 = SST_DTYPEG(stkp);
    if ((DT_ISNUMERIC_ARR(dtype1) && !DT_ISNUMERIC(dtype2)) ||
        DTYG(dtype1) !=
            DTYG(dtype2)) { // TODO: check type against input array ???
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }

    if ((stkp = ARG_STK(4)) && SST_IDG(stkp) == S_CONST) { /* KIND */
      dtyper2 = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper2) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper2 = 0;
    }

    dim = 0;
    mask = 0;

    if ((stkp = ARG_STK(2))) {
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (DT_ISLOG(dtype2)) {
        /* mask && no dim */
        mask = stkp;
        ARG_STK(2) = 0;
      } else if (DT_ISINT(dtype2)) {
        dim = stkp;
      } else {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }

    if (dim) {
      ARG_STK(2) = dim;
      shaper = reduc_shape((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp),
                           (int)STD_PREV(0));
      if (shaper)
        dtyper = aux.dt_iarray;
      else
        dtyper = (!dtyper2) ? stb.user.dt_int : dtyper2;
      XFR_ARGAST(2);
    } else {
      dtyper = get_array_dtype(1, (!dtyper2) ? stb.user.dt_int : dtyper2);
      ad = AD_DPTR(dtyper);
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) =
          mk_isz_cval(rank_of_ast(ARG_AST(0)), astb.bnd.dtype);
      ARG_AST(2) = 0;
    }

    if ((stkp = ARG_STK(3))) {
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISLOG(dtype2) || mask) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      mask = ARG_STK(3);
    }

    if (mask) {
      ARG_STK(3) = mask;
      if (!chkshape(mask, ARG_STK(0), FALSE)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      ARG_AST(3) = SST_ASTG(mask);
    }

    /* back */
    if ((stkp = ARG_STK(5))) {
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      ARG_AST(4) = SST_ASTG(ARG_STK(5));
    } else {
      ARG_AST(4) = mk_cval(SCFTN_FALSE, DT_LOG);
    }
    break;

  case PD_minloc:
  case PD_maxloc:
    if (count == 0 || count > 5) {
      E74_CNT(pdsym, count, 1, 5);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 5, KWDARGSTR(pdsym)))
      goto exit_;

    if ((stkp = ARG_STK(3))) { /* KIND */
      dtyper2 = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper2) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper2 = 0;
    }

    /* back */
    if ((stkp = ARG_STK(4))) {
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      ARG_AST(3) = SST_ASTG(ARG_STK(4));
    } else {
      ARG_AST(3) = mk_cval(SCFTN_FALSE, DT_LOG);
    }

    stkp = ARG_STK(0);
    argt_count = 4;
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISNUMERIC_ARR(dtype1) &&
        !(DTY(dtype1) == TY_ARRAY &&
          (DTYG(dtype1) == TY_CHAR || DTYG(dtype1) == TY_NCHAR))) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if ((stkp = ARG_STK(2))) { /* mask */
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      if (!chkshape(stkp, ARG_STK(0), FALSE)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      XFR_ARGAST(2);
    }
    if ((stkp = ARG_STK(1))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (count == 2 && DT_ISLOG(DDTG(dtype2)) &&
          chkshape(stkp, ARG_STK(0), FALSE)) {
        XFR_ARGAST(1);
        /* shift args over */
        ARG_AST(2) = ARG_AST(1); /* mask */
        ARG_AST(1) = 0;          /* dim is 'null' */
        goto minloc_nodim;
      }
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      shaper = reduc_shape((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp),
                           (int)STD_PREV(0));
      if (shaper)
        dtyper = aux.dt_iarray;
      else
        dtyper = (!dtyper2) ? stb.user.dt_int : dtyper2;
    } else {
    minloc_nodim:
      dtyper = get_array_dtype(1, (!dtyper2) ? stb.user.dt_int : dtyper2);
      ad = AD_DPTR(dtyper);
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) =
          mk_isz_cval(rank_of_ast(ARG_AST(0)), astb.bnd.dtype);
    }
    break;
  case PD_minval:
  case PD_maxval:
  case PD_product:
  case PD_sum:
  case PD_norm2:
    if (count == 0 || count > 3) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }

    // norm2 intrinsic does not have a mask arg
    argt_count = pdtype == PD_norm2 ? 2 : 3;
    if (evl_kwd_args(list, argt_count, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (!DT_ISNUMERIC_ARR(dtype1)) {
      if (pdtype == PD_minval || pdtype == PD_maxval) {
        if (!(DTY(dtype1) == TY_ARRAY &&
              (DTYG(dtype1) == TY_CHAR || DTYG(dtype1) == TY_NCHAR))) {
          E74_ARG(pdsym, 0, NULL);
          goto call_e74_arg;
        }

      } else {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    if (pdtype == PD_minval || pdtype == PD_maxval) {
      if ((!DT_ISINT_ARR(dtype1) && !DT_ISREAL_ARR(dtype1) &&
           !(DTY(dtype1) == TY_ARRAY &&
             (DTYG(dtype1) == TY_CHAR || DTYG(dtype1) == TY_NCHAR))) ||
          DT_ISLOG_ARR(dtype1)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }

    if (pdtype == PD_norm2) {
      if (!DT_ISREAL_ARR(dtype1)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      if (ARG_STK(1)) {
        // dim arg
        ast = SST_ASTG(ARG_STK(1));
        sptr = ast_is_sym(ast) ? memsym_of_ast(ast) : 0;

        // If symbol, disallow if optional dummy arguments used as dim
        if (sptr && OPTARGG(sptr)) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
      }
    }

    dtyper = DTY(dtype1 + 1);
    if ((stkp = ARG_STK(2))) { /* mask */
      dtype2 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      if (!chkshape(stkp, ARG_STK(0), FALSE)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      XFR_ARGAST(2);
    }
    if ((stkp = ARG_STK(1))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        if (count == 2) {
          if (DT_ISLOG(DDTG(dtype2)) && chkshape(stkp, ARG_STK(0), FALSE)) {
            XFR_ARGAST(1);
            /* shift args over */
            ARG_AST(2) = ARG_AST(1); /* mask */
            ARG_AST(1) = 0;          /* dim is 'null' */
            break;
          }
        }
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }

      if (rank_of_ast(ARG_AST(0)) != 1) {
        shaper = reduc_shape((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp),
                             (int)STD_PREV(0));
        if (shaper)
          dtyper = dtype1;
      } else
        check_dim_error((int)A_SHAPEG(ARG_AST(0)), (int)SST_ASTG(stkp));
    }
    break;
  case PD_dlbound:
    if (!XBIT(49, 0x40)) /* if xbit set, CM fortran intrinsics allowed */
      goto bad_args;
    pdtype = PD_lbound;
    goto lbound_ubound;
  case PD_dubound:
    if (!XBIT(49, 0x40)) /* if xbit set, CM fortran intrinsics allowed */
      goto bad_args;
    pdtype = PD_ubound;
    FLANG_FALLTHROUGH;
  case PD_lbound:
  case PD_ubound:
  lbound_ubound:
    if (count == 0 || count > 3) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    dont_issue_assumedsize_error = 1;
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    /* Check the dtype of DIM. */
    if ((stkp = ARG_STK(1))) {
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    if ((stkp = ARG_STK(2))) { /* KIND */
      dtyper2 = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper2) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper2 = stb.user.dt_int;
    }

    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtype1) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    /* get the rank of source array */
    arg1 = ARG_AST(0);
    sptr = get_whole_array_sym(arg1);
    if (sptr && STYPEG(sptr) == ST_ARRAY && SCG(sptr) == SC_DUMMY &&
        ASSUMRANKG(sptr)) {
      /* assumed-rank */
      if (!SDSCG(sptr))
        get_static_descriptor(sptr);
      rank = get_desc_rank(SDSCG(sptr));
    } else {
      rank = mk_isz_cval(rank_of_ast(ARG_AST(0)), astb.bnd.dtype);
    }

    if (ARG_AST(1)) { /* DIM */
      if (A_ALIASG(ARG_AST(1))) {
        /* dim is a constant */
        int rank_val = MAXDIMS;
        i = get_int_cval(A_SPTRG(A_ALIASG(ARG_AST(1))));
        if (A_ALIASG(rank)) {
          rank_val = get_isz_cval(A_SPTRG(A_ALIASG(rank)));
          if (pdtype == PD_ubound && sptr && ASUMSZG(sptr) && i == rank_val) {
            error(84, 3, gbl.lineno, SYMNAME(sptr),
                  "- ubound of assumed size array is unknown");
          }
        }
        if (i < 1 || i > rank_val) {
          error(423, 3, gbl.lineno, NULL, NULL);
          ARG_AST(1) = astb.bnd.one;
        }
      }
      dtyper = dtyper2;
      /* lbound/ubound (array, dim) */
      argt_count = 2;
    } else {
      if (pdtype == PD_ubound && sptr && ASUMSZG(sptr)) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- ubound of assumed size array is unknown");
      }
      dtyper = get_array_dtype(1, dtyper2);
      ADD_LWBD(dtyper, 0) = ADD_LWAST(dtyper, 0) = astb.bnd.one;
      ADD_UPBD(dtyper, 0) = ADD_UPAST(dtyper, 0) = rank;
      ADD_NUMELM(dtyper) = ADD_EXTNTAST(dtyper, 0) = ADD_UPBD(dtyper, 0);
      /* lbound/ubound (array) */
      argt_count = 1;
    }
    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, FALSE);
      return 0;
    }
    break;

  case PD_cshift:
    if (XBIT(49, 0x40)) { /* if xbit set, CM fortran intrinsics allowed */
      argpos_t swap;
      if (count != 3) {
        E74_CNT(pdsym, count, 3, 3);
        goto call_e74_cnt;
      }
      if (evl_kwd_args(list, 3, "array dim shift"))
        goto exit_;
      /* array dim shift --> array shift dim */
      swap = sem.argpos[1];          /* dim */
      sem.argpos[1] = sem.argpos[2]; /* shift */
      sem.argpos[2] = swap;          /* dim */
    } else if (count < 2 || count > 3) {
      E74_CNT(pdsym, count, 2, 3);
      goto call_e74_cnt;
    } else if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;
    dtyper = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtyper) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = A_SHAPEG(ARG_AST(0));

    if ((stkp = ARG_STK(2))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2) && !DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else
      ARG_AST(2) = astb.i1;

    stkp = ARG_STK(1); /* shift */
    dtype1 = SST_DTYPEG(stkp);
    dtype2 = DDTG(dtype1);
    if (!DT_ISINT(dtype2) && !DT_ISLOG(dtype2)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    if (DTY(dtype1) != TY_ARRAY ||
        rank_of_ast(ARG_AST(1)) == (SHD_NDIM(shaper) - 1))
      /* legal cases */;
    else {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    break;
  case PD_eoshift:
    if (XBIT(49, 0x40)) { /* if xbit set, CM fortran intrinsics allowed */
      argpos_t swap;
      if (count < 3 || count > 4) {
        E74_CNT(pdsym, count, 3, 4);
        goto call_e74_cnt;
      }
      if (evl_kwd_args(list, 4, "array dim shift *boundary"))
        goto exit_;
      /* array dim shift boundary --> array shift boundary dim */
      swap = sem.argpos[1];          /* dim */
      sem.argpos[1] = sem.argpos[2]; /* shift */
      sem.argpos[2] = sem.argpos[3]; /* boundary */
      sem.argpos[3] = swap;          /* dim */
    } else if (count < 2 || count > 4) {
      E74_CNT(pdsym, count, 2, 4);
      goto call_e74_cnt;
    } else if (evl_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 4;
    dtyper = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtyper) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = A_SHAPEG(ARG_AST(0));

    if ((stkp = ARG_STK(3))) { /* dim */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2) && !DT_ISLOG(dtype2)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    } else
      ARG_AST(3) = astb.i1;

    stkp = ARG_STK(1); /* shift */
    dtype1 = SST_DTYPEG(stkp);
    dtype2 = DDTG(dtype1);
    if (!DT_ISINT(dtype2) && !DT_ISLOG(dtype2)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    if (DTY(dtype1) != TY_ARRAY ||
        rank_of_ast(ARG_AST(1)) == (SHD_NDIM(shaper) - 1))
      /* legal cases */;
    else {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    if ((stkp = ARG_STK(2))) { /* boundary */
      dtype1 = SST_DTYPEG(stkp);
      dtype2 = DDTG(dtype1);
      if (dtype2 != DDTG(dtyper)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      if (DTY(dtype1) != TY_ARRAY ||
          rank_of_ast(ARG_AST(2)) == (SHD_NDIM(shaper) - 1))
        /* legal cases */;
      else {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    break;
  case PD_number_of_processors:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtyper = stb.user.dt_int;
    if ((stkp = ARG_STK(0))) { /* dim */
      dtype1 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype1)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }

      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_number_of_processors),
                                  stb.user.dt_int);
      argt_count = 2;
      ARG_AST(1) = mk_cval(size_of(dtype1), DT_INT);
      break;
    }
    /* something hpf-specific here. */
    hpf_sym = sym_mknproc();
    ast = mk_id(hpf_sym);
    SST_IDP(stktop, S_EXPR);
    SST_DTYPEP(stktop, dtyper);
    SST_SHAPEP(stktop, 0);
    SST_ASTP(stktop, ast);
    return 1;
  case PD_ran:
    if (count != 1)
      goto bad_args;
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto bad_args;
    if (!is_varref(ARG_STK(0)) || SST_DTYPEG(ARG_STK(0)) != DT_INT) {
      goto bad_args;
    }
    (void)mkarg(ARG_STK(0), &dum);
    dtyper = stb.user.dt_real;
    XFR_ARGAST(0);
    sptr = sym_of_ast(ARG_AST(0)); /*  intent OUT arg */
    ADDRTKNP(sptr, 1);
    break;
  case PD_secnds:
    if (count != 1) {
      goto bad_args;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (dtype1 == DT_FLOAT) {
      (void)mkexpr(ARG_STK(0));
      dtyper = DT_FLOAT;
    } else if (dtype1 == DT_DBLE) {
      (void)mkexpr(ARG_STK(0));
      dtyper = DT_DBLE;
    } else {
      goto bad_args;
    }
    XFR_ARGAST(0);
    break;
  case PD_shift:
    /* Validate the number of arguments and their data types */
    if (count != 2)
      goto bad_args;
    if (get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    dtyper = SST_DTYPEG(ARG_STK(0));
    if (!TYPELESS(dtyper) || !DT_ISINT(SST_DTYPEG(ARG_STK(1)))) {
      goto bad_args;
    }
    /*
       Choose size of operation and thus the result from the first
     * argument having the largest size.  Then cast first argument
     * to this size.
     */
    dtyper = (size_of(dtyper) > 4) ? DT_DWORD : DT_WORD;
    (void)casttyp(ARG_STK(0), dtyper);
    XFR_ARGAST(0);
    (void)chktyp(ARG_STK(1), DT_INT, FALSE);
    XFR_ARGAST(1);
    break;
  case PD_transpose:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtyper = SST_DTYPEG(ARG_STK(0));
    shaper = A_SHAPEG(ARG_AST(0));
    if (shaper == 0 || SHD_NDIM(shaper) != 2) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    add_shape_rank(2);
    add_shape_spec((int)SHD_LWB(shaper, 1), (int)SHD_UPB(shaper, 1),
                   (int)SHD_STRIDE(shaper, 1));
    add_shape_spec((int)SHD_LWB(shaper, 0), (int)SHD_UPB(shaper, 0),
                   (int)SHD_STRIDE(shaper, 0));
    shaper = mk_shape();
    break;
  case PD_spread:
    if (count != 3) {
      E74_CNT(pdsym, count, 3, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0); /* source */
    shape1 = SST_SHAPEG(stkp);
    if (shape1 && SHD_NDIM(shape1) == 7) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtype1 = SST_DTYPEG(stkp);
    /* assertion: it shouldn't matter that the result dtype doesn't have
     * the correct number of bounds.
     */
    dtyper = get_array_dtype(1, (int)DDTG(dtype1));

    if (!DT_ISINT(SST_DTYPEG(ARG_STK(2)))) { /* ncopies */
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }

    stkp = ARG_STK(1); /* dim */
    dtype2 = SST_DTYPEG(stkp);
    if (!DT_ISINT(dtype2)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    /* store max(ncopies, 0) into temporay */

    tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_INT, sem.sc);
    i = ast_intr(I_MAX, DT_INT, 2, (int)ARG_AST(2), astb.i0);
    ast = mk_assn_stmt(mk_id(tmp), i, DT_INT);
    (void)add_stmt(ast);

    shaper = increase_shape(shape1, (int)SST_ASTG(stkp), mk_id(tmp),
                            (int)STD_PREV(0));
    break;
  case PD_pack:
    if (count < 2 || count > 3) {
      E74_CNT(pdsym, count, 2, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;

    stkp = ARG_STK(0); /* array */
    dtyper = SST_DTYPEG(stkp);
    if (DTY(dtyper) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shape1 = SST_SHAPEG(stkp);

    stkp = ARG_STK(1); /* mask */
    dtype2 = SST_DTYPEG(stkp);
    if (!DT_ISLOG(DDTG(dtype2))) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    if (!chkshape(stkp, ARG_STK(0), FALSE)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    if (A_TYPEG(SST_ASTG(stkp)) != A_ID && DTY(dtype2) == TY_ARRAY) {
      /*
         Compute mask into a temp array and use this temp as the argument
         - first we need a dtype for the temp
       */
      int tmp_dtype = dtype2;

      ad = AD_DPTR(dtype2);

      if (!AD_NUMDIM(ad)) {
        tmp_dtype = dtype_with_shape(dtype2, A_SHAPEG(SST_ASTG(stkp)));
      } else {
        tmp_dtype = dtype_with_shape(DDTG(dtype2), A_SHAPEG(SST_ASTG(stkp)));
      }

      tmp = get_arr_temp(tmp_dtype, FALSE, FALSE, FALSE);
      arrtmp_ast = mk_id(tmp);
      ast = mk_assn_stmt(arrtmp_ast, SST_ASTG(stkp), tmp_dtype);
      (void)add_stmt(ast);
      ARG_AST(1) = arrtmp_ast;
    } else {
      XFR_ARGAST(1);
    }

    if ((stkp = ARG_STK(2))) { /* vector */
      if (!eq_dtype(DDTG(SST_DTYPEG(stkp)), DTY(dtyper + 1))) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      if (rank_of_ast((int)ARG_AST(2)) != 1) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }

    tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, astb.bnd.dtype, sem.sc);
    add_shape_rank(1);
    add_shape_spec(astb.bnd.one, mk_id(tmp), astb.bnd.one);
    shaper = mk_shape();

    if (stkp != NULL)
      /* use size of vector */
      ast = size_of_ast(ARG_AST(2));
    else if (DTY(dtype2) != TY_ARRAY) {
      /* mask is a scalar; use (size of array * (- (int)mask) ) */
      int temp;
      ast = mk_convert(ARG_AST(1), DT_INT);
      temp = mk_binop(OP_SUB, astb.bnd.zero, ast, astb.bnd.dtype);
      ast = mk_binop(OP_MUL, size_of_ast(ARG_AST(0)), temp, astb.bnd.dtype);
    } else {
      /* else compute size by the expression  'count(mask)' */
      int t1;
      t1 = mk_argt(2);              /* space for arguments */
      ARGT_ARG(t1, 0) = ARG_AST(1); /* mask */
      ARGT_ARG(t1, 1) = 0;          /* no dim argument */

      func_ast = mk_id(intast_sym[I_COUNT]);
      ast = mk_func_node(A_INTR, func_ast, 2, t1);
      A_DTYPEP(ast, DT_INT);
      A_OPTYPEP(ast, I_COUNT);
      A_SHAPEP(ast, 0);
    }
    ast = mk_assn_stmt(mk_id(tmp), ast, DT_INT);
    (void)add_stmt(ast);
    break;
  case PD_unpack:
    if (count != 3) {
      E74_CNT(pdsym, count, 3, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0); /* vector: any rank 1 array */
    dtyper = SST_DTYPEG(stkp);
    shape1 = SST_SHAPEG(stkp);
    if (DTY(dtyper) != TY_ARRAY || SHD_NDIM(shape1) != 1) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    stkp = ARG_STK(1); /* mask: logical array */
    dtype1 = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);
    if (!DT_ISLOG_ARR(dtype1)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    stkp = ARG_STK(2);         /* field: same type as vector */
    dtype2 = SST_DTYPEG(stkp); /*        same shape as mask */
    shape2 = SST_SHAPEG(stkp);
    if (!eq_dtype(DDTG(dtype2), DTY(dtyper + 1))) {
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }
    if (!chkshape(stkp, ARG_STK(1), FALSE)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    XFR_ARGAST(2);
    break;
  case PD_dshape:
    if (!XBIT(49, 0x40)) /* if xbit set, CM fortran intrinsics allowed */
      goto bad_args;
    FLANG_FALLTHROUGH;
  case PD_shape:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    if ((stkp = ARG_STK(1))) { /* KIND */
      dtyper2 = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper2) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper2 = 0;
    }

    dtype1 = (!dtyper2) ? stb.user.dt_int : dtyper2;

    dtyper = get_array_dtype(1, dtype1);

    if (sem.dinit_data) {
      int rank;

      /* build return type */
      stkp = ARG_STK(0);
      ad = AD_DPTR(SST_DTYPEG(stkp));
      rank = AD_NUMDIM(ad); /* rank of array arg, upper bound of result array */
      sem.arrdim.ndim = 1;
      sem.arrdim.ndefer = 0;
      sem.bounds[0].lowtype = S_CONST;
      sem.bounds[0].lowb = 1;
      sem.bounds[0].lwast = 0;
      sem.bounds[0].uptype = S_CONST;
      sem.bounds[0].upb = rank;
      sem.bounds[0].upast =
          mk_cval(rank, (!dtyper2) ? stb.user.dt_int : dtyper2);
      dtyper = mk_arrdsc();
      DTY(dtyper + 1) = (!dtyper2) ? stb.user.dt_int : dtyper2;

      gen_init_intrin_call(stktop, pdsym, count, dtyper, FALSE);
      return 0;
    }

    ad = AD_DPTR(dtyper);
    count = rank_of_ast(ARG_AST(0));
    AD_NUMELM(ad) = AD_UPBD(ad, 0) = AD_UPAST(ad, 0) =
        mk_isz_cval(count, astb.bnd.dtype);
    shape1 = A_SHAPEG(ARG_AST(0));
    argt_count = 3 * count + 2;
    tmp = get_arr_temp(dtyper, FALSE, FALSE, FALSE);
    arrtmp_ast = mk_id(tmp);
    shaper = A_SHAPEG(arrtmp_ast);
    sptr = find_pointer_variable(ARG_AST(0));
    if (sptr && (POINTERG(sptr) || (ALLOCG(sptr) && SDSCG(sptr)))) {
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_shapeDsc), DT_NONE);
      ast = begin_call(A_CALL, hpf_sym, 2);
      add_arg(arrtmp_ast);
      add_arg(check_member(ARG_AST(0), mk_id(SDSCG(sptr)))); /* rank */
    } else {
      switch (dtyper2) {
      case 0:
        rtlRtn = RTE_shape;
        break;
      case DT_BINT:
        rtlRtn = RTE_shape1;
        break;
      case DT_SINT:
        rtlRtn = RTE_shape2;
        break;
      case DT_INT4:
        rtlRtn = RTE_shape4;
        break;
      case DT_INT8:
        rtlRtn = RTE_shape8;
        break;
      default:
        error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
              "invalid kind argument for shape");
      }
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE);
      ast = begin_call(A_CALL, hpf_sym, argt_count);
      add_arg(arrtmp_ast);
      add_arg(mk_isz_cval((INT)count, astb.bnd.dtype)); /* rank */
      for (i = 0; i < count; i++) {
        add_arg((int)SHD_LWB(shape1, i));
        add_arg((int)SHD_UPB(shape1, i));
        add_arg((int)SHD_STRIDE(shape1, i));
      }
    }
    (void)add_stmt(ast);
    ast = arrtmp_ast;
    goto expr_val;

  case PD_reshape:
    if (count < 2 || count > 4) {
      E74_CNT(pdsym, count, 2, 4);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(1); /* shape */
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISINT_ARR(dtype1)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    shape_acl = NULL;
    if (SST_IDG(stkp) == S_ACONST) {
      shape_acl = SST_ACLG(stkp);
    }

    if (shape_acl && shape_acl->is_const) {
      shape_acl = SST_ACLG(stkp);
      count = get_int_cval(sym_of_ast(AD_NUMELM(AD_DPTR(dtype1))));
      if (count < 0 || count > 7) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else
      shape_acl = NULL;

    stkp = ARG_STK(0);
    dtyper = SST_DTYPEG(stkp); /* source */
    if (DTY(dtyper) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    if (SST_IDG(stkp) == S_IDENT) {
      int allo_sptr = SST_SYMG(stkp);
      if (ALLOCATTRG(allo_sptr)) {
        ALLOCDESCP(allo_sptr, TRUE);
      }
    }
    argt_count = 4;

    stkp = ARG_STK(1); /* shape */

    (void)mkexpr(ARG_STK(1));
    XFR_ARGAST(1);
    if (shape_acl == NULL) {
      ast = ARG_AST(1);
      if (sem.dinit_data && !SST_SHAPEG(stkp)) {
        if (ADD_NUMDIM(A_DTYPEG(ast)) != 1) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
        tmp = ADD_NUMELM(A_DTYPEG(ast));
      } else {
        shape1 = SST_SHAPEG(stkp);
        if (shape1 == 0 || SHD_NDIM(shape1) != 1) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
        tmp = size_of_ast(ast);
      }

      if (A_TYPEG(tmp) != A_CNST) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      count = get_int_cval(A_SPTRG(tmp));
      if (count < 0 || count > 7) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }

    if ((stkp = ARG_STK(2))) { /* pad */
      (void)mkexpr(stkp);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(stkp);
      if (DTY(dtype2) != TY_ARRAY || DTY(dtype2 + 1) != DTY(dtyper + 1)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    if ((stkp = ARG_STK(3))) { /* order */
      (void)mkexpr(stkp);
      XFR_ARGAST(3);
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(DTY(dtype2 + 1)) ||
          ((STYPEG(sym_of_ast(AD_NUMELM(AD_DPTR(dtype2)))) == ST_CONST) && 
          count != get_int_cval(sym_of_ast(AD_NUMELM(AD_DPTR(dtype2)))))) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }

    sem.arrdim.ndim = 1;
    (void)mkexpr(ARG_STK(0));

    XFR_ARGAST(0);

    if (sem.dinit_data) {
      ACL *aclp = shape_acl;

      if (!DT_ISINT(DTY(SST_DTYPEG(ARG_STK(1)) + 1))) { /* shape */
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }

      if ((stkp = ARG_STK(2))) { /* pad */
        if (DTY(SST_DTYPEG(stkp) + 1) != DTY(dtyper + 1)) {
          sem.dinit_error = TRUE;
          E74_ARG(pdsym, 2, NULL);
          goto call_e74_arg;
        }
      }

      if ((stkp = ARG_STK(3))) { /* order */
        dtype2 = SST_DTYPEG(ARG_STK(3));
        if (!DT_ISINT(DTY(dtype2 + 1)) ||
            count != get_int_cval(sym_of_ast(AD_NUMELM(AD_DPTR(dtype2))))) {
          sem.dinit_error = TRUE;
          E74_ARG(pdsym, 3, NULL);
          goto call_e74_arg;
        }
      }

      if (!aclp) {
        aclp = construct_acl_from_ast(SST_ASTG(ARG_STK(1)), 0, 0);
      }
      aclp = eval_init_expr(aclp);

      add_shape_rank(count);
      sem.arrdim.ndim = count;
      sem.arrdim.ndefer = 0;
      aclp = (aclp->id == AC_ACONST ? aclp->subc : aclp);
      if (!aclp) {
        return 0;
      }
      for (i = 0; i < count; i++) {
        int ubast = mk_bnd_int(aclp->u1.ast);
        add_shape_spec(astb.bnd.one, ubast, astb.bnd.one);

        sem.bounds[i].lowtype = S_CONST;
        sem.bounds[i].lowb = 1;
        sem.bounds[i].lwast = 0;
        sem.bounds[i].uptype = S_CONST;
        sem.bounds[i].upb = get_int_cval(A_SPTRG(aclp->u1.ast));
        sem.bounds[i].upast = ubast;
        sem.bounds[i].upast = ubast;

        aclp = aclp->next;
      }
      shaper = mk_shape();
      dtyper = mk_arrdsc();
      DTY(dtyper + 1) = DDTG(SST_DTYPEG(ARG_STK(0)));

      gen_init_intrin_call(stktop, pdsym, argt_count, dtyper, FALSE);

      A_SHAPEP(SST_ASTG(stktop), shaper);

      return 0;
    }

    if (shape_acl != NULL) {
      add_shape_rank(count);
      shape_acl = shape_acl->subc; /* go down to element list */
      for (i = 0; i < count; i++) {
        add_shape_spec(astb.bnd.one, mk_bnd_int(shape_acl->u1.ast),
                       astb.bnd.one);
        shape_acl = shape_acl->next;
      }
      shaper = mk_shape();
    } else {
      /*
       * compute the shape for the result of 'reshape':
       * o   count is the size of the shape argument and represents the
       *     rank of the result.
       * o   for each dimension in the result, its upper bound is the
       *     value of the corresponding element in the shape argument.
       * o   to access an element of the shape argument, a subscripted
       *     reference of the shape argument must be generated; the
       *     subscript will consist of any non-triple subscripts; the
       *     triple subscript will be replaced with the 'current' index.
       * o   the shape descriptor is used to generate a sequence of
       *     indices; e.g.,   lwb : upb : stride.
       */
      int arr;
      int subs[7];
      int asd;
      int dim = 0;
      int nsubs = 1;
      int ix;
      int shp[7];
      int eldtype;

      eldtype = DDTG(A_DTYPEG(ast));
      arr = ast;
      if (A_TYPEG(ast) == A_SUBSCR) {
        arr = A_LOPG(ast);
        asd = A_ASDG(ast);
        nsubs = ASD_NDIM(asd);
        for (i = 0; i < nsubs; i++) {
          tmp = ASD_SUBS(asd, i);
          if (A_TYPEG(tmp) == A_TRIPLE)
            dim = i;
          else
            subs[i] = tmp;
        }
      }

      ix = SHD_LWB(shape1, 0);
      for (i = 0; i < count; i++) {
        int src;
        int asn;

        subs[dim] = ix;
        ix = mk_binop(OP_ADD, ix, (int)SHD_STRIDE(shape1, 0), astb.bnd.dtype);
        shp[i] = mk_id(get_temp(astb.bnd.dtype));
        src = mk_subscr(arr, subs, nsubs, eldtype);
        asn = mk_assn_stmt(shp[i], src, astb.bnd.dtype);
        (void)add_stmt(asn);
      }
      add_shape_rank(count);
      for (i = 0; i < count; i++)
        add_shape_spec(astb.bnd.one, shp[i], astb.bnd.one);
      shaper = mk_shape();
    }
    break;

  case PD_merge:
    if (count != 3) {
      E74_CNT(pdsym, count, 3, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(2);
    if (!DT_ISLOG(DDTG(SST_DTYPEG(stkp)))) { /* mask */
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }
    dtype2 = SST_DTYPEG(stkp);
    shape2 = SST_SHAPEG(stkp);

    stkp = ARG_STK(0); /* tsource */
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);

    stkp = ARG_STK(1); /* fsource */
    dtype1 = SST_DTYPEG(stkp);
    shape1 = SST_SHAPEG(stkp);
    if (DDTG(dtyper) != DDTG(dtype1)) {
      if (DTYG(dtyper) == TY_CHAR || DTYG(dtyper) == TY_NCHAR) {
        if (DTYG(dtyper) != DTYG(dtype1)) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
      } else {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    shaper = set_shape_result(shaper, shape1);
    if (shaper < 0) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    sptr = (shaper == shape1 ? SST_SYMG(ARG_STK(1)) : SST_SYMG(ARG_STK(0)));

    shaper = set_shape_result(shaper, shape2);
    if (shaper < 0) {
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }
    sptr = (shaper == shape2 ? SST_SYMG(ARG_STK(2)) : sptr);

    if (shaper && DTY(dtyper) != TY_ARRAY) {
      dtyper = get_array_dtype(SHD_NDIM(shaper), dtyper);
      ad = AD_DPTR(dtyper);
      for (i = 0; i < (int)SHD_NDIM(shaper); i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = SHD_LWB(shaper, i);
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = SHD_UPB(shaper, i);
        AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
      }
    }

    ast = ARG_AST(2);
    hpf_sym = getMergeSym((int)DDTG(dtyper), IK_ELEMENTAL);
    switch (DTYG(dtyper)) {
    case TY_CHAR:
    case TY_NCHAR:
      dtype1 = DDTG(dtyper);
      if (dtype1 == DT_ASSCHAR || dtype1 == DT_DEFERCHAR) {
        tmp = ast_intr(I_LEN, DT_INT4, 1, ARG_AST(0));
        dtype1 = get_type(2, TY_CHAR, tmp);
        if (DTY(dtyper) != TY_ARRAY) {
          dtyper = dtype1;
        } else {
          dtyper = dup_array_dtype(dtyper);
          DTY(dtyper + 1) = dtype1;
        }
      } else if (dtype1 == DT_ASSNCHAR || dtype1 == DT_DEFERCHAR) {
        tmp = ast_intr(I_LEN, DT_INT4, 1, ARG_AST(0));
        dtype1 = get_type(2, TY_NCHAR, tmp);
        if (DTY(dtyper) != TY_ARRAY) {
          dtyper = dtype1;
        } else {
          dtyper = dup_array_dtype(dtyper);
          DTY(dtyper + 1) = dtype1;
        }
      }
      arrtmp_ast = mk_id(get_ch_temp(dtyper));
      func_ast = begin_call(A_ICALL, hpf_sym, 5);
      A_OPTYPEP(func_ast, INTASTG(pdsym));
      add_arg(arrtmp_ast);
      add_arg(ARG_AST(0));
      add_arg(ARG_AST(1));
      add_arg(ast);
      add_arg(mk_cval(size_of(DDTG(A_DTYPEG(ast))), DT_INT));
      (void)add_stmt(func_ast);
      ast = arrtmp_ast;
      break;
    case TY_DERIVED:
      if (shaper)
        arrtmp_ast = mk_id(get_arr_temp(dtyper, FALSE, FALSE, FALSE));
      else
        arrtmp_ast = mk_id(get_temp(dtyper));
      func_ast = begin_call(A_ICALL, hpf_sym, 6);
      A_OPTYPEP(func_ast, INTASTG(pdsym));
      add_arg(arrtmp_ast);
      add_arg(ARG_AST(0));
      add_arg(ARG_AST(1));
      add_arg(
          mk_cval(size_of(DDTG(dtyper)), DT_INT)); /* size of derived type */
      add_arg(ast);
      add_arg(mk_cval(size_of(DDTG(A_DTYPEG(ast))), DT_INT));
      (void)add_stmt(func_ast);
      ast = arrtmp_ast;
      break;
    default:
      argt = mk_argt(4); /* space for arguments */
      ARGT_ARG(argt, 0) = ARG_AST(0);
      ARGT_ARG(argt, 1) = ARG_AST(1);
      ARGT_ARG(argt, 2) = ast;
      ARGT_ARG(argt, 3) = mk_cval(size_of(DDTG(A_DTYPEG(ast))), DT_INT);
      func_ast = mk_id(hpf_sym);
      ast = mk_func_node(A_INTR, func_ast, 4, argt);
      A_DTYPEP(ast, dtyper);
      A_OPTYPEP(ast, INTASTG(pdsym));
      if (shaper == 0)
        shaper = mkshape(dtyper);
    }
    goto expr_val;

  case PD_dsize:
    if (!XBIT(49, 0x40)) /* if xbit set, CM fortran intrinsics allowed */
      goto bad_args;
    FLANG_FALLTHROUGH;
  case PD_size:
    if (count == 0 || count > 3) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    (void)mkarg(ARG_STK(0), &dum);
    XFR_ARGAST(0);
    argt_count = 2;
    shaper = 0;
    if ((stkp = ARG_STK(2))) { /* KIND */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else {
      if (XBIT(68, 0x1) && XBIT(68, 0x2))
        dtyper = DT_INT8;
      else
        dtyper = stb.user.dt_int;
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, FALSE);
      return 0;
    }

    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtype1) != TY_ARRAY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    asumsz = 0;
    ast = ARG_AST(0);
    if (A_TYPEG(ast) == A_INTR) {
      switch (A_OPTYPEG(ast)) {
      case I_ADJUSTL: /* adjustl(string) */
      case I_ADJUSTR: /* adjustr(string) */
        /*  len is just len(string) */
        ast = ARGT_ARG(A_ARGSG(ast), 0);
        ARG_AST(0) = ast;
        break;
      }
    }
    switch (A_TYPEG(ast)) {
    case A_ID:
      asumsz = A_SPTRG(ast);
      if (SCG(asumsz) != SC_DUMMY || !ASUMSZG(asumsz))
        asumsz = 0;
      break;
    case A_MEM:
      /* elide any scalar members */
      while (TRUE) {
        sptr = A_SPTRG(A_MEMG(ast));
        if (DTY(DTYPEG(sptr)) == TY_ARRAY)
          break;
        ast = A_PARENTG(ast);
        if (A_TYPEG(ast) == A_ID)
          break;
        if (A_TYPEG(ast) == A_SUBSCR)
          break;
      }
      ARG_AST(0) = ast;
      break;
    default:
      break;
    }
    sptr = find_pointer_variable(ast);
    if (sptr && (POINTERG(sptr) || (ALLOCG(sptr) && SDSCG(sptr)) || ASSUMRANKG(sptr))) {
      if ((stkp = ARG_STK(1))) { /* dim */
        (void)mkexpr(stkp);
        XFR_ARGAST(1);
        dtype2 = SST_DTYPEG(stkp);
        if (!DT_ISINT(dtype2)) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
        ARG_AST(1) = mk_bnd_int(ARG_AST(1));
      } else
        ARG_AST(1) = astb.ptr0;

      if (XBIT(68, 0x1))
        hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
      else
        hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
      ast = begin_call(A_FUNC, hpf_sym, 2);
      A_DTYPEP(ast, dtyper);
      add_arg(ARG_AST(1));
      add_arg(check_member(ARG_AST(0), mk_id(SDSCG(sptr)))); /* rank */
      goto expr_val;
    }
    shape1 = A_SHAPEG(ARG_AST(0));
    count = SHD_NDIM(shape1);  /* rank of array arg */
    if ((stkp = ARG_STK(1))) { /* dim */
      (void)mkexpr(stkp);
      XFR_ARGAST(1);
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      if ((ast = A_ALIASG(ARG_AST(1)))) {
        /* dim is a constant */
        i = get_int_cval(A_SPTRG(ast));
        if (i < 1 || i > count) {
          error(423, 3, gbl.lineno, NULL, NULL);
          i = 1;
        }
        if (asumsz && i == count)
          error(84, 3, gbl.lineno, SYMNAME(asumsz),
                "- size of assumed size array is unknown");
        /*
         * Before computing the extent, ensure that an upper bound
         * for this dimension exists.  The upper bound may be zero
         * if the array is an argument declared in an interface
         * within a module.
         */
        if (SHD_UPB(shape1, i - 1)) {
          ast = extent_of_shape(shape1, i - 1);
          if (A_ALIASG(ast)) {
            ast = A_ALIASG(ast);
            iszval = get_isz_cval(A_SPTRG(ast));
            goto const_isz_val;
          } else {

            (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_size), stb.user.dt_int);

            goto gen_call;
          }
        }
        if (sem.interface) {
          /*
           * if this expression is rewritten (i.e., when this
           * function specified by this interface is invoked),
           * ast_rewrite() will select the size based on the
           * constant dim value.
           */

          (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_size), stb.user.dt_int);

          goto gen_call;
        }
        goto expr_val;
      }
    } else {
      if (asumsz)
        error(84, 3, gbl.lineno, SYMNAME(asumsz),
              "- size of assumed size array is unknown");
      else {
        for (i = 0; i < count; i++) {
          if (SHD_LWB(shape1, i) == 0 || A_ALIASG(SHD_LWB(shape1, i)) == 0 ||
              SHD_UPB(shape1, i) == 0 || A_ALIASG(SHD_UPB(shape1, i)) == 0 ||
              (SHD_STRIDE(shape1, i) != 0 &&
               A_ALIASG(SHD_STRIDE(shape1, i)) == 0)) {
            goto PD_size_nonconstant;
          }
        }
        ast = extent_of_shape(shape1, 0);
        for (i = 1; i < count; i++) {
          int e;
          e = extent_of_shape(shape1, i);
          if (A_ALIASG(e)) { /* should be constant, but ... */
            if (get_isz_cval(A_SPTRG(e)) <= 0) {
              ast = astb.bnd.zero;
              break;
            }
          } else
            goto PD_size_nonconstant;
          ast = mk_binop(OP_MUL, ast, e, astb.bnd.dtype);
        }
        if (A_ALIASG(ast)) { /* should be constant, but ... */
          ast = A_ALIASG(ast);
          iszval = get_isz_cval(A_SPTRG(ast));
          goto const_isz_val;
        }
      }
    PD_size_nonconstant:
      ARG_AST(1) = astb.ptr0;
    }

    (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_size), dtyper);
    break;

  case PD_allocated:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 1;
    ast = SST_ASTG(ARG_STK(0));
    if (A_TYPEG(ast) != A_ID && A_TYPEG(ast) != A_MEM) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    i = memsym_of_ast(ast);
    dtype1 = DTYPEG(i);
    if (!ALLOCG(i) || TPALLOCG(i)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    ad = AD_DPTR(dtype1);
    if (DTY(dtype1) == TY_ARRAY) {
      ad = AD_DPTR(dtype1);
      if (AD_DEFER(ad) == 0) {
        E74_CNT(pdsym, count, 1, 1);
        goto call_e74_cnt;
      }
    }
    dtyper = stb.user.dt_log;

    break;

  case PD_present:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    dont_issue_assumedsize_error = 1;
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dont_issue_assumedsize_error = 0;
    argt_count = 1;
    ast = SST_ASTG(ARG_STK(0));
    if (A_TYPEG(ast) != A_ID) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    i = A_SPTRG(ast);
    if (gbl.internal > 1 && !INTERNALG(i) && NEWARGG(i)) {
      i = NEWARGG(i);
      ARG_AST(0) = mk_id(i);
    } else if (SCG(i) != SC_DUMMY) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (!OPTARGG(i))
      error(84, 3, gbl.lineno, SYMNAME(i), "- must be an OPTIONAL argument");
    dtyper = stb.user.dt_log;

    if (DTYG(DTYPEG(i)) == TY_CHAR || DTYG(DTYPEG(i)) == TY_NCHAR)
      (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_presentc), stb.user.dt_log);
    else if (!XBIT(57, 0x80000) && POINTERG(i))
      (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_present_ptr), stb.user.dt_log);
    else
      (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    break;

  case PD_kind:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    conval = kind_of(dtype1);
    if (conval <= 0) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_selected_int_kind:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, stb.user.dt_int, FALSE);
      return 0;
    }

    ast = SST_ASTG(stkp);
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      con1 = A_SPTRG(ast);
      con1 = CONVAL2G(con1);
      conval = 4;
      if (con1 > 18 || (con1 > 9 && XBIT(57, 2)))
        conval = -1;
      else if (con1 > 9)
        conval = 8;
      else if (con1 > 4)
        conval = 4;
      else if (con1 > 2)
        conval = 2;
      else
        conval = 1;
      goto const_default_int_val; /*return default integer*/
    }
    /* nonconstant argument, call RTE_sel_int_kind(r,descr) */
    XFR_ARGAST(0);
    func_type = A_FUNC;

    hpf_sym = sym_mkfunc(mkRteRtnNm(RTE_sel_int_kind), stb.user.dt_int);

    dtyper = stb.user.dt_int;
    break;

  case PD_selected_real_kind:
#ifdef PD_ieee_selected_real_kind
  case PD_ieee_selected_real_kind:
#endif
    if (count > MAX_ARGS_NUMBER || count == MIN_ARGS_NUMBER) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, MAX_ARGS_NUMBER, KWDARGSTR(pdsym)))
      goto exit_;

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, ARGS_NUMBER, stb.user.dt_int, FALSE);
      return 0;
    }

    stkp = ARG_STK(0);
    is_constant = TRUE;
    conval = 4;
    if (!stkp) {
      ARG_AST(0) = astb.ptr0;
    } else {
      dtype1 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype1)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      XFR_ARGAST(0);
      ast = SST_ASTG(stkp);
      if (!A_ALIASG(ast)) {
        is_constant = FALSE;
      } else {
        ast = A_ALIASG(ast);
        con1 = A_SPTRG(ast);
        con1 = CONVAL2G(con1);
        if (con1 <= 6)
          conval = 4;
        else if (con1 <= 15)
          conval = 8;
        else if (con1 <= MAX_EXP_OF_QMANTISSA && !XBIT(57, 4))
          conval = 16;
        else
          conval = -1;
      }
    }
    stkp = ARG_STK(1);
    if (!stkp) {
      ARG_AST(1) = astb.ptr0;
    } else {
      dtype1 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype1)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      XFR_ARGAST(1);
      ast = SST_ASTG(stkp);
      if (!A_ALIASG(ast)) {
        is_constant = FALSE;
      } else {
        ast = A_ALIASG(ast);
        con1 = A_SPTRG(ast);
        con1 = CONVAL2G(con1);
        if (XBIT(49, 0x40000)) {
          /* Cray C90 */
          if (con1 <= 37) {
            if (conval > 0 && conval < 4)
              conval = 4;
          } else if (con1 <= 2465) {
            if (conval > 0 && conval < 8)
              conval = 8;
          } else {
            if (conval > 0)
              conval = 0;
            conval -= 2;
          }
        } else {
          /* ANSI */
          if (con1 <= 37) {
            if (conval > 0 && conval < 4)
              conval = 4;
          } else if (con1 <= 307) {
            if (conval > 0 && conval < 8)
              conval = 8;
          } else if (con1 <= 4931 && !XBIT(57, 4)) {
            if (conval > 0 && conval < 16)
              conval = 16;
          } else {
            if (conval > 0)
              conval = 0;
            conval -= 2;
          }
        }
      }
    }

    /* add radix argument (f2008) */
    stkp = ARG_STK(2);
    if (stkp == NULL) {
      ARG_AST(2) = astb.ptr0;
    } else {
      dtype1 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype1)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      XFR_ARGAST(2);
      ast = SST_ASTG(stkp);
      if (!A_ALIASG(ast)) {
        is_constant = FALSE;
      } else {
        ast = A_ALIASG(ast);
        con1 = A_SPTRG(ast);
        con1 = CONVAL2G(con1);
        if (con1 != 2)
          conval = -5;
      }
    }

    if (is_constant) {
      goto const_default_int_val; /*return default integer*/
    }
    /* nonconstant argument, call RTE_sel_int_kind(r,descr) */
    func_type = A_FUNC;

    hpf_sym = sym_mkfunc(mkRteRtnNm(RTE_sel_real_kind), stb.user.dt_int);
    dtyper = stb.user.dt_int;
    /* add radix argument, so the args is 3  (f2008) */
    argt_count = 3;
    break;

  case PD_selected_char_kind:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (DTY(dtype1) != TY_CHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, stb.user.dt_int, FALSE);
      return 0;
    }
    ast = SST_ASTG(stkp);
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      con1 = A_SPTRG(ast);
      conval = _selected_char_kind(con1);
      goto const_default_int_val; /*return default integer*/
    }
    /* nonconstant argument, call RTE_sel_char_kind(r,descr) */
    XFR_ARGAST(0);
    func_type = A_FUNC;

    hpf_sym = sym_mkfunc(mkRteRtnNm(RTE_sel_char_kinda), stb.user.dt_int);

    dtyper = stb.user.dt_int;
    break;

  case PD_new_line:
    if (count == 0 || count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = dtype1;
    ch = 10;
    conval = getstring(&ch, 1);
    goto const_return;
    break;
  case PD_is_iostat_end:
  case PD_is_iostat_eor:
    if (count < 1 || count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISINT(DDTG(dtype1))) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    ast = ARG_AST(0);
    shaper = SST_SHAPEG(stkp);
    dtyper = stb.user.dt_log; /* default logical */
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);

    if (pdtype == PD_is_iostat_end) {
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_is_iostat_end), dtyper);
    } else {
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_is_iostat_eor), dtyper);
    }
    ELEMENTALP(hpf_sym, 1);
    EXTSYMP(pdsym, hpf_sym);
    DTYPEP(hpf_sym, dtyper);

    argt_count = 1;
    ast = mk_convert(ast, DT_INT4);
    ast = mk_unop(OP_VAL, ast, DT_INT4);
    argt = mk_argt(1);
    ARGT_ARG(argt, 0) = ast;
    func_ast = mk_id(hpf_sym);
    A_DTYPEP(func_ast, dtyper);
    func_type = A_FUNC;
    ast = mk_func_node(func_type, func_ast, 1, argt);
    if (shaper)
      dtyper = dtype_with_shape(dtyper, shaper);
    A_DTYPEP(ast, dtyper);
    if (shaper == 0)
      shaper = mkshape(dtyper);

    goto expr_val;

    break;
  case PD_achar:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    /* TBD - array argument */
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (!DT_ISINT(DDTG(dtype1))) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(stkp);
    ast = ARG_AST(0);
    dtyper = DT_CHAR; /* default kind */
    if ((stkp = ARG_STK(1))) {
      dtyper = set_kind_result(stkp, DT_CHAR, TY_CHAR);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }

    if (shaper) {
      dtyper = get_array_dtype(SHD_NDIM(shaper), dtyper);
      ad = AD_DPTR(dtyper);
      for (i = 0; i < (int)SHD_NDIM(shaper); i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = SHD_LWB(shaper, i);
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = SHD_UPB(shaper, i);
        AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
      }
    } else if (A_ALIASG(ast)) {
      ch = get_int_cval(A_SPTRG(A_ALIASG(ast)));
      conval = getstring(&ch, 1);
      goto const_return;
    }
    if (DTY(dtyper) == TY_NCHAR) {
      sptr = intast_sym[I_NCHAR];
      ast = begin_call(A_INTR, sptr, 1);
      add_arg(ARG_AST(0));
      A_DTYPEP(ast, dtyper);
      A_OPTYPEP(ast, I_NCHAR);
    } else
    {
      sptr = intast_sym[I_ACHAR];
      ast = begin_call(A_INTR, sptr, 1);
      add_arg(ARG_AST(0));
      A_DTYPEP(ast, dtyper);
      A_OPTYPEP(ast, I_ACHAR);
    }
    goto expr_val;

  case PD_adjustl:
  case PD_adjustr:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    dtyper = dtype1;
    shaper = SST_SHAPEG(stkp);
    if (DTYG(dtype1) != TY_CHAR && DTYG(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    ast = ARG_AST(0);
    if (A_ALIASG(ast)) {
      if (pdtype == PD_adjustl)
        sptr = _adjustl(A_SPTRG(A_ALIASG(ast)));
      else
        sptr = _adjustr(A_SPTRG(A_ALIASG(ast)));
      goto const_str_val;
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, DDTG(dtype1), TRUE);
      return 0;
    }

    /* check if the dtype warrants an allocatable temp; if so,
     * need indicate this so that if the context is a relational
     * expression, the expression will be evaluated an assigned
     * to a temp.
     */
    (void)need_alloc_ch_temp(dtyper);
    break;

  case PD_bit_size:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtyper = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtyper)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
      conval = bits_in(dtyper);
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    goto const_kind_int_val;

  case PD_digits:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_BINT:
      conval = 7;
      break;
    case TY_SINT:
      conval = 15;
      break;
    case TY_INT:
      conval = 31;
      break;
    case TY_INT8:
      conval = 63;
      break;
    /* values for real/double taken from float.h <type>_MANT_DIG */
    case TY_REAL:
      conval = 24;
      break;
    case TY_DBLE:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 47;
      else
        conval = 53;
      break;
    case TY_QUAD:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 95;
      else
        conval = 113;
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_epsilon:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_REAL:
      val[0] = 0x34000000;
      sname = "epsilon(1.0_4)";
      goto const_real_val;
    case TY_DBLE:
      if (XBIT(49, 0x40000)) { /* C90 */
#define C90_EPSILON "0.1421085471520200e-13"
        atoxd(C90_EPSILON, &val[0], strlen(C90_EPSILON));
      } else {
        val[0] = 0x3cb00000;
        val[1] = 0;
      }
      sname = "epsilon(1.0_8)";
      goto const_dble_val;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      val[0] = 0x3f8f0000;
      val[1] = 0;
      val[2] = 0;
      val[3] = EPSILON_BIT96_127;
      sname = "epsilon(1.0_16)";
      goto const_quad_val;
#endif
    default:
      break;
    }
    E74_ARG(pdsym, 0, NULL);
    goto call_e74_arg;

  case PD_exponent:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (DTY(dtype1) == TY_REAL)
      rtlRtn = RTE_expon;
    else if(DTY(dtype1) == TY_DBLE) /* TY_DBLE */
      rtlRtn = RTE_expond;
    else
      rtlRtn = RTE_exponq;

    fsptr = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), stb.user.dt_int);
    ELEMENTALP(fsptr, 1);
    shaper = SST_SHAPEG(stkp);
    if (shaper == 0)
      dtyper = stb.user.dt_int;
    else
      dtyper = aux.dt_iarray;
    break;

  case PD_fraction:
  case PD_rrspacing:
  case PD_spacing:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (DTY(dtype1) == TY_REAL) {
      switch (pdtype) {
      case PD_fraction:
        rtlRtn = RTE_frac;
        break;
      case PD_rrspacing:
        rtlRtn = RTE_rrspacing;
        break;
      case PD_spacing:
        rtlRtn = RTE_spacing;
        break;
      default:
        interr("PD_spacing, pdtype", pdtype, 3);
      }
    } else if (DTY(dtype1) == TY_DBLE) { /* TY_DBLE */
      switch (pdtype) {
      case PD_fraction:
        rtlRtn = RTE_fracd;
        break;
      case PD_rrspacing:
        rtlRtn = RTE_rrspacingd;
        break;
      case PD_spacing:
        rtlRtn = RTE_spacingd;
        break;
      default:
        interr("PD_spacingd, pdtype", pdtype, 3);
      }
    } else {
      switch (pdtype) {
      case PD_fraction:
        rtlRtn = RTE_fracq;
        break;
      case PD_rrspacing:
        rtlRtn = RTE_rrspacingq;
        break;
      case PD_spacing:
        rtlRtn = RTE_spacingq;
        break;
      default:
        interr("PD_spacingq or PD_rrspacingq or PD_fraction, pdtype", pdtype,
               3);
      }
    }
    (void)sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtype1);
    break;

  case PD_erf:
  case PD_erfc:
  case PD_erfc_scaled:
  case PD_gamma:
  case PD_log_gamma:
  case PD_acosh:
  case PD_asinh:
  case PD_atanh:
  case PD_bessel_j0:
  case PD_bessel_j1:
  case PD_bessel_y0:
  case PD_bessel_y1:
    /* TODO: where are the names for these set? */
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    break;
  case PD_bessel_jn:
  case PD_bessel_yn:
    if (count < 2 || count > 3) {
      E74_CNT(pdsym, count, 2, 3);
      goto call_e74_cnt;
    }
    if (count == 2) {
      if (evl_kwd_args(list, 2, "n x"))
        goto exit_;

      dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
      dtype2 = DDTG(SST_DTYPEG(ARG_STK(1)));
      if (!DT_ISINT(dtype1) || !DT_ISREAL(dtype2)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      shaper = A_SHAPEG(ARG_AST(1));
      if (shaper < 0) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      if (shaper) {
        dtyper = get_array_dtype(SHD_NDIM(shaper), dtype2);
      } else {
        dtyper = dtype2;
      }

      if (DTY(dtype1) != TY_INT) {
        ast = ARG_AST(0);
        ast = mk_convert(ast, dtype1);
        ARG_AST(0) = ast;
      }
    } else if (count == 3) {
      if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
        goto exit_;

      if (!DT_ISINT(DDTG(SST_DTYPEG(ARG_STK(0)))) ||
          !DT_ISINT(DDTG(SST_DTYPEG(ARG_STK(1)))) ||
          !DT_ISREAL(DDTG(SST_DTYPEG(ARG_STK(2))))) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }

      dtype2 = DDTG(SST_DTYPEG(ARG_STK(2)));

      argt = mk_argt(4);

      sem.arrdim.ndim = 1;
      sem.arrdim.ndefer = 0;
      sem.bounds[0].lowtype = S_CONST;
      sem.bounds[0].lowb = 1;
      sem.bounds[0].lwast = 0;
      sem.bounds[0].uptype = S_EXPR;
      sem.bounds[0].upb = 0;
      sem.bounds[0].upast =
          mk_binop(OP_ADD, mk_binop(OP_SUB, ARG_AST(1), ARG_AST(0), DT_INT),
                   astb.bnd.one, DT_INT);
      dtyper = mk_arrdsc();
      DTY(dtyper + 1) = dtype2;

      shaper = mkshape(dtyper);
      arrtmp_ast = mk_id(get_arr_temp(dtyper, FALSE, FALSE, FALSE));
      ARGT_ARG(argt, 0) = arrtmp_ast;

      dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
      ARGT_ARG(argt, 1) = SST_ASTG(ARG_STK(0));
      if (DTY(dtype1) != TY_INT) {
        ast = ARG_AST(0);
        ast = mk_convert(ast, dtype1);
        ARGT_ARG(argt, 1) = ast;
      }
      dtype1 = DDTG(SST_DTYPEG(ARG_STK(1)));
      ARGT_ARG(argt, 2) = SST_ASTG(ARG_STK(1));
      if (DTY(dtype1) != TY_INT) {
        ast = ARG_AST(1);
        ast = mk_convert(ast, dtype1);
        ARGT_ARG(argt, 2) = ast;
      }

      ARGT_ARG(argt, 3) = SST_ASTG(ARG_STK(2));

      if (DTY(dtype2) == TY_REAL) {
        switch (pdtype) {
        case PD_bessel_jn:
          name = "f90_bessel_jn";
          break;
        case PD_bessel_yn:
          name = "f90_bessel_yn";
          break;
        }
      } else { /* TY_DBLE */
        switch (pdtype) {
        case PD_bessel_jn:
          name = "f90_dbessel_jn";
          break;
        case PD_bessel_yn:
          name = "f90_dbessel_yn";
          break;
        }
      }

      hpf_sym = sym_mkfunc_nodesc(name, dtyper);
      func_ast = mk_id(hpf_sym);
      A_DTYPEP(func_ast, dtyper);
      ast = mk_func_node(A_CALL, func_ast, 4, argt);
      add_stmt(ast);
      dtyper = dtype1;
      A_DTYPEP(ast, dtyper);
      A_DTYPEP(func_ast, dtyper);
      A_SHAPEP(ast, shaper);

      SST_ASTP(stktop, arrtmp_ast);
      SST_SHAPEP(stktop, shaper);
      SST_DTYPEP(stktop, dtyper);
      SST_IDP(stktop, S_EXPR);

      EXPSTP(hpf_sym, 1);
      return 1;
    }
    break;
  case PD_hypot:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    dtype2 = DDTG(SST_DTYPEG(ARG_STK(1)));
    if (!DT_ISREAL(dtype1) || !DT_ISREAL(dtype2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(ARG_STK(0));
    shape2 = SST_SHAPEG(ARG_STK(1));
    shaper = set_shape_result(shaper, shape2);
    if (shaper < 0) {
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }
    if (shaper) {
      dtyper = get_array_dtype(SHD_NDIM(shaper), dtype1);
    } else {
      dtyper = dtype1;
    }
    switch (DTY(dtype1)) {
    case TY_REAL:
      rtlRtn = RTE_hypot;
      break;
    case TY_DBLE:
      rtlRtn = RTE_hypotd;
      break;
    case TY_QUAD:
      rtlRtn = RTE_hypotq;
      break;
    default:
      break;
    }
    /* TODO: where is the call generated */
    break;

  case PD_huge:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    ast = ast_intr(I_HUGE, dtype1, 0); /* returns a constant ast */
    switch (DTY(dtype1)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
      goto const_int_ast;
    case TY_INT8:
      goto const_int8_ast;
    case TY_REAL:
      goto const_real_ast;
    case TY_DBLE:
      goto const_dble_ast;
    case TY_QUAD:
      goto const_quad_ast;
    default:
      break;
    }
    E74_ARG(pdsym, 0, NULL);
    goto call_e74_arg;

  case PD_iachar:
    if (count == 0 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (DTYG(dtype1) != TY_CHAR && DTYG(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(stkp);
    if ((stkp = ARG_STK(1))) { /* KIND */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper = stb.user.dt_int;
    }
    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }
    if (shaper) {
      dtyper = get_array_dtype(SHD_NDIM(shaper), dtyper);
      ad = AD_DPTR(dtyper);
      for (i = 0; i < (int)SHD_NDIM(shaper); i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = SHD_LWB(shaper, i);
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = SHD_UPB(shaper, i);
        AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
      }
    } else if (A_ALIASG(ARG_AST(0))) { /* constant character */
      conval = stb.n_base[CONVAL1G(A_SPTRG(A_ALIASG(ARG_AST(0))))] & 0xff;
      conval = cngcon(conval, DT_INT4, dtyper);
      goto const_return;
    }

    break;

  case PD_ceiling:
  case PD_floor:
    if (count < 1 || count > 2) {
        E74_CNT(pdsym, count, 0, 2);
        goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    dtyper = dtype1; /* initial result of call is type of argument */

    /* for this case dtype2 is used for conversion; the actual floor/ceiling 
     * calls we use return real, but the Fortran declaration returns int. 
     * We need to calculate final type for conversion to correct int kind.
     */

    if ((stkp = ARG_STK(1))) { /* kind */
      dtype2 = set_kind_result(stkp, DT_INT, TY_INT); 
      if (!dtype2) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      dtype2 = stb.user.dt_int;  /* default return type for floor/ceiling */
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtype2, TRUE);
      return 0;
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    (void)mkexpr(ARG_STK(0));
    shaper = SST_SHAPEG(ARG_STK(0));
    XFR_ARGAST(0);
    argt_count = 1;
    if (ARG_STK(1)) {
      (void)mkexpr(ARG_STK(1));
      argt_count = 2;
      ARG_AST(1) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_aint:
  case PD_anint:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    if (SST_ISNONDECC(stkp))
      cngtyp(stkp, DT_INT);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if ((stkp = ARG_STK(1))) { /* kind */
      dtyper = set_kind_result(stkp, DT_REAL, TY_REAL);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else
      dtyper = dtype1; /* result is type of argument */
    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    (void)mkexpr(ARG_STK(0));
    shaper = SST_SHAPEG(ARG_STK(0));
    XFR_ARGAST(0);
    argt_count = 1;
    if (ARG_STK(1)) {
      (void)mkexpr(ARG_STK(1));
      argt_count = 2;
      ARG_AST(1) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_int:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0);
    stkp1 = ARG_STK(1);

    if (stkp1) { /* kind */
      dtyper = set_kind_result(stkp1, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper = stb.user.dt_int; /* default integer*/
    }

    if (SST_ISNONDECC(stkp) || SST_DTYPEG(stkp) == DT_DWORD)
      cngtyp(stkp, dtyper);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISNUMERIC(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    if (is_sst_const(stkp)) {
      con1 = get_sst_cval(stkp);
      conval = cngcon(con1, dtype1, dtyper);
      goto const_return;
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }

    (void)mkexpr(stkp);
    shaper = SST_SHAPEG(stkp);
    XFR_ARGAST(0);
    argt_count = 1;
    if (stkp1) {
      (void)mkexpr(stkp1);
      argt_count = 2;
      ARG_AST(1) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (dtyper == dtype1) {
      ast = ARG_AST(0);
      if (shaper)
        dtyper = get_array_dtype(1, dtyper);
      goto expr_val;
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_nint:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    if (SST_ISNONDECC(stkp))
      cngtyp(stkp, DT_INT);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = stb.user.dt_int;  /* default int */
    if ((stkp = ARG_STK(1))) { /* kind */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    stkp = ARG_STK(0);
    if (is_sst_const(stkp)) {
      con1 = get_sst_cval(stkp);
      switch (DTY(dtype1)) {
      case TY_REAL:
        num1[0] = CONVAL2G(stb.flt0);
        if (xfcmp(con1, num1[0]) >= 0) {
          INT fv2_23 = 0x4b000000;
          if (xfcmp(con1, fv2_23) >= 0)
            xfadd(con1, CONVAL2G(stb.flt0), &res[0]);
          else
            xfadd(con1, CONVAL2G(stb.flthalf), &res[0]);
        } else {
          INT fvm2_23 = 0xcb000000;
          if (xfcmp(con1, fvm2_23) <= 0)
            xfsub(con1, CONVAL2G(stb.flt0), &res[0]);
          else
            xfsub(con1, CONVAL2G(stb.flthalf), &res[0]);
        }
        break;
      case TY_DBLE:
        if (const_fold(OP_CMP, con1, stb.dbl0, DT_REAL8) >= 0) {
          INT dv2_52[2] = {0x43300000, 0x00000000};
          INT d2_52;
          d2_52 = getcon(dv2_52, DT_DBLE);
          if (const_fold(OP_CMP, con1, d2_52, DT_REAL8) >= 0)
            res[0] = const_fold(OP_ADD, con1, stb.dbl0, DT_REAL8);
          else
            res[0] = const_fold(OP_ADD, con1, stb.dblhalf, DT_REAL8);
        } else {
          INT dvm2_52[2] = {0xc3300000, 0x00000000};
          INT dm2_52;
          dm2_52 = getcon(dvm2_52, DT_DBLE);
          if (const_fold(OP_CMP, con1, dm2_52, DT_REAL8) >= 0)
            res[0] = const_fold(OP_SUB, con1, stb.dblhalf, DT_REAL8);
          else
            res[0] = const_fold(OP_SUB, con1, stb.dbl0, DT_REAL8);
        }
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        if (const_fold(OP_CMP, con1, stb.quad0, DT_QUAD) >= 0) {
          /* 0x406f0000:IEEE754_QUAD_BIAS  + 112 */
          INT qv4_112[4] = {0x406f0000, 0x00000000, 0x00000000, 0x00000000};
          INT q4_112;
          q4_112 = getcon(qv4_112, DT_QUAD);
          if (const_fold(OP_CMP, con1, q4_112, DT_QUAD) >= 0)
            res[0] = const_fold(OP_ADD, con1, stb.quad0, DT_QUAD);
          else
            res[0] = const_fold(OP_ADD, con1, stb.quadhalf, DT_QUAD);
        } else {
          INT qvm4_112[QVM4_SIZE] = {MMAX_MANTI_BIT0_31, MMAX_MANTI_BIT32_63,
                                     MMAX_MANTI_BIT64_95, MMAX_MANTI_BIT96_127};
          INT qm4_112;
          qm4_112 = getcon(qvm4_112, DT_QUAD);
          if (const_fold(OP_CMP, con1, qm4_112, DT_QUAD) >= 0)
            res[0] = const_fold(OP_SUB, con1, stb.quadhalf, DT_QUAD);
          else
            res[0] = const_fold(OP_SUB, con1, stb.quad0, DT_QUAD);
        }
        break;
#endif
      }
      conval = cngcon(res[0], dtype1, dtyper);
      goto const_return;
    }
    (void)mkexpr(ARG_STK(0));
    shaper = SST_SHAPEG(ARG_STK(0));
    XFR_ARGAST(0);
    argt_count = 1;
    if (ARG_STK(1)) {
      (void)mkexpr(ARG_STK(1));
      argt_count = 2;
      ARG_AST(1) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_cmplx:
    if (count < 1 || count > 3) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0);
    stkp1 = ARG_STK(1);
    stkp2 = ARG_STK(2);

    if (stkp2) { /* kind */
      dtyper = set_kind_result(stkp2, DT_CMPLX, TY_CMPLX);
      dtype1 = (dtyper == DT_CMPLX16) ? DT_REAL8
#ifdef TARGET_SUPPORTS_QUADFP
             : (dtyper == DT_QCMPLX)  ? DT_QUAD
#endif
                                      : DT_REAL4;
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper = stb.user.dt_cmplx; /* default complex */
      dtype1 = stb.user.dt_real;  /* default real    */
    }

    /* f2003 says that a boz literal can appear as an argument to
     * the real, dble, cmplx, and dcmplx intrinsics and its value
     * is used as the respective internal respresentation
     */
    if (SST_ISNONDECC(stkp) || SST_DTYPEG(stkp) == DT_DWORD)
      cngtyp(stkp, dtype1);
    if (stkp1 && (SST_ISNONDECC(stkp1) || SST_DTYPEG(stkp1) == DT_DWORD))
      cngtyp(stkp1, dtype1);

    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISNUMERIC(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    if (is_sst_const(stkp) && (!stkp1 || is_sst_const(stkp1))) {
      con1 = get_sst_cval(stkp);
      con1 = cngcon(con1, dtype1, dtyper);
      if (stkp1) {
        con2 = get_sst_cval(stkp1);
        con2 = cngcon(con2, DDTG(SST_DTYPEG(stkp1)), dtyper);
        num1[0] = CONVAL1G(con1);
        num1[1] = CONVAL1G(con2);
        conval = getcon(num1, dtyper);
      } else
        conval = con1;
      goto const_return;
    }
    (void)mkexpr(stkp);
    shaper = SST_SHAPEG(stkp);
    XFR_ARGAST(0);
    if (stkp1) {
      (void)mkexpr(stkp1);
      if (shaper == 0 && SST_SHAPEG(stkp1))
        shaper = SST_SHAPEG(stkp1);
      XFR_ARGAST(1);
    } else {
      ARG_AST(1) = 0;
    }
    argt_count = 3;
    ARG_AST(2) = 0;
    if (stkp2) { /* kind is present */
      (void)mkexpr(stkp2);
      ARG_AST(2) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_real:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0);
    stkp1 = ARG_STK(1);

    if (stkp1) { /* kind */
      dtyper = set_kind_result(stkp1, DT_REAL, TY_REAL);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      switch (DTY(DDTG(SST_DTYPEG(stkp)))) {
      case TY_CMPLX:
        dtyper = stb.user.dt_real;
        break;
      case TY_DCMPLX:
        dtyper = DT_REAL8;
        (void)mk_coercion_func(dtyper);
        break;
      case TY_QCMPLX:
        dtyper = DT_QUAD;
        (void)mk_coercion_func(dtyper);
        break;
      default:
        dtyper = stb.user.dt_real; /* default real */
        break;
      }
    }

    /* f2003 says that a boz literal can appear as an argument to
     * the real, dble, cmplx, and dcmplx intrinsics and its value
     * is used as the respective internal respresentation
     */
    if (SST_ISNONDECC(stkp) || SST_DTYPEG(stkp) == DT_DWORD)
      cngtyp(stkp, dtyper);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISNUMERIC(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    if (is_sst_const(stkp)) {
      con1 = get_sst_cval(stkp);
      conval = cngcon(con1, dtype1, dtyper);
      goto const_return;
    }
    (void)mkexpr(stkp);
    shaper = SST_SHAPEG(stkp);
    XFR_ARGAST(0);
    argt_count = 1;
    if (stkp1) {
      (void)mkexpr(stkp1);
      argt_count = 2;
      ARG_AST(1) = mk_cval1(target_kind(dtyper), DT_INT4);
    }
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  case PD_char:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    if (SST_ISNONDECC(stkp))
      cngtyp(stkp, DT_INT);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    dtyper = DT_CHAR;          /* default char */
    if ((stkp = ARG_STK(1))) { /* kind */
      dtyper = set_kind_result(stkp, DT_CHAR, TY_CHAR);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }

    /* If this is f90, leave the kind argument in. Otherwise issue
     * a warning and leave it -- we'll get to it someday
     */
    stkp = ARG_STK(0);
    if (is_sst_const(stkp)) {
      con1 = get_sst_cval(stkp);
      if (SST_DTYPEG(stkp) == DT_INT8)
        /* con1 is an sptr */
        con1 = get_int_cval(con1);
      ch = con1;
      conval = getstring(&ch, 1);
      goto const_return;
    }

    if (sem.dinit_data) {
      if (dtyper == DT_CHAR)
        dtyper = get_type(2, TY_CHAR, astb.i1);
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }
    (void)mkexpr(ARG_STK(0));
    shaper = SST_SHAPEG(ARG_STK(0));
    XFR_ARGAST(0);
    argt_count = 1;
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    goto gen_call;

  const_return:
    SST_IDP(stktop, S_CONST);
    SST_DTYPEP(stktop, dtyper);
    SST_CVALP(stktop, conval);
    EXPSTP(pdsym, 1); /* freeze generic or specific name */
    SST_SHAPEP(stktop, 0);
    ast = mk_cval1(conval, dtyper);
    SST_ASTP(stktop, ast);
    return conval;

    SST_IDP(stktop, S_CONST);
    SST_DTYPEP(stktop, dtyper);
    /* call cngcon to convert the constant from type native integer to the
     * user defined integer type -- if the types are the same cngcon will
     * just return.
     */
    conval = cngcon(conval, DT_INT, dtyper);
    SST_CVALP(stktop, conval);
    EXPSTP(pdsym, 1); /* freeze generic or specific name */
    SST_SHAPEP(stktop, 0);
    ast = mk_cval1(conval, dtyper);
    SST_ASTP(stktop, ast);
    return conval;

  case PD_logical:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISLOG(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = stb.user.dt_log;  /* default logical */
    if ((stkp = ARG_STK(1))) { /* kind */
      dtyper = set_kind_result(stkp, DT_LOG, TY_LOG);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    (void)mkexpr(ARG_STK(0));
    cngtyp(ARG_STK(0), dtyper);
    XFR_ARGAST(0);
    stkp = ARG_STK(0);
    shaper = SST_SHAPEG(stkp);
    ast = ARG_AST(0);
    if (dtype1 != dtyper) {
      argt_count = 1;
      goto gen_call;
    }
    goto expr_val;

  case PD_maxexponent:
  case PD_minexponent:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_REAL:
      conval = pdtype == PD_maxexponent ? 128 : -125;
      break;
    case TY_DBLE:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = pdtype == PD_maxexponent ? 8189 : -8188;
      else
        conval = pdtype == PD_maxexponent ? 1024 : -1021;
      break;
    case TY_QUAD:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = pdtype == PD_maxexponent ? 8189 : -8188;
      else
        conval = pdtype == PD_maxexponent ? 16384 : -16381;
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_nearest:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    dtype2 = DDTG(SST_DTYPEG(ARG_STK(1)));
    if (!DT_ISREAL(dtype1) || !DT_ISREAL(dtype2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shape2 = SST_SHAPEG(ARG_STK(1));
    shaper = set_shape_result(shaper, shape2);
    if (shaper < 0) {
      E74_ARG(pdsym, 2, NULL);
      goto call_e74_arg;
    }
    ast = ARG_AST(1);
    if (shape2)
      dtyper = get_array_dtype(1, DT_LOG);
    else
      dtyper = DT_LOG;
    if (DTY(dtype2) == TY_REAL) {
      ast = mk_binop(OP_GE, ast, mk_cnst(stb.flt0), dtyper);
    } else if (DTY(dtype2) == TY_DBLE) {
      ast = mk_binop(OP_GE, ast, mk_cnst(stb.dbl0), dtyper);
    } else {
      ast = mk_binop(OP_GE, ast, mk_cnst(stb.quad0), dtyper);
    }
    ARG_AST(1) = ast;
    if (DTY(dtype1) == TY_REAL)
      rtlRtn = RTE_nearest;
    else if (DTY(dtype1) == TY_DBLE) /* TY_DBLE */
      rtlRtn = RTE_nearestd;
    else
      rtlRtn = RTE_nearestq;
    (void)sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtype1);
    dtyper = SST_DTYPEG(stkp);
    if (shaper && DTY(dtyper) != TY_ARRAY)
      dtyper = get_array_dtype(1, dtyper);
    break;

  case PD_precision:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    /* values for real/double taken from float.h <type>_DIG */
    case TY_REAL:
    case TY_CMPLX:
      conval = 6;
      break;
    case TY_DBLE:
    case TY_DCMPLX:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 13;
      else
        conval = 15;
      break;
    case TY_QUAD:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 28;
      else
        conval = 33;
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_radix:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
    case TY_REAL:
    case TY_DBLE:
    case TY_QUAD:
      conval = 2;
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_range:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_BINT:
      conval = 2;
      break;
    case TY_SINT:
      conval = 4;
      break;
    case TY_INT:
      conval = 9;
      break;
    case TY_INT8:
      conval = 18;
      break;
    case TY_REAL:
    case TY_CMPLX:
      conval = 37;
      break;
    case TY_DBLE:
    case TY_DCMPLX:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 2465;
      else
        conval = 307;
      break;
    case TY_QUAD:
    case TY_QCMPLX:
      if (XBIT(49, 0x40000)) /* C90 */
        conval = 2465;
      else
        conval = 4931;
      break;
    default:
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    goto const_default_int_val; /*return default integer*/

  case PD_scale:
  case PD_set_exponent:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtype2 = SST_DTYPEG(ARG_STK(1));
    if (!DT_ISINT(DDTG(dtype2))) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    shape1 = SST_SHAPEG(ARG_STK(1));
    shaper = set_shape_result(shaper, shape1);
    if (shaper < 0) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    if (shaper && DTY(dtyper) != TY_ARRAY)
      dtyper = get_array_dtype(1, dtyper);
    if (DTY(dtype1) == TY_REAL) {
      if (pdtype == PD_scale)
        rtlRtn = RTE_scale;
      else
        rtlRtn = RTE_setexp;
    } else if (DTY(dtype1) == TY_DBLE) { /* TY_DBLE */
      if (pdtype == PD_scale)
        rtlRtn = RTE_scaled;
      else
        rtlRtn = RTE_setexpd;
    } else {
      if (pdtype == PD_scale)
        rtlRtn = RTE_scaleq;
      else
        rtlRtn = RTE_setexpq;
    }
    (void)sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtype1);
    break;

  case PD_tiny:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dtype1 = DDTG(SST_DTYPEG(ARG_STK(0)));
    switch (DTY(dtype1)) {
    case TY_REAL:
      /* 1.175494351E-38 */
      val[0] = 0x00800000; /* was 0x00400000 */
      sname = "tiny(1.0_4)";
      goto const_real_val;
    case TY_DBLE:
      if (XBIT(49, 0x40000)) {            /* C90 */
#define C90_TINY "0.73344154702194e-2465" /* 0200044000000000000000 */
        atoxd(C90_TINY, &val[0], strlen(C90_TINY));
      } else {
        /* 2.22507385850720138E-308 */
        val[0] = 0x00100000; /* was 0x00080000 */
        val[1] = 0x00000000;
      }
      sname = "tiny(1.0_8)";
      if (XBIT(51, 0x10))
        goto const_dword_val;
      goto const_dble_val;
    case TY_QUAD:
      val[0] = 0x00010000;
      val[1] = 0x00000000;
      val[2] = 0x00000000;
      val[3] = MIN_QUAD_VALUE_BIT96_127;
      goto const_quad_val;
    default:
      break;
    }
    E74_ARG(pdsym, 0, NULL);
    goto call_e74_arg;

  case PD_index:
#ifdef PD_kindex
  case PD_kindex:
#endif
    if (count < 2 || count > 4) {
      E74_CNT(pdsym, count, 2, 4);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0); /* string */
    if (DTY(DDTG(SST_DTYPEG(stkp))) != TY_CHAR &&
        DTY(DDTG(SST_DTYPEG(stkp))) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    shaper = SST_SHAPEG(stkp);
    stkp = ARG_STK(1); /* substring */
    if (DTY(DDTG(SST_DTYPEG(stkp))) != TY_CHAR &&
        DTY(DDTG(SST_DTYPEG(stkp))) != TY_NCHAR) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    shape1 = SST_SHAPEG(stkp);
    shaper = set_shape_result(shaper, shape1);
    if (shaper < 0) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    if ((stkp = ARG_STK(2))) { /* back */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISLOG(DDTG(dtype2))) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      shape2 = SST_SHAPEG(stkp);
      shaper = set_shape_result(shaper, shape2);
      if (shaper < 0) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else
      ARG_AST(2) = mk_cval((INT)SCFTN_FALSE, DT_LOG);

    dtyper = stb.user.dt_int;
    if ((stkp = ARG_STK(3))) { /* kind */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }

    if (A_ALIASG(ARG_AST(0)) && A_ALIASG(ARG_AST(1)) && A_ALIASG(ARG_AST(2))) {
      conval =
          _index(A_SPTRG(A_ALIASG(ARG_AST(0))), A_SPTRG(A_ALIASG(ARG_AST(1))),
                 A_SPTRG(A_ALIASG(ARG_AST(2))));
      goto const_kind_int_val; /*return kind,default integer*/
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }

    hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_indexa), dtyper);

    argt_count = 4;
    /* pass the kind of the logical argument back */
    ARG_AST(3) = (mk_cval(size_of(DDTG(A_DTYPEG(ARG_AST(2)))), astb.bnd.dtype));

    if (shaper)
      dtyper = get_array_dtype(1, dtyper);

    break;

  case PD_repeat:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* string */
    dtype1 = SST_DTYPEG(stkp);
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    stkp = ARG_STK(1); /* ncopies */
    dtype2 = SST_DTYPEG(stkp);
    if (!DT_ISINT(dtype2)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    ast = ARG_AST(1);
    if (A_ALIASG(ARG_AST(0)) && A_ALIASG(ast)) {
      sptr = _repeat(A_SPTRG(A_ALIASG(ARG_AST(0))), A_SPTRG(A_ALIASG(ast)));
      goto const_str_val;
    }
    if (sem.dinit_data) {
      int ncopies = get_int_cval(A_SPTRG(A_ALIASG(ast)));
      int cvlen = string_length(dtype1);
      int dtypeintr =
          get_type(2, DTYG(dtype1), mk_cval(ncopies * cvlen, stb.user.dt_int));
      gen_init_intrin_call(stktop, pdsym, count, dtypeintr, FALSE);
      return 0;
    }
    ARG_AST(2) = mk_cval(size_of(DDTG(A_DTYPEG(ast))), astb.bnd.dtype);

    ast = mk_id(get_temp(DT_INT));
    if (dtype1 != DT_ASSCHAR && dtype1 != DT_ASSNCHAR) {
      tmp = DTY(dtype1 + 1);
    } else {
      sptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), DT_INT);
      tmp = begin_call(A_FUNC, sptr, 1);
      add_arg(ARG_AST(0));
    }
    tmp = mk_binop(OP_MUL, tmp, ARG_AST(1), DT_INT);
    tmp = mk_assn_stmt(ast, tmp, DT_INT);
    (void)add_stmt(tmp);

    if (DTY(dtype1) == TY_CHAR) {
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_repeata), astb.bnd.dtype);
      dtyper = get_type(2, TY_CHAR, ast);
    } else {
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_nrepeat), DT_INT);
      dtyper = get_type(2, TY_NCHAR, ast);
    }
    arrtmp_ast = mk_id(get_ch_temp(dtyper));
    func_ast = begin_call(A_CALL, hpf_sym, 4);
    add_arg(arrtmp_ast);
    add_arg(ARG_AST(0));
    add_arg(ARG_AST(1));
    add_arg(ARG_AST(2));
    (void)add_stmt(func_ast);
    ast = mk_substr(arrtmp_ast, 0, ast, dtype1);
    shaper = 0;
    goto expr_val;

  case PD_len:
    if (count == 0 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    dont_issue_assumedsize_error = 1;
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    dont_issue_assumedsize_error = 0;
    if ((stkp = ARG_STK(1))) { /* KIND */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper = stb.user.dt_int;
    }
    goto len_shared;

#ifdef PD_klen
  case PD_klen:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    dont_issue_assumedsize_error = 1;
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    dont_issue_assumedsize_error = 0;
    dtyper = DT_INT8;
#endif
  len_shared:
    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    ast = ARG_AST(0);
    if (A_TYPEG(ast) == A_INTR) {
      switch (A_OPTYPEG(ast)) {
      case I_ADJUSTL: /* adjustl(string) */
      case I_ADJUSTR: /* adjustr(string) */
        /*  len is just len(string) */
        ast = ARGT_ARG(A_ARGSG(ast), 0);
        ARG_AST(0) = ast;
        break;
      }
    }
    if (A_ALIASG(ast)) {
      conval = string_length(dtype1);
      goto const_kind_int_val; /*return dtyper integer*/
    }
    switch (A_TYPEG(ast)) {
      int clen;
      int sym = 0;
    case A_ID:
    case A_MEM:
    case A_SUBSCR:
#ifdef USELENG
      sym = memsym_of_ast(ast);
      if (A_TYPEG(ast) == A_MEM && LENG(sym) && USELENG(sym)) {
        if (SETKINDG(sym) && !USEKINDG(sym)) {
          clen = LENG(sym);
        } else {
          clen = get_len_parm_by_number(LENG(sym), ENCLDTYPEG(sym), 0);
        }
        if (clen) {
          clen = mk_member(A_PARENTG(ast), clen, ENCLDTYPEG(sym));
        } else {
          clen = DTY(dtype1 + 1);
        }
      } else
#endif
      {
        if (!sym)
          sym = memsym_of_ast(ast);
        if (ADJLENG(sym)) {
          //Convert if return type of len differs from len attribute of string
          clen = convert_int(mk_id(CVLENG(sym)),dtyper);
        } else {
          clen = DTY(dtype1 + 1);
        }
      }
      if (clen && A_ALIASG(clen)) {
        /* not assumed-size */
        conval = string_length(dtype1);
        goto const_kind_int_val; /*return dtyper integer*/
      } else if (clen) {
        ast = clen;
        goto expr_val;
      }
      break;
    }
    if (DTY(SST_DTYPEG(stkp)) == TY_ARRAY) {
      if (pdtype == PD_len) {
        hpf_sym =
            sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_lena), stb.user.dt_int);
        /*
         * need to generete the call here since gen_call assumes that
         * the type of result of the function is the type of the
         * intrinsic.
         */
        argt = mk_argt(1);
        ARGT_ARG(argt, 0) = ARG_AST(0);
        func_ast = mk_id(hpf_sym);
        ast = mk_func_node(A_FUNC, func_ast, 1, argt);
        A_DTYPEP(ast, stb.user.dt_int);
        A_DTYPEP(func_ast, stb.user.dt_int);
        if (dtyper != stb.user.dt_int)
          ast = mk_convert(ast, dtyper);
        goto expr_val;
      }
      hpf_sym = sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_lena), DT_INT8);
      func_type = A_FUNC;
    }
    argt_count = 1;
    break;

  case PD_len_trim:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    stkp = ARG_STK(0);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    shaper = SST_SHAPEG(stkp);
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    dtyper = stb.user.dt_int;
    if ((stkp = ARG_STK(1))) {
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    ast = ARG_AST(0);
    if (A_ALIASG(ast)) {
      conval = _len_trim(A_SPTRG(A_ALIASG(ast)));
      goto const_kind_int_val;
    }
    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, FALSE);
      return 0;
    }
    argt_count = 1;
    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    break;

  case PD_trim:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0);
    dtype1 = SST_DTYPEG(stkp);
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (A_ALIASG(ARG_AST(0))) {
      sptr = _trim(A_SPTRG(A_ALIASG(ARG_AST(0))));
      goto const_str_val;
    }
    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtype1, FALSE);
      return 0;
    }
    if (DTY(dtype1) == TY_CHAR)
      dtyper = DT_ASSCHAR;
    else
      dtyper = DT_ASSNCHAR;
    /* check if the dtype warrants an allocatable temp; if so,
     * need indicate this so that if the context is a relational
     * expression, the expression will be evaluated an assigned
     * to a temp.
     */
    (void)need_alloc_ch_temp(dtyper);
    break;

  case PD_transfer:
    if (count < 2 || count > 3) {
      E74_CNT(pdsym, count, 2, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;

    stkp = ARG_STK(1); /* mold */
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);

    if ((stkp = ARG_STK(2))) { /* size */
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }

    if (sem.dinit_data) {
      /* If the result is array-valued, we need to determine its type. */
      if (shaper != 0 || stkp != NULL) {
        int size_ast;
        ISZ_T size;

        if (stkp != NULL)
          size_ast = ARG_AST(2); /* use size */
        else {
          /* No size specified.
           * Make result big enough to hold all of source.
           */
          size = size_of(DDTG(dtyper));
          size = (size_of(SST_DTYPEG(ARG_STK(0))) + size - 1) / size;
          size_ast = mk_isz_cval(size, astb.bnd.dtype);
        }
        add_shape_rank(1);
        add_shape_spec(astb.bnd.one, size_ast, astb.bnd.one);
        shaper = mk_shape();
        if (DTY(dtyper) != TY_ARRAY)
          dtyper = get_array_dtype(1, dtyper);
        dtyper = dtype_with_shape(dtyper, shaper);
        ADD_NUMELM(dtyper) = size_ast;
      }
      gen_init_intrin_call(stktop, pdsym, argt_count, dtyper, FALSE);
      return 0;
    }

    if (shaper == 0 && stkp == NULL) {
      /* result is the 'scalar' type of mold */
      shaper = 0;
      dtyper = DDTG(dtyper);
    } else {
      tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, astb.bnd.dtype, sem.sc);
      add_shape_rank(1);
      add_shape_spec(astb.bnd.one, mk_id(tmp), astb.bnd.one);
      shaper = mk_shape();
      if (DTY(dtyper) != TY_ARRAY)
        dtyper = get_array_dtype(1, dtyper);
      if (stkp != NULL)
        ast = ARG_AST(2); /* use size */
      else {
        /* else compute size by the expression
         *   (t1 + t2 - 1) / t2
         *
         * t1 = (#elements source) * size_of(element type of source)
         * t2 = size_of(element type of mold).
         */
        int t1, t2;
        t1 = size_of_ast(ARG_AST(0)); /* #elements in source */
        t1 = mk_binop(OP_MUL, t1, elem_size_of_ast(ARG_AST(0)), astb.bnd.dtype);
        t2 = elem_size_of_ast(ARG_AST(1));
        ast = mk_binop(OP_ADD, t1, t2, astb.bnd.dtype);
        ast = mk_binop(OP_SUB, ast, astb.bnd.one, astb.bnd.dtype);
        ast = mk_binop(OP_DIV, ast, t2, astb.bnd.dtype);
      }
      ast = mk_assn_stmt(mk_id(tmp), ast, astb.bnd.dtype);
      (void)add_stmt(ast);
    }
    break;

  case PD_scan:
  case PD_verify:
    if (count < 2 || count > 4) {
      E74_CNT(pdsym, count, 2, 4);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;

    stkp = ARG_STK(0); /* string */
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (DTY(dtype1) != TY_CHAR && DTY(dtype1) != TY_NCHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(stkp);

    stkp = ARG_STK(1); /* set */
    if (DTY(DDTG(SST_DTYPEG(stkp))) != DTY(dtype1)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    shape1 = SST_SHAPEG(stkp);
    shaper = set_shape_result(shaper, shape1);
    if (shaper < 0) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    dtype2 = DT_LOG;
    if ((stkp = ARG_STK(2))) { /* back */
      ast = ARG_AST(2);
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISLOG(DDTG(dtype2))) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      shape2 = SST_SHAPEG(stkp);
      shaper = set_shape_result(shaper, shape2);
      if (shaper < 0) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else
      ast = mk_cval((INT)SCFTN_FALSE, DT_LOG);

    dtyper = stb.user.dt_int;
    if ((stkp = ARG_STK(3))) { /* kind */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }

    if (DTY(dtype1) == TY_CHAR && A_ALIASG(ARG_AST(0)) &&
        A_ALIASG(ARG_AST(1)) && A_ALIASG(ast)) {
      if (pdtype == PD_verify)
        conval = _verify(A_SPTRG(A_ALIASG(ARG_AST(0))),
                         A_SPTRG(A_ALIASG(ARG_AST(1))), A_SPTRG(A_ALIASG(ast)));
      else
        conval = _scan(A_SPTRG(A_ALIASG(ARG_AST(0))),
                       A_SPTRG(A_ALIASG(ARG_AST(1))), A_SPTRG(A_ALIASG(ast)));
      goto const_kind_int_val; /*return default integer*/
    }

    if (sem.dinit_data) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, TRUE);
      return 0;
    }

    ARG_AST(2) = ast;
    ARG_AST(3) = mk_cval(size_of(DDTG(dtype2)), astb.bnd.dtype);
    argt_count = 4;
    if (DTY(dtype1) == TY_CHAR) {
      if (pdtype == PD_verify)
        rtlRtn = RTE_verifya;
      else
        rtlRtn = RTE_scana;
    } else { /* TY_NCHAR */
      if (pdtype == PD_verify)
        rtlRtn = RTE_nverify;
      else
        rtlRtn = RTE_nscan;
    }

    hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtyper);

    if (shaper)
      dtyper = get_array_dtype(1, dtyper);
    break;

  case PD_ilen:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* i */
    dtyper = SST_DTYPEG(stkp);
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (is_sst_const(stkp)) {
      /*
       * if i is nonnegative,
       *     ilen(i) = ceiling(log2(i+1))
       * if i is negative,
       *     ilen(i) = ceiling(log2(-i))
       */
      INT tmp[2];
      INT zero[2];
      INT vval[2];
      int len;

      con1 = get_sst_cval(stkp);
      if (DTY(dtype1) == TY_INT8 || DTY(dtype1) == TY_LOG8) {
        val[0] = CONVAL1G(con1);
        val[1] = CONVAL2G(con1);
      } else {
        if (con1 < 0)
          val[0] = -1;
        else
          val[0] = 0;
        val[1] = con1;
      }
      zero[0] = zero[1] = 0;
      if (cmp64(val, zero) < 0)
        neg64(val, val);
      else {
        tmp[0] = 0;
        tmp[1] = 1;
        add64(val, tmp, val);
      }
      vval[0] = val[0];
      vval[1] = val[1];
      len = -1;
      while (cmp64(val, zero) != 0) {
        ushf64((UINT *)val, -1, (UINT *)val);
        ++len;
      }
      tmp[0] = 0;
      tmp[1] = 1;
      shf64(tmp, len, tmp);
      /* if number is larger than 2**(bit pos), increase by one */
      xor64(tmp, vval, tmp);
      if (cmp64(tmp, zero) != 0)
        ++len;
      conval = len;
      goto const_default_int_val; /*return default integer*/
    }
    (void)mkexpr(ARG_STK(0));
    XFR_ARGAST(0);
    ast = ARG_AST(0);
    ARG_AST(1) = mk_cval(size_of(DDTG(A_DTYPEG(ast))), astb.bnd.dtype);
    argt_count = 2;
    fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_ilen), astb.bnd.dtype);
    EXTSYMP(pdsym, fsptr);
    break;

  case PD_processors_shape:
    if (count) {
      E74_CNT(pdsym, count, 0, 0);
      goto call_e74_cnt;
    }
    tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, DT_INT, sem.sc);
    add_shape_rank(1);
    add_shape_spec(astb.i1, mk_id(tmp), astb.i1);
    shaper = mk_shape();
    dtyper = aux.dt_iarray;

    sptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_processors_rank), stb.user.dt_int);
    ast = mk_func_node(A_FUNC, mk_id(sptr), 0, 0);
    A_DTYPEP(ast, DT_INT);

    ast = mk_assn_stmt(mk_id(tmp), ast, DT_INT);

    (void)add_stmt(ast);

    argt_count = 0;
    break;

  case PD_same_type_as:
  case PD_extends_type_of: {
    int dt1, dt2, sptrsdsc, argsptr, argsptr2, fsptr, flag, mast1, mast2;
    int decl1, decl2, flag_con;
    static int tmp = 0;

    if (count != 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    dt1 = A_DTYPEG(ARG_AST(0));
    dt2 = A_DTYPEG(ARG_AST(1));
    if (DTY(dt1) == TY_ARRAY) {
      dt1 = DTY(dt1 + 1);
    }

    if (DTY(dt2) == TY_ARRAY) {
      dt2 = DTY(dt2 + 1);
    }

    if (DTY(dt1) != TY_DERIVED) {
      /* TBD - Probably need to fix this condition when we implement
       * unlimited polymorphic types.
       */
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (DTY(dt2) != TY_DERIVED) {
      /* TBD - Probably need to fix this condition when we implement
       * unlimited polymorphic types.
       */
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }

    mast1 = ARG_AST(0);
    if (A_TYPEG(mast1) == A_SUBSCR) {
      /* To avoid lower error - bad OP type */
      mast1 = A_LOPG(mast1);
    }
    argsptr = memsym_of_ast(mast1);
    mast2 = ARG_AST(1);
    if (A_TYPEG(mast2) == A_SUBSCR) {
      /* To avoid lower error - bad OP type */
      mast2 = A_LOPG(mast2);
    }
    argsptr2 = memsym_of_ast(mast2);
    if (!CLASSG(argsptr) && !CLASSG(argsptr2)) {
      /* we can statically compute the type comparison */
      flag = eq_dtype2(dt2, dt1, (pdtype == PD_extends_type_of));
      if (flag)
        flag = gbl.ftn_true;
      ast = mk_cval1(flag, DT_INT);
      goto finish_type_cmp;
    }

    argt = mk_argt(7);
    ARGT_ARG(argt, 0) = mast1;
    ARGT_ARG(argt, 2) = mast2;

    if (CLASSG(argsptr)) {
      if (POINTERG(argsptr)) {
        flag = 1;
      } else if (ALLOCATTRG(argsptr)) {
        flag = 2;
      } else {
        flag = 0;
      }
    } else {
      flag = 0;
    }

    if (flag & (1 | 2)) {
      /* get declared type of arg1 */
      decl1 = getccsym('D', tmp++, ST_VAR);
      DTYPEP(decl1, DTYPEG(argsptr));
      decl1 = get_static_type_descriptor(decl1);
    } else {
      decl1 = 0;
    }

    if (CLASSG(argsptr) && STYPEG(argsptr) == ST_MEMBER) {
      int src_ast, std;
      int sdsc_mem = get_member_descriptor(argsptr);
      if (CLASSG(argsptr)) {
        sptrsdsc = get_type_descr_arg(gbl.currsub, argsptr);
      } else {
        sptrsdsc = getccsym('D', tmp++, ST_VAR);
        DTYPEP(sptrsdsc, DTYPEG(argsptr));
        sptrsdsc = get_static_type_descriptor(sptrsdsc);
      }
      ARGT_ARG(argt, 1) = mk_id(sptrsdsc);

      src_ast = mk_member(A_PARENTG(mast1), mk_id(sdsc_mem), A_DTYPEG(mast1));
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
      gen_set_type(mk_id(sptrsdsc), src_ast, std, FALSE, FALSE);
    } else {
      if (CLASSG(argsptr)) {
        sptrsdsc = get_type_descr_arg(gbl.currsub, argsptr);
      } else {
        sptrsdsc = getccsym('D', tmp++, ST_VAR);
        DTYPEP(sptrsdsc, DTYPEG(argsptr));
        sptrsdsc = get_static_type_descriptor(sptrsdsc);
      }
      ARGT_ARG(argt, 1) = mk_id(sptrsdsc);
    }

    if (CLASSG(argsptr2)) {
      if (POINTERG(argsptr2)) {
        flag |= 4;
      } else if (ALLOCATTRG(argsptr2)) {
        flag |= 8;
      }
    }

    if (flag & (4 | 8)) {
      /* get declared type of arg2 */
      decl2 = getccsym('D', tmp++, ST_VAR);
      DTYPEP(decl2, DTYPEG(argsptr2));
      decl2 = get_static_type_descriptor(decl2);
    } else {
      decl2 = 0;
    }
    if (CLASSG(argsptr2) && STYPEG(argsptr2) == ST_MEMBER) {
      int src_ast, std;
      int sdsc_mem = get_member_descriptor(argsptr2);
      if (CLASSG(argsptr2)) {
        sptrsdsc = get_type_descr_arg(gbl.currsub, argsptr2);
      } else {
        sptrsdsc = getccsym('D', tmp++, ST_VAR);
        DTYPEP(sptrsdsc, DTYPEG(argsptr2));
        sptrsdsc = get_static_type_descriptor(sptrsdsc);
      }

      ARGT_ARG(argt, 3) = mk_id(sptrsdsc);
      src_ast = mk_member(A_PARENTG(mast2), mk_id(sdsc_mem), A_DTYPEG(mast2));
      std = add_stmt(mk_stmt(A_CONTINUE, 0));
      gen_set_type(mk_id(sptrsdsc), src_ast, std, FALSE, FALSE);

    } else {
      if (CLASSG(argsptr2)) {
        sptrsdsc = get_type_descr_arg(gbl.currsub, argsptr2);
      } else {
        sptrsdsc = getccsym('D', tmp++, ST_VAR);
        DTYPEP(sptrsdsc, DTYPEG(argsptr2));
        sptrsdsc = get_static_type_descriptor(sptrsdsc);
      }

      ARGT_ARG(argt, 3) = mk_id(sptrsdsc);
    }

    flag_con = mk_cval1(flag, DT_INT);
    flag_con = mk_unop(OP_VAL, flag_con, DT_INT);
    ARGT_ARG(argt, 4) = flag_con;
    argt_count = 5;
    if (decl1) {
      ARGT_ARG(argt, 5) = mk_id(decl1);
      ++argt_count;
    }
    if (decl2) {
      ARGT_ARG(argt, argt_count) = mk_id(decl2);
      ++argt_count;
    }
    if (pdtype == PD_extends_type_of) {
      if (XBIT(68, 0x1)) {
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_extends_type_of), DT_LOG);
      } else
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_extends_type_of), DT_LOG);
    } else {
      if (XBIT(68, 0x1)) {
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_same_type_as), DT_LOG);

      } else
        fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_same_type_as), DT_LOG);
    }
    func_ast = mk_id(fsptr);
    ast = mk_func_node(A_FUNC, func_ast, argt_count, argt);
  finish_type_cmp:
    dtyper = stb.user.dt_log;
    A_DTYPEP(ast, dtyper);
    A_OPTYPEP(ast, INTASTG(pdsym));
    goto expr_val;
  }
  case PD_associated:
    if (count < 1 || count > 2) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    pvar = find_pointer_variable(ARG_AST(0));
    if (pvar == 0 || !POINTERG(pvar)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if ((stkp = ARG_STK(1))) { /* target */
      find_pointer_target(ARG_AST(1), &baseptr, &sptr);
      /* target may be variable, subarray, or derived-type member;
       * if variable or subarray, it must be POINTER or TARGET.
       * if derived-type member, the base must be a TARGET,
       * or the base or member must be POINTER */
      if (baseptr == 0 || (!TARGETG(baseptr) && !POINTERG(sptr) &&
                           !any_pointer_source(ARG_AST(1)))) {
        if (STYPEG(sptr) != ST_PROC || !is_procedure_ptr(pvar)) {
          E74_ARG(pdsym, 1, NULL);
          goto call_e74_arg;
        }
      }
    }
    argt_count = 2;

    dtyper = stb.user.dt_log;
    break;

  case PD_is_contiguous:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    ast = SST_ASTG(ARG_STK(0));
    if (A_TYPEG(ast) != A_ID && A_TYPEG(ast) != A_MEM) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    i = memsym_of_ast(ast);
    dtype1 = DTYPEG(i);
    if (DTY(dtype1) != TY_ARRAY) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    dtyper = stb.user.dt_log;
    if (CONTIGATTRG(i) || (!ASSUMSHPG(i) && !POINTERG(i))) {
      conval = TRUE;
      goto const_kind_int_val;
    }
    argt_count = 2;
    if (!SDSCG(i)) {
      get_static_descriptor(i);
    }
    if (STYPEG(SDSCG(i)) == ST_MEMBER) {
      ARG_AST(1) = check_member(ast, mk_id(SDSCG(i)));
    } else {
      ARG_AST(1) = mk_id(SDSCG(i));
    }
    break;

  case PD_ranf:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    argt_count = 0; /* ignore argument if present */
    dtyper = stb.user.dt_real;
    break;
  case PD_ranget:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (REFG(pdsym) && !FUNCG(pdsym))
      goto ill_call; /* can be CALL'd, but must be consistent */
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((stkp = ARG_STK(0))) { /* i */
      if (!is_varref(stkp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(stkp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(stkp);
      if (dtype2 != DT_INT) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    dtyper = DT_DWORD;
    REFP(pdsym, 1);
    FUNCP(pdsym, 1);
    break;
  case PD_ranset:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (REFG(pdsym) && !FUNCG(pdsym))
      goto ill_call; /* can be CALL'd, but must be consistent */
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((stkp = ARG_STK(0))) { /* i */
      (void)mkarg(stkp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2) && dtype2 != DT_REAL) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    dtyper = DT_DWORD;
    REFP(pdsym, 1);
    FUNCP(pdsym, 1);
    break;
  case PD_unit:
  case PD_length:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* unit number */
    (void)mkarg(stkp, &dum);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(stkp);
    if (!DT_ISINT(dtype2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (pdtype == PD_unit)
      dtyper = DT_REAL;
    else
      dtyper = DT_INT;
    break;

  case PD_int_mult_upper:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* i */
    shaper = SST_SHAPEG(stkp);
    dtyper = SST_DTYPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (dtype1 != DT_INT) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    stkp = ARG_STK(1); /* j */
    dtype2 = DDTG(SST_DTYPEG(stkp));
    if (dtype2 != DT_INT) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    shape2 = SST_SHAPEG(stkp);
    if (shaper == 0) {
      /* i is scalar - assume the shape of j */
      shaper = shape2;
      dtyper = SST_DTYPEG(stkp);
    } else if (shape2 && !conform_shape(shaper, shape2)) {
      /* both i and j have shape */
      error(155, 3, gbl.lineno, "Nonconformable arrays passed to intrinsic",
            SYMNAME(pdsym));
      goto exit_;
    }
    break;

  case PD_cot:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* x */
    shaper = SST_SHAPEG(stkp);
    dtyper = SST_DTYPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    break;

  case PD_dcot:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* x */
    shaper = SST_SHAPEG(stkp);
    dtyper = SST_DTYPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (dtype1 != DT_QUAD) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    break;

  case PD_shiftl:
  case PD_shiftr:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* i */
    shaper = SST_SHAPEG(stkp);
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISINT(dtype1) && !DT_ISREAL(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    stkp = ARG_STK(1); /* j */
    dtype1 = DDTG(SST_DTYPEG(stkp));
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (shaper)
      dtyper = get_array_dtype(SHD_NDIM(shaper), DT_DWORD);
    else
      dtyper = DT_DWORD;
    break;

  case PD_dshiftl:
  case PD_dshiftr:
    if (count != 3) {
      E74_CNT(pdsym, count, 3, 3);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    shaper = 0;
    for (i = 0; i < 3; i++) {
      stkp = ARG_STK(i); /* i, j, k */
      dtype1 = DDTG(SST_DTYPEG(stkp));
      if (!DT_ISINT(dtype1)) {
        E74_ARG(pdsym, i, NULL);
        goto call_e74_arg;
      }
      if (shaper) {
        if ((shape1 = SST_SHAPEG(stkp)) &&
            SHD_NDIM(shaper) != SHD_NDIM(shape1)) {
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
      } else
        shaper = SST_SHAPEG(stkp);
    }
    if (shaper)
      dtyper = get_array_dtype(SHD_NDIM(shaper), DT_INT);
    else
      dtyper = DT_INT;
    break;

  case PD_mask:
  /* Mask is a cray intrinsic */
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* i */
    dtyper = SST_DTYPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(stkp);
    break;

  case PD_null:
    argt_count = 0;
    if (count > 1) {
      E74_CNT(pdsym, count, 1, 2);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if (count == 1) {
      if (SST_IDG(ARG_STK(0)) == S_IDENT) {
        sptr = SST_SYMG(ARG_STK(0));
      } else {
        sptr = memsym_of_ast(SST_ASTG(ARG_STK(0)));
      }
      if (!POINTERG(sptr)) {
        errsev(458);
        if (INSIDE_STRUCT) {
          sem.dinit_error = TRUE;
        }
        return (fix_term(stktop, stb.i0));
      }
      dtyper = SST_DTYPEG(ARG_STK(0));
      shaper = SST_SHAPEG(ARG_STK(0));
      argt_count = 1;
    } else {
      dtyper = DT_WORD;
    }
    if (sem.dinit_data || INSIDE_STRUCT) {
      gen_init_intrin_call(stktop, pdsym, count, dtyper, FALSE);
      return 0;
    }
    break;

  case PD_int_ptr_kind:
    if (count) {
      E74_CNT(pdsym, count, 0, 2);
      goto call_e74_cnt;
    }
    conval = size_of(DT_PTR);
    goto const_default_int_val; /*return default integer*/

  case PD_c_sizeof:
  case PD_sizeof:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;

    (void)mkarg(ARG_STK(0), &dum);
    XFR_ARGAST(0);
    ast = ARG_AST(0);

    if (pdtype == PD_c_sizeof) {
      sptr = 0;
      if (A_TYPEG(ast) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(ast));
      } else if (A_TYPEG(ast) == A_ID) {
        sptr = A_SPTRG(ast);
      }
      if (sptr) {
        if (POINTERG(sptr) || ALLOCG(sptr) || CLASSG(sptr) || ASSUMSHPG(sptr) ||
            ASUMSZG(sptr) ||
            (DTY(DTYPEG(sptr)) == TY_DERIVED &&
             !(CFUNCG(DTY(DTYPEG(sptr) + 3)) || is_iso_cptr(DTYPEG(sptr)) ||
               is_iso_c_funptr(DTYPEG(sptr))))) {
          error(4, 3, gbl.lineno,
                "Illegal argument: must be interoperable with a C type", NULL);
          goto exit_;
        }
      }
      dtyper = 0;
      sptr = refsym(getsymbol("c_size_t"), OC_OTHER);
      if (STYPEG(sptr) == ST_PARAM) {
        dtyper =
            select_kind(DT_INT, TY_INT, get_isz_cval(A_SPTRG(CONVAL2G(sptr))));
      } else {
        dtyper = select_kind(DT_INT, TY_INT, 8);
      }
    } else {
      if (XBIT(68, 0x1) && XBIT(68, 0x2))
        dtyper = DT_INT8;
      else
        dtyper = stb.user.dt_int;
    }
    asumsz = 0;
    shaper = 0;
    dtype1 = SST_DTYPEG(ARG_STK(0));
    if (DTY(dtype1) == TY_ARRAY) {
      eltype = DTY(dtype1 + 1);
      /* FIRST, compute SIZE(arg) */
      switch (A_TYPEG(ast)) {
      case A_ID:
        asumsz = A_SPTRG(ast);
        if (SCG(asumsz) != SC_DUMMY || !ASUMSZG(asumsz))
          asumsz = 0;
        break;
      default:
        break;
      }
      sptr = find_pointer_variable(ast);
      if (sptr && (POINTERG(sptr) || (ALLOCG(sptr) && SDSCG(sptr)))) {
        /* pghpf_size(dim, static_descriptor) */
        if (XBIT(68, 0x1))
          hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
        else
          hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_sizeDsc), dtyper);
        nelems = begin_call(A_FUNC, hpf_sym, 2);
        A_DTYPEP(nelems, dtyper);
        add_arg(astb.ptr0);
        add_arg(check_member(ARG_AST(0), mk_id(SDSCG(sptr))));
        goto mul_by_eltype;
      }
      shape1 = A_SHAPEG(ARG_AST(0));
      count = SHD_NDIM(shape1); /* rank of array arg */
      if (asumsz)
        error(84, 3, gbl.lineno, SYMNAME(asumsz),
              "- size of assumed size array is unknown");
      else {
        for (i = 0; i < count; i++) {
          if (SHD_LWB(shape1, i) == 0 || A_ALIASG(SHD_LWB(shape1, i)) == 0 ||
              SHD_UPB(shape1, i) == 0 || A_ALIASG(SHD_UPB(shape1, i)) == 0 ||
              (SHD_STRIDE(shape1, i) != 0 &&
               A_ALIASG(SHD_STRIDE(shape1, i)) == 0)) {
            goto call_size_intr;
          }
        }
        nelems = extent_of_shape(shape1, 0);
        for (i = 1; i < count; i++) {
          int e;
          e = extent_of_shape(shape1, i);
          if (A_ALIASG(e)) { /* should be constant, but ... */
            if (get_isz_cval(A_SPTRG(e)) <= 0) {
              nelems = astb.bnd.zero;
              break;
            }
          } else
            goto call_size_intr;
          nelems = mk_binop(OP_MUL, nelems, e, astb.bnd.dtype);
        }
        goto mul_by_eltype;
      }
    call_size_intr:
      (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_size), dtyper);
      argt = mk_argt(2);
      ARGT_ARG(argt, 0) = ARG_AST(0);
      ARGT_ARG(argt, 1) = astb.ptr0;
      func_ast = mk_id(intast_sym[I_SIZE]);
      nelems = mk_func_node(A_INTR, func_ast, 2, argt);
      A_DTYPEP(nelems, dtyper);
      A_DTYPEP(func_ast, dtyper);
      A_OPTYPEP(nelems, I_SIZE);
    } else {
      nelems = mk_cval(1, dtyper);
      eltype = dtype1;
    }

  mul_by_eltype:
    if (eltype == DT_ASSCHAR || eltype == DT_ASSNCHAR ||
        eltype == DT_DEFERCHAR || eltype == DT_DEFERNCHAR) {
      ast = ast_intr(I_LEN, dtyper, 1, ast);
    } else
      ast = size_ast_of(ast, eltype);
    ast = mk_binop(OP_MUL, ast, nelems, dtyper);
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      iszval = get_isz_cval(A_SPTRG(ast));
      goto const_isz_val;
    }
    goto expr_val;

  case PD_storage_size:
    if (count == 0 || count > 2) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;

    if ((stkp = ARG_STK(1))) { /* KIND */
      dtyper = set_kind_result(stkp, DT_INT, TY_INT);
      if (!dtyper) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    } else {
      dtyper = stb.user.dt_int;
    }

    if (SST_IDG(ARG_STK(0)) == S_IDENT) {
      sptr = SST_SYMG(ARG_STK(0));
    } else {
      sptr = memsym_of_ast(SST_ASTG(ARG_STK(0)));
    }

    dtype1 = DTYPEG(sptr);
    eltype = DTY(dtype1) == TY_ARRAY ? DTY(dtype1 + 1) : dtype1;
    if (CLASSG(sptr)) {
      ast = gen_call_class_obj_size(sptr);
      ast = mk_binop(OP_MUL, ast, mk_cval(BITS_PER_BYTE, DT_INT8), DT_INT8);
      if (dtyper != DT_INT8)
        ast = mk_convert(ast, dtyper);
      goto expr_val;
    } else if (eltype == DT_ASSCHAR || eltype == DT_ASSNCHAR ||
               eltype == DT_DEFERCHAR || eltype == DT_DEFERNCHAR) {
      (void)mkarg(ARG_STK(0), &dum);
      XFR_ARGAST(0);
      ast = ast_intr(I_LEN, dtyper, 1, ARG_AST(0));
      ast = mk_binop(OP_MUL, ast, mk_cval(BITS_PER_BYTE, dtyper), dtyper);
      if (A_ALIASG(ast)) {
        ast = A_ALIASG(ast);
        iszval = get_isz_cval(A_SPTRG(ast));
        goto const_isz_val;
      }
      goto expr_val;
    } else {
      dtype1 = SST_DTYPEG(ARG_STK(0));
      if (DTY(dtype1) == TY_ARRAY) {
        conval = size_of(DTY(dtype1 + 1));
        conval = ALIGN(conval, alignment(dtype1));
      } else {
        conval = size_of(dtype1);
      }
      conval *= BITS_PER_BYTE;
      goto const_kind_int_val;
    }
    break;
  case PD_leadz:
  case PD_trailz:
  case PD_popcnt:
  case PD_poppar:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    stkp = ARG_STK(0); /* i */
    dtyper = SST_DTYPEG(stkp);
    dtype1 = DDTG(dtyper);
    if (!DT_ISINT(dtype1)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    shaper = SST_SHAPEG(stkp);
    break;

  case PD_compiler_version:
    if (count != 0) {
      E74_CNT(pdsym, count, 0, 0);
      goto call_e74_cnt;
    }

    sprintf(verstr, "flang %s", get_version_string());
    sptr = getstring(verstr, strlen(verstr));

    goto const_str_val;

  case PD_compiler_options:
    if (count != 0) {
      E74_CNT(pdsym, count, 0, 0);
      goto call_e74_cnt;
    }
    sname = flg.cmdline;
    if (sname != NULL) {
      for (; !isspace(*sname); ++sname)
        ;
      for (; isspace(*sname); ++sname)
        ;
      sptr = getstring(sname, strlen(sname));
    } else {
      interr("compiler_options: command line not available", 0, 3);
    }

    goto const_str_val;

  case PD_command_argument_count:
    if (count != 0) {
      E74_CNT(pdsym, count, 0, 0);
      goto call_e74_cnt;
    }
    dtyper = stb.user.dt_int;
    func_type = A_FUNC;
    argt_count = 0;
    rtlRtn = RTE_cmd_arg_cnt;
    hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), stb.user.dt_int);
    goto gen_call;

    /* cases where predeclared subroutines are called as functions */

  default:
    if ((pdsym = newsym(pdsym))) {
      SST_SYMP(stktop, pdsym);
      return mkvarref(stktop, list);
    }
    return fix_term(stktop, stb.i0);

  } /* End of switch */

  /* generate call where args stored in argpos */

gen_call:
  argt = mk_argt(argt_count + argt_extra); /* space for arguments */
  for (i = 0; i < argt_count; i++)
    ARGT_ARG(argt, i) = ARG_AST(i);
  for (; i < argt_count + argt_extra; i++)
    ARGT_ARG(argt, i) = 0;
  if (hpf_sym)
    func_ast = mk_id(hpf_sym);
  else
    func_ast = mk_id(pdsym);
  ast = mk_func_node(func_type, func_ast, argt_count + argt_extra, argt);
  if (shaper)
    dtyper = dtype_with_shape(dtyper, shaper);
  A_DTYPEP(ast, dtyper);
  A_DTYPEP(func_ast, dtyper);
  if (func_type == A_INTR)
    A_OPTYPEP(ast, INTASTG(pdsym));
  if (shaper == 0)
    shaper = mkshape(dtyper);

expr_val:
  /* dtyper, shaper, ast 'define' the result of the expression */
  A_SHAPEP(ast, shaper);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_EXPR);
  SST_DTYPEP(stktop, dtyper);
  SST_ASTP(stktop, ast);
  SST_SHAPEP(stktop, shaper);
  /* Fortran floor/ceiling take real arguments and return integer values.
   * But we want to use the same ILM/ILI as C/C++ (which return integral
   * values in real format), so as to have common optimization and 
   * vectorization techniques and routines. Thus do an explicit convert here.
   */
  if(pdtype == PD_floor || pdtype == PD_ceiling) 
    cngtyp(stktop, dtype2); /* dtype2 from PD_floor/PD_ceiling case above */
  return 1;

/*
 * result is a 32-bit constant value, but the result is any
 * integer kind.
 */
const_default_int_val:
  dtyper = stb.user.dt_int; /*return default integer*/
                            /*
                             *  FALL THRU !!!!
                             */
const_kind_int_val:
  ast = mk_cval(conval, dtyper);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, dtyper);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  if (DTY(dtyper) != TY_INT8)
    SST_CVALP(stktop, conval);
  else
    SST_CVALP(stktop, A_SPTRG(ast));
  return SST_CVALG(stktop);

const_isz_val:
  ast = mk_isz_cval(iszval, dtyper);
  EXPSTP(pdsym, 1);
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, dtyper);
  SST_ASTP(stktop, ast);
  SST_SHAPEP(stktop, 0);
  if (DTY(dtyper) == TY_INT)
    SST_CVALP(stktop, iszval);
  else
    SST_CVALP(stktop, A_SPTRG(ast));
  return iszval;
const_real_val:
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_REAL4);
  SST_CVALP(stktop, val[0]);
  SST_SHAPEP(stktop, 0);
  ast = mk_cval1(val[0], DT_REAL4);
  SST_ASTP(stktop, ast);
  sptr = A_SPTRG(ast);
  return val[0];

const_dble_val:
  tmp = getcon(val, DT_REAL8);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_REAL8);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, mk_cnst(tmp));
  return tmp;

const_dword_val:
  tmp = getcon(val, DT_DWORD);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_DWORD);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, mk_cnst(tmp));
  return tmp;

const_quad_val:
  tmp = getcon(val, DT_QUAD);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_QUAD);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, mk_cnst(tmp));
  return tmp;

const_str_val:
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DTYPEG(sptr));
  SST_CVALP(stktop, sptr);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, mk_cnst(sptr));
  return sptr;

const_int_ast:
  val[0] = CONVAL2G(A_SPTRG(ast));
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_INT4);
  SST_CVALP(stktop, val[0]);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  return val[0];

const_int8_ast:
  tmp = A_SPTRG(ast);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_INT8);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  return tmp;

const_real_ast:
  val[0] = CONVAL2G(A_SPTRG(ast));
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_REAL4);
  SST_CVALP(stktop, val[0]);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  return val[0];

const_dble_ast:
  tmp = A_SPTRG(ast);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_REAL8);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  return tmp;

const_quad_ast:
  tmp = A_SPTRG(ast);
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_CONST);
  SST_DTYPEP(stktop, DT_QUAD);
  SST_CVALP(stktop, tmp);
  SST_SHAPEP(stktop, 0);
  SST_ASTP(stktop, ast);
  return tmp;

bad_args:
  if (EXPSTG(pdsym)) {
    /* Intrinsic frozen, therefore user misused intrinsic */
    error(74, 3, gbl.lineno, SYMNAME(pdsym), CNULL);
    return (fix_term(stktop, stb.i0));
  }
  /* Intrinsic not frozen, try to interpret as a function call */
  SST_SYMP(stktop, newsym(pdsym));
  return (mkvarref(stktop, list));

call_e74_cnt:
  e74_cnt(_e74_sym, _e74_cnt, _e74_l, _e74_u);
  goto exit_;
call_e74_arg:
  e74_arg(_e74_sym, _e74_pos, _e74_kwd);
exit_:
  dont_issue_assumedsize_error = 0;
  EXPSTP(pdsym, 1); /* freeze predeclared */
  SST_IDP(stktop, S_EXPR);
  SST_DTYPEP(stktop, DT_INT);
  SST_ASTP(stktop, astb.i0);
  SST_SHAPEP(stktop, 0);
  return 1;
ill_call:
  error(84, 3, gbl.lineno, SYMNAME(pdsym),
        "- attempt to use a subroutine intrinsic as a function");
  return (fix_term(stktop, stb.i0));
}

static int
getMergeSym(int dt, int ikind)
{
  int sym;
  FtnRtlEnum rtlRtn = 0;
  int localDt = dt;

  switch (DTY(dt)) {
  case TY_BINT:
    rtlRtn = RTE_mergei1;
    break;
  case TY_SINT:
    rtlRtn = RTE_mergei2;
    break;
  case TY_INT:
    rtlRtn = RTE_mergei;
    break;
  case TY_INT8:
    rtlRtn = RTE_mergei8;
    break;
  case TY_REAL:
    rtlRtn = RTE_merger;
    break;
  case TY_DBLE:
    rtlRtn = RTE_merged;
    break;
  case TY_QUAD:
    rtlRtn = RTE_mergeq;
    break;
  case TY_CMPLX:
    rtlRtn = RTE_mergec;
    break;
  case TY_DCMPLX:
    rtlRtn = RTE_mergedc;
    break;
  case TY_BLOG:
    rtlRtn = RTE_mergel1;
    break;
  case TY_SLOG:
    rtlRtn = RTE_mergel2;
    break;
  case TY_LOG:
    rtlRtn = RTE_mergel;
    break;
  case TY_LOG8:
    rtlRtn = RTE_mergel8;
    break;
  case TY_CHAR:
    rtlRtn = RTE_mergecha;
    localDt = DT_NONE;
    break;
  case TY_DERIVED:
    rtlRtn = RTE_mergedt;
    localDt = DT_NONE;
    break;
  default:
    interr("getMergeSym:unexp.dt", DTY(dt), 3);
    break;
  }
  sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), localDt);
  INKINDP(sym, ikind);
  return sym;
}

static void
ref_pd_subr(SST *stktop, ITEM *list)
{
  int count, pdsym, dtype;
  int sptr, sptr2;
  int dtype1, dtype2;
  int shape, shape1;
  int i, dum;
  ITEM *ip1;
  int ast, lop;
  int argt;
  int argt_count;
  SST *sp;
  SST *stkp;

  /* Count the number of arguments to function */
  count = 0;
  pdsym = SST_SYMG(stktop);
  for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
    count++;
  }

  argt_count = count;
  switch (PDNUMG(pdsym)) {
  case PD_exit:
    if (count > 1 || (count == 1 && evl_kwd_args(list, 1, KWDARGSTR(pdsym))))
      goto bad_args;
    EXPSTP(pdsym, 1); /* freeze predeclared */
    ast =
        begin_call(A_CALL, sym_mkfunc_nodesc(mkRteRtnNm(RTE_exit), DT_NONE), 1);
    if (count == 0)
      add_arg(astb.i0);
    else
      add_arg(ARG_AST(0));
    SST_ASTP(stktop, ast);
    return;

  case PD_date:
    if (count != 1 || get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto bad_args;
    goto time_shared;
  case PD_time:
    if (count != 1 || get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto bad_args;
  time_shared:
    if (!is_varref(ARG_STK(0)))
      goto bad_args;
    (void)mkarg(ARG_STK(0), &dum);
    XFR_ARGAST(0);
    break;

  case PD_idate:
    if (count != 3 || get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto bad_args;
    dtype = SST_DTYPEG(ARG_STK(0));
    if ((dtype != DT_INT && dtype != DT_SINT) || !is_varref(ARG_STK(0)))
      goto bad_args;
    (void)mkarg(ARG_STK(0), &dum);
    XFR_ARGAST(0);
    for (i = 1; i <= 2; i++) {
      if (SST_DTYPEG(ARG_STK(i)) != dtype || !is_varref(ARG_STK(i)))
        goto bad_args;
      (void)mkarg(ARG_STK(i), &dum);
      XFR_ARGAST(i);
    }
    break;

  case PD_move_alloc:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    sp = ARG_STK(0);
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(0);
    sptr = memsym_of_ast(ARG_AST(0));
    if (!ALLOCATTRG(sptr)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }

    sp = ARG_STK(1);
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(1);
    sptr2 = memsym_of_ast(ARG_AST(1));
    if (!ALLOCATTRG(sptr2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (CLASSG(sptr) && !CLASSG(sptr2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    NOALLOOPTP(sptr2, 1);
    dtype1 = A_DTYPEG(ARG_AST(0));
    dtype2 = A_DTYPEG(ARG_AST(1));
    if (rank_of(dtype1) != rank_of(dtype2)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    dtype1 = DDTG(dtype1);
    dtype2 = DDTG(dtype2);
    /*
     * type compatible here means character of any length?
     */
    if (DTY(dtype1) == TY_CHAR && DTY(dtype2) == TY_CHAR)
      break;
    if (DTY(dtype1) == TY_NCHAR && DTY(dtype2) == TY_NCHAR)
      break;
    if (!eq_dtype2(dtype2, dtype1, CLASSG(sptr2))) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    break;

  case PD_mvbits:
    /* call mvbits(from, frompos, len, to, topos) */
    if (count != 5) {
      E74_CNT(pdsym, count, 5, 5);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 5, KWDARGSTR(pdsym)))
      goto exit_;

    for (i = 0; i <= 4; i++) {
      dtype = DDTG(SST_DTYPEG(ARG_STK(i)));
      if (!DT_ISINT(dtype)) {
        E74_ARG(pdsym, i, NULL);
        goto call_e74_arg;
      }
    }

    sp = ARG_STK(0); /* from */
    dtype = DDTG(SST_DTYPEG(sp));

    sp = ARG_STK(3); /* to */
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 3, NULL);
      goto call_e74_arg;
    }
    dtype1 = DDTG(SST_DTYPEG(sp));
    if (dtype != dtype1) {
      E74_ARG(pdsym, 3, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(3);
    shape = SST_SHAPEG(sp);

    for (i = 0; i <= 4; i++) {
      sp = ARG_STK(i);
      (void)mkexpr(sp);
      XFR_ARGAST(i);
      shape1 = SST_SHAPEG(sp);
      if (shape) {
        if (shape1 && !conform_shape(shape, shape1)) {
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
      } else
        shape = shape1;
    }
    break;

  case PD_date_and_time:
    if (count > 4) {
      E74_CNT(pdsym, count, 0, 4);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 4;
    for (i = 0; i <= 2; i++) /* date, time, zone */
      if ((sp = ARG_STK(i))) {
        if (!is_varref(sp) || DTY(SST_DTYPEG(sp)) != TY_CHAR) {
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
        (void)mkarg(sp, &dum);
        XFR_ARGAST(i);
      } else
        ARG_AST(i) = astb.ptr0c;
    if ((sp = ARG_STK(3))) { /* values */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(3);
      dtype = SST_DTYPEG(sp);
      if (!DT_ISINT_ARR(dtype) || rank_of_ast(ARG_AST(3)) != 1) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }
    break;

  case PD_cpu_time:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((sp = ARG_STK(0))) {
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      dtype = SST_DTYPEG(sp);
      if (!DT_ISREAL(dtype)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(0);
    }
    break;

  case PD_random_number:
    if (count != 1) {
      E74_CNT(pdsym, count, 1, 1);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((sp = ARG_STK(0))) {
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      dtype = SST_DTYPEG(sp);
      if (!DT_ISREAL(DDTG(dtype))) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(0);
      sptr = sym_of_ast(ARG_AST(0)); /* the HARVEST arg */
      ADDRTKNP(sptr, 1);
    }
    break;
  case PD_random_seed:
    if (count > 3) {
      E74_CNT(pdsym, count, 0, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;
    for (i = 1; i <= 2; i++)
      if ((sp = ARG_STK(i))) {
        if (i == 2 && !is_varref(sp)) {
          /* get argument must be variable */
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
        dtype = SST_DTYPEG(sp);
        (void)mkarg(sp, &dum);
        XFR_ARGAST(i);
        if (!DT_ISINT_ARR(dtype) || rank_of_ast(ARG_AST(i)) != 1) {
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
        if (i == 2) {
          sptr = sym_of_ast(ARG_AST(2)); /* intent OUT arg */
          ADDRTKNP(sptr, 1);
        }
      }
    if ((sp = ARG_STK(0))) {
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      dtype = SST_DTYPEG(sp);
      if (!DT_ISINT(dtype)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(0);
      sptr = sym_of_ast(ARG_AST(0)); /* intent OUT arg */
      ADDRTKNP(sptr, 1);
    }
    break;
  case PD_system_clock:
    if (count > 3) {
      E74_CNT(pdsym, count, 0, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    argt_count = 3;
    for (i = 0; i <= 2; i++)
      if ((sp = ARG_STK(i))) {
        if (!is_varref(sp)) {
          E74_ARG(pdsym, i, NULL);
          goto call_e74_arg;
        }
        dtype = SST_DTYPEG(sp);
        if (!DT_ISINT(dtype)) {
          /* f2003 allows count_rate to be integer or real */
          if (i != 1 || !DT_ISREAL(dtype)) {
            E74_ARG(pdsym, i, NULL);
            goto call_e74_arg;
          }
        }
        (void)mkarg(sp, &dum);
        XFR_ARGAST(i);
      }
    break;

  case PD_ranget:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (REFG(pdsym) && FUNCG(pdsym))
      goto ill_call; /* can be CALL'd, but must be consistent */
    if (get_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((stkp = ARG_STK(0))) { /* i */
      if (!is_varref(stkp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(stkp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(stkp);
      if (dtype2 != DT_INT) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    REFP(pdsym, 1);
    break;
  case PD_ranset:
    if (count > 1) {
      E74_CNT(pdsym, count, 0, 1);
      goto call_e74_cnt;
    }
    if (REFG(pdsym) && FUNCG(pdsym))
      goto ill_call; /* can be CALL'd, but must be consistent */
    if (evl_kwd_args(list, 1, KWDARGSTR(pdsym)))
      goto exit_;
    if ((stkp = ARG_STK(0))) { /* i */
      (void)mkarg(stkp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(stkp);
      if (!DT_ISINT(dtype2) && dtype2 != DT_REAL) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    REFP(pdsym, 1);
    break;

  case PD_getarg:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 2, KWDARGSTR(pdsym)))
      goto exit_;
    sp = ARG_STK(0); /* pos */
    (void)mkexpr(sp);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(sp);
    if (!DT_ISINT(dtype2)) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if (dtype2 != stb.user.dt_int)
      ARG_AST(0) = mk_convert(SST_ASTG(sp), stb.user.dt_int);
    sp = ARG_STK(1); /* value */
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(1);
    dtype2 = SST_DTYPEG(sp);
    if (DTY(dtype2) != TY_CHAR) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    argt_count = 2;
    break;
  case PD_get_command_argument:
    if (count < 1 || count > 4) {
      E74_CNT(pdsym, count, 1, 4);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 4, KWDARGSTR(pdsym)))
      goto exit_;
    sp = ARG_STK(0); /* number */
    (void)mkexpr(sp);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(sp);
    if (dtype2 != stb.user.dt_int) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if ((sp = ARG_STK(1))) { /* value */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(1);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_CHAR) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(2))) { /* length */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(3))) { /* status */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(3);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }
    argt_count = 4;
    break;

  case PD_execute_command_line:
    if (count < 1 || count > 5) {
      E74_CNT(pdsym, count, 1, 5);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 5, KWDARGSTR(pdsym)))
      goto exit_;
    sp = ARG_STK(0);

    if ((sp = ARG_STK(0))) { /* command */
      (void)mkarg(sp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_CHAR) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }

    if ((sp = ARG_STK(1))) { /* wait */
      (void)mkexpr(sp);
      XFR_ARGAST(1);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_log) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(2))) { /* exitstatus */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(3))) { /* cmdstat */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(3);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(4))) { /* cmdmsg */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 4, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(4);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_CHAR) {
        E74_ARG(pdsym, 4, NULL);
        goto call_e74_arg;
      }
    }
  argt_count = 5;
  break;

  case PD_get_command:
    if (count > 3) {
      E74_CNT(pdsym, count, 0, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 3, KWDARGSTR(pdsym)))
      goto exit_;
    if ((sp = ARG_STK(0))) { /* command */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(0);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_CHAR) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(1))) { /* length */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(1);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(2))) { /* status */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    argt_count = 3;
    break;

  case PD_get_environment_variable:
    if (count < 1 || count > 5) {
      E74_CNT(pdsym, count, 1, 5);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, 5, KWDARGSTR(pdsym)))
      goto exit_;
    sp = ARG_STK(0); /* name */
    (void)mkexpr(sp);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(sp);
    if (DTY(dtype2) != TY_CHAR) {
      E74_ARG(pdsym, 0, NULL);
      goto call_e74_arg;
    }
    if ((sp = ARG_STK(1))) { /* value */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(1);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_CHAR) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(2))) { /* length */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(3))) { /* status */
      if (!is_varref(sp)) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(3);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_int) {
        E74_ARG(pdsym, 3, NULL);
        goto call_e74_arg;
      }
    }
    if ((sp = ARG_STK(4))) { /* trim_name */
      (void)mkexpr(sp);
      XFR_ARGAST(4);
      dtype2 = SST_DTYPEG(sp);
      if (dtype2 != stb.user.dt_log) {
        E74_ARG(pdsym, 4, NULL);
        goto call_e74_arg;
      }
    }
    argt_count = 5;
    break;

    /* cases where predeclared functions are CALL'd */

  default:
    if ((pdsym = newsym(pdsym))) {
      SST_SYMP(stktop, pdsym);
      subr_call(stktop, list);
    }
    return;

  } /* End of switch */

  /*  generate call */

  EXPSTP(pdsym, 1);           /* freeze predeclared */
  argt = mk_argt(argt_count); /* space for arguments */
  for (i = 0; i < argt_count; i++)
    ARGT_ARG(argt, i) = ARG_AST(i);
  ast = mk_stmt(A_ICALL, 0);
  lop = mk_id(pdsym);
  A_LOPP(ast, lop);
  A_OPTYPEP(ast, INTASTG(pdsym));
  A_ARGCNTP(ast, argt_count);
  A_ARGSP(ast, argt);
  SST_ASTP(stktop, ast);
  return;

bad_args:
  /*  if a non-stanrard intrinsic, attempt to override intrinsic property */
  if (EXPSTG(pdsym)) {
    error(74, 3, gbl.lineno, SYMNAME(pdsym), CNULL);
  } else {
    /* Intrinsic not frozen, interpret as a subroutine call */
    SST_SYMP(stktop, newsym(pdsym));
    subr_call(stktop, list);
  }
  return;
call_e74_cnt:
  e74_cnt(_e74_sym, _e74_cnt, _e74_l, _e74_u);
  goto exit_;
call_e74_arg:
  e74_arg(_e74_sym, _e74_pos, _e74_kwd);
exit_:
  return;
ill_call:
  error(84, 3, gbl.lineno, SYMNAME(pdsym),
        "- attempt to CALL a function intrinsic");
}

static void
ref_intrin_subr(SST *stktop, ITEM *list)
{
  int count, pdsym;
  int sptr;
  int dtype2;
  int i, dum;
  ITEM *ip1;
  int ast, lop;
  int argt;
  int argt_count;
  SST *sp;

  /* Count the number of arguments to function */
  count = 0;
  pdsym = SST_SYMG(stktop);
  for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
    count++;
  }

  argt_count = count;
  switch (INTASTG(pdsym)) {
  case I_C_F_POINTER:
    if (count < 2 || count > 3) {
      E74_CNT(pdsym, count, 1, 3);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    sp = ARG_STK(0); /* CPTR */
    (void)mkarg(sp, &dum);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(sp);
    if (!is_iso_c_loc(ARG_AST(0))) {
      if (!is_iso_c_ptr(dtype2)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    sp = ARG_STK(1); /* fptr */
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(1);
    sptr = find_pointer_variable(ARG_AST(1));
    if (!sptr || !POINTERG(sptr)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    if ((sp = ARG_STK(2))) { /* shape */
      if (DTY(SST_DTYPEG(ARG_STK(1))) != TY_ARRAY) {
        E74_ARG(pdsym, 1, NULL);
        goto call_e74_arg;
      }
      (void)mkarg(sp, &dum);
      XFR_ARGAST(2);
      dtype2 = SST_DTYPEG(sp);
      if (DTY(dtype2) != TY_ARRAY || !DT_ISINT(DTY(dtype2 + 1))) {
        E74_ARG(pdsym, 2, NULL);
        goto call_e74_arg;
      }
    } else if (DTY(SST_DTYPEG(ARG_STK(1))) == TY_ARRAY) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    break;
  case I_C_F_PROCPOINTER:
    if (count != 2) {
      E74_CNT(pdsym, count, 2, 2);
      goto call_e74_cnt;
    }
    if (get_kwd_args(list, count, KWDARGSTR(pdsym)))
      goto bad_args;
    sp = ARG_STK(0); /* CPTR */
    (void)mkarg(sp, &dum);
    XFR_ARGAST(0);
    dtype2 = SST_DTYPEG(sp);
    if (!is_iso_c_funloc(ARG_AST(0))) {
      if (!is_iso_c_funptr(dtype2)) {
        E74_ARG(pdsym, 0, NULL);
        goto call_e74_arg;
      }
    }
    sp = ARG_STK(1); /* fptr */
    if (!is_varref(sp)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    (void)mkarg(sp, &dum);
    XFR_ARGAST(1);
    sptr = find_pointer_variable(ARG_AST(1));
    if (!sptr || !is_procedure_ptr(sptr)) {
      E74_ARG(pdsym, 1, NULL);
      goto call_e74_arg;
    }
    break;
  /* cases where predeclared functions are CALL'd */
  default:
    if ((pdsym = newsym(pdsym))) {
      SST_SYMP(stktop, pdsym);
      subr_call(stktop, list);
    }
    return;

  } /* End of switch */

  /*  generate call */

  EXPSTP(pdsym, 1);           /* freeze predeclared */
  argt = mk_argt(argt_count); /* space for arguments */
  for (i = 0; i < argt_count; i++)
    ARGT_ARG(argt, i) = ARG_AST(i);
  ast = mk_stmt(A_ICALL, 0);
  lop = mk_id(pdsym);
  A_LOPP(ast, lop);
  A_OPTYPEP(ast, INTASTG(pdsym));
  A_ARGCNTP(ast, argt_count);
  A_ARGSP(ast, argt);
  SST_ASTP(stktop, ast);
  return;

bad_args:
  /*  if a non-stanrard intrinsic, attempt to override intrinsic property */
  if (EXPSTG(pdsym)) {
    error(74, 3, gbl.lineno, SYMNAME(pdsym), CNULL);
  } else {
    /* Intrinsic not frozen, interpret as a subroutine call */
    SST_SYMP(stktop, newsym(pdsym));
    subr_call(stktop, list);
  }
  return;
call_e74_cnt:
  e74_cnt(_e74_sym, _e74_cnt, _e74_l, _e74_u);
  goto exit_;
call_e74_arg:
  e74_arg(_e74_sym, _e74_pos, _e74_kwd);
exit_:
  return;
  error(84, 3, gbl.lineno, SYMNAME(pdsym),
        "- attempt to CALL a function intrinsic");
}

/*
 * Compare the two shapes and check for conformance.  Return:
 * 1.  if one shape is scalar and the other is array, return the shape
 *     of the array;
 * 2.  if both are arrays and are not conformant, return -1 (error);
 * 3.  otherwise, return the first shape.
 */
static int
set_shape_result(int shape1, int shape2)
{
  if (shape1) {
    if (shape2 && !conform_shape(shape1, shape2))
      return -1;
  } else if (shape2)
    return shape2;

  return shape1;
}

/*
 * a kind argument is present in an intrinsic and is used to select
 * the result of the intrinsic
 */
static int
set_kind_result(SST *stkp, int dt, int ty)
{
  int kind;
  int dtype2;

  dtype2 = SST_DTYPEG(stkp);
  if (!DT_ISINT(dtype2))
    return 0; /* ERROR */
  if (is_sst_const(stkp))
    kind = cngcon(get_sst_cval(stkp), dtype2, DT_INT4);
  else if (SST_IDG(stkp) == S_EXPR) {
    int ast;
    ast = SST_ASTG(stkp);
    if (A_ALIASG(ast))
      kind = get_int_cval(A_SPTRG(ast));
    else
      return 0;
  } else {
    return 0; /* ERROR */
  }
  dtype2 = select_kind(dt, ty, kind);
  return dtype2;
}

static int
mk_array_type(int arr_spec_dt, int base_dtype)
{
  int rdtype;
  int rank;
  ADSC *ad;
  int ubound;
  int lbound;
  int i;

  ad = AD_DPTR(arr_spec_dt);
  rank = AD_NUMDIM(ad);

  sem.arrdim.ndim = rank;
  sem.arrdim.ndefer = 0;
  for (i = 0; i < rank; i++) {
    ubound = AD_UPAST(ad, i);
    lbound = AD_LWAST(ad, i);
    if (A_TYPEG(ubound) != A_CNST || A_TYPEG(lbound) != A_CNST) {
      error(87, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    }

    sem.bounds[i].lowtype = S_CONST;
    sem.bounds[i].lowb = get_int_cval(A_SPTRG(lbound));
    sem.bounds[i].lwast = 0;
    sem.bounds[i].uptype = S_CONST;
    sem.bounds[i].upb = get_int_cval(A_SPTRG(ubound));
    sem.bounds[i].upast = ubound;
  }
  rdtype = mk_arrdsc();
  DTY(rdtype + 1) = base_dtype;

  return rdtype;
}

static int
_adjustl(int string)
{
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen, result;
  int dtyper, dtype;
  INT val[2];

  dtyper = dtype = DTYPEG(string);
  if (DTY(dtyper) == TY_NCHAR) {
    string = CONVAL1G(string);
    dtype = DTYPEG(string);
  }
  p = stb.n_base + CONVAL1G(string);
  cvlen = string_length(dtype);
  origlen = cvlen;
  str = cp = getitem(0, cvlen + 1); /* +1 just in case cvlen is 0 */
  i = 0;
  /* left justify string - skip leading blanks */
  while (cvlen-- > 0) {
    ch = *p++;
    if (ch != ' ') {
      *cp++ = ch;
      break;
    }
    i++;
  }
  while (cvlen-- > 0)
    *cp++ = *p++;
  /* append blanks */
  while (i-- > 0)
    *cp++ = ' ';
  result = getstring(str, origlen);
  if (DTY(dtyper) == TY_NCHAR) {
    val[0] = result;
    val[1] = 0;
    result = getcon(val, dtyper);
  }
  return result;
}

static int
_adjustr(int string)
{
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen, result;
  int dtyper, dtype;
  INT val[2];

  dtyper = dtype = DTYPEG(string);
  if (DTY(dtyper) == TY_NCHAR) {
    string = CONVAL1G(string);
    dtype = DTYPEG(string);
  }
  p = stb.n_base + CONVAL1G(string);
  origlen = cvlen = string_length(dtype);
  str = cp = getitem(0, cvlen + 1); /* +1 just in case cvlen is 0 */
  i = 0;
  p += cvlen - 1;
  cp += cvlen - 1;
  /* right justify string - skip trailing blanks */
  while (cvlen-- > 0) {
    ch = *p--;
    if (ch != ' ') {
      *cp-- = ch;
      break;
    }
    i++;
  }
  while (cvlen-- > 0)
    *cp-- = *p--;
  /* insert blanks */
  while (i-- > 0)
    *cp-- = ' ';
  result = getstring(str, origlen);
  if (DTY(dtyper) == TY_NCHAR) {
    val[0] = result;
    val[1] = 0;
    result = getcon(val, dtyper);
  }
  return result;
}

static int
_index(int string, int substring, int back)
{
  int i, n;
  int l_string, l_substring;
  char *p_string, *p_substring;

  p_string = stb.n_base + CONVAL1G(string);
  l_string = string_length(DTYPEG(string));
  p_substring = stb.n_base + CONVAL1G(substring);
  l_substring = string_length(DTYPEG(substring));
  n = l_string - l_substring;
  if (n < 0)
    return 0;
  if (get_int_cval(back) == 0) {
    if (l_substring == 0)
      return 1;
    for (i = 0; i <= n; ++i) {
      if (p_string[i] == p_substring[0] &&
          strncmp(p_string + i, p_substring, l_substring) == 0)
        return i + 1;
    }
  } else {
    if (l_substring == 0)
      return l_string + 1;
    for (i = n; i >= 0; --i) {
      if (p_string[i] == p_substring[0] &&
          strncmp(p_string + i, p_substring, l_substring) == 0)
        return i + 1;
    }
  }
  return 0;
}

static int
_len_trim(int string)
{
  char *p;
  int i, cvlen, result;
  int dtype;

  dtype = DTYPEG(string);
  if (DTY(dtype) == TY_NCHAR) {
    string = CONVAL1G(string);
    dtype = DTYPEG(string);
  }
  p = stb.n_base + CONVAL1G(string);
  result = cvlen = string_length(dtype);
  i = 0;
  p += cvlen - 1;
  /* skip trailing blanks */
  while (cvlen-- > 0) {
    if (*p-- != ' ')
      break;
    result--;
  }
  return result;
}

static int
_repeat(int string, int ncopies)
{
  char *p, *cp;
  const char *str;
  int i, j, cvlen, newlen, result;
  int dtyper, dtype;
  INT val[2];

  ncopies = get_int_cval(ncopies);
  dtyper = dtype = DTYPEG(string);
  if (DTY(dtyper) == TY_NCHAR) {
    string = CONVAL1G(string);
    dtype = DTYPEG(string);
  }
  cvlen = string_length(dtype);
  newlen = cvlen * ncopies;
  if (newlen == 0) {
    str = "";
    result = getstring(str, strlen(str));
    if (DTY(dtyper) == TY_NCHAR) {
      dtype = get_type(2, TY_NCHAR, strlen(str));
      val[0] = result;
      val[1] = 0;
      result = getcon(val, dtype);
    }
    return result;
  }
  str = cp = getitem(0, newlen);
  j = ncopies;
  while (j-- > 0) {
    p = stb.n_base + CONVAL1G(string);
    i = cvlen;
    while (i-- > 0)
      *cp++ = *p++;
  }
  result = getstring(str, newlen);
  if (DTY(dtyper) == TY_NCHAR) {
    val[0] = result;
    val[1] = 0;
    dtyper = get_type(2, TY_NCHAR,
                      mk_cval(ncopies * string_length(dtyper), DT_INT4));
    result = getcon(val, dtyper);
  }
  return result;
}

static int
_scan(int string, int set, int back)
{
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;

  p_string = stb.n_base + CONVAL1G(string);
  l_string = string_length(DTYPEG(string));
  p_set = stb.n_base + CONVAL1G(set);
  l_set = string_length(DTYPEG(set));
  if (get_int_cval(back) == 0) {
    for (i = 0; i < l_string; ++i)
      for (j = 0; j < l_set; ++j)
        if (p_set[j] == p_string[i])
          return i + 1;
  } else {
    for (i = l_string - 1; i >= 0; --i)
      for (j = 0; j < l_set; ++j)
        if (p_set[j] == p_string[i])
          return i + 1;
  }
  return 0;
}

static int
_trim(int string)
{
  char *p, *cp;
  const char *str;
  int i, cvlen, newlen, result;
  int dtyper, dtype;
  INT val[2];

  dtyper = dtype = DTYPEG(string);
  if (DTY(dtyper) == TY_NCHAR) {
    string = CONVAL1G(string);
    dtype = DTYPEG(string);
  }
  p = stb.n_base + CONVAL1G(string);
  newlen = cvlen = string_length(dtype);
  i = 0;
  p += cvlen - 1;
  /* skip trailing blanks */
  while (cvlen-- > 0) {
    if (*p-- != ' ')
      break;
    newlen--;
  }
  if (newlen == 0) {
    str = "";
    result = getstring(str, strlen(str));
    if (DTY(dtyper) == TY_NCHAR) {
      dtype = get_type(2, TY_NCHAR, strlen(str));
      val[0] = result;
      val[1] = 0;
      result = getcon(val, dtype);
    }
    return result;
  }
  str = cp = getitem(0, newlen);
  i = newlen;
  cp += newlen - 1;
  p++;
  while (i-- > 0) {
    *cp-- = *p--;
  }
  result = getstring(str, newlen);
  if (DTY(dtyper) == TY_NCHAR) {
    i = kanji_len((const unsigned char *)str, newlen);
    dtype = get_type(2, TY_NCHAR, i);
    val[0] = result;
    val[1] = 0;
    result = getcon(val, dtype);
  }
  return result;
}

static int
_verify(int string, int set, int back)
{
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;

  p_string = stb.n_base + CONVAL1G(string);
  l_string = string_length(DTYPEG(string));
  p_set = stb.n_base + CONVAL1G(set);
  l_set = string_length(DTYPEG(set));
  if (get_int_cval(back) == 0) {
    for (i = 0; i < l_string; ++i) {
      for (j = 0; j < l_set; ++j)
        if (p_set[j] == p_string[i])
          goto contf;
      return i + 1;
    contf:;
    }
  } else {
    for (i = l_string - 1; i >= 0; --i) {
      for (j = 0; j < l_set; ++j)
        if (p_set[j] == p_string[i])
          goto contb;
      return i + 1;
    contb:;
    }
  }
  return 0;
}

/** \brief Check charset
 *
 * Make sure this routine is consistent with
 * - f90:         dinit.c:_selected_char_kind()
 * - runtime/f90: miscsup_com.c:_selected_char_kind()
 */
int
_selected_char_kind(int con)
{
  if (sem_eq_str(con, "ASCII"))
    return 1;
  else if (sem_eq_str(con, "DEFAULT"))
    return 1;
  return -1;
}

/*if astdim is constant and out of range, give error messages */
static void
check_dim_error(int shape, int astdim)
{
  int dim;
  int ndim;

  /* dim is a constant */
  if (A_ALIASG(astdim)) {
    ndim = 0;
    if (shape)
      ndim = SHD_NDIM(shape);
    dim = get_int_cval(A_SPTRG(A_ALIASG(astdim)));
    if (dim < 1 || dim > ndim) {
      error(423, 3, gbl.lineno, NULL, NULL);
    }
  }
}

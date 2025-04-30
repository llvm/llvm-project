/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Utility routines used by Fortran Semantic Analyzer.
*/

#include "gbldefs.h"
#include "global.h"
#include "gramtk.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "semstk.h"
#include "machar.h"
#include "ast.h"
#include "dinit.h"
#include "interf.h"
#include "tokdf.h"
#include "scan.h"
#include "pd.h"
#include "rte.h"
#include "state.h"
#include "ccffinfo.h"
#include "rtlRtns.h"
#include "llmputil.h" /* for llmp_create_uplevel, llmp_uplevel_set_parent */

/*
 * before the END for the subprogram is generated, check how/where
 * adjustable & assumed shape arrays were declared.
 *
 * An assumed shape array may be declared before its ENTRY, in which
 * case its assumed shape attribute needs to be set.
 *
 * The entry's assumed size, adjustable, or assumed shape flags are set
 * if there are corresponding array arguments.
 */

static void to_assumed_shape(int);
static int compute_width_dtype(DTYPE in_dtype);
static void compute_size(bool add_flag, ACL *aclp, DTYPE dtype);
static void compute_size_ast(bool add_flag, ACL *aclp, DTYPE dtype);
static DTYPE compute_size_expr(bool add_flag, ACL *aclp, DTYPE dtype);
static void compute_size_ido(bool add_flag, ACL *aclp, DTYPE dtype);
static void compute_size_sconst(bool add_flag, ACL *aclp, DTYPE dtype);
static void add_etmp(int sptr);
static void add_auto_array(int);
static void add_auto_char(int);
static void add_autobj(int);
static void put_prefix(char *, int, FILE *);
static void _dmp_acl(ACL *, int, FILE *);
static ACL *clone_init_const(ACL *original, int temp);
static ACL *clone_init_const_list(ACL *original, int temp);
static ACL *eval_init_expr_item(ACL *cur_e);
static ACL *eval_do(ACL *ido);
static INT get_default_int_val(INT);
static int ast_rewrite_indices(int ast);
static INT get_const_from_ast(int ast);
static ACL *eval_array_constructor(ACL *);
static ISZ_T get_ival(DTYPE, INT);
static ACL *get_exttype_struct_constructor(ACL *, DTYPE, ACL **);
static ACL *get_struct_default_init(int sptr);
static void add_alloc_mem_initialize(int);
static int genPolyAsn(int dest, int src, int std, int parent);
static void save_dim_specs(SEM_DIM_SPECS *aa);
static void restore_dim_specs(SEM_DIM_SPECS *aa);
static void dinit_constructor(SPTR, ACL *);
static AC_INTRINSIC map_I_to_AC(int intrin);
static AC_INTRINSIC map_PD_to_AC(int pdnum);
static bool is_illegal_expr_in_init(SPTR, int ast, DTYPE);
static int init_intrin_type_desc(int ast, SPTR sptr, int std);

/*
 * semant-created temporaries which are re-used across statements.
 */

static int temps_ctr[3];
#define TEMPS_CTR(n) (temps_ctr[n]++)
#define TEMPS_STK(n) ((sem.doif_depth << 10) + temps_ctr[n]++)

void
chk_adjarr(void)
{
  int entsym;
  int *dscptr, cnt, arg;
  LOGICAL is_first;
  int stype;

  // An RU_PROG can contain adjustable arrays in blocks.
  if (gbl.rutype != RU_FUNC && gbl.rutype != RU_SUBR && gbl.rutype != RU_PROG)
    return;
  if (gbl.currsub <= NOSYM)
    return;
  is_first = TRUE;
  /*  scan all entries. NOTE: gbl.entries not yet set  */
  for (entsym = gbl.currsub; entsym != NOSYM; entsym = SYMLKG(entsym)) {
    ADDRESSP(entsym, 0);
    dscptr = aux.dpdsc_base + DPDSCG(entsym);
    for (cnt = PARAMCTG(entsym); cnt > 0; cnt--) {
      arg = *dscptr++;
      if (arg == 0)
        continue;
      stype = STYPEG(arg);
      /*
       * continue processing if
       *     ST_ARRAY | (ST_DERIVED && TY_ARRAY)
       */
      if (stype != ST_ARRAY)
        continue;
      if (ALLOCG(arg) && !ALLOCATTRG(arg)) {
        to_assumed_shape(arg);
      }
      if (ASSUMSHPG(arg))
        ASSUMSHPP(entsym, 1);
      if (ASUMSZG(arg))
        ASUMSZP(entsym, 1);
      if (ADJARRG(arg) || RUNTIMEG(arg)) {
        ADJARRP(entsym, 1); /* tell expand adj. arrays in entry */
        if (!is_first || AFTENTG(arg))
          AFTENTP(entsym, 1); /* tell expand adj. code generated */
      }
    }
    /*
     * repeat for any adjustable arrays which are pointers-based
     * objects.
     */
    for (arg = gbl.p_adjarr; arg > NOSYM; arg = SYMLKG(arg)) {
      if (SCG(arg) == SC_BASED && (ADJARRG(arg) || RUNTIMEG(arg))) {
        ADJARRP(entsym, 1); /* tell expand adj. arrays in entry */
        if (!is_first || AFTENTG(arg))
          AFTENTP(entsym, 1); /* tell expand adj. code generated */
      }
    }
    is_first = FALSE;
  }
}

static void
to_assumed_shape(int arg)
{
  ADSC *ad;
  int ndim;
  int i;

  AFTENTP(arg, 1);
  ASSUMSHPP(arg, 1);
  if (!XBIT(54, 2) && !XBIT(58, 0x400000))
    SDSCS1P(arg, 1);
  ALLOCP(arg, 0);
  ad = AD_DPTR(DTYPEG(arg));
  AD_ASSUMSHP(ad) = 1;
  /* change the lower bound if one was not specifier. */
  ndim = AD_NUMDIM(ad);
  for (i = 0; i < ndim; i++)
    if (AD_LWBD(ad, i) == AD_LWAST(ad, i) && !XBIT(54, 2) &&
        !XBIT(58, 0x400000))
      AD_LWBD(ad, i) = astb.bnd.one;
}

/** \brief Return TRUE if the expression at 'ast' is composed of constants
           and the special symbol 'hpf_np$'. In this case, even though the
           bound is not a literal constant, it is a runtime constant.
 */
int
runtime_array(int ast)
{
  int sym;
#if DEBUG
  if (DBGBIT(3, 32))
    fprintf(gbl.dbgfil, "runtime_array(ast=%d)\n", ast);
#endif
  if (!ast)
    return TRUE;
  switch (A_TYPEG(ast)) {
  case A_ID:
    /* check for named parameter, or hpf_np$ */
    sym = A_SPTRG(ast);
    if (sym == gbl.sym_nproc) {
      return TRUE;
    }
    if (STYPEG(sym) == ST_CONST || STYPEG(sym) == ST_PARAM) {
      return TRUE;
    }
    break;
  case A_CNST:
    return TRUE;
  case A_BINOP:
    if (runtime_array(A_LOPG(ast)) && runtime_array(A_ROPG(ast))) {
      return TRUE;
    }
    break;
  case A_UNOP:
  case A_PAREN:
    if (runtime_array(A_LOPG(ast))) {
      return TRUE;
    }
    break;
  } /* switch */
#if DEBUG
  if (DBGBIT(3, 32))
    fprintf(gbl.dbgfil, "runtime_array(ast=%d): NO\n", ast);
#endif
  return FALSE;
} /* runtime_array */

/* Checks to see if array bound ast is an expression that uses a type parameter.
 * This function is mirrored in lowersym.c
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

/*----------------------------------------------------------------------
 * _mk_arrdsc:
 *  A dimension list has been parsed and all bounds information has been
 *  deposited into a few semant global data structures.  From this
 *  information, create an array record along with the array's array
 *  descriptor, and return the pointer to the array data record.
 * 	The contents of the array record are as follows:
 *
 *  Deferred / assumed-shape arrays:
 *  --------------------------------
 *  AD_LWBD == AD_LWAST, and AD_UPBD == AD_UPAST:
 *	= AST of compiler-generated temp vars, *except*:
 *	-- in a module they're undefined;
 *	-- if the lower bound is explicit (assumed shape array),
 *	   AD_LWBD = AST of lower bound = sem.bounds[i].lwast,
 *	   and the others are as above.
 *
 *  Explicit-shape arrays:
 *  ----------------------
 *  AD_LWBD / AD_UPBD:
 * 	= sem.bounds[i].lwast / upast
 *	= AST of lower / upper bound as it appears in the program.
 * 	  AD_LWBD = NULL for default lower bound.
 * 	  AD_UPBD = NULL for '*' (assumed size).
 *
 *  AD_LWAST / AD_UPAST:
 * 	= AST of lower / upper bound, *except*:
 *	-- if the bound is non-constant and we're not in a module,
 *	   it's the AST of a compiler-generated temp var;
 * 	-- AD_UPAST = NULL for '*' (assumed size).
 */
static DTYPE
_mk_arrdsc(int start_of_base)
{
  DTYPE dtype;
  ISZ_T last_mp, last_lb, last_ub, zbase;
  LOGICAL last_mp_const, last_lb_const, last_ub_const, zbase_const;
  ADSC *ad;
  int i;
  int adjarr, runtime;
  int ast;
  LOGICAL need_temps, struct_base_dim;

  need_temps = TRUE;
  /*
   * don't create any bounds temps if in a module specification or
   * if within an interface block in the module specification
   */
  if (IN_MODULE_SPEC || (IN_MODULE && sem.interface &&
                         sem.interf_base[sem.interface - 1].currsub == 0))
    need_temps = FALSE;

  /* adjustable array for interface we need temp */
  if (need_temps == FALSE && sem.interface)
    need_temps = TRUE;

  dtype = get_array_dtype(sem.arrdim.ndim, DT_NONE);
  ad = AD_DPTR(dtype);

  /* these inits shut lint up */
  last_lb_const = last_ub_const = 0;
  last_lb = last_ub = 0;

  if (sem.arrdim.ndefer) {
    /* A deferred or assumed-shape array.
     * sem.bounds[i] is defined as follows:
     *
     * bounds	lowtype	lowb	lwast	uptype	upb	upast
     * ----------------------------------------------------------
     * ( : )	 S_NULL	 --	 --	 --	 --	 --
     * (<e>: )	 S_EXPR	 --	 <ast>	 --	 --	 --
     * ----------------------------------------------------------
     */
    if (sem.arrdim.ndefer != sem.arrdim.ndim) {
      errsev(152);
      sem.arrdim.ndefer = 0;
    }
    if (need_temps) {
      /* Create temporaries for the lower and upper bounds,
       * the multipliers, and the zero base offset.
       */
      for (i = 0;; i++) {
        int lowtype;
        if (i == 0)
          last_mp = astb.bnd.one;
        else
          last_mp = mk_bnd_ast();
        AD_MLPYR(ad, i) = last_mp;

        if (i == sem.arrdim.ndim)
          break; /* -- loop exit point-- */

        lowtype = sem.bounds[i].lowtype;
        if (i < start_of_base) { /* normal case */
          if (lowtype != S_EXPR) {
            AD_LWBD(ad, i) = AD_LWAST(ad, i) = mk_bnd_ast();
          } else {
            AD_LWBD(ad, i) = sem.bounds[i].lwast;
            AD_LWAST(ad, i) = mk_bnd_ast();
            AD_ASSUMSHP(ad) = 1;
          }
          AD_UPBD(ad, i) = AD_UPAST(ad, i) = mk_bnd_ast();
        } else { /* in a structure base */
          AD_LWBD(ad, i) = sem.bounds[i].lowb;
          AD_LWAST(ad, i) = sem.bounds[i].lwast;
          AD_UPBD(ad, i) = sem.bounds[i].upb;
          AD_UPAST(ad, i) = sem.bounds[i].upast;
          if (lowtype == S_EXPR)
            AD_ASSUMSHP(ad) = 1;
        }
        last_lb = AD_LWAST(ad, i);
        last_ub = AD_UPAST(ad, i);
      }
      AD_ZBASE(ad) = mk_bnd_ast();
    } else {
      /* temps aren't created for the bounds; just propagate any
       * assumed-shape lower bounds.
       */
      for (i = 0; i < sem.arrdim.ndim; i++) {
        if (sem.bounds[i].lowtype == S_EXPR) {
          AD_LWBD(ad, i) = sem.bounds[i].lwast;
          AD_ASSUMSHP(ad) = 1;
        }
      }
    }
    for (i = 0; i < sem.arrdim.ndim; i++) {
      AD_EXTNTAST(ad, i) =
          mk_shared_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
    }
    AD_NOBOUNDS(ad) = AD_DEFER(ad) = 1;
    return dtype;
  }

  adjarr = runtime = 0;
  for (i = 0; i < sem.arrdim.ndim; i++) {
    if (sem.bounds[i].lowtype == S_EXPR) {
      if (chk_len_parm_expr(sem.bounds[i].lwast, sem.stag_dtype, 1) ||
          chk_kind_parm_expr(sem.bounds[i].lwast, sem.stag_dtype, 1, 0)) {
        need_temps = FALSE;
      }
      if (!adjarr && runtime_array(sem.bounds[i].lwast))
        ++runtime;
      else
        ++adjarr;
    }
    if (sem.bounds[i].uptype == S_EXPR) {
      if (chk_len_parm_expr(sem.bounds[i].upast, sem.stag_dtype, 1) ||
          chk_kind_parm_expr(sem.bounds[i].upast, sem.stag_dtype, 1, 0)) {
        need_temps = FALSE;
      }
      if (!adjarr && runtime_array(sem.bounds[i].upast))
        ++runtime;
      else
        ++adjarr;
    }
  }
  if (adjarr)
    AD_ADJARR(ad) = 1;

  zbase_const = TRUE;
  zbase = 0;
  for (i = 0;; i++) {
    /* compute multiplier for this dimension: */

    if (i == 0) {
      last_mp = 1;
      AD_MLPYR(ad, 0) = astb.bnd.one;
      last_mp_const = TRUE;
    } else if (last_mp_const && last_lb_const && last_ub_const) {
      last_mp = last_mp * (last_ub - last_lb + 1);
      AD_MLPYR(ad, i) = mk_isz_cval(last_mp, astb.bnd.dtype);
    } else if (!last_ub_const && last_ub == 0)
      AD_MLPYR(ad, i) = 0;
    else {
      /* don't generate an expression, use a temporary */
      if (AD_LWAST(ad, i - 1) == astb.bnd.one &&
          AD_MLPYR(ad, i - 1) == astb.bnd.one && last_ub) {
        last_mp = last_ub;
        last_mp_const = last_ub_const;
      } else {
        ast = mk_mlpyr_expr(AD_LWAST(ad, i - 1), AD_UPAST(ad, i - 1),
                            AD_MLPYR(ad, i - 1));
        last_mp = mk_shared_bnd_ast(ast);
        last_mp_const = FALSE;
      }
      AD_MLPYR(ad, i) = last_mp;
    }
    if (i == sem.arrdim.ndim)
      break; /* ----- loop exit point ----- */

    /* Process lower bound for this dimension.
     * sem.bounds[i] is defined as follows:
     *
     * lower-bound                lowtype        lowb         lwast
     * --------------------------------------------------------------
     * <NULL>                     S_CONST          1          0 (!)
     * <literal or named const>   S_CONST      <const-val>    <ast>
     * <non const expr>           S_EXPR          1 (!)       <ast>
     * --------------------------------------------------------------
     */

    struct_base_dim = (i >= start_of_base);
    last_lb = sem.bounds[i].lowb;
    last_lb_const = (sem.bounds[i].lowtype != S_EXPR);

    AD_LWBD(ad, i) = struct_base_dim ? sem.bounds[i].lowb : sem.bounds[i].lwast;

    switch (sem.bounds[i].lowtype) {
    case S_EXPR:
      if (need_temps)
        /* create a temp for this bound */
        if (A_TYPEG(sem.bounds[i].lwast) == A_CONV &&
            valid_kind_parm_expr(sem.bounds[i].lwast)) {
          AD_LWAST(ad, i) = last_lb =
              struct_base_dim ? A_LOPG(sem.bounds[i].lwast)
                              : mk_shared_bnd_ast(sem.bounds[i].lwast);
        } else
          AD_LWAST(ad, i) = last_lb =
              struct_base_dim ? mk_bnd_int(sem.bounds[i].lwast)
                              : mk_shared_bnd_ast(sem.bounds[i].lwast);
      else {
        /* don't create a temp; the bound is what was declared */
        if (A_TYPEG(sem.bounds[i].lwast) == A_CONV &&
            valid_kind_parm_expr(sem.bounds[i].lwast)) {
          AD_LWAST(ad, i) = A_LOPG(sem.bounds[i].lwast);
        } else
          AD_LWAST(ad, i) = mk_bnd_int(sem.bounds[i].lwast);
        last_lb = astb.bnd.one;
      }
      break;
    default:
      /* S_CONST: this lower bound is a constant. */
      AD_LWAST(ad, i) = (sem.bounds[i].lowb == 1)
                            ? astb.bnd.one
                            : mk_bnd_int(sem.bounds[i].lwast);
      break;
    }

    if (zbase_const && last_lb_const && last_mp_const)
      zbase = zbase + sem.bounds[i].lowb * last_mp;
    else
      zbase_const = FALSE;

    /* Process upper bound for this dimension.
     * sem.bounds[i] is defined as follows:
     *
     * upper-bound                uptype          upb        upast
     * --------------------------------------------------------------
     *  *                         S_STAR           0           0
     * <literal or named const>   S_CONST     <const-val>    <ast>
     * <non const expr>           S_EXPR         1 (!)       <ast>
     * --------------------------------------------------------------
     */
    last_ub = sem.bounds[i].upb;
    last_ub_const = (sem.bounds[i].uptype == S_CONST);

    AD_UPBD(ad, i) = struct_base_dim ? sem.bounds[i].upb
                                     : sem.bounds[i].upast; /* 0 for '*'*/
    switch (sem.bounds[i].uptype) {
    case S_EXPR:
      if (need_temps)
        /* create a temp for this bound */
        if (A_TYPEG(sem.bounds[i].upast) == A_CONV &&
            valid_kind_parm_expr(sem.bounds[i].upast)) {
          AD_UPAST(ad, i) = last_ub =
              struct_base_dim ? A_LOPG(sem.bounds[i].upast)
                              : mk_shared_bnd_ast(sem.bounds[i].upast);
        } else
          AD_UPAST(ad, i) = last_ub =
              struct_base_dim ? mk_bnd_int(sem.bounds[i].upast)
                              : mk_shared_bnd_ast(sem.bounds[i].upast);
      else {
        /* don't create a temp; the bound is what was declared */
        if (A_TYPEG(sem.bounds[i].upast) == A_CONV &&
            valid_kind_parm_expr(sem.bounds[i].upast)) {
          AD_UPAST(ad, i) = A_LOPG(sem.bounds[i].upast);
        } else
          AD_UPAST(ad, i) = mk_bnd_int(sem.bounds[i].upast);
        last_ub = astb.bnd.one;
      }
      break;
    case S_CONST:
      /* this upper bound is a constant. */
      AD_UPAST(ad, i) = mk_bnd_int(sem.bounds[i].upast);
      break;
    default:
      /* S_STAR: "*" was specified for this upper bound. */
      if (i + 1 != sem.arrdim.ndim && !is_parameter_context())
        error(48, 3, gbl.lineno, CNULL, CNULL);
      AD_UPAST(ad, i) = sem.bounds[i].upast; /* == NULL */
      AD_ASSUMSZ(ad) = 1;
      break;
    }

    AD_EXTNTAST(ad, i) = mk_shared_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
  } /* end of for loop */

  if (!need_temps && (adjarr || runtime) && sem.interface) {
    AD_NUMELM(ad) = 0;
  }

  if (zbase_const)
    AD_ZBASE(ad) = mk_isz_cval(zbase, astb.bnd.dtype);
  else {
    ast = mk_zbase_expr(ad);
    AD_ZBASE(ad) = mk_shared_bnd_ast(ast);
  }
  return dtype;
}

DTYPE
mk_arrdsc(void)
{
  return _mk_arrdsc(99);
}

static void
save_dim_specs(SEM_DIM_SPECS *aa)
{
  if (sem.in_dim) {
    BCOPY(aa, &sem.bounds[0], struct _sem_bounds, MAXDIMS);
    aa->arrdim = sem.arrdim;
  }
}

static void
restore_dim_specs(SEM_DIM_SPECS *aa)
{
  if (sem.in_dim) {
    BCOPY(&sem.bounds[0], aa, struct _sem_bounds, MAXDIMS);
    sem.arrdim = aa->arrdim;
  }
}

/** \brief Process an explicit shape list has been parsed and all bounds
          information has been deposited into a few semant global data
          structures.
    \param sptr sptr of the deferred array
    \param astparent ast of the parent pointer
    \param savedelete ?

    From this collection of information:
    + Generate assignments which define the lower and upper bounds for the
      deferred array; where the bounds are stored (asts) are located in the
      array descriptor.
    + Create a subscript AST which is used to represent the explicit shape;
      the bounds for the explicit shape use the bounds asts which are the
      destinations of the generated assignments; note that each subscript
      is represented as a triple.
 */
int
gen_defer_shape(int sptr, int astparent, int savedelete)
{
  int dt;
  int numdim;
  int subs[MAXDIMS];
  int i;
  int ast, std;
  int src, lb, ub;
  int extent;
  ITEM *itemp;

  dt = DTYPEG(sptr);
  numdim = ADD_NUMDIM(dt);
  for (i = 0; i < numdim; i++) {
    if (sem.bounds[i].lwast)
      src = sem.bounds[i].lwast;
    else
      src = astb.bnd.one;
    if (ADD_DEFER(dt)) {

      lb = ADD_LWBD(dt, i);
    } else {
      lb = ADD_LWAST(dt, i);
    }
    if (lb && A_TYPEG(lb) != A_CNST) {
      ast = mk_assn_stmt(check_member(astparent, lb), src, astb.bnd.dtype);
      std = add_stmt(ast);
      ASSNP(sym_of_ast(lb), 1);
      if (savedelete) {
        itemp = (ITEM *)getitem(1, sizeof(ITEM));
        itemp->ast = mk_id(sptr);
        itemp->t.ilm = std;
        itemp->next = sem.p_dealloc_delete;
        sem.p_dealloc_delete = itemp;
      }
    }

    if (ADD_DEFER(dt)) {
      ub = ADD_UPBD(dt, i);
    } else {
      ub = ADD_UPAST(dt, i);
    }
    if (ub && A_TYPEG(ub) != A_CNST) {
      int ext, useub;
      useub = sem.bounds[i].upast;
      if (A_TYPEG(ub) == A_ID || A_TYPEG(ub) == A_SUBSCR) {
        ast = mk_assn_stmt(check_member(astparent, ub), sem.bounds[i].upast,
                           astb.bnd.dtype);
        std = add_stmt(ast);
        ASSNP(sym_of_ast(ub), 1);
        if (savedelete) {
          itemp = (ITEM *)getitem(1, sizeof(ITEM));
          itemp->ast = mk_id(sptr);
          itemp->t.ilm = std;
          itemp->next = sem.p_dealloc_delete;
          sem.p_dealloc_delete = itemp;
        }
        useub = ub;
      }

      /* Need to make an assignment to the extent also */
      if (src == astb.bnd.one) {
        extent = useub;
      } else {
        extent =
            mk_extent_expr(check_member(astparent, lb), sem.bounds[i].upast);
      }
      ext = ADD_EXTNTAST(dt, i);
      if (A_TYPEG(ext) == A_ID || A_TYPEG(ext) == A_SUBSCR) {
        ast = mk_assn_stmt(check_member(astparent, ext),
                           check_member(astparent, extent), astb.bnd.dtype);

        std = add_stmt(ast);
        ASSNP(sym_of_ast(ADD_EXTNTAST(dt, i)), 1);
        if (savedelete) {
          itemp = (ITEM *)getitem(1, sizeof(ITEM));
          itemp->ast = mk_id(sptr);
          itemp->t.ilm = std;
          itemp->next = sem.p_dealloc_delete;
          sem.p_dealloc_delete = itemp;
        }
      }
    }
  }

  for (i = 0; i < sem.arrdim.ndim; i++) {
    if (ADD_DEFER(dt)) {
      lb = ADD_LWBD(dt, i);
      ub = ADD_UPBD(dt, i);
    } else {
      lb = ADD_LWAST(dt, i);
      ub = ADD_UPAST(dt, i);
    }
    if (lb == 0)
      lb = astb.bnd.one;
    subs[i] =
        mk_triple(check_member(astparent, lb), check_member(astparent, ub), 0);
  }
  ast = check_member(astparent, mk_id(sptr));
  ast = mk_subscr(ast, subs, sem.arrdim.ndim, (int)DTYPEG(sptr));

  return ast;
}

void
add_p_dealloc_item(int sptr)
{
  int depth;
  ITEM *itemp;

  if (sem.use_etmps) {
    /* Add allocatable temps created for an expression to the 'etmp'
     * list; they need to deallocated at the end of processing the
     * expression.
     */
    add_etmp(sptr);
    return;
  }

  /* Don't add it twice */
  for (itemp = sem.p_dealloc; itemp; itemp = itemp->next)
    if (A_SPTRG(itemp->ast) == sptr)
      return;

  for (depth = sem.doif_depth; depth > 0 && DI_ID(depth) == DI_FORALL; --depth)
    ;

  itemp = (ITEM *)getitem(1, sizeof(ITEM));
  itemp->ast = mk_id(sptr);
  itemp->next = sem.p_dealloc;
  itemp->t.conval = depth;
  sem.p_dealloc = itemp;
}


/** \brief Generate deallocates for the temporary arrays in the sem.p_delloc
 * list.
 */
void
gen_deallocate_arrays()
{
  if (sem.p_dealloc) {
    ITEM *p, *t;
    int depth;
    for (depth = sem.doif_depth; depth > 0 && DI_ID(depth) == DI_FORALL;
         --depth)
      ;
    p = NULL; /* p points to last item on remaining list */
    for (t = sem.p_dealloc; t; t = t->next) {
      if (t->t.conval == depth) {
        (void)gen_alloc_dealloc(TK_DEALLOCATE, t->ast, 0);
      } else {
        /* leave on the list */
        if (p != NULL) {
          p->next = t;
        } else {
          sem.p_dealloc = t;
        }
        p = t;
      }
    }
    /* p points to last item on remaining list, if any */
    if (p) {
      p->next = NULL;
    } else {
      sem.p_dealloc = NULL;
    }
  }
}

/*
 * For certain expression, such as if expressions, it's necessary to keep
 * a list of any allocatable temps created while processing the expression.
 * These temps, if they're deallocated at the end of the statement a memory
 * leak may occur because the statement may actually change the control
 * flow.  These temps must be deallocated at the end of the processing
 * the expression.
 */
static void
add_etmp(int sptr)
{
  ITEM *x;

  x = (ITEM *)getitem(0, sizeof(ITEM));
  x->next = sem.etmp_list;
  sem.etmp_list = x;
  x->t.sptr = sptr;
}

void
mk_defer_shape(SPTR sptr)
{
  int i;
  int dt;
  int numdim;
  ADSC *ad;

  dt = DTYPEG(sptr);
  ad = AD_DPTR(dt);
  numdim = AD_NUMDIM(ad);

  if (AD_LWAST(ad, 0))
    return;

  if (IN_MODULE_SPEC)
    MDALLOCP(sptr, 1); /* mark global allocatable array */
  else
    for (i = 0; i < numdim; i++) {
      AD_LWAST(ad, i) = mk_bnd_ast();
      AD_UPAST(ad, i) = mk_bnd_ast();
      AD_EXTNTAST(ad, i) = mk_bnd_ast();
    }
}

/*
 * return '1' if astx is a A_ID of a compiler-created temp
 */
static int
tempvar(int astx)
{
  if (A_TYPEG(astx) == A_ID &&
      (CCSYMG(A_SPTRG(astx)) || HCCSYMG(A_SPTRG(astx))))
    return 1;
  return 0;
} /* tempvar */

void
mk_assumed_shape(SPTR sptr)
{
  int i;
  DTYPE dt = DTYPEG(sptr);
  ADSC *ad = AD_DPTR(dt);
  int numdim = AD_NUMDIM(ad);

  for (i = 0; i < numdim; i++)
    if (AD_LWBD(ad, i) == AD_LWAST(ad, i) &&
        A_TYPEG(AD_LWBD(ad, i)) != A_CNST && tempvar(AD_LWBD(ad, i)) &&
        !XBIT(54, 2) && !XBIT(58, 0x400000)) {
      AD_LWBD(ad, i) = astb.bnd.one;
      AD_LWAST(ad, i) = astb.bnd.one;
    }
  AD_ASSUMSHP(ad) = 1;
  if (sem.arrdim.assumedrank) {
    AD_ASSUMRANK(ad) = 1;
  }
}

/** \brief Get a compiler array temporary of type dtype which is used to
           represent array constants.
 */
SPTR
get_arr_const(DTYPE dtype)
{
  static int iavl;
  /* stype will get changed to ST_ARRAY when it's dinit'd */
  SPTR sptr = getcctmp('c', iavl++, ST_UNKNOWN, dtype);
  SCP(sptr, SC_LOCAL);
  NODESCP(sptr, 0);
  return sptr;
}

DTYPE
select_kind(DTYPE dtype, int ty, INT kind_val)
{
  int out_dtype;

  if (kind_val < 0) {
    error(81, 3, gbl.lineno, "- KIND value must be non-negative", CNULL);
    return dtype;
  }
  out_dtype = -1;
  switch (ty) {
  case TY_INT:
  case TY_INT8:
    switch (kind_val) {
    case 8:
      if (!XBIT(57, 0x2))
        out_dtype = DT_INT8;
      break;
    case 4:
      out_dtype = DT_INT4;
      break;
    case 2:
      out_dtype = DT_SINT;
      break;
    case 1:
      out_dtype = DT_BINT;
      break;
    }
    break;
  case TY_CMPLX:
  case TY_DCMPLX:
    switch (kind_val) {
    case 16:
      if (!XBIT(57, 0x8))
        out_dtype = DT_QCMPLX;
      if (XBIT(57, 0x10)) {
        error(437, 2, gbl.lineno, "COMPLEX(16)", "COMPLEX(8)");
        out_dtype = DT_CMPLX16;
      }
      break;
    case 8:
      out_dtype = DT_CMPLX16;
      break;
    case 4:
      out_dtype = DT_CMPLX8;
      break;
    }
    break;
  case TY_REAL:
  case TY_DBLE:
    switch (kind_val) {
    case 16:
      if (!XBIT(57, 0x4))
        out_dtype = DT_QUAD;
      if (XBIT(57, 0x10)) {
        error(437, 2, gbl.lineno, "REAL(16)", "REAL(8)");
        out_dtype = DT_REAL8;
      }
      break;
    case 8:
      out_dtype = DT_REAL8;
      break;
    case 4:
      out_dtype = DT_REAL4;
      break;
    }
    break;
  case TY_LOG:
  case TY_LOG8:
    switch (kind_val) {
    case 8:
      if (!XBIT(57, 0x2))
        out_dtype = DT_LOG8;
      break;
    case 4:
      out_dtype = DT_LOG4;
      break;
    case 2:
      out_dtype = DT_SLOG;
      break;
    case 1:
      out_dtype = DT_BLOG;
      break;
    }
    break;
  case TY_CHAR:
    if (kind_val == 2)
      out_dtype = DT_NCHAR;
    if (kind_val == 1)
      out_dtype = DT_CHAR;
    break;
  default:
    error(81, 3, gbl.lineno, "- KIND = specified with a non-intrinsic type",
          CNULL);
    return dtype;
  }
  if (out_dtype == -1) {
    error(81, 3, gbl.lineno, "- KIND parameter has unknown value for data type",
          CNULL);
    return dtype;
  }
  return out_dtype;
}

typedef struct _ido_info {
  DOINFO *doinfo;
  struct _ido_info *next;
} IDO_INFO;

typedef struct {
  LOGICAL is_const;
  INT scalar_cnt;           /* # of scalar expressions */
  int aggr_cnt;             /* ast expr of # of elements in implied do or array
                               expression.  */
  int eltype;               /* element dtype */
  int zln;                  /* element dtype is zero length char */
  int arrtype;              /* array dtype record */
  int tmp;                  /* sptr of temp array */
  int tmpid;                /* id ast of array tmp */
  int subs[MAXDIMS];        /* current subscripts - used in _construct() */
  int indx[MAXDIMS];        /* current subscript value */
  INT element_cnt[MAXDIMS]; /* # of scalar expressions */
  int indx_tmpid[MAXDIMS];  /* id ast of subscripting temporary */
  int level;                /* implied do nesting level */
  int width;
  LOGICAL func_in_do;       /* func call found in ac-value-list */
  IDO_INFO *ido_list;       /* track the implied-do loop encoutered */
  int ido_level;            /* track the depth of implied-do loop encoutered */
  struct {
    DOINFO *doinfo;         /* the outermost implied-do loop
                             * on which the generated loop depends.
                             */
    int start;              /* start std of the generated loop */
    int end;                /* end std of the generated loop */
    int level;              /* the depth of the outermost implied-do loop
                             * that has generated a loop.
                             */
    int sumid;              /* the temp variable used to compute
                             * size of implied-do loop.
                             */
  } loop_stmts;
} _ACS;

static _ACS acs;
static LOGICAL _can_fold(int);
static void constructf90(int, ACL *);
static void _dinit_acl(ACL *, LOGICAL);

static int acl_array_num = 0;

static const char *_iexpr_op[] = {
    "?0?",       "ADD",      "SUB",       "MUL",  "DIV",    "EXP",  "NEG",
    "INTR_CALL", "ARRAYREF", "MEMBR_SEL", "CONV", "CAT",    "EXPK", "LEQV",
    "LNEQV",     "LOR",      "LAND",      "EQ",   "GE",     "GT",   "LE",
    "LT",        "NE",       "LNOT",      "EXPX", "TRIPLE",
};

static const char *
iexpr_op(int op)
{
  if (op <= sizeof(_iexpr_op) / sizeof(char *))
    return _iexpr_op[op];
  return "?N?";
}

/** \brief Given an allocatable array and an explicit shape list which has been
           deposited in the semant 'bounds' structure, generate assignments to
           the arrays bounds temporaries, and allocate the array.  Save the id
   ast
           of the array for an ensuing deallocate of the array.
 */
void
gen_allocate_array(int arr)
{
  int alloc_obj = gen_defer_shape(arr, 0, arr);
  if (is_deferlenchar_dtype(acs.arrtype)) {
    get_static_descriptor(arr);
    get_all_descriptors(arr);
  }
  gen_alloc_dealloc(TK_ALLOCATE, alloc_obj, 0);
  add_p_dealloc_item(arr);
}

#if DEBUG
static void
_printacl(int in_array, ACL *aclp, FILE *f)
{
  SST *stkp;
  ACL *member_aclp;
  DTYPE dtype;
  int sptr;
  int save_array_num;

  /* print a list of aclps */

  for (; aclp != NULL; aclp = aclp->next) {
    switch (aclp->id) {
    case AC_AST:
      fprintf(f, "%d:", acl_array_num);
      fprintf(f, "ast%d", aclp->u1.ast);
      dtype = A_DTYPEG(aclp->u1.ast);
      if (!in_array)
        acl_array_num += compute_width_dtype(dtype);
      break;
    case AC_EXPR:
      fprintf(f, "%d:", acl_array_num);
      stkp = aclp->u1.stkp;
      dtype = SST_DTYPEG(stkp);
      switch (SST_IDG(stkp)) {
      case S_ACONST:
        fprintf(f, "missed aconst");
        break;
      case S_CONST:
        fprintf(f, "const");
        break;
      case S_SCONST:
        fprintf(f, "missed sconst");
        break;
      case S_EXPR:
        fprintf(f, "expr");
        break;
      case S_IDENT:
        fprintf(f, "ident");
        break;
      default:
        fprintf(f, "?SST_ID%d", SST_IDG(stkp));
        break;
      }
      if (!in_array)
        acl_array_num += compute_width_dtype(dtype);
      break;
    case AC_ACONST:
      fprintf(f, "(/");
      _printacl(1, aclp->subc, f);
      fprintf(f, "/)");
      dtype = aclp->dtype;
      if (!in_array)
        acl_array_num += compute_width_dtype(dtype);
      break;
    case AC_SCONST:
      save_array_num = acl_array_num;

      dtype = aclp->dtype;
      sptr = DTY(dtype + 3); /* tag sptr */
      fprintf(f, "%s(", SYMNAME(sptr));
      member_aclp = aclp->subc;
      _printacl(0, member_aclp, f);
      fprintf(f, ")");

      if (in_array)
        acl_array_num = save_array_num;
      break;
    case AC_IDO:
      fprintf(f, "(");
      _printacl(in_array, aclp->subc, f);
      fprintf(f, ",i=l,u,s)");
      break;
    case AC_REPEAT:
      fprintf(f, "REPEAT[%d](", aclp->u1.count);
      _printacl(in_array, aclp->subc, f);
      fprintf(f, ")");
      break;
    case AC_IEXPR:
      dtype = aclp->dtype;
      fprintf(f, "AC_IEXPR(dtype %d, op %s)", dtype,
              iexpr_op(aclp->u1.expr->op));
      break;
    default:
      interr("_printacl .id", aclp->id, 3);
      break;
    }
    if (aclp->next)
      fprintf(f, ",");
  }
}

void
printacl(const char *s, ACL *aclp, FILE *f)
{
  if (f == NULL)
    f = stderr;
  acl_array_num = 0;
  fprintf(f, "%s-line %d: ", s, gbl.lineno);
  _printacl(1, aclp, f);
  fprintf(f, "\n");
}

static void
_dumpacl(int nest, ACL *aclp, FILE *f)
{
  /* dump a list of aclps */
  for (; aclp != NULL; aclp = aclp->next) {
    int sptr, dtype, ast, astinit, astlimit, aststep, astcount;
    SST *stkp;
    DOINFO *doinfo;

    fprintf(f, "\n%*.*s", 2 * nest, 2 * nest, "                           ");
    switch (aclp->id) {
    case AC_AST:
      dtype = A_DTYPEG(aclp->u1.ast);
      ast = aclp->u1.ast;
      fprintf(f, "dtype %d, ast(%d) ", dtype, ast);
      if (ast) {
        printast(ast);
        if (A_ALIASG(ast)) {
          fprintf(f, " [alias");
          if (A_ALIASG(ast) != ast) {
            ast = A_ALIASG(ast);
            fprintf(f, "(%d) ", ast);
            printast(ast);
          }
          fprintf(f, "]");
        }
      }
      break;
    case AC_EXPR:
      stkp = aclp->u1.stkp;
      dtype = SST_DTYPEG(stkp);
      ast = SST_ASTG(stkp);
      switch (SST_IDG(stkp)) {
      case S_ACONST:
        fprintf(f, "expr aconst, dtype %d", dtype);
        ast = 0;
        break;
      case S_CONST:
        fprintf(f, "expr const, dtype %d", dtype);
        break;
      case S_SCONST:
        fprintf(f, "expr sconst, dtype %d", dtype);
        break;
      case S_EXPR:
        fprintf(f, "expr expr, dtype %d", dtype);
        break;
      case S_IDENT:
        sptr = SST_SYMG(stkp);
        fprintf(f, "expr ident %d=%s, dtype %d", sptr,
                (sptr > 0 && sptr < stb.stg_avail) ? SYMNAME(sptr) : "", dtype);
        break;
      default:
        fprintf(f, "expr unknown, dtype %d", dtype);
        break;
      }
      if (ast) {
        fprintf(f, ", ast(%d) ", ast);
        printast(ast);
        if (A_ALIASG(ast)) {
          fprintf(f, " [alias");
          if (A_ALIASG(ast) != ast) {
            ast = A_ALIASG(ast);
            fprintf(f, "(%d) ", ast);
            printast(ast);
          }
          fprintf(f, "]");
        }
      }
      break;
    case AC_CONST:
      fprintf(f, "const dtype %d conval %d", aclp->dtype, aclp->conval);
      break;
    case AC_ACONST:
      dtype = aclp->dtype;
      fprintf(f, "array, dtype %d", dtype);
      _dumpacl(nest + 1, aclp->subc, f);
      break;
    case AC_SCONST:
      dtype = aclp->dtype;
      sptr = DTY(dtype + 3); /* tag sptr */
      fprintf(f, "structure %s dtype %d", SYMNAME(sptr), dtype);
      _dumpacl(nest + 1, aclp->subc, f);
      break;
    case AC_IDO:
      doinfo = aclp->u1.doinfo;
      sptr = doinfo->index_var;
      astinit = doinfo->init_expr;
      astlimit = doinfo->limit_expr;
      aststep = doinfo->step_expr;
      astcount = doinfo->count;
      fprintf(f, "DO [ast(%d)] %s = ast(%d), ast(%d), ast(%d) = [", astcount,
              SYMNAME(sptr), astinit, astlimit, aststep);
      if (astcount)
        printast(astcount);
      fprintf(f, "] ");
      if (astinit)
        printast(astinit);
      fprintf(f, ", ");
      if (astlimit)
        printast(astlimit);
      fprintf(f, ", ");
      if (aststep)
        printast(aststep);
      _dumpacl(nest + 1, aclp->subc, f);
      break;
    case AC_REPEAT:
      fprintf(f, "REPEAT*%d", aclp->u1.count);
      _dumpacl(nest + 1, aclp->subc, f);
      break;
    case AC_IEXPR:
      dtype = aclp->dtype;
      fprintf(f, "AC_IEXPR dtype %d, op %s", dtype,
              iexpr_op(aclp->u1.expr->op));
      break;
    case AC_CONVAL:
      dtype = aclp->dtype;
      fprintf(f, "AC_CONVAL dtype %d, conval %d", dtype, aclp->conval);
      break;
    default:
      fprintf(f, "unknown aclp->id %d", aclp->id);
      break;
    }
  }
}

void
dumpacl(const char *s, ACL *aclp, FILE *f)
{
  if (f == NULL)
    f = stderr;
  acl_array_num = 0;
  fprintf(f, "ACL(%s):", s);
  _dumpacl(1, aclp, f);
  fprintf(f, "\n");
}
#endif

static int
compute_width_dtype(DTYPE in_dtype)
{
  int sum;
  int member_dtype;
  int sptr;
  int stag;
  DTYPE dtype = DDTG(in_dtype);

  if (DTY(dtype) != TY_DERIVED)
    return 1;
  stag = DTY(dtype + 3);
  if (VISITG(stag))
    return 1;
  VISITP(stag, 1);
  sum = 0;
  /* for each member */
  sptr = DTY(dtype + 1);
  for (; sptr != NOSYM; sptr = SYMLKG(sptr)) {
    member_dtype = DTYPEG(sptr);
    if (DTYG(member_dtype) == TY_DERIVED)
      sum += compute_width_dtype(member_dtype);
    else
      sum++;
  }
  VISITP(stag, 0);
  return sum;
}

/*  This code computes the number of arrays that are going to be
    created to store the aclp (== 1, unless this is an array of
    derived types.
 */
static int cw_array_num = 0;
static int max_cw_array_num = 0;

static void
_compute_width(int in_array, ACL *aclp)
{
  int save_cw_array_num;
  DTYPE dtype;

  /* if we are !in_array then we are in a structure, and
     the following element (or array) will represent a new
     mangled component, so increment cw_array_num  */

  for (; aclp != NULL; aclp = aclp->next) {
    switch (aclp->id) {
    case AC_AST:
    case AC_CONST:
      dtype = A_DTYPEG(aclp->u1.ast);
      goto have_dtype;
    case AC_EXPR:
      dtype = SST_DTYPEG(aclp->u1.stkp);
    have_dtype:
      aclp->u2.array_i = cw_array_num; /* save index */
      if (!in_array)
        cw_array_num += compute_width_dtype(dtype);
      if ((cw_array_num - 1) > max_cw_array_num)
        max_cw_array_num = (cw_array_num - 1);
      break;
    case AC_ACONST:
      _compute_width(1, aclp->subc); /* element list */
      dtype = aclp->dtype;
      if (!in_array)
        cw_array_num += compute_width_dtype(dtype);
      if ((cw_array_num - 1) > max_cw_array_num)
        max_cw_array_num = (cw_array_num - 1);
      break;
    case AC_SCONST:
      save_cw_array_num = cw_array_num;

      _compute_width(0, aclp->subc); /* member list */

      if (in_array)
        cw_array_num = save_cw_array_num;
      break;
    case AC_IDO:
      _compute_width(in_array, aclp->subc); /* IDO ac list */
      break;
    case AC_REPEAT:
      _compute_width(in_array, aclp->subc); /* item repeated */
      break;
    case AC_IEXPR:
      _compute_width(in_array, aclp->subc);
      break;
    default:
      interr("compute width aclp->id", aclp->id, 3);
      break;
    }
  }
}

/** \brief Check if array has zero size.

    It expects lowerbound and upper bound to be constant asts.
    Don't use NUM_ELEM because it could return 1 as number of element,
    If dtype is zero, it loosely check aggregate size which must be done
    after chk_constructor/(2).
 */
ISZ_T
size_of_array(DTYPE dtype)
{
  int i;
  ADSC *ad;
  int numdim;
  ISZ_T dim_size;
  ISZ_T size = 1;
  ISZ_T d;

  if (dtype) {
#define DEFAULT_DIM_SIZE 127

#if DEBUG
    assert(DTY(dtype) == TY_ARRAY, "extent_of, expected TY_ARRAY", dtype, 3);
#endif
    if ((d = DTY(dtype + 2)) <= 0) {
      interr("extent_of: no array descriptor", (int)d, 3);
      return 0;
    }

    switch (DTY(dtype)) {
    case TY_ARRAY:
      if (DTY(dtype + 2) != 0) {
        ad = AD_DPTR(dtype);
        numdim = AD_NUMDIM(ad);
        if (numdim < 1 || numdim > 7) {
          interr("extent_of: bad numdim", 0, 1);
          numdim = 0;
        }
        for (i = 0; i < numdim; i++) {
          if (A_TYPEG(AD_LWAST(ad, i)) != A_CNST &&
              A_TYPEG(AD_UPAST(ad, i)) != A_CNST) {
            dim_size = DEFAULT_DIM_SIZE;
          } else {
            dim_size = ad_val_of(sym_of_ast(AD_UPAST(ad, i))) -
                       ad_val_of(sym_of_ast(AD_LWAST(ad, i))) + 1;
          }
          size *= dim_size;
        }
      }
      break;

    default:
      return size;
    }
  } else if (acs.aggr_cnt == astb.bnd.zero && acs.scalar_cnt == 0) {
    return 0;
  }
  return size;
}

static int
compute_width(ACL *aclp)
{
  cw_array_num = 0;
  max_cw_array_num = 0;
  _compute_width(1, aclp);
  return (max_cw_array_num + 1);
}

/** \brief Check the array constructor and decide the dtype.

    It is called when we first recognize an array constructor.
 */
DTYPE
chk_constructor(ACL *aclp, DTYPE dtype)
{
  SEM_DIM_SPECS dim_specs_tmp;
#if DEBUG
  if (DBGBIT(3, 64))
    printacl("chk_constructor", aclp, gbl.dbgfil);
  assert(aclp->id == AC_ACONST, "chk_constructor aclp->id:", aclp->id, 3);
#endif

  save_dim_specs(&dim_specs_tmp);
  BZERO(&acs, _ACS, 1);
  acs.aggr_cnt = astb.bnd.zero;
  acs.is_const = TRUE;

  sem.top = &sem.dostack[0];
  compute_size(true, aclp->subc, dtype);
  if (dtype) {
    acs.eltype = dtype;
  }

  switch (DTY(acs.eltype)) {
  case TY_CHAR:
  case TY_NCHAR:
    if (!A_ALIASG(DTY(acs.eltype + 1))) {
    } else if (acs.zln) {
      /* should be an error */
      acs.eltype = get_type(2, DTY(acs.eltype), astb.i1);
    }
    break;
  }

  sem.arrdim.ndim = 1;

  sem.bounds[0].lowtype = S_CONST;
  sem.bounds[0].lowb = 1;
  sem.bounds[0].lwast = 0;

  if (acs.aggr_cnt == astb.bnd.zero) {
    sem.bounds[0].uptype = S_CONST;
    sem.bounds[0].upb = acs.scalar_cnt;
    sem.bounds[0].upast = mk_isz_cval((INT)acs.scalar_cnt, astb.bnd.dtype);
    sem.arrdim.ndefer = 0;
  } else {
    sem.bounds[0].uptype = S_EXPR;
    sem.bounds[0].upb = 0;
    sem.bounds[0].upast = mk_binop(
        OP_ADD, acs.aggr_cnt, mk_isz_cval((INT)acs.scalar_cnt, astb.bnd.dtype),
        astb.bnd.dtype);
    sem.arrdim.ndefer = 1;
    acs.is_const = FALSE;
  }
  if (sem.gcvlen && is_deferlenchar_dtype(acs.eltype)) {
    sem.arrdim.ndefer = 1;
  }
  aclp->size = sem.bounds[0].upast;

  acs.arrtype = mk_arrdsc();
  DTY(acs.arrtype + 1) = acs.eltype;
  restore_dim_specs(&dim_specs_tmp);

  aclp->is_const = acs.is_const; /* store in acl */
  aclp->dtype = acs.arrtype;     /* store in acl and also return*/
  return acs.arrtype;
}

/** \brief Initialize a named array constant (array PARAMETER), ensuring that
           it's only being done within the context of its host subprogram.
 */
void
init_named_array_constant(int arr, int host)
{
  if (ENCLFUNCG(arr) == 0 || ENCLFUNCG(arr) == host)
    /* emit the data inits for the named array constant */
    init_sptr_w_acl((int)CONVAL1G(arr), (ACL *)get_getitem_p(CONVAL2G(arr)));
}

static int ALLOCATE_ARRAYS = TRUE;

SPTR
get_param_alias_var(SPTR param_sptr, DTYPE dtype)
{
  char *np = mangle_name(SYMNAME(param_sptr), "ac");
  SPTR alias_sptr = getsymbol(np);

  STYPEP(alias_sptr, ST_VAR);
  DTYPEP(alias_sptr, dtype);
  DCLDP(alias_sptr, 1);
  SCP(alias_sptr, SC_STATIC);
  SCOPEP(alias_sptr, stb.curr_scope);
  CONVAL1P(param_sptr, alias_sptr);
  PARAMP(alias_sptr, PARAMG(param_sptr));
  PARAMVALP(alias_sptr, PARAMVALG(param_sptr));
  DINITP(alias_sptr, 1);
  HCCSYMP(alias_sptr, 1);
  NMCNSTP(alias_sptr, param_sptr);
  sym_is_refd(alias_sptr);
  REFP(alias_sptr, 1);
  return alias_sptr;
}

static int
convert_ctmp_array_to_param(int cctmpsptr, ACL *aclp)
{
  /* A temp has been generated to hold the value of an array
   * constructor and this temp is used in an expression.  Convert
   * the temp to a named constant so that the initialization
   * values are available (in the associated A_INIT list) for use
   * in expression evaluation (esp. named constant initialization
   * expressions) */

  SST tmp_sst;
  SST init;
  DTYPE dtype = DTYPEG(cctmpsptr);
  int aliassptr;

  PARAMP(cctmpsptr, 1);
  STYPEP(cctmpsptr, ST_ARRAY);
  SCP(cctmpsptr, SC_NONE);

  BZERO(&tmp_sst, SST, 1);
  SST_IDP(&tmp_sst, S_IDENT);
  SST_SYMP(&tmp_sst, cctmpsptr);
  dinit_struct_param(cctmpsptr, aclp, aclp->dtype);

  STYPEP(cctmpsptr, ST_PARAM);
  SCOPEP(cctmpsptr, stb.curr_scope);

  aliassptr = get_param_alias_var(cctmpsptr, dtype);
  STYPEP(aliassptr, ST_ARRAY);

  BZERO(&init, SST, 1);
  SST_IDP(&init, S_ACONST);
  SST_DTYPEP(&init, aclp->dtype);
  SST_ACLP(&init, aclp);

  construct_acl_for_sst(&init, DTYPEG(cctmpsptr));

  if (sem.interface == 0) {
    CONVAL2P(cctmpsptr, put_getitem_p(aclp));
  } else {
    IGNOREP(cctmpsptr, 0);
  }

  return aliassptr;
}

/** \brief Assign \a aclp values to \a in_sptr.

    If \a in_sptr is 0, it assigns values to temporaries.  init_sptr_w_acl() is
    called at the point we are trying to use (a possibly array/struct nested)
    constructor; eg. in mkexpr1().  If acl is constant, dinit_constructor()
    uses data initialization to assign the values; otherwise, _construct is
    called to generate runtime code to assign values.  (is_const means: is
    constant, we can do it, and we're in the right context to do it.)
 */
int
init_sptr_w_acl(int in_sptr, ACL *aclp)
{
  int sptr_supplied;
  int sptr;
  int ast;
  SST tmp_sst;
  VAR *ivl;
  SEM_DIM_SPECS dim_specs_tmp;

#if DEBUG
  if (DBGBIT(3, 64))
    printacl("init_sptr_w_acl", aclp, gbl.dbgfil);
#endif

  if (in_sptr && DINITG(in_sptr))
    return in_sptr;

  if (in_sptr && ENCLFUNCG(in_sptr) &&
      STYPEG(ENCLFUNCG(in_sptr)) == ST_MODULE) {
    /* the DINIT flag used to be enough. But now interf.c sets DINIT to
       zero.  So for MODULE var$ac referenced outside the module, we can
       assume the initialization has already been done. */
    return in_sptr;
  }

  if (aclp->id != AC_ACONST) {
    interr("init_sptr_w_acl aclp->id:", aclp->id, 3);
    return 0;
  }

  save_dim_specs(&dim_specs_tmp);
  BZERO(&acs, _ACS, 1);

  sptr_supplied = (in_sptr != 0);
  sptr = in_sptr;

  /* chk_constructor() was called earlier and set up this information */
  acs.is_const = aclp->is_const;
  acs.arrtype = aclp->dtype;
  sem.arrdim.ndefer = AD_DEFER(AD_DPTR(acs.arrtype));

  if (sem.dinit_data) {
    if (sptr_supplied) {
      acs.tmp = 0;
    } else {
      acs.tmp = get_arr_const(acs.arrtype);
    }

    sptr = acs.tmp;
    /* converts to AC_AST ACL */
    aclp->subc = rewrite_acl(aclp->subc, aclp->dtype, aclp->id);

    if (!sptr_supplied) {
      acs.tmp = sptr = convert_ctmp_array_to_param(sptr, aclp);
    }

    ast = mk_id(sptr);
    SST_IDP(&tmp_sst, S_IDENT);
    SST_ASTP(&tmp_sst, ast);
    SST_DTYPEP(&tmp_sst, DTYPEG(sptr));
    SST_SHAPEP(&tmp_sst, A_SHAPEG(ast));
    ivl = dinit_varref(&tmp_sst);
    dinit(ivl, aclp);
  } else if (acs.is_const) {
    if (sptr_supplied) {
      acs.tmp = 0;
    } else {
      acs.tmp = get_arr_const(acs.arrtype);
    }

    /* converts AC_AST to AC_IEXPR. */
    aclp->subc = rewrite_acl(aclp->subc, aclp->dtype, aclp->id);
  } else {
    int std;
    if (sem.arrdim.ndefer) {
      ALLOCATE_ARRAYS = 0; /* allocate for these array temps is done here */
    }

    if (sem.arrdim.ndefer && sem.arrfn.sptr > NOSYM &&
        sem.arrfn.return_value &&
        ADD_DEFER(A_DTYPEG(sem.arrfn.return_value)) &&
        aclp->id == AC_ACONST && aclp->subc->id == AC_IDO) {
      /* The ACL is an array constructor that contains implied-do loop, and
       * there is a function call that returns a deferred-length array.
       * Create an allocatable array temp in case of that the function call
       * appears in the loop body and causes the size of the resulting array to
       * be determined at runtime.
       */
      sptr = acs.tmp = get_adjlr_arr_temp(acs.arrtype);
      get_static_descriptor(acs.tmp);
      get_all_descriptors(acs.tmp);
    } else
      sptr = acs.tmp = get_arr_temp(acs.arrtype, FALSE, FALSE, FALSE);
    ALLOCATE_ARRAYS = 1;
    if (sem.arrdim.ndefer) {
      sem.bounds[0].lwast = astb.bnd.one;
      sem.bounds[0].upast = aclp->size;
      /* assign values to the bounds temporaries and allocate the
       * array.
       */
      std = STD_PREV(0);
      gen_allocate_array(acs.tmp);
      std = STD_NEXT(std);
    }

    acs.func_in_do = FALSE;
    /* generate code to assign aclp values to the temporary */
    constructf90(acs.tmp, aclp);

    /* If the function call returns a deferred-length array or adjustable
     * array, and it appears in the loop body, the bounds of the function
     * return array is uninitialized when used in the allocation of array temp.
     * Here we set the function return array to be zero-sized before the
     * allocation.
     */
    if (acs.func_in_do && sem.arrdim.ndefer && sem.arrfn.return_value) {
      int ret_sptr, dtype, ndim, i;

      ret_sptr = A_SPTRG(sem.arrfn.return_value);
      dtype = DTYPEG(ret_sptr);
      ndim = ADD_NUMDIM(dtype);
      if (ADJARRG(ret_sptr)) {
        for (i = 0; i < ndim; i++) {
          int lb = ADD_LWBD(dtype, i);
          int ub = ADD_UPBD(dtype, i);
          assert(A_TYPEG(lb) == A_ID, "init_sptr_w_acl: lb not id", lb,
                 ERR_Fatal);
          assert(A_TYPEG(ub) == A_ID, "init_sptr_w_acl: ub not id", ub,
                 ERR_Fatal);
          (void)add_stmt_before(
              mk_assn_stmt(lb, astb.bnd.one, astb.bnd.dtype), std);
          (void)add_stmt_before(
              mk_assn_stmt(ub, astb.bnd.zero, astb.bnd.dtype), std);
        }
      } else if (SDSCG(ret_sptr)) {
        int sdsc = SDSCG(ret_sptr);
        for (i = 0; i < ndim; i++) {
          int lb = lbound_of(dtype, i);
          int extnt = ADD_EXTNTAST(dtype, i);
          assert(A_TYPEG(lb) == A_SUBSCR, "init_sptr_w_acl: lb not subs", lb,
                 ERR_Fatal);
          assert(memsym_of_ast(lb) == sdsc, "init_sptr_w_acl: lb not sdsc", lb,
                 ERR_Fatal);
          assert(A_TYPEG(extnt) == A_SUBSCR, "init_sptr_w_acl: extnt not subs",
                 extnt, ERR_Fatal);
          assert(memsym_of_ast(extnt) == sdsc,
                 "init_sptr_w_acl: extnt not sdsc", extnt, ERR_Fatal);
          (void)add_stmt_before(
              mk_assn_stmt(lb, astb.bnd.one, astb.bnd.dtype), std);
          (void)add_stmt_before(
              mk_assn_stmt(extnt, astb.bnd.zero, astb.bnd.dtype), std);
        }
      }
    }
    acs.tmp = sptr; /* if we recursed, asc.tmp may have changed */
  }

  /* if the user didn't supply an sptr, use the temporary
     created above. */
  if (!sptr_supplied) {
    sptr = acs.tmp;
  }

  if (acs.is_const) {
    if (!sem.dinit_data) {
      dinit_constructor(sptr, aclp);
    } else if (sptr_supplied) {
      interr("acl not resolved as constant", sptr, 2);
    }
  }
  restore_dim_specs(&dim_specs_tmp);
  return sptr;
}

/* add_flag gets set to false, when we see a SCONST.  We want to
   recurse on structure constructor to set acs.is_const, but we
   don't want to add to the counts for any components of the
   structure constructor.
   Convert the dtype to the dtype passed as argument.
 */
static void
compute_size(bool add_flag, ACL *aclp, DTYPE dtype)
{
  for (; aclp != NULL; aclp = aclp->next) {
    switch (aclp->id) {
    case AC_AST:
      compute_size_ast(add_flag, aclp, dtype);
      break;
    case AC_EXPR:
      dtype = compute_size_expr(add_flag, aclp, dtype);
      break;
    case AC_ACONST:
      compute_size(add_flag, aclp->subc, dtype);
      break;
    case AC_SCONST:
      compute_size_sconst(add_flag, aclp, dtype);
      break;
    case AC_IDO: {
      int save_start, save_end, save_level;
      DOINFO *save_doinfo = NULL;
      LOGICAL saved = FALSE;
      IDO_INFO *ido = (IDO_INFO *)getitem(0, sizeof(IDO_INFO));
      ido->doinfo = aclp->u1.doinfo;
      ido->next = acs.ido_list;
      /* start processing of implied-do */
      acs.ido_list = ido;
      acs.ido_level++;
      if (acs.loop_stmts.level != 0 &&
          acs.ido_level == acs.loop_stmts.level) {
        /* save information before computing size of implied-do */
        save_start = acs.loop_stmts.start;
        save_end = acs.loop_stmts.end;
        save_level = acs.loop_stmts.level;
        save_doinfo = acs.loop_stmts.doinfo;
        saved = TRUE;
        acs.loop_stmts.start = 0;
        acs.loop_stmts.end = 0;
        acs.loop_stmts.level = 0;
        acs.loop_stmts.doinfo = NULL;
      }
      compute_size_ido(add_flag, aclp, dtype);
      if (saved) {
        if (acs.loop_stmts.level == 0) {
          /* restore information after computing size of implied-do */
          acs.loop_stmts.start = save_start;
          acs.loop_stmts.end = save_end;
          acs.loop_stmts.level = save_level;
          acs.loop_stmts.doinfo = save_doinfo;
        } else {
          move_range_before(save_start, save_end, acs.loop_stmts.start);
          acs.loop_stmts.start = save_start;
          /* update acs.loop_stmts.doinfo */
          for (IDO_INFO *iter = acs.ido_list; iter; iter = iter->next) {
            if (iter->doinfo == save_doinfo) {
              break;
            }
            if (iter->doinfo == acs.loop_stmts.doinfo) {
              acs.loop_stmts.doinfo = save_doinfo;
              break;
            }
          }
        }
      }
      /* finish processing of implied-do */
      acs.ido_level--;
      acs.ido_list = acs.ido_list->next;
      if (sem.dinit_error) {
        return;
      }
      break;
    }
    default:
      interr("compute_size,ill.id", aclp->id, 3);
    }
  }
}

static void
compute_size_ast(bool add_flag, ACL *aclp, DTYPE dtype)
{
  if (acs.eltype == 0 || acs.zln) {
    if (acs.eltype != 0) {
      acs.zln = 0;
    }
    if (dtype == 0) {
      dtype = DDTG(A_DTYPEG(aclp->u1.ast));
    }
    if (A_TYPEG(aclp->u1.ast) == A_ID) {
      dtype = fix_dtype(A_SPTRG(aclp->u1.ast), dtype);
    }
    acs.eltype = dtype;
    switch (DTY(acs.eltype)) {
    case TY_CHAR:
    case TY_NCHAR:
      if (A_ALIASG(DTY(acs.eltype + 1)) &&
          get_isz_cval(A_SPTRG(A_ALIASG(DTY(acs.eltype + 1)))) == 0) {
        acs.zln = 1;
      }
    }
  }
  if (add_flag)
    acs.scalar_cnt++;
}

static DTYPE
compute_size_expr(bool add_flag, ACL *aclp, DTYPE dtype)
{
  DTYPE dt2, dtype2;
  SST *stkp = aclp->u1.stkp;
  LOGICAL specified_dtype = dtype != 0;
  DTYPE dt = DDTG(dtype);
  dtype2 = SST_DTYPEG(stkp);
  dt2 = DDTG(SST_DTYPEG(stkp));
  if (!specified_dtype) {
    dtype = dtype2;
    dt = dt2;
  }

  if (acs.eltype == 0 || acs.zln) {
    int id = SST_IDG(stkp);
    if (acs.eltype != 0) {
      acs.zln = 0;
    }
    if (id == S_IDENT) {
      dt = fix_dtype(SST_SYMG(stkp), dt);
    } else if (id == S_EXPR || id == S_LVALUE) {
      if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
          || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
      ) {
        dt = adjust_ch_length(dt, SST_ASTG(stkp));
      } else if (dt == DT_ASSCHAR || dt == DT_DEFERCHAR
          || dt == DT_ASSNCHAR || dt == DT_DEFERNCHAR
      ) {
        dt = fix_dtype(SST_SYMG(stkp), dt);
      }
    }
    /* need to change the type for the first element too */
    if (specified_dtype && acs.eltype == 0 &&
        add_flag) { /* if we're in a struct, don't do */
      if (DTY(dt) == TY_CHAR && DTY(dtype) == TY_CHAR) {
        if (dtype2 != DT_DEFERCHAR && dtype2 != DT_DEFERNCHAR)
          dtype = SST_DTYPEG(stkp);
      } else if (DTY(dt) == TY_NCHAR && DTY(dtype) == TY_NCHAR) {
        if (dtype2 != DT_DEFERCHAR && dtype2 != DT_DEFERNCHAR)
          dtype = SST_DTYPEG(stkp);
      } else if (DTY(dtype) == TY_ARRAY) {
        if (DDTG(dtype) != dt) {
          errsev(95);
        }
      } else {
        cngtyp(stkp, acs.eltype);
        dtype = SST_DTYPEG(stkp);
      }
    }
    acs.eltype = dt;
    switch (DTY(acs.eltype)) {
    case TY_CHAR:
    case TY_NCHAR:
      if (A_ALIASG(DTY(acs.eltype + 1)) &&
          get_isz_cval(A_SPTRG(A_ALIASG(DTY(acs.eltype + 1)))) == 0) {
        acs.zln = 1;
      }
    }
  } else {
    /* don't use chktyp here; chktyp evals semantic stack entry
     * causes S_CONST to become S_EXPR.
     */
    if (add_flag) { /* if we're in a struct, don't do */
      if (DTY(dt) == TY_CHAR && DTY(dtype) == TY_CHAR) {
        if (dtype2 != DT_DEFERCHAR && dtype2 != DT_DEFERNCHAR)
          dtype = SST_DTYPEG(stkp);
      } else if (DTY(dt) == TY_NCHAR && DTY(dtype) == TY_NCHAR) {
        if (dtype2 != DT_DEFERCHAR && dtype2 != DT_DEFERNCHAR)
          dtype = SST_DTYPEG(stkp);
      } else if (DTY(dtype) == TY_ARRAY) {
        if (is_dtype_runtime_length_char(dtype) &&
            is_dtype_runtime_length_char(acs.eltype)) {
          // AC element could have adjusted length. The length mismatch should
          // be checked in runtime when -fcheck=bounds is enabled.
        } else if (!eq_dtype(DDTG(dtype), acs.eltype)) {
          errsev(95);
        }
      } else {
        cngtyp(stkp, acs.eltype);
        dtype = SST_DTYPEG(stkp);
      }
    }
  }
  switch (SST_IDG(stkp)) {
  case S_ACONST:
    interr("compute_size, AC_ACONST in AC_EXPR", 0, 3);
    if (add_flag)
      acs.scalar_cnt += CONVAL2G(sym_of_ast(AD_NUMELM(AD_DPTR(dtype))));
    break;
  case S_CONST:
    mkexpr(stkp);
    if (add_flag)
      acs.scalar_cnt++;
    break;
  default:
    mkexpr(stkp);
    if (DTY(dtype) != TY_ARRAY) {
      int ast = SST_ASTG(stkp);
      if (add_flag)
        acs.scalar_cnt++;
      if (!ast) {
        acs.is_const = FALSE;
      } else if (A_ALIASG(ast) || (acs.level && _can_fold(ast))) {
        /* do nothing */
      } else if (A_TYPEG(ast) == A_ID) {
        int sptr = A_SPTRG(ast);
        if (STYPEG(sptr) != ST_VAR || !PARAMVALG(sptr)) {
          acs.is_const = FALSE;
        }
      } else {
        acs.is_const = FALSE;
      }
    } else {
      int ast;
      if (add_flag) {
        int sz = size_of_ast((int)SST_ASTG(stkp));
        if (A_ALIASG(sz))
          acs.scalar_cnt += ad_val_of(A_SPTRG(A_ALIASG(sz)));
        else
          acs.aggr_cnt = mk_binop(OP_ADD, acs.aggr_cnt, sz, astb.bnd.dtype);
      }
      ast = SST_ASTG(stkp);
      if (!ast) {
        acs.is_const = FALSE;
      } else if (A_TYPEG(ast) == A_ID) {
        int sptr = A_SPTRG(ast);
        if (STYPEG(sptr) != ST_ARRAY || !PARAMVALG(sptr)) {
          acs.is_const = FALSE;
        }
      } else if (!_can_fold(ast)) {
        acs.is_const = FALSE;
      }
    }
  }
  return specified_dtype ? dtype : DT_NONE;
}

/* Check dependencies and find the outermost implied-do loop
 * on which the array size depends.
 */
static void
check_dependence_ido(ACL *aclp)
{
  DOINFO *cur_ido = aclp->u1.doinfo;
  STD_RANGE *range = aclp->u2.std_range;
  int level = acs.ido_level - 1;

  for (IDO_INFO *ido = acs.ido_list->next; ido; ido = ido->next) {
    int dovar_id = mk_id(ido->doinfo->index_var);
    LOGICAL depend = FALSE;
    if (contains_ast(acs.aggr_cnt, dovar_id) ||
        contains_ast(cur_ido->init_expr, dovar_id) ||
        contains_ast(cur_ido->limit_expr, dovar_id) ||
        contains_ast(cur_ido->step_expr, dovar_id)) {
      /* direct dependence */
      depend = TRUE;
    }

    if (!depend && range != NULL && range->mid != range->end) {
      for (int std = STD_NEXT(range->mid); std; std = STD_NEXT(std)) {
        if (contains_ast(STD_AST(std), dovar_id)) {
          /* indirect dependence */
          depend = TRUE;
          break;
        }
        if (std == range->end)
          break;
      }
    }
    /* update information about the loop on which current implied-do depends */
    if (depend && (acs.loop_stmts.level == 0 ||
                   acs.loop_stmts.level > level)) {
      acs.loop_stmts.doinfo = ido->doinfo;
    }
    level--;
  }
}

/* When computing the size of AC_IDO, clone the stmts on which the bounds of
 * the current implied-do loop depend, and generate loop to compute size.
 */
static void
handle_dependence_ido(ACL *aclp)
{
  int sumid;
  int prev_std;
  STD_RANGE *range = aclp->u2.std_range;
  DOINFO *doinfo = aclp->u1.doinfo;

  if (acs.loop_stmts.start == 0 && acs.loop_stmts.end == 0) {
    prev_std = STD_LAST;
  } else {
    prev_std = STD_PREV(acs.loop_stmts.start);
  }

  /* create temp for size at the beginning and initialize it with zero */
  if (acs.loop_stmts.sumid == 0) {
    acs.loop_stmts.sumid = mk_id(get_temp(astb.bnd.dtype));
  }
  sumid = acs.loop_stmts.sumid;
  if (acs.loop_stmts.doinfo == doinfo) {
    (void)add_stmt(mk_assn_stmt(sumid, astb.bnd.zero, astb.bnd.dtype));
  }

  /* clone stmts on which the bounds of current implied-do depend */
  if (range != NULL && range->mid != range->end) {
    for (int std = STD_NEXT(range->mid); std; std = STD_NEXT(std)) {
      (void)add_stmt(STD_AST(std));
      if (std == range->end)
        break;
    }
  }

  if (acs.loop_stmts.level == 0) {
    /* current implied-do is the outermost one */
    int st, ast;
    st = doinfo->step_expr == 0 ? astb.bnd.one : doinfo->step_expr;
    ast = mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr,
                   astb.bnd.dtype);
    ast = mk_binop(OP_ADD, ast, st, astb.bnd.dtype);
    ast = mk_binop(OP_DIV, ast, st, astb.bnd.dtype);
    ast = mk_binop(OP_MUL, ast, acs.aggr_cnt, astb.bnd.dtype);
    ast = mk_binop(OP_ADD, sumid, ast, astb.bnd.dtype);
    (void)add_stmt(mk_assn_stmt(sumid, ast, astb.bnd.dtype));
  } else {
    /* generate loop to compute size */
    SPTR dovar, odovar;
    int newid, ast;
    int do_beg_std;

    odovar = doinfo->index_var;
    dovar = get_temp(DTYPEG(odovar));
    HIDDENP(dovar, 1);
    newid = mk_id(dovar);
    ast = mk_stmt(A_DO, 0);
    A_DOVARP(ast, newid);
    A_M1P(ast, doinfo->init_expr);
    A_M2P(ast, doinfo->limit_expr);
    A_M3P(ast, doinfo->step_expr);
    A_M4P(ast, 0);
    do_beg_std = add_stmt(ast);

    ast_visit(1, 1);
    ast_replace(mk_id(odovar), newid);
    move_range_after(acs.loop_stmts.start, acs.loop_stmts.end, do_beg_std);
    for (int std = STD_NEXT(do_beg_std); std; std = STD_NEXT(std)) {
      STD_AST(std) = ast_rewrite(STD_AST(std));
    }

    /* add size */
    if (acs.aggr_cnt != astb.bnd.zero) {
      ast = ast_rewrite(acs.aggr_cnt);
      ast = mk_binop(OP_ADD, sumid, ast, astb.bnd.dtype);
      (void)add_stmt(mk_assn_stmt(sumid, ast, astb.bnd.dtype));
    }
    ast_unvisit();
    (void)add_stmt(mk_stmt(A_ENDDO, 0));
  }

  if (acs.loop_stmts.doinfo == aclp->u1.doinfo) {
    /* remove information about dependence of implied-do */
    acs.aggr_cnt = sumid;
    acs.loop_stmts.start = 0;
    acs.loop_stmts.end = 0;
    acs.loop_stmts.level = 0;
    acs.loop_stmts.sumid = 0;
    acs.loop_stmts.doinfo = NULL;
  } else {
    acs.aggr_cnt = astb.bnd.zero;
    acs.loop_stmts.start = STD_NEXT(prev_std);
    acs.loop_stmts.end = STD_LAST;
    acs.loop_stmts.level = acs.ido_level;
  }
}

static void
compute_size_ido(bool add_flag, ACL *aclp, DTYPE dtype)
{
  DOINFO *doinfo = aclp->u1.doinfo;
  INT initval, limitval, stepval;
  int save_scalar_cnt, save_aggr_cnt;
  int id;
  if (sem.dinit_data) {
    /* set up for the possibility that a nested implied
     * do will require counting the number of elements
     */
    sem.top->sptr = aclp->u1.doinfo->index_var;
    sem.top->currval = initval = dinit_eval(doinfo->init_expr);
    if (sem.dinit_error) {
      return;
    }
    sem.top->upbd = limitval = dinit_eval(doinfo->limit_expr);
    if (sem.dinit_error) {
      return;
    }
    sem.top->step = stepval = dinit_eval(doinfo->step_expr);
    if (sem.dinit_error) {
      return;
    }
    sem.top++;

    if (A_ALIASG(doinfo->count)) {
      acs.level++;
      DOVARP(doinfo->index_var, 1);
    }
  }
  if (add_flag) {
    save_scalar_cnt = acs.scalar_cnt;
    save_aggr_cnt = acs.aggr_cnt;
    /*
     * scalar_cnt & aggr_cnt will reflect the number of items
     * immediately contained by this implied do.
     */
    acs.scalar_cnt = 0;
    acs.aggr_cnt = astb.bnd.zero;
  }
  compute_size(add_flag, aclp->subc, dtype);
  /*
   *  size is the 'cnt*scalar_cnt + cnt*aggr_cnt'
   */
  id = mk_id(doinfo->index_var);
  check_dependence_ido(aclp);
  if (add_flag && (contains_ast(acs.aggr_cnt, id) ||
                   acs.loop_stmts.doinfo != NULL)) {
    /* The size expression depends on the loop index variable.
     * This is tricky because we need the size to allocate
     * the temporary before we generate the loop.  First,
     * if there is a scalar_cnt, convert it to an expression
     * to be added later (size can't be a constant now).
     */
    if (acs.scalar_cnt != 0) {
      acs.aggr_cnt =
          mk_binop(OP_ADD, acs.aggr_cnt,
                   mk_isz_cval(acs.scalar_cnt, astb.bnd.dtype), astb.bnd.dtype);
      acs.scalar_cnt = 0;
    }
    /* Now we need to evaluate the size expression for each
     * value of the loop index variable and add the results.
     * There are two cases:
     */
    if (A_ALIASG(doinfo->init_expr) && A_ALIASG(doinfo->limit_expr) &&
        A_ALIASG(doinfo->step_expr) && acs.loop_stmts.doinfo == NULL) {
      int i;
      int ast;

      /* In the easy case, the loop control expressions are
       * constants, so we can iterate at compile time,
       * substituting each value of the loop variable and
       * adding the sizes.
       */
      initval = CONVAL2G(A_SPTRG(A_ALIASG(doinfo->init_expr)));
      limitval = CONVAL2G(A_SPTRG(A_ALIASG(doinfo->limit_expr)));
      stepval = CONVAL2G(A_SPTRG(A_ALIASG(doinfo->step_expr)));
      ast = astb.bnd.zero;
      if (stepval >= 0) {
        for (i = initval; i <= limitval; i += stepval) {
          ast_visit(1, 1);
          ast_replace(id, mk_cval(i, astb.bnd.dtype));
          ast =
              mk_binop(OP_ADD, ast, ast_rewrite(acs.aggr_cnt), astb.bnd.dtype);
          ast_unvisit();
        }
      } else {
        for (i = initval; i >= limitval; i += stepval) {
          ast_visit(1, 1);
          ast_replace(id, mk_cval(i, astb.bnd.dtype));
          ast =
              mk_binop(OP_ADD, ast, ast_rewrite(acs.aggr_cnt), astb.bnd.dtype);
          ast_unvisit();
        }
      }
      acs.aggr_cnt = ast;
    } else if (acs.loop_stmts.doinfo != NULL) {
      handle_dependence_ido(aclp);
    } else {
      /* Non-constant loop control expression(s).
       * Must generate a run-time loop to add sizes.
       */
      int odovar, dovar, sum, sumid, newid, doif;
      DOINFO newdoinfo;
      int ast;

      /* Duplicate loop info, but substitute a new index var. */
      newdoinfo = *doinfo;
      odovar = doinfo->index_var;
      dovar = get_temp(DDTG(DTYPEG(odovar)));
      STYPEP(dovar, STYPEG(odovar));
      DTYPEP(dovar, DTYPEG(odovar));
      if (SCG(odovar) == SC_PRIVATE) {
        SCP(dovar, SC_PRIVATE);
      } else {
        SCP(dovar, SC_LOCAL);
      }
      HIDDENP(dovar, 1);
      newdoinfo.index_var = dovar;
      newid = mk_id(dovar);

      /* Get a temp for the sum and initialize to zero. */
      sum = get_temp(astb.bnd.dtype);
      sumid = mk_id(sum);
      ast = mk_assn_stmt(sumid, astb.bnd.zero, astb.bnd.dtype);
      add_stmt(ast);

      /* Rewrite the size expression to use the new index var. */
      ast_visit(1, 1);
      ast_replace(id, newid);
      ast = ast_rewrite(acs.aggr_cnt);
      ast_unvisit();

      /* Generate the loop. */
      NEED_DOIF(doif, DI_DO);
      add_stmt(do_begin(&newdoinfo));
      ast = mk_binop(OP_ADD, sumid, ast, astb.bnd.dtype);
      ast = mk_assn_stmt(sumid, ast, astb.bnd.dtype);
      add_stmt(ast);
      do_end(&newdoinfo);

      /* Size is now in our sum temporary. */
      acs.aggr_cnt = sumid;
    }
  } else if (A_ALIASG(doinfo->count)) {
    if (add_flag) {
      int v = CONVAL2G(A_SPTRG(A_ALIASG(doinfo->count)));
      acs.scalar_cnt *= v;
      acs.aggr_cnt = mk_binop(OP_MUL, acs.aggr_cnt, mk_cval(v, astb.bnd.dtype),
                              astb.bnd.dtype);
    }
    if (sem.dinit_data) {
      acs.level--;
      DOVARP(doinfo->index_var, 0);
    } else
      acs.is_const = FALSE;
  } else if (sem.dinit_data) {
    /* TODO: why is this not a simple division?? */
    /* must count them */
    int i, v = 0;
    for (i = initval; i <= limitval; i += stepval, v++)
      ;

    acs.scalar_cnt *= v;
    if (v) {
      acs.aggr_cnt = mk_binop(OP_MUL, acs.aggr_cnt, mk_cval(v, astb.bnd.dtype),
                              astb.bnd.dtype);
      acs.level--;
      DOVARP(doinfo->index_var, 0);
    }
  } else {
    if (add_flag) {
      if (acs.scalar_cnt != 0) {
        acs.aggr_cnt = mk_binop(OP_ADD, acs.aggr_cnt,
                                mk_isz_cval(acs.scalar_cnt, astb.bnd.dtype),
                                astb.bnd.dtype);
        acs.scalar_cnt = 0;
      }
      acs.aggr_cnt =
          mk_binop(OP_MUL, doinfo->count, acs.aggr_cnt, astb.bnd.dtype);
    }
    acs.is_const = FALSE;
  }
  if (add_flag) {
    /*
     * fold counts due to the implied do into the totals
     */
    acs.scalar_cnt += save_scalar_cnt;
    acs.aggr_cnt =
        mk_binop(OP_ADD, acs.aggr_cnt, save_aggr_cnt, astb.bnd.dtype);
  }
  if (sem.dinit_data) {
    sem.top--;
  }
}

static void
compute_size_sconst(bool add_flag, ACL *aclp, DTYPE dtype)
{
  if (add_flag) {
    acs.scalar_cnt++;
  }
  if (acs.eltype == 0) {
    acs.eltype = dtype != 0 ? dtype : aclp->dtype;
  }
  compute_size(false, aclp->subc, dtype);
  if (ALLOCFLDG(DTY(aclp->dtype + 3))) {
    acs.is_const = FALSE;
  }
}

static LOGICAL
_can_fold(int ast)
{
  int sptr, asd, ndim, i, b;

  if (ast == 0)
    return FALSE;
  if (A_ALIASG(ast))
    return TRUE;
  switch (A_TYPEG(ast)) {
  case A_ID:
    /*  see if this ident is an active do index variable: */
    sptr = A_SPTRG(ast);
    if (DOVARG(sptr))
      return TRUE;

    /* if the ID has PARAMVAL, subscripts are foldable */
    if (PARAMVALG(sptr))
      return TRUE;
    break;

  case A_MEM:
    return _can_fold(A_PARENTG(ast));

  case A_SUBSCR:
    if (!_can_fold(A_LOPG(ast)))
      return FALSE;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      int ss = ASD_SUBS(asd, i);
      if (!_can_fold(ss))
        return FALSE;
    }
    return TRUE;
    break;

  case A_TRIPLE:
    b = A_LBDG(ast);
    if (b == 0 || !_can_fold(b))
      return FALSE;
    b = A_UPBDG(ast);
    if (b == 0 || !_can_fold(b))
      return FALSE;
    b = A_STRIDEG(ast);
    if (b != 0 && !_can_fold(b))
      return FALSE;
    return TRUE;
    break;

  case A_CNST:
    return TRUE;

  case A_UNOP:
    if (!DT_ISINT(A_DTYPEG(ast)))
      return FALSE;
    if (A_OPTYPEG(ast) == OP_SUB)
      return _can_fold((int)A_LOPG(ast));
    break;

  case A_BINOP:
    if (!DT_ISINT(A_DTYPEG(ast)))
      return FALSE;
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
      if (!_can_fold((int)A_LOPG(ast)))
        return FALSE;
      return _can_fold((int)A_ROPG(ast));
    }
    break;

  case A_CONV:
  case A_PAREN:
    return _can_fold((int)A_LOPG(ast));

  default:
    break;
  }
  return FALSE;
}

/* ------------------------------------------------------------------------- */
/* small routines used by constructf90(). generate subscripts as they are
 * needed. */

static int sub_i = 7;
static int tmpids[MAXDIMS];

static void
init_constructf90()
{
  int i;

  for (i = 0; i < 7; i++) {
    acs.element_cnt[i] = 0;     /* # of individual constructor items  */
    acs.indx[i] = astb.bnd.one; /* subscript of first element */
    acs.indx_tmpid[i] = 0;      /* no subscripting temporary yet */
    acs.subs[i] = astb.bnd.one;
    tmpids[i] = 0;
  }
  sub_i = 7;
}

static int
add_subscript(int base_id, int indexast, DTYPE dtype)
{
  int dest;

  acs.subs[sub_i] = indexast;
  /* generate subscripts as they are seen */
  dest = mk_subscr(base_id, &acs.subs[sub_i], 1, dtype);
  return dest;
}

static int
apply_shape_subscripts(int base_id, int shp, DTYPE dtype)
{
  int dest;
  int i, ndim;
  int ast;
  int subs[MAXDIMS];

  ndim = SHD_NDIM(shp);
  for (i = 0; i < ndim; i++) {
    ast = mk_triple(SHD_LWB(shp, i), SHD_UPB(shp, i), SHD_STRIDE(shp, i));
    subs[i] = ast;
  }
  dest = mk_subscr(base_id, subs, ndim, dtype);
  return dest;
}

static void
push_subscript()
{
  sub_i--;
}

static void
pop_subscript()
{
  sub_i++;
}

static void
clear_element_cnt()
{
  acs.element_cnt[sub_i] = 0;
}

static void
incr_element_cnt()
{
  acs.element_cnt[sub_i]++;
}

static INT
get_element_cnt()
{
  return acs.element_cnt[sub_i];
}

static int
get_subscripting_tmp(int indexast)
{
  int ast;

  if (!tmpids[sub_i])
    tmpids[sub_i] = mk_id(get_temp(astb.bnd.dtype));
  if (indexast != tmpids[sub_i]) {
    ast = mk_assn_stmt(tmpids[sub_i], indexast, astb.bnd.dtype);
    add_stmt(ast);
  }
  return (tmpids[sub_i]);
}

static void
incr_tmp(int tmpid)
{
  int ast;

  ast = mk_binop(OP_ADD, tmpid, astb.bnd.one, astb.bnd.dtype);
  ast = mk_assn_stmt(tmpid, ast, astb.bnd.dtype);
  add_stmt(ast);
}

#define THRESHHOLD 20

static int
size_of_shape_dim(int shape, int i)
{
  int sz;
  if (SHD_LWB(shape, i) == SHD_STRIDE(shape, i)) {
    sz = SHD_UPB(shape, i);
  } else {
    sz = mk_binop(OP_SUB, SHD_UPB(shape, i), SHD_LWB(shape, i), astb.bnd.dtype);
    sz = mk_binop(OP_ADD, sz, SHD_STRIDE(shape, i), astb.bnd.dtype);
  }
  if (SHD_STRIDE(shape, i) != astb.bnd.one) {
    sz = mk_binop(OP_DIV, sz, SHD_STRIDE(shape, i), astb.bnd.dtype);
  }
  return sz;
} /* size_of_shape_dim */

static int
get_shape_arraydtype(int shape, int eltype)
{
  int arrtype, i, n;
  int sz;
  LOGICAL need_alloc = FALSE;

  n = sem.arrdim.ndim = SHD_NDIM(shape);
  sem.arrdim.ndefer = 0;

  for (i = 0; i < n; ++i) {
    sem.bounds[i].lowtype = S_CONST;
    sem.bounds[i].lowb = 1;
    sem.bounds[i].lwast = 0;

    sz = size_of_shape_dim(shape, i);
    if (A_ALIASG(sz) && (ad_val_of(A_SPTRG(A_ALIASG(sz))) < THRESHHOLD)) {
      /* small constant size */
      sem.bounds[i].uptype = S_CONST;
      sem.bounds[i].upb = ad_val_of(A_SPTRG(A_ALIASG(sz)));
      sem.bounds[i].upast = sz;
    } else {
      sem.bounds[i].uptype = S_EXPR;
      sem.bounds[i].upb = 0;
      sem.bounds[i].upast = sz;
      need_alloc = TRUE;
    }
  }

  if (is_deferlenchar_dtype(acs.arrtype))
    need_alloc = TRUE;

  if (need_alloc)
    sem.arrdim.ndefer = n;
  arrtype = mk_arrdsc();
  DTY(arrtype + 1) = eltype;
  return arrtype;
} /* get_shape_arraydtype */

static void
mkexpr_assign_temp(SST *stkptr)
{
  int ast, a, simple;
  DTYPE dtype;
  int dest;
  int id;

  mkexpr(stkptr);
  /* may have to change to create temp based on shape if we are in
     structure and doing array assignment of a multiple dimension array. */

  simple = 1;
  ast = SST_ASTG(stkptr);
  for (a = ast; a > 0;) {
    switch (A_TYPEG(a)) {
    case A_ID:
      a = 0;
      break;
    case A_MEM:
      a = A_PARENTG(a);
      break;
    default:
      simple = 0;
      a = 0;
      break;
    }
  }
  /* if we have an array expression, we need to assign it to
     a temporary so that we can subscript it. */
  if (DTY(dtype = SST_DTYPEG(stkptr)) == TY_ARRAY && !simple) {
    if (is_deferlenchar_ast(ast)) {
      dtype = get_shape_arraydtype(A_SHAPEG(ast), DTY(acs.arrtype + 1));
    } else {
      dtype = get_shape_arraydtype(A_SHAPEG(ast), DTY(dtype + 1));
    }
    id = get_arr_temp(dtype, FALSE, FALSE, FALSE);
    if (sem.arrdim.ndefer)
      gen_allocate_array(id);
    ast = ast_rewrite_indices(ast);
    dest = mk_id(id);
    ast = mk_assn_stmt(dest, ast, dtype);
    add_stmt(ast);
    SST_ASTP(stkptr, dest);
  }
}

/* if we have a%b, a and b are arrays, subscripts i,j,
 * turn this into a(i)%b(j); this is overkill, since F90
 * only allows one vector subscript in a member tree */
static int
add_dt_subscr(int ast, int *subs, int numdim)
{
  int lop, dtype;
  switch (A_TYPEG(ast)) {
  case A_SUBSCR:
    /* already have the subscripts */
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      assert(numdim == 0, "add_dt_subscr: too many subscripts", numdim, 3);
    } else if (A_TYPEG(lop) == A_MEM) {
      int parent, mem, asd, ndim, i, oldsubs[MAXDIMS];
      parent = add_dt_subscr(A_PARENTG(lop), subs, numdim);
      mem = A_MEMG(lop);
      dtype = DTYPEG(A_SPTRG(mem));
      mem = mk_member(parent, mem, dtype);
      asd = A_ASDG(ast);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i) {
        oldsubs[i] = ASD_SUBS(asd, i);
      }
      ast = mk_subscr(mem, oldsubs, ndim, DTY(dtype + 1));
    } else {
      interr("add_dt_subscr: unexpected subscript parent", A_TYPEG(lop), 3);
    }
    break;

  case A_MEM:
    dtype = DTYPEG(A_SPTRG(A_MEMG(ast)));
    /* apply subscripts? */
    if (DTY(dtype) != TY_ARRAY) {
      int parent;
      parent = add_dt_subscr(A_PARENTG(ast), subs, numdim);
      ast = mk_member(parent, A_MEMG(ast), dtype);
    } else {
      int parent, ndim, odim;
      /* take some subscripts here */
      ndim = ADD_NUMDIM(dtype);
      odim = numdim - ndim;
      assert(odim >= 0, "add_dt_subscr: not enough subscripts", numdim - ndim,
             3);
      parent = add_dt_subscr(A_PARENTG(ast), subs, odim);
      ast = mk_member(parent, A_MEMG(ast), dtype);
      ast = mk_subscr(ast, subs + odim, ndim, DTY(dtype + 1));
    }
    break;
  case A_ID:
    dtype = DTYPEG(A_SPTRG(ast));
    /* apply subscripts? */
    if (DTY(dtype) != TY_ARRAY) {
      assert(numdim == 0, "add_dt_subscr: too many subscripts", numdim, 3);
    } else {
      int ndim;
      /* take rest of subscripts here */
      ndim = ADD_NUMDIM(dtype);
      assert(ndim == numdim, "add_dt_subscr: wrong number of subscripts",
             numdim - ndim, 3);
      ast = mk_subscr(ast, subs, ndim, DTY(dtype + 1));
    }
    break;
  }
  return ast;
} /* add_dt_subscr */

static int oldindex[MAXDIMS], newindex[MAXDIMS], numindex;

static void
ast_replace_index(int old, int new)
{
  oldindex[numindex] = old;
  newindex[numindex] = new;
  ++numindex;
} /* ast_replace_index */

static int
ast_rewrite_indices(int ast)
{
  int i, newast;
  ast_visit(1, 1);
  for (i = 0; i < numindex; ++i) {
    ast_replace(oldindex[i], newindex[i]);
  }
  newast = ast_rewrite(ast);
  ast_unvisit();
  return newast;
} /* ast_rewrite_indices */

static ACL *
acl_rewrite_asts(ACL *aclp)
{
  int ast, initast, limitast, countast, stepast;
  SST *stkp, *sst;
  DOINFO *doinfo;
  ACL *newaclp, *subc, *next;

  newaclp = 0;
  if (aclp->next) {
    next = acl_rewrite_asts(aclp->next);
    if (next != aclp->next) {
      newaclp = GET_ACL(15);
      *newaclp = *aclp;
      newaclp->next = next;
    }
  }
  switch (aclp->id) {
  case AC_AST:
    ast = ast_rewrite(aclp->u1.ast);
    if (ast != aclp->u1.ast) {
      if (newaclp == 0) {
        newaclp = GET_ACL(15);
        *newaclp = *aclp;
      }
      newaclp->u1.ast = ast;
    }
    break;
  case AC_EXPR:
    stkp = aclp->u1.stkp;
    ast = SST_ASTG(stkp);
    switch (SST_IDG(stkp)) {
    case S_ACONST:
      break;
    case S_CONST:
      ast = ast_rewrite(ast);
      break;
    case S_SCONST:
      ast = ast_rewrite(ast);
      break;
    case S_EXPR:
      ast = ast_rewrite(ast);
      break;
    case S_LVALUE:
      ast = ast_rewrite(ast);
      break;
    case S_IDENT:
      ast = ast_rewrite(ast);
      break;
    default:
      interr("acl_rewrite_asts: unknown expr type", SST_IDG(stkp), 3);
      break;
    }
    if (ast != SST_ASTG(stkp)) {
      NEW(sst, SST, SST_SIZE);
      if (sst == NULL)
        error(7, 4, 0, CNULL, CNULL);
      *sst = *stkp;
      SST_ASTP(sst, ast);
      if (newaclp == 0) {
        newaclp = GET_ACL(15);
        *newaclp = *aclp;
      }
      newaclp->u1.stkp = sst;
    }
    break;
  case AC_ACONST:
  case AC_SCONST:
  case AC_REPEAT:
    subc = acl_rewrite_asts(aclp->subc);
    if (subc != aclp->subc) {
      if (newaclp == 0) {
        newaclp = GET_ACL(15);
        *newaclp = *aclp;
      }
      newaclp->subc = subc;
    }
    break;
  case AC_IDO:
    doinfo = aclp->u1.doinfo;
    initast = ast_rewrite(doinfo->init_expr);
    limitast = ast_rewrite(doinfo->limit_expr);
    stepast = ast_rewrite(doinfo->step_expr);
    countast = ast_rewrite(doinfo->count);
    if (initast != doinfo->init_expr || limitast != doinfo->limit_expr ||
        stepast != doinfo->step_expr || countast != doinfo->count) {
      doinfo = get_doinfo(15);
      *doinfo = *(aclp->u1.doinfo);
      doinfo->init_expr = initast;
      doinfo->limit_expr = limitast;
      doinfo->step_expr = stepast;
      doinfo->count = countast;
    }
    subc = acl_rewrite_asts(aclp->subc);
    if (doinfo != aclp->u1.doinfo || subc != aclp->subc) {
      if (newaclp == 0) {
        newaclp = GET_ACL(15);
        *newaclp = *aclp;
      }
      newaclp->subc = subc;
      newaclp->u1.doinfo = doinfo;
    }
    break;
  default:
    interr("acl_rewrite_asts: unknown ACL id", aclp->id, 3);
    break;
  }
  return newaclp ? newaclp : aclp;
} /* acl_rewrite_asts */

static int
gen_null_intrin()
{
  int func_ast, ast;
  func_ast = mk_id(intast_sym[I_NULL]);
  ast = mk_func_node(A_INTR, func_ast, 0, 0);
  A_DTYPEP(ast, DT_WORD);
  EXPSTP(intast_sym[I_NULL], 1);
  A_OPTYPEP(ast, I_NULL);
  return ast;
}

static int
_constructf90(int base_id, int in_indexast, bool in_array, ACL *aclp)
{
  int i;
  SST *stkp;
  DOINFO *doinfo;
  int ast;
  DTYPE dtype;
  int odovar, dovar;
  int dest;
  int src_subs[MAXDIMS];
  int src;
  int tmpsptr;
  int mem_sptr, mem_sptr_id, cmem_sptr;
  ACL *mem_aclp;
  ACL *tmp;
  int tmpid;
  int indexast;
  INT cnt;
  LOGICAL sdscismbr;
  int argt = 0;
  int argt_count = 0;

  indexast = in_indexast;

#if DEBUG
  if (DBGBIT(3, 64))
    printacl("_constructf90", aclp, gbl.dbgfil);
#endif

  for (; aclp != NULL; aclp = aclp->next) {
    switch (aclp->id) {
    case AC_ACONST:
      if (in_array) {
        indexast = _constructf90(base_id, indexast, true, aclp->subc);
      } else {
        push_subscript();
        indexast = _constructf90(base_id, SHD_LWB(A_SHAPEG(base_id), 0), true,
                                 aclp->subc);
        pop_subscript();
      }
      break;
    case AC_SCONST:
      mem_aclp = aclp->subc;
      dtype = aclp->dtype;
      if (in_array)
        dest = add_subscript(base_id, indexast, dtype);
      else
        dest = base_id;
      dtype = DDTG(dtype);

      mem_sptr = DTY(dtype + 1);
      for (; mem_sptr != NOSYM; mem_sptr = SYMLKG(mem_sptr)) {
        if (!is_unl_poly(mem_sptr) && no_data_components(DTYPEG(mem_sptr)))
          continue;
        /* skip $td */
        if (CLASSG(mem_sptr) && DESCARRAYG(mem_sptr))
          continue;
        if (XBIT(58, 0x10000) && POINTERG(mem_sptr) && !F90POINTERG(mem_sptr)) {
          SST *astkp;
          int aast;
          int stmtast, asptr;
          if (!mem_aclp) {
            /* Check to see if there's a default
             * initialization for this missing element in the
             * structure constructor. If not, then issue an
             * error message.
             */
            mem_aclp = get_struct_default_init(mem_sptr);
            if (!mem_aclp) {
              error(155, 3, gbl.lineno, "No default initialization for",
                    SYMNAME(mem_sptr));
              mem_aclp = GET_ACL(15);
              mem_aclp->id = AC_AST;
              mem_aclp->dtype = DT_PTR;
              mem_aclp->u1.ast = astb.i0;
            }
          }
          if (mem_aclp->id == AC_AST &&
             (mem_aclp->dtype == DT_PTR || POINTERG(mem_sptr)) &&
              mem_aclp->u1.ast == astb.i0) {
            /* Convert this to NULL then assign ptr */
            aast = gen_null_intrin();
          } else if (DTY(DTYPEG(mem_sptr)) == TY_PTR &&
                     DTY(DTY(DTYPEG(mem_sptr) + 1)) == TY_PROC) {
            /* cannot call mkexpr which later call mkexpr1
             * for procedure(subroutine) assignment of
             * derived type in structure constructor.
             */
            mkexpr2(mem_aclp->u1.stkp);
            astkp = mem_aclp->u1.stkp;
            aast = SST_ASTG(astkp);
          } else {
            mkexpr(mem_aclp->u1.stkp);
            astkp = mem_aclp->u1.stkp;
            aast = SST_ASTG(astkp);
          }
          if ((A_TYPEG(aast) == A_INTR && A_OPTYPEG(aast) == I_NULL) ||
              (DTY(DTYPEG(mem_sptr)) == TY_PTR &&
               DTY(DTY(DTYPEG(mem_sptr) + 1)) == TY_PROC)) {

            if (!(A_TYPEG(aast) == A_INTR && A_OPTYPEG(aast) == I_NULL))
              (void)chk_pointer_target(mem_sptr, aast);

            stmtast = add_ptr_assign(mkmember(dtype, dest, NMPTRG(mem_sptr)),
                                     aast, 0);
            add_stmt(ast_rewrite_indices(stmtast));
            mem_aclp = mem_aclp->next;
            if (SDSCG(mem_sptr) && STYPEG(SDSCG(mem_sptr)) == ST_MEMBER) {
              cmem_sptr = mem_sptr;
              if (SYMLKG(mem_sptr) == MIDNUMG(cmem_sptr)) {
                /* point to pointer */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (SYMLKG(mem_sptr) == PTROFFG(cmem_sptr)) {
                /* point to offset */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (SYMLKG(mem_sptr) == SDSCG(cmem_sptr)) {
                /* point to sdsc */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (CLASSG(cmem_sptr) && DESCARRAYG(mem_sptr)) {
                /* points to $td */
                mem_sptr = SYMLKG(mem_sptr);
              }
            } else if (MIDNUMG(mem_sptr)) {
              mem_sptr = MIDNUMG(mem_sptr); /* skip $o, $sd, $p */
            }
          } else if (SDSCG(mem_sptr)) {
            (void)chk_pointer_target(mem_sptr, aast);
            astkp = mem_aclp->u1.stkp;
            i = NMPTRG(mem_sptr);
            if (SST_IDG(astkp) == S_IDENT) {
              asptr = SST_SYMG(astkp);
              aast = mk_id(asptr);
            } else if (SST_IDG(astkp) == S_LVALUE) {
              aast = mem_aclp->u1.stkp->ast;
              if (aast == 0) {
                asptr = SST_LSYMG(astkp);
                aast = mk_id(asptr);
              }
            } else {
              aast = mem_aclp->u1.stkp->ast;
            }
            if (STYPEG(SDSCG(mem_sptr)) == ST_MEMBER) {
              /* do a 'pointer-assign' here. skip over
               * base pointer/offset/descriptor */
              stmtast = add_ptr_assign(mkmember(dtype, dest, i), aast, 0);
              (void)add_stmt(ast_rewrite_indices(stmtast));
              cmem_sptr = mem_sptr;
              if (SYMLKG(mem_sptr) == MIDNUMG(cmem_sptr)) {
                /* point to pointer */
                mem_sptr = SYMLKG(mem_sptr);
              }
              mem_aclp = mem_aclp->next;
              if (SYMLKG(mem_sptr) == PTROFFG(cmem_sptr)) {
                /* point to offset */
                mem_sptr = SYMLKG(mem_sptr);
              }
              mem_aclp = mem_aclp->next;
              if (SYMLKG(mem_sptr) == SDSCG(cmem_sptr)) {
                /* point to sdsc */
                mem_sptr = SYMLKG(mem_sptr);
              }
              mem_aclp = mem_aclp->next;
              if (CLASSG(cmem_sptr) && DESCARRAYG(mem_sptr)) {
                /* points to $td, no aclp, part of sdsc */
                mem_sptr = SYMLKG(mem_sptr);
              }
              mem_aclp = mem_aclp->next; /* past sdsc */
            } else {
              stmtast = add_ptr_assign(mkmember(dtype, dest, i), aast, 0);
              (void)add_stmt(ast_rewrite_indices(stmtast));
              mem_aclp = mem_aclp->next;
              mem_sptr = MIDNUMG(mem_sptr); /* skip $o, $sd, $p */
            }
          } else {
            mem_aclp = mem_aclp->next; /* skip pointee */
          }
          continue;
        } else if (ALLOCATTRG(mem_sptr)) {
          int stmt, orig_mem_sptr;
          ast = mk_id(mem_sptr);
          orig_mem_sptr = mem_sptr;
          if (mem_aclp->id == AC_ACONST) {
            mem_sptr_id = mk_member(dest, ast, DTYPEG(mem_sptr));
            tmpsptr = getcctmp_sc('f', sem.dtemps++, ST_ARRAY, mem_aclp->dtype,
                                  SC_STATIC);
            NODESCP(tmpsptr, 0);
            tmp = clone_init_const(mem_aclp, FALSE);
            init_sptr_w_acl(tmpsptr, tmp);
            acs.is_const = 0;
            ast = mk_id(tmpsptr);
            ast = mk_assn_stmt(mem_sptr_id, ast, mem_aclp->dtype);
            stmt = add_stmt(ast);
            /* need init $p $sd */
            (void)add_stmt_before(add_nullify_ast(mem_sptr_id), stmt);
          } else if (mem_aclp->id == AC_SCONST) {
            if (is_unl_poly(mem_sptr)) {
              mem_sptr_id = mk_member(dest, ast, mem_aclp->dtype);
            } else {
              mem_sptr_id = mk_member(dest, ast, DTYPEG(mem_sptr));
            }
            tmpsptr = getcctmp_sc('f', sem.dtemps++, ST_VAR, mem_aclp->dtype,
                                  SC_STATIC);
            NODESCP(tmpsptr, 0);
            tmp = clone_init_const(mem_aclp, FALSE);
            init_derived_w_acl(tmpsptr, tmp);
            acs.is_const = 0;
            ast = mk_id(tmpsptr);
            ast = mk_assn_stmt(mem_sptr_id, ast, mem_aclp->dtype);
            stmt = add_stmt(ast);

          } else if (mem_aclp->id == AC_EXPR &&
                     A_TYPEG(mem_aclp->u1.stkp->ast) == A_INTR &&
                     A_OPTYPEG(mem_aclp->u1.stkp->ast) == I_NULL) {
            mem_sptr_id = mk_member(dest, ast, DTYPEG(mem_sptr));
            ast = add_nullify_ast(mem_sptr_id);
            stmt = add_stmt(ast);
          } else if ((DTYPEG(mem_sptr)) == DT_DEFERCHAR ||
                     (DTYPEG(mem_sptr)) == DT_DEFERNCHAR) {

            mem_sptr_id = mk_member(dest, ast, DTYPEG(mem_sptr));
            if (mem_aclp->id == AC_AST && mem_aclp->u1.ast == astb.i0) {
              ast = add_nullify_ast(mem_sptr_id);
            } else {
              ast = add_nullify_ast(mem_sptr_id);
              stmt = add_stmt(ast);
              mkexpr(mem_aclp->u1.stkp);
              ast = mem_aclp->u1.stkp->ast;
              ast = mk_assn_stmt(mem_sptr_id, ast, A_DTYPEG(ast));
            }

            stmt = add_stmt(ast);

            if (SDSCG(mem_sptr) && STYPEG(SDSCG(mem_sptr)) == ST_MEMBER) {
              cmem_sptr = mem_sptr;
              if (SYMLKG(mem_sptr) == MIDNUMG(cmem_sptr)) {
                /* point to pointer */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (SYMLKG(mem_sptr) == PTROFFG(cmem_sptr)) {
                /* point to offset */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (SYMLKG(mem_sptr) == SDSCG(cmem_sptr)) {
                /* point to sdsc */
                mem_sptr = SYMLKG(mem_sptr);
              }
              if (CLASSG(cmem_sptr) && DESCARRAYG(mem_sptr)) {
                /* points to $td */
                mem_sptr = SYMLKG(mem_sptr);
              }
            } else {
              mem_sptr = MIDNUMG(mem_sptr); /* skip $o, $sd, $p */
            }
            mem_aclp = mem_aclp->next;
            continue;

          } else {
            if (mem_aclp->id == AC_EXPR && is_unl_poly(mem_sptr)) {
              mem_sptr_id = mk_member(dest, ast, SST_DTYPEG(mem_aclp->u1.stkp));
            } else {
              mem_sptr_id = mk_member(dest, ast, DTYPEG(mem_sptr));
            }
            if (mem_aclp->id == AC_AST && mem_aclp->u1.ast == astb.i0) {
              ast = add_nullify_ast(mem_sptr_id);
            } else {
              mkexpr(mem_aclp->u1.stkp);
              ast = mem_aclp->u1.stkp->ast;
              ast = mk_assn_stmt(mem_sptr_id, ast, A_DTYPEG(ast));
            }
            stmt = add_stmt(ast);
          }

          sdscismbr = (SDSCG(mem_sptr) && STYPEG(SDSCG(mem_sptr)) == ST_MEMBER);
          mem_sptr = SYMLKG(mem_sptr); /* point to pointer */
          mem_aclp = mem_aclp->next;
          if (sdscismbr) {
            mem_sptr = SYMLKG(mem_sptr); /* point to offset */
            if (DTY(DTYPEG(orig_mem_sptr)) == TY_ARRAY)
              mem_sptr = SYMLKG(mem_sptr); /* point to sdsc */
          }
          continue;
        }
        i = NMPTRG(mem_sptr);
        mem_sptr_id = mkmember(dtype, dest, i);
        if (mem_aclp == 0) {
          /* interr("ran out of aclp",sptr,2); */
          break;
        }
        tmp = mem_aclp->next;
        mem_aclp->next = 0; /* decouple aclp */
        i = _constructf90(mem_sptr_id, 0, false, mem_aclp);
        mem_aclp->next = tmp; /* relink behind us */
        mem_aclp = tmp;
      }
      if (in_array) {
        indexast = mk_binop(OP_ADD, indexast, astb.bnd.one, astb.bnd.dtype);
        incr_element_cnt();
      }
      break;
    case AC_EXPR:
      stkp = aclp->u1.stkp;
      if (in_array)
        mkexpr_assign_temp(stkp);
      else
        mkexpr(stkp);
      dtype = SST_DTYPEG(stkp);
      if (DTY(dtype) == TY_ARRAY) {
        /* constructor item is an array */
        int shp;
        int shpdest;
        int ndim;
        int iv;
        int alloc_obj = 0; /* used to dealloc array function return variable */

        if (!in_array) {
          /* handle case where a (possibly multiple dimensioned
             array is assigned to a structure element. */
          src = SST_ASTG(stkp);
          shp = A_SHAPEG(src);
          dest = base_id;
          shpdest = A_SHAPEG(dest);
          ndim = SHD_NDIM(shp);
          add_shape_rank(ndim);
          for (i = 0; i < ndim; i++) {
            ast = extent_of_shape(shp, i);
            ast = mk_binop(
                OP_SUB,
                mk_binop(OP_ADD, SHD_LWB(shpdest, i), ast, astb.bnd.dtype),
                astb.i1, astb.bnd.dtype);
            add_shape_spec(SHD_LWB(shpdest, i), ast, astb.i1);
          }
          shpdest = mk_shape();
          dest = apply_shape_subscripts(base_id, shpdest, dtype);
          ast = mk_assn_stmt(dest, src, dtype);
          ast = ast_rewrite_indices(ast);
          (void)add_stmt(ast);
          break;
        }

        /* In the loop generated from implied-do loop, we encoutered the AST
         * used as the return value to replace the function call.
         */
        if (acs.level && sem.arrfn.sptr > NOSYM &&
            sem.arrfn.return_value == SST_ASTG(stkp)) {
          int ret_sptr = A_SPTRG(sem.arrfn.return_value);
          acs.func_in_do = TRUE;
          /* STDs generated in func_call2(semfunc.c) have been move in loop
           * when processing AC_IDO. We do not need move in sem.arrfn.alloc_std
           * and sem.arrfn.call_std again.
           */
          if (sem.arrfn.alloc_std) {
            /* remove return variable from sem.p_dealloc */
            ITEM *pre = NULL;
            for (ITEM *itemp = sem.p_dealloc; itemp; itemp = itemp->next) {
              if (A_SPTRG(itemp->ast) == ret_sptr) {
                if (pre != NULL)
                  pre->next = itemp->next;
                else
                  sem.p_dealloc = itemp->next;
                alloc_obj = itemp->ast;
                break;
              }
              pre = itemp;
            }
          }
          if (ADD_DEFER(SST_DTYPEG(stkp))) {
            /* If the function call returns deferred-length array or adjustable
             * array, we use the sum of the current length of the array temp
             * and the length of the array returned by the function as the
             * length of the array temp after reallocation.
             */
            int src_array = memsym_of_ast(base_id);
            if (!SDSCG(ret_sptr))
              get_static_descriptor(ret_sptr);
            argt_count = 3;
            argt = mk_argt(argt_count);
            ARGT_ARG(argt, 0) = mk_id(MIDNUMG(src_array));
            ARGT_ARG(argt, 1) = mk_id(SDSCG(src_array));
            ARGT_ARG(argt, 2) = mk_id(SDSCG(ret_sptr));
            DESCUSEDP(src_array, 1);
            DESCUSEDP(ret_sptr, 1);
            ast = mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(
                RTE_realloc_arr_in_impiled_do), DT_ADDR)), argt_count, argt);
            (void)add_stmt(ast);
          }
        }

        tmpid = get_subscripting_tmp(indexast);

        /*  get do begins for src array objects */
        shp = A_SHAPEG(SST_ASTG(stkp));
        ndim = SHD_NDIM(shp);
        for (i = ndim - 1; i >= 0; i--) {
          iv = get_temp(astb.bnd.dtype);
          ast = mk_stmt(A_DO, 0);
          dovar = mk_id(iv);
          A_DOVARP(ast, dovar);
          A_M1P(ast, SHD_LWB(shp, i));
          A_M2P(ast, SHD_UPB(shp, i));
          A_M3P(ast, SHD_STRIDE(shp, i));
          ast = ast_rewrite_indices(ast);
          (void)add_stmt(ast);
          src_subs[i] = A_DOVARG(ast);
        }

        src = add_dt_subscr(SST_ASTG(stkp), src_subs, ndim);

        dest = add_subscript(base_id, tmpid, DTY(dtype + 1));

        ast = mk_assn_stmt(dest, src, DTY(dtype + 1));
        ast = ast_rewrite_indices(ast);
        (void)add_stmt(ast);

        /* increment the subscripting temporary */
        incr_tmp(tmpid);

        for (i = 0; i < ndim; i++) {
          ast = mk_stmt(A_ENDDO, 0);
          (void)add_stmt(ast);
        }

        /* dealloc array function return variable */
        if (alloc_obj)
          (void)gen_alloc_dealloc(TK_DEALLOCATE, alloc_obj, 0);
        clear_element_cnt();
        indexast = tmpid;
      } else {
        /* constructor item is a scalar */
        src = SST_ASTG(stkp);
        dest = base_id;
        dtype = A_DTYPEG(dest);
        if (in_array) {
          dtype = DDTG(dtype);
          dest = add_subscript(dest, indexast, dtype);
        }
        if (DTY(dtype) != TY_ARRAY && ast_is_sym(src) &&
            has_layout_desc(memsym_of_ast(src))) {
          int argt, dest_td_sym, src_td_sym;
          dest_td_sym = getccsym('d', sem.dtemps++, ST_VAR);
          DTYPEP(dest_td_sym, dtype);
          src_td_sym = getccsym('d', sem.dtemps++, ST_VAR);
          DTYPEP(src_td_sym, A_DTYPEG(src));
          argt = mk_argt(5);
          ARGT_ARG(argt, 0) = dest;
          ARGT_ARG(argt, 1) = mk_id(get_static_type_descriptor(dest_td_sym));
          ARGT_ARG(argt, 2) = src;
          ARGT_ARG(argt, 3) = mk_id(get_static_type_descriptor(src_td_sym));
          ARGT_ARG(argt, 4) = mk_unop(OP_VAL, mk_cval1(1, DT_INT), DT_INT);
          ast = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(RTE_poly_asn), DT_NONE));
          ast = mk_func_node(A_CALL, ast, 5, argt);
        } else {
          ast = mk_assn_stmt(dest, src, dtype);
        }
        ast = ast_rewrite_indices(ast);
        (void)add_stmt(ast);
        if (in_array) {
          indexast = mk_binop(OP_ADD, indexast, astb.bnd.one, astb.bnd.dtype);
          incr_element_cnt();
        }
      }
      break;
    case AC_IDO:
      tmpid = get_subscripting_tmp(indexast);

      acs.level++;
      clear_element_cnt();
      doinfo = aclp->u1.doinfo;
      /* for array constructor, we must create a new symbol
       * for the implied 'do' loop */
      odovar = doinfo->index_var;
      /* insert a new one */
      dovar = get_temp(DDTG(DTYPEG(odovar)));
      STYPEP(dovar, STYPEG(odovar));
      DTYPEP(dovar, DTYPEG(odovar));
      if (SCG(odovar) == SC_PRIVATE) {
        SCP(dovar, SC_PRIVATE);
      } else {
        SCP(dovar, SC_LOCAL);
      }
      HIDDENP(dovar, 1);
      ast_replace_index(mk_id(odovar), mk_id(dovar));
      doinfo->index_var = dovar;
      ast = do_begin(doinfo);
      ast = ast_rewrite_indices(ast);

      /* Folling line of code is an extension, where we allow
       * a ac-do-variable to be referenced in limit expression.
       * Do not rewrite ast of limit_expr. For example,
       * do i = 1, n
       *   x = (/i,i = 1,fox(i)/)
       * end do
       * i in fox(i) is from do i=1, not implied-do-variable i
       */

      if (!XBIT(57, 0x4000))
        A_M2P(ast, doinfo->limit_expr);

      (void)add_stmt(ast);
      /* Value-list must be rewritten too. */
      ast_visit(1, 1);
      ast_replace(mk_id(odovar), mk_id(dovar));
      if (aclp->u2.std_range != NULL && aclp->u2.std_range->start != 0) {
        /* move in stmts generated by loop body and replace dovar */
        int do_std = STD_LAST;
        move_range_after(aclp->u2.std_range->start, aclp->u2.std_range->mid,
                         do_std);
        for (int std = STD_NEXT(do_std); std; std = STD_NEXT(std)) {
          STD_AST(std) = ast_rewrite(STD_AST(std));
        }
      }
      aclp->subc = acl_rewrite_asts(aclp->subc);
      ast_unvisit();

      _constructf90(base_id, tmpid, in_array, aclp->subc);

      if ((cnt = get_element_cnt())) {
        /* increment the subscripting temporary */
        i = mk_isz_cval(cnt, astb.bnd.dtype);
        i = mk_binop(OP_ADD, tmpid, i, astb.bnd.dtype);
        ast = mk_assn_stmt(tmpid, i, astb.bnd.dtype);
        ast = ast_rewrite_indices(ast);
        (void)add_stmt(ast);
      }

      NEED_DOIF(i, DI_DO); /* need a loop stack entry for do_end() */
      do_end(doinfo);
      --numindex; /* done with this loop */
      indexast = tmpid;
      clear_element_cnt();
      acs.level--;
      break;
    case AC_AST: /* default init */
      ast = aclp->u1.ast;
      dtype = A_DTYPEG(ast);

      if (is_iso_cptr(dtype)) {
        mem_sptr = DTY(dtype + 1);
        ast = mkmember(dtype, ast, NMPTRG(mem_sptr));
      }

      if (in_array) {
        dtype = DDTG(A_DTYPEG(base_id));
        dest = add_subscript(base_id, indexast, dtype);
      } else {
        dtype = A_DTYPEG(base_id);
        dest = base_id;
      }

      ast = mk_assn_stmt(dest, ast, dtype);

      ast = ast_rewrite_indices(ast);
      (void)add_stmt(ast);
      if (in_array) {
        indexast = mk_binop(OP_ADD, indexast, astb.bnd.one, astb.bnd.dtype);
        incr_element_cnt();
      }
      break;
    case AC_IEXPR:
      break;
    default:
      interr("_construct,ill.id", aclp->id, 3);
      break;
    }
  }

  return indexast;
}

static void
constructf90(int arr, ACL *aclp)
{
  DTYPE dtype;
  int lower;
  bool inarray;

  init_constructf90();

  acs.level = 0;
  acs.width = compute_width(aclp);

  dtype = DTYPEG(arr);
  inarray = DTY(dtype) == TY_ARRAY;
  if (inarray) {
    lower = ADD_LWAST(dtype, 0);
    if (lower == 0)
      lower = astb.bnd.one;
    push_subscript();
  } else {
    lower = astb.bnd.one;
  }

  acs.tmpid = mk_id(arr);

  numindex = 0;
  _constructf90(acs.tmpid, lower, inarray, aclp);

  if (DTY(dtype) == TY_ARRAY) {
    pop_subscript();
  }

  if (sub_i != 7)
    interr("sub_i in constructf90 is not back", sub_i, 2);
}

ACL *
mk_init_intrinsic(AC_INTRINSIC init_intr)
{
  AEXPR *aexpr;
  ACL *expracl = GET_ACL(15);

  expracl->id = AC_IEXPR;
  expracl->u1.expr = aexpr = (AEXPR *)getitem(15, sizeof(AEXPR));
  BZERO(aexpr, AEXPR, 1);
  aexpr->op = AC_INTR_CALL;
  aexpr->lop = GET_ACL(15);
  aexpr->lop->id = AC_ICONST;
  aexpr->lop->u1.i = init_intr;

  return expracl;
}

static ACL *
mk_ulbound_intrin(AC_INTRINSIC intrin, int ast)
{
  ACL *argacl;
  ACL *dimval;
  ACL **r;
  AEXPR *aexpr;
  int ubound[MAXDIMS];
  int lbound[MAXDIMS];
  int i;
  LOGICAL must_convert;
  ACL *expracl = mk_init_intrinsic(intrin);
  int arg_count = A_ARGCNTG(ast);
  int argt = A_ARGSG(ast);
  int argast = ARGT_ARG(argt, 0);
  int shape = A_SHAPEG(argast);
  int rank = SHD_NDIM(shape);
  int dtyper;

  for (i = 0; i < rank; i++) {
    if (A_TYPEG(argast) == A_ID) {
      ubound[i] = ubound_of_shape(shape, i);
      lbound[i] = lbound_of_shape(shape, i);
    } else {
      ubound[i] = extent_of_shape(shape, i);
      lbound[i] = astb.i1;
    }
  }

  aexpr = expracl->u1.expr;

  argacl = aexpr->rop = GET_ACL(15);
  argacl->id = AC_ACONST;
  sem.arrdim.ndim = 1;
  sem.arrdim.ndefer = 0;
  sem.bounds[0].lowtype = S_CONST;
  sem.bounds[0].lowb = 1;
  sem.bounds[0].lwast = 0;
  sem.bounds[0].uptype = S_CONST;
  sem.bounds[0].upb = rank;
  sem.bounds[0].upast = mk_cval(rank, stb.user.dt_int);
  dtyper = mk_arrdsc();
  DTY(dtyper + 1) = stb.user.dt_int;
  argacl->dtype = dtyper;

  must_convert = FALSE;
  if (arg_count == 2 && argacl->dtype != stb.user.dt_int)
    must_convert = TRUE;

  r = &argacl->subc;
  for (i = 0; i < rank; i++) {
    *r = GET_ACL(15);
    (*r)->id = AC_AST;
    (*r)->dtype = stb.user.dt_int;
    (*r)->is_const = TRUE;
    if (intrin == AC_I_ubound) {
      (*r)->u1.ast = ubound[i];
    } else {
      (*r)->u1.ast = lbound[i];
    }
    if (must_convert) {
      (*r)->u1.ast = mk_convert((*r)->u1.ast, stb.user.dt_int);
    }
    r = &(*r)->next;
  }

  if (arg_count == 2) {
    argast = ARGT_ARG(argt, 1);
    if (!_can_fold(argast)) {
      error(87, 3, gbl.lineno, NULL, NULL);
    }
    argacl = construct_acl_from_ast(argast, stb.user.dt_int, 0);
    if (!argacl) {
      return 0;
    }
    aexpr->rop->next = argacl;
    expracl->dtype = stb.user.dt_int;

    dimval = eval_init_expr_item(argacl);
    if (!dimval) {
      return 0;
    }
    i = dimval->conval;
    if (dimval->dtype == DT_INT8)
      i = get_int_cval(i);
    if ((intrin == AC_I_ubound && !_can_fold(ubound[i - 1])) ||
        (intrin == AC_I_lbound && !_can_fold(lbound[i - 1]))) {
      error(87, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    }
  } else {
    for (i = 0; i < rank; i++) {
      if ((intrin == AC_I_ubound && !_can_fold(ubound[i])) ||
          (intrin == AC_I_lbound && !_can_fold(lbound[i]))) {
        error(87, 3, gbl.lineno, NULL, NULL);
        sem.dinit_error = TRUE;
        return 0;
      }
    }
    expracl->dtype = A_DTYPEG(ast);
    ;
  }

  return expracl;
}

static ACL *
mk_transpose_intrin(int ast)
{
  ACL *expracl = mk_init_intrinsic(AC_I_transpose);
  expracl->dtype = A_DTYPEG(ast);

  AEXPR *aexpr;
  aexpr = expracl->u1.expr;

  int argt = A_ARGSG(ast);
  int srcast = ARGT_ARG(argt, 0);
  aexpr->rop = construct_acl_from_ast(srcast, A_DTYPEG(srcast), 0);
  if (!aexpr->rop) {
    return 0;
  }

  return expracl;
}

static ACL *
mk_reshape_intrin(int ast)
{
  ACL *expracl;
  int arg_count;
  int argt;
  AEXPR *aexpr;
  int srcast;
  int shapeast;
  int padast = 0;
  int orderast = 0;
  ACL *a;
  int new_sz, old_sz;

  expracl = mk_init_intrinsic(AC_I_reshape);
  aexpr = expracl->u1.expr;

  arg_count = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);

  /* Ignore arg2, the shape was built and plugged in ref_pd */
  shapeast = ARGT_ARG(argt, 1);
  srcast = ARGT_ARG(argt, 0);

  new_sz = get_int_cval(sym_of_ast(ADD_NUMELM(A_DTYPEG(ast))));
  old_sz = get_int_cval(sym_of_ast(ADD_NUMELM(A_DTYPEG(srcast))));
  if (arg_count > 2) {
    padast = ARGT_ARG(argt, 2);
    if (arg_count > 3) {
      orderast = ARGT_ARG(argt, 3);
    }
  }

  /* compute the number of elements in the source */
  if (new_sz > old_sz && !padast) {
    error(4, 3, gbl.lineno,
          "Source and shape argument size mismatch, too few source constants",
          NULL);
    sem.dinit_error = TRUE;
    return 0;
  }

  expracl->dtype = A_DTYPEG(ast);

  aexpr->rop = construct_acl_from_ast(srcast, A_DTYPEG(srcast), 0);
  if (!aexpr->rop) {
    return 0;
  }
  aexpr->rop->next = construct_acl_from_ast(shapeast, A_DTYPEG(shapeast), 0);
  if (!aexpr->rop->next) {
    return 0;
  }

  if (arg_count > 2) {
    if (padast) {
      aexpr->rop->next->next =
          construct_acl_from_ast(padast, A_DTYPEG(padast), 0);
      if (!aexpr->rop->next->next) {
        return 0;
      }
    } else {
      a = GET_ACL(15);
      a->id = AC_AST;
      a->dtype = stb.user.dt_int;
      a->u1.ast = astb.i0;
      aexpr->rop->next->next = a;
    }

    if (arg_count > 3 && orderast) {
      aexpr->rop->next->next->next =
          construct_acl_from_ast(orderast, A_DTYPEG(orderast), 0);
      if (!aexpr->rop->next->next->next) {
        return 0;
      }
    }
  }

  return expracl;
}

static ACL *
mk_shape_intrin(int ast)
{
  ACL *expracl;
  ACL *argacl;
  int argast;
  ACL **r;
  AEXPR *aexpr;
  int rank;
  int shape;
  int argt;
  int ubound[MAXDIMS];
  int lbound[MAXDIMS];
  int i;

  expracl = mk_init_intrinsic(AC_I_shape);
  expracl->dtype = A_DTYPEG(ast);

  argt = A_ARGSG(ast);

  argast = ARGT_ARG(argt, 0);
  shape = A_SHAPEG(argast);
  rank = SHD_NDIM(shape);

  for (i = 0; i < rank; i++) {
    if (A_TYPEG(argast) == A_ID) {
      ubound[i] = ubound_of_shape(shape, i);
      lbound[i] = lbound_of_shape(shape, i);
      if (lbound[i] != astb.i1 || lbound[i] != astb.i0) {
        ubound[i] = extent_of_shape(shape, i);
      }
    } else {
      ubound[i] = extent_of_shape(shape, i);
      lbound[i] = astb.i1;
    }
  }

  aexpr = expracl->u1.expr;

  argacl = aexpr->rop = GET_ACL(15);
  argacl->id = AC_ACONST;
  argacl->dtype = A_DTYPEG(argast);

  r = &argacl->subc;
  for (i = 0; i < rank; i++) {
    *r = GET_ACL(15);
    (*r)->id = AC_AST;
    (*r)->dtype = stb.user.dt_int;
    (*r)->is_const = TRUE;
    (*r)->u1.ast = ubound[i];
    r = &(*r)->next;
  }

  return expracl;
}

static ACL *
mk_size_intrin(int ast)
{
  ACL *expracl;
  ACL **csub_acl;
  ACL *c_acl;
  ACL *arg2acl;
  ACL *dimval;
  int arg1ast;
  int arg2ast;
  DTYPE dtype;
  int shape;
  int rank;
  int i;
  int argt;
  int arg_count;

  /* Build a new arg list that contains:
   *   1) array size (possible astb.i0)
   *   2) array constructor containing the size of each dimension
   *   3) original DIM arg (optional)
   * (athough I'm not sure why, it would be much easier to just
   * plug the size value).
   */

  expracl = mk_init_intrinsic(AC_I_size);
  expracl->dtype = stb.user.dt_int;

  arg_count = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);

  arg1ast = ARGT_ARG(argt, 0);
  shape = A_SHAPEG(arg1ast);
  rank = SHD_NDIM(shape);

  if (arg_count == 1) {
    if (A_TYPEG(arg1ast) == A_ID &&
        (ASUMSZG(A_SPTRG(arg1ast)) || ASSUMSHPG(A_SPTRG(arg1ast)))) {
      error(87, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    }
  } else {
    arg2ast = ARGT_ARG(argt, 1);
    if (!_can_fold(arg2ast)) {
      error(422, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    }
    arg2acl = construct_acl_from_ast(arg2ast, A_DTYPEG(arg2ast), 0);
    if (!arg2acl) {
      return 0;
    }
    dimval = eval_init_expr_item(arg2acl);
    if (!dimval) {
      return 0;
    }
    i = dimval->conval;
    if (i > rank) {
      error(423, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    }
  }

  expracl->u1.expr->rop = c_acl = GET_ACL(15);
  c_acl->id = AC_AST;
  c_acl->dtype = stb.user.dt_int;
  if (A_TYPEG(arg1ast) == A_ID &&
      (ASUMSZG(A_SPTRG(arg1ast)) || ASSUMSHPG(A_SPTRG(arg1ast)))) {
    c_acl->u1.ast = astb.i0;
  } else {
    c_acl->u1.ast = size_of_ast(arg1ast);
  }
  if (c_acl->dtype != A_DTYPEG(c_acl->u1.ast))
    c_acl->u1.ast = mk_convert(c_acl->u1.ast, c_acl->dtype);

  /* shape/dtype for arg 2 */
  sem.arrdim.ndim = 1;
  sem.arrdim.ndefer = 0;
  sem.bounds[0].lowtype = S_CONST;
  sem.bounds[0].lowb = 1;
  sem.bounds[0].lwast = 0;
  sem.bounds[0].uptype = S_CONST;
  sem.bounds[0].upb = rank;
  sem.bounds[0].upast = mk_cval(rank, stb.user.dt_int);
  dtype = mk_arrdsc();
  DTY(dtype + 1) = stb.user.dt_int;

  c_acl->next = GET_ACL(15);
  c_acl = c_acl->next;
  c_acl->id = AC_ACONST;
  c_acl->dtype = dtype;
  csub_acl = &c_acl->subc;
  for (i = 0; i < rank; i++) {
    *csub_acl = c_acl = GET_ACL(15);
    c_acl->id = AC_AST;
    c_acl->dtype = stb.user.dt_int;

    if (_can_fold(SHD_LWB(shape, i)) && _can_fold(SHD_UPB(shape, i))) {
      c_acl->u1.ast = extent_of_shape(shape, i);
    } else if (arg_count == 1 || i == dimval->conval - 1) {
      error(87, 3, gbl.lineno, NULL, NULL);
      sem.dinit_error = TRUE;
      return 0;
    } else {
      c_acl->u1.ast = astb.i0;
    }

    csub_acl = &(*csub_acl)->next;
  }

  if (arg_count == 2) {
    expracl->u1.expr->rop->next->next = arg2acl;
  }

  return expracl;
}

static ACL *
mk_transfer_intrin(int ast)
{
  int argt;
  int argast;
  ACL *expracl;
  ACL *arglist;

  expracl = mk_init_intrinsic(AC_I_transfer);

  argt = A_ARGSG(ast);
  argast = ARGT_ARG(argt, 0);
  arglist = construct_acl_from_ast(argast, A_DTYPEG(argast), 0);
  if (arglist == 0) {
    sem.dinit_error = TRUE;
    return 0;
  }

#ifdef try_without_this
  /* Maybe we don't need the 2nd and 3rd args.
     A_DTYPEG(ast) gives the type of the result.
  */
  /* Can't call construct_acl_from_ast() for the mold argument because
   * it need not be a constant.  All we really need is the element type.
   */
  argast = ARGT_ARG(argt, 1);
  aclp = GET_ACL(15);
  aclp->id = AC_AST;
  aclp->dtype = DDTG(A_DTYPEG(argast));
  aclp->u1.ast = mk_cval(0, aclp->dtype);
  arglist->next = aclp;

  /* size of result */
  argast = ARGT_ARG(argt, 2);
  aclp = construct_acl_from_ast(argast, A_DTYPEG(argast), 0);
  if (aclp == 0) {
    sem.dinit_error = TRUE;
    return 0;
  }
  arglist->next->next = aclp;
#endif

  expracl->dtype = A_DTYPEG(ast);
  expracl->u1.expr->rop = arglist;
  return expracl;
}

static ACL *
construct_arg_list(int ast)
{
  int argt = A_ARGSG(ast);
  ACL *argroot = NULL;
  ACL **curarg = &argroot;
  int i;

  for (i = 0; i < A_ARGCNTG(ast); i++) {
    int argast = ARGT_ARG(argt, i);
    /* argast is 0 for optional args */
    if (argast) {
      *curarg = construct_acl_from_ast(argast, A_DTYPEG(argast), 0);
      if (!*curarg) {
        return 0;
      }
      curarg = &(*curarg)->next;
    }
  }
  return argroot;
}

static ACL *
mk_nonelem_init_intrinsic(AC_INTRINSIC init_intr, int ast, DTYPE dtype)
{
  ACL *expracl = mk_init_intrinsic(init_intr);
  ACL *arglist = construct_arg_list(ast);

  if (sem.dinit_error) {
    return 0;
  }
  expracl->dtype = dtype;
  expracl->u1.expr->rop = arglist;
  return expracl;
}

static ACL *
mk_elem_init_intrinsic(AC_INTRINSIC init_intr, int ast, DTYPE dtype,
                       int parent_acltype)
{
  ACL *arg1acl;
  ACL *a;
  DTYPE arg1dtype;
  DTYPE dtypebase = DDTG(dtype);
  ACL *expracl = mk_init_intrinsic(init_intr);
  ACL *arglist = construct_arg_list(ast);

  if (!arglist) {
    sem.dinit_error = TRUE;
    return 0;
  }

  arg1acl = arglist;
  arg1dtype = arg1acl->dtype;
  expracl->dtype = dtypebase;
  expracl->u1.expr->rop = arglist;

  if (DTY(dtype) == TY_ARRAY) {
    if (DTY(arg1dtype) != TY_ARRAY && parent_acltype != AC_ACONST)
      expracl->repeatc = ADD_NUMELM(dtype);
    a = GET_ACL(15);
    a->id = AC_ACONST;
    a->dtype = dtype;
    a->subc = expracl;
    expracl = a;
  }
  return expracl;
}

static AC_INTRINSIC
get_ac_intrinsic(int ast)
{
  SPTR sptr = A_SPTRG(A_LOPG(ast));
  switch (STYPEG(sptr)) {
  case ST_PD:
    return map_PD_to_AC(PDNUMG(sptr));
  case ST_INTRIN:
  case ST_GENERIC:
    return map_I_to_AC(INTASTG(sptr));
  case ST_PROC:
    if (A_TYPEG(ast) == A_INTR) {
      return map_I_to_AC(A_OPTYPEG(ast));
    } else {
      return AC_I_NONE;
    }
  default:
    return AC_I_NONE;
  }
}

/* Map I_* to AC_I_* constants. */
static AC_INTRINSIC
map_I_to_AC(int intrin)
{
  switch (intrin) {
  case I_ICHAR:
    return AC_I_ichar;
  case I_IISHFT:
  case I_JISHFT:
  case I_KISHFT:
    return AC_I_ishft;
  case I_LSHIFT:
    return AC_I_lshift;
  case I_RSHIFT:
    return AC_I_rshift;
  case I_IMIN0:
  case I_MIN0:
  case I_AMIN1:
  case I_DMIN1:
  case I_KMIN0:
  case I_JMIN0:
  case I_AMIN0:
  case I_AIMIN0:
  case I_MIN1:
  case I_IMIN1:
  case I_JMIN1:
  case I_KMIN1:
  case I_AJMIN0:
  case I_MIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMIN:
#endif
    return AC_I_min;
  case I_IMAX0:
  case I_MAX0:
  case I_AMAX1:
  case I_DMAX1:
  case I_KMAX0:
  case I_JMAX0:
  case I_AMAX0:
  case I_AIMAX0:
  case I_MAX1:
  case I_IMAX1:
  case I_JMAX1:
  case I_KMAX1:
  case I_AJMAX0:
  case I_MAX:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMAX:
#endif
    return AC_I_max;
  case I_ABS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QABS:
#endif
    return AC_I_abs;
  case I_DBLE:
  case I_DFLOAT:
  case I_FLOAT:
  case I_REAL:
    return AC_I_fltconvert;
  case I_MOD:
  case I_AMOD:
  case I_DMOD:
    return AC_I_mod;
  case I_SQRT:
  case I_DSQRT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSQRT:
#endif
    return AC_I_sqrt;
  case I_EXP:
  case I_DEXP:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QEXP:
#endif
    return AC_I_exp;
  case I_LOG:
  case I_ALOG:
  case I_DLOG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG:
#endif
    return AC_I_log;
  case I_LOG10:
  case I_ALOG10:
  case I_DLOG10:
    return AC_I_log10;
  case I_SIN:
  case I_DSIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIN:
#endif
    return AC_I_sin;
  case I_COS:
  case I_DCOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOS:
#endif
    return AC_I_cos;
  case I_TAN:
  case I_DTAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTAN:
#endif
    return AC_I_tan;
  case I_ASIN:
  case I_DASIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QASIN:
#endif
    return AC_I_asin;
  case I_ACOS:
  case I_DACOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOS:
#endif
    return AC_I_acos;
  case I_ATAN:
  case I_DATAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN:
#endif
    return AC_I_atan;
  case I_ATAN2:
  case I_DATAN2:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2:
#endif
    return AC_I_atan2;
  case I_IAND:
    return AC_I_iand;
  case I_IOR:
    return AC_I_ior;
  case I_IEOR:
    return AC_I_ieor;
  case I_MERGE:
    return AC_I_merge;
  case I_SCALE:
    return AC_I_scale;
  case I_MAXLOC:
    return AC_I_maxloc;
  case I_MAXVAL:
    return AC_I_maxval;
  case I_MINLOC:
    return AC_I_minloc;
  case I_MINVAL:
    return AC_I_minval;
  default:
    return AC_I_NONE;
  }
}

/* Map PD_* to AC_I_* constants. */
static AC_INTRINSIC
map_PD_to_AC(int pdnum)
{
  switch (pdnum) {
  case PD_lbound:
    return AC_I_lbound;
  case PD_ubound:
    return AC_I_ubound;
  case PD_reshape:
    return AC_I_reshape;
  case PD_size:
    return AC_I_size;
  case PD_selected_int_kind:
    return AC_I_selected_int_kind;
  case PD_selected_real_kind:
#ifdef PD_ieee_selected_real_kind
  case PD_ieee_selected_real_kind:
#endif
    return AC_I_selected_real_kind;
  case PD_selected_char_kind:
    return AC_I_selected_char_kind;
  case PD_adjustl:
    return AC_I_adjustl;
  case PD_adjustr:
    return AC_I_adjustr;
  case PD_achar:
    return AC_I_char;
  case PD_iachar:
    return AC_I_ichar;
  case PD_int:
    return AC_I_int;
  case PD_nint:
    return AC_I_nint;
  case PD_char:
    return AC_I_char;
  case PD_index:
    return AC_I_index;
  case PD_repeat:
    return AC_I_repeat;
  case PD_len_trim:
    return AC_I_len_trim;
  case PD_trim:
    return AC_I_trim;
  case PD_scan:
    return AC_I_scan;
  case PD_verify:
    return AC_I_verify;
  case PD_null:
    return AC_I_null;
  case PD_shape:
    return AC_I_shape;
  case PD_real:
    return AC_I_fltconvert;
  case PD_floor:
    return AC_I_floor;
  case PD_ceiling:
    return AC_I_ceiling;
  case PD_transfer:
    return AC_I_transfer;
  case PD_transpose:
    return AC_I_transpose;
  case PD_scale:
    return AC_I_scale;
  case PD_maxloc:
    return AC_I_maxloc;
  case PD_maxval:
    return AC_I_maxval;
  case PD_minloc:
    return AC_I_minloc;
  case PD_minval:
    return AC_I_minval;
  default:
    return AC_I_NONE;
  }
}

static ACL *
construct_intrinsic_acl(int ast, DTYPE dtype, int parent_acltype)
{
  AC_INTRINSIC intrin = get_ac_intrinsic(ast);
  switch (intrin) {
  case AC_I_char:
  case AC_I_adjustl:
  case AC_I_adjustr:
  case AC_I_ichar:
  case AC_I_index:
  case AC_I_int:
  case AC_I_ishft:
  case AC_I_max:
  case AC_I_min:
  case AC_I_nint:
  case AC_I_len_trim:
  case AC_I_ishftc:
  case AC_I_fltconvert:
  case AC_I_scan:
  case AC_I_verify:
  case AC_I_floor:
  case AC_I_ceiling:
  case AC_I_mod:
  case AC_I_sqrt:
  case AC_I_exp:
  case AC_I_log:
  case AC_I_log10:
  case AC_I_sin:
  case AC_I_cos:
  case AC_I_tan:
  case AC_I_asin:
  case AC_I_acos:
  case AC_I_atan:
  case AC_I_atan2:
  case AC_I_abs:
  case AC_I_iand:
  case AC_I_ior:
  case AC_I_ieor:
  case AC_I_merge:
  case AC_I_scale:
    return mk_elem_init_intrinsic(intrin, ast, dtype, parent_acltype);
  case AC_I_maxloc:
  case AC_I_maxval:
  case AC_I_minloc:
  case AC_I_minval:
    return mk_elem_init_intrinsic(intrin, ast, dtype, parent_acltype);
  case AC_I_lshift:
    /* LSHIFT(i, shift) == ISHFT(i, shift) */
    return mk_elem_init_intrinsic(AC_I_ishft, ast, dtype, parent_acltype);
  case AC_I_rshift: {
    /* RSHIFT(i, shift) == ISHFT(-i, shift) */
    int argt = A_ARGSG(ast);
    int val = ARGT_ARG(argt, 0);
    int shift = ARGT_ARG(argt, 1);
    int new_shift = mk_unop(OP_SUB, shift, A_DTYPEG(shift));
    int new_ast = ast_intr(I_ISHFT, A_DTYPEG(ast), 2, val, new_shift);
    return mk_elem_init_intrinsic(AC_I_ishft, new_ast, dtype, parent_acltype);
  }
  case AC_I_len:
  case AC_I_lbound:
  case AC_I_ubound:
    return mk_ulbound_intrin(intrin, ast);
  case AC_I_null:
  case AC_I_repeat:
  case AC_I_trim:
  case AC_I_selected_int_kind:
  case AC_I_selected_real_kind:
  case AC_I_selected_char_kind:
    return mk_nonelem_init_intrinsic(intrin, ast, A_DTYPEG(ast));
  case AC_I_size:
    return mk_size_intrin(ast);
  case AC_I_transpose:
    return mk_transpose_intrin(ast);
  case AC_I_reshape:
    return mk_reshape_intrin(ast);
  case AC_I_shape:
    return mk_shape_intrin(ast);
  case AC_I_transfer:
    return mk_transfer_intrin(ast);
  default:
    error(155, ERR_Severe, gbl.lineno,
          "Intrinsic not supported in initialization:",
          SYMNAME(A_SPTRG(A_LOPG(ast))));
    sem.dinit_error = TRUE;
    return 0;
  }
}

static int
get_ast_op(int op)
{
  int ast_op = 0;

  switch (op) {
  case AC_NEG:
    ast_op = OP_NEG;
    break;
  case AC_ADD:
    ast_op = OP_ADD;
    break;
  case AC_SUB:
    ast_op = OP_SUB;
    break;
  case AC_MUL:
    ast_op = OP_MUL;
    break;
  case AC_DIV:
    ast_op = OP_DIV;
    break;
  case AC_CAT:
    ast_op = OP_CAT;
    break;
  case AC_LEQV:
    ast_op = OP_LEQV;
    break;
  case AC_LNEQV:
    ast_op = OP_LNEQV;
    break;
  case AC_LOR:
    ast_op = OP_LOR;
    break;
  case AC_LAND:
    ast_op = OP_LAND;
    break;
  case AC_EQ:
    ast_op = OP_EQ;
    break;
  case AC_GE:
    ast_op = OP_GE;
    break;
  case AC_GT:
    ast_op = OP_GT;
    break;
  case AC_LE:
    ast_op = OP_LE;
    break;
  case AC_LT:
    ast_op = OP_LT;
    break;
  case AC_NE:
    ast_op = OP_NE;
    break;
  case AC_LNOT:
    ast_op = OP_LNOT;
    break;
  case AC_EXP:
  case AC_EXPK:
    ast_op = OP_XTOI;
    break;
  case AC_EXPX:
    ast_op = OP_XTOX;
    break;
  default:
    interr("get_ast_op: unexpected operator in initialization expr", op, 3);
  }
  return ast_op;
}

static int
get_ac_op(int ast)
{
  int ac_op = 0;

  switch (A_OPTYPEG(ast)) {
  case OP_NEG:
    ac_op = AC_NEG;
    break;
  case OP_ADD:
    ac_op = AC_ADD;
    break;
  case OP_SUB:
    ac_op = AC_SUB;
    break;
  case OP_MUL:
    ac_op = AC_MUL;
    break;
  case OP_DIV:
    ac_op = AC_DIV;
    break;
  case OP_CAT:
    ac_op = AC_CAT;
    break;
  case OP_LEQV:
    ac_op = AC_LEQV;
    break;
  case OP_LNEQV:
    ac_op = AC_LNEQV;
    break;
  case OP_LOR:
    ac_op = AC_LOR;
    break;
  case OP_LAND:
    ac_op = AC_LAND;
    break;
  case OP_EQ:
    ac_op = AC_EQ;
    break;
  case OP_GE:
    ac_op = AC_GE;
    break;
  case OP_GT:
    ac_op = AC_GT;
    break;
  case OP_LE:
    ac_op = AC_LE;
    break;
  case OP_LT:
    ac_op = AC_LT;
    break;
  case OP_NE:
    ac_op = AC_NE;
    break;
  case OP_LNOT:
    ac_op = AC_LNOT;
    break;
  case OP_XTOI:
    switch (DDTG(A_DTYPEG(A_ROPG(ast)))) {
    case DT_INT8:
      ac_op = AC_EXPK;
      break;
    case DT_REAL4:
    case DT_REAL8:
#ifdef TARGET_SUPPORTS_QUADFP
    case DT_QUAD:
    case DT_CMPLX8:
    case DT_CMPLX16:
    case DT_QCMPLX:
#endif
      ac_op = AC_EXPX;
      break;
    default:
      ac_op = AC_EXP;
      break;
    }
    break;
  default:
    interr("get_ac_op: unexpected operator in initialization expr",
           A_OPTYPEG(ast), 3);
  }
  return ac_op;
}

static ACL *
eval_do_idx(int ast)
{
  ACL *aclp = NULL;
  DOSTACK *p;
  int sptr = A_SPTRG(ast);

  if (!sptr)
    return aclp;

  for (p = sem.dostack; p < sem.top; p++) {
    if (p->sptr == sptr) {
      aclp = GET_ACL(15);
      aclp->id = AC_CONST;
      aclp->dtype = A_DTYPEG(ast);
      aclp->is_const = 1;
      aclp->u1.ast = ast;

      if (DT_ISWORD(A_DTYPEG(ast)))
        aclp->u1.ast = mk_cval1(p->currval, A_DTYPEG(ast));
      else
        aclp->u1.ast = mk_cnst(p->currval);
      return aclp;
    }
  }
  return aclp;
}

ACL *
construct_acl_from_ast(int ast, DTYPE dtype, int parent_acltype)
{
  ACL *aclp = NULL, *subscr_aclp;
  ACL *u, *l, *s;
  ACL *prev;
  int lParent_acltype;
  int sptr;
  int asd;
  int sub_ast;
  int ndim;
  int i;
  int m_sptr;
  int p_dtype;

  if (!ast) {
    errsev(457);
    sem.dinit_error = TRUE;
    return 0;
  }
  if (!_can_fold(ast) &&
      (A_TYPEG(ast) == A_ID && !DOVARG(A_SPTRG(ast)) &&
       !(STYPEG(A_SPTRG(ast)) == ST_MEMBER) &&
       !(STYPEG(A_SPTRG(ast)) == ST_PARAM || PARAMG(A_SPTRG(ast)))) &&
      !(HCCSYMG(A_SPTRG(ast)) && DINITG(A_SPTRG(ast)))) {
    ACL *acl = eval_do_idx(ast);
    if (acl)
      return acl;
    errsev(87);
    sem.dinit_error = TRUE;
    return 0;
  }

  switch (A_TYPEG(ast)) {
  case A_FUNC:
    if (sem.equal_initializer) {
      errsev(87);
      sem.dinit_error = TRUE;
      return 0;
    }
    aclp = GET_ACL(15);
    aclp->id = AC_IDENT;
    aclp->dtype = A_DTYPEG(ast);
    aclp->is_const = 1;
    aclp->u1.ast = A_LOPG(ast);
    break;
  case A_ID:
    aclp = GET_ACL(15);
    aclp->id = AC_AST;
    aclp->dtype = A_DTYPEG(ast);
    aclp->is_const = 1;
    aclp->u1.ast = ast;

    if (DTY(DDTG(dtype)) == TY_DERIVED &&
        (parent_acltype != AC_SCONST || DDTG(A_DTYPEG(ast)) != DDTG(dtype)) &&
        !(DTY(dtype) == TY_ARRAY && DTY(A_DTYPEG(ast)) == TY_ARRAY)) {
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_SCONST;
      aclp->dtype = DDTG(A_DTYPEG(ast));
      aclp->is_const = 1;
      aclp->subc = prev;
    }
    if (DTY(dtype) == TY_ARRAY && DTY(A_DTYPEG(ast)) != TY_ARRAY &&
        parent_acltype != AC_ACONST) {
      aclp->repeatc = ADD_NUMELM(dtype);
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_ACONST;
      aclp->dtype = dtype;
      aclp->is_const = 1;
      aclp->subc = prev;
    }
    break;
  case A_CNST:
    aclp = GET_ACL(15);
    aclp->id = AC_AST;
    aclp->dtype = A_DTYPEG(ast);
    aclp->is_const = 1;
    aclp->u1.ast = ast;
    if (DTY(dtype) == TY_ARRAY && DTY(A_DTYPEG(ast)) != TY_ARRAY &&
        parent_acltype != AC_ACONST) {
      aclp->repeatc = ADD_NUMELM(dtype);
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_ACONST;
      aclp->dtype = dtype;
      aclp->is_const = 1;
      aclp->subc = prev;
    }
    break;
  case A_BINOP:
    aclp = GET_ACL(15);
    aclp->id = AC_IEXPR;
    aclp->dtype = A_DTYPEG(ast);
    aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
    aclp->u1.expr->op = get_ac_op(ast);
    /* this ACL may become the child of an AC_ACONST; set the last argument of
     * call to construct_acl_from_ast appropriately
     */
    lParent_acltype =
        (DTY(dtype) == TY_ARRAY && parent_acltype != AC_ACONST) ? AC_ACONST : 0;
    aclp->u1.expr->lop = construct_acl_from_ast(
        A_LOPG(ast), A_DTYPEG(A_LOPG(ast)), lParent_acltype);
    aclp->u1.expr->rop = construct_acl_from_ast(
        A_ROPG(ast), A_DTYPEG(A_ROPG(ast)), lParent_acltype);

    if (!aclp->u1.expr->lop || !aclp->u1.expr->rop) {
      return 0;
    }
    if (DTY(dtype) == TY_ARRAY && parent_acltype != AC_ACONST) {
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_ACONST;
      aclp->dtype = dtype;
      aclp->is_const = 1;
      aclp->subc = prev;
    }
    break;
  case A_UNOP:
    aclp = GET_ACL(15);
    aclp->id = AC_IEXPR;
    aclp->dtype = A_DTYPEG(ast);
    aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
    aclp->u1.expr->op = AC_NEG;
    if (get_ac_op(ast) == AC_LNOT)
      aclp->u1.expr->op = AC_LNOT;
    aclp->u1.expr->lop = construct_acl_from_ast(A_LOPG(ast), A_DTYPEG(ast), 0);
    if (!aclp->u1.expr->lop) {
      return 0;
    }
    aclp->u1.expr->rop = NULL;
    if (DTY(dtype) == TY_ARRAY && parent_acltype != AC_ACONST) {
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_ACONST;
      aclp->dtype = dtype;
      aclp->is_const = 1;
      aclp->subc = prev;
    }
    break;
  case A_CONV:
    if (DDTG(A_DTYPEG(ast)) == DDTG(A_DTYPEG(A_LOPG(ast)))) {
      aclp = construct_acl_from_ast(A_LOPG(ast), 0, 0);
      if (!aclp) {
        return 0;
      }
    } else {
      aclp = GET_ACL(15);
      aclp->id = AC_IEXPR;
      aclp->dtype = A_DTYPEG(ast);
      aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
      aclp->u1.expr->op = AC_CONV;
      aclp->u1.expr->lop =
          construct_acl_from_ast(A_LOPG(ast), DDTG(A_DTYPEG(ast)), 0);
      if (!aclp->u1.expr->lop) {
        return 0;
      }
      aclp->u1.expr->rop = NULL;
      if (DTY(dtype) == TY_ARRAY && parent_acltype != AC_ACONST) {
        prev = aclp;
        aclp = GET_ACL(15);
        aclp->id = AC_ACONST;
        aclp->dtype = dtype;
        aclp->is_const = 1;
        aclp->subc = prev;
      }
    }
    break;
  case A_SUBSCR:
    aclp = GET_ACL(15);
    aclp->id = AC_IEXPR;
    aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
    aclp->u1.expr->op = AC_ARRAYREF;
    aclp->u1.expr->lop = construct_acl_from_ast(A_LOPG(ast), 0, 0);
    if (!aclp->u1.expr->lop) {
      return 0;
    }
    aclp->dtype = A_DTYPEG(ast);
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    prev = NULL;
    for (i = 0; i < ndim; i++) {
      sub_ast = ASD_SUBS(asd, i);
      subscr_aclp = GET_ACL(15);
      subscr_aclp->id = AC_IEXPR;
      subscr_aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
      subscr_aclp->u1.expr->op = AC_TRIPLE;
      subscr_aclp->dtype = A_DTYPEG(sub_ast);
      subscr_aclp->u1.expr->lop = NULL;
      if (prev == NULL) {
        aclp->u1.expr->rop = subscr_aclp;
      } else {
        prev->next = subscr_aclp;
      }
      prev = subscr_aclp;

      l = GET_ACL(15);
      l->id = AC_AST;
      l->dtype = astb.bnd.dtype;
      l->is_const = 1;

      u = GET_ACL(15);
      u->id = AC_AST;
      u->dtype = astb.bnd.dtype;
      u->is_const = 1;

      s = GET_ACL(15);
      s->id = AC_AST;
      s->dtype = astb.bnd.dtype;
      s->is_const = 1;

    again:
      switch (A_TYPEG(sub_ast)) {
      case A_TRIPLE:
        l->u1.ast = A_LBDG(sub_ast);
        l->dtype = A_DTYPEG(A_LBDG(sub_ast));
        u->u1.ast = A_UPBDG(sub_ast);
        u->dtype = A_DTYPEG(A_UPBDG(sub_ast));
        if (A_STRIDEG(sub_ast) == 0) {
          s->u1.ast = astb.bnd.one;
          u->dtype = A_DTYPEG(astb.bnd.one);
        } else {
          s->u1.ast = A_STRIDEG(sub_ast);
          u->dtype = A_DTYPEG(A_STRIDEG(sub_ast));
        }
        break;
      case A_SUBSCR:
        /* This needs updated for sub_ast that is an array section
         * of multi-dimension array with rank one.
         */
        ast = sub_ast;
        asd = A_ASDG(ast);
        sub_ast = ASD_SUBS(asd, 0);
        subscr_aclp->u1.expr->lop = construct_acl_from_ast(A_LOPG(ast), 0, 0);
        goto again;
        break;
      case A_CONV:
        ast = sub_ast;
        sub_ast = A_LOPG(sub_ast);
        goto again;
        break;
      case A_ID:
        if (DTY(A_DTYPEG(sub_ast)) == TY_ARRAY) {
          int shape;
          shape = A_SHAPEG(sub_ast);
          if (SHD_LWB(shape, 0)) {
            l->u1.ast = SHD_LWB(shape, 0);
            l->dtype = A_DTYPEG(SHD_LWB(shape, 0));
          } else {
            l->u1.ast = astb.bnd.one;
            l->dtype = A_DTYPEG(astb.bnd.one);
          }
          u->u1.ast = SHD_UPB(shape, 0);
          u->dtype = A_DTYPEG(SHD_UPB(shape, 0));
          s->u1.ast = astb.bnd.one;
          s->dtype = A_DTYPEG(astb.bnd.one);
          subscr_aclp->u1.expr->lop = construct_acl_from_ast(sub_ast, 0, 0);
          break;
        }
        FLANG_FALLTHROUGH;
      default:
        l->u1.ast = sub_ast;
        l->dtype = A_DTYPEG(sub_ast);
        u->u1.ast = sub_ast;
        u->dtype = A_DTYPEG(sub_ast);
        s->u1.ast = astb.bnd.one;
        s->dtype = A_DTYPEG(astb.bnd.one);
        break;
      }

      l->next = u;
      u->next = s;
      s->next = NULL;
      subscr_aclp->u1.expr->rop = l;
    }
    break;
  case A_MEM:
    aclp = GET_ACL(15);
    aclp->id = AC_IEXPR;
    aclp->dtype = A_DTYPEG(ast);
    aclp->u1.expr = (AEXPR *)getitem(15, sizeof(AEXPR));
    aclp->u1.expr->op = AC_MEMBR_SEL;
    aclp->u1.expr->lop = construct_acl_from_ast(A_PARENTG(ast), 0, 0);
    if (!aclp->u1.expr->lop) {
      return 0;
    }

    /* find the field number */
    p_dtype = A_DTYPEG(A_PARENTG(ast));
    m_sptr = A_SPTRG(A_MEMG(ast));
    for (sptr = DTY(p_dtype + 1), i = 0; sptr > NOSYM && sptr != m_sptr;
         sptr = SYMLKG(sptr), i++)
      ;
    l = GET_ACL(15);
    l->id = AC_AST;
    l->dtype = DT_INT4;
    l->u1.ast = mk_cval(i, l->dtype);

    aclp->u1.expr->rop = l;
    break;
  case A_INTR:
    aclp = construct_intrinsic_acl(ast, dtype, parent_acltype);
    if (aclp && DTY(dtype) == TY_ARRAY && DTY(A_DTYPEG(ast)) != TY_ARRAY &&
        parent_acltype != AC_ACONST &&
        !(STYPEG(A_SPTRG(A_LOPG(ast))) == ST_PD &&
          PDNUMG(A_SPTRG(A_LOPG(ast))) == PD_null)) {
      if (aclp->dtype == dtype) {
        if (aclp->subc && aclp->subc->repeatc == ADD_NUMELM(dtype))
          break;
      }
      aclp->repeatc = ADD_NUMELM(dtype);
      prev = aclp;
      aclp = GET_ACL(15);
      aclp->id = AC_ACONST;
      aclp->dtype = dtype;
      aclp->is_const = 1;
      aclp->subc = prev;
    }

    break;
  default:
    interr("unexpected ast type in initialization expr", ast, 3);
  }

  return aclp;
}

static int
next_member(int member)
{
  int new_mbr = SYMLKG(member);

  if (POINTERG(member) || ALLOCATTRG(member))
    while (new_mbr != NOSYM && HCCSYMG(new_mbr))
      new_mbr = SYMLKG(new_mbr);

  return new_mbr == NOSYM ? 0 : new_mbr;
}

ACL *
rewrite_acl(ACL *aclp, DTYPE dtype, int parent_acltype)
{
  SST *stkp;
  int ast;
  int sptr;
  int mbr_sptr;
  int wrk_dtype = dtype;
  DOINFO *doinfo;
  ACL *cur_aclp;
  ACL *wrk_aclp;
  ACL *prev_aclp = NULL;
  ACL *ret_aclp = aclp;
  ACL *sav_aclp = NULL;
  if (no_data_components(dtype) && !is_zero_size_typedef(dtype)) {
    return 0;
  }
  if (parent_acltype == AC_SCONST) {
    mbr_sptr = DTY(DDTG(dtype) + 1);
    wrk_dtype = DTYPEG(mbr_sptr);
  }

  for (cur_aclp = aclp; cur_aclp != NULL; cur_aclp = cur_aclp->next) {
    wrk_aclp = cur_aclp;
    switch (cur_aclp->id) {
    case AC_EXPR:
      stkp = cur_aclp->u1.stkp;
    again:
      ast = SST_ASTG(stkp);
      if (SST_IDG(stkp) == S_ACONST) {
        /* attempt to avoid ICE by calling mkexpr() on
         * S_ACONST
         */
        mkexpr(stkp);
        if (SST_IDG(stkp) != S_ACONST)
          goto again;
        interr("rewrite_acl: unexpected S_ACONST", 0, 3);
        wrk_aclp->subc = SST_ACLG(stkp);
        wrk_aclp->id = AC_ACONST;
        wrk_aclp->repeatc = 0;
      } else if (SST_IDG(stkp) == S_IDENT) {
        sptr = SST_SYMG(stkp);
        if (STYPEG(sptr) == ST_PARAM || PARAMG(sptr)) {
          ast = mk_id(sptr);
          wrk_aclp = construct_acl_from_ast(ast, wrk_dtype, parent_acltype);
          wrk_aclp->u1.ast = ast;
        }
        /* MORE is this necessary */
        else if (STYPEG(sptr) == ST_PD || STYPEG(sptr) == ST_INTRIN) {
          wrk_aclp = SST_ACLG(stkp);
        } else {
          errsev(87);
          sem.dinit_error = TRUE;
          continue;
        }
      } else if (SST_IDG(stkp) == S_CONST) {
        wrk_aclp =
            construct_acl_from_ast(SST_ASTG(stkp), wrk_dtype, parent_acltype);
      } else if (SST_IDG(stkp) == S_EXPR &&
                 (A_TYPEG(ast) == A_ID || A_TYPEG(ast) == A_CNST)) {
        wrk_aclp =
            construct_acl_from_ast(SST_ASTG(stkp), wrk_dtype, parent_acltype);
      } else
        wrk_aclp = construct_acl_from_ast(ast, wrk_dtype, parent_acltype);
      break;
    case AC_IDO:
      /* must make a copy of DOINFO because we don't know where
       * the current one was allocated or when it will be freed.
       */
      doinfo = get_doinfo(15);
      *doinfo = *cur_aclp->u1.doinfo;
      wrk_aclp->u1.doinfo = doinfo;

      DOVARP(cur_aclp->u1.doinfo->index_var, 1);
      wrk_aclp->subc = rewrite_acl(cur_aclp->subc, DDTG(dtype), 0);
      if (!wrk_aclp->subc) {
        return 0;
      }
      DOVARP(cur_aclp->u1.doinfo->index_var, 0);
      wrk_aclp->repeatc = 0;

      break;
    case AC_SCONST:
    case AC_TYPEINIT:
      wrk_aclp->subc =
          rewrite_acl(cur_aclp->subc, cur_aclp->dtype, cur_aclp->id);
      if (!wrk_aclp->subc && !is_zero_size_typedef(wrk_dtype)) {
        return 0;
      }
      if (DTY(wrk_dtype) == TY_ARRAY && parent_acltype != AC_ACONST) {
        wrk_aclp->repeatc = ADD_NUMELM(wrk_dtype);
        sav_aclp = wrk_aclp;
        wrk_aclp = GET_ACL(15);
        wrk_aclp->id = AC_ACONST;
        wrk_aclp->dtype = wrk_dtype;
        wrk_aclp->is_const = 1;
        wrk_aclp->subc = sav_aclp;
      }
      break;
    case AC_ACONST:
      wrk_aclp->subc =
          rewrite_acl(cur_aclp->subc, cur_aclp->dtype, cur_aclp->id);
      if (!wrk_aclp->subc) {
        break;
      }
      wrk_aclp->repeatc = aclp->repeatc;
      break;
    case AC_AST:
      wrk_aclp = construct_acl_from_ast(cur_aclp->u1.ast, cur_aclp->dtype,
                                        parent_acltype);
      if (wrk_aclp) {
        wrk_aclp->repeatc = cur_aclp->repeatc;
        wrk_aclp->sptr = cur_aclp->sptr;
      }
      break;
    case AC_IEXPR:
      wrk_aclp = cur_aclp;
      break;
    case AC_REPEAT:
    default:
      interr("unexpected acl expresion type", cur_aclp->id, 3);
      break;
    }

    if (wrk_aclp) {
      if (prev_aclp) {
        prev_aclp->next = wrk_aclp;
      } else {
        ret_aclp = wrk_aclp;
      }
      prev_aclp = wrk_aclp;
    }

    if (parent_acltype == AC_SCONST) {
      mbr_sptr = next_member(mbr_sptr);
      wrk_dtype = DTYPEG(mbr_sptr);
    }
  }

  if (sem.dinit_error) {
    ret_aclp = 0;
  }

  return ret_aclp;
}

static int
init_types_compatable(SST *istkp, DTYPE dtype, int sptr)
{

  if (STYPEG(sptr) == ST_PD && PDNUMG(sptr) == PD_null &&
      SST_DTYPEG(istkp) == DT_WORD) {
    return TRUE;
  }

  if ((DTY(dtype) != TY_ARRAY && DTY(dtype) != DTY(SST_DTYPEG(istkp))) ||
      (DTY(dtype) == TY_ARRAY && DTY(SST_DTYPEG(istkp)) == TY_ARRAY &&
       !cmpat_dtype_with_size(dtype, SST_DTYPEG(istkp))) ||
      (DTY(dtype) == TY_ARRAY && DTY(SST_DTYPEG(istkp)) != TY_ARRAY &&
       DDTG(dtype) != SST_DTYPEG(istkp))) {
    return FALSE;
  }
  return TRUE;
}

void
construct_acl_for_sst(SST *istkp, DTYPE dtype)
{
  ACL *aclp = 0;
  int sptr = 0;

  switch (SST_IDG(istkp)) {
  case S_IDENT:
    /* the ident must be a named constant or an alias for a named constant */
    aclp = SST_ACLG(istkp);
    if (aclp) {
      sptr = A_SPTRG(aclp->u1.ast);
    } else {
      sptr = SST_SYMG(istkp);
    }
    if ((!sptr || !(STYPEG(sptr) == ST_PARAM || PARAMG(sptr))) &&
        (!has_type_parameter(dtype) || !sem.param_struct_constr)) {
      if (!no_data_components(dtype)) {
        errsev(87);
      }
      sem.dinit_error = TRUE;
      SST_ACLP(istkp, 0);
      return;
    }
    /* the types must be compatable */
    if (!init_types_compatable(istkp, dtype, sptr)) {
      errsev(91);
      sem.dinit_error = TRUE;
      SST_ACLP(istkp, 0);
      return;
    }
    if (!aclp) {
      /* PARAMETER defined in a module, already processed */
      SST_ACLP(istkp, (ACL *)get_getitem_p(CONVAL2G(NMCNSTG(sptr))));
    } else if (DTY(DDTG(dtype)) == TY_DERIVED) {
      SST_ACLP(istkp, construct_acl_from_ast(aclp->u1.ast, dtype, 0));
    }
    if (DTY(dtype) == TY_ARRAY && (aclp = SST_ACLG(istkp)) &&
        DTY(aclp->dtype) != TY_ARRAY && aclp->id == AC_IEXPR &&
        aclp->u1.expr->op == AC_INTR_CALL) {
      aclp->repeatc = ADD_NUMELM(dtype);
    }
    break;
  case S_EXPR:
  case S_CONST:
  case S_LVALUE:
    SST_ACLP(istkp, construct_acl_from_ast(SST_ASTG(istkp), dtype, 0));
    break;
  case S_ACONST:
    SST_ACLP(istkp, rewrite_acl(SST_ACLG(istkp), dtype, 0));
    break;
  case S_SCONST:
    if (DDTG(dtype) != SST_DTYPEG(istkp)) {
      if (DTY(DDTG(dtype)) == TY_DERIVED &&
          DTY(SST_DTYPEG(istkp)) == TY_DERIVED) {

        /* For parameterized derived types, the following from F2008 spec
         * applies (there's similar language in F2003 spec):
         * Section 5.2.3 ...
         * If initialization is = constant-expr, the variable is initially
         * defined with the value specified by the constant-expr; if
         * necessary, the value is converted according to the rules of
         * intrinsic assignment (7.2.1.3) to a value that agrees in type,
         * type parameters, and shape with the variable.
         *
         * Therefore, if the type on the LHS is a parameterized derived
         * type, check its "base type" with the type on the RHS. If they
         * are identical, then we have a legal initialization since the
         * value is to be "converted".
         */

        int tag1, dty1, dty2;
        tag1 = DTY(DDTG(dtype) + 3);
        dty1 = (BASETYPEG(tag1)) ? BASETYPEG(tag1) : DDTG(dtype);
        dty2 = SST_DTYPEG(istkp);
        if (dty1 == dty2)
          goto sconst_ok;
      }
      errsev(91);
      sem.dinit_error = TRUE;
      SST_ACLP(istkp, 0);
      return;
    }
  sconst_ok:
    SST_ACLP(istkp, rewrite_acl(SST_ACLG(istkp), dtype, 0));
    break;
  default:
    interr("unexpected sst type for initialization list", SST_IDG(istkp), 3);
  }
}

ACL *
get_acl(int area)
{
  ACL *a;
  a = (ACL *)getitem(area, sizeof(ACL));
  BZERO(a, ACL, 1);
  return a;
}

ACL *
save_acl(ACL *oldp)
{
  ACL *rootp, *newp;
  SST *stkp;
  DOINFO *doinfo;

  if (oldp == NULL)
    return NULL;

  rootp = newp = GET_ACL(15);

  while (TRUE) {
    *newp = *oldp;
    switch (oldp->id) {
    case AC_EXPR:
      stkp = oldp->u1.stkp;
      if (SST_IDG(stkp) == S_ACONST) {
        newp->subc = SST_ACLG(stkp);
        newp->id = AC_ACONST;
      } else if (oldp->repeatc && oldp->size) {
      } else {
        newp->u1.ast = SST_ASTG(stkp);
        newp->id = AC_AST;
      }
      break;
    case AC_IDO:
      newp->subc = save_acl(oldp->subc);
      doinfo = get_doinfo(ACL_SAVE_AREA);
      *doinfo = *oldp->u1.doinfo;
      newp->u1.doinfo = doinfo;
      break;
    case AC_REPEAT:
    case AC_SCONST:
    case AC_ACONST:
      newp->subc = save_acl(oldp->subc);
      break;
    case AC_AST:
    case AC_ICONST:
    case AC_CONST:
      break;
    case AC_IEXPR:
      if (newp->u1.expr->lop) {
        newp->u1.expr->lop = save_acl(oldp->u1.expr->lop);
      }
      if (newp->u1.expr->rop) {
        newp->u1.expr->rop = save_acl(oldp->u1.expr->rop);
      }
      break;
    default:
      interr("save_acl,ill.id", oldp->id, 3);
      break;
    }
    oldp = oldp->next;
    if (oldp == NULL)
      break;
    newp->next = GET_ACL(15);
    newp = newp->next;
  }

  return rootp;
}

static int dinit_array = 0;
static void
dinit_constructor(SPTR arr, ACL *aclp)
{
  if (DINITG(arr))
    return;

  {
    VAR *ivl = (VAR *)getitem(15, sizeof(VAR));
    int ast = mk_id(arr);
    SCP(arr, SC_STATIC);
    STYPEP(arr, ST_ARRAY);
    ivl->id = Varref;
    ivl->u.varref.ptr = ast;
    ivl->u.varref.id = S_IDENT;
    ivl->u.varref.dtype = A_DTYPEG(ast);
    ivl->u.varref.shape = A_SHAPEG(ast);
    ivl->u.varref.subt = NULL;
    ivl->next = NULL;
    DINITP(arr, 1);
    if (SCG(arr) != SC_NONE)
      sym_is_refd(arr);

    dinit(ivl, aclp);
  }
  DINITP(arr, 1); /* will set for ST_DERIVED arrays, too -  to indicate that
                     components have been inited.  */
}

static void
put_a_init_tree(int ast, int dinit_array)
{
  ACL temp;
  for (; ast; ast = A_RIGHTG(ast)) {
    if (A_TYPEG(ast) != A_INIT) {
      interr("put_a_init_tree: unknown ast type", A_TYPEG(ast), 3);
    } else {
      DTYPE dtype = A_DTYPEG(ast);
      switch (DTY(dtype)) {
      case TY_ARRAY:
        put_a_init_tree(A_LEFTG(ast), dinit_array);
        break;
      case TY_DERIVED:
        dinit_put(DINIT_TYPEDEF, DTY(dtype + 3));
        put_a_init_tree(A_LEFTG(ast), dinit_array);
        dinit_put(DINIT_ENDTYPE, 0);
        break;
      default:
        temp.id = AC_AST;
        temp.u1.ast = A_LEFTG(ast);
        temp.next = NULL;
        temp.subc = NULL;
        temp.dtype = A_DTYPEG(A_LEFTG(ast));
        temp.u2.array_i = dinit_array;
        _dinit_acl(&temp, FALSE);
        break;
      }
    }
  }
} /* put_a_init_tree */

static void
_dinit_acl(ACL *aclp, LOGICAL optimpldo)
{
  SST *stkp;
  DOINFO *doinfo;
  int ast, last, lastright;
  DTYPE dtype;
  int sptr;
  INT count, step;
  DOSTACK *tp;

  for (; aclp != NULL; aclp = aclp->next) {
    switch (aclp->id) {
    case AC_EXPR:
      stkp = aclp->u1.stkp;
      if (SST_IDG(stkp) == S_IDENT) {
        _dinit_acl(stkp->value.cnval.acl, FALSE);
      } else {
        /* the only AC_EXPR's left are those with A_INIT trees */
        ast = aclp->repeatc;
        last = aclp->size;
        /* break the list at 'last' */
        lastright = A_RIGHTG(last);
        A_RIGHTP(last, 0);
        put_a_init_tree(ast, dinit_array);
        /* restore the list at 'last' */
        A_RIGHTP(last, lastright);
      }
      break;
    case AC_AST:
      ast = aclp->u1.ast;
      sptr = 0;
      dtype = A_DTYPEG(ast);
      if (ast && A_TYPEG(ast) == A_ID) {
        sptr = A_SPTRG(ast);
      }
      if (sptr && (STYPEG(sptr) == ST_VAR || STYPEG(sptr) == ST_ARRAY) &&
          PARAMVALG(sptr)) {
        /* put out the initialization values */
        put_a_init_tree(PARAMVALG(sptr), dinit_array);
      } else if (DTY(dtype) == TY_ARRAY) {
        /* constructor item is an array */
        interr("_dinit_acl,array", ast, 3);
      } else if (A_ALIASG(ast)) {
        /* constructor item is a scalar constant */
        ast = A_ALIASG(ast);
        sptr = A_SPTRG(ast);
        switch (DTY(dtype)) {
        case TY_WORD:
        case TY_BINT:
        case TY_SINT:
        case TY_INT:
        case TY_BLOG:
        case TY_SLOG:
        case TY_LOG:
        case TY_REAL:
          dinit_put(dtype, CONVAL2G(sptr));
          break;
        case TY_CHAR:
          dinit_put(DINIT_STR, (INT)sptr);
          break;
        default:
          dinit_put(dtype, (INT)sptr);
          break;
        }
      } else if (DTY(astb.bnd.dtype) == TY_INT8) {
        /* constructor item is a scalar expression*/
        INT v[2];

        /* NOTE: dinit_eval() returns a 4-byte int. this is
           wrong, but until it gets fixed, this will have to
           do. */
        v[1] = dinit_eval(ast);
        if (v[1] < 0)
          v[0] = -1;
        else
          v[0] = 0;
        dinit_put(astb.bnd.dtype, getcon(v, astb.bnd.dtype));
      } else
        /* constructor item is a scalar expression*/
        dinit_put(astb.bnd.dtype, dinit_eval(ast));

      break;
    case AC_SCONST:
      dinit_put(DINIT_TYPEDEF, DTY(aclp->dtype + 3));
      _dinit_acl(aclp->subc, FALSE);
      dinit_put(DINIT_ENDTYPE, 0);
      break;
    case AC_ACONST:
      dinit_put(DINIT_STARTARY, 0);
      _dinit_acl(aclp->subc, FALSE);
      dinit_put(DINIT_ENDARY, 0);
      break;
    case AC_IDO:
      doinfo = aclp->u1.doinfo;
      if (sem.top == &sem.dostack[MAX_DOSTACK]) {
        /*  nesting maximum exceeded.  */
        errsev(34);
        return;
      }
      count = CONVAL2G(A_SPTRG(A_ALIASG(doinfo->count)));
      tp = sem.top;
      tp->sptr = doinfo->index_var;
      tp->currval = dinit_eval(doinfo->init_expr);
      step = dinit_eval(doinfo->step_expr);
      ++sem.top;
      /*
       * optimize the case where the initializer controlled by the
       * implied do is a single scalar constant
       */
      if (optimpldo && aclp->subc->id == AC_AST && aclp->subc->next == NULL &&
          DTY(A_DTYPEG(aclp->subc->u1.ast)) != TY_ARRAY &&
          A_ALIASG(aclp->subc->u1.ast)) {
        dinit_put(DINIT_REPEAT, count);
        _dinit_acl(aclp->subc, optimpldo);
        tp->currval += count * step;
      } else
        while (count-- > 0) {
          _dinit_acl(aclp->subc, optimpldo);
          tp->currval += step;
        }
      --sem.top;
      break;
    case AC_REPEAT:
      dinit_put(DINIT_REPEAT, aclp->u1.count);
      ast = aclp->subc->u1.ast;
      dtype = A_DTYPEG(ast);
      ast = A_ALIASG(ast);
      sptr = A_SPTRG(ast);
      if (DT_ISWORD(dtype))
        dinit_put(dtype, CONVAL2G(sptr));
      else
        dinit_put(dtype, (INT)sptr);
      break;
    default:
      interr("_dinit_acl,ill.id", aclp->id, 3);
      break;
    }
  }
}

typedef struct struct_init {
  int default_count; /* is sptr+1, sptr is the last default init */
  int dt_count;      /* number of members */
  ACL **default_acl; /* if sptr is inited, points to default acl*/
  ACL **dt_acl;      /* points to all inited acl */
} struct_init;

static struct_init dt_init = {0, 0, NULL, NULL};

#define DTC_DEFAULT_HEAD dt_init.default_acl
#define DTC_ACL_HEAD dt_init.dt_acl
#define DTC_DEFAULT(i) dt_init.default_acl[i]
#define DTC_ACL(i) dt_init.dt_acl[i]
#define DTC_DEFAULT_CNT dt_init.default_count
#define DTC_DT_CNT dt_init.dt_count

static char *
make_structkwd_str(DTYPE dtype, int *num_of_member, int *is_extend)
{
  int i;
  char *name;
  int optional = 1; /* all are optional */
  int len;
  int size;
  int avl;
  int member_sptr, ptr_sptr = 0, thissptr, myparent;
  char *kwd_str = NULL;
  char *first_str = NULL;
  int num, is_extend2, num_of_member2;
  int possible_ext = 1;

  num = 0;
  avl = 0;
  i = 0;
  len = 0;
  size = 100;
  NEW(kwd_str, char, size);
  *kwd_str = '\0';
  member_sptr = DTY(dtype + 1);
  ptr_sptr = member_sptr;
  for (; member_sptr > NOSYM; member_sptr = SYMLKG(member_sptr)) {
    if (POINTERG(member_sptr))
      ptr_sptr = member_sptr;
    if (is_tbp_or_final(member_sptr)) {
      possible_ext = 0;
      continue; /* skip tbp */
    }
    name = SYMNAME(member_sptr);
    len = strlen(name);
    if (ptr_sptr &&
        (member_sptr == MIDNUMG(ptr_sptr) || member_sptr == PTROFFG(ptr_sptr) ||
         member_sptr == SDSCG(ptr_sptr) ||
         (CLASSG(member_sptr) && DESCARRAYG(member_sptr)))) {
      /* skip pointer related members */
      possible_ext = 0;
      continue;
    }
    ptr_sptr =
        USELENG(member_sptr) || POINTERG(member_sptr) || ALLOCATTRG(member_sptr)
            ? member_sptr
            : 0;

    /* NOTE: should make kwd_str static  */
    thissptr = DTY(dtype + 1);
    myparent = PARENTG(thissptr);
    if (myparent && myparent == PARENTG(member_sptr) && possible_ext &&
        (DTY(DTYPEG(member_sptr)) == TY_DERIVED ||
         DTY(DTYPEG(member_sptr)) == TY_STRUCT)) {
      *is_extend = 1;
      first_str =
          make_structkwd_str(DTYPEG(member_sptr), &num_of_member2, &is_extend2);
      len = strlen(first_str);
      i = 0;
      num += num_of_member2;
      avl += len; /* len chars in name, 1 for ' ', 1 for null */
      if (avl > size) {
        NEED(avl, kwd_str, char, size, size + avl + 100);
      }
      strcpy(kwd_str, first_str);
      FREE(first_str);
    } else {
      if (member_sptr <= DTC_DEFAULT_CNT - 1 && DTC_DEFAULT(member_sptr))
        optional = 1;
      else
        optional = 0;
      i = avl;
      avl +=
          (optional + len + 2); /* len chars in name, 1 for ' ', 1 for null */
      NEED(avl, kwd_str, char, size, size + 100);
      if (optional)
        kwd_str[i++] = '*';
      strcpy(kwd_str + i, name);
      kwd_str[i + len] = ' ';
      kwd_str[i + len + 1] = '\0';
      ++num;
      avl--;
    }
    possible_ext = 0; /* only the first member is extended type member */
  }

  *num_of_member = num;

  /* Allocate ACL pointers to all members , reuse if possible*/
  if (DTC_DT_CNT < num) {
    NEED(num, DTC_ACL_HEAD, ACL *, DTC_DT_CNT, num);
  }
  BZERO(DTC_ACL_HEAD, ACL *, DTC_DT_CNT);
  return kwd_str;
}

void
clean_struct_default_init(int sptr)
{
  int i;
  if (sptr == 0) {
    FREE(DTC_DEFAULT_HEAD);
    FREE(DTC_ACL_HEAD);
    DTC_DEFAULT_HEAD = NULL;
    DTC_ACL_HEAD = NULL;
    DTC_DEFAULT_CNT = 0;
    DTC_DT_CNT = 0;
  } else {
    /* only clean from the sptr, this is a case of contained routine */
    if (DTC_DEFAULT_CNT == 0)
      return;
    for (i = sptr; i < DTC_DEFAULT_CNT; ++i) {
      DTC_DEFAULT(i) = NULL;
    }
    DTC_DT_CNT = 0;
    FREE(DTC_ACL_HEAD);
    DTC_ACL_HEAD = NULL;
  }
}

int
has_init_value(SPTR sptr)
{
  if (sptr < DTC_DEFAULT_CNT) {
    if (DTC_DEFAULT(sptr))
      return 1;
  }
  return 0;
}

static ACL *
rewrite_typeinit_to_sconst(ACL *ict)
{
  ACL *newacl = ict;
  if (ict->id == AC_TYPEINIT) {
    newacl = GET_ACL(15);
    newacl->id = AC_SCONST;
    newacl->dtype = ict->dtype;
    newacl->next = ict->next;
    newacl->repeatc = ict->repeatc;
    newacl->subc = rewrite_typeinit_to_sconst(ict->subc);
  }
  return newacl;
}

/** \brief Duplicate a derived type component's default initializations.
 *
 * \param new_sptr is the component that receives the initialization copy.
 * \param old_sptr has the default initialization that we want to duplicate.
 *
 * We need to duplicate the initialization of a derived type component when
 * we create new instances of the derived type with different kind/len
 * type parameters.
 */
void
dup_struct_init(int new_sptr, int old_sptr)
{

  if (!has_init_value(old_sptr))
    return;

  if (DTC_DEFAULT_CNT == 0) {
    NEED(new_sptr + 1, DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT, new_sptr + 10);
    BZERO(DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT);
  } else if (DTC_DEFAULT_CNT - 1 < new_sptr) {
    int oldcnt = DTC_DEFAULT_CNT;
    NEED(new_sptr + 1, DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT, new_sptr + 10);
    BZERO((DTC_DEFAULT_HEAD + oldcnt), ACL *, DTC_DEFAULT_CNT - oldcnt);
  }

  DTC_DEFAULT(new_sptr) = DTC_DEFAULT(old_sptr);
}

void
save_struct_init(ACL *ict)
{
  ACL *newacl = ict;

  if (DTC_DEFAULT_CNT == 0) {
    NEED(ict->sptr + 1, DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT,
         ict->sptr + 10);
    BZERO(DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT);
  } else if (DTC_DEFAULT_CNT - 1 < ict->sptr) {
    int oldcnt = DTC_DEFAULT_CNT;
    NEED(ict->sptr + 1, DTC_DEFAULT_HEAD, ACL *, DTC_DEFAULT_CNT,
         ict->sptr + 10);
    BZERO((DTC_DEFAULT_HEAD + oldcnt), ACL *, DTC_DEFAULT_CNT - oldcnt);
  }
#if DEBUG
#endif

  if (ict->id == AC_TYPEINIT) {
    newacl = rewrite_typeinit_to_sconst(ict);
  }

  /* in module, the ..$p  is put in .mod file instead of member symbol */
  if (HCCSYMG(ict->sptr) && NEEDMODG(SCOPEG(ict->sptr))) {
    int sptr = VARIANTG(ict->sptr);
    if ((POINTERG(sptr) || ALLOCATTRG(sptr))) {
      if (MIDNUMG(sptr) == ict->sptr && SYMLKG(sptr) == ict->sptr) {
        DTC_DEFAULT(sptr) = newacl;
        return;
      }
    }
  }
  DTC_DEFAULT(ict->sptr) = newacl;
}

static ACL *
get_struct_default_init(int sptr)
{
  if (sptr > 0 && sptr <= DTC_DEFAULT_CNT - 1) {
    ACL *init_acl = DTC_DEFAULT(sptr);
    if (init_acl) {
      return clone_init_const(init_acl, 0);
    }
    return init_acl;
  } else {
    return NULL;
  }
}

/** \brief Check whether derived type has components with default
 *  initializations.
 *
 * \param dtype is the derived type we want to check.
 *
 * \return pointer to first default initializer, else NULL.
 */
ACL *
all_default_init(DTYPE dtype)
{
  int mem, myparent, thissptr;
  ACL *rslt, *dflt;
  int possible_ext = 1;

  rslt = dflt = NULL;
  if (DTY(dtype) != TY_DERIVED && DTY(dtype) != TY_STRUCT &&
      DTY(dtype) != TY_UNION) {
    return NULL;
  }

  thissptr = DTY(dtype + 1);
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (POINTERG(mem))
      thissptr = mem;
    myparent = PARENTG(thissptr);
    if (myparent && myparent == PARENTG(mem) && possible_ext &&
        DTY(DTYPEG(mem)) == TY_DERIVED) {
      dflt = all_default_init(DTYPEG(mem));
      if (dflt)
        return dflt;
    } else {
      if (is_tbp_or_final(mem))
        continue; /* skip tbp */
      if (thissptr &&
          (mem == MIDNUMG(thissptr) || mem == PTROFFG(thissptr) ||
           mem == SDSCG(thissptr) || (CLASSG(mem) && DESCARRAYG(mem)))) {
        /* skip pointer related members */
        possible_ext = 0;
        continue;
      }
      if (mem > 0 && mem <= DTC_DEFAULT_CNT - 1) {
        rslt = DTC_DEFAULT(mem);
        if (rslt == NULL) {
          return NULL;
        } else if (!dflt) {
          dflt = clone_init_const(rslt, 0);
        }
      } else {
        return NULL;
      }
    }
    possible_ext = 0;
  }
  return dflt;
}

static ACL *
get_exttype_list(int cnt)
{
  int i;
  ACL *first = NULL;
  ACL *prev = NULL;
  for (i = 0; i < cnt; ++i) {
    if (DTC_ACL(i)) {
      if (first == NULL) {
        first = DTC_ACL(i);
        prev = first;
        prev->next = NULL;
      } else {
        prev->next = DTC_ACL(i);
        prev = prev->next;
        prev->next = NULL;
      }
    }
  }
  return first;
}

static int
set_exttype_list(ACL *aclp)
{
  int i;
  ACL *first = aclp;
  for (i = 0; first != NULL; ++i) {
    DTC_ACL(i) = first;
    first = first->next;
  }
  for (; i < DTC_DT_CNT; ++i) {
    DTC_ACL(i) = 0;
  }
  return i;
}

static int
get_exttype_default(DTYPE dtype, int pos)
{
  int ptr_sptr = 0, thissptr, myparent;
  int member_sptr = DTY(dtype + 1);
  int possible_ext = 1;
  if (pos >= DTC_DT_CNT)
    return pos;

  ptr_sptr = member_sptr;
  for (; member_sptr > NOSYM; member_sptr = SYMLKG(member_sptr)) {
    if (no_data_components(DTYPEG(member_sptr))) {
      possible_ext = 0;
      continue;
    }
    if (CLASSG(member_sptr) && VTABLEG(member_sptr) && BINDG(member_sptr)) {
      possible_ext = 0;
      continue;
    }
    if (POINTERG(member_sptr))
      ptr_sptr = member_sptr;
    if (ptr_sptr &&
        (member_sptr == MIDNUMG(ptr_sptr) || member_sptr == PTROFFG(ptr_sptr) ||
         member_sptr == SDSCG(ptr_sptr) ||
         (CLASSG(member_sptr) && DESCARRAYG(member_sptr)))) {
      /* skip pointer related members */
      possible_ext = 0;
      continue;
    }
    ptr_sptr =
        USELENG(member_sptr) || POINTERG(member_sptr) || ALLOCATTRG(member_sptr)
            ? member_sptr
            : 0;

    thissptr = DTY(dtype + 1);
    myparent = PARENTG(thissptr);
    if (myparent && myparent == PARENTG(member_sptr) && possible_ext &&
        DTY(DTYPEG(member_sptr)) == TY_DERIVED) {
      if (!no_data_components(DTYPEG(member_sptr)))
        pos = get_exttype_default(DTYPEG(member_sptr), pos);
    } else {
      if (DTC_ACL(pos) == NULL)
        DTC_ACL(pos) = get_struct_default_init(member_sptr);
      ++pos;
    }

    possible_ext = 0; /* only the first member is extended type member */
    if (pos > DTC_DT_CNT)
      return pos;
  }
  return pos;
}

/* Also create a new ACL of base type if the initialization list of
 * extended type is not in the form of base_type(..).
 * This is getting complicated.
 */

static LOGICAL
get_keyword_components(ACL *in_aclp, int cnt, char *kwdarg, DTYPE dtype,
                       int is_extend)
{
  SST *stkp;
  int pos;
  int i;
  char *kwd, *np;
  int kwd_len;
  char *actual_kwd; /* name of keyword used with the actual arg */
  int actual_kwd_len;
  LOGICAL kwd_present;
  ACL *t_aclp, *aclp = in_aclp->subc;
  int member_sptr;

  /* convention for the keyword 'variable' arguments ---
   * the keyword specifier is of the form
   *     #<pos>#<base>#<kwd>
   * where,
   *      <pos>  = digit indicating the zero-relative positional index where
   *               the variable arguments begin in the argument list.
   *      <base> = digit indicating value to be subtracted from the digit
   *               string suffix of the keyword.
   *      <kwd>  = name of the keyword which varies (i.e., the prefix).
   */

  if (*kwdarg == '\0' || *kwdarg == ' ')
    return TRUE;
  kwd_present = FALSE;
  for (i = 0; i < cnt; i++) {
    DTC_ACL(i) = NULL;
  }

  for (pos = 0; aclp != NULL; pos++) {
    if (aclp->id == AC_EXPR) {
      stkp = aclp->u1.stkp;
      if (SST_IDG(stkp) == S_KEYWORD) {
        kwd_present = TRUE;
        actual_kwd = scn.id.name + SST_CVALG(stkp);
        actual_kwd_len = strlen(actual_kwd);
        kwd = kwdarg;
        for (i = 0; TRUE; i++) {
          if (*kwd == '*')
            kwd++;
          kwd_len = 0;
          for (np = kwd; TRUE; np++, kwd_len++)
            if (*np == ' ' || *np == '\0')
              break;
          if (kwd_len == actual_kwd_len &&
              strncmp(kwd, actual_kwd, actual_kwd_len) == 0)
            break;
          if (*np == '\0')
            goto ill_keyword;
          kwd = np + 1; /* skip over blank */
        }
        if (i > cnt)
          error(155, 3, gbl.lineno,
                "Too many elements in structure constructor", CNULL);
        if (DTC_ACL(i))
          goto ill_keyword;
        stkp = SST_E3G(stkp);
        aclp->u1.stkp = stkp; /* Should this be done?*/
        if (SST_IDG(stkp) == S_SCONST)
          DTC_ACL(i) = SST_ACLG(stkp);
        else
          DTC_ACL(i) = aclp; /* should SST_IDG change?*/
      } else {
        if (kwd_present) {
          error(155, 4, gbl.lineno,
                "Positional components must not follow keyword arguments",
                CNULL);
          return TRUE;
        }
        if (pos > cnt)
          error(155, 3, gbl.lineno,
                "Too many elements in structure constructor", CNULL);
        if (DTC_ACL(pos)) {
          char print[22];
          kwd = kwdarg;
          for (i = 0; TRUE; i++) {
            if (*kwd == '*' || *kwd == ' ')
              kwd++;
            if (*kwd == '\0') {
              error(155, 3, gbl.lineno,
                    "Invalid element in structure constructor", CNULL);
              return TRUE;
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
          if (kwd_len > 21)
            kwd_len = 21;
          strncpy(print, kwd, kwd_len);
          print[kwd_len] = '\0';
          error(79, 3, gbl.lineno, print, CNULL);
          return TRUE;
        }
        DTC_ACL(pos) = aclp;
      }
    } else {
      if (kwd_present) {
        error(155, 4, gbl.lineno,
              "Positional components must not follow keyword components",
              CNULL);
        return TRUE;
      }
      DTC_ACL(pos) = aclp;
    }
    aclp = aclp->next;
    if (pos > cnt)
      errsev(67);
  }

  if (is_extend) {
    /* for extended type, the first member is the base type. */
    /* if kwd_present, then it must list all members in base type(s). */
    aclp = in_aclp->subc;
    member_sptr = DTY(dtype + 1);
    if (!no_data_components(DTYPEG(member_sptr))) {
      if (kwd_present || pos < cnt) {
        /* get default value here if keyword is present */
        pos = get_exttype_default(dtype, 0);
      }
      aclp = get_exttype_list(cnt);
      if (!(aclp->id == AC_SCONST &&
            cmpat_dtype_with_size(aclp->dtype, DTYPEG(member_sptr)))) {
        aclp = get_exttype_struct_constructor(aclp, dtype, &t_aclp);
      }
      in_aclp->subc = aclp;
      return kwd_present;
    }
  }

  /* determine if required component is not present.  */

  kwd = kwdarg;
  for (pos = 0; pos < cnt; pos++, kwd = np) {
    if (*kwd == ' ')
      kwd++;
    kwd_len = 0;
    for (np = kwd; TRUE; np++) {
      if (*np == ' ' || *np == '\0')
        break;
      kwd_len++;
    }
    if (DTC_ACL(pos) && sem.new_param_dt) {
      /* We have an initializer in a type parameter position...
       * skip over the type parameter since it is not defined in
       * the structure constructor portion of the syntax. Instead,
       * set the next component to this value and the type parameter
       * to its default value.
       */
      int i;
      char *buf = getitem(0, kwd_len + 1);
      strncpy(buf, kwd, kwd_len);
      buf[kwd_len] = '\0';

      if (*buf == '*')
        ++buf;

      put_default_kind_type_param(sem.new_param_dt, 0, 0);
      put_length_type_param(sem.new_param_dt, 0);

      i = get_kind_parm_by_name(buf, sem.new_param_dt);
      if (i != 0) {
        SST *e1;
        int j;

        for (j = (cnt - 1); j > pos; --j)
          DTC_ACL(j) = DTC_ACL(j - 1);

        e1 = (SST *)getitem(ACL_SAVE_AREA, sizeof(SST));
        if (i < 0) {
          int val = 0;
          i = get_len_set_parm_by_name(buf, sem.new_param_dt, &val);
          if (val) {
            SST_IDP(e1, S_EXPR);
            SST_DTYPEP(e1, DT_INT);
            SST_ASTP(e1, val);
          } else {
            SST_IDP(e1, S_CONST);
            SST_DTYPEP(e1, DT_INT);
            SST_CVALP(e1, i);
            SST_ASTP(e1, mk_cval1(i, DT_INT));
            SST_SHAPEP(e1, 0);
          }
        } else {

          SST_IDP(e1, S_CONST);
          SST_DTYPEP(e1, DT_INT);
          SST_CVALP(e1, i);
          SST_ASTP(e1, mk_cval1(i, DT_INT));
          SST_SHAPEP(e1, 0);
        }

        t_aclp = GET_ACL(15);
        t_aclp->id = AC_EXPR;
        t_aclp->repeatc = t_aclp->size = 0;
        t_aclp->next = NULL;
        t_aclp->subc = NULL;
        t_aclp->u1.stkp = e1;
        DTC_ACL(pos) = t_aclp;
        continue;
      }
    } else if (DTC_ACL(pos) == NULL) {
      /* If missing value in structure constructor is a type parameter,
       * then fill in the value here.
       */
      int i;
      char *buf = getitem(0, kwd_len + 1);
      strncpy(buf, kwd, kwd_len);
      buf[kwd_len] = '\0';
      if (*buf == '*')
        ++buf;
      if (sem.new_param_dt) {
        /* Make sure the default values are initialized */
        put_default_kind_type_param(sem.new_param_dt, 0, 0);
        put_length_type_param(sem.new_param_dt, 0);
      }
      if ((sem.new_param_dt &&
           (i = get_kind_parm_by_name(buf, sem.new_param_dt)))) {
        SST *e1;
        e1 = (SST *)getitem(ACL_SAVE_AREA, sizeof(SST));
        if (i < 0) {
          int val = 0;
          i = get_len_set_parm_by_name(buf, sem.new_param_dt, &val);
          if (val) {
            SST_IDP(e1, S_EXPR);
            SST_DTYPEP(e1, DT_INT);
            SST_ASTP(e1, val);
          } else {
            SST_IDP(e1, S_CONST);
            SST_DTYPEP(e1, DT_INT);
            SST_CVALP(e1, i);
            SST_ASTP(e1, mk_cval1(i, DT_INT));
            SST_SHAPEP(e1, 0);
          }
        } else {

          SST_IDP(e1, S_CONST);
          SST_DTYPEP(e1, DT_INT);
          SST_CVALP(e1, i);
          SST_ASTP(e1, mk_cval1(i, DT_INT));
          SST_SHAPEP(e1, 0);
        }

        t_aclp = GET_ACL(15);
        t_aclp->id = AC_EXPR;
        t_aclp->repeatc = t_aclp->size = 0;
        t_aclp->next = NULL;
        t_aclp->subc = NULL;
        t_aclp->u1.stkp = e1;

        DTC_ACL(pos) = t_aclp;
        continue;
      }
    }
    if (*kwd == '*') {
      continue;
    }

    if (DTC_ACL(pos) == NULL) {
      char print[22];
      if (kwd_len > 21)
        kwd_len = 21;
      strncpy(print, kwd, kwd_len);
      print[kwd_len] = '\0';
      error(155, 4, gbl.lineno,
            "No default initialization in structure constructor- member",
            print);

      return kwd_present;
    }
  }

  return kwd_present;

ill_keyword:
  error(155, 4, gbl.lineno,
        "Invalid component initialization in structure constructor", CNULL);
  return kwd_present;
}

/* Put in_aclp in a form similar its datatype.
 * Also check the default init value here.
 */
static ACL *
get_exttype_struct_constructor(ACL *in_aclp, DTYPE dtype, ACL **prev_aclp)
{
  int member_dtype, field_dtype;
  int member_sptr;
  int ptr_sptr = 0, thissptr, myparent;
  ACL *aclp, *head_aclp, *curr_aclp;
  SST *stkp;
  int ast, possible_ext = 1;

  aclp = in_aclp;
  head_aclp = in_aclp;
  curr_aclp = NULL;

#if DEBUG
  if (DBGBIT(3, 64))
    printacl("get_exttype_struct_constructor", aclp, gbl.dbgfil);
#endif

  member_sptr = DTY(dtype + 1);
  ptr_sptr = member_sptr;
  if (member_sptr == 0) {
    error(155, 3, gbl.lineno, "Use of derived type name before definition:",
          SYMNAME(DTY(dtype + 3)));
    return in_aclp;
  }
  for (; member_sptr != NOSYM && aclp != NULL;
       member_sptr = SYMLKG(member_sptr)) {
    if (no_data_components(DTYPEG(member_sptr))) {
      possible_ext = 0;
      continue;
    }
    if (is_tbp_or_final(member_sptr)) {
      possible_ext = 0;
      continue; /* skip tbp */
    }

    if (POINTERG(member_sptr))
      ptr_sptr = member_sptr;
    if (ptr_sptr &&
        (member_sptr == MIDNUMG(ptr_sptr) || member_sptr == PTROFFG(ptr_sptr) ||
         member_sptr == SDSCG(ptr_sptr) ||
         (CLASSG(member_sptr) && DESCARRAYG(member_sptr)))) {
      /* skip pointer related members */
      possible_ext = 0;
      continue;
    }
    ptr_sptr =
        USELENG(member_sptr) || POINTERG(member_sptr) || ALLOCATTRG(member_sptr)
            ? member_sptr
            : 0;
    thissptr = DTY(dtype + 1);
    myparent = PARENTG(thissptr);
    member_dtype = DTYPEG(member_sptr);
    field_dtype = member_dtype;
    if (possible_ext) {
      switch (aclp->id) {
      case AC_AST:
        ast = aclp->u1.ast;
        field_dtype = A_DTYPEG(ast);
        break;
      case AC_EXPR:
        stkp = aclp->u1.stkp;
        field_dtype = SST_DTYPEG(stkp);
        if (SST_IDG(stkp) == S_IDENT || SST_IDG(stkp) == S_LVALUE ||
            (SST_IDG(stkp) == S_EXPR && A_TYPEG(SST_ASTG(stkp)) == A_ID)) {
          SPTR sptr;
          if (SST_IDG(stkp) == S_IDENT) {
            sptr = SST_SYMG(stkp);
          } else if (SST_IDG(stkp) == S_EXPR &&
                     A_TYPEG(SST_ASTG(stkp)) == A_ID) {
            sptr = A_SPTRG(SST_ASTG(stkp));
          } else {
            sptr = SST_LSYMG(stkp);
          }
          if (DESCARRAYG(sptr) && DESCARRAYG(member_sptr)) {
            field_dtype = DDTG(field_dtype);
          }
          if (SCG(member_sptr) == SC_BASED &&
              (SCG(sptr) == SC_BASED || TARGETG(sptr) ||
               (SCG(sptr) == SC_CMBLK && POINTERG(sptr) &&
                !F90POINTERG(sptr)))) {
            field_dtype = DDTG(field_dtype);
          }
        } else if (SST_IDG(stkp) == S_EXPR) {
          field_dtype = 0;
        }
        break;
      case AC_ACONST:
      case AC_SCONST:
        field_dtype = aclp->dtype;
        break;
      default:
        field_dtype = 0;
        break;
      }
    }

    if (myparent && myparent == PARENTG(member_sptr) && possible_ext &&
        DTY(member_dtype) == TY_DERIVED &&
        !no_data_components(DTYPEG(member_dtype))) {
      if (!cmpat_dtype_with_size(field_dtype, member_dtype)) {
        head_aclp = GET_ACL(15);
        head_aclp->id = AC_SCONST;
        head_aclp->dtype = DDTG(member_dtype);
        head_aclp->next = NULL;
        *prev_aclp = aclp;
        head_aclp->subc = get_exttype_struct_constructor(
            aclp, DDTG(DTYPEG(member_sptr)), prev_aclp);
        if (*prev_aclp) {
          aclp = (*prev_aclp)->next;
          (*prev_aclp)->next = NULL;
          *prev_aclp = aclp;
        }
        curr_aclp = head_aclp;
        head_aclp->next = NULL;
      } else {
        *prev_aclp = aclp;
        if (curr_aclp)
          curr_aclp->next = aclp;
        curr_aclp = aclp;
        aclp = aclp->next;
      }
    } else {
      *prev_aclp = aclp;
      if (curr_aclp)
        curr_aclp->next = aclp;
      curr_aclp = aclp;
      aclp = aclp->next;
    }

    possible_ext = 0;
  }
  return head_aclp;
}

void
chk_struct_constructor(ACL *in_aclp)
{
  DTYPE dtype, member_dtype, field_dtype;
  int field_rank, member_rank;
  int member_sptr, memnum, cnt;
  int ptr_sptr = 0;
  ACL *aclp, *prev_aclp;
  SST *stkp;
  int ast, shape;
  int is_extend = 0;
  char *keyword;

  aclp = in_aclp;
#if DEBUG
  if (DBGBIT(3, 64))
    printacl("chk_struct_constructor", aclp, gbl.dbgfil);
#endif
  assert(aclp->id == AC_SCONST, "bad id in chk_struct_constructor", aclp->id,
         3);

  dtype = aclp->dtype;
  aclp = aclp->subc; /* go down to member list */
  member_sptr = DTY(dtype + 1);
  ptr_sptr = member_sptr;
  if (member_sptr == 0) {
    error(155, 3, gbl.lineno, "Use of derived type name before definition:",
          SYMNAME(DTY(dtype + 3)));
    return;
  }
  keyword = make_structkwd_str(dtype, &memnum, &is_extend);
  if (get_keyword_components(in_aclp, memnum, keyword, dtype, is_extend)) {
    ;
  }
  FREE(keyword);
  if (is_extend) {
    cnt = set_exttype_list(in_aclp->subc);
  }

  cnt = 0;
  prev_aclp = NULL;
  for (; member_sptr != NOSYM; member_sptr = SYMLKG(member_sptr)) {
    if (POINTERG(member_sptr))
      ptr_sptr = member_sptr;
    if (no_data_components(DTYPEG(member_sptr)))
      continue;
    if (is_tbp_or_final(member_sptr))
      continue; /* skip tbp */
    if (ptr_sptr &&
        (member_sptr == MIDNUMG(ptr_sptr) || member_sptr == PTROFFG(ptr_sptr) ||
         member_sptr == SDSCG(ptr_sptr) ||
         (CLASSG(member_sptr) && DESCARRAYG(member_sptr)))) {
      continue; /* skip pointer-related members */
    }
    ptr_sptr =
        USELENG(member_sptr) || POINTERG(member_sptr) || ALLOCATTRG(member_sptr)
            ? member_sptr
            : 0;

    aclp = DTC_ACL(cnt);
    if (aclp == NULL) {
      aclp = get_struct_default_init(member_sptr);
    }
    if (aclp)
      aclp->next = NULL;
    else
      error(155, 4, gbl.lineno,
            "No default initialization in structure constructor- member",
            SYMNAME(member_sptr));

    if (prev_aclp == NULL) {
      prev_aclp = aclp;
      in_aclp->subc = aclp;
    } else {
      prev_aclp->next = aclp;
      prev_aclp = aclp;
    }
    member_dtype = DTYPEG(member_sptr);
    member_rank = rank_of(member_dtype);

    ast = 0;
    switch (aclp->id) {
    case AC_AST:
      ast = aclp->u1.ast;
      field_dtype = A_DTYPEG(ast);
      shape = A_SHAPEG(ast);
      field_rank = (shape == 0) ? 0 : SHD_NDIM(shape);
      if ((POINTERG(member_sptr) || ALLOCATTRG(member_sptr))) {
        if (aclp->dtype == DT_PTR) {
          int tdtype = aclp->ptrdtype;
          if (DTY(tdtype) == TY_PTR) {
            field_dtype = DTY(tdtype + 1);
          }
        }
      }
      break;
    case AC_EXPR:
      stkp = aclp->u1.stkp;
      field_dtype = SST_DTYPEG(stkp);
      if (field_dtype)
        field_rank = rank_of(field_dtype);
      if (SST_IDG(stkp) == S_IDENT || SST_IDG(stkp) == S_LVALUE ||
          (SST_IDG(stkp) == S_EXPR && A_TYPEG(SST_ASTG(stkp)) == A_ID)) {
        int newast, sptr;
        if (SST_IDG(stkp) == S_IDENT) {
          sptr = SST_SYMG(stkp);
        } else if (SST_IDG(stkp) == S_EXPR && A_TYPEG(SST_ASTG(stkp)) == A_ID) {
          sptr = A_SPTRG(SST_ASTG(stkp));
        } else {
          sptr = SST_LSYMG(stkp);
        }
        if (DESCARRAYG(sptr) && DESCARRAYG(member_sptr)) {
          field_dtype = DDTG(field_dtype);
          member_dtype = DDTG(member_dtype);
        }
        if (SCG(member_sptr) == SC_BASED &&
            (SCG(sptr) == SC_BASED || TARGETG(sptr) ||
             (SCG(sptr) == SC_CMBLK && POINTERG(sptr) && !F90POINTERG(sptr)))) {
          /* add ACLs for pointer/offset/descriptor */
          ACL *naclp;
          SST *sp;
          int sdsc, ptroff, midnum;
          ast = SST_ASTG(stkp);
          if (ast) {
            shape = A_SHAPEG(ast);
            field_rank = (shape == 0) ? 0 : SHD_NDIM(shape);
          }
          field_dtype = DDTG(field_dtype);
          member_dtype = DDTG(member_dtype);
          if ((TARGETG(sptr) || POINTERG(sptr)) && SDSCG(sptr) == 0 &&
              !F90POINTERG(sptr)) {
            get_static_descriptor(sptr);
            if (POINTERG(sptr) || (ALLOCATTRG(sptr) && TARGETG(sptr))) {
              get_all_descriptors(sptr);
            }
          }
          sdsc = SDSCG(sptr);
          if (sdsc && SDSCG(member_sptr) &&
              STYPEG(SDSCG(member_sptr)) == ST_MEMBER) {

            sp = (SST *)getitem(ACL_AREA, sizeof(SST));
            if (SST_IDG(stkp) == S_IDENT) {
              SST_IDP(sp, S_IDENT);
              SST_SYMP(sp, sdsc);
            } else {
              SST_IDP(sp, S_LVALUE);
              SST_SYMP(sp, SST_SYMG(stkp));
              SST_LSYMP(sp, sdsc);
              newast = check_member(ast, mk_id(sdsc));
              SST_ASTP(sp, newast);
              SST_SHAPEP(sp, A_SHAPEG(newast));
            }
            SST_DTYPEP(sp, DTYPEG(sdsc));
            naclp = GET_ACL(ACL_AREA);
            naclp->id = AC_EXPR;
            naclp->repeatc = naclp->size = 0;
            naclp->next = prev_aclp->next;
            naclp->subc = NULL;
            naclp->u1.stkp = sp;
            prev_aclp->next = naclp;
            prev_aclp = naclp;

            sp = (SST *)getitem(ACL_AREA, sizeof(SST));
            ptroff = PTROFFG(sptr);
            if (ptroff == 0) {
              SST_IDP(sp, S_CONST);
              SST_SYMP(sp, stb.i0);
              SST_DTYPEP(sp, DTYPEG(stb.i0));
            } else if (SST_IDG(stkp) == S_IDENT) {
              SST_IDP(sp, S_IDENT);
              SST_SYMP(sp, ptroff);
              SST_DTYPEP(sp, DTYPEG(ptroff));
            } else {
              SST_IDP(sp, S_LVALUE);
              SST_SYMP(sp, SST_SYMG(stkp));
              SST_LSYMP(sp, ptroff);
              newast = check_member(ast, mk_id(ptroff));
              SST_ASTP(sp, newast);
              SST_SHAPEP(sp, A_SHAPEG(newast));
              SST_DTYPEP(sp, DTYPEG(ptroff));
            }
            naclp = GET_ACL(ACL_AREA);
            naclp->id = AC_EXPR;
            naclp->repeatc = naclp->size = 0;
            naclp->next = prev_aclp->next;
            naclp->subc = NULL;
            naclp->u1.stkp = sp;
            prev_aclp->next = naclp;
            prev_aclp = naclp;

            sp = (SST *)getitem(ACL_AREA, sizeof(SST));
            midnum = MIDNUMG(sptr);
            if (midnum == 0) {
              SST_IDP(sp, S_CONST);
              SST_SYMP(sp, stb.i0);
              SST_DTYPEP(sp, DTYPEG(stb.i0));
            } else if (SST_IDG(stkp) == S_IDENT) {
              SST_IDP(sp, S_IDENT);
              SST_SYMP(sp, midnum);
              SST_DTYPEP(sp, DTYPEG(midnum));
            } else {
              SST_IDP(sp, S_LVALUE);
              SST_SYMP(sp, SST_SYMG(stkp));
              SST_LSYMP(sp, midnum);
              newast = check_member(ast, mk_id(midnum));
              SST_ASTP(sp, newast);
              SST_SHAPEP(sp, A_SHAPEG(ast));
              SST_DTYPEP(sp, DTYPEG(midnum));
            }
            naclp = GET_ACL(ACL_AREA);
            naclp->id = AC_EXPR;
            naclp->repeatc = naclp->size = 0;
            naclp->next = prev_aclp->next;
            naclp->subc = NULL;
            naclp->u1.stkp = sp;
            prev_aclp->next = naclp;
            prev_aclp = naclp;
          }
        }
      } else if (SST_IDG(stkp) == S_EXPR) {
        /* handle call to NULL() */
        ast = SST_ASTG(stkp);
        field_dtype = 0;
        field_rank = 0;
        if (A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_NULL) {
          field_dtype = A_DTYPEG(ast);
          if (POINTERG(member_sptr) || ALLOCATTRG(member_sptr)) {
            member_dtype = DT_PTR;
          }
        }
      }
      break;
    case AC_ACONST:
    case AC_SCONST:
      field_dtype = aclp->dtype;
      field_rank = rank_of(field_dtype);
      break;
    default:
      field_dtype = 0;
      field_rank = 0;
      break;
    }
    if ((field_rank && member_rank && field_rank != member_rank) ||
        (field_dtype && !cmpat_dtype_with_size(field_dtype, member_dtype))) {
      if (DTY(DTYPEG(member_sptr)) != TY_PTR &&
          DTY(DTY(DTYPEG(member_sptr) + 1)) != TY_PROC)
        error(155, 2, gbl.lineno, "Mismatched data type for member",
              SYMNAME(member_sptr));
    }
    if (is_illegal_expr_in_init(member_sptr, ast, aclp->dtype)) {
      error(457, 3, gbl.lineno, CNULL, CNULL);
    }

    cnt++;
  }
  if (cnt > memnum)
    error(155, 4, gbl.lineno,
          "Too many elements in structure constructor- type",
          SYMNAME(DTY(dtype + 3)));

  /* may want to set is_const flag in aclp if all members are constant */
}

static bool
is_illegal_expr_in_init(SPTR member_sptr, int ast, DTYPE acl_dtype)
{
  if (!sem.dinit_data)
    return false;
  if (!POINTERG(member_sptr) && !ALLOCATTRG(member_sptr))
    return false;
  if (ast == 0)
    return true;
  if (A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_NULL)
    return false;
  if (ast != astb.i0 || acl_dtype != DT_PTR ||
      DTY(ENCLDTYPEG(member_sptr)) != TY_DERIVED)
    return true;
  return false;
}

int
init_derived_w_acl(int in_sptr, ACL *sconst)
{
  int sptr, dtype, tag;

  if (in_sptr)
    sptr = in_sptr;
  else {
    dtype = sconst->dtype;
    tag = DTY(dtype + 3);
    sptr = get_next_sym(SYMNAME(tag), "d");
    STYPEP(sptr, ST_VAR);
    DCLDP(sptr, 1);
    SCP(sptr, sem.sc);
    DTYPEP(sptr, dtype);
    add_alloc_mem_initialize(sptr);
  }

  constructf90(sptr, sconst);

  return sptr;
}

/*
 * keep track of an initialization ast tree.
 * this is a list of ast nodes linked by A_RIGHT fields;
 * the A_TYPE is A_INIT
 * the A_LEFT field points to the initialization value.
 * the A_SPTR field, if set, points to the variable or member symbol.
 */

typedef struct {
  int head, tail;
} ASTLIST;

static void
append_init_list(ASTLIST *target, ASTLIST *src)
{
  if (target->head == 0) {
    *target = *src;
  } else {
    A_RIGHTP(target->tail, src->head);
    target->tail = src->tail;
  }
}

static void
add_init(ASTLIST *list, int left, DTYPE dtype, int sptr)
{
  int ast;
  ast = mk_init(left, dtype);
  A_SPTRP(ast, sptr);
  if (list->head == 0) {
    list->head = ast;
  } else {
    A_RIGHTP(list->tail, ast);
  }
  list->tail = ast;
} /* add_init */

static LOGICAL out_of_elements_message;

/*
 * Evaluate a constant expression.  Code borrowed from dinit_eval() and
 * changed to allow expression types other than integer.
 * Part of the fix for FS2281.
 */
static INT
const_eval(int ast)
{
  DOSTACK *p;
  int sptr;
  INT val;
  int lop, rop;
  INT term;
  INT lv, rv;
  int count;
  int sign;

  if (ast == 0)
    return 1L;
  if (A_ALIASG(ast)) {
    ast = A_ALIASG(ast);
    goto eval_cnst;
  }
  switch (A_TYPEG(ast) /* opc */) {
  case A_ID:
    if (!DT_ISINT(A_DTYPEG(ast)))
      goto cnst_err;
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      goto eval_cnst;
    }
    /*  see if this ident is an active do index variable: */
    sptr = A_SPTRG(ast);
    for (p = sem.dostack; p < sem.top; p++)
      if (p->sptr == sptr)
        return p->currval;
    /*  else - illegal use of variable: */
    error(64, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    sem.dinit_error = TRUE;
    return 1L;

  case A_CNST:
    goto eval_cnst;

  case A_UNOP:
    val = const_eval((int)A_LOPG(ast));
    if (A_OPTYPEG(ast) == OP_SUB)
      val = negate_const(val, A_DTYPEG(ast));
    if (A_OPTYPEG(ast) == OP_LNOT)
      val = ~(val);
    return val;

  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
      return const_fold(A_OPTYPEG(ast), const_eval((int)A_LOPG(ast)),
                        const_eval((int)A_ROPG(ast)), A_DTYPEG(ast));

    case OP_EQ:
    case OP_GE:
    case OP_GT:
    case OP_LE:
    case OP_LT:
    case OP_NE:
      val = const_fold(OP_CMP, const_eval((int)A_LOPG(ast)),
                       const_eval((int)A_ROPG(ast)), A_DTYPEG(A_LOPG(ast)));
      switch (A_OPTYPEG(ast)) {
      case OP_EQ:
        val = (val == 0);
        break;
      case OP_GE:
        val = (val >= 0);
        break;
      case OP_GT:
        val = (val > 0);
        break;
      case OP_LE:
        val = (val <= 0);
        break;
      case OP_LT:
        val = (val < 0);
        break;
      case OP_NE:
        val = (val != 0);
        break;
      }
      val = val ? SCFTN_TRUE : SCFTN_FALSE;
      return val;

    case OP_LEQV:
    case OP_LNEQV:
    case OP_LOR:
    case OP_LAND:
      lv = const_eval((int)A_LOPG(ast));
      rv = const_eval((int)A_ROPG(ast));
      switch (A_OPTYPEG(ast)) {
      case OP_LEQV:
        val = (lv == rv) ? SCFTN_TRUE : SCFTN_FALSE;
        FLANG_FALLTHROUGH;
      case OP_LNEQV:
        val = (lv == rv) ? SCFTN_FALSE : SCFTN_TRUE;
        FLANG_FALLTHROUGH;
      case OP_LOR:
        val = (lv == SCFTN_TRUE || rv == SCFTN_TRUE) ? SCFTN_TRUE : SCFTN_FALSE;
        FLANG_FALLTHROUGH;
      case OP_LAND:
        val = (lv == SCFTN_TRUE && rv == SCFTN_TRUE) ? SCFTN_TRUE : SCFTN_FALSE;
      }
      return val;
    case OP_XTOI:
      lop = A_LOPG(ast);
      rop = A_ROPG(ast);
      if (A_DTYPEG(rop) == DT_INT8) {
        term = stb.k1;
        if (A_DTYPEG(lop) != DT_INT8)
          term = cngcon(term, DT_INT8, A_DTYPEG(lop));
        val = term;
        lv = const_eval(lop);
        rv = const_eval(rop);
        count = get_int_cval(rv);
        count = (count < 0) ? -count : count;
        while (count--)
          val = const_fold(OP_MUL, val, lv, A_DTYPEG(lop));
        if (get_int_cval(rv) < 0) {
          /* exponentiation to a negative power */
          val = const_fold(OP_DIV, term, val, A_DTYPEG(lop));
        }
      } else if (DT_ISINT(A_DTYPEG(rop))) {
        term = 1;
        if (A_DTYPEG(lop) != DT_INT4)
          term = cngcon(term, DT_INT4, A_DTYPEG(lop));
        val = term;
        lv = const_eval(lop);
        rv = const_eval(rop);
        if (A_DTYPEG(rop) != DT_INT4)
          rv = cngcon(rv, A_DTYPEG(rop), DT_INT4);
        if (rv >= 0)
          sign = 0;
        else {
          rv = -rv;
          sign = 1;
        }
        while (rv--)
          val = const_fold(OP_MUL, val, lv, A_DTYPEG(lop));
        if (sign) {
          /* exponentiation to a negative power */
          val = const_fold(OP_DIV, term, val, A_DTYPEG(lop));
        }
      } else {
        lv = const_eval(lop);
        rv = const_eval(rop);
        val = const_fold(OP_XTOI, lv, rv, A_DTYPEG(lop));
      }
      return val;
    }
    break;

  case A_CONV:
    val = const_eval((int)A_LOPG(ast));
    return cngcon(val, A_DTYPEG(A_LOPG(ast)), A_DTYPEG(ast));

  case A_PAREN:
    return const_eval((int)A_LOPG(ast));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_NULL:
      return 0;
    case I_NCHAR:

      /* kanji/international character sets */

      val = A_ARGSG(ast);
      val = ARGT_ARG(val, 0);
      if (A_TYPEG(val) == A_CNST) {
        int con1, con2, bytes;
        con1 = A_SPTRG(val);
        con2 = CONVAL1G(con1);
        count = size_of(DTYPEG(con2));
        val = kanji_char((unsigned char *)stb.n_base + CONVAL1G(con2), count,
                         &bytes);
        return val;
      }
      break;
    case I_ICHAR:
    case I_IACHAR:
      val = A_ARGSG(ast);
      val = ARGT_ARG(val, 0);
      if (A_TYPEG(val) == A_CNST) {
        val = A_SPTRG(val);
        count = size_of(DTYPEG(val));
        if (count == 1) {
          val = stb.n_base[CONVAL1G(val)] & 0xff;
          return val;
        }
      }
      break;
    case I_INT:
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      val = const_eval(ast);
      return cngcon(val, A_DTYPEG(ast), DT_INT);
    case I_INT8:
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      val = const_eval(ast);
      return cngcon(val, A_DTYPEG(ast), DT_INT8);
    case I_INT4:
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      val = const_eval(ast);
      return cngcon(val, A_DTYPEG(ast), DT_INT4);
    case I_INT2:
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      val = const_eval(ast);
      return cngcon(val, A_DTYPEG(ast), DT_SINT);
    case I_INT1:
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      val = const_eval(ast);
      return cngcon(val, A_DTYPEG(ast), DT_BINT);
    case I_SIZE: {
      int sz;
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      ast = ADD_NUMELM(A_DTYPEG(ast));
      sz = get_const_from_ast(ast);
      if (XBIT(68, 0x1) && A_ALIASG(ast) && !DT_ISWORD(A_DTYPEG(ast))) {
        sz = get_int_cval(sz);
      }
      return sz;
    }
    case I_LBOUND: {
      int lwb, dim;
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      dim = get_const_from_ast(ARGT_ARG(val, 1));
      ast = ADD_LWAST(A_DTYPEG(ast), dim - 1);
      lwb = get_const_from_ast(ast);
      if (XBIT(68, 0x1) && A_ALIASG(ast) && !DT_ISWORD(A_DTYPEG(ast))) {
        lwb = get_int_cval(lwb);
      }
      return lwb;
    }
    case I_UBOUND: {
      int upb, dim;
      val = A_ARGSG(ast);
      ast = ARGT_ARG(val, 0);
      dim = get_const_from_ast(ARGT_ARG(val, 1));
      ast = ADD_UPAST(A_DTYPEG(ast), dim - 1);
      upb = get_const_from_ast(ast);
      if (XBIT(68, 0x1) && A_ALIASG(ast) && !DT_ISWORD(A_DTYPEG(ast))) {
        upb = get_int_cval(upb);
      }
      return upb;
    }
    case I_MAX0: {
      int max, i, tmp;
      val = A_ARGSG(ast);
      max = get_const_from_ast(ARGT_ARG(val, 0));
      for (i = 1; i < A_ARGCNTG(ast); ++i) {
        tmp = get_const_from_ast(ARGT_ARG(val, i));
        if (tmp > max) {
          max = tmp;
        }
      }
      return max;
    }
    case I_MIN0: {
      int min, i, tmp;
      val = A_ARGSG(ast);
      min = get_const_from_ast(ARGT_ARG(val, 0));
      for (i = 1; i < A_ARGCNTG(ast); ++i) {
        tmp = get_const_from_ast(ARGT_ARG(val, i));
        if (tmp < min) {
          min = tmp;
        }
      }
      return min;
    }
    }
    break;
  default:
    break;
  }
cnst_err:
  errsev(69);
  sem.dinit_error = TRUE;
  A_DTYPEP(ast, DT_INT);
  return 1L;

eval_cnst:
  val = A_SPTRG(ast);
  if (DT_ISWORD(DTY(A_DTYPEG(ast))))
    val = CONVAL2G(val);
  return val;
}

/*
 * make sure 'ast' is a constant of the proper datatype
 */
static int
dinit_getval(int ast, DTYPE dtype)
{
  DTYPE adtype;
  int aval, val;
  if (!A_ALIASG(ast)) {
    /* nothing to do right now */
    if (dtype == 0)
      dtype = A_DTYPEG(ast);
    aval = dinit_eval(ast);
    ast = mk_cval(aval, DT_INT);
  }
  if (dtype == 0)
    return ast;
  adtype = A_DTYPEG(ast);
  if (adtype == dtype)
    return ast;
  if (!DT_ISSCALAR(adtype) || !DT_ISSCALAR(dtype)) {
    return 0;
  }
  ast = A_ALIASG(ast);
  aval = A_SPTRG(ast);
  adtype = DTYPEG(aval);
  if (DT_ISWORD(adtype))
    aval = CONVAL2G(aval);
  val = cngcon(aval, adtype, dtype);
  ast = mk_cval1(val, dtype);
  return ast;
} /* dinit_getval */

/*
 * Similar to dinit_getval, above, but allows types other than integer.
 * Part of the fix for FS2281.
 */
static int
dinit_getval1(int ast, DTYPE dtype)
{
  DTYPE adtype;
  INT aval, val;
  if (!A_ALIASG(ast)) {
    if (dtype == 0)
      dtype = A_DTYPEG(ast);
    aval = const_eval(ast);
    ast = mk_cval(aval, A_DTYPEG(ast));
  }
  if (dtype == 0)
    return ast;
  adtype = A_DTYPEG(ast);
  if (adtype == dtype)
    return ast;
  if (!DT_ISSCALAR(adtype) || !DT_ISSCALAR(dtype)) {
    return 0;
  }
  ast = A_ALIASG(ast);
  aval = A_SPTRG(ast);
  adtype = DTYPEG(aval);
  if (DT_ISWORD(adtype))
    aval = CONVAL2G(aval);
  val = cngcon(aval, adtype, dtype);
  ast = mk_cval1(val, dtype);
  return ast;
} /* dinit_getval1 */

static int
unop_init_list(int llist, int optype)
{
  int ll, list, last, nlist;
  list = last = 0;
  if (!llist) {
    /* error return */
    interr("unop_init_list, no llist", 0, 3);
    return 0;
  }
  for (ll = llist; ll; ll = A_RIGHTG(ll)) {
    int le;
    le = A_LEFTG(ll);
    if (A_TYPEG(le) == A_INIT) {
      nlist = unop_init_list(le, optype);
    } else {
      /* do the operation */
      nlist = mk_unop(optype, le, A_DTYPEG(le));
    }
    nlist = mk_init(nlist, A_DTYPEG(nlist));
    if (last) {
      A_RIGHTP(last, nlist);
    } else {
      list = nlist;
    }
    last = nlist;
  }
  return list;
} /* unop_init_list */

static int
binop_init_list(int llist, int rlist, int lop, int rop, int optype)
{
  int ll, rl, list, last, nlist;
  list = last = 0;
  if (lop && rop) {
    /* error return */
    interr("binop_init_list, lop&&rop", 0, 3);
    return 0;
  }
  if (!lop && !llist) {
    /* error return */
    interr("binop_init_list, neither lop nor llist", 0, 3);
    return 0;
  }
  if (!rop && !rlist) {
    /* error return */
    interr("binop_init_list, neither rop nor rlist", 0, 3);
    return 0;
  }
  if (!llist && !rlist) {
    /* error return */
    interr("binop_init_list, neither llist nor rlist", 0, 3);
    return 0;
  }
  if (llist && rlist) {
    for (ll = llist, rl = rlist; ll && rl;
         ll = A_RIGHTG(ll), rl = A_RIGHTG(rl)) {
      /* ll and rl are at an 'A_INIT' */
      int le, re;
      le = A_LEFTG(ll);
      re = A_LEFTG(rl);
      if (A_TYPEG(le) == A_INIT && A_TYPEG(re) == A_INIT) {
        nlist = binop_init_list(le, re, 0, 0, optype);
      } else if (A_TYPEG(le) == A_INIT) {
        nlist = binop_init_list(le, 0, 0, re, optype);
      } else if (A_TYPEG(re) == A_INIT) {
        nlist = binop_init_list(0, re, le, 0, optype);
      } else {
        /* do the operation */
        nlist = mk_binop(optype, le, re, A_DTYPEG(le));
      }
      nlist = mk_init(nlist, A_DTYPEG(nlist));
      if (last) {
        A_RIGHTP(last, nlist);
      } else {
        list = nlist;
      }
      last = nlist;
    }
  } else if (llist) {
    for (ll = llist; ll; ll = A_RIGHTG(ll)) {
      int le;
      le = A_LEFTG(ll);
      if (A_TYPEG(le) == A_INIT) {
        nlist = binop_init_list(le, 0, 0, rop, optype);
      } else {
        /* do the operation */
        nlist = mk_binop(optype, le, rop, A_DTYPEG(le));
      }
      nlist = mk_init(nlist, A_DTYPEG(nlist));
      if (last) {
        A_RIGHTP(last, nlist);
      } else {
        list = nlist;
      }
      last = nlist;
    }
  } else if (rlist) {
    for (rl = rlist; rl; rl = A_RIGHTG(rl)) {
      int re;
      re = A_LEFTG(rl);
      if (A_TYPEG(re) == A_INIT) {
        nlist = binop_init_list(0, re, lop, 0, optype);
      } else {
        /* do the operation */
        nlist = mk_binop(optype, lop, re, A_DTYPEG(re));
      }
      nlist = mk_init(nlist, A_DTYPEG(nlist));
      if (last) {
        A_RIGHTP(last, nlist);
      } else {
        list = nlist;
      }
      last = nlist;
    }
  }
  return list;
} /* binop_init_list */

static void
add_subscript_list(ASTLIST *list, int ast, int arraylist, int ssval[], int ndim)
{
  /* find shape for array at 'ast', use that plus values of ssval[]
   * to pick a value from 'arraylist' */
  int a, sh, i, offset, o;
  a = A_LOPG(ast);
  sh = A_SHAPEG(a);
  assert(SHD_NDIM(sh) == ndim,
         "add_subscript_list, shape rank != subscript rank",
         SHD_NDIM(sh) - ndim, 3);
  offset = 0;
  for (i = 0; i < SHD_NDIM(sh); ++i) {
    int l, lsptr, lb, u, usptr, ub, ss, ssptr, ssv;
    l = SHD_LWB(sh, i);
    assert(A_ALIASG(l), "add_subscript_list: nonconstant array lower bound", l,
           3);
    l = A_ALIASG(l);
    lsptr = A_SPTRG(l);
    lb = CONVAL2G(lsptr);
    u = SHD_UPB(sh, i);
    assert(A_ALIASG(u), "add_subscript_list: nonconstant array upper bound", u,
           3);
    u = A_ALIASG(u);
    usptr = A_SPTRG(u);
    ub = CONVAL2G(usptr);
    ss = ssval[i];
    assert(A_ALIASG(ss), "add_subscript_list: nonconstant subscript", ss, 3);
    ss = A_ALIASG(ss);
    ssptr = A_SPTRG(ss);
    ssv = CONVAL2G(ssptr);
    if (ub >= lb)
      offset *= (ub - lb + 1);
    if (ssv >= lb)
      offset += ssv - lb;
  }
  /* skip 'offset' items from the arraylist, add that value to 'list' */
  for (o = arraylist; o && offset; o = A_RIGHTG(o), --offset)
    ;
  if (o) {
    DTYPE dtype = DDTG(A_DTYPEG(ast));
    add_init(list, A_LEFTG(o), dtype, 0);
  }
} /* add_subscript_list */

static void
build_subscript_list(ASTLIST *list, int ast, int arraylist, int ssval[],
                     int sslist[], int dim, int ndim)
{
  if (sslist[dim] == 0) {
    /* only one value for dimension 'dim' */
    if (dim > 0) {
      build_subscript_list(list, ast, arraylist, ssval, sslist, dim - 1, ndim);
    } else {
      add_subscript_list(list, ast, arraylist, ssval, ndim);
    }
  } else {
    /* step dimension 'dim' through all of its values */
    int l;
    for (l = sslist[dim]; l; l = A_RIGHTG(l)) {
      ssval[dim] = A_LEFTG(l);
      if (dim > 0) {
        build_subscript_list(list, ast, arraylist, ssval, sslist, dim - 1,
                             ndim);
      } else {
        add_subscript_list(list, ast, arraylist, ssval, ndim);
      }
    }
  }
} /* build_subscript_list */

static void
build_array_list(ASTLIST *list, int ast, DTYPE dtype, int sptr)
{
  int asptr, lop, rop, asd, ndim, i;
  int lower, upper, stride, d, ssval[MAXDIMS], sslist[MAXDIMS];
  ASTLIST larray;
  int fldsptr, past;
  list->head = 0;
  list->tail = 0;
  switch (A_TYPEG(ast)) {
  case A_CNST:
    add_init(list, ast, dtype, 0);
    break;
  case A_MEM: {
    DTYPE dtype;
    int a;
    fldsptr = A_SPTRG(A_MEMG(ast));
    past = A_PARENTG(ast);
    asptr = A_SPTRG(past);
    for (a = A_LEFTG(PARAMVALG(asptr)); a; a = A_RIGHTG(a)) {
      if (A_SPTRG(a) == fldsptr) {
        break;
      }
    }
    if (!a) {
      interr("field initializer not found", 0, 3);
      sem.dinit_error = TRUE;
      break;
    }
    dtype = DDTG(DTYPEG(A_SPTRG(a)));
    for (a = A_LEFTG(a); a; a = A_RIGHTG(a)) {
      add_init(list, A_LEFTG(a), dtype, 0);
    }
  } break;
  case A_ID:
    /* an array name */
    asptr = A_SPTRG(ast);
    switch (STYPEG(asptr)) {
    case ST_ARRAY:
    case ST_IDENT:
    case ST_VAR:
      if (PARAMVALG(asptr)) {
        DTYPE dtype = DDTG(DTYPEG(asptr));
        int a;
        for (a = A_LEFTG(PARAMVALG(asptr)); a; a = A_RIGHTG(a)) {
          add_init(list, A_LEFTG(a), dtype, 0);
        }
      }
      break;

    default:
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    }
    break;
  case A_SUBSCR:
    /* subscripted array */
    build_array_list(&larray, A_LOPG(ast), dtype, sptr);
    /* get the subscript; take the one element, or the
     * sequence of elements requested */
    if (sem.dinit_error)
      break;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    assert(ndim <= 7, "build_array_list, >7 dimensions", ndim, 3);
    assert(A_SHAPEG(A_LOPG(ast)), "build_array_list, shapeless array", 0, 3);
    for (i = 0; i < ndim; ++i) {
      int ss;
      ss = ASD_SUBS(asd, i);
      if (A_SHAPEG(ss) || A_TYPEG(ss) == A_TRIPLE) {
        ASTLIST ssl;
        build_array_list(&ssl, ss, astb.bnd.dtype, 0);
        ssval[i] = 0;
        sslist[i] = ssl.head;
      } else {
        ssval[i] = dinit_getval(ss, astb.bnd.dtype);
        sslist[i] = 0;
      }
    }
    build_subscript_list(list, ast, larray.head, ssval, sslist, ndim - 1, ndim);
    break;
  case A_UNOP:
    /* get the right operand */
    build_array_list(list, A_LOPG(ast), dtype, sptr);
    if (sem.dinit_error)
      break;
    /* negate? */
    switch (A_OPTYPEG(ast)) {
    case OP_SUB:
      /* negate everything on the list */
      unop_init_list(list->head, A_OPTYPEG(ast));
      break;
    case OP_ADD:
      break;
    default:
      errsev(69);
      sem.dinit_error = TRUE;
    }
    break;
  case A_BINOP:
    /* get right operand */
    lop = A_LOPG(ast);
    while (A_TYPEG(lop) == A_CONV)
      lop = A_LOPG(lop);
    rop = A_ROPG(ast);
    while (A_TYPEG(rop) == A_CONV)
      rop = A_LOPG(rop);
    if (A_SHAPEG(lop) && !A_SHAPEG(rop)) {
      build_array_list(list, lop, dtype, sptr);
      if (sem.dinit_error)
        break;
      binop_init_list(list->head, 0, 0, rop, A_OPTYPEG(ast));
    } else if (!A_SHAPEG(lop) && A_SHAPEG(rop)) {
      build_array_list(list, rop, dtype, sptr);
      if (sem.dinit_error)
        break;
      binop_init_list(0, list->head, lop, 0, A_OPTYPEG(ast));
    } else {
      ASTLIST list2;
      build_array_list(list, lop, dtype, sptr);
      if (sem.dinit_error)
        break;
      list2.head = list2.tail = 0;
      build_array_list(&list2, rop, dtype, sptr);
      if (sem.dinit_error)
        break;
      binop_init_list(list->head, list2.head, 0, 0, A_OPTYPEG(ast));
    }
    break;
  case A_CONV:
  case A_PAREN:
    build_array_list(list, A_LOPG(ast), dtype, sptr);
    break;
  case A_TRIPLE:
    /* build a list of items from the triplet */
    lower = dinit_getval(A_LBDG(ast), astb.bnd.dtype);
    upper = dinit_getval(A_UPBDG(ast), astb.bnd.dtype);
    if (A_STRIDEG(ast)) {
      stride = dinit_getval(A_STRIDEG(ast), astb.bnd.dtype);
    } else {
      stride = astb.bnd.one;
    }
    if (lower == 0 || upper == 0 || stride == 0) {
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    }
    lower = A_ALIASG(lower);
    upper = A_ALIASG(upper);
    stride = A_ALIASG(stride);
    if (lower == 0 || upper == 0 || stride == 0) {
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    }
    lower = A_SPTRG(lower);
    upper = A_SPTRG(upper);
    stride = A_SPTRG(stride);
    lower = CONVAL2G(lower);
    upper = CONVAL2G(upper);
    stride = CONVAL2G(stride);
    if (stride == 0) {
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    } else if (stride > 0 && lower > upper) {
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    } else if (stride < 0 && lower < upper) {
      errsev(69);
      sem.dinit_error = TRUE;
      break;
    }
    if (lower <= upper) {
      for (d = lower; d <= upper; d += stride) {
        /* make a constant with value 'd'; add to A_INIT list */
        int a = mk_isz_cval(d, astb.bnd.dtype);
        add_init(list, a, astb.bnd.dtype, 0);
      }
    } else {
      for (d = lower; d >= upper; d += stride) {
        /* make a constant with value 'd'; add to A_INIT list */
        int a = mk_isz_cval(d, astb.bnd.dtype);
        add_init(list, a, astb.bnd.dtype, 0);
      }
    }
    break;
  default:
    errsev(69);
    sem.dinit_error = TRUE;
    break;
  }
} /* build_array_list */

static void
add_array_init(ASTLIST *list, int ast, DTYPE dtype, int sptr)
{
  /* given an array-shaped expression ast, add 'init' items */
  ASTLIST newlist;
  newlist.head = 0;
  newlist.tail = 0;

  build_array_list(&newlist, ast, dtype, sptr);
  if (newlist.head) {
    if (list->head == 0) {
      list->head = newlist.head;
    } else {
      A_RIGHTP(list->tail, newlist.head);
    }
    list->tail = newlist.tail;
  }
} /* add_array_init */

static ACL *
dinit_fill_struct(ASTLIST *list, ACL *aclp, int sdtype, int sptr,
                  int memberlist, int init_single)
{
  int i, idx_sptr, aa, tmpcon;
  ACL *a;
  ACL *b;
  INT initval, limitval, stepval, save_conval1;
  INT num[2];
  ASTLIST newlist = {0, 0};
  if (aclp == NULL)
    return NULL;
#if DEBUG
  if (DBGBIT(3, 64))
    dumpacl("dinit_fill_struct", aclp, gbl.dbgfil);
#endif
  for (a = aclp; a; a = a->next) {
    SST *stkp;
    DOINFO *doinfo;
    int aast, dtype, ddtype, member, count;
    if (memberlist && sptr == 0 && !out_of_elements_message) {
      interr("dinit_fill_struct, out of derived type elements", 0, 0);
      out_of_elements_message = TRUE;
    }
    switch (a->id) {
    case AC_AST:
      dtype = A_DTYPEG(a->u1.ast);
      aast = a->u1.ast;
      if (A_TYPEG(aast) == A_ID && PARAMG(A_SPTRG(aast))) {
        if (PARAMVALG(A_SPTRG(aast))) {
          add_init(list, A_LEFTG(PARAMVALG(A_SPTRG(aast))), dtype, sptr);
        }
      } else {
        aast = dinit_getval(aast, sdtype);
        add_init(list, aast, dtype, sptr);
      }
      break;
    case AC_EXPR:
      /* get the AST */
      stkp = a->u1.stkp;
      dtype = SST_DTYPEG(stkp);
      a->repeatc = a->size = 0;
      aast = SST_ASTG(stkp);
      if (SST_IDG(stkp) == S_ACONST) {
        interr("dinit_fill_struct, unexpected S_ACONST", 0, 3);
        aast = 0;
      } else if (A_TYPEG(aast) == A_INTR || A_TYPEG(aast) == A_BINOP) {
        ACL *iaclp = construct_acl_from_ast(aast, sdtype, 0);
        if (!iaclp) {
          return 0;
        }
        iaclp = eval_init_expr_item(iaclp);
        if (!iaclp) {
          return 0;
        }
        newlist.head = newlist.tail = 0;
        dinit_fill_struct(&newlist, iaclp, sdtype, sptr, memberlist,
                          init_single);
        append_init_list(list, &newlist);
      } else {
        int save;
        aast = SST_ASTG(stkp);
        if (A_SHAPEG(aast) != 0 || A_TYPEG(aast) == A_SUBSCR) {
          save = list->tail;
          add_array_init(list, aast, dtype, sptr);
          if (save) {
            a->repeatc = A_RIGHTG(save);
          } else {
            a->repeatc = list->head;
          }
          a->size = list->tail;
        } else if (A_TYPEG(aast) == A_ID && PARAMVALG(A_SPTRG(aast))) {
          aa = mk_init(PARAMVALG(A_SPTRG(aast)), dtype);
          A_SPTRP(aa, sptr);
          add_init(list, aast, dtype, sptr);
        } else {
          if (DTY(sdtype) == TY_ARRAY) {
            aast = dinit_getval1(aast, DTY(sdtype + 1));
          } else
            aast = dinit_getval1(aast, sdtype);

          if (A_TYPEG(SST_ASTG(stkp)) == A_CNST &&
              A_DTYPEG(aast) != A_DTYPEG(SST_ASTG(stkp))) {
            /* constant initialization value needed type conversion,
             * rewrite the ACL instance to use converted value */
            a->id = AC_AST;
            a->dtype = sdtype;
            a->u1.ast = aast;
          }
          add_init(list, aast, dtype, sptr);
        }
      }
      break;
    case AC_IEXPR:
      if (POINTERG(sptr)) {
        /*  maybe this should always be done  */
        a->sptr = sptr;
      }
      b = eval_init_expr_item(a);
      if (!b) {
        return 0;
      }
      newlist.head = newlist.tail = 0;
      if (POINTERG(sptr)) {
        /*  And, MUST be ST_MEMBER */
        b = dinit_fill_struct(&newlist, b, b->dtype, MIDNUMG(sptr), 1,
                              init_single);
      } else {
        if (DTY(b->dtype) == TY_ARRAY)
          dtype = b->dtype;
        else
          dtype = sdtype;
        b = dinit_fill_struct(&newlist, b, dtype, sptr, 0, init_single);
      }
      append_init_list(list, &newlist);
      break;
    case AC_ACONST:
      dtype = a->dtype;
      if (DTY(dtype) != TY_ARRAY) {
        interr("dinit_fill_struct, expecting ARRAY type", dtype, 1);
        ddtype = dtype;
      } else {
        ddtype = DDTG(sdtype);
      }
      newlist.head = newlist.tail = 0;
      b = dinit_fill_struct(&newlist, a->subc, ddtype, sptr, 0, FALSE);
      if (list && DTY(sdtype) != TY_ARRAY)
        append_init_list(list, &newlist);
      else {
        if (DTY(ddtype) == TY_DERIVED) {
          add_init(list, newlist.head, ddtype, sptr);
        } else
          add_init(list, newlist.head, dtype, sptr);
      }
      break;
    case AC_SCONST:
      dtype = a->dtype;
      if (DTY(dtype) != TY_DERIVED) {
        interr("dinit_fill_struct, expecting DERIVED type", dtype, 1);
        member = 0;
        ddtype = 0;
      } else {
        member = DTY(dtype + 1);
        if (member) {
          ddtype = DTYPEG(member);
          if (no_data_components(ddtype)) {
            member = next_member(member);
            if (member)
              ddtype = DTYPEG(member);
            else
              ddtype = 0;
          }
        } else {
          ddtype = 0;
        }
      }
      newlist.head = newlist.tail = 0;
      b = dinit_fill_struct(&newlist, a->subc, ddtype, member, 1, member != 0);
      add_init(list, newlist.head, dtype, sptr);
      if (sdtype && dtype != sdtype) {
        /* coerce */
        interr("initialization coercion needed", sdtype, 1);
      }
      break;
    case AC_IDO:
      if (sem.top == &sem.dostack[MAX_DOSTACK]) {
        /*  nesting maximum exceeded.  */
        errsev(34);
        return NULL;
      }
      doinfo = a->u1.doinfo;
      ++sem.top;
      newlist.head = newlist.tail = 0;
      idx_sptr = doinfo->index_var;
      initval = dinit_eval(doinfo->init_expr);
      limitval = dinit_eval(doinfo->limit_expr);
      stepval = dinit_eval(doinfo->step_expr);
      save_conval1 = CONVAL1G(idx_sptr);
      if (stepval >= 0) {
        for (i = initval; i <= limitval; i += stepval) {
          switch (DTY(DTYPEG(idx_sptr))) {
          case TY_INT8:
          case TY_LOG8:
            ISZ_2_INT64(i, num);
            tmpcon = getcon(num, DTYPEG(idx_sptr));
            CONVAL1P(idx_sptr, tmpcon);
            break;
          default:
            CONVAL1P(idx_sptr, i);
            break;
          }
          b = dinit_fill_struct(&newlist, a->subc, sdtype, sptr, 0, sptr != 0);
        }
      } else {
        for (i = initval; i >= limitval; i += stepval) {
          switch (DTY(DTYPEG(idx_sptr))) {
          case TY_INT8:
          case TY_LOG8:
            ISZ_2_INT64(i, num);
            tmpcon = getcon(num, DTYPEG(idx_sptr));
            CONVAL1P(idx_sptr, tmpcon);
            break;
          default:
            CONVAL1P(idx_sptr, i);
            break;
          }
          b = dinit_fill_struct(&newlist, a->subc, sdtype, sptr, 0, sptr != 0);
        }
      }
      append_init_list(list, &newlist);
      CONVAL1P(idx_sptr, save_conval1);
      --sem.top;
      break;
    case AC_REPEAT:
      count = a->u1.count;
      while (--count >= 0) {
        b = dinit_fill_struct(list, a->subc, sdtype, sptr, 0, sptr != 0);
      }
      break;
    case AC_CONVAL:
      if (a->conval == 0) {
        aast = a->u1.ast;
      } else if (DT_ISWORD(a->dtype)) {
        aast = mk_cval1(a->conval, a->dtype);
      } else {
        aast = mk_cnst(a->conval);
      }
      dtype = A_DTYPEG(aast);
      aast = dinit_getval(aast, sdtype);
      add_init(list, aast, dtype, sptr);
      break;
    }
    if (memberlist && sptr) {
      /* move 'sptr' along the member list */
      if (STYPEG(sptr) != ST_MEMBER) {
        interr("dinit_fill_struct, expecing member", sptr, 1);
        return a->next;
      }
      sptr = next_member(sptr);
      if (sptr <= NOSYM) {
        return a->next;
      }
      sdtype = DTYPEG(sptr);
    } else if (init_single) {
      /* initializing a single symbol */
      return a->next;
    }
  }
  return NULL;
} /* dinit_fill_struct */

void
dinit_struct_param(SPTR sptr, ACL *sconst, DTYPE dtype)
{
  ASTLIST newlist;
  /* set up 'sptr' as having a parameter value */
  PARAMP(sptr, 1);
  /* put the 'parameter' value in the ASTs */
  out_of_elements_message = FALSE;
  sem.top = &sem.dostack[0];
  newlist.head = newlist.tail = 0;
  dinit_fill_struct(&newlist, sconst, dtype, sptr, 0, sptr != 0);
  PARAMVALP(sptr, newlist.head);
} /* dinit_struct_param */

/** \brief In DATA statement, do the stuff in dinit_struct_const in two steps.
 */
ACL *
dinit_struct_vals(ACL *sconst, DTYPE dtype, SPTR component_sptr)
{
  SST *item_stkp;
  int ast;
  ACL *aclp;
  ACL *ict; /* Initializer Constant Tree */
  ACL *last;
  ACL *first;
  /* need to check for number of entries */
  /* allocate and init an Initializer Constant Tree */
  int count = 0;
  SPTR member_sptr = DTY(dtype + 1);
  SPTR sptr = component_sptr != NOSYM ? component_sptr : DTY(dtype + 3);
  last = NULL;
  for (aclp = sconst->subc; aclp != NULL; aclp = aclp->next) {
    if (aclp->id == AC_ACONST) {
      ict = aclp;
      ict->sptr = member_sptr;
    } else if (aclp->id == AC_SCONST) {
      ict = dinit_struct_vals(aclp, aclp->dtype, member_sptr);
      ict->sptr = member_sptr;
    } else if (aclp->id == AC_EXPR && SST_IDG(aclp->u1.stkp) == S_IDENT &&
               STYPEG(SST_SYMG(aclp->u1.stkp)) == ST_PD &&
               PDNUMG(SST_SYMG(aclp->u1.stkp)) == PD_null) {
      ict = SST_ACLG(aclp->u1.stkp);
    } else {
      item_stkp = aclp->u1.stkp;
      ast = item_stkp->ast;
      if (!ast || (!A_ALIASG(ast) &&
                   (A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) != I_NULL))) {
        int errsptr;
        errsptr = SST_SYMG(item_stkp);
        if (ast == 0 && errsptr) {
          error(155, 3, gbl.lineno,
                "DATA initialization with nonconstant value -",
                SYMNAME(errsptr));
          sem.dinit_error = TRUE;
        } else {
          error(155, 3, gbl.lineno,
                "DATA initialization with nonconstant expression", "");
          sem.dinit_error = TRUE;
          return NULL;
        }
        ict = NULL;
      } else {
        ict = GET_ACL(15);
        ict->id = AC_AST;
        ict->next = NULL;
        ict->subc = NULL;
        ict->u1.ast = SST_ASTG(item_stkp); /* the data constant */
        ict->repeatc = 0;                  /* no repeat count */
        ict->sptr = member_sptr;
        ict->dtype = SST_DTYPEG(item_stkp);
      }
    }
    if (ict != NULL) {
      if (last == NULL)
        first = ict;
      else
        last->next = ict;
      last = ict;
    }
    if (member_sptr != 0)
      member_sptr = SYMLKG(member_sptr);
  }
  ict = GET_ACL(15);
  ict->id = AC_SCONST;
  ict->next = NULL;
  ict->subc = first;
  ict->u1.ast = count;
  ict->repeatc = astb.bnd.one; /* repeat count */
  ict->sptr = sptr;
  ict->dtype = dtype;
  return ict;
}

/** \brief Create an initialization node for a variable reference in a data
   statement.

    If the variable reference is an array section (tpr1652) an implied do is
    generated for each subscript which is a triple.   For example, the array
    section:
    <pre>
        A(i1, L2:U2, L3:U3, i4)
    </pre>
    is transformed into:
    <pre>
        ( ( A(i1, j2, j3, i4) j2 = L2, U2 ), j3 = L3, U3 )
    </pre>
    Each triple subscript is replaced by an implied do index variable, and
    the expressions in the triplet becomes the bounds of the implied do.
    Sections are to be initialized in array element order (i.e., column major).
    An implied do nest is produced by a left to right scan of the subscripts
    (the leftmost triple represents the innermost implied do).

    If the variable reference is a member of a whole array, turn the whole
    array reference into a subscripted reference where each subscript is
    a triple.  Then, the subscripted referenced is handled as described
    above.

    For other variable references, a single initialization node is created.
 */
VAR *
dinit_varref(SST *stkp)
{
  VAR *ivl;
  int ast;
  ITEM *mhd, *p; /* mhd = "member of the whole array" ? */
  int i;
  int ndim;
  int subs[MAXDIMS];

  mhd = NULL;
  for (ast = SST_ASTG(stkp); A_TYPEG(ast) == A_MEM; ast = A_PARENTG(ast)) {
    p = (ITEM *)getitem(0, sizeof(ITEM));
    p->next = mhd;
    p->ast = ast;
    mhd = p;
  }
  if (mhd && A_TYPEG(ast) == A_ID && DTY(A_DTYPEG(ast)) == TY_ARRAY) {
    int ss;
    ADSC *ad;
    ss = A_SPTRG(ast);
    ad = AD_DPTR(DTYPEG(ss));
    ndim = AD_NUMDIM(ad);
    i = 0;
    while (i < ndim) {
      subs[i] = mk_triple(AD_LWAST(ad, i), AD_UPAST(ad, i), 0);
      i++;
    }
    ast = mk_subscr(ast, subs, ndim, A_DTYPEG(ast));
  }
  if (A_TYPEG(ast) == A_SUBSCR) {
    /*
     * the variable reference is subscripted; check if any of the subcripts
     * are triples.
     */
    int asd;
    int triple[MAXDIMS];
    LOGICAL any_triple;
    int newast;

    any_triple = FALSE;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      /*
       * If a subscript in dimension 'i' (zero-based) is a triple:
       * 1.  save the ast of the triple in triple[i].
       * 2.  create an integer variable which will be the implied do
       *     index in dimension 'i'.
       * 3.  create the ast of the do variable which will be the
       *     subscript in dimension 'i' and save in subs[i].
       *
       * Otherwise, triple[i] is set to 0 (subscript in the dimension
       * 'i' is not a triple).
       */
      subs[i] = ASD_SUBS(asd, i);
      if (A_TYPEG(subs[i]) == A_TRIPLE) {
        any_triple = TRUE;
        triple[i] = subs[i];
        subs[i] = mk_id(get_temp(astb.bnd.dtype));
      } else
        triple[i] = 0;
    }
    if (any_triple) {
      VAR *newivl;
      VAR *endl;
      /*
       * Create a subscripted reference, where the triples are replaced
       * by their respective index variables; the other subscripts
       * are used as is.  This subscripted reference becomes the object
       * in a variable reference initialization node.
       */
      newast = mk_subscr(A_LOPG(ast), subs, ndim, DTY(A_DTYPEG(ast) + 1));
      for (p = mhd; p != NULL; p = p->next) {
        newast = mk_member(newast, A_MEMG(p->ast), A_DTYPEG(p->ast));
      }
      ivl = (VAR *)getitem(15, sizeof(VAR));
      ivl->id = Varref;
      ivl->u.varref.ptr = newast;
      ivl->u.varref.id = S_LVALUE;
      ivl->u.varref.dtype = A_DTYPEG(newast);
      ivl->u.varref.shape = A_SHAPEG(newast);
      ivl->u.varref.subt = NULL;
      ivl->next = NULL;
      if (SCG(SST_LSYMG(stkp)) == SC_BASED) {
        error(116, 3, gbl.lineno, SYMNAME(SST_LSYMG(stkp)), "(DATA)");
        sem.dinit_error = TRUE;
      }
      /* keep track of the 'end' (outer) init. node; note that 'ivl'
       * represents the current init. node
       */
      endl = ivl;
      for (i = 0; i < ndim; i++) {
        if (triple[i]) {
          /* build a doend element for the dinit var list */
          newivl = (VAR *)getitem(15, sizeof(VAR));
          endl->next = newivl; /* current -> Doend */
          newivl->id = Doend;
          newivl->next = NULL;
          endl = newivl; /* end of this do is the Doend */
                         /*
                          * Create the dostart element, link it to the doend element,
                          * and link all in the order dostart -> current node ->
                          * doend.
                          */
          newivl->u.doend.dostart = (VAR *)getitem(15, sizeof(VAR));
          newivl = newivl->u.doend.dostart;
          newivl->id = Dostart;
          newivl->u.dostart.indvar = subs[i];
          newivl->u.dostart.lowbd = A_LBDG(triple[i]);
          newivl->u.dostart.upbd = A_UPBDG(triple[i]);
          newivl->u.dostart.step = A_STRIDEG(triple[i]);
          newivl->next = ivl; /* Dostart -> current */
          ivl = newivl;       /* Dostart is the new current node */
        }
      }
      SST_VLBEGP(stkp, ivl);  /* Dostart of the outermost implied do*/
      SST_VLENDP(stkp, endl); /* Doend of the outermost implied do*/
      sem.dinit_data = TRUE;
      return NULL; /* tell semant that a section was initialized */
    }
  }
  /* build a single element for the dinit var list */
  ivl = (VAR *)getitem(15, sizeof(VAR));
  ivl->id = Varref;
  ivl->u.varref.ptr = SST_ASTG(stkp);
  ivl->u.varref.id = SST_IDG(stkp);
  ivl->u.varref.dtype = SST_DTYPEG(stkp);
  ivl->u.varref.shape = SST_SHAPEG(stkp);
  ivl->u.varref.subt = NULL;
  ivl->next = NULL;
  return ivl;
}

/** \brief Get a compiler temporary of any scalar dtype.
 */
SPTR
get_temp(DTYPE dtype)
{
  SPTR sptr;
  DTYPE dt;
#if DEBUG
  assert(DT_ISSCALAR(dtype) || DTY(dtype) == TY_DERIVED,
         "get_temp:nonscalar dt", dtype, 3);
#endif
  if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR)
    return get_ch_temp(dtype);

  if (!sem.temps_reset) {
    BZERO(temps_ctr, char, sizeof(temps_ctr));
    sem.temps_reset = TRUE;
  }

  do {
    sptr = getcctmp_sc('i', TEMPS_CTR(0), ST_VAR, dtype, sem.sc);
    dt = DTYPEG(sptr);
  } while (dt != dtype);

  return sptr;
}

DTYPE
get_temp_dtype(DTYPE dtype, int expr)
{
  if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR || dtype == DT_DEFERCHAR ||
      dtype == DT_DEFERNCHAR) {
    int len;
    if (A_TYPEG(expr) == A_INTR && A_OPTYPEG(expr) == I_TRIM)
      len = ast_intr(I_LEN_TRIM, astb.bnd.dtype, 1, ARGT_ARG(A_ARGSG(expr), 0));
    else {
      len = ast_intr(I_LEN, astb.bnd.dtype, 1, expr);
    }
    dtype = get_type(2, DTY(dtype), len);
  }
  return dtype;
}

SPTR
get_itemp(DTYPE dtype)
{
  SPTR sptr = getccsym_sc('i', sem.itemps++, ST_VAR, sem.sc);
  DTYPEP(sptr, dtype);
  return sptr;
}

static void
allocate_temp(SPTR sptr)
{
  DTYPE dtype;
  int subs[MAXDIMS], i, n, ast;

  add_p_dealloc_item(sptr);

  dtype = DTYPEG(sptr);
  ast = mk_id(sptr);
  /* char length variable? */
  if (DTYG(dtype) == TY_CHAR || DTYG(dtype) == TY_NCHAR) {
    int cvlen, len, rhs, asn, dty;
    dty = DDTG(dtype);
    cvlen = CVLENG(sptr);
    if (cvlen) {
      len = mk_id(cvlen);
      rhs = DTY(dty + 1);
      rhs = mk_convert(rhs, DTYPEG(cvlen));
      rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));
      asn = mk_assn_stmt(len, rhs, DTYPEG(cvlen));
      (void)add_stmt(asn);
    }
  }
  if (DTY(dtype) == TY_ARRAY) {
    ADD_DEFER(dtype) = 1;
    /* insert allocate statement */
    n = ADD_NUMDIM(dtype);
    for (i = 0; i < n; ++i) {
      subs[i] = mk_triple(ADD_LWBD(dtype, i), ADD_UPBD(dtype, i), 0);
    }
    ast = mk_subscr(ast, subs, n, dtype);
  }
  gen_alloc_dealloc(TK_ALLOCATE, ast, 0);
} /* allocate_temp */

/** \brief Get a compiler array temporary of type dtype.
 */
SPTR
get_arr_temp(DTYPE dtype, LOGICAL nodesc, LOGICAL alloc_deferred,
             LOGICAL constructor)
{
  SPTR sptr;
  int needalloc;
  SC_KIND sc = sem.sc;
  DTYPE dt = DTY(dtype + 1);

  if (DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR)
    return get_ch_temp(dtype);
  if (!sem.temps_reset) {
    BZERO(temps_ctr, char, sizeof(temps_ctr));
    sem.temps_reset = TRUE;
  }

  /*
   * Examine dtype to determine if an allocatable temp is needed:
   * o  has deferred shape, or
   * o  the size is not constant.
   *
   * If an allocatable temp is needed, its storage class is always
   * SC_LOCAL or SC_PRIVATE.
   */
  needalloc = 0;
  if (ADD_DEFER(dtype)) {
    needalloc = 1;
  } else {
    int d;
    /* if the size is not constant, mark it as adjustable */
    for (d = 0; d < ADD_NUMDIM(dtype); ++d) {
      int lb, ub;
      lb = ADD_LWBD(dtype, d);
      if (lb && A_ALIASG(lb) == 0) {
        needalloc = 1;
        break;
      }
      ub = ADD_UPBD(dtype, d);
      if (ub && A_ALIASG(ub) == 0) {
        needalloc = 1;
        break;
      }
    }
  }
  if (needalloc && sc != SC_PRIVATE)
    sc = SC_LOCAL;

  do {
    int tmpc;
    if (!needalloc)
      tmpc = TEMPS_CTR(1);
    else
      tmpc = TEMPS_STK(1);
    if (constructor)
      /* Creating a temporary for an array constructor within an OpenACC region.
       * Mark this by using letter 'x' in the name of the temporary so that it
       * can be identified by the accelerator backend.
       * Caution: Any change to this naming scheme must also be reflected in
       * routine add_implicit_private in accel.c.
       */
      sptr = getcctmp_sc('x', tmpc, ST_ARRAY, dtype, sc);
    else
      sptr = getcctmp_sc('a', tmpc, ST_ARRAY, dtype, sc);
    dt = DTYPEG(sptr);
    if (DTY(dt + 1) == DTY(dtype + 1) && ADD_DEFER(dtype) == ADD_DEFER(dt) &&
        nodesc == NODESCG(sptr) && conformable(dt, dtype))
      break;
  } while (dt != dtype);

  if (needalloc) {
    ALLOCP(sptr, 1);
    if (!alloc_deferred && ADD_DEFER(dtype)) {
      /* if deferred shape, temp will be treated as allocatable */
      ;
    } else if (ALLOCATE_ARRAYS) {
      int d;
      /* if the size is not constant, allocate it, but
       * first ensure that each dimension has a lower bound.
       */
      for (d = 0; d < ADD_NUMDIM(dtype); ++d) {
        if (ADD_LWBD(dtype, d) == 0)
          ADD_LWBD(dtype, d) = astb.bnd.one;
      }
      allocate_temp(sptr);
    }
  }
  NODESCP(sptr, nodesc);
  return sptr;
}

/** \brief Get a compiler-created allocatable array temp to represent the
           result of run-time function computing the adjustl/adjustr intrinsic.

    The result of the run-time is the length (which we don't actually use), but
    it's needed to effect array/forall processing in the compiler.  Eventually,
    in outconv.c, the temp is discarded, as well as the return value of the
    runtime routine.
 */
SPTR
get_adjlr_arr_temp(DTYPE dtype)
{
  SPTR sptr;
  ALLOCATE_ARRAYS = 0; /* no need to generate an allocate of the temp*/
  sptr = get_arr_temp(dtype, TRUE, FALSE, FALSE);
  ALLOCATE_ARRAYS = 1;
  return sptr;
}

/** \brief Get a compiler array temporary of from a shape of an ast.
 */
SPTR
get_shape_arr_temp(int arg)
{
  int shape = A_SHAPEG(arg);
  DTYPE dtype = get_shape_arraydtype(shape, DTY(A_DTYPEG(arg) + 1));
  SPTR tmp = get_arr_temp(dtype, FALSE, FALSE, FALSE);
  if (sem.arrdim.ndefer)
    gen_allocate_array(tmp);
  return tmp;
}

/** \brief Get a character compiler temporary of type dtype.
 */
SPTR
get_ch_temp(DTYPE dtype)
{
  SPTR sptr;
  DTYPE dt;
  SYMTYPE stype;
  int len;
  bool needalloc = false;
  SC_KIND sc = sem.sc;

  if (!sem.temps_reset) {
    BZERO(temps_ctr, char, sizeof(temps_ctr));
    sem.temps_reset = TRUE;
  }

  /*
   * Examine dtype to determine if an allocatable temp is needed:
   * o  the length is not a constant, or
   * o  if array, the size is not constant.
   *
   * If an allocatable temp is needed, its storage class is always
   * SC_LOCAL.
   */
  dt = DDTG(dtype);
  /* This is pretty bogus, _INF_CLEN for temps, ugh. */
  if (dt == DT_ASSCHAR || dt == DT_DEFERCHAR) {
    dt = get_type(2, TY_CHAR, mk_cval(_INF_CLEN, DT_INT4));
    error(310, 2, gbl.lineno,
          "Unsafe fixed-length string temporary*500 being used", CNULL);
  } else if (dt == DT_ASSNCHAR || dt == DT_DEFERNCHAR) {
    dt = get_type(2, TY_NCHAR, mk_cval(_INF_CLEN, DT_INT4));
    error(310, 2, gbl.lineno,
          "Unsafe fixed-length string temporary*500 being used", CNULL);
  }

  /* if the length is not a constant, make it 'adjustable' */
  len = DTY(dt + 1);
  if (A_ALIASG(len) == 0) {
    /* will fill in CVLEN field */
    needalloc = true;
  }
  stype = ST_VAR;
  if (DTY(dtype) == TY_ARRAY) {
    int d;
    /* if the size is not constant, mark it as adjustable */
    stype = ST_ARRAY;
    for (d = 0; d < ADD_NUMDIM(dtype); ++d) {
      int lb, ub;
      lb = ADD_LWBD(dtype, d);
      if (lb && A_ALIASG(lb) == 0) {
        needalloc = true;
        break;
      }
      ub = ADD_UPBD(dtype, d);
      if (ub && A_ALIASG(ub) == 0) {
        needalloc = true;
        break;
      }
    }
  }
  if (needalloc)
    sc = SC_LOCAL;

  do {
    int tmpc;
    if (!needalloc)
      tmpc = TEMPS_CTR(1);
    else
      tmpc = TEMPS_STK(1);
    sptr = getcctmp_sc('s', tmpc, stype, dtype, sc);
    dt = DTYPEG(sptr);
  } while (dt != dtype);

  if (needalloc) {
    int clen;
    ALLOCP(sptr, 1);
    /* if the length is not a constant, make it 'adjustable' */
    if (sem.gcvlen && is_deferlenchar_dtype(dtype)) {
      clen = ast_intr(I_LEN, astb.bnd.dtype, 1, mk_id(sptr));
    } else if (A_ALIASG(len) == 0) {
      /* fill in CVLEN field */
      ADJLENP(sptr, 1);
      if (CVLENG(sptr) == 0) {
        clen = sym_get_scalar(SYMNAME(sptr), "len", astb.bnd.dtype);
        CVLENP(sptr, clen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(clen, 1);
      }
    }
    if (DTY(dtype) == TY_ARRAY) {
      if (ALLOCATE_ARRAYS) {
        int d;
        /* if the size is not constant, allocate it, but need to
         * first ensure that each dimension has a lower bound.
         */
        for (d = 0; d < ADD_NUMDIM(dtype); ++d) {
          if (ADD_LWBD(dtype, d) == 0)
            ADD_LWBD(dtype, d) = astb.bnd.one;
        }
        if (!sem.arrdim.ndefer || ADJLENG(sptr))
          allocate_temp(sptr);
      }
    } else {
      allocate_temp(sptr);
    }
  }
  return sptr;
}

int
need_alloc_ch_temp(DTYPE dtype)
{
  if (sem.use_etmps) {
    /*
     * if the dtype warrants an allocatable temp, need to add a fake
     * etmp entry so that its expression context, such as a relational
     * expression, is fully evaluated and assigned to a temp.
     */
    if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR || dtype == DT_DEFERCHAR ||
        dtype == DT_DEFERNCHAR || !A_ALIASG(DTY(dtype + 1))) {
      add_etmp(0);
      return 1;
    }
  }
  return 0;
}

/** \brief Compare \a str and \a pattern like strcmp() but ignoring the case of
   str.
           \a pattern is all lower case.
 */
int
sem_strcmp(const char *str, const char *pattern)
{
  const char *p1, *p2;
  int ch;

  p1 = str;
  p2 = pattern;
  do {
    ch = *p1;
    if (ch >= 'A' && ch <= 'Z')
      ch += ('a' - 'A'); /* to lower case */
    if (ch != *p2)
      return (ch - *p2);
    if (ch == '\0')
      return 0;
    p1++;
    p2++;
  } while (1);
}

/** \brief Return TRUE if fortran character constant is equal to pattern
           (pattern is always uppercase).
  */
LOGICAL
sem_eq_str(int con, const char *pattern)
{
  char *p1;
  const char *p2;
  int len;
  int c1, c2;

  p1 = stb.n_base + CONVAL1G(con);
  p2 = pattern;
  for (len = string_length(DTYPEG(con)); len > 0; len--) {
    c1 = *p1;
    if (c1 >= 'a' && c1 <= 'z') /* convert to upper case */
      c1 = c1 + ('A' - 'a');
    c2 = *p2;
    if (c2 == '\0' || c1 != c2)
      break;
    p1++;
    p2++;
  }

  if (len == 0)
    return TRUE;

  /*  verify that remaining characters of con are blank:  */
  while (len--)
    if (*p1++ != ' ')
      return FALSE;
  return TRUE;
}

/** \brief Allocate a temporary, assign it the value, and return the assignment
 * ast.
 */
int
sem_tempify(SST *stkptr)
{
  int argtyp;
  SST tmpsst;
  int tmpsym;
  int assn;
  argtyp = SST_DTYPEG(stkptr);
  argtyp = get_temp_dtype(argtyp, SST_ASTG(stkptr));
  if (DTY(argtyp) != TY_ARRAY) {
    tmpsym = get_temp(argtyp);
  } else {
    tmpsym = get_arr_temp(argtyp, FALSE, A_SHAPEG(SST_ASTG(stkptr)), FALSE);
  }
  mkident(&tmpsst);
  SST_SYMP(&tmpsst, tmpsym);
  SST_DTYPEP(&tmpsst, argtyp);
  assn = assign(&tmpsst, stkptr);
  return assn;
}

/** \brief Update the SWEL list for a `SELECTCASE` construct represented by
           the \a doif structure.

    A new SWEL item is created for a case value or a range of case
    values denoted by the arguments \a lc and \a uc.  The order of the items in
    the list will correspond to the case values in ascending order.

     Kind of case   |   lc   |  uc
     ---------------|--------|------
     case (:c)      |   c    |  -1
     case (c)       |   c    |  0       (c is a sym pointer)
     case (c:)      |   c    |  1
     case (c:d)     |   c    |  d       (c and d are sym pointers)
 */
void
add_case_range(int doif, int lc, int uc)
{
  SWEL *swel;
  int ni;
  int bef;
  int i;
  int (*p_cmp)(int, int);

  ni = sem.switch_avl++; /* relative ptr to new SWEL item */
  NEED(sem.switch_avl, switch_base, SWEL, sem.switch_size,
       sem.switch_size + 300);

  /* The first SWEL item's next field locates the head of the list */
  bef = DI_SWEL_HD(doif);
  if (DT_ISLOG(DI_DTYPE(doif))) {
    for (i = switch_base[bef].next; i != 0; i = switch_base[i].next) {
      if (switch_base[i].val == lc)
        goto dup_error;
    }
    switch_base[ni].val = lc;
    switch_base[ni].next = switch_base[bef].next;
    switch_base[bef].next = ni;
    return;
  }
  if (DI_DTYPE(doif) == DT_INT8)
    p_cmp = _i8_cmp;
  else if (DT_ISINT(DI_DTYPE(doif)))
    p_cmp = _i4_cmp;
  else {
/* character */
    if (DTY(DI_DTYPE(doif)) == TY_NCHAR)
      p_cmp = _nchar_cmp;
    else
      p_cmp = _char_cmp;
  }

  for (i = switch_base[bef].next; i != 0; i = switch_base[i].next) {
    swel = switch_base + i;
    if ((*p_cmp)(lc, swel->val) < 0) {
      /*  lc < current case value 'val' */
      if (swel->uval == -1)
        /* (lc) in (:val) */
        goto range_error;
      if (uc == 1)
        /* (lc :) in (val ...) */
        goto range_error;
      if (uc > 1 && (*p_cmp)(uc, swel->val) >= 0)
        /* (lc:uc), lc < val, uc >= val */
        goto range_error;
      break;
    }
    if ((*p_cmp)(lc, swel->val) == 0) {
      /* lc == current case value */
      if (uc == 0 && swel->uval == 0)
        goto dup_error;
      goto range_error;
    }

    /*  lc > current case value */
    if (uc == -1)
      /* lc > val, (:lc) specified */
      goto range_error;
    if (swel->uval == 1)
      /* lc in (val:) */
      goto range_error;
    if (swel->uval > 1) {
      if ((*p_cmp)(lc, swel->uval) <= 0)
        /* lc in (val:uval) */
        goto range_error;
    }
    bef = i;
  }

  /* insert new swel item into list */
  switch_base[ni].val = lc;
  switch_base[ni].uval = uc;
  switch_base[ni].next = switch_base[bef].next;
  switch_base[bef].next = ni;
  return;

dup_error:
  error(310, 3, gbl.lineno, "Duplicate case value", CNULL);
  sem.switch_avl--;
  return;

range_error:
  error(310, 3, gbl.lineno, "Overlapping case value", CNULL);
  sem.switch_avl--;
}

/** \brief Compare functions whose arguments are pointers to ST_CONST
           symbol table entries.
    \return a number less than, equal to, or greater than 0, depending on the
   comparison
 */
int
_i4_cmp(int l, int r)
{
  INT v1, v2;

  v1 = CONVAL2G(l);
  v2 = CONVAL2G(r);
  if (v1 < v2)
    return -1;
  if (v1 == v2)
    return 0;
  return 1;
}

int
_i8_cmp(int l, int r)
{
  DBLINT64 v1, v2;

  v1[0] = CONVAL1G(l);
  v1[1] = CONVAL2G(l);
  v2[0] = CONVAL1G(r);
  v2[1] = CONVAL2G(r);
  return cmp64(v1, v2);
}

int
_char_cmp(int l, int r)
{
  char *v1, *v2;

  v1 = stb.n_base + CONVAL1G(l);
  v2 = stb.n_base + CONVAL1G(r);
  return strcmp(v1, v2);
}

int
_nchar_cmp(int l, int r)
{
#define KANJI_BLANK 0xA1A1
  int bytes, val1, val2;
  int cvlen1, cvlen2;
  char *p, *q;

  cvlen1 = string_length(DTYPEG(l));
  cvlen2 = string_length(DTYPEG(r));
  p = stb.n_base + CONVAL1G(l);
  q = stb.n_base + CONVAL1G(r);

  while (cvlen1 > 0 && cvlen2 > 0) {
    val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
    p += bytes, cvlen1 -= bytes;
    val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
    q += bytes, cvlen2 -= bytes;
    if (val1 != val2)
      return (val1 - val2);
  }

  while (cvlen1 > 0) {
    val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
    p += bytes, cvlen1 -= bytes;
    if (val1 != KANJI_BLANK)
      return (val1 - KANJI_BLANK);
  }

  while (cvlen2 > 0) {
    val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
    q += bytes, cvlen2 -= bytes;
    if (val2 != KANJI_BLANK)
      return (KANJI_BLANK - val2);
  }
  return 0;
}

/** \brief Check if we are currently in a block FORALL scope;
           if so, issue an error message.
*/
LOGICAL
not_in_forall(const char *stmttype)
{
  if (sem.doif_depth > 0 && DI_ID(sem.doif_depth) == DI_FORALL) {
    error(441, 3, gbl.lineno, stmttype, CNULL);
    return TRUE;
  }
  return FALSE;
} /* not_in_forall */

/** \brief If we are accepting cuda syntax return TRUE.
          Otherwise issue an error message and return FALSE.
 */
LOGICAL
cuda_enabled(const char *at_or_near)
{
  error(34, 3, gbl.lineno, at_or_near, CNULL);
  return FALSE;
} /* cuda_enabled */

LOGICAL
in_device_code(int sptr)
{
  return FALSE;
}

static void
add_to_list(ACL *val, ACL **root)
{
  ACL *tail;
  if (*root) {
    for (tail = *root; tail->next; tail = tail->next)
      ;
    tail->next = val;
  } else {
    *root = val;
  }
}

static ACL *
clone_init_const(ACL *original, int temp)
{
  ACL *clone;

  if (!original)
    return NULL;
  clone = GET_ACL(15);
  *clone = *original;
  if (clone->subc) {
    clone_init_const_list(clone->subc, temp);
  }
  if (clone->id == AC_IEXPR) {
    if (clone->u1.expr->lop) {
      clone_init_const_list(clone->u1.expr->lop, temp);
    }
    if (clone->u1.expr->rop) {
      clone_init_const_list(clone->u1.expr->rop, temp);
    }
  }
  clone->next = NULL;
  return clone;
}

static ACL *
clone_init_const_list(ACL *original, int temp)
{
  ACL *clone;

  clone = clone_init_const(original, temp);
  for (original = original->next; original; original = original->next) {
    add_to_list(clone_init_const(original, temp), &clone);
  }
  return clone;
}

static INT
get_int_from_init_conval(ACL *aclp)
{
  INT ret;

  if (DT_ISWORD(aclp->dtype)) {
    ret = aclp->conval;
  } else {
    ret = CONVAL2G(aclp->conval);
  }
  return ret;
}

/* Intrinsic evaluation routines for data initialization
 *  Stolen from semfunc.c and hacked to generate ACL initialization values.
 */
static ACL *
eval_ishft(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  ACL *arg2;
  INT val;
  INT conval;
  INT res[4];
  INT shftval;

  arg = eval_init_expr(arg);
  rslt = clone_init_const(arg, TRUE);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  arg2 = arg->next;
  shftval = get_int_from_init_conval(arg2);
  if (shftval > bits_in(wrkarg->dtype)) {
    error(4, 3, gbl.lineno, "ISHFT SHIFT argument too big for I argument\n",
          NULL);
    return 0;
  }

  for (; wrkarg; wrkarg = wrkarg->next) {
    val = get_int_from_init_conval(wrkarg);
    switch (size_of(wrkarg->dtype)) {
    case 2:
      val = get_int_from_init_conval(wrkarg);
      if (shftval >= 0) {
        if (shftval >= 16)
          conval = 0;
        else {
          conval = ULSHIFT(val, shftval);
          conval = ULSHIFT(conval, 16);
          conval = ARSHIFT(conval, 16);
        }
      } else {
        if (shftval <= -16)
          conval = 0;
        else {
          val &= 0xffff;
          conval = URSHIFT(val, -shftval);
        }
      }
      conval = cngcon(conval, DT_WORD, DDTG(dtype));
      break;
    case 4:
      /*
       * because this ilm is used for the ISHFT intrinsic, count
       * is defined for values -32 to 32; some hw (i.e., n10) shifts
       * by cnt mod 32.
       */
      val = get_int_from_init_conval(wrkarg);
      if (shftval >= 0) {
        if (shftval >= 32)
          conval = 0;
        else
          conval = ULSHIFT(val, shftval);
      } else {
        if (shftval <= -32)
          conval = 0;
        else
          conval = URSHIFT(val, -shftval);
      }
      conval = cngcon(conval, DT_WORD, DDTG(dtype));

      break;
    case 8:
      /* val and shftval are symbol pointers */
      /* get the value for shftval */
      res[0] = CONVAL1G(wrkarg->conval);
      res[1] = CONVAL2G(wrkarg->conval);
      if (shftval >= 0) {
        if (shftval >= 64) {
          res[0] = 0;
          res[1] = 0;
        } else if (shftval >= 32) {
          /* shift val by 32 bits or more */
          res[0] = ULSHIFT(res[1], shftval - 32);
          res[1] = 0;
        } else {
          /* shift by less than 32 bits; shift high-order
           * bits of low-order word into high-order word */
          res[0] = ULSHIFT(res[0], shftval) | URSHIFT(res[1], 32 - shftval);
          res[1] = ULSHIFT(res[1], shftval);
        }
      } else {
        shftval = -shftval;
        if (shftval >= 64) {
          res[0] = 0;
          res[1] = 0;
        } else if (shftval >= 32) {
          /* shift val by 32 bits or more */
          res[1] = URSHIFT(res[0], shftval - 32);
          res[0] = 0;
        } else {
          /* shift by less than 32 bits; shift low-order
           * bits of high-order word into low-order word */
          res[1] = URSHIFT(res[1], shftval) | ULSHIFT(res[0], 32 - shftval);
          res[0] = URSHIFT(res[0], shftval);
        }
      }
      conval = getcon(res, DT_INT8);

      break;
    }
    wrkarg->id = AC_CONVAL;
    wrkarg->conval = conval;
    wrkarg->dtype = dtype;
  }

  return rslt;
}

#define INTINTRIN2(iname, ent, op)                               \
  static ACL *ent(ACL *arg, DTYPE dtype)                         \
  {                                                              \
    ACL *arg1 = eval_init_expr_item(arg);                        \
    ACL *arg2 = eval_init_expr_item(arg->next);                  \
    ACL *rslt = clone_init_const(arg1, TRUE);                    \
    arg1 = rslt->id == AC_ACONST ? rslt->subc : rslt;            \
    arg2 = arg2->id == AC_ACONST ? arg2->subc : arg2;            \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {         \
      int con1 = arg1->conval;                                   \
      int con2 = arg2->conval;                                   \
      int num1[2], num2[2], res[2], conval;                      \
      if (DT_ISWORD(arg1->dtype)) {                              \
        num1[0] = 0, num1[1] = con1;                             \
      } else {                                                   \
        num1[0] = CONVAL1G(con1), num1[1] = CONVAL2G(con1);      \
      }                                                          \
      if (DT_ISWORD(arg2->dtype)) {                              \
        num2[0] = 0, num2[1] = con2;                             \
      } else {                                                   \
        num2[0] = CONVAL1G(con2), num2[1] = CONVAL2G(con2);      \
      }                                                          \
      res[0] = num1[0] op num2[0];                               \
      res[1] = num1[1] op num2[1];                               \
      conval = DT_ISWORD(dtype) ? res[1] : getcon(res, DT_INT8); \
      arg1->conval = conval;                                     \
      arg1->dtype = dtype;                                       \
    }                                                            \
    return rslt;                                                 \
  }

INTINTRIN2("iand", eval_iand, &)
INTINTRIN2("ior", eval_ior, |)
INTINTRIN2("ieor", eval_ieor, ^)

static ACL *
eval_ichar(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  int srcdty;
  int rsltdtype = DDTG(dtype);
  int clen;
  INT c;
  int dum;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  srcdty = DTY(wrkarg->dtype);
  for (; wrkarg; wrkarg = wrkarg->next) {
    if (srcdty == TY_NCHAR) {
      c = CONVAL1G(wrkarg->conval);
      clen = size_of(DTYPEG(c));
      c = kanji_char((unsigned char *)stb.n_base + CONVAL1G(c), clen, &dum);
    } else {
      c = stb.n_base[CONVAL1G(wrkarg->conval)] & 0xff;
    }
    if (DTY(rsltdtype) == TY_INT8) {
      INT res[4];
      INT conval;
      res[0] = 0;
      res[1] = c;
      conval = getcon(res, DT_INT8);
      dtype = DT_INT8;
      wrkarg->conval = A_SPTRG(mk_cval1(conval, dtype));
    } else {
      wrkarg->conval = c;
    }
    wrkarg->id = AC_CONVAL;
    wrkarg->dtype = rsltdtype;
  }
  if (rslt->id == AC_ACONST) {
    rslt->dtype = dup_array_dtype(arg->dtype);
    DTY(rslt->dtype + 1) = dtype;
  } else
    rslt->dtype = dtype;
  return rslt;
}

static ACL *
eval_char(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  char c;
  int sptr;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    c = get_int_from_init_conval(wrkarg);
    sptr = getstring(&c, 1);
    wrkarg->dtype = dtype;
    wrkarg->conval = sptr;
    wrkarg->u1.ast = mk_cnst(sptr);
  }
  return rslt;
}

static ACL *
eval_int(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    wrkarg->conval = cngcon(wrkarg->conval, wrkarg->dtype, DDTG(dtype));
    wrkarg->dtype = dtype;
  }
  return rslt;
}

static ACL *
eval_fltconvert(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  int rsltdtype = DDTG(dtype);

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    wrkarg->conval = cngcon(wrkarg->conval, wrkarg->dtype, rsltdtype);
    wrkarg->dtype = rsltdtype;
  }
  return rslt;
}

#define GET_DBLE(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y)
#define GET_QUAD(x, y) \
  x[0] = CONVAL1G(y);  \
  x[1] = CONVAL2G(y);  \
  x[2] = CONVAL3G(y);  \
  x[3] = CONVAL4G(y);
#define GETVALI64(x, b) \
  x[0] = CONVAL1G(b);   \
  x[1] = CONVAL2G(b);

static ACL *
eval_abs(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  INT con1, res[4], num1[4], num2[4];
  DTYPE rsltdtype = dtype;
  float f1, f2;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    switch (DTY(wrkarg->dtype)) {
    case TY_SINT:
    case TY_BINT:
    case TY_INT:
      con1 = wrkarg->conval;
      con1 = con1 >= 0 ? con1 : -con1;
      break;
    case TY_INT8:
      con1 = wrkarg->conval; /* sptr */
      GETVALI64(num1, con1);
      GETVALI64(num2, stb.k0);
      if (cmp64(num1, num2) == -1) {
        neg64(num1, res);
        con1 = getcon(res, DT_INT8);
      }
      break;
    case TY_REAL:
      con1 = wrkarg->conval;
      res[0] = 0;
      xfabsv(con1, &res[1]);
      con1 = res[1];
      break;
    case TY_DBLE:
      con1 = wrkarg->conval;
      GET_DBLE(num1, con1);
      xdabsv(num1, res);
      con1 = getcon(res, dtype);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      con1 = wrkarg->conval;
      GET_QUAD(num1, con1);
      xqabsv(num1, res);
      con1 = getcon(res, dtype);
      break;
#endif
    case TY_CMPLX:
      con1 = wrkarg->conval;
      f1 = CONVAL1G(con1);
      f2 = CONVAL2G(con1);
      f1 = f1 * f1;
      f2 = f2 * f2;
      f2 = f1 + f2;
      xfsqrt(f2, &con1);
      dtype = rsltdtype = DT_REAL;
      wrkarg->dtype = dtype;
      break;
    case TY_DCMPLX:
      con1 = wrkarg->conval;
      rsltdtype = DT_REAL;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      con1 = wrkarg->conval;
      GET_QUAD(num1, CONVAL1G(con1));
      GET_QUAD(num2, CONVAL2G(con1));
      xqmul(num1, num1, num1);
      xqmul(num2, num2, num2);
      xqadd(num1, num2, num2);
      xqsqrt(num2, num1);
      con1 = getcon(num1, dtype);
      dtype = rsltdtype = DT_QUAD;
      wrkarg->dtype = dtype;
      break;
#endif
    default:
      con1 = wrkarg->conval;
      break;
    }

    wrkarg->conval = cngcon(con1, wrkarg->dtype, rsltdtype);
    wrkarg->dtype = dtype;
  }
  return rslt;
}

/* scale(X, I) = X * 2 **I, X is real type, I is an integer */
static ACL *
eval_scale(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *arg2;
  INT i, conval1, conval2, conval;
  DBLINT64 inum1, inum2;
  INT e;
#ifdef TARGET_SUPPORTS_QUADFP
  INT qnum1[4], qnum2[4];
  QUAD qconval;
#endif
  DBLE dconval;

  rslt = arg = eval_init_expr(arg);
  conval1 = arg->conval;
  arg2 = arg->next;

  if (arg2->dtype == DT_INT8)
    error(205, ERR_Warning, gbl.lineno, SYMNAME(arg2->conval),
          "- Illegal specification of scale factor");

  i = arg2->dtype == DT_INT8 ? CONVAL2G(arg2->conval) : arg2->conval;

  switch (size_of(arg->dtype)) {
  case 4:
    /* 8-bit exponent (127) to get an exponent value in the
     * range -126 .. +127 */
    e = 127 + i;
    if (e < 0)
      e = 0;
    else if (e > 255)
      e = 255;

    /* calculate decimal value from it's IEEE 754 form*/
    conval2 = e << 23;
    xfmul(conval1, conval2, &conval);
    rslt->conval = conval;
    break;

  case 8:
    e = 1023 + i;
    if (e < 0)
      e = 0;
    else if (e > 2047)
      e = 2047;

    inum1[0] = CONVAL1G(conval1);
    inum1[1] = CONVAL2G(conval1);

    inum2[0] = e << 20;
    inum2[1] = 0;
    xdmul(inum1, inum2, dconval);
    rslt->conval = getcon(dconval, DT_REAL8);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case 16:
    e = 16383 + i;
    if (e < 0)
      e = 0;
    else if (e > 32767)
      e = 32767;

    qnum1[0] = CONVAL1G(conval1);
    qnum1[1] = CONVAL2G(conval1);
    qnum1[2] = CONVAL3G(conval1);
    qnum1[3] = CONVAL4G(conval1);

    qnum2[0] = e << 16;
    qnum2[1] = 0;
    qnum2[2] = 0;
    qnum2[3] = 0;
    xqmul(qnum1, qnum2, qconval);
    rslt->conval = getcon(qconval, DT_QUAD);
    break;
#endif
  }

  return rslt;
}

static ACL *
eval_merge(ACL *arg, DTYPE dtype)
{
  ACL *tsource = eval_init_expr_item(arg);
  ACL *fsource = eval_init_expr_item(arg->next);
  ACL *mask = eval_init_expr_item(arg->next->next);
  ACL *result = clone_init_const(tsource, TRUE);
  ACL *r = result;
  if (tsource->id == AC_ACONST)
    tsource = tsource->subc;
  if (fsource->id == AC_ACONST)
    fsource = fsource->subc;
  if (mask->id == AC_ACONST)
    mask = mask->subc;
  if (r->id == AC_ACONST)
    r = r->subc;
  for (; r != 0; r = r->next) {
    int cond = DT_ISWORD(mask->dtype) ? mask->conval : CONVAL2G(mask->conval);
    r->conval = cond ? tsource->conval : fsource->conval;
    r->dtype = dtype;
    tsource = tsource->next;
    fsource = fsource->next;
    mask = mask->next;
  }
  return result;
}

/* Compare two constant ACLs. Return x > y or x < y depending on want_max.
 */
static bool
cmp_acl(DTYPE dtype, ACL *x, ACL *y, bool want_max, bool back)
{
  int cmp;
  switch (DTY(dtype)) {
  case TY_CHAR:
    cmp = strcmp(stb.n_base + CONVAL1G(x->conval),
                 stb.n_base + CONVAL1G(y->conval));
    break;
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    if (x->conval == y->conval) {
      cmp = 0;
    } else if (x->conval > y->conval) {
      cmp = 1;
    } else {
      cmp = -1;
    }
    break;
  case TY_REAL:
    cmp = xfcmp(x->conval, y->conval);
    break;
  case TY_INT8:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
    cmp = const_fold(OP_CMP, x->conval, y->conval, dtype);
    break;
  default:
    interr("cmp_acl: bad dtype", dtype, ERR_Severe);
    return false;
  }
  if (back) {
    return want_max ? cmp >= 0 : cmp <= 0;
  } else {
    return want_max ? cmp > 0 : cmp < 0;
  }
}

/* An index into a Fortran array. ndims is in [1,MAXDIMS], index[] is the
 * index itself, extent[] is the extent in each dimension.
 * index[i] is in [1,extent[i]] for i in 1..ndims
 */
typedef struct {
  unsigned ndims;
  unsigned index[MAXDIMS + 1];
  unsigned extent[MAXDIMS + 1];
} INDEX;

/* Increment an array index starting at the left and carrying to the right. */
static bool
incr_index(INDEX *index)
{
  unsigned d;
  for (d = 1; d <= index->ndims; ++d) {
    if (index->index[d] < index->extent[d]) {
      index->index[d] += 1;
      return true; /* no carry needed */
    }
    index->index[d] = 1;
  }
  return false;
}

static unsigned
get_offset_without_dim(INDEX *index, unsigned dim)
{
  if (dim == 0) {
    return 0;
  } else {
    unsigned result = 0;
    unsigned d;
    for (d = index->ndims; d > 0; --d) {
      if (d != dim) {
        result *= index->extent[d];
        result += index->index[d] - 1;
      }
    }
    return result;
  }
}

/* Create an array dtype from the extents in index, omitting dimension dim. */
static DTYPE
mk_dtype_without_dim(INDEX *index, unsigned dim, DTYPE elem_dtype)
{
  DTYPE array_dtype;
  unsigned i, j;
  for (i = 1, j = 0; i <= index->ndims; ++i) {
    if (i != dim) {
      sem.bounds[j].lowtype = S_CONST;
      sem.bounds[j].lowb = 1;
      sem.bounds[j].lwast = 0;
      sem.bounds[j].uptype = S_CONST;
      sem.bounds[j].upb = index->extent[i];
      sem.bounds[j].upast = mk_cval(index->extent[i], stb.user.dt_int);
      ++j;
    }
  }
  sem.arrdim.ndim = index->ndims - 1;
  sem.arrdim.ndefer = 0;
  array_dtype = mk_arrdsc();
  DTY(array_dtype + 1) = elem_dtype;
  return array_dtype;
}

/* Get an ACL representing the smallest/largest value of this type. */
static ACL *
get_minmax_val(DTYPE dtype, bool want_max)
{
  int ast = want_max ? mk_smallest_val(dtype) : mk_largest_val(dtype);
  return eval_init_expr_item(construct_acl_from_ast(ast, dtype, 0));
}

static ACL *
convert_acl_dtype(ACL *head, int oldtype, int newtype)
{
  DTYPE dtype;
  ACL *cur_lop;
  if (DTY(oldtype) == TY_DERIVED || DTY(oldtype) == TY_STRUCT ||
      DTY(oldtype) == TY_CHAR || DTY(oldtype) == TY_NCHAR ||
      DTY(oldtype) == TY_UNION) {
    return head;
  }
  dtype = DDTG(newtype);

  /* make sure all are AC_CONST */
  for (cur_lop = head; cur_lop; cur_lop = cur_lop->next) {
    if (cur_lop->id != AC_CONST)
      return head;
  }

  for (cur_lop = head; cur_lop; cur_lop = cur_lop->next) {
    if (cur_lop->dtype != dtype) {
      cur_lop->dtype = dtype;
      cur_lop->conval = cngcon(cur_lop->conval, DDTG(oldtype), DDTG(newtype));
    }
  }
  return head;
}

/* Evaluate {min,max}{val,loc}{elems, dim, mask, back).
 * index describes the shape of the array; elem_dt the type of elems.
 */
static ACL *
do_eval_minval_or_maxval(INDEX *index, DTYPE elem_dt, DTYPE loc_dt, ACL *elems,
                         unsigned dim, ACL *mask, bool back,
                         AC_INTRINSIC intrin)
{
  unsigned ndims = index->ndims;
  unsigned i;
  ACL **vals;
  unsigned *locs;
  unsigned vals_size = 1;
  unsigned locs_size;
  bool want_max = intrin == AC_I_maxloc || intrin == AC_I_maxval;
  bool want_val = intrin == AC_I_minval || intrin == AC_I_maxval;

  /* vals[vals_size] contains the result for {min,max}val()
   * locs[locs_size] contains the result for {min,max}loc() */
  if (dim == 0) {
    locs_size = ndims;
  } else {
    unsigned d;
    for (d = 1; d <= ndims; ++d) {
      if (d != dim)
        vals_size *= index->extent[d];
    }
    locs_size = vals_size;
  }
  NEW(vals, ACL *, vals_size);
  for (i = 0; i < vals_size; ++i) {
    vals[i] = get_minmax_val(elem_dt, want_max);
  }

  NEW(locs, unsigned, locs_size);
  BZERO(locs, unsigned, locs_size);

  { /* iterate over elements computing min/max into vals[] and locs[] */
    ACL *elem;
    for (elem = elems; elem != 0; elem = elem->next) {
      if (elem->dtype != elem_dt) {
        elem = convert_acl_dtype(elem, elem->dtype, elem_dt);
      }

      if (mask->conval) {
        ACL *val = eval_init_expr_item(elem);
        unsigned offset = get_offset_without_dim(index, dim);
        ACL *prev_val = vals[offset];
        if (cmp_acl(elem_dt, val, prev_val, want_max, back)) {
          vals[offset] = val;
          if (dim == 0) {
            BCOPY(locs, &index->index[1], int, ndims);
          } else {
            locs[offset] = index->index[dim];
          }
        }
      }
      if (mask->next)
        mask = mask->next;
      incr_index(index);
    }
  }

  { /* build result from vals[] or locs[] */
    ACL *result;
    ACL *subc = NULL; /* elements of result array */
    if (!want_val) {
      for (i = 0; i < locs_size; i++) {
        ACL *elem = GET_ACL(15);
        BZERO(elem, ACL, 1);
        elem->id = AC_CONST;
        elem->dtype = loc_dt;
        elem->is_const = true;
        elem->conval = locs[i];
        elem->u1.ast = mk_cval(locs[i], loc_dt);
        add_to_list(elem, &subc);
      }
    } else if (dim > 0) {
      for (i = 0; i < vals_size; i++) {
        add_to_list(vals[i], &subc);
      }
    } else {
      return vals[0]; /* minval/maxval with no dim has scalar result */
    }

    result = GET_ACL(15);
    BZERO(result, ACL, 1);
    result->id = AC_ACONST;
    result->dtype =
        mk_dtype_without_dim(index, dim, want_val ? elem_dt : loc_dt);
    result->is_const = 1;
    result->subc = subc;
    return result;
  }
}

static ACL *
eval_minval_or_maxval(ACL *arg, DTYPE dtype, AC_INTRINSIC intrin)
{
  DTYPE elem_dt = array_element_dtype(dtype);
  DTYPE loc_dtype = DT_INT;
  ACL *array = eval_init_expr_item(arg);
  unsigned dim = 0; /* 0 means no DIM specified, otherwise in 1..ndims */
  ACL *mask = 0;
  INDEX index;
  unsigned d;
  ACL *arg2;
  bool back = FALSE;

  while ((arg = arg->next)) {
    if (DT_ISLOG(arg->dtype)) { /* back */
      arg2 = eval_init_expr_item(arg);
      back = arg2->conval;
    } else if (DT_ISINT(arg->dtype)) { /* dim */
      arg2 = eval_init_expr_item(arg);
      dim = arg2->conval;
      assert(dim == arg2->conval, "DIM value must be an integer!", 0,
             ERR_Fatal);
    } else { //(DT_ISLOG_ARR(arg->dtype))
      mask = eval_init_expr_item(arg);
      if (mask != 0 && mask->id == AC_ACONST)
        mask = mask->subc;
    }
  }

  if (mask == 0) {
    /* mask defaults to .true. */
    mask = GET_ACL(15);
    BZERO(mask, ACL, 1);
    mask->id = AC_CONST;
    mask->dtype = DT_LOG;
    mask->is_const = 1;
    mask->conval = 1;
    mask->u1.ast = mk_cval(gbl.ftn_true, DT_LOG);
  }
  /* index contains the rank and extents of the array dtype */
  BZERO(&index, INDEX, 1);
  index.ndims = ADD_NUMDIM(dtype);
  for (d = 1; d <= index.ndims; ++d) {
    int lwbd = get_int_cval(A_SPTRG(ADD_LWAST(dtype, d - 1)));
    int upbd = get_int_cval(A_SPTRG(ADD_UPAST(dtype, d - 1)));
    int extent = upbd - lwbd + 1;
    index.extent[d] = extent;
    index.index[d] = 1;
  }
  return do_eval_minval_or_maxval(&index, elem_dt, loc_dtype, array->subc, dim,
                                  mask, back, intrin);
}

/* evaluate min or max, depending on want_max flag */
static ACL *
eval_min_or_max(ACL *arg, DTYPE dtype, LOGICAL want_max)
{
  ACL *rslt;
  ACL *wrkarg1, *head, *c;
  ACL **arglist;
  int nargs;
  int nelems = 0;
  int i, j, repeatc1, repeatc2;
  ADSC *adsc;
  ACL *root = NULL;

  /* at this point we only know argument types but we don't know the
   * lhs of min(...) type
   * Therefore, create a result based on the result of args.
   */

  rslt = GET_ACL(15);
  BZERO(rslt, ACL, 1);
  rslt->dtype = arg->dtype;

  for (wrkarg1 = arg, nargs = 0; wrkarg1; wrkarg1 = wrkarg1->next, nargs++)
    ;

  NEW(arglist, ACL *, nargs);
  for (i = 0, wrkarg1 = arg; i < nargs; i++, wrkarg1 = wrkarg1->next) {
    head = arglist[i] = eval_init_expr(wrkarg1);
    if (DTY(head->dtype) == TY_ARRAY) {
      int num;
      adsc = AD_DPTR(head->dtype);
      num = get_int_cval(A_SPTRG(AD_NUMELM(adsc)));
      if (nelems == 0) {
        nelems = num;
      } else if (nelems != num) {
        /* error */
      }
      rslt->id = AC_ACONST;
      rslt->dtype = head->dtype;
    }
  }
  if (nelems == 0) {
    nelems = 1; /* scalar only */
    c = rslt;
    c->id = AC_CONST;
    c->repeatc = astb.bnd.one;
    c->next = NULL;
    add_to_list(c, &root);
  } else {
    for (j = 0; j < nelems; j++) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->repeatc = astb.bnd.one;
      c->next = NULL;
      add_to_list(c, &root);
    }
    rslt->subc = root;
    rslt->repeatc = 0;
  }

  wrkarg1 = arglist[0];
  for (i = 1; i < nargs; i++) {
    ACL *wrkarg2 = arglist[i];
    if (wrkarg2->id == AC_ACONST) {
      wrkarg2 = wrkarg2->subc;
      if (wrkarg2->repeatc)
        repeatc2 = get_int_cval(A_SPTRG(wrkarg2->repeatc));
      else
        repeatc2 = 1;
    } else {
      repeatc2 = nelems;
    }
    if (wrkarg1->id == AC_ACONST) {
      wrkarg1 = wrkarg1->subc;
      if (wrkarg1->repeatc)
        repeatc1 = get_int_cval(A_SPTRG(wrkarg1->repeatc));
      else
        repeatc1 = 1;
    } else {
      repeatc1 = nelems;
    }

    c = root;
    for (j = 0; j < nelems; j++) {
      if (cmp_acl(dtype, wrkarg2, wrkarg1, want_max, FALSE)) {
        c->u1 = wrkarg2->u1;
        c->conval = wrkarg2->conval;
        c->dtype = wrkarg2->dtype;
      } else if (root != wrkarg1) {
        c->u1 = wrkarg1->u1;
        c->conval = wrkarg1->conval;
        c->dtype = wrkarg1->dtype;
      }
      if (--repeatc2 <= 0) {
        wrkarg2 = wrkarg2->next;
        if (wrkarg2 && wrkarg2->repeatc)
          repeatc2 = get_int_cval(A_SPTRG(wrkarg2->repeatc));
        else
          repeatc2 = 1;
      }
      c = c->next;
      if (wrkarg1 == root) { /* result becomes argument on next
                              * iteration of outer loop
                              */
        wrkarg1 = c;
        repeatc1 = 1;
      } else if (--repeatc1 <= 0) {
        wrkarg1 = wrkarg1->next;
        if (wrkarg2 && wrkarg2->repeatc)
          repeatc2 = get_int_cval(A_SPTRG(wrkarg2->repeatc));
        else
          repeatc2 = 1;
      }
    }
    wrkarg1 = c = root;
  }
  return rslt;
}

static ACL *
eval_nint(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  int conval;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT res[4];
    INT con1;
    DTYPE dtype1 = wrkarg->dtype;

    switch (DTY(dtype1)) {
    case TY_REAL:
      con1 = wrkarg->conval;
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) >= 0) {
        INT fv2_23;
        xffloat(1 << 23, &fv2_23);
        if (xfcmp(con1, fv2_23) >= 0)
          xfadd(con1, CONVAL2G(stb.flt0), &res[0]);
        else
          xfadd(con1, CONVAL2G(stb.flthalf), &res[0]);
      } else {
        INT fvm2_23;
        xffloat(-(1 << 23), &fvm2_23);
        if (xfcmp(con1, fvm2_23) <= 0)
          xfsub(con1, CONVAL2G(stb.flt0), &res[0]);
        else
          xfsub(con1, CONVAL2G(stb.flthalf), &res[0]);
      }
      break;
    case TY_DBLE:
      con1 = wrkarg->conval;
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
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      con1 = wrkarg->conval;
      if (const_fold(OP_CMP, con1, stb.quad0, DT_QUAD) >= 0) {
        INT qv2_112[4] = {MAX_MANTI_BIT0_31, MAX_MANTI_BIT32_63, MAX_MANTI_BIT64_95, MAX_MANTI_BIT96_127};
        INT q2_112;
        q2_112 = getcon(qv2_112, DT_QUAD);
        if (const_fold(OP_CMP, con1, q2_112, DT_QUAD) >= 0)
          res[0] = const_fold(OP_ADD, con1, stb.quad0, DT_QUAD);
        else
          res[0] = const_fold(OP_ADD, con1, stb.quadhalf, DT_QUAD);
      } else {
        INT qvm2_112[4] = {MMAX_MANTI_BIT0_31, MMAX_MANTI_BIT32_63, MMAX_MANTI_BIT64_95, MMAX_MANTI_BIT96_127};
        INT qm2_112;
        qm2_112 = getcon(qvm2_112, DT_QUAD);
        if (const_fold(OP_CMP, con1, qm2_112, DT_QUAD) <= 0)
          res[0] = const_fold(OP_SUB, con1, stb.quadhalf, DT_QUAD);
        else
          res[0] = const_fold(OP_SUB, con1, stb.quad0, DT_QUAD);
      }
      break;
#endif
    }
    conval = cngcon(res[0], dtype1, dtype);
    wrkarg->dtype = dtype;
    wrkarg->conval = conval;
  }
  return rslt;
}

static ACL *
eval_floor(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  int conval;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT con1;
    int adjust;

    adjust = 0;
    con1 = wrkarg->conval;
    switch (DTY(wrkarg->dtype)) {
    case TY_REAL:
      conval = cngcon(con1, DT_REAL4, dtype);
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) < 0) {
        con1 = cngcon(conval, dtype, DT_REAL4);
        if (xfcmp(con1, wrkarg->conval) != 0)
          adjust = 1;
      }
      break;
    case TY_DBLE:
      conval = cngcon(con1, DT_REAL8, dtype);
      if (const_fold(OP_CMP, con1, stb.dbl0, DT_REAL8) < 0) {
        con1 = cngcon(conval, dtype, DT_REAL8);
        if (const_fold(OP_CMP, con1, wrkarg->conval, DT_REAL8) != 0)
          adjust = 1;
      }
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      conval = cngcon(con1, DT_QUAD, dtype);
      if (const_fold(OP_CMP, con1, stb.quad0, DT_QUAD) < 0) {
        con1 = cngcon(conval, dtype, DT_QUAD);
        if (const_fold(OP_CMP, con1, wrkarg->conval, DT_QUAD) != 0)
          adjust = 1;
      }
      break;
#endif
    }
    if (adjust) {
      if (DT_ISWORD(dtype))
        conval--;
      else {
        num1[0] = 0;
        num1[1] = 1;
        con1 = getcon(num1, dtype);
        conval = const_fold(OP_SUB, conval, con1, dtype);
      }
    }
    wrkarg->conval = conval;
    wrkarg->dtype = dtype;
  }
  return rslt;
}

static ACL *
eval_ceiling(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  int conval;

  rslt = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT con1;
    int adjust;

    adjust = 0;
    con1 = wrkarg->conval;
    switch (DTY(wrkarg->dtype)) {
    case TY_REAL:
      conval = cngcon(con1, DT_REAL4, dtype);
      num1[0] = CONVAL2G(stb.flt0);
      if (xfcmp(con1, num1[0]) > 0) {
        con1 = cngcon(conval, dtype, DT_REAL4);
        if (xfcmp(con1, wrkarg->conval) != 0)
          adjust = 1;
      }
      break;
    case TY_DBLE:
      conval = cngcon(con1, DT_REAL8, dtype);
      if (const_fold(OP_CMP, con1, stb.dbl0, DT_REAL8) > 0) {
        con1 = cngcon(conval, dtype, DT_REAL8);
        if (const_fold(OP_CMP, con1, wrkarg->conval, DT_REAL8) != 0)
          adjust = 1;
      }
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      conval = cngcon(con1, DT_QUAD, dtype);
      if (const_fold(OP_CMP, con1, stb.quad0, DT_QUAD) > 0) {
        con1 = cngcon(conval, dtype, DT_QUAD);
        if (const_fold(OP_CMP, con1, wrkarg->conval, DT_QUAD) != 0)
          adjust = 1;
      }
      break;
#endif
    }
    if (adjust) {
      if (DT_ISWORD(dtype))
        conval++;
      else {
        num1[0] = 0;
        num1[1] = 1;
        con1 = getcon(num1, dtype);
        conval = const_fold(OP_ADD, conval, con1, dtype);
      }
    }
    wrkarg->conval = conval;
    wrkarg->dtype = dtype;
  }
  return rslt;
}

static ACL *
eval_mod(ACL *arg, DTYPE dtype)
{
  ACL *rslt, *arg1, *arg2;
  int conval1, conval2, conval3;

  rslt = arg = eval_init_expr(arg);
  arg1 = arg->id == AC_ACONST ? arg->subc : arg;
  arg2 = arg->next->id == AC_ACONST ? arg->next->subc : arg->next;
  arg->next = 0;
  dtype = DDTG(dtype);
  for (; arg1; arg1 = arg1->next) {
    /*  mod(a,p) == a-int(a/p)*p  */
    conval1 = cngcon(arg1->conval, arg1->dtype, dtype);
    conval2 = cngcon(arg2->conval, arg2->dtype, dtype);
    conval3 = const_fold(OP_DIV, conval1, conval2, dtype);
    conval3 = cngcon(conval3, dtype, DT_INT8);
    conval3 = cngcon(conval3, DT_INT8, dtype);
    conval3 = const_fold(OP_MUL, conval3, conval2, dtype);
    conval3 = const_fold(OP_SUB, conval1, conval3, dtype);
    arg1->conval = conval3;
    arg1->dtype = dtype;
    if (arg2->next)
      arg2 = arg2->next;
  }
  return rslt;
}

static ACL *
eval_repeat(ACL *arg, DTYPE dtype)
{
  ACL *rslt = NULL;
  ACL *arg1;
  ACL *arg2;
  int i, j, cvlen, newlen;
  INT ncopies;
  char *p, *cp, *str;

  arg = eval_init_expr(arg);
  arg1 = arg;
  arg2 = arg->next;
  ncopies = get_int_from_init_conval(arg2);
  newlen = size_of(dtype);
  cvlen = size_of(arg1->dtype);

  NEW(str, char, newlen);
  cp = str;
  j = ncopies;
  while (j-- > 0) {
    i = cvlen;
    p = stb.n_base + CONVAL1G(arg1->conval);
    while (i-- > 0)
      *cp++ = *p++;
  }

  rslt = GET_ACL(15);
  rslt->id = AC_CONVAL;
  rslt->dtype = dtype;
  rslt->repeatc = astb.i1;
  rslt->conval = getstring(str, newlen);

  FREE(str);
  return rslt;
}

/* Store the value 'conval' of type 'dtype' into 'destination'. */
static void
transfer_store(INT conval, DTYPE dtype, char *destination)
{
  int *dest = (int *)destination;
  INT real, imag;

  if (DT_ISWORD(dtype)) {
    dest[0] = conval;
    return;
  }

  switch (DTY(dtype)) {
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
    dest[0] = CONVAL2G(conval);
    dest[1] = CONVAL1G(conval);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    dest[0] = CONVAL4G(conval);
    dest[1] = CONVAL3G(conval);
    dest[2] = CONVAL2G(conval);
    dest[3] = CONVAL1G(conval);
    break;
#endif

  case TY_CMPLX:
    dest[0] = CONVAL1G(conval);
    dest[1] = CONVAL2G(conval);
    break;

  case TY_DCMPLX:
    real = CONVAL1G(conval);
    imag = CONVAL2G(conval);
    dest[0] = CONVAL2G(real);
    dest[1] = CONVAL1G(real);
    dest[2] = CONVAL2G(imag);
    dest[3] = CONVAL1G(imag);
    break;

  case TY_CHAR:
    memcpy(dest, stb.n_base + CONVAL1G(conval), size_of(dtype));
    break;

  default:
    interr("transfer_store: unexpected dtype", dtype, 3);
  }
}

/* Get a value of type 'dtype' from buffer 'source'. */
static INT
transfer_load(DTYPE dtype, char *source)
{
  int *src = (int *)source;
  INT num[4], real[2], imag[2];

  if (DT_ISWORD(dtype))
    return src[0];

  switch (DTY(dtype)) {
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
    num[1] = src[0];
    num[0] = src[1];
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    num[3] = src[0];
    num[2] = src[1];
    num[1] = src[2];
    num[0] = src[3];
    break;
#endif

  case TY_CMPLX:
    num[0] = src[0];
    num[1] = src[1];
    break;

  case TY_DCMPLX:
    real[1] = src[0];
    real[0] = src[1];
    imag[1] = src[2];
    imag[0] = src[3];
    num[0] = getcon(real, DT_REAL8);
    num[1] = getcon(imag, DT_REAL8);
    break;

  case TY_CHAR:
    return getstring(source, size_of(dtype));

  default:
    interr("transfer_load: unexpected dtype", dtype, 3);
  }

  return getcon(num, dtype);
}

static ACL *
eval_transfer(ACL *arg, DTYPE dtype)
{
  ACL *src;
  ACL *rslt;
  int ssize, sdtype, rsize, rdtype;
  int need, avail;
  char value[256];
  char *buffer = value;
  char *bp;
  INT pad;

  arg = eval_init_expr(arg);
  src = clone_init_const(arg, TRUE);
  /* Find the type and size of the source and result. */
  sdtype = DDTG(arg->dtype);
  ssize = size_of(sdtype);
  rdtype = DDTG(dtype);
  rsize = size_of(rdtype);

  /* Be sure we have enough space. */
  need = (rsize > ssize ? rsize : ssize) * 2;
  if (sizeof(value) < need) {
    NEW(buffer, char, need);
    return 0;
  }

  /* Get a pad value in case we have to fill. */
  if (DTY(sdtype) == TY_CHAR)
    memset(buffer, ' ', ssize);
  else
    BZERO(buffer, char, ssize);
  pad = transfer_load(sdtype, buffer);

  src->next = 0;
  if (DTY(src->dtype) == TY_ARRAY)
    src = src->subc;
  bp = buffer;
  avail = 0;
  if (DTY(dtype) != TY_ARRAY) {
    /* Result is scalar. */
    while (avail < rsize) {
      if (src) {
        transfer_store(src->conval, sdtype, bp);
        src = src->next;
      } else
        transfer_store(pad, sdtype, bp);
      bp += ssize;
      avail += ssize;
    }
    rslt = GET_ACL(15);
    rslt->id = AC_CONVAL;
    rslt->dtype = rdtype;
    rslt->conval = transfer_load(rdtype, buffer);
  } else {
    /* Result is array. */
    ACL *root, **current;
    ISZ_T i, nelem;
    int j;

    nelem = get_const_from_ast(ADD_NUMELM(dtype));
    root = NULL;
    current = &root;
    for (i = 0; i < nelem; i++) {
      while (avail < rsize) {
        if (src) {
          transfer_store(src->conval, sdtype, bp);
          src = src->next;
        } else
          transfer_store(pad, sdtype, bp);
        bp += ssize;
        avail += ssize;
      }
      rslt = GET_ACL(15);
      rslt->id = AC_CONVAL;
      rslt->dtype = rdtype;
      rslt->conval = transfer_load(rdtype, buffer);
      *current = rslt;
      current = &(rslt->next);
      bp -= rsize;
      avail -= rsize;
      for (j = 0; j < avail; j++)
        buffer[j] = buffer[rsize + j];
    }
    rslt = GET_ACL(15);
    rslt->id = AC_ACONST;
    rslt->dtype = dtype;
    rslt->subc = root;
  }

  if (buffer != value)
    FREE(buffer);
  return rslt;
}

static ACL *
eval_len_trim(ACL *arg)
{
  ACL *rslt;
  ACL *wrkarg;
  char *p;
  int cvlen, result;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p = stb.n_base + CONVAL1G(wrkarg->conval);
    result = cvlen = size_of(wrkarg->dtype);
    p += cvlen - 1;
    /* skip trailing blanks */
    while (cvlen-- > 0) {
      if (*p-- != ' ')
        break;
      result--;
    }
    wrkarg->dtype = stb.user.dt_int;
    rslt->conval = get_default_int_val(result);
  }
  return rslt;
}

static ACL *
eval_selected_real_kind(ACL *arg)
{
  ACL *rslt;
  ACL *wrkarg;
  int r;
  INT con;

  r = 4;

  wrkarg = arg = eval_init_expr(arg);
  con = get_int_from_init_conval(wrkarg);
  if (con <= 6)
    r = 4;
  else if (con <= 15)
    r = 8;
#ifdef TARGET_SUPPORTS_QUADFP
  else if (con <= MAX_EXP_OF_QMANTISSA)
    r = REAL_16;
#endif
  else
    r = -1;

  if (arg->next) {
    wrkarg = arg = arg->next;
    con = get_int_from_init_conval(wrkarg);
    if (con <= 37) {
      if (r > 0 && r < 4)
        r = 4;
    } else if (con <= 307) {
      if (r > 0 && r < 8)
        r = 8;
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (con <= MAX_EXP_QVALUE) {
      if (r > REAL_0 && r < REAL_16)
        r = REAL_16;
#endif
    } else {
      if (r > 0)
        r = 0;
      r -= 2;
    }
  }

  if (arg->next) {
    wrkarg = arg->next;
    con = get_int_from_init_conval(wrkarg);
    if (con != RADIX2) {
      if (con == NOT_GET_VAL && !ARG_STK(KEYWD_ARGS2)) {
      } else {
        r = NO_REAL;
      }
    }
  }

  rslt = GET_ACL(15);
  rslt->id = AC_CONVAL;
  rslt->dtype = stb.user.dt_int;
  rslt->repeatc = astb.i1;
  rslt->conval = get_default_int_val(r);

  return rslt;
}

static ACL *
eval_selected_int_kind(ACL *arg)
{
  ACL *rslt;
  int r;
  INT con;

  rslt = eval_init_expr(arg);
  con = get_int_from_init_conval(rslt);
  if (con > 18 || (con > 9 && XBIT(57, 2)))
    r = -1;
  else if (con > 9)
    r = 8;
  else if (con > 4)
    r = 4;
  else if (con > 2)
    r = 2;
  else
    r = 1;
  rslt->id = AC_CONVAL;
  rslt->dtype = stb.user.dt_int;
  rslt->repeatc = astb.i1;
  rslt->conval = get_default_int_val(r);

  return rslt;
}

static ACL *
eval_selected_char_kind(ACL *arg)
{
  ACL *rslt;
  int r;

  rslt = eval_init_expr(arg);
  r = _selected_char_kind(rslt->conval);
  rslt->id = AC_CONVAL;
  rslt->dtype = stb.user.dt_int;
  rslt->repeatc = astb.i1;
  rslt->conval = get_default_int_val(r);

  return rslt;
}

static ACL *
eval_scan(ACL *arg)
{
  ACL *rslt = NULL;
  ACL *c;
  ACL *wrkarg;
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;
  INT back = 0;

  arg = eval_init_expr(arg);
  p_set = stb.n_base + CONVAL1G(arg->next->conval);
  l_set = size_of(arg->next->dtype);

  if (arg->next->next) {
    back = get_int_from_init_conval(arg->next->next);
  }

  wrkarg = clone_init_const(arg, TRUE);
  wrkarg = (wrkarg->id == AC_ACONST ? wrkarg->subc : wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p_string = stb.n_base + CONVAL1G(wrkarg->conval);
    l_string = size_of(wrkarg->dtype);

    c = GET_ACL(15);
    c->id = AC_CONVAL;
    c->dtype = stb.dt_int;
    c->repeatc = wrkarg->repeatc;

    if (back == 0) {
      for (i = 0; i < l_string; ++i)
        for (j = 0; j < l_set; ++j)
          if (p_set[j] == p_string[i]) {
            c->conval = i + 1;
            goto addtolist;
          }
    } else {
      for (i = l_string - 1; i >= 0; --i)
        for (j = 0; j < l_set; ++j)
          if (p_set[j] == p_string[i]) {
            c->conval = i + 1;
            goto addtolist;
          }
    }
    c->conval = 0;

  addtolist:
    add_to_list(c, &rslt);
  }
  rslt->repeatc = arg->repeatc;
  return rslt;
}

static ACL *
eval_verify(ACL *arg)
{
  ACL *rslt = NULL;
  ACL *c;
  ACL *wrkarg;
  int i, j;
  int l_string, l_set;
  char *p_string, *p_set;
  INT back = 0;

  arg = eval_init_expr(arg);
  p_set = stb.n_base + CONVAL1G(arg->next->conval);
  l_set = size_of(arg->next->dtype);

  if (arg->next->next) {
    back = get_int_from_init_conval(arg->next->next);
  }

  wrkarg = clone_init_const(arg, TRUE);
  wrkarg = (wrkarg->id == AC_ACONST ? wrkarg->subc : wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p_string = stb.n_base + CONVAL1G(wrkarg->u1.ast);
    l_string = size_of(wrkarg->dtype);

    c = GET_ACL(15);
    c->id = AC_CONVAL;
    c->dtype = stb.dt_int;
    c->conval = 0;
    c->repeatc = wrkarg->repeatc;

    if (back == 0) {
      for (i = 0; i < l_string; ++i) {
        for (j = 0; j < l_set; ++j) {
          if (p_set[j] == p_string[i])
            goto contf;
        }
        c->conval = i + 1;
        break;
      contf:;
      }
    } else {
      for (i = l_string - 1; i >= 0; --i) {
        for (j = 0; j < l_set; ++j) {
          if (p_set[j] == p_string[i])
            goto contb;
        }
        c->conval = i + 1;
        break;
      contb:;
      }
    }

    add_to_list(c, &rslt);
  }
  rslt->repeatc = arg->repeatc;
  return rslt;
}

static ACL *
eval_index(ACL *arg)
{
  ACL *rslt = NULL;
  ACL *c;
  ACL *wrkarg;
  int i, n;
  int l_string, l_substring;
  char *p_string, *p_substring;
  INT back = 0;

  arg = eval_init_expr(arg);
  p_substring = stb.n_base + CONVAL1G(arg->next->conval);
  l_substring = size_of(arg->next->dtype);

  if (arg->next->next) {
    back = get_int_from_init_conval(arg->next->next);
  }

  wrkarg = clone_init_const(arg, TRUE);
  wrkarg = (wrkarg->id == AC_ACONST ? wrkarg->subc : wrkarg);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p_string = stb.n_base + CONVAL1G(wrkarg->conval);
    l_string = size_of(wrkarg->dtype);

    c = GET_ACL(15);
    c->id = AC_CONST;
    c->dtype = stb.dt_int;
    c->repeatc = wrkarg->repeatc;

    n = l_string - l_substring;
    if (n < 0)
      c->conval = 0;
    if (back == 0) {
      if (l_substring == 0)
        c->conval = 0;
      for (i = 0; i <= n; ++i) {
        if (p_string[i] == p_substring[0] &&
            strncmp(p_string + i, p_substring, l_substring) == 0)
          c->conval = i + 1;
      }
    } else {
      if (l_substring == 0)
        c->conval = l_string + 1;
      for (i = n; i >= 0; --i) {
        if (p_string[i] == p_substring[0] &&
            strncmp(p_string + i, p_substring, l_substring) == 0)
          c->conval = i + 1;
      }
    }
    add_to_list(c, &rslt);
  }
  rslt->repeatc = arg->repeatc;
  return rslt;
}

static ACL *
eval_trim(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  char *p, *cp;
  const char *str;
  int i, cvlen, newlen;

  rslt = eval_init_expr(arg);
  p = stb.n_base + CONVAL1G(rslt->conval);
  cvlen = newlen = size_of(rslt->dtype);

  i = 0;
  p += cvlen - 1;
  /* skip trailing blanks */
  while (cvlen-- > 0) {
    if (*p-- != ' ')
      break;
    newlen--;
  }

  if (newlen == 0) {
    str = " ";
    rslt->conval = getstring(str, strlen(str));
  } else {
    str = cp = getitem(0, newlen);
    i = newlen;
    cp += newlen - 1;
    p++;
    while (i-- > 0) {
      *cp-- = *p--;
    }

    rslt->conval = getstring(str, newlen);
  }

  rslt->dtype = get_type(2, DTY(dtype), newlen);
  return rslt;
}

static ACL *
eval_adjustl(ACL *arg)
{
  ACL *rslt;
  ACL *wrkarg;
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen;

  arg = eval_init_expr(arg);
  rslt = clone_init_const(arg, TRUE);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p = stb.n_base + CONVAL1G(wrkarg->conval);
    cvlen = size_of(wrkarg->dtype);
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
    wrkarg->conval = getstring(str, origlen);
  }

  return rslt;
}

static ACL *
eval_adjustr(ACL *arg)
{
  ACL *rslt;
  ACL *wrkarg;
  char *p, *cp, *str;
  char ch;
  int i, cvlen, origlen;

  arg = eval_init_expr(arg);
  rslt = clone_init_const(arg, TRUE);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    p = stb.n_base + CONVAL1G(wrkarg->conval);
    origlen = cvlen = size_of(wrkarg->dtype);
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
    wrkarg->id = AC_CONVAL;
    wrkarg->conval = getstring(str, origlen);
  }

  return rslt;
}

static ACL *
eval_shape(ACL *arg, DTYPE dtype)
{
  ACL *rslt;

  rslt = clone_init_const(arg, TRUE);
  rslt->dtype = dtype;
  return rslt;
}

static ACL *
eval_size(ACL *arg)
{
  ACL *arg1;
  ACL *arg2;
  ACL *arg3;
  ACL *rslt;
  int dim;
  int i;

  arg = eval_init_expr(arg);
  arg1 = arg;
  arg2 = arg->next;
  if ((arg3 = arg->next->next)) {
    arg3 = eval_init_expr_item(arg3);
    if (!arg3) {
      return 0;
    }
    dim = arg3->conval;

    for (i = 1, arg2 = arg2->subc; i < dim && arg2; i++, arg2 = arg2->next)
      ;
    rslt = clone_init_const(arg2, TRUE);
  } else {
    rslt = clone_init_const(arg1, TRUE);
  }

  return rslt;
}

static ACL *
eval_ul_bound(ACL *arg)
{
  ACL *arg1;
  ACL *arg2;
  INT arg2const;
  ACL *rslt;
  ADSC *adsc;
  int rank;
  int i;

  arg = arg1 = eval_init_expr(arg);
  adsc = AD_DPTR(arg1->dtype);
  rank = AD_UPBD(adsc, 0);
  if (arg->next) {
    arg2 = arg->next;
    arg2const = get_int_from_init_conval(arg2);

    if (arg2const > rank) {
      error(155, 3, gbl.lineno, "DIM argument greater than the array rank",
            CNULL);
      return 0;
    }
    rslt = arg1->subc;
    for (i = 1; rslt && i < arg2const; i++) {
      rslt = rslt->next;
    }
    rslt = clone_init_const(rslt, TRUE);
  } else {
    rslt = clone_init_const(arg1, TRUE);
  }
  return rslt;
}

static int
copy_initconst_to_array(ACL **arr, ACL *c, int count)
{
  int i;
  int acnt;
  ACL *acl;

  for (i = 0; i < count;) {
    if (c == NULL)
      break;
    switch (c->id) {
    case AC_ACONST:
      acnt = copy_initconst_to_array(arr, c->subc,
                                     count - i); /* MORE: count - i??? */
      i += acnt;
      arr += acnt;
      break;
    case AC_CONST:
    case AC_AST:
      acl = *arr = clone_init_const(c, TRUE);
      /* if there is a repeat */
      if (acl->repeatc > 0) {
        acnt = get_int_cval(A_SPTRG(acl->repeatc));
        arr += acnt;
        i += acnt;
      } else {
        arr++;
        i++;
      }
      break;
    default:
      interr("copy_initconst_to_array: unexpected const type", c->id, 3);
      return count;
    }
    c = c->next;
  }
  return i;
}

static ACL *
eval_reshape(ACL *arg, DTYPE dtype, LOGICAL transpose)
{
  ACL *srclist;
  ACL *tacl;
  ACL *pad = NULL;
  ACL *wrklist = NULL;
  ACL *orderarg = NULL;
  ACL **old_val = NULL;
  ACL **new_val = NULL;
  ACL *c = NULL;
  ADSC *adsc = AD_DPTR(dtype);
  int *new_index;
  int src_sz, dest_sz;
  int rank;
  INT order[MAXDIMS];
  int lwb[MAXDIMS];
  int upb[MAXDIMS];
  int mult[MAXDIMS];
  int i;
  int count;

  arg = eval_init_expr(arg);
  srclist = clone_init_const(arg, TRUE);
  if (arg->next && arg->next->next) {
    pad = arg->next->next;
    if (pad->id == AC_ACONST) {
      pad = eval_init_expr_item(pad);
    }
    if (arg->next->next->next && arg->next->next->next->id == AC_ACONST) {
      orderarg = eval_init_expr_item(arg->next->next->next);
    }
  }

  src_sz = get_int_cval(A_SPTRG(ADD_NUMELM(arg->dtype)));
  dest_sz = 1;

  rank = AD_NUMDIM(adsc);
  for (i = 0; i < rank; i++) {
    lwb[i] = 0;
    upb[i] = get_int_cval(A_SPTRG(AD_UPBD(adsc, i)));
    mult[i] = dest_sz;
    dest_sz *= upb[i];
  }

  if (orderarg == NULL) {
    if (transpose) {
        order[0] = 1;
        order[1] = 0;
    } else {
      if (src_sz == dest_sz) {
        return srclist;
      }
      for (i = 0; i < rank; i++) {
        order[i] = i;
      }
    }
  } else {
    LOGICAL out_of_order;

    out_of_order = FALSE;
    c = (orderarg->id == AC_ACONST ? orderarg->subc : orderarg);
    for (i = 0; c && i < rank; c = c->next, i++) {
      order[i] =
          DT_ISWORD(c->dtype) ? c->conval - 1 : get_int_cval(c->conval) - 1;
      if (order[i] != i)
        out_of_order = TRUE;
    }
    if (!out_of_order && src_sz == dest_sz) {
      return srclist;
    }
  }

  NEW(old_val, ACL *, dest_sz);
  if (old_val == NULL)
    return 0;
  BZERO(old_val, ACL *, dest_sz);
  /* MORE use GET_ACL for new_value */
  NEW(new_val, ACL *, dest_sz);
  NEW(new_index, int, dest_sz);
  if (new_val == NULL || new_index == NULL) {
    return 0;
  }
  BZERO(old_val, ACL *, dest_sz);
  BZERO(new_index, int, dest_sz);

  count = dest_sz > src_sz ? src_sz : dest_sz;
  wrklist = srclist->id == AC_ACONST ? srclist->subc : srclist;
  (void)copy_initconst_to_array(old_val, wrklist, count);

  if (dest_sz > src_sz) {
    count = dest_sz - src_sz;
    wrklist = pad->id == AC_ACONST ? pad->subc : pad;
    while (count > 0) {
      i = copy_initconst_to_array(old_val + src_sz, wrklist, count);
      count -= i;
      src_sz += i;
    }
  }

  /* index to access source in linear order */
  i = 0;
  while (TRUE) {
    int index; /* index where to store each element of new val */
    int j;

    index = 0;
    for (j = 0; j < rank; j++)
      index += lwb[j] * mult[j];

    /* new_index contains old_val index */
    new_index[index] = i;

    /* update loop indices */
    for (j = 0; j < rank; j++) {
      int loop;
      loop = order[j];
      lwb[loop]++;
      if (lwb[loop] < upb[loop])
        break;
      lwb[loop] = 0; /* reset and go on to the next loop */
    }
    if (j >= rank)
      break;
    i++;
  }

  for (i = 0; i < dest_sz; i++) {
    ACL *tacl, *tail;
    int idx, start, end;
    int index = new_index[i];
    int repeatc;
    if (old_val[index]) {
      if (old_val[index]->repeatc)
        repeatc = get_int_cval(A_SPTRG(old_val[index]->repeatc));
      else
        repeatc = 1;
      if (repeatc <= 1) {
        new_val[i] = old_val[index];
        new_val[i]->id = AC_CONVAL;
      } else {
        idx = index + 1;
        start = i;
        end = repeatc - 1;
        while (new_index[++start] == idx) {
          ++idx;
          if (end <= 0 || start > dest_sz - 1)
            break;
        }
        old_val[index]->next = NULL;
        tacl = clone_init_const(old_val[index], TRUE);
        tacl->repeatc = mk_cval(idx - index, DT_INT);
        tacl->id = AC_CONVAL;
        old_val[index]->repeatc = mk_cval(index - (idx - index), DT_INT);
        new_val[i] = tacl;
      }
    } else {
      tail = old_val[index];
      idx = index;
      while (tail == NULL && idx >= 0) {
        tail = old_val[idx--];
      }
      tail->next = NULL;
      tacl = clone_init_const(tail, TRUE);
      start = i;
      end = get_int_cval(A_SPTRG(tail->repeatc)) - 1;
      idx = index + 1;
      while (new_index[++start] == idx) {
        ++idx;
        --end;
        if (end <= 0 || start > dest_sz - 1)
          break;
      }
      tail->repeatc = mk_cval(index - (idx - index), DT_INT);
      tacl->repeatc = mk_cval(idx - index, DT_INT);
      tacl->id = AC_CONVAL;
      new_val[i] = tacl;
    }
  }
  tacl = new_val[0];
  for (i = 0; i < dest_sz - 1; ++i) {
    if (new_val[i + 1] == NULL) {
      continue;
    } else {
      tacl->next = new_val[i + 1];
      tacl = new_val[i + 1];
    }
  }
  if (new_val[dest_sz - 1])
    (new_val[dest_sz - 1])->next = NULL;
  srclist = *new_val;

  FREE(old_val);
  FREE(new_index);

  return srclist;
}

static ACL *
eval_null(int sptr)
{
  ACL *root = NULL;
  ACL *c;

  /* for <ptr>$p */
  c = GET_ACL(15);
  c->id = AC_CONVAL;
  c->dtype = DT_PTR;
  c->u1.ast = astb.bnd.zero;
  c->conval = 0;
  add_to_list(c, &root);
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    /* for <ptr>$o */
    c = GET_ACL(15);
    c->id = AC_CONVAL;
    c->dtype = DT_PTR;
    c->sptr = PTROFFG(sptr);
    c->u1.ast = astb.bnd.zero;
    c->conval = 0;
    add_to_list(c, &root);
    /* for <ptr>$sd[1] */
    c = GET_ACL(15);
    c->id = AC_CONVAL;
    c->dtype = astb.bnd.dtype;
    c->sptr = SDSCG(sptr);
    c->u1.ast = astb.bnd.zero;
    c->conval = 0;
    add_to_list(c, &root);
  }

  return root;
}

static ACL *
eval_sqrt(ACL *arg, DTYPE dtype)
{
  ACL *rslt;
  ACL *wrkarg;
  INT conval;

  rslt = arg = eval_init_expr(arg);
  wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);
  for (; wrkarg; wrkarg = wrkarg->next) {
    INT num1[4];
    INT res[4];
    INT con1;

    con1 = wrkarg->conval;
    switch (DTY(wrkarg->dtype)) {
    case TY_REAL:
      xfsqrt(con1, &res[0]);
      conval = res[0];
      break;
    case TY_DBLE:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      xdsqrt(num1, res);
      conval = getcon(res, DT_DBLE);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      num1[0] = CONVAL1G(con1);
      num1[1] = CONVAL2G(con1);
      num1[2] = CONVAL3G(con1);
      num1[3] = CONVAL4G(con1);
      xqsqrt(num1, res);
      conval = getcon(res, DT_QUAD);
      break;
#endif
    case TY_CMPLX:
    case TY_DCMPLX:
      /*
          a = sqrt(real**2 + imag**2);  "hypot(real,imag)
          if (a == 0) {
              x = 0;
              y = 0;
          }
          else if (real > 0) {
              x = sqrt(0.5 * (a + real));
              y = 0.5 * (imag / x);
          }
          else {
              y = sqrt(0.5 * (a - real));
              if (imag < 0)
                  y = -y;
              x = 0.5 * (imag / y);
          }
          res.real = x;
          res.imag = y;
      */

      error(155, 3, gbl.lineno,
            "Intrinsic not supported in initialization:", "sqrt");
      break;
    default:
      error(155, 3, gbl.lineno,
            "Intrinsic not supported in initialization:", "sqrt");
      break;
    }
    conval = cngcon(conval, wrkarg->dtype, dtype);
    wrkarg->conval = conval;
    wrkarg->dtype = dtype;
  }
  return rslt;
}

/*----------------------------------------------------------------------------*/

#ifdef TARGET_SUPPORTS_QUADFP
#define FPINTRIN1(iname, ent, fscutil, dscutil, qscutil)                       \
  static ACL *ent(ACL *arg, DTYPE dtype)                                       \
  {                                                                            \
    ACL *rslt;                                                                 \
    ACL *wrkarg;                                                               \
    INT conval;                                                                \
    rslt = arg = eval_init_expr(arg);                                          \
    wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);                      \
    for (; wrkarg; wrkarg = wrkarg->next) {                                    \
      INT num1[4];                                                             \
      INT res[4];                                                              \
      INT con1;                                                                \
      con1 = wrkarg->conval;                                                   \
      switch (DTY(wrkarg->dtype)) {                                            \
      case TY_REAL:                                                            \
        fscutil(con1, &res[0]);                                                \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        dscutil(num1, res);                                                    \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_QUAD:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num1[2] = CONVAL3G(con1);                                              \
        num1[3] = CONVAL4G(con1);                                              \
        qscutil(num1, res);                                                    \
        conval = getcon(res, DT_QUAD);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      case TY_HALF:                                                            \
        /* fallthrough to error */                                             \
      default:                                                                 \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, wrkarg->dtype, dtype);                           \
      wrkarg->conval = conval;                                                 \
      wrkarg->dtype = dtype;                                                   \
    }                                                                          \
    return rslt;                                                               \
  }
#else
#define FPINTRIN1(iname, ent, fscutil, dscutil, qscutil)                       \
  static ACL *ent(ACL *arg, DTYPE dtype)                                       \
  {                                                                            \
    ACL *rslt;                                                                 \
    ACL *wrkarg;                                                               \
    INT conval;                                                                \
    rslt = arg = eval_init_expr(arg);                                          \
    wrkarg = (rslt->id == AC_ACONST ? rslt->subc : rslt);                      \
    for (; wrkarg; wrkarg = wrkarg->next) {                                    \
      INT num1[4];                                                             \
      INT res[4];                                                              \
      INT con1;                                                                \
      con1 = wrkarg->conval;                                                   \
      switch (DTY(wrkarg->dtype)) {                                            \
      case TY_REAL:                                                            \
        fscutil(con1, &res[0]);                                                \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        dscutil(num1, res);                                                    \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      case TY_HALF:                                                            \
        /* fallthrough to error */                                             \
      default:                                                                 \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, wrkarg->dtype, dtype);                           \
      wrkarg->conval = conval;                                                 \
      wrkarg->dtype = dtype;                                                   \
    }                                                                          \
    return rslt;                                                               \
  }
#endif

FPINTRIN1("exp", eval_exp, xfexp, xdexp, xqexp)

FPINTRIN1("log", eval_log, xflog, xdlog, xqlog)

FPINTRIN1("log10", eval_log10, xflog10, xdlog10, xqlog10)

FPINTRIN1("sin", eval_sin, xfsin, xdsin, xqsin)

FPINTRIN1("cos", eval_cos, xfcos, xdcos, xqcos)

FPINTRIN1("tan", eval_tan, xftan, xdtan, xqtan)

FPINTRIN1("asin", eval_asin, xfasin, xdasin, xqasin)

FPINTRIN1("acos", eval_acos, xfacos, xdacos, xqacos)

FPINTRIN1("atan", eval_atan, xfatan, xdatan, xqatan)

#ifdef TARGET_SUPPORTS_QUADFP
#define FPINTRIN2(iname, ent, fscutil, dscutil, qscutil)                       \
  static ACL *ent(ACL *arg, DTYPE dtype)                                       \
  {                                                                            \
    ACL *rslt = arg;                                                           \
    ACL *arg1, *arg2;                                                          \
    INT conval;                                                                \
    arg1 = eval_init_expr_item(arg);                                           \
    arg2 = eval_init_expr_item(arg->next);                                     \
    rslt = clone_init_const(arg1, TRUE);                                       \
    arg1 = (rslt->id == AC_ACONST ? rslt->subc : rslt);                        \
    arg2 = (arg2->id == AC_ACONST ? arg2->subc : arg2);                        \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {                       \
      INT num1[4], num2[4];                                                    \
      INT res[4];                                                              \
      INT con1, con2;                                                          \
      con1 = arg1->conval;                                                     \
      con2 = arg2->conval;                                                     \
      switch (DTY(arg1->dtype)) {                                              \
      case TY_REAL:                                                            \
        fscutil(con1, con2, &res[0]);                                          \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        dscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_QUAD:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num1[2] = CONVAL3G(con1);                                              \
        num1[3] = CONVAL4G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        num2[2] = CONVAL3G(con2);                                              \
        num2[3] = CONVAL4G(con2);                                              \
        qscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_QUAD);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      case TY_HALF:                                                            \
        /* fallthrough to error */                                             \
      default:                                                                 \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, arg1->dtype, dtype);                             \
      arg1->conval = conval;                                                   \
      arg1->dtype = dtype;                                                     \
    }                                                                          \
    return rslt;                                                               \
  }
#else
#define FPINTRIN2(iname, ent, fscutil, dscutil, qscutil)                       \
  static ACL *ent(ACL *arg, DTYPE dtype)                                       \
  {                                                                            \
    ACL *rslt = arg;                                                           \
    ACL *arg1, *arg2;                                                          \
    INT conval;                                                                \
    arg1 = eval_init_expr_item(arg);                                           \
    arg2 = eval_init_expr_item(arg->next);                                     \
    rslt = clone_init_const(arg1, TRUE);                                       \
    arg1 = (rslt->id == AC_ACONST ? rslt->subc : rslt);                        \
    arg2 = (arg2->id == AC_ACONST ? arg2->subc : arg2);                        \
    for (; arg1; arg1 = arg1->next, arg2 = arg2->next) {                       \
      INT num1[4], num2[4];                                                    \
      INT res[4];                                                              \
      INT con1, con2;                                                          \
      con1 = arg1->conval;                                                     \
      con2 = arg2->conval;                                                     \
      switch (DTY(arg1->dtype)) {                                              \
      case TY_REAL:                                                            \
        fscutil(con1, con2, &res[0]);                                          \
        conval = res[0];                                                       \
        break;                                                                 \
      case TY_DBLE:                                                            \
        num1[0] = CONVAL1G(con1);                                              \
        num1[1] = CONVAL2G(con1);                                              \
        num2[0] = CONVAL1G(con2);                                              \
        num2[1] = CONVAL2G(con2);                                              \
        dscutil(num1, num2, res);                                              \
        conval = getcon(res, DT_DBLE);                                         \
        break;                                                                 \
      case TY_CMPLX:                                                           \
      case TY_DCMPLX:                                                          \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      case TY_HALF:                                                            \
        /* fallthrough to error */                                             \
      default:                                                                 \
        error(155, 3, gbl.lineno,                                              \
              "Intrinsic not supported in initialization:", iname);            \
        break;                                                                 \
      }                                                                        \
      conval = cngcon(conval, arg1->dtype, dtype);                             \
      arg1->conval = conval;                                                   \
      arg1->dtype = dtype;                                                     \
    }                                                                          \
    return rslt;                                                               \
  }
#endif

FPINTRIN2("atan2", eval_atan2, xfatan2, xdatan2, xqatan2)

static INT
get_const_from_ast(int ast)
{
  DTYPE dtype = A_DTYPEG(ast);
  INT c = 0;

  if (A_TYPEG(ast) == A_ID) {

    if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
      c = A_SPTRG(ast);
    } else {
      c = CONVAL1G(A_SPTRG(ast));
    }
  } else if (A_ALIASG(ast)) {
    if (DT_ISWORD(A_DTYPEG(ast))) {
      c = CONVAL2G(A_SPTRG(A_ALIASG(ast)));
    } else {
      c = A_SPTRG(A_ALIASG(ast));
    }
  } else {
    if (A_TYPEG(ast) == A_BINOP || A_TYPEG(ast) == A_INTR) {
      return const_eval(ast);
    }
    interr("get_const_from_ast: can't get const value", 0, 3);
  }

  return c;
}

static struct {
  ACL *root;
  ACL *roottail;
  ACL *arrbase;
  int ndims;
  struct {
    DTYPE dtype;
    ISZ_T idx;
    ACL *subscr_base;
    ISZ_T lowb;
    ISZ_T upb;
    ISZ_T stride;
  } sub[MAXDIMS];
  struct {
    ISZ_T lowb;
    ISZ_T upb;
    ISZ_T mplyr;
  } dim[MAXDIMS];
} sb;

static ISZ_T
eval_sub_index(int dim)
{
  int repeatc;
  ISZ_T o_lowb, elem_offset;
  ACL *subscr_base;
  ADSC *adsc = AD_DPTR(sb.sub[dim].dtype);
  o_lowb = ad_val_of(sym_of_ast(AD_LWAST(adsc, 0)));
  subscr_base = sb.sub[dim].subscr_base;

  elem_offset = (sb.sub[dim].idx - o_lowb);
  while (elem_offset && subscr_base) {
    if (subscr_base->repeatc)
      repeatc = get_int_cval(A_SPTRG(subscr_base->repeatc));
    else
      repeatc = 1;
    if (repeatc > 1) {
      while (repeatc > 0 && elem_offset) {
        --repeatc;
        --elem_offset;
      }
    } else {
      subscr_base = subscr_base->next;
      --elem_offset;
    }
  }
  return get_ival(subscr_base->dtype, subscr_base->conval);
}

static int
eval_sb(int d)
{
  int i;
  int t_ub = 0;
  ISZ_T sub_idx;
  ISZ_T elem_offset;
  ISZ_T repeat;
  ACL *v;
  ACL *c;
  ACL tmp;

#define TRACE_EVAL_SB 0
  if (d == 0) {
#if TRACE_EVAL_SB
    printf("-----\n");
#endif
    sb.sub[0].idx = sb.sub[0].lowb;
    if (sb.sub[0].stride > 0)
      t_ub = 1;
    while ((t_ub ? sb.sub[0].idx <= sb.sub[0].upb
                 : sb.sub[0].idx >= sb.sub[0].upb)) {
      /* compute element offset */
      elem_offset = 0;
      for (i = 0; i < sb.ndims; i++) {
        sub_idx = sb.sub[i].idx;
        if (sb.sub[i].subscr_base) {
          sub_idx = eval_sub_index(i);
        }
        assert(sub_idx >= sb.dim[i].lowb && sub_idx <= sb.dim[i].upb,
               "Subscript for array is out-of-bounds", sub_idx, 0);

        elem_offset += (sub_idx - sb.dim[i].lowb) * sb.dim[i].mplyr;
#if TRACE_EVAL_SB
        printf("%3d ", sub_idx);
#endif
      }
#if TRACE_EVAL_SB
      printf(" elem_offset - %ld\n", elem_offset);
#endif
      /* get initialization value at element offset */
      v = sb.arrbase;
      while (v && elem_offset) {
        if (v->repeatc)
          repeat = get_int_cval(A_SPTRG(v->repeatc));
        else
          repeat = 1;
        if (repeat > 1) {
          while (v && repeat > 0 && elem_offset) {
            --elem_offset;
            --repeat;
          }
        } else {
          v = v->next;
          --elem_offset;
        }
      }
      if (v == NULL) {
        interr("initialization expression: invalid array subscripts\n",
               elem_offset, 3);
        return 1;
      }
      /*
       * evaluate initialization value and add (repeat copies) to
       * initialization list
       */
      tmp = *v;
      tmp.next = 0;
      tmp.repeatc = astb.i1;
      c = eval_init_expr_item(clone_init_const(&tmp, TRUE));
      c->next = NULL;

      add_to_list(c, &sb.root);
      sb.sub[0].idx += sb.sub[0].stride;
    }
#if TRACE_EVAL_SB
    printf("-----\n");
#endif
    return 0;
  }
  if (sb.sub[d].stride > 0) {
    for (sb.sub[d].idx = sb.sub[d].lowb; sb.sub[d].idx <= sb.sub[d].upb;
         sb.sub[d].idx += sb.sub[d].stride) {
      if (eval_sb(d - 1))
        return 1;
    }
  } else {
    for (sb.sub[d].idx = sb.sub[d].lowb; sb.sub[d].idx >= sb.sub[d].upb;
         sb.sub[d].idx += sb.sub[d].stride) {
      if (eval_sb(d - 1))
        return 1;
    }
  }
  return 0;
}

static ACL *
eval_const_array_section(ACL *lop, int ldtype)
{
  ADSC *adsc = AD_DPTR(ldtype);
  int ndims = 0;
  int i;

  sb.root = sb.roottail = NULL;
  if (lop->id == AC_ACONST) {
    sb.arrbase = eval_array_constructor(lop);
  } else {
    sb.arrbase = lop;
  }

  if (sb.ndims != AD_NUMDIM(adsc)) {
    interr("initialization expression: subscript/dimension mis-match\n", ldtype,
           3);
    return 0;
  }
  ndims = AD_NUMDIM(adsc);
  for (i = 0; i < ndims; i++) {
    sb.dim[i].lowb = ad_val_of(sym_of_ast(AD_LWAST(adsc, i)));
    sb.dim[i].upb = ad_val_of(sym_of_ast(AD_UPAST(adsc, i)));
    sb.dim[i].mplyr = ad_val_of(sym_of_ast(AD_MLPYR(adsc, i)));
  }

  sb.ndims = ndims;
  if (eval_sb(ndims - 1))
    return 0;

  return sb.root;
}

static ISZ_T
get_ival(DTYPE dtype, INT conval)
{
  switch (DTY(dtype)) {
  case TY_INT8:
  case TY_LOG8:
    return get_isz_cval(conval);
  default:
    return conval;
  }
}

static ACL *
eval_const_array_triple_section(ACL *curr_e)
{
  ACL *c, *lop, *rop, *t_lop;
  ACL *v;
  int ndims = 0;

  sb.root = sb.roottail = NULL;
  c = curr_e;
  do {
    rop = c->u1.expr->rop;
    lop = c->u1.expr->lop;
    sb.sub[ndims].subscr_base = 0;
    sb.sub[ndims].dtype = 0;
    if (lop) {
      t_lop = eval_init_expr(lop);
      sb.sub[ndims].dtype = t_lop->dtype;
      if (t_lop->id == AC_ACONST)
        sb.sub[ndims].subscr_base = eval_array_constructor(t_lop);
      else
        sb.sub[ndims].subscr_base = t_lop;
    }
    if (rop == 0) {
      interr("initialization expression: missing array section lb\n", 0, 3);
      return 0;
    }
    v = eval_init_expr(rop);
    if (!v || !v->is_const) {
      interr("initialization expression: non-constant lb\n", 0, 3);
      return 0;
    }
    sb.sub[ndims].lowb = get_ival(v->dtype, v->conval);

    if ((rop = rop->next) == 0) {
      interr("initialization expression: missing array section ub\n", 0, 3);
      return 0;
    }
    v = eval_init_expr(rop);
    if (!v || !v->is_const) {
      interr("initialization expression: non-constant ub\n", 0, 3);
      return 0;
    }

    sb.sub[ndims].upb = get_ival(v->dtype, v->conval);

    if ((rop = rop->next) == 0) {
      interr("initialization expression: missing array section stride\n", 0, 3);
      return 0;
    }
    v = eval_init_expr(rop);
    if (!v || !v->is_const) {
      interr("initialization expression: non-constant stride\n", 0, 3);
      return 0;
    }

    sb.sub[ndims].stride = get_ival(v->dtype, v->conval);

    if (++ndims >= 7) {
      interr("initialization expression: too many dimensions\n", 0, 3);
      return 0;
    }
    c = c->next;
  } while (c);

  sb.ndims = ndims;
  return sb.root;
}

static void
mk_cmp(ACL *c, int op, INT l_conval, INT r_conval, int rdtype, int dt)
{
  switch (get_ast_op(op)) {
  case OP_EQ:
  case OP_GE:
  case OP_GT:
  case OP_LE:
  case OP_LT:
  case OP_NE:
    l_conval = const_fold(OP_CMP, l_conval, r_conval, rdtype);
    switch (get_ast_op(op)) {
    case OP_EQ:
      l_conval = l_conval == 0;
      break;
    case OP_GE:
      l_conval = l_conval >= 0;
      break;
    case OP_GT:
      l_conval = l_conval > 0;
      break;
    case OP_LE:
      l_conval = l_conval <= 0;
      break;
    case OP_LT:
      l_conval = l_conval < 0;
      break;
    case OP_NE:
      l_conval = l_conval != 0;
      break;
    }
    l_conval = l_conval ? SCFTN_TRUE : SCFTN_FALSE;
    c->conval = l_conval;
    break;
  case OP_LEQV:
    l_conval = const_fold(OP_CMP, l_conval, r_conval, rdtype);
    c->conval = l_conval == 0;
    break;
  case OP_LNEQV:
    l_conval = const_fold(OP_CMP, l_conval, r_conval, rdtype);
    c->conval = l_conval != 0;
    break;
  case OP_LOR:
    c->conval = l_conval | r_conval;
    break;
  case OP_LAND:
    c->conval = l_conval & r_conval;
    break;
  default:
    c->conval = const_fold(get_ast_op(op), l_conval, r_conval, dt);
  }
}

static ACL *
eval_init_op(int op, ACL *lop, DTYPE ldtype, ACL *rop, DTYPE rdtype, SPTR sptr,
             DTYPE dtype)
{
  ACL *root = NULL;
  ACL *c;
  ACL *cur_lop;
  ACL *cur_rop;
  DTYPE dt = DDTG(dtype);
  DTYPE e_dtype;
  int l_repeatc;
  int r_repeatc;
  INT l_conval;
  INT r_conval;
  int count;
  int lsptr;
  int rsptr;
  char *s;
  int llen;
  int rlen;

  if (!lop) {
    return 0;
  }

  if (op == AC_NEG || op == AC_LNOT) {
    cur_lop = (lop->id == AC_ACONST ? lop->subc : lop);
    for (; cur_lop; cur_lop = cur_lop->next) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = astb.i1;
      l_conval = cur_lop->conval;
      if (dt != cur_lop->dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), dt);
      }
      if (op == AC_LNOT)
        c->conval = ~(l_conval);
      else
        c->conval = negate_const(l_conval, dt);
      add_to_list(c, &root);
    }
  } else if (op == AC_ARRAYREF) {
    root = eval_const_array_section(lop, ldtype);
  } else if (op == AC_CAT) {
    lsptr = lop->conval;
    rsptr = rop->conval;
    llen = string_length(DTYPEG(lsptr));
    rlen = string_length(DTYPEG(rsptr));
    s = getitem(0, llen + rlen);
    BCOPY(s, stb.n_base + CONVAL1G(lsptr), char, llen);
    BCOPY(s + llen, stb.n_base + CONVAL1G(rsptr), char, rlen);

    c = GET_ACL(15);
    c->id = AC_CONST;
    c->dtype =
        get_type(2, DTY(DDTG(DTYPEG(lsptr))), mk_cval(llen + rlen, DT_INT4));
    c->repeatc = astb.i1;
    c->conval = c->sptr = getstring(s, llen + rlen);
    c->u1.ast = mk_cnst(c->conval);
    add_to_list(c, &root);
  } else if (op == AC_CONV) {
    cur_lop = (lop->id == AC_ACONST ? lop->subc : lop);
    if (cur_lop->repeatc)
      l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
    else
      l_repeatc = 1;
    for (; cur_lop;) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = astb.i1;
      c->conval = cngcon(cur_lop->conval, cur_lop->dtype, DDTG(dtype));
      add_to_list(c, &root);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          if (cur_lop->repeatc)
            l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
          else
            l_repeatc = 1;
        }
      }
    }
  } else if (op == AC_MEMBR_SEL) {
    sptr = A_SPTRG(lop->u1.ast);
    if (DTY(DTYPEG(sptr)) != TY_DERIVED || !PARAMG(sptr)) {
      error(
          4, 3, gbl.lineno,
          "Left hand side of % operator must be a named constant derived type",
          NULL);
      return 0;
    }

    sptr = NMCNSTG(sptr);
    c = clone_init_const(get_getitem_p(CONVAL2G(sptr)), TRUE);

    if (c->id != AC_SCONST) {
      interr("Malformed member select operator, lhs not a derived type "
             "initializaer",
             op, 3);
      return 0;
    }

    for (c = c->subc, count = CONVAL2G(A_SPTRG(rop->u1.ast)); c && count;
         c = c->next, --count)
      ;

    if (!c || count != 0) {
      interr("Malformed member select operator, invalid member specifier", op,
             3);
      return 0;
    }

    root = clone_init_const(c, TRUE);
    root = eval_init_expr(root);
  } else if (op == AC_INTR_CALL) {
    AC_INTRINSIC intrin = lop->u1.i;
    switch (intrin) {
    case AC_I_adjustl:
      root = eval_adjustl(rop);
      break;
    case AC_I_adjustr:
      root = eval_adjustr(rop);
      break;
    case AC_I_char:
      root = eval_char(rop, dtype);
      break;
    case AC_I_ichar:
      root = eval_ichar(rop, dtype);
      break;
    case AC_I_index:
      root = eval_index(rop);
      break;
    case AC_I_int:
      root = eval_int(rop, dtype);
      break;
    case AC_I_ishft:
      root = eval_ishft(rop, dtype);
      break;
    case AC_I_len_trim:
      root = eval_len_trim(rop);
      break;
    case AC_I_ubound:
    case AC_I_lbound:
      root = eval_ul_bound(rop);
      break;
    case AC_I_min:
      root = eval_min_or_max(rop, dtype, /*want_max*/ FALSE);
      break;
    case AC_I_max:
      root = eval_min_or_max(rop, dtype, /*want_max*/ TRUE);
      break;
    case AC_I_nint:
      root = eval_nint(rop, dtype);
      break;
    case AC_I_null:
      root = eval_null(sptr);
      break;
    case AC_I_fltconvert:
      root = eval_fltconvert(rop, dtype);
      break;
    case AC_I_repeat:
      root = eval_repeat(rop, dtype);
      break;
    case AC_I_transfer:
      root = eval_transfer(rop, dtype);
      break;
    case AC_I_transpose:
      root = eval_reshape(rop, dtype, /*transpose*/ TRUE);
      break;
    case AC_I_reshape:
      root = eval_reshape(rop, dtype, /*transpose*/ FALSE);
      break;
    case AC_I_selected_int_kind:
      root = eval_selected_int_kind(rop);
      break;
    case AC_I_selected_real_kind:
      root = eval_selected_real_kind(rop);
      break;
    case AC_I_selected_char_kind:
      root = eval_selected_char_kind(rop);
      break;
    case AC_I_scan:
      root = eval_scan(rop);
      break;
    case AC_I_shape:
      root = eval_shape(rop, dtype);
      break;
    case AC_I_size:
      root = eval_size(rop);
      break;
    case AC_I_trim:
      root = eval_trim(rop, dtype);
      break;
    case AC_I_verify:
      root = eval_verify(rop);
      break;
    case AC_I_floor:
      root = eval_floor(rop, dtype);
      break;
    case AC_I_ceiling:
      root = eval_ceiling(rop, dtype);
      break;
    case AC_I_mod:
      root = eval_mod(rop, dtype);
      break;
    case AC_I_sqrt:
      root = eval_sqrt(rop, dtype);
      break;
    case AC_I_exp:
      root = eval_exp(rop, dtype);
      break;
    case AC_I_log:
      root = eval_log(rop, dtype);
      break;
    case AC_I_log10:
      root = eval_log10(rop, dtype);
      break;
    case AC_I_sin:
      root = eval_sin(rop, dtype);
      break;
    case AC_I_cos:
      root = eval_cos(rop, dtype);
      break;
    case AC_I_tan:
      root = eval_tan(rop, dtype);
      break;
    case AC_I_asin:
      root = eval_asin(rop, dtype);
      break;
    case AC_I_acos:
      root = eval_acos(rop, dtype);
      break;
    case AC_I_atan:
      root = eval_atan(rop, dtype);
      break;
    case AC_I_atan2:
      root = eval_atan2(rop, dtype);
      break;
    case AC_I_abs:
      root = eval_abs(rop, dtype);
      break;
    case AC_I_iand:
      root = eval_iand(rop, dtype);
      break;
    case AC_I_ior:
      root = eval_ior(rop, dtype);
      break;
    case AC_I_ieor:
      root = eval_ieor(rop, dtype);
      break;
    case AC_I_merge:
      root = eval_merge(rop, dtype);
      break;
    case AC_I_scale:
      root = eval_scale(rop, dtype);
      break;
    case AC_I_maxloc:
    case AC_I_maxval:
    case AC_I_minloc:
    case AC_I_minval:
      root = eval_minval_or_maxval(rop, rdtype, intrin);
      break;
    default:
      interr("eval_init_op(semutil2.c): intrinsic not supported in "
             "initialization",
             intrin, ERR_Severe);
      /* Try to avoid a seg fault by returning something reasonable */
      root = GET_ACL(15);
      root->id = AC_CONST;
      root->repeatc = astb.i1;
      root->dtype = dtype;
      root->conval = cngcon(0, DT_INT, dtype);
    }
  } else if (DTY(ldtype) == TY_ARRAY && DTY(rdtype) == TY_ARRAY) {
    /* array <binop> array */
    cur_lop = (lop->id == AC_ACONST ? lop->subc : lop);
    cur_rop = (rop->id == AC_ACONST ? rop->subc : rop);
    if (cur_lop->repeatc)
      l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
    else
      l_repeatc = 1;
    if (cur_rop->repeatc)
      r_repeatc = get_int_cval(A_SPTRG(cur_rop->repeatc));
    else
      r_repeatc = 1;
    e_dtype = DDTG(dtype);
    for (; cur_rop && cur_lop;) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->dtype = dt;
      l_conval = cur_lop->conval;
      if (DDTG(cur_lop->dtype) != e_dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), e_dtype);
      }
      r_conval = cur_rop->conval;
      if (DDTG(cur_rop->dtype) != e_dtype) {
        r_conval = cngcon(r_conval, DDTG(cur_rop->dtype), e_dtype);
      }
      c->conval = const_fold(get_ast_op(op), l_conval, r_conval, dt);
      add_to_list(c, &root);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          if (cur_lop->repeatc)
            l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
          else
            l_repeatc = 1;
        }
      }
      if (--r_repeatc <= 0) {
        cur_rop = cur_rop->next;
        if (cur_rop) {
          if (cur_rop->repeatc)
            r_repeatc = get_int_cval(A_SPTRG(cur_rop->repeatc));
          else
            r_repeatc = 1;
        }
      }
    }
  } else if (DTY(ldtype) == TY_ARRAY) {
    /* array <binop> scalar */
    cur_lop = (lop->id == AC_ACONST ? lop->subc : lop);
    if (cur_lop->repeatc)
      l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
    else
      l_repeatc = 1;
    e_dtype = DDTG(dtype) != DT_LOG ? DDTG(dtype) : DDTG(rop->dtype);
    r_conval = rop->conval;
    if (rop->dtype != e_dtype) {
      r_conval = cngcon(r_conval, rop->dtype, e_dtype);
    }
    for (; cur_lop;) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = astb.i1;
      l_conval = cur_lop->conval;
      if (DDTG(cur_lop->dtype) != e_dtype) {
        l_conval = cngcon(l_conval, DDTG(cur_lop->dtype), e_dtype);
      }

      mk_cmp(c, op, l_conval, r_conval, rdtype, dt);
      add_to_list(c, &root);
      if (--l_repeatc <= 0) {
        cur_lop = cur_lop->next;
        if (cur_lop) {
          if (cur_lop->repeatc)
            l_repeatc = get_int_cval(A_SPTRG(cur_lop->repeatc));
          else
            l_repeatc = 1;
        }
      }
    }
  } else if (DTY(rdtype) == TY_ARRAY) {
    /* scalar <binop> array */
    cur_rop = (rop->id == AC_ACONST ? rop->subc : rop);
    if (cur_rop->repeatc)
      r_repeatc = get_int_cval(A_SPTRG(cur_rop->repeatc));
    else
      r_repeatc = 1;
    e_dtype = DDTG(dtype) != DT_LOG ? DDTG(dtype) : DDTG(lop->dtype);
    l_conval = lop->conval;
    if (lop->dtype != e_dtype) {
      l_conval = cngcon(l_conval, lop->dtype, e_dtype);
    }
    for (cur_rop = rop; cur_rop;) {
      c = GET_ACL(15);
      c->id = AC_CONST;
      c->dtype = dt;
      c->repeatc = astb.i1;
      r_conval = cur_rop->conval;
      if (DDTG(cur_rop->dtype) != e_dtype) {
        r_conval = cngcon(r_conval, DDTG(cur_rop->dtype), e_dtype);
      }
      mk_cmp(c, op, l_conval, r_conval, rdtype, dt);
      add_to_list(c, &root);
      if (--r_repeatc <= 0) {
        cur_rop = cur_rop->next;
        if (cur_rop) {
          if (cur_rop->repeatc)
            r_repeatc = get_int_cval(A_SPTRG(cur_rop->repeatc));
          else
            r_repeatc = 1;
        }
      }
    }
  } else {
    /* scalar <binop> scalar */
    root = GET_ACL(15);
    root->id = AC_CONST;
    root->repeatc = astb.i1;
    root->dtype = dt;
    op = get_ast_op(op);
    switch (op) {
    case OP_EQ:
    case OP_GE:
    case OP_GT:
    case OP_LE:
    case OP_LT:
    case OP_NE:
      l_conval = const_fold(OP_CMP, lop->conval, rop->conval, ldtype);
      switch (op) {
      case OP_EQ:
        l_conval = (l_conval == 0);
        break;
      case OP_GE:
        l_conval = (l_conval >= 0);
        break;
      case OP_GT:
        l_conval = (l_conval > 0);
        break;
      case OP_LE:
        l_conval = (l_conval <= 0);
        break;
      case OP_LT:
        l_conval = (l_conval < 0);
        break;
      case OP_NE:
        l_conval = (l_conval != 0);
        break;
      }
      l_conval = l_conval ? SCFTN_TRUE : SCFTN_FALSE;
      root->conval = l_conval;
      break;
    case OP_LEQV:
      l_conval = const_fold(OP_CMP, lop->conval, rop->conval, ldtype);
      root->conval = (l_conval == 0);
      break;
    case OP_LNEQV:
      l_conval = const_fold(OP_CMP, lop->conval, rop->conval, ldtype);
      root->conval = (l_conval != 0);
      break;
    case OP_LOR:
      root->conval = lop->conval | rop->conval;
      break;
    case OP_LAND:
      root->conval = lop->conval & rop->conval;
      break;
    default:
      l_conval = lop->conval;
      if (lop->dtype != dt) {
        l_conval = cngcon(l_conval, lop->dtype, dt);
      }
      r_conval = rop->conval;
      if (rop->dtype != dt) {
        r_conval = cngcon(r_conval, rop->dtype, dt);
      }
      root->conval = const_fold(op, l_conval, r_conval, dt);
      break;
    }
  }
  return root;
}

static ACL *
eval_array_constructor(ACL *e)
{
  ACL *root = NULL;
  ACL *cur_e;
  ACL *new_e;

  /* collapse nested array contstructors */
  for (cur_e = e->subc; cur_e; cur_e = cur_e->next) {
    if (cur_e->id == AC_ACONST) {
      new_e = eval_array_constructor(cur_e);
    } else {
      new_e = eval_init_expr_item(cur_e);
      if (!new_e) {
        return 0;
      }
      if (new_e->id == AC_ACONST) {
        new_e = eval_array_constructor(new_e);
      }
    }
    add_to_list(new_e, &root);
  }
  return root;
}

static ACL *
eval_init_expr_item(ACL *cur_e)
{
  ACL *new_e = NULL;
  ACL *lop = NULL;
  ACL *rop = NULL;
  ACL *temp = NULL;
  int sptr;

  switch (cur_e->id) {
  case AC_AST:
    if (A_TYPEG(cur_e->u1.ast) == A_ID &&
        DTY(A_DTYPEG(cur_e->u1.ast)) == TY_ARRAY) {
      sptr = A_SPTRG(cur_e->u1.ast);
      if (PARAMG(sptr)) {
        if (STYPEG(sptr) != ST_PARAM) {
          sptr = NMCNSTG(sptr);
        }
        new_e = clone_init_const(get_getitem_p(CONVAL2G(sptr)), TRUE);
        new_e = eval_init_expr(new_e);
        break;
      } else {
        return 0;
      }
    }
    FLANG_FALLTHROUGH;
  case AC_CONST:
    new_e = clone_init_const(cur_e, TRUE);
    if (new_e->id == AC_AST) {
      new_e->id = AC_CONST;
      new_e->conval = get_const_from_ast(new_e->u1.ast);
    }
    break;
  case AC_ICONST:
    new_e = clone_init_const(cur_e, TRUE);
    break;
  case AC_IEXPR:
    if (cur_e->u1.expr->op != AC_INTR_CALL) {
      lop = eval_init_expr(cur_e->u1.expr->lop);
      rop = temp = cur_e->u1.expr->rop;
      if (temp && cur_e->u1.expr->op == AC_ARRAYREF &&
          temp->u1.expr->op == AC_TRIPLE) {
        rop = eval_const_array_triple_section(temp);
      } else if (temp)
        rop = eval_init_expr(temp);
    } else {
      lop = cur_e->u1.expr->lop;
      rop = cur_e->u1.expr->rop;
    }
    new_e = eval_init_op(cur_e->u1.expr->op, lop, cur_e->u1.expr->lop->dtype,
                         rop, rop ? cur_e->u1.expr->rop->dtype : 0, cur_e->sptr,
                         cur_e->dtype);
    break;
  case AC_ACONST:
    new_e = clone_init_const(cur_e, TRUE);
    new_e->subc = eval_array_constructor(cur_e);
    if (new_e->subc)
      new_e->subc = convert_acl_dtype(new_e->subc, DDTG(new_e->subc->dtype),
                                      DDTG(new_e->dtype));
    break;
  case AC_SCONST:
    new_e = clone_init_const(cur_e, TRUE);
    new_e->subc = eval_init_expr(new_e->subc);
    break;
  case AC_IDO:
    new_e = eval_do(cur_e);
    break;
  case AC_CONVAL:
    new_e = cur_e;
    break;
  default:
    /* MORE internal error */
    break;
  }

  return new_e;
}

ACL *
eval_init_expr(ACL *e)
{
  ACL *root = NULL;
  ACL *cur_e;
  ACL *new_e;

  for (cur_e = e; cur_e; cur_e = cur_e->next) {
    switch (cur_e->id) {
    case AC_SCONST:
      new_e = clone_init_const(cur_e, TRUE);
      new_e->subc = eval_init_expr(new_e->subc);
      if (!new_e->subc) {
        return 0;
      }
      if (new_e->subc->dtype == cur_e->dtype) {
        new_e->subc = new_e->subc->subc;
      }
      break;
    case AC_ACONST:
      new_e = clone_init_const(cur_e, TRUE);
      new_e->subc = eval_array_constructor(cur_e);
      if (new_e->subc)
        new_e->subc = convert_acl_dtype(new_e->subc, DDTG(new_e->subc->dtype),
                                        DDTG(new_e->dtype));
      break;
    default:
      new_e = eval_init_expr_item(cur_e);
      break;
    }
    if (!new_e) {
      return 0;
    }
    add_to_list(new_e, &root);
  }

  return root;
}

static ACL *
eval_do(ACL *ido)
{
  INT i;
  DOINFO *di = ido->u1.doinfo;
  INT initval;
  INT limitval;
  INT stepval;
  int idx_sptr = di->index_var;
  ACL *root = NULL;
  ACL *ict;
  INT num[2];
  INT sav_conval1 = CONVAL1G(idx_sptr);
  int inflag = 0;

  initval = dinit_eval(di->init_expr);
  if (sem.dinit_error) {
    interr("Non-constant implied DO initial value", di->init_expr, 3);
    return 0;
  }

  limitval = dinit_eval(di->limit_expr);
  if (sem.dinit_error) {
    interr("Non-constant implied DO limit value", di->init_expr, 3);
    return 0;
  }

  stepval = dinit_eval(di->step_expr);
  if (sem.dinit_error) {
    interr("Non-constant implied DO step value", di->init_expr, 3);
    return 0;
  }

  if (stepval >= 0) {
    for (i = initval; i <= limitval; i += stepval) {
      switch (DTY(DTYPEG(idx_sptr))) {
      case TY_INT8:
      case TY_LOG8:
        ISZ_2_INT64(i, num);
        /* implied do loop index variable is not A_CNST,
         * it is A_ID, so put it in CONVAL1P, so that
         * get_const_from_ast get it right.
         */
        CONVAL1P(idx_sptr, getcon(num, DTYPEG(idx_sptr)));
        break;
      default:
        CONVAL1P(idx_sptr, i);
        break;
      }

      ict = eval_init_expr(ido->subc);
      if (!ict) {
        return 0;
      }
      ict->u1.ast = mk_cval1(ict->conval, ict->dtype);
      add_to_list(ict, &root);
      inflag = 1;
    }
  } else {
    for (i = initval; i >= limitval; i += stepval) {
      switch (DTY(DTYPEG(idx_sptr))) {
      case TY_INT8:
      case TY_LOG8:
        ISZ_2_INT64(i, num);
        CONVAL1P(idx_sptr, getcon(num, DTYPEG(idx_sptr)));
        break;
      default:
        CONVAL1P(idx_sptr, i);
        break;
      }
      ict = eval_init_expr(ido->subc);
      if (!ict) {
        return 0;
      }
      ict->u1.ast = mk_cval1(ict->conval, ict->dtype);
      add_to_list(ict, &root);
      inflag = 1;
    }
  }
  if (inflag == 0 && ido->subc) {
    ict = eval_init_expr(ido->subc);
    add_to_list(ict, &root);
  }

  CONVAL1P(idx_sptr, sav_conval1);

  return root;
}

static INT
get_default_int_val(INT r)
{
  INT tmp[2];
  if (DTY(stb.user.dt_int) != TY_INT8) {
    return r;
  }
  tmp[1] = r;
  if (r >= 0)
    tmp[0] = 0;
  else
    tmp[0] = -1;
  return getcon(tmp, DT_INT8);
}

VAR *
gen_varref_var(int ast, DTYPE dtype)
{
  SST tmp_sst;
  VAR *ivl;

  SST_IDP(&tmp_sst, S_IDENT);
  SST_ASTP(&tmp_sst, ast);
  SST_DTYPEP(&tmp_sst, dtype);
  SST_SHAPEP(&tmp_sst, A_SHAPEG(ast));
  ivl = dinit_varref(&tmp_sst);

  return ivl;
}

/** \brief Process an AC_TYPEINIT.

    Look for an initialization template for this type.  If one already exists
    then return it.  Otherwise build one (and return it).
 */
SPTR
get_dtype_init_template(DTYPE dtype)
{
  DTYPE element_dtype =
      is_array_dtype(dtype) ? array_element_dtype(dtype) : dtype;
  SPTR tag_sptr = get_struct_tag_sptr(element_dtype);
  int init_ict = get_struct_initialization_tree(element_dtype);
  ACL *aclp, *tmpl_aclp;
  SPTR sptr = NOSYM;
  char namebuf[128];
  const char prefix[] = "_dtInit";

  assert(DTY(element_dtype) == TY_DERIVED,
         "get_dtype_init_template: element dtype not derived", dtype,
         ERR_Fatal);
  aclp = get_getitem_p(init_ict);
  if (aclp) {
    assert(eq_dtype(DDTG(aclp->dtype), element_dtype),
           "get_dtype_init_template: element dtype mismatch", dtype, ERR_Fatal);
  }

  if (is_unresolved_parameterized_dtype(element_dtype))
    return NOSYM;

  if (tag_sptr > NOSYM) {
    if ((sptr = TYPDEF_INITG(tag_sptr)) > NOSYM &&
        (SCG(sptr) == SC_STATIC || SCG(sptr) == SC_CMBLK)) {
      /* Reuse an existing initialization template object. */
      return sptr;
    }
  }
  snprintf(namebuf, sizeof namebuf, ".%s%04d", prefix, (int)element_dtype);
  namebuf[sizeof namebuf - 1] = '\0'; /* Windows snprintf bug workaround */

  /* no existing initialization template yet for this derived type; build one */
  if (aclp) {
    sptr = getccssym_sc(prefix, (int)element_dtype, ST_VAR, SC_STATIC);
    DTYPEP(sptr, element_dtype);
    DCLDP(sptr, TRUE);
    INITIALIZERP(sptr, TRUE);

    tmpl_aclp = GET_ACL(15);
    *tmpl_aclp = *aclp;
    tmpl_aclp->sptr = sptr;
    dinit((VAR *)NULL, tmpl_aclp);
    if (tag_sptr > NOSYM)
      TYPDEF_INITP(tag_sptr, sptr);
  }
  return sptr;
}

void
gen_derived_type_alloc_init(ITEM *itemp)
{
  int ast = itemp->ast;
  DTYPE dtype = A_DTYPEG(ast);
  ACL *aclp;
  SPTR prototype;
  int ict = get_struct_initialization_tree(dtype);

  if (ict == 0)
    return;

  if ((aclp = get_getitem_p(ict)) && aclp->dtype &&
      (!dtype || !has_type_parameter(aclp->dtype)))
    dtype = aclp->dtype;

  /* TODO: use init_derived_type() from semfin.c here instead? */
  prototype = get_dtype_init_template(dtype);
  if (prototype > NOSYM) {
    int src_ast = mk_id(prototype);
    add_stmt(mk_assn_stmt(itemp->ast, src_ast, A_DTYPEG(itemp->ast)));
  }
}

static int firstalloc;

void
check_dealloc_clauses(ITEM *list, ITEM *spec)
{
  ITEM *itemp;
  int stat = 0;
  int errmsg = 0;

  if (list == 0)
    list = ITEM_END;
  if (spec == 0)
    spec = ITEM_END;
  firstalloc = 1;
  for (itemp = spec; itemp != ITEM_END; itemp = itemp->next) {
    switch (itemp->t.conval) {
    case TK_STAT:
      if (stat == 1)
        error(155, 2, gbl.lineno, "Multiple STAT specifiers", CNULL);
      stat++;
      break;
    case TK_ERRMSG:
      if (errmsg == 1)
        error(155, 2, gbl.lineno, "Multiple ERRMSG specifiers", CNULL);
      errmsg++;
      break;
    default:
      error(155, 3, gbl.lineno, tokname[itemp->t.conval],
            "specifier invalid in DEALLOCATE");
    }
  }
}

void
check_alloc_clauses(ITEM *list, ITEM *spec, int *srcast, int *mold_or_src)
{
  ITEM *itemp;
  int stat = 0;
  int errmsg = 0;
  int source = 0;

  *srcast = 0;
  *mold_or_src = 0;

  if (list == 0)
    list = ITEM_END;
  if (spec == 0)
    spec = ITEM_END;
  firstalloc = 1;
  for (itemp = spec; itemp != ITEM_END; itemp = itemp->next) {
    switch (itemp->t.conval) {
    case TK_STAT:
      if (stat == 1)
        error(155, 2, gbl.lineno, "Multiple STAT specifiers", CNULL);
      stat++;
      break;
    case TK_ERRMSG:
      if (errmsg == 1)
        error(155, 2, gbl.lineno, "Multiple ERRMSG specifiers", CNULL);
      errmsg++;
      break;
    case TK_SOURCE:
    case TK_MOLD:
      if (source == 1)
        error(155, 2, gbl.lineno, "Multiple SOURCE/MOLD specifiers", CNULL);
      source++;
      *srcast = itemp->ast;
      *mold_or_src = itemp->t.conval;
      break;
    case TK_ALIGN:
      break;
    }
  }
}

int
gen_alloc_dealloc(int stmtyp, int object, ITEM *spec)
{
  int ast;
  ITEM *itemp;
  int sptr, objectsptr;
  DTYPE dtype;
  int stmt;
  int store_stat = 0;
  int store_pinned = 0;
  int len_stmt;

  if (spec == 0)
    spec = ITEM_END;
  objectsptr = sym_of_ast(object);
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, stmtyp); /* TK_ALLOCATE/TK_DEALLOCATE */
  A_SRCP(ast, object); /* object (ast) to be allocated/deallocated */
  A_FIRSTALLOCP(ast, firstalloc);
  firstalloc = 0;
  for (itemp = spec; itemp != ITEM_END; itemp = itemp->next) {
    switch (itemp->t.conval) {
    case TK_STAT:
      sptr = sym_of_ast(itemp->ast);
      dtype = DTYPEG(sptr);
      if (DTYG(dtype) == TY_INT8) {
        int tmp;
        tmp = mk_id(get_temp(DT_INT4));
        store_stat = mk_assn_stmt(itemp->ast, tmp, dtype);
        itemp->ast = tmp;
      }
      if (dtype != DT_INT && flg.standard && !XBIT(124, 0x10))
        error(155, 2, gbl.lineno, "Invalid type for STATUS specifier",
              SYMNAME(sptr));
      A_LOPP(ast, itemp->ast);
      break;
    case TK_ERRMSG:
      A_M3P(ast, itemp->ast);
      break;
    case TK_SOURCE:
    case TK_MOLD:
      A_STARTP(ast, itemp->ast);
      break;
    case TK_ALIGN:
      A_ALIGNP(ast, itemp->ast);
      break;
    }
  }
  stmt = add_stmt(ast);

  sem.alloc_std = stmt; /* std of allocate */

  /* This is for allocate statement, must set length before allocate
   * sem.gcvlen supposedly gets set only when it is character
   */
  if (is_deferlenchar_ast(object) &&
      stmtyp == TK_ALLOCATE) {
    if (sem.gcvlen) {
      len_stmt =
          mk_assn_stmt(get_len_of_deferchar_ast(object), sem.gcvlen, DT_INT);
      stmt = add_stmt_before(len_stmt, stmt);
    } else {
#if DEBUG
      assert(sem.gcvlen != 0, "gen_alloc_dealloc: character size missing", 3,
             object);
#endif
    }
  }

  if (store_stat) {
    stmt = add_stmt_after(store_stat, stmt);
  }
  if (store_pinned) {
    add_stmt_after(store_pinned, stmt);
  }

  return ast;
}

/** \brief If temps were allocated while processing the expression, the
   expression
           needs to be assigned to a temp, the allocatable temps need to be
           deallocated, and the use of the expression is replaced by the temp.
 */
int
check_etmp(SST *stkp)
{
  int new, ast;

  sem.use_etmps = FALSE;
  if (sem.etmp_list == NULL)
    return SST_ASTG(stkp);
  /*
   * Create a new temp, generate an assignment of the expression to
   * the temp.
   */
  ast = sem_tempify(stkp);
  (void)add_stmt(ast);
  new = A_DESTG(ast);
  gen_dealloc_etmps();
  return new;
}

void
gen_dealloc_etmps(void)
{
  int sptr;

  while (sem.etmp_list) {
    /* insert a deallocate for the symbol at this item */
    sptr = sem.etmp_list->t.sptr;
    if (sptr)
      gen_alloc_dealloc(TK_DEALLOCATE, mk_id(sptr), 0);
    sem.etmp_list = sem.etmp_list->next;
  }
  sem.use_etmps = FALSE;
}

void
check_and_add_auto_dealloc_from_ast(int ast)
{
  int sptr = sym_of_ast(ast);

  check_and_add_auto_dealloc(sptr);
}

void
check_and_add_auto_dealloc(int sptr)
{
  if (gbl.rutype != RU_FUNC && gbl.rutype != RU_SUBR && !CONSTRUCTSYMG(sptr))
    return;
  if (SCG(sptr) != SC_BASED)
    return;
  if (!ALLOCG(sptr) || POINTERG(sptr) || SAVEG(sptr) || sem.savall)
    return;
  if (!ALLOCATTRG(sptr) && MIDNUMG(sptr) && PTRVG(MIDNUMG(sptr)))
    return;
  if (MIDNUMG(sptr))
    switch (SCG(MIDNUMG(sptr))) {
    case SC_CMBLK:
    case SC_PRIVATE:
      return;
    default:
      break;
    }
  if (sem.scope_stack &&
      SCOPEG(sptr) == sem.scope_stack[sem.scope_level].sptr) {
    add_auto_dealloc(sptr);
  }
}

void
add_auto_dealloc(int sptr)
{
  ITEM *itemp;
  for (itemp = sem.auto_dealloc; itemp; itemp = itemp->next) {
    if (itemp->t.sptr == sptr) {
      return;
    }
  }
  itemp = (ITEM *)getitem(15, sizeof(ITEM));
  itemp->t.sptr = sptr;
  itemp->next = sem.auto_dealloc;
  sem.auto_dealloc = itemp;
}

static void
add_alloc_mem_initialize(int sptr)
{
  ITEM *itemp;

  if (DTY(DTYPEG(sptr)) != TY_DERIVED || ALLOCATTRG(sptr) || POINTERG(sptr) ||
      !allocatable_member(sptr))
    return;

  for (itemp = sem.alloc_mem_initialize; itemp; itemp = itemp->next) {
    if (itemp->t.sptr == sptr) {
      return;
    }
  }
  itemp = (ITEM *)getitem(15, sizeof(ITEM));
  itemp->t.sptr = sptr;
  itemp->next = sem.alloc_mem_initialize;
  sem.alloc_mem_initialize = itemp;
}

void
add_type_param_initialize(int sptr)
{
  ITEM *itemp;
  DTYPE dtype = DTYPEG(sptr);
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  if (DTY(dtype) != TY_DERIVED || !has_type_parameter(dtype))
    return;
  for (itemp = sem.type_initialize; itemp; itemp = itemp->next) {
    if (itemp->t.sptr == sptr) {
      return;
    }
  }
  itemp = (ITEM *)getitem(15, sizeof(ITEM));
  itemp->t.sptr = sptr;
  itemp->next = sem.type_initialize;
  sem.type_initialize = itemp;
}

void
add_auto_finalize(int sptr)
{
  ITEM *itemp;
  for (itemp = sem.auto_finalize; itemp; itemp = itemp->next) {
    if (itemp->t.sptr == sptr) {
      return;
    }
  }
  itemp = (ITEM *)getitem(15, sizeof(ITEM));
  itemp->t.sptr = sptr;
  itemp->next = sem.auto_finalize;
  sem.auto_finalize = itemp;
}

int
gen_finalization_for_sym(int sptr, int std, int memAst)
{
  int fsptr;
  int argt;
  int ast;
  int desc;
  DTYPE dtype;
  int tag, st_type;
  FtnRtlEnum rtlRtn;

  if (SAVEG(sptr) || sem.savall || !has_finalized_component(sptr))
    return std; /* no finalization needed */

  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    if (SDSCG(sptr) == 0) {
      get_static_descriptor(sptr);
      std = add_stmt_after(mk_stmt(A_CONTINUE, 0), std);
      std = init_sdsc(sptr, DTYPEG(sptr), std, 0);
    }
    desc = SDSCG(sptr);

    dtype = DTYPEG(sptr);

    dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED) {
      int arg0;
      tag = DTY(dtype + 3);
      st_type = get_static_type_descriptor(tag);
      arg0 = check_member(memAst, mk_id(desc));
      std = gen_set_type(arg0, mk_id(st_type), std, FALSE, FALSE);
    }
  } else {
    desc = get_type_descr_arg(gbl.currsub, sptr);
  }
  rtlRtn = RTE_finalize;
  fsptr = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE);
  argt = mk_argt(2);

  ARGT_ARG(argt, 0) = check_member(memAst, mk_id(sptr));
  ARGT_ARG(argt, 1) = check_member(memAst, mk_id(desc));

  ast = mk_id(fsptr);
  ast = mk_func_node(A_CALL, ast, 2, argt);
  std = add_stmt_after(ast, std);
  return std;
}

static int
get_parm_ast(int parent, SPTR sptr, DTYPE dtype)
{
  int mem, rslt, ast;
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      ast = mk_member(parent, mk_id(mem), dtype);
      rslt = get_parm_ast(ast, sptr, DTYPEG(mem));
      if (rslt)
        return rslt;
    }
    if (strcmp(SYMNAME(sptr), SYMNAME(mem)) == 0) {
      ast = mk_member(parent, mk_id(mem), /*dtype*/ DTYPEG(mem));
      return ast;
    }
  }
  /* Field not found, so try again and take more recursive approach */
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    /* Act only on the cases that were not considered before */
    if ((!PARENTG(mem)) && strcmp(SYMNAME(sptr), SYMNAME(mem))) {
      ast = mk_member(parent, mk_id(mem), dtype);
      rslt = get_parm_ast(ast, sptr, DTYPEG(mem)); /* recurse here */
      if (rslt)
        return rslt;
    }
  }
  return 0;
}

static int
remove_parent_from_ast(int ast)
{
  int i, newast, newast2, nargs, newargs, orig_args;
  int asd;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      orig_args = A_ARGSG(ast);
      newast = remove_parent_from_ast(ARGT_ARG(orig_args, 0));
      newast2 = mk_stmt(A_INTR, A_DTYPEG(ast));
      A_OPTYPEP(newast2, A_OPTYPEG(ast));
      nargs = A_ARGCNTG(ast);
      newargs = mk_argt(nargs);
      ARGT_ARG(newargs, 0) = newast;
      for (i = 1; i < nargs; ++i)
        ARGT_ARG(newargs, i) = ARGT_ARG(orig_args, i);
      A_ARGSP(newast2, newargs);
      A_ARGCNTP(newast2, nargs);
      ast = newast;
    }
    break;
  case A_MEM:
    ast = mk_id(memsym_of_ast(ast));
    break;
  case A_CNST:
    break;
  case A_ID:
    break;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    newast = remove_parent_from_ast(A_LOPG(ast));
    ast = mk_subscr_copy(newast, asd, A_DTYPEG(newast));
    break;
  case A_UNOP:
    newast = remove_parent_from_ast(A_LOPG(ast));
    ast = mk_unop(A_OPTYPEG(ast), newast, A_DTYPEG(ast));
    break;
  case A_CONV:
    newast = remove_parent_from_ast(A_LOPG(ast));
    ast = mk_convert(newast, A_DTYPEG(ast));
    break;
  case A_BINOP:
    newast = remove_parent_from_ast(A_LOPG(ast));
    newast2 = remove_parent_from_ast(A_ROPG(ast));
    ast = mk_binop(A_OPTYPEG(ast), newast, newast2, A_DTYPEG(ast));
    break;
  default:
    interr("remove_parent_from_ast: unexpected ast type", A_TYPEG(ast), 3);
  }
  return ast;
}

int
add_parent_to_bounds(int parent, int ast)
{
  int newast, i;
  if (parent == 0)
    return ast;
  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      newast = add_parent_to_bounds(parent, ARGT_ARG(i, 0));
      ARGT_ARG(i, 0) = newast;
    }
    break;
  case A_MEM:
    if (A_PARENTG(ast) == parent) {
      break;
    }

    if (!A_PARENTG(ast)) {
      A_PARENTP(ast, parent);
      break;
    }

    newast = add_parent_to_bounds(parent, A_PARENTG(ast));
    if (newast)
      A_PARENTP(ast, newast);

    break;
  case A_CNST:
    break;
  case A_ID:
    newast = get_parm_ast(parent, sym_of_ast(ast), DTYPEG(sym_of_ast(parent)));
    if (newast)
      ast = newast;
    break;
  case A_SUBSCR:
  case A_UNOP:
  case A_CONV:
    newast = add_parent_to_bounds(parent, A_LOPG(ast));
    A_LOPP(ast, newast);
    break;
  case A_BINOP:
    newast = add_parent_to_bounds(parent, A_LOPG(ast));
    A_LOPP(ast, newast);
    newast = add_parent_to_bounds(parent, A_ROPG(ast));
    A_ROPP(ast, newast);
    break;
  default:
    interr("add_parent_to_bounds: unexpected ast type", A_TYPEG(ast), 3);
  }
  return ast;
}

int
fix_mem_bounds(int parent, int mem)
{
  ADSC *ad;
  int numdim, i, bndast;
  int all_cnst;
  int zbase;

  ad = AD_DPTR(DTYPEG(mem));
  numdim = AD_NUMDIM(ad);
  all_cnst = 1;
  zbase = AD_ZBASE(ad);
  if (zbase && A_TYPEG(zbase)) {
    AD_ZBASE(ad) = add_parent_to_bounds(parent, zbase);
  }
  for (i = 0; i < numdim; i++) {
    bndast = AD_LWAST(ad, i);
    if (bndast) {
      AD_LWAST(ad, i) = add_parent_to_bounds(parent, bndast);
      if (A_TYPEG(AD_LWAST(ad, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = AD_UPAST(ad, i);
    if (bndast) {
      AD_UPAST(ad, i) = add_parent_to_bounds(parent, bndast);
      if (A_TYPEG(AD_UPAST(ad, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = AD_EXTNTAST(ad, i);
    if (bndast) {
      AD_EXTNTAST(ad, i) = add_parent_to_bounds(parent, bndast);
    }
  }

  return all_cnst;
}

int
fix_mem_bounds2(int parent, int mem)
{
  ADSC *ad, *bd;
  int numdim, i, bndast;
  int all_cnst;
  int zbase;
  int mem_dtype;
  int new_dtype;

  /* This function is the same as fix_mem_bounds() above except we
   * assign a new dtype with mem that includes a new array descriptor.
   * Otherwise, we may overwrite a shared array descriptor with new
   * bounds information.
   */

  mem_dtype = new_dtype = DTYPEG(mem);
  new_dtype = dup_array_dtype(new_dtype);

  numdim = ADD_NUMDIM(mem_dtype);
  get_aux_arrdsc(new_dtype, numdim);
  bd = AD_DPTR(new_dtype);
  ad = AD_DPTR(mem_dtype);

  /* Step 1: Construct bd w/ fields from mem_dtype minus any existing parent */

  all_cnst = 1;
  zbase = ADD_ZBASE(mem_dtype);
  if (zbase && A_TYPEG(zbase)) {
    AD_ZBASE(bd) = remove_parent_from_ast(zbase);
  }

  for (i = 0; i < numdim; i++) {
    bndast = ADD_LWAST(mem_dtype, i);
    if (bndast) {
      AD_LWBD(bd, i) = AD_LWAST(bd, i) = remove_parent_from_ast(bndast);
      if (A_TYPEG(ADD_LWAST(mem_dtype, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = ADD_UPAST(mem_dtype, i);
    if (bndast) {
      AD_UPBD(bd, i) = AD_UPAST(bd, i) = remove_parent_from_ast(bndast);
      if (A_TYPEG(ADD_UPAST(mem_dtype, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = ADD_EXTNTAST(mem_dtype, i);
    if (bndast) {
      AD_EXTNTAST(bd, i) = remove_parent_from_ast(bndast);
    }
  }

  if (all_cnst)
    return 1;

  AD_DEFER(bd) = AD_DEFER(ad);
  /* Step 2: Fill in parent into new array descriptor */
  ad = bd;

  all_cnst = 1;
  zbase = AD_ZBASE(ad);
  if (zbase && A_TYPEG(zbase)) {
    AD_ZBASE(ad) = add_parent_to_bounds(parent, zbase);
  }
  for (i = 0; i < numdim; i++) {
    bndast = AD_LWAST(ad, i);
    if (bndast) {
      AD_LWAST(ad, i) = add_parent_to_bounds(parent, bndast);
      if (A_TYPEG(AD_LWAST(ad, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = AD_UPAST(ad, i);
    if (bndast) {
      AD_UPAST(ad, i) = add_parent_to_bounds(parent, bndast);
      if (A_TYPEG(AD_UPAST(ad, i)) != A_CNST)
        all_cnst = 0;
    }
    bndast = AD_EXTNTAST(ad, i);
    if (bndast) {
      AD_EXTNTAST(ad, i) = add_parent_to_bounds(parent, bndast);
    }
  }

  DTYPEP(mem, new_dtype);

  return all_cnst;
}

/*
 * insert an assignment statement
 */
static int
insert_assign(int lhs, int rhs, int std)
{
  int newasn, newstd;
  if (lhs == rhs)
    return std;
  newasn = mk_assn_stmt(lhs, rhs, 0);
  newstd = add_stmt_after(newasn, std);
  return newstd;
} /* insert_assign */

static int
get_header_member(int sdsc_ast, int info)
{
  int ast;
  int subs[1];

  subs[0] = mk_isz_cval(info, astb.bnd.dtype);
  ast = mk_subscr(sdsc_ast, subs, 1, astb.bnd.dtype);
  return ast;
}

static int
size_of_dtype(DTYPE dtype, SPTR sptr, int memberast)
{
  int sizeAst;
  if (DTY(dtype) == TY_CHAR) {
    /* assumed length character */
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR) {
      sizeAst = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), astb.bnd.dtype);
      sizeAst = begin_call(A_FUNC, sizeAst, 1);
      add_arg(check_member(memberast, mk_id(sptr)));
    } else {
      int clen;
      clen = DTY(dtype + 1);
      if (A_ALIASG(clen)) {
        sizeAst = A_ALIASG(clen);
      } else {
        sizeAst = clen;
      }
      sizeAst = mk_bnd_int(sizeAst);
    }
  } else {
    sizeAst = mk_isz_cval(size_of(dtype), astb.bnd.dtype);
  }
  return sizeAst;
}

int
init_sdsc(int sptr, DTYPE dtype, int before_std, int parent_sptr)
{
  int sptrsdsc = SDSCG(sptr);
  ADSC *ad = AD_DPTR(dtype);
  int ndims = AD_NUMDIM(ad);
  int nargs = 5 + ndims * 2;
  int argt = mk_argt(nargs);
  int fsptr = sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE);
  int sptrsdsc_arg, ast, i, std;

  assert(sptrsdsc > NOSYM, "init_sdsc: sptr has no SDSC", sptr, ERR_Fatal);
  sptrsdsc_arg = mk_id(sptrsdsc);
  if (STYPEG(sptrsdsc) == ST_MEMBER) {
    assert(STYPEG(sptrsdsc) != ST_MEMBER || parent_sptr > NOSYM,
           "init_sdsc: sptrdsc is member but no parent sptr", sptrsdsc,
           ERR_Fatal);
    sptrsdsc_arg = mk_member(mk_id(parent_sptr), sptrsdsc_arg, dtype);
  }

  /* call RTE_template(desc, rank, flags, kind, len,  {lb, ub}+) */
  ARGT_ARG(argt, 0) = sptrsdsc_arg;
  ARGT_ARG(argt, 1) = mk_isz_cval(ndims, astb.bnd.dtype);
  ARGT_ARG(argt, 2) = mk_isz_cval(0, astb.bnd.dtype);
  ARGT_ARG(argt, 3) = mk_isz_cval(dtype_to_arg(dtype + 1), astb.bnd.dtype);
  ARGT_ARG(argt, 4) = size_of_dtype(DDTG(dtype), sptr, 0);

  for (i = 0; i < ndims; ++i) {
    ARGT_ARG(argt, 5 + 2 * i) = AD_LWAST(ad, i);
    ARGT_ARG(argt, 6 + 2 * i) = AD_UPAST(ad, i);
  }

  ast =
      mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE)),
                   nargs, argt);
  SDSCINITP(sptr, TRUE);
  A_DTYPEP(ast, DT_INT);
  NODESCP(fsptr, TRUE);
  std = add_stmt_before(ast, before_std);

  /* call pghpf_instance(dest desc, targ desc, kind,len, 0) */
  argt = mk_argt(nargs = 5);
  ARGT_ARG(argt, 0) = sptrsdsc_arg;
  ARGT_ARG(argt, 1) = sptrsdsc_arg;
  ARGT_ARG(argt, 2) = mk_isz_cval(dtype_to_arg(dtype + 1), astb.bnd.dtype);
  ARGT_ARG(argt, 3) = size_of_dtype(DDTG(dtype), sptr, ast);
  ARGT_ARG(argt, 4) = mk_isz_cval(0, astb.bnd.dtype);

  ast =
      mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_instance), DT_NONE)),
                   nargs, argt);
  return add_stmt_after(ast, std);
}

/** \brief Similar to init_sdsc() above, but it's also used to initialize 
 *         a descriptor's bounds from a subscript expression.
 *
 * \param sptr is the symbol table pointer of the symbol with the descriptor
 *        to initialize.
 * \param dtype is the dtype used for initializing the descriptor.
 * \param before_std is the statement descriptor where we want to insert the
 *        initialization code (inserted before this std).
 * \param parent_sptr is the symbol table pointer of the enclosing object 
 *        if sptr is an ST_MEMBER. Otherwise, it can be 0.
 * \param subscr is an AST representing the subscript expression that contains
 *        the array bounds. If it's not an A_SUBSCR, then init_sdsc() is
 *        called instead.
 * \param td_ast is an AST representing the descriptor that we are creating
 *        an instance of.
 *
 * \return a statement descriptor of the generated statements.
 */
int
init_sdsc_bounds(SPTR sptr, DTYPE dtype, int before_std, SPTR parent_sptr,
                 int subscr, int td_ast)
{
  SPTR sptrsdsc = SDSCG(sptr);
  ADSC *ad = AD_DPTR(dtype);
  int ndims = AD_NUMDIM(ad);
  int nargs = 5 + ndims * 2;
  int argt = mk_argt(nargs);
  SPTR fsptr = sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE);
  int sptrsdsc_arg, ast, i, std;
  int asd, triplet, stride;

  if (!subscr || A_TYPEG(subscr) != A_SUBSCR) {
    return init_sdsc(sptr, dtype, before_std, parent_sptr);
  }
  assert(sptrsdsc > NOSYM, "init_sdsc_bounds: sptr has no SDSC", sptr, 
         ERR_Fatal);
  sptrsdsc_arg = mk_id(sptrsdsc);
  if (STYPEG(sptrsdsc) == ST_MEMBER) {
    assert(STYPEG(sptrsdsc) != ST_MEMBER || parent_sptr > NOSYM,
           "init_sdsc_bounds: sptrdsc is member but no parent sptr", sptrsdsc,
           ERR_Fatal);
    sptrsdsc_arg = mk_member(mk_id(parent_sptr), sptrsdsc_arg, dtype);
  }

  /* call RTE_template(desc, rank, flags, kind, len,  {lb, ub}+) */
  ARGT_ARG(argt, 0) = sptrsdsc_arg;
  ARGT_ARG(argt, 1) = mk_isz_cval(ndims, astb.bnd.dtype);
  ARGT_ARG(argt, 2) = mk_isz_cval(0, astb.bnd.dtype);
  ARGT_ARG(argt, 3) = mk_isz_cval(dtype_to_arg(dtype + 1), astb.bnd.dtype);
  ARGT_ARG(argt, 4) = size_of_dtype(DDTG(dtype), sptr, 0);

  asd = A_ASDG(subscr);
  for (i = 0; i < ndims; ++i) {
    triplet = ASD_SUBS(asd, i);
    if ((stride = A_STRIDEG(triplet)) != 0 && A_TYPEG(stride) == A_CNST &&
          ad_val_of(A_SPTRG(stride)) < 0) {
      ARGT_ARG(argt, 5 + 2 * i) = mk_bnd_int(A_UPBDG(triplet));
      ARGT_ARG(argt, 6 + 2 * i) = mk_bnd_int(A_LBDG(triplet));
    } else {
      ARGT_ARG(argt, 5 + 2 * i) = mk_bnd_int(A_LBDG(triplet));
      ARGT_ARG(argt, 6 + 2 * i) = mk_bnd_int(A_UPBDG(triplet));
    }
  }

  ast =
      mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE)),
                   nargs, argt);
  SDSCINITP(sptr, TRUE);
  A_DTYPEP(ast, DT_INT);
  NODESCP(fsptr, TRUE);
  std = add_stmt_before(ast, before_std);

  /* call pghpf_instance(dest desc, targ desc, kind,len, 0) */
  argt = mk_argt(nargs = 5);
  ARGT_ARG(argt, 0) = td_ast != 0 ? td_ast  : sptrsdsc_arg;
  ARGT_ARG(argt, 1) = sptrsdsc_arg;
  ARGT_ARG(argt, 2) = mk_isz_cval(dtype_to_arg(dtype + 1), astb.bnd.dtype);
  ARGT_ARG(argt, 3) = size_of_dtype(DDTG(dtype), sptr, ast);
  ARGT_ARG(argt, 4) = mk_isz_cval(0, astb.bnd.dtype);

  ast =
      mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_instance), DT_NONE)),
                   nargs, argt);
  return add_stmt_after(ast, std);
}

static int
genPolyAsn(int dest, int src, int std, int parentMem)
{
  int argt, flag_con, astdest, dest_sdsc_ast, astsrc, src_sdsc_ast, fsptr;
  int ast;

  astsrc = mk_id(src);

  if (!parentMem) {
    if (!SDSCG(dest))
      get_static_descriptor(dest);

    dest_sdsc_ast = mk_id(SDSCG(dest));

    astdest = mk_id(dest);
  } else {
    int sdsc_mem = get_member_descriptor(dest);
    if (sdsc_mem > NOSYM) {
      int parentDty = DTYPEG(sym_of_ast(parentMem));
      if (DTY(parentDty) == TY_ARRAY)
        parentDty = DTY(parentDty + 1);
      dest_sdsc_ast = check_member(parentMem, mk_id(sdsc_mem));
    } else {
      if (!SDSCG(dest)) {
        get_static_descriptor(dest);
      }
      dest_sdsc_ast = mk_id(SDSCG(dest));
    }

    astdest = check_member(parentMem, mk_id(dest));
  }

  src_sdsc_ast = mk_id(get_static_type_descriptor(src));
  if (dest_sdsc_ast) {
    std = gen_set_type(dest_sdsc_ast, src_sdsc_ast, std, FALSE, FALSE);
  }

  std = add_stmt_after(mk_stmt(A_CONTINUE, 0), std);
  std = init_sdsc(dest, parentMem ? A_DTYPEG(parentMem) : DTYPEG(dest), std,
                  parentMem ? sym_of_ast(parentMem) : 0);

  fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_poly_asn), DT_NONE);
  argt = mk_argt(5);
  flag_con = mk_cval1(1, DT_INT);
  flag_con = mk_unop(OP_VAL, flag_con, DT_INT);
  ARGT_ARG(argt, 4) = flag_con;

  ARGT_ARG(argt, 0) = astdest;
  ARGT_ARG(argt, 1) = dest_sdsc_ast;
  ARGT_ARG(argt, 2) = astsrc;
  ARGT_ARG(argt, 3) = src_sdsc_ast;
  ast = mk_id(fsptr);
  ast = mk_func_node(A_CALL, ast, 5, argt);
  std = add_stmt_after(ast, std);

  return std;
}

static int
gen_kind_parm_assignments(SPTR sptr, DTYPE dtype, int std, int flag)
{
  int mem, val, con;
  int ast, ast2;
  int sdsc_mem, i, j;
  int pass;
  int memDtype;
  int orig_dtype;
  int exit_std;
  static int parentMem = 0;
  static int firstAllocStd = 0;

  orig_dtype = dtype;
  if (DTY(dtype) == TY_ARRAY) {
    dtype = DTY(dtype + 1);
  }
  if (DTY(dtype) != TY_DERIVED ||
      (!flag && (ALLOCATTRG(sptr) || POINTERG(sptr)) && SCG(sptr) != SC_DUMMY))
    return std;
  if (STYPEG(sptr) == ST_ARRAY || DTY(orig_dtype) == TY_ARRAY) {
    /* This code creates an array of PDTs. It first creates a scalar PDT object.
     * We then recursively call gen_kind_parm_assignments() on that object to
     * initialize the components that use the PDT's type parameters.
     * The firstAllocStd static variable is set to the std of the first
     * init code of a component that uses one of more type parameters. If
     * firstAllocStd is not set (i.e., it's -1) after the call to
     * gen_kind_parm_assignments(), then just return std. In this case, we
     * have a PDT with type parameters, but no components that use those type
     * parameters. If firstAllocStd > -1, then we have a PDT that uses
     * the type parameters. We use our temporary PDT (i.e., tmp) to create an
     * array of these by cloning it into each element of the array. This is very
     * similar to sourced allocation (e.g.,allocate(pdt_array(n),source=pdt)).
     * In fact, we clone tmp by calling the RTE_poly_asn() rte routine.
     * This routine is also called when we perform sourced allocation.
     * Although our technique is similar to source allocation, this code also
     * works with non-allocatable arrays.
     */
    int tmp = getccsym_sc('d', sem.dtemps++, ST_VAR, SC_STATIC);
    DTYPEP(tmp, dtype);
    firstAllocStd = -1;
    gen_kind_parm_assignments(tmp, dtype, std, flag);
    if (firstAllocStd > -1) {
      std = firstAllocStd;
      std = genPolyAsn(sptr, tmp, std, parentMem);
    }
    firstAllocStd = std;
    return std;
  }
  for (pass = 0; pass <= 1; ++pass) {
    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      memDtype = DTYPEG(mem);
      if (pass && DTY(memDtype) == TY_ARRAY && has_type_parameter(memDtype)) {
        int origParentMem = parentMem;
        int eleDtype = DTY(memDtype + 1);
        parentMem = (!parentMem) ? mk_member(mk_id(sptr), mk_id(mem), eleDtype)
                                 : mk_member(parentMem, mk_id(mem), eleDtype);
        std = gen_kind_parm_assignments(mem, memDtype, std, flag);
        parentMem = origParentMem;
        continue;
      }
      if (SCG(sptr) == SC_DUMMY && !flag) {
        continue;
      }
      if (PARENTG(mem)) {
        std = gen_kind_parm_assignments(sptr, DTYPEG(mem), std, flag);
        continue;
      }
      if ((!LENPARMG(mem) || A_TYPEG(LENG(mem)) == A_CNST) && SETKINDG(mem) &&
          !USEKINDG(mem) && (val = KINDG(mem))) {
        if (!pass) {
          con = mk_cval1(val, DT_INT);
          ast = add_parent_to_bounds(mk_id(sptr), mk_id(mem));
          ast = mk_assn_stmt(ast, con, DT_INT);
          std = add_stmt_after(ast, std);
        }
      } else if (LENPARMG(mem) && SETKINDG(mem) && !USEKINDG(mem) &&
                 (val = KINDG(mem)) && LENG(mem)) {
        if (!pass) {
          ast = add_parent_to_bounds(mk_id(sptr), mk_id(mem));
          ast2 = LENG(mem);
          ast = mk_assn_stmt(ast, ast2, DT_INT);
          std = add_stmt_after(ast, std);
        }
      } else if (SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) &&
                 (val = PARMINITG(mem))) {
        if (!pass) {
          con = mk_cval1(val, DT_INT);
          ast = add_parent_to_bounds(mk_id(sptr), mk_id(mem));
          ast = mk_assn_stmt(ast, con, DT_INT);
          std = add_stmt_after(ast, std);
        }
      } else if (INITKINDG(mem) && (val = PARMINITG(mem))) {
        if (!pass) {
          if (!chk_kind_parm_expr(val, dtype, 0, 1)) {
            char *buf;
            int len;
            len = strlen("Initialization must be a constant"
                         " expression for component  in object") +
                  strlen(SYMNAME(mem)) + 1;
            buf = getitem(0, len);
            sprintf(buf,
                    "Initialization must be a constant"
                    " expression for component %s in object",
                    SYMNAME(mem));
            error(155, 3, gbl.lineno, buf, SYMNAME(sptr));
          } else {
            val = chk_kind_parm_set_expr(val, dtype);
            if (A_TYPEG(val) == A_CNST) {
              if (USELENG(mem)) {
                error(155, 4, gbl.lineno,
                      "Length type parameters may not be "
                      "used with type components that have default "
                      "initialization -",
                      SYMNAME(mem));
              }
              ast = add_parent_to_bounds(mk_id(sptr), mk_id(mem));
              ast = mk_assn_stmt(ast, val, DT_INT);
              std = add_stmt_after(ast, std);
            } else {
              char *buf;
              int len;
              len = strlen("Initialization must be a constant"
                           " expression for component  in object") +
                    strlen(SYMNAME(mem)) + 1;
              buf = getitem(0, len);
              sprintf(buf,
                      "Initialization must be a constant"
                      " expression for component %s in object",
                      SYMNAME(mem));
              error(155, 3, gbl.lineno, buf, SYMNAME(sptr));
            }
          }
        }
      } else if (USELENG(mem) &&
                 /*ALLOCG(mem) &&*/ DTY(DTYPEG(mem)) == TY_ARRAY) {
        if (pass) {
          i = mk_id(sptr);
          if (flag)
            fix_mem_bounds2(i, mem);

          ast = mk_stmt(A_ALLOC, 0);
          A_TKNP(ast, TK_ALLOCATE);
          j = mk_member(i, mk_id(mem), dtype);
          A_SRCP(ast, j);
          std = add_stmt_after(ast, std);
          if (firstAllocStd < 0)
            firstAllocStd = std;
          std = add_stmt_before(mk_stmt(A_CONTINUE, 0), std);
          std = init_sdsc(mem, DTYPEG(mem), std, sptr);

          if (!flag && (gbl.rutype != RU_PROG || CONSTRUCTSYMG(sptr))) {
            exit_std = CONSTRUCTSYMG(sptr) ? STD_PREV(BLOCK_EXIT_STD(sptr)) :
                                             gbl.exitstd;
            i = mk_stmt(A_ALLOC, 0);
            A_TKNP(i, TK_DEALLOCATE);
            A_SRCP(i, j);
            A_DALLOCMEMP(i, 1);
            add_stmt_after(i, exit_std);
          }
        }
      } else if (USELENG(mem) && ALLOCG(mem) && DTY(DTYPEG(mem)) == TY_CHAR &&
                 LENG(mem)) {
        if (pass) {
          int src_ast;

          sdsc_mem = SDSCG(mem);
          sdsc_mem = mk_member(mk_id(sptr), mk_id(sdsc_mem), dtype);
          sdsc_mem = get_header_member(sdsc_mem, get_byte_len_indx());

          ast = mk_stmt(A_ALLOC, 0);
          A_TKNP(ast, TK_ALLOCATE);
          src_ast = add_parent_to_bounds(mk_id(sptr), mk_id(mem));
          A_SRCP(ast, src_ast);
          std = add_stmt_after(ast, std);
          if (firstAllocStd < 0)
            firstAllocStd = std;

          std = insert_assign(sdsc_mem, LENG(mem), std);

          if (!flag && (gbl.rutype != RU_PROG || CONSTRUCTSYMG(sptr))) {
            exit_std = CONSTRUCTSYMG(sptr) ? STD_PREV(BLOCK_EXIT_STD(sptr)) :
                                             gbl.exitstd;
            i = mk_stmt(A_ALLOC, 0);
            A_TKNP(i, TK_DEALLOCATE);
            A_SRCP(i, A_SRCG(ast));
            A_DALLOCMEMP(i, 1);
            add_stmt_after(i, exit_std);
          }
        }
      } else if (!SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) &&
                 !PARMINITG(mem)) {
        int len;
        char *buf;
        len = strlen(SYMNAME(mem)) + strlen(SYMNAME(sptr)) +
              strlen("Missing value for kind type parameter  in") + 1;
        buf = getitem(0, len);
        sprintf(buf, "Missing value for kind type parameter %s in %s",
                SYMNAME(mem), SYMNAME(sptr));
        error(155, 3, gbl.lineno, buf, CNULL);
      }
    }
  }
  return std;
}

void
fix_type_param_members(SPTR sptr, DTYPE dtype)
{

  int mem, i, ast;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (USELENG(mem) && ALLOCG(mem) && DTY(DTYPEG(mem)) == TY_ARRAY) {
      i = mk_id(sptr);
      fix_mem_bounds(i, mem);
    } else if (USELENG(mem) && ALLOCG(mem) && DTY(DTYPEG(mem)) == TY_CHAR &&
               LENG(mem)) {
      ast = add_parent_to_bounds(mk_id(sptr), LENG(mem));
      LENP(mem, ast);
      DTY(DTYPEG(mem) + 1) = ast;
    }
  }
}

void
gen_type_initialize_for_sym(SPTR sptr, int std, int flag, DTYPE dtype2)
{
  DTYPE orig_dtype = dtype2 ? dtype2 : DTYPEG(sptr);
  DTYPE dtype = orig_dtype;

  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  if (DTY(dtype) == TY_DERIVED) {
    if (std < 0) {
      int ast = mk_stmt(A_CONTINUE, 0);
      std = add_stmt(ast);
    }
    gen_kind_parm_assignments(sptr, orig_dtype, std, flag);
  }
}

static void
gen_alloc_mem_initialize_for_sym2(int sptr, int std, int ast, int visit_flag)
{
  typedef struct visitDty {
    int dty;
    struct visitDty *next;
  } VISITDTY;

  static VISITDTY *visit_list = 0;
  VISITDTY *curr, *new_visit, *prev;

  int sptrmem, aast, mem_sptr_id, dtype, bast;

  dtype = (sptr) ? DTYPEG(sptr) : DTYPEG(memsym_of_ast(ast));

  if (DTY(dtype) != TY_DERIVED)
    return;

  if (visit_list) {
    for (curr = visit_list; curr; curr = curr->next) {
      if (curr->dty == dtype)
        return;
    }
  }

  NEW(new_visit, VISITDTY, 1);
  new_visit->dty = dtype;
  new_visit->next = visit_list;
  visit_list = new_visit;

  for (sptrmem = DTY(DDTG(dtype) + 1); sptrmem > NOSYM;
       sptrmem = SYMLKG(sptrmem)) {
    if (ALLOCATTRG(sptrmem)) {
      aast = mk_id(sptrmem);
      bast = (ast) ? ast : mk_id(sptr);
      mem_sptr_id = mk_member(bast, aast, DTYPEG(sptrmem));
      add_stmt_after(add_nullify_ast(mem_sptr_id), std);
    } else if (allocatable_member(sptrmem)) {
      aast = mk_id(sptrmem);
      bast = (ast) ? ast : mk_id(sptr);
      bast = mk_member(bast, aast, DTYPEG(sptrmem));
      gen_alloc_mem_initialize_for_sym2(0, std, bast, 1);
    }
  }

  if (!visit_flag && visit_list) {
    for (prev = curr = visit_list; curr;) {
      curr = curr->next;
      FREE(prev);
      prev = curr;
    }
    visit_list = 0;
  }
}

void
gen_alloc_mem_initialize_for_sym(int sptr, int std)
{
  gen_alloc_mem_initialize_for_sym2(sptr, std, 0, 0);
}

static void
__gen_conditional_dealloc(int do_cond, int dealloc_item, int after,
                          int test_presence)
{
  int argt;
  int ifast;
  int ast;
  int tsptr;
  int std;

  std = after;
  if (do_cond) {
    /* generate
     * if( allocated(itemp->t.sptr ) then
     *   deallocate(itemp->t.sptr)
     * ifend
     */
    int present;
    if (test_presence) {
      present = ast_intr(I_PRESENT, stb.user.dt_log, 1, dealloc_item);
      ifast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ifast, present);
      std = add_stmt_after(ifast, std);
    }
    argt = mk_argt(1);
    ARGT_ARG(argt, 0) = dealloc_item;
    tsptr = getsymbol("allocated");
    ast = mk_id(tsptr);
    A_DTYPEP(ast, A_DTYPEG(dealloc_item));
    ast = mk_func_node(A_INTR, ast, 1, argt);
    A_DTYPEP(ast, stb.user.dt_log);
    A_OPTYPEP(ast, I_ALLOCATED);
    ifast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifast, ast);
    std = add_stmt_after(ifast, std);
  }

  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_DEALLOCATE);
  A_SRCP(ast, dealloc_item);
  std = add_stmt_after(ast, std);

  if (do_cond) {
    std = add_stmt_after(mk_stmt(A_ENDIF, 0), std);
    if (test_presence)
      std = add_stmt_after(mk_stmt(A_ENDIF, 0), std);
  }
}

void
gen_conditional_dealloc(int do_cond, int dealloc_item, int after)
{
  __gen_conditional_dealloc(do_cond, dealloc_item, after, 0);
}

int
gen_conditional_alloc(int cond, int alloc_item, int after)
{
  int argt;
  int ifast;
  int ast;
  int tsptr;

  /* generate
   * if( allocated(cond) ) then
   *   allocate(alloc_item)
   * ifend
   */
  if (cond) {
    argt = mk_argt(1);
    ARGT_ARG(argt, 0) = cond;
    tsptr = getsymbol("allocated");
    ast = mk_id(tsptr);
    A_DTYPEP(ast, A_DTYPEG(cond));
    ast = mk_func_node(A_INTR, ast, 1, argt);
    A_DTYPEP(ast, stb.user.dt_log);
    A_OPTYPEP(ast, I_ALLOCATED);
    ifast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifast, ast);
    after = add_stmt_after(ifast, after);
  }

  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_ALLOCATE);
  A_LOPP(ast, 0);
  A_SRCP(ast, alloc_item);
  after = add_stmt_after(ast, after);

  if (cond)
    after = add_stmt_after(mk_stmt(A_ENDIF, 0), after);
  return after;
}

void
gen_conditional_dealloc_for_sym(int sptr, int std)
{
  int idast = mk_id(sptr);
  if (SCG(sptr) != SC_LOCAL) {
    if (flg.smp && gbl.internal > 1) {
      int scope = SCOPEG(sptr);
      if (scope && scope == SCOPEG(gbl.currsub)) {
        return;
      }
    }
    if (SCG(sptr) == SC_DUMMY && OPTARGG(sptr))
      __gen_conditional_dealloc(1, idast, std, 1);
    else
      __gen_conditional_dealloc(1, idast, std, 0);
  } else {
    /* must be derived type scalar or array which contains allocatable
     * components.
     */
    int ast;
    ast = mk_stmt(A_ALLOC, 0);
    A_TKNP(ast, TK_DEALLOCATE);
    A_SRCP(ast, idast);
    (void)add_stmt_after(ast, std);
  }
}

int
gen_dealloc_for_sym(int sptr, int std)
{
  int idast;
  int ast;
  int ss;

  idast = mk_id(sptr);
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_DEALLOCATE);
  A_SRCP(ast, idast);
  ss = add_stmt_after(ast, std);
  return ss;
}

/** \brief This function initializes the type in a descriptor for an object
 *         with an intrinsic type.
 *
 *  This function generates a call to set_intrin_type() before the statement
 *  descriptor, \param std. 
 *
 *  \param ast is the ast of the object that has a descriptor that needs to be
 *         initialized.
 *  \param sptr is the symbol table pointer of the object that has a descriptor
 *         that needs to be initialized.
 *  \param std is the statement descriptor that indicates where to add the call
 *         to set_intrin_type(). 
 *
 *  \return the std after the set_intrin_type() call.
 */
static int 
init_intrin_type_desc(int ast, SPTR sptr, int std)
{


  int type_ast;
  SPTR sdsc = STYPEG(sptr) == ST_MEMBER ? get_member_descriptor(sptr) : 
              SDSCG(sptr);
  int sdsc_ast = STYPEG(sptr) == ST_MEMBER ? 
                 check_member(ast, mk_id(sdsc)) :
                 mk_id(sdsc);
  DTYPE dtype = DDTG(DTYPEG(sptr));
  int intrin_type;

#if DEBUG
  assert(DT_ISBASIC(dtype), "init_intrin_type_desc: not basic dtype for ast", 
         ast, 4);
#endif
  intrin_type = mk_cval(dtype_to_arg(dtype), astb.bnd.dtype); 
  intrin_type = mk_unop(OP_VAL, intrin_type, astb.bnd.dtype);
  type_ast = mk_set_type_call(sdsc_ast, intrin_type, TRUE);
  std = add_stmt_after(type_ast, std);
  return std;
}

/** \brief Generate (re)allocation code for deferred length character objects
 *         and traditional character objects that are allocatable scalars.
 *
 *         This is typically used in generating (re)allocation code in
 *         an assignment to an allocatable/deferred length character object.
 *
 *         Reallocation code is generated for deferred length character
 *         objects.
 *
 *         For traditional character allocatable scalars, we allocate
 *         the object if it has not already been allocated; we do not
 *         generate reallocation code since the amount of space allocated
 *         is fixed with traditional character allocatable objects.
 *
 *         We update the character length descriptor information for
 *         both deferred length and traditional character objects. This
 *         is needed for proper I/O such as namelist processing.
 *
 *  \param lhs is the ast of the object getting (re)allocated.
 *  \param rhs is the ast of the object that supplies the character length.
 *  \param std is the statement descriptor where we insert the (re)allocation
 *         and/or length assignment code.
 */
void
gen_automatic_reallocation(int lhs, int rhs, int std)
{

  int ast, len_stmt;
  int tsptr;
  int argt;
  int ifast, innerifast, binopast;
  int lhs_len, rhs_len;
  DTYPE dtypedest = A_DTYPEG(lhs);

  /* generate the following for deferred length character objects:
   *
   * if( allocated(lhs) ) then
   *     if(len(lhs) .ne. len(rhs)) then
   *         deallocate(lhs)
   *         lhs$len = rhs$len
   *         allocate(lhs, len=lhs$len)
   *     ifend
   * else
   *   lhs$len = rhs$len
   *   allocate(lhs, len=lhs$len)
   * ifend
   *
   * generate the following for traditional character allocatable objects:
   *
   * if( allocated(lhs) ) then
   *     if(len(lhs) .ne. len(rhs)) then
   *       lhs$len = rhs$len
   *     ifend
   * else
   *   lhs$len = rhs$len
   *   allocate(lhs, len=the_declared_length)
   * ifend
   */

  ifast = mk_stmt(A_IFTHEN, 0);

  argt = mk_argt(1);
  ARGT_ARG(argt, 0) = lhs;
  tsptr = getsymbol("allocated");
  ast = mk_id(tsptr);
  A_DTYPEP(ast, A_DTYPEG(lhs));
  ast = mk_func_node(A_INTR, ast, 1, argt);
  A_DTYPEP(ast, stb.user.dt_log);
  A_OPTYPEP(ast, I_ALLOCATED);
  A_IFEXPRP(ifast, ast);
  std = add_stmt_before(ifast, std);

  innerifast = mk_stmt(A_IFTHEN, 0);
  A_IFSTMTP(ifast, innerifast);

  lhs_len = size_ast_of(lhs, DDTG(A_DTYPEG(lhs)));
  if (A_TYPEG(rhs) == A_FUNC) {
    /* need to get the interface from the A_FUNC ast. */
    int sym, iface = 0;
    sym = procsym_of_ast(A_LOPG(rhs));
    proc_arginfo(sym, NULL, NULL, &iface);
    rhs_len = string_expr_length(mk_id(iface));
  } else {
    rhs_len = string_expr_length(rhs);
  }
  binopast = mk_binop(OP_NE, lhs_len, rhs_len, DT_LOG);
  A_IFEXPRP(innerifast, binopast);
  std = add_stmt_after(innerifast, std);

  if (dtypedest == DT_DEFERCHAR || dtypedest == DT_DEFERNCHAR) {
    /* reallocation is only required for deferred length character objects */
    ast = mk_stmt(A_ALLOC, 0);
    A_IFSTMTP(innerifast, ast);

    A_TKNP(ast, TK_DEALLOCATE);
    A_SRCP(ast, lhs);
    std = add_stmt_after(ast, std);
  }

  len_stmt = mk_assn_stmt(get_len_of_deferchar_ast(lhs), rhs_len, DT_INT);
  std = add_stmt_after(len_stmt, std);

  if (dtypedest == DT_DEFERCHAR || dtypedest == DT_DEFERNCHAR) {
    /* reallocation is only required for deferred length character objects */
    ast = mk_stmt(A_ALLOC, 0);
    A_TKNP(ast, TK_ALLOCATE);
    A_SRCP(ast, lhs);
    A_FIRSTALLOCP(ast, 1);
    std = add_stmt_after(ast, std);
  }

  std = add_stmt_after(mk_stmt(A_ENDIF, 0), std);
  std = add_stmt_after(mk_stmt(A_ELSE, 0), std);

  len_stmt = mk_assn_stmt(get_len_of_deferchar_ast(lhs), rhs_len, DT_INT);
  std = add_stmt_after(len_stmt, std);
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_ALLOCATE);
  A_SRCP(ast, lhs);
  A_FIRSTALLOCP(ast, 1);
  std = add_stmt_after(ast, std);

  std = init_intrin_type_desc(lhs, memsym_of_ast(lhs), std);

  add_stmt_after(mk_stmt(A_ENDIF, 0), std);

  check_and_add_auto_dealloc_from_ast(lhs);
}

/** \brief Check whether there is a subprogram statement; if not, create a
           dummy program symbol, and use that as the program.
 */
void
dummy_program()
{
  if (sem.scope_level == 0) {
    const char *tname;
    int sptr;
    /* get a symbol to be the outer scope */
    tname = "MAIN";
    sptr = declref(getsymbol(tname), ST_ENTRY, 'd');
    SYMLKP(sptr, NOSYM);
    SCP(sptr, SC_EXTERN);
    PARAMCTP(sptr, 0);
    FUNCLINEP(sptr, gbl.funcline);
    DTYPEP(sptr, DT_NONE);
    push_scope_level(sptr, SCOPE_NORMAL);
    push_scope_level(sptr, SCOPE_SUBPROGRAM);
    gbl.currsub = sptr;
    /* if the first statement was labelled, set the scope of the label */
    if (scn.currlab) {
      SCOPEP(scn.currlab, sptr);
    }
  }
} /* dummy_program */

static void
rw_host_state(int wherefrom, int (*p_rw)(), FILE *fd)
{
  if (wherefrom & 0x1) {
    rw_semant_state(p_rw, fd);
  }
  if (wherefrom & 0x10) {
    rw_gnr_state(p_rw, fd);
  }
  if (wherefrom & 0x2) {
    rw_sym_state(p_rw, fd);
    rw_dtype_state(p_rw, fd);
    rw_ast_state(p_rw, fd);
    rw_dinit_state(p_rw, fd);
    rw_dpmout_state(p_rw, fd);
    rw_import_state(p_rw, fd);
  }
  if (wherefrom & 0x4) {
    rw_mod_state(p_rw, fd);
  }
  if (wherefrom & 0x20) {
    rw_semant_state(p_rw, fd);
    rw_sym_state(p_rw, fd);
    rw_dtype_state(p_rw, fd);
    rw_ast_state(p_rw, fd);
    rw_dinit_state(p_rw, fd);
    rw_dpmout_state(p_rw, fd);
    rw_import_state(p_rw, fd);
  }
} /* rw_host_state */

static FILE *state_file = NULL;
static FILE *state_append_file = NULL;
static int saved_symavl = 0;
static int saved_astavl = 0;
static int saved_dtyavl = 0;
static LOGICAL state_still_pass_one = FALSE;
static LOGICAL state_append_file_full = FALSE;
static long state_file_position = 0;
static int state_last_routine = 0;

/* labels for internal subprograms are saved in pass 1, and restored
 * in pass 2; they are saved as C strings in a char array;
 * the structure of the C array is:
 *  s u b 1 \000 . L 0 0 1 0 0 \000 s u b 2 \000 . L 0 0 2 0 0 \000
 *  . L 0 0 3 0 0 \000 s u b 3 \000 s u b 4 \000 . L 0 0 1 0 0 \000 ;
 * for four internal subprograms:
 *  sub1 with label 100
 *  sub2 with labels 200 and 300
 *  sub3 with no labels
 *  sub4 with another label 100
 * the semicolon at the end is used to tell when to stop for the last
 * subprogram's label list.
 */
static char *saved_labels = NULL;
static int saved_labels_size = 0, saved_labels_avail = 0, saved_labels_pos = 0;

/** \brief Called from semant.c to save the semant, sym, dtype, ast, and other
    'state' information from a host routine for internal subprograms, for 'pass
   1'.
    Also, for 'pass 2', save_host_state is called to overwrite the semant state
   information.
*/
void
save_host_state(int wherefrom)
{
  /* use quick binary read/write */
  if (state_file) {
    if (wherefrom & 0x21) {
      /* seek to the beginning before writing first data */
      fseek(state_file, 0L, 0);
    }
  } else {
    state_file = tmpfile();
    if (state_file == NULL)
      errfatal(5);
  }
  if (wherefrom & 0x2) {
    /* clear the SECD field of ST_ARRDSC symbols */
    int sptr;
    for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
      if (STYPEG(sptr) == ST_ARRDSC) {
        /* clear SECD field */
        SECDP(sptr, 0);
        ALNDP(sptr, 0);
      }
    }
  }
  rw_host_state(wherefrom, (int (*)())fwrite, state_file);
  saved_symavl = stb.stg_avail;
  saved_astavl = astb.stg_avail;
  saved_dtyavl = stb.dt.stg_avail;
} /* save_host_state */

#ifdef CLASSG
static void
fix_invobj(int sptr)
{
  /* Called by fix_symtab() below. Decrements INVOBJ field of type bound
   * procedure due to fix_symtab() removing result argument of function.
   */
  int sptr2;
  for (sptr2 = 1; sptr2 < stb.stg_avail; ++sptr2) {
    int bind_sptr;
    if (STYPEG(sptr2) == ST_MEMBER && CLASSG(sptr2) && VTABLEG(sptr2) == sptr &&
        !NOPASSG(sptr2) && (bind_sptr = BINDG(sptr2)) > NOSYM &&
        STYPEG(bind_sptr) == ST_PROC && INVOBJINCG(bind_sptr)) {
      INVOBJINCP(bind_sptr, FALSE);
      INVOBJP(bind_sptr, INVOBJG(bind_sptr) - 1);
    }
  }
}
#endif

/* look through restored symbol for array-valued, pointer-valued,
 * or other functions that were turned into subprograms. */
static void
fix_symtab()
{
  int sptr, i;
  for (sptr = aux.list[ST_PROC]; sptr > NOSYM; sptr = SLNKG(sptr)) {
    if (!FUNCG(sptr) && FVALG(sptr) > NOSYM) {
      /* remake into a function */
      FUNCP(sptr, TRUE);
      /* Remove first parameter only if it is the
       * return value symbol.
       */
      if (aux.dpdsc_base[DPDSCG(sptr)] == FVALG(sptr)) {
#ifdef CLASSG
        fix_invobj(sptr);
#endif
        PARAMCTP(sptr, PARAMCTG(sptr) - 1);
        aux.dpdsc_base[DPDSCG(sptr)] = 0; /* clear the reserved fval field */
        DPDSCP(sptr, DPDSCG(sptr) + 1);
      }
      DTYPEP(sptr, DTYPEG(FVALG(sptr)));
    }
  }
#if DEBUG
  /* aux.list[ST_PROC] must be terminated with NOSYM, not 0 */
  assert(sptr == NOSYM, "fix_symtab: corrupted aux.list[ST_PROC]", sptr, 3);
#endif
  /* fixing up procedure pointers that contain interfaces and converting it
   * back from subroutine to functions.
   */
  for (i = sem.typroc_avail; sptr > NOSYM; i++) {
    int procdt, fval;
    procdt = sem.typroc_base[i];
    fval = DTY(procdt + 5);
    if (!fval)
      continue;
    sptr = DTY(procdt + 2);
    if (!FUNCG(sptr) && FVALG(sptr) > NOSYM) {
      FUNCP(sptr, TRUE);
      if (aux.dpdsc_base[DPDSCG(sptr)] == FVALG(sptr)) {
#ifdef CLASSG
        fix_invobj(sptr);
#endif
        PARAMCTP(sptr, PARAMCTG(sptr) - 1);
        aux.dpdsc_base[DPDSCG(sptr)] = 0; /* clear the reserved fval field */
        DPDSCP(sptr, DPDSCG(sptr) + 1);
      }
      DTYPEP(sptr, DTYPEG(FVALG(sptr)));
    }
  }
} /* fix_symtab */

/** \brief Called at the end of an internal subprogram.

   In pass 1:
     - Save the internal subprogram information, kind of like interface blocks.
       If there is more than 1 internal subprogram, the information is
       exported collectively, that is, all subprograms are exported each time.
     - Save any labels for internal subprograms.
       These labels are restored by restore_internal_subprograms() below.
     - Restores the host information for the next subprogram.
     - Reimport any internal subprograms as contained subprograms.
       This information will be reimported in pass 2 by
       restore_internal_subprograms() below.

   In pass 2:
     - Restore the host information for the next subprogram,
       as this will have been saved by the save_host_state
       call for the host subprogram and will include all the contained
       subprograms information as imported in restore_internal_subprograms()
       for the host routine.
 */
void
restore_host_state(int whichpass)
{
  if (state_file == NULL)
    interr("no state file to restore", 0, 4);

  if (whichpass == 2) {
    fseek(state_file, 0L, 0);
    rw_host_state(0x13, (int (*)())fread, state_file);
    /*astb.firstuast = astb.stg_avail;*/
    /* ### don't reset firstusym for main program */
    stb.firstusym = stb.stg_avail;
    state_still_pass_one = 0;
    fix_symtab();
  } else if (whichpass == 4) { /* for ipa import */
    fseek(state_file, 0L, 0);
    rw_host_state(0x2, (int (*)())fread, state_file);
    /*astb.firstuast = astb.stg_avail;*/
    /* ### don't reset firstusym for main program */
    stb.firstusym = stb.stg_avail;
    state_still_pass_one = 0;
    fix_symtab();
  } else {
    int nw, modbase, smodbase, len, lab, saved_scope;
    long end_of_file;
    char Mname[100], Sname[100], MMname[100], SSname[100];
    /* pass one */
    /* write the 'append' symbols into the 'append_file' */
    state_append_file_full = TRUE;
    if (!state_append_file) {
      state_append_file = tmpfile();
      if (state_append_file == NULL)
        errfatal(5);
      state_file_position = 0;
    } else {
      if (!state_still_pass_one) {
        state_file_position = 0;
        fseek(state_append_file, state_file_position, 0);
        saved_labels_avail = 0;
        saved_labels_pos = 0;
      } else {
        /* what is the containing subprogram;
         * this is the subprogram on the top of the scope stack */
        if (state_last_routine == sem.scope_stack[sem.scope_level].sptr) {
          /* rewind to the last position */
          fseek(state_append_file, state_file_position, 0);
        } else {
          /* leave at the end */
        }
      }
    }
    state_last_routine = sem.scope_stack[sem.scope_level].sptr;
    modbase = 0;
    strcpy(Mname, "--");
    strcpy(Sname, SYMNAME(state_last_routine));
    if (sem.mod_sym) {
      modbase = CMEMFG(sem.mod_sym);
      strcpy(Mname, SYMNAME(sem.mod_sym));
    }
    fflush(state_append_file);
    state_file_position = ftell(state_append_file);
    /* write identifier to the file */
    fprintf(state_append_file, "- %s %s %d %d %d %d %d\n", Mname, Sname,
            SCOPEG(gbl.currsub), saved_symavl, saved_astavl, saved_dtyavl,
            modbase);
    export_append_host_sym(gbl.currsub);
    export_host_subprogram(state_append_file, gbl.currsub, saved_symavl,
                           saved_astavl, saved_dtyavl);
    end_of_file = ftell(state_append_file); /* get position */

    /* save labels from the internal subprogram */
    if (saved_labels == NULL) {
      saved_labels_size = 512;
      NEW(saved_labels, char, saved_labels_size);
      saved_labels_avail = 0;
      saved_labels_pos = 0;
    }
    len = strlen(SYMNAME(gbl.currsub));
    /* need len+1 char positions for the null char at the end of the
     * string; also need one more for the 'end everything' marker */
    NEED(saved_labels_avail + len + 2, saved_labels, char, saved_labels_size,
         saved_labels_size + 512);
    strcpy(saved_labels + saved_labels_avail, SYMNAME(gbl.currsub));
    saved_labels_avail += len + 1;
    for (lab = sem.flabels; lab > NOSYM; lab = SYMLKG(lab)) {
      len = strlen(SYMNAME(lab));
      NEED(saved_labels_avail + len + 2, saved_labels, char, saved_labels_size,
           saved_labels_size + 512);
      strcpy(saved_labels + saved_labels_avail, SYMNAME(lab));
      saved_labels_avail += len + 1;
    }
    sem.flabels = 0;
    saved_labels[saved_labels_avail] = ';';

    fseek(state_file, 0L, 0);
    rw_host_state(0x3, (int (*)())fread, state_file);
    /*astb.firstuast = astb.stg_avail;*/

    fseek(state_append_file, state_file_position, 0);
    nw = fscanf(state_append_file, "- %s %s %d %d %d %d %d\n", MMname, SSname,
                &saved_scope, &saved_symavl, &saved_astavl, &saved_dtyavl,
                &smodbase);
    if (strcmp(MMname, Mname) != 0 || strcmp(SSname, Sname) != 0 || nw != 7) {
      interr("unknown state file error", 0, 4);
    }
    /* import the contained subprogram symbols */
    import_host_subprogram(state_append_file, "state file", saved_symavl,
                           saved_astavl, saved_dtyavl, 0, 0);
    state_still_pass_one = 1;
    /* move file for read and write to end of file */
    fseek(state_append_file, end_of_file, 0);
  }
} /* restore_host_state */

/** \brief Called at the beginning of a subprogram in pass 2.

    - Checks whether there is information available for subprograms
      contained in this one, as saved by restore_host_state().
    - If so, restores that more or less like an interface block.
    - If the current routine is an internal subprogram, its labels are
      restored.  This is so FORMAT labels that appear in both the inner
      and outer subprogram are properly resolved.
 */
void
restore_internal_subprograms(void)
{
  if (gbl.currsub == 0)
    dummy_program();
  if (state_append_file && state_append_file_full) {
    int nw, last_routine, modbase, nmodbase, moddiff;
    int saved_scope;
    char Mname[100], Sname[100], MMname[100], SSname[100];
    if (state_still_pass_one) {
      state_still_pass_one = 0;
      state_file_position = 0;
      exterf_init_host();
    }
    nw = fseek(state_append_file, state_file_position, 0);
    nw = fscanf(state_append_file, "- %s %s %d %d %d %d %d\n", MMname, SSname,
                &saved_scope, &saved_symavl, &saved_astavl, &saved_dtyavl,
                &modbase);
    /* import the contained subprogram symbols */
    if (sem.scope_level) {
      last_routine = sem.scope_stack[sem.scope_level].sptr;
      strcpy(Sname, SYMNAME(last_routine));
    } else {
      strcpy(Sname, "MAIN");
    }
    /* adjust symbols in case they were moved around by module importing */
    nmodbase = 0;
    strcpy(Mname, "--");
    if (sem.mod_sym) {
      nmodbase = CMEMFG(sem.mod_sym);
      strcpy(Mname, SYMNAME(sem.mod_sym));
    }
    if (nw == 7 && strcmp(Mname, MMname) == 0 && strcmp(Sname, SSname) == 0) {
      moddiff = nmodbase - modbase;
      /* this is the information for this routine */
      import_host(state_append_file, "state file", saved_symavl, saved_astavl,
                  saved_dtyavl, modbase, moddiff, saved_scope, stb.curr_scope);
      state_file_position = ftell(state_append_file);
    }
  }
  if (gbl.internal > 1) {
    /* restore any labels found */
    /* compare subprogram name */
    char *cp;
    cp = saved_labels + saved_labels_pos;
    if (strcmp(cp, SYMNAME(gbl.currsub))) {
      interr("unknown internal subprogram state error (labels)", gbl.currsub,
             4);
    }
    saved_labels_pos += strlen(cp) + 1;
    cp = saved_labels + saved_labels_pos;
    while (*cp == '.') {
      /* have a label */
      int sptr = getsymbol(cp);
      if (STYPEG(sptr) != ST_UNKNOWN &&
          (STYPEG(sptr) != ST_LABEL || SCOPEG(sptr) != stb.curr_scope)) {
        /* this was not a label for this subprogram already */
        sptr = insert_sym(sptr);
      }
      STYPEP(sptr, ST_LABEL);
      FMTPTP(sptr, 0);
      REFP(sptr, 0);
      ADDRESSP(sptr, 0);
      SYMLKP(sptr, NOSYM);
      SCOPEP(sptr, stb.curr_scope);
      saved_labels_pos += strlen(cp) + 1;
      cp = saved_labels + saved_labels_pos;
    }
  }
} /* restore_internal_subprograms */

void
reset_internal_subprograms()
{
  state_still_pass_one = 0;
  state_file_position = 0;
  state_append_file_full = FALSE;
} /* reset_internal_subprograms */

static FILE *modstate_file = NULL;
static FILE *modstate_append_file = NULL;
static int modsaved_symavl, modsaved_astavl, modsaved_dtyavl;
static int modstate_append_file_full = 0;
static int mod_clear_init = 0;
static LOGICAL modsave_ieee_features;

/** \brief Called at a CONTAINS clause

    Writes the module information out quickly.
    It is split into two pieces: the first only writes out the semant
    information, before semfin() deallocates it, and the second appends
    everything else, including the module.c tables.
 */
void
save_module_state1()
{
  if (modstate_file) {
    fseek(modstate_file, 0L, 0);
  } else {
    modstate_file = tmpfile();
    if (modstate_file == NULL)
      errfatal(5);
  }
  rw_host_state(0x1, (int (*)())fwrite, modstate_file);
} /* save_module_state1 */

void
save_module_state2()
{
  rw_host_state(0x16, (int (*)())fwrite, modstate_file);
  modsaved_symavl = stb.stg_avail;
  modsaved_astavl = astb.stg_avail;
  modsaved_dtyavl = stb.dt.stg_avail;
  modstate_append_file_full = 0;
  mod_clear_init = 1;
  modsave_ieee_features = sem.ieee_features;
} /* save_module_state2 */

static FILE *modsave_file = NULL;

void
save_imported_modules_state()
{
  if (modsave_file) {
    fseek(modsave_file, 0L, 0);
  } else {
    modsave_file = tmpfile();
    if (modsave_file == NULL)
      errfatal(5);
  }
  rw_host_state(0x20, (int (*)())fwrite, modsave_file);
} /* save_imported_modules_state */

void
restore_imported_modules_state()
{
  fseek(modsave_file, 0L, 0);
  rw_host_state(0x20, (int (*)())fread, modsave_file);
} /* restore_imported_modules_state */

/*
 * consider:
 *  module b
 *   public :: f << at this point, we add a variable 'f'
 *  contains
 *   integer function f << now here, we add function 'f', hide variable 'f'
 * ...
 * the problem is that hiding variable 'f' happens too late, we've already
 * got all the information for 'f' in modstate_file; so we keep
 * track of this situation (semsym.c:replace_variable) and when it
 * arises, and we restore the module state, we re-hide 'f'.
 * We only need to keep track of a single variable at a time.
 */
static int module_must_hide_this_symbol_sptr = 0;

void
module_must_hide_this_symbol(int sptr)
{
  module_must_hide_this_symbol_sptr = sptr;
} /* module_must_hide_this_symbol */

/** \brief Called at start of module-contained subprogram, restores state.
           If this is the first 'restore' since the last 'reset',
           the 'module append' file is full and needs to be imported.
 */
void
restore_module_state()
{
  if (modstate_file == NULL)
    errfatal(5);
  /* First, read the binary-saved information */
  fseek(modstate_file, 0L, 0);
  rw_host_state(0x17, (int (*)())fread, modstate_file);
  /* for TPR 1654, if we need to set NEEDMOD for internal
   * subprograms, this is the place to set it
   * NEEDMODP( stb.curr_scope, 1 );
   */
  if (modstate_append_file_full) {

    /* Next, import the module-contained subprogram */
    fseek(modstate_append_file, 0L, 0);
    import_host(modstate_append_file, "module state file", modsaved_symavl,
                modsaved_astavl, modsaved_dtyavl, 0, 0, 0, 0);
  }
  if (module_must_hide_this_symbol_sptr) {
    HIDDENP(module_must_hide_this_symbol_sptr, 1);
    module_must_hide_this_symbol_sptr = 0;
  }
  if (mod_clear_init) {
    /* clear the data-initialized bit for any module-initialized commons */
    int sptr;
    for (sptr = gbl.cmblks; sptr > NOSYM; sptr = SYMLKG(sptr)) {
      DINITP(sptr, 0);
    }
  }
  if (mod_clear_init || modstate_append_file_full) {
    modstate_append_file_full = 0;
    mod_clear_init = 0;
    /* Lastly, rewrite the module state file */
    fseek(modstate_file, 0L, 0);
    rw_host_state(0x17, (int (*)())fwrite, modstate_file);
    modsaved_symavl = stb.stg_avail;
    modsaved_astavl = astb.stg_avail;
    modsaved_dtyavl = stb.dt.stg_avail;
  }
  sem.ieee_features = modsave_ieee_features;
} /* restore_module_state */

/** \brief Called at the end of a module-contained subprogram;
           rearranges the data structures for the module.
*/
void
reset_module_state()
{
  if (modstate_file == NULL)
    interr("no module state file to restore", 0, 4);
  if (sem.which_pass == 1) {
    fseek(modstate_file, 0L, 0);
    rw_host_state(0x17, (int (*)())fread, modstate_file);
  } else {
    /* export the module-contained subprogram */
    if (!modstate_append_file) {
      modstate_append_file = tmpfile();
      if (modstate_append_file == NULL)
        errfatal(5);
    } else {
      fseek(modstate_append_file, 0L, 0);
    }
    export_module_subprogram(modstate_append_file, gbl.currsub, modsaved_symavl,
                             modsaved_astavl, modsaved_dtyavl);
    modstate_append_file_full = 1;
  }
} /* reset_module_state */

int
have_module_state()
{
  if (modstate_file == NULL)
    return 0;
  return 1;
}

/** \brief Compilation is finished - deallocate storage, close files, etc.
 */
void
sem_fini(void)
{
  if (state_file)
    fclose(state_file);
  state_file = NULL;
  if (state_append_file)
    fclose(state_append_file);
  state_append_file = NULL;
  if (saved_labels) {
    FREE(saved_labels);
    saved_labels = NULL;
    saved_labels_size = 0;
    saved_labels_avail = 0;
    saved_labels_pos = 0;
  }
  if (sem.eqv_base) {
    FREE(sem.eqv_base);
    sem.eqv_base = NULL;
  }
  if (sem.eqv_ss_base) {
    FREE(sem.eqv_ss_base);
    sem.eqv_ss_base = NULL;
  }
  import_fini();
  if (sem.non_private_base) {
    FREE(sem.non_private_base);
    sem.non_private_base = NULL;
  }
} /* sem_fini */

void
sem_set_storage_class(int sptr)
{
  if (STYPEG(sptr) == ST_ARRAY) {
    if (ALLOCG(sptr)) {
      SCP(sptr, SC_BASED);
    } else if (ASUMSZG(sptr)) {
      {
        error(50, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SCP(sptr, SC_DUMMY);
      }
    } else if (ASSUMLENG(sptr)) {
      error(452, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      SCP(sptr, SC_DUMMY);
    } else {
      SCP(sptr, SC_LOCAL);
      if (ADJARRG(sptr) || RUNTIMEG(sptr)) {
        add_auto_array(sptr);
        if (has_allocattr(sptr)) {
          add_auto_dealloc(sptr);
        }
      } else if (ADJLENG(sptr))
        add_auto_char(sptr);
    }
  } else if (STYPEG(sptr) == ST_PROC)
    SCP(sptr, SC_EXTERN);
  else if (POINTERG(sptr)) {
    SCP(sptr, SC_BASED);
    if (ADJLENG(sptr))
      add_auto_char(sptr);
  } else if (!IS_INTRINSIC(STYPEG(sptr))) {
/* if an intrinsic, this processing must be deferred until an
 * actual scalar reference confirms a nonintrinsic context.
 */
    SCP(sptr, SC_LOCAL);
    if (ADJLENG(sptr))
      add_auto_char(sptr);
  }
}

/* ensure that the list of automatic arrays is in
 * the order they're declared
 */
static void
add_auto_array(int sptr)
{
  SCP(sptr, SC_LOCAL);
  add_autobj(sptr);
  AD_NOBOUNDS(AD_DPTR(DTYPEG(sptr))) = 1;
}

/* ensure that the list of automatic arrays is in
 * the order they're declared
 */
static void
add_auto_char(int sptr)
{
  SCP(sptr, SC_LOCAL);
  add_autobj(sptr);
}

static void
add_autobj(int sptr)
{
  static int last_autobj;

  if (last_autobj > NOSYM && AUTOBJG(last_autobj) > NOSYM) {
    /* last_autobj is invalid */
    int next;
    last_autobj = 0;
    next = gbl.autobj;
    while (next > NOSYM) {
      last_autobj = next;
      next = AUTOBJG(next);
    }
  }
  if (gbl.autobj == NOSYM)
    /* first automatic array */
    gbl.autobj = sptr;
  else
    AUTOBJP(last_autobj, sptr);
  last_autobj = sptr;
  AUTOBJP(sptr, NOSYM);
}

void
dmp_var(VAR *var, int indent, FILE *f)
{
  int i;
  if (f == NULL)
    f = stderr;
  for (i = 0; i < indent; ++i)
    fprintf(f, "  ");
  switch (var->id) {
  case Dostart:
    fprintf(f, "Dostart: indvar=%d lowbd=%d upbd=%d step=%d (ASTs)\n",
            var->u.dostart.indvar, var->u.dostart.lowbd, var->u.dostart.upbd,
            var->u.dostart.step);
    break;
  case Doend:
    fprintf(f, "Doend for:\n");
    dmp_var(var->u.doend.dostart, indent + 1, f);
    break;
  case Varref: {
    char typebuf[300];
    DTYPE dtype = var->u.varref.dtype;
    VAR *members = var->u.varref.subt;
    FILE *save_dbgfil = gbl.dbgfil;
    getdtype(dtype, typebuf);
    /* id is S_* constant */
    fprintf(f, "Varref: id=%d ptr=AST:%d:", var->u.varref.id,
            var->u.varref.ptr);
    gbl.dbgfil = f;
    printast(var->u.varref.ptr);
    gbl.dbgfil = save_dbgfil;
    fprintf(f, " dtype=%d:%s shape=%d\n", dtype, typebuf, var->u.varref.shape);
    for (; members != 0; members = members->next) {
      dmp_var(members, indent + 1, f);
    }
  } break;
  default:
    interr("dmp_var: bad id", var->id, ERR_Severe);
  }
}

void
dvar(VAR *var)
{
  dmp_var(var, 0, stderr);
}

void
dmp_acl(ACL *acl, int indent)
{
  _dmp_acl(acl, indent, NULL);
}

static void
_dmp_acl(ACL *acl, int indent, FILE *f)
{
  ACL *c_aclp;
  char two_spaces[3] = "  ";

  if (!acl) {
    return;
  }

  if (f == NULL)
    f = stderr;
  for (c_aclp = acl; c_aclp; c_aclp = c_aclp->next) {
    switch (c_aclp->id) {
    case AC_IDENT:
      put_prefix(two_spaces, indent, f);
      fprintf(
          f,
          "AC_IDENT: %d, repeatc=%d, is_const=%d, dtype=%d, sptr=%d, size=%d\n",
          c_aclp->u1.ast, c_aclp->repeatc, c_aclp->is_const, c_aclp->dtype,
          c_aclp->sptr, c_aclp->size);
      break;
    case AC_CONST:
      put_prefix(two_spaces, indent, f);
      fprintf(
          f,
          "AC_CONST: %d, repeatc=%d, is_const=%d, dtype=%d, sptr=%d, size=%d\n",
          c_aclp->u1.ast, c_aclp->repeatc, c_aclp->is_const, c_aclp->dtype,
          c_aclp->sptr, c_aclp->size);
      break;
    case AC_AST:
      put_prefix(two_spaces, indent, f);
      fprintf(
          f,
          "AC_AST: %d, repeatc=%d, is_const=%d, dtype=%d, sptr=%d, size=%d\n",
          c_aclp->u1.ast, c_aclp->repeatc, c_aclp->is_const, c_aclp->dtype,
          c_aclp->sptr, c_aclp->size);
      break;
    case AC_EXPR:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "**** AC_EXPR: SST id %d ***\n", SST_IDG(c_aclp->u1.stkp));
      break;
    case AC_IEXPR:
      put_prefix(two_spaces, indent, f);
      fprintf(f,
              "AC_IEXPR: op %s, repeatc=%d, is_const=%d, dtype=%d, sptr=%d, "
              "size=%d\n",
              iexpr_op(c_aclp->u1.expr->op), c_aclp->repeatc, c_aclp->is_const,
              c_aclp->dtype, c_aclp->sptr, c_aclp->size);
      _dmp_acl(c_aclp->u1.expr->lop, indent + 1, f);
      _dmp_acl(c_aclp->u1.expr->rop, indent + 1, f);
      break;
    case AC_IDO:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "AC_IDO: , dtype=%d, sptr=%d, size=%d\n", c_aclp->dtype,
              c_aclp->sptr, c_aclp->size);
      fprintf(f,
              "        index var sptr %d, init expr ast %d, "
              "limit expr ast %d, step_expr ast %d, repeatc %d\n",
              c_aclp->u1.doinfo->index_var, c_aclp->u1.doinfo->init_expr,
              c_aclp->u1.doinfo->limit_expr, c_aclp->u1.doinfo->step_expr,
              c_aclp->repeatc);
      put_prefix(two_spaces, indent, f);
      fprintf(f, " Initialization Values:\n");
      _dmp_acl(c_aclp->subc, indent + 1, f);
      break;
    case AC_ACONST:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "AC_ACONST: repeatc %d, dtype=%d, sptr=%d\n", c_aclp->repeatc,
              c_aclp->dtype, c_aclp->sptr);
      put_prefix(two_spaces, indent, f);
      fprintf(f, " Initialization Values:\n");
      _dmp_acl(c_aclp->subc, indent + 1, f);
      break;
    case AC_SCONST:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "AC_SCONST: repeatc %d, dtype=%d, sptr=%d\n", c_aclp->repeatc,
              c_aclp->dtype, c_aclp->sptr);
      put_prefix(two_spaces, indent, f);
      fprintf(f, " Initialization Values:\n");
      _dmp_acl(c_aclp->subc, indent + 1, f);
      break;
    case AC_TYPEINIT:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "AC_TYPEINIT: repeatc %d, dtype=%d, sptr=%d\n",
              c_aclp->repeatc, c_aclp->dtype, c_aclp->sptr);
      put_prefix(two_spaces, indent, f);
      fprintf(f, " Initialization Values:\n");
      _dmp_acl(c_aclp->subc, indent + 1, f);
      break;
    case AC_ICONST:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "AC_ICONST: value %d\n", c_aclp->u1.i);
      break;
    case AC_REPEAT:
    case AC_LIST:
    default:
      put_prefix(two_spaces, indent, f);
      fprintf(f, "*** UNKNOWN/UNUSED ACL ID %d\n", c_aclp->id);
      break;
    }
  }
}

static void
put_prefix(char *str, int cnt, FILE *f)
{
  int i;

  fprintf(f, "    ");
  for (i = 0; i < cnt; i++)
    fprintf(f, "%s", str);
}

int
mp_create_bscope(int reuse)
{
  int ast = 0, i;
  int astid;
  int uplevel_sptr = 0;
  int scope_sptr = 0;
  SPTR parent_sptr;

  if (reuse) {
    i = sem.scope_level;
    scope_sptr = BLK_SCOPE_SPTR(i);
    while (scope_sptr == 0 && i) {
      scope_sptr = BLK_SCOPE_SPTR(i);
      --i;
    }
    if (scope_sptr == 0) {
      goto newscope;
    }
    ast = mk_stmt(A_MP_BMPSCOPE, 0);
    astid = mk_id(scope_sptr);
    A_STBLKP(ast, astid);
    (void)add_stmt(ast);
    return ast;
  }
newscope:
  scope_sptr  = getccssym("uplevel", sem.blksymnum++, ST_BLOCK);
  PARSYMSCTP(scope_sptr, 0);
  PARSYMSP(scope_sptr, 0);
  BLK_SCOPE_SPTR(sem.scope_level) = scope_sptr;

  /* create a new uplevel_sptr per outlined region */
  uplevel_sptr = getccssym("uplevel", sem.blksymnum++, ST_BLOCK);
  PARSYMSCTP(uplevel_sptr, 0);
  PARSYMSP(uplevel_sptr, 0);
  PARUPLEVELP(scope_sptr, uplevel_sptr);
  BLK_UPLEVEL_SPTR(sem.scope_level) = uplevel_sptr;
  i = sem.scope_level - 1;
  parent_sptr = BLK_UPLEVEL_SPTR(i);
  while (i > 0 && parent_sptr == 0) {
    --i;
    parent_sptr = BLK_UPLEVEL_SPTR(i);
  }
  (void)llmp_create_uplevel(uplevel_sptr);
  if (parent_sptr) {
    llmp_uplevel_set_parent((SPTR)uplevel_sptr, parent_sptr);
  }
  ast = mk_stmt(A_MP_BMPSCOPE, 0);
  astid = mk_id(scope_sptr);
  A_STBLKP(ast, astid);
  (void)add_stmt(ast);
  return ast;
}

int
mp_create_escope()
{
  int ast = 0;

  ast = mk_stmt(A_MP_EMPSCOPE, 0);
  (void)add_stmt(ast);
  BLK_UPLEVEL_SPTR(sem.scope_level) = 0;

  return ast;
}

int
enter_lexical_block(int gen_debug)
{
  int sptr;
  int sptr1;
  int ast, std;

  sptr = BLK_SCOPE_SPTR(sem.scope_level - 1);

  if (gen_debug) {
    if (!sptr) {
      sptr = getccssym("uplevel", sem.blksymnum++, ST_BLOCK);
      PARSYMSCTP(sptr, 0);
      PARSYMSP(sptr, 0);
    }
    STARTLINEP(sptr, gbl.lineno);
    if (sptr != BLK_SYM(sem.scope_level - 1))
      ENCLFUNCP(sptr, BLK_SYM(sem.scope_level - 1));
    sptr1 = getlab();
    RFCNTI(sptr1);
    VOLP(sptr1, 1); /* so block is never deleted */
    STARTLABP(sptr, sptr1);
    ENCLFUNCP(sptr1, sptr);
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, (int)STD_PREV(0));
    STD_LABEL(std) = sptr1;
  }
  BLK_SYM(sem.scope_level) = sptr;
  return sptr;
}

void
exit_lexical_block(int gen_debug)
{
  int sptr1;
  int blksym;
  int ast, std;

  blksym = BLK_SYM(sem.scope_level);
  ENDLINEP(blksym, gbl.lineno);
  if (gen_debug) {
    sptr1 = getlab();
    RFCNTI(sptr1);
    VOLP(sptr1, 1); /* so block is never deleted */
    ENDLABP(blksym, sptr1);
    ENCLFUNCP(sptr1, blksym);
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, (int)STD_PREV(0));
    STD_LABEL(std) = sptr1;
  }
}

static const char *di_name[] = { // order by DI_KIND enum in semant.h
    "block IF",
    "IFELSE",
    "DO",
    "DOCONCURRENT",
    "DOWHILE",
    "WHERE",
    "ELSEWHERE",
    "FORALL",
    "SELECTCASE",
    "SELECT TYPE",
    "ASSOCIATE",
    "BLOCK",
    "PARALLEL directive",
    "PARALLELDO directive",
    "OMP DO directive",
    "DOACROSS directive",
    "PARALLELSECTIONS directive",
    "SECTIONS directive",
    "SINGLE directive",
    "CRITICAL directive",
    "MASTER directive",
    "ORDERED directive",
    "WORKSHARE directive",
    "PARALLELWORKSHARE directive",
    "TASK directive",
    "ACC ATOMIC CAPTURE construct",
    "SIMD",
    "TASKGROUP",
    "TASKLOOP",
    "TARGET",
    "TARGETENTERDATA",
    "TARGETEXITDATA",
    "TARGETDATA",
    "TARGETUPDATE",
    "DISTRIBUTE",
    "TEAMS",
    "DECLARE TARGET",
    "DISTRIBUTE PARALLEL DO",
    "TARGET PARALLEL DO",
    "TARGET SIMD",
    "TARGET TEAMS DISTRIBUTE",
    "TEAMS DISTRIBUTE",
    "TARGET TEAMS DISTRIBUTE PARALLEL DO",
    "TEAMS DISTRIBUTE PARALLEL DO",
    "CUDA KERNEL directive",
    "ACC REGION directive",
    "ACC KERNELS construct",
    "ACC PARALLEL construct",
    "ACC DO directive",
    "ACC LOOP directive",
    "ACC REGION DO directive",
    "ACC REGION LOOP directive",
    "ACC KERNELS DO directive",
    "ACC KERNELS LOOP directive",
    "ACC PARALLEL DO directive",
    "ACC PARALLEL LOOP directive",
    "ACC KERNEL construct",
    "ACC DATA REGION construct",
    "ACC HOST DATA construct",
    "ACC SERIAL",
    "ACC SERIAL LOOP",
};

void
sem_err104(int df, int lineno, const char *str)
{
  if (df) {
    int id;
    id = DI_ID(df);
    if (id < sizeof(di_name) / sizeof(char *)) {
      char buff[256];
      sprintf(buff, "- %s %s", str, di_name[id]);
      error(104, 3, lineno, buff, CNULL);
      return;
    }
    interr("sem_err104:unk doif->ID", DI_ID(df), 3);
  }
}

void
sem_err105(int df)
{
  if (df) {
    int id;
    id = DI_ID(df);
    if (id < sizeof(di_name) / sizeof(char *)) {
      sem_err104(df, gbl.lineno, "unterminated");
      return;
    }
  }
  errsev(105);
}

#if DEBUG
void
_dmp_doif(int df, FILE *f)
{
  int id;
  if (f == NULL)
    f = stderr;
  id = DI_ID(df);
  if (id >= sizeof(di_name) / sizeof(char *)) {
    fprintf(f, "Unknown DI_ID(%d) == %d\n", df, id);
    return;
  }
  fprintf(f, "[%3d] %s - Line=%d", df, di_name[id], DI_LINENO(df));
  if (DI_NAME(df))
    fprintf(f, " ConstructName=%d", DI_NAME(df));
  if (id == DI_DO && DI_DOINFO(df)->collapse)
    fprintf(f, " Collapse=%d", DI_DOINFO(df)->collapse);
  fprintf(f, "\n");
  if (DI_NEST(df)) {
    int i;
    fprintf(f, "      Nest:0x%08lx ", DI_NEST(df));
    for (i = 0; i <= DI_MAXID; i++) {
      if (DI_B(i) & DI_NEST(df))
        fprintf(f, "|%s", di_name[i]);
    }
  }
  fprintf(f, "\n");
}

void
dmp_doif(FILE *f)
{
  int df;
  if (f == NULL)
    f = stderr;
  fprintf(f, "----- DOIF (%d entries)\n", sem.doif_depth);
  for (df = 1; df <= sem.doif_depth; df++) {
    _dmp_doif(df, f);
  }
}
#endif

LOGICAL
is_alloc_ast(int ast)
{
  if (ast)
    return (A_TYPEG(ast) == A_ALLOC && A_TKNG(ast) == TK_ALLOCATE);
  else
    return FALSE;
}

LOGICAL
is_dealloc_ast(int ast)
{
  if (ast)
    return (A_TYPEG(ast) == A_ALLOC && A_TKNG(ast) == TK_DEALLOCATE);
  else
    return FALSE;
}

LOGICAL
is_alloc_std(int std)
{
  int ast;
  if (std) {
    ast = STD_AST(std);
    return (A_TYPEG(ast) == A_ALLOC && A_TKNG(ast) == TK_ALLOCATE);
  } else {
    return FALSE;
  }
}

LOGICAL
is_dealloc_std(int std)
{
  int ast;
  if (std) {
    ast = STD_AST(std);
    return (A_TYPEG(ast) == A_ALLOC && A_TKNG(ast) == TK_DEALLOCATE);
  } else {
    return FALSE;
  }
}

/** \brief Creates an ast that represents a call to a set type runtime routine.
 *
 * \param arg0 is the ast of the descriptor that receives the type from arg1.
 *
 * \param arg1 is the ast of the source descriptor. The type of arg1 is copied
 * into the arg0 descriptor.
 *
 * \param intrin_type is true when you want to use the RTE_set_intrin_type()
 * routine instead of the RTE_set_type() routine.
 *
 * \returns the call ast
 */
int
mk_set_type_call(int arg0, int arg1, LOGICAL intrin_type)
{
  int newargt, func, astnew;

  newargt = mk_argt(2);
  ARGT_ARG(newargt, 0) = arg0;
  ARGT_ARG(newargt, 1) = arg1;

  func = mk_id(sym_mkfunc_nodesc(
      mkRteRtnNm((intrin_type) ? RTE_set_intrin_type : RTE_set_type), DT_NONE));
  astnew = mk_func_node(A_CALL, func, 2, newargt);

  return astnew;
}

/** \brief Generates calls to RTE_set_type() or RTE_set_intrin_type() which
 * set the type descriptor field of an object's descriptor.
 *
 * \param dest_ast is the descriptor expression that's getting its type
 *  descriptor set. Note: dest_ast may be a descriptor expression or an
 *  expression that has a descriptor.
 *
 * \param src_ast is the expression that has the type descriptor that we are
 *  copying to dest_ast. Note: src_ast may be a descriptor expression or an
 *  expession that has a descriptor.
 *
 * \param std is the std where we will insert the call.
 *
 * \param insert_before is true when you want to insert the call before std,
 * otherwise we insert it after std.
 *
 * \param intrin_type is true when you want to use the RTE_set_intrin_type()
 * routine instead of the RTE_set_type() routine.
 *
 * \returns the new std after inserting the call.
 */
int
gen_set_type(int dest_ast, int src_ast, int std, LOGICAL insert_before,
             LOGICAL intrin_type)
{
  int astnew, arg0, arg1, sptr, sdsc;
  int atype;

  /* Walk the ast expression to find the invoking object (an A_MEM or A_ID) */
  for (atype = A_TYPEG(src_ast);
       atype == A_FUNC || atype == A_SUBSCR || atype == A_CONV ||
       atype == A_CALL || atype == A_MEM;
       atype = A_TYPEG(src_ast)) {

    if (atype == A_MEM) {
      sptr = memsym_of_ast(src_ast);
      if (is_tbp(sptr)) {
        src_ast = A_PARENTG(src_ast);
      } else {
        break;
      }
    } else {
      src_ast = A_LOPG(src_ast);
    }
  }

  /* get descriptor expression for dest_ast */
  sptr = memsym_of_ast(dest_ast);
  if (DESCARRAYG(sptr) || SCG(sptr) == SC_DUMMY) {
    arg0 = dest_ast;
  } else if (A_TYPEG(src_ast) == A_MEM) {
    sdsc = get_member_descriptor(sptr);
    arg0 = check_member(dest_ast, mk_id(sdsc));
  } else {
    sdsc = SDSCG(sptr);
    if (sdsc == 0) {
      arg0 = dest_ast;
    } else {
      arg0 = mk_id(sdsc);
    }
  }

  /* get descriptor expression for src_ast */
  if (intrin_type) {
    arg1 = src_ast;
  } else {
    sptr = memsym_of_ast(src_ast);
    if (DESCARRAYG(sptr) || SCG(sptr) == SC_DUMMY) {
      arg1 = src_ast;
    } else if (A_TYPEG(src_ast) == A_MEM) {
      sdsc = get_member_descriptor(sptr);
      arg1 = check_member(src_ast, mk_id(sdsc));
    } else {
      sdsc = SDSCG(sptr);
      if (sdsc == 0) {
        arg1 = src_ast;
      } else {
        arg1 = mk_id(sdsc);
      }
    }
  }

  astnew = mk_set_type_call(arg0, arg1, intrin_type);

  if (insert_before) {
    std = add_stmt_before(astnew, std);
  } else {
    std = add_stmt_after(astnew, std);
  }

  return std;
}

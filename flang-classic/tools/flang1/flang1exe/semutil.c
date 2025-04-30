/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Utility routines used by Semantic Analyzer.
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
#include "dinit.h"
#include "semstk.h"
#include "machar.h"
#include "ast.h"
#define RTE_C
#include "rte.h"
#include "pd.h"
#include "fdirect.h"
#include "go.h"
#include "rtlRtns.h"

#define ERR170(s1, s2) error(170, 2, gbl.lineno, s1, s2)

#define UFCHAR \
  error(0, 3, gbl.lineno, "character array expressions not supported", CNULL)

/*
 * Need quick ways of getting data types (for code readablity):
 *   1) actual type e.g. integer or array of integer
 *   2) basic type (no arrays) e.g. real, integer, logical, ...
 */
#define TYPE_OF(s) (SST_DTYPEG(s))
#define TY_OF(s) (DTYG(TYPE_OF(s)))
#define PT_OF(s) (DDTG(TYPE_OF(s))) /* pointer to data type */

#define ERRMSG_BUFFSIZE 200

static int ref_array(SST *, ITEM *);
static INT clog_to_log(INT);
static int mkunion(int, int, int);
static INT const_xtoi(INT, int, int);
static INT _xtok(INT, BIGINT64, int);

static void error83(int);
static LOGICAL subst_lhs_arrfn(int, int, int);
static LOGICAL subst_lhs_pointer(int, int, int);
static LOGICAL not_in_arrfn(int, int);
static int find_pointer_variable_assign(int, int);

static int inline_contig_check(int src, SPTR src_sptr, SPTR sdsc, int std);
static bool is_selector(SPTR sptr);


/*---------------------------------------------------------------------*/

/** \brief If \a stkptr is an LVALUE that has a constant value, replace it with
           the constant value
 */
void
constant_lvalue(SST *stkptr)
{
  int ast, sptr, dtype;
  if (SST_IDG(stkptr) == S_LVALUE) {
    ast = SST_ASTG(stkptr);
    if (ast > 0 && ast < astb.stg_avail && A_ALIASG(ast)) {
      /* make into an S_CONST */
      ast = A_ALIASG(ast);
      sptr = A_SPTRG(ast);
      dtype = DTYPEG(sptr);
      SST_DTYPEP(stkptr, dtype);
      if (DT_ISWORD(dtype)) {
        SST_SYMP(stkptr, CONVAL2G(sptr));
      } else {
        SST_SYMP(stkptr, sptr);
      }
      SST_ASTP(stkptr, ast);
      SST_IDP(stkptr, S_CONST);
      return;
    }
  }
} /* constant_lvalue */

/** \brief Check that the indicated semantic stack entry is a constant of the
           specified type and convert constant to new type if possible.
    \return 32-bit constant value or symbol table pointer.
 */
INT
chkcon(SST *stkptr, int dtype, LOGICAL warnflg)
{
  INT oldval, oldtyp, oldast, oldid;

  constant_lvalue(stkptr);
  oldval = SST_SYMG(stkptr);
  oldtyp = SST_DTYPEG(stkptr);
  oldast = SST_ASTG(stkptr);
  oldid = SST_IDG(stkptr);
  if (oldid == S_EXPR && oldast && A_TYPEG(oldast) == A_CNST) {
    oldid = S_CONST;
    if ((DT_ISINT(oldtyp) && oldtyp != DT_INT8) || DT_ISREAL(oldtyp)) {
    } else {
      /* logical, complex, etc., use sptrs */
      oldval = A_SPTRG(oldast);
    }
  }
  if (oldid != S_CONST) {
    errsev(87);
    if (DTY(dtype) == TY_CHAR) {
      oldval = getstring(" ", 1);
      oldtyp = DT_CHAR;
    }
    else if (DTY(dtype) == TY_NCHAR) {
      oldval = getstring(" ", 1);
      oldtyp = DT_NCHAR;
    }
    else if (dtype == DT_LOG) {
      oldval = SCFTN_TRUE; /* VMS */
      oldtyp = DT_LOG4;
    } else {
      oldtyp = DT_INT4;
      oldval = 1;
    }
  }

  if (oldtyp != dtype) {
    if (warnflg) {
      if (flg.standard) {
        if (TY_ISINT(oldtyp) && TY_ISINT(dtype)) {
          /* any integer, treated identical */
        } else if (TY_ISLOG(oldtyp) && TY_ISLOG(dtype)) {
          /* any logical, treated identical */
        } else {
          errwarn(91);
        }
      } else {
        if ((TY_ISINT(oldtyp) || TY_ISLOG(oldtyp)) &&
            (TY_ISINT(dtype) || TY_ISLOG(dtype))) {
          /* any integer, any logical, treated identical */
        } else {
          errwarn(91);
        }
      }
    }
    return cngcon(oldval, oldtyp, dtype);
  }
  return oldval;
}

/** \brief Check that the indicated semantic stack entry is a constant of any
           integer type.
    \return the integer value as type ISZ_T.

    Issue an error message if stkptr is not a constant of the correct type.
 */
ISZ_T
chkcon_to_isz(SST *stkptr, LOGICAL warnflg)
{
  int dtype;
  INT cval;
  ISZ_T iszv;

  if (!XBIT(68, 0x1))
    return chkcon(stkptr, DT_INT, warnflg);
  if (SST_IDG(stkptr) == S_CONST) {
    dtype = SST_DTYPEG(stkptr);
    if (DT_ISINT(dtype))
      cval = SST_CVALG(stkptr);
    else {
      cval = chkcon(stkptr, DT_INT8, warnflg);
      dtype = DT_INT8;
    }
    if (size_of(dtype) > 4) {
      INT num[2];
      num[0] = CONVAL1G(cval);
      num[1] = CONVAL2G(cval);
      INT64_2_ISZ(num, iszv);
      return iszv;
    }
    return cval;
  }
  errsev(91);
  return 1;
}

/** \brief Convert expression pointed to by stkptr from its current data type
           to data type dtype.
    \return ILM pointer
 */
INT
chktyp(SST *stkptr, int dtype, LOGICAL warnflg)
{
  int oldtyp;

  /* Change non-decimal constants to integer before mkexpr call */
  /* this might need to change! -nzm */
  if (SST_ISNONDECC(stkptr))
    cngtyp(stkptr, DT_INT);
  if (SST_IDG(stkptr) == S_CONST) {
    oldtyp = SST_DTYPEG(stkptr);
    cngtyp(stkptr, dtype);
    mkexpr1(stkptr);
  } else {
    mkexpr1(stkptr);
    oldtyp = SST_DTYPEG(stkptr);
    cngtyp(stkptr, dtype);
  }
  if (warnflg && (DTYG(oldtyp) != DTYG(dtype)) && DTY(dtype) != TY_NUMERIC &&
      (!(TY_ISINT(DTYG(oldtyp)) || TY_ISLOG(DTYG(oldtyp))) ||
       !(TY_ISINT(DTYG(dtype)) || TY_ISLOG(DTYG(dtype)))))
    errwarn(93);
  return 1;
}

/** \brief Same as chktyp() with the restriction that the expression must be a
           scalar (i.e., not an array/vector form).
 */
INT
chk_scalartyp(SST *stkptr, int dtype, LOGICAL warnflg)
{
  int oldtyp;

  oldtyp = SST_DTYPEG(stkptr);
  if (DTY(oldtyp) == TY_ARRAY)
    errsev(83);
  return (chktyp(stkptr, dtype, warnflg));
}

/** \brief Same as chktyp() with the restriction that the expression must be a
           scalar (i.e., not an array/vector form) and integer (i.e., not
   logical).
 */
INT
chk_scalar_inttyp(SST *stkptr, int dtype, const char *msg)
{
  int oldtyp;

  oldtyp = SST_DTYPEG(stkptr);
  if (DTY(oldtyp) == TY_ARRAY)
    errsev(83);
  else if (!DT_ISNUMERIC(oldtyp) || DT_ISLOG(oldtyp))
    error(155, 3, gbl.lineno, msg, "must be numeric");
  else if (flg.standard && !DT_ISINT(oldtyp))
    error(170, 2, gbl.lineno, msg, "is not integer");
  return (chktyp(stkptr, dtype, FALSE));
}

/** \brief Restrict the expression to be suitable for an array extent.
 */
INT
chk_arr_extent(SST *stkptr, const char *msg)
{
  if (flg.standard)
    return chk_scalar_inttyp(stkptr, astb.bnd.dtype, msg);
  else
    return chk_scalartyp(stkptr, astb.bnd.dtype, FALSE);
}

/** \brief Convert expression pointed to by stkptr from its current data type to
           a data type consistent with subscripting.
    \return the ILM pointer
 */
INT
chksubscr(SST *stkptr, int sptr)
{
  /* Change non-decimal constants to integer before mkexpr call */
  if (SST_ISNONDECC(stkptr))
    cngtyp(stkptr, astb.bnd.dtype);
  mkexpr1(stkptr);
  if (!TY_ISINT(DTYG(SST_DTYPEG(stkptr))))
    error(103, 2, gbl.lineno, SYMNAME(sptr), CNULL);
  if (rank_of_ast(SST_ASTG(stkptr)) > 1)
    errsev(161);
  if (DTYG(SST_DTYPEG(stkptr)) != TY_INT8 && DTY(SST_DTYPEG(stkptr)) != TY_ARRAY)
    cngtyp(stkptr, astb.bnd.dtype);
  return 1;
}

/** \brief Cast a given semantic entry into a desired type.
           No type conversion is done.

    \return 1 if no error, -1 if error

    The following casts are ok:
      1. Cast any of the data types to TY_WORD or TY_DWORD (necessary for the
         bitwise intrinsics and relational comparisons)
      2. Cast a TY_WORD or TY_DWORD to any of the data types (necessary for
         casting the bitwise intrinsics back to a data type)

    Since this is used primarily for the bitwise intrinsics, there is no need
    to support TY_DBLE, TY_CHAR, TY_CMPX, or TY_DCMPX since these types are
    illegal for these intrinsics.  However, cast of TY_DWORD and TY_WORD to
    TY_CMPX and TY_DCMPLX is needed for relational comparisons.
    Comparisons between vector typed  and typeless operands require typed
    vectors to be casted to typeless vectors.
 */
int
casttyp(SST *old, int newcast)
{
  int im, from, isvector;

  from = SST_DTYPEG(old);
  if (SST_IDG(old) == S_ACONST && DTY(from) == TY_ARRAY &&
      DTY(newcast) == TY_ARRAY &&
      size_of(DTY(from + 1)) == size_of(DTY(newcast + 1))) {
    ACL *aclp;
    aclp = SST_ACLG(old);
    aclp->dtype = newcast;
    SST_DTYPEP(old, newcast);
    return 1;
  }
  isvector = FALSE;
  if ((from > DT_LOG8 && DTY(from) != TY_ARRAY) || newcast > DT_LOG8)
    goto err_exit;
  /*
  if (from > DT_LOG || newcast > DT_LOG)
      goto err_exit;
  */

  if (DTY(from) == TY_ARRAY) {
    isvector = TRUE;
    from = DTYG(from);
    im = 1;
  } else if (newcast == DT_WORD || newcast == DT_DWORD)
    im = cast_types[from][newcast - 1][0];
  else if (from == DT_WORD || from == DT_DWORD)
    im = cast_types[from][from - 1][1];
  else
    goto err_exit;

  if (im < 0)
    goto err_exit;

  if (from == DT_HOLL) {
    /* default int is integer*8 and 64-bit precision, convert to DT_INT8. */
    if (DTY(stb.user.dt_int) == TY_INT8) {
      cngtyp(old, DT_INT8);
      from = DT_INT8;
    } else if (newcast == DT_WORD)
      cngtyp(old, DT_INT);
    else
      cngtyp(old, DT_REAL8);
  }
  /*  -nzm must not make it look like an integer
  if (from == DT_WORD)
      SST_DTYPEP(old, DT_INT);	keep mkexpr1 happy
  */
  mkexpr1(old);
  if (isvector)
    DTY(SST_DTYPEP(old, get_type(3, TY_ARRAY, newcast)) + 2) = 0;
  else
    SST_DTYPEP(old, newcast);
  return 1;

err_exit:
  errsev(95);
  return (-1);
}

/** \brief Convert expression pointed-to by old to the data type newtyp.

    If newtyp points to a TY_ARRAY entry or newshape is true then old is
    converted to an array.

    \param old points to the semantic stack entry with the old data type.
    \param newtyp is the new dtype for the old semantic stack entry.
    \param allowPolyExpr is true when we want to allow type extension in our
           type comparison. 
 */
static void
cngtyp2(SST *old, DTYPE newtyp, bool allowPolyExpr)
{
  DTYPE oldtyp;
  int to, from;
  int fromisv;
  int ast;
  bool have_unl_poly;

  if (newtyp == 0)
    return;
  oldtyp = SST_DTYPEG(old);

  have_unl_poly = allowPolyExpr && is_dtype_unlimited_polymorphic(newtyp);

  /* handle constants elsewhere */
  if (SST_IDG(old) == S_CONST && !have_unl_poly) {
    /* if not scalar as in structure=constant then cngcon will fail
     * so we will assume type of integer.
     */
    newtyp = DDTG(newtyp);
    if (TY_ISSCALAR(DTY(newtyp)))
      SST_DTYPEP(old, newtyp);
    else
      SST_DTYPEP(old, DT_INT);
    SST_CVALP(old, cngcon(SST_CVALG(old), oldtyp, newtyp));
    if (newtyp == DT_NUMERIC)
      SST_DTYPEP(old, oldtyp);
    else if (oldtyp != newtyp) {
      ast = mk_convert((int)SST_ASTG(old), newtyp);
      SST_ASTP(old, ast);
      mk_alias(ast, mk_cval1(SST_CVALG(old), newtyp));
      SST_SHAPEP(old, A_SHAPEG(ast));
    }
    return;
  }

  to = DTYG(newtyp);
  from = DTY(oldtyp);

  if (from == TY_ARRAY) {
    fromisv = TRUE;
    from = DTYG(oldtyp);
  } else
    fromisv = FALSE;

  /* If the conversion is FROM or TO a typeless value, perform a
   * casting operation.
   */
  if (from == TY_WORD || from == TY_DWORD || to == TY_WORD || to == TY_DWORD) {
    (void)casttyp(old, newtyp);
    return;
  }

  if (from == to) {
    if (from == TY_CHAR) {
      if (DDTG(oldtyp) == DDTG(newtyp))
        return;
    } else if (from == TY_NCHAR) {
      if (DDTG(oldtyp) == DDTG(newtyp))
        return;
    } else if (from != TY_STRUCT && from != TY_DERIVED)
      return;
  }

  if (F77OUTPUT) {
    if (TY_ISLOG(to) && (!TY_ISLOG(from)))
      /* "Illegal type conversion $" */
      error(432, 2, gbl.lineno, "to logical", CNULL);
    if (TY_ISLOG(from) && (!TY_ISLOG(to)))
      error(432, 2, gbl.lineno, "from logical", CNULL);
  }

  switch (to) {

  case TY_BLOG:
  case TY_SLOG:
    cngtyp(old, DT_LOG);
    SST_DTYPEP(old, DT_LOG);
    break;
  case TY_BINT:
  case TY_SINT:
    cngtyp(old, DT_INT);
    SST_DTYPEP(old, DT_INT);
    break;

  case TY_LOG:
  case TY_INT:
    switch (from) {
    case TY_LOG:
    case TY_INT:
      goto done;
    case TY_BLOG:
    case TY_BINT:
      break;
    case TY_SLOG:
    case TY_SINT:
      break;
    case TY_LOG8:
    case TY_INT8:
      break;
    case TY_CMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      break;
    case TY_DCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      break;
#endif
    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;
    default:
      goto type_error;
    }
    FLANG_FALLTHROUGH;
  case TY_LOG8:
  case TY_INT8:
    switch (from) {
    case TY_LOG8:
    case TY_INT8:
      goto done;
    case TY_BLOG:
    case TY_BINT:
      break;
    case TY_SLOG:
    case TY_SINT:
      break;
    case TY_LOG:
    case TY_INT:
      break;
    case TY_CMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      break;
    case TY_DCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      break;
#endif
    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;
    default:
      goto type_error;
    }
    break;
  case TY_REAL:
    switch (from) {
    case TY_BLOG:
    case TY_BINT:
    case TY_SLOG:
    case TY_SINT:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      break;
    case TY_CMPLX:
      break;
    case TY_DCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      break;
#endif
    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;
    default:
      goto type_error;
    }
    break;

  case TY_DBLE:
    switch (from) {
    case TY_BLOG:
    case TY_BINT:
    case TY_SLOG:
    case TY_SINT:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      break;
    case TY_DCMPLX:
      break;
    case TY_CMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      break;
#endif
    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;
    default:
      goto type_error;
    }
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    switch (from) {
    case TY_BLOG:
    case TY_BINT:
    case TY_SLOG:
    case TY_SINT:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      break;
    case TY_QCMPLX:
      break;
    case TY_DCMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      break;
    case TY_CMPLX:
      mkexpr1(old);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      break;
    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;
    default:
      goto type_error;
    }
    break;
#endif

  case TY_CMPLX:
    switch (from) {
    case TY_BINT:
    case TY_BLOG:
    case TY_SINT:
    case TY_SLOG:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
#endif
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      cngtyp(old, DT_REAL);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      if (fromisv)
        mkexpr1(old);
      else
        mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
      mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;

    default:
      goto type_error;
    }

  case TY_DCMPLX:
    switch (from) {
    case TY_BINT:
    case TY_BLOG:
    case TY_SINT:
    case TY_SLOG:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    case TY_REAL:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
#endif
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      cngtyp(old, DT_REAL8);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      if (fromisv)
        mkexpr1(old);
      else
        mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_CMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
#endif
      mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
      FLANG_FALLTHROUGH;

    default:
      goto type_error;
    }

  case TY_CHAR:
  case TY_NCHAR:
    if (from != to) {
      goto type_error;
    }
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    switch (from) {
    case TY_BINT:
    case TY_BLOG:
    case TY_SINT:
    case TY_SLOG:
      cngtyp(old, DT_INT);
      SST_DTYPEP(old, DT_INT);
      FLANG_FALLTHROUGH;
    /* fall thru ... */
    case TY_REAL:
    case TY_DBLE:
    case TY_LOG:
    case TY_INT:
    case TY_LOG8:
    case TY_INT8:
      cngtyp(old, DT_QUAD);
      FLANG_FALLTHROUGH;
    /* fall thru ... */
    case TY_QUAD:
      mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_CMPLX:
    case TY_DCMPLX:
      mkexpr1(old);
      SST_IDP(old, S_EXPR);
      goto done;

    case TY_CHAR:
    case TY_NCHAR:
    case TY_STRUCT:
    case TY_DERIVED:
    /* fall thru ... */

    default:
      goto type_error;
    }
#endif

  case TY_STRUCT:
    if (DDTG(newtyp) != DDTG(oldtyp)) {
      if (from == TY_STRUCT) {
        error(99, 3, gbl.lineno, "RECORD", CNULL);
      } else {
        error(148, 3, gbl.lineno, "RECORD", CNULL);
      }
    }
    return;

  case TY_DERIVED:
    if (DDTG(newtyp) != DDTG(oldtyp)) {
      int new;
      int old;
      new = DDTG(newtyp);
      old = DDTG(oldtyp);

      /* module processing may duplicate dtypes, but if tag is
         the same, then allow them to be considered equal */
      if (same_dtype(old, new))
        return;

      if (DTY(new) == TY_DERIVED) {
        int iso_dt;
        iso_dt = is_iso_cptr(new);
        if (iso_dt) {
          if (is_iso_c_ptr(iso_dt)) {
            error(148, 3, gbl.lineno, "TYPE(C_PTR) expression", CNULL);
            return;
          }
          if (is_iso_c_funptr(iso_dt)) {
            error(148, 3, gbl.lineno, "TYPE(C_FUNPTR) expression", CNULL);
            return;
          }
        }
      }
      if (allowPolyExpr && from == TY_DERIVED && 
          (have_unl_poly || eq_dtype2(oldtyp, newtyp, TRUE) || 
           eq_dtype2(newtyp, oldtyp, TRUE))) {
          return;
      }
      if (from == TY_DERIVED)
        error(99, 3, gbl.lineno, "derived type", CNULL);
      else if (to == TY_DERIVED && UNLPOLYG(DTY(new + 3)) &&
               ((DTY(newtyp) != TY_ARRAY && DTY(oldtyp) != TY_ARRAY) ||
                (DTY(newtyp) == TY_ARRAY && DTY(oldtyp) == TY_ARRAY &&
                 ADD_NUMDIM(newtyp) == ADD_NUMDIM(oldtyp))))
        return;
      else
        error(148, 3, gbl.lineno, "derived type", CNULL);
    }
    return;

  case TY_NUMERIC:
    if (!TY_ISNUMERIC(from))
      goto type_error;
    mkexpr1(old);
    return;

  default:
    goto type_error;
  }

  mkexpr1(old);

done:
  if (flg.standard) {
    if ((to == TY_BLOG || to == TY_SLOG || to == TY_LOG || to == TY_LOG8) &&
        (from == TY_BINT || from == TY_SINT || from == TY_INT ||
         from == TY_INT8 || from == TY_REAL || from == TY_DCMPLX ||
         from == TY_DBLE || from == TY_CMPLX
#ifdef TARGET_SUPPORTS_QUADFP
         || from == TY_QUAD || from == TY_QCMPLX
#endif
         ))
      goto type_error;
    if ((from == TY_BLOG || from == TY_SLOG || from == TY_LOG ||
         from == TY_LOG8) &&
        (to == TY_BINT || to == TY_SINT || to == TY_INT || to == TY_INT8 ||
         to == TY_REAL || to == TY_DCMPLX || to == TY_DBLE || to == TY_CMPLX
#ifdef TARGET_SUPPORTS_QUADFP
         || to == TY_QUAD || to == TY_QCMPLX
#endif
         ))
      goto type_error;
  }

  if (fromisv) {
    newtyp = get_type(3, TY_ARRAY, DDTG(newtyp));
    DTY(newtyp + 2) = DTY(oldtyp + 2);
    SST_DTYPEP(old, newtyp);
  } else
    SST_DTYPEP(old, DDTG(newtyp));
  if (SST_ASTG(old)) {
    SST_ASTP(old, mk_convert(SST_ASTG(old), newtyp));
    SST_SHAPEP(old, A_SHAPEG(SST_ASTG(old)));
  }
  return;

type_error:
  /* assertion:  we get here when user mixes character or record data
   * with numeric data or for unsupported data types such as QUAD.
   */
  if (to == TY_STRUCT)
    error(148, 3, gbl.lineno, "RECORD", CNULL);
  else if (to == TY_DERIVED)
    error(148, 3, gbl.lineno, "derived type", CNULL);
  else if (from == TY_STRUCT || from == TY_DERIVED || from == TY_CHAR ||
           to == TY_CHAR) {
    if (from == TY_STRUCT)
      error(99, 3, gbl.lineno, "RECORD", CNULL);
    else if (from == TY_DERIVED)
      error(99, 3, gbl.lineno, "derived type", CNULL);

    if (from == TY_CHAR)
      errsev(147);
    else if (to == TY_CHAR)
      errsev(146);

    SST_IDP(old, S_EXPR);
    fromisv = FALSE;
    goto done;
  } else
    errsev(95);
  /* prevent further errors */
  SST_DTYPEP(old, newtyp);
}

/**\brief Convert expression pointed-to by old to the data type newtyp.
 *
 * Main entry point for cngtyp2() that assumes no polymorphic expressions.
 * 
 * \param old points to the semantic stack entry with the old data type.
 * \param newtyp is the new dtype for the old semantic stack entry.
 *
 */
void
cngtyp(SST *old, DTYPE newtyp)
{
   cngtyp2(old, newtyp, false);
}

void
cngshape(SST *old, SST *new)
{
  int from, to;
  LOGICAL fromisv, toisv;
  int ast;
  int newtyp;

  fromisv = (DTY(SST_DTYPEG(old)) == TY_ARRAY) ? TRUE : FALSE;
  from = DTYG(SST_DTYPEG(old));

  newtyp = SST_DTYPEG(new);
  toisv = (DTY(newtyp) == TY_ARRAY) ? TRUE : FALSE;
  to = DTYG(newtyp);

  if (!toisv && !fromisv)
    return; /* both scalars */

  if (fromisv && !toisv) { /* && !is_iso_c_loc(SST_ASTG(old)) */
                           /* can't demote an array to a scalar */
#if DEBUG
    if (is_iso_c_loc(SST_ASTG(old))) {
      interr("cngshape: array-value c_loc", SST_ASTG(old), 3);
    }
#endif
    errsev(83);
    SST_IDP(old, S_EXPR);
    SST_DTYPEP(old, DT_INT);
  } else if (!fromisv && toisv) {
    /* scalar promotion */
    if (!TY_ISVEC(to)) {
      if (to == TY_CHAR)
        UFCHAR;
      else
        errsev(100);
    } else if (!TY_ISVEC(from))
      error83(from);
    else {
      mkexpr1(old);
      if (SST_SHAPEG(new) == 0)
        (void)mkexpr1(new);
      if (to == TY_CHAR) {
        /* scalar character to array of character -- don't change
         * the element type.
         */
        newtyp = dup_array_dtype(newtyp);
        DTY(newtyp + 1) = SST_DTYPEG(old);
      }
      ast = mk_promote_scalar((int)SST_ASTG(old), newtyp, (int)SST_SHAPEG(new));
      SST_ASTP(old, ast);
      SST_DTYPEP(old, newtyp);
      SST_SHAPEP(old, A_SHAPEG(ast));
    }
  } else {
#if DEBUG
    assert(fromisv && toisv, "chgshape:both vectors", 0, 3);
#endif
    if (SST_SHAPEG(old) == 0)
      (void)mkexpr1(old);
    if (SST_SHAPEG(new) == 0)
      (void)mkexpr1(new);
    if (!conform_shape((int)SST_SHAPEG(old), (int)SST_SHAPEG(new)))
      error(153, 3, gbl.lineno, CNULL, CNULL);
  }
}

/** \brief Semantically check an operand (old) for array conformance with
           operand new. If the operand is a scalar, change the shape of the
           operand to conform with the expected shape.  If the operand is an
           array, check for conformance.
    \param old     operand to check
    \param new     operand to conform with
    \param promote if true, promote scalar to vector
    \return TRUE if shapes are conformant; false, otherwise.
 */
LOGICAL
chkshape(SST *old, SST *new, LOGICAL promote)
{
  int from;

  from = SST_DTYPEG(old);
  if (DTY(from) == TY_ARRAY)
    return conform_shape((int)SST_SHAPEG(old), (int)SST_SHAPEG(new));

  /* old is scalar */

  if (promote) {
    int ast;
    int newtyp;
    newtyp = dup_array_dtype((int)SST_DTYPEG(new));
    DTY(newtyp + 1) = from;
    ast = mk_promote_scalar((int)SST_ASTG(old), newtyp, (int)SST_SHAPEG(new));
    SST_ASTP(old, ast);
    SST_DTYPEP(old, newtyp);
    SST_SHAPEP(old, A_SHAPEG(ast));
  }

  return TRUE;
}

int
chklog(SST *stkptr)
{
  LOGICAL notlog;

  notlog = (flg.standard) ? (!TY_ISLOG(DTYG(SST_DTYPEG(stkptr))))
                          : (!TY_ISINT(DTYG(SST_DTYPEG(stkptr))));
  if (SST_IDG(stkptr) != S_CONST) {
    if (notlog) {
      errsev(121);
      SST_IDP(stkptr, S_CONST);
      SST_CVALP(stkptr, 0);
      SST_DTYPEP(stkptr, DT_LOG);
      mkexpr1(stkptr);
    } else {
      mkexpr1(stkptr);
/*  Change only different sizes of logicals to a
 *  logical. Change to integer data type is done
 *  in chkopnds since only at that point if either of
 *  the operands is an integer we want to change the
 *  operation to bitwise logical.
 */
    }
  } else {
    /* the operand is a constant */
    if (!flg.standard && DTYG(SST_DTYPEG(stkptr)) == TY_DWORD)
      cngtyp(stkptr, DT_INT8);
    else if (!flg.standard && DTYG(SST_DTYPEG(stkptr)) == TY_CHAR)
      cngtyp(stkptr, DT_LOG);
    else {
      if (!SST_ISNONDECC(stkptr) && notlog) {
        /* Fix constants that are not ultimately int, char or log */
        errsev(121);
        SST_CVALP(stkptr, 0);
        SST_DTYPEP(stkptr, DT_LOG);
      }
    }
  }

  return 1;
}

void
mkident(SST *stkptr)
{
  SST_IDP(stkptr, S_IDENT);
  SST_ALIASP(stkptr, 0);
  SST_CVLENP(stkptr, 0);
  SST_SHAPEP(stkptr, 0);
}

int
mkexpr(SST *stkptr)
{
  mkexpr1(stkptr);
  mklogint4(stkptr);
  return 1;
}

/*---------------------------------------------------------------------*/

/** \brief Given a semantic stack entry, write ILM's for the expression
           represented by the stack entry if they have not already been written.
    \return pointer to ILM
 */
int
mkexpr1(SST *stkptr)
{
  int dtype;
  int sptr;
  INT num[2];
  int shape;
  extern int dont_issue_assumedsize_error;

again:
  switch (SST_IDG(stkptr)) {
  case S_STFUNC: /* delayed var ref */
    mkident(stkptr);
    (void)mkvarref(stkptr, SST_ENDG(stkptr));
    goto again;

  case S_CONST:
    SST_CVLENP(stkptr, 0);
    dtype = SST_DTYPEG(stkptr);
    sptr = SST_SYMG(stkptr);
    /* generate constant ILM */
    switch (DTY(dtype)) {
    case TY_DWORD:
      dtype = DT_DWORD;
      SST_DTYPEP(stkptr, DT_DWORD);
      break;
    case TY_WORD:
      dtype = DT_WORD;
      SST_DTYPEP(stkptr, DT_WORD);
      break;
    case TY_INT:
    case TY_BINT:
    case TY_SINT:
      break;
    case TY_INT8:
    case TY_LOG8:
      break;
    case TY_LOG:
    case TY_BLOG:
    case TY_SLOG:
      break;
    case TY_REAL:
      break;
    case TY_DBLE:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      break;
#endif
    case TY_CMPLX:
      break;
    case TY_DCMPLX:
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      break;
#endif
    case TY_CHAR:
      break;
    case TY_NCHAR:
      /*  replace sptr to TY_CHAR const by TY_NCHAR constant: */
      num[0] = sptr;
      num[1] = 0;
      sptr = getcon(num, dtype);
      break;
    default:
      interr("mkexpr1: bad const", dtype, 3);
      SST_IDP(stkptr, S_EXPR);
      return 1;
    }
    SST_IDP(stkptr, S_EXPR);
    return 1;

  case S_ACONST:
    shape = 0;
    if (SST_ACLG(stkptr) == 0) {
      int sdtype;
      sptr = sym_get_array("zs", "array", SST_DTYPEG(stkptr), 1);
      sdtype = DTYPEG(sptr);
      ADD_LWBD(sdtype, 0) = ADD_LWAST(sdtype, 0) = astb.bnd.one;
      ADD_UPBD(sdtype, 0) = ADD_UPAST(sdtype, 0) = astb.bnd.zero;
      ADD_EXTNTAST(sdtype, 0) =
          mk_extent(ADD_LWAST(sdtype, 0), ADD_UPAST(sdtype, 0), 0);
      mkident(stkptr);
      SST_SYMP(stkptr, sptr);
      SST_DTYPEP(stkptr, dtype = DTYPEG(sptr));
    } else {
      sptr = init_sptr_w_acl(0, SST_ACLG(stkptr));
      SST_IDP(stkptr, S_LVALUE);
      SST_DTYPEP(stkptr, dtype = DTYPEG(sptr));
      SST_LSYMP(stkptr, sptr);
    }
    SST_ASTP(stkptr, mk_id(sptr));
    goto lval;

  case S_IDENT:
    /* need to set data type, stack type */
    dtype = 0;
    sptr = SST_SYMG(stkptr);
    shape = 0;
    get_next_hash_link(sptr, 0);
  retry:
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
      if (SCG(sptr) == SC_DUMMY && ASUMSZG(sptr) &&
          !dont_issue_assumedsize_error)
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- extent of assumed size array is unknown");
      if (ALLOCATTRG(sptr) && STYPEG(sptr) == ST_MEMBER && SDSCG(sptr) == 0 &&
          !F90POINTERG(sptr)) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
        ASSUMSHPP(sptr, 0);
        SDSCS1P(sptr, 1);
      }
      goto var_primary;
    case ST_PD:
#ifdef I_N_PES
      if (sptr == intast_sym[I_N_PES])
        return ref_pd(stkptr, ITEM_END);
#endif
      FLANG_FALLTHROUGH;
    case ST_INTRIN:
    case ST_GENERIC:
      if (sem.dinit_data) {
        return 1;
      }
      if (EXPSTG(sptr)) { /* Frozen as an intrinsic */
        return (mkvarref(stkptr, ITEM_END));
      }
      /* Not a frozen intrinsic, so assume its a variable */
      sptr = newsym(sptr);
      sem_set_storage_class(sptr);
      FLANG_FALLTHROUGH;
    case ST_UNKNOWN:
    case ST_IDENT:
      STYPEP(sptr, ST_VAR);
      FLANG_FALLTHROUGH;
    case ST_VAR:
    case ST_STRUCT:
    case ST_MEMBER:
      if (((ALLOCATTRG(sptr) && STYPEG(sptr) == ST_MEMBER) || POINTERG(sptr)) &&
          SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
        if (SCG(sptr) == SC_NONE)
          SCP(sptr, SC_BASED);
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
      FLANG_FALLTHROUGH;
    case ST_DESCRIPTOR:
    var_primary:
      SST_IDP(stkptr, S_LVALUE);
      sptr = ref_object(sptr);
      SST_DTYPEP(stkptr, dtype = DTYPEG(sptr));
      SST_LSYMP(stkptr, sptr);
      SST_ASTP(stkptr, mk_id(sptr));
      SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
      goto lval;
    case ST_ENTRY:
      if (gbl.rutype == RU_FUNC) {
        SST_IDP(stkptr, S_EXPR);
        SST_DTYPEP(stkptr, dtype = DTYPEG(sptr));
        sptr = ref_entry(sptr);
        SST_ASTP(stkptr, mk_id(sptr));
        goto lval;
      }
      error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      return 1;
    case ST_PROC:
      dtype = DTYPEG(sptr);
      if (dtype == 0) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- attempt to use a SUBROUTINE as a FUNCTION");
        SST_DTYPEP(stkptr, DT_INT);
        return 1;
      }
      SST_DTYPEP(stkptr, dtype);
      SST_ASTP(stkptr, mk_id(sptr));
      return func_call(stkptr, (ITEM *)NULL);
    case ST_USERGENERIC:
      do {
        /* This symbol might be overloading the intended symbol.
         * Attempt to locate it.
         */
        sptr = get_next_hash_link(sptr, 2);
        if (test_scope(sptr)) {
          if (STYPEG(sptr) == ST_PARAM) {
            dtype = DTYPEG(sptr);
            SST_IDP(stkptr, S_CONST);
            SST_SYMP(stkptr, sptr);
            SST_DTYPEP(stkptr, dtype);
            SST_CVLENP(stkptr, 0);
            SST_ASTP(stkptr, mk_cnst(sptr));
            goto again;
          }
          goto retry;
        }
      } while (sptr > NOSYM);
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- attempt to use a GENERIC subprogram as a FUNCTION");
      SST_DTYPEP(stkptr, DT_INT);
      return 1;
    default:
      error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      SST_DTYPEP(stkptr, DT_INT);
      SST_IDP(stkptr, S_EXPR);
      return 1;
    }
    /* NOTREACHED */;

  case S_LVALUE:
    dtype = SST_DTYPEG(stkptr);
    sptr = SST_LSYMG(stkptr);
  lval:
    SST_CVLENP(stkptr, 0);
    if (dtype == 0)
      interr("mkexpr1: 0 dtype", dtype, 3);
    else if ((DTY(dtype) == TY_STRUCT) || (DTY(dtype) == TY_UNION) ||
             ((DTY(dtype) == TY_DERIVED)))
      return 1;
    else if (DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) {
      if (!DTY(dtype + 1) ||
          !A_ALIASG(DTY(dtype + 1))) { /* nonconstant char length */
        SST_CVLENP(stkptr, size_ast(sptr, dtype));
      }
      return 1;
    } else if (DT_ISBASIC(dtype))
      ;
    else if (DTY(dtype) == TY_ARRAY) {
      /* base - element handled separately; don't use 'TY_ISVEC' here!
       * We don't know the intended usage of this expression; it still
       * could be an argument and we want to allow character arrays
       * as arguments.
       */
      int dd;
      dd = DTY(dtype + 1);
      if (DTY(dd) == TY_STRUCT) {
        error83(DTY(dd));
        SST_DTYPEP(stkptr, DDTG(dtype));
        return 1;
      }
      if ((DTY(dd) == TY_CHAR || DTY(dd) == TY_NCHAR)) {
        if (!DTY(dd + 1) ||
            !A_ALIASG(DTY(dd + 1))) { /* nonconstant char length */
          SST_CVLENP(stkptr, size_ast(sptr, dd));
        }
      }
    } else
      interr("mkexpr1: bad dtype", dtype, 3);

    if (DTY(dtype) != TY_ARRAY) {
      shape = 0;
    } else {
      shape = A_SHAPEG(SST_ASTG(stkptr));
    }
    SST_DTYPEP(stkptr, dtype);
    SST_IDP(stkptr, S_EXPR);
    SST_SHAPEP(stkptr, shape);
    return 1;

  case S_LOGEXPR: /* ILMs have been written */
  case S_EXPR:    /* ILMs have been written */
    return 1;

  case S_VAL:
  case S_REF:
    /* %val(x) -- shouldn't appear here */
    errsev(53);
    SST_IDP(stkptr, S_EXPR);
    return 1;
  case S_STAR:
  /* (*) -- shouldn't appear here */
  default:
    interr("mkexpr1: bad id", SST_IDG(stkptr), 3);
    return 1;
  }
}

/** \brief Same as mkexpr1(), but the expression is the target of a pointer
           assignment.  Must handle ST_PROCs as identifiers; otherwise, just
           call mkexpr1().
 */
int
mkexpr2(SST *stkptr)
{
  int dt;
  int sptr;

  switch (SST_IDG(stkptr)) {
  case S_IDENT:
    sptr = SST_SYMG(stkptr);
    switch (STYPEG(sptr)) {
    case ST_PROC:
      sptr = ref_object(sptr);
      SST_DTYPEP(stkptr, DTYPEG(sptr));
      SST_ASTP(stkptr, mk_id(sptr));
      SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
      SST_CVLENP(stkptr, 0);
      dt = DDTG(DTYPEG(sptr)); /* element dtype record */
      if ((DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) && ADJLENG(sptr)) {
        SST_CVLENP(stkptr, size_ast(sptr, dt));
      }
      return 1;
    default:;
    }
  }
  return mkexpr1(stkptr);
}

/** \brief Convert all sizes of logicals and integers to 4 byte versions.
 */
void
mklogint4(SST *stkptr)
{
}

/** \brief Check for legal variable to be assigned to.
    \param stkptr    the variable to check
    \param stmt_type type of statement we are processing, from the table below
    \return The sptr of the variable if \a stmt_type indicates an index
   variable.<br>
        Otherwise the ILM pointer to address expression for the destination.<br>
        Zero is returned for cases where we want to avoid assignment code
   generation.

    Possible values for \a stmt_type:
    <pre>
        0 - Do index var
        1 - Assignment statement
        2 - Data statement
        3 - LOC reference
        4 - Implied do index var
        5 - Forall index var
    </pre>
 */
int
mklvalue(SST *stkptr, int stmt_type)
{
  int dcld, lval;
  DTYPE dtype;
  SPTR sptr;
  bool is_index_var = stmt_type == 0 || stmt_type == 4 || stmt_type == 5;

  lval = 0;
  SST_CVLENP(stkptr, 0);
  switch (SST_IDG(stkptr)) {
  case S_IDENT: /* Scalar or whole array references */
    // DO CONCURRENT and FORALL index vars are construct entities that are
    // not visible outside of the construct.  If sptr is external to the
    // construct, get a new var.  Use an explicit type if there is one.
    sptr = SST_SYMG(stkptr);
    SST_SHAPEP(stkptr, 0);
    if (stmt_type == 0 && sem.doconcurrent_symavl) {
      dtype = sem.doconcurrent_dtype ? sem.doconcurrent_dtype : DTYPEG(sptr);
      dcld  = sem.doconcurrent_dtype || DCLDG(sptr);
      if (sptr < sem.doconcurrent_symavl)
        sptr = insert_sym(sptr);
      DTYPEP(sptr, dtype);
      DCLDP(sptr, dcld);
      DCLCHK(sptr);
    } else if (stmt_type == 5) {
      int doif = sem.doif_depth;
      dtype = DI_FORALL_DTYPE(doif) ? DI_FORALL_DTYPE(doif) : DTYPEG(sptr);
      dcld  = DI_FORALL_DTYPE(doif) || DCLDG(sptr);
      if (sptr < DI_FORALL_SYMAVL(doif))
        sptr = insert_sym(sptr);
      DTYPEP(sptr, dtype);
      DCLDP(sptr, dcld);
      DCLCHK(sptr);
    }

    switch (STYPEG(sptr)) {
    case ST_ENTRY:
      if (stmt_type == 3) {
        SST_ASTP(stkptr, mk_id(sptr));
        return 1;
      }
      if (gbl.rutype == RU_FUNC && stmt_type != 2) {
        dtype = DTYPEG(sptr); /* use dtype of entry, not func val */
        sptr = ref_entry(sptr);
        DTYPEP(sptr, dtype);
      } else {
        if (is_index_var)
          goto do_error;
        if (stmt_type == 2)
          sem.dinit_error = TRUE;
        error(72, 3, gbl.lineno, "entry point", SYMNAME(sptr));
      }
      break;

    case ST_UNKNOWN:
    case ST_IDENT:
      STYPEP(sptr, ST_VAR);
      FLANG_FALLTHROUGH;
    case ST_VAR:
      if (POINTERG(sptr) && SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
        if (SCG(sptr) == SC_NONE)
          SCP(sptr, SC_BASED);
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
      break;

    case ST_STRUCT:
    struct_error:
      if (flg.standard)
        error(179, 2, gbl.lineno, SYMNAME(sptr), CNULL);
      if (stmt_type == 2 || is_index_var) {
        sem.dinit_error = TRUE;
        if (is_index_var)
          goto do_error;
        error(150, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      }
      break;

    case ST_ARRAY:
      if (is_index_var)
        goto do_error;
      else if (stmt_type == 2 && DTYG(SST_DTYPEG(stkptr)) == TY_STRUCT)
        goto struct_error;
      else if (stmt_type == 1 && SCG(sptr) == SC_DUMMY && ASUMSZG(sptr))
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- extent of assumed size array is unknown");
      break;

    case ST_PD:
    case ST_GENERIC:
    case ST_INTRIN:
      if (!EXPSTG(sptr)) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        /* need storage class (local) */
        sem_set_storage_class(sptr);
        break;
      }
      /* ERROR, intrinsic is frozen - give lvalue valid data type */
      if (STYPEG(sptr) == ST_GENERIC && DTYPEG(sptr) == DT_NONE) {
        if (GSAMEG(sptr))
          /* Specific of same name so use its data type */
          DTYPEP(sptr, DTYPEG(GSAMEG(sptr)));
        else
          setimplicit(sptr);
      }
      FLANG_FALLTHROUGH;

    case ST_PROC: /* Function/intrinsic reference used as an lvalue */
      if (stmt_type == 3) {
        SST_ASTP(stkptr, mk_id(sptr));
        return 1;
      }
      if (is_index_var)
        goto do_error;
      error(72, 3, gbl.lineno, "external procedure", SYMNAME(sptr));
      if (stmt_type == 2)
        sem.dinit_error = TRUE;
      return (0);

    case ST_USERGENERIC:
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- attempt to use a generic subprogram name as a variable");
      SST_DTYPEP(stkptr, DT_INT);
      return 1;

    default:
      error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      SST_DTYPEP(stkptr, DT_INT);
      SST_ASTP(stkptr, mk_id(sptr));
      SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
      return sptr;
    }

    if (sem.parallel || sem.task || sem.target || sem.teams
        || sem.orph
        ) {
      if (stmt_type == 0) {
        switch (DI_ID(sem.doif_depth)) {
        case DI_TARGTEAMSDIST:
        case DI_TEAMSDIST:
        case DI_TARGTEAMSDISTPARDO:
        case DI_TEAMSDISTPARDO:
        case DI_DISTRIBUTE:
        case DI_DISTPARDO:
        case DI_SIMD:
        case DI_PARDO:
        case DI_TASKLOOP:
          /* parallel and those work-sharing do variables must be private */
          sptr = decl_private_sym(sptr);
          if (SCG(sptr) != SC_PRIVATE) {
            /*
             * the symbol created isn't private presumably
             * because there was an explicit shared declaration
             * of the index variable on the parallel do.
             * Just insert a new symbol (ST_UNKNOWN) and
             * declare as private.  Another solution to this
             * problem is to push 2 par scopes when the
             * parallel do is processed by semsmp.c.
             */
            int new;
            new = insert_sym(sptr);
            DTYPEP(new, DTYPEG(sptr));
            sptr = decl_private_sym(new);
          }
          break;
        case DI_PDO:
          /* parallel work-sharing do variables must be private */
          sptr = decl_private_sym(sptr);
          break;
        case DI_TASK:
          /* do variables within tasks must be private */
          sptr = decl_private_sym(sptr);
          break;
        case DI_ATOMIC_CAPTURE:
          /* no special handling for atomic capture. */
          break;
        default:
          /* a sequential do index variable within a parallel region,
           * if otherwise shared based on default rules, must be
           * private.
           * First, call sem_check_scope() to see if was explicitly
           * declared shared or private -- if the returned symbol has
           * scope 0, then must create a private copy.
           */
          sem.ignore_default_none = TRUE;
          sptr = sem_check_scope(sptr, sptr);
          sem.ignore_default_none = FALSE;
          if (sem.parallel || sem.task) {
            sptr = decl_private_sym(sptr);
          } else if (SCOPEG(sptr) == gbl.currsub) {
            sptr = decl_private_sym(sptr);
          } else if (SCOPEG(sptr) == stb.curr_scope) {
            sptr = decl_private_sym(sptr);
#if DEBUG
            if (XBIT(69, 0x80000000))
              error(155, 2, gbl.lineno,
                    "DO variable in contained procedure is PRIVATE -",
                    SYMNAME(sptr));
#endif
          }
#if DEBUG
          else if (SCG(sptr) != SC_PRIVATE) {
            if (XBIT(69, 0x80000000))
              error(155, 2, gbl.lineno, "DO variable is not PRIVATE -",
                    SYMNAME(sptr));
          }
#endif
          break;
        }
      } else if (stmt_type == 4) {
        /* Implied do variables must be private */
        /* We currently have a bug where if a private variable is
         * created, it will not be reflected in any of the ILMs which
         * have already been generated for the I/O items referencing
         * the do variable. For now, don't create a new symbol; just
         * use whatever symbol is in scope -- at least the I/O
         * code is within a critical section and the user can just
         * add a PRIVATE clause as a workaround.
        sptr = decl_private_sym(sptr);
         */
        ;
      } else if (stmt_type == 5) {
        /* Forall variables must be private */
        /* if variable is already private, create another
         * private sptr for this forall. We call pop_sym(sptr)
         * hash table in check_no_scope_sptr()
         * once it exists forall construct.
         * !omp parallel private(i)
         * print *, i
         * forall(i=1:N) b(i) = k(i)
         * print *, i
         * the value of i before and after forall should be the same
         * i inside forall has it forall scope.
         */
        if (SCG(sptr) == SC_PRIVATE)
          sptr = insert_sym(sptr);
        sptr = decl_private_sym(sptr);
      }
    } else if (stmt_type == 0 && DI_ID(sem.doif_depth) == DI_PDO) {
      sptr = decl_private_sym(sptr);
    } else if (stmt_type == 0 && (DI_ID(sem.doif_depth) == DI_SIMD)) {
      sptr = decl_private_sym(sptr);
    }
    /*    Induction variables can be inside of struct frame pointer that is passed
       by caller subroutine. To use them, the compiler needs to extract them inside
       of the loop. It might the compiler to think there are additional codes
       between the loops even though the loops are tightly nested. In this case, the
       compiler might not generate parallel code. Here, we create a new variable
       with the same name of induction variables.
    */
    if (stmt_type == 0 && flg.smp && (SCG(sptr) != SC_PRIVATE) && 
            sem.expect_cuf_do ) {
       int newsptr;
       newsptr = insert_sym(sptr);
       DCLDP(newsptr, TRUE);
       DTYPEP(newsptr, DTYPEG(sptr));
       STYPEP(newsptr, STYPEG(sptr));
       sptr = newsptr;
       sem.index_sym_to_pop = newsptr;
     }

    sptr = ref_object(sptr);
    SST_DTYPEP(stkptr, DTYPEG(sptr));
    dtype = DDTG(DTYPEG(sptr)); /* element dtype record */
    if (stmt_type == 1) {
      DOCHK(sptr);
    }
    SST_ASTP(stkptr, mk_id(sptr));
    SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
    if ((DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) && ADJLENG(sptr)) {
      SST_CVLENP(stkptr, size_ast(sptr, dtype));
    }
    if (stmt_type == 3) {
      int subs[MAXDIMS], numdim, i, ast;
      ADSC *ad;
      if (SCG(sptr) == SC_DUMMY && ASSUMSHPG(sptr)) {
        ad = AD_DPTR(DTYPEG(sptr));
        numdim = AD_NUMDIM(ad);
        for (i = 0; i < numdim; i++) {
          subs[i] = AD_LWBD(ad, i);
          if (subs[i] == 0 || STYPEG(subs[i]) != ST_CONST) {
            subs[i] = AD_LWAST(ad, i);
          }
        }
        ast = SST_ASTG(stkptr);
        ast = mk_subscr(ast, subs, numdim, dtype);
        SST_ASTP(stkptr, ast);
      } else if (POINTERG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY) {
        ad = AD_DPTR(DTYPEG(sptr));
        numdim = AD_NUMDIM(ad);
        for (i = 0; i < numdim; i++) {
          subs[i] = AD_LWAST(ad, i);
        }
        ast = SST_ASTG(stkptr);
        ast = mk_subscr(ast, subs, numdim, dtype);
        SST_ASTP(stkptr, ast);
      }
    }
    break;

  case S_LVALUE:
    /*
     * We have any combination of the following: 1) subscripted array,
     * 2) char substring,  3) member ref.
     * These references are disallowed as DO index variables.
     */
    sptr = SST_LSYMG(stkptr);
    lval = SST_ASTG(stkptr);
    if (is_index_var) {
      if (STYPEG(sptr) != ST_VAR)
        goto do_error;
      return sptr; /* SST_OPTYPE field is correct */
    }

    /* If LOC applied to an array section, build a new A_SUBSCR
     * replacing triples with the triplet lbound */
    if (stmt_type == 3) {
      if (A_TYPEG(lval) == A_SUBSCR) {
        int i;
        int asd;
        int ast = lval;
        int subs[MAXDIMS] = {0};
        LOGICAL array_section = FALSE;

        asd = A_ASDG(ast);
        for (i = 0; i < (int)(ASD_NDIM(asd)); ++i) {
          if (A_TYPEG(ASD_SUBS(asd, i)) == A_TRIPLE) {
            subs[i] = A_LBDG(ASD_SUBS(asd, i));
            array_section = TRUE;
          } else {
            subs[i] = ASD_SUBS(asd, i);
          }
        }
        if (array_section) {
          ast = mk_subscr(A_LOPG(ast), subs, ASD_NDIM(asd), A_DTYPEG(ast));
          SST_ASTP(stkptr, ast);
        }
      } else if (A_TYPEG(lval) == A_MEM && DTY(A_DTYPEG(lval)) == TY_ARRAY &&
                 POINTERG((sptr = memsym_of_ast(lval)))) {
        int subs[MAXDIMS], numdim, i, ast;
        ADSC *ad;
        ad = AD_DPTR(DTYPEG(sptr));
        numdim = AD_NUMDIM(ad);
        for (i = 0; i < numdim; i++) {
          subs[i] = check_member(lval, AD_LWAST(ad, i));
        }
        ast = mk_subscr(lval, subs, numdim, DTY(DTYPEG(sptr) + 1));
        SST_ASTP(stkptr, ast);
      }
    }

    /* Catch structure references  in DATA stmts */
    if (stmt_type == 2 && DTY(SST_DTYPEG(stkptr)) == TY_STRUCT) {
      sem.dinit_error = TRUE;
      error(150, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    }

    if (DTY(SST_DTYPEG(stkptr)) == TY_ARRAY && !SST_SHAPEG(stkptr))
      SST_SHAPEP(stkptr, mkshape((int)SST_DTYPEG(stkptr)));
    dtype = DDTG(DTYPEG(sptr)); /* element dtype record */
    if ((DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) && ADJLENG(sptr)) {
      SST_CVLENP(stkptr, size_ast(sptr, dtype));
    }
    break;

  case S_CONST:
    /* If TEMP has value then constant was a PARAMETER (so get name) */
    if (is_index_var)
      goto do_error;
    if (SST_ERRSYMG(stkptr) && STYPEG(SST_SYMG(stkptr)) == ST_PARAM)
      error(33, 3, gbl.lineno, SYMNAME(SST_ERRSYMG(stkptr)), CNULL);
    else
      error(33, 3, gbl.lineno, prtsst(stkptr), CNULL);
    if (stmt_type == 2)
      sem.dinit_error = TRUE;
    else if (stmt_type == 3)
      return (0);
    break;

  case S_EXPR:
    if (is_index_var)
      goto do_error;
    if (stmt_type == 3)
      errsev(52);
    else {
      /* For now assume left side was ref to external procedure */
      sptr = SST_ERRSYMG(stkptr);
      if (!sptr)
        sptr = getbase((int)SST_ASTG(stkptr));
      error(72, 3, gbl.lineno, "external procedure", SYMNAME(sptr));
      /*
       * (f21763) attempt to avoid any further errors/ICEs for the symbol,
       * just re-classify the symbol as a 'var' -- if resetting causes
       * worse errors down-stream, just delete thie STYPEP and set the
       * above error to 'fatal'
       */
      STYPEP(sptr, ST_VAR);
      if (stmt_type == 2)
        sem.dinit_error = TRUE;
    }
    return (0);

  case S_ACONST:
    if (is_index_var)
      goto do_error;
    error(33, 3, gbl.lineno, SYMNAME(SST_SYMG(stkptr)), CNULL);
    if (stmt_type == 2)
      sem.dinit_error = TRUE;
    else if (stmt_type == 3)
      return (0);
    break;

  default:
    interr("mklvalue: Unexpected semantic stack entry id", SST_IDG(stkptr), 3);
    break;

  } /* End of switch on semantic stack id */

  if (is_index_var) {
    if (stmt_type == 5 && !PRIVATEG(sptr) && INTENTG(sptr) == 1)
      ; /* we always create a new index variable for forall statement and never
           set ASSNG flag */
    else
      set_assn(sptr);
  } else if (stmt_type == 1 && !POINTERG(lval ? memsym_of_ast(lval) : sptr)) {
    if (!lval) {
        set_assn(sptr);
    }
    else
      set_assn(sym_of_ast(lval));
  } else if (stmt_type == 3)
    ADDRTKNP(sptr, 1);
  if (is_index_var) {
    /* DOCHK(sptr);  perform this check in do_begin() */
    return (sptr);
  }
  return 1;

do_error:
  errsev(106);
  sptr = getccsym('.', 0, ST_VAR);
  DTYPEP(sptr, DT_INT);
  return (sptr);
}

static INT
const_xtoi(INT conval1, INT cnt, int dtype)
{
  union {
    DBLINT64 i64;
    BIGINT64 bgi;
  } u;

  u.bgi = 1;
  if (u.i64[0]) {
    /*  little endian */
    u.i64[0] = CONVAL2G(cnt);
    u.i64[1] = CONVAL1G(cnt);
  } else {
    u.i64[0] = CONVAL1G(cnt);
    u.i64[1] = CONVAL2G(cnt);
  }
  return _xtok(conval1, u.bgi, dtype);
}

/** \brief Link parents for type extension by adding parent as a member to
           the type.
*/
void
link_parents(STSK *stsk, int sptr)
{
  int sptr1;
  int tag;
  if (!sptr)
    return;
  /* Need to call insert_sym() and use the new symbol because a component in
   * another derived type can have the same name as the derived type we're
   * processing. Otherwise, we may have the wrong symbol for our parent
   * symbol in the type extension. Also a derived type name can be overloaded by
   * a generic interface.
   */
  sptr1 = insert_sym(sptr);
  STYPEP(sptr1, ST_MEMBER);
  DTYPEP(sptr1, DTYPEG(sptr));
  /* for the parent member, we just mark it with the PARENT flag assigned
   * to itself since this also works for base types.
   */
  PARENTP(sptr1, sptr1);
  tag = DTY(DTYPEG(sptr) + 3);
  PRIVATEP(sptr1, PRIVATEG(tag));
  DINITP(sptr1, DINITG(sptr));
  if (DINITG(sptr))
    DINITP(stsk->sptr, DINITG(sptr));
  link_members(stsk, sptr1);
}

/** \brief Check parents of type extension for duplicate symbols.

   To Do: Take into account attributes such as access (private/public)
   and overridable.
 */
int
check_parent(int sptr1, int sptr2)
{
  int sptr3;
  for (sptr3 = DTY(DTYPEG(sptr2) + 1); sptr3 != NOSYM; sptr3 = SYMLKG(sptr3)) {
    if (NMPTRG(sptr1) == NMPTRG(sptr3)) {
      return 0;
    } else if (PARENTG(sptr3) == sptr3) {
      int rslt = check_parent(sptr1, sptr3);
      if (!rslt)
        return 0;
    }
  }
  return 1;
}

/** \brief Link together members of a structure.
    \param stsk the structure stack item representing the structure to which
   members are added
    \param sptr points to a list of new members linked via symlk

    The new member list is added to the end of the existing member list watching
    out for duplicate member names.
 */
void
link_members(STSK *stsk, int sptr)
{
  int dtype;
  int sptr1, sptr2, sptr_end;
  int count;
  int member_access;
  int entity_access;

  dtype = stsk->dtype;

  assert((DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_UNION ||
          DTY(dtype) == TY_DERIVED),
         "link_members, unexp. dtype", dtype, 3);
  /*
   * loop thru list of symbols to be added and add them to the LIFO
   * list which represents a flattened list of all the members which
   * occur at the same level.  Recall that we create special members
   * for each union and for each map, where each map is represented
   * by a structure and belongs to a union which contains as members
   * all maps.  the LIFO is created so that we can easily search for
   * conflicts.
   */
  sptr_end = stsk->last; /* current end of LIFO for struct */
  member_access = (stsk->mem_access == 'v');
  entity_access = get_entity_access();
  for (sptr1 = sptr; sptr1 != NOSYM; sptr1 = SYMLKG(sptr1)) {

    /*  loop thru members (LIFO) currently in the structure  */
    for (sptr2 = sptr_end; sptr2 != NOSYM; sptr2 = VARIANTG(sptr2)) {
      if (NMPTRG(sptr1) == NMPTRG(sptr2))
        error(138, 2, gbl.lineno, SYMNAME(sptr1), CNULL);
      if (DTY(DTYPEG(sptr2)) == TY_DERIVED && PARENTG(sptr2) == sptr2 &&
          PARENTG(sptr2) && !check_parent(sptr1, sptr2)) {
        /* type extension */
        error(138, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      }
    }
    VARIANTP(sptr1, sptr_end); /* add new member to LIFO */

    PRIVATEP(sptr1,
             (member_access && entity_access != 'u') ||
                 (!member_access && entity_access == 'v'));
    ENCLDTYPEP(sptr1, dtype);
    sptr_end = sptr1; /* current end */
  }
  stsk->last = sptr_end; /* new last */
                         /*
                          * loop thru all symbols which currently belong to the structure.
                          * Find the last member so that the sptr list is added to the end
                          * of the structure.
                          */
  count = 0;
  if ((sptr2 = DTY(dtype + 1)) == NOSYM)
    /*  first time members are added */
    DTY(dtype + 1) = sptr;
  else {
    /*  find end of members, add list to the end */
    do {
      sptr_end = sptr1 = sptr2;
      sptr2 = SYMLKG(sptr2);
    } while (sptr2 != NOSYM);
    SYMLKP(sptr_end, sptr);
  }
}

/* called if RESULTG(sptr) is set.
 * this must be a recursive reference; find the matching entry point */
static int
test_really_an_entry(int sptr)
{
  int ent;
  /*  scan all entries. NOTE: gbl.entries not yet set  */
  for (ent = gbl.currsub; ent > NOSYM; ent = SYMLKG(ent)) {
    if (FVALG(ent) == sptr) {
      return ent;
    }
  }
  if (sptr == FVALG(gbl.outersub)) {
    /* recursive call to host */
    return gbl.outersub;
  }
  /* no such entry point found, must be an error */
  interr("dangling RESULT variable reference", sptr, 3);
  return 0;
} /* test_really_an_entry */

/** \brief Make a var ref of the form: `<var primary> ( [<ssa list>] )`

    Determine if a function call, array reference, or substring reference, and
    generate appropriate ILMs, shapes, data types. \a stktop is input and
   output.
 */
int
mkvarref(SST *stktop, ITEM *list)
{
  int sptr, dtype, entry;
  int ast;

  switch (SST_IDG(stktop)) {

  case S_ACONST:
    /* I don't think we should get here anymore, but if we do,
       give error and go ahead and process. Leave code in for now
       - it may be needed later for processing named constants */
    interr("mkvarref: array constructor seen", 0, 3);
    sptr = init_sptr_w_acl(0, SST_ACLG(stktop));
    mkident(stktop);
    goto varref_ident;

  case S_DERIVED:
    sptr = SST_SYMG(stktop);
    dtype = DTYPEG(sptr);
    FLANG_FALLTHROUGH;
  case S_IDENT: /* dtype has not been set in semantic stack yet */
    sptr = SST_SYMG(stktop);
  varref_ident:
    switch (STYPEG(sptr)) {
    case ST_UNKNOWN:
    case ST_IDENT:
      dtype = DTYPEG(sptr);
      /* A non-array identifier used with (<ssa list>) notation.  Check
       * for a character substring otherwise it must be a function call.
       */
      if (IS_CHAR_TYPE(DTYG(dtype))) {
        if (list && list != ITEM_END && SST_IDG(list->t.stkp) == S_TRIPLE) {
          STYPEP(sptr, ST_VAR);
          SST_ASTP(stktop, mk_id(sptr));
          chksubstr(stktop, list);
          SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
          return 1;
        }
      }
      if (RESULTG(sptr) && (entry = test_really_an_entry(sptr))) {
        sptr = entry;
        SST_SYMP(stktop, sptr);
        goto really_an_entry;
      }
      if (STYPEG(sptr) == ST_IDENT && SCG(sptr) == SC_LOCAL && AUTOBJG(sptr)) {
        /* Remove from automatic data list */
        int curr = gbl.autobj;
        if (curr == sptr) {
          gbl.autobj = AUTOBJG(sptr);
        } else {
          while (curr > NOSYM) {
            int next = AUTOBJG(curr);
            if (next == sptr)
              break;
            curr = next;
          }
          if (curr > NOSYM) {
            AUTOBJP(curr, AUTOBJG(sptr));
          }
        }
        AUTOBJP(sptr, 0);
      }
      /* must be a function reference */
      STYPEP(sptr, ST_PROC);
      FWDREFP(sptr, 1); /* FS1551, see resolve_fwd_refs() below */
      if (SCG(sptr) == SC_DUMMY) {
        /* dummy procedure not declared external: */
        error(125, 1, gbl.lineno, SYMNAME(sptr), CNULL);
      } else /* if (SCG(sptr) == SC_NONE) */
             /*
              * <var ref> ::= <ident> sets the storage class to SC_LOCAL;
              * make it extern.
              */
        SCP(sptr, SC_EXTERN);
      SST_ASTP(stktop, mk_id(sptr));
      return func_call(stktop, list);

    case ST_VAR:
      dtype = DTYPEG(sptr);

      if (IS_CHAR_TYPE(DTYG(dtype))) {
        SST_ASTP(stktop, mk_id(sptr));
        chksubstr(stktop, list);
        SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
        return 1;
      }
      if (RESULTG(sptr) && (entry = test_really_an_entry(sptr))) {
        sptr = entry;
        SST_SYMP(stktop, sptr);
        goto really_an_entry;
      }
      if (is_procedure_ptr(sptr)) {
        return ptrfunc_call(stktop, list);
      }
      /* subscripts specified for non-array variable */
      error(76, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      goto add_base;
    case ST_MODPROC:
    case ST_PROC:
      if (FVALG(sptr) == 0 && DTYPEG(sptr) == 0) {
        error(84, 3, gbl.lineno, SYMNAME(sptr),
              "- attempt to use a SUBROUTINE as a FUNCTION");
        dtype = DT_INT;
        SST_IDP(stktop, S_EXPR);
        break;
      }
      if (GSAMEG(sptr)) {
        /* generic has same name as specific, treat as generic call */
        return generic_func(GSAMEG(sptr), stktop, list);
      }
      SST_ASTP(stktop, mk_id(sptr));
      return func_call(stktop, list);

    case ST_USERGENERIC:
      return generic_func(sptr, stktop, list);

    case ST_ARRAY:
      return (ref_array(stktop, list));

    case ST_TYPEDEF:
      interr("mkvarref: structure constructor seen", 0, 3);
      SST_IDP(stktop, S_EXPR);
      return 0;

    case ST_STRUCT:
      if (!sem.dinit_error)
        dinit((VAR *)NULL, SST_CLBEGG(stktop));
      sem.dinit_error = FALSE;
      return (0);
    /* ??????
                error(76, 3, gbl.lineno, SYMNAME(sptr), CNULL);
                goto add_base;
    */

    case ST_ENTRY:
    /* Possible recursive function call */
    really_an_entry:
      dtype = DTYPEG(sptr);
      if ((sptr == gbl.currsub && gbl.rutype == RU_FUNC) ||
          (sptr == gbl.outersub && STYPEG(sptr) == ST_ENTRY)) {
        if (GSAMEG(sptr))
          return generic_func(GSAMEG(sptr), stktop, list);
        if (DTYG(dtype) == TY_CHAR || DTYG(dtype) == TY_NCHAR) {
          if (list && list != ITEM_END && SST_IDG(list->t.stkp) == S_TRIPLE) {
            /* Character substring of character function okay */
            SST_ASTP(stktop, mk_id(sptr));
            SST_SYMP(stktop, ref_entry(sptr));
            chksubstr(stktop, list);
            SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
            return 1;
          }
        }
        if (list && SST_ALIASG(stktop) && DTY(dtype) == TY_ARRAY)
          return (ref_array(stktop, list));
        if (flg.recursive || RECURG(sptr)) {
          if (flg.standard && RECURG(sptr) && !RESULTG(sptr)) {
            error(155, 2, gbl.lineno, "An explicit RESULT variable should be "
                                      "present for RECURSIVE function",
                  SYMNAME(sptr));
          }
          SST_ASTP(stktop, mk_id(sptr));
          return func_call(stktop, list);
        }
        if (list && DTY(dtype) == TY_ARRAY)
          return (ref_array(stktop, list));
        error(88, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      } else { /* illegal use */
        switch (gbl.rutype) {
        case RU_SUBR:
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- SUBROUTINE name used as function");
          break;
        case RU_PROG:
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "- PROGRAM name used as function");
          break;
        default:
          error(84, 3, gbl.lineno, SYMNAME(sptr), "- used as a function");
          break;
        }
        /* give it a datatype, prevent further errors? */
        dtype = DT_INT;
      }
      sptr = ref_entry(sptr);
    add_base:
      sptr = ref_object(sptr);
      SST_IDP(stktop, S_LVALUE);
      SST_LSYMP(stktop, sptr);
      SST_ASTP(stktop, mk_id(sptr));
      SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
      break;

    case ST_STFUNC:
      dtype = DTYPEG(sptr);
      ref_stfunc(stktop, list);
      break;

    case ST_INTRIN:
    case ST_GENERIC:
      dtype = DTYPEG(sptr);
      /*
       * watch for case where an intrinsic was declared as a character
       * variable (array is already handled) and its first reference is
       * a substring reference.
       */
      if (!EXPSTG(sptr) && IS_CHAR_TYPE(DTY(dtype)) && list &&
          list != ITEM_END && SST_IDG(list->t.stkp) == S_TRIPLE) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        SST_SYMP(stktop, sptr);
        SST_ASTP(stktop, mk_id(sptr));
        chksubstr(stktop, list);
        SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
        return 1;
      }
      ref_intrin(stktop, list);
      return 1;

    case ST_PD:
      dtype = DTYPEG(sptr);
      if (!EXPSTG(sptr) && list && list != ITEM_END &&
          SST_IDG(list->t.stkp) == S_TRIPLE && IS_CHAR_TYPE(DTY(dtype))) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        SST_SYMP(stktop, sptr);
        SST_ASTP(stktop, mk_id(sptr));
        chksubstr(stktop, list);
        SST_SHAPEP(stktop, A_SHAPEG(SST_ASTG(stktop)));
        return 1;
      }
      ref_pd(stktop, list);
      return 1;

    default:
      dtype = DTYPEG(sptr);
      /* illegal use */
      SST_IDP(stktop, S_EXPR);
      error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      break;
    }
    SST_DTYPEP(stktop, dtype);
    return 1;

  case S_LVALUE:
    /* this must be array or substring reference */
    ast = SST_ASTG(stktop);
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr = memsym_of_ast(ast);
      dtype = DTYPEG(sptr);
      if (CLASSG(sptr)) {
        sptr = BINDG(sptr);
        if (VTOFFG(sptr)) {
          int ss;
          ss = sym_skip_construct(SST_SYMG(stktop));
          SST_SYMP(stktop, ss);
          if (A_TYPEG(ast) == A_MEM && A_TYPEG(A_PARENTG(ast)) == A_SUBSCR) {
            int ast2, asd, ndim, i;
            ast2 = A_PARENTG(ast);
            asd = A_ASDG(ast2);
            ndim = ASD_NDIM(asd);
            for (i = 0; i < ndim; i++) {
              if (A_TYPEG(ASD_SUBS(asd, i)) == A_TRIPLE) {
                /* Subscript has a triple, so remove it from the
                 * member portion of the expression to prevent
                 * an invalid ast type during lowering.
                 */
                A_PARENTP(ast, A_LOPG(ast2));
                break;
              }
            }
          }
          return func_call(stktop, list);
        }
      }
    }
    sptr = SST_LSYMG(stktop);
    dtype = SST_DTYPEG(stktop);

    if (IS_CHAR_TYPE(DTY(dtype))) {
      /* substring */
      if (A_TYPEG(ast) == A_SUBSTR)
        error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      else
        chksubstr(stktop, list);
    } else if (DTY(dtype) == TY_ARRAY) {
      int ddtype;
      ddtype = DTY(dtype + 1);
      if (ast && A_TYPEG(ast) == A_SUBSCR) {
        if (IS_CHAR_TYPE(DTY(ddtype))) {
          chksubstr(stktop, list);
        } else {
          /* double subscripting with vector subscripts */
          error(75, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        }
      } else if (ast && A_TYPEG(ast) == A_MEM) {
        int dtmem;
        dtmem = DTYPEG(A_SPTRG(A_MEMG(ast)));

        if (IS_CHAR_TYPE(DTY(dtmem)))
          chksubstr(stktop, list);
        else
          ref_array(stktop, list);
      } else {
        ref_array(stktop, list);
      }
    } else if (STYPEG(sptr) == ST_MEMBER && is_procedure_ptr(sptr)) {
      return ptrfunc_call(stktop, list);
    } else
      error(75, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    return 1;

  case S_CONST:
    dtype = SST_DTYPEG(stktop);
    if (list && list != ITEM_END && (DTY(dtype) == TY_NCHAR)) {
      SST *sp;
      sp = list->t.stkp;
      if (SST_IDG(sp) != S_TRIPLE || SST_IDG(SST_E3G(sp)) != S_NULL ||
          list->next != ITEM_END) {
        INT val[2];
        error(75, 3, gbl.lineno, "'constant'", CNULL);
        SST_DTYPEP(stktop, DT_NCHAR);
        val[0] = getstring(" ", 1);
        val[1] = 0;
        SST_IDP(stktop, S_CONST);
        SST_CVALP(stktop, getcon(val, DT_NCHAR));
        SST_ASTP(stktop, mk_cnst(SST_CVALG(stktop)));
        SST_SHAPEP(stktop, 0);
        break;
      }
      ch_substring(stktop, SST_E1G(sp), SST_E2G(sp));
      break;
    }
    if (list && list != ITEM_END && (DTY(dtype) == TY_CHAR)) {
      SST *sp;
      sp = list->t.stkp;
      if (SST_IDG(sp) != S_TRIPLE || SST_IDG(SST_E3G(sp)) != S_NULL ||
          list->next != ITEM_END) {
        error(75, 3, gbl.lineno, "'constant'", CNULL);
        SST_DTYPEP(stktop, DT_CHAR);
        SST_CVALP(stktop, getstring(" ", 1));
        SST_ASTP(stktop, mk_cnst(SST_CVALG(stktop)));
        SST_SHAPEP(stktop, 0);
        break;
      }
      ch_substring(stktop, SST_E1G(sp), SST_E2G(sp));
      break;
    }
    error(75, 3, gbl.lineno, "'constant'", CNULL);
    break;
  default:
    /* So far, we get here if SST_ID is S_EXPR.  This means that an
     * expression has an argument list as in rs(1)(2).  Give syntax error.
     * If a compiler created symbol (ie. a char function) look up real name.
     */
    sptr = getbase((int)SST_ASTG(stktop));
    if (CCSYMG(sptr))
      sptr = SST_ERRSYMG(stktop);
    if (STYPEG(sptr) == ST_ARRAY)
      return (ref_array(stktop, list));
    error(75, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    break;
  }
  return (1);
}

/**
    \brief Resolve forward references: try to find the declaration symbol
           and replace the reference symbol with it.

    F95 allows forward references to pure functions from within
    specification expressions.  A symbol will be created at the
    reference which must be fixed later after the function declaration
    has been seen.  Possible forward references are marked FWDREF in
    mkvarref() above.
 */
void
resolve_fwd_refs()
{
  int ref, mod, decl, hashlk;

  for (ref = stb.firstusym; ref < stb.stg_avail; ref++) {
    if (STYPEG(ref) == ST_PROC && FWDREFG(ref)) {

      /* Find the module that contains the reference. */
      for (mod = SCOPEG(ref); mod; mod = SCOPEG(mod))
        if (STYPEG(mod) == ST_MODULE)
          break;
      if (mod == 0)
        continue; /* Not in a module. */

      /* Look for the matching declaration. */
      for (decl = first_hash(ref); decl; decl = HASHLKG(decl)) {
        if (NMPTRG(decl) != NMPTRG(ref))
          continue;
        if (STYPEG(decl) == ST_PROC && ENCLFUNCG(decl) == mod) {
          hashlk = HASHLKG(ref);
          *(stb.stg_base + ref) = *(stb.stg_base + decl);
          HASHLKP(ref, hashlk);
          break;
        }
      }
    }
  }
}

/* \brief Return the predicate:
 *        \sptr is in the scope of a SAVE statement with no SAVE list.
 * \param sptr symbol (index) to check. */
bool
in_save_scope(SPTR sptr)
{
  return CONSTRUCTSYMG(sptr) ? SAVEG(ENCLFUNCG(sptr)) : sem.savall;
}

#ifdef FLANG_SEMUTIL_UNUSED
/* returns 1 if array dtype has one too many subscripts and the first
   subscript in the list is a S_TRIPLE.  Otherwise, returns 0;
*/
static int
is_substring(ITEM *list, int dtype)
{
  int numdim;
  ITEM *tmplist;
  int i;

  if (!list || list == ITEM_END)
    return 0;

  if (DTY(dtype) != TY_ARRAY)
    return 0;

  if (SST_IDG(list->t.stkp) != S_TRIPLE)
    return 0;

  numdim = AD_NUMDIM(AD_DPTR(dtype));

  tmplist = list;
  i = 0;
  while (tmplist != ITEM_END) {
    i++;
    tmplist = tmplist->next;
  }
  if (i == numdim + 1)
    return 1;

  return 0;
}
#endif

/** \brief Check if a stack entry represents a constant or an expression
           evaluated to a constant.
 */
LOGICAL
is_sst_const(SST *stk)
{
  switch (SST_IDG(stk)) {
  case S_CONST:
    return TRUE;
  case S_EXPR:
    if (A_ALIASG(SST_ASTG(stk)))
      return TRUE;
    break;
  default:
    break;
  }
  return FALSE;
}

/** \brief Get the SST_CVAL-like value for a semantic stack entry already
   determined
   to be a constant (i.e., is_sst_const() is true).

   SST_CVAL-like means just the the sst's CVAL field.  If the stack has been
   evaluated (is an S_EXPR), need to get CVAL from the ast.
 */
INT
get_sst_cval(SST *stkp)
{
  int ast;
  int sptr;

  if (SST_IDG(stkp) == S_CONST)
    return SST_CVALG(stkp);
  ast = SST_ASTG(stkp);
#if DEBUG
  assert(SST_IDG(stkp) == S_EXPR && A_ALIASG(ast),
         "get_sst_cval, expected S_EXPR with ALIAS", ast, 4);
#endif
  ast = A_ALIASG(ast);
  sptr = A_SPTRG(ast);
  switch (DTY(A_DTYPEG(ast))) {
  case TY_WORD:
  case TY_INT:
  case TY_LOG:
  case TY_REAL:
  case TY_SINT:
  case TY_BINT:
  case TY_SLOG:
  case TY_BLOG:
    /*  coordinate with ast.c:mk_cval1() */
    return CONVAL2G(sptr);
  default:
    break;
  }
  return sptr;
}

/** \brief Check if a stack entry is a legal variable reference.

    This routine is used when it's known that a variable reference is required
    and a check is necessary before calling routines like like mkvarref and
    mklvalue.
 */
LOGICAL
is_varref(SST *stk)
{
  switch (SST_IDG(stk)) {
  case S_IDENT:
  case S_LVALUE:
    return TRUE;
  default:
    break;
  }
  return FALSE;
}

/** \brief Access the address of the object (sym).
 */
int
ref_object(int sptr)
{
  /* Check the current scope for a default clause */
  if (sem.parallel || sem.task || sem.target || sem.teams
      || sem.orph
      )
    sptr = sem_check_scope(sptr, sptr);
  if (SCG(sptr) == SC_BASED)
    ref_based_object(sptr);

  return sptr;
}

LOGICAL
ast_isparam(int ast)
{
  INT val;
  int ndim;
  int i, asd;
  int argt;

  if (ast == 0)
    return FALSE;
  switch (A_TYPEG(ast) /* opc */) {
  case A_ID:
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      return TRUE;
    }
    if (PARAMG(A_SPTRG(ast)))
      return TRUE;
    return FALSE;

  case A_CNST:
    return TRUE;

  case A_UNOP:
    val = ast_isparam((int)A_LOPG(ast));
    return val;

  case A_BINOP:
    if (ast_isparam((int)A_LOPG(ast)) == FALSE)
      return FALSE;
    return ast_isparam((int)A_ROPG(ast));

  case A_PAREN:
  case A_CONV:
    return ast_isparam((int)A_LOPG(ast));

  case A_MEM:
    if (A_MEM == A_TYPEG(A_PARENTG(ast))) /* don't evaluate at this point */
      return FALSE;
    if (ALLOCATTRG(A_SPTRG(A_MEMG(ast))) || POINTERG(A_SPTRG(A_MEMG(ast))))
      return FALSE;
    return ast_isparam(A_PARENTG(ast));

  case A_SUBSCR:
    if (ast_isparam(A_LOPG(ast)) == FALSE)
      return FALSE;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      int ss;
      ss = ASD_SUBS(asd, i);
      if (ast_isparam(ss) == FALSE)
        return FALSE;
    }
    return TRUE;
  case A_TRIPLE:
    if (ast_isparam(A_LBDG(ast)) == FALSE)
      return FALSE;
    if (ast_isparam(A_UPBDG(ast)) == FALSE)
      return FALSE;
    if (A_STRIDEG(ast))
      return (ast_isparam(A_STRIDEG(ast)));
    return TRUE;

  /* don't do A_INTR for now except for
     maxval, maxloc, minval, minloc */
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
      case I_MAXVAL:
      case I_MAXLOC:
      case I_MINVAL:
      case I_MINLOC:
        argt = A_ARGSG(ast);
        for (i = 0; i < A_ARGCNTG(ast); ++i) {
          int argast = ARGT_ARG(argt, i);
          if (argast && !ast_isparam(argast))
            return FALSE; 
        }
        return TRUE;

      default:
        return FALSE;
    }
  default:
    return FALSE;
    break;
  }
  return FALSE;
}

/** \brief Checks whether a symbol is used in a select type or associate
 *         construct as a selector.
 *
 *  \param sptr is the symbol we are checking.
 *
 *  \return true if symbol is a selector in an associate/select type
 *          construct; else false.
 */
static bool
is_selector(SPTR sptr)
{

  int i;
  ITEM *itemp;
  int doif = sem.doif_depth;

  for(i=doif; i > 0; --i) {
    if (DI_ID(i) == DI_ASSOC) { 
      for (itemp = DI_ASSOCIATIONS(doif); itemp != NULL; 
           itemp = itemp->next) {
        if (itemp->t.sptr == sptr) {
          return true;
        }
      }
    } else if (DI_ID(i) == DI_SELECT_TYPE && 
               strcmp(SYMNAME(sptr), SYMNAME(DI_SELECTOR(i))) == 0) {
      return true;
    }
  } 
  return false;
}

static int
ref_array(SST *stktop, ITEM *list)
{
  int sptr, dtype;
  int count;
  ITEM *ip1;
  SST *sp;
  int numdim, isvec;
  int nummissing;
  ADSC *ad;
  int subs[MAXDIMS], ast;
  int triple[3]; /* asts for triple notation */
  int tmp;
  ast = SST_ASTG(stktop);
  if (SST_IDG(stktop) == S_LVALUE) {
    /* pointer to an ILM */
    dtype = SST_DTYPEG(stktop);
    sptr = SST_LSYMG(stktop);
  } else {
    /* symbol table entry */
    sptr = SST_SYMG(stktop);
    dtype = DTYPEG(sptr);
    sptr = ref_object(sptr);
    if (SST_IDG(stktop) != S_DERIVED)
      SST_LSYMP(stktop, sptr);
    if (STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC)
      sptr = ref_entry(sptr);
    ast = mk_id(sptr);
  }
  ad = AD_DPTR(dtype);
  numdim = AD_NUMDIM(ad);

  /*
   * we must make two passes through the subscript list to
   * determine if it is vector or element
   */
  isvec = FALSE;
  count = 0;
  for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
    count++;
    /* will be marked as illegal */
    if (SST_IDG(ip1->t.stkp) == S_KEYWORD)
      continue;
    if (SST_IDG(ip1->t.stkp) == S_LABEL)
      continue;
    if (SST_IDG(ip1->t.stkp) == S_TRIPLE) {
      isvec = TRUE;
      continue;
    }
    if (DTY(SST_DTYPEG(ip1->t.stkp)) == TY_ARRAY) {
      isvec = TRUE;
      continue;
    }
  }

  /* for NULL triples in derived type references, we have to be
     sure to grab array bounds from the correct place.
     We assert that any missing subscripts apply to the inner
     component array (whose subscripts come first.) Subscripts
     in subs[] array  will get shifted over later */
  nummissing = 0;
  if (SST_IDG(stktop) == S_DERIVED) {
    if (count < numdim)
      nummissing = numdim - count;
  }

  if (!isvec) {
    count = 0;
    for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
      count++;
      if (count == numdim && ip1->next != ITEM_END) {
        error(78, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        ip1->next = ITEM_END; /* Truncate # of subscripts */
      }
      /* process each subscript: */
      sp = ip1->t.stkp;
      if (SST_IDG(sp) == S_KEYWORD) {
        /* <ident> = <expr> illegal */
        errsev(79);
        subs[count - 1] = astb.bnd.one;
      } else if (SST_IDG(sp) == S_LABEL) {
        error(155, 3, gbl.lineno, "Illegal use of alternate return specifier",
              CNULL);
        subs[count - 1] = astb.bnd.one;
      } else {
        /* single subscript */
        chksubscr(sp, sptr);
        subs[count - 1] = SST_ASTG(sp);
      }
    }
    /* generate scalar load */
    dtype = DTY(dtype + 1);
  } else {
    /* A vector slice reference */
    if (!TY_ISVEC(DTYG(dtype))) {
      error83(DTYG(dtype));
      sem.dinit_error = TRUE;
      return (0);
    }
    count = 0;
    for (ip1 = list; ip1 != ITEM_END; ip1 = ip1->next) {
      count++;
      if (count == numdim && ip1->next != ITEM_END) {
        error(78, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        sem.dinit_error = TRUE;
        ip1->next = ITEM_END;
      }
      /* process each subscript: */
      triple[0] = triple[1] = triple[2] = 0;
      sp = ip1->t.stkp;
      if (SST_IDG(sp) == S_KEYWORD) {
        /* <ident> = <expression> is illegal */
        errsev(79);
        subs[count - 1] = astb.bnd.one;
      } else if (SST_IDG(sp) == S_LABEL) {
        error(155, 3, gbl.lineno, "Illegal use of alternate return specifier",
              CNULL);
        subs[count - 1] = astb.bnd.one;
      } else if (SST_IDG(sp) == S_TRIPLE) {
        sp = SST_E1G(sp);
        /* triplet subscript */
        if (SST_IDG(sp) == S_NULL) {
          triple[0] = tmp =
              check_member(ast, lbound_of(dtype, (count - 1) + nummissing));
        again:
          switch (A_TYPEG(tmp)) {
          case A_ID:
          case A_CNST:
          case A_BINOP: /*ptr reshape*/
            tmp = A_SPTRG(tmp);
            break;
          case A_SUBSCR:
            tmp = A_LOPG(tmp);
            goto again;
          default:
            if (A_ALIASG(tmp))
              tmp = A_SPTRG(A_ALIASG(tmp));
            break;
          }
        } else {
          chksubscr(sp, sptr);
          triple[0] = SST_ASTG(sp);
        }
        sp = SST_E2G(ip1->t.stkp);
        if (SST_IDG(sp) == S_NULL) {
          if (!SST_DIMFLAGG(stktop) &&
              AD_UPBD(ad, (count - 1) + nummissing) == 0) {
            /* '*' specified */
            error(84, 3, gbl.lineno, SYMNAME(sptr),
                  "- extent of assumed size array is unknown");
          } else {
            triple[1] = tmp =
                check_member(ast, AD_UPAST(ad, (count - 1) + nummissing));

            switch (A_TYPEG(tmp)) {
            case A_ID:
            case A_CNST:
            case A_BINOP: /*ptr reshape*/
              tmp = A_SPTRG(tmp);
              break;
            default:
              if (A_ALIASG(tmp))
                tmp = A_SPTRG(A_ALIASG(tmp));
              break;
            }
          }
        } else {
          chksubscr(sp, sptr);
          triple[1] = SST_ASTG(sp);
        }

        sp = SST_E3G(ip1->t.stkp);
        if (SST_IDG(sp) != S_NULL) {
          chksubscr(sp, sptr);
          triple[2] = SST_ASTG(sp);
          if (triple[2] == astb.bnd.zero)
            error(155, 3, gbl.lineno, "Illegal zero stride",
                  "in array subscript triplet");
        }
        subs[count - 1] = mk_triple(triple[0], triple[1], triple[2]);
        A_MASKP(subs[count - 1], SST_DIMFLAGG(stktop));
      } else {
        /* single subscript */
        chksubscr(sp, sptr);
        subs[count - 1] = SST_ASTG(sp);
      }
    }

    if (!DT_ISVEC(DTY(dtype + 1))) {
      interr("mkvarref: non-vec type", dtype, 3);
    }
  }

  if (count != numdim) {
    if (SST_IDG(stktop) == S_DERIVED && count < numdim) {
      /* a member reference of a subscripted derived type -
       * insert the remaining subscripts as triples derived from the
       * bounds of the beginning dimensions.
       */
      int i, j;
      /* shift subscripts over */
      j = numdim - 1;
      for (i = count - 1; i >= 0; i--)
        subs[j--] = subs[i];
      i = 0;
      while (count < numdim) {
        subs[i] = mk_triple(AD_LWAST(ad, i), AD_UPAST(ad, i), 0);
        count++;
        i++;
      }
      dtype = DTYPEG(sptr);
    } else if (!ALIGNG(sptr) && !DISTG(sptr)) {
      /* 'overindexed' subscript reference
       * T3D/T3E or C90 Cray targets, scalar reference of unmapped
       * array.
       */
      while (count < numdim) {
        if (AD_LWAST(ad, count) == 0)
          subs[count] = astb.bnd.one;
        else
          subs[count] = AD_LWAST(ad, count);
        count++;
      }
      if (flg.standard)
        ERR170("The number of subscripts is less than the rank of",
               SYMNAME(sptr));
      else
        error(155, 2, gbl.lineno,
              "The number of subscripts is less than the rank of",
              SYMNAME(sptr));
    } else {
      error(78, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      while (count < numdim)
        subs[count++] = astb.bnd.one;
    }
  }

  SST_IDP(stktop, S_LVALUE);
  /* can't overwrite list item in w4 until list is processed ????*/
  /*SST_SHAPEP(stktop, A_SHAPEG(ast));*/
  SST_LSYMP(stktop, sptr);
  ast = mk_subscr(ast, subs, numdim, dtype);
  dtype = A_DTYPEG(ast); /* derived types may change dtype */
  SST_DTYPEP(stktop, dtype);
  SST_ASTP(stktop, ast);
  SST_SHAPEP(stktop, A_SHAPEG(ast));
  if (sem.dinit_data) {
    constant_lvalue(stktop);
  }
  /* evaluate to constant here if it is a dimension and all is param */
  if (!isvec && numdim == 1 &&
      (sem.dinit_data || sem.in_dim || INSIDE_STRUCT)) {
    if (DT_ISINT(A_DTYPEG(ast)) && ast_isparam(ast)) {
      INT conval;
      ACL *acl = construct_acl_from_ast(ast, A_DTYPEG(ast), 0);

      acl = eval_init_expr(acl);
      conval = cngcon(acl->conval, acl->dtype, A_DTYPEG(ast));
      ast = mk_cval1(conval, (int)A_DTYPEG(ast));
      SST_IDP(stktop, S_CONST);
      SST_LSYMP(stktop, 0);
      SST_ASTP(stktop, ast);
      SST_ACLP(stktop, 0);
      if (DT_ISWORD(A_DTYPEG(ast)))
        SST_SYMP(stktop, CONVAL2G(A_SPTRG(ast)));
      else
        SST_SYMP(stktop, A_SPTRG(ast));
    }
  }
  if (!isvec && CLASSG(sptr) && !MONOMORPHICG(sptr) && 
      !is_selector(sptr) && !is_unl_poly(sptr) && sem.array_const_level == 0) {
    /* Provide polymorphic address for the polymorphic subscripted reference.
     *
     * Note the following expressions are handled separately:
     *
     * 1. selectors that are a part of a select type or associate construct.
     * 2. unlimited polymorphic objects.
     * 3. expressions inside an array constructor.
     *
     */
    int std = add_stmt(mk_stmt(A_CONTINUE, 0));
    int astnew = gen_poly_element_arg(ast, sptr, std);
    A_ORIG_EXPRP(astnew, ast);
    SST_ASTP(stktop, astnew);
  } 
  return 1;
}

/*---------------------------------------------------------------------*/

/** \brief Check that substring specifier is correct, write SUBS (substring)
           ILM and return pointer to it.
 */
int
chksubstr(SST *stktop, ITEM *item)
{
  SST *sp;
  int sptr;
  int cvlen;
  int ast, lb_ast, ub_ast;
  int odtype, dtype;
  INT t;
  int ityp; /* integer type for substring positions */

  ityp = stb.user.dt_int;
  if (astb.bnd.dtype == DT_INT8)
    ityp = DT_INT8;
  SST_CVLENP(stktop, 0);
  lb_ast = ub_ast = 0;
  odtype = SST_DTYPEG(stktop);
  dtype = DDTG(odtype);
  if (SST_IDG(stktop) == S_LVALUE) {
    /* Probably substringing an array reference e.g. ca(1)(1:2) */
    sptr = SST_LSYMG(stktop);
  } else if (SST_IDG(stktop) == S_DERIVED) {
    sptr = SST_SYMG(stktop);
    dtype = DDTG(DTYPEG(sptr));
  } else {
    sptr = SST_SYMG(stktop);
    SST_LSYMP(stktop, sptr);
    SST_IDP(stktop, S_LVALUE);
    sptr = ref_object(sptr);
  }
  ast = SST_ASTG(stktop);

  if (item == ITEM_END) {
    /* Neither upper nor lower bound given, default both */
    goto no_upbound;
  }

  /* Validate that we process only a subscript triplet, of which, only the
   * form e1:e2 is valid for substring references.
   */
  if (SST_IDG(item->t.stkp) != S_TRIPLE) {
    error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    return 1;
  }

  /* Validate lower bound and generate ast's for it */
  sp = SST_E1G(item->t.stkp);
  if (SST_IDG(sp) == S_NULL) {
    /* No lower bound, default to 1 */
  } else {
    if (!DT_ISINT(SST_DTYPEG(sp)))
      chk_scalartyp(sp, ityp, TRUE);
    else {
      if (DTY(SST_DTYPEG(sp)) == TY_INT8)
        ityp = DT_INT8;
      if (SST_IDG(sp) == S_CONST) {
        t = SST_CVALG(sp);
        if (DTY(SST_DTYPEG(sp)) == TY_INT8)
          t = cngcon(t, DT_INT8, ityp);
        if (t < 1) {
          error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          SST_DTYPEP(sp, ityp);
          SST_CVALP(sp, 1);
        }
      }
      chktyp(sp, ityp, FALSE); /* just to mkexpr() & set dtype */
    }
    lb_ast = SST_ASTG(sp);
  }

  /* Validate upper bound and generate ast's for it.  If user didn't
   * specify an upper bound use the variable's character length.
   */
  sp = SST_E2G(item->t.stkp);

  cvlen = 0;
  if (SST_IDG(sp) == S_NULL) { /* upper bound not specified */
  no_upbound:
    if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR || dtype == DT_DEFERCHAR ||
        dtype == DT_DEFERNCHAR) {
      /* Don't really know if character length assumption works */
      if (STYPEG(sptr) == ST_ENTRY)
        sptr = ref_entry(sptr);
    } else if (ADJLENG(sptr))
      ub_ast = size_ast(sptr, dtype);
    else {
      cvlen = string_length(dtype);
      if (cvlen < 0)
        interr("chksubstr: bad cvlen", cvlen, 3);
    }
  } else { /* upper bound specified */
           /* no need to check value of upper bound since F90 allows the lower
            * bound to exceed the upper bound.
            */
    if (DTY(SST_DTYPEG(sp)) == TY_INT8)
      ityp = DT_INT8;
    chk_scalartyp(sp, ityp, TRUE);
    ub_ast = SST_ASTG(sp);
  }

  /* Make sure user didn't specify a 3rd expression i.e. e1:e2:e3, or
   * more than one argument.
   */
  if (item != ITEM_END &&
      (SST_IDG(SST_E3G(item->t.stkp)) != S_NULL || item->next != ITEM_END))
    error(82, 3, gbl.lineno, SYMNAME(sptr), CNULL);

  if (lb_ast == ub_ast && (lb_ast == 0 || !A_CALLFGG(lb_ast))) {
    cvlen = 1;
    dtype = get_type(2, (int)DTY(dtype), mk_cval(cvlen, DT_INT4));
  } else if (A_TYPEG(lb_ast) == A_CNST && A_TYPEG(ub_ast) == A_CNST) {
    cvlen = CONVAL2G(A_SPTRG(ub_ast)) - CONVAL2G(A_SPTRG(lb_ast)) + 1;
    if (cvlen < 0)
      cvlen = 0;
    dtype = get_type(2, (int)DTY(dtype), mk_cval(cvlen, DT_INT4));
  } else if (ub_ast) {
    cvlen = ub_ast;
    if (lb_ast) {
      lb_ast = mk_convert(lb_ast, ityp); /* lb may have narrow type */
      cvlen = mk_binop(OP_SUB, cvlen, lb_ast, ityp);
      cvlen = mk_binop(OP_ADD, cvlen, astb.i1, ityp);
    }
    if (ityp == DT_INT8)
      cvlen = mk_convert(cvlen, DT_INT4);
    if (!A_ALIASG(cvlen))
      cvlen = ast_intr(I_MAX, DT_INT4, 2, cvlen, mk_cval(0, DT_INT4));
    dtype = get_type(2, (int)DTY(dtype), cvlen);
    SST_CVLENP(stktop, cvlen);
  } else if (cvlen && A_TYPEG(lb_ast) == A_CNST) {
    cvlen = cvlen - CONVAL2G(A_SPTRG(lb_ast)) + 1;
    if (cvlen < 0)
      cvlen = 0;
    dtype = get_type(2, (int)DTY(dtype), mk_cval(cvlen, DT_INT4));
  } else {
    cvlen = 0;
    if (DTY(dtype) == TY_CHAR) {
      dtype = DT_ASSCHAR;
    } else if (DTY(dtype) == TY_NCHAR) {
      dtype = DT_ASSNCHAR;
    } else {
      interr("chksubstr: bad character type", dtype, 3);
    }
  }
  /* should this be an array type? */
  if (DTY(odtype) == TY_ARRAY) {
    /* make a new array type, same bounds as parent type */
    dtype = get_type(3, TY_ARRAY, dtype);
    DTY(dtype + 2) = DTY(odtype + 2);
  }
  ast = mk_substr(ast, lb_ast, ub_ast, dtype);
  SST_ASTP(stktop, ast);
  if (SST_IDG(stktop) != S_DERIVED) {
    SST_SHAPEP(stktop, A_SHAPEG(ast));
    SST_DTYPEP(stktop, dtype);
  }
  return 1;
}

/** \brief Substring of a character constant.
 */
void
ch_substring(SST *stktop, SST *lb_sp, SST *ub_sp)
{
  int cnst_sptr; /* symbol table pointer of character constant */
  int lb_ast;
  int ub_ast;
  int dtype;
  int cvlen;
  char *cp;
  int new_var;
  int ast;
  INT val[2];

  dtype = SST_DTYPEG(stktop);
  cnst_sptr = SST_CVALG(stktop);
  if (SST_IDG(lb_sp) != S_NULL) {
    if (!DT_ISINT(SST_DTYPEG(lb_sp)))
      (void)chk_scalartyp(lb_sp, DT_INT, TRUE);
  }
  if (SST_IDG(ub_sp) != S_NULL) {
    if (!DT_ISINT(SST_DTYPEG(ub_sp)))
      (void)chk_scalartyp(ub_sp, DT_INT, TRUE);
  }
  if (SST_IDG(stktop) == S_CONST &&
      (SST_IDG(lb_sp) == S_NULL || SST_IDG(lb_sp) == S_CONST) &&
      (SST_IDG(ub_sp) == S_NULL || SST_IDG(ub_sp) == S_CONST)) {
    cvlen = string_length(dtype);
    if (SST_IDG(lb_sp) == S_NULL)
      lb_ast = 1;
    else {
      lb_ast = CONVAL2G(A_SPTRG(SST_ASTG(lb_sp)));
      if (lb_ast < 1) {
        errsev(82);
        lb_ast = 1;
      }
    }
    if (SST_IDG(ub_sp) == S_NULL)
      ub_ast = cvlen;
    else {
      ub_ast = CONVAL2G(A_SPTRG(SST_ASTG(ub_sp)));
      if (ub_ast > cvlen) {
        errsev(82);
        ub_ast = cvlen;
      }
    }
    cvlen = ub_ast - lb_ast + 1;
    if (cvlen < 1) {
      const char *str = "";
      cnst_sptr = getstring(str, strlen(str));
      if (DTY(dtype) == TY_NCHAR) {
        dtype = get_type(2, TY_NCHAR, mk_cval(strlen(str), DT_INT4));
        val[0] = cnst_sptr;
        val[1] = 0;
        cnst_sptr = getcon(val, dtype);
      }
      SST_DTYPEP(stktop, DTYPEG(cnst_sptr));
      SST_CVALP(stktop, cnst_sptr);
      SST_ASTP(stktop, mk_cnst(cnst_sptr));
      return;
    }
    if (cvlen != string_length(dtype)) {
      if (DTY(dtype) == TY_NCHAR) {
        int char_cnst;
        int blen; /* length in bytes of new kanji constant */
        char *p;

        char_cnst = CONVAL1G(cnst_sptr);
        p = stb.n_base + CONVAL1G(char_cnst);
        /*
         * get char position of lower bnd and char length of resulting
         * string.
         */
        lb_ast = kanji_len((unsigned char *)p, lb_ast - 1);
        blen = kanji_len((unsigned char *)p + lb_ast, cvlen);
        cp = getitem(0, blen);
        BCOPY(cp, p + lb_ast, char, blen);
        char_cnst = getstring(cp, blen);
        dtype = get_type(2, TY_NCHAR, mk_cval(cvlen, DT_INT4));
        val[0] = char_cnst;
        val[1] = 0;
        SST_DTYPEP(stktop, dtype);
        SST_ASTP(stktop, mk_cnst(getcon(val, dtype)));
        return;
      }
      cp = getitem(0, cvlen);
      BCOPY(cp, stb.n_base + CONVAL1G(cnst_sptr) + lb_ast - 1, char, cvlen);
      dtype = get_type(2, TY_CHAR, mk_cval(cvlen, DT_INT4));
      SST_DTYPEP(stktop, dtype);
      SST_CVALP(stktop, getstring(cp, cvlen));
      SST_ASTP(stktop, mk_cnst(SST_CVALG(stktop)));
    }
    return;
  }
  if (SST_IDG(lb_sp) != S_NULL) {
    (void)chktyp(lb_sp, DT_INT, FALSE); /* just to mkexpr() & set dtype */
    lb_ast = SST_ASTG(lb_sp);
  } else
    lb_ast = 0;
  if (SST_IDG(ub_sp) != S_NULL) {
    (void)chktyp(ub_sp, DT_INT, FALSE); /* just to mkexpr() & set dtype */
    ub_ast = SST_ASTG(ub_sp);
  } else
    ub_ast = 0;
  new_var = getcctmp('t', cnst_sptr, ST_UNKNOWN, dtype);
  if (STYPEG(new_var) == ST_UNKNOWN) {
    STYPEP(new_var, ST_VAR);
    DINITP(new_var, 1);
    sym_is_refd(new_var);
    dinit_put(DINIT_LOC, new_var);
    dinit_put(DINIT_STR, (INT)cnst_sptr);
    dinit_put(DINIT_END, (INT)0);
  }
  ast = mk_id(new_var);
  ast = mk_substr(ast, lb_ast, ub_ast, dtype);
  SST_IDP(stktop, S_EXPR);
  SST_ASTP(stktop, ast);
}

/** \brief Repair a bad term in an expression.

    Done by using the constant (sptr) passed to this routine.  An xCON ILM is
    generated referencing this constant.
 */
int
fix_term(SST *stktop, int sptr)
{
  SST_IDP(stktop, S_EXPR);
  SST_DTYPEP(stktop, DTYPEG(sptr));
  switch (DTY(DTYPEG(sptr))) {
  case TY_INT:
    break;
  case TY_REAL:
    break;
  case TY_DBLE:
    break;
  case TY_INT8:
    break;
  default:
    interr("fix_term: Unexpected dtype:", DTYPEG(sptr), 0);
    break;
  }

  return 1;
}

/** \brief Called when array of derived type = scalar derived type, but the
           scalar derived type has an array component.
*/
int
assign_array_w_forall(int dest_ast, int src_ast, int dtype, int ndim)
{
  int i;
  ADSC *ad;
  int subs[MAXDIMS];
  int ast, ast2;
  int list;
  int forallast;
  int sptr;

  /*  generate code
        forall(i's) dest(:'s,i's) = src(:'s)
           where there are ndim :'s representing the component array
           and the i's represent the shape of the dest ary
           of derived type

      we already have
        dest(*) = src(*);
  */

  /* first ndim are o.k. */
  for (i = 0; i < ndim; i++) {
    subs[i] = ASD_SUBS(A_ASDG(dest_ast), i);
  }
  if (DTY(dtype) != TY_ARRAY)
    interr("assign_array_w_forall(), bad dtype", dtype, 3);
  ad = AD_DPTR(dtype);
  if (AD_NUMDIM(ad) <= ndim)
    interr("assign_array_w_forall(), bad dtype dim", dtype, 3);
  start_astli();
  /* i retains its value from prior loop */
  for (; i < AD_NUMDIM(ad); i++) {
    /* get temp var for forall index var */
    sptr = get_temp(astb.bnd.dtype);
    ast2 = mk_id(sptr);
    /* use subscript for forall index var */
    list = add_astli();
    ASTLI_SPTR(list) = sptr;
    ASTLI_TRIPLE(list) = ASD_SUBS(A_ASDG(dest_ast), i);
    /* and use forall index var for subscript */
    subs[i] = ast2;
  }
  forallast = mk_stmt(A_FORALL, 0);
  A_LISTP(forallast, ASTLI_HEAD);

  /* change dest subscript to subs which uses forall vars */
  dest_ast = mk_subscr(A_LOPG(dest_ast), subs, AD_NUMDIM(ad), dtype);

  /* add assign and make forall point to it ?? */
  ast = mk_assn_stmt(dest_ast, src_ast, dtype);
  A_IFSTMTP(forallast, ast);

  return forallast;
}

/** \brief Give error message for reference like a(:)%b(:)
 */
void
check_derived_type_array_section(int ast)
{
  int mem, parent, subscr;
  for (mem = ast; mem;) {
    switch (A_TYPEG(mem)) {
    case A_MEM:
      parent = A_PARENTG(mem);
      /* if this is an array member, and the parent has nontrivial shape,
       * give an error message */
      if (A_SHAPEG(parent)) {
        int sptr;
        sptr = A_SPTRG(A_MEMG(mem));
        if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
          error(455, 3, gbl.lineno, SYMNAME(memsym_of_ast(mem)), "");
        }
      }
      mem = parent;
      break;
    case A_SUBSCR:
      subscr = mem;
      parent = mem = A_LOPG(subscr);
      if (A_TYPEG(mem) == A_MEM) {
        parent = A_PARENTG(mem);
        if (A_SHAPEG(parent)) {
          /* if any subscripts are triplets or have shape, give error */
          int asd, i, ndim, ss;
          asd = A_ASDG(subscr);
          ndim = ASD_NDIM(asd);
          for (i = 0; i < ndim; ++i) {
            ss = ASD_SUBS(asd, i);
            if (A_SHAPEG(ss) || A_TYPEG(ss) == A_TRIPLE) {
              error(455, 3, gbl.lineno, SYMNAME(memsym_of_ast(mem)), "");
              break;
            }
          }
        }
      }
      mem = parent;
      FLANG_FALLTHROUGH;
    default:
      return;
    }
  }
} /* check_derived_type_array_section */

/** \brief Assign stktop to newtop.
 */
int
assign(SST *newtop, SST *stktop)
{
  int dtype;
  int shape;
  int ast;

  if (mklvalue(newtop, 1) == 0)
    /* Avoid assignment ILM's if lvalue is illegal */
    return 0;
  dtype = SST_DTYPEG(newtop);
  shape = SST_SHAPEG(newtop);

  if (shape != 0 && SST_DTYPEG(stktop) == DT_HOLL)
    errsev(100);

  /* If the left and right sides of the assign. stmt. have unequal data
   * types or if equal, they are records then change the type of the right
   * side to the type of the left side.
   */
  if (SST_IDG(stktop) == S_STFUNC)
    chktyp(stktop, dtype, FALSE);

  if (SST_IDG(stktop) == S_EXPR && SST_ASTG(stktop) && SST_ASTG(newtop) &&
      (A_TYPEG(SST_ASTG(stktop)) == A_FUNC) &&
      is_iso_cptr(A_DTYPEG(SST_ASTG(stktop))) &&
      is_iso_cptr(A_DTYPEG(SST_ASTG(newtop)))) {

  } else if (DTYG(dtype) == TY_STRUCT || DTYG(dtype) == TY_DERIVED) {
    SPTR sptr;
    if (SST_IDG(newtop) == S_LVALUE || SST_IDG(newtop) == S_EXPR) {
      sptr = SST_LSYMG(newtop);
    } else {
      sptr = SST_SYMG(newtop);
    }
    cngtyp2(stktop, dtype, (CLASSG(sptr) && ALLOCATTRG(sptr)));
  } else if (DTYG(dtype) != DTYG(SST_DTYPEG(stktop))) {
    if (DTY(dtype) == TY_ARRAY && DTY(SST_DTYPEG(stktop)) != TY_ARRAY)
      /*
       * array = scalar and the element type is not the same as the
       *    type of the scalar; first convert the scalar.
       */
      cngtyp(stktop, DTY(dtype + 1));
    else {
      cngtyp(stktop, dtype);
    }
  }

  mkexpr1(stktop);
  cngshape(stktop, newtop);

  if (DTY(dtype) == TY_ARRAY && !DT_ISVEC(DTY(dtype + 1)))
    error83(DTYG(dtype));

  check_derived_type_array_section(SST_ASTG(newtop));

  {
    int lhs;
    int rhs;
    int call;

    lhs = SST_ASTG(newtop);
    rhs = SST_ASTG(stktop);
    call = STD_AST(sem.arrfn.call_std);
    if (gbl.maxsev < 3 && sem.arrfn.try && DTY(dtype) == TY_ARRAY &&
        rhs == sem.arrfn.return_value && subst_lhs_arrfn(lhs, rhs, call)) {
      /*
       * The RHS of the assignment is a function call for which
       * the result temp can be replaced by the lhs.
       */
      int argt;
      int arr_tmp;

      arr_tmp = A_SPTRG(rhs);
      argt = A_ARGSG(call);
      ARGT_ARG(argt, 0) = lhs;
      if (ALLOCG(arr_tmp)) {
        /*
         * if the temp was allocated, delete its allocation
         * and remove the temp from the dealloc list.  Note
         * that if the temp is not found in the dealloc list,
         * then the allocate is left.
         */
        ITEM *p, *t;

        p = NULL;
        for (t = sem.p_dealloc; t != NULL; t = t->next) {
          if (t->ast == rhs) {
            ast_to_comment(STD_AST(sem.arrfn.alloc_std));
            if (p == NULL)
              sem.p_dealloc = t->next;
            else
              p->next = t->next;
            break;
          }
          p = t;
        }
        for (t = sem.p_dealloc_delete; t != NULL; t = t->next) {
          if (t->ast == rhs) {
            delete_stmt(t->t.ilm);
          }
        }
      }
      sem.arrfn.try
        = 0;
      return 0;
    }
    ast = mk_assn_stmt(lhs, rhs, dtype);

    if (DTY(dtype) == TY_ARRAY) {
      direct_loop_enter();
      direct_loop_end(gbl.lineno, gbl.lineno);
    }
  }

  return ast;
}

/*
 * Can the result temp by substituted with the LHS?
 * The LHS cannot:
 * -  have the POINTER attribute
 * -  have adjustable length if character
 * -  have the allocatable attribute if the 2003 allocatable semantics are
 *    enabled
 * -  have different length than the function result
 * -  appear as an argument to the function
 * The LHS must be 'whole';  for hpf, the LHS must also be an ident.
 * The RHS (function result) cannot have the POINTER attribute (POINTER
 *    functions can be seen in assign() because of the work for
 *    p => func() (i.e., assign_pointer())
 */
static LOGICAL
subst_lhs_arrfn(int lhs, int rhs, int call)
{
  int sym;
  int arr_tmp;
  int dtype, eldt;
  int func_sptr;

  if (XBIT(47, 0x800000))
    return FALSE;
  if (DI_IN_NEST(sem.doif_depth, DI_WHERE)) {
    /* WHERE processing must see the assignment! */
    return FALSE;
  }
  func_sptr = sem.arrfn.sptr;
  if (!PUREG(func_sptr)) {
/*
 * f1565 6-
 * substituting the result of an array-valued function to the array
 * on the lhs is an unsafe optimization since the function could
 * define the array.  This optimization was added for polyhedron-
 * channel (f12457), and fixing 15656 means that the optimization
 * will no longer occur in channel ...
 * In addition to the constraints above, need:
 * o  calling a contained function from the host
 * o  calling a function from a contained function and the lhs is
 *    not local
 * o  calling a function and the lhs is 'global'
 * We can do better if:
 * o  for internal procedures, we somehow record what variables
 *    (host-associated & globals) are possibly defined =>
 *    IDEAS: enhance how we process internal procedures so that
 *           we can collect information; use IPA
 * o  for external functions, what global symbols are possibly
 *    defined => use IPA
 */
    sym = sym_of_ast(lhs);
    if (gbl.internal > 1 && !INTERNALG(sym))
      return FALSE;
    if (INTERNALG(func_sptr) && gbl.internal <= 1 &&
        (GSCOPEG(sym) || XBIT(7, 0x200000))) {
      return FALSE;
    }
    if ((SCG(sym) == SC_CMBLK) || (SCG(sym) == SC_EXTERN))
      return FALSE;
  }
  sym = memsym_of_ast(lhs);
  if (POINTERG(sym) || ADJLENG(sym) || (ALLOCATTRG(sym) && XBIT(54, 1)))
    return FALSE;
  arr_tmp = A_SPTRG(rhs);
  if (POINTERG(arr_tmp))
    return FALSE;
  dtype = DTYPEG(sym);
  eldt = DTY(DTYPEG(arr_tmp) + 1);
  if (DTY(eldt) == TY_CHAR || DTY(eldt) == TY_NCHAR) {
    int d1;
    /* warning - use DDTG for the lhs, since the member itself doesn't
     * need to be an array.
     */
    if (ADJLENG(arr_tmp))
      return FALSE;
    d1 = DDTG(dtype);
    if (DTY(eldt + 1) != DTY(d1 + 1))
      return FALSE;
  }
  if (A_TYPEG(lhs) == A_ID)
    return not_in_arrfn(lhs, call);
  if (A_TYPEG(lhs) == A_MEM) {
    /*
     * If the LHS is a member, then the member must be an array in
     * order for it to be 'whole'.
     */
    if (DTY(dtype) == TY_ARRAY)
      return not_in_arrfn(A_PARENTG(lhs), call);
    return FALSE;
  }
  if (A_TYPEG(lhs) == A_SUBSCR && A_SHAPEG(lhs) && DTY(dtype) == TY_ARRAY) {
    /*
     * If subscripted, the LHS is 'whole' if its triples are just ':'.
     */
    ADSC *ad;
    int shd, nd, ii;
    int asd, sub;

    ad = AD_DPTR(dtype);
    shd = A_SHAPEG(lhs);
    nd = SHD_NDIM(shd);
    if (nd > AD_NUMDIM(ad))
      return FALSE;
    asd = A_ASDG(lhs);
    for (ii = 0; ii < nd; ++ii) {
      sub = ASD_SUBS(asd, ii);
      if (A_TYPEG(sub) != A_TRIPLE)
        return FALSE;
      if (A_STRIDEG(sub) && A_STRIDEG(sub) != astb.bnd.one)
        return FALSE;
      if (A_LBDG(sub) != AD_LWAST(ad, ii))
        return FALSE;
      if (A_UPBDG(sub) != AD_UPAST(ad, ii))
        return FALSE;
    }
    return not_in_arrfn(A_LOPG(lhs), call);
  }

  return FALSE;
}

/*
 * Can the result temp by substituted with the LHS?
 * The LHS must have POINTER attribute.
 * the LHS must not appear as an argument to the function
 * the LHS must be 'whole'
 * the LHS must not be adjustable length, if character
 * the LHS must match in datatype and rank to the function
 */
static LOGICAL
subst_lhs_pointer(int lhs, int rhs, int call)
{
  int sym, tmp, symdtype, symddtype, tmpdtype, tmpddtype;
  if (XBIT(47, 0x800000))
    return FALSE;
  sym = memsym_of_ast(lhs);
  if (!POINTERG(sym) || ADJLENG(sym))
    return FALSE;
  symdtype = DTYPEG(sym);
  tmp = A_SPTRG(rhs);
  tmpdtype = DTYPEG(tmp);
  if (DTY(tmpdtype) != DTY(symdtype))
    return FALSE;

  symddtype = DDTG(symdtype);
  tmpddtype = DDTG(tmpdtype);
  if (DTY(symddtype) != DTY(tmpddtype))
    return FALSE;

  if (DTY(tmpddtype) == TY_CHAR || DTY(tmpddtype) == TY_NCHAR) {
    /* warning - use DDTG for the lhs, since the member itself doesn't
     * need to be an array.  */
    if (ADJLENG(tmp)) /* return temp is adjustable length */
      return FALSE;
    if (DTY(symddtype + 1) != DTY(tmpddtype + 1)) /* not same char length */
      return FALSE;
  }
  if (A_TYPEG(lhs) == A_ID)
    return not_in_arrfn(lhs, call);
  if (A_TYPEG(lhs) == A_MEM)
    return not_in_arrfn(A_PARENTG(lhs), call);

  return FALSE;
} /* subst_lhs_pointer */

static LOGICAL
not_in_arrfn(int memref, int call)
{
  int i;
  int nargs;
  int argt;
  int arg;

  nargs = A_ARGCNTG(call);
  argt = A_ARGSG(call);
  for (i = 1; i < nargs; i++) {
    arg = ARGT_ARG(argt, i);
    if (contains_ast(arg, memref))
      return FALSE;
  }
  return TRUE;
}

static void
update_proc_ptr_dtype_from_interface(int func_sptr)
{
  if (is_procedure_ptr(func_sptr)) {
    int paramct, dpdsc, iface_sptr;
    proc_arginfo(func_sptr, &paramct, &dpdsc, &iface_sptr);
    if (iface_sptr > NOSYM) {
      if (STYPEG(iface_sptr) != 0 && STYPEG(iface_sptr) != ST_PROC) {
        int found_iface_sptr =
            findByNameStypeScope(SYMNAME(iface_sptr), ST_PROC, 0);
        if (found_iface_sptr > NOSYM && STYPEG(found_iface_sptr) == ST_PROC) {
          iface_sptr = found_iface_sptr;
          proc_arginfo(iface_sptr, &paramct, &dpdsc, NULL);
        }
      }
    }
    if (iface_sptr > NOSYM && STYPEG(iface_sptr) == ST_PROC) {
      int dtproc = DTY(DTYPEG(func_sptr) + 1);
      CHECK(DTY(dtproc) == TY_PROC);
      DTY(dtproc + 1) = DTYPEG(iface_sptr);
      DTY(dtproc + 2) = iface_sptr;
      DTY(dtproc + 3) = paramct;
      DTY(dtproc + 4) = dpdsc;
    }
  }
}

/*
 * pointer assignment - assign stktop to newtop.
 */
static LOGICAL
valid_assign_pointer_types(SST *newtop, SST *stktop)
{
  LOGICAL is_proc_ptr = FALSE;
  int dest = SST_ASTG(newtop);
  int source = SST_ASTG(stktop);
  DTYPE d1, d2, dtype;

  d1 = DDTG(SST_DTYPEG(newtop)); /* Check for procedure ptr */
  if (!is_procedure_ptr_dtype(d1) && rank_of_ast(dest) != rank_of_ast(source)) {
    if (A_TYPEG(dest) != A_SUBSCR) {
      error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
            "rank mismatch");
      return FALSE;
    }
    if (rank_of_ast(source) != 1 && !bnds_remap_list(dest)) {
      error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
            "rank of pointer target must be 1 or equal to rank of pointer "
            "object");
      return FALSE;
    }
  }
  if (rank_of_ast(source) != 1 && bnds_remap_list(dest) &&
      !simply_contiguous(source)) {
    error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
          "pointer target must be simply contiguous");
    return FALSE;
  }
  dtype = SST_DTYPEG(newtop);
  d1 = DDTG(dtype);
  d2 = DDTG(SST_DTYPEG(stktop));
  is_proc_ptr = is_procedure_ptr_dtype(d1);
  if (is_proc_ptr) {
    update_proc_ptr_dtype_from_interface(get_ast_sptr(SST_ASTG(newtop)));
    d1 = proc_ptr_result_dtype(d1);
    if (is_procedure_ptr_dtype(d2)) {
      d2 = proc_ptr_result_dtype(d2);
    } else {
      int rhs_sptr = get_ast_sptr(SST_ASTG(stktop));
      if (rhs_sptr > NOSYM) {
        int dpdsc = 0, iface_sptr;
        proc_arginfo(rhs_sptr, NULL, &dpdsc, &iface_sptr);
        if (iface_sptr <= NOSYM)
          iface_sptr = rhs_sptr;
        if (dpdsc > 0) {
          d2 = DTYPEG(iface_sptr);
        } else if (iface_sptr > NOSYM && STYPEG(iface_sptr) == ST_PROC &&
                   SCG(iface_sptr) == SC_EXTERN) {
          /* Assume this is a procedure declared with the external
           * statement and therefore, does not have an interface. Fortran spec
           * allows assignment of external procedures to procedure pointers.
           */
          d2 = DT_NONE;
        }
      }
    }
  }

  switch (DTY(d1)) {
  case TY_CHAR:
  case TY_NCHAR:
    if (DTY(d1) != DTY(d2)) {
      error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
            "type mismatch");
      return FALSE;
    }
    if (d1 == DT_ASSCHAR || d2 == DT_DEFERCHAR)
      break;
    if (d1 == DT_ASSNCHAR || d2 == DT_DEFERNCHAR)
      break;
    if (DTY(d1 + 1) && DTY(d2 + 1) && A_ALIASG(DTY(d1 + 1)) &&
        A_ALIASG(DTY(d2 + 1)) && DTY(d1 + 1) != DTY(d2 + 1)) {
      error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
            "type mismatch");
      return FALSE;
    }
    break;
  default:
    if (!eq_dtype2(d1, d2, TRUE)) { /* TRUE for polymorphic ptrs */
      if (UNLPOLYG(DTY(d1 + 3)))    /* true for CLASS(*) ptrs */
        return TRUE;
      if (is_proc_ptr && d2 == DT_NONE)
        return TRUE;
      error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
            "type mismatch");
      return FALSE;
    }
  }

  if (DTY(dtype) == TY_ARRAY && !DT_ISVEC(array_element_dtype(dtype))) {
    error83(DTYG(dtype));
    return FALSE;
  }

  return TRUE;
}

static int
assign_intrinsic_to_pointer(SST *newtop, SST *stktop)
{
  int dest, source;
  int pvar;

  dest = SST_ASTG(newtop);
  source = SST_ASTG(stktop);

  if (PDNUMG(A_SPTRG(A_LOPG(source))) != PD_null) {
    error(155, 3, gbl.lineno, "Illegal POINTER assignment", CNULL);
    if (INSIDE_STRUCT) {
      sem.dinit_error = TRUE;
    }
    return 0;
  }

  pvar = find_pointer_variable_assign(dest, SST_DIMFLAGG(newtop));
  if (pvar == 0) {
    error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
          "non-POINTER object");
    return 0;
  }
  if (!POINTERG(pvar)) {
    error(72, 3, gbl.lineno, SYMNAME(pvar), "- must be a POINTER variable");
    return 0;
  }

  set_assn(sym_of_ast(dest));

  if (DTY(A_DTYPEG(source)) == TY_WORD) {
    A_DTYPEP(source, A_DTYPEG(dest));
    A_SHAPEP(source, A_SHAPEG(dest));
  } else if (!valid_assign_pointer_types(newtop, stktop)) {
    if (INSIDE_STRUCT) {
      sem.dinit_error = TRUE;
    }
    return 0;
  }

  return add_ptr_assign(dest, source, 0);
}

int
assign_pointer(SST *newtop, SST *stktop)
{
  int dtype;
  int shape;
  int ast;
  int dest, source, call;
  int pvar;

  ast = 0;

  if (mklvalue(newtop, 1) == 0)
    /* Avoid assignment ILM's if lvalue is illegal */
    return 0;

  if (A_TYPEG(SST_ASTG(stktop)) == A_INTR) {
    set_assn(sym_of_ast(A_LOPG(SST_ASTG(stktop))));
    return assign_intrinsic_to_pointer(newtop, stktop);
  }

  if (SST_IDG(stktop) == S_IDENT) {
    int sptr = SST_SYMG(stktop), sp2;
    switch (STYPEG(sptr)) {
    case ST_GENERIC:
      if (!select_gsame(sptr))
        break;
      FLANG_FALLTHROUGH;
    case ST_PD:
    case ST_INTRIN:
      sp2 = intrinsic_as_arg(sptr);
      if (sp2 == 0)
        break;
      TARGETP(sp2, 1);
      SST_IDP(stktop, S_EXPR);
      SST_ASTP(stktop, mk_id(sp2));
      SST_DTYPEP(stktop, DTYPEG(sp2));
      SST_SHAPEP(stktop, 0);
      break;
    default:;
    }
  }

  dtype = SST_DTYPEG(newtop);
  shape = SST_SHAPEG(newtop);

  mkexpr2(stktop);

  /* both sides of the assignment must be of the same type, type parameters
   * and rank.
   */
  dest = SST_ASTG(newtop);
  source = SST_ASTG(stktop);

  pvar = find_pointer_variable_assign(dest, SST_DIMFLAGG(newtop));
  if (pvar == 0) {
    error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
          "non-POINTER object");
    return 0;
  }
  if (!POINTERG(pvar)) {
    error(72, 3, gbl.lineno, SYMNAME(pvar), "- must be a POINTER variable");
    return 0;
  }
  if (is_protected(pvar)) {
    err_protected(pvar, "be assigned");
    return 0;
  }
 
  if (is_procedure_ptr(pvar)) {
    int iface=0;
    proc_arginfo(pvar, NULL, NULL, &iface);
    if (ELEMENTALG(iface) && !IS_INTRINSIC(STYPEG(iface)) && !CCSYMG(iface)) {
      error(1010, ERR_Severe, gbl.lineno, SYMNAME(pvar), CNULL);
    }
  }
 
  if (chk_pointer_intent(pvar, dest))
    return 0;

  if (chk_pointer_target(pvar, source))
    return 0;

  if (!valid_assign_pointer_types(newtop, stktop))
    return 0;

  call = STD_AST(sem.arrfn.call_std);
  if (gbl.maxsev < 3 && sem.arrfn.try && source == sem.arrfn.return_value &&
      subst_lhs_pointer(dest, source, call)) {
    /*
     * The RHS of the assignment is a function call for which
     * the result temp can be replaced by the lhs.
     */
    int argt;
    int arr_tmp;

    arr_tmp = A_SPTRG(source);
    argt = A_ARGSG(call);
    ARGT_ARG(argt, 0) = dest;
    sem.arrfn.try
      = 0;
    return 0;
  }

  return add_ptr_assign(dest, source, 0);
}

/** \brief Generates a call to a poly_element_addr runtime routine that
 *         computes the address of a polymorphic array element.
 *
 *         This is required when our passed object argument of a type bound
 *         procedure call is an array element.
 *
 *  \param ast is the ast of the passed object argument (an A_SUBSCR ast).
 *  \param sptr is the symbol table pointer of the passed object argument.
 *  \param std is the current statement descriptor.
 *
 *  \return an ast that represents the pointer that holds the address of the
 *          polymorphic array element. 
 */
int
gen_poly_element_arg(int ast, SPTR sptr, int std) 
{

  SPTR func, tmp, ptr, sdsc, ptr_sdsc;
  int astnew, args;
  int asd, numdim, i, ss;
  int tmp_ast, ptr_ast, sdsc_ast, ptr_sdsc_ast;
  DTYPE dtype;
  FtnRtlEnum rtlRtn;

  dtype = DTYPEG(sptr);

  assert(DTY(dtype) == TY_ARRAY, "gen_poly_element_arg: Expected array dtype",
             dtype, 4);

  dtype = DTY(dtype+1);

  asd = A_ASDG(ast);
  numdim = ASD_NDIM(asd);
  args = mk_argt(3+numdim);

  for (i = 0; i < numdim; ++i) {
    ss = ASD_SUBS(asd, i);
    ARGT_ARG(args, 3+i) = ss;
  }

  ARGT_ARG(args, 0) = A_LOPG(ast);
  if (SCG(sptr) == SC_DUMMY && (needs_descriptor(sptr) || CLASSG(sptr))) {
    fix_class_args(gbl.currsub);
    sdsc = get_type_descr_arg(gbl.currsub, sptr);
  } else {
    sdsc = 0;
  }
  if (sdsc <= NOSYM) {
    do {
      if (STYPEG(sptr) == ST_MEMBER) {
        sdsc = get_member_descriptor(sptr);
      } else {
        sdsc = SDSCG(sptr);
      }
      if (sdsc > NOSYM) {
        break;
      }
      get_static_descriptor(sptr);
      assert(SDSCG(sptr) > NOSYM, "gen_poly_element_arg: get_static_descriptor"
             " failed", sptr, 4); /* sanity check */
    } while(true); 
  }

  sdsc_ast = mk_id(sdsc);
  sdsc_ast = check_member(ast, sdsc_ast);

  ptr = getccsym_sc('d', sem.dtemps++, ST_VAR, SC_LOCAL);
  DTYPEP(ptr, dtype);
  POINTERP(ptr, 1);
  CLASSP(ptr, CLASSG(sptr));
  ADDRTKNP(ptr, 1);  
  set_descriptor_rank(1);
  get_static_descriptor(ptr);
  set_descriptor_rank(0);
  ptr_sdsc = SDSCG(ptr);
  ptr_sdsc_ast = mk_id(ptr_sdsc);
  
  if (DTY(dtype) == TY_DERIVED) { 
    astnew = mk_set_type_call(ptr_sdsc_ast, sdsc_ast, 0);
  } else {
    int type_code = dtype_to_arg(DTY(dtype));
    type_code = mk_cval1(type_code, DT_INT);
    type_code = mk_unop(OP_VAL, type_code, DT_INT);
    astnew = mk_set_type_call(ptr_sdsc_ast, type_code, 1);
  }

  std = add_stmt_before(astnew, std);
  
  ARGT_ARG(args, 1) = sdsc_ast;

  switch(numdim) {
  case 1:
    rtlRtn = RTE_poly_element_addr1;
    break;
  case 2:
    rtlRtn = RTE_poly_element_addr2;
    break;
  case 3:
    rtlRtn = RTE_poly_element_addr3;
    break;
  default:
    rtlRtn = RTE_poly_element_addr;
  }
    
  func = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE));

  tmp = getccsym_sc('d', sem.dtemps++, ST_VAR, SC_LOCAL);
  DTYPEP(tmp, dtype);
  POINTERP(tmp, 1);
  tmp_ast = mk_id(tmp);
  A_DTYPEP(tmp_ast, dtype);
  A_PTRREFP(tmp_ast, 1);
  ARGT_ARG(args, 2) = tmp_ast;

  astnew = mk_func_node(A_CALL, func, 3+numdim, args);
      
  std = add_stmt_after(astnew, std);
 
  ptr_ast = mk_id(ptr);
  astnew = add_ptr_assign(ptr_ast, tmp_ast, std);
  add_stmt_after(astnew, std);
  return  ptr_ast;
}

int
add_ptr_assign(int dest, int src, int std)
{
  int func;
  int ast;
  int dtype, tag;
  int dtype2, tag2, dtype3;
  SPTR dest_sptr, src_sptr, sdsc;
  int newargt, astnew;

  /* Check if the dest is scalar, if so assign len to descriptor
   * For array, it was done in runtime.
   * Also, check if it is assigned to NULL, then don't assign len
   */

  dtype = A_DTYPEG(dest);
  dtype2 = A_DTYPEG(src);

  if (DTY(dtype) == TY_DERIVED) {
    tag = DTY(dtype + 3);
  } else if (DTY(dtype) == TY_ARRAY) {
    dtype3 = DTY(dtype + 1);
    if (DTY(dtype3) == TY_DERIVED) {
      tag = DTY(dtype3 + 3);
    } else {
      tag = 0;
    }
  } else {
    tag = 0;
  }

  if (DTY(dtype2) == TY_DERIVED) {
    tag2 = DTY(dtype2 + 3);
  } else if (DTY(dtype2) == TY_ARRAY) {
    dtype3 = DTY(dtype2 + 1);
    if (DTY(dtype3) == TY_DERIVED) {
      tag2 = DTY(dtype3 + 3);
    } else {
      tag2 = 0;
    }
  } else {
    tag2 = 0;
  }

  if (tag && tag2 && has_type_parameter(dtype2) && has_type_parameter(dtype) &&
      !BASETYPEG(tag) && BASETYPEG(tag2)) {
    /* The parameterized derived type (PDT) for the destination
     * pointer is currently set to the default/base type. Now that it's
     * being used, we need to instantiate it with the source type.
     */
    if (DTY(dtype2) == TY_ARRAY) {
      dtype2 = DTY(dtype2 + 1);
    }
    dtype3 = create_parameterized_dt(dtype2, 1);
    if (DTY(dtype) == TY_ARRAY) {
      dtype = dup_array_dtype(dtype);
      DTY(dtype + 1) = dtype3;
    } else {
      dtype = dtype3;
    }
    A_DTYPEP(dest, dtype);
    DTYPEP(memsym_of_ast(dest), dtype);
  }

  if ((dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR ||
       (UNLPOLYG(tag) && DTY(A_DTYPEG(src)) == TY_CHAR)) &&
      !is_dtype_unlimited_polymorphic(A_DTYPEG(src))) {
    int dest_len_ast = get_len_of_deferchar_ast(dest);
    int src_len_ast, cvlen;
    if (A_TYPEG(src) == A_INTR && A_OPTYPEG(src) == I_NULL)
      src_len_ast = mk_cval(0, astb.bnd.dtype);
    else
      src_len_ast = string_expr_length(src);
    cvlen = mk_assn_stmt(dest_len_ast, src_len_ast, astb.bnd.dtype);
    if (std)
      add_stmt_before(cvlen, std);
    else
      add_stmt(cvlen);
  }

  if (ast_is_sym(src)) {
    src_sptr = memsym_of_ast(src);
  } else {
    src_sptr = 0;
  }
  
  dest_sptr = memsym_of_ast(dest);

  if (DTY(dtype) == TY_PTR) {

    if (STYPEG(src_sptr) == ST_PROC) { 
      int iface=0, iface2=0, dpdsc=0, dpdsc2=0;
      proc_arginfo(src_sptr, NULL, &dpdsc, &iface); 
      proc_arginfo(dest_sptr, NULL, &dpdsc2, &iface2);
      if (iface > NOSYM && iface2 > NOSYM && dpdsc != 0 && dpdsc2 != 0 &&
          !cmp_interfaces_strict(iface2, iface, 
                                (IGNORE_ARG_NAMES|RELAX_STYPE_CHK))) {
        /* issue an error if src_sptr is not declared with an external 
         * statement and its interface does not match dest_sptr's interface.
         */
        error(1008, ERR_Severe, gbl.lineno, SYMNAME(dest_sptr), CNULL);
      }
    }
    if (STYPEG(src_sptr) == ST_PROC && INTERNALG(src_sptr)) {
       sdsc = SDSCG(dest_sptr);
       if (sdsc == 0)
         get_static_descriptor(dest_sptr);
       if (STYPEG(dest_sptr) == ST_MEMBER)
         sdsc = get_member_descriptor(dest_sptr);
       if (sdsc <= NOSYM)
         sdsc = SDSCG(dest_sptr);
       /* Note: closure pointer register argument to RTE_asn_closure is added
        * in exp_rte.c.
        */
       newargt = mk_argt(1);
       ARGT_ARG(newargt, 0) = STYPEG(sdsc) != ST_MEMBER ? mk_id(sdsc) :
                             check_member(dest, mk_id(sdsc));
       func = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(RTE_asn_closure), DT_NONE));
       /* Setting the recursive flag on the host subprogram forces the contains
        * subprograms to use the closure pointer register and not a direct
        * uplevel memory reference (which does not work with pointers
        * to internal procedures).
        */ 
       RECURP(gbl.currsub, 1);
       astnew = mk_func_node(A_CALL, func, 1, newargt);
       if (std)
         add_stmt_before(astnew, std);
       else
         add_stmt(astnew);
    }
  }
  func = intast_sym[I_PTR2_ASSIGN];
  ast = begin_call(A_ICALL, func, 2);
  A_OPTYPEP(ast, I_PTR2_ASSIGN);
  add_arg(dest);
  add_arg(src);
  if (XBIT(54, 0x40) && ast_is_sym(dest) && CONTIGATTRG(memsym_of_ast(dest))) {
    /* Add contiguity pointer check. We add the check after the pointer
     * assignment so we will get the correct section descriptor for dest.
     */
    if (std) {
      std = add_stmt_before(ast, std);
    } else {
      std = add_stmt(ast);
    }
    ast = mk_stmt(A_CONTINUE, 0);
    std = add_stmt_after(ast, std);
    gen_contig_check(dest, dest, 0, gbl.lineno, false, std);
    ast = mk_stmt(A_CONTINUE, 0); /* return a continue statement */
  }
  return ast;
}

/** \brief Generate contiguity check test inline (experimental)
 *  
 *  Called by gen_contig_check() below to generate the contiguity check inline.
 *  This is an experimental test since it looks at the descriptor flags, 
 *  data type, and src_sptr if src_sptr is an optional dummy argument. The
 *  endif asts are generated in gen_contig_check().
 *
 *  \param src is the source/pointer target ast.
 *  \param src_sptr is the source/pointer target sptr.
 *  \param sdsc is the source/pointer target's descriptor
 *  \param std is the optional statement descriptor for adding the check (0 
 *         if not applicable).
 *  
 *  \return the statement descriptor (std) of the generated code.
 */
static int  
inline_contig_check(int src, SPTR src_sptr, SPTR sdsc, int std)
{
  int flagsast = get_header_member_with_parent(src, sdsc, DESC_HDR_FLAGS);
  int lenast = get_header_member_with_parent(src, sdsc, DESC_HDR_BYTE_LEN);
  int sizeast = size_ast(src_sptr, DDTG(DTYPEG(src_sptr)));
  int cmp, astnew, seqast;

  /* Step 1: Add insertion point in AST */
  astnew = mk_stmt(A_CONTINUE, 0);
  if (std)
    std = add_stmt_before(astnew, std);
  else
   std = add_stmt(astnew);

  /* Step 2: If src_sptr is an optional argument, then generate an 
   * argument "present" check. Also generate this check if XBIT(54, 0x200)
   * is set which says to ignore null pointer targets.
   */
  if (XBIT(54, 0x200) || (SCG(src_sptr) == SC_DUMMY && OPTARGG(src_sptr))) {
    int present = ast_intr(I_PRESENT, stb.user.dt_log, 1, src);
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    std = add_stmt_after(astnew, std);
  }
   
  /* Step 3: Check descriptor flag to see if it includes
   * __SEQUENTIAL_SECTION.
   */
  seqast = mk_isz_cval(__SEQUENTIAL_SECTION, DT_INT);
  flagsast = ast_intr(I_AND, astb.bnd.dtype, 2, flagsast, seqast);
  cmp = mk_binop(OP_EQ, flagsast, astb.i0, DT_INT);
  astnew = mk_stmt(A_IFTHEN, 0);
  A_IFEXPRP(astnew, cmp); 
  std = add_stmt_after(astnew, std);

  /* Step 4: Check element size to see if it matches descriptor 
   * element size (i.e., check for a noncontiguous array subobject like 
   * p => dt(:)%m where dt has more than one component).
   */
  cmp = mk_binop(OP_EQ, lenast, sizeast, DT_INT);
  astnew = mk_stmt(A_IFTHEN, 0);
  A_IFEXPRP(astnew, cmp);
  std = add_stmt_after(astnew, std);

  return std;
}

/** \brief Generate a contiguous pointer check on a pointer assignment
 * when applicable.
 *
 * \param dest is the destination pointer.
 * \param src is the pointer target.
 * \param sdsc is an optional descriptor argument to pass to the check 
 * function (0 to use src's descriptor).
 * \param srcLine is the line number associated with the check.
 * \param cs is true when we are generating the check at a call-site.
 * \param std is the optional statement descriptor for adding the check (0 
 * if not applicable).
 */
void
gen_contig_check(int dest, int src, SPTR sdsc, int srcLine, bool cs, int std)
{
  int newargt, astnew;
  SPTR src_sptr, dest_sptr, func;
  bool isFuncCall, inlineContigCheck, ignoreNullTargets;
  int argFlags;

  if (ast_is_sym(src)) {
    src_sptr = memsym_of_ast(src);
  } else {
    interr("gen_contig_check: invalid src ast", src, 3);
    src_sptr = 0; 
  }
 
  if (ast_is_sym(dest)) {
    dest_sptr = memsym_of_ast(dest);
  } else {
    interr("gen_contig_check: invalid dest ast", dest, 3);
    dest_sptr = 0;
  }
  isFuncCall = (RESULTG(dest_sptr) && FVALG(gbl.currsub) != dest_sptr);
  /* If XBIT(54, 0x200) is set, we ignore null pointer targets. If
   * we have an optional argument, then we need to igore it if it's 
   * null (i.e., not present).
   */
  ignoreNullTargets = (XBIT(54, 0x200) || (SCG(dest_sptr) == SC_DUMMY && 
                                          OPTARGG(dest_sptr)));
  if (CONTIGATTRG(dest_sptr) || (CONTIGATTRG(src_sptr) && isFuncCall)) {
    int lineno, ptrnam, srcfil;
    if (sdsc <= NOSYM)
      sdsc = SDSCG(src_sptr);
    if (sdsc <= NOSYM)
      get_static_descriptor(src_sptr);
    if (STYPEG(src_sptr) == ST_MEMBER)
      sdsc = get_member_descriptor(src_sptr);
    if (sdsc <= NOSYM)
      sdsc = SDSCG(src_sptr);
    lineno = mk_cval1(srcLine, DT_INT);
    lineno = mk_unop(OP_VAL, lineno, DT_INT);
    ptrnam = !isFuncCall ? getstring(SYMNAME(dest_sptr), 
                                     strlen(SYMNAME(dest_sptr))+1) :
             getstring(SYMNAME(src_sptr), strlen(SYMNAME(src_sptr))+1);
    srcfil = getstring(gbl.curr_file, strlen(gbl.curr_file)+1);
    /* Check to see if we should inline the contiguity check. We do not
     * currently inline it if the user is also generating checks at the
     * call-site. Currently the inlining routine uses an argument structure
     * that may conflict with the call-site (but not when we're generating
     * checks for pointer assignments or arguments inside a callee). 
     * We could possibly support inlining at the call-site by deferring the
     * check after we generate the call-site code. However, this may be
     * a lot of work for something that probably will not be used too often.
     * Generating checks for pointer assignments and for arguments inside a
     * callee are typically sufficient. The only time one needs to check
     * the call-site is when the called routine is inside a library that was
     * not compiled with contiguity checking.
     */ 
    inlineContigCheck = (XBIT(54, 0x100) && !cs);
    if (inlineContigCheck) {
      std = inline_contig_check(src, src_sptr, sdsc, std);
    }
    newargt = mk_argt(6);
    ARGT_ARG(newargt, 0) = A_TYPEG(src) == A_SUBSCR ? A_LOPG(src) : src;
    ARGT_ARG(newargt, 1) = STYPEG(sdsc) != ST_MEMBER ? mk_id(sdsc) :
                           check_member(src, mk_id(sdsc));
    ARGT_ARG(newargt, 2) = lineno;
    ARGT_ARG(newargt, 3) = mk_id(ptrnam);
    ARGT_ARG(newargt, 4) = mk_id(srcfil);
    /* We can pass some flags about src here. For now, the flag is 1 if
     * dest_sptr is an optional argument or if we do not want to flag null
     * pointer targets. That way, we do not indicate a contiguity error
     * if the argument is not present or if the pointer target is null.
     */
    argFlags = mk_cval1( ignoreNullTargets ? 1 : 0, DT_INT);
    argFlags = mk_unop(OP_VAL, argFlags, DT_INT);
    ARGT_ARG(newargt, 5) = argFlags;
       
    func = mk_id(sym_mkfunc_nodesc(inlineContigCheck ? 
                                   mkRteRtnNm(RTE_contigerror) :
                                   mkRteRtnNm(RTE_contigchk), DT_NONE));
    astnew = mk_func_node(A_CALL, func, 6, newargt);
    if (inlineContigCheck) {
      /* generate endifs for inline contiguity checks */
      std = add_stmt_after(astnew, std);
      std = add_stmt_after(mk_stmt(A_ENDIF,0), std);
      if (ignoreNullTargets) {
        std = add_stmt_after(mk_stmt(A_ENDIF,0), std);
      }
      add_stmt_after(mk_stmt(A_ENDIF,0), std);
    } else if (std) {
      add_stmt_before(astnew, std);
    } else {
      add_stmt(astnew);
    }
  }
}

int
mk_component_ast(int leaf, int parent, int src_ast)
{
  int new_src_ast;
  int new_src_dt;
  int i, i2;
  int dt, nsubs, ndim, add, subs[MAXDIMS];
  ADSC *ad;

  new_src_ast = mk_id(leaf);
  new_src_dt = DTYPEG(leaf);
  dt = DDTG(new_src_dt);
  nsubs = 0;
  if (A_TYPEG(src_ast) == A_SUBSCR) {
    ad = AD_DPTR(DTYPEG(parent));
    nsubs = AD_NUMDIM(ad);
  }

  /* now check to see if we have to add subscripts because the
     component itself was originally an array. (Now the component
     will still be an array, but may have more dimensions.) */
  i2 = 0;
  ndim = 0;
  if (DTY(new_src_dt) == TY_ARRAY) {
    ad = AD_DPTR(new_src_dt);
    ndim = AD_NUMDIM(ad);
    if (nsubs != ndim) {
      /* we have to add subscripts. */
      add = ndim - nsubs;
      if (add <= 0)
        interr("mk_component_ast: derived type assign src", leaf, 3);
      else
        dt = new_src_dt; /* want array of ... */
      for (; i2 < add; i2++) {
        subs[i2] = mk_triple(AD_LWAST(ad, i2), AD_UPAST(ad, i2), 0);
      }
    }
  }
  if (nsubs) {
    add = i2 + nsubs;
    i = 0;
    for (; i2 < add; i2++) {
      subs[i2] = ASD_SUBS(A_ASDG(src_ast), i++);
    }
  }
  if (ndim) {
    new_src_ast = mk_subscr(new_src_ast, subs, ndim, dt);
    A_DTYPEP(new_src_ast, dt);
  }

  return new_src_ast;
}

/* Similar to ast.c:find_pointer_variable(), but it also looks for a
 * special case where we're performing pointer reshaping (e.g.
 * ptr(1:n) => x or ptr(1:) => x). Therefore, this function only gets
 * called by assign_pointer() and assign_intrinsic_to_pointer().
 */
static int
find_pointer_variable_assign(int ast, int dimFlag)
{
  if (A_TYPEG(ast) == A_SUBSCR) { /* ptr reshape */
    int shd, nd, asd, i, sub, ubast, lbast, ast2;
    int bounds_spec_list, bounds_remapping_list;
    shd = A_SHAPEG(ast);
    nd = SHD_NDIM(shd);
    asd = A_ASDG(ast);
    ast2 = A_LOPG(ast);
    if (A_TYPEG(ast2) == A_MEM)
      ast2 = A_MEMG(ast2);
    for (bounds_spec_list = bounds_remapping_list = i = 0; i < nd; ++i) {
      sub = ASD_SUBS(asd, i);
      ubast = A_UPBDG(sub);
      lbast = A_LBDG(sub);
      if (A_STRIDEG(sub)) {
        error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
              "stride specification not allowed in destination pointer "
              "section");
        return 0; /* p(l:u:s) => ... not valid for specified stride */
      }
      if (dimFlag & (0x2 << (i * 3))) {
        /* p(l:) => or p(:) =>
         * need to discard compiler inserted expr for upperbound.
         */
        A_UPBDP(sub, 0);
        ubast = 0;
      }
      if (dimFlag & (0x1 << (i * 3))) {
        /* p(:u) => or p(:) =>
         * need to discard compiler inserted expr for lowerbound.
         */
        A_LBDP(sub, 0);
        lbast = 0;
      }
      if (!lbast) {
        error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
              "illegal implied lowerbound in destination pointer "
              "section");
        return 0; /*p(:) => or p(:u) => not valid for implied lowerbound */
      }
      if (ubast) {
        if (bounds_spec_list) {
          /* cannot mix bounds-spec-list dimensions with
           * bounds-remapping-list dimensions (e.g., x(l:u,l:) is
           * not valid). See 7.4.2 Pointer Assignment in F2003 spec.
           */
          error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
                "inconsistent dimension specification in "
                "destination pointer section");

          return 0;
        }
        bounds_remapping_list = 1;
      } else {
        if (bounds_remapping_list) {
          /* cannot mix bounds-spec-list dimensions with
           * bounds-remapping-list dimensions (e.g., x(l:u,l:) is
           * not valid) See 7.4.2 Pointer Assignment in F2003 spec.
           */
          error(155, 3, gbl.lineno, "Illegal POINTER assignment -",
                "inconsistent dimension specification in "
                "destination pointer section");

          return 0;
        }
        bounds_spec_list = 1;
      }
    }
    ast = ast2;
  }
  return find_pointer_variable(ast);
}

int
chk_pointer_intent(int pvar, int refast)
{
  if (STYPEG(pvar) == ST_MEMBER) {
    if (refast) {
      int ss;
      ss = getbase(refast);
      if (SCG(ss) == SC_DUMMY && !POINTERG(ss) && !ALLOCATTRG(ss) &&
          INTENTG(ss) == INTENT_IN) {
        error(155, 3, gbl.lineno,
              "Derived type argument cannot be INTENT(IN) --", SYMNAME(ss));
        return 1;
      }
    }
  } else if (SCG(pvar) == SC_DUMMY && INTENTG(pvar) == INTENT_IN) {
    error(155, 3, gbl.lineno, "POINTER argument cannot be INTENT(IN) --",
          SYMNAME(pvar));
    return 1;
  }
  return 0;
}

int
any_pointer_source(int ast)
{
again:
  switch (A_TYPEG(ast)) {
  case A_ID:
    if (POINTERG(A_SPTRG(ast)))
      return 1;
    break;
  case A_FUNC:
  case A_SUBSCR:
  case A_SUBSTR:
    ast = A_LOPG(ast);
    goto again;
  case A_MEM:
    if (POINTERG(A_SPTRG(A_MEMG(ast))))
      return 1;
    ast = A_PARENTG(ast);
    goto again;
  default:
    break;
  }
  return 0;
}

int
chk_pointer_target(int pvar, int source)
{
  int targetbase;
  int target;

  find_pointer_target(source, &targetbase, &target);
  if (target == 0 || targetbase == 0) {
    error(155, 3, gbl.lineno, "Illegal target of a POINTER assignment", CNULL);
    return 1;
  }
  if (STYPEG(target) == ST_PROC) {
    if (is_procedure_ptr(pvar)) {
      ADDRTKNP(target, 1);
      return 0;
    }
    error(155, 3, gbl.lineno, "Illegal target of a POINTER assignment", CNULL);
    return 1;
  }
  if (!TARGETG(targetbase) && !POINTERG(target) &&
      !any_pointer_source(source)) {
    error(84, 3, gbl.lineno, SYMNAME(target),
          "- must have the TARGET or POINTER attribute");
    return 1;
  }
  if (TARGETG(targetbase)) {
    ADDRTKNP(targetbase, 1);
#ifdef PTRRHSG
    PTRRHSP(targetbase, 1);
#endif
    if (F77OUTPUT && XBIT(49, 0x8000) && DT_ISCMPLX(DDTG(DTYPEG(target))))
      error(155, 2, gbl.lineno, "Complex TARGET may not be properly aligned -",
            SYMNAME(target));
    if (is_protected(targetbase)) {
      err_protected(targetbase, "be a pointer target");
    }
  }
  return 0;
}

LOGICAL
is_protected(int sptr)
{
  if (PROTECTEDG(sptr) && ENCLFUNCG(sptr) != sem.mod_sym)
    return TRUE;
  return FALSE;
}

void
err_protected(int sptr, const char *context)
{
  char bf[ERRMSG_BUFFSIZE];
  sprintf(bf, "%s %s -",
          "A use-associated object with the PROTECTED attribute cannot",
          context);
  error(155, 3, gbl.lineno, bf, SYMNAME(sptr));
}

void
set_assn(int sptr)
{
  ASSNP(sptr, 1);
  /* it's legal for inherited submodules to access protected variables 
     defined parent modules, otherwise it's illegal */
  if (is_protected(sptr) && !is_used_by_submod(gbl.currsub, sptr)) {
    err_protected(sptr, "be assigned");
  }
}

static void
cast_to_typeless(SST *op, int typ)
{
  int conv_ast;

  (void)casttyp(op, typ);

  if (typ != TY_WORD && typ != TY_DWORD)
    return;

  if (SST_ASTG(op)) {
    conv_ast = mk_convert(SST_ASTG(op), typ);
    if (conv_ast != SST_ASTG(op)) {
      SST_ASTP(op, conv_ast);
    }
  }
}

/** \brief Make two operands conform in a binary operation.  The sequence of
           events is crucial to correct interpretation of expression.
 */
void
chkopnds(SST *lop, SST *operator, SST *rop)
{
  int dltype, drtype; /* data type */
  int opc, opl;

#define ARITH(o) \
  (o == OP_ADD || o == OP_SUB || o == OP_MUL || o == OP_DIV || o == OP_XTOI)
#define OK_LTYP(t)                                                          \
  ((t) == TY_WORD || (t) == TY_DWORD || (t) == TY_BINT || (t) == TY_SINT || \
   (t) == TY_INT || (t) == TY_CHAR || (t) == TY_NCHAR)

/* define OP_ macros not defined in ast.h which will represent the bit-wise
 * variants of OP_LOR, OP_LAND, OP_EQV, OP_XOR, respectively.
 */
#define OP_OR -1
#define OP_AND -2
#define OP_EQV -3
#define OP_XOR -4

  opc = SST_OPTYPEG(operator);

  /*
   * Rules for logical expressions: non-decimal constants assume
   * the data type of integer.  If at least one operand is
   * an integer the other operand becomes an integer and operation
   * is bitwise.  Handle logicals first since left operand is already
   * checked by semant and right must be checked here.
   */
  if (opc == OP_LOG) {
    int ty_lop, ty_rop;

    opl = (int)SST_OPCG(operator);
    chklog(rop);
    ty_lop = TY_OF(lop);
    ty_rop = TY_OF(rop);
    if (flg.standard) {
      if (!TY_ISLOG(ty_lop) || !TY_ISLOG(ty_rop))
        errwarn(95);
    }
    if (OK_LTYP(ty_lop) || OK_LTYP(ty_rop)) {
      /* if one operand an integer make other operand
       * and operator an integer.
       */
      cngtyp(lop, DT_INT);
      cngtyp(rop, DT_INT);
      if (opl == OP_LAND || opl == OP_LOR)
        opl = (opl == OP_LAND) ? OP_AND : OP_OR;
      else
        opl = (opl == OP_LEQV) ? OP_EQV : OP_XOR;
    }
    SST_OPCP(operator, opl);
    goto shape;
  } else {
    if (flg.standard) {
      if (TY_ISLOG(TY_OF(lop)) || TY_ISLOG(TY_OF(rop)))
        errwarn(95);
    }
  }
  /* catch use of structures and convert to other opnd's type or integer */
  if (((TY_OF(lop) == TY_STRUCT) && (TY_OF(rop) == TY_STRUCT)) ||
      ((TY_OF(lop) == TY_DERIVED) && (TY_OF(rop) == TY_DERIVED))) {
    cngtyp(lop, DT_INT);
    cngtyp(rop, DT_INT);
  }
  if ((TY_OF(lop) == TY_STRUCT) || (TY_OF(lop) == TY_DERIVED))
    cngtyp(lop, (int)PT_OF(rop));
  if ((TY_OF(rop) == TY_STRUCT) || (TY_OF(rop) == TY_DERIVED))
    cngtyp(rop, (int)PT_OF(lop));

  /*
   * Look for special case of 'double op complex' which should result
   * in both operands converted to double complex.
   */
  if ((TY_OF(lop) == TY_DBLE && TY_OF(rop) == TY_CMPLX) ||
      (TY_OF(lop) == TY_CMPLX && TY_OF(rop) == TY_DBLE)) {
    cngtyp(rop, DT_CMPLX16);
    cngtyp(lop, DT_CMPLX16);
  }

#ifdef TARGET_SUPPORTS_QUADFP
  /*
   * Look for special case of 'quad op complex or complex*16' which should result
   * in both operands converted to quad complex.
   */
  if (((TY_OF(rop) == TY_CMPLX || TY_OF(rop) == TY_DCMPLX) &&
       TY_OF(lop) == TY_QUAD) ||
      ((TY_OF(lop) == TY_CMPLX || TY_OF(lop) == TY_DCMPLX) &&
       TY_OF(rop) == TY_QUAD)) {
    cngtyp(rop, DT_QCMPLX);
    cngtyp(lop, DT_QCMPLX);
  }
#endif

  if (opc == OP_CMP) {
    /* Rules for relational expressions: nondecimal constants result
     * in a typeless comparison.  Size of the larger operand is used.
     * (per the VMS implementation)
     *
     * first catch illegal relational expressions i.e. mixture of
     * char and numeric
     */
    if ((TY_OF(lop) == TY_CHAR || TY_OF(lop) == TY_NCHAR) &&
        (TY_OF(rop) != TY_CHAR && TY_OF(rop) != TY_NCHAR)) {
      errsev(124);
      SST_IDP(lop, S_CONST);
      SST_DTYPEP(lop, DT_INT);
      SST_CVALP(lop, 0);
    }
    if ((TY_OF(rop) == TY_CHAR || TY_OF(rop) == TY_NCHAR) &&
        (TY_OF(lop) != TY_CHAR && TY_OF(lop) != TY_NCHAR)) {
      errsev(124);
      SST_IDP(rop, S_CONST);
      SST_DTYPEP(rop, DT_INT);
      SST_CVALP(rop, 0);
    }

    /* Catch certain relational operations to avoid type conversion unless
     * the other operand is integer or logical.  For integer/logical,
     * cast the 'word' value to the respective integer/logical type.
     */
    if (TY_OF(lop) == TY_DWORD) {
      if (!TY_ISINT(TY_OF(rop)) && !TY_ISLOG(TY_OF(rop))) {
        /* typeless compare */
        (void)cast_to_typeless(rop, DT_DWORD);
        goto shape;
      }
    }
    if (TY_OF(rop) == TY_DWORD) {
      if (!TY_ISINT(TY_OF(lop)) && !TY_ISLOG(TY_OF(lop))) {
        /* typeless compare */
        (void)cast_to_typeless(lop, DT_DWORD);
        goto shape;
      }
    }
    if (TY_OF(lop) == TY_WORD) {
      /* here comparison must be at least 64-bits */
      if (TY_OF(rop) == TY_DBLE || TY_ISCMPLX(TY_OF(rop))) {
        (void)cast_to_typeless(rop, DT_DWORD);
        (void)casttyp(lop, DT_DWORD);
        goto shape;
      }
      if (!TY_ISINT(TY_OF(rop)) && !TY_ISLOG(TY_OF(rop))) {
        (void)cast_to_typeless(rop, DT_WORD);
        goto shape;
      }
    }
    if (TY_OF(rop) == TY_WORD) {
      /* here comparison must be at least 64-bits */
      if (TY_OF(lop) == TY_DBLE || TY_ISCMPLX(TY_OF(lop))) {
        (void)cast_to_typeless(lop, DT_DWORD);
        (void)casttyp(rop, DT_DWORD);
        goto shape;
      }
      if (!TY_ISINT(TY_OF(lop)) && !TY_ISLOG(TY_OF(lop))) {
        (void)cast_to_typeless(lop, DT_WORD);
        goto shape;
      }
    }
  }
  if (ARITH(opc) || opc == OP_CAT) {
    /* handle nondecimals in arithmetic operations and
     * character expressions
     */
    if ((SST_ISNONDECC(lop) &&
         (SST_ISNONDECC(rop) || TY_OF(rop) == TY_DWORD)) ||
        (TY_OF(lop) == TY_DWORD &&
         (SST_ISNONDECC(rop) || TY_OF(rop) == TY_DWORD))) {
      cngtyp(lop, DT_INT);
      cngtyp(rop, DT_INT);
    }
    if (TY_ISNUMERIC(TY_OF(rop)) &&
        (SST_ISNONDECC(lop) || (TY_OF(lop) == TY_DWORD)))
      cngtyp(lop, (int)PT_OF(rop));

    if (TY_ISNUMERIC(TY_OF(lop)) &&
        (SST_ISNONDECC(rop) || (TY_OF(rop) == TY_DWORD)))
      cngtyp(rop, (int)PT_OF(lop));
  }

  /* Change logical types to integer for
   * arithmetic and relational operations
   */
  if (TY_ISLOG(TY_OF(lop))) {
    if (SST_IDG(lop) != S_CONST)
      mkexpr1(lop);
    dltype = TYPE_OF(lop);
    dltype = DDTG(dltype);
    cngtyp(lop, DT_INT + (dltype - DT_LOG));
  }

  if (TY_ISLOG(TY_OF(rop))) {
    if (SST_IDG(rop) != S_CONST)
      mkexpr1(rop);
    drtype = TYPE_OF(rop);
    drtype = DDTG(drtype);
    cngtyp(rop, DT_INT + (drtype - DT_LOG));
  }

  if (opc == OP_XTOI) {
    /* Exponentiation breaks the normal rule. If exponent is integer,
     * don't change its type.
     */
    if (TY_ISINT(TY_OF(rop))) {
      if (TY_OF(rop) < TY_OF(lop)) {
        /* Check left operand */
        if (!TY_ISNUMERIC(TY_OF(lop)))
          cngtyp(lop, (int)PT_OF(rop));
        if (TY_OF(rop) != TY_INT8)
          cngtyp(rop, DT_INT);
        if (SST_IDG(lop) == S_CONST && SST_IDG(rop) == S_CONST)
          /* scalar constant ** int constant */
          return;
        mkexpr1(lop);
        mkexpr1(rop);
        if (DTY(SST_DTYPEG(lop)) == TY_ARRAY) {
          (void)chkshape(rop, lop, TRUE);
          return;
        }
        if (DTY(SST_DTYPEG(rop)) == TY_ARRAY) {
          (void)chkshape(lop, rop, TRUE);
          return;
        }
        /* scalar ** int scalar */
        return;
      }
    } else if (!XBIT(124, 0x40000) && SST_IDG(rop) == S_CONST) {
      int pw, is_int;
      INT conval;
      INT num[4];
      switch (TY_OF(rop)) {
      case TY_CMPLX:
        conval = SST_CVALG(rop);
        if (CONVAL2G(conval) != 0)
          break;
        conval = CONVAL1G(conval);
        goto ck_real_pw;
      case TY_REAL:
        conval = SST_CVALG(rop);
      ck_real_pw:
        is_int = xfisint(conval, &pw);
        if ((!flg.ieee || pw == 1 || pw == 2) && is_int) {
          if (TY_OF(lop) < TY_OF(rop))
            cngtyp(lop, (int)SST_DTYPEG(rop)); /* Normal rule */
          SST_CVALP(rop, pw);
          SST_DTYPEP(rop, DT_INT4);
          SST_ASTP(rop, mk_cval1(pw, DT_INT4));
          return;
        }
        break;
      case TY_DCMPLX:
        conval = SST_CVALG(rop);
        if (!is_dbl0(CONVAL2G(conval)))
          break;
        conval = CONVAL1G(conval);
        goto ck_dble_pw;
      case TY_DBLE:
        conval = SST_CVALG(rop);
      ck_dble_pw:
        num[0] = CONVAL1G(conval);
        num[1] = CONVAL2G(conval);
        is_int = xdisint(num, &pw);
        if ((!flg.ieee || pw == 1 || pw == 2) && is_int) {
          if (TY_OF(lop) < TY_OF(rop))
            cngtyp(lop, (int)SST_DTYPEG(rop)); /* Normal rule */
          SST_CVALP(rop, pw);
          SST_DTYPEP(rop, DT_INT4);
          SST_ASTP(rop, mk_cval1(pw, DT_INT4));
          return;
        }
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        conval = SST_CVALG(rop);
        if (!is_quad0(CONVAL2G(conval)))
          break;
        conval = CONVAL1G(conval);
        goto ck_quad_pw;
      case TY_QUAD:
        conval = SST_CVALG(rop);
      ck_quad_pw:
        num[0] = CONVAL1G(conval);
        num[1] = CONVAL2G(conval);
        num[2] = CONVAL3G(conval);
        num[3] = CONVAL4G(conval);
        is_int = xqisint(num, &pw);
        if ((!flg.ieee || pw == POW1 || pw == POW2) && is_int) {
          if (TY_OF(lop) < TY_OF(rop))
            cngtyp(lop, (int)SST_DTYPEG(rop)); /* Normal rule */
          SST_CVALP(rop, pw);
          SST_DTYPEP(rop, DT_INT4);
          SST_ASTP(rop, mk_cval1(pw, DT_INT4));
          return;
        }
        break;
#endif
      default:
        break;
      }
    }
  }
  /*
   * Perform type conversion of both operands to a common data type.
   * Remember that character and records are highest data types.  For
   * non-character operations character data should be converted to down
   * rather than follow the normal rule.  When records are used they should
   * always be converted down.  This avoids propagation of errors.
   */
  if (TY_OF(lop) < TY_OF(rop)) {
    if (((TY_OF(rop) == TY_STRUCT) || (TY_OF(rop) == TY_DERIVED)) ||
        (opc != OP_CAT && (TY_OF(rop) == TY_CHAR || TY_OF(rop) == TY_NCHAR)))
      cngtyp(rop, (int)SST_DTYPEG(lop)); /* Break normal rule */
    else
      cngtyp(lop, (int)SST_DTYPEG(rop)); /* Normal rule */
  } else if (TY_OF(rop) < TY_OF(lop)) {
    if (((TY_OF(lop) == TY_STRUCT) || (TY_OF(lop) == TY_DERIVED)) ||
        (opc != OP_CAT && (TY_OF(lop) == TY_CHAR || TY_OF(lop) == TY_NCHAR)))
      cngtyp(lop, (int)SST_DTYPEG(rop)); /* Break normal rule */
    else
      cngtyp(rop, (int)SST_DTYPEG(lop)); /* Normal rule */
  } else if ((TY_OF(lop) == TY_STRUCT) || (TY_OF(lop) == TY_DERIVED)) {
    /* Both are == and structure.  can't do binary operations with
     * structures.
     */
    cngtyp(lop, DT_INT);
    cngtyp(rop, DT_INT);
  } else if (TY_OF(lop) == TY_CHAR || TY_OF(lop) == TY_NCHAR) {
    /* Both are == and character;
     * char op char is only legal for concat and relational operators
     */
    if (opc != OP_CAT && opc != OP_CMP) {
      cngtyp(lop, DT_INT);
      cngtyp(rop, DT_INT);
    } else if (DTY(TYPE_OF(lop)) == TY_ARRAY && !TY_ISVEC(TY_CHAR))
      error83(TY_CHAR);
  }
/*
 * Types of operands are the same now make sure shapes of both
 * operands agree.
 */
shape:
  if (DTY(SST_DTYPEG(lop)) == TY_ARRAY && DTY(SST_DTYPEG(rop)) != TY_ARRAY)
    cngshape(rop, lop);
  else
    cngshape(lop, rop);
}

/** \brief Perform a unary operation on logical rhs.
 */
void
unop(SST *rslt, SST *operator, SST *rop)
{
  int rdtype;         /* data type */
  int lbtype;         /* basic data type (INT, LOG, etc) */
  int opc;            /* operation code */
  int drtype;         /* data type */

  opc = SST_OPTYPEG(operator);
  if (opc != OP_ADD && opc != OP_SUB) {
    return;
  }
  if (!TY_ISLOG(TY_OF(rop))) {
    return;
  }
  if (SST_IDG(rop) == S_STFUNC)
    mkexpr1(rop);
  constant_lvalue(rop);

  if (SST_IDG(rop) != S_CONST)
    mkexpr1(rop);

  drtype = TYPE_OF(rop);
  drtype = DDTG(drtype);
  cngtyp(rop, DT_INT + (drtype - DT_LOG));

  cngshape(rop, rop);

  mkexpr1(rop);
  lbtype = TY_OF(rop);
  rdtype = TYPE_OF(rop);
  SST_IDP(rslt, S_EXPR);
  SST_DTYPEP(rslt, rdtype);
}

/** \brief Perform a binary operation on rhs1 and rhs2.  They both conform in
           data type and shape.
 */
void
binop(SST *rslt, SST *lop, SST *operator, SST *rop)
{
  /* Values for left and right operands */
  int ldtype, rdtype; /* data type */
  int lbtype;         /* basic data type (INT, LOG, etc) */
  int newtyp;
  int lsptr, rsptr;          /* symbol table pointers */
  int klsptr, krsptr, krslt; /* symbol table pointers */
  int llen, rlen;            /* character string lengths */
  int opc, opc1;             /* operation code */

  char *carea; /* temporary area for concatenation */
  int count, condition;
  INT conval;
  LOGICAL is_array;
  ADSC *ad1;
  int numdim;
  INT val1[2], val2[2], res[2], val[4];
  int cvlen;

  /*
   * Step 1: Catch statement functions and call mkexpr1 to process the
   *         linked list (arguments) on the semantic stack.
   */
  if (SST_IDG(lop) == S_STFUNC)
    mkexpr1(lop);
  if (SST_IDG(rop) == S_STFUNC)
    mkexpr1(rop);

  /*
   * Step 2: Catch some illegal cases early.
   */
  /* catch vector ops on hollerith constants before changing their type */
  if ((TYPE_OF(rop) == DT_HOLL && DTY(TYPE_OF(lop)) == TY_ARRAY) ||
      (TYPE_OF(lop) == DT_HOLL && DTY(TYPE_OF(rop)) == TY_ARRAY))
    errsev(100);

  opc = SST_OPTYPEG(operator);
  constant_lvalue(lop);
  constant_lvalue(rop);
  /*
   * Step 3: Ensure that the data types and shapes of both operands agree.
   */
  chkopnds(lop, operator, rop);

  /*
   * Step 4: Shortcut comparisons between typeless and different sized
   *         operands.  A 32-bit typeless is always less than a 64-bit
   *         typeless.
   */
  if (opc == OP_CMP) {
    if ((TYPE_OF(lop) == TY_DWORD) && (TYPE_OF(rop) == TY_WORD)) {
      conval = 1;
      goto shortcut;
    }
    if ((TYPE_OF(rop) == TY_DWORD) && (TYPE_OF(lop) == TY_WORD)) {
      conval = -1;
      goto shortcut;
    }
  }

  /*
   * Step 5: Optimize AND's and OR's in logical expressions by short
   *         circuiting if both operands are logicals and one operand
   *         is a logical constant .false. for an AND operation or a
   *         .true. for an OR operation.  For example  l .or. c
   *	   would avoid the evaluation of l if the constant c were true
   *  	   or would return the evaluation of l if the constant c were false.
   */
  if (opc == OP_LOG && TY_ISLOG(TY_OF(lop)) && TY_ISLOG(TY_OF(rop))) {
    if ((opc1 = SST_OPCG(operator)) == OP_LOR)
      condition = SCFTN_FALSE & 1;
    else if (opc1 == OP_LAND)
      condition = SCFTN_TRUE & 1;
    else
      goto step6;
    if (SST_IDG(lop) == S_CONST) {
      val1[1] = (DTY(TY_OF(lop)) == TY_LOG8) ? CONVAL2G(SST_CVALG(lop))
                                             : SST_CVALG(lop);
      if ((val1[1] & 1) == condition)
        *rslt = *rop;
      else
        *rslt = *lop;
      SST_ASTP(rop, 0); /* short circuit optimization occurred */
      return;
    } else if (SST_IDG(rop) == S_CONST) {
      val1[1] = (DTY(TY_OF(rop)) == TY_LOG8) ? CONVAL2G(SST_CVALG(rop))
                                             : SST_CVALG(rop);
      if ((val1[1] & 1) == condition)
        *rslt = *lop;
      else
        *rslt = *rop;
      SST_ASTP(rop, 0); /* short circuit optimization occurred */
      return;
    }
  }

/* assertion: We have two operands of equal data types, of equal shape,
 *            and an operation to perform.  If constants are involved,
 *            non-decimal constants have assumed a different type.
 * Step 6: Possibly constant fold.
 */
step6:
  if (SST_IDG(lop) == S_CONST && SST_IDG(rop) == S_CONST) {
    /* Perform constant folding based on operator */
    switch (opc) {
    case OP_LOG:
      opc1 = SST_OPCG(operator); /* real logical operator */
      if (DTY(TY_OF(lop)) == TY_LOG8) {
        val1[0] = CONVAL1G(SST_CVALG(lop));
        val1[1] = CONVAL2G(SST_CVALG(lop));
      } else {
        val1[1] = SST_CVALG(lop);
        if (val1[1] < 0)
          val1[0] = -1;
        else
          val1[0] = 0;
      }
      if (DTY(TY_OF(rop)) == TY_LOG8) {
        val2[0] = CONVAL1G(SST_CVALG(rop));
        val2[1] = CONVAL2G(SST_CVALG(rop));
      } else {
        val2[1] = SST_CVALG(rop);
        if (val2[1] < 0)
          val2[0] = -1;
        else
          val2[0] = 0;
      }
      if (opc1 == OP_LEQV || opc1 == OP_EQV) {
        conval = cmp64(val1, val2);
        SST_CVALP(rslt, clog_to_log((INT)(conval == 0)));
      } else if (opc1 == OP_LNEQV) {
        conval = cmp64(val1, val2);
        SST_CVALP(rslt, clog_to_log((INT)(conval != 0)));
      } else if (opc1 == OP_LOR) {
        or64(val1, val2, res);
        SST_CVALP(rslt, clog_to_log(res[0] | res[1]));
      } else if (opc1 == OP_LAND) {
        and64(val1, val2, res);
        SST_CVALP(rslt, clog_to_log(res[0] | res[1]));
      } else if (opc1 == OP_XOR) {
        xor64(val1, val2, res);
        SST_CVALP(rslt, clog_to_log(res[0] | res[1]));
      } else if (opc1 == OP_OR) {
        or64(val1, val2, res);
        SST_CVALP(rslt, clog_to_log(res[0] | res[1]));
      } else if (opc1 == OP_AND) {
        and64(val1, val2, res);
        SST_CVALP(rslt, clog_to_log(res[0] | res[1]));
      } else
        interr("binop: bad opcode in SST_OPC:", opc1, 0);
      SST_DTYPEP(rslt, DT_LOG);
      if (DTY(DT_LOG) == TY_LOG8) {
        res[1] = SST_CVALG(rslt);
        if (res[1] < 0)
          res[0] = -1 & 0xFFFFFFFF;
        else
          res[0] = 0;
        SST_CVALP(rslt, getcon(res, DT_LOG8));
      }
      break;
    case OP_XTOI:
    case OP_XTOX:
      if (TYPE_OF(rop) == DT_INT8) {
        conval = const_xtoi(SST_CVALG(lop), SST_CVALG(rop), TYPE_OF(lop));
        SST_CVALP(rslt, conval);
      } else if (DT_ISINT(TYPE_OF(rop))) {
        count = SST_CVALG(rop);
        if (TYPE_OF(rop) != DT_INT4)
          count = cngcon(count, (int)TYPE_OF(rop), DT_INT4);
        conval = _xtok(SST_CVALG(lop), count, TYPE_OF(lop));
        SST_CVALP(rslt, conval);
      } else {
        /* can't fold if exponent is not an integer constant */
        goto binop_exp;
      }
      break;

    case OP_CAT:
      SST_CVLENP(rslt, 0);
      if (TY_OF(lop) != TY_OF(rop))
        goto error_cat;
      if (TY_OF(lop) != TY_CHAR && TY_OF(lop) != TY_NCHAR)
        goto error_cat;
      klsptr = lsptr = SST_SYMG(lop);
      krsptr = rsptr = SST_SYMG(rop);
      ldtype = DTYPEG(lsptr);
      rdtype = DTYPEG(rsptr);
#if DEBUG
      assert(STYPEG(lsptr) == ST_CONST &&
                 (DTY(ldtype) == TY_CHAR || DTY(ldtype) == TY_NCHAR),
             "binop:CAT1", lsptr, 2);
      assert(STYPEG(rsptr) == ST_CONST &&
                 (DTY(rdtype) == TY_CHAR || DTY(rdtype) == TY_NCHAR),
             "binop:CAT2", rsptr, 2);
#endif
      llen = string_length(ldtype);
      rlen = string_length(rdtype);
      carea = getitem(0, llen + rlen);
      if (TY_OF(lop) == TY_NCHAR) {
        klsptr = CONVAL1G(lsptr);
        krsptr = CONVAL1G(rsptr);
      }
      BCOPY(carea, stb.n_base + CONVAL1G(klsptr), char, llen);
      BCOPY(carea + llen, stb.n_base + CONVAL1G(krsptr), char, rlen);
      krslt = getstring(carea, llen + rlen);
      newtyp = get_type(2, TY_OF(lop), mk_cval(llen + rlen, DT_INT4));
      if (TY_OF(lop) == TY_NCHAR) {
        llen = kanji_len((unsigned char *)stb.n_base + CONVAL1G(klsptr), llen);
        rlen = kanji_len((unsigned char *)stb.n_base + CONVAL1G(krsptr), rlen);
        val[0] = krslt;
        val[1] = 0;
        val[2] = 0;
        val[3] = 0;
        krslt = getcon(val, newtyp);
      }
      SST_SYMP(rslt, krslt);
      SST_DTYPEP(rslt, newtyp);
      break;

    error_cat:
      SST_CVLENP(rslt, 0);
      errsev(146);
      SST_SYMP(rslt, getstring(" ", 1));
      SST_DTYPEP(rslt, DT_CHAR);
      break;

    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
      SST_CVALP(rslt, const_fold(opc, SST_CVALG(lop), SST_CVALG(rop),
                                 (int)TYPE_OF(lop)));
      SST_DTYPEP(rslt, TYPE_OF(lop));
      break;

    case OP_CMP:
      conval =
          const_fold(OP_CMP, SST_CVALG(lop), SST_CVALG(rop), (int)TYPE_OF(lop));
    shortcut:
      switch (SST_OPCG(operator)) {
      case OP_EQ:
        conval = (conval == 0);
        break;
      case OP_GE:
        conval = (conval >= 0);
        break;
      case OP_GT:
        conval = (conval > 0);
        break;
      case OP_LE:
        conval = (conval <= 0);
        break;
      case OP_LT:
        conval = (conval < 0);
        break;
      case OP_NE:
        conval = (conval != 0);
        break;
      }
      conval = conval ? SCFTN_TRUE : SCFTN_FALSE;
      if (DTY(stb.user.dt_log) == TY_LOG8) {
        res[1] = conval;
        if (res[1] < 0)
          res[0] = -1 & 0xFFFFFFFF;
        else
          res[0] = 0;
        SST_CVALP(rslt, getcon(res, DT_LOG8));
      } else
        SST_CVALP(rslt, conval);
      SST_DTYPEP(rslt, stb.user.dt_log);
      break;

    default:
      interr("binop: bad opcode:", opc, 0);
      break;
    }
    return;
  }

  /*
   * assertion: We have two operands that are not both constants
   *            therefore constant folding is not possible.
   * step 7: Make an expression from operands and operator.
   */
  if (opc == OP_XTOI && SST_IDG(rop) == S_CONST && TYPE_OF(rop) == DT_INT &&
      SST_CVALG(rop) == 2) {
    /* optimize x raised to the power of 2 */
    mkexpr(lop);
    SST_IDP(rslt, S_EXPR);
    SST_DTYPEP(rslt, SST_DTYPEG(lop));
  } else if (opc == OP_LOG) {
    /* We have a logical expression */
    mkexpr(lop);
    mkexpr(rop);
    opc = SST_OPCG(operator);
    chklog(lop);
    chklog(rop);

    if (DTY(TYPE_OF(lop)) == TY_ARRAY || DTY(TYPE_OF(rop)) == TY_ARRAY)
      ;
    else {
      /* Normal scalar logical expressions should be LOG*4 */
      mklogint4(lop);
      mklogint4(rop);
    }
    SST_IDP(rslt, S_EXPR);
  } else {
  binop_exp:
    if (opc == OP_CAT) {
      int rdt;

      cvlen = 0;
      if (TY_OF(lop) == TY_CHAR) {
        if (TY_OF(rop) != TY_CHAR)
          goto error_cat;
        mkexpr1(lop);
        mkexpr1(rop);
        rdt = DT_ASSCHAR;
      }
      else if (TY_OF(lop) == TY_NCHAR) { /* kanji */
        if (TY_OF(rop) != TY_NCHAR)
          goto error_cat;
        mkexpr1(lop);
        mkexpr1(rop);
        rdt = DT_ASSNCHAR;
      }
      else
        goto error_cat;
      ldtype = TYPE_OF(lop);
      rdtype = TYPE_OF(rop);
      is_array = FALSE;
      if (DTY(ldtype) == TY_ARRAY) {
        is_array = TRUE;
        ldtype = DTY(ldtype + 1);
      }
      if (DTY(rdtype) == TY_ARRAY) {
        is_array = TRUE;
        rdtype = DTY(rdtype + 1);
      }
      if (ldtype != DT_ASSCHAR && ldtype != DT_DEFERCHAR &&
          ldtype != DT_ASSNCHAR && rdtype != DT_ASSNCHAR &&
          ldtype != DT_DEFERNCHAR && rdtype != DT_DEFERNCHAR &&
          rdtype != DT_ASSCHAR && rdtype != DT_DEFERCHAR) {
        llen = SST_CVLENG(lop);
        rlen = SST_CVLENG(rop);
        if (llen == 0 && !A_ALIASG(DTY(ldtype + 1)))
          goto cat_result;
        if (rlen == 0 && !A_ALIASG(DTY(rdtype + 1)))
          goto cat_result;
        if (llen) {
          if (rlen == 0)
            rlen = mk_cval(string_length(rdtype), DT_INT4);
        } else if (rlen) {
          llen = mk_cval(string_length(ldtype), DT_INT4);
        }
        if (llen) {
          cvlen = mk_binop(OP_ADD, llen, rlen, DT_INT4);
          rdt = get_type(2, (int)DTY(rdt), cvlen);
        } else {
          llen = string_length(ldtype);
          rlen = string_length(rdtype);
          rdt = get_type(2, (int)DTY(rdt), mk_cval(llen + rlen, DT_INT4));
          cvlen = DTY(rdt + 1);
        }
      }
    cat_result:
      if (is_array) {
        if (TY_OF(lop) == TY_CHAR) {
          if (DTY(TYPE_OF(lop)) == TY_ARRAY)
            ad1 = AD_DPTR(TYPE_OF(lop));
          else
            ad1 = AD_DPTR(TYPE_OF(rop));
          numdim = AD_NUMDIM(ad1);
          rdt = get_array_dtype(numdim, rdt);
        } else {
          rdt = get_type(3, TY_ARRAY, rdt);
          DTY(rdt + 2) = 0;
        }
      }
      SST_IDP(rslt, S_EXPR);
      SST_DTYPEP(rslt, rdt);
      SST_CVLENP(rslt, cvlen);
    } else {
      mkexpr1(lop);
      mkexpr1(rop);
      lbtype = TY_OF(lop);
      ldtype = TYPE_OF(lop);

      if (opc == OP_CMP) {
        opc = SST_OPCG(operator);
        if (DTY(TYPE_OF(lop)) == TY_ARRAY || DTY(TYPE_OF(rop)) == TY_ARRAY)
          is_array = TRUE;
        else
          is_array = FALSE;
        if (TY_ISCMPLX(TY_OF(lop)) && (opc != OP_EQ && opc != OP_NE))
          errsev(96);
        if (is_array) {
          ldtype = get_type(3, TY_ARRAY, stb.user.dt_log);
          DTY(ldtype + 2) = 0;
        } else
          ldtype = stb.user.dt_log;
      }

      SST_IDP(rslt, S_EXPR);
      SST_DTYPEP(rslt, ldtype);
    }
  }
}

/* convert C's logical value to pgftn's logical (.true./.false.) */
static INT
clog_to_log(INT clog)
{
  if (clog)
    return SCFTN_TRUE;
  return SCFTN_FALSE;
}

/** \brief Return a new data type based on the rules of applying a length
   specifier to an existing base data type (i.e. LOGICAL*1) passed in as
   a TY_ value.

   \a sptr points to the symbol table entry whose data type is being modified.
   This is for error messages.  If no \a sptr then message is for type
   declaration verb.

   Special case:
   >   When sptr is 0, the data type adjustment is occurring at the time when
   >   the length immediately follows a data type (i.e. when \<data type> is
   >   being processed). When sptr is non-zero, this means that the length
   >   follows the name of the symbol (\<data type> has already been processed)
   >   (i.e. CHARACTER FOO*1); and a length of -1 implies that no length
   >   was specified.
   >
   >   So, when sptr is nonzero and len is -1, we do not attempt to adjust
   >   the data type; if so, we will incorrectly adjust
   >   <pre>
   >       REAL*4  rv</pre>
   >   when the "r8" option has been selected (-x 124 8).
 */
int
mod_type(int dtype, int ty, int kind, int len, int propagated, int sptr)
{
  /*
   * The dtype could be any static or dynamic dtype therefore use the
   * TY_type field for comparisons.  For example, there is the static
   * entry for CHARACTER*1 and the dynamic entries for CHARACTER*number.
   */
  if (sptr && len == -1)
    return dtype;
  /*
   * the possible values of 'ty' are those which can be base types.
   */
  switch (ty) {
  case TY_BINT:
    if (kind != 0)
      error(32, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "byte", CNULL);
    break;
  case TY_INT:
  case TY_INT8:
    if (kind == 0) {
      if (!flg.i4 && dtype == DT_INT)
        return (DT_SINT);
      return dtype;
    }
    if (kind == 1) {
      if (len == 1)
        return (DT_BINT);
      if (len == 2)
        return (DT_SINT);
      if (len == 4)
        return (DT_INT4);
      if (len == 8 && !XBIT(57, 0x2))
        return (DT_INT8);
    }
    error(31, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "integer", CNULL);
    break;
  case TY_LOG:
  case TY_LOG8:
    if (kind == 0) {
      if (!flg.i4 && dtype == DT_LOG)
        return (DT_SLOG);
      return dtype;
    }
    if (kind == 1) {
      if (len == 1)
        return (DT_BLOG);
      if (len == 2)
        return (DT_SLOG);
      if (len == 4)
        return (DT_LOG4);
      if (len == 8 && !XBIT(57, 0x2))
        return (DT_LOG8);
    }
    error(31, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "logical", CNULL);
    break;
  case TY_DBLE:
    if (sem.ogdtype == DT_REAL8 && kind != 0) {
      error(32, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "doubleprecision",
            CNULL);
      break;
    }
    FLANG_FALLTHROUGH;
  case TY_REAL:
    if (kind == 0)
      return dtype;
    if (kind == 1) {
      if (len == 16 && !XBIT(57, 0x4)) {
        if (XBIT(57, 0x10)) {
          if (!propagated)
            error(437, 2, gbl.lineno, "REAL*16", "REAL*8");
          return DT_REAL8;
        } else {
          return DT_QUAD;
        }
      }
      if (len == 8)
        return DT_REAL8;
      if (len == 4)
        return (DT_REAL4);
    }
    error(31, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) :
                                     (ty == TY_HALF ? "real2" : "real"), CNULL);
    break;
  case TY_DCMPLX:
    if (sem.ogdtype == DT_CMPLX16 && kind != 0) {
      error(32, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "doublecomplex", CNULL);
      break;
    }
    FLANG_FALLTHROUGH;
  case TY_CMPLX:
    if (kind == 0)
      return dtype;
    if (kind == 1) {
      if (len == 32 && !XBIT(57, 0x8)) {
        if (XBIT(57, 0x10)) {
          if (!propagated)
            error(437, 2, gbl.lineno, "COMPLEX*32", "COMPLEX*16");
          return DT_CMPLX16;
        } else {
          return DT_QCMPLX;
        }
      }
      if (len == 16)
        return DT_CMPLX16;
      if (len == 8)
        return (DT_CMPLX8);
    }
    error(31, 2, gbl.lineno, (sptr) ? SYMNAME(sptr) : "complex", CNULL);
    break;
  case TY_CHAR:
  case TY_NCHAR:
    switch (kind) {
    case 3: /* zero-size character */
      return get_type(2, DTY(dtype), astb.i0);
    case 5: /* '*(:)' */
      if (DTY(dtype) == TY_CHAR)
        return DT_DEFERCHAR;
      else
        return DT_DEFERNCHAR;
    case 2: /* '*(*)' */
      if (DTY(dtype) == TY_CHAR)
        return DT_ASSCHAR;
      else
        return DT_ASSNCHAR;
    case 1: /* constant length */
      return get_type(2, DTY(dtype), mk_cval(len, DT_INT4));
    case 4: /* adjustable length */
      return get_type(2, DTY(dtype), len);
    case 0: /* no length */
      return get_type(2, DTY(dtype), astb.i1);
    }
    break;
  default:
    interr("mod_type/data type: bad data type:", dtype, 0);
    break;
  }
  return dtype;
}

/** \brief Return the printable representation of a semantic stack entry
 */
const char *
prtsst(SST *stkptr)
{
  static char symbuf[132];
  int val, dtype;

  val = SST_SYMG(stkptr);
  dtype = SST_DTYPEG(stkptr);
  if (SST_IDG(stkptr) == S_CONST) {
    if (dtype == DT_QUAD || dtype == DT_REAL8 || DT_ISCMPLX(dtype)) {
      return (getprint(val));
    } else {
      if (DT_ISREAL(dtype)) {
        sprintf(symbuf, "%f", *(float *)&val);
      } else if (DT_ISLOG(dtype)) {
        if (val == SCFTN_TRUE)
          sprintf(symbuf, ".TRUE.");
        else
          sprintf(symbuf, ".FALSE.");
      } else if (DTYG(dtype) == TY_CHAR)
        sprintf(symbuf, "\"%s\"", stb.n_base + CONVAL1G(val));
      else
        sprintf(symbuf, "%d", val);
    }
  }
  return (symbuf);
}

/** \brief Dereference an ast to determine the base, i.e. its symbol pointer.
 */
int
getbase(int ast)
{
  switch (A_TYPEG(ast)) {
  case A_SUBSTR:
  case A_SUBSCR:
    return (getbase((int)A_LOPG(ast)));

  case A_ID:
    return A_SPTRG(ast);

  case A_MEM:
    return (getbase((int)A_PARENTG(ast)));

  case A_FUNC:
  case A_CALL:
    return (getbase((int)A_LOPG(ast)));

  default:
    return 0;
  }
}

/*---------------------------------------------------------------------*
 * Handle DO statements                                                *
 *---------------------------------------------------------------------*/

/** \brief Generate ILMs which computes the address of the index variable.
           Need to do it this way since the ILMs which were originally
           computed during the parse are not saved across the blocks
 */
int
do_index_addr(int sptr)
{
  return ref_object(sptr);
}

/** \brief Write out block DO AST from doinfo record.  This function assumes
           that they init, limit, and step expressions have already been cast to
           the type of the do index variable.
 */
int
do_begin(DOINFO *doinfo)
{
  int iv;
  int ast, dovar;

  iv = doinfo->index_var;
  doinfo->prev_dovar = DOVARG(iv);
  DOCHK(iv);
  DOVARP(iv, 1);
  ast = mk_stmt(A_DO, 0 /* SST_ASTG(RHS(1)) BLOCKDO */);
  dovar = mk_id(iv);
  A_DOVARP(ast, dovar);
  A_M1P(ast, doinfo->init_expr);
  A_M2P(ast, doinfo->limit_expr);
  A_M3P(ast, doinfo->step_expr);
  A_LASTVALP(ast, 0);

  return ast;
}

/*
 * Compute the last value of a DO index variable.
 */
static int tempify_ast(int);

void
do_lastval(DOINFO *doinfo)
{
  int dtype, sptr;
  int e1, e2, e3;
  int ast, dest_ast;

/* for a simd loop, lastval_var is not used.
 * we need to calculate the last iteration in the
 * compiler.
 */
  doinfo->lastval_var = 0;
  if (!sem.expect_simd_do) {
    sptr = get_itemp(DT_INT);
    ast = astb.i0;
    ADDRTKNP(sptr, 1);
    doinfo->lastval_var = sptr;
    dest_ast = mk_id(sptr);
    ast = mk_assn_stmt(dest_ast, ast, A_DTYPEG(ast));
    (void)add_stmt(ast);
    return;
  }

  dtype = DTYPEG(doinfo->index_var);
  /*
   * A do expression containing a function needs to be assigned to a temp
   * since we're creating multiple uses (here in and in the DO itself),
   * of a do expression.
   */
  e1 = doinfo->init_expr;
  if (A_CALLFGG(e1)) {
    e1 = tempify_ast(e1);
    e1 = doinfo->init_expr = A_DESTG(e1);
  }
  e2 = doinfo->limit_expr;
  if (A_CALLFGG(e2)) {
    e2 = tempify_ast(e2);
    e2 = doinfo->limit_expr = A_DESTG(e2);
  }
  e3 = doinfo->step_expr;
  if (A_CALLFGG(e3)) {
    e3 = tempify_ast(e3);
    e3 = doinfo->step_expr = A_DESTG(e3);
  }

  /* lp_cnt = (e2 - e1 + e3) / e3 */
  ast = mk_binop(OP_SUB, e2, e1, dtype);
  ast = mk_binop(OP_ADD, ast, e3, dtype);
  ast = mk_binop(OP_DIV, ast, e3, dtype);

  /* lastval = lp_cnt*e3 + e1 */
  ast = mk_binop(OP_MUL, ast, e3, dtype);
  ast = mk_binop(OP_ADD, ast, e1, dtype);
  doinfo->lastval_var = get_itemp(dtype);
  dest_ast = mk_id(doinfo->lastval_var);
  ast = mk_assn_stmt(dest_ast, ast, dtype);
  (void)add_stmt(ast);
}

/*
 *  allocate a temporary, assign it the value, and return the assignment
 *  ast
 */
static int
tempify_ast(int src)
{
  int argtyp;
  int tmpsym;
  int ast;

  argtyp = A_DTYPEG(src);
  tmpsym = get_temp(argtyp);
  ast = mk_id(tmpsym);
  ast = mk_assn_stmt(ast, src, argtyp);
  (void)add_stmt(ast);
  return ast;
}

static void
add_taskloopreg(DOINFO *doinfo)
{
  int ast;

  ast = mk_stmt(A_MP_TASKLOOPREG, 0);
  A_M1P(ast, doinfo->init_expr);
  A_M2P(ast, doinfo->limit_expr);
  A_M3P(ast, doinfo->step_expr);
  (void)add_stmt(ast);
}

int
do_parbegin(DOINFO *doinfo)
{
  int iv;
  int ast, dovar;

  iv = doinfo->index_var;
  if (!DT_ISINT(DTYPEG(iv))) {
    error(155, 3, gbl.lineno,
          "The index variable of a parallel DO must be integer -", SYMNAME(iv));
    return do_begin(doinfo);
  }

  if (DI_ID(sem.doif_depth) == DI_TASKLOOP) {
    add_taskloopreg(doinfo);
  }

  doinfo->prev_dovar = DOVARG(iv);
  DOCHK(iv);
  DOVARP(iv, 1);

  ast = mk_stmt(A_MP_PDO, 0 /* SST_ASTG(RHS(1)) BLOCKDO */);
  dovar = mk_id(iv);
  A_DOVARP(ast, dovar);
  A_M1P(ast, doinfo->init_expr);
  A_M2P(ast, doinfo->limit_expr);
  A_M3P(ast, doinfo->step_expr);
#ifdef OMP_OFFLOAD_LLVM
  if(DI_ID(sem.doif_depth) == DI_PARDO &&
     DI_ID(sem.doif_depth-1) == DI_TARGET) {
    int targetast = DI_BTARGET(1);
    int ast_looptc = mk_stmt(A_MP_TARGETLOOPTRIPCOUNT, 0);
    A_LOOPTRIPCOUNTP(targetast, ast_looptc);
    A_DOVARP(ast_looptc, dovar);
    A_M1P(ast_looptc, doinfo->init_expr);
    A_M2P(ast_looptc, doinfo->limit_expr);
    A_M3P(ast_looptc, doinfo->step_expr);
  }
#endif
  if (DI_ID(sem.doif_depth) != DI_TASKLOOP) {
    A_CHUNKP(ast, DI_CHUNK(sem.doif_depth));
    A_DISTCHUNKP(ast, DI_DISTCHUNK(sem.doif_depth)); /* currently unused */
    A_SCHED_TYPEP(ast, DI_SCHED_TYPE(sem.doif_depth));
    A_ORDEREDP(ast, DI_IS_ORDERED(sem.doif_depth));
  } else {
    A_CHUNKP(ast, 0);
    A_DISTCHUNKP(ast, 0);
    A_SCHED_TYPEP(ast, 0);
    A_ORDEREDP(ast, 0);
  }
  if (doinfo->lastval_var) {
    int lv_ast = mk_id(doinfo->lastval_var);
    A_LASTVALP(ast, lv_ast);
  } else {
    A_LASTVALP(ast, 0);
  }
  A_ENDLABP(ast, 0);

  /* set distribute loop flag */
  A_DISTRIBUTEP(ast, 0);
  A_DISTPARDOP(ast, 0);

  if (DI_ID(sem.doif_depth) == DI_TASKLOOP) {
    A_TASKLOOPP(ast, 1);
  } else {
    A_TASKLOOPP(ast, 0);
  }

  return ast;
}

void
save_distloop_info(int lower, int upper, int stride)
{
}

void
restore_distloop_info()
{
}

int
do_simdbegin(DOINFO *doinfo)
{
  int iv;
  int ast, dovar;

  iv = doinfo->index_var;
  if (!DT_ISINT(DTYPEG(iv))) {
    error(155, 3, gbl.lineno,
          "The index variable of a simd DO must be integer -", SYMNAME(iv));
    return do_begin(doinfo);
  }
  doinfo->prev_dovar = DOVARG(iv);
  DOCHK(iv);
  DOVARP(iv, 1);
  ast = mk_stmt(A_DO, 0 /* SST_ASTG(RHS(1)) BLOCKDO */);
  dovar = mk_id(iv);
  A_DOVARP(ast, dovar);
  A_M1P(ast, doinfo->init_expr);
  A_M2P(ast, doinfo->limit_expr);
  A_M3P(ast, doinfo->step_expr);
  if (doinfo->lastval_var) {
    A_LASTVALP(ast, mk_id(doinfo->lastval_var));
  } else {
    A_LASTVALP(ast, 0);
  }
  A_ENDLABP(ast, 0);
  A_DISTRIBUTEP(ast, 0);
  A_CHUNKP(ast, 0);
  A_DISTCHUNKP(ast, 0); /* currently unused */
  A_SCHED_TYPEP(ast, 0);
  A_ORDEREDP(ast, 0);
  A_DISTPARDOP(ast, 0);
  A_TASKLOOPP(ast, 0);

  return ast;
}

/*
 * collapse structure where various information is collected when the
 * omp collapse clause is present.
 */
static struct {
  int itemp;
  int doif_depth; /* doif of the PARDO/PDO specifying COLLAPSE */
  int dtype;      /* dtype of the new index, loop cnt & other temps */
  int index_var;
  int lp_cnt;
  int quo_var;
  int rem_var;
  int tmp_var;
} coll_st;

static int get_collapse_temp(int, const char *);
static int collapse_expr(int, int, const char *);
static void collapse_index(DOINFO *);

/** \brief Begin processing loop collapse.

    Example: the use of the collapse is for 3 loops.
    <pre>
    !$omp ... collapse(3)
        do i1 = in1, l1, s1
          do i2 = in2, l2, s2
            do i3 = in3, l3, s3
            ... SS ...
    </pre>

    The 3 loops are collapsed into a single loop with a new index variable and
    loop count. The new loop defines the iteration space for which the other
    omp clauses are applied; the new loop will appear as:
    <pre>
        n1 = (l1 - in1 + s1)/s1
        n2 = (l2 - in2 + s2)/s2
        n3 = (l3 - in3 + s3)/s3
        nn = n1*n2*n3  !! the product of the loop counts
    !$omp ...
        do ii = 1, nn
            t  = ii-1
            q  = t / n3
            r  = t - q*n3
            i3 = in3 + r*s3

            t  = q
            q  = t / n2
            r  = t - q*n2
            i2 = in2 + r*s2

            t  = q
            q  = t / n1
            r  = t - q*n1
            i2 = in1 + r*s1

            ... SS ...
    </pre>

    Basically, the original index variables are no longer iterated; their
    values are computed as a function of the new index variable and the
    corresponding loops' init, stride, and loop count.

    Prefix of temps created for each loop:
    <pre>
        .Xa - lower bound
        .Xb - stride
        .Xc - loop count
    </pre>
    Collapsed loop:
    <pre>
        .Xd - loop count
        .id - index variable
        .Xe - quotient  of id/loopcnt
        .Xf - remainder of id/loopcnt
        .Xg = temp var
    </pre>
 */
int
collapse_begin(DOINFO *doinfo)
{
  int dtype;
  SST tsst;
  int ast;
  int count_var;

  dtype = DTYPEG(doinfo->index_var);
  if (!DT_ISINT(dtype)) {
    error(155, 3, gbl.lineno,
          "The index variable of a parallel DO must be integer -",
          SYMNAME(doinfo->index_var));
    doinfo->collapse = sem.collapse = sem.collapse_depth = 0;
    ast = do_begin(doinfo);
    DI_DOINFO(sem.doif_depth) = 0; /* remove any chunk info */
    return ast;
  }
  coll_st.doif_depth = sem.doif_depth;

  if (dtype != DT_INT8) /* change type if LOG, SINT, etc.*/
    dtype = DT_INT;     /* see ensuing getccsym() call */
                        /*
                         * if the step expression is not a constant, a temporary variable
                         * must be allocated to hold the value for the do-end.
                         */
  doinfo->step_expr = collapse_expr(doinfo->step_expr, dtype, "Xb");
  /*
   * Same with the init expr.
   */
  doinfo->init_expr = collapse_expr(doinfo->init_expr, dtype, "Xa");
  /*
   *  lp_cnt <-- (e2 - e1 + e3) / e3
   */
  ast = mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr, dtype);
  ast = mk_binop(OP_ADD, ast, doinfo->step_expr, dtype);
  ast = mk_binop(OP_DIV, ast, doinfo->step_expr, dtype);
  SST_IDP(&tsst, S_EXPR);
  SST_ASTP(&tsst, ast);
  SST_DTYPEP(&tsst, dtype);
  chktyp(&tsst, DT_INT8, FALSE);

  count_var = get_collapse_temp(DT_INT8, "Xc");
  doinfo->count = mk_id(count_var);

  /* add store of loop count */
  ast = SST_ASTG(&tsst);
  ast = mk_assn_stmt(doinfo->count, ast, DT_INT8);
  (void)add_stmt(ast);

  coll_st.dtype = DT_INT8;
  coll_st.lp_cnt = get_collapse_temp(coll_st.dtype, "Xd");
  coll_st.index_var = get_collapse_temp(coll_st.dtype, "id");
  coll_st.quo_var = get_collapse_temp(coll_st.dtype, "Xe");
  coll_st.rem_var = get_collapse_temp(coll_st.dtype, "Xf");
  coll_st.tmp_var = get_collapse_temp(coll_st.dtype, "Xg");
  ENCLFUNCP(count_var, BLK_SYM(sem.scope_level));  
  ENCLFUNCP(coll_st.lp_cnt, BLK_SYM(sem.scope_level));  
  ENCLFUNCP(coll_st.index_var, BLK_SYM(sem.scope_level));  
  ENCLFUNCP(coll_st.quo_var, BLK_SYM(sem.scope_level));  
  ENCLFUNCP(coll_st.rem_var, BLK_SYM(sem.scope_level));  
  ENCLFUNCP(coll_st.tmp_var, BLK_SYM(sem.scope_level));  
  /*
   * initialize the new loop count as the loop count of the first loop.
   */
  SST_IDP(&tsst, S_IDENT);
  SST_SYMP(&tsst, count_var);
  SST_DTYPEP(&tsst, DT_INT8);
  chktyp(&tsst, coll_st.dtype, FALSE);
  mkexpr1(&tsst);
  ast = SST_ASTG(&tsst);
  ast = mk_assn_stmt(mk_id(coll_st.lp_cnt), ast, coll_st.dtype);
  (void)add_stmt(ast);
  coll_st.itemp++;
  sem.collapse_depth--;

  return 0;
}

/** \brief Process an ensuing loop which is being collapsed.
 */
int
collapse_add(DOINFO *doinfo)
{
  int dtype;
  SST tsst;
  int ast, dest_ast, std;
  int count_var;

  dtype = DTYPEG(doinfo->index_var);
  if (DT_ISINT(dtype) && dtype != DT_INT8) /* change type if LOG, SINT, etc.*/
    dtype = DT_INT;                        /* see ensuing getccsym() call */
                                           /*
                                            * if the step expression is not a constant, a temporary variable
                                            * must be allocated to hold the value for the do-end.
                                            */
  doinfo->step_expr = collapse_expr(doinfo->step_expr, dtype, "Xb");
  /*
   * Same with the init expr.
   */
  doinfo->init_expr = collapse_expr(doinfo->init_expr, dtype, "Xa");
  /*
   *  lp_cnt <-- (e2 - e1 + e3) / e3
   */
  ast = mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr, dtype);
  ast = mk_binop(OP_ADD, ast, doinfo->step_expr, dtype);
  ast = mk_binop(OP_DIV, ast, doinfo->step_expr, dtype);
  SST_IDP(&tsst, S_EXPR);
  SST_ASTP(&tsst, ast);
  SST_DTYPEP(&tsst, dtype);

  chktyp(&tsst, DT_INT8, FALSE);
  ast = SST_ASTG(&tsst);

  count_var = get_collapse_temp(DT_INT8, "Xc");
  ENCLFUNCP(count_var, BLK_SYM(sem.scope_level));  
  doinfo->count = mk_id(count_var);
  coll_st.itemp++;

  /* add store of loop count */
  ast = SST_ASTG(&tsst);
  ast = mk_assn_stmt(doinfo->count, ast, DT_INT8);
  (void)add_stmt(ast);

  /*
   * update the new loop count by multiplying the loop count of the
   * current loop.
   */
  SST_IDP(&tsst, S_IDENT);
  SST_SYMP(&tsst, count_var);
  SST_DTYPEP(&tsst, DT_INT8);
  chktyp(&tsst, coll_st.dtype, FALSE);
  mkexpr1(&tsst);
  ast = SST_ASTG(&tsst);
  dest_ast = mk_id(coll_st.lp_cnt);
  ast = mk_binop(OP_MUL, dest_ast, ast, coll_st.dtype);
  ast = mk_assn_stmt(dest_ast, ast, coll_st.dtype);
  (void)add_stmt(ast);

  if (doinfo->collapse == 1) {
    DOINFO *dinf;
    int sv;
    int i;
    /*
     * The last loop to be collapsed is now processed. Create the new
     * new loop and pass to do_parbegin() which will apply the remaining
     * omp clauses.
     */
    dinf = get_doinfo(1);
    dinf->index_var = coll_st.index_var;
    dinf->prev_dovar = 0;
    if (coll_st.dtype != DT_INT8)
      dinf->init_expr = dinf->step_expr = astb.i1;
    else
      dinf->init_expr = dinf->step_expr = astb.k1;
    dinf->limit_expr = mk_id(coll_st.lp_cnt);
    do_lastval(dinf);
    sv = sem.doif_depth;
    /*
     * DI_DOINFO(coll_st.doif_depth) locates the DOINFO record for
     * the PARDO/PDO; DI_DOINFO(coll_st.doif_depth+1) is the DOINFO
     * for its corresponding DO.
     */
    sem.doif_depth = coll_st.doif_depth;
    if (DI_ID(sem.doif_depth) == DI_SIMD)
      ast = do_simdbegin(dinf);
    else
      ast = do_parbegin(dinf);
    std = add_stmt(ast);
    sem.doif_depth = sv;
    if (DI_ID(sv) == DI_DOCONCURRENT)
      STD_BLKSYM(std) = DI_CONC_BLOCK_SYM(sv);
    /*
     * Compute the values for index variables in the collapsed do loops in
     * the order from inner to outer.
     * DI_DOINFO(sem.doif_depth) locates the DOINFO record for loop
     * immediately enclosing the current loop.
     */
    collapse_index(doinfo); /* innermost first */
    for (i = sem.doif_depth; TRUE; i--) {
      DOINFO *dd;
      dd = DI_DOINFO(i);
      collapse_index(dd);
      if (dd->collapse == sem.collapse)
        break;
    }

    DI_DOINFO(coll_st.doif_depth + 1) = dinf;
  }
  sem.collapse_depth--;

  return 0;
}

static int
get_collapse_temp(int dtype, const char *pfx)
{
  int sptr;
  sptr = getccssym_sc(pfx, coll_st.itemp, ST_VAR, sem.sc);
  DTYPEP(sptr, dtype);
  return sptr;
}

static int
collapse_expr(int ast, int dtype, const char *pfx)
{
  int sptr, dest_ast;
  if (A_ALIASG(ast))
    return ast;
  sptr = getccssym_sc(pfx, coll_st.itemp, ST_VAR, sem.sc);
  DTYPEP(sptr, dtype);
  dest_ast = mk_id(sptr);
  ast = mk_assn_stmt(dest_ast, ast, dtype);
  (void)add_stmt(ast);
  return dest_ast;
}

/*
 * Compute the values of the index variables of the collapsed DO loops.
 * The index variables will be computed in the order of inner to
 * outer.
 */
static void
collapse_index(DOINFO *dd)
{
  int dt_index;
  int q, r, cnt;
  int qpr, tmp;
  SST tsst;

  dt_index = DTYPEG(dd->index_var);
  if (dd->collapse == 1) {
    /*
     * initialize for a new set of collapsed loops; compute
     *   qpr <-- (id-1) / cnt
     */
    qpr = mk_id(coll_st.index_var);
    if (coll_st.dtype != DT_INT8)
      qpr = mk_binop(OP_SUB, qpr, astb.i1, coll_st.dtype);
    else
      qpr = mk_binop(OP_SUB, qpr, astb.k1, coll_st.dtype);
    qpr = mk_assn_stmt(mk_id(coll_st.tmp_var), qpr, coll_st.dtype);
    (void)add_stmt(qpr);
  }
  /*
   * Compute
   *     q <-- qpr / cnt
   */
  qpr = mk_id(coll_st.tmp_var);
  SST_IDP(&tsst, S_IDENT);
  SST_SYMP(&tsst, A_SPTRG(dd->count));
  SST_DTYPEP(&tsst, dt_index);
  chktyp(&tsst, coll_st.dtype, FALSE);
  mkexpr1(&tsst);
  cnt = SST_ASTG(&tsst);
  tmp = mk_binop(OP_DIV, qpr, cnt, coll_st.dtype);
  q = mk_id(coll_st.quo_var);
  tmp = mk_assn_stmt(q, tmp, coll_st.dtype);
  (void)add_stmt(tmp);
  /*
   * Compute
   *     r <-- qpr - q * cnt
   */
  tmp = mk_binop(OP_MUL, q, cnt, coll_st.dtype);
  tmp = mk_binop(OP_SUB, qpr, tmp, coll_st.dtype);
  r = mk_id(coll_st.rem_var);
  tmp = mk_assn_stmt(r, tmp, coll_st.dtype);
  (void)add_stmt(tmp);
  /*
   * Compute
   *    i <-- init + r*step
   */
  SST_IDP(&tsst, S_IDENT);
  SST_SYMP(&tsst, coll_st.rem_var);
  SST_DTYPEP(&tsst, coll_st.dtype);
  chktyp(&tsst, dt_index, FALSE);
  mkexpr1(&tsst);
  r = SST_ASTG(&tsst);
  tmp = mk_binop(OP_MUL, r, dd->step_expr, dt_index);
  tmp = mk_binop(OP_ADD, tmp, dd->init_expr, dt_index);
  tmp = mk_assn_stmt(mk_id(dd->index_var), tmp, dt_index);
  (void)add_stmt(tmp);
  /*
   * Compute, iff not the last index variable
   *     qpr <-- q
   */
  if (dd->collapse != sem.collapse) {
    tmp = mk_assn_stmt(qpr, q, coll_st.dtype);
    (void)add_stmt(tmp);
  }
}

void
do_end(DOINFO *doinfo)
{
  int ast, i, orig_doif, par_doif, std, symi, astlab;
  SPTR lab, sptr;

  orig_doif = sem.doif_depth; // original loop index

  // Close do concurrent mask.
  // Don't emit scn.currlab here.  (Don't use add_stmt.)
  if (DI_ID(orig_doif) == DI_DOCONCURRENT && DI_CONC_MASK_STD(orig_doif))
    (void)add_stmt_after(mk_stmt(A_ENDIF, 0), STD_LAST);

  // Loop body is done; emit loop cycle label.
  // Don't emit scn.currlab here.  (Don't use add_stmt.)
  if (DI_CYCLE_LABEL(orig_doif)) {
    std = add_stmt_after(mk_stmt(A_CONTINUE, 0), STD_LAST);
    STD_LABEL(std) = DI_CYCLE_LABEL(orig_doif);
    DEFDP(DI_CYCLE_LABEL(orig_doif), 1);
  }

  // Finish do concurrent inner loop processing and move to the outermost loop.
  if (DI_ID(orig_doif) == DI_DOCONCURRENT) {
    check_doconcurrent(orig_doif); // innermost loop has constraint check info
    std = add_stmt_after(mk_stmt(A_CONTINUE, 0), STD_LAST);
    STD_LINENO(std) = gbl.lineno;
    STD_LABEL(std) = lab = getlab();
    RFCNTI(lab);
    VOLP(lab, true);
    ENDLINEP(sem.construct_sptr, gbl.lineno);
    ENDLABP(sem.construct_sptr, lab);
    LABSTDP(lab, std);
    for (i = DI_CONC_COUNT(orig_doif), symi = DI_CONC_SYMS(orig_doif); i;
         --i, symi = SYMI_NEXT(symi)) {
      sptr = SYMI_SPTR(symi);
      pop_sym(sptr); // do concurrent index construct var
    }
    for (++sptr; sptr < stb.stg_avail; ++sptr)
      switch (STYPEG(sptr)) {
      default:
        break;
      case ST_UNKNOWN:
      case ST_IDENT:
      case ST_VAR:
      case ST_ARRAY:
        if (SAVEG(sptr))
          break;
        if (!CCSYMG(sptr) && !HCCSYMG(sptr))
          DCLCHK(sptr);
        pop_sym(sptr); // do concurrent non-index construct var
        if (ENCLFUNCG(sptr) == 0)
          ENCLFUNCP(sptr, sem.construct_sptr);
      }
    for (; DI_CONC_COUNT(orig_doif) > 1; --orig_doif)
      if (!DI_DOINFO(orig_doif)->collapse) {
        std = add_stmt(mk_stmt(A_ENDDO, 0));
        STD_BLKSYM(std) = sem.construct_sptr;
      }
    doinfo = DI_DOINFO(orig_doif);
    sem.doif_depth = orig_doif;
  }

  if (doinfo->index_var)
    /*
     * If there is an index variable, set its DOVAR flag to its 'state'
     * before entering the DO which is about to be popped.
     */
    DOVARP(doinfo->index_var, doinfo->prev_dovar);

  par_doif = orig_doif - 1; // parallel loop index (if it exists)

  switch (DI_ID(par_doif)) {
  case DI_PDO:
    (void)add_stmt(mk_stmt(A_MP_ENDPDO, 0));
    if (scn.currlab && scn.stmtyp != TK_ENDDO)
      (void)add_stmt(mk_stmt(A_MP_BARRIER, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    par_pop_scope();
    sem.collapse = 0;
    break;

  case DI_TASKLOOP:
    ast = mk_stmt(A_MP_ENDPDO, 0);
    A_TASKLOOPP(ast, 1);
    (void)add_stmt(ast);
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    --sem.task;
    par_pop_scope();
    add_stmt(mk_stmt(A_MP_ETASKLOOPREG, 0));
    ast = mk_stmt(A_MP_ETASKLOOP, 0);
    A_LOPP(DI_BEGINP(par_doif), ast);
    A_LOPP(ast, DI_BEGINP(par_doif));
    add_stmt(ast);
    if (sem.task < 0)
      sem.task = 0;
    mp_create_escope();
    sem.collapse = 0;
    break;

  case DI_DOACROSS:
  case DI_PARDO:
    /* For DOACROSS & PARALLEL DO, need to end the parallel section. */
    (void)add_stmt(mk_stmt(A_MP_ENDPDO, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    A_LOPP(DI_BPAR(par_doif), ast);
    A_LOPP(ast, DI_BPAR(par_doif));
    mp_create_escope();
    sem.collapse = 0;
    break;

  case DI_TEAMSDIST:
  case DI_TARGTEAMSDIST:
  case DI_DISTRIBUTE:
    (void)add_stmt(mk_stmt(A_MP_ENDPDO, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    par_pop_scope();
    ast = mk_stmt(A_MP_ENDDISTRIBUTE, 0);
    A_LOPP(DI_BDISTRIBUTE(par_doif), ast);
    A_LOPP(ast, DI_BDISTRIBUTE(par_doif));
    (void)add_stmt(ast);
    sem.collapse = 0;
    break;

  case DI_TEAMSDISTPARDO:
  case DI_TARGTEAMSDISTPARDO:
  case DI_DISTPARDO:
    (void)add_stmt(mk_stmt(A_MP_ENDPDO, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;

    /* We create 2 scopes for distributed loop so that
     * lastprivate(dovar) is not the same as dovar for
     * distributed loop, therefore we need to double pop
     * one for do scope and another is for lastprivate
     * which is DISTPARDO scope.
     */

    par_pop_scope();
    par_pop_scope();
    ast = mk_stmt(A_MP_ENDDISTRIBUTE, 0);
    A_LOPP(DI_BDISTRIBUTE(par_doif), ast);
    A_LOPP(ast, DI_BDISTRIBUTE(par_doif));
    (void)add_stmt(ast);
    sem.collapse = 0;
    break;

  case DI_TARGPARDO:
    (void)add_stmt(mk_stmt(A_MP_ENDPDO, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    --sem.parallel;
    par_pop_scope();
    ast = emit_epar();
    A_LOPP(DI_BPAR(par_doif), ast);
    A_LOPP(ast, DI_BPAR(par_doif));
    mp_create_escope();
    sem.collapse = 0;
    end_parallel_clause(orig_doif);
    sem.doif_depth--; /* leave_dir(DI_TARGPARDO, .. */
    par_doif--;
    sem.target--;
    par_pop_scope();
    ast = emit_etarget();
    mp_create_escope();
    A_LOPP(DI_BTARGET(par_doif), ast);
    A_LOPP(ast, DI_BTARGET(par_doif));
    sem.collapse = 0;
    break;

  case DI_SIMD:
    /* Standalone simd construct and target simd too? */
    (void)add_stmt(mk_stmt(A_ENDDO, 0));
    end_parallel_clause(par_doif);
    sem.close_pdo = TRUE;
    par_pop_scope();
    sem.collapse = 0;
    break;

  case DI_ACCDO:
  case DI_ACCLOOP:
  case DI_ACCREGDO:
  case DI_ACCREGLOOP:
  case DI_ACCKERNELSDO:
  case DI_ACCKERNELSLOOP:
  case DI_ACCPARALLELDO:
  case DI_ACCPARALLELLOOP:
  case DI_ACCSERIALLOOP:
  case DI_CUFKERNEL:
    (void)add_stmt(mk_stmt(A_ENDDO, 0));
    sem.close_pdo = TRUE;
    /* Pop the inserted new symbol for the induction var*/
    if (flg.smp && (SCG(doinfo->index_var) != SC_PRIVATE)) {
      if (DI_DO_POPINDEX(sem.doif_depth) > SPTR_NULL)
        pop_sym(DI_DO_POPINDEX(sem.doif_depth));
    }
    break;

  default:
    // No parallel loop; process the original loop.
    if (doinfo->collapse > 0)
      // This is an intermediate loop in a collapsed loop nest.
      break;

    switch (DI_ID(orig_doif)) {
    default:
      break;
    case DI_DO:
      (void)add_stmt(mk_stmt(A_ENDDO, 0));
      break;
    case DI_DOCONCURRENT:
      std = add_stmt(mk_stmt(A_ENDDO, 0));
      STD_BLKSYM(std) = sem.construct_sptr;

      sem.construct_sptr = ENCLFUNCG(sem.construct_sptr);
      if (STYPEG(sem.construct_sptr) != ST_BLOCK)
        sem.construct_sptr = 0; // not in a construct

      break;
    case DI_DOWHILE:
      ast = mk_stmt(A_GOTO, 0);
      // Do not place mk_label inside A_L1P(ast, mk_label(...))
      // due to undefined behavior of C compiler for evaluation order
      // between the calculation of the address of the target of an
      // assignment and the computation of the value being assigned.
      astlab = mk_label(DI_TOP_LABEL(orig_doif));
      A_L1P(ast, astlab);
      RFCNTI(DI_TOP_LABEL(orig_doif));
      (void)add_stmt(ast);
      (void)add_stmt(mk_stmt(A_ENDIF, 0));
      break;
    }
  }

  // Loop code is done; emit loop exit label.
  if (DI_EXIT_LABEL(orig_doif)) {
    std = add_stmt(mk_stmt(A_CONTINUE, 0));
    STD_LABEL(std) = DI_EXIT_LABEL(orig_doif);
    DEFDP(DI_EXIT_LABEL(orig_doif), 1);
  }

  --sem.doif_depth;
}

DOINFO *
get_doinfo(int area)
{
  DOINFO *doinfo;
  doinfo = (DOINFO *)getitem(area, sizeof(DOINFO));
  doinfo->collapse = 0;
  doinfo->distloop = 0;
  return doinfo;
}

/**
    \param structd dtype record of parent structure
    \param base    ast ptr of parent structure
    \param nmx     index into "names" area of member
    \return ast or 0 if not found
 */
int
mkmember(int structd, int base, int nmx)
{
  int sptr; /* next member of structure to search */
  int dtype;
  for (sptr = DTY(structd + 1); sptr > NOSYM; sptr = SYMLKG(sptr)) {
    dtype = DTYPEG(sptr);
    /*
     * special case:  if member is a union, then we must look at
     * all maps which belong to the union; recall that each map is
     * just a struct.
     */
    if (DTY(dtype) == TY_UNION) {
      int ast;
      ast = mkunion(dtype, base, nmx);
      if (ast)
        return (ast);
    } else if (NMPTRG(sptr) == nmx) {
      int ast, member;
      if (flg.xref)
        xrefput(sptr, 'r');
      member = mk_id(sptr);
      ast = mk_member(base, mk_id(sptr), dtype);
      return ast;
    } else if (PARENTG(sptr)) { /* type extension */
      int ast = mkmember(DTYPEG(sptr), base, nmx);
      if (ast)
        return ast;
    }
  }
  return 0; /* not found */
}

/**
    \param uniond dtype record of parent structure
    \param base   ast ptr of parent structure
    \param nmx    index into "names" area of member
    \return ast or 0 if not found
 */
static int
mkunion(int uniond, int base, int nmx)
{
  int sptr; /* next member of structure to search */
  int dtype;
  int ast;
  /*
   * scan the MAPs (each "member" is a struct and represents
   * one map)
   */
  for (sptr = DTY(uniond + 1); sptr != NOSYM; sptr = SYMLKG(sptr)) {
    dtype = DTYPEG(sptr);
#if DEBUG
    assert(DTY(dtype) == TY_STRUCT, "mkunion, dt not struct", sptr, 3);
#endif
    /*  look at all members of the map (a struct)  */
    ast = mkmember(dtype, base, nmx);
    if (ast)
      return ast;
  }
  return 0; /* not found */
}

/** \brief Given an ast which computes the address of the label variable or
           loads the label variable, create the variable of indicated dtype.
 */
int
mklabelvar(SST *stkptr)
{
  int ast;
  int sptr;
  int dtype;

  mkexpr(stkptr);
  ast = SST_ASTG(stkptr);
#if DEBUG
  if (A_TYPEG(ast) != A_ID) {
    interr("mklabelvar: ast not id", ast, 3);
    return 0;
  }
#endif
  sptr = A_SPTRG(ast);
  /*
   * When targeting llvm, always create a temp variable of ptr-size
   * integer type.
   */
  if (XBIT(49, 0x100))
    dtype = DT_INT8;
  else
    dtype = DT_INT4;
  sptr = getcctmp_sc('l', sptr, ST_VAR, dtype, sem.sc);
  SST_DTYPEP(stkptr, DTYPEG(sptr));
  SST_ASTP(stkptr, mk_id(sptr));
  return sptr;
}

LOGICAL
legal_labelvar(int dtype)
{
  if (dtype == stb.user.dt_int)
    return TRUE;
  if (dtype == DT_INT4 || dtype == DT_INT8)
    return TRUE;
  return FALSE;
}

static INT
_xtok(INT conval1, BIGINT64 count, int dtype)
{
  INT conval;
  INT one;
  int isneg;
#ifdef TARGET_SUPPORTS_QUADFP
  IEEE128 qtemp, qnum1, qresult;
  IEEE128 qreal1, qrealrs, qimag1, qimagrs;
  IEEE128 qrealpv, qtemp1;
#endif
  DBLE dtemp, dresult, num1;
  DBLE dreal1, drealrs, dimag1, dimagrs;
  DBLE drealpv, dtemp1;
  SNGL temp;
  SNGL real1, realrs, imag1, imagrs;
  SNGL realpv, temp1;
  DBLINT64 inum1, ires;
  int overr;
  UINT uval, uoldval;

  overr = 0;
  isneg = 0;
  if (count < 0) {
    isneg = 1;
    count = -count;
  }
  one = 1;
  if (dtype != DT_INT4)
    one = cngcon(one, DT_INT4, dtype);
  switch (DTY(dtype)) {
  case TY_WORD:
  case TY_DWORD:
    error(33, 3, gbl.lineno, " ", CNULL);
    return (0);

  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    uval = 1;
    {
      int do_neg;
      int sg;
      sg = 0;
      do_neg = 0;
      if (conval1 < 0) {
        do_neg = 1;
        conval1 = -conval1;
      }
      uoldval = conval1;
      while (count--) {
        sg ^= 1;
        uval = uval * conval1;
        if (!sem.which_pass && !overr && uval < uoldval) {
          /*
           * generally, warnings are inhibited during the 2nd parse
           */
          overr = 1;
        }
        uoldval = uval;
      }
      conval = *((INT *)&uval);
      if (do_neg) {
        conval1 = -conval1;
        if (sg)
          conval = -conval;
      } else if (conval & 0x80000000)
        overr = 1;
      if (overr) {
        error(155, 2, gbl.lineno, "Integer overflow occurred when evaluating",
              "**");
      }
    }
    break;

  case TY_INT8:
    inum1[0] = CONVAL1G(conval1);
    inum1[1] = CONVAL2G(conval1);
    ires[0] = CONVAL1G(stb.k1);
    ires[1] = CONVAL2G(stb.k1);
    while (count--)
      mul64(inum1, ires, ires);
    conval = getcon(ires, DT_INT8);
    break;

  case TY_REAL:
    conval = CONVAL2G(stb.flt1);
    while (count--)
      xfmul(conval1, conval, &conval);
    break;

  case TY_DBLE:
    num1[0] = CONVAL1G(conval1);
    num1[1] = CONVAL2G(conval1);
    dresult[0] = CONVAL1G(stb.dbl1);
    dresult[1] = CONVAL2G(stb.dbl1);
    while (count--)
      xdmul(num1, dresult, dresult);
    conval = getcon(dresult, DT_REAL8);
    break;

  case TY_CMPLX:
    real1 = CONVAL1G(conval1);
    imag1 = CONVAL2G(conval1);
    realrs = CONVAL1G(one);
    imagrs = CONVAL2G(one);
    while (count--) {
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      realpv = realrs;
      xfmul(real1, realrs, &temp1);
      xfmul(imag1, imagrs, &temp);
      xfsub(temp1, temp, &realrs);
      xfmul(real1, imagrs, &temp1);
      xfmul(realpv, imag1, &temp);
      xfadd(temp1, temp, &imagrs);
    }
    num1[0] = realrs;
    num1[1] = imagrs;
    conval = getcon(num1, DT_CMPLX8);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    qnum1[0] = CONVAL1G(conval1);
    qnum1[1] = CONVAL2G(conval1);
    qnum1[2] = CONVAL3G(conval1);
    qnum1[3] = CONVAL4G(conval1);
    qresult[0] = CONVAL1G(stb.quad1);
    qresult[1] = CONVAL2G(stb.quad1);
    qresult[2] = CONVAL3G(stb.quad1);
    qresult[3] = CONVAL4G(stb.quad1);
    while (count--)
      xqmul(qnum1, qresult, qresult);
    conval = getcon(qresult, DT_QUAD);
    break;
#endif

  case TY_DCMPLX:
    dreal1[0] = CONVAL1G(CONVAL1G(conval1));
    dreal1[1] = CONVAL2G(CONVAL1G(conval1));
    dimag1[0] = CONVAL1G(CONVAL2G(conval1));
    dimag1[1] = CONVAL2G(CONVAL2G(conval1));
    drealrs[0] = CONVAL1G(CONVAL1G(one));
    drealrs[1] = CONVAL2G(CONVAL1G(one));
    dimagrs[0] = CONVAL1G(CONVAL2G(one));
    dimagrs[1] = CONVAL2G(CONVAL2G(one));
    while (count--) {
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      drealpv[0] = drealrs[0];
      drealpv[1] = drealrs[1];
      xdmul(dreal1, drealrs, dtemp1);
      xdmul(dimag1, dimagrs, dtemp);
      xdsub(dtemp1, dtemp, drealrs);
      xdmul(dreal1, dimagrs, dtemp1);
      xdmul(drealpv, dimag1, dtemp);
      xdadd(dtemp1, dtemp, dimagrs);
    }
    num1[0] = getcon(drealrs, DT_REAL8);
    num1[1] = getcon(dimagrs, DT_REAL8);
    conval = getcon(num1, DT_CMPLX16);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    qreal1[0] = CONVAL1G(CONVAL1G(conval1));
    qreal1[1] = CONVAL2G(CONVAL1G(conval1));
    qreal1[2] = CONVAL3G(CONVAL1G(conval1));
    qreal1[3] = CONVAL4G(CONVAL1G(conval1));
    qimag1[0] = CONVAL1G(CONVAL2G(conval1));
    qimag1[1] = CONVAL2G(CONVAL2G(conval1));
    qimag1[2] = CONVAL3G(CONVAL2G(conval1));
    qimag1[3] = CONVAL4G(CONVAL2G(conval1));
    qrealrs[0] = CONVAL1G(CONVAL1G(one));
    qrealrs[1] = CONVAL2G(CONVAL1G(one));
    qrealrs[2] = CONVAL3G(CONVAL1G(one));
    qrealrs[3] = CONVAL4G(CONVAL1G(one));
    qimagrs[0] = CONVAL1G(CONVAL2G(one));
    qimagrs[1] = CONVAL2G(CONVAL2G(one));
    qimagrs[2] = CONVAL3G(CONVAL2G(one));
    qimagrs[3] = CONVAL4G(CONVAL2G(one));
    while (count--) {
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      qrealpv[0] = qrealrs[0];
      qrealpv[1] = qrealrs[1];
      qrealpv[2] = qrealrs[2];
      qrealpv[3] = qrealrs[3];
      xqmul(qreal1, qrealrs, qtemp1);
      xqmul(qimag1, qimagrs, qtemp);
      xqsub(qtemp1, qtemp, qrealrs);
      xqmul(qreal1, qimagrs, qtemp1);
      xqmul(qrealpv, qimag1, qtemp);
      xqadd(qtemp1, qtemp, qimagrs);
    }
    num1[0] = getcon(qrealrs, DT_QUAD);
    num1[1] = getcon(qimagrs, DT_QUAD);
    conval = getcon(num1, DT_QCMPLX);
    break;
#endif

  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
  case TY_NCHAR:
  case TY_CHAR:
    errsev(91);
    return 0;
  }
  if (isneg) {
    /* exponentiation to a negative power */
    conval = const_fold(OP_DIV, one, conval, dtype);
  }

  return conval;
}

static void
error83(int ty)
{
  if (ty == TY_CHAR)
    UFCHAR;
  else
    errsev(83);
}


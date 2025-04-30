/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran data type utility functions.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "machar.h"
#include "machardf.h"
#include "ast.h"
#include "rte.h"
#include "symutl.h"

static TY_KIND get_ty_kind(DTYPE);
static LOGICAL get_kind_set_parm(int, DTYPE, int *);
static int get_len_set_parm(int, DTYPE, int *);
static DTYPE get_iso_derivedtype(DTYPE);
static DTYPE get_cuf_derivedtype(DTYPE);
static int ic_strcmp(const char *str, const char *pattern);

static int size_sym = 0;

const char *
target_name(DTYPE dtype)
{
  TY_KIND ty = get_ty_kind(dtype);
  switch (ty) {
  case TY_DCMPLX:
    if (XBIT(57, 0x200)) {
      return "complex*16";
    }
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
  case TY_CHAR:
  case TY_NCHAR:
    return dtypeinfo[ty].target_type;

  default:
    interr("target_name: bad dtype ", ty, 3);
    return "";
  }
}

int
target_kind(DTYPE dtype)
{
  TY_KIND ty = get_ty_kind(dtype);
  switch (ty) {
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
  case TY_CHAR:
  case TY_NCHAR:
    return dtypeinfo[ty].target_kind;

  default:
    interr("target_kind: bad dtype ", ty, 3);
    return 0;
  }
}

ISZ_T
size_of(DTYPE dtype)
{
  ISZ_T d, nelems, sz;
  INT clen;
  ADSC *ad;

  TY_KIND ty = get_ty_kind(dtype);
  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_PTR:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
    return dtypeinfo[ty].size;

  case TY_HOLL:
    /* treat like default integer type */
    return dtypeinfo[DTY(DT_INT)].size;

  case TY_CHAR:
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR)
      interr("size_of: attempt to get size of assumed size character", 0, 3);
    clen = string_length(dtype);
    return clen;

  case TY_NCHAR:
    if (dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR)
      interr("size_of: attempt to get size of assumed size ncharacter", 0, 3);
    clen = string_length(dtype);
    return 2 * clen;

  case TY_ARRAY:
    if ((d = DTY(dtype + 2)) <= 0) {
      interr("size_of: no array descriptor", (int)d, 3);
      return size_of((int)DTY(dtype + 1));
    }
    ad = AD_DPTR(dtype);
    if (AD_DEFER(ad)) {
      return dtypeinfo[DTY(DT_PTR)].size;
    }
    if (AD_NUMELM(ad) == 0) {
/* illegal use of adjustable or assumed-size array:
   should have been caught in semant.  */
/* errsev(50); */
      if (XBIT(68, 0x1)) {
        AD_NUMELM(ad) = astb.k1;
        d = stb.i1;
      } else {
        AD_NUMELM(ad) = astb.i1;
        d = stb.i1;
      }
    } else {
      switch (A_TYPEG(AD_NUMELM(ad))) {
      case A_BINOP:
      case A_UNOP:
        /* FS#20474: Occurs with length type parameters. Treat as
         * as if AD_DEFER(ad) is set (see case above).
         * This also avoids an ICE from call to sym_of_ast() below.
         */
        return dtypeinfo[DTY(DT_PTR)].size;
      }
      d = AD_NUMELM(ad);
      if (A_TYPEG(d) == A_INTR) {
        switch (A_OPTYPEG(d)) {
        case I_INT1:
        case I_INT2:
        case I_INT4:
        case I_INT8:
        case I_INT:
          /* FS#22205: This can occur with -mcmodel=medium */
          d = A_ARGSG(d);
          break;
        default:
          interr("size_of: unexpected intrinsic optype ", A_OPTYPEG(d), 3);
        }
      }
      d = sym_of_ast(d);
      if (d == stb.i0 || STYPEG(d) != ST_CONST) {
        /* illegal use of adjustable or assumed-size array:
           should have been caught in semant.  */
        /* errsev(50); */
        AD_NUMELM(ad) = astb.i1;
        d = stb.i1;
      }
      if (XBIT(68, 0x1) && d == stb.k0) {
        AD_NUMELM(ad) = astb.k1;
        d = stb.k1;
      }
    }
    if (XBIT(68, 0x1)) {
      INT num[2];
      num[0] = CONVAL1G(d);
      num[1] = CONVAL2G(d);
      INT64_2_ISZ(num, d);
    } else
      d = CONVAL2G(d);
    nelems = d;
    sz = size_of((int)DTY(dtype + 1));
    d = d * sz;
    if (size_sym && (d < nelems || d < sz) && nelems && sz) {
      return -1;
    }
    return d;

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    if (DTY(dtype + 1) == 0) {
      errsev(151);
      return 4;
    } else
      return DTY(dtype + 2);

  default:
    interr("size_of: bad dtype ", ty, 3);
    return 1;
  }
}

/** \brief Return length of constant char string data type */
int
string_length(DTYPE dtype)
{
  int clen;
  switch (DTY(dtype)) {
  case TY_CHAR:
  case TY_NCHAR:
    break;
  default:
    interr("string length applied to nonstring datatype", dtype, 2);
    return 1;
  }
  clen = DTY(dtype + 1); /* get length ast */
  clen = A_ALIASG(clen); /* get constant alias */
  clen = A_SPTRG(clen);  /* get constant symbol */
  clen = CONVAL2G(clen); /* get constant value */
  if (clen < 0)
    return 0;
  return clen;
}

/*
 *  A framework for recursively scanning derived types and their members
 *  with pluggable predicates follows.  The results are not cached,
 *  but probably could be.  This code is used below to replace a series
 *  of functions that used boilerplate to implement their own variant
 *  frameworks.
 */

/* Visitation lists record the datatypes that have been visited and
 * whether those visits remain active.
 */
struct visit_list {
  DTYPE dtype;
  LOGICAL is_active;
  struct visit_list *next;
};

static struct visit_list *
visit_list_scan(struct visit_list *list, DTYPE dtype)
{
  for (; list; list = list->next) {
    if (list->dtype == dtype)
      break;
  }
  return list;
}

static void
visit_list_push(struct visit_list **list, DTYPE dtype)
{
  struct visit_list *newlist;
  NEW(newlist, struct visit_list, 1);
  newlist->dtype = dtype;
  newlist->is_active = TRUE;
  newlist->next = *list;
  *list = newlist;
}

static void
visit_list_free(struct visit_list **list)
{
  struct visit_list *p;
  while ((p = *list)) {
    *list = p->next;
    FREE(p);
  }
}

static LOGICAL
is_container_dtype(DTYPE dtype)
{
  if (dtype > 0) {
    if (is_array_dtype(dtype))
      dtype = array_element_dtype(dtype);
    switch (DTYG(dtype)) {
    case TY_DERIVED:
    case TY_STRUCT:
    case TY_UNION:
      return TRUE;
    }
  }
  return FALSE;
}

/* Forward declare is_recursive() here so that search_type_members()
 * below can identify it as a special case.
 */
static LOGICAL is_recursive(int sptr, struct visit_list **visited);

typedef LOGICAL (*stm_predicate_t)(int member_sptr,
                                   struct visit_list **visited);

/* search_type_members() is a potentially recursive scanner of
 * derived types that applies a given predicate function against the
 * component members.  Returns TRUE if any application of the predicate
 * is satisfied.  The predicate function can (and usually does) indirectly
 * recursively call search_type_members() to scan the types of
 * members, but it has the option to recurse conditionally or not at all.
 */
static LOGICAL
search_type_members(DTYPE dtype, stm_predicate_t predicate,
                    struct visit_list **visited)
{
  LOGICAL result = FALSE;

  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  if (is_container_dtype(dtype)) {
    int member_sptr = DTY(dtype + 1);
    struct visit_list *active = visit_list_scan(*visited, dtype);

    if (active) {
      /* This dtype has already been scanned or is in process.
       * Cut off the scan, and return FALSE unless the search
       * is for recursive types and we've just found one.
       */
      return predicate == is_recursive && active->is_active;
    }

    visit_list_push(visited, dtype);
    active = *visited;

    /* Traverse the members of the derived type. */
    while (member_sptr > NOSYM && !(result = predicate(member_sptr, visited))) {
      member_sptr = SYMLKG(member_sptr);
    }

    /* The scan of this data type is complete. Leave it on the visited
     * list to forestall another failed pass later.
     */
    active->is_active = FALSE;
  }
  return result;
}

/* Wraps a call to search_type_members() above with the construction
 * and destruction of its visitation list.
 */
static LOGICAL
search_type_members_wrapped(DTYPE dtype, stm_predicate_t predicate)
{
  struct visit_list *visited = NULL;
  LOGICAL result = search_type_members(dtype, predicate, &visited);
  visit_list_free(&visited);
  return result;
}

/* Driver for predicates that use search_type_members(); it sets up and
 * tears down the visitation list that search_type_members() uses.
 * N.B. this function will succeed if the supplied predicate is true on
 * the initial symbol table index, even if it's not a component.
 */
static LOGICAL
test_sym_and_components(int sptr, stm_predicate_t predicate)
{
  struct visit_list *visited = NULL;
  LOGICAL result = predicate(sptr, &visited);
  visit_list_free(&visited);
  return result;
}

static LOGICAL
test_sym_components_only(int sptr, stm_predicate_t predicate)
{
  return sptr > NOSYM && search_type_members_wrapped(DTYPEG(sptr), predicate);
}

/** \brief Check for special case of empty typedef which has a size of 0
    but one member of type DT_NONE to indicate that the type is
    empty and not incomplete, a forward reference, etc.
 */
LOGICAL
is_empty_typedef(DTYPE dtype)
{
  SPTR sptr;
  if (dtype) {
    if (is_array_dtype(dtype))
      dtype = array_element_dtype(dtype);
    switch (DTY(dtype)) {
    case TY_DERIVED:
    case TY_UNION:
    case TY_STRUCT:
      for (sptr = DTY(dtype + 1); sptr > NOSYM;
           sptr = SYMLKG(sptr)) {
        /* Type parameters are not data components. Skip type parameters. */
        if (SETKINDG(sptr) || LENPARMG(sptr)) {
          continue;
        }
        return FALSE;
      }
      return TRUE;
    }
  }
  return FALSE;
}

/** \brief Check for special case of zero-size typedef which may nest have
    zero-size typedef compnents or zero-size array compnents.
 */
LOGICAL
is_zero_size_typedef(DTYPE dtype)
{
  if (dtype <= DT_NONE)
    return FALSE;
  dtype = is_array_dtype(dtype) ? DTY(dtype + 1) : dtype;

  switch (DTY(dtype)) {
  case TY_DERIVED:
  case TY_UNION:
  case TY_STRUCT:
    return (DTY(dtype + 2) == 0);
  default:
    return FALSE;
  }
  return FALSE;
}

static LOGICAL
is_recursive_dtype(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         search_type_members(sptr, is_recursive_dtype, visited);
}

/* N.B., no_data_components_recursive() will only ever be true for
 * data types that are containers, so types like INTEGER will map to FALSE.
 */
static bool
no_data_components_recursive(DTYPE dtype, stm_predicate_t predicate, struct visit_list **visited)
{
  /* For the derived type in dtype: Returns true if dtype is empty or
   * if it does not contain any data components (i.e., a derived type with
   * type bound procedures returns false). Otherwise, returns false.
   */
  int mem_sptr;
  struct visit_list *active;
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  active = visit_list_scan(*visited, dtype);
  if (is_empty_typedef(dtype))
    return TRUE;
  if (!is_container_dtype(dtype))
    return FALSE;
  if (active) {
    /* This dtype has already been scanned or is in process.
     * Cut off the scan, and return FALSE unless the search
     * is for recursive types and we've just found one.
     */
    return predicate == is_recursive_dtype && active->is_active;
  }

  visit_list_push(visited, dtype);
  active = *visited;

  for (mem_sptr = DTY(dtype + 1); mem_sptr > NOSYM;
       mem_sptr = SYMLKG(mem_sptr)) {
    if (DTYG(DTYPEG(mem_sptr)) == TY_DERIVED) {
      if (!no_data_components_recursive(DTYPEG(mem_sptr), is_recursive_dtype, visited)) {
        active->is_active = FALSE;
        return FALSE;
      }
    } else if (!CLASSG(mem_sptr) || !BINDG(mem_sptr) || !VTABLEG(mem_sptr)) {
      active->is_active = FALSE;
      return FALSE;
    }
  }
  return TRUE;
}

/* Wrapper to no_data_components_recursive() to detect cycles. */
LOGICAL
no_data_components(DTYPE dtype)
{
  struct visit_list *visited = NULL;
  LOGICAL result = no_data_components_recursive(dtype, is_recursive_dtype, &visited);
  visit_list_free(&visited);
  return result;
}

/** \brief Return the size of this variable, taking into account
    such things like whether the variable is a pointer */
ISZ_T
size_of_var(int sptr)
{
  DTYPE dtype;
  ISZ_T sz;

  if (is_tbp_or_final(sptr)) {
    return 0; /* type bound procedure */
  }
  dtype = DTYPEG(sptr);

  if (POINTERG(sptr) || ALLOCG(sptr)) {
    if (DTY(dtype) == TY_ARRAY) {
      /* array pointer, size of pointer+offset+descriptor */
      /* pointer is a pointer, offset and descriptor are DT_INT */
      int rank = ADD_NUMDIM(dtype);
      int descsize = get_descriptor_len(rank);
      return dtypeinfo[TY_PTR].size +
             (descsize + 1) * dtypeinfo[DTY(stb.dt_int)].size;
    }
    /* scalar pointer, size of pointer */
    return dtypeinfo[TY_PTR].size;
  }
  /* not a pointer, just return the size of the type of the variable */
  if (STYPEG(sptr) == ST_PLIST) {
    sz = PLLENG(sptr) * size_of(dtype);
    return sz;
  }

  /* normal size of */
  size_sym = sptr;
  sz = size_of(dtype);
  if (sz < 0) {
    error(219, 3, gbl.lineno, SYMNAME(sptr), NULL);
    sz = 1;
  }
  size_sym = 0;
  return sz;
} /* size_of_var */

INT
size_ast(int sptr, DTYPE dtype)
{
  INT d, len, clen, mlpyr = 1;
  ISZ_T val1;
  TY_KIND ty = get_ty_kind(dtype);

  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_PTR:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
    return mk_isz_cval(dtypeinfo[ty].size, astb.bnd.dtype);

  case TY_HOLL:
    /* treat like default integer type */
    return mk_isz_cval(dtypeinfo[DTY(DT_INT)].size, astb.bnd.dtype);

  case TY_NCHAR:
    mlpyr = 2;
    FLANG_FALLTHROUGH;
  case TY_CHAR:
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
        || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
        ) {
      if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
       || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
      ) {
        clen = ast_intr(I_LEN, astb.bnd.dtype, 1, mk_id(sptr));
      } else {
        clen = DTY(dtype+1);
      }
    } else if (ADJLENG(sptr) && !F90POINTERG(sptr)) {
      /* don't add CVLEN for local automatic character */
      clen = CVLENG(sptr);
      if (clen == 0) {
        clen = sym_get_scalar(SYMNAME(sptr), "len", astb.bnd.dtype);
        CVLENP(sptr, clen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(clen, 1);
      }
      clen = mk_id(clen);
    } else {
      clen = DTY(dtype + 1);
      if (A_ALIASG(clen)) {
        clen = A_ALIASG(clen);
        clen = A_SPTRG(clen);
        clen = CONVAL2G(clen);
        return mk_isz_cval(mlpyr * clen, astb.bnd.dtype);
      }
      clen = mk_convert(clen, astb.bnd.dtype);
    }
    if (mlpyr != 1) {
      len = mk_isz_cval(mlpyr, astb.bnd.dtype);
      clen = mk_binop(OP_MUL, len, clen, astb.bnd.dtype);
    }
    return clen;

  case TY_ARRAY:
    len = size_ast(sptr, DTY(dtype + 1));
    if (DTY(dtype + 2) <= 0) {
      interr("size_ast: no array descriptor", dtype, 3);
      return len;
    }
    if (ADD_DEFER(dtype)) {
      return mk_isz_cval(dtypeinfo[DTY(DT_PTR)].size, astb.bnd.dtype);
    }
    if (ADD_NUMELM(dtype) == 0) {
      /* illegal use of adjustable or assumed-size array:
         should have been caught in semant.  */
      /* errsev(50); */
      ADD_NUMELM(dtype) = astb.bnd.one;
      d = stb.i1;
    } else {
      d = sym_of_ast(ADD_NUMELM(dtype));
      if (d == stb.i0 || STYPEG(d) != ST_CONST) {
        /* illegal use of adjustable or assumed-size array:
           should have been caught in semant.  */
        /* errsev(50); */
        ADD_NUMELM(dtype) = astb.bnd.one;
        d = stb.i1;
      }
    }
    val1 = ad_val_of(d);
    if (A_TYPEG(len) == A_CNST) {
      int dd;
      ISZ_T val2;

      dd = sym_of_ast(len);
      if (STYPEG(dd) != ST_CONST) {
        dd = stb.i1;
      }
      val2 = ad_val_of(dd);
      return mk_isz_cval(val1 * val2, astb.bnd.dtype);
    }
    d = mk_cval(val1, astb.bnd.dtype);
    return mk_binop(OP_MUL, d, len, astb.bnd.dtype);

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    if (DTY(dtype + 2) <= 0 && (!CLASSG(sptr) || !DTY(dtype + 1)) &&
        !has_tbp_or_final(dtype) && !UNLPOLYG(DTY(dtype + 3))) {
      return mk_isz_cval(4, astb.bnd.dtype);
    } else {
      return mk_isz_cval(DTY(dtype + 2), astb.bnd.dtype);
    }

  default:
    interr("size_ast: bad dtype", ty, 3);
    return mk_isz_cval(1, astb.bnd.dtype);
  }
}

/** \brief Like size_ast(), but pass an AST, allowing for member references */
INT
size_ast_of(int ast, DTYPE dtype)
{
  INT d, len, clen, mlpyr = 1, sptr = 0, concat;
  ISZ_T val;
  TY_KIND ty = get_ty_kind(dtype);

  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_PTR:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
    return mk_isz_cval(dtypeinfo[ty].size, astb.bnd.dtype);

  case TY_HOLL:
    /* treat like default integer type */
    return mk_isz_cval(dtypeinfo[DTY(DT_INT)].size, astb.bnd.dtype);

  case TY_NCHAR:
    mlpyr = 2;
    FLANG_FALLTHROUGH;
  case TY_CHAR:
    concat = 0;
    if (ast) {
      if (A_TYPEG(ast) == A_SUBSTR)
        ast = A_LOPG(ast);
      if (A_TYPEG(ast) == A_SUBSCR)
        ast = A_LOPG(ast);
      if (A_TYPEG(ast) == A_FUNC)
        ast = A_LOPG(ast);
      if (A_TYPEG(ast) == A_CNST) {
        sptr = A_SPTRG(ast);
      } else if (A_TYPEG(ast) == A_ID) {
        sptr = A_SPTRG(ast);
      } else if (A_TYPEG(ast) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(ast));
      } else if (A_TYPEG(ast) == A_BINOP && A_OPTYPEG(ast) == OP_CAT) {
        sptr = 0;
        concat = 1;
      } else {
        interr("size_ast_of: unexpected ast type", A_TYPEG(ast), 3);
        sptr = 0;
      }
    } else {
      sptr = 0;
    }
    if (sptr && (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
                 || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
                 )) {
      clen = ast_intr(I_LEN, astb.bnd.dtype, 1, ast);
    } else if (sptr && ADJLENG(sptr) && !F90POINTERG(sptr)) {
      /* don't add CVLEN for local automatic character */
      clen = CVLENG(sptr);
      if (clen == 0) {
        clen = sym_get_scalar(SYMNAME(sptr), "len", astb.bnd.dtype);
        CVLENP(sptr, clen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(clen, 1);
      }
      clen = mk_id(clen);
    } else {
      clen = DTY(dtype + 1);
      if (clen == 0 && concat) {
        /* get the length of the concatenation operands */
        int lsize, rsize;
        lsize = size_ast_of(A_LOPG(ast), A_DTYPEG(A_LOPG(ast)));
        rsize = size_ast_of(A_ROPG(ast), A_DTYPEG(A_ROPG(ast)));
        return mk_binop(OP_ADD, lsize, rsize, astb.bnd.dtype);
      }
      if (A_ALIASG(clen)) {
        clen = A_ALIASG(clen);
        clen = A_SPTRG(clen);
        clen = CONVAL2G(clen);
        return mk_isz_cval(mlpyr * clen, astb.bnd.dtype);
      }
      clen = mk_convert(clen, astb.bnd.dtype);
      clen =
          ast_intr(I_MAX, astb.bnd.dtype, 2, clen, mk_cval(0, astb.bnd.dtype));
    }
    if (mlpyr != 1) {
      len = mk_cval(mlpyr, astb.bnd.dtype);
      clen = mk_binop(OP_MUL, len, clen, astb.bnd.dtype);
    }
    return clen;

  case TY_ARRAY:
    len = size_ast_of(ast, DTY(dtype + 1));
    if (DTY(dtype + 2) <= 0) {
      interr("size_ast_of: no array descriptor", dtype, 3);
      return len;
    }
    if (ADD_DEFER(dtype)) {
      return mk_cval(dtypeinfo[DTY(DT_PTR)].size, DT_INT);
    }
    if (ADD_NUMELM(dtype) == 0) {
      /* illegal use of adjustable or assumed-size array:
         should have been caught in semant.  */
      /* errsev(50); */
      ADD_NUMELM(dtype) = astb.bnd.one;
      d = stb.i1;
    } else {
      d = sym_of_ast(ADD_NUMELM(dtype));
      if (d == stb.i0 || STYPEG(d) != ST_CONST) {
        /* illegal use of adjustable or assumed-size array:
           should have been caught in semant.  */
        /* errsev(50); */
        ADD_NUMELM(dtype) = astb.bnd.one;
        d = stb.i1;
      }
    }
    val = ad_val_of(d);
    if (A_TYPEG(len) == A_CNST) {
      int dd;
      ISZ_T val2;
      dd = sym_of_ast(len);
      if (STYPEG(dd) != ST_CONST) {
        dd = stb.i1;
      }
      val2 = ad_val_of(dd);
      return mk_isz_cval(val * val2, astb.bnd.dtype);
    }
    d = mk_isz_cval(d, astb.bnd.dtype);
    return mk_binop(OP_MUL, d, len, astb.bnd.dtype);

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    if (!sptr)
      sptr = DTY(dtype + 1);
    if (DTY(dtype + 2) <= 0 && !UNLPOLYG(DTY(dtype+3)) && 
        (!CLASSG(sptr) || !DTY(dtype + 1))) {
      errsev(151);
      return mk_isz_cval(4, astb.bnd.dtype);
    } else {
      return mk_isz_cval(DTY(dtype + 2), astb.bnd.dtype);
    }

  default:
    interr("size_ast_of: bad dtype", ty, 3);
    return mk_isz_cval(1, astb.bnd.dtype);
  }
} /* size_ast_of */

INT
string_expr_length(int ast)
{
  int len, al, ar;
  DTYPE dt_int;
  int sym, iface;
  /* must be constant reference, symbol reference, or concatenation */
  if (ast <= 0)
    return astb.i0;
  dt_int = DT_INT;
  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_CNST:
  case A_MEM:
    return size_ast_of(ast, DDTG(A_DTYPEG(ast)));
  case A_SUBSTR:
    if (A_DTYPEG(A_LEFTG(ast)) == DT_INT8)
      dt_int = DT_INT8;
    else if (A_DTYPEG(A_RIGHTG(ast)) == DT_INT8)
      dt_int = DT_INT8;
    if (A_RIGHTG(ast)) {
      if (A_LEFTG(ast) == 0) {
        len = A_RIGHTG(ast);
      } else {
        int l1;
        l1 = mk_binop(OP_SUB, A_LEFTG(ast), astb.i1, dt_int);
        len = mk_binop(OP_SUB, A_RIGHTG(ast), l1, dt_int);
      }
    } else {
      if (A_LEFTG(ast) == 0) {
        return string_expr_length(A_LOPG(ast));
      } else {
        int l1, l2;
        l1 = mk_binop(OP_SUB, A_LEFTG(ast), astb.i1, dt_int);
        l2 = string_expr_length(A_LOPG(ast));
        len = mk_binop(OP_SUB, l2, l1, dt_int);
      }
    }
    if (A_ALIASG(len)) {
      int cvlen;
      cvlen = get_int_cval(A_SPTRG(len));
      if (cvlen < 0)
        len = mk_cval(0, DT_INT4);
    } else if (dt_int != DT_INT8)
      len = ast_intr(I_MAX, DT_INT4, 2, len, mk_cval(0, DT_INT4));
    else
      len = ast_intr(I_MAX, DT_INT8, 2, len, mk_cval(0, DT_INT8));
    return len;
  case A_SUBSCR:
    /* subscripted reference, just get the length of the symbol */
    return string_expr_length(A_LOPG(ast));
  case A_PAREN:
  case A_CONV:
    return string_expr_length(A_LOPG(ast));
  case A_BINOP:
    if (A_OPTYPEG(ast) != OP_CAT) {
      interr("string_expr_length: operator not concatenation", A_OPTYPEG(ast),
             3);
      return astb.i0;
    }
    al = string_expr_length(A_LOPG(ast));
    ar = string_expr_length(A_ROPG(ast));
    if (A_DTYPEG(al) == DT_INT8)
      dt_int = DT_INT8;
    else if (A_DTYPEG(ar) == DT_INT8)
      dt_int = DT_INT8;
    len = mk_binop(OP_ADD, al, ar, dt_int);
    return ast_intr(I_MAX, dt_int, 2, len, mk_cval(0, dt_int));
  case A_FUNC:
    /* FS#21600: need to get the interface from the A_FUNC ast. */
    sym = procsym_of_ast(A_LOPG(ast));
    iface = 0;
    proc_arginfo(sym, NULL, NULL, &iface);
    return string_expr_length(mk_id(iface));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_TRIM:
      return ast_intr(I_LEN_TRIM, astb.bnd.dtype, 1, ARGT_ARG(A_ARGSG(ast), 0));
    case I_RESHAPE:
      return ast_intr(I_LEN, astb.bnd.dtype, 1, ARGT_ARG(A_ARGSG(ast), 0));
    case I_ACHAR:
    case I_CHAR:
      return ast_intr(I_INT, astb.bnd.dtype, 1, ARGT_ARG(A_ARGSG(ast), 0));
    }
  /* else fall thru */
  default:
    interr("string_expr_length: ast not string op", A_TYPEG(ast), 3);
    return astb.i0;
  }
} /* string_expr_length */

/** \brief Change \p dtype from assumed-length to length of \p ast */
DTYPE
adjust_ch_length(DTYPE dtype, int ast)
{
  int len;
  /* if 'dtype' is assumed-length character, create a new
   * datatype with same character type, but with length equal
   * to length of the expression ast 'ast' */
  dtype = DDTG(dtype);
  if (dtype != DT_ASSNCHAR && dtype != DT_ASSCHAR && dtype != DT_DEFERNCHAR &&
      dtype != DT_DEFERCHAR) {
    return dtype;
  }
  len = string_expr_length(ast);
  if (len) {
    dtype = get_type(2, DTY(dtype), len);
  }
  return dtype;
} /* adjust_ch_length */

/** \brief Fix array and char dtypes.

    Given datatype \p dtype and symbol \p sptr, return a datatype
    equivalent to the given datatype, but with any array bounds
    filled in from the symbol's array bounds, and char length
    filled in from the symbol's char length, using
    calls to LBOUND, UBOUND, or LEN, as necessary.
 */
DTYPE
fix_dtype(int sptr, DTYPE dtype)
{
  DTYPE elemdt;
  int sym;
  TY_KIND ty = get_ty_kind(dtype);

  if (sptr <= NOSYM)
    return dtype;

  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_PTR:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
  case TY_HOLL:
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    return dtype;

  case TY_NCHAR:
  case TY_CHAR:
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
        || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
        ) {
      int clen = ast_intr(I_LEN, astb.bnd.dtype, 1, mk_id(sptr));
      return get_type(2, ty, clen);
    }
    if (ADJLENG(sptr) && !F90POINTERG(sptr)) {
      /* don't add CVLEN for local automatic character */
      int cvlen;
      int clen = CVLENG(sptr);
      if (clen == 0) {
        clen = sym_get_scalar(SYMNAME(sptr), "len", DT_INT);
        CVLENP(sptr, clen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(clen, 1);
      }
      cvlen = CVLENG(sptr);
      clen = mk_id(clen);
      clen = ast_intr(I_MAX, DTYPEG(cvlen), 2, clen, mk_cval(0, DTYPEG(cvlen)));
      return get_type(2, ty, clen);
    }
    return dtype;

  case TY_ARRAY:
    if (DTY(dtype + 2) <= 0) {
      interr("fix_dtype: no array descriptor", dtype, 3);
      return dtype;
    }
    elemdt = fix_dtype(sptr, DTY(dtype + 1));
    sym = mk_id(sptr);
    if (ADD_ASSUMSHP(dtype) == 1) {
      /* get bounds from that of sptr */
      int ndim = ADD_NUMDIM(dtype);
      DTYPE dt = get_array_dtype(ndim, elemdt);
      int i;
      for (i = 0; i < ndim; ++i) {
        int up, ext;
        ADD_MLPYR(dt, i) = 0;
        ADD_LWBD(dt, i) = ADD_LWBD(dtype, i);
        ADD_LWAST(dt, i) = ADD_LWAST(dtype, i);
        up = ADD_UPBD(dtype, i);
        if (up == 0) {
          up = ast_intr(I_UBOUND, DT_INT, 2, sym, i + 1);
          ADD_UPBD(dt, i) = up;
          ADD_UPAST(dt, i) = up;
        } else {
          ADD_UPBD(dt, i) = ADD_UPBD(dtype, i);
          ADD_UPAST(dt, i) = ADD_UPAST(dtype, i);
        }
        ext = mk_extent(ADD_LWAST(dt, i), ADD_UPAST(dtype, i), i);
        ADD_EXTNTAST(dt, i) = ext;
      }
      ADD_NUMELM(dt) = 0;
      ADD_ZBASE(dt) = 0;
      return dt;
    }
    if (elemdt != DTY(dtype + 1)) {
      /* same array bounds, different subtype */
      int ndim = ADD_NUMDIM(dtype);
      DTYPE dt = get_array_dtype(ndim, elemdt);
      int i;
      for (i = 0; i < ndim; ++i) {
        ADD_MLPYR(dt, i) = ADD_MLPYR(dtype, i);
        ADD_LWBD(dt, i) = ADD_LWBD(dtype, i);
        ADD_UPBD(dt, i) = ADD_UPBD(dtype, i);
        ADD_LWAST(dt, i) = ADD_LWAST(dtype, i);
        ADD_UPAST(dt, i) = ADD_UPAST(dtype, i);
        ADD_EXTNTAST(dt, i) = ADD_EXTNTAST(dtype, i);
      }
      ADD_NUMELM(dt) = ADD_NUMELM(dtype);
      ADD_ZBASE(dt) = ADD_ZBASE(dtype);
      return dt;
    }
    return dtype;

  default:
    interr("fix_dtype: bad dtype", dtype * 1000 + ty, 3);
    return dtype;
  }
} /* fix_dtype */

ISZ_T
extent_of(DTYPE dtype)
{
  ISZ_T d;
  ADSC *ad;
  int numelem;

#if DEBUG
  assert(DTY(dtype) == TY_ARRAY, "extent_of, expected TY_ARRAY", dtype, 3);
#endif
  if ((d = DTY(dtype + 2)) <= 0) {
    interr("extent_of: no array descriptor", (int)d, 3);
    return 0;
  }
  ad = AD_DPTR(dtype);
  numelem = AD_NUMELM(ad);
  if (numelem == 0)
    return 0;
  if (A_ALIASG(numelem) == 0)
    return 0;
  d = sym_of_ast(AD_NUMELM(ad));
  if (d == stb.i0 || STYPEG(d) != ST_CONST)
    return 0;
  d = CONVAL2G(d);
  return d;
}

DTYPE
dtype_with_shape(DTYPE dtype, int shape)
{
  int ndim, i, last_mp, last_mp_const;
  ISZ_T last_mp_val;
  DTYPE dtyper;
  /* if the shape is scalar, either return the old scalar datatype
   * or the base type if the old datatype was an array */
  if (shape == 0 || SHD_NDIM(shape) == 0) {
    if (DTY(dtype) == TY_ARRAY) {
      return DTY(dtype + 1);
    } else {
      return dtype;
    }
  }
  /* if the datatype is an array and the shape has dimensions,
   * and the dimensionality and sizes match, no need to change */
  ndim = SHD_NDIM(shape);
  if (DTY(dtype) == TY_ARRAY) {
    if (ADD_NUMDIM(dtype) == ndim) {
      /* check the upper/lower bounds */
      for (i = 0; i < ndim; ++i) {
        /* skip if the stride is not one */
        if (SHD_STRIDE(shape, i) != astb.bnd.one)
          return dtype;
        if (SHD_LWB(shape, i) != ADD_LWAST(dtype, i))
          break;
        if (SHD_UPB(shape, i) != ADD_UPAST(dtype, i))
          break;
      }
      if (i == ndim) {
        /* all bounds matched */
        return dtype;
      }
    }
  }
  /* must make a new datatype */
  dtyper = get_array_dtype(ndim, DT_NONE);
  if (DTY(dtype) == TY_ARRAY) {
    /* copy base type */
    DTY(dtyper + 1) = DTY(dtype + 1);
  } else {
    /* make this the new base type */
    DTY(dtyper + 1) = dtype;
  }
  last_mp_const = 1;
  last_mp_val = 1;
  last_mp = astb.bnd.one;
  for (i = 0; i < ndim; ++i) {
    int lb, ub;
    ISZ_T lbval, ubval;
    ADD_LWAST(dtyper, i) = ADD_LWBD(dtyper, i) = SHD_LWB(shape, i);
    ADD_UPAST(dtyper, i) = ADD_UPBD(dtyper, i) = SHD_UPB(shape, i);
    ADD_EXTNTAST(dtyper, i) =
        mk_extent(ADD_LWAST(dtyper, i), ADD_UPAST(dtyper, i), i);
    ADD_MLPYR(dtyper, i) = last_mp;
    lb = ADD_LWAST(dtyper, i);
    if (!A_ALIASG(lb)) {
      lb = -1;
    } else {
      lb = A_ALIASG(lb);
      lbval = ad_val_of(A_SPTRG(lb));
    }
    ub = ADD_UPAST(dtyper, i);
    if (!A_ALIASG(ub)) {
      ub = -1;
    } else {
      ub = A_ALIASG(ub);
      ubval = ad_val_of(A_SPTRG(ub));
    }
    if (last_mp_const && lb > 0 && ub > 0) {
      last_mp_val = (last_mp_val) * (ubval - lbval + 1);
      last_mp = mk_isz_cval(last_mp_val, astb.bnd.dtype);
    } else {
      last_mp_const = 0;
      last_mp = mk_bnd_ast();
    }
  }

  ADD_NUMELM(dtyper) = last_mp;
  ADD_ZBASE(dtyper) = mk_bnd_ast();
  return dtyper;
} /* dtype_with_shape */

ISZ_T
ad_val_of(int sym)
{
  if (XBIT(68, 0x1)) {
    INT num[2];
    ISZ_T v;
    num[0] = CONVAL1G(sym);
    num[1] = CONVAL2G(sym);
    INT64_2_ISZ(num, v);
    return v;
  }
  return CONVAL2G(sym);
}

/** \brief Create a constant sym entry which reflects the type of an array
    bound/extent.
 */
int
get_bnd_con(ISZ_T v)
{
  INT num[2];

  if (XBIT(68, 0x1)) {
    ISZ_2_INT64(v, num);
    return getcon(num, DT_INT8);
  }
  num[0] = 0;
  num[1] = v;
  return getcon(num, DT_INT);
}

int
alignment(DTYPE dtype)
{
  TY_KIND ty = get_ty_kind(dtype);
  int align_val;

  switch (ty) {
  case TY_DWORD:
  case TY_DBLE:
  case TY_DCMPLX:
  case TY_QCMPLX:
    if (!flg.dalign)
      return dtypeinfo[TY_INT].align;
    FLANG_FALLTHROUGH;
  case TY_QUAD:
  case TY_WORD:
  case TY_HOLL:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_REAL:
  case TY_CMPLX:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_CHAR:
  case TY_NCHAR:
  case TY_PTR:
    return dtypeinfo[ty].align;
  case TY_INT8:
  case TY_LOG8:
    if (!flg.dalign || XBIT(119, 0x100000))
      return dtypeinfo[TY_INT].align;
    return dtypeinfo[ty].align;

  case TY_ARRAY:
    align_val = alignment((int)DTY(dtype + 1));
    return align_val;

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    return DTY(dtype + 4);

  default:
    interr("alignment: bad dtype ", ty, 3);
    return 0;
  }
}

/** \brief Like alignment(), but takes into account whether the var is a pointer
 */
int
alignment_of_var(int sptr)
{
  DTYPE dtype = DTYPEG(sptr);
  int align;

  if (POINTERG(sptr) || ALLOCG(sptr)) {
    align = dtypeinfo[TY_PTR].align;
  } else {
    align = alignment(dtype);
  }
#ifdef QALNG
  if (QALNG(sptr)) {
    int ta;
    ta = dtypeinfo[TY_QUAD].align;
    if (align < ta)
      align = ta;
  }
#endif
#ifdef PDALN_IS_DEFAULT
  if (!PDALN_IS_DEFAULT(sptr)) {
    /* PDALNG==3 means align to 2^3==8 byte address, align == 7 */
    int ta = (1 << PDALNG(sptr)) - 1;
    if (align < ta)
      align = ta;
  }
#endif
  /*
   * If alignment of variable set by `!DIR$ ALIGN alignment`
   * in flang1 is smaller than its original, then this pragma
   * should have no effect.
   */
  if (align < PALIGNG(sptr)) {
    align = PALIGNG(sptr) - 1;
  }
  return align;
} /* alignment_of_var */

int
bits_in(DTYPE dtype)
{
  TY_KIND ty = get_ty_kind(dtype);

  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_HOLL:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_CHAR:
  case TY_NCHAR:
  case TY_INT8:
  case TY_LOG8:
  case TY_PTR:
    return dtypeinfo[ty].bits;

  default:
    interr("bits_in: bad type ", ty, 3);
    return 0;
  }
}

/*---------------------------------------------------------*/

/*
 * Data structure to hold TY_CHAR entries: linked list off of
 * array chartab; entries that are equal module CHARTABSIZE are
 * linked.  Relative pointers (integers) are used.
 */
#define CHARTABSIZE 40
static int chartab[CHARTABSIZE];
struct chartab {
  int next;
  DTYPE dtype;
};
static int chartabavail, chartabsize;
static struct chartab *chartabbase = 0;

void
init_chartab(void)
{
  int i;

  for (i = 0; i < CHARTABSIZE; ++i)
    chartab[i] = 0;
  if (chartabbase == 0) {
    /* allocate new */
    chartabsize = CHARTABSIZE;
    NEW(chartabbase, struct chartab, chartabsize);
  }
  chartabavail = 1;
  chartabbase[0].next = 0;
  chartabbase[0].dtype = 0;

  rehost_machar(flg.x[45]);

  if (XBIT(52, 1)) {
    /* complex must be double-aligned */
    dtypeinfo[TY_CMPLX].align = dtypeinfo[TY_DBLE].align;
  }
}

void
fini_chartab()
{
  FREE(chartabbase);
  chartabsize = 0;
  chartabavail = 0;
} /* fini_chartab */

/** Find or allocate a slot in dtype array for the new datatype. For strings,
 * check if the data type is already present.
 *
 * \param n number of datatype entries we want to occupy
 * \param v1 data type
 * \param v2 second value, meaning depends on data type, for strings it is
 *           the length, for pointers - target type, etc
 */
DTYPE
get_type(int n, TY_KIND v1, int v2)
{
  int i, j;
  DTYPE dtype = 0;
  LOGICAL is_nchar = FALSE;
  is_nchar = (v1 == TY_NCHAR);

  /* For a string try to find a matching type first */
  if (v1 == TY_CHAR || is_nchar) {
    if (v2 < 0 || v2 >= astb.stg_avail) {
      interr("char string length is wrong.", v2, 2);
      v2 = astb.i1;
    }
    i = v2 % CHARTABSIZE;
    if (chartab[i]) {
      /* check list for this length */
      for (j = chartab[i]; j != 0; j = chartabbase[j].next) {
        int k = chartabbase[j].dtype;
        if (DTY(k + 1) == v2 && /* same length */
            DTY(k) == v1 /*TY_CHAR vs TY_NCHAR*/) {
          dtype = chartabbase[j].dtype;
          goto found;
        }
      }
    }
    if (v2 == astb.i1) {
      if (v1 == TY_CHAR) {
        dtype = DT_CHAR;
      } else if (v1 == TY_NCHAR) {
        dtype = DT_NCHAR;
      }
    }
  }
  if (dtype == 0) {
    dtype = STG_NEXT_SIZE(stb.dt, n);
    DTY(dtype) = v1;
    DTY(dtype + 1) = v2;
    if (v1 == TY_CHAR || is_nchar) {
      /* not found */
      NEED(chartabavail + n, chartabbase, struct chartab, chartabsize,
           chartabsize + CHARTABSIZE);
      chartabbase[chartabavail].dtype = dtype;
      chartabbase[chartabavail].next = chartab[i];
      chartab[i] = chartabavail++;
    }
  }
found:
  return dtype;
}

/** \brief Return true if the data types for two functions are compatible.

    Two functions are compatible if a single local variable can be
    used to hold their return values and therefore implying that the
    same return mechanism can be used for the functions.
 */
LOGICAL
cmpat_func(DTYPE d1, DTYPE d2)
{
  if (d1 == d2) {
    return TRUE;
  } else {
    int fv1 = dtypeinfo[DTY(d1)].fval;
    int fv2 = dtypeinfo[DTY(d2)].fval;
    assert(fv1 >= 0, "cmpat_func1: bad dtype", d1, 3);
    assert(fv2 >= 0, "cmpat_func2: bad dtype", d2, 3);
    return fv1 == fv2;
  }
}

/** \brief Return TRUE if the scalar data types of an actual argument matches
    the formal with respect to the type & kind rules.
 */
LOGICAL
tk_match_arg(int formal_dt, int actual_dt, LOGICAL flag)
{
  int f_len;
  int a_len;
  LOGICAL unk = FALSE;
  int f_dt = DDTG(formal_dt);
  int a_dt = DDTG(actual_dt);

  if (DTY(f_dt) == TY_CHAR) {
    if (DTY(a_dt) != TY_CHAR)
      return FALSE;
    /*
     * if formal is not assumed length, the length of the formal must be
     * be less than or equal to the length of the actual.
     */
    if (f_dt != DT_ASSCHAR && a_dt != DT_ASSCHAR && f_dt != DT_DEFERCHAR &&
        a_dt != DT_DEFERCHAR) {
      f_len = DTY(f_dt + 1);
      if (!A_ALIASG(f_len)) {
        f_len = 0;
      } else {
        f_len = A_ALIASG(f_len);
        f_len = A_SPTRG(f_len);
        f_len = CONVAL2G(f_len);
      }
      a_len = DTY(a_dt + 1);
      if (!A_ALIASG(a_len)) {
        a_len = 0;
        unk = TRUE;
      } else {
        a_len = A_ALIASG(a_len);
        a_len = A_SPTRG(a_len);
        a_len = CONVAL2G(a_len);
      }
      if (DTY(formal_dt) == TY_ARRAY) {
        int f_nelems = extent_of(formal_dt);
        int a_nelems;
        if (DTY(actual_dt) == TY_ARRAY)
          a_nelems = extent_of(actual_dt);
        else
          a_nelems = 1;
        if (f_nelems && a_nelems && f_len && a_len) {
          if (f_nelems * f_len > a_nelems * a_len)
            return FALSE;
        }
      } else if (!unk && f_len > a_len)
        return FALSE;
    }
  }
  else if (DTY(f_dt) == TY_NCHAR) {
    if (DTY(a_dt) != TY_NCHAR)
      return FALSE;
    f_len = DTY(f_dt + 1);
    if (!A_ALIASG(f_len)) {
      f_len = 0;
      unk = TRUE;
    } else {
      f_len = A_ALIASG(f_len);
      f_len = A_SPTRG(f_len);
      f_len = CONVAL2G(f_len);
    }
    a_len = DTY(a_dt + 1);
    if (!A_ALIASG(a_len)) {
      a_len = 0;
    } else {
      a_len = A_ALIASG(a_len);
      a_len = A_SPTRG(a_len);
      a_len = CONVAL2G(a_len);
    }
    if (f_dt != DT_ASSNCHAR && f_dt != DT_DEFERNCHAR) {
      if (!unk && f_len > a_len)
        return FALSE;
    }
  }
  else if (!eq_dtype2(f_dt, a_dt, flag)) {
    if (DTY(f_dt) == TY_PTR && DTY(a_dt) == TY_PTR && 
        DTY(DTY(f_dt + 1)) == TY_PROC && DTY(DTY(a_dt + 1)) == TY_PROC) {
      /* eq_dtype2 checks equality of the procedure pointers.
       * If they are not the same (including the same name), then
       * it returns false. This is correct for an equality test.
       * However, in this case, we don't care about the names being
       * the same if all other attributes are equal.
       */
       DTYPE d1 = DTY(f_dt + 1);
       DTYPE d2 = DTY(a_dt + 1);
       if (cmp_interfaces(DTY(d1 + 2), DTY(d2 + 2), FALSE)) {
         return TRUE;
       }
    } 
    return FALSE;
  }

  return TRUE;
}

#if defined(PARENTG)
LOGICAL
extends_type(int tg1, int tg2)
{
  /* Returns true if derived type tag tg2 extends derived type tag tg1 */
  int sptr;

  if (!tg2 || !tg1 || DTY(DTYPEG(tg1)) != TY_DERIVED ||
      DTY(DTYPEG(tg2)) != TY_DERIVED)
    return FALSE;
  if (strcmp(SYMNAME(tg1), SYMNAME(tg2)) == 0)
    return TRUE;
  sptr = DTY(DTYPEG(tg2) + 1);
  if (PARENTG(sptr))
    return extends_type(tg1, sptr);
  return FALSE;
}

static LOGICAL
same_parameterized_dt(DTYPE d1, DTYPE d2)
{

  /* Used in conjunction with same_dtype().
   * Returns TRUE if both d1 and d2 are the same
   * parameterized derived type
   */
  int base_type1, base_type2, mem1, mem2, val1, val2;
  int rslt;

  if (d1 == d2)
    return TRUE;
  if (DTY(d1) == TY_DERIVED && DTY(d2) == TY_DERIVED) {
    base_type1 = BASETYPEG(DTY(d1 + 3));
    base_type2 = BASETYPEG(DTY(d2 + 3));
    if (!base_type1)
      base_type1 = d1;
    if (!base_type2)
      base_type2 = d2;
    if (base_type1 && base_type2 && base_type1 == base_type2) {
      for (mem1 = DTY(d1 + 1), mem2 = DTY(d2 + 1); mem1 > NOSYM && mem2 > NOSYM;
           mem1 = SYMLKG(mem1), mem2 = SYMLKG(mem2)) {
        if (PARENTG(mem1)) {
          if (!PARENTG(mem2)) {
            return FALSE;
          }
          rslt = same_parameterized_dt(DTYPEG(mem1), DTYPEG(mem2));
          if (!rslt)
            return FALSE;
        } else if (PARENTG(mem2)) {
          return FALSE;
        }
        if (!SETKINDG(mem1) && !USEKINDG(mem1) && KINDG(mem1) &&
            PARMINITG(mem1)) {
          val1 = PARMINITG(mem1);
        } else if (SETKINDG(mem1) && !USEKINDG(mem1) && KINDG(mem1)) {
          if (LENPARMG(mem1)) {
            val1 = chk_kind_parm_set_expr(LENG(mem1), 0);
            if (val1 > 0 && A_TYPEG(val1) == A_CNST) {
              val1 = CONVAL2G(A_SPTRG(val1));
            } else {
              continue;
            }
          } else {
            val1 = KINDG(mem1);
          }
        } else {
          val1 = 0;
        }
        if (!SETKINDG(mem2) && !USEKINDG(mem2) && KINDG(mem2) &&
            PARMINITG(mem2)) {
          val2 = PARMINITG(mem2);
        } else if (SETKINDG(mem2) && !USEKINDG(mem2) && KINDG(mem2)) {
          if (LENPARMG(mem2)) {
            val2 = chk_kind_parm_set_expr(LENG(mem2), 0);
            if (val2 > 0 && A_TYPEG(val2) == A_CNST) {
              val2 = CONVAL2G(A_SPTRG(val2));
            } else {
              continue;
            }
          } else {
            val2 = KINDG(mem2);
          }
        } else {
          val2 = 0;
        }
        if (val1 != val2) {
          return FALSE;
        }
      }
      return TRUE;
    }
  }
  return FALSE;
}
#endif

/** \brief In the presence of modules and interface blocks, it's possible that
   two
    identical derived types are not represented by the same data type record.
    If this occurs, eq_dtype() will check the types of the members.
 */
LOGICAL
eq_dtype2(DTYPE d1, DTYPE d2, LOGICAL flag)
{
  int s1, s2;
  int tg1, tg2;

  if (d1 == d2)
    return TRUE;
  if (DTY(d1) != DTY(d2))
    return FALSE;
  switch (DTY(d1)) {
  case TY_ARRAY:
    /* check rank and element type */
    if (ADD_NUMDIM(d1) != ADD_NUMDIM(d2))
      return FALSE;
    return (eq_dtype2((int)DTY(d1 + 1), (int)DTY(d2 + 1), flag));

  case TY_DERIVED:
    tg1 = DTY(d1 + 3);
    tg2 = DTY(d2 + 3);
    if (tg1 == tg2)
      /* tags are the same => equal types */
      return TRUE;
#if defined(PARENTG)
    if (flag && extends_type(tg1, tg2)) /* type extension */
      return TRUE;
#endif
    if (same_parameterized_dt(d1, d2))
      return TRUE;
    if (strcmp(SYMNAME(tg1), SYMNAME(tg2)) != 0)
      return FALSE;

    if (VISITG(tg1) && VISITG(tg2)) {
      /* have a self-referential derived type */
      return TRUE;
    }
    if (VISITG(tg1) || VISITG(tg2)) {
      return FALSE;
    }
    VISITP(tg1, 1);
    VISITP(tg2, 1);
    /* traverse the members */
    for (s1 = DTY(d1 + 1), s2 = DTY(d2 + 1); s1 > NOSYM && s2 > NOSYM;
         s1 = SYMLKG(s1), s2 = SYMLKG(s2)) {
      if (HCCSYMG(s1) && HCCSYMG(s2)) {
      } else if (HCCSYMG(s1) || HCCSYMG(s2)) {
        break; /* return FALSE; */
      } else if (strcmp(SYMNAME(s1), SYMNAME(s2)) != 0)
        break; /* return FALSE; */
      /* if one member is private, both must be */
      if (PRIVATEG(s1) != PRIVATEG(s2))
        break; /* return FALSE; */
      /* member types are different => different types */
      if (!eq_dtype2(DTYPEG(s1), DTYPEG(s2), flag))
        break; /* return FALSE; */
    }
    VISITP(tg1, 0);
    VISITP(tg2, 0);
    /*  more members in either record? */
    if (s2 > NOSYM || s1 > NOSYM)
      return FALSE;
    return TRUE;

  case TY_CHAR:
  case TY_NCHAR:
    /* compare lengths */
    if (DTY(d1 + 1) == DTY(d2 + 1))
      return TRUE;
    break;

  case TY_PTR:
    if (DTY(DTY(d1 + 1)) == TY_PROC) {
      return eq_dtype2(DTY(d1 + 1), DTY(d2 + 1), flag);
    }
    break;
  case TY_PROC:
    if ((DTY(d1 + 2) && (DTY(d1 + 2) == DTY(d2 + 2))) ||
        (cmp_interfaces(DTY(d1 + 2), DTY(d2 + 2), TRUE))) {
      /* identical interfaces */
      return TRUE;
    }
    if (!DTY(d1 + 2) && !DTY(d1 + 2)) {
      /* no interfaces; check result dtypes */
      return eq_dtype2(DTY(d1 + 1), DTY(d2 + 1), flag);
    }
    break;
  default:
    break;
  }
  return FALSE;
}

LOGICAL
eq_dtype(DTYPE d1, DTYPE d2)
{
  return eq_dtype2(d1, d2, FALSE);
}

/** \brief Check to see if two types are extensions from the same ancestor */
LOGICAL
same_ancestor(DTYPE dtype1, DTYPE dtype2)
{
  int mem1, mem2;
  int next_dtype1, next_dtype2;

  if (DTY(dtype1) == TY_ARRAY)
    dtype1 = DTY(dtype1 + 1);

  if (DTY(dtype2) == TY_ARRAY)
    dtype2 = DTY(dtype2 + 1);

  if (DTY(dtype1) != TY_DERIVED || DTY(dtype2) != TY_DERIVED)
    return FALSE;

  if (eq_dtype2(dtype1, dtype2, 1) || eq_dtype2(dtype2, dtype1, 1))
    return TRUE;

  mem1 = DTY(dtype1 + 1);
  mem2 = DTY(dtype2 + 1);

  if (PARENTG(mem1))
    next_dtype1 = DTYPEG(mem1);
  else
    next_dtype1 = dtype1;

  if (PARENTG(mem2))
    next_dtype2 = DTYPEG(mem2);
  else
    next_dtype2 = dtype2;

  if (dtype1 == next_dtype1 && dtype2 == next_dtype2)
    return FALSE;

  return same_ancestor(next_dtype1, next_dtype2);
}

/*
 *  These data type tests use search_type_members() above, so they're
 *  each written in terms of (1) a predicate function to be applied to the
 *  component members of a symbol's derived type and (2) a wrapper that
 *  applies test_sym_and_components() to that predicate function.
 */

static LOGICAL
is_recursive(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         search_type_members(DTYPEG(sptr), is_recursive, visited);
}

LOGICAL
has_recursive_component(int sptr)
{
  return test_sym_and_components(sptr, is_recursive);
}

static LOGICAL
is_finalized(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         ((STYPEG(sptr) == ST_MEMBER &&
           (FINALG(sptr) != 0 || FINALIZEDG(sptr))) ||
          search_type_members(DTYPEG(sptr), is_finalized, visited));
}

LOGICAL
has_finalized_component(SPTR sptr)
{
  return test_sym_components_only(sptr, is_finalized);
}

static LOGICAL
is_impure_finalizer(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         ((STYPEG(sptr) == ST_MEMBER &&
           FINALG(sptr) && is_impure(VTABLEG(sptr))) ||
           search_type_members(DTYPEG(sptr), is_impure_finalizer, visited));
}

LOGICAL
has_impure_finalizer(SPTR sptr)
{
  return test_sym_and_components(sptr, is_impure_finalizer);
}

static LOGICAL
is_layout_desc(SPTR sptr, struct visit_list **visited)
{
  return sptr > NOSYM && ((/* STYPEG(sptr) == ST_MEMBER && */
                           (POINTERG(sptr) || ALLOCATTRG(sptr) ||
                            DTYG(DTYPEG(sptr)) ==
                                TY_PTR /* procedure pointer */ ||
                            (/* PARENTG(sptr) -- why only parents?  why not
                                other components? && */
                             search_type_members(DTYPEG(sptr), is_layout_desc,
                                                 visited)))) ||
                          has_finalized_component(sptr) ||
                          has_recursive_component(sptr) ||
                          is_or_has_derived_allo(sptr));
}

LOGICAL
has_layout_desc(SPTR sptr)
{
  return test_sym_components_only(sptr, is_layout_desc);
}

static LOGICAL
is_poly(SPTR sptr, struct visit_list **visited)
{
  return sptr > NOSYM && ((CLASSG(sptr) && !is_tbp_or_final(sptr)) ||
                          search_type_members(DTYPEG(sptr), is_poly, visited));
}

LOGICAL
is_or_has_poly(SPTR sptr)
{
  return test_sym_and_components(sptr, is_poly);
}

static LOGICAL
is_derived_type_allo(SPTR sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         ((ALLOCATTRG(sptr) && is_container_dtype(DTYPEG(sptr))) ||
          search_type_members(DTYPEG(sptr), is_derived_type_allo, visited));
}

LOGICAL
is_or_has_derived_allo(SPTR sptr)
{
  return test_sym_and_components(sptr, is_derived_type_allo);
}

/** \brief Similar to eq_dtype(), except all scalar integer types are compatible
   with
    each other, all scalar logical types are compatible with each other, all
    character types are compatible with each other.
 */
LOGICAL
cmpat_dtype(DTYPE d1, DTYPE d2)
{
  if (d1 == d2)
    return TRUE;
  if (DTY(d1) != DTY(d2)) {
    /* check for any logical first since logical types also have the
     * _TY_INT attribute.
     */
    if (DT_ISLOG(d1) && DT_ISLOG(d2))
      return TRUE;
    if (DT_ISINT(d1) && DT_ISINT(d2))
      return TRUE;
    return FALSE;
  }
  if (DTY(d1) == TY_CHAR || DTY(d1) == TY_NCHAR)
    return TRUE;
  return eq_dtype(d1, d2);
}

/** \brief Similar to compat_dtype(), except all scalar integer types are
    compatible with each other, all scalar logical types are compatible
    with each other, all character types are compatible with each other.
    Also, array extents are checked.
 */
LOGICAL
cmpat_dtype_with_size(DTYPE d1, DTYPE d2)
{
  int s1, s2, i, n;

  if (is_iso_cptr(d1)) {
    d1 = DTYPEG(DTY(d1 + 1));
  }
  if (is_iso_cptr(d2)) {
    d2 = DTYPEG(DTY(d2 + 1));
  }

  if (d1 == d2)
    return TRUE;
  if (d1 <= 0 || d2 <= 0)
    return FALSE;
  if (DTY(d1) != DTY(d2)) {
    /* check for any logical first since logical types also have the
     * _TY_INT attribute.
     */
    if (DT_ISLOG(d1) && DT_ISLOG(d2))
      return TRUE;
    if (DT_ISNUMERIC(d2) && DT_ISNUMERIC(d1))
      return TRUE;
    /* allow array of blah to match with scalar blah */
    if (DTY(d1) == TY_ARRAY && cmpat_dtype_with_size(DTY(d1 + 1), d2))
      return TRUE;
    if (DTY(d2) == TY_ARRAY && cmpat_dtype_with_size(d1, DTY(d2 + 1)))
      return TRUE;
    return FALSE;
  }
  /* here, DTY(d1) == DTY(d2) */
  if (DTY(d1) == TY_CHAR || DTY(d1) == TY_NCHAR)
    return TRUE;
  switch (DTY(d1)) {
  case TY_ARRAY:
    /* check rank, extents and element type */
    if (ADD_NUMDIM(d1) != ADD_NUMDIM(d2))
      return FALSE;
    n = ADD_NUMDIM(d1);
    for (i = 0; i < n; ++i) {
      /* check bounds */
      int l1, l2, u1, u2, e1, e2;
      l1 = ADD_LWBD(d1, i);
      u1 = ADD_UPBD(d1, i);
      l2 = ADD_LWBD(d2, i);
      u2 = ADD_UPBD(d2, i);
      /* if not constant upper bound, give up */
      if (!u1 || !A_ALIASG(u1))
        continue;
      if (A_DTYPEG(u1) != DT_INT4)
        continue;
      e1 = get_int_cval(A_SPTRG(u1));
      if (l1) {
        if (!A_ALIASG(l1))
          continue;
        if (A_DTYPEG(l1) != DT_INT4)
          continue;
        e1 -= get_int_cval(A_SPTRG(l1)) - 1;
      }
      if (!u2 || !A_ALIASG(u2))
        continue;
      if (A_DTYPEG(u2) != DT_INT4)
        continue;
      e2 = get_int_cval(A_SPTRG(u2));
      if (l2) {
        if (!A_ALIASG(l2))
          continue;
        if (A_DTYPEG(l2) != DT_INT4)
          continue;
        e2 -= get_int_cval(A_SPTRG(l2)) - 1;
      }
      /* different extents */
      if (e1 != e2)
        return FALSE;
    }
    return cmpat_dtype_with_size(DTY(d1 + 1), DTY(d2 + 1));

  case TY_DERIVED:
    /* tags are the same => equal types */
    if (DTY(d1 + 3) == DTY(d2 + 3))
      return TRUE;
    /* traverse the members */
    for (s1 = DTY(d1 + 1), s2 = DTY(d2 + 1); s1 > NOSYM && s2 > NOSYM;
         s1 = SYMLKG(s1), s2 = SYMLKG(s2)) {
      /* member types are different => different types */
      if (!cmpat_dtype_with_size(DTYPEG(s1), DTYPEG(s2)))
        return FALSE;
      if (PRIVATEG(s1) != PRIVATEG(s2))
        return FALSE;
    }
    /*  more members in either record? */
    if (s2 > NOSYM || s1 > NOSYM)
      return FALSE;
    return TRUE;

  case TY_CHAR:
  case TY_NCHAR:
    return TRUE;
  default:
    break;
  }
  return FALSE;
}

/** \brief Similar to eq_dtype(), except types must match exactly,
    and derived types must be SEQUENCE and match in name also.
    Also, array extents are checked.
 */
LOGICAL
same_dtype(DTYPE d1, DTYPE d2)
{
  int s1, s2, i, n;
  int tg1, tg2;

  if (d1 == d2)
    return TRUE;
  if (d1 <= 0 || d2 <= 0)
    return FALSE;
  if (DTY(d1) != DTY(d2))
    return FALSE;
  /* here, DTY(d1) == DTY(d2) */
  switch (DTY(d1)) {
  case TY_ARRAY:
    /* check rank, extents and element type */
    if (ADD_NUMDIM(d1) != ADD_NUMDIM(d2))
      return FALSE;
    n = ADD_NUMDIM(d1);
    for (i = 0; i < n; ++i) {
      /* check bounds */
      int l1, l2, u1, u2, e1, e2;
      l1 = ADD_LWBD(d1, i);
      u1 = ADD_UPBD(d1, i);
      l2 = ADD_LWBD(d2, i);
      u2 = ADD_UPBD(d2, i);
      /* if not constant upper bound, give up */
      if (!u1 || !A_ALIASG(u1))
        continue;
      if (A_DTYPEG(u1) != DT_INT4)
        continue;
      e1 = get_int_cval(A_SPTRG(u1));
      if (l1) {
        if (!A_ALIASG(l1))
          continue;
        if (A_DTYPEG(l1) != DT_INT4)
          continue;
        e1 -= get_int_cval(A_SPTRG(l1)) - 1;
      }
      if (!u2 || !A_ALIASG(u2))
        continue;
      if (A_DTYPEG(u2) != DT_INT4)
        continue;
      e2 = get_int_cval(A_SPTRG(u2));
      if (l2) {
        if (!A_ALIASG(l2))
          continue;
        if (A_DTYPEG(l2) != DT_INT4)
          continue;
        e2 -= get_int_cval(A_SPTRG(l2)) - 1;
      }
      /* different extents */
      if (e1 != e2)
        return FALSE;
    }
    return same_dtype(DTY(d1 + 1), DTY(d2 + 1));

  case TY_DERIVED:
    /* tags are the same => equal types */
    tg1 = DTY(d1 + 3);
    tg2 = DTY(d2 + 3);
    if (tg1 == tg2)
      return TRUE;
    /* both must be SEQUENCE or both must be BIND(C) */
    if ((SEQG(tg1) && SEQG(tg2)) || (CFUNCG(tg1) && CFUNCG(tg2)))
      ;
    else if (same_parameterized_dt(d1, d2))
      return TRUE;
    else {
      return FALSE;
    }
    if (VISITG(tg1) && VISITG(tg2)) {
      /* have a self-referential derived type */
      return TRUE;
    }
    if (VISITG(tg1) || VISITG(tg2)) {
      return FALSE;
    }
    VISITP(tg1, 1);
    VISITP(tg2, 1);
    /* traverse the members */
    for (s1 = DTY(d1 + 1), s2 = DTY(d2 + 1); s1 > NOSYM && s2 > NOSYM;
         s1 = SYMLKG(s1), s2 = SYMLKG(s2)) {
      /* neither member can be PRIVATE */
      if (PRIVATEG(s1) || PRIVATEG(s2))
        break; /* return FALSE; */
      /* member types are different => different types */
      if (!same_dtype(DTYPEG(s1), DTYPEG(s2)))
        break; /* return FALSE; */
    }
    VISITP(tg1, 0);
    VISITP(tg2, 0);
    /*  more members in either record? */
    if (s2 > NOSYM || s1 > NOSYM)
      return FALSE;
    return TRUE;

  case TY_CHAR:
  case TY_NCHAR:
    /* compare lengths */
    if (DTY(d1 + 1) == DTY(d2 + 1))
      return TRUE;
    break;

  default:
    break;
  }
  return FALSE;
}

/** \brief Similar to eq_dtype(), except all scalar integer types are compatible
    with each other, all scalar logical types are compatible with each other,
    all character types are compatible with each other and we allow
    casting arrays when types fit into each other.
 */
LOGICAL
cmpat_dtype_array_cast(DTYPE d1, DTYPE d2)
{
  if (d1 == d2)
    return TRUE;
  if (DTY(d1) != DTY(d2)) {
    /* check for any logical first since logical types also have the
     * _TY_INT attribute.
     */
    if (DT_ISLOG(d1) && DT_ISLOG(d2))
      return TRUE;
    if (DT_ISINT(d1) && DT_ISINT(d2))
      return TRUE;
    return FALSE;
  }

  /* At this place DTY(d1) = DTY(d2) */
  if (DTY(d1) == TY_CHAR || DTY(d1) == TY_NCHAR)
    return TRUE;

  /* Perform checks for array casting possibilities */
  if (DTY(d1) == TY_ARRAY) {
    DTYPE array_type_1 = DDTG(d1);
    DTYPE array_type_2 = DDTG(d2);

    /* We want to allow all casts apart from REAL to INT without further checks */
    if ((DT_ISINT(array_type_1)  && DT_ISINT(array_type_2)) ||
        (DT_ISREAL(array_type_1) && DT_ISREAL(array_type_2)) ||
        (DT_ISREAL(array_type_1) && DT_ISINT(array_type_2)))
      return true; // cngcon will later check if numbers actually fit into each other and warn if not
    else
      /* Not REAL or INT, continue normal checks */
      return cmpat_dtype(array_type_1, array_type_2);
  }

  /* Continue checks for other cases */
  return eq_dtype(d1, d2);
}

static int
priority(int op)
{
  switch (op) {
  case OP_CMP:
  case OP_AIF:
  case OP_FUNC:
  case OP_CON:
  case OP_LOG:
    /* I don't know what these are, anyway */
    return 0;
  /* missing priority 110 is for user-defined binary operators */
  case OP_LEQV:
  case OP_LNEQV:
    return 20;
  case OP_LOR:
    return 30;
  case OP_LAND:
    return 40;
  case OP_LNOT:
    return 50;
  case OP_EQ:
  case OP_GE:
  case OP_GT:
  case OP_LE:
  case OP_LT:
  case OP_NE:
    return 60;
  case OP_CAT:
    return 70;
  case OP_NEG:
  case OP_ADD:
  case OP_SUB:
    return 80;
  case OP_MUL:
  case OP_DIV:
    return 90;
  case OP_XTOI:
  case OP_XTOX:
    return 101; /* right associative */
  /* missing priority 110 is for user-defined unary operators */
  case OP_LD:
  case OP_ST:
  case OP_LOC:
  case OP_REF:
  case OP_VAL:
  case OP_BYVAL:
  case OP_SCAND:
    return 120;
  }
  interr("priority. Unexpected op", op, 2);
  return 0;
} /* priority */

static int
leftparens(int ast, int astleft)
{
  int prio, prioleft;
  if (A_TYPEG(ast) != A_BINOP && A_TYPEG(ast) != A_UNOP) {
    return FALSE;
  }
  if (A_TYPEG(astleft) != A_BINOP && A_TYPEG(astleft) != A_UNOP) {
    return FALSE;
  }
  prio = priority(A_OPTYPEG(ast));
  prioleft = priority(A_OPTYPEG(astleft));
  if (prio < prioleft)
    return FALSE;
  if (prioleft < prio)
    return TRUE;
  /* if the same priority, check if the operator is right-associative */
  if (prio & 0x1)
    return TRUE;
  return FALSE;
} /* leftparens */

static int
rightparens(int ast, int astright)
{
  int prio, prioright;
  if (A_TYPEG(ast) != A_BINOP && A_TYPEG(ast) != A_UNOP) {
    return FALSE;
  }
  if (A_TYPEG(astright) != A_BINOP && A_TYPEG(astright) != A_UNOP) {
    return FALSE;
  }
  prio = priority(A_OPTYPEG(ast));
  prioright = priority(A_OPTYPEG(astright));
  if (prio < prioright)
    return FALSE;
  if (prioright < prio)
    return TRUE;
  /* if the same priority, check if the operator is right-associative */
  if (prio & 0x1)
    return FALSE;
  return TRUE;
} /* rightparens */

static void
getop(int op, char *string)
{
  const char *s;
  switch (op) {
  case OP_CMP:
    s = ".cmp.";
    break;
  case OP_AIF:
    s = ".aif.";
    break;
  case OP_FUNC:
    s = ".func.";
    break;
  case OP_CON:
    s = ".con.";
    break;
  case OP_LOG:
    s = ".aif.";
    break;
  case OP_LEQV:
    s = ".eqv.";
    break;
  case OP_LNEQV:
    s = ".neqv.";
    break;
  case OP_LOR:
    s = ".or.";
    break;
  case OP_LAND:
    s = ".and.";
    break;
  case OP_LNOT:
    s = ".not.";
    break;
  case OP_EQ:
    s = ".eq.";
    break;
  case OP_GE:
    s = ".ge.";
    break;
  case OP_GT:
    s = ".gt.";
    break;
  case OP_LE:
    s = ".le.";
    break;
  case OP_LT:
    s = ".lt.";
    break;
  case OP_NE:
    s = ".ne.";
    break;
  case OP_CAT:
    s = "//";
    break;
  case OP_NEG:
    s = "-";
    break;
  case OP_ADD:
    s = "+";
    break;
  case OP_SUB:
    s = "-";
    break;
  case OP_MUL:
    s = "*";
    break;
  case OP_DIV:
    s = "/";
    break;
  case OP_XTOI:
    s = "**";
    break;
  case OP_XTOX:
    s = "**";
    break;
  case OP_LD:
    s = ".load.";
    break;
  case OP_ST:
    s = ".store.";
    break;
  case OP_LOC:
    s = ".loc.";
    break;
  case OP_REF:
    s = ".ref.";
    break;
  case OP_VAL:
    s = ".val.";
    break;
  case OP_BYVAL:
    s = "(byval)";
    break;
  case OP_SCAND:
    s = ".scand.";
    break;
  default:
    s = ".??.";
    break;
  }
  strcat(string, s);
} /* getop */

/** \brief Given an AST and a string pointer, append a printable representation
    of the AST expression onto the string */
void
getast(int ast, char *string)
{
  int asd, ndim, i, lp, rp;
  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_CNST:
    strcat(string, getprint(sym_of_ast(ast)));
    if (DBGBIT(5, 0x40) && A_TYPEG(ast) == A_ID) {
      char b[64];
      sprintf(b, "\\%d", sym_of_ast(ast));
      strcat(string, b);
    }
    break;
  case A_SUBSCR:
    /*strcat( string, getprint(sym_of_ast(ast)) );*/
    getast((int)A_LOPG(ast), string);
    strcat(string, "(");
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      if (i)
        strcat(string, ",");
      getast((int)ASD_SUBS(asd, i), string);
    }
    strcat(string, ")");
    break;
  case A_MEM:
    getast((int)A_PARENTG(ast), string);
    strcat(string, "%");
    getast((int)A_MEMG(ast), string);
    break;
  case A_BINOP:
    lp = rp = 0;
    if (leftparens(ast, A_LOPG(ast))) {
      strcat(string, "(");
      lp = 1;
    }
    getast((int)A_LOPG(ast), string);
    if (lp)
      strcat(string, ")");
    getop(A_OPTYPEG(ast), string);
    if (rightparens(ast, A_ROPG(ast))) {
      strcat(string, "(");
      rp = 1;
    }
    getast((int)A_ROPG(ast), string);
    if (rp)
      strcat(string, ")");
    break;
  case A_UNOP:
    rp = 0;
    getop(A_OPTYPEG(ast), string);
    if (rightparens(ast, A_LOPG(ast))) {
      strcat(string, "(");
      rp = 1;
    }
    getast((int)A_LOPG(ast), string);
    if (rp)
      strcat(string, ")");
    break;
  case A_PAREN:
    strcat(string, "(");
    getast((int)A_LOPG(ast), string);
    strcat(string, ")");
    break;
  case A_CONV:
    strcat(string, "conv(");
    getast((int)A_LOPG(ast), string);
    strcat(string, ")");
    break;
  case A_CMPLXC:
    strcat(string, "(");
    getast((int)A_LOPG(ast), string);
    strcat(string, ",");
    getast((int)A_ROPG(ast), string);
    strcat(string, ")");
    break;
  default:
    strcat(string, "??");
    break;
  } /* switch */
} /* getast */

/** \brief Check if ast is deferred-length character  */
bool
is_deferlenchar_ast(int ast)
{
  DTYPE dt;
  SPTR sym = 0;

  dt = DDTG(A_DTYPEG(ast));
  if (DTY(dt) != TY_CHAR && DTY(dt) != TY_NCHAR) {
    return false;
  }

  if (dt ==  DT_ASSCHAR || dt ==  DT_ASSNCHAR) {
    return false;
  } else if (dt == DT_DEFERCHAR || dt == DT_DEFERNCHAR) {
    return true;
  }

  if (ast_is_sym(ast)) {
    sym = memsym_of_ast(ast);
  }

  /* adjustable length character */
  if ((sym > NOSYM) && ADJLENG(sym)) {
    return false;
  }

  if (DTY(A_DTYPEG(ast)) == TY_ARRAY) {
    if (ADD_DEFER(A_DTYPEG(ast))) {
      dt = DTY(DDTG(A_DTYPEG(ast)) + 1);
      if (A_TYPEG(dt) != A_CNST) {
        return true;
      }
    }
  }
  return false;
}

/** \brief Check if dtype is deferred-length character */
bool
is_deferlenchar_dtype(DTYPE dtype)
{
  DTYPE dt;

  dt = DDTG(dtype);
  if (DTY(dt) != TY_CHAR && DTY(dt) != TY_NCHAR) {
    return false;
  }

  if (dt == DT_DEFERCHAR || dt == DT_DEFERNCHAR) {
    return true;
  }
  dt = DTY(dt+1);
  if (DTY(dtype) == TY_ARRAY) {
    if (!ADD_DEFER(dtype)) {
      return false;
    }
  }

  if (A_TYPEG(dt) == A_ID) {
    /* i.e. character(len=newlen) */
    if (ASSNG(A_SPTRG(dt))) {
      return true;
    }
  } else if (A_TYPEG(dt) == A_SUBSCR) {
    /* i.e. character(len=newlen(1)) */
    if (ASSNG(memsym_of_ast(dt))) {
      return true;
    }
  }

  /* i.e. character(len=len(a)) */
  if ((A_TYPEG(dt) == A_FUNC || A_TYPEG(dt) == A_INTR)
    && is_deferlenchar_ast(ARGT_ARG(A_ARGSG(dt), 0))) {
    return true;
  }
  return false;
}


/** \brief Put into the character array pointed to by ptr, the print
   representation
    of dtype.
 */
void
getdtype(DTYPE dtype, char *ptr)
{
  int i;
  ADSC *ad;
  int numdim;
  char *p;

  p = ptr;
  *p = 0;
  for (; dtype != 0 && p - ptr <= 150; dtype = DTY(dtype + 1)) {
    if (dtype <= 0 || dtype >= stb.dt.stg_avail) {
      sprintf(p, "bad dtype(%d)", dtype);
      break;
    }
    if (DTY(dtype) <= 0 || DTY(dtype) > TY_MAX) {
      sprintf(p, "bad dtype(%d[%d])", dtype, (int)DTY(dtype));
      break;
    }
    strcpy(p, stb.tynames[DTY(dtype)]);
    p += strlen(p);

    switch (DTY(dtype)) {
    case TY_STRUCT:
    case TY_UNION:
    case TY_DERIVED:
      i = DTY(dtype + 3);
      if (i) {
        if (i <= NOSYM || i >= stb.stg_avail) {
          sprintf(p, "/bad tag=%d/", i);
        } else {
          sprintf(p, "/%s/", SYMNAME(i));
        }
        p += strlen(p);
      }
      return;

    case TY_ARRAY:
      if (DTY(dtype + 2) == 0) {
        *p++ = ' ';
        *p++ = '(';
      } else {
        ad = AD_DPTR(dtype);
        numdim = AD_NUMDIM(ad);
        if (numdim < 1 || numdim > 7) {
          sprintf(p, "ndim=%d", numdim);
          numdim = 0;
          p += strlen(p);
        }
        if (AD_DEFER(ad)) {
          strcpy(p, " deferred");
          p += strlen(p);
        }
        if (AD_ASSUMSHP(ad) == 1) {
          strcpy(p, " assumedshape");
          p += strlen(p);
        }
        if (AD_ASSUMRANK(ad) == 1) {
          strcpy(p, " assumedrank");
          p += strlen(p);
        }
        if (AD_ASSUMSHP(ad) == 2) {
          strcpy(p, " wasassumedshape");
          p += strlen(p);
        }
        if (AD_ADJARR(ad)) {
          strcpy(p, " adjustable");
          p += strlen(p);
        }
        if (AD_ASSUMSZ(ad)) {
          strcpy(p, " assumedsize");
          p += strlen(p);
        }
        if (AD_NOBOUNDS(ad)) {
          strcpy(p, " nobounds");
          p += strlen(p);
        }
        *p++ = ' ';
        *p++ = '(';
        for (i = 0; i < numdim; i++) {
          if (i)
            *p++ = ',';
          if (AD_LWAST(ad, i)) {
            *p = '\0';
            getast(AD_LWAST(ad, i), p);
            p += strlen(p);
            if (AD_LWBD(ad, i) != AD_LWAST(ad, i)) {
              *p++ = '[';
              if (AD_LWBD(ad, i)) {
                *p = '\0';
                getast(AD_LWBD(ad, i), p);
                p += strlen(p);
              }
              *p++ = ']';
            }
            *p++ = ':';
          } else if (AD_LWBD(ad, i)) {
            *p++ = '[';
            *p = '\0';
            getast(AD_LWBD(ad, i), p);
            p += strlen(p);
            *p++ = ']';
            *p++ = ':';
          }
          if (AD_UPAST(ad, i)) {
            *p = '\0';
            getast(AD_UPAST(ad, i), p);
            p += strlen(p);
          } else {
            *p++ = '*';
          }
          if (AD_UPBD(ad, i) != AD_UPAST(ad, i)) {
            *p++ = '[';
            if (AD_UPBD(ad, i)) {
              *p = '\0';
              getast(AD_UPBD(ad, i), p);
              p += strlen(p);
            }
            *p++ = ']';
          }
        }
      }
      strcpy(p, ") of ");
      p += 5;
      break;

    case TY_PTR:
      *p++ = ' ';
      break;

    case TY_CHAR:
    case TY_NCHAR:
      if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR)
        sprintf(p, "*(*)");
      else if (dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR) {
        sprintf(p, "*(:)");
      } else {
        sprintf(p, "*");
        p += strlen(p);
        *p = '\0';
        getast(DTY(dtype + 1), p);
      }
      return;

    default:
      return;
    }
  }
}

void
dmp_dtype(void)
{
  int i;

  fprintf(gbl.dbgfil, "\n------------------------\nDTYPE DUMP:\n");
  fprintf(gbl.dbgfil, "\ndt_base: %lx   dt_size: %d   dt_avail: %d\n\n",
          (long)(stb.dt.stg_base), stb.dt.stg_size, stb.dt.stg_avail);
  i = 1;
  fprintf(gbl.dbgfil, "index   dtype\n");
  while (i < stb.dt.stg_avail) {
    i += dmp_dent(i);
  }
  fprintf(gbl.dbgfil, "\n------------------------\n");
}

int
dlen(int ty)
{
  switch (ty) {
  case TY_NONE:
  case TY_WORD:
  case TY_DWORD:
  case TY_HOLL:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_INT8:
  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
  case TY_NUMERIC:
  case TY_ANY:
  case TY_128:
  case TY_256:
  case TY_512:
  case TY_INT128:
  case TY_LOG128:
  case TY_FLOAT128:
  case TY_CMPLX128:
    return 1;

  case TY_CHAR:
  case TY_NCHAR:
  case TY_PTR:
    return 2;

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    return 6;

  case TY_ARRAY:
    return 3;

  case TY_PROC:
    return 6;

  default:
    return 1;
  }
} /* dlen */

int
_dmp_dent(DTYPE dtypeind, FILE *outfile)
{
  char buf[1024];
  int retval;
  ADSC *ad;
  int numdim;
  int i;
  int paramct, dpdsc;

  if (outfile == NULL)
    outfile = stderr;

  if (dtypeind < 1 || dtypeind >= stb.dt.stg_avail) {
    fprintf(outfile, "dtype index (%d) out of range in dmp_dent\n", dtypeind);
    return 1;
  }
  buf[0] = '\0';
  fprintf(outfile, " %5d  ", dtypeind);
  switch (DTY(dtypeind)) {
  case TY_WORD:
  case TY_DWORD:
  case TY_HOLL:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_NUMERIC:
  case TY_ANY:
  case TY_INT8:
  case TY_LOG8:
  case TY_128:
  case TY_256:
  case TY_512:
  case TY_INT128:
  case TY_LOG128:
  case TY_FLOAT128:
  case TY_CMPLX128:
    retval = 1;
    break;

  case TY_CHAR:
  case TY_NCHAR:
    retval = 2;
    break;

  case TY_PTR:
    fprintf(outfile, "ptr     dtype=%5d\n        ", (int)DTY(dtypeind + 1));
    retval = 2;
    break;

  case TY_ARRAY:
    retval = 3;
    fprintf(outfile, "array   dtype=%5d   desc   =%" ISZ_PF "d\n",
            (int)DTY(dtypeind + 1), DTY(dtypeind + 2));
    if (DTY(dtypeind + 2) == 0) {
      fprintf(outfile, "        (No array desc)\n        ");
      break;
    }
    ad = AD_DPTR(dtypeind);
    numdim = AD_NUMDIM(ad);
    fprintf(outfile,
            "        numdim:%d  defer:%d  adjarr:%d  assumz:%d  nobounds:%d",
            numdim, AD_DEFER(ad), AD_ADJARR(ad), AD_ASSUMSZ(ad),
            AD_NOBOUNDS(ad));
    fprintf(outfile, "  assumshp:%d\n", AD_ASSUMSHP(ad));
    fprintf(outfile, "  assumrank:%d\n", AD_ASSUMRANK(ad));
    fprintf(outfile, "        zbase: %d   numelm: %d\n", AD_ZBASE(ad),
            AD_NUMELM(ad));
    if (numdim < 1 || numdim > 7)
      numdim = 0;
    for (i = 0; i < numdim; i++)
      fprintf(outfile, "        %1d:  mlpyr: %d  lwbd: %d  upbd: %d"
                       "  lwast: %d  upast: %d  extntast: %d\n",
              i + 1, AD_MLPYR(ad, i), AD_LWBD(ad, i), AD_UPBD(ad, i),
              AD_LWAST(ad, i), AD_UPAST(ad, i), AD_EXTNTAST(ad, i));
    break;
  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    fprintf(outfile, "%s  sptr =%5d   size  =%" ISZ_PF "d",
            stb.tynames[DTY(dtypeind)], (int)DTY(dtypeind + 1),
            DTY(dtypeind + 2));
    fprintf(outfile, "   tag=%5d   align=%3d", (int)DTY(dtypeind + 3),
            (int)DTY(dtypeind + 4));
    fprintf(outfile, "   ict=%08lx\n        ",
            (long)(get_getitem_p(DTY(dtypeind + 5))));
    retval = 6;
    break;
  case TY_PROC:
    paramct = DTY(dtypeind + 3);
    dpdsc = DTY(dtypeind + 4);
    fprintf(outfile, "proc    dtype=%5" BIGIPFSZ "d  interface=%5" BIGIPFSZ
                     "d  paramct=%3d"
                     "  dpdsc=%5d  fval=%5" BIGIPFSZ "d\n",
            DTY(dtypeind + 1), DTY(dtypeind + 2), paramct, dpdsc,
            DTY(dtypeind + 5));
    for (i = 0; i < paramct; i++) {
      fprintf(outfile, "     arg %d: %d\n", i + 1, aux.dpdsc_base[dpdsc + i]);
    }
    retval = 6;
    break;
  default:
    /* function param thing ?? */
    fprintf(outfile, "????  %5d\n", (int)DTY(dtypeind));
    retval = 1;
    dtypeind = 0;
    break;
  }
  if (dtypeind) {
    getdtype(dtypeind, buf);
    fprintf(outfile, "%s\n", buf);
  }
  return retval;
}

int
dmp_dent(DTYPE dtypeind)
{

  FILE *outfile;
  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }
  return _dmp_dent(dtypeind, outfile);
}

void
pr_dent(DTYPE dt, FILE *f)
{
  int ss;
  if (f == NULL)
    f = stderr;
  if (dt < 1 || dt >= stb.dt.stg_avail) {
    fprintf(f, "dtype index (%d) out of range in pr_dent\n", dt);
    return;
  }
  _dmp_dent(dt, f);
  switch (DTY(dt)) {
  case TY_PTR:
    pr_dent(DTY(dt + 1), f);
    break;
  case TY_DERIVED:
    for (ss = DTY(dt + 1); ss > NOSYM; ss = SYMLKG(ss)) {
      fprintf(f, " +++++ MEMBER %d(%s)\n", ss, SYMNAME(ss));
      pr_dent(DTYPEG(ss), f);
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }
}

#if DEBUG
void
dumpdtype(DTYPE dtype)
{
  dmp_dent(dtype);
} /* dumpdtype */
#endif

/** \brief Compute the size of a data type.

    This machine dependent routine computes the size of a data type
    in terms of two quantities:
        - size  - number of elements in the data type (returned thru size).
        - scale - number of bytes in each element, expressed as a power
                  of two (the return value of scale_of).

    This routine will be used to take advantage of the machines that
    have the ability to add a scaled expression (multiplied by a power
    of two) to an address.  This is particularly useful for incrementing
    a pointer variable and array subscripting.

    Note that for those machines that do not have this feature, scale_of
    returns a scale of 0 and size_of for size.
 */
int
scale_of(DTYPE dtype, INT *size)
{
  INT d;
  int tmp;
  int scale, clen;
  INT tmpsiz;
  TY_KIND ty = get_ty_kind(dtype);

  switch (ty) {
  case TY_WORD:
  case TY_DWORD:
  case TY_LOG:
  case TY_INT:
  case TY_FLOAT:
  case TY_PTR:
  case TY_SLOG:
  case TY_SINT:
  case TY_BINT:
  case TY_BLOG:
  case TY_DBLE:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_INT8:
  case TY_LOG8:
    scale = dtypeinfo[ty].scale;
    *size = (unsigned)dtypeinfo[ty].size >> scale;
    return scale;

  case TY_HOLL:
  case TY_CHAR:
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR)
      interr("scale_of: attempt to size assumed size character", 0, 3);
    clen = string_length(dtype);
    *size = clen;
    return 0;

  case TY_NCHAR:
    if (dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR)
      interr("scale_of: attempt to size assumed size ncharacter", 0, 3);
    clen = string_length(dtype);
    *size = 2 * clen;
    return 0;

  case TY_ARRAY:
    if ((d = DTY(dtype + 2)) <= 0) {
      interr("scale_of: no array descriptor", (int)d, 3);
      d = DTY(dtype + 2) = 1;
    }
    tmp = scale_of((int)DTY(dtype + 1), &tmpsiz);
    *size = d * tmpsiz;
    return tmp;

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    if (DTY(dtype + 2) <= 0) {
      interr("scale_of: 0 size struct", 0, 3);
      *size = 4;
      return 0;
    } else {
      *size = DTY(dtype + 2);
      return 0;
    }

  default:
    interr("scale_of: bad dtype", ty, 3);
    *size = 1;
    return 0;
  }
}

/** \brief Return 0 if reg, 1 if mem.  */
int
fval_of(DTYPE dtype)
{
  TY_KIND ty = get_ty_kind(dtype);
  int fv = dtypeinfo[ty].fval & 0x3;
  assert(fv <= 1, "fval_of: bad dtype, dt is", dtype, 3);
  return fv;
}

static TY_KIND
get_ty_kind(DTYPE dtype)
{
  assert(dtype > 0 && dtype < stb.dt.stg_avail, "bad dtype", dtype, ERR_Severe);
  return DTY(dtype);
}

#define SS2 0x8e
#define SS3 0x8f

/** \brief Return number of kanji characters
    \param p character string
    \param len length in bytes of \p p
 */
int
kanji_len(const unsigned char *p, int len)
{
  int count = 0;
  int val;

  while (len > 0) {
    val = *p;
    count++;
    if ((val & 0x80) == 0 || len <= 1) /* ASCII */
      len--, p++;
    else if (val == SS2) /* JIS 8-bit character */
      len -= 2, p += 2;
    else if (val == SS3 && len >= 3) /* Graphic Character */
      len -= 3, p += 3;
    else /* Kanji */
      len -= 2, p += 2;
  }

  return count;
}

/** \brief Extract necessary bytes from character string in order to return
    integer (16-bit) representation of one kanji char.
    \param p     ptr to EUC string
    \param len   number of bytes in \p p
    \param bytes return number of EUC bytes used up
*/
int
kanji_char(unsigned char *p, int len, int *bytes)
{
  int val = *p;

  if ((val & 0x80) == 0 || len <= 1) /* ASCII */
    *bytes = 1;
  else if (val == SS2) /* JIS 8-bit character */
    *bytes = 2, val = *(p + 1);
  else if (val == SS3 && len >= 3) /* Graphic Character */
    *bytes = 3, val = ((*(p + 1) << 8) | (*(p + 2) & 0x7F));
  else /* Kanji */
    *bytes = 2, val = ((val << 8) | *(p + 1));

  return val;
}

/** \brief Return number of bytes required for newlen chars.
    \param p      ptr to EUC string
    \param newlen number of kanji chars required from string prefix
    \param len    total number of bytes in string
 */
int
kanji_prefix(unsigned char *p, int newlen, int len)
{
  unsigned char *begin;
  int bytes;

  begin = p;
  while (newlen-- > 0) {
    (void)kanji_char(p, len, &bytes);
    p += bytes;
    len -= bytes;
  }

  return (p - begin);
}

#define _FP 4
#define _VP 6
/** \brief Create a dtype record for an array of rank numdim including its array
    descriptor.
    \param numdim number of dimensions
    \param eltype data type of the array element

    The layout of an array descriptor is:
    <pre>
        int    numdim;  --+
        int    zbase;     |
        UINT16 *ilmp;     |
        char   defer;     +-- 4 ints (fixed part)
        char   adjarr;    |
        char   assumsz;   |
        char   pad;     --+
        struct {
            int mlpyr;  --+
            int lwbd;     |
            int upbd;     +-- 6 ints (variable part)
            int lwast;    |
            int upast;    |
            int exntnast;-+
        } b[numdim];
        int    numelm;  --+-- 1 int
    </pre>
    Any change in the size of the structure requires a change to one or both
    of the macros _FP and _VP.  Also the size assertion in symtab.c needs
    to be changed.
 */
DTYPE
get_array_dtype(int numdim, DTYPE eltype)
{
  DTYPE dtype;

  dtype = get_type(3, TY_ARRAY, eltype);
  get_aux_arrdsc(dtype, numdim);

  return dtype;
}

void
get_aux_arrdsc(DTYPE dtype, int numdim)
{
  ADSC *ad;

  DTY(dtype + 2) = aux.arrdsc_avl;
  aux.arrdsc_avl += (_FP + 1) + (_VP * numdim);
  NEED(aux.arrdsc_avl, aux.arrdsc_base, int, aux.arrdsc_size,
       aux.arrdsc_avl + 240);
  ad = AD_DPTR(dtype);
  BZERO(ad, int, (_FP + 1) + _VP * numdim);
  AD_NUMDIM(ad) = numdim;
}

/** \brief Duplicate a dtype array record and its array descriptor.
 */
DTYPE
dup_array_dtype(DTYPE o_dt)
{
  ADSC *ad;
  ADSC *o_ad = AD_DPTR(o_dt);
  int numdim = AD_NUMDIM(o_ad);
  DTYPE dtype = get_type(3, TY_ARRAY, DT_NONE);

  DTY(dtype + 2) = aux.arrdsc_avl;
  aux.arrdsc_avl += (_FP + 1) + (_VP * numdim);
  NEED(aux.arrdsc_avl, aux.arrdsc_base, int, aux.arrdsc_size,
       aux.arrdsc_avl + 240);
  o_ad = AD_DPTR(o_dt); /* recreate pointer after possible realloc */
  ad = AD_DPTR(dtype);

  BCOPY(ad, o_ad, int, (_FP + 1) + _VP * numdim);

  /* make it the same element type; the caller may change the type */
  DTY(dtype + 1) = DTY(o_dt + 1);
  return dtype;
}

/** \brief Duplicate a dtype array record and its array descriptor, excluding a
    dimension.
    \param o_dt old array dtype
    \param elem_dt element dtype
    \param astdim ast of dimension to be excluded
    \param after std after which code is produced to create the bounds
        descriptor (if dim is not a constant)
 */
DTYPE
reduc_rank_dtype(DTYPE o_dt, DTYPE elem_dt, int astdim, int after)
{
  DTYPE dtype;
  int numdim;
  int o_numdim;
  int dim;
  ADSC *o_ad;
  ADSC *ad;
  int i, j;

  o_ad = AD_DPTR(o_dt);
  o_numdim = AD_NUMDIM(o_ad);
  numdim = o_numdim - 1;
  if (numdim <= 0)
    return DTY(o_dt + 1);

  dtype = get_type(3, TY_ARRAY, elem_dt);
  DTY(dtype + 2) = aux.arrdsc_avl;
  aux.arrdsc_avl += (_FP + 1) + (_VP * numdim);
  NEED(aux.arrdsc_avl, aux.arrdsc_base, int, aux.arrdsc_size,
       aux.arrdsc_avl + 240);
  o_ad = AD_DPTR(o_dt); /* recreate pointer after possible realloc */
  ad = AD_DPTR(dtype);
  if (A_ALIASG(astdim) == 0) {
    error(422, 3, gbl.lineno, NULL, NULL);
    dim = 1;
    /* TBD insert code after 'after' to compute bound excluding dim
          at run-time */
  } else {
    /* dim is a constant */

    dim = CONVAL2G(A_SPTRG(A_ALIASG(astdim)));
    if (dim < 1 || dim > o_numdim) {
      error(423, 3, gbl.lineno, NULL, NULL);
      dim = 1;
    }
  }
  ad = AD_DPTR(dtype);
  BZERO(ad, int, (_FP + 1) + _VP * numdim);
  j = 0;
  for (i = 0; i < o_numdim; i++)
    if (i != dim - 1) {
      AD_LWBD(ad, j) = AD_LWBD(o_ad, i);
      AD_UPBD(ad, j) = AD_UPBD(o_ad, i);
      AD_LWAST(ad, j) = AD_LWAST(o_ad, i);
      AD_UPAST(ad, j) = AD_UPAST(o_ad, i);
      AD_EXTNTAST(ad, j) = AD_EXTNTAST(o_ad, i);
      j++;
    }
  AD_NUMDIM(ad) = numdim;

  return dtype;
}

/** \brief Return number of dimensions of array dtype */
int
rank_of(DTYPE dtype)
{

#if DEBUG
  assert(dtype != DT_NONE, "rank_of:DT_NONE", dtype, 2);
#endif
  if (DTY(dtype) != TY_ARRAY) {
    /* must be scalar */;
    return 0;
  }
  if (DTY(dtype + 2) == 0) {
    interr("rank_of: no array descriptor", dtype, 2);
    return 1;
  }
  return AD_NUMDIM(AD_DPTR(dtype));
}

/** \brief Return number of dimensions of symbol sptr */
int
rank_of_sym(int sptr)
{
  return rank_of((int)DTYPEG(sptr));
}

/** \brief Return AST representing the lower bound of array dtype for
    dimension dim (dim is relative to 0).
 */
int
lbound_of(DTYPE dtype, int dim)
{
  int rank;
  ADSC *ad;
  int ast;

#if DEBUG
  assert(DTY(dtype) == TY_ARRAY, "lbound_of: not array", dtype, 0);
  assert(DTY(dtype + 2), "lbound_of: no arrdsc", dtype, 0);
#endif

  rank = rank_of(dtype);
#if DEBUG
  assert(dim >= 0 && dim < rank, "lbound_of: illegal dimension", dim, 0);
#endif

  ad = AD_DPTR(dtype);

  ast = AD_LWAST(ad, dim);

  if (AD_ASSUMSHP(ad) == 1) {
    int lwb1;
    lwb1 = AD_LWBD(ad, dim);
    if (A_TYPEG(lwb1) == A_CNST)
      ast = lwb1;
  }
  if (ast == 0)
    ast = astb.bnd.one;
  return ast;
}

/** \brief Return AST representing the lower bound of array symbol for
    dimension dim (dim is relative to 0).
 */
int
lbound_of_sym(int sptr, int dim)
{
  return lbound_of((int)DTYPEG(sptr), dim);
}

/** \brief Return AST representing the upper bound of array dtype for
    dimension dim (dim is relative to 0).
 */
int
ubound_of(DTYPE dtype, int dim)
{
  int rank;
  ADSC *ad;
  int ast;

#if DEBUG
  assert(DTY(dtype) == TY_ARRAY, "ubound_of: not array", dtype, 0);
  assert(DTY(dtype + 2), "ubound_of: no arrdsc", dtype, 0);
#endif

  rank = rank_of(dtype);
#if DEBUG
  assert(dim >= 0 && dim < rank, "ubound_of: illegal dimension", dim, 0);
#endif

  ad = AD_DPTR(dtype);

  ast = AD_UPAST(ad, dim);
  if (ast == 0) {
    interr("ubound_of: *dim", dtype, 3);
    ast = astb.bnd.one;
  }
  return ast;
}

/** \brief Return AST representing the upper bound of array symbol for
    dimension dim (dim is relative to 0).
 */
int
ubound_of_sym(int sptr, int dim)
{
  return ubound_of((int)DTYPEG(sptr), dim);
}

/** \brief Return true if the data types for two arrays are conformable
    (have the same shape).  Shape is defined to be the rank and
    the extents of each dimension.
 */
LOGICAL
conformable(DTYPE d1, DTYPE d2)
{
  int ndim;
  int i;
  int bnd;
  INT lb1, lb2;
  INT ub1, ub2;
  ADSC *ad1, *ad2;

  ad1 = AD_DPTR(d1);
  ad2 = AD_DPTR(d2);
  ndim = AD_NUMDIM(ad1);
  if (ndim != AD_NUMDIM(ad2))
    return FALSE;

  for (i = 0; i < ndim; i++) {
    bnd = AD_LWAST(ad1, i);
    if (bnd) {
      bnd = A_ALIASG(bnd);
      if (bnd == 0)
        continue; /* nonconstant bound => skip this dimension */
      lb1 = get_int_cval(A_SPTRG(bnd));
    } else {
      lb1 = 1; /* no lower bound => 1 */
    }

    bnd = AD_UPAST(ad1, i);
    if (bnd) {
      bnd = A_ALIASG(bnd);
      if (bnd == 0)
        continue; /* nonconstant bound => skip this dimension */
      ub1 = get_int_cval(A_SPTRG(bnd));
    } else {
      continue; /* no upper bound => skip this dimension */
    }

    bnd = AD_LWAST(ad2, i);
    if (bnd) {
      bnd = A_ALIASG(bnd);
      if (bnd == 0)
        continue; /* nonconstant bound => skip this dimension */
      lb2 = get_int_cval(A_SPTRG(bnd));
    } else {
      lb2 = 1; /* no lower bound => 1 */
    }

    bnd = AD_UPAST(ad2, i);
    if (bnd) {
      bnd = A_ALIASG(bnd);
      if (bnd == 0)
        continue; /* nonconstant bound => skip this dimension */
      ub2 = get_int_cval(A_SPTRG(bnd));
    } else {
      continue; /* no upper bound => skip this dimension */
    }

    /* upper and lower bounds in this dimension are constants */

    if ((ub1 - lb1) != (ub2 - lb2))
      return FALSE;
  }

  return TRUE;
}

/* Define mapping from compiler ty entries to type values used by the library */

typedef enum {
  __NONE = 0,        /*   type of an absent optional argument */
  __SHORT = 1,       /* C   signed short */
  __USHORT = 2,      /* C   unsigned short */
  __INT = 3,         /* C   signed int */
  __UINT = 4,        /* C   unsigned int */
  __LONG = 5,        /* C   signed long int */
  __ULONG = 6,       /* C   unsigned long int */
  __FLOAT = 7,       /* C   float */
  __DOUBLE = 8,      /* C   double */
  __CPLX = 9,        /*   F complex*8 (2x real*4) */
  __DCPLX = 10,      /*   F complex*16 (2x real*8) */
  __CHAR = 11,       /* C   signed char */
  __UCHAR = 12,      /* C   unsigned char */
  __LONGDOUBLE = 13, /* C   long double */
  __STR = 14,        /*   F character */
  __LONGLONG = 15,   /* C   long long */
  __ULONGLONG = 16,  /* C   unsigned long long */
  __BLOG = 17,       /*   F logical*1 */
  __SLOG = 18,       /*   F logical*2 */
  __LOG = 19,        /*   F logical*4 */
  __LOG8 = 20,       /*   F logical*8 */
  __WORD = 21,       /*   F typeless */
  __DWORD = 22,      /*   F double typeless */
  __NCHAR = 23,      /*   F ncharacter - kanji */

  /* new fortran data types */
  __INT2 = 24,    /*   F integer*2 */
  __INT4 = 25,    /*   F integer*4, integer */
  __INT8 = 26,    /*   F integer*8 */
  __REAL2 = 45,   /*   F real*2, half */
  __REAL4 = 27,   /*   F real*4, real */
  __REAL8 = 28,   /*   F real*8, double precision */
  __REAL16 = 29,  /*   F real*16 */
  __CPLX32 = 30,  /*   F complex*32 (2x real*16) */
  __WORD16 = 31,  /*   F quad typeless */
  __INT1 = 32,    /*   F byte (integer*1) */
  __DERIVED = 33, /*   F90 derived type */

  /* run-time descriptor types */

  __PROC = 34, /* processors descriptor */
  __DESC = 35, /* template/array/section descriptor */
  __SKED = 36, /* communication schedule */

  /* more new fortran data types */

  __M128 = 37,    /* 128-bit type */
  __M256 = 38,    /* 256-bit type */
  __INT16 = 39,   /* F integer(16) */
  __LOG16 = 40,   /* F logical(16) */
  __QREAL16 = 41, /* F real(16) */
  __QCPLX32 = 42, /* F complex(32) */
  __POLY = 43,    /* F polymorphic variable */
  __PROCPTR = 44, /* F procedure pointer descriptor */

  /* number of data types */
  __NTYPES = 46 /* MUST BE LAST */

} _pghpf_type;

int ty_to_lib[] = {
    __NONE,    /* TY_NONE */
    __WORD,    /* TY_WORD */
    __DWORD,   /* TY_DWORD */
    __NONE,    /* TY_HOLL */
    __INT1,    /* TY_BINT */
    __INT2,    /* TY_SINT */
    __INT4,    /* TY_INT */
    __INT8,    /* TY_INT8 */
    __REAL2,   /* TY_HALF */
    __REAL4,   /* TY_REAL */
    __REAL8,   /* TY_DBLE */
    __REAL16,  /* TY_QUAD */
    __CPLX,    /* TY_HCMPLX */
    __CPLX,    /* TY_CMPLX */
    __DCPLX,   /* TY_DCMPLX */
    __CPLX32,  /* TY_QCMPLX */
    __BLOG,    /* TY_BLOG */
    __SLOG,    /* TY_SLOG */
    __LOG,     /* TY_LOG */
    __LOG8,    /* TY_LOG8 */
    __STR,     /* TY_CHAR */
    __NCHAR,   /* TY_NCHAR */
    __NONE,    /* TY_PTR */
    __NONE,    /* TY_ARRAY */
    __NONE,    /* TY_STRUCT */
    __NONE,    /* TY_UNION */
    __DERIVED, /* TY_DERIVED */
    __NONE,    /* TY_NUMERIC */
    __NONE,    /* TY_ANY */
    __NONE,    /* TY_PROC */
    __M128,    /* TY_128 */
    __M256,    /* TY_256 */
    __NONE,    /* TY_512 */
    __INT16,   /* TY_INT128 */
    __LOG16,   /* TY_LOG128 */
    __QREAL16, /* TY_FLOAT128 */
    __QCPLX32, /* TY_CMPLX128 */
};

static int ty_to_base_ty[] = {
    TY_NONE,    /* TY_NONE */
    TY_WORD,    /* TY_WORD */
    TY_DWORD,   /* TY_DWORD */
    TY_HOLL,    /* TY_HOLL */
    TY_INT,     /* TY_BINT */
    TY_INT,     /* TY_SINT */
    TY_INT,     /* TY_INT */
    TY_INT,     /* TY_INT8 */
    TY_REAL,    /* TY_HALF */
    TY_REAL,    /* TY_REAL */
    TY_REAL,    /* TY_DBLE */
    TY_REAL,    /* TY_QUAD */
    TY_CMPLX,   /* TY_HCMPLX */
    TY_CMPLX,   /* TY_CMPLX */
    TY_CMPLX,   /* TY_DCMPLX */
    TY_CMPLX,   /* TY_QCMPLX */
    TY_LOG,     /* TY_BLOG */
    TY_LOG,     /* TY_SLOG */
    TY_LOG,     /* TY_LOG */
    TY_LOG,     /* TY_LOG8 */
    TY_CHAR,    /* TY_CHAR */
    TY_CHAR,    /* TY_NCHAR */
    TY_PTR,     /* TY_PTR */
    TY_ARRAY,   /* TY_ARRAY */
    TY_STRUCT,  /* TY_STRUCT */
    TY_UNION,   /* TY_UNION */
    TY_DERIVED, /* TY_DERIVED */
    TY_NUMERIC, /* TY_NUMERIC */
    TY_ANY,     /* TY_ANY */
    TY_PROC,    /* TY_PROC */
    TY_128,     /* TY_128 */
    TY_256,     /* TY_256 */
    TY_512,     /* TY_512 */
    TY_INT,     /* TY_INT128 */
    TY_LOG,     /* TY_LOG128 */
    TY_REAL,    /* TY_FLOAT128 */
    TY_CMPLX,   /* TY_CMPLX128 */
};

#if TY_MAX != 36
#error \
    "Need to edit dtypeutl.c to add new TY_... data types to ty_to_lib and ty_to_base_ty"
#endif

/** \brief Map compiler's DT_ values to the values expected by the run-time. */
int
dtype_to_arg(DTYPE dtype)
{
  return ty_to_lib[DTY(dtype)];
}

/** \brief For intrinsic types, return same value as the KIND intrinsic */
int
kind_of(DTYPE d1)
{
  int ty1;
  int k;

  ty1 = DTY(d1);
  if (ty1 < 0 || ty1 >= TY_MAX)
    return 0;
  if (!TY_ISBASIC(ty1))
    return 0;
  switch (ty1) {
  case TY_CHAR:
    k = 1;
    break;
  case TY_NCHAR:
    k = 2;
    break;
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
    k = size_of(d1) / 2;
    break;
  default:
    k = size_of(d1);
    break;
  }
  return k;
}

LOGICAL
same_type_different_kind(DTYPE d1, DTYPE d2)
{
  int ty1, ty2;
  ty1 = DTY(d1);
  ty2 = DTY(d2);
  if (ty1 < 0 || ty2 < 0 || ty1 >= TY_MAX || ty2 >= TY_MAX)
    return FALSE;
  if (ty_to_base_ty[ty1] == ty_to_base_ty[ty2])
    return TRUE;
  return FALSE;
} /* same_type_different_kind */

LOGICAL
different_type_same_kind(DTYPE d1, DTYPE d2)
{
  int ty1, ty2;
  int k1, k2;
  ty1 = DTY(d1);
  ty2 = DTY(d2);
  if (ty1 < 0 || ty2 < 0 || ty1 >= TY_MAX || ty2 >= TY_MAX)
    return FALSE;
  /* at least the TYs must be different */
  if (ty1 == ty2)
    return FALSE;
  k1 = kind_of(d1);
  k2 = kind_of(d2);
  if (k1 != k2)
    return FALSE;
  return TRUE;
} /* different_type_same_kind */

#define RW_FD(b, s, n)                         \
  {                                            \
    nw = (*p_rw)((char *)b, sizeof(s), n, fd); \
    if (nw != (n))                             \
      error(10, 40, 0, "(state file)", CNULL); \
  }

void
rw_dtype_state(int (*p_rw)(void *, size_t, size_t, FILE *), FILE *fd)
{
  int nw;

  RW_FD(&stb.dt.stg_avail, stb.dt.stg_avail, 1);
  RW_FD(&stb.dt.stg_cleared, stb.dt.stg_cleared, 1);
  RW_FD(stb.dt.stg_base, ISZ_T, stb.dt.stg_avail);
  RW_FD(chartab, chartab, 1);
  RW_FD(&chartabavail, chartabavail, 1);
  RW_FD(chartabbase, struct chartab, chartabavail);
}

/* Predicate: is dtype is a derived type with type bound procedures? */
static LOGICAL
is_tbp_component(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         (is_tbp(sptr) ||
          (/* PARENTG(sptr) && */
           search_type_members(DTYPEG(sptr), is_tbp_component, visited)));
}

LOGICAL
has_tbp(DTYPE dtype)
{
  return search_type_members_wrapped(dtype, is_tbp_component);
}

/* Predicate: is dtype is a derived type with type bound/final procedures? */
static LOGICAL
is_tbp_or_final_component(int sptr, struct visit_list **visited)
{
  return sptr > NOSYM &&
         (is_tbp_or_final(sptr) ||
          (/* PARENTG(sptr) && */
           search_type_members(DTYPEG(sptr), is_tbp_or_final_component,
                               visited)));
}

LOGICAL
has_tbp_or_final(DTYPE dtype)
{
  return search_type_members_wrapped(dtype, is_tbp_or_final_component);
}

int
chk_kind_parm_set_expr(int ast, DTYPE dtype)
{
  int sptr, newast1, newast2, i, val;

  switch (A_TYPEG(ast)) {
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      newast1 = chk_kind_parm_set_expr(ARGT_ARG(i, 0), dtype);
      return newast1 < 0 ? -1 : newast1;
    }
    break;
  case A_CNST:
    break;
  case A_ID:
    if (dtype == 0)
      break;
    sptr = A_SPTRG(ast);
    if (get_kind_set_parm(sptr, dtype, &val)) {
      return mk_cval1(val, DT_INT);
    }
    val = 0;
    i = get_len_set_parm(sptr, dtype, &val);
    if (i || val) {
      if (val && A_TYPEG(val) == A_ID) {
        return ast;
      } else if (val) {
        return chk_kind_parm_set_expr(val, dtype);
      } else {
        return mk_cval1(i, DT_INT);
      }
    } else {
      return -1;
    }
    break;
  case A_UNOP:
    newast1 = chk_kind_parm_set_expr(A_LOPG(ast), dtype);
    if (newast1 < 0)
      return -1;
    A_LOPP(ast, newast1);
    break;
  case A_BINOP:
    newast1 = chk_kind_parm_set_expr(A_LOPG(ast), dtype);
    if (newast1 < 0)
      return -1;
    newast2 = chk_kind_parm_set_expr(A_ROPG(ast), dtype);
    if (newast2 < 0)
      return -1;
    A_LOPP(ast, newast1);
    A_ROPP(ast, newast2);

    if (A_TYPEG(newast1) == A_CNST && A_TYPEG(newast2) == A_CNST) {
      i = const_fold(A_OPTYPEG(ast), CONVAL2G(A_SPTRG(newast1)),
                     CONVAL2G(A_SPTRG(newast2)), A_DTYPEG(ast));
      ast = mk_cval1(i, DT_INT);
    }
    break;
  default:
    return -1;
  }

  return ast;
}

static LOGICAL
get_kind_set_parm(int sptr, DTYPE dtype, int *val)
{
  int mem;

  if (DTY(dtype) != TY_DERIVED)
    return FALSE;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      if (get_kind_set_parm(sptr, DTYPEG(mem), val)) {
        return TRUE;
      }
    }
    if (!LENPARMG(mem) && SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0) {
      *val = KINDG(mem);
      return TRUE;
    }
  }

  return FALSE;
}

static int
get_len_set_parm(int sptr, DTYPE dtype, int *val)
{
  int rslt, mem;

  if (DTY(dtype) != TY_DERIVED)
    return 0;

  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    if (PARENTG(mem)) {
      rslt = get_len_set_parm(sptr, DTYPEG(mem), val);
      if (rslt)
        return rslt;
    }
    if (LENPARMG(mem) && SETKINDG(mem) && !USEKINDG(mem) && KINDG(mem) &&
        strcmp(SYMNAME(mem), SYMNAME(sptr)) == 0) {
      *val = LENG(mem);
      return KINDG(mem);
    }
  }

  return 0;
}

/** \brief Compute size and alignment of struct and union types and their
 * members.
 */
void
chkstruct(DTYPE dtype)
{
  int m, m_prev = NOSYM, m_next = NOSYM;
  ISZ_T symlk;

  if (DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_DERIVED) {
    int offset = 0;  /* byte offset from beginning of struct */
    int maxa = 0;    /* maximum alignment req'd by any member */
    int distmem = 0; /* any distributed members? */
    int ptrmem = 0;  /* any pointer members? */

    for (m = DTY(dtype + 1); m != NOSYM; m_prev = m, m = symlk) {
      int oldoffset, a;
      symlk = SYMLKG(m);
      m_next = symlk;
      if (DTYPEG(m) == DT_NONE) {
        continue; /* Occurs w/ empty typedef */
      }
      if (is_tbp_or_final(m)) {
        /* skip tbp */
        continue;
      }
      a = alignment_of_var(m);
      offset = ALIGN(offset, a);
      oldoffset = offset;
      ADDRESSP(m, offset);
      if (DTY(DTYPEG(m)) == TY_ARRAY
          && !MIDNUMG(m) && !ADJARRG(m) && !POINTERG(m)
          && !RUNTIMEG(m)) {
        if (extent_of(DTYPEG(m)) != 0)
          offset += size_of_var(m);
      } else
        offset += size_of_var(m);
/* if this is a pointer member, and the next member
 * is the actual pointer, let the pointer/offset/descriptor
 * overlap */
      if ((POINTERG(m) || ALLOCG(m) || ADJARRG(m) || RUNTIMEG(m)) &&
          MIDNUMG(m) == symlk)
        offset = oldoffset;
      if (POINTERG(m))
        ptrmem = 1;
      if (a > maxa)
        maxa = a;
      PSMEMP(m, m);
      if (ALIGNG(m) || DISTG(m)) {
        distmem = 1;
      } else {
        DTYPE d = DTYPEG(m);
        if (DTY(d) == TY_DERIVED) {
          int tag = DTY(d + 3);
          if (tag) {
            if (DISTMEMG(tag))
              distmem = 1;
            if (POINTERG(tag))
              ptrmem = 1;
          }
        }
      }
    }
    /* compute final size and alignment of struct: */

    DTY(dtype + 2) = ALIGN(offset, maxa);
    if (distmem && DTY(dtype + 3)) {
      DISTMEMP(DTY(dtype + 3), 1);
    }
    DTY(dtype + 4) = maxa;
    if (ptrmem && DTY(dtype + 3)) {
      POINTERP(DTY(dtype + 3), 1);
    }
  } else {
    /*
     * Size and alignment of a union are the maximum of the sizes and
     * alignments of its members:
     */
    int maxa = 0;
    ISZ_T size = 1;
    assert(DTY(dtype) == TY_UNION && DTY(dtype + 1), "chkstruct:bad dt", dtype,
           3);
    for (m = DTY(dtype + 1); m != NOSYM; m_prev = m, m = symlk) {
      symlk = SYMLKG(m);
      m_next = symlk;
      ISZ_T s = size_of_var(m);
      int a = alignment_of_var(m);
      if (s > size)
        size = s;
      if (a > maxa)
        maxa = a;
    }
    DTY(dtype + 2) = ALIGN(size, maxa);
    DTY(dtype + 4) = maxa;
  }
}

/* Return the dtype if this derived type was defined in iso_c_bind_decl
   Return 0 otherwise.
   Differentiate c_ptr and c_funptr from possible user defined
   derived types.
   NOTE:
   We still need to mark these symbols as compiler generated to
   that the user could define his own iso_c_bind_decl that would
   not conflict with this
 */

DTYPE
is_iso_cptr(DTYPE d_dtype)
{
  return get_iso_derivedtype(d_dtype);
}

LOGICAL
is_iso_c_ptr(DTYPE d_dtype)
{
  DTYPE dtype = get_iso_derivedtype(d_dtype);
  return dtype && ic_strcmp(SYMNAME(DTY(dtype + 3)), "c_ptr") == 0;
}

LOGICAL
is_iso_c_funptr(DTYPE d_dtype)
{
  DTYPE dtype = get_iso_derivedtype(d_dtype);
  return dtype && ic_strcmp(SYMNAME(DTY(dtype + 3)), "c_funptr") == 0;
}

static DTYPE
get_iso_derivedtype(DTYPE d_dtype)
{
  int check_mod;
  DTYPE dtype = d_dtype;
  int mod = lookupsymbol("iso_c_binding");

  if (mod == 0)
    return 0;

  /* yes to array of c_ptrs */
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);

  if (DTY(dtype) == TY_DERIVED && DTY(dtype + 3))
    check_mod = DTY(dtype + 3); /* tag */
  else
    check_mod = 0;

  if (gbl.internal > 1) {
    /* If a contained subprogram has USEd iso_c_binding, a temporary,
     * incomplete symbol table entry may have been generated. Find the
     * correct (complete) one.
     */
    for (; mod && STYPEG(mod) != ST_MODULE; mod = HASHLKG(mod)) {
      if (strcmp(SYMNAME(mod), "iso_c_binding") == 0 &&
          STYPEG(mod) == ST_MODULE) {
        break;
      }
    }
  }

  if (check_mod <= 0 || ENCLFUNCG(check_mod) <= 0)
    return 0;

  if (ENCLFUNCG(check_mod) == mod) {
    ISOCTYPEP(check_mod, 1);
    return dtype;
  }

  return 0;
}

LOGICAL
is_cuf_c_devptr(DTYPE d_dtype)
{
  DTYPE dtype = get_cuf_derivedtype(d_dtype);
  return dtype && ic_strcmp(SYMNAME(DTY(dtype + 3)), "c_devptr") == 0;
}

static DTYPE
get_cuf_derivedtype(DTYPE d_dtype)
{
  int check_mod;
  DTYPE dtype = d_dtype;
  int mod = lookupsymbol("pgi_acc_common");
  if (mod == 0 || STYPEG(mod) == ST_UNKNOWN)
    mod = lookupsymbol("cudafor");
  if (mod == 0 || STYPEG(mod) == ST_UNKNOWN)
    mod = lookupsymbol("cudafor_la");
  if (mod == 0 || STYPEG(mod) == ST_UNKNOWN)
    return 0;

  /* yes to array of c_ptrs */
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);
  if (DTY(dtype) == TY_DERIVED && DTY(dtype + 3))
    check_mod = DTY(dtype + 3); /* tag */
  else
    check_mod = 0;
  if (check_mod <= 0 || ENCLFUNCG(check_mod) <= 0)
    return 0;
  if (ENCLFUNCG(check_mod) == mod)
    return dtype;
  return 0;
}

DTYPE
get_iso_ptrtype(const char *name)
{
  int s, sptr;
  int mod;
  int check_mod;

  mod = lookupsymbol("iso_c_binding");
  if (mod == 0)
    return 0;

  s = getsymbol(name);
  for (sptr = first_hash(s); sptr; sptr = HASHLKG(sptr)) {
    if (NMPTRG(sptr) != NMPTRG(s))
      continue;
    if (STYPEG(sptr) != ST_TYPEDEF)
      continue;
    check_mod = DTY(DTYPEG(sptr) + 3); /* tag */
    if ((check_mod <= 0) || (ENCLFUNCG(check_mod) <= 0))
      continue;
    if (ENCLFUNCG(check_mod) == mod) {
      return DTYPEG(sptr);
    }
  }
  return DT_NONE;
}

DTYPE
get_iso_c_ptr(void)
{
  return get_iso_ptrtype("c_ptr");
}

/* FIXME: create common utility that can be shared with sem_strcmp */
/** \brief Compare \a str and \a pattern like strcmp() but ignoring the case of
   str.
           \a pattern is all lower case.
 */
static int
ic_strcmp(const char *str, const char *pattern)
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

LOGICAL
is_array_dtype(DTYPE dtype)
{
  return dtype > DT_NONE && get_ty_kind(dtype) == TY_ARRAY;
}

DTYPE
array_element_dtype(DTYPE dtype)
{
  return is_array_dtype(dtype) ? DTY(dtype + 1) : DT_NONE;
}

LOGICAL
is_dtype_runtime_length_char(DTYPE dtype)
{
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  return dtype > DT_NONE &&
         DT_ISCHAR(dtype) &&
         string_length(dtype) == 0;
}

LOGICAL
is_dtype_unlimited_polymorphic(DTYPE dtype)
{
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  return dtype > DT_NONE &&
         DTY(dtype) == TY_DERIVED &&
         UNLPOLYG(DTY(dtype + 3 /*tag*/));
}

/** \brief Test if a data type index corresponds with a procedure pointer
 * \param dtype data type index to check
 */
LOGICAL
is_procedure_ptr_dtype(DTYPE dtype)
{
  return ((dtype > DT_NONE) && (get_ty_kind(dtype) == TY_PTR) &&
           is_procedure_dtype(DTY(dtype + 1)));
}

/** \brief Get return type from a procedure pointer dtype
 * \param dtype data type index for procedure pointer
 */
DTYPE
proc_ptr_result_dtype(DTYPE dtype)
{
  return is_procedure_ptr_dtype(dtype) ? DTY(DTY(dtype + 1) + 1) : DT_NONE;
}

/** \brief Set return type for a procedure pointer
 * \param ptr_dtype data type index for the procedure pointer
 * \param result_dtype data type index for the return type
 */
void
set_proc_ptr_result_dtype(DTYPE ptr_dtype, DTYPE result_dtype)
{
  assert(is_procedure_ptr_dtype(ptr_dtype), "type is not a procedure pointer",
         ptr_dtype, 3);

  set_proc_result_dtype(DTY(ptr_dtype + 1), result_dtype);
}

/** \brief Set paramter count for a procedure pointer type 
 * \param ptr_dtype data type index for the procedure pointer
 * \param param_count paramter count to set
 */
void
set_proc_ptr_param_count_dtype(DTYPE ptr_dtype, int param_count)
{
  assert(is_procedure_ptr_dtype(ptr_dtype), "type is not a procedure pointer",
         ptr_dtype, 3);

  set_proc_param_count_dtype(DTY(ptr_dtype + 1), param_count);
}

/** \brief Test if a data type index corresponds with a procedure
 * \param dtype data type index to check
 */
LOGICAL
is_procedure_dtype(DTYPE dtype)
{
  return dtype > DT_NONE && get_ty_kind(dtype) == TY_PROC;
}

/** \brief Set return type for a procedure type
 * \param proc_dtype data type index for the procedure type
 * \param result_dtype data type index for the return type
 */
void
set_proc_result_dtype(DTYPE proc_dtype, DTYPE result_dtype)
{
  assert(is_procedure_dtype(proc_dtype), "type is not a procedure", proc_dtype,
         3);

  DTY(proc_dtype + 1) = result_dtype;
}

/** \brief Set paramter count for a procedure type
 * \param proc_dtype data type index for the procedure type
 * \param param_count paramter count to set
 */
void
set_proc_param_count_dtype(DTYPE proc_dtype, int param_count)
{
  assert(is_procedure_dtype(proc_dtype), "type is not a procedure", proc_dtype,
         3);

  DTY(proc_dtype + 3) = param_count;
}

static int
get_struct_dtype_field(DTYPE dtype, int offset, int default_result)
{
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  if (dtype > DT_NONE) {
    switch (get_ty_kind(dtype)) {
    case TY_STRUCT:
    case TY_UNION:
    case TY_DERIVED:
      return DTY(dtype + offset);
    default:
      break;
    }
  }
  return default_result;
}

SPTR
get_struct_tag_sptr(DTYPE dtype)
{
  return get_struct_dtype_field(dtype, 3 /* tag */, 0);
}

SPTR
get_struct_members(DTYPE dtype)
{
  return get_struct_dtype_field(dtype, 1 /* members */, 0);
}

int
get_struct_initialization_tree(DTYPE dtype)
{
  return get_struct_dtype_field(dtype, 5 /* i.c.t. */, 0);
}

LOGICAL
is_unresolved_parameterized_dtype(DTYPE dtype)
{
  SPTR tag;
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);
  tag = get_struct_tag_sptr(dtype);
  if (tag > NOSYM) {
    SPTR member;
    if (BASETYPEG(tag) > DT_NONE)
      return FALSE; /* the BASETYPE here means the original p.d.t. */
    for (member = get_struct_members(dtype);
         member > NOSYM; member = SYMLKG(member)) {
      if (!USEKINDG(member) && !SETKINDG(member) && KINDG(member) != 0)
        return TRUE;
    }
  }
  return FALSE;
}

/* Correct TYPE IS(CHARACTER(LEN=*)) to TYPE IS(CHARACTER(LEN=:))
 * so that semant3 can create a pointer or allocatable for construct
 * association.
 */
DTYPE
change_assumed_char_to_deferred(DTYPE dtype)
{
  switch (dtype) {
  case DT_ASSCHAR:
    return DT_DEFERCHAR;
  case DT_ASSNCHAR:
    return DT_DEFERNCHAR;
  default:
    return dtype;
  }
}

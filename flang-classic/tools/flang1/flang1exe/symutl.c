/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Fortran symbol utilities.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symfun.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "soc.h"
#include "ast.h"
#include "gramtk.h"
#include "comm.h"
#include "extern.h"
#include "hpfutl.h"
#include "rte.h"
#include "semant.h"

#define CONTIGUOUS_ARR(sptr) (ALLOCG(sptr) || CONTIGATTRG(sptr))
static int find_alloc_size(int ast, int foralllist, int *allocss,
                           int *allocdtype, int *allocdim);
static int do_check_member_id(int astmem, int astid);

SYMUTL symutl;

static int symutl_sc = SC_LOCAL; /* should symutl.sc be used instead?? */

void
set_symutl_sc(int sc)
{
  symutl_sc = sc;
  symutl.sc = sc;
}

int
get_next_sym(const char *basename, const char *purpose)
{
  int sptr;
  char *p;

  p = mangle_name(basename, purpose);
  sptr = getsymbol(p);
  HCCSYMP(sptr, 1);
  HIDDENP(sptr, 1); /* can't see this, if in the parser */
  SCOPEP(sptr, stb.curr_scope);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  return sptr;
}

int
get_next_sym_dt(const char *basename, const char *purpose, int encldtype)
{
  int sptr;
  char *p;

  p = mangle_name_dt(basename, purpose, encldtype);
  sptr = getsymbol(p);
  HCCSYMP(sptr, 1);
  HIDDENP(sptr, 1); /* can't see this, if in the parser */
  SCOPEP(sptr, stb.curr_scope);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  return sptr;
}

/* rename to get_ast_of_deferlen? */
int
get_len_of_deferchar_ast(int ast)
{
  int sdsc, sdsc_ast, sptr;
  int sdscofmem_ast;
  int first;
  int subs[1];

  /* Need to add a check for subscript type */
  first = first_element(ast);
  if (A_TYPEG(first) == A_SUBSCR) {
    first = A_LOPG(first);
  }
  if (A_TYPEG(first) != A_MEM) {
    sdsc = SDSCG(A_SPTRG(first));
    assert(sdsc != 0, "Deferred-length character symbol must have descriptor",
           A_SPTRG(ast), 0);
    return get_byte_len(sdsc);
  }

  /* The length is set in type descriptor for a polymorphic derived type
   * member, and it is set in section descriptor for other cases. */
  sptr = A_SPTRG(A_MEMG(first));
  if (CLASSG(sptr))
    sdsc = get_member_descriptor(sptr);
  else
    sdsc = SDSCG(sptr);
  sdsc_ast = mk_id(sdsc);
  sdscofmem_ast = mk_member(A_PARENTG(first), sdsc_ast, A_DTYPEG(sdsc_ast));

  subs[0] = mk_isz_cval(get_byte_len_indx(), astb.bnd.dtype);
  return mk_subscr(sdscofmem_ast, subs, 1, astb.bnd.dtype);
}

/** \brief Get the sptr of a specific name & SYMTYPE in the hash list
    \param stype  the SYMTYPE
    \param first  where to start the search (also establishes the name )
 */
SPTR
get_symtype(SYMTYPE stype, SPTR first)
{
  SPTR sptr;
  for (sptr = first; sptr > NOSYM; sptr = HASHLKG(sptr)) {
    if (NMPTRG(sptr) != NMPTRG(first))
      continue;
    if (STYPEG(sptr) == stype)
      return sptr;
  }
  return 0;
}

int
sym_get_scalar(const char *basename, const char *purpose, int dtype)
{
  int sptr;

  sptr = get_next_sym(basename, purpose);
  DTYPEP(sptr, dtype);
  STYPEP(sptr, ST_VAR);
  DCLDP(sptr, 1);
  SCP(sptr, symutl.sc);
  NODESCP(sptr, 1);
  SCOPEP(sptr, stb.curr_scope);
  return sptr;
}

int
sym_get_ptr(int base)
{
  int sptr;
  char *basename;

  basename = SYMNAME(base);
  if (STYPEG(base) == ST_MEMBER) {
    sptr = get_next_sym_dt(basename, "p", ENCLDTYPEG(base));
  } else {
    sptr = get_next_sym(basename, "p");
  }
  DTYPEP(sptr, DT_PTR);
  STYPEP(sptr, ST_VAR);
  SCP(sptr, symutl_sc);
  NODESCP(sptr, 1);
  PTRVP(sptr, 1);
  if (CONSTRUCTSYMG(base)) {
    CONSTRUCTSYMP(sptr, true);
    ENCLFUNCP(sptr, ENCLFUNCG(base));
  }
  return sptr;
}

int
sym_get_ptr_name(char *basename)
{
  int sptr;

  sptr = get_next_sym(basename, "p");
  DTYPEP(sptr, DT_PTR);
  STYPEP(sptr, ST_VAR);
  SCP(sptr, symutl_sc);
  NODESCP(sptr, 1);
  PTRVP(sptr, 1);
  return sptr;
}

int
sym_get_offset(int base)
{
  int sptr;
  char *basename;

  basename = SYMNAME(base);
  if (STYPEG(base) == ST_MEMBER) {
    sptr = get_next_sym_dt(basename, "o", ENCLDTYPEG(base));
  } else {
    sptr = get_next_sym(basename, "o");
  }
  DTYPEP(sptr, DT_PTR);
  STYPEP(sptr, ST_VAR);
  SCP(sptr, symutl_sc);
  NODESCP(sptr, 1);
  if (CONSTRUCTSYMG(base)) {
    CONSTRUCTSYMP(sptr, true);
    ENCLFUNCP(sptr, ENCLFUNCG(base));
  }
  return sptr;
}

/** \brief Make a temporary array, not deferred shape.  Bounds need to be
           filled in later.
 */
int
sym_get_array(const char *basename, const char *purpose, int dtype, int ndim)
{
  int sptr;
  ADSC *ad;
  int i;

  sptr = get_next_sym(basename, purpose);
  dtype = get_array_dtype(ndim, dtype);
  ALLOCP(sptr, 1);
  ad = AD_DPTR(dtype);
  AD_NOBOUNDS(ad) = 1;
  for (i = 0; i < ndim; ++i) {
    AD_LWAST(ad, i) = AD_UPAST(ad, i) = 0;
    AD_LWBD(ad, i) = AD_UPBD(ad, i) = 0;
    AD_EXTNTAST(ad, i) = 0;
  }
  DTYPEP(sptr, dtype);
  STYPEP(sptr, ST_ARRAY);
  DCLDP(sptr, 1);
  SCP(sptr, symutl_sc);
  return sptr;
}

/** \brief Create a function ST item given a name */
int
sym_mkfunc(const char *nmptr, int dtype)
{
  register int sptr;

  sptr = getsymbol(nmptr);
  STYPEP(sptr, ST_PROC);
  DTYPEP(sptr, dtype);
  if (dtype != DT_NONE) {
    DCLDP(sptr, 1);
    FUNCP(sptr, 1);
    if (XBIT(57, 0x2000))
      TYPDP(sptr, 1);
  }
  SCP(sptr, SC_EXTERN);
  SCOPEP(sptr, 0);
  HCCSYMP(sptr, 1);
  /*NODESCP(sptr,1);*/
  return sptr;
}

/** \brief Create a function ST item given a name; set its NODESC flag */
int
sym_mkfunc_nodesc(const char *nmptr, int dtype)
{
  register int sptr;

  sptr = sym_mkfunc(nmptr, dtype);
  NODESCP(sptr, 1);
  PUREP(sptr, 1);
  return sptr;
}

/** \brief Create a function ST item given a name; set its NODESC and EXPST
   flag.

    Could replace EXPST with a  new flag. If the flag is set
    we still need transform_call() to fix arguments which are
    array sections.
 */
int
sym_mkfunc_nodesc_expst(const char *nmptr, int dtype)
{
  register int sptr;

  sptr = sym_mkfunc_nodesc(nmptr, dtype);
  EXPSTP(sptr, 1);
  return sptr;
}

/** \brief Create a function ST item given a name; set its NODESC and NOCOMM
 * flag */
int
sym_mkfunc_nodesc_nocomm(const char *nmptr, int dtype)
{
  register int sptr;

  sptr = sym_mkfunc(nmptr, dtype);
  NODESCP(sptr, 1);
  NOCOMMP(sptr, 1);
  PUREP(sptr, 1);
  return sptr;
}

int
sym_mknproc(void)
{
  STYPEP(gbl.sym_nproc, ST_VAR);
  return gbl.sym_nproc;
}

/* This create  descriptor and section descriptor
   for each user defined array                    */

void
trans_mkdescr(int sptr)
{
  int descr;
  char *p;

  if (DESCRG(sptr) != 0)
    return;
  /* save the basename in case it is a SYMNAME (might be realloc'd) */
  p = sym_strsave(SYMNAME(sptr));
  descr = get_next_sym(p, "arrdsc");
  STYPEP(descr, ST_ARRDSC);
  /*
   * 2nd try for f15624 - use the storage class field of the arrdsc so
   * that the SC_PRIVATE of arrdsc is propagated to the actual descriptor.
   */
  SCP(descr, symutl_sc);
  ARRAYP(descr, sptr);
  ALNDP(descr, 0);
  SECDP(descr, 0);
  SECDSCP(descr, 0);
  DESCRP(sptr, descr);
  NODESCP(sptr, 0);
  if (XBIT(57, 0x10000) && SCG(sptr) == SC_DUMMY && NEWDSCG(sptr)) {
    SECDSCP(descr, NEWDSCG(sptr));
  }
  FREE(p);
}

/** \brief Create a section descriptor */
int
sym_get_sec(const char *basename, int is_dummy)
{
  int sec, sec_ptr;
  ADSC *ad;
  int dtype;
  char *p;

  /* save the basename in case it is a SYMNAME (might be realloc'd) */
  p = sym_strsave(basename);
  sec = get_next_sym(p, "s");
  if (!is_dummy)
    sec_ptr = get_next_sym(p, "sp");
  FREE(p);

  /* make sec be array(1) */
  STYPEP(sec, ST_ARRAY);
  dtype = aux.dt_iarray_int;
  ad = AD_DPTR(dtype);
  AD_LWAST(ad, 0) = 0;
  AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_isz_cval(1, astb.bnd.dtype);
  AD_EXTNTAST(ad, 0) = mk_isz_cval(1, astb.bnd.dtype);
  DTYPEP(sec, dtype);
  DCLDP(sec, 1);

  /* array section for dummy doesn't have a pointer */
  if (!is_dummy) {
    SCP(sec, SC_BASED);
    /* make the pointer point to sec */
    STYPEP(sec_ptr, ST_VAR);
    DTYPEP(sec_ptr, DT_PTR);
    SCP(sec_ptr, symutl_sc);
    MIDNUMP(sec, sec_ptr);
  } else
    SCP(sec, SC_DUMMY);

  NODESCP(sec, 1);
  return sec;
}

/** \brief Create a channel pointer (cp) */
int
sym_get_cp(void)
{
  int cp_ptr;

  /* save the basename in case it is a SYMNAME (might be realloc'd) */
  cp_ptr = trans_getbound(0, 11);

  /* make the pointer point to cp */
  STYPEP(cp_ptr, ST_VAR);
  DTYPEP(cp_ptr, DT_ADDR);
  SCP(cp_ptr, SC_LOCAL);
  DCLDP(cp_ptr, 1);
  return cp_ptr;
}

/** \brief Create a channel pointer (xfer) */
int
sym_get_xfer(void)
{
  int xfer_ptr;

  xfer_ptr = trans_getbound(0, 12);
  STYPEP(xfer_ptr, ST_VAR);
  DTYPEP(xfer_ptr, DT_ADDR);
  SCP(xfer_ptr, SC_LOCAL);
  DCLDP(xfer_ptr, 1);
  return xfer_ptr;
}

/** \brief Create a section descriptor for a dummy argument */
int
sym_get_arg_sec(int sptr)
{
  int sec;
  ADSC *ad;
  int dtype;
  char *p;
  char *basename;
  int sdsc;

  if (XBIT(57, 0x10000)) {
    sdsc = sym_get_sdescr(sptr, -1);
    /* sym_get_sdescr will return existing descriptor but we don't want that if
     * this is
     * an interface.  It is possible that the current routine has same name
     * descriptor
     * due to use associate.
     */
    if (SCOPEG(sptr)) {
      int scope = SCOPEG(sptr);
      if (STYPEG(scope) == ST_ALIAS)
        scope = SYMLKG(scope);
      if (SCG(scope) == SC_EXTERN && STYPEG(scope) == ST_PROC) {
        sdsc = get_next_sym(SYMNAME(sptr), "sd");
      }
    }
    SCP(sdsc, SC_DUMMY);
    HCCSYMP(sdsc, 1);
    return sdsc;
  }

  basename = SYMNAME(sptr);
  /* save the basename in case it is a SYMNAME (might be realloc'd) */
  p = sym_strsave(basename);
  sec = get_next_sym(p, "s0");
  FREE(p);

  dtype = DDTG(DTYPEG(sptr));

  if ((STYPEG(sptr) != ST_ARRAY || DTY(dtype) == TY_CHAR ||
       DTY(dtype) == TY_NCHAR) &&
      !POINTERG(sptr)) {
    /* make sec be integer scalar */
    DTYPEP(sec, DT_INT);
    STYPEP(sec, ST_VAR);
  } else {
    /* make sec be array(1) */
    STYPEP(sec, ST_ARRAY);
    dtype = aux.dt_iarray_int;
    ad = AD_DPTR(dtype);
    AD_LWAST(ad, 0) = 0;
    AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_isz_cval(1, astb.bnd.dtype);
    AD_EXTNTAST(ad, 0) = mk_isz_cval(1, astb.bnd.dtype);
    DTYPEP(sec, dtype);
  }

  DCLDP(sec, 1);
  SCP(sec, SC_DUMMY);
  HCCSYMP(sec, 1);

  NODESCP(sec, 1);
  return sec;
}

/** \brief Get a symbol for the base address of the formal argument */
int
sym_get_formal(int basevar)
{
  int formal;
  int dtype;
  char *basename;

  basename = SYMNAME(basevar);
  formal = get_next_sym(basename, "bs");

  dtype = DTYPEG(basevar);
  if (DTY(dtype) != TY_ARRAY) {
    /* declare as pointer to the datatype */
    STYPEP(formal, ST_VAR);
    DTYPEP(formal, dtype);
    /*POINTERP( formal, 1 );*/
  } else {
    /* make sec be array(1) */
    STYPEP(formal, ST_ARRAY);
    dtype = DDTG(dtype);
    dtype = get_array_dtype(1, dtype);
    ADD_LWBD(dtype, 0) = 0;
    ADD_LWAST(dtype, 0) = ADD_NUMELM(dtype) = ADD_UPBD(dtype, 0) =
        ADD_UPAST(dtype, 0) = ADD_EXTNTAST(dtype, 0) =
            mk_isz_cval(1, astb.bnd.dtype);
    DTYPEP(formal, dtype);
  }
  DCLDP(formal, 1);
  SCP(formal, SC_DUMMY);
  HCCSYMP(formal, 1);
  OPTARGP(formal, OPTARGG(basevar));
  INTENTP(formal, INTENTG(basevar));
  return formal;
}

/*-------------------------------------------------------------------------*/

/** \brief Return TRUE if the ast (triplet or shape stride)
    has a constant value, and return its constant value
 */
int
constant_stride(int a, int *value)
{
  int sptr;
  /* ast of zero is treated as constant one */
  if (a == 0) {
    *value = 1;
    return TRUE;
  }
  if (A_TYPEG(a) != A_CNST)
    return FALSE;
  sptr = A_SPTRG(a);
  if (!DT_ISINT(DTYPEG(sptr)))
    return FALSE;
  if ((CONVAL1G(sptr) == 0 && CONVAL2G(sptr) >= 0) ||
      (CONVAL1G(sptr) == -1 && CONVAL2G(sptr) < 0)) {
    *value = CONVAL2G(sptr);
    return TRUE;
  }
  return FALSE;
} /* constant_stride */

/* Temporary allocation */
/* subscripts (triples) for temp */
int
mk_forall_sptr(int forall_ast, int subscr_ast, int *subscr, int elem_dty)
{
  int astli;
  int submap[MAXSUBS], arr_sptr, memberast;
  int i, ndims, lwbnd[MAXSUBS], upbnd[MAXSUBS];
  int sptr, sdtype;

  assert(A_TYPEG(forall_ast) == A_FORALL, "mk_forall_sptr: ast not forall",
         forall_ast, 4);
  /* get the forall index list */
  astli = A_LISTG(forall_ast);

  ndims = 0;
  do {
    if (A_TYPEG(subscr_ast) == A_MEM) {
      subscr_ast = A_PARENTG(subscr_ast);
    } else if (A_TYPEG(subscr_ast) == A_SUBSCR) {
      int lop, dtype;
      int asd, n, i;
      for (i = 0; i < ndims; ++i)
        submap[i] = -1;
      memberast = 0;
      lop = A_LOPG(subscr_ast);
      if (A_TYPEG(lop) == A_MEM) {
        memberast = lop;
        arr_sptr = A_SPTRG(A_MEMG(memberast));
      } else if (A_TYPEG(lop) == A_ID) {
        arr_sptr = A_SPTRG(lop);
      } else {
        interr("mk_forall_sptr: subscript has no member/id", subscr_ast, 3);
      }
      dtype = DTYPEG(arr_sptr);
      /* determine how many dimensions are needed, and which ones they are */
      asd = A_ASDG(subscr_ast);
      n = ASD_NDIM(asd);
      for (i = 0; i < n; ++i) {
        /* need to include the dimension if it is vector as well */
        int k, ast, allocss, allocdtype, allocdim, c, stride, lw, up;
        allocss = 0;
        allocdtype = 0;
        allocdim = 0;
        ast = ASD_SUBS(asd, i);
        if (ASUMSZG(arr_sptr) || XBIT(58, 0x20000)) {
          if (A_TYPEG(ast) == A_TRIPLE) {
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            lw = check_member(memberast, A_LBDG(ast));
            up = check_member(memberast, A_UPBDG(ast));
            c = constant_stride(A_STRIDEG(ast), &stride);
            if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
              lwbnd[ndims] = astb.i1;
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                           stride, stb.user.dt_int),
                  stride, stb.user.dt_int);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else if (c && stride == 1) {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (c && stride == -1) {
              lwbnd[ndims] = up;
              upbnd[ndims] = lw;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (XBIT(58, 0x20000)) {
              lwbnd[ndims] = astb.i1;
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                           stride, stb.user.dt_int),
                  stride, stb.user.dt_int);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            }
            submap[ndims] = i;
            ++ndims;
          } else if (A_SHAPEG(ast)) {
            int shd;
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            shd = A_SHAPEG(ast);
            lw = check_member(memberast, SHD_LWB(shd, i));
            up = check_member(memberast, SHD_UPB(shd, i));
            c = constant_stride(SHD_STRIDE(shd, i), &stride);
            if (c && stride == 1) {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (c && stride == -1) {
              lwbnd[ndims] = up;
              upbnd[ndims] = lw;
              subscr[ndims] = mk_triple(lw, up, A_STRIDEG(ast));
            } else if (XBIT(58, 0x20000)) {
              lwbnd[ndims] = astb.bnd.one;
              stride = SHD_STRIDE(shd, i);
              if (stride == 0)
                stride = astb.bnd.one;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, astb.bnd.dtype),
                           stride, astb.bnd.dtype),
                  stride, astb.bnd.dtype);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            }
            submap[ndims] = i;
            ++ndims;
          } else if ((k = search_forall_var(ast, astli)) != 0) {
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            /* make sure the bounds don't have other forall indices */
            lw = A_LBDG(ASTLI_TRIPLE(k));
            up = A_UPBDG(ASTLI_TRIPLE(k));
            if (search_forall_var(lw, astli) || search_forall_var(up, astli)) {
              /* can't use forall indices, they are triangular.
               * use the bounds of the host array */
              lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
              upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else if (other_forall_var(ast, astli, k)) {
              lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
              upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              c = constant_stride(A_STRIDEG(ASTLI_TRIPLE(k)), &stride);
              if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
                lwbnd[ndims] = astb.i1;
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                upbnd[ndims] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
                subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
              } else if (c && stride == 1) {
                lwbnd[ndims] = lw;
                upbnd[ndims] = up;
                subscr[ndims] = mk_triple(lw, up, 0);
              } else if (c && stride == -1) {
                lwbnd[ndims] = up;
                upbnd[ndims] = lw;
                subscr[ndims] = mk_triple(up, lw, 0);
              } else if (XBIT(58, 0x20000)) {
                lwbnd[ndims] = astb.i1;
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                upbnd[ndims] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
                subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
              } else {
                lwbnd[ndims] = lw;
                upbnd[ndims] = up;
                subscr[ndims] = mk_triple(lw, up, 0);
              }
            }
            submap[ndims] = i;
            ++ndims;
          }
        } else if (A_TYPEG(ast) == A_TRIPLE) {
          /* include this dimension */
          /* build a triplet for the allocate statement off of the
           * dimensions for the array */
          assert(ndims < MAXDIMS, "temporary has too many dimensions",
            ndims, 4);
          lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
          upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        } else if (find_alloc_size(ast, astli, &allocss, &allocdtype,
                                   &allocdim)) {
          /* make this dimension the same size as dimension
           * allocdim of datatype allocdtype for which the subscript
           * is at allocss */
          assert(ndims < MAXDIMS, "temporary has too many dimensions",
            ndims, 4);
          if (allocdtype == 0) {
            allocdtype = dtype;
            allocdim = i;
            allocss = memberast;
          }
          lwbnd[ndims] = check_member(allocss, ADD_LWAST(allocdtype, allocdim));
          upbnd[ndims] = check_member(allocss, ADD_UPAST(allocdtype, allocdim));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        } else if (A_SHAPEG(ast) || search_forall_var(ast, astli)) {
          /* include this dimension */
          /* build a triplet for the allocate statement off of the
           * dimensions for the array */
          assert(ndims < MAXDIMS, "temporary has too many dimensions",
            ndims, 4);
          lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
          upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        }
      }
      subscr_ast = A_LOPG(subscr_ast);
    } else {
      interr("mk_forall_sptr: not member or subscript", subscr_ast, 3);
    }
  } while (A_TYPEG(subscr_ast) != A_ID);

  /* get the temporary */
  assert(ndims > 0, "mk_forall_sptr: not enough dimensions", ndims, 4);
  sptr = sym_get_array(SYMNAME(arr_sptr), "f", elem_dty, ndims);
  /* set the bounds to the correct bounds from the array */
  sdtype = DTYPEG(sptr);
  if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
    for (i = 0; i < ndims; ++i) {
      ADD_LWBD(sdtype, i) = ADD_LWAST(sdtype, i) = astb.bnd.one;
      ADD_UPBD(sdtype, i) = ADD_UPAST(sdtype, i) =
          mk_binop(OP_ADD, mk_binop(OP_SUB, upbnd[i], lwbnd[i], astb.bnd.dtype),
                   astb.bnd.one, astb.bnd.dtype);
      ADD_EXTNTAST(sdtype, i) =
          mk_extent(ADD_LWAST(sdtype, i), ADD_UPAST(sdtype, i), i);
    }
  } else {
    for (i = 0; i < ndims; ++i) {
      ADD_LWBD(sdtype, i) = ADD_LWAST(sdtype, i) = lwbnd[i];
      ADD_UPBD(sdtype, i) = ADD_UPAST(sdtype, i) = upbnd[i];
      ADD_EXTNTAST(sdtype, i) =
          mk_extent(ADD_LWAST(sdtype, i), ADD_UPAST(sdtype, i), i);
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  return sptr;
}

/* get the subscript forall sptr, reuse temporary */
/* subscripts (triples) for temp */
int
get_forall_subscr(int forall_ast, int subscr_ast, int *subscr, int elem_dty)
{
  int astli;
  int submap[MAXSUBS], arr_sptr, memberast;
  int ndims, lwbnd[MAXSUBS], upbnd[MAXSUBS];
  int sptr = 0;

  assert(A_TYPEG(forall_ast) == A_FORALL, "get_forall_subscr: ast not forall",
         forall_ast, 4);
  /* get the forall index list */
  astli = A_LISTG(forall_ast);

  ndims = 0;
  do {
    if (A_TYPEG(subscr_ast) == A_MEM) {
      subscr_ast = A_PARENTG(subscr_ast);
    } else if (A_TYPEG(subscr_ast) == A_SUBSCR) {
      int lop, dtype;
      int asd, n, i;
      for (i = 0; i < ndims; ++i)
        submap[i] = -1;
      memberast = 0;
      lop = A_LOPG(subscr_ast);
      if (A_TYPEG(lop) == A_MEM) {
        memberast = lop;
        arr_sptr = A_SPTRG(A_MEMG(memberast));
      } else if (A_TYPEG(lop) == A_ID) {
        arr_sptr = A_SPTRG(lop);
      } else {
        interr("get_forall_subscr: subscript has no member/id", subscr_ast, 3);
      }
      dtype = DTYPEG(arr_sptr);
      /* determine how many dimensions are needed, and which ones they are */
      asd = A_ASDG(subscr_ast);
      n = ASD_NDIM(asd);
      for (i = 0; i < n; ++i) {
        /* need to include the dimension if it is vector as well */
        int k, ast, allocss, allocdtype, allocdim, c, stride, lw, up;
        allocss = 0;
        allocdtype = 0;
        allocdim = 0;
        ast = ASD_SUBS(asd, i);
        if (ASUMSZG(arr_sptr) || XBIT(58, 0x20000)) {
          if (A_TYPEG(ast) == A_TRIPLE) {
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            lw = check_member(memberast, A_LBDG(ast));
            up = check_member(memberast, A_UPBDG(ast));
            c = constant_stride(A_STRIDEG(ast), &stride);
            if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
              lwbnd[ndims] = astb.i1;
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                           stride, stb.user.dt_int),
                  stride, stb.user.dt_int);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else if (c && stride == 1) {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (c && stride == -1) {
              lwbnd[ndims] = up;
              upbnd[ndims] = lw;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (XBIT(58, 0x20000)) {
              lwbnd[ndims] = astb.i1;
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                           stride, stb.user.dt_int),
                  stride, stb.user.dt_int);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            }
            submap[ndims] = i;
            ++ndims;
          } else if (A_SHAPEG(ast)) {
            int shd;
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            shd = A_SHAPEG(ast);
            lw = check_member(memberast, SHD_LWB(shd, i));
            up = check_member(memberast, SHD_UPB(shd, i));
            c = constant_stride(SHD_STRIDE(shd, i), &stride);
            if (c && stride == 1) {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            } else if (c && stride == -1) {
              lwbnd[ndims] = up;
              upbnd[ndims] = lw;
              subscr[ndims] = mk_triple(lw, up, A_STRIDEG(ast));
            } else if (XBIT(58, 0x20000)) {
              lwbnd[ndims] = astb.bnd.one;
              stride = SHD_STRIDE(shd, i);
              if (stride == 0)
                stride = astb.bnd.one;
              upbnd[ndims] = mk_binop(
                  OP_DIV,
                  mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, astb.bnd.dtype),
                           stride, astb.bnd.dtype),
                  stride, astb.bnd.dtype);
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              lwbnd[ndims] = lw;
              upbnd[ndims] = up;
              subscr[ndims] = mk_triple(lw, up, 0);
            }
            submap[ndims] = i;
            ++ndims;
          } else if ((k = search_forall_var(ast, astli)) != 0) {
            assert(ndims < MAXDIMS, "temporary has too many dimensions",
              ndims, 4);
            /* make sure the bounds don't have other forall indices */
            lw = A_LBDG(ASTLI_TRIPLE(k));
            up = A_UPBDG(ASTLI_TRIPLE(k));
            if (search_forall_var(lw, astli) || search_forall_var(up, astli)) {
              /* can't use forall indices, they are triangular.
               * use the bounds of the host array */
              lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
              upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else if (other_forall_var(ast, astli, k)) {
              lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
              upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
              subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
            } else {
              c = constant_stride(A_STRIDEG(ASTLI_TRIPLE(k)), &stride);
              if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
                lwbnd[ndims] = astb.i1;
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                upbnd[ndims] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
                subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
              } else if (c && stride == 1) {
                lwbnd[ndims] = lw;
                upbnd[ndims] = up;
                subscr[ndims] = mk_triple(lw, up, 0);
              } else if (c && stride == -1) {
                lwbnd[ndims] = up;
                upbnd[ndims] = lw;
                subscr[ndims] = mk_triple(up, lw, 0);
              } else if (XBIT(58, 0x20000)) {
                lwbnd[ndims] = astb.i1;
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                upbnd[ndims] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw, stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
                subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
              } else {
                lwbnd[ndims] = lw;
                upbnd[ndims] = up;
                subscr[ndims] = mk_triple(lw, up, 0);
              }
            }
            submap[ndims] = i;
            ++ndims;
          }
        } else if (A_TYPEG(ast) == A_TRIPLE) {
          /* include this dimension */
          /* build a triplet for the allocate statement off of the
           * dimensions for the array */
          assert(ndims < MAXDIMS, "temporary has >MAXDIMS dimensions",
            ndims, 4);
          lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
          upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        } else if (find_alloc_size(ast, astli, &allocss, &allocdtype,
                                   &allocdim)) {
          /* make this dimension the same size as dimension
           * allocdim of datatype allocdtype for which the subscript
           * is at allocss */
          assert(ndims < MAXDIMS, "temporary has >MAXDIMS dimensions",
            ndims, 4);
          if (allocdtype == 0) {
            allocdtype = dtype;
            allocdim = i;
            allocss = memberast;
          }
          lwbnd[ndims] = check_member(allocss, ADD_LWAST(allocdtype, allocdim));
          upbnd[ndims] = check_member(allocss, ADD_UPAST(allocdtype, allocdim));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        } else if (A_SHAPEG(ast) || search_forall_var(ast, astli)) {
          /* include this dimension */
          /* build a triplet for the allocate statement off of the
           * dimensions for the array */
          assert(ndims < MAXDIMS, "temporary has >MAXDIMS dimensions",
            ndims, 4);
          lwbnd[ndims] = check_member(memberast, ADD_LWAST(dtype, i));
          upbnd[ndims] = check_member(memberast, ADD_UPAST(dtype, i));
          subscr[ndims] = mk_triple(lwbnd[ndims], upbnd[ndims], 0);
          submap[ndims] = i;
          ++ndims;
        }
      }
      subscr_ast = A_LOPG(subscr_ast);
    } else {
      interr("get_forall_subscr: not member or subscript", subscr_ast, 3);
    }
  } while (A_TYPEG(subscr_ast) != A_ID);

  return sptr;
}

/** \brief Allocate a temporary to hold an array.  Create the symbol pointer,
    add the allocate and deallocate statements, and return the array.
    \param forall_ast   ast for forall
    \param subscr_ast   ast for subscript expression
    \param alloc_stmt   statement before which to allocate temp
    \param dealloc_stmt statement after which to deallocate temp
    \param dty          datatype, or zero
    \param ast_dty      ast with data type of element required
    \return symbol table pointer for array

    The dimensions and mapping for the array are determined from the
    subscr_ast and the forall_ast.  The subscr_ast has dimensions which
    are indexed by forall variables, and dimensions that are not.  Those
    that are not are excluded from the temp.  The caller can use the
    same index expressions to index this temp, as are used in the subscr_ast.

    The dimensions included in the temp are taken from the array referenced
    by the subscr ast.  Alignments for those dimensions are also taken from
    this array.

    The allocate for the temporary is placed before alloc_stmt, and
    the deallocate is placed after dealloc_stmt.  The name of the temporary
    is derived from the name of the array in the subscr_ast.
 */
int
get_temp_forall(int forall_ast, int subscr_ast, int alloc_stmt,
                int dealloc_stmt, int dty, int ast_dty)
{
  int sptr;
  int subscr[MAXSUBS];
  int par;
  int save_sc;
  int astd, dstd;

  par = STD_PAR(alloc_stmt) || STD_TASK(alloc_stmt);
  if (par) {
    save_sc = symutl.sc;
    set_descriptor_sc(SC_PRIVATE);
  }
  if (dty) {
    sptr = mk_forall_sptr(forall_ast, subscr_ast, subscr, dty);
  } else {
    sptr = mk_forall_sptr(forall_ast, subscr_ast, subscr,
                          DDTG(A_DTYPEG(ast_dty)));
    if (ast_dty > 0 &&
        sptr > NOSYM &&
        A_TYPEG(ast_dty) == A_SUBSCR &&
        is_dtype_runtime_length_char(A_DTYPEG(ast_dty)) &&
        SDSCG(sptr) <= NOSYM) {
      int length_ast = string_expr_length(ast_dty);
      if (length_ast > 0) {
        int descr_length_ast;
        get_static_descriptor(sptr);
        descr_length_ast = symbol_descriptor_length_ast(sptr, 0);
        if (descr_length_ast > 0) {
          add_stmt_before(mk_assn_stmt(descr_length_ast, length_ast,
                                       astb.bnd.dtype), alloc_stmt);
        }
      }
    }
  }
  if (par) {
    set_descriptor_sc(save_sc);
  }
  astd = mk_mem_allocate(mk_id(sptr), subscr, alloc_stmt, ast_dty);
  dstd = mk_mem_deallocate(mk_id(sptr), dealloc_stmt);
  if (STD_ACCEL(alloc_stmt))
    STD_RESCOPE(astd) = 1;
  if (STD_ACCEL(dealloc_stmt))
    STD_RESCOPE(dstd) = 1;
  return sptr;
}

/** \brief This is almost identical to get_temp_forall() except that it has
    one more parameter, \p rhs.
    \param forall_ast   ast for forall
    \param lhs          ast for LHS
    \param rhs          ast for RHS
    \param alloc_stmt   statement before which to allocate temp
    \param dealloc_stmt statement after which to deallocate temp
    \param ast_dty      ast with data type of element required
    \return symbol table pointer for array

    For copy_section, we would like to decide the rank of temp
    according to rhs and distribution will be according to lhs.
    This case arise since we let copy_section to do also multicasting
    For example a(i,j) = b(2*i,3) kind of cases.
    tmp will be 1 dimensional and that will be distribute according
    to the fist dim of a.
 */
int
get_temp_copy_section(int forall_ast, int lhs, int rhs, int alloc_stmt,
                      int dealloc_stmt, int ast_dty)
{
  int sptr, dty, subscr[MAXSUBS];
  dty = DDTG(A_DTYPEG(ast_dty));
  sptr = mk_forall_sptr_copy_section(forall_ast, lhs, rhs, subscr, dty);
  mk_mem_allocate(mk_id(sptr), subscr, alloc_stmt, ast_dty);
  mk_mem_deallocate(mk_id(sptr), dealloc_stmt);
  return sptr;
}

/**
    \param forall_ast   ast for forall
    \param lhs          ast for LHS
    \param rhs          ast for RHS
    \param alloc_stmt   statement before which to allocate temp
    \param dealloc_stmt statement after which to deallocate temp
    \param ast_dty      ast with data type of element required
 */
int
get_temp_pure(int forall_ast, int lhs, int rhs, int alloc_stmt,
              int dealloc_stmt, int ast_dty)
{
  int sptr, dty, subscr[MAXSUBS];
  dty = DDTG(A_DTYPEG(ast_dty));
  sptr = mk_forall_sptr_pure(forall_ast, lhs, rhs, subscr, dty);
  mk_mem_allocate(mk_id(sptr), subscr, alloc_stmt, ast_dty);
  mk_mem_deallocate(mk_id(sptr), dealloc_stmt);
  return sptr;
}

/** \brief Get a temp sptr1 which will be as big as sptr
    it will be replicated and allocated
    \param sptr         sptr to replicate
    \param alloc_stmt   statement before which to allocate temp
    \param dealloc_stmt statement after which to deallocate temp
    \param astmem - tbw.
 */
int
get_temp_pure_replicated(int sptr, int alloc_stmt, int dealloc_stmt, int astmem)
{
  int sptr1;
  int subscr[MAXSUBS];
  int i, ndim;
  ADSC *ad, *ad1;

  ndim = rank_of_sym(sptr);
  sptr1 = sym_get_array(SYMNAME(sptr), "pure$repl", DDTG(DTYPEG(sptr)), ndim);
  ad = AD_DPTR(DTYPEG(sptr));
  ad1 = AD_DPTR(DTYPEG(sptr1));
  for (i = 0; i < ndim; i++) {
    AD_LWAST(ad1, i) = check_member(astmem, AD_LWAST(ad, i));
    AD_UPAST(ad1, i) = check_member(astmem, AD_UPAST(ad, i));
    AD_LWBD(ad1, i) = check_member(astmem, AD_LWBD(ad, i));
    AD_UPBD(ad1, i) = check_member(astmem, AD_UPBD(ad, i));
    AD_EXTNTAST(ad1, i) = check_member(astmem, AD_EXTNTAST(ad, i));
    subscr[i] = mk_triple(AD_LWAST(ad1, i), AD_UPAST(ad1, i), 0);
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr1);

  mk_mem_allocate(mk_id(sptr1), subscr, alloc_stmt, astmem);
  mk_mem_deallocate(mk_id(sptr1), dealloc_stmt);
  return sptr1;
}

/**
    \param arr_ast      ast for arr_ast
    \param alloc_stmt   statement before which to allocate temp
    \param dealloc_stmt statement after which to deallocate temp
    \param dty          ast with data type of element required
 */
int
get_temp_remapping(int arr_ast, int alloc_stmt, int dealloc_stmt, int dty)
{
  int sptr;
  int subscr[MAXSUBS];

  sptr = mk_shape_sptr(A_SHAPEG(arr_ast), subscr, dty);
  mk_mem_allocate(mk_id(sptr), subscr, alloc_stmt, arr_ast);
  mk_mem_deallocate(mk_id(sptr), dealloc_stmt);
  return sptr;
}

static LOGICAL
chk_temp_bnds(int lhs, int arr_sptr, int *subscr, int ndim)
{
  ADSC *tad;
  int sptr;
  int i;

  if (A_TYPEG(lhs) != A_ID)
    return FALSE;
  sptr = A_SPTRG(lhs);
  tad = AD_DPTR(DTYPEG(sptr));
  /* runtime can't handle dest==src */
  if (arr_sptr == sptr)
    return FALSE;
  if (ndim != AD_NUMDIM(tad))
    return FALSE;
  for (i = 0; i < ndim; ++i) {
    if (AD_LWAST(tad, i) != A_LBDG(subscr[i]))
      return FALSE;
    if (AD_UPAST(tad, i) != A_UPBDG(subscr[i]))
      return FALSE;
  }
  return TRUE;
}

/*
 * Make a symbol pointer from an array or subscripted array, assuming
 * that that symbol will be assigned the array
 */
int
mk_assign_sptr(int arr_ast, const char *purpose, int *subscr, int elem_dty,
               int *retval)
{
  return chk_assign_sptr(arr_ast, purpose, subscr, elem_dty, 0, retval);
}

/*
 * Find the sptr of the dummy at position 'pos' for subprogram ent
 */
int
find_dummy(int entry, int pos)
{
  int dscptr;

  proc_arginfo(entry, NULL, &dscptr, NULL);
  if (!dscptr)
    return 0;
  return aux.dpdsc_base[dscptr + pos];
}

/*
 * return the symbol pointer to the array symbol,
 * and in *returnast, return the pointer to the A_SUBSCR
 * (or A_MEM or A_ID, if an unsubscripted array reference)
 */
int
find_array(int ast, int *returnast)
{
  int sptr = 0;

  if (A_TYPEG(ast) == A_SUBSCR) {
    int lop;
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      if (returnast)
        *returnast = ast;
      sptr = A_SPTRG(lop);
    } else if (A_TYPEG(lop) == A_MEM) {
      /* child or parent? */
      int parent = A_PARENTG(lop);
      if ((A_SHAPEG(ast) != 0 && A_SHAPEG(ast) == A_SHAPEG(parent)) ||
          A_SHAPEG(lop) == 0) {
        return find_array(parent, returnast);
      }
      if (returnast)
        *returnast = ast;
      sptr = A_SPTRG(A_MEMG(lop));
    } else {
      interr("find_array: subscript parent is not id or member", ast, 3);
    }
  } else if (A_TYPEG(ast) == A_MEM) {
    int parent = A_PARENTG(ast);
    assert(A_SHAPEG(ast) != 0, "find_array: member ast has no shape", ast, 4);

    if (A_SHAPEG(ast) == A_SHAPEG(parent)) {
      return find_array(parent, returnast);
    }
    if (returnast)
      *returnast = ast;
    sptr = A_SPTRG(A_MEMG(ast));
  } else if (A_TYPEG(ast) == A_ID) {
    assert(A_SHAPEG(ast) != 0, "find_array: ast has no shape", ast, 4);
    if (returnast)
      *returnast = ast;
    sptr = A_SPTRG(ast);
  } else {
    interr("find_array: not subscript or id or member", ast, 3);
  }
  assert(DTY(DTYPEG(sptr)) == TY_ARRAY, "find_array: symbol is not ARRAY", sptr,
         4);
  return sptr;
}

/* ast is ast to search */
static LOGICAL
found_forall_var(int ast)
{
  int argt, n, i;
  int asd;

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if (found_forall_var(A_LOPG(ast)))
      return TRUE;
    return found_forall_var(A_ROPG(ast));
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return found_forall_var(A_LOPG(ast));
  case A_CMPLXC:
  case A_CNST:
    return FALSE;
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if (found_forall_var(ARGT_ARG(argt, i)))
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    if (found_forall_var(A_LBDG(ast)))
      return TRUE;
    if (found_forall_var(A_UPBDG(ast)))
      return TRUE;
    if (A_STRIDEG(ast) && found_forall_var(A_STRIDEG(ast)))
      return TRUE;
    return FALSE;
  case A_MEM:
    return found_forall_var(A_PARENTG(ast));
  case A_SUBSCR:
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i) {
      if (found_forall_var(ASD_SUBS(asd, i)))
        return TRUE;
    }
    return found_forall_var(A_LOPG(ast));
  case A_ID:
    if (FORALLNDXG(A_SPTRG(ast)))
      return TRUE;
    return FALSE;
  default:
    interr("found_forall_index: bad opc", ast, 3);
    return FALSE;
  }
}

static void
fixup_allocd_tmp_bounds(int *subscr, int *newsubscr, int ndim)
{
  int i;
  int c_subscr;

  /*
   * As per the Fortran spec, ALLOCATE allocates an array of size
   * zero when lb>ub.  If the variable being allocated is a compiler
   * generated temp to hold the result of an expression that has a
   * negative stride, then the lb>ub.  Reset the ub, lb, and stride
   * for this case (tpr3551)
   *
   * Update -- resetting the ub, lb, and stride has the effect of
   * computing the exact size needed for the temp.  However, the
   * subscripts for the temp are not normalized with respect to
   * the actual size -- the original strided subscripts are used
   * and therefore, array bounds violations will occur.  The computed
   * size just needs the direction of the stride (1 or -1) factored in;
   * the direction just needs to be computed as sign(1,stride).
   */

  for (i = 0; i < ndim; ++i) {
    c_subscr = subscr[i];
    if (A_TYPEG(c_subscr) == A_TRIPLE && A_STRIDEG(c_subscr) != astb.bnd.one &&
        A_STRIDEG(c_subscr) != 0) {
      int ub;
      int stride;

      stride = A_STRIDEG(c_subscr);
      if (A_ALIASG(stride)) {
        ISZ_T v;
        v = get_isz_cval(A_SPTRG((A_ALIASG(stride))));
        stride = astb.bnd.one;
        if (v < 0)
          stride = mk_isz_cval(-1, astb.bnd.dtype);

      } else {
        int isign;
        isign = I_ISIGN;
        if (astb.bnd.dtype == DT_INT8) {
          isign = I_KISIGN;
        }
        stride = ast_intr(isign, astb.bnd.dtype, 2, astb.bnd.one, stride);
      }
      ub = mk_binop(OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, A_UPBDG(c_subscr),
                                              A_LBDG(c_subscr), astb.bnd.dtype),
                             stride, astb.bnd.dtype),
                    stride, astb.bnd.dtype);
      newsubscr[i] = mk_triple(astb.bnd.one, ub, 0);
    } else {
      newsubscr[i] = subscr[i];
    }
  }
}

void
fixup_srcalloc_bounds(int *subscr, int *newsubscr, int ndim)
{
  int i;
  int c_subscr;
  for (i = 0; i < ndim; ++i) {
    c_subscr = subscr[i];
    if (A_TYPEG(c_subscr) == A_TRIPLE) {
      int ub;
      int stride;

      stride = A_STRIDEG(c_subscr);
      if (stride == 0)
        stride = astb.bnd.one;

      if (A_ALIASG(stride)) {
        ISZ_T v;
        v = get_isz_cval(A_SPTRG((A_ALIASG(stride))));
        if (v < 0)
          stride = mk_isz_cval(-1, astb.bnd.dtype);
      }
      ub = mk_binop(OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, A_UPBDG(c_subscr),
                                              A_LBDG(c_subscr), astb.bnd.dtype),
                             stride, astb.bnd.dtype),
                    stride, astb.bnd.dtype);

      newsubscr[i] = mk_triple(astb.bnd.one, ub, 0);
    } else {
      newsubscr[i] = subscr[i];
    }
  }
}

/* This routine is just like old routine above.
 * However, it does not try to create temporary
 * based on in indirection, because that is wrong.
 * because distribution becomes wrong.
 * You can not align temporary with one dimension
 * aligned with a template the other dimension aligned with another
 * template.
 */

int
chk_assign_sptr(int arr_ast, const char *purpose, int *subscr, int elem_dty,
                int lhs, int *retval)
{
  int arr_sptr;
  int ast;
  int submap[MAXSUBS];
  int newsubs[MAXSUBS];
  int i, n, j;
  int asd;
  int sptr, ssast;
  ADSC *ad;
  int dtype;
  ADSC *tad;
  int lb, ub;
  int lb1, ub1, st1;
  int vsubsptr[MAXSUBS];
  int extent;

  /* find the array */
  arr_sptr = find_array(arr_ast, &ssast);
  if (ASUMSZG(arr_sptr)) {
    sptr = mk_shape_sptr(A_SHAPEG(ssast), subscr, elem_dty);
    *retval = mk_id(sptr);
    return sptr;
  }

  dtype = DTYPEG(arr_sptr);
  ad = AD_DPTR(dtype);

  /* determine how many dimensions are needed, and which ones they are */
  if (A_TYPEG(ssast) == A_SUBSCR) {
    asd = A_ASDG(ssast);
    n = ASD_NDIM(asd);
  } else {
    asd = 0;
    n = AD_NUMDIM(ad);
  }

  j = 0;
  assert(n <= MAXDIMS, "chk_assign_sptr: too many dimensions", n, 4);
  for (i = 0; i < n; ++i) {
    lb = AD_LWAST(ad, i);
    if (lb == 0)
      lb = mk_isz_cval(1, astb.bnd.dtype);
    lb = check_member(ssast, lb);
    ub = AD_UPAST(ad, i);
    ub = check_member(ssast, ub);
    vsubsptr[j] = 0;
    /* If this is a pointer member, we need to use the shape */
    if (A_TYPEG(ssast) == A_ID && STYPEG(arr_sptr) == ST_MEMBER &&
        POINTERG(arr_sptr)) {
      int shape;
      shape = A_SHAPEG(ssast);
      subscr[j] = mk_triple(SHD_LWB(shape, i), SHD_UPB(shape, i), 0);
      submap[j] = i;
      if (asd)
        ast = ASD_SUBS(asd, i);
      else
        ast = subscr[j];
      newsubs[j] = ast;
      ++j;
    } else if (asd) {
      ast = ASD_SUBS(asd, i);
      /* if it is from where-block
       * include each dimension */
      if (!XBIT(58, 0x20000) && !strcmp(purpose, "ww")) {
        subscr[j] = mk_triple(lb, ub, 0);
        newsubs[j] = ast;
        submap[j] = i;
        ++j;
        continue;
      }
      if (A_TYPEG(ast) == A_TRIPLE) {
        /* include this one */
        if (XBIT(58, 0x20000)) {
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), A_STRIDEG(ast));
          newsubs[j] = ast;
          submap[j] = i;
          ++j;
        } else {
          /* would like to allocate to full size for hpf
           * so all processors get a chunk, even if ignored */
          subscr[j] = mk_triple(lb, ub, 0);
          newsubs[j] = ast;
          submap[j] = i;
          ++j;
        }
      } else if (A_SHAPEG(ast)) {
        /* vector subscript */
        /* (ub-lb+s)/st = extent */
        extent = extent_of_shape(A_SHAPEG(ast), 0);
        lb1 = lb;
        st1 = astb.i1;
        ub1 = opt_binop(OP_SUB, extent, st1, astb.bnd.dtype);
        ub1 = opt_binop(OP_ADD, ub1, lb1, astb.bnd.dtype);
        newsubs[j] = mk_triple(lb1, ub1, 0);
        subscr[j] = newsubs[j];
        submap[j] = i;
        ++j;
      } else if (found_forall_var(ast)) {
        /* a forall index appears in the subscript */
        /* allocate to full size instead of trying to
         * decipher the max/min size of the expression */
        subscr[j] = mk_triple(lb, ub, 0);
        newsubs[j] = ast;
        submap[j] = i;
        ++j;
      }
      /* else don't include scalar dims */
    } else {
      subscr[j] = mk_triple(lb, ub, 0);
      submap[j] = i;
      newsubs[j] = subscr[j];
      ++j;
    }
  }

  assert(j > 0, "chk_assign_sptr: not enough dimensions", j, 4);

  if (lhs && chk_temp_bnds(lhs, arr_sptr, subscr, j)) {
    *retval = lhs;
    return 0;
  }

  /* get the temporary */
  sptr = sym_get_array(SYMNAME(arr_sptr), purpose, elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  ad = AD_DPTR(dtype);
  tad = AD_DPTR(DTYPEG(sptr));
  fixup_allocd_tmp_bounds(newsubs, newsubs, j);
  for (i = 0; i < j; ++i) {
    if (A_TYPEG(newsubs[i]) == A_TRIPLE) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(newsubs[i]);
      AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(newsubs[i]);
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    } else {
      /* assuming A_TYPE is A_CNST or A_ID */
      AD_LWBD(tad, i) = AD_LWAST(tad, i) = AD_UPBD(tad, i) = AD_UPAST(tad, i) =
          newsubs[i];
      AD_EXTNTAST(tad, i) = astb.bnd.one;
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
#ifdef NOEXTENTG
  if ((!HCCSYMG(arr_sptr) && CONTIGUOUS_ARR(arr_sptr)) ||
      (HCCSYMG(arr_sptr) && SCG(arr_sptr) == SC_LOCAL &&
       CONTIGUOUS_ARR(arr_sptr) && NOEXTENTG(arr_sptr))) {
    NOEXTENTP(sptr, 1);
  }
#endif
  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  /* make the subscript expression */
  *retval = mk_subscr(mk_id(sptr), newsubs, j, DTYPEG(sptr));
  return sptr;
}

/*
 * Make a symbol pointer from an array or subscripted array, assuming
 * that that symbol will be assigned the array
 */
int
mk_shape_sptr(int shape, int *subscr, int elem_dty)
{
  int i, n, size, notshort;
  int sptr;
  ADSC *tad;
  int ub;

  /* determine how many dimensions are needed, and which ones they are */
  n = SHD_NDIM(shape);
  assert(n <= MAXDIMS, "mk_assign_sptr: too many dimensions", n, 4);
  for (i = 0; i < n; ++i) {
    /* (ub - lb + stride) / stride */
    assert(SHD_LWB(shape, i), "mk_assign_sptr: lower bound missing", 0, 4);
    assert(SHD_UPB(shape, i), "mk_assign_sptr: upper bound missing", 0, 4);
    if (SHD_STRIDE(shape, i) == astb.i1)
      ub = mk_binop(OP_ADD, mk_binop(OP_SUB, SHD_UPB(shape, i),
                                     SHD_LWB(shape, i), astb.bnd.dtype),
                    astb.bnd.one, astb.bnd.dtype);
    else
      ub = mk_binop(
          OP_DIV, mk_binop(OP_ADD, mk_binop(OP_SUB, SHD_UPB(shape, i),
                                            SHD_LWB(shape, i), astb.bnd.dtype),
                           SHD_STRIDE(shape, i), astb.bnd.dtype),
          SHD_STRIDE(shape, i), astb.bnd.dtype);
    subscr[i] = mk_triple(astb.bnd.one, ub, 0);
  }
  /* get the temporary */
  sptr = sym_get_array("tmp", "r", elem_dty, n);
  /* set the bounds to the correct bounds from the array */
  tad = AD_DPTR(DTYPEG(sptr));
  AD_MLPYR(tad, 0) = astb.bnd.one;
  notshort = 0;
  size = 1;
  for (i = 0; i < n; ++i) {
    AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
    AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
    AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    AD_MLPYR(tad, i + 1) =
        mk_binop(OP_MUL, AD_MLPYR(tad, i), AD_UPBD(tad, i), astb.bnd.dtype);
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  return sptr;
}

/*
 * If this is a temporary allocatable array,
 * see if it is small enough that we should just leave it on the stack.
 */
void
check_small_allocatable(int sptr)
{
  int i, n, ex, small;
  int eldt;
  ISZ_T size;
  ADSC *ad;
  if (!XBIT(2, 0x1000))
    return;
  eldt = DTY(DTYPEG(sptr) + 1);
  if (DTY(eldt) == TY_CHAR
      || DTY(eldt) == TY_NCHAR
      ) {
    if (eldt == DT_ASSCHAR || eldt == DT_DEFERCHAR
        || eldt == DT_ASSNCHAR || eldt == DT_DEFERNCHAR
        )
      return;
    if (!A_ALIASG(DTY(eldt + 1)))
      return;
  }
  ad = AD_DPTR(DTYPEG(sptr));
  n = AD_NUMDIM(ad);
  small = 1;
  size = 1;
  for (i = 0; i < n; ++i) {
    ex = AD_EXTNTAST(ad, i);
    if (!A_ALIASG(ex)) {
      return;
    }
    ex = A_ALIASG(ex);
    size *= ad_val_of(A_SPTRG(ex));
    if (size > 20) {
      return;
    }
  }
  /* still small enough */
  ALLOCP(sptr, 0);
  if (MIDNUMG(sptr)) {
    SCP(sptr, SCG(MIDNUMG(sptr)));
    MIDNUMP(sptr, 0);
  }
} /* check_small_allocatable */

/* if non-constant DIM
 * This routine is to handle non-constant dimension for reduction and spread.
 * Idea is to create temporary with right dimension, and
 * rely on pghpf_reduce_descriptor and pghpf_spread_descriptor
 * for tempoary bounds and alignment. Mark temp as if DYNAMIC since
 * compiler does not know the alignment of temporary.
 * This routine is alo try to use lhs
 */

static int
handle_non_cnst_dim(int arr_ast, const char *purpose, int *subscr, int elem_dty,
                    int dim, int lhs, int *retval, int ndim)
{
  int arr_sptr;
  int newsubs[MAXSUBS];
  int i;
  int sptr;
  int dtype;
  int desc;
  int lb, ub, ssast;

  /* find the array */
  arr_sptr = find_array(arr_ast, &ssast);
  dtype = DTYPEG(arr_sptr);

  /* constant DIM */
  assert(dim != 0, "handle_non_cnst_dim: no dim", 0, 4);
  assert(!A_ALIASG(dim), "handle_non_cnst_dim: dim must be non-constant", 0, 4);

  sptr = sym_get_array(SYMNAME(arr_sptr), purpose, elem_dty, ndim);
  desc = sym_get_sdescr(sptr, ndim);
  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  NODESCP(sptr, 1);
  SECDSCP(DESCRG(sptr), desc);
  /* mark as compiler created */
  HCCSYMP(sptr, 1);
  for (i = 0; i < ndim; ++i) {
    int a;
    lb = get_global_lower(desc, i);
    a = get_extent(desc, i);
    a = mk_binop(OP_SUB, a, mk_cval(1, A_DTYPEG(a)), A_DTYPEG(a));
    ub = mk_binop(OP_ADD, lb, a, A_DTYPEG(lb));
    subscr[i] = newsubs[i] = mk_triple(lb, ub, 0);
  }

  /* *retval = mk_id(sptr);*/
  *retval = mk_subscr(mk_id(sptr), newsubs, ndim, DTYPEG(sptr));
  return sptr;
}

/*
 * Make a symbol pointer from an array or subscripted array, assuming
 * that that symbol will be used as the result of a reduction expression
 * that reduces the array.  One dimension is squeezed out.
 */
int
chk_reduc_sptr(int arr_ast, const char *purpose, int *subscr, int elem_dty,
               int dim, int lhs, int *retval)
{
  int arr_sptr;
  int ast;
  int submap[MAXSUBS];
  int newsubs[MAXSUBS];
  int i, n, j, k;
  int asd;
  int sptr, ssast;
  ADSC *ad;
  int dtype;
  ADSC *tad;

  /* find the array */
  arr_sptr = find_array(arr_ast, &ssast);
  dtype = DTYPEG(arr_sptr);
  ad = AD_DPTR(dtype);

  /* determine how many dimensions are needed, and which ones they are */
  if (A_TYPEG(ssast) == A_SUBSCR) {
    asd = A_ASDG(ssast);
    n = ASD_NDIM(asd);
  } else {
    asd = 0;
    n = AD_NUMDIM(ad);
  }

  /* constant DIM */
  assert(dim != 0, "chk_reduc_sptr: dim must be constant", 0, 4);
  /* if non-constant DIM */
  if (!A_ALIASG(dim))
    return handle_non_cnst_dim(ssast, purpose, subscr, elem_dty, dim, lhs,
                               retval, n - 1);

  dim = get_int_cval(A_SPTRG(A_ALIASG(dim)));

  j = 0; /* dimension counter in temp */
  k = 0; /* vector dimensions in array */
  assert(n <= MAXDIMS, "chk_reduc_sptr: too many dimensions", n, 4);
  for (i = 0; i < n; ++i) {
    if (asd) {
      ast = ASD_SUBS(asd, i);
      if (A_TYPEG(ast) == A_TRIPLE) {
        k++;
        if (k == dim)
          continue;
        if (ASUMSZG(arr_sptr))
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), 0);
        else
          subscr[j] = mk_triple(check_member(arr_ast, AD_LWAST(ad, i)),
                                check_member(arr_ast, AD_UPAST(ad, i)), 0);
        submap[j] = i;
        newsubs[j] = ast;
        ++j;
      }
    } else {
      k++;
      if (k == dim)
        continue;
      subscr[j] = mk_triple(check_member(arr_ast, AD_LWAST(ad, i)),
                            check_member(arr_ast, AD_UPAST(ad, i)), 0);
      submap[j] = i;
      newsubs[j] = subscr[j];
      ++j;
    }
  }
  /* get the temporary */
  assert(k > 1, "chk_reduc_sptr: not enough dimensions", 0, 4);
  assert(j == k - 1, "chk_reduc_sptr: dim out of range", 0, 4);

  if (lhs && chk_temp_bnds(lhs, arr_sptr, subscr, j)) {
    *retval = lhs;
    return 0;
  }

  sptr = sym_get_array(SYMNAME(arr_sptr), purpose, elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  ad = AD_DPTR(dtype);
  tad = AD_DPTR(DTYPEG(sptr));
  if (!ASUMSZG(arr_sptr)) {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) =
          check_member(arr_ast, AD_LWAST(ad, submap[i]));
      AD_UPBD(tad, i) = AD_UPAST(tad, i) =
          check_member(arr_ast, AD_UPAST(ad, submap[i]));
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    }
  } else {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
      AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  *retval = mk_subscr(mk_id(sptr), newsubs, j, DTYPEG(sptr));

  return sptr;
}

static void
mk_temp_based(int sptr)
{
  int tempbase;
  /* create a pointer variable */
  tempbase = get_next_sym(SYMNAME(sptr), "cp");

  /* make the pointer point to sptr */
  STYPEP(tempbase, ST_VAR);
  DTYPEP(tempbase, DT_PTR);
  SCP(tempbase, symutl_sc);

  MIDNUMP(sptr, tempbase);
  SCP(sptr, SC_BASED);
}

/*
 * Make a symbol pointer from an array or subscripted array, assuming
 * that that symbol will be used as the result of a spread expression
 * that adds a dimension to the array.  One dimension is added.
 */
int
mk_spread_sptr(int arr_ast, const char *purpose, int *subscr, int elem_dty,
               int dim, int ncopies, int lhs, int *retval)
{
  int arr_sptr;
  int ast, shape;
  int submap[MAXSUBS];
  int newsubs[MAXSUBS];
  int i, n, j;
  int asd;
  int sptr, ssast;
  ADSC *ad = NULL;
  int dtype;
  ADSC *tad = NULL;
  int ttype = 0;

  /* if it has a scalar spread(3, dim, ncopies) */
  if (A_TYPEG(arr_ast) != A_SUBSCR && A_SHAPEG(arr_ast) == 0) { /*scalar */
    sptr = sym_get_array("spread", purpose, elem_dty, 1);
    /* set the bounds to the correct bounds from the array */
    tad = AD_DPTR(DTYPEG(sptr));
    AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = mk_isz_cval(1, astb.bnd.dtype);
    AD_NUMELM(tad) = AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = ncopies;
    AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0);
    if (elem_dty == DT_ASSCHAR || elem_dty == DT_DEFERCHAR
        || elem_dty == DT_ASSNCHAR || elem_dty == DT_DEFERNCHAR
        ) {
      /* make the temporary a based symbol; mk_mem_allocate() will compute
       * the length
       */
      mk_temp_based(sptr);
    }
    /* make the descriptors for the temporary */
    trans_mkdescr(sptr);
    check_small_allocatable(sptr);
    /* mark as compiler created */
    HCCSYMP(sptr, 1);
    subscr[0] = mk_triple(mk_isz_cval(1, astb.bnd.dtype), ncopies, 0);
    newsubs[0] = subscr[0];
    *retval = mk_subscr(mk_id(sptr), newsubs, 1, DTYPEG(sptr));
    return sptr;
  }

  switch (A_TYPEG(arr_ast)) {
  case A_SUBSCR:
    asd = A_ASDG(arr_ast);
    n = ASD_NDIM(asd);
    for (j = 0; j < n; j++) {
      int sb;
      sb = ASD_SUBS(asd, j);
      if (A_TYPEG(sb) != A_TRIPLE && A_SHAPEG(sb)) {
        /*  has index vector */
        goto no_arr_sptr;
      }
    }
    FLANG_FALLTHROUGH;
  case A_ID:
  case A_MEM:
    arr_sptr = find_array(arr_ast, &ssast);
    ttype = dtype = DTYPEG(arr_sptr);
    ad = AD_DPTR(dtype);
    shape = 0;

    /* determine how many dimensions are needed, and which ones they are */
    if (A_TYPEG(ssast) == A_SUBSCR) {
      asd = A_ASDG(ssast);
      n = ASD_NDIM(asd);
    } else {
      asd = 0;
      n = AD_NUMDIM(ad);
    }
    break;
  default:
  no_arr_sptr:
    arr_sptr = 0;
    dtype = A_DTYPEG(arr_ast);
    ad = NULL;
    asd = 0;
    shape = A_SHAPEG(arr_ast);
    n = SHD_NDIM(shape);
    break;
  }

  /* constant DIM */
  assert(dim != 0, "chk_reduc_sptr: dim must be constant", 0, 4);
  /* if non-constant DIM */
  if (!A_ALIASG(dim))
    return handle_non_cnst_dim(ssast, purpose, subscr, elem_dty, dim, lhs,
                               retval, n + 1);

  dim = get_int_cval(A_SPTRG(A_ALIASG(dim)));

  j = 0;
  assert(n <= MAXDIMS, "chk_spread_sptr: too many dimensions", n, 4);
  for (i = 0; i < n; ++i) {
    if (asd) {
      ast = ASD_SUBS(asd, i);
      if (A_TYPEG(ast) == A_TRIPLE) {
        if (j == dim - 1) {
          /* add before this dimension */
          subscr[j] = mk_triple(mk_isz_cval(1, astb.bnd.dtype), ncopies, 0);
          submap[j] = -1;
          newsubs[j] = subscr[j];
          ++j;
        }
        /* include this one */
        if (ASUMSZG(arr_sptr))
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), 0);
        else
          subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, i)),
                                check_member(ssast, AD_UPAST(ad, i)), 0);
        submap[j] = i;
        newsubs[j] = ast;
        ++j;
      }
    } else {
      if (j == dim - 1) {
        /* add before this dimension */
        subscr[j] = mk_triple(mk_isz_cval(1, astb.bnd.dtype), ncopies, 0);
        submap[j] = -1;
        newsubs[j] = subscr[j];
        ++j;
      }
      if (ad) {
        subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, i)),
                              check_member(ssast, AD_UPAST(ad, i)), 0);
      } else if (shape) {
        subscr[j] = mk_triple(SHD_LWB(shape, i), SHD_UPB(shape, i), 0);
      } else {
        interr("spread with no shape", ast, 3);
      }
      submap[j] = i;
      newsubs[j] = subscr[j];
      ++j;
    }
  }
  if (j == dim - 1) {
    /* add after last dimension */
    subscr[j] = mk_triple(mk_cval(1, DT_INT), ncopies, 0);
    submap[j] = -1;
    newsubs[j] = subscr[j];
    ++j;
  }

  /* get the temporary */
  assert(j > 0, "chk_spread_sptr: not enough dimensions", 0, 4);

  if (arr_sptr) {
    sptr = sym_get_array(SYMNAME(arr_sptr), purpose, elem_dty, j);
  } else {
    sptr = sym_get_array("sprd", purpose, elem_dty, j);
  }
  /* set the bounds to the correct bounds from the array */
  tad = AD_DPTR(DTYPEG(sptr));
  if (ad) {
    if (ttype) {
      ad = AD_DPTR(ttype);
    }
    for (i = 0; i < j; ++i) {
      if (ASUMSZG(arr_sptr)) {
        AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
        AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
        AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
      } else if (submap[i] != -1) {
        AD_LWBD(tad, i) = AD_LWAST(tad, i) =
            check_member(ssast, AD_LWAST(ad, submap[i]));
        AD_UPBD(tad, i) = AD_UPAST(tad, i) =
            check_member(ssast, AD_UPAST(ad, submap[i]));
        AD_EXTNTAST(tad, i) = check_member(ssast, AD_EXTNTAST(ad, submap[i]));
      } else {
        AD_LWBD(tad, i) = AD_LWAST(tad, i) = mk_isz_cval(1, astb.bnd.dtype);
        AD_UPBD(tad, i) = AD_UPAST(tad, i) = ncopies;
        AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
      }
    }
  } else {
    for (i = 0; i < j; ++i) {
      if (submap[i] != -1) {
        AD_LWBD(tad, i) = AD_LWAST(tad, i) = SHD_LWB(shape, submap[i]);
        AD_UPBD(tad, i) = AD_UPAST(tad, i) = SHD_UPB(shape, submap[i]);
        AD_EXTNTAST(tad, i) = mk_extent_expr(SHD_LWB(shape, submap[i]),
                                             SHD_UPB(shape, submap[i]));
      } else {
        AD_LWBD(tad, i) = AD_LWAST(tad, i) = mk_isz_cval(1, astb.bnd.dtype);
        AD_UPBD(tad, i) = AD_UPAST(tad, i) = ncopies;
        AD_EXTNTAST(tad, i) =
            mk_extent_expr(AD_LWAST(tad, i), AD_UPAST(tad, i));
      }
    }
  }
  if (elem_dty == DT_ASSCHAR || elem_dty == DT_DEFERCHAR
      || dtype == DT_ASSNCHAR || elem_dty == DT_DEFERNCHAR
      ) {
    /* make the temporary a based symbol; mk_mem_allocate() will compute
     * the length
     */
    mk_temp_based(sptr);
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  *retval = mk_subscr(mk_id(sptr), newsubs, j, DTYPEG(sptr));
  return sptr;
}

/*
 * Make sptr for matmul
 */
int
mk_matmul_sptr(int arg1, int arg2, const char *purpose, int *subscr,
               int elem_dty, int *retval)
{
  int arr_sptr1, arr_sptr2;
  int ast;
  int submap1[MAXSUBS], submap2[MAXSUBS];
  int subscr1[MAXSUBS], subscr2[MAXSUBS];
  int newsubs1[MAXSUBS], newsubs2[MAXSUBS];
  int newsubs[MAXSUBS];
  int rank1, rank2, rank;
  int i, n, j;
  int asd;
  int sptr, ssast1, ssast2;
  ADSC *ad1, *ad2;
  int dtype;
  ADSC *tad;

  arr_sptr1 = find_array(arg1, &ssast1);
  dtype = DTYPEG(arr_sptr1);
  ad1 = AD_DPTR(dtype);

  /* find the first vector dimension of the first arg */
  if (A_TYPEG(ssast1) == A_SUBSCR) {
    asd = A_ASDG(ssast1);
    n = ASD_NDIM(asd);
  } else {
    asd = 0;
    n = AD_NUMDIM(ad1);
  }
  assert(n <= MAXDIMS, "mk_matmul_sptr: too many dimensions", n, 4);
  j = 0;
  for (i = 0; i < n; ++i) {
    if (asd) {
      ast = ASD_SUBS(asd, i);
      if (A_TYPEG(ast) == A_TRIPLE) {
        int lb, ub;
        if (ASUMSZG(arr_sptr1)) {
          lb = A_LBDG(ast);
          ub = A_UPBDG(ast);
        } else {
          lb = check_member(ssast1, AD_LWAST(ad1, i));
          ub = check_member(ssast1, AD_UPAST(ad1, i));
        }
        subscr1[j] = mk_triple(lb, ub, 0);
        submap1[j] = i;
        newsubs1[j] = ast;
        ++j;
      }
    } else {
      int lb, ub;
      lb = check_member(ssast1, AD_LWAST(ad1, i));
      ub = check_member(ssast1, AD_UPAST(ad1, i));
      subscr1[j] = mk_triple(lb, ub, 0);
      submap1[j] = i;
      newsubs1[j] = subscr1[j];
      ++j;
    }
  }
  rank1 = j;

  arr_sptr2 = find_array(arg2, &ssast2);
  dtype = DTYPEG(arr_sptr2);
  ad2 = AD_DPTR(dtype);

  /* find the second vector dimension of the second arg */
  if (A_TYPEG(ssast2) == A_SUBSCR) {
    asd = A_ASDG(ssast2);
    n = ASD_NDIM(asd);
  } else {
    asd = 0;
    n = AD_NUMDIM(ad2);
  }
  assert(n <= MAXDIMS, "mk_matmul_sptr: too many dimensions", n, 4);
  j = 0;
  for (i = 0; i < n; ++i) {
    if (asd) {
      ast = ASD_SUBS(asd, i);
      if (A_TYPEG(ast) == A_TRIPLE) {
        int lb, ub;
        if (ASUMSZG(arr_sptr2)) {
          lb = A_LBDG(ast);
          ub = A_UPBDG(ast);
        } else {
          lb = check_member(ssast2, AD_LWAST(ad2, i));
          ub = check_member(ssast2, AD_UPAST(ad2, i));
        }
        subscr2[j] = mk_triple(lb, ub, 0);
        submap2[j] = i;
        newsubs2[j] = ast;
        ++j;
      }
    } else {
      int lb, ub;
      lb = check_member(ssast2, AD_LWAST(ad2, i));
      ub = check_member(ssast2, AD_UPAST(ad2, i));
      subscr2[j] = mk_triple(lb, ub, 0);
      submap2[j] = i;
      newsubs2[j] = subscr2[j];
      ++j;
    }
  }
  rank2 = j;

  if (rank1 == 1) {
    /* dimension is second dimension of second array */
    assert(rank2 == 2, "mk_matmul_sptr: rank mismatch (1,2)", 0, 4);
    rank = 1;
    subscr[0] = subscr2[1];
    newsubs[0] = newsubs2[1];
    /* get the temporary */
    sptr = sym_get_array(SYMNAME(arr_sptr1), purpose, elem_dty, rank);
    dtype = DTYPEG(arr_sptr2);
    ad2 = AD_DPTR(dtype);
    /* set the bounds to the correct bounds from the arrays */
    tad = AD_DPTR(DTYPEG(sptr));
    if (ASUMSZG(arr_sptr2)) {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = A_LBDG(subscr[0]);
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = A_UPBDG(subscr[0]);
      AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), i);
    } else {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) =
          check_member(ssast2, AD_LWAST(ad2, submap2[1]));
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) =
          check_member(ssast2, AD_UPAST(ad2, submap2[1]));
      AD_EXTNTAST(tad, 0) = check_member(ssast2, AD_EXTNTAST(ad2, submap2[1]));
    }
  } else if (rank2 == 1) {
    /* dimension is first dimension of first array */
    assert(rank1 == 2, "mk_matmul_sptr: rank mismatch (2,1)", 0, 4);
    rank = 1;
    subscr[0] = subscr1[0];
    newsubs[0] = newsubs1[0];
    /* get the temporary */
    sptr = sym_get_array(SYMNAME(arr_sptr1), purpose, elem_dty, rank);
    dtype = DTYPEG(arr_sptr1);
    ad1 = AD_DPTR(dtype);
    /* set the bounds to the correct bounds from the arrays */
    tad = AD_DPTR(DTYPEG(sptr));
    if (ASUMSZG(arr_sptr1)) {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = A_LBDG(subscr[0]);
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = A_UPBDG(subscr[0]);
      AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), i);
    } else {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) =
          check_member(ssast1, AD_LWAST(ad1, submap1[0]));
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) =
          check_member(ssast1, AD_UPAST(ad1, submap1[0]));
      AD_EXTNTAST(tad, 0) = check_member(ssast1, AD_EXTNTAST(ad1, submap1[0]));
    }
  } else {
    /* dimension is 1st of 1st and 2nd of 2nd */
    assert(rank1 == 2 && rank2 == 2, "mk_matmul_sptr: rank mismatch (2,2)", 0,
           4);
    rank = 2;
    subscr[0] = subscr1[0];
    newsubs[0] = newsubs1[0];
    subscr[1] = subscr2[1];
    newsubs[1] = newsubs2[1];
    /* get the temporary */
    sptr = sym_get_array(SYMNAME(arr_sptr1), purpose, elem_dty, rank);
    dtype = DTYPEG(arr_sptr1);
    ad1 = AD_DPTR(dtype);
    dtype = DTYPEG(arr_sptr2);
    ad2 = AD_DPTR(dtype);
    /* set the bounds to the correct bounds from the arrays */
    tad = AD_DPTR(DTYPEG(sptr));
    if (ASUMSZG(arr_sptr1)) {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = A_LBDG(subscr[0]);
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = A_UPBDG(subscr[0]);
      AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0);
    } else {
      AD_LWBD(tad, 0) = AD_LWAST(tad, 0) =
          check_member(ssast1, AD_LWAST(ad1, submap1[0]));
      AD_UPBD(tad, 0) = AD_UPAST(tad, 0) =
          check_member(ssast1, AD_UPAST(ad1, submap1[0]));
      AD_EXTNTAST(tad, 0) = check_member(ssast1, AD_EXTNTAST(ad1, submap1[0]));
    }
    if (ASUMSZG(arr_sptr2)) {
      AD_LWBD(tad, 1) = AD_LWAST(tad, 1) = A_LBDG(subscr[1]);
      AD_UPBD(tad, 1) = AD_UPAST(tad, 1) = A_UPBDG(subscr[1]);
      AD_EXTNTAST(tad, 1) = mk_extent(AD_LWAST(tad, 1), AD_UPAST(tad, 1), 1);
    } else {
      AD_LWBD(tad, 1) = AD_LWAST(tad, 1) =
          check_member(ssast2, AD_LWAST(ad2, submap2[1]));
      AD_UPBD(tad, 1) = AD_UPAST(tad, 1) =
          check_member(ssast2, AD_UPAST(ad2, submap2[1]));
      AD_EXTNTAST(tad, 1) = check_member(ssast2, AD_EXTNTAST(ad2, submap2[1]));
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);
#ifdef NOEXTENTG
  if (ALLOCG(sptr)) {
    if ((!HCCSYMG(arr_sptr1) && CONTIGUOUS_ARR(arr_sptr1)) ||
        (HCCSYMG(arr_sptr1) && SCG(arr_sptr1) == SC_LOCAL &&
         CONTIGUOUS_ARR(arr_sptr1) && NOEXTENTG(arr_sptr1))) {
      NOEXTENTP(sptr, 1);
    }
  }
#endif
  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  *retval = mk_subscr(mk_id(sptr), newsubs, rank, DTYPEG(sptr));

  return sptr;
}

/*
 * Make sptr for transpose
 */
int
mk_transpose_sptr(int arr_ast, const char *purpose, int *subscr, int elem_dty,
                  int *retval)
{
  int arr_sptr;
  int ast;
  int submap[MAXSUBS];
  int newsubs[MAXSUBS];
  int i, n, j;
  int asd;
  int sptr, ssast;
  ADSC *ad;
  int dtype;
  ADSC *tad;

  arr_sptr = find_array(arr_ast, &ssast);
  dtype = DTYPEG(arr_sptr);
  ad = AD_DPTR(dtype);

  /* determine how many dimensions are needed, and which ones they are */
  if (A_TYPEG(ssast) == A_SUBSCR) {
    asd = A_ASDG(ssast);
    n = ASD_NDIM(asd);
  } else {
    asd = 0;
    n = AD_NUMDIM(ad);
  }
  j = 0;
  assert(n <= MAXDIMS, "mk_transpose_sptr: too many dimensions", n, 4);
  for (i = 0; i < n; ++i) {
    if (asd) {
      ast = ASD_SUBS(asd, i);
      if (A_TYPEG(ast) == A_TRIPLE) {
        /* include this one */
        if (ASUMSZG(arr_sptr))
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), 0);
        else
          subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, i)),
                                check_member(ssast, AD_UPAST(ad, i)), 0);
        submap[j] = i;
        newsubs[j] = ast;
        ++j;
      }
    } else {
      subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, i)),
                            check_member(ssast, AD_UPAST(ad, i)), 0);
      submap[j] = i;
      newsubs[j] = subscr[j];
      ++j;
    }
  }
  /* get the temporary */
  assert(j == 2, "mk_transpose_sptr: not enough dimensions", 0, 4);
  sptr = sym_get_array(SYMNAME(arr_sptr), purpose, elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  ad = AD_DPTR(dtype);
  tad = AD_DPTR(DTYPEG(sptr));
  if (ASUMSZG(arr_sptr)) {
    AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = subscr[1];
    AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = subscr[1];
    AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0);
    AD_LWBD(tad, 1) = AD_LWAST(tad, 1) = subscr[0];
    AD_UPBD(tad, 1) = AD_UPAST(tad, 1) = subscr[0];
    AD_EXTNTAST(tad, 1) = mk_extent(AD_LWAST(tad, 1), AD_UPAST(tad, 1), 1);
  } else {
    AD_LWBD(tad, 0) = AD_LWAST(tad, 0) =
        check_member(arr_ast, AD_LWAST(ad, submap[1]));
    AD_UPBD(tad, 0) = AD_UPAST(tad, 0) =
        check_member(arr_ast, AD_UPAST(ad, submap[1]));
    AD_EXTNTAST(tad, 0) =
        check_member(arr_ast, mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0));
    AD_LWBD(tad, 1) = AD_LWAST(tad, 1) =
        check_member(arr_ast, AD_LWAST(ad, submap[0]));
    AD_UPBD(tad, 1) = AD_UPAST(tad, 1) =
        check_member(arr_ast, AD_UPAST(ad, submap[0]));
    AD_EXTNTAST(tad, 1) =
        check_member(arr_ast, mk_extent(AD_LWAST(tad, 1), AD_UPAST(tad, 1), 1));
  }

  i = newsubs[1];
  newsubs[1] = newsubs[0];
  newsubs[0] = i;
  i = subscr[1];
  subscr[1] = subscr[0];
  subscr[0] = i;

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  *retval = mk_subscr(mk_id(sptr), newsubs, j, DTYPEG(sptr));

  return sptr;
}

int
mk_pack_sptr(int shape, int elem_dty)
{
  int sptr;
  ADSC *tad;

  assert(SHD_NDIM(shape) == 1, "mk_pack_sptr: not rank 1", 0, 4);
  sptr = sym_get_array("pack", "r", elem_dty, 1);
  tad = AD_DPTR(DTYPEG(sptr));
  AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = SHD_LWB(shape, 0);
  AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = SHD_UPB(shape, 0);
  AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0);
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);
  return sptr;
}

/*
 * Replicated array to hold result of 'scalar' min/maxloc
 */
int
mk_maxloc_sptr(int shape, int elem_dty)
{
  int sptr;
  int dtype;
  ADSC *tad;

  assert(SHD_NDIM(shape) == 1, "mk_maxloc_sptr: not rank 1", 0, 4);
  sptr = get_next_sym("mloc", "r");
  dtype = get_array_dtype(1, elem_dty);
  tad = AD_DPTR(dtype);
  AD_LWBD(tad, 0) = AD_LWAST(tad, 0) = SHD_LWB(shape, 0);
  AD_UPBD(tad, 0) = AD_UPAST(tad, 0) = SHD_UPB(shape, 0);
  AD_EXTNTAST(tad, 0) = mk_extent(AD_LWAST(tad, 0), AD_UPAST(tad, 0), 0);
  DTYPEP(sptr, dtype);
  STYPEP(sptr, ST_ARRAY);
  DCLDP(sptr, 1);
  SCP(sptr, symutl.sc);

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);

  return sptr;
}

/* ast to search
 * list pointer of forall indices
 */
int
search_forall_var(int ast, int list)
{
  int argt, n, i;
  int asd;
  int j;

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if ((j = search_forall_var(A_LOPG(ast), list)) != 0)
      return j;
    return search_forall_var(A_ROPG(ast), list);
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return search_forall_var(A_LOPG(ast), list);
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if ((j = search_forall_var(ARGT_ARG(argt, i), list)) != 0)
        return j;
    }
    break;
  case A_TRIPLE:
    if ((j = search_forall_var(A_LBDG(ast), list)) != 0)
      return j;
    if ((j = search_forall_var(A_UPBDG(ast), list)) != 0)
      return j;
    if (A_STRIDEG(ast) && (j = search_forall_var(A_STRIDEG(ast), list)) != 0)
      return j;
    break;
  case A_MEM:
    return search_forall_var(A_PARENTG(ast), list);
  case A_SUBSCR:
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i)
      if ((j = search_forall_var(ASD_SUBS(asd, i), list)) != 0)
        return j;
    return search_forall_var(A_LOPG(ast), list);
  case A_ID:
    for (i = list; i != 0; i = ASTLI_NEXT(i)) {
      if (A_SPTRG(ast) == ASTLI_SPTR(i))
        return i;
    }
    break;
  default:
    interr("search_forall_var: bad opc", ast, 3);
    break;
  }
  return 0;
}

int
other_forall_var(int ast, int list, int fnd)
{
  /* ast to search
   * list pointer of forall indices
   * fnd - astli item of a forall index which appears in ast
   * f2731.
   */
  int argt, n, i;
  int asd;
  int j;

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if ((j = other_forall_var(A_LOPG(ast), list, fnd)) != 0)
      return j;
    return other_forall_var(A_ROPG(ast), list, fnd);
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return other_forall_var(A_LOPG(ast), list, fnd);
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if ((j = other_forall_var(ARGT_ARG(argt, i), list, fnd)) != 0)
        return j;
    }
    break;
  case A_TRIPLE:
    if ((j = other_forall_var(A_LBDG(ast), list, fnd)) != 0)
      return j;
    if ((j = other_forall_var(A_UPBDG(ast), list, fnd)) != 0)
      return j;
    if (A_STRIDEG(ast) &&
        (j = other_forall_var(A_STRIDEG(ast), list, fnd)) != 0)
      return j;
    return 0;
  case A_MEM:
    return other_forall_var(A_PARENTG(ast), list, fnd);
  case A_SUBSCR:
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i)
      if ((j = other_forall_var(ASD_SUBS(asd, i), list, fnd)) != 0)
        return j;
    return other_forall_var(A_LOPG(ast), list, fnd);
  case A_ID:
    for (i = list; i != 0; i = ASTLI_NEXT(i)) {
      if (i == fnd)
        continue;
      if (A_SPTRG(ast) == ASTLI_SPTR(i))
        return i;
    }
    break;
  default:
    interr("other_forall_var: bad opc", ast, 3);
  }
  return 0;
}

static int
find_alloc_size(int ast, int foralllist, int *ss, int *dtype, int *dim)
{
  int argt, n, i;
  int asd;
  int j;

  if (ast <= 0)
    return 0;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    j = find_alloc_size(A_LOPG(ast), foralllist, ss, dtype, dim);
    if (j != 0)
      return j;
    return find_alloc_size(A_ROPG(ast), foralllist, ss, dtype, dim);
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return find_alloc_size(A_LOPG(ast), foralllist, ss, dtype, dim);
  case A_CMPLXC:
  case A_CNST:
    return 0;

  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      j = find_alloc_size(ARGT_ARG(argt, i), foralllist, ss, dtype, dim);
      if (j != 0)
        return j;
    }
    return 0;

  case A_TRIPLE:
    j = find_alloc_size(A_LBDG(ast), foralllist, ss, dtype, dim);
    if (j != 0)
      return j;
    j = find_alloc_size(A_UPBDG(ast), foralllist, ss, dtype, dim);
    if (j != 0)
      return j;
    if (A_STRIDEG(ast)) {
      j = find_alloc_size(A_STRIDEG(ast), foralllist, ss, dtype, dim);
      if (j != 0)
        return j;
    }
    return 0;
  case A_MEM:
    return find_alloc_size(A_PARENTG(ast), foralllist, ss, dtype, dim);
  case A_SUBSCR:
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i) {
      j = find_alloc_size(ASD_SUBS(asd, i), foralllist, ss, dtype, dim);
      if (j != 0) {
        /* this subscript?  another? */
        if (*ss == 0) {
          *ss = ast;
          *dim = i;
          *dtype = DTYPEG(memsym_of_ast(ast));
        }
        return j;
      }
    }
    return find_alloc_size(A_LOPG(ast), foralllist, ss, dtype, dim);
  case A_ID:
    for (i = foralllist; i != 0; i = ASTLI_NEXT(i)) {
      if (A_SPTRG(ast) == ASTLI_SPTR(i))
        return i;
    }
    return 0;
  default:
    interr("find_alloc_size: bad opc", ast, 3);
    return 0;
  }
}

/* subscripts (triples) for temp */
int
mk_forall_sptr_copy_section(int forall_ast, int lhs, int rhs, int *subscr,
                            int elem_dty)
{
  int arr_sptr, ssast;
  int ast;
  int submap[MAXSUBS];
  int i, n, j;
  int asd;
  int sptr;
  ADSC *ad;
  int dtype;
  ADSC *tad;
  int list;
  int asd1;
  int n1;
  int k;
  LOGICAL found;
  int astli, astli1;
  int nidx, nidx1;

  /* find the array */
  assert(A_TYPEG(lhs) == A_SUBSCR,
         "mk_forall_sptr_copy_section: ast not subscript", lhs, 4);
  arr_sptr = find_array(lhs, &ssast);
  dtype = DTYPEG(arr_sptr);
  assert(DTY(dtype) == TY_ARRAY,
         "mk_forall_sptr_copy_section: subscr sym not ARRAY", arr_sptr, 4);
  ad = AD_DPTR(dtype);

  /* get the forall index list */
  assert(A_TYPEG(forall_ast) == A_FORALL,
         "mk_forall_sptr_copy_section: ast not forall", forall_ast, 4);
  list = A_LISTG(forall_ast);

  /* determine how many dimensions are needed, and which ones they are */
  asd = A_ASDG(lhs);
  n = ASD_NDIM(asd);
  asd1 = A_ASDG(rhs);
  n1 = ASD_NDIM(asd1);

  j = 0;
  assert(n <= MAXDIMS && n1 <= MAXDIMS,
    "mk_forall_sptr_copy_section: too many dimensions", 0, 4);
  for (k = 0; k < n1; ++k) {
    astli1 = 0;
    nidx1 = 0;
    search_forall_idx(ASD_SUBS(asd1, k), list, &astli1, &nidx1);
    assert(nidx1 < 2, "mk_forall_sptr_copy_section: something is wrong", 2,
           rhs);
    if (nidx1 == 1 && astli1) {
      found = FALSE;
      for (i = 0; i < n; i++) {
        astli = 0;
        nidx = 0;
        search_forall_idx(ASD_SUBS(asd, i), list, &astli, &nidx);
        if (astli != 0) {
          assert(nidx == 1, "mk_forall_sptr_copy_section: something is wrong",
                 2, lhs);
          if (astli == astli1) {
            found = TRUE;
            break;
          }
        }
      }

      assert(found, "mk_forall_sptr_copy_section: something is wrong", 0, 4);

      /* include this dimension */
      /* build a triplet for the allocate statement off of the
       * dimensions for a */
      if (ASUMSZG(arr_sptr)) {
        ast = ASD_SUBS(asd, i);
        if (A_TYPEG(ast) == A_TRIPLE)
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), 0);
        else if (A_SHAPEG(ast)) {
          int shd;
          shd = A_SHAPEG(ast);
          subscr[j] = mk_triple(SHD_LWB(shd, i), SHD_UPB(shd, i), 0);
        } else
          subscr[j] = mk_triple(A_LBDG(ASTLI_TRIPLE(astli)),
                                A_UPBDG(ASTLI_TRIPLE(astli)), 0);
        submap[j] = i;
        ++j;
      } else {
        subscr[j] = mk_triple(check_member(lhs, AD_LWAST(ad, i)),
                              check_member(lhs, AD_UPAST(ad, i)), 0);
        submap[j] = i;
        ++j;
      }
    }
  }
  /* get the temporary */
  assert(j > 0, "mk_forall_sptr_copy_section: not enough dimensions", 0, 4);
  sptr = sym_get_array(SYMNAME(arr_sptr), "cs", elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  ad = AD_DPTR(dtype); /* may have realloc'd */
  tad = AD_DPTR(DTYPEG(sptr));
  if (!ASUMSZG(arr_sptr)) {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) =
          check_member(lhs, AD_LWAST(ad, submap[i]));
      AD_UPBD(tad, i) = AD_UPAST(tad, i) =
          check_member(lhs, AD_UPAST(ad, submap[i]));
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    }
  } else {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
      AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  return sptr;
}

/*  This is just like mk_forall_sptr_copy_section expect that
 *  It also includes scalar dimension.
 */
int
mk_forall_sptr_gatherx(int forall_ast, int lhs, int rhs, int *subscr,
                       int elem_dty)
{
  int arr_sptr;
  int ast, ssast;
  int submap[MAXSUBS];
  int i, n, j;
  int asd;
  int sptr;
  ADSC *ad;
  int dtype;
  ADSC *tad;
  int list;
  int asd1;
  int n1;
  int k;
  LOGICAL found;
  int astli, astli1;
  int nidx, nidx1;

  /* find the array */
  assert(A_TYPEG(lhs) == A_SUBSCR, "mk_forall_sptr_gatherx: ast not subscript",
         lhs, 4);
  arr_sptr = find_array(lhs, &ssast);
  dtype = DTYPEG(arr_sptr);
  assert(DTY(dtype) == TY_ARRAY, "mk_forall_sptr_gatherx: subscr sym not ARRAY",
         arr_sptr, 4);
  ad = AD_DPTR(dtype);

  /* get the forall index list */
  assert(A_TYPEG(forall_ast) == A_FORALL,
         "mk_foral_sptr_gatherx: ast not forall", forall_ast, 4);
  list = A_LISTG(forall_ast);

  /* determine how many dimensions are needed, and which ones they are */
  asd = A_ASDG(lhs);
  n = ASD_NDIM(asd);
  asd1 = A_ASDG(rhs);
  n1 = ASD_NDIM(asd1);

  j = 0;
  assert(n <= MAXDIMS && n1 <= MAXDIMS,
    "mk_forall_sptr_gatherx: too many dimensions", 0, 4);
  for (k = 0; k < n1; ++k) {
    astli1 = 0;
    nidx1 = 0;
    search_forall_idx(ASD_SUBS(asd1, k), list, &astli1, &nidx1);
    assert(nidx1 < 2, "mk_forall_sptr_gatherx: something is wrong", 2, rhs);
    if (nidx1 == 1 && astli1) {
      found = FALSE;
      for (i = 0; i < n; i++) {
        astli = 0;
        nidx = 0;
        search_forall_idx(ASD_SUBS(asd, i), list, &astli, &nidx);
        if (astli != 0) {
          assert(nidx == 1, "mk_forall_sptr_gatherx: something is wrong", 2,
                 lhs);
          if (astli == astli1) {
            found = TRUE;
            break;
          }
        }
      }

      assert(found, "mk_forall_sptr_gatherx: something is wrong", 0, 4);

      /* include this dimension */
      /* build a triplet for the allocate statement off of the
       * dimensions for a */
      if (ASUMSZG(arr_sptr)) {
        ast = ASD_SUBS(asd, i);
        if (A_TYPEG(ast) == A_TRIPLE)
          subscr[j] = mk_triple(A_LBDG(ast), A_UPBDG(ast), 0);
        else if (A_SHAPEG(ast)) {
          int shd;
          shd = A_SHAPEG(ast);
          subscr[j] = mk_triple(SHD_LWB(shd, i), SHD_UPB(shd, i), 0);
        } else
          subscr[j] = mk_triple(A_LBDG(ASTLI_TRIPLE(astli)),
                                A_UPBDG(ASTLI_TRIPLE(astli)), 0);
        submap[j] = i;
        ++j;
      } else {
        subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, i)),
                              check_member(ssast, AD_UPAST(ad, i)), 0);
        submap[j] = i;
        ++j;
      }
    } else if (nidx1 == 0 && astli1 == 0) {
      /* include scalar dimension too */
      subscr[j] = mk_triple(check_member(ssast, AD_LWAST(ad, k)),
                            check_member(ssast, AD_UPAST(ad, k)), 0);
      submap[j] = k;
      ++j;
    }
  }
  /* get the temporary */
  assert(j > 0, "mk_forall_sptr_gatherx: not enough dimensions", 0, 4);
  sptr = sym_get_array(SYMNAME(arr_sptr), "g", elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  ad = AD_DPTR(dtype); /* may have realloc'd */
  tad = AD_DPTR(DTYPEG(sptr));
  if (!ASUMSZG(arr_sptr)) {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) =
          check_member(ssast, AD_LWAST(ad, submap[i]));
      AD_UPBD(tad, i) = AD_UPAST(tad, i) =
          check_member(ssast, AD_UPAST(ad, submap[i]));
      AD_EXTNTAST(tad, i) = check_member(ssast, AD_EXTNTAST(ad, submap[i]));
    }
  } else {
    for (i = 0; i < j; ++i) {
      AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
      AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
      AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
    }
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  return sptr;
}

/* the size and shape of TMP will be based on both LHS and RHS.
 * There are two rules for TMP:
 *        1-) heading dimensions size and distribution from LHS
 *        2-) tailling dimensions size from shape of arg with no distribution
 */
int
mk_forall_sptr_pure(int forall_ast, int lhs, int rhs, int *subscr, int elem_dty)
{
  int submap[MAXSUBS];
  int i, j;
  int asd;
  int sptr;
  int lhs_sptr, rhs_sptr, lhsmem, rhsmem;
  int ndim;
  ADSC *ad;
  ADSC *tad;
  int list;

  assert(A_TYPEG(lhs) == A_SUBSCR, "mk_forall_sptr_pure: ast not subscript",
         lhs, 4);
  assert(A_TYPEG(rhs) == A_SUBSCR, "mk_forall_sptr_pure: ast not subscript",
         rhs, 4);
  lhs_sptr = memsym_of_ast(lhs);
  lhsmem = A_LOPG(lhs);
  if (A_TYPEG(lhsmem) != A_MEM)
    lhsmem = 0;
  assert(DTY(DTYPEG(lhs_sptr)) == TY_ARRAY,
         "mk_forall_sptr_pure: subscr sym not ARRAY", lhs_sptr, 4);
  rhs_sptr = memsym_of_ast(rhs);
  rhsmem = A_LOPG(rhs);
  if (A_TYPEG(rhsmem) != A_MEM)
    rhsmem = 0;
  assert(DTY(DTYPEG(rhs_sptr)) == TY_ARRAY,
         "mk_forall_sptr_pure: subscr sym not ARRAY", rhs_sptr, 4);

  /* get the forall index list */
  assert(A_TYPEG(forall_ast) == A_FORALL, "mk_forall_sptr_pure: ast not forall",
         forall_ast, 4);
  list = A_LISTG(forall_ast);

  /* determine how many dimensions are needed, and which ones they are */
  asd = A_ASDG(lhs);
  ndim = ASD_NDIM(asd);
  ad = AD_DPTR(DTYPEG(lhs_sptr));
  j = 0;
  /* find size and distribution of heading dimension from lhs */
  for (i = 0; i < ndim; i++) {
    /* include this dimension */
    if (search_forall_var(ASD_SUBS(asd, i), list)) {
      subscr[j] = mk_triple(check_member(lhsmem, AD_LWAST(ad, i)),
                            check_member(lhsmem, AD_UPAST(ad, i)), 0);
      submap[j] = i;
      ++j;
    }
  }

  /* find tailling dimension from rhs with no distribution*/
  ad = AD_DPTR(DTYPEG(rhs_sptr));
  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    if (A_TYPEG(ASD_SUBS(asd, i)) == A_TRIPLE || A_SHAPEG(ASD_SUBS(asd, i))) {
      /* include this dimension */
      subscr[j] = mk_triple(check_member(rhsmem, AD_LWAST(ad, i)),
                            check_member(rhsmem, AD_UPAST(ad, i)), 0);
      submap[j] = -1;
      ++j;
    }
  }

  /* get the temporary */
  assert(j > 0 && j <= MAXDIMS, "mk_forall_sptr_pure: not enough dimensions",
    j, 4);
  sptr = sym_get_array(SYMNAME(lhs_sptr), "pure", elem_dty, j);
  /* set the bounds to the correct bounds from the array */
  tad = AD_DPTR(DTYPEG(sptr));
  for (i = 0; i < j; ++i) {
    AD_LWBD(tad, i) = AD_LWAST(tad, i) = A_LBDG(subscr[i]);
    AD_UPBD(tad, i) = AD_UPAST(tad, i) = A_UPBDG(subscr[i]);
    AD_EXTNTAST(tad, i) = mk_extent(AD_LWAST(tad, i), AD_UPAST(tad, i), i);
  }

  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  check_small_allocatable(sptr);

  /* mark as compiler created */
  HCCSYMP(sptr, 1);

  return sptr;
}

int
first_element(int ast)
{
  int i, n, subs[MAXSUBS], asd, ss, doit, lop, dtype, parent, sptr;
  switch (A_TYPEG(ast)) {
  case A_SUBSCR:
    /* if any subscript is a triplet, take the first one */
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    doit = 0;
    for (i = 0; i < n; ++i) {
      ss = ASD_SUBS(asd, i);
      if (A_TYPEG(ss) == A_TRIPLE) {
        subs[i] = A_LBDG(ss);
        doit = 1;
      } else {
        subs[i] = ss;
      }
    }
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_MEM) {
      parent = first_element(A_PARENTG(lop));
      if (parent != A_PARENTG(lop)) {
        doit = 1;
        lop = mk_member(parent, A_MEMG(lop), A_DTYPEG(lop));
      }
    }
    if (doit) {
      ast = mk_subscr(lop, subs, n, A_DTYPEG(ast));
    }
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    parent = 0;
    goto hit;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(ast));
    parent = first_element(A_PARENTG(ast));
    if (parent != A_PARENTG(ast)) {
      ast = mk_member(parent, A_MEMG(ast), A_DTYPEG(ast));
    }
  hit:
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY) {
      n = ADD_NUMDIM(dtype);
      for (i = 0; i < n; ++i) {
        if (ADD_LWAST(dtype, i))
          subs[i] = check_member(ast, ADD_LWAST(dtype, i));
        else
          subs[i] = astb.bnd.one;
      }
      ast = mk_subscr(ast, subs, n, DTY(dtype + 1));
    }
  }
  return ast;
} /* first_element */

int
mk_mem_allocate(int in_ast, int *subscr, int alloc_stmt, int ast_len_from)
{
  int n, ast, shape, dtype, eldtype, sptr;
  int atp;
  int newstd = 0;
  int par;
  int task;

  par = STD_PAR(alloc_stmt);
  task = STD_TASK(alloc_stmt);
  shape = A_SHAPEG(in_ast);
  assert(shape != 0, "mk_mem_allocate: no shape", in_ast, 4);
  n = SHD_NDIM(shape);
  if (A_TYPEG(in_ast) == A_ID) {
    sptr = A_SPTRG(in_ast);
  } else if (A_TYPEG(in_ast) == A_MEM) {
    sptr = A_SPTRG(A_MEMG(in_ast));
  } else {
    interr("mk_mem_allocate: not id/member", in_ast, 4);
    sptr = 0;
  }
  if (sptr && !ALLOCG(sptr) && !POINTERG(sptr) && !ADJARRG(sptr) &&
      !ADJLENG(sptr))
    return 0;
  dtype = A_DTYPEG(in_ast);
  eldtype = DDTG(dtype);
  if (ast_len_from && (eldtype == DT_ASSCHAR || eldtype == DT_ASSNCHAR ||
                       eldtype == DT_DEFERCHAR || eldtype == DT_DEFERNCHAR) &&
      !EARLYSPECG(sptr)) {
    int cvsptr, cvast, cvlenast;
    int cvlen = CVLENG(sptr);
    if (eldtype == DT_ASSCHAR || eldtype == DT_ASSNCHAR) {
      if (cvlen == 0) {
        cvlen = sym_get_scalar(SYMNAME(sptr), "len", DT_INT);
        CVLENP(sptr, cvlen);
        if (SCG(sptr) == SC_DUMMY)
          CCSYMP(cvlen, 1);
      }
      ADJLENP(sptr, 1);
      cvlenast = mk_id(cvlen);
    } else {
      cvlenast = get_len_of_deferchar_ast(in_ast);
    }
    ast = mk_stmt(A_ASN, 0);
    A_DESTP(ast, cvlenast);

    /* see if the source length can be resolved a little */
    cvsptr = 0;
    cvast = ast_len_from;
    if (A_TYPEG(cvast) == A_SUBSCR) {
      cvast = A_LOPG(cvast);
    }
    if (A_TYPEG(cvast) == A_ID) {
      cvsptr = A_SPTRG(cvast);
    } else if (A_TYPEG(cvast) == A_MEM) {
      cvsptr = A_SPTRG(A_MEMG(cvast));
    }
    if (cvsptr) {
      int cvdtype = DDTG(DTYPEG(cvsptr));
      if (cvdtype == DT_ASSCHAR || cvdtype == DT_ASSNCHAR) {
        if (CVLENG(cvsptr)) {
          atp = mk_id(CVLENG(cvsptr));
          A_SRCP(ast, atp);
        } else { /* formal argument */
          atp = size_ast(cvsptr, cvdtype);
          A_SRCP(ast, atp);
        }
      } else if (cvdtype == DT_DEFERCHAR || cvdtype == DT_DEFERNCHAR) {
        cvsptr = 0;
      } else if (DTY(cvdtype) == TY_CHAR || DTY(cvdtype) == TY_NCHAR) {
        A_SRCP(ast, DTY(cvdtype + 1));
      } else {
        cvsptr = 0;
      }
    }

    if (cvsptr == 0) {
      ast_len_from = first_element(ast_len_from);
      atp = ast_intr(I_LEN, DT_INT, 1, ast_len_from);
      A_SRCP(ast, atp);
    }
    newstd = add_stmt_before(ast, alloc_stmt);
    STD_PAR(newstd) = par;
    STD_TASK(newstd) = task;
  } else if ((DTY(eldtype) == TY_CHAR || DTY(eldtype) == TY_NCHAR) &&
             (DTY(eldtype + 1) == 0 ||
              (DTY(eldtype + 1) > 0 && !A_ALIASG(DTY(eldtype + 1)))) &&
             !EARLYSPECG(sptr)) {
    /* nonconstant length */
    int rhs;
    int cvlen = CVLENG(sptr);
    if (cvlen == 0) {
      cvlen = sym_get_scalar(SYMNAME(sptr), "len", DT_INT);
      CVLENP(sptr, cvlen);
      if (SCG(sptr) == SC_DUMMY)
        CCSYMP(cvlen, 1);
    }
    ADJLENP(sptr, 1);
    ast = mk_stmt(A_ASN, 0);
    atp = mk_id(cvlen);
    A_DESTP(ast, atp);
    rhs = DTY(eldtype + 1);
    rhs = mk_convert(rhs, DTYPEG(cvlen));
    rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));
    A_SRCP(ast, rhs);
    newstd = add_stmt_before(ast, alloc_stmt);
    STD_PAR(newstd) = par;
    STD_TASK(newstd) = task;
  }
  /* build and insert the allocate statement */
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_ALLOCATE);
  A_LOPP(ast, 0);
  if (subscr != 0) {
    /*
     * As per the Fortran spec, ALLOCATE allocates an array of size
     * zero when lb>ub.  If the variable being allocated is a compiler
     * generated temp to hold the result of an expression that has a
     * negative stride, then the lb>ub.  Reset the ub, lb, and stride
     * for this case (tpr3551)
     *
     * Update -- resetting the ub, lb, and stride has the effect of
     * computing the exact size needed for the temp.  However, the
     * subscripts for the temp are not normalized with respect to
     * the actual size -- the original strided subscripts are used
     * and therefore, array bounds violations will occur.  The computed
     * size just needs the direction of the stride (1 or -1) factored in;
     * the direction just needs to be computed as sign(1,stride).
     */
    if (A_TYPEG(in_ast) == A_ID && (HCCSYMG(sptr) || CCSYMG(sptr))) {
      int newsubscr[MAXSUBS];

      fixup_allocd_tmp_bounds(subscr, newsubscr, n);

      atp = mk_subscr(in_ast, newsubscr, n, DDTG(A_DTYPEG(in_ast)));
    } else {
      atp = mk_subscr(in_ast, subscr, n, DDTG(A_DTYPEG(in_ast)));
    }
    A_SRCP(ast, atp);
  } else
    A_SRCP(ast, in_ast);
  newstd = add_stmt_before(ast, alloc_stmt);
  STD_PAR(newstd) = par;
  STD_TASK(newstd) = task;
  return newstd;
}

/* see mk_mem_allocate -- should be always called */
int
mk_mem_deallocate(int in_ast, int dealloc_stmt)
{
  int ast, sptr;
  int par, task;
  int newstd = 0;

  /* build and insert the deallocate statement */
  assert(A_SHAPEG(in_ast), "mk_mem_deallocate: var not array", in_ast, 4);
  sptr = memsym_of_ast(in_ast);
  if (sptr && !ALLOCG(sptr) && !POINTERG(sptr) && !ADJARRG(sptr) &&
      !ADJLENG(sptr))
    return 0;
  par = STD_PAR(dealloc_stmt);
  task = STD_TASK(dealloc_stmt);
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_DEALLOCATE);
  A_LOPP(ast, 0);
  A_SRCP(ast, in_ast);
  newstd = add_stmt_after(ast, dealloc_stmt);
  STD_PAR(newstd) = par;
  STD_TASK(newstd) = task;
  return newstd;
}

typedef struct {
  int count0;
  int count1;
  int count2;
  int count3;
  int count4;
  int count5;
  int count6;
  int count7;
  int count8;
  int count9;
  int count10;
  int count11;
  int count12;
} BOUND;

static BOUND bound;

void
init_bnd(void)
{
  if (gbl.internal)
    return;
  bound.count0 = 0;
  bound.count1 = 0;
  bound.count2 = 0;
  bound.count3 = 0;
  bound.count4 = 0;
  bound.count5 = 0;
  bound.count6 = 0;
  bound.count7 = 0;
  bound.count8 = 0;
  bound.count9 = 0;
  bound.count10 = 0;
  bound.count11 = 0;
  bound.count12 = 0;
}

int
getbnd(const char *basename, const char *purpose, int n, int dtype)
{
  int sptr;

#if DEBUG
  assert(n >= 0 && n <= 99999, "getbnd-n too large", n, 0);
#endif
  if (n) {
    if (purpose)
      sptr = getsymf("%s%s%s%d", basename, "$$", purpose, n);
    else
      sptr = getsymf("%s%d", basename, n);
  } else {
    if (purpose)
      sptr = getsymf("%s%s%s", basename, "$$", purpose);
    else
      sptr = getsymbol(basename);
  }

  if (gbl.internal > 1 && !INTERNALG(sptr))
    sptr = insert_sym(sptr);
  assert(STYPEG(sptr) == ST_UNKNOWN, "getbnd: name crash", sptr, 2);
  DTYPEP(sptr, dtype);
  STYPEP(sptr, ST_VAR);
  DCLDP(sptr, 1);
  SCP(sptr, SC_LOCAL);
  NODESCP(sptr, 1);
  HCCSYMP(sptr, 1);
  return sptr;
}

int
trans_getbound(int sym, int type)
{
  int sptr;

  switch (type) {
  case 0:
    sptr = getbnd("i", "l", bound.count0, DT_INT);
    bound.count0++;
    return sptr;
  case 1:
    sptr = getbnd("i", "u", bound.count1, DT_INT);
    bound.count1++;
    return sptr;
  case 2:
    sptr = getbnd("i", "s", bound.count2, DT_INT);
    bound.count2++;
    return sptr;
  case 3:
    sptr = getbnd("c", "l", bound.count3, DT_INT);
    bound.count3++;
    return sptr;
  case 4:
    sptr = getbnd("c", "u", bound.count4, DT_INT);
    bound.count4++;
    return sptr;
  case 5:
    sptr = getbnd("c", "cs", bound.count5, DT_INT);
    bound.count5++;
    return sptr;
  case 6:
    sptr = getbnd("c", "lo", bound.count6, DT_INT);
    bound.count6++;
    return sptr;
  case 7:
    sptr = getbnd("c", "ls", bound.count7, DT_INT);
    bound.count7++;
    return sptr;
  case 8:
    sptr = getbnd("i", "c", bound.count8, DT_INT);
    bound.count8++;
    return sptr;
  case 9:
    sptr = getbnd("l", "b", bound.count9, DT_INT);
    bound.count9++;
    return sptr;
  case 10:
    sptr = getbnd("u", "b", bound.count10, DT_INT);
    bound.count10++;
    return sptr;
  case 11:
    sptr = getbnd("cp", "com", bound.count11, DT_INT);
    bound.count11++;
    return sptr;
  case 12:
    sptr = getbnd("xfer", "com", bound.count12, DT_INT);
    bound.count12++;
    return sptr;
  default:
    assert(TRUE, "trans_getbound: unknown type", 0, 4);
    sptr = getbnd("i", "l", bound.count0, DT_INT);
    bound.count0++;
    return sptr;
  }
}

/* astmem is either zero, or an A_MEM */
/* astid is an A_ID */
/* if astmem is zero, return astid; otherwise, return an A_MEM with
 * the same parent as astmem, and with astid as member */
int
check_member(int astmem, int astid)
{
  if (astmem != 0 && A_TYPEG(astmem) == A_SUBSCR) {
    astmem = A_LOPG(astmem);
  }
  if (astmem == 0 || A_TYPEG(astmem) != A_MEM) {
    int sptr, stype;
    /* error check */
    /* astid may be A_ID or A_SUBSCR */
    if (A_TYPEG(astid) == A_ID) {
      sptr = A_SPTRG(astid);
    } else if (A_TYPEG(astid) == A_SUBSCR) {
      int lop;
      lop = A_LOPG(astid);
      if (A_TYPEG(lop) != A_ID)
        return astid;
      sptr = A_SPTRG(lop);
    } else {
      return astid;
    }
    stype = STYPEG(sptr);
    if (stype == ST_ARRDSC) {
      int secdsc;
      secdsc = SECDSCG(sptr);
      if (secdsc) {
        stype = STYPEG(secdsc);
      } else {
        stype = STYPEG(ARRAYG(sptr));
      }
    }
    if (stype == ST_MEMBER && !DESCARRAYG(sptr)) {
      interr("check_member: cannot match member with derived type", sptr, 3);
    }
    return astid;
  }

  /* In the new array/pointer runtime descriptor, the extent may be an
   * expression of lower bound and upper bound.  Handle this case.
   */
  if (A_TYPEG(astid) == A_BINOP) {
    /* get the values we need, in case astb.stg_base gets reallocated! */
    int lop = check_member(astmem, A_LOPG(astid));
    int rop = check_member(astmem, A_ROPG(astid));
    return mk_binop(A_OPTYPEG(astid), lop, rop, A_DTYPEG(astid));
  }

  /* check that the ID is a ST_MEMBER of the same datatype */
  if (A_TYPEG(astid) == A_ID) {
    return do_check_member_id(astmem, astid);
  }
  if (A_TYPEG(astid) == A_SUBSCR) {
    int lop = A_LOPG(astid);
    if (A_TYPEG(lop) != A_ID) {
      return astid;
    } else {
      int subs[MAXSUBS];
      int i;
      int lop2 = do_check_member_id(astmem, lop);
      int asd = A_ASDG(astid);
      int n = ASD_NDIM(asd);
      for (i = 0; i < n; ++i) {
        subs[i] = ASD_SUBS(asd, i);
      }
      return mk_subscr(lop2, subs, n, A_DTYPEG(astid));
    }
  }
  return astid;
} /* check_member */

static int
do_check_member_id(int astmem, int astid)
{
  int sptr2, checksptr;
  int sptr = A_SPTRG(astid);
  SYMTYPE stype = STYPEG(sptr);
  assert(A_TYPEG(astid) == A_ID, "expecting A_TYPE == A_ID",
    A_TYPEG(astid), ERR_Fatal);
  if (XBIT(58, 0x10000) && stype == ST_ARRDSC) {
    checksptr = ARRAYG(sptr);
    if (checksptr && SDSCG(checksptr)) {
      checksptr = SDSCG(checksptr);
      stype = STYPEG(checksptr);
    } else if (checksptr) {
      /* see if the array is a member and needs a local
        * section descriptor */
      DTYPE dtype = DTYPEG(checksptr);
      if (ALIGNG(checksptr) || DISTG(checksptr) || POINTERG(checksptr) ||
          ADD_DEFER(dtype) || ADD_ADJARR(dtype) || ADD_NOBOUNDS(dtype)) {
        stype = STYPEG(checksptr);
      }
    }
  } else {
    checksptr = sptr;
  }
  if (stype != ST_MEMBER) {
    return astid;
  }
  sptr2 = A_SPTRG(A_MEMG(astmem));
  if (ENCLDTYPEG(checksptr) != ENCLDTYPEG(sptr2)) {
    interr("check_member: member arrived with wrong derived type", sptr, 3);
  }
  return mk_member(A_PARENTG(astmem), astid, A_DTYPEG(astid));
}

/* get the first symbol with the same hash link */
int
first_hash(int sptr)
{
  char *name;
  int len, hashval;
  name = SYMNAME(sptr);
  len = strlen(name);
  HASH_ID(hashval, name, len);
  if (hashval < 0)
    hashval = -hashval;
  return stb.hashtb[hashval];
} /* first_hash */

LOGICAL
has_allocattr(int sptr)
{
  int dtype;
  if (ALLOCATTRG(sptr))
    return TRUE;
  dtype = DTYPEG(sptr);
  dtype = DDTG(dtype);
  if (DTY(dtype) == TY_DERIVED && ALLOCFLDG(DTY(dtype + 3)))
    return TRUE;
  return FALSE;
}

/**
  * \brief utility function for visiting symbols of a specified name.
  *
  * This function is used to visit symbols of a specified name. The user of
  * this function should first initialize the search by calling the function
  * with task == 0.
  *
  * After initializing the search, the user can call this function with the
  * same sptr and a task == 1 or task == 2. The function
  * will return the sptr of the next unvisited symbol based on the criteria
  * specified in the task argument. Below summarizes the values for task:
  *
  * If task == 0, then this function will unset the VISIT flag for all symbols
  * with the same name as sptr.
  *
  * If task == 1, then return symbol must have the same symbol table type
  * as the sptr argument.
  *
  * If task == 2, then the returned symbol can have any symbol table type.
  *
  * Caveat: Do not use this function when another phase is using the
  *         the VISIT field. For example, during the lower phase.
  *
  * \param sptr is the symbol table pointer of the name you wish to find.
  * \param task is the task the function will perform (see comments above).
  *
  * \return symbol table pointer of next unvisited symbol or 0 if no more
  *         symbols have been found.
  */
int
get_next_hash_link(int sptr, int task)
{
  int hash, hptr, len;
  char *symname;

  if (!sptr)
    return 0;
  symname = SYMNAME(sptr);
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  if (task == 0) {
    /* init visit flag for all symbols with same name as sptr */
    for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
      if (STYPEG(hptr) == STYPEG(sptr) && strcmp(symname, SYMNAME(hptr)) == 0) {
        VISITP(hptr, 0);
      }
    }
  } else if (task == 1) {
    VISITP(sptr, 1);
    for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
      if (hptr != sptr && !VISITG(hptr) && STYPEG(hptr) == STYPEG(sptr) &&
          strcmp(symname, SYMNAME(hptr)) == 0) {
        return hptr;
      }
    }
  } else if (task == 2) {
    VISITP(sptr, 1);
    for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
      if (hptr != sptr && !VISITG(hptr) &&
          strcmp(symname, SYMNAME(hptr)) == 0) {
        return hptr;
      }
    }
  }
  return 0;
}

/** \brief utility function for finding a symbol in the symbol table that
  * has a specified name.
  *
  * Note: The first symbol that meets the criteria specified in the arguments
  * is returned. If you need to make multiple queries of the symbol table,
  * consider using the function get_next_hash_link() instead.
  *
  * \param symname is a C string that specifies the name of the symbol to find
  * \param stype specifies a particular symbol type to find or 0 will locate
  *        any symbol type.
  * \param scope specifies the scope symbol table pointer to search for the
  *        symbol. If scope is 0 then the symbol can appear in any scope.
  *        If scope is -1, then the first symbol that's in scope is returned.
  *
  * \return symbol table pointer of the first symbol found that meets the
  *         criteria mentioned above; else 0.
  */
int
findByNameStypeScope(char *symname, int stype, int scope)
{
  int hash, hptr, len;
  len = strlen(symname);
  HASH_ID(hash, symname, len);
  for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
    if ((stype == 0 || STYPEG(hptr) == stype) &&
        strcmp(SYMNAME(hptr), symname) == 0) {
      if (scope == 0 || (scope == -1 && test_scope(hptr) > 0) ||
          scope == SCOPEG(hptr)) {
        return hptr;
      }
    }
  }
  return 0;
}

LOGICAL
is_array_sptr(int sptr)
{
  if (sptr > NOSYM) {
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
      return TRUE;
    case ST_VAR:
      return is_array_dtype(DTYPEG(sptr));
    default:
      return FALSE;
    }
  }
  return FALSE;
}

LOGICAL
is_unl_poly(int sptr)
{
  return sptr > NOSYM &&
         CLASSG(sptr) &&
         is_dtype_unlimited_polymorphic(DTYPEG(sptr));
}

bool
is_impure(int sptr)
{
  if ((STYPEG(sptr) == ST_INTRIN || STYPEG(sptr) == ST_PD) &&
      INKINDG(sptr) == IK_ELEMENTAL)
    return false;
  return IMPUREG(sptr) || (!PUREG(sptr) && !ELEMENTALG(sptr));
}

LOGICAL
needs_descriptor(int sptr)
{
  if (sptr > NOSYM) {
    if (IS_PROC_DUMMYG(sptr)) {
      return TRUE;
    }
    if (ST_ISVAR(STYPEG(sptr)) || STYPEG(sptr) == ST_IDENT) {
      DTYPE dtype = DTYPEG(sptr);
      return ASSUMSHPG(sptr) || POINTERG(sptr) || ALLOCATTRG(sptr) ||
             IS_PROC_DUMMYG(sptr) ||
             (is_array_dtype(dtype) && ADD_ASSUMSHP(dtype));
    }
  }
  /* N.B. Scalar CLASS polymorphic dummy arguments get type descriptors only,
   * not full descriptors, as a special case in add_class_arg_descr_arg().
   */
  return FALSE;
}

/* \brief Returns true if a procedure dummy argument needs a procedure
 *        descriptor.
 *
 * By default, we do not use a descriptor argument for dummy arguments
 * declared EXTERNAL since they could be non-Fortran procedures.
 * If the procedure dummy argument is an interface, not declared
 * EXTERNAL, or a part of an internal procedure, then we assume it is a Fortran
 * procedure and we will use a descriptor argument.
 *
 * XBIT(54, 0x20) overrides this restriction. That is, we will always use a
 * procedure descriptor when XBIT(54, 0x20) is enabled.
 *
 * \param symfunc is the procedure dummy argument we are testing.
 *
 * \return true if procedure dummy needs a descriptor; else false.
 */
bool
proc_arg_needs_proc_desc(SPTR symfunc)
{
  return IS_PROC_DUMMYG(symfunc) && (XBIT(54, 0x20) ||
         IS_INTERFACEG(symfunc) || !TYPDG(symfunc) || INTERNALG(gbl.currsub));
}

/* This function encloses an idiom that appears more than once in the
 * Fortran front-end to follow the symbol linkage convention
 * used to locate descriptor members in derived types.
 */
SPTR
get_member_descriptor(int sptr)
{
  SPTR mem;
  assert(sptr > NOSYM && STYPEG(sptr) == ST_MEMBER,
         "get_member_descriptor: bad member", sptr, ERR_Severe);
  for (mem = SYMLKG(sptr); mem > NOSYM && HCCSYMG(mem); mem = SYMLKG(mem)) {
    if (DESCARRAYG(mem))
      return mem;
  }
  return NOSYM;
}

int
find_member_descriptor(int sptr)
{
  if (sptr > NOSYM && STYPEG(sptr) == ST_MEMBER &&
      (CLASSG(sptr) || FINALIZEDG(sptr))) {
    int dsc_mem = get_member_descriptor(sptr);
    if (dsc_mem > NOSYM && DESCARRAYG(dsc_mem))
      return dsc_mem;
  }
  return 0;
}

/* Ferret out a variable's descriptor from any of the places in which
 * the front-end might have hidden it.  Represent it as an AST
 * if it exists.
 */
int
find_descriptor_ast(int sptr, int ast)
{
  int desc_ast, desc_sptr;

  if (sptr <= NOSYM)
    return 0;
  if ((desc_ast = DSCASTG(sptr)))
    return desc_ast;
  if ((desc_sptr = find_member_descriptor(sptr)) > NOSYM ||
      (desc_sptr = SDSCG(sptr)) > NOSYM ||
      (desc_sptr = DESCRG(sptr)) > NOSYM) {
    if (STYPEG(desc_sptr) != ST_MEMBER || ast > 0) {
      desc_ast = mk_id(desc_sptr);
      if (STYPEG(desc_sptr) == ST_MEMBER)
        desc_ast = check_member(ast, desc_ast);
      DESCUSEDP(sptr, TRUE);
      return desc_ast;
    }
  }
  if (SCG(sptr) == SC_DUMMY && CLASSG(sptr)) {
    /* Identify a type descriptor argument */
    int scope_sptr = gbl.currsub;
    if (gbl.internal > 0)
      scope_sptr = resolve_sym_aliases(SCOPEG(sptr));
    desc_sptr = get_type_descr_arg(scope_sptr, sptr);
    if (desc_sptr > NOSYM) {
      DESCUSEDP(sptr, TRUE);
      return mk_id(desc_sptr);
    }
  }
  return 0;
}

/* Scan a dummy argument list for a specific symbol's name (if valid) and
 * return its 1-based position if it's present in the list, else 0.
 * Done the hard way by comparing names, ignoring case.
 */
int
find_dummy_position(int proc_sptr, int arg_sptr)
{
  if (proc_sptr > NOSYM && arg_sptr > NOSYM) {
    const char *name = SYMNAME(arg_sptr);
    int paramct, dpdsc, iface;
    proc_arginfo(proc_sptr, &paramct, &dpdsc, &iface);
    if (dpdsc > 0) {
      int j, *argument = &aux.dpdsc_base[dpdsc];
      for (j = 0; j < paramct; ++j) {
        if (argument[j] > NOSYM && strcmp(SYMNAME(argument[j]), name) == 0)
          return 1 + j; /* 1-based list position */
      }
    }
  }
  return 0;
}

/* Scan the whole symbol table(!) and return the maximum value of
 * the INVOBJ field for every valid binding of a t.b.p. for which
 * the argument is an implementation without NOPASS.
 */
int
max_binding_invobj(int impl_sptr, int invobj)
{
  int sptr;
  for (sptr = 1; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_MEMBER && CLASSG(sptr) &&
        VTABLEG(sptr) == impl_sptr && !NOPASSG(sptr)) {
      int bind_sptr = BINDG(sptr);
      if (bind_sptr > NOSYM && INVOBJG(bind_sptr) > invobj)
        invobj = INVOBJG(bind_sptr);
    }
  }
  return invobj;
}

static LOGICAL
test_tbp_or_final(int sptr)
{
  /* Subtlety: For type-bound procedures, BIND and VTABLE are nonzero,
   * but might be set to NOSYM (1).  The FINAL field's value is documented
   * as being the rank of the final procedure's argument plus one, but
   * it can also be -1 to represent a forward reference to a final subroutine.
   */
  return sptr > NOSYM && STYPEG(sptr) == ST_MEMBER && CCSYMG(sptr) &&
         CLASSG(sptr) && VTABLEG(sptr) != 0;
}

LOGICAL
is_tbp(int sptr)
{
  return test_tbp_or_final(sptr) && (BINDG(sptr) != 0 || IS_TBP(sptr));
}

LOGICAL
is_final_procedure(int sptr)
{
  return test_tbp_or_final(sptr) && FINALG(sptr) != 0;
}

LOGICAL
is_tbp_or_final(int sptr)
{
  return is_tbp(sptr) || is_final_procedure(sptr);
}

/** \brief create a temporary variable that holds a temporary descriptor.
  *
  * \param dtype is the data type of the temporary variable.
  *
  * \returns the temporary variable.
  */
int
get_tmp_descr(DTYPE dtype)
{
  int tmpv = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, sem.sc);
  if (DTY(dtype) != TY_ARRAY && !SDSCG(tmpv)) {
     set_descriptor_rank(1); /* force a full (true) descriptor on a scalar */
     get_static_descriptor(tmpv);
     set_descriptor_rank(0);
   } else if (!SDSCG(tmpv)) {
     get_static_descriptor(tmpv);
  }
  return tmpv;
}

/** \brief get a temporary procedure pointer to a specified procedure.
 *
 *  \param sptr is the ST_PROC pointer target.
 *
 *  \returns the procedure pointer.
 */
SPTR
get_proc_ptr(SPTR sptr)
{
  DTYPE dtype;
  SPTR tmpv;
  int sc;

  if (!IS_PROC(STYPEG(sptr)))
    return NOSYM;

  dtype = DTYPEG(sptr);
  tmpv  = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, sem.sc);

  dtype = get_type(6, TY_PROC, dtype);
  DTY(dtype + 2) = sptr; /* interface */
  DTY(dtype + 3) = PARAMCTG(sptr); /* PARAMCT */
  DTY(dtype + 4) = DPDSCG(sptr); /* DPDSC */
  DTY(dtype + 5) = FVALG(sptr); /* FVAL */

  dtype = get_type(2, TY_PTR, dtype);

  POINTERP(tmpv, 1);
  DTYPEP(tmpv, dtype);
  sc = get_descriptor_sc();
  set_descriptor_sc(SC_LOCAL);
  get_static_descriptor(tmpv);
  set_descriptor_sc(sc);
  return tmpv;
}


/* Build an AST that references the byte length field in a descriptor,
 * if it exists and can be subscripted, else return 0.
 */
int
get_descriptor_length_ast(int descriptor_ast)
{
  if (descriptor_ast > 0 && is_array_dtype(A_DTYPEG(descriptor_ast))) {
    int subs = mk_isz_cval(get_byte_len_indx(), astb.bnd.dtype);
    return mk_subscr(descriptor_ast, &subs, 1, astb.bnd.dtype);
  }
  return 0;
}

/* If a symbol has a descriptor that might need its byte length field
 * defined, return an AST to which the length should be stored, else 0.
 */
int
symbol_descriptor_length_ast(SPTR sptr, int ast)
{
  int descr_ast = find_descriptor_ast(sptr, ast);
  if (descr_ast > 0) {
    DTYPE dtype = DTYPEG(sptr);
    if (DT_ISCHAR(dtype) ||
        is_unl_poly(sptr) ||
        is_array_dtype(dtype)) {
      return get_descriptor_length_ast(descr_ast);
    }
  }
  return 0;
}

/* Build an AST to characterize the length of a value.
 * Pass values for arguments when they're known, or the
 * appropriate invalid value (NOSYM, DT_NONE, &c.) when not.
 */
int
get_value_length_ast(DTYPE value_dtype, int value_ast,
                     SPTR sptr, DTYPE sptr_dtype,
                     int value_descr_ast)
{
  int ast;
  if (value_dtype > DT_NONE) {
    if (is_array_dtype(value_dtype))
      value_dtype = array_element_dtype(value_dtype);
    if (DT_ISCHAR(value_dtype)) {
      int len = string_length(value_dtype);
      if (len > 0) {
        return mk_isz_cval(len, astb.bnd.dtype);
      }
      if ((ast = DTY(value_dtype + 1)) > 0) {
        return ast;
      }
      if (value_ast > 0 &&
          (ast = string_expr_length(value_ast)) > 0) {
        return ast;
      }
    }
  }
  if (sptr > NOSYM && sptr_dtype > DT_NONE &&
      (ast = size_ast(sptr, sptr_dtype)) > 0)
    return ast;
  return get_descriptor_length_ast(value_descr_ast);
}

void
add_auto_len(int sym, int Lbegin)
{
  int dtype, cvlen;
  int lhs, rhs, ast, std;

  dtype = DDTG(DTYPEG(sym));
  if (DTY(dtype) != TY_CHAR && DTY(dtype) != TY_NCHAR)
    return;
  cvlen = CVLENG(sym);
#if DEBUG
  assert(
      (DDTG(DTYPEG(sym)) != DT_DEFERCHAR && DDTG(DTYPEG(sym)) != DT_DEFERNCHAR),
      "set_auto_len: arg is deferred-length character", sym, 4);
#endif
  if (cvlen == 0) {
    cvlen = sym_get_scalar(SYMNAME(sym), "len", DT_INT);
    CVLENP(sym, cvlen);
    ADJLENP(sym, 1);
    if (SCG(sym) == SC_DUMMY)
      CCSYMP(cvlen, 1);
  }
  /* If EARLYSPEC is set, the length assignment was done earlier. */
  if (!EARLYSPECG(CVLENG(sym))) {
    lhs = mk_id(cvlen);
    rhs = DTyCharLength(dtype);

    rhs = mk_convert(rhs, DTYPEG(cvlen));
    rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));

    ast = mk_assn_stmt(lhs, rhs, DTYPEG(cvlen));
    std = add_stmt_before(ast, Lbegin);
  }
} /* add_auto_len */

